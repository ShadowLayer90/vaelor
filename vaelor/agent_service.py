"""The root-side LAUNCH surface for one deployed cluster Agent (F4b-ii-B).

F4b-ii-A already built the loopback OpenAI runtime the appliance runs -
:mod:`vaelor.agent_server`, started as ``/opt/vaelor/venv/bin/python -m
vaelor.agent_server``, reading its 0600 JSON config from the ``VAELOR_AGENT_CONFIG``
path and binding a loopback port. This module is how the ROOT hardware bridge
brings one such runtime up and takes it down, exactly as
:mod:`vaelor.phoenix_service` and :mod:`vaelor.llm_server_proxy` launch their
processes: a ``*Process`` class with ``start``/``stop``/``alive`` and an injected
``run`` seam so the launch is verifiable without systemd or docker.

Two things run per agent, and this class owns both:

* **The agent unit.** ``vaelor-agent-<name>.service`` is written by THIS code with
  an ExecStart it fixes itself - ``/opt/vaelor/venv/bin/python -m
  vaelor.agent_server`` - so no caller can inject an argv. The unit runs as the
  UNPRIVILEGED ``vaelor-research`` account (never root), the one existing appliance
  identity that is neither in the ``docker`` group (which is root-equivalent) nor
  the ``vaelor-credentials`` group (the broker), so a model-driven tool loop cannot
  reach the docker socket or the credential broker. A new dedicated user is NOT
  minted here on purpose: the bridge runs under ``ProtectSystem=strict`` with
  ``/etc`` read-only save ``/etc/systemd/system``, so it could not write
  ``/etc/passwd`` even as root, and a user added by the installer would need a
  shipped unit change the wheel deploy cannot carry.

* **The inbound gate.** One nginx container (the :mod:`vaelor.llm_server_proxy`
  gate, reused byte-for-byte) fronts the agent's loopback port on the LAN,
  admitting ANY of the agent endpoint's CURRENT ``vsk_`` keys. The gate's
  upstream is FIXED to the agent's validated loopback port here - never a host from
  the caller - so this launch can never be turned into a relay to an arbitrary
  address. An agent endpoint with no active key runs NO gate (the no-keyless-door
  rule the LLM Server keeps), and :meth:`AgentServerProcess.rekey` replaces only
  the gate when the key set changes, leaving the runtime serving.

What is RUNNING is reported off the running agent itself, never off what a
caller last asked for (ACC-070, the LLM Server's VD-127 amendment applied to
agents): the gate config carries the SET-hash of the keys rendered into it on its
first line (:data:`KEY_SET_MARKER`), and the agent config carries the control
plane's ``surface_digest`` of the tools, skills, instructions and backing model it
was started with. :meth:`AgentServerProcess.status` reads both back, so the
control plane's reconcile compares the live gate and runtime with the broker's
current keys and the agent's current surface, and re-keys or relaunches on drift.

The agent config carries the model key and the MCP injected data, so it is written
0600 to ``/run/vaelor/agents`` (tmpfs, in the bridge's ``ReadWritePaths``; the state
root is EROFS under ``ProtectSystem=strict``) and then chowned to the runtime
account so the key is readable by that one uid and root alone - never on an argv, in
the unit, in the environment value, or to the rest of the ``vaelor`` group. The
``/run/vaelor`` root is ``0770`` group-writable (so gate configs can be dropped
there), which is the whole symlink-attack surface, closed on four fronts: the
``agents`` subdirectory is created ``0710 root:vaelor`` (an in-group process cannot
create anything, symlink included, inside it); the config is opened ``O_NOFOLLOW``
(a symlink at the final path fails the open rather than being followed); the
parent run root is given the STICKY bit (a non-owner cannot ``rename`` the
root-owned ``agents`` entry aside to plant a symlink at its path - the installer
sets ``1770`` from boot, and this code re-asserts it on an already-running box); and
the ensure step REFUSES a subdir that is already a symlink or a non-directory rather
than ``chmod``/``chown`` through it.

Two review findings are recorded here without a behaviour change:

* SHOULD-FIX (deferred to an installer follow-up): the agent unit shares the
  ``vaelor-research`` uid with the untrusted-web application-research broker, so
  that broker - same uid - can read the agent's 0600 model-key config. The proper
  fix is a dedicated ``vaelor-agent`` system user (group ``vaelor`` only, not
  ``docker``/``vaelor-credentials``), but minting one needs an installer change the
  wheel deploy cannot carry, so reusing ``vaelor-research`` is the deploy-compatible
  first cut and the dedicated user is a tracked follow-up.
* ACCEPTED under the bridge's existing trust model: the name-keyed verbs carry no
  per-caller ownership, so any group-vaelor caller can hijack or re-key an agent by
  a name collision. This matches every other bridge launch verb - the bridge has no
  caller identity and trusts in-group callers equally - so it is in-scope-accepted,
  not a new hole this surface opens.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .flm_service import _validate_port
from .gpu_pool_units import deployment_name
from .llm_server_proxy import (
    DOCKER_PULL_TIMEOUT,
    DOCKER_QUERY_TIMEOUT,
    DOCKER_RUN_TIMEOUT,
    PROXY_IMAGE,
    ProxyImageMissingError,
    proxy_container_command,
    render_proxy_config,
    validate_api_key,
    write_proxy_config,
)
from .llm_server_state import applied_marker
from .runtime_paths import RUN_ROOT, run_path

#: The loopback band a deployed agent's OpenAI runtime binds. Deliberately its
#: OWN range, clear of every other serving allocation so a launch here can never
#: collide with one of those: the single-node GPU fork/replica band (8000-8079),
#: the managed model server's loopback allocation (8080-8099), the cluster LLM and
#: application deploy band (8100-8199), the LLM Server auth proxy (11434) and the
#: Phoenix trace UI (6006) are all outside it. The control plane picks a free port
#: inside this band for each agent and passes it here; this boundary re-checks it.
AGENT_PORT_BAND_START = 8300
AGENT_PORT_BAND_END = 8379

#: The UNPRIVILEGED account the agent unit runs as, and its group. ``vaelor-research``
#: exists on every appliance (the installer creates it as the bounded public-egress
#: broker identity) and is in the ``vaelor`` group ONLY - not ``docker``, not
#: ``vaelor-credentials`` - which is precisely the trust an agent runtime needs:
#: outbound HTTP to the model and MCP, and nothing else. Reused rather than minted
#: because the bridge cannot write ``/etc/passwd`` under ``ProtectSystem=strict``.
AGENT_UNIT_USER = "vaelor-research"
AGENT_UNIT_GROUP = "vaelor"

#: Where the agent unit files are written. In the bridge's ``ReadWritePaths`` (the
#: one ``/etc`` subtree it may write under ``ProtectSystem=strict``), so no shipped
#: unit change is needed to land one.
AGENT_UNIT_DIR = "/etc/systemd/system"

#: The python module the unit runs, and the interpreter that runs it. The ExecStart
#: is assembled from these HERE, in code the caller cannot influence, so the launch
#: argv is fixed no matter what a verb payload carries. A LITERAL POSIX path (not a
#: ``pathlib`` join): the unit is a Linux artifact, so the interpreter is the
#: appliance's fixed venv path exactly as the bridge's own unit and
#: :mod:`vaelor.appliance_upgrade` spell it, never a host-shaped path.
AGENT_SERVER_MODULE = "vaelor.agent_server"
AGENT_VENV_PYTHON = "/opt/vaelor/venv/bin/python"

#: The environment variable the unit sets to name the agent's 0600 config. Only the
#: PATH travels in the environment; the secret is in the file the path names.
AGENT_CONFIG_ENV = "VAELOR_AGENT_CONFIG"

#: A root-owned, group-NON-writable subdirectory under ``/run/vaelor`` where the two
#: 0600 configs live. The ``/run/vaelor`` root itself is ``0770 root:vaelor`` (the
#: installer grants the group write so gate configs can be dropped there), which
#: would let any group-vaelor process PLANT A SYMLINK at a config path and steer a
#: root writer into an arbitrary target. This subdirectory is created ``0710
#: root:vaelor`` instead - root owns it, the group gets ``--x`` (traverse only, no
#: write), so the runtime account can descend and read its own 0600 file but cannot
#: create anything here, symlink included.
AGENT_CONFIG_SUBDIR = "agents"

#: The one host the gate's upstream may ever be: the agent's own loopback. Fixed so
#: the gate fronts the validated loopback agent port and nothing else - no relay to
#: an address a caller supplied.
AGENT_LOOPBACK_HOST = "127.0.0.1"

#: How long ``systemctl`` calls may take before they are abandoned.
SYSTEMCTL_TIMEOUT = 30

#: The largest agent config this launch will land, a defence-in-depth bound behind
#: the socket frame the payload already crosses.
MAX_CONFIG_BYTES = 256 * 1024

#: The first line of every agent gate config: an nginx comment carrying the
#: SET-hash of the keys rendered below it (never a key - the hash of the per-key
#: fingerprints, :func:`vaelor.llm_server_state.applied_marker`'s basis). The
#: config is rewritten only immediately before its container is replaced, so
#: this line is what the RUNNING gate admits.
KEY_SET_MARKER = "# vaelor-key-set "

#: How far a stored gate generation may run ahead of the bridge's own clock
#: before it is read as a clock step back rather than a newer key set (N2).
GENERATION_CLOCK_SKEW_SECONDS = 5.0

#: The most keys one agent gate renders - a bound on the config, far above any
#: owner's real count.
MAX_GATE_KEYS = 64


def _read_at(gate: Any) -> float:
    """When the caller read the gate's key set (``keys_read_at``), or 0 if unsaid."""
    try:
        value = float((gate or {}).get("keys_read_at") or 0)
    except (TypeError, ValueError, AttributeError):
        return 0.0
    return value if value == value and 0 <= value < 1e12 else 0.0


class AgentLaunchError(RuntimeError):
    """The agent unit or its gate could not be brought up. A presentable message."""


def _default_run(command: Any, *, timeout: Optional[int] = None):
    """Run one systemctl/docker command, capturing output, never raising on non-zero.

    stdout/stderr are captured, not discarded: a failed ``systemctl start`` or
    ``docker run`` explains itself on stderr, which the caller lifts into the raised
    error rather than swallowing (LESSONS 8).
    """
    return subprocess.run(
        list(command), capture_output=True, text=True, check=False, timeout=timeout
    )


def _default_chown_to_user(path: str, user: str) -> None:
    """Give ``path`` to ``user`` (uid + primary gid), best-effort.

    The agent config is created 0600 root-owned; this hands it to the runtime
    account so that one uid, and root, can read the key - and the rest of the
    ``vaelor`` group cannot. Best-effort like the proxy config's ``chmod``: a box
    without the account (or a non-POSIX test host with no ``os.lchown``) leaves the
    file root-owned rather than failing the launch, and on the appliance where the
    account exists the chown lands.

    ``os.lchown`` (never ``os.chown``) so a symlink at ``path`` is chowned as the
    link, not followed to its target: even though the file was just opened
    ``O_NOFOLLOW`` (so ``path`` is a regular file this process created), chowning
    the link rather than a target it points at removes any lingering way to hand
    an attacker-chosen file to the runtime uid.
    """
    try:
        import pwd

        entry = pwd.getpwnam(user)
        os.lchown(path, entry.pw_uid, entry.pw_gid)
    except (ImportError, KeyError, OSError, AttributeError):
        pass


def _default_ensure_config_dir(directory: str) -> None:
    """Create the root-owned, group-non-writable agent config dir, best-effort.

    ``0710 root:vaelor``: root owns it, the ``vaelor`` group gets ``--x`` (traverse
    only, NEVER write), everyone else nothing. That denies a group-vaelor process
    the ability to plant a symlink at a config path - which the ``0770``
    group-writable ``/run/vaelor`` root would otherwise allow - while the runtime
    account can still descend and read its own 0600 file. Best-effort like the
    config chown: a non-POSIX host or a missing ``vaelor`` group leaves the
    directory at whatever ``makedirs`` produced rather than failing the launch, and
    on the appliance the tightening lands.

    Two further symlink-class defences. FAIL-CLOSED: if the ``agents`` path is
    already a symlink (or exists as a non-directory), refuse rather than proceed -
    ``chmod``/``chown`` FOLLOW a symlink, so acting on a planted link would let root
    tighten and write through an attacker-chosen directory. And the STICKY BIT is
    set on the parent ``/run/vaelor``: that root is ``0770`` group-writable, so
    without sticky an in-group process could ``rename`` the root-owned ``agents``
    entry aside (rename needs write on the parent, which the group has) and plant a
    symlink at its path; sticky restricts rename/delete of an entry to its owner.
    Every ``/run/vaelor`` entry is root-created, so root and the directory's other
    users (the broker/workload sockets, the proxy and balancer configs) are
    unaffected - sticky only bars a non-owner from removing another's entry.
    """
    if os.path.islink(directory) or (
        os.path.exists(directory) and not os.path.isdir(directory)
    ):
        raise AgentLaunchError(
            "SECURITY / cluster agent: the config directory {!r} is a symlink or "
            "not a directory; refusing to write through it.".format(directory)
        )
    # Sticky the parent run root so a non-owner group member can no longer rename or
    # delete the root-owned subdir out from under this launch. Best-effort.
    try:
        parent = os.path.dirname(directory.rstrip("/")) or directory
        parent_mode = stat.S_IMODE(os.stat(parent).st_mode)
        os.chmod(parent, parent_mode | stat.S_ISVTX)
    except (OSError, ValueError, AttributeError):
        pass
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        return
    try:
        os.chmod(directory, 0o710)
    except OSError:
        pass
    try:
        import grp

        os.chown(directory, 0, grp.getgrnam(AGENT_UNIT_GROUP).gr_gid)
    except (ImportError, KeyError, OSError, AttributeError):
        pass


def agent_unit_name(name: str) -> str:
    """``vaelor-agent-<name>.service`` for a validated agent name.

    The name is held to the shared cluster/agent slug rule
    (:func:`vaelor.gpu_pool_units.deployment_name`), the same one the agent
    deployment store validates a record's name against, so it is safe in a unit
    filename and a container name without a second rule that could drift.
    """
    return "vaelor-agent-{}.service".format(deployment_name(name))


def agent_config_path(name: str) -> str:
    """The 0600 config the agent unit reads, under ``/run/vaelor/agents``."""
    return run_path("{}/agent-{}.conf".format(AGENT_CONFIG_SUBDIR, deployment_name(name)))


def gate_config_path(name: str) -> str:
    """The 0600 nginx config the agent's inbound gate reads, under ``/run/vaelor/agents``."""
    return run_path(
        "{}/agent-{}-gate.conf".format(AGENT_CONFIG_SUBDIR, deployment_name(name))
    )


def gate_container_name(name: str) -> str:
    """``vaelor-agent-<name>-gate``: the nginx container fronting the agent."""
    return "vaelor-agent-{}-gate".format(deployment_name(name))


def validate_agent_port(port: Any) -> int:
    """Return the agent's loopback port, or raise unless it is in the agent band.

    Layered on :func:`vaelor.flm_service._validate_port` (an integer in 1024-65535,
    never a reserved control-plane port), then narrowed to the agent-serving band so
    a launch reached by any path cannot bind an agent onto the GPU, model-server,
    cluster-deploy, proxy or trace ports.
    """
    value = _validate_port(port)
    if not AGENT_PORT_BAND_START <= value <= AGENT_PORT_BAND_END:
        raise ValueError(
            "A cluster agent binds a loopback port from {} to {}; {} is outside "
            "that band.".format(AGENT_PORT_BAND_START, AGENT_PORT_BAND_END, value)
        )
    return value


def _validate_loopback(value: Any) -> str:
    """Return the host, or raise unless it is the agent's own loopback.

    The gate's upstream is built from this and nothing a caller sends, so the one
    place the gate argv is assembled refuses any target but ``127.0.0.1`` - the
    launch cannot be widened into a relay to an arbitrary endpoint.
    """
    text = str(value or "").strip()
    if text != AGENT_LOOPBACK_HOST:
        raise ValueError(
            "SECURITY / cluster agent: the inbound gate may front the agent's own "
            "loopback ({}) only, never {!r}.".format(AGENT_LOOPBACK_HOST, text)
        )
    return text


def agent_unit_text(*, name: str, port: int, config_path: str) -> str:
    """Render the systemd unit that supervises one agent's loopback runtime.

    Assembled line by line rather than from a template blob, the convention the
    sibling unit author :func:`vaelor.gpu_pool_units.unit_text` follows, so no
    duplicated body is shared between two unit writers.

    The three security-critical lines are fixed HERE, by this code:

    * ``ExecStart`` is the interpreter and module spelled from module constants, so
      a verb payload can never influence what runs;
    * ``User``/``Group`` are the unprivileged runtime account, so the process is
      never root and cannot reach the docker socket or the credential broker;
    * ``Environment`` carries ONLY the config PATH - the model key lives in the file
      that path names, never in the unit.

    The sandbox mirrors the appliance's other host-process units: ``ProtectSystem=
    strict`` with ``/run/vaelor`` the one writable path (so the process reads its
    config), ``NoNewPrivileges``, ``PrivateTmp`` and the ``Protect*``/``Restrict*``
    set - but deliberately NOT ``PrivateNetwork``, because the loopback runtime must
    reach the model endpoint and its MCP servers. ``Restart=on-failure`` with
    ``StartLimitIntervalSec=0`` means systemd relaunches a crashed agent without
    ever giving up on the start-rate limit.
    """
    exec_start = "{} -m {}".format(AGENT_VENV_PYTHON, AGENT_SERVER_MODULE)
    lines = [
        "[Unit]",
        "Description=Vaelor cluster agent runtime for {}".format(deployment_name(name)),
        "After=network-online.target",
        "Wants=network-online.target",
        "",
        "[Service]",
        "Type=simple",
        "User={}".format(AGENT_UNIT_USER),
        "Group={}".format(AGENT_UNIT_GROUP),
        "ExecStart={}".format(exec_start),
        "Environment={}={}".format(AGENT_CONFIG_ENV, config_path),
        "Restart=on-failure",
        "RestartSec=3",
        "StartLimitIntervalSec=0",
        "UMask=0007",
        "NoNewPrivileges=yes",
        "PrivateTmp=yes",
        "ProtectSystem=strict",
        "ProtectHome=yes",
        "ProtectControlGroups=yes",
        "ProtectKernelModules=yes",
        "ProtectKernelTunables=yes",
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "RestrictRealtime=yes",
        "LockPersonality=yes",
        "ReadWritePaths=/run/vaelor",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ]
    # The runtime's loopback port is documented in a comment, not an argv element:
    # the agent server reads it from the config, and the gate's upstream is derived
    # from the same validated value, so the port need not (and must not) reach the
    # ExecStart where a reader could mistake it for an injectable argument.
    header = "# Loopback runtime port {}, fronted by {}.\n".format(
        int(port), gate_container_name(name)
    )
    return header + "\n".join(lines)


def _write_unit(unit_dir: str, unit: str, body: str) -> str:
    """Write a unit file 0644 root-owned and return its full path.

    A unit file is not secret (it names the config path, not the key), so it is a
    plain 0644 create - distinct from the agent and gate configs, which carry the
    key and are landed 0600.
    """
    target = Path(unit_dir) / unit
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(body)
    return str(target)


class AgentServerProcess:
    """A supervised cluster-agent runtime and its inbound gate, root-side.

    The agent analogue of :class:`vaelor.phoenix_service.PhoenixServerProcess` and
    :class:`vaelor.llm_server_proxy.LlmServerProxyProcess`: ``start``/``stop``/
    ``alive`` over an injected ``run`` seam, so a test drives the whole launch with
    no systemd and no docker. Unlike those two - each one container under a fixed
    name - an agent is a systemd unit (the loopback python runtime) PLUS a docker
    gate, both keyed by the agent name, so every method takes the name.
    """

    #: What this launch is called in its own failure sentence.
    label = "cluster agent"

    def __init__(
        self,
        *,
        run: Optional[Callable[..., Any]] = None,
        docker: Optional[str] = None,
        systemctl: Optional[str] = None,
        unit_dir: str = AGENT_UNIT_DIR,
        run_root: Optional[str] = None,
        image: str = PROXY_IMAGE,
        chown: Callable[[str, str], None] = _default_chown_to_user,
        ensure_dir: Callable[[str], None] = _default_ensure_config_dir,
        clock: Callable[[], float] = time.time,
    ):
        self._run = run or _default_run
        self._docker = docker
        self._systemctl = systemctl
        self._unit_dir = unit_dir
        #: Where the two 0600 configs are landed. Defaults to the production
        #: ``/run/vaelor`` (tmpfs, in the bridge's ``ReadWritePaths``); a test
        #: injects a tempdir so a launch writes nowhere real. The configs go in the
        #: ``agents`` subdirectory under it, tightened to 0710 root:vaelor.
        self._run_root = str(run_root) if run_root else str(RUN_ROOT)
        self._image = image
        self._chown = chown
        self._ensure_dir = ensure_dir
        self._clock = clock

    def _config_dir(self) -> str:
        """The root-owned, group-non-writable directory both configs live in."""
        return os.path.join(self._run_root, AGENT_CONFIG_SUBDIR)

    def _config_file(self, name: str) -> str:
        """The agent's 0600 config path under the tightened config directory."""
        return os.path.join(
            self._config_dir(), "agent-{}.conf".format(deployment_name(name))
        )

    def _gate_config_file(self, name: str) -> str:
        """The gate's 0600 config path under the tightened config directory."""
        return os.path.join(
            self._config_dir(), "agent-{}-gate.conf".format(deployment_name(name))
        )

    def _docker_binary(self) -> str:
        return self._docker or shutil.which("docker") or "/usr/bin/docker"

    def _systemctl_binary(self) -> str:
        return self._systemctl or shutil.which("systemctl") or "/usr/bin/systemctl"

    def ensure_gate_image(self) -> None:
        """Make sure the gate's nginx image is present, pulling it once if not.

        Called by the bridge verb BEFORE it takes the lock (an image pull can run
        for minutes and must never be held across, per
        :ref:`same-allowlist-is-not-same-boundary`), so :meth:`start` then does only
        the fast launch. ``docker image inspect`` is the cheap presence probe; only
        a miss triggers a ``docker pull``, and a failed fetch raises
        :class:`vaelor.llm_server_proxy.ProxyImageMissingError` naming the image.
        """
        docker = self._docker_binary()
        inspected = self._run(
            [docker, "image", "inspect", self._image], timeout=DOCKER_QUERY_TIMEOUT
        )
        if getattr(inspected, "returncode", 1) == 0:
            return
        pulled = self._run([docker, "pull", self._image], timeout=DOCKER_PULL_TIMEOUT)
        if getattr(pulled, "returncode", 1) != 0:
            raise ProxyImageMissingError(
                "The cluster agent gate image {} is not present and could not be "
                "pulled: {}".format(
                    self._image, (getattr(pulled, "stderr", "") or "").strip()
                )
            )

    def start(
        self,
        *,
        name: str,
        port: Any,
        config_content: str,
        gate: Dict[str, Any],
        skip_ensure: bool = False,
    ) -> dict:
        """Validate, land the config, write and start the unit, then start the gate.

        Everything that could refuse a bad request - the name, the port band, the
        gate's key and listen host, the upstream-is-loopback rule, the config size -
        is checked BEFORE anything is written, so a malformed request never leaves a
        half-started agent behind. Then, in order:

        1. the agent config (:paramref:`config_content`, the opaque 0600 JSON the
           runtime reads - model key and MCP data included) is written to
           ``/run/vaelor/agent-<name>.conf`` and chowned to the runtime account;
        2. the unit is written with the ExecStart THIS code fixes and started
           (``daemon-reload`` first; a prior failed unit of the same name is
           ``reset-failed`` so a redeploy starts clean);
        3. the inbound single-key gate is launched, its upstream FIXED to the
           agent's own validated loopback port.

        ``skip_ensure`` lets the bridge verb pull the gate image off the lock and
        pass ``True`` so only the fast launch is serialized (mirrors
        :meth:`vaelor.phoenix_service.PhoenixServerProcess.start`).
        """
        agent = deployment_name(name)
        agent_port = validate_agent_port(port)
        content = str(config_content or "")
        if not content.strip():
            raise ValueError("A cluster agent launch requires a config body.")
        if len(content.encode("utf-8")) > MAX_CONFIG_BYTES:
            raise ValueError("The cluster agent config is larger than is accepted.")
        gate_listen_host, gate_listen_port, gate_keys = self._validate_gate(gate)
        # The gate config is rendered before anything is written, so a bad key or
        # host is refused here (the key/coupling lives in llm_server_proxy). No
        # key means no gate at all, never a keyless one.
        gate_config = self._render_gate(
            gate_listen_host, gate_listen_port, agent_port, gate_keys
        )

        # Land both 0600 configs in the tightened 0710 root:vaelor subdirectory,
        # created here (root) before any write so a group-vaelor process never had
        # a group-writable directory to plant a symlink in.
        self._ensure_dir(self._config_dir())
        config_file = self._config_file(agent)
        write_proxy_config(config_file, content)
        self._chown(config_file, AGENT_UNIT_USER)

        unit = agent_unit_name(agent)
        _write_unit(
            self._unit_dir, unit,
            agent_unit_text(name=agent, port=agent_port, config_path=config_file),
        )
        self._start_unit(unit)

        # The gate is left alone when it already runs exactly this key set
        # (a unit-only relaunch must not bounce a converged door, S3) or a
        # NEWER set than this caller read (S4).
        outcome = self._apply_gate(agent, gate_config, gate_listen_port, gate_keys,
                                   _read_at(gate), skip_ensure)
        return {**self.status(agent), "gate_change": outcome}

    def rekey(
        self, *, name: str, port: Any, gate: Dict[str, Any], skip_ensure: bool = False,
    ) -> dict:
        """Replace ONLY the gate, with the endpoint's current key set; the runtime keeps serving.

        The key change path (ACC-070): a key minted, rotated or revoked reaches
        the running gate through here, without a runtime restart. Every
        validation :meth:`start` makes of the gate is made again; an EMPTY key
        set stops the gate outright (no keyless door) rather than leaving the
        last keys admitted.
        """
        agent = deployment_name(name)
        agent_port = validate_agent_port(port)
        listen_host, listen_port, keys = self._validate_gate(gate)
        config = self._render_gate(listen_host, listen_port, agent_port, keys)
        # N1: a re-key that lands after a Remove (the console's request runs
        # outside the executor's lifecycle lock) must not start a gate for an
        # agent that no longer exists. `stop` unlinks the unit and the config,
        # so either missing means there is nothing here to re-key.
        if not self._deployed_here(agent):
            raise AgentLaunchError(
                "The {} {} is not deployed here, so its gate was not "
                "started.".format(self.label, agent)
            )
        self._ensure_dir(self._config_dir())
        outcome = self._apply_gate(agent, config, listen_port, keys, _read_at(gate), skip_ensure)
        return {**self.status(agent), "gate_change": outcome}

    def _apply_gate(
        self, agent: str, config: str, listen_port: int, keys: List[str],
        read_at: float, skip_ensure: bool,
    ) -> str:
        """Bring the gate to ``config``, or say why it was left: the one gate decision.

        * ``stale`` - the key set now running was read AFTER this caller read
          its own (the console's re-key and the executor's reconcile race; the
          later reading wins whichever request lands last, S4);
        * ``unchanged`` - the gate already runs exactly this config;
        * ``closed`` / ``replaced`` - the gate was stopped (no keys) or replaced.

        ``read_at`` is when the caller read the key set from the broker; it is
        recorded beside the gate (not a secret) whatever the outcome.
        """
        if self._gate_generation(agent) > read_at:
            return "stale"
        running = self.gate_running(agent)
        if config and running and self._gate_config_text(agent) == config:
            self._record_generation(agent, read_at)
            return "unchanged"
        if not config:
            self._stop_gate(agent)
            self._record_generation(agent, read_at)
            return "closed"
        self._start_gate(agent, config, listen_port, keys, skip_ensure)
        self._record_generation(agent, read_at)
        return "replaced"

    def _deployed_here(self, name: str) -> bool:
        """Whether this agent's unit file and runtime config both exist."""
        return (
            Path(self._unit_dir, agent_unit_name(name)).is_file()
            and os.path.isfile(self._config_file(name))
        )

    def _gate_generation_file(self, name: str) -> str:
        return os.path.join(
            self._config_dir(), "agent-{}-gate.generation".format(deployment_name(name))
        )

    def _gate_generation(self, name: str) -> float:
        """When the running gate's key set was read (0 when never recorded).

        A stored time later than this bridge's own clock (plus a small skew)
        was written before the wall clock stepped back (an NTP correction,
        N2); it is read as 0 rather than making every later change "stale"
        until the clock catches up.
        """
        try:
            with open(self._gate_generation_file(name), encoding="utf-8") as handle:
                stored = float(handle.read().strip() or 0)
        except (OSError, ValueError):
            return 0.0
        return 0.0 if stored > self._clock() + GENERATION_CLOCK_SKEW_SECONDS else stored

    def _record_generation(self, name: str, read_at: float) -> None:
        write_proxy_config(self._gate_generation_file(name), repr(float(read_at)))

    def _gate_config_text(self, name: str) -> str:
        try:
            with open(self._gate_config_file(name), encoding="utf-8") as handle:
                return handle.read()
        except OSError:
            return ""

    def gate_image_present(self) -> bool:
        """Whether the gate image is already local (no pull)."""
        inspected = self._run(
            [self._docker_binary(), "image", "inspect", self._image],
            timeout=DOCKER_QUERY_TIMEOUT,
        )
        return getattr(inspected, "returncode", 1) == 0

    @staticmethod
    def _render_gate(
        listen_host: str, listen_port: int, agent_port: int, keys: List[str],
    ) -> str:
        """The gate config with its key-set line first, or ``""`` for no keys."""
        if not keys:
            return ""
        body = render_proxy_config(
            listen_host=listen_host, listen_port=listen_port,
            model_port=agent_port, api_keys=keys,
        )
        marker = applied_marker(True, keys).key_fingerprint
        return KEY_SET_MARKER + marker + "\n" + body

    def _validate_gate(self, gate: Any) -> tuple:
        """Read the gate's listen host, port and key SET off the payload, validated.

        The gate payload carries ONLY where the LAN door listens and the
        ``vsk_`` keys it admits (``api_keys``; a lone ``api_key`` from an older
        control plane is read as a one-key set) - never an upstream, which this
        launch fixes to the agent's loopback. Each key is validated
        (:func:`vaelor.llm_server_proxy.validate_api_key`) so it cannot break out
        of the nginx config string; the listen host is validated where the
        config is rendered. An empty set is allowed and means "no gate".
        """
        if not isinstance(gate, dict):
            raise ValueError("The cluster agent gate settings must be an object.")
        listen_host = str(gate.get("listen_host") or "").strip()
        if not listen_host:
            raise ValueError("The cluster agent gate needs a LAN listen host.")
        listen_port = _validate_port(gate.get("listen_port"))
        raw = gate.get("api_keys")
        if raw is None:
            raw = [gate.get("api_key")] if "api_key" in gate else []
        if not isinstance(raw, list) or len(raw) > MAX_GATE_KEYS:
            raise ValueError("The cluster agent gate keys must be a short list.")
        keys = sorted({validate_api_key(item) for item in raw})
        return listen_host, listen_port, keys

    def _start_unit(self, unit: str) -> None:
        """``daemon-reload``, clear any prior failure, then ``restart`` the unit.

        ``restart`` rather than ``start``: a ``start`` of a unit that is already
        active is a no-op, so a relaunch that landed a CHANGED config (a tool,
        skill or model change, ACC-074) would leave the runtime serving the old
        one. ``restart`` starts a stopped unit and re-reads a running one.
        """
        systemctl = self._systemctl_binary()
        self._run([systemctl, "daemon-reload"], timeout=SYSTEMCTL_TIMEOUT)
        # A prior run of the same name that failed leaves the unit in `failed`,
        # which blocks a restart until it is reset (the reset-failed-before-retry
        # rule the hot-patch lesson records); tolerated so a first launch is fine.
        self._run([systemctl, "reset-failed", unit], timeout=SYSTEMCTL_TIMEOUT)
        result = self._run([systemctl, "restart", unit], timeout=SYSTEMCTL_TIMEOUT)
        if getattr(result, "returncode", 1) != 0:
            raise AgentLaunchError(
                "The {} unit {} failed to start: {}".format(
                    self.label, unit, (getattr(result, "stderr", "") or "").strip()
                )
            )

    def _start_gate(
        self, name: str, config: str, listen_port: int, api_keys: List[str],
        skip_ensure: bool,
    ) -> None:
        """Replace the agent's gate container with one running ``config``.

        The gate is the same nginx launch as the LLM Server proxy
        (:func:`vaelor.llm_server_proxy.proxy_container_command`), so the key is in
        the mounted 0600 config and NOWHERE on the argv, and ``_require_keys`` there
        refuses a keyless gate. Its upstream is the agent's loopback port, asserted
        here.
        """
        _validate_loopback(AGENT_LOOPBACK_HOST)
        docker = self._docker_binary()
        container = gate_container_name(name)
        config_file = self._gate_config_file(name)
        command = proxy_container_command(
            listen_port=int(listen_port), api_keys=api_keys, docker=docker,
            image=self._image, container_name=container,
            config_host_file=config_file,
        )
        if not skip_ensure:
            self.ensure_gate_image()
        self._remove_container(container)
        # S1: the old gate must be GONE before its config is rewritten - a
        # rewrite in place under a still-running nginx would read as the new
        # key set while the old one keeps admitting the revoked key.
        self._require_gone(container)
        write_proxy_config(config_file, config)
        result = self._run(command, timeout=DOCKER_RUN_TIMEOUT)
        if getattr(result, "returncode", 1) != 0:
            # Nothing runs this config: unlink it so the key set reads "" and
            # the reconcile retries, rather than reading as applied.
            write_proxy_config(config_file, "")
            raise AgentLaunchError(
                "The {} gate container {} could not be started: {}".format(
                    self.label, container, (getattr(result, "stderr", "") or "").strip()
                )
            )

    def _container_exists(self, container: str) -> bool:
        """Whether docker still knows the container. Unanswerable reads as present."""
        try:
            result = self._run(
                [self._docker_binary(), "inspect", "-f", "{{.State.Running}}", container],
                timeout=DOCKER_QUERY_TIMEOUT,
            )
        except (OSError, subprocess.SubprocessError):
            return True
        return getattr(result, "returncode", 1) == 0

    def _require_gone(self, container: str) -> None:
        if self._container_exists(container):
            raise AgentLaunchError(
                "The {} gate container {} could not be removed, so its keys were "
                "not changed.".format(self.label, container)
            )

    def _stop_gate(self, name: str) -> None:
        """Take the gate down and unlink its key-bearing config, best-effort."""
        container = gate_container_name(name)
        try:
            self._run(
                [self._docker_binary(), "stop", "--time", "5", container],
                timeout=DOCKER_RUN_TIMEOUT,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        self._remove_container(container)
        write_proxy_config(self._gate_config_file(name), "")
        self._require_gone(container)

    def _remove_container(self, container: str) -> None:
        """``docker rm -f`` a stale gate of this name, best-effort."""
        docker = self._docker_binary()
        try:
            self._run([docker, "rm", "-f", container], timeout=DOCKER_QUERY_TIMEOUT)
        except (OSError, subprocess.SubprocessError):
            pass

    def stop(self, name: str) -> dict:
        """Stop and remove BOTH the agent unit and its gate, and unlink both configs.

        Best-effort throughout so a stop never raises: the unit is stopped,
        disabled and its file removed (``daemon-reload`` after), its failure state
        cleared; the gate container is stopped and removed; and both 0600 configs
        are unlinked so a removed agent leaves no key behind.
        """
        agent = deployment_name(name)
        unit = agent_unit_name(agent)
        was_active = self.alive(agent)
        systemctl = self._systemctl_binary()
        docker = self._docker_binary()
        for argv in (
            [systemctl, "stop", unit],
            [systemctl, "disable", unit],
        ):
            try:
                self._run(argv, timeout=SYSTEMCTL_TIMEOUT)
            except (OSError, subprocess.SubprocessError):
                pass
        try:
            Path(self._unit_dir, unit).unlink()
        except OSError:
            pass
        for argv in (
            [systemctl, "daemon-reload"],
            [systemctl, "reset-failed", unit],
        ):
            try:
                self._run(argv, timeout=SYSTEMCTL_TIMEOUT)
            except (OSError, subprocess.SubprocessError):
                pass
        container = gate_container_name(agent)
        try:
            self._run(
                [docker, "stop", "--time", "5", container], timeout=DOCKER_RUN_TIMEOUT
            )
            self._run([docker, "rm", "-f", container], timeout=DOCKER_QUERY_TIMEOUT)
        except (OSError, subprocess.SubprocessError):
            pass
        write_proxy_config(self._config_file(agent), "")
        write_proxy_config(self._gate_config_file(agent), "")
        write_proxy_config(self._gate_generation_file(agent), "")
        return {"stopped": True, "was_running": was_active}

    def alive(self, name: str) -> bool:
        """Whether the agent unit is active, per ``systemctl is-active``. Fail-safe.

        The unit is the runtime that actually serves the agent, so its liveness is
        the honest answer; any trouble reading it is reported as not-alive, so a
        caller applies rather than trusts a broken read.
        """
        systemctl = self._systemctl_binary()
        try:
            result = self._run(
                [systemctl, "is-active", agent_unit_name(name)],
                timeout=SYSTEMCTL_TIMEOUT,
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            return False
        return (getattr(result, "stdout", "") or "").strip() == "active"

    def gate_running(self, name: str) -> bool:
        """Whether the agent's gate container is running, per ``docker inspect``. Fail-safe."""
        docker = self._docker_binary()
        try:
            result = self._run(
                [docker, "inspect", "-f", "{{.State.Running}}", gate_container_name(name)],
                timeout=DOCKER_QUERY_TIMEOUT,
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            return False
        return (
            getattr(result, "returncode", 1) == 0
            and (getattr(result, "stdout", "") or "").strip() == "true"
        )

    def status(self, name: str) -> dict:
        """The honest launch status, read off the running agent, by name.

        The unit's liveness and the gate's, plus what each is RUNNING WITH:
        ``key_set`` is the set-hash on the gate config's first line (``""``
        when no gate runs, so a stopped gate never reads as carrying keys) and
        ``surface_digest`` is the digest the control plane stamped into the
        agent config it started with (``""`` when unreadable). Neither is a
        secret; neither file's key is ever returned.
        """
        agent = deployment_name(name)
        gate_running = self.gate_running(agent)
        return {
            "name": agent,
            "unit": agent_unit_name(agent),
            "running": self.alive(agent),
            "gate": gate_container_name(agent),
            "gate_running": gate_running,
            "key_set": self._running_key_set(agent) if gate_running else "",
            "surface_digest": self._running_digest(agent),
            "config": self._config_file(agent),
        }

    def _running_key_set(self, name: str) -> str:
        """The set-hash on the gate config's first line, or ``""``."""
        try:
            with open(self._gate_config_file(name), encoding="utf-8") as handle:
                first = handle.readline().strip()
        except OSError:
            return ""
        marker = KEY_SET_MARKER.strip()
        return first[len(marker):].strip() if first.startswith(marker) else ""

    def _running_digest(self, name: str) -> str:
        """The ``surface_digest`` the agent config was started with, or ``""``."""
        try:
            with open(self._config_file(name), encoding="utf-8") as handle:
                body = json.load(handle)
        except (OSError, ValueError):
            return ""
        return str(body.get("surface_digest") or "") if isinstance(body, dict) else ""
