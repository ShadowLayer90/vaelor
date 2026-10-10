"""Install/reconcile the Telegraf telemetry agent onto a worker over SSH (E2b).

The controller ships the **whole** agent stack to a lean worker — the pinned
Telegraf binary, the Vaelor emitter zipapp, its ``0600`` config and the systemd
unit — and nothing but the worker's system ``python3`` is assumed present. This
is the B1 pinned-artifact pattern `gpu_pool_runtime` uses, applied to telemetry:
every artifact is content-verified before it is placed, the secret (the ingest
key) reaches the worker only on ``stdin`` into a ``0600`` file, and every command
this module issues has an allowlisted first argv token
(`ssh_transport.REMOTE_COMMAND_ALLOWLIST`), so it clears the SSH transport
boundary unchanged — Telegraf and the emitter are only ever an ``ExecStart`` or
an ``inputs.exec`` *inside* the unit and config, never an SSH first argv, so no
allowlist widening is needed (the same way ``docker`` and ``vllm`` are).

The names and rendered text live in `worker_telemetry_config`; this module owns
only the SSH sequence — ship, verify, place, enable — and the reconcile that
brings a down agent back without evicting an unreachable worker.
"""

from __future__ import annotations

import base64
import hashlib
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import worker_telemetry_config as cfg
from .ssh_transport import SshTransportError
from .tls_paths import WORKER_OS_TRUST_FILE
from .telemetry_store import REPORTING_WINDOW_SECONDS
from .worker_telemetry_bundle import (
    build_sampler_bytes, build_zipapp_bytes, bundle_digest, sampler_bundle_digest,
)
from .worker_telemetry_reconcile_scheduler import DEFAULT_INTERVAL_SECONDS

#: Where the controller stages the pinned Telegraf release tarballs it ships. A
#: controller-local, Vaelor-owned directory; the default resolver reads the
#: arch-appropriate artifact from here.
TELEGRAF_STAGING_DIR = "/var/lib/vaelor/telegraf"

#: How systemd reports a unit it has never heard of (mirrors `gpu_pool_runtime`).
_LOAD_STATE = "LoadState"
_ACTIVE_STATE = "ActiveState"
_UNIT_NOT_FOUND = "not-found"

#: How long an ACTIVE agent may go without an accepted sample before the
#: controller reinstalls it. This is a different question from "is the machine
#: reporting?" (``telemetry_store.REPORTING_WINDOW_SECONDS``, which every
#: screen and the alert engine share): a machine can stop reporting for a
#: minute through a network blip that heals itself, and a reinstall is a heavy,
#: disruptive repair that must not churn on one. So the repair waits three
#: reporting windows - an agent that has posted nothing for this long is
#: failing every POST (a stale ingest key, or a CA the live process never
#: reloaded) and only a full re-render of its config heals it.
REINSTALL_AFTER_SILENT_SECONDS = 3 * REPORTING_WINDOW_SECONDS
SAMPLE_FRESHNESS_WINDOW_SECONDS = REINSTALL_AFTER_SILENT_SECONDS

#: After one reinstall for drift (an installed bundle or unit that differs from
#: what this controller ships), how long the SAME difference is left alone
#: before another reinstall is tried. Four automatic-repair intervals: the
#: repair runs every `DEFAULT_INTERVAL_SECONDS`, so a window of exactly one
#: interval would let the very next tick retry and turn a difference a
#: reinstall cannot fix into a reinstall every fifteen minutes - the loop this
#: guard exists to stop (VD-147, N-B1). A new shipped bundle, a Recheck, or the
#: window passing clears it.
DRIFT_RETRY_WINDOW_SECONDS = 4 * DEFAULT_INTERVAL_SECONDS

#: How many reinstalls one unchanged difference gets before Vaelor stops trying
#: on its own (review S-2). Each retry waits twice as long as the one before
#: (one window, then two); after the last, only a Recheck or a newer bundle on
#: the controller tries again. A difference a reinstall cannot fix would
#: otherwise re-mint the reporting key and restart the agent every hour forever.
DRIFT_MAX_ATTEMPTS = 3

#: How many times one stopped sampler is started again before Vaelor stops and
#: says so (pass-2 review SC-D): a sampler that keeps dying is not fixed by a
#: restart every fifteen minutes, and each one was an audit row.
SAMPLER_RESTART_LIMIT = 3

#: What the owner reads when the sampler could not be landed intact. Plain
#: cause, no command output: the input stream may have carried a password.
SAMPLER_LANDING_FAILED = (
    "The GPU sampler did not arrive on the machine as it was sent, so it was "
    "not installed. The rest of the telemetry agent is unaffected."
)
#: The same for the sampler's service file, which is removed rather than left.
SAMPLER_UNIT_LANDING_FAILED = (
    "The GPU sampler's service file did not arrive on the machine as it was "
    "sent, so it was removed and the sampler was not started. The rest of the "
    "telemetry agent is unaffected."
)

#: The program that lands the sampler's bundle on the worker: base64 on stdin,
#: bytes into a fresh ``0600`` file that must not already exist and is never
#: reached through a symlink. It runs under the worker's ``python3`` as root, in
#: the root-owned library directory - the bundle is never staged under ``/tmp``
#: (VD-147). The emitter, the Telegraf tarball and the controller's CA are
#: staged in a private random directory (`SshTransport.stage_private_artifact`,
#: review A10), never at a ``/tmp`` name another account could compute.
#:
#: **It checks what arrived before writing a byte** (review B3): the input must
#: be base64 and nothing else, and decode to exactly the digest the controller
#: passes as the second argument. Anything else - a sudo password sharing the
#: input stream on a worker whose sudo did not ask for it, a truncated stream -
#: writes nothing and exits non-zero with :data:`LANDING_REFUSED` on stderr,
#: never the input itself.
LANDING_REFUSED = "vaelor-landing: the bundle did not arrive as it was sent"
_LAND_PROGRAM = (
    "import base64,binascii,hashlib,os,sys\n"
    "p,want=sys.argv[1],sys.argv[2]\n"
    "raw=''.join(sys.stdin.read().split())\n"
    "try:\n"
    "    data=base64.b64decode(raw,validate=True)\n"
    "except (binascii.Error,ValueError):\n"
    "    data=None\n"
    "if data is None or hashlib.sha256(data).hexdigest()!=want:\n"
    "    sys.stderr.write('" + LANDING_REFUSED + "\\n')\n"
    "    sys.exit(3)\n"
    "os.path.lexists(p) and os.unlink(p)\n"
    "f=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)\n"
    "with os.fdopen(f,'wb') as handle:\n"
    "    handle.write(data)\n"
)


def _drift_detail(drift: Optional[Dict[str, str]]) -> Dict[str, str]:
    """A pending drift finding carried beside another repair's action, so it is not lost."""
    if not drift:
        return {}
    return {"drift": drift["action"], "differences": drift.get("differences", "")}


class CannotCheck(RuntimeError):
    """A file on the worker could not be read, which is not the same as "differs" (review S-3)."""


def _absent(error: BaseException) -> bool:
    """Whether a failed read means the file is not there (as opposed to unreadable)."""
    return "no such file" in str(error).lower()

#: The Telegraf resolver's return: local tarball path, its pinned sha256, and
#: the binary's path inside the archive.
TelegrafArtifact = Tuple[str, str, str]


def _local_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_telegraf_resolver(architecture: str) -> TelegrafArtifact:
    """Locate the staged, pinned Telegraf tarball for an architecture.

    Reads the arch's artifact facts from `worker_telemetry_config` and expects
    the tarball to have been staged under :data:`TELEGRAF_STAGING_DIR`. The
    pinned sha256 travels with it so :meth:`WorkerTelemetryRuntime.install`
    verifies the local file before it ever leaves the controller.
    """
    artifact = cfg.telegraf_artifact(architecture)
    source = str(Path(TELEGRAF_STAGING_DIR) / artifact["filename"])
    return source, artifact["sha256"], artifact["member"]


class WorkerTelemetryRuntime:
    """Ship and supervise the worker telemetry agent over a pinned transport."""

    def __init__(
        self,
        telegraf_resolver: Callable[[str], TelegrafArtifact] = default_telegraf_resolver,
    ):
        self._telegraf_resolver = telegraf_resolver

    # -- reads ------------------------------------------------------------
    @staticmethod
    def _remote_sha(transport, path: str) -> str:
        """The sha256 of a file on the worker, or ``""`` when it is absent."""
        try:
            return str(transport.run(["sha256sum", path], sudo=True)).split()[0]
        except (SshTransportError, IndexError):
            return ""

    @staticmethod
    def _installed_sha(transport, path: str) -> str:
        """The sha256 of an installed file, ``""`` when it is absent.

        Raises :class:`CannotCheck` when the worker could not be asked (a
        timeout, a refused command): that is not a difference.
        """
        try:
            return str(transport.run(["sha256sum", path], sudo=True)).split()[0]
        except IndexError:
            return ""
        except SshTransportError as error:
            if _absent(error):
                return ""
            raise CannotCheck(str(error)) from error

    @staticmethod
    def _installed_text(transport, path: str) -> str:
        """An installed text file, ``""`` when absent; :class:`CannotCheck` when unreadable."""
        try:
            return str(transport.run(["cat", path], sudo=True))
        except SshTransportError as error:
            if _absent(error):
                return ""
            raise CannotCheck(str(error)) from error

    @staticmethod
    def _unit_properties(transport, properties: str, unit: str = cfg.UNIT_NAME) -> Dict[str, str]:
        """One ``systemctl show`` of a unit (the agent's by default), parsed.

        ``systemctl show`` exits 0 even for a unit systemd has never heard of, so
        this raises only when the worker itself cannot be asked — which is the
        caller's real failure, not a missing unit.
        """
        raw = transport.run(
            ["systemctl", "show", unit, "--property={}".format(properties)]
        )
        parsed: Dict[str, str] = {}
        for line in str(raw).splitlines():
            key, separator, value = line.partition("=")
            if separator:
                parsed[key.strip()] = value.strip()
        return parsed

    # -- install ----------------------------------------------------------
    def install(
        self,
        transport,
        *,
        node_id: str,
        architecture: str,
        ingest_url: str,
        ingest_key: str,
        controller_ca_source: Optional[str] = None,
        gpu_sampler: bool = False,
    ) -> Dict[str, str]:
        """Ship, verify, place and enable the whole agent on one worker.

        Returns the paths written. Every artifact is content-verified before it
        is placed, and a re-run is idempotent: an artifact whose worker copy
        already matches is not re-shipped (content-hashed skip), the config is
        re-rendered (it is cheap and carries the possibly-rotated key), and the
        unit is re-enabled. ``gpu_sampler`` also installs the non-root GPU
        vendor sampler; the caller sets it only for a worker with an AMD
        integrated GPU (capability discovery, VD-147).

        **The sampler goes last, and its failure is not the agent's** (review
        B3). It used to be installed before the config and the unit, so a
        sampler that could not be landed aborted the whole telemetry install.
        Now the agent is installed and running first; a sampler that fails is
        reported in ``sampler_error`` (a plain sentence) and the agent reports
        "the sampler is missing" until the next repair pass tries again.

        **Nothing to pin on an installed controller is a refusal** (VD-212):
        with no ``controller_ca_source`` the agent would post without
        certificate checks, so on a controller whose household authority or
        TLS folder is installed this raises ``ValueError`` with the reason
        (`cluster_worker_profile.NOTHING_TO_PIN`) before anything is touched.
        Only a development box keeps the explicit insecure opt-in.
        """
        if not controller_ca_source or not Path(controller_ca_source).is_file():
            from .cluster_worker_profile import controller_trust

            refusal = controller_trust().refusal
            if refusal:
                raise ValueError(refusal)
        self._ensure_dirs(transport)
        self._ship_telegraf(transport, architecture)
        self._ship_emitter(transport)
        tls_ca_path = self._ship_controller_ca(transport, controller_ca_source)
        self._write_config(
            transport, node_id=node_id, ingest_url=ingest_url,
            ingest_key=ingest_key, tls_ca_path=tls_ca_path,
        )
        self._write_unit(transport)
        result = {
            "binary": cfg.TELEGRAF_BINARY_PATH,
            "emitter": cfg.EMITTER_PATH,
            "config": cfg.CONFIG_PATH,
            "unit": cfg.UNIT_NAME,
        }
        if gpu_sampler:
            try:
                self._install_sampler(transport)
            except SshTransportError as error:
                result["sampler_error"] = str(error)
        return result

    @staticmethod
    def _ensure_dirs(transport) -> None:
        for directory in (cfg.TELEGRAF_LIB_DIR, cfg.CONFIG_DIR):
            transport.run(["install", "-d", "-m", "0755", directory], sudo=True)

    def _ship_telegraf(self, transport, architecture: str) -> None:
        """Ship the pinned Telegraf binary: verify local, upload, verify remote, extract.

        The binary is never installed unless its tarball's sha256 matches the
        pinned constant on BOTH sides — the controller-local copy before it
        leaves, and the staged copy on the worker before it is extracted. An
        unpinned placeholder hash therefore refuses to install rather than
        shipping an unverified build (the safe default until a real hash is
        pinned).
        """
        source, expected_sha, member = self._telegraf_resolver(architecture)
        if not Path(source).is_file():
            raise SshTransportError(
                "The pinned Telegraf artifact is not staged on the controller."
            )
        local_sha = _local_sha256(source)
        if local_sha != expected_sha:
            raise SshTransportError(
                "The staged Telegraf tarball does not match its pinned checksum."
            )
        # Review A10: staged privately and uploaded every time - a staged copy
        # left at a digest-named /tmp path was the reuse an attacker could plant.
        staged = transport.stage_private_artifact(source, "telegraf.tar.gz")
        try:
            if self._remote_sha(transport, staged) != expected_sha:
                raise SshTransportError(
                    "The Telegraf tarball on the worker failed checksum verification."
                )
            # Extract just the binary, stripping the archive's leading
            # directories so it lands at the fixed binary path.
            strip = str(member.count("/"))
            transport.run(
                ["tar", "-xzf", staged, "-C", cfg.TELEGRAF_LIB_DIR,
                 "--strip-components", strip, member],
                sudo=True,
            )
            # Then fix its mode and owner - inside the try, so nothing the
            # clean-up below does can leave the binary as tar wrote it.
            transport.run(["chmod", "0755", cfg.TELEGRAF_BINARY_PATH], sudo=True)
            transport.run(["chown", "root:root", cfg.TELEGRAF_BINARY_PATH], sudo=True)
        finally:
            transport.discard_private_artifact(staged)

    def _ship_emitter(self, transport) -> None:
        """Ship the emitter zipapp, skipping the upload when it already matches.

        The installed ``.pyz`` IS the bytes the controller would ship, so its
        digest equals :func:`bundle_digest`; a worker already holding the current
        bundle is left untouched (content-hashed skip). Otherwise the bytes are
        written to a private temp file, uploaded into a private staging
        directory on the worker (review A10), checksum-verified there, and
        installed ``0755``.
        """
        digest = bundle_digest()
        if self._remote_sha(transport, cfg.EMITTER_PATH) == digest:
            return
        payload = build_zipapp_bytes()
        with tempfile.NamedTemporaryFile(suffix=".pyz", delete=False) as handle:
            handle.write(payload)
            local = handle.name
        try:
            staged = transport.stage_private_artifact(local, "vaelor-telemetry-emitter.pyz")
        finally:
            Path(local).unlink(missing_ok=True)
        try:
            if self._remote_sha(transport, staged) != digest:
                raise SshTransportError(
                    "The telemetry emitter on the worker failed checksum verification."
                )
            transport.run(
                ["install", "-m", "0755", staged, cfg.EMITTER_PATH], sudo=True
            )
        finally:
            transport.discard_private_artifact(staged)
        transport.run(["chown", "root:root", cfg.EMITTER_PATH], sudo=True)

    def _install_sampler(self, transport) -> None:
        """Land, verify and start the GPU vendor sampler (VD-147).

        The bundle is streamed on stdin into a fresh ``0600`` file in the
        root-owned library directory (:data:`_LAND_PROGRAM`), its digest is
        checked there, it is installed ``0755``, and the digest of the
        INSTALLED file is checked again - so what runs is what was shipped. A
        worker already holding the current bundle is not re-shipped. The unit
        text is always rewritten and the service restarted, so a hand-edited
        unit is put back.
        """
        digest = sampler_bundle_digest()
        if self._remote_sha(transport, cfg.SAMPLER_PATH) != digest:
            payload = base64.b64encode(build_sampler_bytes()).decode("ascii")
            try:
                transport.run(
                    ["python3", "-c", _LAND_PROGRAM, cfg.SAMPLER_STAGING_PATH, digest],
                    sudo=True, stdin_text=payload,
                )
            except SshTransportError as error:
                transport.run(["rm", "-f", cfg.SAMPLER_STAGING_PATH], sudo=True)
                # The landing program's own refusal names no cause beyond
                # "altered on the way"; anything else is the command's error.
                raise SshTransportError(
                    SAMPLER_LANDING_FAILED if LANDING_REFUSED in str(error) else str(error)
                ) from error
            staged_ok = self._remote_sha(transport, cfg.SAMPLER_STAGING_PATH) == digest
            if staged_ok:
                transport.run(
                    ["install", "-m", "0755", cfg.SAMPLER_STAGING_PATH, cfg.SAMPLER_PATH],
                    sudo=True,
                )
                transport.run(["chown", "root:root", cfg.SAMPLER_PATH], sudo=True)
            transport.run(["rm", "-f", cfg.SAMPLER_STAGING_PATH], sudo=True)
            if not staged_ok or self._remote_sha(transport, cfg.SAMPLER_PATH) != digest:
                raise SshTransportError(SAMPLER_LANDING_FAILED)
        self._write_verified_unit(transport, cfg.SAMPLER_UNIT_PATH, cfg.render_sampler_unit())
        transport.run(["systemctl", "daemon-reload"], sudo=True)
        transport.run(["systemctl", "enable", cfg.SAMPLER_UNIT_NAME], sudo=True)
        transport.run(["systemctl", "restart", cfg.SAMPLER_UNIT_NAME], sudo=True)

    def _write_verified_unit(self, transport, path: str, text: str) -> None:
        """Write a unit file with ``tee`` and read it back; remove it if it differs.

        ``tee`` writes whatever arrives on its input. On a worker whose sudo
        does not ask for a password while the transport sends one anyway, that
        password would become the file's first line (review B3). What landed is
        read back and compared; a file that is not what was sent is removed,
        never left in place, and the cause is reported plainly - without the
        file's contents.

        The transport no longer sends a password to a sudo that does not ask
        for one (w3-sshfix, VD-164, merged), and the file is still created
        ``0600`` first and made ``0644`` only once it has been read back as
        exactly what was sent (pass-2 review SC-A): a write that lands wrong
        for any other reason is never readable by anyone else.
        """
        transport.run(["install", "-m", "0600", "/dev/null", path], sudo=True)
        transport.run(["tee", path], sudo=True, stdin_text=text)
        try:
            landed = self._installed_text(transport, path)
        except CannotCheck:
            landed = None
        if landed is None or landed.strip() != text.strip():
            transport.run(["rm", "-f", path], sudo=True)
            raise SshTransportError(SAMPLER_UNIT_LANDING_FAILED)
        transport.run(["chmod", "0644", path], sudo=True)

    def _remove_sampler(self, transport) -> None:
        """Stop, disable and delete the GPU sampler; a no-op where it never was.

        ``DynamicUser`` leaves no account behind and systemd removes the
        runtime directory when the service stops, so the unit file and the
        bundle are all there is to delete.
        """
        state = self._unit_properties(transport, _LOAD_STATE, cfg.SAMPLER_UNIT_NAME)
        if state.get(_LOAD_STATE) != _UNIT_NOT_FOUND:
            transport.run(
                ["systemctl", "disable", "--now", cfg.SAMPLER_UNIT_NAME], sudo=True
            )
        for path in (cfg.SAMPLER_UNIT_PATH, cfg.SAMPLER_PATH, cfg.SAMPLER_STAGING_PATH):
            transport.run(["rm", "-f", path], sudo=True)

    def _ship_controller_ca(
        self, transport, source: Optional[str]
    ) -> Optional[str]:
        """Place what the worker pins on the worker, if given.

        ``source`` is `cluster_worker_profile.controller_ca_source`: the
        household authority's trust bundle (one or more PEM certificates), or
        the served certificate before the authority has run (VD-212). Returns
        the on-worker CA path when it was shipped, else ``None`` - which, on a
        development box only, tells the config renderer to use the explicit
        insecure opt-in (:meth:`install` refuses it on an installed controller).
        """
        if not source or not Path(source).is_file():
            return None
        staged = transport.stage_private_artifact(source, "vaelor-controller-ca.pem")
        try:
            transport.run(
                ["install", "-m", "0644", staged, cfg.CONTROLLER_CA_PATH], sudo=True
            )
        finally:
            transport.discard_private_artifact(staged)
        return cfg.CONTROLLER_CA_PATH

    @staticmethod
    def _write_config(
        transport, *, node_id: str, ingest_url: str, ingest_key: str,
        tls_ca_path: Optional[str],
    ) -> None:
        """Land the ``0600`` config, key on stdin, never on argv.

        The file is pre-created empty at ``0600`` from ``/dev/null``, then the
        rendered config, with the key in it, is streamed into that existing
        file with ``tee`` (which does not change an existing file's mode), so
        the config is ``0600`` from its first byte. This mirrors
        `gpu_pool_runtime._write_gate_config`: ``install -m 0600 /dev/stdin
        <path>`` was replaced because it intermittently fails to open
        ``/dev/stdin`` over an SSH exec channel once the payload passes about a
        kilobyte (this config runs ~1.4 KB), aborting the whole install; ``tee``
        reads the inherited stdin descriptor directly and is immune to that
        size race. The key still travels only on ``stdin``, so it appears on no
        argv and in no process listing; ``tee``'s stdout echo is discarded.
        """
        config = cfg.render_config(
            node_id=node_id, ingest_url=ingest_url, ingest_key=ingest_key,
            tls_ca_path=tls_ca_path, allow_insecure_tls=tls_ca_path is None,
        )
        transport.run(
            ["install", "-m", "0600", "/dev/null", cfg.CONFIG_PATH], sudo=True
        )
        transport.run(["tee", cfg.CONFIG_PATH], sudo=True, stdin_text=config)

    @staticmethod
    def _write_unit(transport) -> None:
        """Land the unit, enable it, and (re)start it so a fresh CA takes effect.

        ``enable --now`` does NOT restart an already-active unit, so a re-pin
        that ships a new controller CA would leave the running Telegraf on the
        old CA it read once at startup (Go reads ``tls_ca`` only then). An
        explicit ``restart`` after ``enable`` starts a stopped unit and reloads a
        running one, so the CA and config just written always take effect. This
        runs only on install/re-pin, never per reconcile tick, so it is not a
        restart loop.
        """
        transport.run(
            ["tee", cfg.UNIT_PATH], sudo=True, stdin_text=cfg.render_unit()
        )
        transport.run(["systemctl", "daemon-reload"], sudo=True)
        transport.run(["systemctl", "enable", cfg.UNIT_NAME], sudo=True)
        transport.run(["systemctl", "restart", cfg.UNIT_NAME], sudo=True)

    # -- uninstall --------------------------------------------------------
    def uninstall(self, transport) -> None:
        """Disable and remove the agent, and every file it placed.

        Shaped like `gpu_pool_runtime.stop_unit`: systemd is asked first, so a
        unit it never knew about is a no-op rather than a failure, then the unit
        and its ``0600`` config (which carries the key) and the binary and
        emitter are removed. The controller-side ingest-key hash is cleared by
        the manager; this clears the copy on the worker.

        The household root comes out of the worker's system trust store too
        (VD-212), and the store is rebuilt without it. A worker with no
        ``update-ca-certificates`` (not Ubuntu or Debian) never had the root
        placed there, so its absence is not an error.
        """
        state = self._unit_properties(transport, _LOAD_STATE)
        if state.get(_LOAD_STATE) != _UNIT_NOT_FOUND:
            transport.run(["systemctl", "disable", "--now", cfg.UNIT_NAME], sudo=True)
        transport.run(["rm", "-f", cfg.UNIT_PATH], sudo=True)
        # The GPU sampler goes with the agent it feeds (VD-147).
        self._remove_sampler(transport)
        transport.run(["systemctl", "daemon-reload"], sudo=True)
        for path in (
            cfg.CONFIG_PATH, cfg.EMITTER_PATH, cfg.TELEGRAF_BINARY_PATH,
            cfg.CONTROLLER_CA_PATH, WORKER_OS_TRUST_FILE,
        ):
            transport.run(["rm", "-f", path], sudo=True)
        try:
            transport.run(["update-ca-certificates"], sudo=True)
        except SshTransportError as error:
            # sudo's own words when the tool is not installed.
            if "command not found" not in str(error).lower() and not _absent(error):
                raise

    # -- reconcile --------------------------------------------------------
    def reconcile(
        self, transport, *, controller_ca_source: Optional[str] = None,
        node_id: Optional[str] = None,
        last_sample_time: Optional[Callable[[str], Optional[int]]] = None,
        reprovision: Optional[Callable[[], object]] = None,
        now: Optional[float] = None,
        heal_absent: bool = False,
        clock_refused: Optional[Callable[[], bool]] = None,
        backlog_refused: Optional[Callable[[], bool]] = None,
        gpu_sampler: Optional[bool] = None,
        drift_guard: Optional[Dict[str, Dict[str, Any]]] = None,
        retry_drift: bool = False,
    ) -> Dict[str, str]:
        """Heal a down agent, a stale controller CA, OR a reporting-stale agent.

        Reads the unit's state. A loaded-but-inactive unit is brought back with
        ``enable --now``. An ACTIVE unit is checked two further ways, both of
        which leave a healthy agent untouched so this is not a per-tick restart
        loop:

        * ``_controller_ca_is_stale`` compares the on-worker CA against the cert
          the controller now serves; on a mismatch it re-ships the CA and
          restarts the unit (a re-pin the running Telegraf never reloaded).
        * a liveness check: when a ``last_sample_time`` accessor and a
          ``reprovision`` callback are wired, an active unit that has had no
          sample accepted within :data:`SAMPLE_FRESHNESS_WINDOW_SECONDS` is
          failing every POST on a layer the on-disk CA cannot reveal (a stale
          ingest key outlives a correct CA). That agent is REPROVISIONED (the
          reprovision callback re-drives the whole install path: re-mint key,
          re-render config, re-ship CA, restart), not reported healthy.

        The liveness branch is guarded against a reprovision loop: it fires only
        when the unit has already been active at least one freshness window
        (``_unit_active_seconds``), so an agent this reconcile has just restarted
        (which resets ``ActiveEnterTimestamp``) is left ``warming`` until it has
        had a window to post its first sample. A worker that cannot be reached
        raises through the transport, which the caller reports rather than
        treating as a reason to remove the node.

        Two cases the caller now decides for it (ACC-121, ACC-126):

        * ``heal_absent`` - the caller holds a key for this agent, so a unit
          systemd has never heard of was REMOVED since the owner installed it,
          and is reinstalled (``reinstalled-absent``). Without it an absent
          agent is reported and left alone, as before: a worker the owner never
          set up is not given one by a Recheck.
        * ``clock_refused`` - the controller's ingest route refused this
          agent's newest post because the worker's clock is outside the
          accepted window. A reinstall cannot move a clock, and doing one every
          15 minutes was a loop; the answer is ``clock-skew`` with no change.
        * ``backlog_refused`` - the newest posts are refused because the agent
          is resending readings it buffered while the controller was away, too
          old to accept. The clock is fine; a restart drops the backlog
          (``restarted-backlog``).

        **What is installed is compared with what this controller ships**
        (VD-147, N-B1), whenever a ``reprovision`` callback is wired: the
        emitter bundle's digest and the agent unit's text, and - when
        ``gpu_sampler`` says this worker should run one - the sampler bundle's
        digest, its unit's text, and that it is active. Any difference
        reinstalls once (``reprovisioned-drift``). This is what brings a worker
        up to a newer controller with nobody pressing Recheck, and what puts a
        hand-edited unit back. If the SAME difference is still there after that
        reinstall, within :data:`DRIFT_RETRY_WINDOW_SECONDS`, nothing is changed
        and the answer is ``drift-unresolved`` - not a reinstall every tick.
        ``drift_guard`` holds that memory per node (the caller owns the dict);
        ``retry_drift`` (a Recheck) tries again regardless.
        """
        state = self._unit_properties(
            transport, "{},{}".format(_LOAD_STATE, _ACTIVE_STATE)
        )
        load = state.get(_LOAD_STATE, "")
        active = state.get(_ACTIVE_STATE, "")
        if load == _UNIT_NOT_FOUND:
            if heal_absent and reprovision is not None:
                reprovision()
                return {
                    "action": "reinstalled-absent", "load_state": load,
                    "active_state": active,
                }
            return {"action": "absent", "load_state": load, "active_state": active}
        if active != "active":
            transport.run(["systemctl", "enable", "--now", cfg.UNIT_NAME], sudo=True)
            return {"action": "restarted", "load_state": load, "active_state": active}
        if self._controller_ca_is_stale(transport, controller_ca_source):
            self._ship_controller_ca(transport, controller_ca_source)
            transport.run(["systemctl", "restart", cfg.UNIT_NAME], sudo=True)
            return {"action": "reshipped-ca", "load_state": load, "active_state": active}
        if clock_refused is not None and clock_refused():
            return {"action": "clock-skew", "load_state": load, "active_state": active}
        drift = self._drift_action(
            transport, node_id, gpu_sampler, reprovision, drift_guard, retry_drift, now,
        )
        if drift is not None and drift["action"] == "reprovisioned-drift":
            # A whole reinstall happened: nothing else is left to heal.
            return {**drift, "load_state": load, "active_state": active}
        # A drift that is waiting, stopped or just restarted the sampler does
        # not stand in the way of the other repairs (review S-1); it is
        # reported only when nothing else acted.
        if backlog_refused is not None and backlog_refused():
            transport.run(["systemctl", "restart", cfg.UNIT_NAME], sudo=True)
            return {
                "action": "restarted-backlog", "load_state": load,
                "active_state": active, **_drift_detail(drift),
            }
        verdict = self._liveness_action(
            transport, node_id, last_sample_time, reprovision, now
        )
        if verdict is not None:
            return {"action": verdict, "load_state": load, "active_state": active, **_drift_detail(drift)}
        if drift is not None:
            return {**drift, "load_state": load, "active_state": active}
        return {"action": "healthy", "load_state": load, "active_state": active}

    def installed_differences(self, transport, gpu_sampler: Optional[bool]) -> List[str]:
        """What on the worker differs from what this controller ships, by name.

        Empty when the emitter bundle and the agent unit (and, for a worker
        that should run one, the sampler bundle, its unit and its running
        state) are exactly what would be installed now. Raises
        :class:`CannotCheck` when a file could not be read: unknown is not
        "differs" (review S-3).
        """
        found: List[str] = []
        if self._installed_sha(transport, cfg.EMITTER_PATH) != bundle_digest():
            found.append("emitter")
        if self._installed_text(transport, cfg.UNIT_PATH).strip() != cfg.render_unit().strip():
            found.append("agent-unit")
        if gpu_sampler:
            if self._installed_sha(transport, cfg.SAMPLER_PATH) != sampler_bundle_digest():
                found.append("sampler")
            installed_unit = self._installed_text(transport, cfg.SAMPLER_UNIT_PATH).strip()
            if installed_unit != cfg.render_sampler_unit().strip():
                found.append("sampler-unit")
            state = self._unit_properties(transport, _ACTIVE_STATE, cfg.SAMPLER_UNIT_NAME)
            if state.get(_ACTIVE_STATE) != "active":
                found.append("sampler-stopped")
        return found

    def _drift_action(
        self, transport, node_id, gpu_sampler, reprovision, guard, retry, now,
    ) -> Optional[Dict[str, str]]:
        """Repair what differs, or say why not; None when nothing differs or it cannot be checked.

        * a sampler that is only STOPPED is restarted, not reinstalled
          (``restarted-sampler``; review S-4) - a reinstall would re-mint the
          reporting key for a service that only needed starting;
        * anything else is reinstalled (``reprovisioned-drift``), and the SAME
          difference is tried again only after a window that doubles each time
          (:data:`DRIFT_RETRY_WINDOW_SECONDS`, then twice that), and not at all
          after :data:`DRIFT_MAX_ATTEMPTS` until a Recheck or a newer bundle
          (review S-2): ``drift-unresolved`` while waiting, ``drift-stopped``
          after giving up. ``newly`` marks the first pass that reports each,
          which is what gets audited (review S-5);
        * a file that could not be READ is not a difference: nothing is done
          this pass (review S-3).
        """
        if reprovision is None:
            return None
        try:
            differences = self.installed_differences(transport, gpu_sampler)
        except (CannotCheck, SshTransportError):
            return None
        key = str(node_id or "")
        if not differences:
            if guard is not None:
                guard.pop(key, None)
            return None
        if differences == ["sampler-stopped"]:
            return self._restart_sampler(transport, guard, key)
        # The shipped digests are part of the fingerprint, so a newer bundle on
        # the controller is a new difference and is tried straight away.
        fingerprint = "{}@{}:{}".format(
            ",".join(differences), bundle_digest()[:16], sampler_bundle_digest()[:16],
        )
        moment = time.time() if now is None else float(now)
        record = (guard or {}).get(key) or {}
        same = record.get("fingerprint") == fingerprint
        attempts = int(record.get("attempts") or 0) if same else 0
        window = DRIFT_RETRY_WINDOW_SECONDS * (2 ** max(0, attempts - 1))
        waiting = same and moment - float(record.get("last_attempt_at") or 0.0) < window
        if same and not retry and (attempts >= DRIFT_MAX_ATTEMPTS or waiting):
            action = "drift-stopped" if attempts >= DRIFT_MAX_ATTEMPTS else "drift-unresolved"
            newly = record.get("reported") != action
            record["reported"] = action
            return {
                "action": action, "differences": ",".join(differences),
                "error": str(record.get("error") or ""), "newly": "yes" if newly else "",
            }
        attempt = {
            "fingerprint": fingerprint, "attempts": attempts + 1,
            "last_attempt_at": moment, "error": "", "reported": "",
        }
        if guard is not None:
            guard[key] = attempt
        try:
            outcome = reprovision()
        except Exception as error:
            # Kept for the sentence the next pass shows; the failure itself is
            # raised to the caller, which reports and audits it.
            attempt["error"] = " ".join(str(error).split())[:200]
            raise
        if isinstance(outcome, dict) and outcome.get("sampler_error"):
            attempt["error"] = str(outcome["sampler_error"])[:200]
        return {"action": "reprovisioned-drift", "differences": ",".join(differences)}

    def _restart_sampler(self, transport, guard, key: str) -> Dict[str, str]:
        """Start a stopped sampler, at most :data:`SAMPLER_RESTART_LIMIT` times, then say so once."""
        fingerprint = "sampler-stopped@{}".format(sampler_bundle_digest()[:16])
        record = (guard or {}).get(key) or {}
        restarts = int(record.get("attempts") or 0) if record.get("fingerprint") == fingerprint else 0
        if restarts >= SAMPLER_RESTART_LIMIT:
            newly = record.get("reported") != "sampler-restarts-stopped"
            record["reported"] = "sampler-restarts-stopped"
            return {"action": "sampler-restarts-stopped", "differences": "sampler-stopped",
                    "error": "", "newly": "yes" if newly else ""}
        if guard is not None:
            guard[key] = {"fingerprint": fingerprint, "attempts": restarts + 1, "last_attempt_at": time.time(),
                          "error": "", "reported": ""}
        transport.run(["systemctl", "restart", cfg.SAMPLER_UNIT_NAME], sudo=True)
        return {"action": "restarted-sampler", "differences": "sampler-stopped"}

    def _controller_ca_is_stale(
        self, transport, source: Optional[str]
    ) -> bool:
        """True when the worker's on-disk controller CA is not the served cert.

        Compares the sha256 of the CA the runtime would ship (the controller's
        current serving cert, ``VAELOR_TLS_CERT``) against the sha256 of
        ``/etc/vaelor/controller-ca.pem`` on the worker. With no source cert to
        compare - none configured or unreadable - it reports not-stale, so a
        reconcile never restarts an agent it cannot prove is out of date. This is
        the minimal safe check: it detects a re-pinned CA the running agent never
        reloaded (the live stale-CA incident) without reaching into the agent's
        POST results.
        """
        if not source or not Path(source).is_file():
            return False
        return self._remote_sha(transport, cfg.CONTROLLER_CA_PATH) != _local_sha256(source)

    def _liveness_action(
        self, transport, node_id, last_sample_time, reprovision, now,
    ) -> Optional[str]:
        """The action an active unit reporting-liveness calls for, or None.

        Returns None (leave the agent healthy) whenever liveness cannot be
        judged or acted on: no accessor, no reprovision callback, a store that
        cannot answer, or a sample inside the window. An agent is never churned
        on a signal that could not be read. Only a definitely
        active-but-not-reporting agent that has been up at least one window
        returns ``reprovisioned-stale`` (after driving the reprovision); one
        that has not yet been up a window returns ``warming``.
        """
        if node_id is None or last_sample_time is None or reprovision is None:
            return None
        try:
            last = last_sample_time(node_id)
        except (RuntimeError, LookupError, OSError, ValueError, TypeError):
            return None
        current = time.time() if now is None else now
        window = SAMPLE_FRESHNESS_WINDOW_SECONDS
        if last is not None and current - float(last) <= window:
            return None
        active_seconds = self._unit_active_seconds(transport)
        if active_seconds is None or active_seconds < window:
            return "warming"
        reprovision()
        return "reprovisioned-stale"

    def _unit_active_seconds(self, transport) -> Optional[float]:
        """How long the agent unit has been active, in seconds, or None.

        Computed on the worker, so no controller/worker time skew enters the
        loop guard: ``/proc/uptime`` (seconds since boot) minus
        ``ActiveEnterTimestampMonotonic`` (microseconds since boot). The two
        are not quite the same clock across a suspend -- uptime's span counts
        suspended time and CLOCK_MONOTONIC does not -- so a suspend/resume can
        make this over-estimate the active span. That is harmless here: the
        supported appliances do not suspend, and an over-estimate only lets
        the loop guard clear sooner, never keeps a genuinely stale agent from
        being reprovisioned. None when either read is missing or unparseable,
        which the caller reads as cannot-prove-long-running and so declines
        to reprovision.
        """
        props = self._unit_properties(transport, "ActiveEnterTimestampMonotonic")
        raw = props.get("ActiveEnterTimestampMonotonic", "")
        try:
            entered_us = int(raw)
        except (TypeError, ValueError):
            return None
        if entered_us <= 0:
            return None
        try:
            uptime = float(str(transport.run(["cat", "/proc/uptime"])).split()[0])
        except (SshTransportError, IndexError, ValueError):
            return None
        return uptime - entered_us / 1_000_000.0
