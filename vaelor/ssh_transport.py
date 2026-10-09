"""Fingerprint-pinned SSH transport with a fixed cluster command surface."""

from __future__ import annotations

import base64
import errno
import hashlib
import ipaddress
import json
import os
import re
import shlex
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from . import cluster_link, gpu_memory_pool, gpu_node_facts, ssh_sudo_stream
from .credential_use import note_credential_use
from .platforms import remote_machine_class
from .platforms.kfd_topology import GFX_READER_SOURCE
from .platforms.accelerators import VENDORS, unified_memory_verdict

#: How long `SshTransport._cluster_address` waits for a hostname to resolve.
#: `socket.gethostbyname` takes NO timeout argument - it blocks inside the C
#: library's resolver, which retries a silent nameserver for as long as
#: ``/etc/resolv.conf`` tells it to (five seconds times two attempts times every
#: listed server, by default). That is once per node on every enrolment AND
#: every refresh, so a nameserver that stopped answering would stall the whole
#: fleet's inventory rather than one field of it. Five seconds is longer than a
#: healthy LAN or cache answer by orders of magnitude and short enough that a
#: dead resolver costs a refresh nothing it cannot afford.
_RESOLVE_TIMEOUT_SECONDS = 5

#: The overall wait for a command given no ``timeout`` of its own. Long enough
#: for any reviewed command that sets none (a ``tar`` of a runtime archive, a
#: ``systemctl`` start); short enough that a PAM prompt nobody can answer ends.
_RUN_DEADLINE_SECONDS = 600


class SshTransportError(RuntimeError):
    """The one transport error type, whatever carries the command.

    The name is historical - it was written when SSH was the only way a
    cluster command left this process. Since the controller can serve its own
    GPU, `vaelor.bridge_transport.BridgeTransport` carries the identical
    command surface over the root hardware bridge and raises THIS class, so
    every ``except SshTransportError`` in the runtime and the operations layer
    keeps working without knowing which transport it holds. Do not add a second
    error type for the bridge path: the whole point is that the callers cannot
    tell them apart.
    """


#: The first-argv tokens a reviewed cluster command may use. ONE home, because
#: two boundaries now enforce it: `SshTransport.run` on the way out to an
#: enrolled worker, and the root hardware bridge's ``run_argv`` handler on the
#: way into the controller's own shell (`vaelor.hardware_bridge`), with
#: `vaelor.bridge_transport.BridgeTransport` refusing client-side first so a bad
#: argv never crosses the socket. A copy per boundary would drift, and a widened
#: copy at the ROOT boundary would be a privilege escalation rather than a bug.
REMOTE_COMMAND_ALLOWLIST = frozenset({
    "cat", "docker", "getconf", "id", "journalctl", "nft", "nproc", "stat", "systemctl",
    "uname", "apt-get", "install", "mkdir", "tee", "tar",
    "sha256sum", "chmod", "python3", "rm", "useradd", "chown",
    # The boot-image rebuild a GPU memory pool change needs (VD-161); the
    # controller's bridge has its own verb for it and does not take this.
    "update-initramfs",
    # One shape only - `mv -f PATH.vaelor-new-<16 hex> PATH`, a checked file renamed
    # onto its own target (`ssh_sudo_stream.check_rename`, VD-164).
    "mv",
})


#: What `SshTransport.run` raises with when a command ended non-zero and
#: wrote no error text of its own - which includes a connection that closed
#: under it. Named so a caller can tell "the machine refused and said why"
#: from "the channel went away" (`gpu_memory_pool_nodes.reboot_worker`).
COMMAND_ENDED_WITHOUT_REASON = "The remote command failed."


#: The socket errnos that mean the network or the far host, by name, so a
#: platform without one simply leaves it out.
_LINK_ERRNOS = frozenset(
    getattr(errno, name) for name in (
        "ENETUNREACH", "ENETDOWN", "ENETRESET", "EHOSTUNREACH", "EHOSTDOWN", "ECONNREFUSED",
        "ECONNRESET", "ECONNABORTED", "ETIMEDOUT", "EPIPE",
    ) if hasattr(errno, name)
)


class MachineUnreachable(ConnectionError):
    """A socket error from opening the SSH connection: the link, not this controller."""


def channel_drops() -> tuple:
    """Every way the SSH channel itself can die under a command (review S4).

    paramiko's own errors, an end of stream, and the socket errors of a link
    that went away. NOT every ``OSError``: a file this controller could not
    read, or a full disk, is not a dropped connection, and calling it one
    records the machine as unreachable when it is not (LESSONS 8).
    """
    import socket

    drops: tuple = (EOFError, ConnectionError, TimeoutError, socket.gaierror)
    try:
        import paramiko
    except ImportError:
        return drops
    return drops + (paramiko.SSHException, paramiko.ssh_exception.NoValidConnectionsError)


def machine_failures() -> tuple:
    """Every way the machine, its stored sign-in or the connection to it can fail.

    Anything else raised while asking a machine - this controller's own file
    reads among it - is this controller's own fault, and is logged as one
    rather than worded as the machine failing.
    """
    from .credential_broker_client import CredentialError

    return (SshTransportError, CredentialError) + channel_drops()


def _paramiko():
    try:
        import paramiko
    except ImportError as error:
        raise SshTransportError("Install the Vaelor SSH support package first.") from error
    return paramiko


def host_key_fingerprint(host: str, port: int = 22, timeout: int = 8) -> Dict[str, str]:
    """Read an SSH host key without authenticating."""
    paramiko = _paramiko()
    connection = socket.create_connection((host, int(port)), timeout=timeout)
    transport = paramiko.Transport(connection)
    try:
        transport.start_client(timeout=timeout)
        key = transport.get_remote_server_key()
        digest = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode("ascii")
        return {
            "algorithm": key.get_name(),
            "fingerprint": f"SHA256:{digest.rstrip('=')}",
        }
    except Exception as error:
        raise SshTransportError("The SSH host key could not be inspected.") from error
    finally:
        transport.close()
        connection.close()


#: The two ``nft`` shapes a worker may be asked for (ACC-163): load one split's
#: staged ruleset, or delete one split's table. Spelled here, like
#: `gpu_ray_plane.RAY_TOKEN_ROOT`, because this module may not import it; a
#: test ties the spellings.
_NFT_RULESET = r"/etc/vaelor-ray/[a-z0-9][a-z0-9-]{0,38}\.nft"
_NFT_TABLE = r"vaelor_ray_[a-z0-9][a-z0-9_]{0,38}"
#: A split's own slice (`gpu_ray_plane.slice_name`, ACC-187): ``systemctl``
#: may start or stop one, and do nothing else to any slice.
_RAY_SLICE = r"vaelor-ray-[a-z0-9][a-z0-9_]{0,38}\.slice"
#: Glob characters and whitespace, which no systemctl argument Vaelor sends carries.
_SYSTEMCTL_PATTERN = re.compile(r"[*?\[\]\s]")


def remote_command_refusal(args):
    """Why ``args`` may not run on a worker, or ``None`` when it may.

    The first word must be on :data:`REMOTE_COMMAND_ALLOWLIST`; ``nft``, which
    can rewrite a machine's whole firewall, is further held to its two exact
    shapes.
    """
    if not args or args[0] not in REMOTE_COMMAND_ALLOWLIST:
        return "The requested remote command is not permitted."
    if args[0] == "nft":
        load = len(args) == 3 and args[1] == "-f" and re.fullmatch(_NFT_RULESET, args[2])
        drop = (len(args) == 5 and args[1:4] == ["delete", "table", "inet"]
                and re.fullmatch(_NFT_TABLE, args[4]))
        if not (load or drop):
            return "Only a split's own nftables table may be loaded or removed."
    if args[0] == "systemctl" and any(
        "slice" in str(arg) or _SYSTEMCTL_PATTERN.search(str(arg)) for arg in args[1:]
    ):
        # systemctl takes unit-name globs, so a slice is named whole or not at
        # all (review 1): exactly start/stop of a split's own slice.
        if not (len(args) == 3 and args[1] in ("start", "stop")
                and re.fullmatch(_RAY_SLICE, str(args[2]))):
            return "Only a split's own slice may be started or stopped."
    return None


class SshTransport:
    """Execute only reviewed argv-style commands after host-key verification."""

    def __init__(
        self,
        profile: Dict[str, Any],
        timeout: int = 20,
        resolve_host: Optional[Callable[[str], str]] = None,
    ):
        self.profile = profile
        self.timeout = max(5, min(int(timeout), 120))
        # A node enrolled by hostname still has to record the IPv4 the cluster
        # reaches it on, so `_cluster_address` resolves one. Injectable purely
        # so the probe's cluster-interface fact is testable without DNS.
        self._resolve_host = resolve_host or socket.gethostbyname

    def _connect(self):
        paramiko = _paramiko()
        expected = self.profile["host_key_fingerprint"]

        class PinnedHostKeyPolicy(paramiko.MissingHostKeyPolicy):
            def missing_host_key(self, client, hostname, key):
                digest = base64.b64encode(
                    hashlib.sha256(key.asbytes()).digest()
                ).decode("ascii").rstrip("=")
                if f"SHA256:{digest}" != expected:
                    raise SshTransportError(
                        "The SSH host key changed. Remove and re-enroll this node before connecting."
                    )
                client.get_host_keys().add(hostname, key.get_name(), key)

        client = paramiko.SSHClient()
        try:
            client.set_missing_host_key_policy(PinnedHostKeyPolicy())
            client.connect(
                hostname=self.profile["host"],
                port=self.profile["port"],
                username=self.profile["username"],
                password=self.profile["password"],
                timeout=self.timeout,
                auth_timeout=self.timeout,
                look_for_keys=False,
                allow_agent=False,
            )
        except OSError as error:
            client.close()
            if isinstance(error, channel_drops()) or error.errno not in _LINK_ERRNOS:
                # Only a network errno is the link. A refused sign-in, a
                # missing key file or a local descriptor limit is not, and
                # is never said as "unreachable" (merged-tree review).
                raise
            # Review round 2: paramiko re-raises some socket errors raw - a
            # pulled cable is "Network is unreachable", a sleeping host
            # "Host is down". Raised by connect() they can only be the link,
            # so they are said as the link, never as this controller's fault.
            raise MachineUnreachable(str(error) or type(error).__name__) from error
        except BaseException:
            # A refused password or a changed host key raises from connect():
            # the half-open client (socket and transport thread) is closed here
            # rather than left for the caller, which never received it.
            client.close()
            raise
        return client

    def verify_login(self) -> None:
        """Sign in against the pinned host key and disconnect, running nothing.

        The credential Test button's SSH check
        (`vaelor.credential_provider_probe.probe_ssh_sign_in`): the same
        connection every cluster command opens, so a pass means a command would
        have been able to sign in, and a changed host key fails exactly as it
        does there.
        """
        client = self._connect()
        try:
            transport = client.get_transport()
            if transport is None or not transport.is_authenticated():
                raise SshTransportError("The SSH sign-in did not complete.")
        finally:
            client.close()

    def _session(self):
        """A signed-in connection for a command: the node credential's USE (ACC-107).

        Every command path opens its connection here, so the use is recorded
        where the credential signs in to do work. A sign-in check that runs
        nothing (the credential Test) calls `_connect` directly and is not a
        use. The lease carries its credential id; a profile built by hand does
        not, and is simply not reported.
        """
        client = self._connect()
        note_credential_use(self.profile)
        return client

    @staticmethod
    def _command(arguments: Iterable[str]) -> str:
        return " ".join(shlex.quote(str(argument)) for argument in arguments)

    def _exec(
        self,
        client,
        arguments: List[str],
        *,
        sudo: bool,
        password: Optional[str],
        stdin_text: str = "",
        timeout: Optional[int] = None,
        write: bool = False,
        c_locale: bool = False,
    ) -> str:
        """Run one argv on an open connection; stdout, or a plain error.

        Elevation is `ssh_sudo_stream.elevated`'s (VD-164). The wait has an
        overall deadline - the call's ``timeout``, else
        `_RUN_DEADLINE_SECONDS` - because ``recv_exit_status`` waits forever
        and a PAM module that waits (a fingerprint reader) would otherwise hang
        the call. Input that could not be delivered in full fails a command
        that reported success, unless it is a staged ``write``, whose digest
        check already decides whether it landed.
        """
        command = self._command(arguments)
        input_text = stdin_text
        if sudo:
            command = ssh_sudo_stream.elevated(command, password, c_locale)
            if password is not None:
                input_text = f"{password}\n{stdin_text}"
        elif c_locale:
            command = f"LC_ALL=C {command}"
        stdin, stdout, stderr = client.exec_command(
            command, timeout=timeout or self.timeout
        )
        undelivered = False
        try:
            if input_text:
                stdin.write(input_text)
                stdin.flush()
            stdin.channel.shutdown_write()
        except (OSError, EOFError):
            # The far side stopped reading (sudo -n refused before the command
            # started, or the command exited early). The status decides.
            undelivered = True
        channel = stdout.channel
        limit = timeout or _RUN_DEADLINE_SECONDS
        deadline = time.monotonic() + limit
        while not channel.exit_status_ready():
            if time.monotonic() >= deadline:
                channel.close()
                raise SshTransportError(
                    f"The command did not finish within {limit} seconds, so "
                    "Vaelor stopped waiting. It may still be running on the "
                    "machine (apt holding the package lock, for example), so "
                    "an immediate retry may fail on that lock. If this "
                    "machine's sudo waits for something such as a "
                    "fingerprint, let the account use its password for sudo."
                )
            time.sleep(0.05)
        status = channel.recv_exit_status()
        output = stdout.read(1024 * 1024).decode("utf-8", "replace").strip()
        if status == 0:
            if undelivered and input_text and not write:
                raise SshTransportError(
                    "The command finished, but its input could not be "
                    "delivered in full, so its result cannot be trusted."
                )
            return output
        error = stderr.read(16 * 1024).decode("utf-8", "replace")
        raise ssh_sudo_stream.failure(error, password)

    def run(
        self,
        arguments: Iterable[str],
        *,
        sudo: bool = False,
        stdin_text: str = "",
        timeout: Optional[int] = None,
    ) -> str:
        """Stdout of one allowlisted argv, run on the worker.

        ``tee PATH`` is not run as given: `ssh_sudo_stream.land_file` writes
        beside the target, checks the digest and renames into place, and
        returns ``""``. ``mv`` is accepted only in the one shape that rename
        uses (`ssh_sudo_stream.check_rename`).
        """
        args = [str(item) for item in arguments]
        refusal = remote_command_refusal(args)
        if refusal is not None:
            raise SshTransportError(refusal)
        target = ssh_sudo_stream.tee_target(args)
        ssh_sudo_stream.check_rename(args)
        password = ssh_sudo_stream.sudo_password(self.profile) if sudo else None
        client = self._session()
        try:
            def execute(argv, stdin_text="", write=False, c_locale=False):
                return self._exec(
                    client, argv, sudo=sudo, password=password,
                    stdin_text=stdin_text, timeout=timeout, write=write,
                    c_locale=c_locale,
                )

            if target is not None:
                return ssh_sudo_stream.land_file(execute, target, stdin_text)
            return execute(args, stdin_text)
        finally:
            client.close()

    # -- Private staging (review A10, VD-172): see `vaelor.ssh_private_staging`.

    def stage_private_artifact(self, source: str, filename: str, **bounds: Any) -> str:
        """Upload ``source`` into a fresh, verified private directory; answer its path."""
        from . import ssh_private_staging

        return ssh_private_staging.stage_private_artifact(self, source, filename, **bounds)

    def make_private_directory(self) -> str:
        """An empty, verified private staging directory on the worker."""
        from . import ssh_private_staging

        return ssh_private_staging.make_private_directory(self)

    def discard_private_artifact(self, remote: str) -> None:
        """Remove a staged file and its directory; logs, never raises."""
        from . import ssh_private_staging

        ssh_private_staging.discard_private_artifact(self, remote)

    def remove_private_directory(self, directory: str) -> None:
        """Remove a staging directory root took over (``sudo rm -rf``); logs, never raises."""
        from . import ssh_private_staging

        ssh_private_staging.remove_private_directory(self, directory)

    def _login_uid(self) -> int:
        from . import ssh_private_staging

        return ssh_private_staging.login_uid(self)

    def get_cluster_archive(
        self,
        remote_name: str,
        destination: str,
        *,
        maximum_bytes: int = 10 * 1024 ** 3,
    ) -> str:
        from .ssh_private_staging import is_private_staged_path

        # VD-172 (round 1): only from a private staging directory; the bare
        # /tmp/vaelor-cluster- path root's tar left 0644 is gone.
        if not is_private_staged_path(remote_name):
            raise SshTransportError("The cluster transfer path is not permitted.")
        remote = str(remote_name)
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        client = self._session()
        try:
            with client.open_sftp() as sftp:
                size = int(sftp.stat(remote).st_size)
                if size <= 0 or size > int(maximum_bytes):
                    raise SshTransportError(
                        "The remote cluster backup size is not permitted."
                    )
                sftp.get(remote, str(path))
            os.chmod(path, 0o640)
            return str(path)
        finally:
            client.close()

    #: Counts PHYSICAL cores over SSH from the kernel's CPU topology - the set
    #: of distinct (physical_package_id, core_id) pairs, which is cores even on
    #: a multi-socket or SMT part. `nproc` would be simpler but returns logical
    #: CPUs (threads); the coordinator's `lscpu -p=core | ... | wc -l` and a
    #: raw `/sys/.../core_id` glob are both blocked here because `_command`
    #: shlex-quotes every argv token, so the glob runs inside python3 instead of
    #: the remote shell. `python3` is already the transport's install language.
    _CORE_TOPOLOGY_PROBE = (
        "import glob\n"
        "pairs=set()\n"
        "for d in glob.glob('/sys/devices/system/cpu/cpu[0-9]*/topology'):\n"
        "    try:\n"
        "        pkg=open(d+'/physical_package_id').read().strip()\n"
        "        core=open(d+'/core_id').read().strip()\n"
        "    except OSError:\n"
        "        continue\n"
        "    pairs.add((pkg, core))\n"
        "print(len(pairs))\n"
    )

    def _remote_physical_cores(self, execute) -> Optional[int]:
        """Physical cores, or None when the topology cannot be read.

        #113: the caller must NOT substitute the `nproc` thread count for a
        missing core count - a thread count published as `cpu_count` is threads
        masquerading as cores, the defect this fixes. None says "cores unknown"
        honestly; the fleet UI already renders that.
        """
        raw = execute(["python3", "-c", self._CORE_TOPOLOGY_PROBE], optional=True)
        try:
            cores = int(str(raw).strip())
        except (TypeError, ValueError):
            return None
        return cores if cores > 0 else None

    #: Reads accelerator facts from sysfs over the existing SSH path, without
    #: rocm-smi/amd-smi (the Z2 host lacks them). Run through `python3 -c` for
    #: the same reason `_CORE_TOPOLOGY_PROBE` is: `_command` shlex-quotes every
    #: argv token, so a shell glob (`/sys/class/drm/card*/...`) would be passed
    #: literally instead of expanded — and `python3` is already the transport's
    #: install language, so no new allowlist command is needed. It reports what
    #: it read; it never guesses an absent device present. Fields mirror
    #: `vaelor.platforms.accelerators._card_record` so a node and the controller
    #: describe a GPU the same way. MemTotal is captured so a later phase can
    #: size a raisable GTT ceiling.
    _GPU_PROBE = (
        "import glob, json, os, re\n"
        "def num(path):\n"
        "    try:\n"
        "        with open(path) as handle:\n"
        "            return int(handle.read().strip())\n"
        "    except (OSError, ValueError):\n"
        "        return None\n"
        "def text(path):\n"
        "    try:\n"
        "        with open(path) as handle:\n"
        "            return handle.read().strip().lower()\n"
        "    except OSError:\n"
        "        return ''\n"
        "def mem_total():\n"
        "    try:\n"
        "        with open('/proc/meminfo') as handle:\n"
        "            for line in handle:\n"
        "                if line.startswith('MemTotal:'):\n"
        "                    return int(line.split()[1]) * 1024\n"
        "    except (OSError, ValueError, IndexError):\n"
        "        return None\n"
        "    return None\n"
        # The GPU family, by the one reader the controller also runs
        # (`platforms.kfd_topology`): the same source text, not a copy of it.
        + GFX_READER_SOURCE +
        "kfd = os.path.exists('/dev/kfd')\n"
        "renders = sorted(glob.glob('/dev/dri/renderD*'))\n"
        "devices = []\n"
        "try:\n"
        "    names = sorted(os.listdir('/sys/class/drm'))\n"
        "except OSError:\n"
        "    names = []\n"
        "for name in names:\n"
        "    if not re.match('card[0-9]+$', name):\n"
        "        continue\n"
        "    root = '/sys/class/drm/' + name + '/device'\n"
        "    record = {\n"
        "        'id': name,\n"
        "        'vendor': text(root + '/vendor'),\n"
        "        'device': text(root + '/device'),\n"
        "        'vram_total_bytes': num(root + '/mem_info_vram_total'),\n"
        "        'vram_used_bytes': num(root + '/mem_info_vram_used'),\n"
        "        'gtt_total_bytes': num(root + '/mem_info_gtt_total'),\n"
        "        'gtt_used_bytes': num(root + '/mem_info_gtt_used'),\n"
        "        'temperature_label': next((text(path) for path in "
        "sorted(glob.glob(root + '/hwmon/hwmon*/temp1_label'))), ''),\n"
        "    }\n"
        "    if record['vram_total_bytes'] is not None "
        "or record['gtt_total_bytes'] is not None:\n"
        "        devices.append(record)\n"
        "print(json.dumps({\n"
        "    'kfd': kfd,\n"
        "    'render_nodes': renders,\n"
        "    'gfx_target_version': gfx_target_version(),\n"
        "    'devices': devices,\n"
        "    'system_ram_bytes': mem_total(),\n"
        "}))\n"
    )

    @staticmethod
    def _gpu_absent(reason: str) -> Dict[str, Any]:
        return {
            "present": False,
            "reason": reason,
            "gfx_target_version": "",
            "device_count": 0,
            "vram_total_bytes": 0,
            "vram_used_bytes": 0,
            "gtt_total_bytes": 0,
            "gtt_used_bytes": 0,
            "system_ram_bytes": None,
            "unified_memory": None,
            "unified_memory_source": "",
        }

    def _remote_gpu(self, execute, found: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Accelerator facts read from the node's sysfs, or an honest absence.

        Never guesses a device present: when the probe cannot be read at all,
        or reads no GPU, the result says so with a reason. `present` requires a
        compute device (`/dev/kfd` with a render node) or a card reporting
        addressable memory, so a display-only framebuffer does not read as a
        GPU capable of holding weights.

        **Whether the aperture counts towards a model budget is answered here,
        not left to the ledger's bytes** (VD-125). The probe reads the card's
        PCI `vendor` and `device` out of the same sysfs directory it already
        opens, and `platforms.accelerators.unified_memory_verdict` - the one
        home of that question, whose docstring records why byte figures cannot
        answer it - turns them into `unified_memory`. The controller's own facts
        carry the identical pair, so a worker and the head describe a GPU the
        same way and `cluster_capacity.gpu_memory_model` believes both.
        """
        raw = execute(["python3", "-c", self._GPU_PROBE], optional=True)
        try:
            facts = json.loads(str(raw))
        except (TypeError, ValueError):
            return self._gpu_absent(
                "This node's accelerator sysfs could not be read."
            )
        if not isinstance(facts, dict):
            return self._gpu_absent(
                "This node returned invalid accelerator data."
            )
        devices = facts.get("devices") or []
        present = bool(
            (facts.get("kfd") and facts.get("render_nodes")) or devices
        )

        def total(field: str) -> int:
            return sum(
                int(device.get(field) or 0)
                for device in devices
                if isinstance(device, dict)
            )

        if not present:
            return self._gpu_absent(
                "No GPU compute device was found on this node (no /dev/kfd "
                "with a render node, and no DRM card reporting memory)."
            )
        system_ram = facts.get("system_ram_bytes")
        primary = next(
            (device for device in devices if isinstance(device, dict)), {}
        )
        if found is not None:
            # The vendor by name, as local discovery reports it (VD-147). Handed
            # back beside the facts, not inside them: the facts' field set is
            # held equal to the controller's own by `cluster_capacity`.
            found["vendor"] = VENDORS.get(str(primary.get("vendor", "") or "").lower(), "")
            # The driver's own name for the GPU temperature sensor (review S-20).
            found["temperature_label"] = str(primary.get("temperature_label", "") or "")[:32]
        verdict = unified_memory_verdict(
            str(primary.get("vendor", "") or ""),
            str(primary.get("device", "") or ""),
            total("vram_total_bytes"),
            total("gtt_total_bytes"),
        )
        return {
            "present": True,
            "reason": "",
            "gfx_target_version": str(facts.get("gfx_target_version", "") or ""),
            "device_count": len(devices),
            "vram_total_bytes": total("vram_total_bytes"),
            "vram_used_bytes": total("vram_used_bytes"),
            "gtt_total_bytes": total("gtt_total_bytes"),
            "gtt_used_bytes": total("gtt_used_bytes"),
            "system_ram_bytes": (
                int(system_ram) if isinstance(system_ram, int) else None
            ),
            "unified_memory": bool(verdict["unified_memory"]),
            "unified_memory_source": str(verdict["unified_memory_source"]),
        }

    def _bounded_resolve(self, host: str) -> str:
        """``host`` resolved to a name-service answer, or "" if none arrives.

        The lookup runs in a daemon worker and is ABANDONED after
        `_RESOLVE_TIMEOUT_SECONDS` rather than waited on, because the resolver
        it calls has no timeout of its own to pass down (see that constant). An
        abandoned worker costs nothing further - it holds one blocked libc call
        and no lock of ours, and it dies with the process - and the caller then
        reports the address as unresolved, which is the same honest absence a
        name that does not exist produces. A resolver that is merely slow is
        NOT reported as a wrong answer: it is reported as no answer.
        """
        answer: List[str] = []

        def lookup() -> None:
            try:
                answer.append(str(self._resolve_host(host)))
            except Exception:
                pass

        worker = threading.Thread(target=lookup, daemon=True)
        worker.start()
        worker.join(_RESOLVE_TIMEOUT_SECONDS)
        return answer[0] if answer else ""

    def _cluster_address(self) -> Tuple[str, str]:
        """The IPv4 the cluster reaches this node on, or an empty pair reason.

        Whatever this transport connects to IS that address: an IP literal is
        the answer already, and a node enrolled by hostname is resolved once,
        here, so the recorded fact is the address a route table can be matched
        against rather than a name it says nothing about. THE ADDRESS IS THE
        SSH CREDENTIAL'S, which is why the fact records it beside the name -
        see `gpu_node_facts`' module docstring for what a management-NIC
        credential means for the deploy and for the fleet card.
        """
        host = str(self.profile.get("host", "") or "").strip()
        if not host:
            return "", "This node has no cluster address on record."
        try:
            return str(ipaddress.IPv4Address(host)), ""
        except ipaddress.AddressValueError:
            pass
        # One refusal for three endings - no answer, a timed-out resolver, and
        # an answer that is not IPv4 - because to a node record they are the
        # same absent fact, and three spellings of it would drift apart.
        try:
            resolved = str(ipaddress.IPv4Address(self._bounded_resolve(host)))
        except (ipaddress.AddressValueError, ValueError):
            return "", (
                "The cluster address of {} could not be resolved to an IPv4 "
                "address.".format(host)
            )
        return resolved, ""

    def _remote_cluster_interface(self, execute) -> Dict[str, Any]:
        """The NIC carrying this node's cluster address, captured at enrolment.

        VD-125: the cluster interface is a NODE FACT fixed when the cluster is
        formed, not a deploy-time guess. It is keyed on the address - the NIC
        the kernel's own table says carries the address the cluster talks to
        this node on - so a box whose default route belongs to a VPN, a tunnel
        or a bridge still reports the link the collectives must bind.

        Absent with a reason, never guessed, exactly as the accelerator block
        is: an unreadable routing table or an address on no connected subnet
        leaves ``name`` empty and says why, and does not abort discovery - the
        deploy is where that becomes a refusal, and it names the node.
        """
        address, reason = self._cluster_address()
        if reason:
            return {"name": "", "address": address, "reason": reason}
        route = execute(["cat", "/proc/net/route"], optional=True)
        if not str(route).strip():
            return {
                "name": "", "address": address,
                "reason": (
                    "This node's routing table could not be read, so the "
                    "interface carrying its cluster address is unknown."
                ),
            }
        try:
            name = gpu_node_facts.interface_for_address(route, address)
        except ValueError as error:
            return {"name": "", "address": address, "reason": str(error)}
        return {
            "name": name, "address": address,
            **self._remote_link(execute, name),
        }

    def _remote_link(self, execute, name: str) -> Dict[str, Any]:
        """The cluster NIC's link speed and medium, over the same SSH path.

        Two unprivileged, allowlisted ``cat`` reads of the link's own
        world-readable sysfs files, each ``optional`` so an enrolment still
        delivers the rest of the fact when a driver publishes neither: the speed
        then records UNKNOWN with a reason and the medium follows, never a
        guess. The name is validated into the path by `gpu_node_facts`
        (`interface_name`), so nothing this issues can leave ``/sys/class/net``.
        This replaces the serve form's static "2.5G / Wi-Fi" label with the real
        per-node link (VD-125).
        """
        speed = execute(
            ["cat", gpu_node_facts.link_speed_path(name)], optional=True
        )
        uevent = execute(
            ["cat", gpu_node_facts.link_uevent_path(name)], optional=True
        )
        return gpu_node_facts.link_facts(speed, uevent)

    def probe(self) -> Dict[str, Any]:
        client = self._session()
        try:
            def execute(arguments, *, optional=False):
                _, stdout, stderr = client.exec_command(
                    self._command(arguments), timeout=self.timeout
                )
                status = stdout.channel.recv_exit_status()
                output = stdout.read(64 * 1024).decode("utf-8", "replace").strip()
                if status and not optional:
                    message = stderr.read(4096).decode("utf-8", "replace").strip()
                    raise SshTransportError(message or "Node discovery failed.")
                return output if status == 0 else ""

            gpu_found: Dict[str, str] = {}
            gpu = self._remote_gpu(execute, gpu_found)
            pages = int(execute(["getconf", "_PHYS_PAGES"]))
            # Read once and used twice: the GPU's own facts, and whether its
            # memory is shared, which the pool reading below needs.
            gpu = self._remote_gpu(execute)
            page_size = int(execute(["getconf", "PAGE_SIZE"]))
            blocks = execute(["stat", "-f", "-c", "%a %S", "/"]).split()
            os_release = self._parse_os_release(execute(["cat", "/etc/os-release"]))
            docker_binary = execute(
                ["stat", "-c", "%n", "/usr/bin/docker"], optional=True
            )
            docker_version = execute(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                optional=True,
            )
            swarm_state = execute(
                ["docker", "info", "--format", "{{.Swarm.LocalNodeState}}"],
                optional=True,
            )
            return {
                "reachable": True,
                "hostname": execute(["uname", "-n"])[:253],
                "architecture": execute(["uname", "-m"])[:32],
                "kernel": execute(["uname", "-r"])[:128],
                # #113: `cpu_count` is PHYSICAL CORES; `cpu_threads` is the
                # logical count `nproc` reports. `nproc` counts threads, so it
                # fills `cpu_threads`, and cores come from a real topology read
                # (`_remote_physical_cores`). When cores cannot be read this is
                # None, not the thread count - threads masquerading as cores is
                # the exact defect being fixed. Do not treat threads as cores.
                "cpu_count": self._remote_physical_cores(execute),
                "cpu_threads": int(execute(["nproc"])),
                "memory_bytes": pages * page_size,
                "root_free_bytes": int(blocks[0]) * int(blocks[1]),
                "os": os_release.get("PRETTY_NAME", "Linux")[:160],
                "os_id": os_release.get("ID", "").lower()[:40],
                "os_version": os_release.get("VERSION_ID", "")[:40],
                "docker": bool(docker_binary),
                "docker_version": docker_version[:80],
                "docker_swarm_state": swarm_state.lower()[:40],
                # Accelerator facts, read from sysfs over this same SSH path so
                # the capacity ledger knows what each node can hold. Absent GPU
                # is reported absent with a reason, never guessed present.
                "gpu": gpu,
                # With `gpu.unified_memory`, what decides whether this worker
                # is given the GPU vendor sampler before its own emitter says.
                "gpu_vendor": gpu_found.get("vendor", ""),
                # The driver's label for the GPU temperature sensor ("edge",
                # "junction", ...), so the sensor is named as the driver names it.
                "gpu_temperature_label": gpu_found.get("temperature_label", ""),
                # VD-125. The NIC a GPU cluster's collectives must bind on this
                # node, fixed at cluster formation and keyed on the address the
                # cluster reaches it on - so a deploy never has to guess. A node
                # with no such NIC still joins (enrolment is generic) and
                # records the reason here; the GPU deploy is what refuses it.
                "cluster_interface": self._remote_cluster_interface(execute),
                # VD-161. This node's GPU memory pool READING: what it is,
                # what the next restart will make it and what it may be set
                # to. The node reports a few facts (the source the controller
                # runs on itself); they are interpreted here and only the
                # reading is stored - never the node's files (review S4). A
                # node that returned nothing stores a reading that says so.
                "gpu_memory_pool": gpu_memory_pool.pool_status(
                    gpu_memory_pool.parse_remote_facts(execute(
                        ["python3", "-c", gpu_memory_pool.REMOTE_FACTS_PROGRAM],
                        optional=True,
                    )),
                    gpu,
                ),
                # VD-162. Every address this node holds and the link that
                # carries it, so the cluster link panel can say whether
                # the node is on the chosen link. The node's own tables;
                # nothing is probed. None when a table could not be read,
                # which is not the same as a node with no addresses.
                "links": cluster_link.node_links_if_read(
                    execute(["cat", cluster_link.ROUTE_PATH], optional=True),
                    execute(["cat", cluster_link.FIB_TRIE_PATH], optional=True),
                ),
                # VD-147. Which class of machine this is, by the same rule the
                # controller's own driver registry applies, so a worker's GPU
                # temperature is judged against ITS class's bands and never
                # the controller's. "" when it could not be read.
                "machine_class": self._machine_class_from(execute),
            }
        except (IndexError, TypeError, ValueError) as error:
            raise SshTransportError("The node returned invalid discovery data.") from error
        finally:
            client.close()

    @staticmethod
    def _machine_class_from(execute) -> str:
        """A node's machine class from its device-tree model and DMI identity."""
        model = execute(["cat", "/proc/device-tree/model"], optional=True)
        vendor = execute(["cat", "/sys/class/dmi/id/sys_vendor"], optional=True)
        product = execute(["cat", "/sys/class/dmi/id/product_name"], optional=True)
        return remote_machine_class(model, vendor, product)

    def gpu_identity(self) -> Dict[str, str]:
        """The GPU's vendor and temperature-sensor label, read now (a reconcile fills missing ones)."""
        def execute(arguments, *, optional=False):
            try:
                return self.run(arguments)
            except SshTransportError:
                if optional:
                    return ""
                raise
        found: Dict[str, str] = {}
        self._remote_gpu(execute, found)
        return {"vendor": found.get("vendor", ""), "temperature_label": found.get("temperature_label", "")}

    def machine_class(self) -> str:
        """This node's machine class, read now (a reconcile fills a missing one)."""
        def execute(arguments, *, optional=False):
            try:
                return self.run(arguments)
            except SshTransportError:
                if optional:
                    return ""
                raise
        return self._machine_class_from(execute)

    @staticmethod
    def _parse_os_release(raw: str) -> Dict[str, str]:
        values = {}
        for line in str(raw).splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            if key and key.replace("_", "").isalnum():
                values[key] = value.strip().strip("\"'").replace(r"\"", '"')
        return values
