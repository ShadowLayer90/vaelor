"""Fingerprint-pinned SSH transport with a fixed cluster command surface."""

from __future__ import annotations

import base64
import hashlib
import os
import shlex
import socket
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class SshTransportError(RuntimeError):
    pass


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


class SshTransport:
    """Execute only reviewed argv-style commands after host-key verification."""

    def __init__(self, profile: Dict[str, Any], timeout: int = 20):
        self.profile = profile
        self.timeout = max(5, min(int(timeout), 120))

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
        return client

    @staticmethod
    def _command(arguments: Iterable[str]) -> str:
        return " ".join(shlex.quote(str(argument)) for argument in arguments)

    def run(
        self,
        arguments: Iterable[str],
        *,
        sudo: bool = False,
        stdin_text: str = "",
        timeout: Optional[int] = None,
    ) -> str:
        args = [str(item) for item in arguments]
        if not args or args[0] not in {
            "cat", "docker", "getconf", "id", "nproc", "stat", "systemctl",
            "uname", "apt-get", "install", "mkdir", "tee", "tar",
            "sha256sum", "chmod", "python3", "rm", "useradd", "chown",
        }:
            raise SshTransportError("The requested remote command is not permitted.")
        command = self._command(args)
        input_text = stdin_text
        if sudo:
            command = f"sudo -S -p '' -- {command}"
            if self.profile.get("sudo_uses_login_password", True):
                input_text = f"{self.profile['password']}\n{stdin_text}"
        client = self._connect()
        try:
            stdin, stdout, stderr = client.exec_command(
                command, timeout=timeout or self.timeout
            )
            if input_text:
                stdin.write(input_text)
                stdin.flush()
            stdin.channel.shutdown_write()
            status = stdout.channel.recv_exit_status()
            output = stdout.read(1024 * 1024).decode("utf-8", "replace").strip()
            error = stderr.read(16 * 1024).decode("utf-8", "replace").strip()
            if status != 0:
                raise SshTransportError(error[:400] or "The remote command failed.")
            return output
        finally:
            client.close()

    def put_runtime_artifact(self, source: str, remote_name: str) -> str:
        path = Path(source)
        if not path.is_file() or path.stat().st_size > 100 * 1024 ** 2:
            raise SshTransportError("The reviewed runtime artifact is unavailable.")
        if (
            not remote_name.startswith("/tmp/vaelor-")
            or "/" in remote_name[len("/tmp/"):]
        ):
            raise SshTransportError("The runtime staging path is not permitted.")
        client = self._connect()
        try:
            with client.open_sftp() as sftp:
                sftp.put(str(path), remote_name)
                sftp.chmod(remote_name, 0o600)
            return remote_name
        finally:
            client.close()

    @staticmethod
    def _managed_transfer_path(remote_name: str) -> str:
        value = str(remote_name)
        if (
            not value.startswith("/tmp/vaelor-cluster-")
            or "/" in value[len("/tmp/"):]
            or not all(
                character.isalnum() or character in "._/-"
                for character in value
            )
        ):
            raise SshTransportError("The cluster transfer path is not permitted.")
        return value

    def get_cluster_archive(
        self,
        remote_name: str,
        destination: str,
        *,
        maximum_bytes: int = 10 * 1024 ** 3,
    ) -> str:
        remote = self._managed_transfer_path(remote_name)
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        client = self._connect()
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

    def put_cluster_archive(
        self,
        source: str,
        remote_name: str,
        *,
        maximum_bytes: int = 10 * 1024 ** 3,
    ) -> str:
        remote = self._managed_transfer_path(remote_name)
        path = Path(source)
        if (
            not path.is_file()
            or path.stat().st_size <= 0
            or path.stat().st_size > int(maximum_bytes)
        ):
            raise SshTransportError("The cluster backup archive is unavailable.")
        client = self._connect()
        try:
            with client.open_sftp() as sftp:
                sftp.put(str(path), remote)
                sftp.chmod(remote, 0o600)
            return remote
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

    def probe(self) -> Dict[str, Any]:
        client = self._connect()
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

            pages = int(execute(["getconf", "_PHYS_PAGES"]))
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
            }
        except (IndexError, TypeError, ValueError) as error:
            raise SshTransportError("The node returned invalid discovery data.") from error
        finally:
            client.close()

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
