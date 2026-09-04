"""Install and supervise the pinned Distributed Llama runtime over pinned SSH."""

from __future__ import annotations

import ipaddress
import re
from typing import Any, Dict

from .ssh_transport import SshTransportError


INSTALL_ROOT = "/opt/vaelor/runtimes/distributed-llama"
SERVICE_USER = "vaelor-inference"


class PooledRuntime:
    @staticmethod
    def _name(value: str) -> str:
        name = str(value).strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,38}", name):
            raise ValueError("Use a short lowercase pooled deployment name.")
        return name

    @staticmethod
    def _address(value: str) -> str:
        address = ipaddress.ip_address(str(value))
        if (
            address.version != 4
            or not address.is_private
            or address.is_loopback
            or address.is_link_local
        ):
            raise ValueError("Pooled inference requires private static IPv4 addresses.")
        return str(address)

    def install(self, transport, artifact: Dict[str, Any]) -> str:
        commit = str(artifact["commit"])
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("The pooled runtime commit is invalid.")
        staged = transport.put_runtime_artifact(
            artifact["path"], "/tmp/vaelor-distributed-llama.tar.gz"
        )
        digest = transport.run(["sha256sum", staged]).split()[0]
        if digest != artifact["sha256"]:
            raise RuntimeError("The worker received a damaged runtime artifact.")
        try:
            transport.run(["id", "-u", SERVICE_USER])
        except SshTransportError:
            transport.run(
                [
                    "useradd", "--system", "--home-dir", "/nonexistent",
                    "--shell", "/usr/sbin/nologin", SERVICE_USER,
                ],
                sudo=True,
            )
        destination = f"{INSTALL_ROOT}/{commit}"
        transport.run(["install", "-d", "-m", "0755", destination], sudo=True)
        transport.run(
            ["tar", "-xzf", staged, "-C", destination], sudo=True
        )
        transport.run(
            [
                "chmod", "0755", f"{destination}/dllama",
                f"{destination}/dllama-api", f"{destination}/launch.py",
            ],
            sudo=True,
        )
        transport.run(
            ["chown", "-R", f"{SERVICE_USER}:{SERVICE_USER}", destination],
            sudo=True,
        )
        return destination

    def download_model(
        self, transport, destination: str, model_id: str
    ) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,80}", model_id):
            raise ValueError("The pooled model preset is invalid.")
        transport.run(
            [
                "python3", f"{destination}/launch.py", model_id,
                "-skip-run", "-skip-script", "-y",
            ],
            sudo=True,
            timeout=1800,
        )
        transport.run(
            ["chown", "-R", f"{SERVICE_USER}:{SERVICE_USER}", destination],
            sudo=True,
        )

    def start_worker(
        self, transport, *, name: str, destination: str,
        address: str, threads: int,
    ) -> str:
        deployment = self._name(name)
        host = self._address(address)
        unit = f"vaelor-pooled-{deployment}-worker.service"
        content = f"""[Unit]
Description=Vaelor pooled inference worker ({deployment})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={SERVICE_USER}
Group={SERVICE_USER}
UMask=0077
ExecStart={destination}/dllama worker --host {host} --port 9999 --nthreads {threads}
Restart=on-failure
RestartSec=3
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={destination}

[Install]
WantedBy=multi-user.target
"""
        transport.run(
            ["tee", f"/etc/systemd/system/{unit}"],
            sudo=True, stdin_text=content,
        )
        transport.run(["systemctl", "daemon-reload"], sudo=True)
        transport.run(["systemctl", "enable", "--now", unit], sudo=True)
        return unit

    def start_root(
        self, transport, *, name: str, destination: str,
        address: str, threads: int, model: Dict[str, Any],
        workers: list[str],
    ) -> str:
        deployment = self._name(name)
        host = self._address(address)
        worker_args = " ".join(f"{self._address(item)}:9999" for item in workers)
        model_id = model["id"]
        unit = f"vaelor-pooled-{deployment}-root.service"
        content = f"""[Unit]
Description=Vaelor pooled inference root ({deployment})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={SERVICE_USER}
Group={SERVICE_USER}
UMask=0077
WorkingDirectory={destination}
ExecStart={destination}/dllama-api --host {host} --port 9998 --model {destination}/models/{model_id}/dllama_model_{model_id}.m --tokenizer {destination}/models/{model_id}/dllama_tokenizer_{model_id}.t --buffer-float-type q80 --nthreads {threads} --max-seq-len {model['max_sequence_length']} --workers {worker_args}
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={destination}

[Install]
WantedBy=multi-user.target
"""
        transport.run(
            ["tee", f"/etc/systemd/system/{unit}"],
            sudo=True, stdin_text=content,
        )
        transport.run(["systemctl", "daemon-reload"], sudo=True)
        transport.run(["systemctl", "enable", "--now", unit], sudo=True)
        return unit

    def stop_unit(self, transport, unit: str) -> None:
        if not re.fullmatch(
            r"vaelor-pooled-[a-z0-9][a-z0-9-]{0,38}-(?:root|worker)\.service",
            str(unit),
        ):
            raise ValueError("The pooled service name is invalid.")
        transport.run(["systemctl", "disable", "--now", unit], sudo=True)
        transport.run(
            ["rm", "-f", f"/etc/systemd/system/{unit}"], sudo=True
        )
        transport.run(["systemctl", "daemon-reload"], sudo=True)
