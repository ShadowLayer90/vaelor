"""Bounded system, storage, network, and service inventory."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import socket
import time
from pathlib import Path
from typing import Any, Callable

from .linux_storage import LinuxStorageProvider
from .network_interfaces import interface_inventory
from .runtime_paths import state_path


def _run(command, **kwargs):
    return subprocess.run(command, **kwargs)


class SystemInventory:
    def __init__(
        self,
        runner: Callable[..., Any] = _run,
        storage_provider=None,
        service_catalog=None,
        package_manager=None,
    ):
        self.runner = runner
        self.storage_provider = storage_provider or LinuxStorageProvider(runner)
        self.service_catalog = dict(service_catalog or {})
        self.package_manager = package_manager

    def _platform_dependencies(self):
        if self.service_catalog and self.package_manager is not None:
            return
        from .platform_drivers import default_platform_drivers

        drivers = default_platform_drivers()
        if not self.service_catalog:
            self.service_catalog = drivers[
                "operating_system"
            ].managed_services()
        if self.package_manager is None:
            self.package_manager = drivers["package_manager"]

    def _command(self, command: list[str], timeout: int = 5):
        try:
            return self.runner(
                command, capture_output=True, text=True, check=False, timeout=timeout
            )
        except (OSError, subprocess.SubprocessError):
            return type("Result", (), {"returncode": 1, "stdout": "", "stderr": ""})()

    def storage(self):
        return self.storage_provider.inventory()

    def memory(self):
        values = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                name, raw = line.split(":", 1)
                values[name] = int(raw.strip().split()[0]) * 1024
        except (OSError, ValueError):
            pass
        try:
            swappiness = int(Path("/proc/sys/vm/swappiness").read_text().strip())
        except (OSError, ValueError):
            swappiness = None
        try:
            zswap = Path("/sys/module/zswap/parameters/enabled").read_text().strip().lower() in {
                "1", "y", "yes",
            }
        except OSError:
            zswap = False
        zram_devices = [
            item.name for item in Path("/sys/block").glob("zram*") if item.is_dir()
        ]
        pressure = {"some_avg10": None, "full_avg10": None}
        try:
            for line in Path("/proc/pressure/memory").read_text().splitlines():
                fields = dict(
                    item.split("=", 1) for item in line.split()[1:] if "=" in item
                )
                if line.startswith("some "):
                    pressure["some_avg10"] = float(fields.get("avg10", 0))
                elif line.startswith("full "):
                    pressure["full_avg10"] = float(fields.get("avg10", 0))
        except (OSError, ValueError):
            pass
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", 0)
        return {
            "total_bytes": total,
            "available_bytes": available,
            "used_bytes": max(0, total - available),
            "used_percent": round((total - available) / total * 100, 1) if total else 0,
            "swap_total_bytes": values.get("SwapTotal", 0),
            "swap_free_bytes": values.get("SwapFree", 0),
            "swappiness": swappiness,
            "zswap_enabled": zswap,
            "zram_devices": zram_devices[:8],
            "compressed_swap": zswap or bool(zram_devices),
            "pressure": pressure,
            "profiles": [
                {
                    "id": "balanced",
                    "name": "Balanced",
                    "swappiness": 60,
                    "description": "Linux default tradeoff for desktop, containers, and normal appliance use.",
                },
                {
                    "id": "ai-latency",
                    "name": "AI low latency",
                    "swappiness": 10,
                    "description": "Avoids swapping model pages until memory pressure is high.",
                },
                {
                    "id": "capacity",
                    "name": "Capacity first",
                    "swappiness": 100,
                    "description": "Reclaims inactive memory earlier to keep more mixed workloads available.",
                },
            ],
        }

    def network(self):
        """Interfaces, routes and DNS - and, now, how well that went.

        The enumeration itself moved to `network_interfaces`, which stopped
        treating "`ip` did not answer" as "this machine has no interfaces".
        Those were the same empty list here, and a workstation serving the
        dashboard over `enp193s0` was told it had no network interfaces.
        """
        report = interface_inventory(self._command, shutil.which)
        dns = []
        try:
            for line in Path("/etc/resolv.conf").read_text().splitlines():
                if line.startswith("nameserver "):
                    dns.append(line.split(None, 1)[1][:100])
        except OSError:
            pass
        return {**report, "dns": dns[:8]}

    def services(self):
        self._platform_dependencies()
        output = []
        systemctl = shutil.which("systemctl")
        for key, candidates in self.service_catalog.items():
            unit = candidates[0]
            if not systemctl:
                output.append({"id": key, "unit": unit, "available": False})
                continue
            values = {}
            for candidate in candidates:
                result = self._command([
                    systemctl, "show", candidate,
                    "--property=LoadState,ActiveState,SubState,UnitFileState,NRestarts,ExecMainStatus",
                    "--no-pager",
                ])
                candidate_values = {}
                for line in result.stdout.splitlines():
                    if "=" in line:
                        name, value = line.split("=", 1)
                        candidate_values[name] = value
                if candidate_values.get("LoadState") == "loaded":
                    unit, values = candidate, candidate_values
                    break
            output.append({
                "id": key, "unit": unit,
                "available": values.get("LoadState") == "loaded",
                "active": values.get("ActiveState", "unknown"),
                "substate": values.get("SubState", "unknown"),
                "enabled": values.get("UnitFileState", "unknown"),
                "restarts": int(values.get("NRestarts") or 0),
                "exit_status": int(values.get("ExecMainStatus") or 0),
            })
        return output

    def service_logs(self, service_id: str, lines: int = 200):
        self._platform_dependencies()
        if service_id not in self.service_catalog:
            raise ValueError("Choose a managed Vaelor service.")
        journalctl = shutil.which("journalctl")
        if not journalctl:
            raise ValueError("System logs are unavailable.")
        lines = max(20, min(int(lines), 500))
        live = next(
            (item for item in self.services() if item["id"] == service_id),
            {"unit": self.service_catalog[service_id][0]},
        )
        result = self._command([
            journalctl, "-q", "-u", live["unit"], "-n", str(lines),
            "--no-pager", "--output=short-iso",
        ], timeout=10)
        # An empty read with a non-zero exit is not an empty log: journalctl
        # exits 1 when the caller lacks journal access (it is not in `adm` or
        # `systemd-journal`) and writes nothing to stdout. Surfacing that as a
        # bare empty payload rendered as "No recent log entries" for a service
        # that was logging fine. Report the reason instead of hiding it.
        if result.returncode != 0 and not result.stdout.strip():
            detail = (result.stderr or "").strip().splitlines()
            reason = detail[-1] if detail else "journalctl could not read the service log."
            raise ValueError(reason[:300])
        return {"service": service_id, "lines": lines, "output": result.stdout[-65536:]}

    def updates(self):
        self._platform_dependencies()
        try:
            command = self.package_manager.list_upgradable_command()
        except (AttributeError, RuntimeError):
            return {"available": False, "count": 0, "packages": [], "staged": False}
        result = self._command(command, timeout=15)
        details = self.package_manager.parse_upgradable(result.stdout)
        packages = [item["name"] for item in details]
        state = {}
        try:
            state = json.loads(
                Path(state_path("system-update-state.json")).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError):
            pass
        archive_count = len(list(Path("/var/cache/apt/archives").glob("*.deb")))
        staged_size = sum(
            item.stat().st_size
            for item in Path("/var/cache/apt/archives").glob("*.deb")
            if item.is_file()
        )
        reboot_likelihood = (
            "likely"
            if any(
                item["name"].startswith(("linux-image", "linux-headers"))
                or item["name"] in {"systemd", "libc6"}
                for item in details
            )
            else "possible"
            if any(item["name"].startswith("linux-firmware") for item in details)
            else "unlikely"
        )
        return {
            "available": bool(packages),
            "count": len(packages),
            "packages": packages[:100],
            "details": details[:100],
            "staged": archive_count > 0 and state.get("action") == "stage",
            "staged_archive_count": archive_count,
            "staged_size_bytes": staged_size,
            "estimated_install_minutes": max(
                1, math.ceil((len(packages) * 2 + staged_size / (20 * 1024 * 1024)) / 60)
            ) if packages else 0,
            "reboot_likelihood": reboot_likelihood,
            "last_action": state.get("action"),
            "last_completed_at": state.get("completed_at"),
            "reboot_required": bool(
                state.get("reboot_required")
                or Path("/var/run/reboot-required").exists()
            ),
        }

    def connectivity(self):
        started = time.monotonic()
        dns_ok = False
        internet_ok = False
        try:
            socket.getaddrinfo("huggingface.co", 443, type=socket.SOCK_STREAM)
            dns_ok = True
        except OSError:
            pass
        try:
            connection = socket.create_connection(("1.1.1.1", 443), timeout=3)
            connection.close()
            internet_ok = True
        except OSError:
            pass
        return {
            "dns": dns_ok, "internet": internet_ok,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "targets": ["huggingface.co:443", "1.1.1.1:443"],
        }

    def snapshot(self):
        return {
            "storage": self.storage(),
            "memory": self.memory(),
            "network": self.network(),
            "services": self.services(),
            "updates": self.updates(),
        }
