"""Approval-gated, fixed-scope factory reset broker for Vaelor."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import socketserver
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from .runtime_paths import (
    data_path, env_value, jobs_group_id, run_path, state_path,
)
from .portable_state import PortableState
from .backup_schedule import BACKUP_SCHEDULE_DB

try:
    import grp
except ImportError:  # pragma: no cover
    grp = None


SOCKET_PATH = env_value(
    "VAELOR_RECOVERY_SOCKET", "PM_RECOVERY_SOCKET",
    run_path("appliance-recovery.sock"),
)
PLAN_PATH = Path(
    env_value(
        "VAELOR_RECOVERY_PLAN", "PM_RECOVERY_PLAN",
        state_path("recovery/factory-reset.json"),
    )
)
IMPORT_ROOT = Path(state_path("recovery/imports"))
EXPORT_ROOT = Path(state_path("recovery/exports"))
IMPORT_PLAN_PATH = Path(state_path("recovery/portable-import.json"))
IMPORT_RESULT_PATH = Path(state_path("recovery/portable-import-result.json"))
FACTORY_RESET_RESULT_PATH = Path(state_path("recovery/factory-reset-result.json"))
CONFIRMATION = "RESET VAELOR"
IMPORT_CONFIRMATION = "IMPORT VAELOR STATE"
UNINSTALL_CONFIRMATION = "REMOVE VAELOR"
UNINSTALL_PLAN_PATH = Path(state_path("recovery/remove-vaelor.json"))
UNINSTALL_RESULT_PATH = Path(state_path("recovery/remove-vaelor-result.json"))
# The installer leaves this teardown script under /usr/lib/vaelor, OUTSIDE every
# tree the teardown removes, so it survives to run. It is copied to /run before
# launch so deleting /usr/lib/vaelor cannot pull the running file out from under
# bash. The runner path sits at /run (not /run/vaelor), which the teardown does
# not remove, and the transient systemd unit keeps it alive.
RELEASE_MAINTAIN_SCRIPT = env_value(
    "VAELOR_MAINTAIN_SCRIPT", "PM_MAINTAIN_SCRIPT",
    "/usr/lib/vaelor/release/maintain-vaelor.sh",
)
UNINSTALL_RUNNER_PATH = "/run/vaelor-uninstall.sh"
# The internal confirmation the teardown script itself demands - distinct from
# the user-typed UNINSTALL_CONFIRMATION checked at the socket, so neither alone
# is enough.
UNINSTALL_SCRIPT_CONFIRMATION = "uninstall-vaelor-and-delete-all-data"
# Internal one-use handoff token. This is not a user-visible staging window.
PLAN_LIFETIME_SECONDS = 2 * 60
MAX_REQUEST_BYTES = 4096
PLAN_ID = re.compile(r"^[a-f0-9]{32}$")

ERASED_CATEGORIES = [
    "local dashboard accounts, sessions, MFA, and audit history",
    "encrypted AI provider credentials and assistant API keys",
    "assistant conversations, memories, skills, automations, specialist tasks, app registrations, grants, connections, and invocation audit",
    "deployment jobs, recovery checkpoints, installed models, and managed app data",
    "temporary VNC sessions, KVM control leases, and staged-update records",
]
RETAINED_CATEGORIES = [
    "the host operating system, installed enclosure software, and OS updates",
    "TLS certificate and network configuration so the dashboard remains reachable",
    "hardware-safe cooling, lighting, display, and selected-enclosure configuration",
]

# The scope of a full REMOVE VAELOR teardown, shown before the typed confirm.
UNINSTALL_REMOVES = [
    "the entire Vaelor control plane, executor, brokers, services, and /opt/vaelor",
    "all dashboard accounts, credentials, AI keys, models, chats, and managed app data",
    "the OS stack Vaelor installed - Docker, InfluxDB, the ROCm gfx1151 packages and /opt/rocm, amd-smi, and novnc",
    "the AMD apt source and keyring, and Vaelor's users, groups, and system units",
]
UNINSTALL_RETAINS = [
    "the host operating system and shared tools it did not install (Python, curl, git, OpenSSL)",
    "any non-Vaelor software; Vaelor's mask on SunFounder's enclosure service is lifted, though SunFounder's own package must be reinstalled for that service to run again",
    "your own files that live outside Vaelor's directories",
]

# Fixed stores OUTSIDE the assistant directory, each in its own location.
# Assistant stores are deliberately NOT enumerated here: they are erased
# wholesale by directory glob (see ASSISTANT_STATE_DIR / perform_factory_reset),
# so a newly added assistant store can never be forgotten the way
# custom-agents, workload-act-grants, connector-audit and rag-chat once were.
# (LESSONS 6 / #222: this was a hand-maintained deletion list that drifted from
# the stores it was meant to cover. Four assistant stores were absent from it,
# so a "factory reset" left a user's custom agents AND their workload-action
# grants on the box for the next first-run admin - the opposite of what the
# reset's own "Will be permanently erased" panel promises. Do not add an
# assistant/*.sqlite3 path here; put nothing assistant-owned outside the
# assistant directory, and the glob covers it by construction.)
SQLITE_FILES = (
    Path(state_path("security.sqlite3")),
    Path(data_path("jobs/jobs.sqlite3")),
    Path(data_path("vnc/sessions.sqlite3")),
    Path(state_path("credentials/vault.sqlite3")),
    Path(state_path("integrations/app-capability-registry.sqlite3")),
    Path(data_path("kvm/control.sqlite3")),
    # Scheduled-backup config + run history is host-bound operational state, so
    # a factory reset that returns the box to first-run must erase it too (its
    # off-site endpoint and credential-purpose references included). The archive
    # files themselves sit under recovery/exports and are encrypted with the
    # broker passphrase, whose vault this same reset erases, so they are left as
    # unreadable orphans rather than kept as recoverable state.
    BACKUP_SCHEDULE_DB,
)
# Every SQLite store the assistant writes lives directly under this directory.
# A factory reset erases all of them, so the erased set is whatever is on disk,
# not a list that can drift out of step with the code that creates the stores.
ASSISTANT_STATE_DIR = Path(state_path("assistant"))
DATA_ROOTS = (
    Path(data_path("workloads")),
    Path(data_path("models")),
    Path(data_path("backups/workloads")),
)
STATE_FILES = (Path(state_path("system-update-state.json")),)
STOP_SERVICES = (
    "vaelor-control-plane.service",
    "vaelor-workload-executor.service",
    "vaelor-workload-broker.service",
    "vaelor-credential-broker.service",
    "vaelor-vnc-gateway.service",
    "vaelor-vnc-tls-proxy.service",
)
START_SERVICES = (
    "vaelor-credential-broker.service",
    "vaelor-vnc-gateway.service",
    "vaelor-vnc-tls-proxy.service",
    "vaelor-workload-executor.service",
    "vaelor-workload-broker.service",
    "vaelor-control-plane.service",
)


def _atomic_write(
    path: Path, payload: Dict[str, Any], mode: int = 0o640
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    os.chmod(temporary, mode)
    temporary.replace(path)


class FactoryResetPlans:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else PLAN_PATH

    def status(self) -> Dict[str, Any]:
        try:
            plan = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            plan = None
        if not isinstance(plan, dict) or int(plan.get("expires_at", 0)) <= int(time.time()):
            if plan is not None:
                self.path.unlink(missing_ok=True)
            plan = None
        return {
            "staged": plan is not None,
            "plan": plan,
            "confirmation": CONFIRMATION,
            "erases": ERASED_CATEGORIES,
            "retains": RETAINED_CATEGORIES,
        }

    def stage(self, actor: str) -> Dict[str, Any]:
        current = self.status()
        if current["staged"]:
            return current
        now = int(time.time())
        plan = {
            "id": uuid.uuid4().hex,
            "actor": str(actor)[:64],
            "created_at": now,
            "expires_at": now + PLAN_LIFETIME_SECONDS,
            "erases": ERASED_CATEGORIES,
            "retains": RETAINED_CATEGORIES,
        }
        _atomic_write(self.path, plan)
        return self.status()

    def cancel(self) -> Dict[str, Any]:
        self.path.unlink(missing_ok=True)
        return self.status()

    def consume(self, plan_id: str) -> Dict[str, Any]:
        status = self.status()
        plan = status.get("plan")
        if not plan or not PLAN_ID.fullmatch(str(plan_id)):
            raise ValueError("The staged reset plan is missing or expired.")
        if plan["id"] != plan_id:
            raise ValueError("The staged reset plan does not match.")
        claimed = self.path.with_suffix(".claimed")
        try:
            os.replace(self.path, claimed)
        except FileNotFoundError as error:
            raise ValueError("The staged reset plan was already consumed.") from error
        claimed.unlink(missing_ok=True)
        return plan


class UninstallPlans:
    """One-use handoff for a full REMOVE VAELOR teardown.

    Mirrors :class:`FactoryResetPlans` - a two-minute one-use plan, consumed
    exactly once - but carries the uninstall scope and reports the last launch
    result. A teardown that started leaves no success result (the box, and this
    file, are being destroyed); only a failure to LAUNCH writes one, so the
    dashboard can say the removal did not begin.
    """

    def __init__(
        self, path: Optional[Path] = None, result_path: Optional[Path] = None
    ):
        self.path = Path(path) if path is not None else UNINSTALL_PLAN_PATH
        self.result_path = (
            Path(result_path) if result_path is not None else UNINSTALL_RESULT_PATH
        )

    def status(self) -> Dict[str, Any]:
        try:
            plan = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            plan = None
        if not isinstance(plan, dict) or int(plan.get("expires_at", 0)) <= int(time.time()):
            if plan is not None:
                self.path.unlink(missing_ok=True)
            plan = None
        try:
            result = json.loads(self.result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            result = None
        return {
            "staged": plan is not None,
            "plan": plan,
            "confirmation": UNINSTALL_CONFIRMATION,
            "removes": UNINSTALL_REMOVES,
            "retains": UNINSTALL_RETAINS,
            "last_result": result if isinstance(result, dict) else None,
        }

    def stage(self, actor: str) -> Dict[str, Any]:
        current = self.status()
        if current["staged"]:
            return current
        now = int(time.time())
        plan = {
            "id": uuid.uuid4().hex,
            "actor": str(actor)[:64],
            "created_at": now,
            "expires_at": now + PLAN_LIFETIME_SECONDS,
            "removes": UNINSTALL_REMOVES,
            "retains": UNINSTALL_RETAINS,
        }
        _atomic_write(self.path, plan)
        return self.status()

    def cancel(self) -> Dict[str, Any]:
        self.path.unlink(missing_ok=True)
        return self.status()

    def consume(self, plan_id: str) -> Dict[str, Any]:
        status = self.status()
        plan = status.get("plan")
        if not plan or not PLAN_ID.fullmatch(str(plan_id)):
            raise ValueError("The staged uninstall plan is missing or expired.")
        if plan["id"] != plan_id:
            raise ValueError("The staged uninstall plan does not match.")
        claimed = self.path.with_suffix(".claimed")
        try:
            os.replace(self.path, claimed)
        except FileNotFoundError as error:
            raise ValueError("The staged uninstall plan was already consumed.") from error
        claimed.unlink(missing_ok=True)
        return plan


class PortableImportPlans:
    def __init__(
        self,
        path: Optional[Path] = None,
        result_path: Optional[Path] = None,
        import_root: Optional[Path] = None,
        portable_state: Optional[PortableState] = None,
        export_root: Optional[Path] = None,
    ):
        self.path = Path(path) if path is not None else IMPORT_PLAN_PATH
        self.result_path = (
            Path(result_path) if result_path is not None else IMPORT_RESULT_PATH
        )
        self.import_root = (
            Path(import_root) if import_root is not None else IMPORT_ROOT
        )
        self.export_root = (
            Path(export_root) if export_root is not None else EXPORT_ROOT
        )
        self.portable_state = portable_state or PortableState()

    def _current(self) -> Optional[Dict[str, Any]]:
        try:
            plan = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(plan, dict) or int(plan.get("expires_at", 0)) <= int(time.time()):
            archive = Path(str(plan.get("archive", ""))) if isinstance(plan, dict) else None
            if archive is not None:
                archive.unlink(missing_ok=True)
            self.path.unlink(missing_ok=True)
            return None
        return plan

    def status(self) -> Dict[str, Any]:
        plan = self._current()
        try:
            result = json.loads(self.result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            result = None
        public_plan = None
        if plan:
            public_plan = {
                key: plan[key]
                for key in (
                    "id", "actor", "created_at", "expires_at",
                    "source_version", "approved_at",
                )
                if key in plan
            }
        return {
            "staged": public_plan is not None,
            "plan": public_plan,
            "confirmation": IMPORT_CONFIRMATION,
            "scope": PortableState.scope(),
            "last_result": result if isinstance(result, dict) else None,
        }

    def stage(
        self,
        actor: str,
        archive: Path,
        passphrase: str,
    ) -> Dict[str, Any]:
        resolved_root = self.import_root.resolve()
        resolved_archive = Path(archive).resolve()
        try:
            resolved_archive.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError("The staged transfer escaped the import directory.") from error
        if not resolved_archive.is_file():
            raise ValueError("The staged transfer archive is missing.")
        manifest = self.portable_state.inspect(resolved_archive, passphrase)
        self.cancel()
        now = int(time.time())
        plan = {
            "id": uuid.uuid4().hex,
            "actor": str(actor)[:64],
            "archive": str(resolved_archive),
            "passphrase": passphrase,
            "source_version": manifest.get("version"),
            "created_at": now,
            "expires_at": now + 10 * 60,
        }
        _atomic_write(self.path, plan, mode=0o600)
        return self.status()

    def cancel(self) -> Dict[str, Any]:
        plan = self._current()
        if plan:
            if plan.get("approved_at"):
                raise ValueError("The approved import is already queued.")
            Path(str(plan.get("archive", ""))).unlink(missing_ok=True)
        self.path.unlink(missing_ok=True)
        return self.status()

    def approve(self, plan_id: str) -> Dict[str, Any]:
        plan = self._current()
        if not plan or plan.get("id") != plan_id:
            raise ValueError("The staged import is missing or expired.")
        if plan.get("approved_at"):
            raise ValueError("The staged import was already approved.")
        plan["approved_at"] = int(time.time())
        _atomic_write(self.path, plan, mode=0o600)
        return self.status()

    def unapprove(self, plan_id: str) -> None:
        plan = self._current()
        if plan and plan.get("id") == plan_id:
            plan.pop("approved_at", None)
            _atomic_write(self.path, plan, mode=0o600)

    def consume(self, plan_id: str) -> Dict[str, Any]:
        plan = self._current()
        if not plan or not PLAN_ID.fullmatch(str(plan_id)):
            raise ValueError("The staged import is missing or expired.")
        if plan["id"] != plan_id:
            raise ValueError("The staged import does not match.")
        claimed = self.path.with_suffix(".claimed")
        try:
            os.replace(self.path, claimed)
        except FileNotFoundError as error:
            raise ValueError("The staged import was already consumed.") from error
        claimed.unlink(missing_ok=True)
        return plan


def _run(command: list[str], timeout: int = 120) -> None:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or "Recovery command failed.")[-1000:])


def _remove_sqlite(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        candidate.unlink(missing_ok=True)


def _clear_root(root: Path) -> None:
    resolved = root.resolve()
    allowed = {item.resolve() for item in DATA_ROOTS}
    if resolved not in allowed:
        raise RuntimeError("Recovery target escaped the fixed allowlist.")
    if not resolved.exists():
        return
    for child in resolved.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            shutil.rmtree(child)


def _managed_containers() -> list[str]:
    result = subprocess.run(
        ["/usr/bin/docker", "ps", "-aq", "--no-trunc"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    identifiers = [
        value for value in result.stdout.splitlines()
        if re.fullmatch(r"[a-f0-9]{12,64}", value)
    ][:200]
    if not identifiers:
        return []
    inspect = subprocess.run(
        ["/usr/bin/docker", "inspect", *identifiers],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    try:
        records = json.loads(inspect.stdout)
    except (TypeError, json.JSONDecodeError):
        return []
    root = Path(data_path("workloads")).resolve()
    managed = []
    for record in records:
        labels = (record.get("Config") or {}).get("Labels") or {}
        working = labels.get("com.docker.compose.project.working_dir", "")
        try:
            Path(working).resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        identifier = str(record.get("Id", ""))
        if re.fullmatch(r"[a-f0-9]{64}", identifier):
            managed.append(identifier)
    return managed


def perform_factory_reset() -> None:
    # Mirror perform_portable_import: the stop -> destructive act -> start
    # sequence is wrapped so the control plane, executor, broker, and VNC are
    # ALWAYS restarted and a result is ALWAYS written, even when a reset step
    # hangs or raises (#Recovery-2). Without this a mid-reset failure - a docker
    # rm timeout, an rmtree error - would leave the START loop unreached and the
    # box stopped with no result file and no in-band recovery.
    # Bound before the try so the finally can ALWAYS write a result, even if the
    # except body itself raises (a recovery-loop _run can raise OSError, which
    # its narrow except would let escape - leaving `outcome` unbound and the
    # result file unwritten, defeating this function's whole guarantee).
    outcome: Dict[str, Any] = {
        "ok": False,
        "completed_at": int(time.time()),
        "error": "factory reset did not complete",
    }
    try:
        for service in STOP_SERVICES:
            _run(["/usr/bin/systemctl", "stop", service], timeout=90)
        managed = _managed_containers()
        if managed:
            _run(["/usr/bin/docker", "rm", "--force", *managed], timeout=180)
        for path in SQLITE_FILES:
            _remove_sqlite(path)
        # Erase every assistant store by directory glob, not by name, so a store
        # added later cannot be left behind (LESSONS 6 / #222). glob() on a
        # missing directory yields nothing, and a non-recursive *.sqlite3 match
        # cannot escape the assistant directory.
        for path in sorted(ASSISTANT_STATE_DIR.glob("*.sqlite3")):
            _remove_sqlite(path)
        for root in DATA_ROOTS:
            _clear_root(root)
        for path in STATE_FILES:
            path.unlink(missing_ok=True)
        for service in START_SERVICES:
            _run(["/usr/bin/systemctl", "start", service], timeout=90)
        outcome = {"ok": True, "completed_at": int(time.time())}
    except Exception as error:
        recovery_error = ""
        for service in START_SERVICES:
            try:
                _run(["/usr/bin/systemctl", "start", service], timeout=90)
            except Exception as restart_error:
                # Broad: a restart failing for ANY reason (including an OSError
                # invoking systemctl) must be recorded and the loop continue, so
                # the result is written rather than the failure escaping.
                recovery_error += f" {service}: {restart_error}"
        outcome = {
            "ok": False,
            "completed_at": int(time.time()),
            "error": f"{error}{recovery_error}"[:500],
        }
    finally:
        _atomic_write(FACTORY_RESET_RESULT_PATH, outcome, mode=0o640)


def perform_portable_import(plan: Dict[str, Any]) -> None:
    time.sleep(2)
    archive = Path(str(plan["archive"]))
    state = PortableState()
    import_result: Optional[Dict[str, Any]] = None
    # Bound before the try so the finally can ALWAYS write a result, even if the
    # except body raises (mirrors perform_factory_reset).
    outcome: Dict[str, Any] = {
        "ok": False,
        "completed_at": int(time.time()),
        "error": "portable import did not complete",
    }
    try:
        for service in STOP_SERVICES:
            _run(["/usr/bin/systemctl", "stop", service], timeout=90)
        import_result = state.import_archive(
            archive,
            str(plan["passphrase"]),
            replace=True,
        )
        for service in START_SERVICES:
            _run(["/usr/bin/systemctl", "start", service], timeout=90)
        outcome = {
            "ok": True,
            "completed_at": int(time.time()),
            "source_version": import_result.get("source_version"),
            "imported": import_result.get("imported"),
            "credentials_restored": False,
            "sessions_restored": False,
        }
    except Exception as error:
        recovery_error = ""
        if import_result is not None:
            try:
                state.rollback_import(import_result)
            except Exception as rollback_error:
                recovery_error = f" Rollback also failed: {rollback_error}"
        for service in START_SERVICES:
            try:
                _run(["/usr/bin/systemctl", "start", service], timeout=90)
            except Exception as restart_error:
                # Broad for the same reason as perform_factory_reset: a restart
                # failing for any reason must be recorded, not escape and leave
                # the result unwritten.
                recovery_error += f" {service}: {restart_error}"
        outcome = {
            "ok": False,
            "completed_at": int(time.time()),
            "error": f"{error}{recovery_error}"[:500],
        }
    finally:
        archive.unlink(missing_ok=True)
        _atomic_write(IMPORT_RESULT_PATH, outcome, mode=0o640)


def perform_uninstall(runner: Any = subprocess.run) -> None:
    """Launch the bare-OS teardown as a transient systemd unit, then return.

    Unlike :func:`perform_factory_reset`, a full uninstall removes ``/opt/vaelor``
    and this recovery daemon itself, so it CANNOT run in the daemon's own thread:
    the teardown stops ``vaelor-appliance-recovery.service`` and would kill the
    thread mid-run. ``systemd-run`` starts the teardown as a transient unit owned
    by pid 1, which survives every Vaelor unit being stopped and removed.

    The release script is copied to ``/run`` first (a path the teardown does not
    delete) so removing ``/usr/lib/vaelor`` cannot pull the running file out from
    under bash, and a ``sleep`` lets the HTTP 202 reach the browser before the
    control plane stops. A teardown that starts leaves no result - the box is
    being destroyed - so only a failure to LAUNCH writes one, letting the
    dashboard report that the removal never began.
    """
    try:
        source = Path(RELEASE_MAINTAIN_SCRIPT)
        if not source.is_file():
            raise RuntimeError(
                "The Vaelor release script is missing at {}, so the teardown "
                "cannot be launched.".format(RELEASE_MAINTAIN_SCRIPT)
            )
        shutil.copyfile(source, UNINSTALL_RUNNER_PATH)
        os.chmod(UNINSTALL_RUNNER_PATH, 0o700)
        # A prior attempt that FAILED (e.g. an old teardown script that rejected
        # the flags, before this was updated) leaves vaelor-uninstall.service in
        # the failed state, and `systemd-run` refuses to reuse a unit name that
        # still exists - so a retry dies with "unit already exists" (exit 1) and
        # looks like the button doing nothing. Clear any such lingering unit
        # first, best-effort; `--collect` handles the clean-exit case.
        runner(
            ["/usr/bin/systemctl", "reset-failed", "vaelor-uninstall.service"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        runner(
            [
                "/usr/bin/systemd-run",
                "--collect",
                "--unit=vaelor-uninstall",
                "--description=Vaelor bare-OS teardown",
                "/bin/bash",
                "-c",
                "sleep 5; exec /bin/bash {runner} uninstall --purge-data "
                "--bare-os --confirm {token} "
                ">/run/vaelor-uninstall.log 2>&1".format(
                    runner=UNINSTALL_RUNNER_PATH,
                    token=UNINSTALL_SCRIPT_CONFIRMATION,
                ),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except Exception as error:  # noqa: BLE001 - any launch failure must be recorded
        _atomic_write(
            UNINSTALL_RESULT_PATH,
            {
                "ok": False,
                "completed_at": int(time.time()),
                "error": "The teardown could not be launched: {}".format(error)[:500],
            },
            mode=0o640,
        )


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        try:
            if len(raw) > MAX_REQUEST_BYTES:
                raise ValueError("Recovery request is too large.")
            request = json.loads(raw.decode("utf-8"))
            if not isinstance(request, dict) or set(request) != {
                "action", "plan_id", "confirmation"
            }:
                raise ValueError("Recovery request is invalid.")
            if request["action"] == "factory-reset":
                if request["confirmation"] != CONFIRMATION:
                    raise ValueError("Factory reset confirmation did not match.")
                FactoryResetPlans().consume(str(request["plan_id"]))
                target, arguments, name = (
                    perform_factory_reset, (), "vaelor-factory-reset",
                )
            elif request["action"] == "portable-import":
                if request["confirmation"] != IMPORT_CONFIRMATION:
                    raise ValueError("Portable import confirmation did not match.")
                plan = PortableImportPlans().consume(str(request["plan_id"]))
                target, arguments, name = (
                    perform_portable_import, (plan,), "vaelor-portable-import",
                )
            elif request["action"] == "uninstall":
                if request["confirmation"] != UNINSTALL_CONFIRMATION:
                    raise ValueError("Remove-Vaelor confirmation did not match.")
                UninstallPlans().consume(str(request["plan_id"]))
                target, arguments, name = (
                    perform_uninstall, (), "vaelor-uninstall",
                )
            else:
                raise ValueError("Unsupported recovery action.")
            worker = threading.Thread(
                target=target,
                args=arguments,
                name=name,
                daemon=False,
            )
            worker.start()
            result = {"ok": True, "result": {"scheduled": True}}
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            result = {"ok": False, "error": str(error)[:1000]}
        self.wfile.write(
            json.dumps(result, separators=(",", ":")).encode("utf-8") + b"\n"
        )


_UnixServerBase = getattr(
    socketserver, "ThreadingUnixStreamServer", socketserver.ThreadingTCPServer
)


class _Server(_UnixServerBase):
    daemon_threads = True


def serve() -> None:
    path = Path(SOCKET_PATH)
    path.unlink(missing_ok=True)
    server = _Server(str(path), _Handler)
    os.chmod(path, 0o660)
    if grp is not None:
        group_id = jobs_group_id(grp)
        if group_id is not None:
            os.chown(path, 0, group_id)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        path.unlink(missing_ok=True)


class ApplianceRecoveryClient:
    def __init__(self, socket_path: str = SOCKET_PATH, timeout: int = 10):
        self.socket_path = socket_path
        self.timeout = timeout

    def reset(self, plan_id: str, confirmation: str) -> Dict[str, Any]:
        return self._request("factory-reset", plan_id, confirmation)

    def portable_import(
        self, plan_id: str, confirmation: str
    ) -> Dict[str, Any]:
        return self._request("portable-import", plan_id, confirmation)

    def uninstall(self, plan_id: str, confirmation: str) -> Dict[str, Any]:
        return self._request("uninstall", plan_id, confirmation)

    def _request(
        self, action: str, plan_id: str, confirmation: str
    ) -> Dict[str, Any]:
        request = {
            "action": action,
            "plan_id": str(plan_id),
            "confirmation": str(confirmation),
        }
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(self.socket_path)
            connection.sendall(
                json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
            )
            response = connection.makefile("rb").readline(64 * 1024)
        payload = json.loads(response.decode("utf-8"))
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error") or "Factory reset was rejected.")
        return payload["result"]


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
