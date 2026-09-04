"""Root broker that applies a staged Vaelor wheel and proves the result.

The executor (``vaelor-workloads``) cannot write ``/opt/vaelor/venv`` and cannot
restart the control plane it is a peer of, so the *apply* half of an upgrade is
a privileged broker job, modelled on :mod:`vaelor.appliance_recovery`: a
root-owned Unix socket, a one-use staged plan the executor consumes, a
background non-daemon worker that stops the Vaelor unit set, acts, and starts it
again, and a written result file the control plane reads back.

Three things here are load-bearing and each is a defect this project has already
paid for:

* **Success is read back from the restarted control plane, never inferred from
  ``pip``'s exit code.** ``pip install`` returning 0 says the *files* changed;
  it says nothing about which code is *running*. On 2026-08-10 an executor kept
  serving 2.1.0a29 while ``pip show`` said a35 and reported healthy the whole
  time (LESSONS 1, #191/#208). So the only thing that promotes this to
  ``completed`` is the live control plane answering with ``to_version``.

* **That readback comes from the running control plane's own API, not from a
  constant this process imports.** ``vaelor.version.__version__`` in *this*
  process is the version this broker was started with - the old code until the
  restart takes. This module imports no such constant on purpose; it asks the
  restarted process (LESSONS 6, #185/#171).

* **The refresh runs after the restart, never beside the install.** An executor
  on the previous release re-renders the previous release's compose while
  reporting success, which is how an alpha-29 rendering came back on an a35 box
  (LESSONS 15, #119/VD-081). ``install-vaelor.sh`` gets this order right after
  its health gate; this broker keeps the same order for the same reason, and
  ``tests/test_appliance_upgrade_apply.py`` ties the two so they cannot drift.

The install itself uses ``--no-deps --force-reinstall`` so re-applying the same
version is a real reinstall and not a silent no-op (#134): an upgrade that
lands on the same version number - a rebuild, a hotfix - must still replace the
bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import socketserver
import ssl
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# The ordered core unit set is imported, not restated: two hand-kept copies of
# an ordered service list is exactly the LESSONS 6 drift (#98), and the recovery
# broker already owns the one true stop/start order for the core serving stack.
# The upgrade set below is DERIVED from it, never re-typed, for the same reason.
from packaging.version import InvalidVersion, Version

from .appliance_recovery import START_SERVICES, STOP_SERVICES
from .runtime_paths import env_value, jobs_group_id, run_path, state_path

try:
    import grp
except ImportError:  # pragma: no cover - deployment target is Linux
    grp = None


SOCKET_PATH = env_value(
    "VAELOR_UPGRADE_SOCKET", "PM_UPGRADE_SOCKET",
    run_path("appliance-upgrade.sock"),
)
PLAN_PATH = Path(
    env_value(
        "VAELOR_UPGRADE_PLAN", "PM_UPGRADE_PLAN",
        state_path("upgrade/apply-plan.json"),
    )
)
RESULT_PATH = Path(
    env_value(
        "VAELOR_UPGRADE_RESULT", "PM_UPGRADE_RESULT",
        state_path("upgrade/apply-result.json"),
    )
)
#: The pointer to the wheel that produced the *current* install, so a failed
#: apply can restore it. The installer writes it for the first upgrade; a
#: successful apply rewrites it.
CURRENT_WHEEL_PATH = Path(
    env_value(
        "VAELOR_UPGRADE_CURRENT_WHEEL", "PM_UPGRADE_CURRENT_WHEEL",
        state_path("upgrade/current-wheel.json"),
    )
)
#: The only directory a staged wheel may live in. The executor stages here and
#: the broker refuses to install anything outside it - a plan must not be able
#: to point the root install at an arbitrary path.
STAGE_ROOT = Path(state_path("upgrade/staged"))

CONFIRMATION = "apply-vaelor-upgrade"
PLAN_LIFETIME_SECONDS = 30 * 60
MAX_REQUEST_BYTES = 4096
PLAN_ID = re.compile(r"^[a-f0-9]{32}$")

PIP = env_value("VAELOR_UPGRADE_PIP", "PM_UPGRADE_PIP", "/opt/vaelor/venv/bin/pip")
PACKAGE_NAME = "vaelor-control-plane"
#: The single reinstall pin. The two flags are the #134 defeat: without
#: ``--force-reinstall`` a same-version wheel is a no-op, and ``--no-deps``
#: keeps an upgrade from silently resolving the dependency graph on the box.
#: ``install-vaelor.sh`` builds the same flags; the tie is enforced in tests.
REINSTALL_FLAGS = ("install", "--no-deps", "--force-reinstall")
#: The venv tree ``pip`` rewrites: the ``vaelor`` package under ``lib`` AND the
#: regenerated console scripts under ``bin`` (``vaelor-vnc-tls-proxy``,
#: ``vaelor-credential-broker``, ``vaelor-application-research`` ...).
VENV_ROOT = env_value("VAELOR_UPGRADE_VENV", "PM_UPGRADE_VENV", "/opt/vaelor/venv")
#: Re-widen the reinstalled package to world-readable AFTER every install.
#:
#: ``install-vaelor.sh`` runs its ``pip`` under the default root umask (022), so
#: the installed venv is world-readable (dirs 755 / files 644). This broker's
#: unit runs ``UMask=0007``, so an unfixed reinstall writes the SAME tree
#: group-only (dirs 770 / files 660, ``root:vaelor-jobs``) - and every service
#: account NOT in ``vaelor-jobs`` (``vaelor-vnc`` for vnc-gateway/vnc-tls-proxy,
#: ``vaelor-secrets`` for the credential broker, ``vaelor-research`` for
#: application-research) can no longer traverse the package or exec the bin
#: scripts, and crash-loops with ``ModuleNotFoundError``. The control plane
#: survives only because it runs as ``vaelor``, a ``vaelor-jobs`` member.
#: Found and reproduced on the Pi (#178 / VD-102, LESSONS 6). ``a+rX`` only ADDS
#: world read (and directory/already-executable traverse); it never clears a
#: bit, so an intentionally-restricted file is not weakened - and it matches
#: exactly the umask-022 result the installer already leaves. The post-install
#: chmod, not a change to the unit's ``UMask``, is the fix: the unit's own
#: files (socket, plan, result) are ``os.chmod``-ed explicitly, so ``UMask``
#: governs nothing but this reinstall, and a surgical chmod cannot regress a
#: hardening directive elsewhere in the unit.
NORMALIZE_PERMS = ("/bin/chmod", "-R", "a+rX")
REFRESH_COMMAND = (
    env_value(
        "VAELOR_REFRESH_MODELS", "PM_REFRESH_MODELS",
        "/opt/vaelor/venv/bin/vaelor-refresh-models",
    ),
    "--apply",
)
CONTROL_PLANE_ORIGIN = env_value(
    "VAELOR_CONTROL_PLANE_ORIGIN", "PM_CONTROL_PLANE_ORIGIN",
    "https://127.0.0.1:34001",
)
HEALTH_PATH = "/api/v2/auth/status"
#: The loopback-only readback surface the running control plane exposes for this
#: broker. See ``api_upgrade_routes.register_upgrade_routes``.
RUNNING_VERSION_PATH = "/api/v2/upgrade/running-version"
HEALTH_TIMEOUT_SECONDS = 300
HEALTH_STREAK = 3

#: Vaelor services that run the package but are NOT the core serving stack the
#: recovery broker bounces. A factory reset recovers the core and deliberately
#: leaves these alone; an *upgrade* replaces the installed bytes, so every one
#: of them would otherwise keep serving the OLD code from memory until the next
#: reboot - the mixed-version-in-memory defect this project pays for
#: (#178 / VD-102; LESSONS 6, #185/#171). They live here rather than in the
#: shared recovery lists so recovery semantics do not change (LESSONS 6 / #98:
#: do not fold two differently-scoped lists into one).
UPGRADE_EXTRA_SERVICES = (
    "vaelor-hardware-bridge.service",
    "vaelor-application-research.service",
    "vaelor-host-desktop.service",
)
#: The privileged brokers an upgrade must NOT restart, stated explicitly so the
#: exclusion is documented rather than accidental. ``vaelor-appliance-upgrade``
#: is THIS process: ``systemctl stop`` on it kills the apply thread mid-flight,
#: before the result file is written or the rollback runs. ``appliance-recovery``
#: and ``system-update`` are peer orchestrators; leaving their refresh to the
#: next reboot/operation keeps an upgrade's success tied to the core serving
#: stack rather than to a peripheral broker's restart. All three pick up new
#: code the next time they start.
UPGRADE_ORCHESTRATOR_EXCLUSIONS = (
    "vaelor-appliance-upgrade.service",
    "vaelor-appliance-recovery.service",
    "vaelor-system-update.service",
)
#: Stop the core plus the extras: the shared recovery stop order stops the
#: control plane first (it stops serving), and the extras follow it down.
UPGRADE_STOP_SERVICES = (*STOP_SERVICES, *UPGRADE_EXTRA_SERVICES)
#: Start the extras first, then the shared recovery start order - which ends
#: with the control plane, so the control plane stays LAST and the health gate
#: still waits on it. ``vaelor-hardware-bridge`` is ``Before=`` the control
#: plane, so bringing it up ahead of the extras block honours that ordering.
UPGRADE_START_SERVICES = (*UPGRADE_EXTRA_SERVICES, *START_SERVICES)


class UpgradeFailure(RuntimeError):
    """The applied release did not become the running release."""


def _atomic_write(path: Path, payload: Dict[str, Any], mode: int = 0o640) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    os.chmod(temporary, mode)
    temporary.replace(path)


class UpgradeApplyPlans:
    """The one-use handoff between the executor's stage step and this broker.

    Modelled on ``appliance_recovery.FactoryResetPlans``: a plan is staged with
    the verified wheel and the versions it moves between, and ``consume`` renames
    it so it can be claimed exactly once. The wheel path and digest are carried
    so the broker can recompute them against the pin at the privileged boundary
    (:func:`_verify_staged_wheel`) - the plan is a handoff, not a trust
    delegation, so the executor's check does not stand in for the broker's.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else PLAN_PATH

    def _current(self) -> Optional[Dict[str, Any]]:
        try:
            plan = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(plan, dict) or int(plan.get("expires_at", 0)) <= int(time.time()):
            self.path.unlink(missing_ok=True)
            return None
        return plan

    def status(self) -> Dict[str, Any]:
        plan = self._current()
        public = None
        if plan:
            public = {
                key: plan[key]
                for key in ("id", "actor", "from_version", "to_version",
                            "sha256", "bytes", "created_at", "expires_at")
                if key in plan
            }
        return {"staged": public is not None, "plan": public}

    def stage(
        self,
        actor: str,
        wheel: Path,
        from_version: str,
        to_version: str,
        sha256: str,
        byte_length: int,
    ) -> Dict[str, Any]:
        now = int(time.time())
        plan = {
            "id": uuid.uuid4().hex,
            "actor": str(actor)[:64],
            "wheel": str(Path(wheel).resolve()),
            "from_version": str(from_version),
            "to_version": str(to_version),
            "sha256": str(sha256),
            "bytes": int(byte_length),
            "created_at": now,
            "expires_at": now + PLAN_LIFETIME_SECONDS,
        }
        _atomic_write(self.path, plan, mode=0o640)
        return plan

    def cancel(self) -> None:
        self.path.unlink(missing_ok=True)

    def consume(self, plan_id: str) -> Dict[str, Any]:
        plan = self._current()
        if not plan or not PLAN_ID.fullmatch(str(plan_id)):
            raise ValueError("The staged upgrade plan is missing or expired.")
        if plan["id"] != plan_id:
            raise ValueError("The staged upgrade plan does not match.")
        claimed = self.path.with_suffix(".claimed")
        try:
            os.replace(self.path, claimed)
        except FileNotFoundError as error:
            raise ValueError("The staged upgrade plan was already consumed.") from error
        claimed.unlink(missing_ok=True)
        return plan


def read_current_wheel(path: Path = CURRENT_WHEEL_PATH) -> Optional[Path]:
    """The wheel that produced the running install, if it was retained."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    wheel = str(record.get("wheel", "")) if isinstance(record, dict) else ""
    candidate = Path(wheel) if wheel else None
    return candidate if candidate and candidate.is_file() else None


def record_current_wheel(version: str, wheel: Path, path: Path = CURRENT_WHEEL_PATH) -> None:
    _atomic_write(path, {"version": str(version), "wheel": str(Path(wheel).resolve())})


def last_result(path: Path = RESULT_PATH) -> Optional[Dict[str, Any]]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return result if isinstance(result, dict) else None


def _run(runner: Callable[..., Any], command: List[str], timeout: int) -> str:
    result = runner(
        command, capture_output=True, text=True, check=False, timeout=timeout,
    )
    if getattr(result, "returncode", 1) != 0:
        raise UpgradeFailure(
            (getattr(result, "stderr", "") or getattr(result, "stdout", "")
             or "An upgrade command failed.")[-1000:]
        )
    return getattr(result, "stdout", "") or ""


def _verify_staged_wheel(
    wheel: Path, plan: Dict[str, Any], stage_root: Path
) -> Path:
    """Re-check the staged wheel against the plan's pin AT THE PRIVILEGE BOUNDARY.

    #130/#178, pin enforced at the privilege boundary: the executor that stages
    the plan runs unprivileged, and the staged wheel plus the plan file sit in a
    workloads-writable directory. If verification lived only in that stager,
    anything able to rewrite the staging area between stage and apply would get
    *root* to force-reinstall arbitrary bytes into ``/opt/vaelor/venv``. So the
    broker recomputes the SHA-256 and byte length here, immediately before the
    install, against the values the consumed plan carried in memory - and it
    refuses a wheel that is not inside ``stage_root``, so a plan can never aim
    the root install at an arbitrary path.
    """
    resolved = Path(wheel).resolve()
    if resolved.parent != Path(stage_root).resolve():
        raise UpgradeFailure(
            "The staged wheel is outside the upgrade staging directory; "
            "refusing to install it as root."
        )
    if not resolved.is_file():
        raise UpgradeFailure("The staged upgrade wheel is no longer present.")
    expected_sha = str(plan.get("sha256", ""))
    expected_bytes = int(plan.get("bytes", -1))
    actual_bytes = resolved.stat().st_size
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if actual_bytes != expected_bytes or digest.hexdigest() != expected_sha:
        raise UpgradeFailure(
            "The staged wheel no longer matches the plan's pinned length or "
            "digest; refusing to install unverified bytes as root."
        )
    return resolved


def _pip_reinstall(runner: Callable[..., Any], wheel: Path) -> None:
    _run(runner, [PIP, *REINSTALL_FLAGS, str(wheel)], timeout=1800)
    # #178 / VD-102 (Pi): this broker's UMask=0007 strips the world-read that
    # install-vaelor.sh leaves, so re-widen the venv after EVERY install. This
    # lives in _pip_reinstall on purpose: both the forward apply and the
    # rollback reinstall go through here, and the rollback failed for exactly
    # this reason - a group-only reinstall of the OLD wheel that then could not
    # start vnc-gateway. See NORMALIZE_PERMS.
    _run(runner, [*NORMALIZE_PERMS, VENV_ROOT], timeout=120)


def _same_version(left: str, right: str) -> bool:
    """True when two version strings denote the same release.

    ``pip show`` reports the PEP 440-canonicalized version while the manifest
    ``to_version`` and the control plane's running version are raw strings. A
    non-canonical-but-equal spelling (``1.0.0-alpha48`` vs ``1.0.0a48``) must not
    read as a mismatch and trigger a needless rollback (#Recovery-9). Compare by
    parsed version; fall back to exact string equality only when a value is not
    PEP 440-parseable.
    """
    try:
        return Version(left) == Version(right)
    except InvalidVersion:
        return left == right


def _installed_version(runner: Callable[..., Any]) -> str:
    """The version ``pip`` records on disk. Reported, never decisive.

    This is the metadata the install wrote. It is carried beside the live
    readback so the two can be compared, but on its own it is exactly the claim
    #191 warns against: files, not the running process.
    """
    output = _run(runner, [PIP, "show", PACKAGE_NAME], timeout=60)
    for line in output.splitlines():
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip()
    return ""


def _restart_units(runner: Callable[..., Any]) -> None:
    # The UPGRADE set, not the recovery core set: an upgrade must refresh every
    # vaelor service that runs the reinstalled package, or the ones it skips
    # keep serving the old code from memory (#178 / VD-102).
    for service in UPGRADE_STOP_SERVICES:
        _run(runner, ["/usr/bin/systemctl", "stop", service], timeout=90)
    for service in UPGRADE_START_SERVICES:
        _run(runner, ["/usr/bin/systemctl", "start", service], timeout=90)


def _verify_units_active(runner: Callable[..., Any]) -> None:
    """Prove every restarted unit is active, not just the control plane.

    #178 / VD-102 (Pi): ``_default_health_gate`` probes ONLY the control plane
    (``https://127.0.0.1:34001``), which runs as ``vaelor`` - a ``vaelor-jobs``
    member - and so survives a group-only reinstall. The four services whose
    accounts are NOT in ``vaelor-jobs`` (vnc-gateway, vnc-tls-proxy,
    credential-broker, application-research) crash-loop on a package they can no
    longer read, while the upgrade still returned ``completed: true`` -
    honest-green over four dead services. So a ``completed`` apply, and a
    ``rolled_back`` rollback, must mean EVERY unit ``_restart_units`` started is
    active. Enumerated here rather than folded into the health gate because the
    gate's control-plane version-readback is correct and load-bearing on its
    own (LESSONS 1, #191/#208); this adds the services it never covered.
    """
    down: List[str] = []
    for service in UPGRADE_START_SERVICES:
        result = runner(
            ["/usr/bin/systemctl", "is-active", service],
            capture_output=True, text=True, check=False, timeout=30,
        )
        state = (getattr(result, "stdout", "") or "").strip()
        if state != "active":
            down.append("{} ({})".format(service, state or "inactive"))
    if down:
        raise UpgradeFailure(
            "the restart left {} of {} services not active: {}".format(
                len(down), len(UPGRADE_START_SERVICES), ", ".join(down)
            )
        )


def _loopback_opener() -> urllib.request.OpenerDirector:
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))


def _default_health_gate() -> None:
    opener = _loopback_opener()
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    streak = 0
    url = CONTROL_PLANE_ORIGIN + HEALTH_PATH
    while time.monotonic() < deadline:
        try:
            with opener.open(url, timeout=5) as response:
                if 200 <= int(response.getcode() or 0) < 300:
                    streak += 1
                    if streak >= HEALTH_STREAK:
                        return
                else:
                    streak = 0
        except (OSError, ValueError):
            streak = 0
        time.sleep(2)
    raise UpgradeFailure("The control plane did not become healthy after the restart.")


def _default_read_version() -> str:
    opener = _loopback_opener()
    url = CONTROL_PLANE_ORIGIN + RUNNING_VERSION_PATH
    with opener.open(url, timeout=10) as response:
        payload = json.loads(response.read(4096).decode("utf-8"))
    data = payload.get("data") if isinstance(payload, dict) else None
    running = str((data or {}).get("running_version", "")).strip()
    if not running:
        raise UpgradeFailure("The control plane did not report a running version.")
    return running


def _default_refresh(runner: Callable[..., Any]) -> None:
    # Non-fatal, exactly as install-vaelor.sh treats it: a refresh that cannot
    # run must not undo an upgrade that otherwise succeeded and verified.
    try:
        _run(runner, list(REFRESH_COMMAND), timeout=600)
    except (UpgradeFailure, OSError, subprocess.SubprocessError):
        pass


def perform_upgrade_apply(
    plan: Dict[str, Any],
    *,
    runner: Callable[..., Any] = subprocess.run,
    health_gate: Callable[[], None] = _default_health_gate,
    read_version: Callable[[], str] = _default_read_version,
    refresh: Optional[Callable[[Callable[..., Any]], None]] = None,
    snapshot_current: Callable[[], Optional[Path]] = read_current_wheel,
    record_current: Callable[[str, Path], None] = record_current_wheel,
    result_path: Path = RESULT_PATH,
    stage_root: Path = STAGE_ROOT,
) -> Dict[str, Any]:
    """Install the staged wheel, restart, and prove the running version.

    The collaborators are injected so the ordering and the success/rollback
    decision are exercised without real ``pip``, ``systemctl`` or the network -
    the parts that can only be proven on the Pi are the collaborators' bodies,
    not this control flow.
    """
    time.sleep(0)
    wheel = Path(str(plan["wheel"]))
    to_version = str(plan["to_version"])
    from_version = str(plan["from_version"])
    refresh = refresh or _default_refresh
    snapshot = snapshot_current()
    outcome: Optional[Dict[str, Any]] = None
    try:
        # Verify the pinned bytes HERE, at the privileged boundary, before the
        # root install touches them (#130/#178).
        verified = _verify_staged_wheel(wheel, plan, stage_root)
        _pip_reinstall(runner, verified)
        _restart_units(runner)
        health_gate()
        # The health gate proves only the control plane. Every OTHER restarted
        # service must be active too, or a reinstall that broke one of them is
        # honest-green completed while it crash-loops (#178 / VD-102).
        _verify_units_active(runner)
        # SUCCESS COMES FROM HERE, NOT FROM pip's exit code above.
        running = read_version()
        installed = _installed_version(runner)
        completed = _same_version(running, to_version) and _same_version(
            installed, to_version
        )
        if not completed:
            raise UpgradeFailure(
                "installed {!r} / running {!r} did not both reach {!r}".format(
                    installed, running, to_version
                )
            )
        record_current(to_version, wheel)
        refresh(runner)
        outcome = {
            "ok": True,
            "completed": True,
            "from_version": from_version,
            "to_version": to_version,
            "installed_version": installed,
            "running_version": running,
            "rolled_back": False,
            "completed_at": int(time.time()),
        }
    except Exception as error:  # noqa: BLE001 - every failure rolls back and is recorded
        outcome = _rollback(
            runner, snapshot, from_version, to_version,
            health_gate, read_version, error,
        )
    finally:
        _atomic_write(result_path, outcome or {"ok": False, "completed": False})
    return outcome


def _rollback(
    runner: Callable[..., Any],
    snapshot: Optional[Path],
    from_version: str,
    to_version: str,
    health_gate: Callable[[], None],
    read_version: Callable[[], str],
    error: BaseException,
) -> Dict[str, Any]:
    """Restore the previous wheel and prove the *old* version is running again.

    The rollback is held to the same bar as the apply: it is not "successful"
    because the reinstall returned 0, but because the live control plane reports
    ``from_version`` again.
    """
    detail = str(error)[:500]
    if snapshot is None or not Path(snapshot).is_file():
        return {
            "ok": False,
            "completed": False,
            "from_version": from_version,
            "to_version": to_version,
            "rolled_back": False,
            "error": detail,
            "rollback_error": "No retained previous wheel to reinstall.",
            "completed_at": int(time.time()),
        }
    try:
        _pip_reinstall(runner, Path(snapshot))
        _restart_units(runner)
        health_gate()
        # Same all-services bar as the apply: rollback only reports rolled_back
        # once every restarted unit is back up, not merely the control plane
        # (#178 / VD-102). A rollback that leaves vnc-gateway down is not a
        # recovered box.
        _verify_units_active(runner)
        running = read_version()
        if running != from_version:
            raise UpgradeFailure(
                "rollback restored {!r}, not {!r}".format(running, from_version)
            )
        return {
            "ok": False,
            "completed": False,
            "from_version": from_version,
            "to_version": to_version,
            "running_version": running,
            "rolled_back": True,
            "error": detail,
            "completed_at": int(time.time()),
        }
    except Exception as rollback_error:  # noqa: BLE001
        return {
            "ok": False,
            "completed": False,
            "from_version": from_version,
            "to_version": to_version,
            "rolled_back": False,
            "error": detail,
            "rollback_error": str(rollback_error)[:500],
            "completed_at": int(time.time()),
        }


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        try:
            if len(raw) > MAX_REQUEST_BYTES:
                raise ValueError("Upgrade request is too large.")
            request = json.loads(raw.decode("utf-8"))
            if not isinstance(request, dict) or set(request) != {
                "action", "plan_id", "confirmation"
            }:
                raise ValueError("Upgrade request is invalid.")
            if request["action"] != "apply":
                raise ValueError("Unsupported upgrade action.")
            if request["confirmation"] != CONFIRMATION:
                raise ValueError("Upgrade confirmation did not match.")
            plan = UpgradeApplyPlans().consume(str(request["plan_id"]))
            worker = threading.Thread(
                target=perform_upgrade_apply,
                args=(plan,),
                name="vaelor-appliance-upgrade",
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


class ApplianceUpgradeClient:
    def __init__(self, socket_path: str = SOCKET_PATH, timeout: int = 10):
        self.socket_path = socket_path
        self.timeout = timeout

    def apply(self, plan_id: str, confirmation: str) -> Dict[str, Any]:
        request = {
            "action": "apply",
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
            raise RuntimeError(payload.get("error") or "The upgrade was rejected.")
        return payload["result"]


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
