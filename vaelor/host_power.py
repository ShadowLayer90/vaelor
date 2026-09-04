"""Host power actions through systemd-logind.

`systemctl reboot` and `systemctl poweroff` are thin clients for the logind
D-Bus API (``org.freedesktop.login1.Manager.Reboot`` / ``PowerOff``). Calling
that API directly would be marginally more precise, but no D-Bus binding is
installed or declared as a dependency of this project, and hand-rolling the
D-Bus wire protocol for three calls is strictly worse than exec'ing the tool
that already speaks it. So: fixed argv, never a shell string, never a PATH
lookup that a caller could influence.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional


SYSTEMCTL = "/usr/bin/systemctl"
CONTROL_PLANE_UNIT = "vaelor-control-plane.service"

#: How long to watch a power command before calling it started. `systemctl
#: reboot`/`poweroff` that genuinely begins does not return within this window;
#: a failure (sudo blocked by NoNewPrivileges, exit 1) returns at once. Two
#: seconds is long enough to catch the fast failure and short enough to stay
#: inside the hardware bridge client's request timeout.
POWER_GRACE_SECONDS = 2.0


class PowerCommandError(RuntimeError):
    """A host power command exited non-zero before it could take effect.

    LESSONS #191 / #208 (VD-098): ``systemctl reboot``/``poweroff`` that
    actually starts never returns, so a fast non-zero exit means the machine is
    doing nothing — on the appliance this was ``sudo`` blocked by
    ``NoNewPrivileges=yes``, exit 1, audited as success while the box stayed up.
    The child's stderr is carried so the caller can audit ``failure`` and show
    the reason instead of reporting ``{"scheduled": True}``.
    """

    def __init__(self, command: Any, returncode: Optional[int], stderr: str):
        self.command = [str(part) for part in command]
        self.returncode = returncode
        self.stderr = stderr or ""
        super().__init__(
            "{} exited {} before taking effect: {}".format(
                " ".join(self.command), returncode, self.stderr or "no error output"
            )
        )

#: Fixed argument vectors. No element is ever taken from a request.
POWER_COMMANDS: Dict[str, tuple[str, ...]] = {
    "restart_service": ("restart", CONTROL_PLANE_UNIT),
    "reboot": ("reboot",),
    "shutdown": ("poweroff",),
}

#: The polkit actions logind checks for a non-root caller. They matter for the
#: privilege report below, not for the call itself.
POLKIT_ACTIONS = {
    "reboot": "org.freedesktop.login1.reboot",
    "shutdown": "org.freedesktop.login1.power-off",
}


class SystemdPowerActions:
    """Reboot, power off, and restart the control plane through systemd."""

    def __init__(
        self,
        *,
        executable: Optional[str] = None,
        finder: Callable[[str], Optional[str]] = shutil.which,
        systemd_root: str = "/run/systemd/system",
        geteuid: Callable[[], int] = lambda: getattr(os, "geteuid", lambda: 0)(),
        runner: Optional[Callable[..., Any]] = None,
        grace_seconds: float = POWER_GRACE_SECONDS,
    ):
        self._executable = executable or (
            SYSTEMCTL if Path(SYSTEMCTL).exists() else finder("systemctl")
        )
        self._systemd_root = Path(systemd_root)
        self._geteuid = geteuid
        self._runner = runner or self._spawn
        self._grace = grace_seconds

    # -- capability -------------------------------------------------------
    def available(self) -> bool:
        """Is systemd the init system here, and is systemctl present?"""
        return bool(self._executable) and self._systemd_root.is_dir()

    def privilege(self) -> Dict[str, Any]:
        """Say what privilege these actions actually need, and whether we have it.

        Running as root needs no polkit rule at all: logind grants a root
        caller unconditionally. An unprivileged system service is the case that
        does not work — it has no active local session, so logind's default
        policy for ``power-off`` and ``reboot`` falls through to
        ``auth_admin_keep``, which cannot be satisfied without an interactive
        agent. That is precisely why the privileged hardware bridge exists, and
        why shipping a polkit rule to let the unprivileged control-plane
        account power the machine off would be a worse answer than routing the
        request through the bridge that already holds the privilege.
        """
        root = self._geteuid() == 0
        return {
            "euid_root": root,
            "sufficient": root,
            "polkit_rule_required": not root,
            "polkit_actions": dict(POLKIT_ACTIONS),
            "reason": (
                ""
                if root
                else (
                    "An unprivileged service has no active local session, so "
                    "logind will not authorize reboot or power off. Route the "
                    "request through the privileged Vaelor hardware bridge."
                )
            ),
        }

    # -- execution --------------------------------------------------------
    def _spawn(self, command: list[str]) -> Dict[str, Any]:
        """Start the command and watch it for the grace window.

        LESSONS #191 / #208: this used to be fire-and-forget — ``Popen`` with
        ``stderr`` to ``DEVNULL`` and no returncode check — so a command that
        exited 1 immediately (``sudo`` blocked by ``NoNewPrivileges``) was
        indistinguishable from one that started, and the failure was audited as
        success. Now stderr is captured and the exit is inspected: a fast
        non-zero exit raises :class:`PowerCommandError`; still running after the
        grace window (a real reboot/poweroff never returns) is success.
        """
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,  # read back only to explain an immediate failure
            close_fds=True,
        )
        try:
            _, stderr = process.communicate(timeout=self._grace)
        except subprocess.TimeoutExpired:
            # Still running: the teardown is under way and will not return.
            # Leave the child alive on purpose — this is the success path.
            return {"scheduled": True, "started": True}
        if process.returncode == 0:
            return {"scheduled": True, "started": True}
        detail = (stderr or b"").decode("utf-8", "replace").strip()
        raise PowerCommandError(command, process.returncode, detail)

    def command(self, action: str) -> list[str]:
        if action not in POWER_COMMANDS:
            raise ValueError("Choose restart_service, reboot, or shutdown.")
        if not self._executable:
            raise RuntimeError("systemctl is not available on this host.")
        return [self._executable, *POWER_COMMANDS[action]]

    def execute(self, action: str) -> Dict[str, Any]:
        command = self.command(action)
        # Run it and watch briefly. A command that fails returns at once and
        # must be reported as a failure, never audited as success (#208,
        # LESSONS #191); a real reboot/poweroff does not return within the
        # grace window, so the response is still delivered before the host
        # tears down. Any PowerCommandError propagates to the caller.
        outcome = self._runner(command)
        result: Dict[str, Any] = {"action": action, "scheduled": True}
        if isinstance(outcome, dict):
            result.update(outcome)
        return result

    def actions(self) -> Dict[str, Dict[str, Any]]:
        """One record per action: a callable when usable, a reason when not."""
        usable = self.available()
        privilege = self.privilege()
        records: Dict[str, Dict[str, Any]] = {}
        for name in POWER_COMMANDS:
            if not usable:
                reason = "This host does not run systemd, so Vaelor cannot power it."
                run = None
            elif not privilege["sufficient"]:
                reason = privilege["reason"]
                run = None
            else:
                reason = ""
                run = (lambda action=name: self.execute(action))
            records[name] = {
                "run": run,
                "reason": reason,
                "id": "systemd-logind",
            }
        return records


def systemd_power_actions(**kwargs: Any) -> Dict[str, Dict[str, Any]]:
    return SystemdPowerActions(**kwargs).actions()
