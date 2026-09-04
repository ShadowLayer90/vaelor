"""How the last boot ended, and what a clean power register does not prove.

The appliance hard-powered off under load. **The owner found out by walking
past a dark machine**, because nothing on it could say what had happened: the
control plane came back up with no memory of the outage, every panel was green,
and the only evidence was in a journal nobody read.

An unclean shutdown is detectable on boot, and this module detects it.

**The evidence is the absence of a shutdown sequence, not the presence of a
fault.** A system going down deliberately writes its way there — PID 1 reaches
a power-off, reboot or halt target, or the kernel prints ``reboot: Power down``
— and a system that lost power simply stops mid-sentence. So the previous
boot's last lines are read and asked one question: *did anything ask this
machine to stop?*

Two traps are wired in deliberately, and both were live in the incident:

* **A user session reaching ``shutdown.target`` is an SSH session ending**, not
  the system going down. ``systemd[1917]: Reached target shutdown.target`` is
  emitted by a *user* manager every time someone logs out. Only PID 1 and the
  kernel are believed here.
* **Absence of a journal is not absence of a shutdown.** With volatile
  journaling, or on the first boot after an install, there is nothing to read —
  and that is ``unknown``, never ``unclean``. Reporting a power loss that did
  not happen is the same class of error as missing one that did.

**This is not the NVMe counter, and the two answer different questions.**
:mod:`vaelor.nvme_smart` reads ``unsafe_shutdowns`` — 65 against 145 power
cycles on the target machine — which is a *lifetime* ratio for one drive,
needs ``CAP_SYS_ADMIN``, and exists only where there is an NVMe controller. It
says this appliance has a history of dirty power-offs. It cannot say *the last
one was unclean*, which is the question an owner in front of a rebooted machine
is actually asking.

**And the register that looks like the answer is not one.** ``vcgencmd
get_throttled`` reports flags *for the current boot*, and **a power loss clears
them**: after a cutout it reads ``0x0`` and describes the boot that has just
started. The obvious reading — *"no undervoltage was ever recorded, so power is
fine"* — clears the actual cause. Absence of evidence that the event itself
destroyed is not evidence of absence, and
:data:`THROTTLED_HISTORY_NOTE` travels with every reading of that register so
nothing downstream can quietly treat a clean flag word as proof.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any, Callable, Dict, List, Optional


#: What the register covers and what it cannot. Carried on the power payload
#: rather than left for a reader to know, because the reader that did not know
#: it cleared the right diagnosis in the real incident.
THROTTLED_HISTORY_NOTE = (
    "These flags cover the current boot only and are cleared by a power loss, "
    "so a clean reading after an unexpected power-off is not evidence that "
    "power was adequate. Read the previous boot's ending instead."
)

#: Messages that mean *something asked this machine to stop*. Matched against
#: the previous boot's last lines, case-insensitively.
#:
#: ``reached target shutdown`` is deliberately absent: a user manager emits it
#: on every logout, and treating it as a system shutdown would call a power
#: loss clean whenever anyone had an SSH session open — which is exactly the
#: wrong turn available in this diagnosis.
SHUTDOWN_MARKERS = (
    "reached target power-off",
    "reached target poweroff",
    "reached target reboot",
    "reached target halt",
    "reached target system power off",
    "shutting down",
    "system is powering down",
    "system is rebooting",
    "power down",
    "powering off",
    "restarting system",
    "unmounting file systems",
    "reboot: system halted",
)

#: Only PID 1 and the kernel are believed. ``systemd[1917]`` is a user manager
#: and its shutdown lines are about a login session, not about the machine.
_SYSTEM_EMITTER = re.compile(r"(?:^|\s)(?:kernel|systemd\[1\]|shutdown\[\d+\]):")

JOURNAL_COMMAND = ("journalctl", "-b", "-1", "-n", "40", "--no-pager", "-o", "short-iso")


def is_system_shutdown_line(line: str) -> bool:
    """Whether one journal line is the machine being told to stop.

    Both halves are required. A marker alone matches an SSH session ending; an
    emitter alone matches every kernel message ever printed.
    """
    text = str(line or "")
    if not _SYSTEM_EMITTER.search(text):
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in SHUTDOWN_MARKERS)


def _previous_boot_lines(runner: Callable[..., Any]) -> tuple[List[str], str]:
    try:
        result = runner(
            list(JOURNAL_COMMAND),
            capture_output=True, text=True, check=False, timeout=10,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return [], "journalctl could not be run on this host."
    output = getattr(result, "stdout", "") or ""
    lines = [line for line in output.splitlines() if line.strip()]
    if getattr(result, "returncode", 0) not in (0, None) and not lines:
        return [], (
            "journalctl returned no lines for the previous boot: {}".format(
                (getattr(result, "stderr", "") or "").strip()[:200]
                or "the journal may not be persistent on this host"
            )
        )
    return lines, ""


#: How the previous boot ended cannot change until this one ends, so it is read
#: once per process. Without this the reading would spawn ``journalctl`` on
#: every telemetry poll — once a second per open browser tab — to re-answer a
#: question whose answer is fixed. Mirrors the vendor-tool cache in
#: :mod:`vaelor.platforms.accelerators`, for the same reason.
_CACHED_SHUTDOWN: Optional[Dict[str, Any]] = None


def cached_previous_shutdown(
    runner: Callable[..., Any] = subprocess.run,
) -> Dict[str, Any]:
    """:func:`previous_shutdown`, read once and held for the process."""
    global _CACHED_SHUTDOWN

    if _CACHED_SHUTDOWN is None:
        _CACHED_SHUTDOWN = previous_shutdown(runner)
    return dict(_CACHED_SHUTDOWN)


def reset_previous_shutdown_cache() -> None:
    """Forget the held reading. For tests, and for a re-read after a restart."""
    global _CACHED_SHUTDOWN

    _CACHED_SHUTDOWN = None


def previous_shutdown(
    runner: Callable[..., Any] = subprocess.run,
) -> Dict[str, Any]:
    """Whether the previous boot ended in a shutdown or simply stopped.

    ``state`` is one of:

    * ``clean`` — the previous boot recorded a shutdown, reboot or halt.
    * ``unclean`` — the journal for the previous boot exists and ends with no
      shutdown sequence at all. Nothing asked the machine to stop; power was
      lost, or the kernel died without getting a word out.
    * ``unknown`` — there is no previous boot to read, or no journal retained
      from it. An unanswered question, and never rendered as either outcome.
    """
    lines, reason = _previous_boot_lines(runner)
    record: Dict[str, Any] = {
        "state": "unknown",
        "clean": None,
        "evidence": "",
        "lines_read": len(lines),
        "detail": "",
        "throttled_history_note": THROTTLED_HISTORY_NOTE,
    }
    if not lines:
        record["detail"] = (
            "How the previous boot ended could not be read, so whether this "
            "machine was shut down or lost power is unknown. {}"
        ).format(reason or "No journal was retained from the previous boot.")
        return record
    for line in reversed(lines):
        if is_system_shutdown_line(line):
            record.update({
                "state": "clean",
                "clean": True,
                "evidence": line.strip()[:200],
                "detail": (
                    "The previous boot ended in an orderly shutdown; the "
                    "journal records the system being brought down."
                ),
            })
            return record
    record.update({
        "state": "unclean",
        "clean": False,
        "evidence": lines[-1].strip()[:200],
        "detail": (
            "This machine lost power or stopped without shutting down. The "
            "previous boot's journal ends mid-run with no shutdown sequence: "
            "nothing asked the system to stop, and the last thing it recorded "
            "was \"{}\". A supply that cannot hold the load under peak draw is "
            "the common cause, and it is fixed by a person at the machine "
            "rather than by this appliance."
        ).format(lines[-1].strip()[:120]),
    })
    return record
