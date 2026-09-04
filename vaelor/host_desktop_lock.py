"""Stopping the managed browser desktop from locking itself out of reach.

**Every function here runs inside the privileged host-desktop broker, as
root**, and writes into the dedicated desktop account's own dconf through
`runuser`. Naming the process that runs what is the cheap habit LESSONS
pattern 14 asks for, and it is why nothing here is reachable from the Flask
process except through the broker's unix socket.

Split out of `host_desktop.py` so the readback below has room to explain
itself without pushing that module past the 1,000-line ceiling — the same
reason `host_desktop_tls.py` exists.

**Why there is a readback at all (#191).** The first version ran `gsettings
set` five times and reported success when the exit status was zero. A zero
from `gsettings set` means the tool was invoked and parsed its arguments; it
is not evidence that the value is now in force. A dconf lock under
`/etc/dconf/db/*/locks` makes the write a no-op and still exits 0, and so does
a write that lands in a profile the session does not read. So the appliance
could tell the owner the screen lock was disabled, and the owner would meet
GNOME's "Password:" prompt on an account with no password — the exact defect
this table exists to prevent, reported as fixed. That is LESSONS pattern 1,
and the answer pattern 1 gives is the one used here: after setting anything,
read back what you actually got.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any, Dict, List

#: What the managed desktop must not do, and the value it must hold instead.
#:
#: **A lock screen on this account is a door with no key.** The dedicated
#: desktop account is created with no password — `passwd -S` reports `L`, and
#: that is deliberate, because nobody should be able to log into it directly.
#: GNOME Flashback still starts `gsd-screensaver-proxy` and still locks on
#: idle, so the appliance's own browser desktop reached a prompt reading
#: "Password:" that **no password on earth satisfies**. Observed on the Pi
#: 2026-08-11: a connected, healthy noVNC session, permanently unusable.
#:
#: Removing the lock is not a weakening. The session is bound to localhost,
#: reached only through Vaelor's authenticated gateway and its TLS proxy, and
#: is advertised as a one-use session; the gate is the control plane's own
#: login. A second lock inside it protects nothing and denies everything.
#:
#: This is also the only list of these keys. The readback compares against the
#: value written here rather than against a second table of expected outputs,
#: because two hand-maintained lists of one fact is LESSONS pattern 6 and the
#: reason `_settings_value` normalises instead.
DESKTOP_LOCK_SETTINGS = (
    ("org.gnome.desktop.screensaver", "lock-enabled", "false"),
    ("org.gnome.desktop.screensaver", "idle-activation-enabled", "false"),
    ("org.gnome.desktop.session", "idle-delay", "0"),
    ("org.gnome.desktop.lockdown", "disable-lock-screen", "true"),
    ("org.gnome.settings-daemon.plugins.power",
     "sleep-inactive-ac-type", "nothing"),
)

#: `gsettings get` prints a GVariant, `gsettings set` takes a bare literal, and
#: they do not spell the same value the same way: `idle-delay` reads back as
#: `uint32 0` and `sleep-inactive-ac-type` as `'nothing'`. Stripping the type
#: tag and the quotes is what lets one table serve both directions.
_GVARIANT_TYPE_TAG = re.compile(
    r"^(?:u?int(?:16|32|64)|byte|double|handle)\s+"
)


def _settings_value(raw: str) -> str:
    """`gsettings get` output reduced to the literal `gsettings set` was given."""
    text = _GVARIANT_TYPE_TAG.sub("", raw.strip())
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1]
    return text


def _gsettings(record, *arguments: str) -> subprocess.CompletedProcess:
    """Run one `gsettings` verb inside the desktop account's own dconf.

    `dbus-run-session` gives `gsettings` a bus without needing a live session,
    so this works at commission time before anyone has connected. Both the
    write and the readback are built here so they cannot drift into addressing
    two different accounts, which would make the readback agree with itself
    and nothing else.
    """
    return subprocess.run(
        [
            "/usr/sbin/runuser", "-u", record.pw_name, "--",
            "dbus-run-session", "--",
            "gsettings", *arguments,
        ],
        capture_output=True, text=True, check=False, timeout=60,
    )


def disable_desktop_locking(record) -> Dict[str, Any]:
    """Write the lock settings, then report which ones are actually in force.

    Written into the account's own dconf, so it survives a session restart and
    applies to every later session without a system-wide dconf profile — one
    of those would reach the owner's real desktop too, which is not ours to
    change.

    **Two different failures, deliberately kept apart.** Raising is reserved
    for the case where not one of the five could be *written*: that is a
    machine where this mechanism does not work at all, and commissioning
    should stop rather than hand over a desktop that will lock itself shut.
    A key that was written but does not read back is reported instead, because
    aborting an install over an unconfirmable reading would be a new way to
    fail on machines that were working — and a GNOME build that has dropped
    one schema must still be able to commission a usable desktop.

    So the caller gets a verdict rather than an absence of an exception:
    `confirmed` is true only when every key was read back holding the value
    this module asked for. "No exception" used to stand in for that, and it
    covered a machine where four of the five had failed.
    """
    unwritten: List[str] = []
    unconfirmed: List[str] = []
    for schema, key, value in DESKTOP_LOCK_SETTINGS:
        name = "{} {}".format(schema, key)
        if _gsettings(record, "set", schema, key, value).returncode != 0:
            unwritten.append(name)
            unconfirmed.append(name)
            continue
        readback = _gsettings(record, "get", schema, key)
        if readback.returncode != 0 or _settings_value(readback.stdout) != value:
            unconfirmed.append(name)
    if len(unwritten) == len(DESKTOP_LOCK_SETTINGS):
        raise RuntimeError(
            "The browser desktop's screen lock could not be disabled, so the "
            "session would lock itself and no password exists to unlock it."
        )
    return {"confirmed": not unconfirmed, "unconfirmed": unconfirmed}


def lock_detail(verdict: Dict[str, Any]) -> str:
    """The sentence an owner gets when the lock is not provably gone.

    Naming the keys rather than saying "some settings failed", because the
    owner's next step differs entirely between a missing schema and a dconf
    lock, and the key name is the only thing that tells them apart.
    """
    if verdict.get("confirmed"):
        return ""
    return (
        "The desktop was ended, but its screen lock is not confirmed "
        "disabled: {} did not read back as set, so this desktop may lock "
        "itself again.".format(", ".join(verdict.get("unconfirmed") or []))
    )
