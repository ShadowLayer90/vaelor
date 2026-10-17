"""One appliance update at a time (VD-217).

An update must never leave a box half upgraded (VD-217). The root broker
(`appliance_upgrade`) runs each apply on a thread of its own - verify,
install, restart, gate, then the refresh with the release's root builds,
up to about 7.5 minutes - and nothing upstream stopped a second apply in that
window: the route checks only that a release is offered and eligible, and a
same-version re-apply is eligible as ``same-build`` the moment the first
apply has recorded its install. Two overlapping applies could reinstall the
package under each other's restart, or race their rollbacks.

So the broker takes an exclusive ``flock`` on a file in its own root-only
folder (``appliance_upgrade.RETAINED_DIR``, ``root:root 0700``: no other
account can open, replace or unlink it) before it consumes a plan, and holds
it for the whole apply. A second apply is refused at once with
:data:`BUSY` and its plan is left unconsumed. The lock lives on an open file
descriptor: it is released when the apply ends, on every path, and by the
kernel if the broker dies. The descriptor is close-on-exec, so no command the
apply starts (pip, systemctl, the refresh and its builds) can hold it past
the apply.

**The broker that applies an update is the previous release's, still in
memory** (R102), so a lock in its handler protects nothing until the broker
restarts. Two more holders close that gap from the first update on, because
both run from the NEW package:

* ``vaelor-refresh-models --apply`` (`deployment_refresh`), which the
  installer and every release's broker run after their health gate, takes
  the same lock for its build steps and its refresh - waiting at most
  :data:`REFRESH_WAIT_SECONDS`, then refusing with :data:`BUSY` - unless the
  broker that started it already holds it (:data:`HELD_ENV`);
* after an apply ends - its result already written - the broker schedules its
  own restart (:func:`restart_self`) and keeps the lock until systemd stops
  it, so no apply can begin in between; the next update runs the new
  handler.

What stays unguarded is said in :data:`FIRST_UPDATE_UNGUARDED`.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - the broker runs on Linux
    fcntl = None

#: The lock file's name in the broker's root-only folder.
LOCK_NAME = "apply.lock"

#: What a second apply is refused with while one runs.
BUSY = "An update is already being applied; wait for it to finish."

#: What the release notes say about the first update into this release (R102).
FIRST_UPDATE_UNGUARDED = (
    "The first update from the console into this release is applied by the "
    "updater of the release it replaces, which does not take the update lock. "
    "While that one update runs, do not start another update or run the "
    "installer. Its model refresh and build steps already take the lock, and "
    "every update after it is fully protected."
)

#: Set by the broker for the refresh it starts while it holds the lock, so the
#: refresh does not wait on its own parent. Root's environment only: the
#: refresh runs as root from the broker or the installer, never for a client.
HELD_ENV = "VAELOR_APPLY_LOCK_HELD"

#: How long the refresh waits for an apply that holds the lock, then refuses.
#: Inside the 600 s every broker gives it, with the build budget.
REFRESH_WAIT_SECONDS = 60

#: How long a broker that scheduled its own restart keeps the lock if systemd
#: has not stopped it by then.
RESTART_GRACE_SECONDS = 60


def acquire(directory: Path, wait_seconds: float = 0.0,
            sleep: Callable[[float], None] = time.sleep,
            clock: Callable[[], float] = time.monotonic) -> Optional[int]:
    """An open descriptor holding the apply lock, or ``None`` when an apply holds it.

    The folder is made ``0700`` if it is missing; the file is opened without
    following a link, close-on-exec, ``0600``. ``wait_seconds`` retries once a
    second for that long before giving up.
    """
    deadline = clock() + float(wait_seconds)
    while True:
        handle = _try(directory)
        if handle is not None or clock() >= deadline:
            return handle
        sleep(1.0)


def _try(directory: Path) -> Optional[int]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    handle = os.open(str(directory / LOCK_NAME), flags, 0o600)
    if fcntl is None:  # pragma: no cover - no flock: the broker does not run here
        return handle
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(handle)
        return None
    return handle


def release(handle: Optional[int]) -> None:
    """Close the descriptor, which releases the lock. Never raises."""
    if handle is None:
        return
    try:
        os.close(handle)
    except OSError:
        pass


def restart_self(unit: str, run: Callable[..., Any] = subprocess.run) -> bool:
    """Ask systemd to restart this broker's own unit without waiting; whether it was queued."""
    try:
        done = run(["/usr/bin/systemctl", "restart", "--no-block", unit],
                   capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return getattr(done, "returncode", 1) == 0


def start_locked(
    directory: Path, consume: Callable[[], Dict[str, Any]],
    apply: Callable[[Dict[str, Any]], Any],
    start: Callable[[Callable[[], None]], None] = lambda body: threading.Thread(
        target=body, name="vaelor-appliance-upgrade", daemon=False).start(),
    after: Optional[Callable[[], bool]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Take the lock, consume the plan, and run ``apply`` on a thread that releases it.

    Raises ``ValueError(BUSY)`` at once while another apply holds the lock,
    with the plan left unconsumed. A plan that cannot be consumed, or a
    thread that cannot start, releases the lock before the error goes back.
    ``after`` runs once the apply has ended (its result written); when it
    reports a restart queued (:func:`restart_self`), the lock is kept until
    the process is stopped, or :data:`RESTART_GRACE_SECONDS` have passed.
    """
    handle = acquire(directory)
    if handle is None:
        raise ValueError(BUSY)
    try:
        plan = consume()

        def body() -> None:
            try:
                apply(plan)
            finally:
                try:
                    if after is not None and after():
                        sleep(RESTART_GRACE_SECONDS)
                finally:
                    release(handle)

        start(body)
    except BaseException:
        release(handle)
        raise
