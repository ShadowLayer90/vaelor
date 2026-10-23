"""The settings Vaelor gives InfluxDB, applied by the installer and by every update (VD-133 1.5.5 #13).

InfluxDB 1.x is installed with two defaults Vaelor overrides through one
systemd drop-in, never by editing the packaged ``influxdb.conf`` (a package
upgrade would silently undo that):

* ``[http] bind-address`` is ``:8086`` - every interface, with no
  authentication - so the API is locked to loopback (Vaelor's own client only
  ever connects to ``127.0.0.1:8086``).
* ``[http] log-enabled`` is true, so InfluxDB writes one journal line per HTTP
  request. The 1.5.4 cold install measured about a million lines a day on the
  controller, from telemetry writes alone. Request logging is turned off;
  InfluxDB's own errors and start-up lines still reach the journal, because
  they come from its logger, not from the request log.
* ``[data] query-log-enabled`` is true, so InfluxDB 1.12 also writes one
  ``msg="Executing query"`` line per query - about 195 a minute (~280k a day)
  measured on the Z2 once the request log was off (VD-133 1.5.5 #13, live:
  query log). It is turned off too; a query that fails still answers its
  caller with the error, and InfluxDB's own errors still reach the journal.

**One text, two paths (VD-217, LESSONS 6).** `deploy/install-vaelor.sh` writes
:data:`DROP_IN_TEXT` before its first InfluxDB start (a test holds the two
equal), and :data:`post_install_steps.STEPS` runs ``apply`` here on every
install and every console update, so a box updated from an older release gets
the setting without re-running the installer.

**Done means the running InfluxDB has it, not that a file was written**
(LESSONS 1). ``apply`` checks every ``systemctl`` it runs, reads the running
daemon's own environment, restarts only an InfluxDB that lacks the settings,
and then proves the restarted one answers on loopback the way the installer's
``influxdb_verify`` does. A restart that failed is therefore retried by the
next update, rather than read back as current from the file alone.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

#: The drop-in's path. Its name predates the log setting and is kept, so a box
#: installed earlier has its one file rewritten rather than a second beside it.
DROP_IN_PATH = Path("/etc/systemd/system/influxdb.service.d/vaelor-localhost.conf")

#: The settings, as the environment the running daemon must carry. InfluxDB 1.x
#: reads ``INFLUXDB_<SECTION>_<KEY>``.
SETTINGS = (
    "INFLUXDB_HTTP_BIND_ADDRESS=127.0.0.1:8086",
    "INFLUXDB_HTTP_LOG_ENABLED=false",
    "INFLUXDB_DATA_QUERY_LOG_ENABLED=false",
)

#: The whole drop-in.
DROP_IN_TEXT = "[Service]\n" + "".join("Environment={}\n".format(s) for s in SETTINGS)

SERVICE = "influxdb.service"
SYSTEMCTL = "/usr/bin/systemctl"
PING_URL = "http://127.0.0.1:8086/ping"

#: Each wait's own bound, and the whole of ``apply`` at its worst: a reload, a
#: restart, the wait for InfluxDB to answer, and the reads around them. The
#: post-install step's bound (`post_install_steps.STEPS`) is at least this.
RELOAD_SECONDS = 20
READ_SECONDS = 10
#: How long a restarted InfluxDB gets to answer, as the installer's
#: ``influxdb_verify`` gives it (90 s). The restart itself is queued
#: (``--no-block``), so the wait is this loop's and never a hung systemctl's.
ANSWER_SECONDS = 90
WORST_CASE_SECONDS = RELOAD_SECONDS + ANSWER_SECONDS + 4 * READ_SECONDS

#: What ``apply`` prints, one line, for the update panel and the journal.
NOT_INSTALLED = "InfluxDB is not installed here; nothing to set"
CURRENT = "InfluxDB settings (loopback only, no request or query log): done"
CURRENT_STOPPED = ("InfluxDB settings (loopback only, no request or query log): done; InfluxDB is "
                   "stopped and reads them when it starts")
CURRENT_UNREAD = ("InfluxDB settings (loopback only, no request or query log): done; the running "
                  "InfluxDB's own settings could not be read to confirm them")
UPDATED = "InfluxDB settings (loopback only, no request or query log): applied and InfluxDB restarted"
STARTING = ("InfluxDB settings (loopback only, no request or query log): applied; InfluxDB is still "
            "starting and had not answered within {} s".format(ANSWER_SECONDS))
NOT_ROOT = "InfluxDB settings: not applied, this must run as root"


class ApplyFailed(RuntimeError):
    """A step of ``apply`` failed; its message says which, for the update panel."""


def _write(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` in one rename, mode 0644, parent 0755."""
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    handle, staging = tempfile.mkstemp(dir=str(path.parent), prefix=".vaelor-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as output:
            output.write(text)
        os.chmod(staging, 0o644)
        os.replace(staging, str(path))
    except BaseException:
        try:
            os.unlink(staging)
        except OSError:
            pass
        raise


def _systemctl(run: Callable[..., object], arguments: Sequence[str], seconds: int) -> str:
    """Run one ``systemctl``; its output, or :class:`ApplyFailed` naming it."""
    result = run([SYSTEMCTL, *arguments], check=False, timeout=seconds,
                 capture_output=True, text=True)
    if getattr(result, "returncode", 1) != 0:
        detail = " ".join(str(getattr(result, "stderr", "") or "").split())[:160]
        raise ApplyFailed("systemctl {} failed{}".format(
            " ".join(arguments), ": " + detail if detail else ""))
    return str(getattr(result, "stdout", "") or "")


def _service_fields(run: Callable[..., object]) -> dict:
    shown = _systemctl(run, ("show", SERVICE, "-p", "MainPID", "-p", "NeedDaemonReload",
                             "-p", "ActiveState"), READ_SECONDS)
    return {key: value.strip() for key, value in
            (line.split("=", 1) for line in shown.splitlines() if "=" in line)}


def _service_state(run: Callable[..., object]) -> Tuple[int, bool]:
    """``(MainPID, NeedDaemonReload)`` of the InfluxDB unit, as systemd reports them."""
    fields = _service_fields(run)
    try:
        pid = int(fields.get("MainPID", "0") or 0)
    except ValueError:
        raise ApplyFailed("systemd reported no readable MainPID for {}".format(SERVICE))
    return pid, fields.get("NeedDaemonReload", "") == "yes"


def running_has_settings(pid: int, proc_root: Path = Path("/proc")) -> Optional[bool]:
    """Whether process ``pid`` runs with every :data:`SETTINGS`; ``None`` if unreadable."""
    try:
        environ = (proc_root / str(pid) / "environ").read_bytes()
    except OSError:
        return None
    present = set(environ.decode("utf-8", "replace").split("\0"))
    return all(setting in present for setting in SETTINGS)


def _ping_once() -> bool:
    try:
        with urllib.request.urlopen(PING_URL, timeout=2) as response:
            return response.status == 204
    except (OSError, urllib.error.URLError, ValueError):
        return False


def _answers(ping: Callable[[], bool], clock: Callable[[], float],
             sleep: Callable[[float], None]) -> bool:
    """Whether InfluxDB answers its ping within :data:`ANSWER_SECONDS`; asks at least once."""
    deadline = clock() + ANSWER_SECONDS
    while True:
        if ping():
            return True
        if clock() >= deadline:
            return False
        sleep(1)


def apply(
    *, path: Path = DROP_IN_PATH,
    which: Callable[[str], Optional[str]] = shutil.which,
    run: Callable[..., object] = subprocess.run,
    proc_root: Path = Path("/proc"),
    ping: Callable[[], bool] = _ping_once,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Bring the drop-in AND the running InfluxDB to :data:`SETTINGS`; one line, or ApplyFailed."""
    if which("influxd") is None:
        return NOT_INSTALLED
    try:
        current = path.read_text(encoding="utf-8")
    except OSError:
        current = None
    changed = current != DROP_IN_TEXT
    if changed:
        _write(path, DROP_IN_TEXT)
    pid, needs_reload = _service_state(run)
    if changed or needs_reload:
        _systemctl(run, ("daemon-reload",), RELOAD_SECONDS)
    if pid <= 0:
        return CURRENT_STOPPED
    live = running_has_settings(pid, proc_root)
    if not changed and live is None:
        return CURRENT_UNREAD
    if not changed and live:
        return CURRENT
    _systemctl(run, ("restart", "--no-block", SERVICE), READ_SECONDS)
    if not _answers(ping, clock, sleep):
        # Applied either way; only a unit systemd calls failed is a failure.
        if _service_fields(run).get("ActiveState") == "failed":
            raise ApplyFailed("InfluxDB failed to start after its restart "
                              "(journalctl -u influxdb says why)")
        return STARTING
    pid, _ = _service_state(run)
    if pid > 0 and running_has_settings(pid, proc_root) is False:
        raise ApplyFailed("InfluxDB restarted but still runs without its loopback and "
                          "request- and query-log settings")
    return UPDATED


def main(argv: Sequence[str] = ()) -> int:
    """``apply``, as a post-install step. Prints one line; never fails an update.

    A failure exits 1, so the step reads as held back on the update panel
    (`post_install_steps.step_status`) rather than done; the runner never lets
    a step's status stop the update, and the next update tries again.
    """
    arguments = list(argv) if argv else sys.argv[1:]
    if arguments != ["apply"]:
        sys.stderr.write("usage: python -I -m vaelor.influxdb_config apply\n")
        return 2
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None or geteuid() != 0:
        print(NOT_ROOT)
        return 0
    try:
        print(apply())
    except (ApplyFailed, OSError, subprocess.SubprocessError) as error:
        print("InfluxDB settings: not applied ({})".format(
            str(error)[:200] or type(error).__name__))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - run as a post-install step
    sys.exit(main())
