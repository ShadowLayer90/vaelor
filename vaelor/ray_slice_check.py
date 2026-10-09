"""Vaelor's slice check, run by systemd at every start of a split's Ray unit (ACC-187).

The split's firewall matches sockets by the systemd slice its containers run
in. That holds only while Docker uses the systemd cgroup driver and the
container really is in the slice - facts a deploy confirmed once, and a reboot,
a Docker reconfiguration or a unit restart can change. So each Ray unit runs
this program twice, from its own unit file:

* ``ExecStartPre=... driver`` - Docker's cgroup driver is exactly ``systemd``,
  before the container starts;
* ``ExecStartPost=... container <name> <slice path>`` - the container's own
  cgroup (``/proc/<pid>/cgroup``) lies under the slice, the whole check
  bounded at :data:`CONTAINER_CHECK_BOUND_SECONDS`.

A failing check exits non-zero, systemd fails the unit, the unit's
``ExecStopPost=-docker rm -f`` removes the container (systemd runs no
``ExecStop`` after a failed start), and the split's firewall stays. The program is staged by Vaelor as a
root-only file beside the split's token (`gpu_ray_plane.SLICE_CHECK_PATH`) and
uses only the standard library, so it runs on any machine of the split.
"""

import re
import subprocess
import sys
import time

DOCKER = "/usr/bin/docker"
#: How long the post-start check waits for the container to have a process,
#: how long one ``docker inspect`` may take, and so the most the whole check
#: can take (review 3): the last inspect may start just before the wait ends.
#: Short: systemd counts it against the unit's start timeout.
CONTAINER_WAIT_SECONDS = 15
INSPECT_TIMEOUT_SECONDS = 3
CONTAINER_CHECK_BOUND_SECONDS = CONTAINER_WAIT_SECONDS + 1 + INSPECT_TIMEOUT_SECONDS
#: The driver check's one ``docker info``, and systemd's default start timeout
#: (``DefaultTimeoutStartSec``) that both checks must stay well inside.
DRIVER_TIMEOUT_SECONDS = 10
SYSTEMD_START_TIMEOUT_SECONDS = 90
_CONTAINER = re.compile(r"vaelor-vllm-[a-z0-9][a-z0-9-]{0,50}")
#: The two Ray container roles a split runs (`gpu_pool_units`).
_ROLES = ("-server", "-ray-worker")
_SLICE_PATH = re.compile(r"vaelor\.slice/vaelor-ray\.slice/vaelor-ray-[a-z0-9][a-z0-9_]{0,38}\.slice")


def _docker(argv):
    """``(exit code, stdout)`` of one fixed docker command, bounded."""
    timeout = INSPECT_TIMEOUT_SECONDS if argv[:1] == ["inspect"] else DRIVER_TIMEOUT_SECONDS
    try:
        completed = subprocess.run(
            [DOCKER, *argv], capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""
    return completed.returncode, completed.stdout


def _read(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read(65536)
    except OSError:
        return ""


def check_driver(run):
    """0 when Docker's cgroup driver is exactly ``systemd``, else 1."""
    code, out = run([DOCKER, "info", "--format", "{{.CgroupDriver}}"])
    driver = str(out or "").strip()
    if code == 0 and driver == "systemd":
        return 0
    print("Docker uses the '{}' cgroup driver; a split's firewall needs systemd.".format(
        driver or "unknown"), file=sys.stderr)
    return 1


def check_container(run, read, name, slice_path, *, sleep, monotonic,
                    wait=CONTAINER_WAIT_SECONDS):
    """0 when container ``name`` runs under ``slice_path``, else 1."""
    end = monotonic() + wait
    while True:
        code, out = run([DOCKER, "inspect", "--format", "{{.State.Pid}}", name])
        pid = str(out or "").strip()
        if code == 0 and pid.isdigit() and pid != "0":
            for line in str(read("/proc/{}/cgroup".format(pid)) or "").splitlines():
                if line.startswith("0::"):
                    if line[3:].strip().startswith("/{}/".format(slice_path)):
                        return 0
                    print("{} runs in {}, outside its split's slice.".format(
                        name, line[3:].strip()), file=sys.stderr)
                    return 1
        # Never sleep past the deadline, and never start an inspect at or
        # after it: the last one starts before the wait ends (review 4).
        left = end - monotonic()
        if left > 0:
            sleep(min(1, left))
        if monotonic() >= end:
            print("{} never showed a running process.".format(name), file=sys.stderr)
            return 1


def main(argv, *, run=None, read=_read, sleep=time.sleep, monotonic=time.monotonic):
    """The command line systemd runs; 2 for any argument of another shape."""
    run = run or (lambda full: _docker(full[1:]))
    if argv == ["driver"]:
        return check_driver(run)
    if (len(argv) == 3 and argv[0] == "container" and _CONTAINER.fullmatch(argv[1])
            and argv[1].endswith(_ROLES)
            and _SLICE_PATH.fullmatch(argv[2])):
        return check_container(run, read, argv[1], argv[2], sleep=sleep, monotonic=monotonic)
    print("usage: driver | container <name> <slice path>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
