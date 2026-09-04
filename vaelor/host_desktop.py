"""Fixed-command broker for native GNOME RDP and optional browser VNC."""

from __future__ import annotations

import json
import os
import re
import socket
import socketserver
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

from .host_desktop_first_run import initial_setup_detail
from .host_desktop_lock import disable_desktop_locking, lock_detail
from .host_desktop_tls import (
    RDP_TLS_CERT,
    RDP_TLS_KEY,
    rdp_certificate_fingerprint,
    install_rdp_tls,
)
from .host_desktop_vnc import (
    MANAGED_DESKTOP_USER,
    VNC_DISPLAY,
    _write_desktop_files,
)
from .platform_drivers import default_platform_drivers
from .runtime_paths import env_value, jobs_group_id, run_path

try:
    import pwd
except ImportError:  # pragma: no cover - deployment target is Linux
    pwd = None

try:
    import grp
except ImportError:  # pragma: no cover
    grp = None


SOCKET_PATH = env_value(
    "VAELOR_HOST_DESKTOP_SOCKET", "PM_HOST_DESKTOP_SOCKET",
    run_path("host-desktop.sock"),
)
MAX_REQUEST_BYTES = 4096
RDP_PORT = 3389
VNC_PORT = 5901
#: How long the browser desktop gets to start accepting connections, whether
#: it is being commissioned or restarted after a session was ended.
#:
#: One number for both, and it is an upper bound rather than a measurement: a
#: warm restart binds in a second or two and a first commission does not, so
#: the generous value only costs time on a machine that is already broken. A
#: second constant for the restart path would be a number nobody measured
#: either, and two unmeasured numbers drift apart (VD-077's shape).
#:
#: It has to stay inside the browser's own budget, because a reply that never
#: arrives is not an honest answer. `apiRequest` aborts a POST at 60 s; the
#: failing end path spends at most this plus the terminate poll, so the owner
#: gets the specific reason rather than a generic timeout.
VNC_LISTEN_SECONDS = 30
MEMORY_POLICY_FILE = Path("/etc/sysctl.d/90-vaelor-memory.conf")
DOCKER_SERVICE_DROP_INS = {
    Path(
        "/etc/systemd/system/"
        "vaelor-workload-executor.service.d/docker.conf"
    ): "[Service]\nSupplementaryGroups=vaelor-credentials docker\n",
    Path(
        "/etc/systemd/system/"
        "vaelor-workload-broker.service.d/docker.conf"
    ): "[Service]\nSupplementaryGroups=vaelor docker\n",
}
RDP_USERNAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$")


def _run(
    command: list[str],
    timeout: int = 900,
    input_text: str | None = None,
) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        input=input_text,
        env={
            **os.environ,
            "DEBIAN_FRONTEND": "noninteractive",
            "APT_LISTCHANGES_FRONTEND": "none",
        },
    )
    if result.returncode:
        raise RuntimeError(
            (result.stderr or result.stdout or "Host desktop setup failed.")[-2000:]
        )
    return result.stdout[-4000:]


def _existing_desktop_user():
    """The managed desktop account if this appliance has one, else ``None``.

    **Looking is separated from creating because one caller must not create.**
    `end_desktop_session` used the resolver below, which runs `useradd` when
    neither account exists — so "End desktop" on an appliance that had never
    commissioned a browser desktop *added a system account* in order to end a
    session that never existed. Asking whether something is there is not a
    reason to bring it into being.

    One name since #168 retired the pre-rename account, and it is still the
    same list here and in `deploy/install-vaelor.sh` — held together by a test
    rather than a shared constant, because bash cannot import one (#166/#168).
    """
    if pwd is None:
        raise RuntimeError("Browser desktop setup requires Linux.")
    try:
        record = pwd.getpwnam(MANAGED_DESKTOP_USER)
    except KeyError:
        return None
    # Refused here rather than in one caller, because every use of this
    # record ends up as `loginctl terminate-user <name>` or `pgrep -u <uid>`,
    # and uid 0 turns both of those into something else entirely.
    if record.pw_uid == 0:
        raise RuntimeError("The managed browser desktop account is invalid.")
    return record


def _managed_desktop_user():
    record = _existing_desktop_user()
    if record is None:
        _run([
            "/usr/sbin/useradd",
            "--create-home",
            "--shell", "/bin/bash",
            "--comment", "Vaelor isolated browser desktop",
            MANAGED_DESKTOP_USER,
        ], timeout=30)
        record = pwd.getpwnam(MANAGED_DESKTOP_USER)
    if record.pw_uid == 0 or not Path(record.pw_dir).is_dir():
        raise RuntimeError("The managed browser desktop account is invalid.")
    return record


def _terminate_managed_desktop_session(record) -> None:
    """End processes escaped into the dedicated account's user manager.

    **`terminate-user` failing is the ordinary case, not an error.** systemd
    exits non-zero when the account has no user manager to terminate, which is
    exactly the state on a first install and after any clean shutdown - so
    routing it through `_run`, which raises on any non-zero exit, aborted the
    whole browser desktop installation on the machines that had nothing wrong
    with them. The account is created a few lines earlier in that path and has
    never logged in.

    What matters is whether processes are still running afterwards, and the
    loop below is the authority on that. The call is best-effort; the check is
    not.
    """
    subprocess.run(
        ["/usr/bin/loginctl", "terminate-user", record.pw_name],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        processes = subprocess.run(
            ["/usr/bin/pgrep", "-u", str(record.pw_uid)],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if processes.returncode == 1:
            return
        if processes.returncode not in {0, 1}:
            raise RuntimeError(
                "The managed browser desktop process state could not be verified."
            )
        time.sleep(0.25)
    raise RuntimeError(
        "The previous managed browser desktop session did not terminate cleanly."
    )


def _wait_for_listener(port: int, seconds: float) -> bool:
    """Whether something accepts a connection on `port` within `seconds`.

    **The one place this module decides that the desktop is up.**
    `commission_vnc` and `end_desktop_session` both have to answer it, and
    before #191 only the first did — the second reported `systemctl restart`'s
    exit status instead, which says the start job completed and nothing about
    a forking Xvnc that died a moment later. Two ways of answering one
    question is LESSONS pattern 6, so there is one.

    It always probes at least once, however small the deadline: a loop written
    as `while monotonic() < deadline` can perform zero attempts and return the
    same answer as a refused connection, which is pattern 8 in one line.
    """
    deadline = time.monotonic() + seconds
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(1)


def rdp_certificate() -> Dict[str, str]:
    """The fingerprint of the certificate this appliance is serving.

    A broker action rather than something the control plane reads, because
    both routes to the value need root: the certificate file is mode 0600
    owned by `gnome-remote-desktop`, and `grdctl --system` is a system-level
    call. `host_desktop_tls` holds the parsing and the reasons.
    """
    return rdp_certificate_fingerprint()


def _verify_rdp_configuration(expected_username: str, expected_password: str) -> None:
    """Verify that grdctl persisted every setting without returning secrets.

    **The output of this command contains the password, so it must never
    leave this function.** It was read through `_run`, which raises
    ``RuntimeError(result.stderr or result.stdout)`` on a non-zero exit - and
    the caller turns a RuntimeError into an HTTP error body. One `grdctl`
    failure that wrote its usage text to stderr *and nothing else* would have
    fallen through to `stdout`, putting `Password: <the owner's password>` on
    the wire and into whatever logs the response. The docstring said "without
    returning secrets" while the error path returned them, which is why it
    survived reading.

    `subprocess` is called directly so the output is bound to a local, matched
    here, and reported only as the booleans below.
    """
    result = subprocess.run(
        ["/usr/bin/grdctl", "--system", "status", "--show-credentials"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    # **A failed instrument is not a failed configuration**, and conflating
    # them was worse than the leak this function was rewritten to close.
    # Treating a non-zero exit as an empty status made every one of the five
    # checks below "missing", so the owner was told remote login, the
    # certificate, the key, the user name and the password had all failed to
    # persist - when grdctl might simply not have reached D-Bus, with the
    # configuration perfectly correct. `stderr` says which it was and was
    # being thrown away; it carries the tool's own diagnostics and never the
    # credentials, which appear on stdout only.
    if result.returncode != 0:
        # Surfacing tool output re-opens the question this function exists to
        # close, so the secret is removed from it by value rather than by
        # assumption. stderr *should* never carry the password - the tool
        # prints credentials on stdout - but "should never" is what put it in
        # the journal in the first place, and a redaction costs nothing.
        detail = " ".join((result.stderr or "").split())[:400]
        if expected_password:
            detail = detail.replace(expected_password, "[redacted]")
        raise RuntimeError(
            "The remote login settings could not be read back, so this "
            "configuration is unverified: grdctl exited {}.{}".format(
                result.returncode, " " + detail if detail else ""
            )
        )
    missing = [
        name
        for name, value in (
            ("remote login is enabled", "Status: enabled"),
            ("the TLS certificate", f"TLS certificate: {RDP_TLS_CERT}"),
            ("the TLS key", f"TLS key: {RDP_TLS_KEY}"),
            ("the user name", f"Username: {expected_username}"),
            # The comparison happens here and the value is never re-emitted.
            ("the password", f"Password: {expected_password}"),
        )
        if value not in result.stdout
    ]
    if missing:
        raise RuntimeError(
            "GNOME Remote Login did not retain {}. No usable RDP setup was "
            "reported.".format(", ".join(missing))
        )


def _set_rdp_credentials(username: str, password: str) -> None:
    """Hand grdctl the password on stdin, never as a command argument.

    **The argument form leaks the password to every reader of the journal.**
    This ran as `grdctl --system rdp set-credentials <user> <password>`, and
    because the call goes through `pkexec`, the whole command line is written
    to the system journal:

        pkexec[...]: root: Executing command ...
        [COMMAND=/usr/bin/grdctl rdp set-credentials vaelor <the password>]

    Found on the appliance 2026-08-11 with the owner's live credential sitting
    in the log in clear text. An argv element is world-readable in
    `/proc/<pid>/cmdline` while the process runs, visible to `ps`, and once
    pkexec has logged it the secret outlives every later rotation. Vaelor
    keeps credentials in the broker precisely so they never travel this way.

    `grdctl` prompts for the password when the argument is omitted, so stdin
    carries it instead: the journal then records the command without the
    secret, and the value exists only in this process's memory and the pipe.

    Two newlines are sent because the tool asks for the password and then for
    a confirmation. `configure_rdp` has already refused any password
    containing a newline, so the input cannot be split into extra answers.

    VD-087 carries the rest, including the part code cannot fix: a journal
    already written has to be rotated and vacuumed by the owner.
    """
    _run(
        ["/usr/bin/grdctl", "--system", "rdp", "set-credentials", username],
        timeout=30,
        input_text="{0}\n{0}\n".format(password),
    )


@contextmanager
def _stopped_for_reconfiguration(unit: str):
    """Stop a unit, run the body, and never leave the unit stopped.

    GNOME Remote Desktop reads its system credentials and TLS material when
    the daemon starts, so updating either while it is active can leave the
    listener online but unable to authenticate old or new credentials. It has
    to come down first — and whatever happens next, it has to go back up.

    **What this guarantees, precisely, because the first version of this
    comment overclaimed and review caught it.** It guarantees the *service is
    running* when this block exits, by success or by failure. It does **not**
    guarantee the configuration is the one that was there before:
    `set-credentials` may already have overwritten the stored password by the
    time a later step fails, so a failure at `enable` or at verification
    leaves the *new* credentials in effect on a running service, not the old
    ones. The owner is not locked out, which is the property worth having on a
    machine reachable only over RDP; they may however need the password they
    just typed rather than the previous one, and the caller's error message
    must not tell them otherwise.

    **The TLS half of that rollback now exists and the credential half still
    does not.** `install_rdp_tls` stages both PEM files in a private directory
    and publishes them only once they are generated, restoring the previous
    pair if the second move fails, so a certificate failure leaves the
    working certificate serving. Snapshotting grdctl's credential store is the
    remaining gap (VD-087); the lockout was the part that could not wait.
    """
    _run(["/usr/bin/systemctl", "stop", unit], timeout=30)
    try:
        yield
    except BaseException:
        # Restoring must not replace the error that explains what went wrong,
        # so its own failure is swallowed deliberately - there is nothing left
        # to try, and the owner needs the original message. `BaseException` is
        # deliberate too: a KeyboardInterrupt mid-reconfiguration must still
        # put the service back, and the bare `raise` re-raises it unchanged.
        try:
            subprocess.run(
                ["/usr/bin/systemctl", "start", unit],
                capture_output=True, text=True, check=False, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        raise


def configure_rdp(username: str, password: str) -> Dict[str, Any]:
    clean_username = str(username or "").strip()
    clean_password = str(password or "")
    if not RDP_USERNAME.fullmatch(clean_username):
        raise ValueError(
            "Use 3–64 letters, numbers, dots, dashes, or underscores for the RDP user name."
        )
    if not 12 <= len(clean_password) <= 128 or "\n" in clean_password or "\0" in clean_password:
        raise ValueError("Use a dedicated RDP password between 12 and 128 characters.")

    with _stopped_for_reconfiguration("gnome-remote-desktop.service"):
        install_rdp_tls(_run)
        commands = [
            ["rdp", "set-port", str(RDP_PORT)],
            ["rdp", "set-auth-methods", "credentials"],
            ["rdp", "set-tls-cert", str(RDP_TLS_CERT)],
            ["rdp", "set-tls-key", str(RDP_TLS_KEY)],
        ]
        for arguments in commands:
            _run(["/usr/bin/grdctl", "--system", *arguments], timeout=30)
        _set_rdp_credentials(clean_username, clean_password)
        _run(["/usr/bin/grdctl", "--system", "rdp", "enable"], timeout=30)
        _verify_rdp_configuration(clean_username, clean_password)
        _run(
            ["/usr/bin/systemctl", "enable", "--now",
             "gnome-remote-desktop.service"],
            timeout=30,
        )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", RDP_PORT), timeout=0.5):
                return {
                    "available": True,
                    "port": RDP_PORT,
                    "username": clean_username,
                    "service": "gnome-remote-desktop.service",
                }
        except OSError:
            time.sleep(1)
    raise RuntimeError("Ubuntu RDP was configured but did not start listening.")


def disable_rdp() -> Dict[str, Any]:
    _run(
        ["/usr/bin/systemctl", "disable", "--now", "gnome-remote-desktop.service"],
        timeout=30,
    )
    _run(["/usr/bin/grdctl", "--system", "rdp", "clear-credentials"], timeout=30)
    _run(["/usr/bin/grdctl", "--system", "rdp", "disable"], timeout=30)
    RDP_TLS_CERT.unlink(missing_ok=True)
    RDP_TLS_KEY.unlink(missing_ok=True)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", RDP_PORT), timeout=0.4):
                time.sleep(0.5)
        except OSError:
            return {"available": False, "port": RDP_PORT, "credentials_cleared": True}
    raise RuntimeError("Ubuntu RDP did not close port 3389 after being disabled.")


def end_desktop_session() -> Dict[str, Any]:
    """End the running browser desktop, and repair what made it unusable.

    **The owner had no way to do this, and needed one.** "Close session" in
    the console only cleared the browser's own state — it set the viewer URL
    to empty and said "Remote desktop session closed" while the session on the
    appliance carried on untouched, so reconnecting landed straight back in
    whatever was wrong with it. On 2026-08-11 that was a GNOME lock screen
    demanding a password for an account that has none; the only route out was
    SSH, which is not a route an owner has.

    Ending it is safe by construction: the desktop is a stated one-use
    session holding no work worth keeping, `tigervncserver` brings a fresh one
    up on the next connection, and `_terminate_managed_desktop_session` is the
    same call commissioning already makes. Restarting the unit rather than
    only killing the user avoids leaving the display socket owned by a session
    that no longer exists.

    **`_disable_desktop_locking` is applied here because commissioning was the
    only place that called it, and commissioning is the one thing an appliance
    with the defect will never do again.** A machine commissioned on alpha 47
    or earlier locks itself out of reach; upgrading brought the repair but no
    way to reach it, since the console replaces "Install browser desktop" with
    static text once the desktop is ready. "End desktop" is the control an
    owner presses when the desktop is wrong, which is exactly the population
    that needs the fix, and the write is five idempotent dconf keys that cost
    nothing to repeat. It is the same function commissioning calls — one
    mechanism, so the two cannot disagree about what "not locked" means
    (LESSONS pattern 6) — and it runs after the session is terminated and
    before the unit restarts, so the next session reads the repaired settings.

    The keys are attempted rather than required: the caller pressed a button
    that means *end this*, and failing to end it because a GNOME schema is
    missing would be the wrong trade. `screen_lock_disabled` says which
    happened, because an owner who is about to meet the lock screen again
    should be told so rather than left to discover it.

    **Three of the four facts in this reply are now read back, and #191 is
    why.** `ended` always was — `_terminate_managed_desktop_session` polls
    `pgrep` and raises unless the account really has no processes left. The
    other two were exit statuses wearing the names of outcomes:
    `screen_lock_disabled` came from `gsettings set` not failing, which a
    dconf lock satisfies while changing nothing, and the restart came from
    `systemctl restart` exiting 0, which a unit that starts and immediately
    dies also satisfies. Both now compare against the machine: the lock keys
    are read back out of the account's dconf, and the desktop is called back
    only when 5901 accepts a connection.

    `service_restarted` stays, and it is deliberately not folded into
    `desktop_listening`. "systemd refused to restart the unit" and "the unit
    restarted and nothing is listening" are different faults with different
    remedies, and collapsing them into one boolean is how they arrive looking
    the same (LESSONS pattern 8).
    """
    service = f"tigervncserver@{VNC_DISPLAY}.service"
    record = _existing_desktop_user()
    if record is None:
        # **Ending nothing must not create something.** This asked
        # `_managed_desktop_user`, which runs `useradd` when neither account
        # exists — so on an appliance that never commissioned a browser
        # desktop, "End desktop" created a system account, restarted a unit
        # that is not installed, and reported `ended: true`. Three untruths in
        # one reply, and the account outlived the request.
        return {
            "ended": False,
            "account": "",
            "service": service,
            "service_restarted": False,
            "desktop_listening": False,
            "screen_lock_disabled": False,
            "detail": (
                "No browser desktop is commissioned on this appliance, so "
                "there was no desktop session to end."
            ),
        }
    _terminate_managed_desktop_session(record)
    try:
        lock_verdict = disable_desktop_locking(record)
        lock_message = lock_detail(lock_verdict)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        lock_verdict = {"confirmed": False, "unconfirmed": []}
        lock_message = (
            "The desktop was ended, but its screen lock could not be "
            "disabled: {}".format(str(error)[:200])
        )
    # The restart is best-effort — a commissioned account whose unit is
    # missing or masked is a partly-built install, not a reason to refuse to
    # end the session that was just terminated. Its outcome is reported rather
    # than discarded, so "restarted" and "there was no unit to restart" do not
    # arrive looking the same (LESSONS pattern 8).
    restart = subprocess.run(
        ["/usr/bin/systemctl", "restart", service],
        capture_output=True, text=True, check=False, timeout=60,
    )
    # Probed whatever the restart said, because the restart's exit status is
    # the claim under test. Gating the readback on it would put the answer
    # back inside the thing being checked.
    listening = _wait_for_listener(VNC_PORT, VNC_LISTEN_SECONDS)
    details = [detail for detail in (
        lock_message,
        "" if listening else (
            "The desktop was ended, but the appliance did not bring a fresh "
            "one back up: nothing is accepting connections on port {}. "
            "Reinstalling the browser desktop is the way back.".format(VNC_PORT)
        ),
    ) if detail]
    return {
        "ended": True,
        "account": record.pw_name,
        "service": service,
        "service_restarted": restart.returncode == 0,
        "desktop_listening": listening,
        "screen_lock_disabled": bool(lock_verdict.get("confirmed")),
        "detail": " ".join(details),
    }


def commission_vnc() -> Dict[str, Any]:
    drivers = default_platform_drivers()
    os_info = drivers["operating_system"].snapshot()
    package_manager = drivers["package_manager"]
    if not package_manager.installation_capability(os_info)["available"]:
        raise RuntimeError(
            "Automatic browser desktop installation is not available for "
            "this operating system."
        )
    record = _managed_desktop_user()
    _run(package_manager.update_command())
    _run(package_manager.install_command([
        "tigervnc-standalone-server", "gnome-session-flashback",
    ]))
    service = f"tigervncserver@{VNC_DISPLAY}.service"
    _run(["/usr/bin/systemctl", "disable", "--now", service], timeout=30)
    _terminate_managed_desktop_session(record)
    written = _write_desktop_files(record)
    first_run = written["initial_setup"]
    # #191. The verdict was discarded here, so commissioning applied the
    # screen-lock repair and then reported `available: True` without ever
    # saying whether it held - the exact readback `end_desktop_session`
    # surfaces, unsurfaced on the path that installs the desktop. A dconf lock
    # accepts every `gsettings set` and changes nothing (LESSONS pattern 1), so
    # "no exception" is not "the lock is off"; the verdict is read and carried
    # into the reply below rather than thrown away (LESSONS pattern 11).
    lock_verdict = disable_desktop_locking(record)
    legacy_unit = Path("/etc/systemd/system/pironman-host-vnc.service")
    if legacy_unit.exists():
        _run(
            ["/usr/bin/systemctl", "disable", "--now", legacy_unit.name],
            timeout=30,
        )
        legacy_unit.unlink()
    _run(["/usr/bin/systemctl", "daemon-reload"], timeout=30)
    _run(["/usr/bin/systemctl", "enable", "--now", service])
    if not _wait_for_listener(VNC_PORT, VNC_LISTEN_SECONDS):
        raise RuntimeError("TigerVNC was installed but did not start listening.")
    return {
        "available": True,
        "port": VNC_PORT,
        "desktop_user": record.pw_name,
        "service": service,
        "session": "gnome-flashback-metacity",
        "initial_setup_suppressed": bool(first_run.get("confirmed")),
        "screen_lock_disabled": bool(lock_verdict.get("confirmed")),
        "retained_paths": written["retained_paths"],
        "detail": " ".join(sentence for sentence in (
            initial_setup_detail(first_run),
            # The keys are named rather than counted, because a dconf lock and a
            # missing schema are the two ways this reads back unconfirmed and
            # they have different remedies (the same reason `lock_detail` names
            # them for the end path). The wording is commission's own, not
            # `lock_detail`'s "was ended" - a shared sentence would be a label
            # that does not match the event (LESSONS pattern 5).
            "" if lock_verdict.get("confirmed") else (
                "Vaelor could not confirm the browser desktop's screen lock is "
                "off ({}); the session may reach a password prompt that this "
                "passwordless account cannot answer.".format(
                    ", ".join(lock_verdict.get("unconfirmed") or [])
                )
            ),
            # Named, not counted: the owner's only useful next step is to look
            # at the directory, and "1 path was retained" does not say which.
            "Vaelor left a previous VNC configuration in place because it did "
            "not write it: {}.".format(", ".join(written["retained_paths"]))
            if written["retained_paths"] else "",
        ) if sentence),
    }


def install_docker() -> Dict[str, Any]:
    """Install the distribution Docker packages on validated Debian-family hosts."""

    drivers = default_platform_drivers()
    os_info = drivers["operating_system"].snapshot()
    package_manager = drivers["package_manager"]
    if not package_manager.installation_capability(os_info)["available"]:
        raise RuntimeError(
            "Automatic Docker installation is not available for this operating system."
        )
    _run(package_manager.update_command())
    _run(package_manager.install_command(["docker.io", "docker-compose-v2"]))
    _run(["/usr/bin/systemctl", "enable", "--now", "docker.service"], timeout=60)
    service_user = env_value(
        "VAELOR_SERVICE_USER", "PM_DASHBOARD_USER", "vaelor"
    )
    service_users = (service_user, "vaelor-workloads")
    if any(
        not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", user)
        for user in service_users
    ):
        raise RuntimeError("A Vaelor service account is invalid.")
    for user in service_users:
        _run(["/usr/sbin/usermod", "-aG", "docker", user], timeout=30)
    for destination, content in DOCKER_SERVICE_DROP_INS.items():
        destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        os.chmod(destination, 0o644)
    version = _run(
        ["/usr/bin/docker", "compose", "version", "--short"], timeout=30
    ).strip()[:64]
    # Refresh supplementary groups after the job result has been persisted.
    if Path("/usr/bin/systemd-run").is_file():
        _run(["/usr/bin/systemctl", "daemon-reload"], timeout=30)
        _run([
            "/usr/bin/systemd-run", "--unit=vaelor-docker-activation",
            "--on-active=8s", "/usr/bin/systemctl", "restart",
            "vaelor-workload-executor.service",
            "vaelor-workload-broker.service",
            "vaelor-control-plane.service",
        ], timeout=30)
    return {
        "installed": True,
        "compose": True,
        "compose_version": version,
        "service": "docker.service",
        "service_accounts": list(service_users),
        "activation_restart_scheduled": True,
    }


def repair_docker() -> Dict[str, Any]:
    """Restart the Docker runtime so it rebuilds a wiped/broken data-root.

    The one remedy for a daemon that answers but cannot create containers (its
    ``/var/lib/docker`` gone from under it) is to restart it: dockerd recreates
    the storage skeleton on start. The command is FIXED and takes no
    request-supplied parameter, so there is no injection surface - containerd
    first, then docker, matching ``deploy/install-vaelor.sh`` ``ensure_docker_ready``.
    """
    _run(
        [
            "/usr/bin/systemctl",
            "restart",
            "containerd.service",
            "docker.service",
        ],
        timeout=120,
    )
    return {"repaired": True}


def optimize_memory(profile: str) -> Dict[str, Any]:
    """Apply one reviewed Linux VM policy without accepting raw sysctl input."""
    profiles = {
        "balanced": 60,
        "ai-latency": 10,
        "capacity": 100,
    }
    if profile not in profiles:
        raise ValueError("Choose a supported memory profile.")
    if not Path("/proc/sys/vm/swappiness").is_file():
        raise RuntimeError("Linux memory tuning is not available on this host.")
    swappiness = profiles[profile]
    MEMORY_POLICY_FILE.write_text(
        "# Managed by Vaelor. Change through the control plane.\n"
        "vm.swappiness={}\n".format(swappiness),
        encoding="utf-8",
    )
    os.chmod(MEMORY_POLICY_FILE, 0o644)
    sysctl = next(
        (path for path in ("/usr/sbin/sysctl", "/sbin/sysctl") if Path(path).is_file()),
        "",
    )
    if not sysctl:
        raise RuntimeError("The operating system does not provide sysctl.")
    _run([sysctl, "-w", "vm.swappiness={}".format(swappiness)], timeout=30)
    try:
        observed = int(Path("/proc/sys/vm/swappiness").read_text().strip())
    except (OSError, ValueError) as error:
        raise RuntimeError("The active Linux memory policy could not be verified.") from error
    if observed != swappiness:
        raise RuntimeError(
            "Linux kept swappiness at {} instead of the requested {}.".format(
                observed, swappiness
            )
        )
    return {
        "profile": profile,
        "swappiness": swappiness,
        "persistent": True,
        "configuration": str(MEMORY_POLICY_FILE),
        "reboot_required": False,
    }


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        try:
            if len(raw) > MAX_REQUEST_BYTES:
                raise ValueError("Host desktop request is too large.")
            request = json.loads(raw.decode("utf-8"))
            action = request.get("action")
            if action == "configure_rdp":
                result = {
                    "ok": True,
                    "result": configure_rdp(
                        request.get("username", ""),
                        request.get("password", ""),
                    ),
                }
            elif action == "disable_rdp":
                result = {"ok": True, "result": disable_rdp()}
            elif action == "rdp_certificate":
                result = {"ok": True, "result": rdp_certificate()}
            elif action == "end_desktop_session":
                result = {"ok": True, "result": end_desktop_session()}
            elif action == "commission_vnc":
                result = {"ok": True, "result": commission_vnc()}
            elif action == "install_docker":
                result = {"ok": True, "result": install_docker()}
            elif action == "repair_docker":
                result = {"ok": True, "result": repair_docker()}
            elif action == "optimize_memory":
                result = {
                    "ok": True,
                    "result": optimize_memory(str(request.get("profile", ""))),
                }
            else:
                raise ValueError("Host desktop request is invalid.")
        # **`subprocess.SubprocessError` was missing, and that is the one an
        # owner meets.** `TimeoutExpired` inherits from it, not from `OSError`,
        # so a helper that hit its timeout escaped this handler entirely: the
        # response was never written, the client's `json.loads("")` raised, and
        # the owner was shown `Expecting value: line 1 column 1 (char 0)` as a
        # 400 — a JSON parser's complaint standing in for "the machine took too
        # long". Every call here has a timeout, so this was always reachable;
        # the RDP work added direct `subprocess.run` calls outside `_run`, which
        # made it more so.
        except (
            KeyError, OSError, RuntimeError, ValueError,
            json.JSONDecodeError, subprocess.SubprocessError,
        ) as error:
            message = str(error)[:1000] or error.__class__.__name__
            if isinstance(error, subprocess.TimeoutExpired):
                # Its `str()` embeds the command, and `.output` can carry
                # whatever the tool printed - including, on the credential
                # path, the password. Neither goes back to the caller.
                message = (
                    "A host setup step took too long and was stopped. The "
                    "appliance may still be finishing it; check the remote "
                    "access settings before trying again."
                )
            result = {"ok": False, "error": message}
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


class HostDesktopClient:
    def __init__(self, socket_path: str = SOCKET_PATH, timeout: int = 1900):
        self.socket_path = socket_path
        self.timeout = timeout

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(self.socket_path)
            connection.sendall(
                json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
            )
            response = connection.makefile("rb").readline(64 * 1024)
        payload = json.loads(response.decode("utf-8"))
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error") or "Host desktop setup failed.")
        return payload["result"]

    def configure_rdp(self, username: str, password: str) -> Dict[str, Any]:
        return self._request({
            "action": "configure_rdp",
            "username": username,
            "password": password,
        })

    def disable_rdp(self) -> Dict[str, Any]:
        return self._request({"action": "disable_rdp"})

    def rdp_certificate(self) -> Dict[str, Any]:
        return self._request({"action": "rdp_certificate"})

    def commission_vnc(self) -> Dict[str, Any]:
        return self._request({"action": "commission_vnc"})

    def end_desktop_session(self) -> Dict[str, Any]:
        return self._request({"action": "end_desktop_session"})

    def install_docker(self) -> Dict[str, Any]:
        return self._request({"action": "install_docker"})

    def repair_docker(self) -> Dict[str, Any]:
        return self._request({"action": "repair_docker"})

    def optimize_memory(self, profile: str) -> Dict[str, Any]:
        return self._request({"action": "optimize_memory", "profile": profile})

    # Compatibility for already-queued browser-desktop jobs.
    def commission(self) -> Dict[str, Any]:
        return self.commission_vnc()


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
