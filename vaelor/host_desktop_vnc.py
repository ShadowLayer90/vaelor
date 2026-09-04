"""Authoring the managed browser desktop's own TigerVNC session files.

**Every function here runs inside the privileged host-desktop broker, as
root**, writing into the dedicated desktop account's home and into the system
TigerVNC user map. Naming the process that runs what is the cheap habit LESSONS
pattern 14 asks for, and it is why the chowns below are not optional: root
writing into another account's `~/.config` leaves files that account cannot
read, which is LESSONS pattern 13's exact shape.

Split out of `host_desktop.py` for the reason `host_desktop_lock.py`,
`host_desktop_tls.py` and `host_desktop_first_run.py` were: that module sat at
the 1,000-line ceiling, and the file-authoring concern — recognising a config
Vaelor wrote, replacing it, refusing to write through a planted symlink, and
reporting what it declined to touch — is a seam with its own long reasons.

This module is a leaf: it imports the first-run suppression it composes and
nothing from `host_desktop`, so the account name and display it needs are
owned here and imported back up, one direction only. `host_desktop` owns the
broker socket, the account resolution, and the service lifecycle.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from .host_desktop_first_run import suppress_initial_setup

try:
    import pwd
except ImportError:  # pragma: no cover - deployment target is Linux
    pwd = None


VNC_DISPLAY = ":1"
MANAGED_DESKTOP_USER = "vaelor-desktop"
#: TigerVNC's system map of display to account, which is where `:1` is bound
#: to the managed user. Named here rather than written inline so
#: `_write_desktop_files` can be driven against a real directory in a test —
#: the alternative is a test that stubs the writer and then has nothing left
#: to check (LESSONS pattern 14).
TIGERVNC_USERS_FILE = Path("/etc/tigervnc/vncserver.users")
#: What marks a TigerVNC file as one Vaelor wrote, so commissioning may
#: replace it instead of refusing to touch an owner's own configuration.
#:
#: A third marker naming the pre-rename product sat here until #168. It only
#: ever recognised files under that era's home directory, which no longer
#: exists on the one machine that had it. `session=ubuntu` is not a product
#: name — it is the session line the older template wrote — so it stays and
#: still recognises a pre-flashback config whatever comment sits above it.
#:
#: Dropping a marker never deletes an owner's file, but the three readers do
#: not answer the same way and it is worth knowing which before editing this:
#: `_write_desktop_files` **raises** on an unrecognised `xstartup` or
#: `config`; it **skips the cleanup** for an unrecognised legacy `~/.vnc`,
#: leaving that directory in place, which is deliberate — the new session
#: reads `~/.config/tigervnc` and does not care — and is reported in the
#: commissioning reply rather than left silent, because an absence nobody is
#: told about is LESSONS pattern 8; `_remove_previous_managed_vnc_config`
#: **does not unlink**, which is the point of its check. Adding a marker to
#: silence a refusal is the wrong repair; find which side moved.
MANAGED_VNC_MARKERS = (
    "Managed by Vaelor",
    "session=ubuntu",
)


def _is_managed_vnc_content(content: str) -> bool:
    return any(marker in content for marker in MANAGED_VNC_MARKERS)


def _remove_previous_managed_vnc_config(username: str) -> None:
    if pwd is None or not username or username == MANAGED_DESKTOP_USER:
        return
    try:
        record = pwd.getpwnam(username)
    except KeyError:
        return
    vnc_dir = Path(record.pw_dir) / ".vnc"
    for name in ("config", "xstartup"):
        candidate = vnc_dir / name
        if not candidate.exists():
            continue
        content = candidate.read_text(encoding="utf-8", errors="replace")
        if _is_managed_vnc_content(content):
            candidate.unlink()


def _write_desktop_files(record) -> Dict[str, Any]:
    """Write the managed account's session files, and report what is neither
    a success nor a failure.

    Most of this succeeds or raises and needs no verdict. Two things do, and
    both would otherwise leave the owner a reply saying more than happened
    (LESSONS pattern 11): the first-run suppression, which is worth doing and
    not worth refusing a desktop over; and a legacy `~/.vnc` this function
    declined to clean up, which was retained *silently* until #168's review.
    """
    home = Path(record.pw_dir)
    # Not `retained`: that name is taken further down by the user map.
    retained_paths: list[str] = []
    legacy_dir = home / ".vnc"
    if legacy_dir.is_dir() and not legacy_dir.is_symlink():
        legacy_config = legacy_dir / "config"
        if legacy_config.exists() and _is_managed_vnc_content(
            legacy_config.read_text(encoding="utf-8", errors="replace")
        ):
            for item in legacy_dir.iterdir():
                if item.is_file() and (
                    item.name == "config" or item.name.endswith(".log")
                ):
                    item.unlink()
            if any(legacy_dir.iterdir()):
                raise RuntimeError(
                    "The legacy TigerVNC directory contains unmanaged files."
                )
            legacy_dir.rmdir()
        else:
            # Not ours, so it stays - and the owner is told. Cleaning it
            # would delete a file Vaelor did not write; raising would refuse
            # a desktop over a directory the new session never reads. Saying
            # nothing was the third option, and the one this branch took.
            retained_paths.append(str(legacy_dir))
    elif legacy_dir.is_symlink():
        retained_paths.append(str(legacy_dir))  # declined for the same reason
    config_root = home / ".config"
    config_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    vnc_dir = config_root / "tigervnc"
    vnc_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Root writes and chowns below, inside an account Vaelor does not run as,
    # so a link planted from the desktop session would have root write its
    # target. `legacy_dir` above was always guarded this way; these were not
    # (#168 review). The residual race is stated in
    # `host_desktop_first_run._refuse_symlink`, which took the same trade.
    for candidate in (vnc_dir, vnc_dir / "xstartup", vnc_dir / "config"):
        if candidate.is_symlink():
            raise RuntimeError(
                "A TigerVNC path in the managed desktop account is a symbolic "
                "link, so Vaelor will not write through it: {}".format(candidate)
            )
    xstartup = vnc_dir / "xstartup"
    if xstartup.exists():
        existing = xstartup.read_text(encoding="utf-8", errors="replace")
        if not _is_managed_vnc_content(existing):
            raise RuntimeError(
                "An existing custom TigerVNC startup file must be reviewed before "
                "Vaelor can commission the browser desktop."
            )
        xstartup.unlink()

    config = vnc_dir / "config"
    if config.exists() and not _is_managed_vnc_content(
        config.read_text(encoding="utf-8", errors="replace")
    ):
        raise RuntimeError(
            "An existing custom TigerVNC configuration must be reviewed before "
            "Vaelor can commission the browser desktop."
        )
    config.write_text(
        """# Managed by Vaelor
session=gnome-flashback-metacity
geometry=1920x1080
localhost
securitytypes=None
alwaysshared
""",
        encoding="utf-8",
    )
    os.chown(config_root, record.pw_uid, record.pw_gid)
    os.chown(vnc_dir, record.pw_uid, record.pw_gid)
    os.chown(config, record.pw_uid, record.pw_gid)
    os.chmod(config, 0o600)

    users = TIGERVNC_USERS_FILE
    existing_users = (
        users.read_text(encoding="utf-8", errors="replace").splitlines()
        if users.exists()
        else []
    )
    previous = next(
        (
            line.split("=", 1)[1].strip()
            for line in existing_users
            if line.startswith(f"{VNC_DISPLAY}=") and "=" in line
        ),
        "",
    )
    _remove_previous_managed_vnc_config(previous)
    retained = [line for line in existing_users if not line.startswith(f"{VNC_DISPLAY}=")]
    retained.append(f"{VNC_DISPLAY}={record.pw_name}")
    users.write_text("\n".join(retained) + "\n", encoding="utf-8")
    os.chmod(users, 0o644)
    # Written here rather than in `commission_vnc` because this is where the
    # account's own configuration is authored, and because the trigger is a
    # fresh home directory: every rebuild of the account re-arms the wizard,
    # and every rebuild passes through this function.
    return {
        "initial_setup": suppress_initial_setup(record),
        "retained_paths": retained_paths,
    }
