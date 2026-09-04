"""Keeping Ubuntu's own onboarding out of Vaelor's one-use browser desktop.

**Runs inside the privileged host-desktop broker, as root**, writing into the
dedicated desktop account's home. Naming the process that runs what is the
cheap habit LESSONS pattern 14 asks for, and it is why the chown below is not
optional: root writing into another account's `~/.config` leaves files that
account cannot read, which is LESSONS pattern 13's exact shape.

Split out of `host_desktop.py` for the reason `host_desktop_lock.py` and
`host_desktop_tls.py` were: that module is near the 1,000-line ceiling and
this needs room to say why it exists.

**What was seen (#168, on the appliance 2026-08-13).** The managed account was
rebuilt, and the first browser-desktop session opened onto GNOME's "Welcome!"
dialog asking for a language, with "Initial Setup" in the taskbar. The owner
reaches this desktop *through* Vaelor's own sign-in, on a machine that is
already set up; nobody should be completing Ubuntu's onboarding inside it.

It is triggered by a fresh home directory rather than by a fresh machine, so
it returns every time the managed account is rebuilt - which is precisely what
"Install browser desktop" does. That makes it commissioning's problem.

**What the first fix got wrong, measured live on the appliance 2026-08-13
(#168).** The stamp and the autostart override both landed correctly and
suppressed nothing, for two reasons that no offline test could have found:

  (a) This Ubuntu build has **no `/etc/xdg/autostart/*initial*` entry at all**.
      The wizard is launched by two **systemd user units**,
      `gnome-initial-setup-first-login.service` and
      `gnome-initial-setup-upgrade-login.service`, so a `.desktop` autostart
      override shadows a system entry that does not exist. This is LESSONS
      pattern 15 - build a suppression for a launcher without reading how the
      thing is actually launched.
  (b) The running process was `gnome-initial-setup --upgrade-user`, and upgrade
      mode exists to show new pages to users who have *already completed*
      setup, so the `-done` stamp is bypassed by design.

**So the suppression that actually works on this build is masking those two
user units for the managed account** - a per-user mask, which is a symlink to
`/dev/null` under the account's `~/.config/systemd/user/`, placed at commission
time and read by the user manager at first login. The stamp and the autostart
override are kept: they cost nothing, they are the correct suppression for a
build that *does* use first-login mode or a `/etc/xdg/autostart` entry, and
this control plane cannot inspect offline which mechanism a given release ships.

**None of this is verified.** The mask is unproven until the managed account is
rebuilt on the appliance and connected, which the owner runs; treat a
`confirmed` verdict as "Vaelor placed the mask it meant to place" and nothing
more. VD-096 records the decision, including that unverified part.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

#: The stamp `gnome-initial-setup` writes when its wizard completes, and the
#: file it looks for before offering to run again. Upstream's
#: `gis_ensure_stamp_files()` writes exactly this path with exactly this body,
#: so writing it ourselves says "already done" in the wizard's own vocabulary
#: rather than in one invented here.
INITIAL_SETUP_STAMP = "gnome-initial-setup-done"
INITIAL_SETUP_STAMP_BODY = "yes\n"

#: The system autostart entry that launches the wizard at an existing user's
#: first login. A per-user file of the same name shadows it, which is the
#: freedesktop autostart rule, and is why this suppression reaches only the
#: managed account: a system-wide edit would reach the owner's real desktop
#: too, and that is not ours to change.
INITIAL_SETUP_AUTOSTART = "gnome-initial-setup-first-login.desktop"
INITIAL_SETUP_AUTOSTART_BODY = """[Desktop Entry]
Type=Application
Name=GNOME Initial Setup
Hidden=true
X-GNOME-Autostart-enabled=false
"""

#: The systemd **user** units that launch the wizard on this Ubuntu build - the
#: one the two suppressions above miss (#168, measured live). `first-login`
#: runs it for an account that has never completed setup; `upgrade-login` runs
#: `--upgrade-user`, which shows new pages to accounts that *have*, and is why
#: the `-done` stamp is bypassed. Masking both - a symlink to `/dev/null` in the
#: account's own `~/.config/systemd/user/` - is `systemctl --user mask` done by
#: hand, so it needs no live user session and holds at the next first login.
INITIAL_SETUP_SYSTEMD_UNITS = (
    "gnome-initial-setup-first-login.service",
    "gnome-initial-setup-upgrade-login.service",
)


def _refuse_symlink(path: Path) -> None:
    """Refuse to write through a symlink planted in the desktop account's home.

    **Everything here runs as root inside that account's directories**, so a
    process in the desktop session could replace the stamp file or the
    `autostart` directory with a link and have the next commissioning run
    write and chown the target as root. That is a real primitive, and the
    precondition — code running in the managed session — is a much lower bar
    than root, because the whole point of that session is that it is a sandbox
    an owner may hand to anything.

    `is_symlink()` and not `O_NOFOLLOW`: this is the idiom the module already
    uses (`_write_desktop_files` guards the legacy `~/.vnc` directory the same
    way), it keeps these writers reachable by the suite on a non-Linux
    developer machine, and LESSONS pattern 13 is emphatic about what happens
    to a path only the appliance can execute.

    **The residual race is stated rather than papered over.** Between this
    check and the write, the link can be re-planted; `os.open` with
    `O_NOFOLLOW` and `dir_fd` is the closure and it is not this change. What
    this removes is the unattended case — plant a link, wait for the next
    commission — and what it leaves is a microsecond window an attacker must
    hit on a run whose timing they do not control.
    """
    if path.is_symlink():
        raise OSError(
            "{} is a symbolic link; refusing to write through it.".format(path)
        )


def _write_owned(path: Path, body: str, record) -> None:
    """Write `body` to `path` and hand the file to the desktop account.

    Unconditional, because both files are Vaelor's to state and neither
    carries anything worth preserving: re-running commissioning must reach the
    same result on a home that already has them as on one that does not. That
    is the whole of the idempotence claim, and it is why nothing here reads
    the existing content first.

    **`mkdir(mode=...)` is not the mode of a directory that already exists**,
    so the parent is chmodded explicitly. Creating it 0700 and then leaving a
    pre-existing 0755 alone is the shape where a second run reaches a
    different state from the first, which is the claim above being quietly
    false.
    """
    _refuse_symlink(path.parent)
    os.chmod(path.parent, 0o700)
    os.chown(path.parent, record.pw_uid, record.pw_gid)
    _refuse_symlink(path)
    path.write_text(body, encoding="utf-8")
    os.chown(path, record.pw_uid, record.pw_gid)
    os.chmod(path, 0o644)


def _mask_unit(mask_dir: Path, unit: str) -> str:
    """Symlink one systemd user unit to `/dev/null`, and read the link back.

    This is exactly what `systemctl --user mask` writes, done by hand so it
    needs no live user manager - the account has never logged in at commission
    time. Idempotent by construction: whatever is at the unit path is removed
    first, so a rebuild reaches the same masked state as a first commission,
    and a stale real unit file left by a package upgrade is replaced.

    **The symlink is left root-owned on purpose.** Chowning it would mean
    `lchown` - a plain `os.chown` follows the link and would hand `/dev/null`
    to the desktop account, which is the wrong file entirely. The parent
    directory is owned by the account (below), which is what lets the account
    read and remove the mask; a symlink's own owner does not gate either.

    **The readback proves the link landed and points where it should**, which
    is LESSONS pattern 1's answer and the honest limit of it - it does not
    prove the user manager will honour it, only rebuilding the account on the
    appliance settles that. Returns "" when the mask is confirmed, or the link
    path when it could not be placed, so the caller names it (pattern 11).
    """
    link = mask_dir / unit
    try:
        # Remove whatever is there before creating the link, rather than
        # writing through it: a planted symlink at this path would otherwise
        # send `os.symlink`'s own replacement nowhere useful, and unlinking a
        # link removes the link, never its target.
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.devnull, link)
        if os.readlink(link) != os.devnull:
            return str(link)
    except OSError:
        return str(link)
    return ""


def suppress_initial_setup(record) -> Dict[str, Any]:
    """Stop GNOME's first-run wizard opening inside the managed desktop.

    **Three suppressions, because none is provable from here.** The stamp is
    the wizard's own completion marker; the autostart override stops the
    `/etc/xdg/autostart` entry that launches it; and the unit masks stop the
    two systemd **user** units that launch it on the build the appliance
    actually runs. They are not answers to one question - LESSONS pattern 6 is
    about two *readings* disagreeing, and these are actions that can only all
    hold or some fail. What they are is cover across the launch mechanisms a
    given `gnome-initial-setup` release may use, which this control plane
    cannot inspect offline. #168 established that on this build only the mask
    bites: the module docstring carries the measurement.

    **The readback proves the write landed, not that GNOME obeys it.** Both
    files are read back and compared, so a home that is read-only, a chown
    that failed, or a path that moved is reported rather than assumed - that
    is LESSONS pattern 1's answer, and it is the honest limit of it. Whether
    this release's session actually honours either file is settled by one
    thing only: rebuilding the managed account on the appliance and connecting.
    Until that happens, treat a `confirmed` verdict as "Vaelor wrote what it
    meant to write" and nothing more.

    Failure is reported, never raised. An owner who presses "Install browser
    desktop" wants a desktop; refusing to commission a working one because a
    stamp file could not be written would trade a wizard for no desktop at
    all. `commission_vnc` carries the verdict into its reply so the outcome is
    read rather than discarded (LESSONS pattern 11). A planted symlink takes
    the same route: `_refuse_symlink` raises, the loop records the path as
    unconfirmed, and the owner is told which file could not be written rather
    than having root follow the link in silence.
    """
    config_root = Path(record.pw_dir) / ".config"
    autostart = config_root / "autostart"
    targets = [
        (config_root / INITIAL_SETUP_STAMP, INITIAL_SETUP_STAMP_BODY),
        (autostart / INITIAL_SETUP_AUTOSTART, INITIAL_SETUP_AUTOSTART_BODY),
    ]
    unconfirmed: List[str] = []
    for path, body in targets:
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _write_owned(path, body, record)
            if path.read_text(encoding="utf-8") != body:
                unconfirmed.append(str(path))
        except OSError:
            unconfirmed.append(str(path))

    # The one that actually bites on this build (#168). `~/.config/systemd` and
    # `~/.config/systemd/user` are both created and handed to the account; each
    # unit is then masked. A failure to prepare the directories leaves both
    # units unconfirmed rather than raising - an owner who pressed "Install
    # browser desktop" still gets a desktop.
    #
    # **Every intermediate is refused, and `mkdir` never uses `parents=True`**
    # (#168 review, a Medium security defect). The first draft created the leaf
    # with `mkdir(parents=True)` and refused only the leaf, then chowned
    # `mask_dir.parent`. A desktop-session process - "a sandbox an owner may
    # hand to anything" - could pre-plant `~/.config/systemd` as a symlink to a
    # directory outside the home: `mkdir(parents=True)` then created `user`
    # *inside the link's target*, the leaf check passed because the created
    # `user` really is a directory, and `os.chown(mask_dir.parent, ...)`
    # **followed the link and chowned the attacker's directory** to the desktop
    # account. That is a chown-arbitrary-path primitive, reachable on the
    # re-commission path. The fix is `_write_owned`'s rule applied to each
    # directory this brings into being: create one level at a time so no
    # `mkdir` can traverse a link, and refuse each directory before chowning
    # it, so root never chowns a path it has not confirmed is a real directory.
    systemd_dir = config_root / "systemd"
    mask_dir = systemd_dir / "user"
    try:
        for directory in (systemd_dir, mask_dir):
            directory.mkdir(mode=0o700, exist_ok=True)
            _refuse_symlink(directory)
            os.chmod(directory, 0o700)
            os.chown(directory, record.pw_uid, record.pw_gid)
    except OSError:
        unconfirmed.extend(
            str(mask_dir / unit) for unit in INITIAL_SETUP_SYSTEMD_UNITS
        )
    else:
        for unit in INITIAL_SETUP_SYSTEMD_UNITS:
            failed = _mask_unit(mask_dir, unit)
            if failed:
                unconfirmed.append(failed)
    return {"confirmed": not unconfirmed, "unconfirmed": unconfirmed}


def initial_setup_detail(verdict: Dict[str, Any]) -> str:
    """The sentence an owner gets when the wizard may still appear.

    Naming the files rather than saying "suppression failed", because the two
    have different remedies and the path is the only thing that tells them
    apart. Empty when there is nothing to say, so callers can join details
    without inventing a reassurance nobody measured.
    """
    if verdict.get("confirmed"):
        return ""
    return (
        "The browser desktop was commissioned, but Ubuntu's first-run setup "
        "wizard is not confirmed suppressed: {} could not be written, so the "
        "first session may open onto a Welcome dialog.".format(
            ", ".join(verdict.get("unconfirmed") or [])
        )
    )
