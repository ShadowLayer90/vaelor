"""The release snapshot a console update leaves in /usr/lib/vaelor/release (VD-217, VD-133 1.5.5).

The installer stages what ``maintain-vaelor.sh`` needs beside itself under
``/usr/lib/vaelor/release``: that is the fixed path the console's Remove
Vaelor runs (`appliance_recovery.RELEASE_MAINTAIN_SCRIPT`), and ``repair`` runs
the ``install-vaelor.sh`` staged there, which reads its constraints, its units,
the NPU model fetcher and AMD's key from its own folder. The console's Update
Vaelor reinstalled the package with pip and never touched the folder, so a box
updated from the console kept the OLD uninstaller (VD-217: an update never
leaves a box half upgraded; memory "hotpatch must include invoked on-box
files").

**Where the new copies come from.** The wheel carries :data:`PAYLOAD` as data
files (`pyproject.toml`, ``share/vaelor/release``), so pip places them in the
venv beside the package it installed - the bytes the installer of that release
stages - and the units come from the package's own ``appliance_units``.
:func:`refresh` copies each one that differs into the snapshot by a rename,
root-owned (a file the update broker's group could write would be a script
root later runs), with the verified wheel beside them, and keeps only the
current and the previous wheel. ``installed-by-vaelor``, the record the
bare-OS teardown reads, is never touched.

**When it does nothing.** No snapshot folder (nothing was installed there: it
is never created here); a box installed from the native ``.deb``, whose
package manager owns these files and updates them itself; and a wheel that
ships no payload (one from before this), so a rollback to such a release keeps
the newer scripts rather than deleting them.

**Two callers.** The update broker calls it beside its unit refresh. And
because the broker that applies the FIRST update to this release is the
previous release's code, which knows nothing of it, :data:`post_install_steps.STEPS`
runs ``refresh`` too - the one command an older broker runs from the new
package (``vaelor-refresh-models --apply``) - taking the wheel from the
retained-current record the broker has just written.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

#: The snapshot (`deploy/install-vaelor.sh` stages it).
RELEASE_DIR = Path("/usr/lib/vaelor/release")

#: Where the wheel's data files put the payload, under the venv.
BUNDLED_SUBDIR = Path("share") / "vaelor" / "release"

#: Everything the staged installer and uninstaller read from their own folder,
#: and the mode each is staged with. `deploy/install-vaelor.sh` stages the same
#: list and a test holds the three (this, the installer, pyproject) equal.
PAYLOAD = (
    ("install-vaelor.sh", 0o755),
    ("maintain-vaelor.sh", 0o755),
    ("fetch-npu-model.sh", 0o755),
    ("amdrocm-keyring.gpg", 0o644),
    ("requirements-release.txt", 0o644),
)
SCRIPTS = tuple(name for name, mode in PAYLOAD if mode == 0o755)

#: The units ``install-vaelor.sh`` installs from ``systemd/`` beside itself.
UNITS_SUBDIR = "systemd"

#: The native package's file list: present, dpkg owns the snapshot.
DPKG_LIST = Path("/var/lib/dpkg/info/vaelor-control-plane.list")

WHEEL_GLOB = "vaelor_control_plane-*.whl"


def _units_dir() -> Path:
    return Path(__file__).parent / "appliance_units"


def _replace(source: Path, target: Path, mode: int) -> None:
    """Copy ``source`` to ``target`` through a sibling temp file and one rename, root-owned."""
    target.parent.mkdir(mode=0o755, exist_ok=True)
    handle, staging = tempfile.mkstemp(dir=str(target.parent), prefix=".vaelor-")
    os.close(handle)
    try:
        shutil.copyfile(str(source), staging)
        os.chmod(staging, mode)
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            os.chown(staging, 0, 0)
        os.replace(staging, str(target))
    except BaseException:
        try:
            os.unlink(staging)
        except OSError:
            pass
        raise


def _differs(source: Path, target: Path) -> bool:
    try:
        return target.read_bytes() != source.read_bytes()
    except OSError:
        return True


def _prune_wheels(release_dir: Path, current: Path) -> List[str]:
    """Keep the current wheel and the newest other one; the names removed."""
    others = sorted((path for path in release_dir.glob(WHEEL_GLOB) if path.name != current.name),
                    key=lambda path: path.stat().st_mtime, reverse=True)
    removed = []
    for path in others[1:]:
        path.unlink()
        removed.append(path.name)
    return removed


def refresh(
    venv_root: Path, *, release_dir: Path = RELEASE_DIR, wheel: Optional[Path] = None,
    units_dir: Optional[Path] = None, dpkg_list: Path = DPKG_LIST,
) -> List[str]:
    """Bring the snapshot to the payload the installed wheel ships; the names written."""
    release_dir = Path(release_dir)
    if not release_dir.is_dir() or Path(dpkg_list).exists():
        return []
    bundled = Path(venv_root) / BUNDLED_SUBDIR
    sources = [(bundled / name, mode) for name, mode in PAYLOAD if (bundled / name).is_file()]
    if not sources:
        return []
    written = []
    for source, mode in sources:
        target = release_dir / source.name
        if _differs(source, target):
            _replace(source, target, mode)
            written.append(source.name)
    units = Path(units_dir) if units_dir is not None else _units_dir()
    for unit in sorted(units.glob("vaelor-*.service")):
        target = release_dir / UNITS_SUBDIR / unit.name
        if _differs(unit, target):
            _replace(unit, target, 0o644)
            written.append(UNITS_SUBDIR + "/" + unit.name)
    if wheel is not None and Path(wheel).is_file():
        target = release_dir / Path(wheel).name
        if _differs(Path(wheel), target):
            _replace(Path(wheel), target, 0o644)
            written.append(target.name)
        _prune_wheels(release_dir, target)
    return written


def refresh_quietly(venv_root: Path, *, wheel: Optional[Path] = None, **options) -> str:
    """:func:`refresh`, never raising: ``""``, or why the snapshot was not refreshed.

    The update broker's caller: a full disk must not stop an update - or a
    rollback - before its restart (review, VD-133 1.5.5). The reason goes into
    the outcome, and the post-install step tries again.
    """
    try:
        refresh(venv_root, wheel=wheel, **options)
    except OSError as error:
        return "The release scripts Remove Vaelor runs were not refreshed: {}".format(
            str(error)[:200] or type(error).__name__)
    return ""


def main(argv: Sequence[str] = ()) -> int:
    """``refresh``, as a post-install step, from the venv this interpreter runs in."""
    arguments = list(argv) if argv else sys.argv[1:]
    if arguments != ["refresh"]:
        sys.stderr.write("usage: python -I -m vaelor.release_snapshot refresh\n")
        return 2
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None or geteuid() != 0:
        print("Release scripts: not refreshed, this must run as root")
        return 0
    from .appliance_upgrade import read_current_wheel

    current = read_current_wheel()
    try:
        written = refresh(Path(sys.prefix), wheel=current["wheel"] if current else None)
    except OSError as error:
        print("Release scripts: not refreshed ({})".format(str(error)[:200]))
        return 1
    print("Release scripts: {}".format(
        "refreshed " + ", ".join(written) if written else "done"))
    return 0


if __name__ == "__main__":  # pragma: no cover - run as a post-install step
    sys.exit(main())
