"""The update's small JSON files, written and read without following a link (VD-217, R106).

The update broker's result (`appliance_upgrade.RESULT_PATH`), its plan, and
the refresh's progress record (`upgrade_progress.RECORD_NAME`) live in
``/var/lib/vaelor/upgrade``: ``2770 vaelor-workloads:vaelor-jobs``, so the
executor and the control plane can create, rename and remove entries there,
and it is not sticky, so ``fs.protected_symlinks`` does not help. A root
writer that opened a fixed temporary name there - ``apply-result.tmp``,
``refresh-record.tmp`` - followed a link planted at that name and overwrote
any file root can write, then chmod'ed it 0640 (reproduced on the Z2, kb R106).

So both writers go through :func:`write_json`: the folder is opened one
component at a time from ``/`` without following a link, the bytes go to a
fresh ``O_CREAT|O_EXCL|O_NOFOLLOW`` temporary with a random name, are
``fchmod``-ed, and are renamed over the target relative to that folder - a
link planted at the target is replaced, never written through
(`bridge_safe_fs.write_file`, the root bridge's own rule; LESSONS 18).

And readers that act on what root wrote use :func:`read_json` with
``owners``: a link, a FIFO, a device, a second hard link or a file another
account owns is refused, never read. What stays possible for an account that
can write the folder: removing or renaming root's file (the panel then says
the update left no record - a denial, not a forgery), or putting back an
older root-written file it moved aside earlier, which the readers' time
filters (`upgrade_progress.progress`) attribute to the update it belongs to.
Both accounts can already start and stop updates, so neither is a new power.

On a platform without ``O_NOFOLLOW``/``dir_fd`` (a developer's Windows
checkout) the plain path is used: the guarantee is a Linux appliance one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional

from . import bridge_safe_fs as safe_fs

#: Who a record a root writer made may belong to. Root only: the executor and
#: the control plane can write the folder, so their files are never trusted.
ROOT_ONLY: FrozenSet[int] = frozenset({0})

#: The largest record read back.
READ_LIMIT = 256 * 1024


def write_json(path: Path, payload: Dict[str, Any], mode: int = 0o640) -> None:
    """Replace ``path`` with ``payload`` as JSON, atomically, following no link."""
    path = Path(path)
    text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    if not safe_fs.supported():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(".{}.{}".format(path.name, os.getpid()))
        temporary.write_text(text, encoding="utf-8")
        os.chmod(temporary, mode)
        temporary.replace(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with safe_fs.directory(str(path.parent)) as parent:
        safe_fs.write_file(parent, path.name, text, mode=mode)


def read_json(path: Path, owners: Optional[FrozenSet[int]] = ROOT_ONLY) -> Optional[Dict[str, Any]]:
    """The JSON object in ``path``, or ``None`` - also for a link, a non-regular
    file, a second hard link or a file ``owners`` did not write."""
    path = Path(path)
    try:
        if not safe_fs.supported():
            text = path.read_text(encoding="utf-8")
        else:
            with safe_fs.directory(str(path.parent)) as parent:
                text = safe_fs.read_file(parent, path.name, limit=READ_LIMIT, owners=owners)
        value = json.loads(text)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None
