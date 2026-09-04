"""A confined file manager over each managed app's browsable data roots.

The config-file editor in :mod:`vaelor.workload_inventory` exposes only the
handful of files a template declares editable. This is the general file
manager: list, mkdir, delete, upload and download over an app's DATA ROOTS -
its volume mount, plus a media mount for media apps.

Every operation is confined the same way, in two independent places. The
control plane resolves the app to a curated template and checks the path with
:func:`~vaelor.app_catalog.data_path_is_within_roots` before acting; the
workload broker re-derives the container and re-checks the path against the
template's own roots, admitting only the fixed :data:`FS_LIST_SCRIPT`/
:data:`FS_MKDIR_SCRIPT`/:data:`FS_DELETE_SCRIPT` (each taking the path as the
positional ``$1``) and the bounded ``docker cp`` shapes. The two checks share
one encoding of the confinement rule so they cannot drift apart.

The bytes of an uploaded or downloaded file never cross the broker's text
channel: they move through a private, group-shared ``docker cp`` temp file, the
same cross-user pattern the config editor uses.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from .app_catalog import data_path_is_within_roots, declared_data_roots
from .workload_broker import (
    FS_CHOWN_SCRIPT,
    FS_DELETE_SCRIPT,
    FS_LIST_SCRIPT,
    FS_MKDIR_SCRIPT,
)

#: Ceiling on a file landed for download before any byte is streamed. Read from
#: the on-disk size AFTER ``docker cp`` lands the file, so an oversized file is
#: refused without being loaded into memory.
FS_DOWNLOAD_MAX_BYTES = 100 * 1024 * 1024

#: Longest single path segment (name) accepted for a new directory or upload.
FS_NAME_MAX_LENGTH = 255

_NOTDIR_SENTINEL = "__vaelor_notdir__"
_OUTSIDE_ROOTS = "That location is outside this app's data folders."
_BAD_NAME = "Choose a name without a slash or path segments."
_NOT_FOUND = "That file was not found in the app."


def _normalize(path: str) -> str:
    """Collapse ``.`` and empty segments to an absolute, slash-free-tail path.

    ``..`` is not handled here because a path containing it is refused by
    :func:`data_path_is_within_roots` before this runs.
    """
    cleaned = [segment for segment in str(path).split("/") if segment not in ("", ".")]
    return "/" + "/".join(cleaned)


def _has_control_char(text: str) -> bool:
    """True if any character is a C0 control (< 0x20), newline and tab included.

    The FS_LIST output is tab-separated and split on newlines, so a name with
    either would forge or split a listing row; a name Vaelor creates must carry
    neither, and the list parser drops any pre-existing entry that does.
    """
    return any(ord(character) < 0x20 for character in text)


def _is_safe_name(name: Any) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and "/" not in name
        and name not in (".", "..")
        and len(name) <= FS_NAME_MAX_LENGTH
        and not _has_control_char(name)
    )


class AppFileBrowser:
    """List, mkdir, delete, upload and download over a managed app's data roots.

    Constructed with a :class:`~vaelor.workload_inventory.WorkloadInventory`;
    it reuses that object's template resolution, brokered ``docker`` runner and
    private-workdir/``docker cp`` primitives so there is one confinement path
    and one cross-user temp-file discipline in the codebase.
    """

    def __init__(self, inventory: Any):
        self._inventory = inventory

    def _resolve(self, app_id: str) -> tuple[str, str, list[str]]:
        """Return (container, template_id, roots) for a managed app, or refuse."""
        _app, template_id, _template = self._inventory._managed_template(app_id)
        roots = declared_data_roots(template_id)
        if not roots:
            raise ValueError("This app has no browsable data folders.")
        return f"vaelor-{template_id}", template_id, roots

    def _within(self, template_id: str, path: str) -> str:
        if not data_path_is_within_roots(template_id, path):
            raise ValueError(_OUTSIDE_ROOTS)
        return _normalize(path)

    def _target_path(self, template_id: str, roots: list[str], path: Any) -> str:
        """The requested directory, defaulting to the first data root."""
        if path is None or path == "":
            return roots[0]
        return self._within(template_id, path)

    def _redact(self, text: str) -> str:
        """Route broker stderr through the inventory's credential/paths redactor.

        The same redaction the config editor applies, so a docker error naming a
        host temp path or container path is not surfaced verbatim to the caller.
        """
        redactor = getattr(self._inventory, "_redact", None)
        return redactor(text) if callable(redactor) else text

    def _list_entries(self, container: str, resolved: str) -> list[dict[str, Any]]:
        """Run the fixed FS_LIST script on one directory and parse its rows.

        The not-a-directory case is the script's ACTUAL signal - exit code 3
        with the sentinel as the whole stripped output - never a mere substring
        of stdout, so a child file literally named ``__vaelor_notdir__`` cannot
        poison its parent's listing. Rows are parsed resiliently: a malformed
        record, or one whose name carries a control character (a newline that
        split one entry into two, a forged tab), is dropped rather than trusted.
        """
        result = self._inventory._run(
            ["docker", "exec", container, "sh", "-c", FS_LIST_SCRIPT, "_", resolved],
            timeout=15,
        )
        stdout = result.stdout or ""
        if result.returncode == 3 and stdout.strip() == _NOTDIR_SENTINEL:
            raise ValueError("That location is not a folder.")
        if result.returncode != 0:
            raise ValueError(
                self._redact((result.stderr or "").strip()) or "That folder could not be listed."
            )
        entries = []
        for line in stdout.splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            kind, size, name = parts
            if kind not in ("d", "f") or not name or _has_control_char(name):
                continue
            try:
                size_bytes = int(size)
            except ValueError:
                size_bytes = 0
            entries.append({"name": name, "type": kind, "size": size_bytes})
        entries.sort(key=lambda entry: (entry["type"] != "d", entry["name"]))
        return entries

    def list_dir(self, app_id: str, path: Any = None) -> dict[str, Any]:
        container, template_id, roots = self._resolve(app_id)
        resolved = self._target_path(template_id, roots, path)
        entries = self._list_entries(container, resolved)
        return {"roots": roots, "path": resolved, "entries": entries}

    def make_dir(self, app_id: str, path: Any, name: Any) -> dict[str, Any]:
        container, template_id, roots = self._resolve(app_id)
        parent = self._target_path(template_id, roots, path)
        if not _is_safe_name(name):
            raise ValueError(_BAD_NAME)
        target = self._within(template_id, f"{parent}/{name}")
        self._exec_script(container, FS_MKDIR_SCRIPT, target, "That folder could not be created.")
        return {"ok": True, "path": target}

    def delete_path(self, app_id: str, path: Any) -> dict[str, Any]:
        container, template_id, roots = self._resolve(app_id)
        if path is None or path == "":
            raise ValueError(_OUTSIDE_ROOTS)
        target = self._within(template_id, path)
        if target in roots:
            raise ValueError("The data folder root itself cannot be deleted.")
        self._exec_script(container, FS_DELETE_SCRIPT, target, "That item could not be deleted.")
        return {"ok": True, "path": target}

    def _exec_script(self, container: str, script: str, target: str, failure: str) -> None:
        result = self._inventory._run(
            ["docker", "exec", container, "sh", "-c", script, "_", target],
            timeout=15,
        )
        if result.returncode != 0:
            raise ValueError(self._redact((result.stderr or "").strip()) or failure)

    @contextmanager
    def download(self, app_id: str, path: Any) -> Iterator[tuple[Any, str]]:
        """Land one data-root file on the host and yield (host_path, filename).

        The size cap is enforced BEFORE the copy: ``docker cp`` writes the whole
        file to host disk (the broker's stdout clamp does not bound it), so an
        oversized file would otherwise land in full before any check. We read the
        authoritative size from a listing of the file's PARENT directory - an
        already-allowlisted FS_LIST call, adding no broker shape - and refuse an
        over-cap or absent file before the copy. The private temp directory
        holding the landed file is removed when the block exits.
        """
        container, template_id, _roots = self._resolve(app_id)
        if path is None or path == "":
            raise ValueError(_OUTSIDE_ROOTS)
        resolved = self._within(template_id, path)
        filename = resolved.rsplit("/", 1)[-1] or "download"
        parent = resolved.rsplit("/", 1)[0] or "/"
        entry = next(
            (
                item for item in self._list_entries(container, self._within(template_id, parent))
                if item["name"] == filename and item["type"] == "f"
            ),
            None,
        )
        if entry is None:
            raise ValueError(_NOT_FOUND)
        if entry["size"] > FS_DOWNLOAD_MAX_BYTES:
            raise ValueError("That file is too large to download here.")
        with self._inventory._copy_from_container(container, resolved) as landed:
            if landed is None:
                raise ValueError(_NOT_FOUND)
            yield landed, filename

    def upload(
        self,
        app_id: str,
        path: Any,
        filename: Any,
        source: Any,
        max_bytes: int,
    ) -> dict[str, Any]:
        """Write one uploaded file into a data-root directory via ``docker cp``.

        ``source`` is raw ``bytes`` or a readable binary stream; it is bounded
        by ``max_bytes`` and reaches the container through a private,
        group-readable temp file so no byte crosses the broker's text channel.

        Two steps, because ``docker cp`` preserves the host temp file's owner
        (the control-plane user, uid ~997) and mode 0660 into the container. An
        app that runs as its own non-root user would then be denied read on its
        own upload. So after the cp we run one fixed, root-only chown inside the
        container (see ``FS_CHOWN_SCRIPT``) that gives the file the owner:group
        of its parent data-root directory - the app's own identity - so the app
        can read AND write it. The host temp file stays 0660 (never made
        world-readable); the fix is native ownership, not looser bits. The chown
        is BEST-EFFORT: the bytes are already in the container, so a non-zero
        chown does not fail the upload (some images may lack ``chown``/``stat``);
        it just leaves the file owned by the copier.
        """
        container, template_id, roots = self._resolve(app_id)
        directory = self._target_path(template_id, roots, path)
        if not _is_safe_name(filename):
            raise ValueError(_BAD_NAME)
        target = self._within(template_id, f"{directory}/{filename}")
        data = self._read_bounded(source, max_bytes)
        with self._inventory._private_workdir() as workdir:
            temporary_path = workdir / "file"
            temporary_path.write_bytes(data)
            # The control plane writes this temp file but the broker
            # (a different OS user) runs `docker cp` and must READ it; 0660
            # grants read to the shared vaelor-jobs group, and nothing outside.
            temporary_path.chmod(0o660)
            result = self._inventory._run(
                ["docker", "cp", str(temporary_path), f"{container}:{target}"],
                timeout=30,
            )
            if result.returncode != 0:
                raise ValueError(
                    self._redact((result.stderr or "").strip())
                    or "That file could not be uploaded."
                )
        # Best-effort: hand the freshly copied file to the app's own user by
        # matching the parent directory's owner:group. A failure here leaves the
        # bytes in place owned by the copier, so it is not raised.
        self._inventory._run(
            ["docker", "exec", "-u", "0", container, "sh", "-c",
             FS_CHOWN_SCRIPT, "_", target, directory],
            timeout=15,
        )
        return {"ok": True, "path": target, "size": len(data)}

    @staticmethod
    def _read_bounded(source: Any, max_bytes: int) -> bytes:
        if hasattr(source, "read"):
            data = source.read(max_bytes + 1)
        else:
            data = bytes(source)
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("Upload the file as binary content.")
        if len(data) > max_bytes:
            raise ValueError("That file is larger than the upload limit.")
        return bytes(data)
