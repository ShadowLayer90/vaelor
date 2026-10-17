"""Build the patched vLLM attention file on the machine that serves it: fetch, patch, edit, verify.

**VD-215, owner decision 2026-10-12: "for the license, the installer should
fetch it".** Vaelor is GPL-2.0-only and the patched file is vLLM's
(Apache-2.0), so neither Vaelor's wheel nor its repository carries any vLLM
code - not the file, not a line of it. Instead, on each machine:

1. **Fetch** (:func:`fetch_diff`, the installer's step): upstream pull request
   vllm-project/vllm#59448's diff, from a URL naming both commits. GitHub
   renders that diff itself (index lines, unrelated files), so the bytes are
   not what is pinned: the parsed hunks for the ONE file this uses are
   (:func:`hunks_digest`), and the diff is kept root-owned in the overlay's
   folder (`vllm_overlays.OVERLAY_ROOT`).
2. **Read the image's own file** out of the pinned image on that machine
   (:func:`image_file_argv`: ``docker run --rm --pull never --network none
   --entrypoint base64``, never a download; base64 so the bytes survive a
   transport that strips its output) and accept it only with the pinned
   pristine sha256.
3. **Apply the PR's hunks for that one file** (:func:`apply_hunks`) - not its
   ``vllm/envs.py`` or test hunks - with an applier that takes no fuzz: every
   hunk's old lines must match exactly; it may sit some lines from where the
   diff says, and an insertion lands where ``patch(1)`` puts it.
4. **Apply Vaelor's own edits** (`vllm_overlay_edits.vaelor_edits`). They
   are located by LINE NUMBER in the patched file and the sha256 of the line
   found there - the pristine file and the hunks are both pinned, so the
   positions are fixed - and they quote no vLLM line: where a vLLM line is
   reused, it is taken from the file being built, by position and hash. The
   edits read the setting from ``os.environ`` (an overlay replaces one file,
   so ``envs.py`` cannot register it), add the gfx1151 prompt-reading launch
   shape (VD-215), and add Vaelor's decode/verify and prompt kernels with
   their dispatch (VD-216).
5. **Accept the result only with the pinned final sha256**
   (`vllm_overlays.VllmOverlay.built_sha256`) and install it root-owned
   ``0644`` in the root-only overlay folder, where the root bridge verifies it
   again before every mount.

Any failure leaves no file and a plain reason (:func:`record_status`); that
machine then serves WITHOUT the draft model (owner, VD-215) and says why.

**Where it runs.** The installer runs ``fetch`` (network) and, when the 0.27
image is already on the box, ``build``. The vLLM image is pulled per machine
at the first cluster deploy, not at install, so on the controller the root
bridge builds the file at the first start that needs it, from the diff the
installer fetched (`vllm_overlays.root_overlay_file`), and on a worker the
controller builds that worker's copy from the worker's OWN image file and
pushes it, re-checking its hash there (`vllm_overlays.ensure_overlay_on_worker`).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import http.client
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from typing import Callable, List, Optional, Sequence, Tuple

from .vllm_overlays import (
    BUILT_NAME, DIFF_NAME, OVERLAY_ROOT, OVERLAYS, OverlayUnavailable,
)
from .vllm_overlay_edits import vaelor_edits

#: The one file of the PR that is applied; its other hunks are refused.
PATCHED_FILE = "vllm/v1/attention/ops/triton_unified_attention.py"

#: The most bytes a diff, or the image's file, may be.
_LIMIT = 1 << 20

#: The whole download's deadline, not only each socket read's.
FETCH_DEADLINE_SECONDS = 120

# --- The diff ---------------------------------------------------------------

Hunk = Tuple[int, List[str], List[str]]


def file_hunks(diff: str, path: str = PATCHED_FILE) -> List[Hunk]:
    """``[(old start, old lines, new lines)]`` of the diff's hunks for ``path`` only.

    A unified diff as git writes it. Lines keep their newline. A file the diff
    does not touch, a malformed hunk header or a hunk whose line counts do not
    add up is refused.
    """
    lines = diff.splitlines(keepends=True)
    hunks: List[Hunk] = []
    index, inside = 0, False
    while index < len(lines):
        line = lines[index]
        if line.startswith("diff --git "):
            inside = line.rstrip("\n").endswith(" b/" + path)
            index += 1
            continue
        if not inside or not line.startswith("@@ "):
            index += 1
            continue
        found = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if not found:
            raise OverlayUnavailable("The vLLM change's diff has a hunk this builder cannot read.")
        old_start = int(found.group(1))
        old_count = int(found.group(2) or 1) if found.group(2) != "0" else 0
        new_count = int(found.group(4) or 1) if found.group(4) != "0" else 0
        old: List[str] = []
        new: List[str] = []
        index += 1
        while index < len(lines) and (len(old) < old_count or len(new) < new_count):
            body = lines[index]
            if body.startswith("\\"):
                index += 1
                continue
            tag, text = body[:1], body[1:]
            if tag == " ":
                old.append(text)
                new.append(text)
            elif tag == "-":
                old.append(text)
            elif tag == "+":
                new.append(text)
            else:
                raise OverlayUnavailable("The vLLM change's diff ends inside a hunk.")
            index += 1
        if len(old) != old_count or len(new) != new_count:
            raise OverlayUnavailable("The vLLM change's diff has a hunk whose lengths do not add up.")
        hunks.append((old_start, old, new))
    if not hunks:
        raise OverlayUnavailable("The vLLM change's diff does not touch the attention file.")
    return hunks


def hunks_digest(hunks: Sequence[Hunk]) -> str:
    """The sha256 of the hunks in one canonical form: what the pin is about.

    Compact JSON of ``[old start, old lines, new lines]`` per hunk. Whatever
    GitHub writes around them - index lines, other files - does not move it.
    """
    canonical = json.dumps(
        [[start, list(old), list(new)] for start, old, new in hunks],
        separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def apply_hunks(text: str, hunks: Sequence[Hunk]) -> str:
    """``text`` with every hunk applied, exactly: no fuzz, no partial context.

    Each hunk's old lines must appear verbatim. It is placed at the exact
    match nearest to where the diff says (an offset, as ``patch --fuzz=0``
    allows), never before the previous hunk's end; no match refuses the build.
    A pure insertion (``@@ -N,0 ...``) goes AFTER old line ``N``, as
    ``patch(1)`` puts it.
    """
    lines = text.splitlines(keepends=True)
    out: List[str] = []
    # ``offset`` is the last hunk's distance from where the diff placed it,
    # carried to the next as ``patch`` does; line numbers are the old file's.
    cursor, offset = 0, 0
    for old_start, old, new in hunks:
        # The 0-based index the old lines start at: line N is index N-1, and
        # an insertion after line N starts at index N.
        stated = old_start if not old else old_start - 1
        expected = stated + offset
        candidates = [
            at for at in range(cursor, len(lines) - len(old) + 1)
            if lines[at:at + len(old)] == old
        ]
        if not old:
            candidates = [expected] if cursor <= expected <= len(lines) else []
        if not candidates:
            raise OverlayUnavailable(
                "The vLLM change does not apply to this image's attention file "
                "exactly (a hunk's lines are not there), so it was not applied."
            )
        at = min(candidates, key=lambda candidate: abs(candidate - expected))
        out.extend(lines[cursor:at])
        out.extend(new)
        offset = at - stated
        cursor = at + len(old)
    out.extend(lines[cursor:])
    return "".join(out)


def build(pristine: bytes, diff: bytes, overlay) -> bytes:
    """The overlay's bytes from the image's own file and the PR's diff, every hash checked."""
    if hashlib.sha256(pristine).hexdigest() != overlay.pristine_sha256:
        raise OverlayUnavailable(
            "The vLLM image's own attention file is not the one Vaelor's patch "
            "was made for (its checksum differs)."
        )
    try:
        hunks = file_hunks(diff.decode("utf-8"), PATCHED_FILE)
        if hunks_digest(hunks) != overlay.hunks_sha256:
            raise OverlayUnavailable(
                "The downloaded vLLM change is not the pinned one (its attention-file "
                "changes differ). Re-run the Vaelor installer to fetch it again."
            )
        text = vaelor_edits(apply_hunks(pristine.decode("utf-8"), hunks))
    except UnicodeDecodeError as error:
        raise OverlayUnavailable("The vLLM change or the image's file is not text.") from error
    built = text.encode("utf-8")
    if hashlib.sha256(built).hexdigest() != overlay.built_sha256:
        raise OverlayUnavailable(
            "The patched attention file built on this machine is not the pinned "
            "result (its checksum differs), so it is not used."
        )
    return built


# --- Fetching, reading the image, installing ----------------------------------

def fetch_diff(
    overlay, *, opener: Callable[..., object] = urllib.request.urlopen,
    clock: Callable[[], float] = time.monotonic,
) -> bytes:
    """The PR's diff from its pinned URL, accepted only when its hunks are the pinned ones.

    Bounded twice: each socket read by ``timeout`` and the whole download by
    :data:`FETCH_DEADLINE_SECONDS`. Every failure, a short read included, is
    an :class:`OverlayUnavailable` in plain words.
    """
    deadline = clock() + FETCH_DEADLINE_SECONDS
    data = b""
    try:
        with opener(overlay.diff_url, timeout=30) as response:
            while len(data) <= _LIMIT:
                if clock() > deadline:
                    raise OverlayUnavailable(
                        "The vLLM change did not finish downloading within {} seconds."
                        .format(FETCH_DEADLINE_SECONDS))
                chunk = response.read(65536)
                if not chunk:
                    break
                data += chunk
    except (OSError, http.client.HTTPException, ValueError) as error:
        raise OverlayUnavailable(
            "The vLLM change could not be downloaded from GitHub ({}).".format(
                str(error)[:160] or type(error).__name__)) from error
    try:
        pinned = len(data) <= _LIMIT and hunks_digest(
            file_hunks(data.decode("utf-8"))) == overlay.hunks_sha256
    except (UnicodeDecodeError, OverlayUnavailable):
        pinned = False
    if not pinned:
        raise OverlayUnavailable(
            "GitHub served a vLLM change that is not the pinned one, so it was not kept."
        )
    return data


def image_file_argv(overlay) -> List[str]:
    """The fixed command that prints the image's own file as base64: no network, no GPU,
    no download. Base64 because the SSH transport strips its output."""
    return [
        "docker", "run", "--rm", "--pull", "never", "--network", "none",
        "--entrypoint", "base64",
        overlay.image, overlay.container_path,
    ]


def decode_image_file(output: object) -> bytes:
    """The bytes :func:`image_file_argv` printed, refused if they are not base64."""
    text = output.decode("ascii", "replace") if isinstance(output, bytes) else str(output)
    packed = "".join(text.split())
    try:
        data = base64.b64decode(packed, validate=True)
    except (binascii.Error, ValueError) as error:
        raise OverlayUnavailable(
            "The vLLM image's attention file could not be read: the image did "
            "not print it as expected."
        ) from error
    if not data or len(data) > _LIMIT:
        raise OverlayUnavailable("The vLLM image's attention file could not be read.")
    return data


def read_image_file(overlay, run: Callable[..., object] = subprocess.run, timeout: int = 60) -> bytes:
    """The image's own file, as bytes, read on THIS machine; refused in plain words."""
    try:
        completed = run(image_file_argv(overlay), capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise OverlayUnavailable(
            "The vLLM image's attention file could not be read ({}).".format(str(error)[:160])
        ) from error
    stdout = getattr(completed, "stdout", b"") or b""
    if getattr(completed, "returncode", 1) != 0 or not stdout:
        raise OverlayUnavailable(
            "The vLLM image's attention file could not be read: is the vLLM 0.27 "
            "image on this machine?"
        )
    return decode_image_file(stdout)


STATUS_FILE = "status.json"


def _write_folder(overlay, root: str, owner: Tuple[int, int], files: Sequence[Tuple[str, str]]) -> None:
    """Write ``files`` into the overlay's root-owned folder, no link followed, atomically each."""
    from . import bridge_safe_fs as safe_fs

    with safe_fs.directory(root) as root_fd:
        handle = safe_fs.ensure_root_directory(root_fd, overlay.key, mode=0o755, owner=owner)
        try:
            for name, text in files:
                safe_fs.write_file(handle, name, text, mode=0o644)
        finally:
            os.close(handle)


def _status(overlay, reason: Optional[str]) -> str:
    body = {"key": overlay.key, "ready": reason is None, "reason": reason or ""}
    return json.dumps(body, sort_keys=True) + "\n"


def record_status(
    overlay, reason: Optional[str], *, root: str = OVERLAY_ROOT, owner: Tuple[int, int] = (0, 0),
) -> None:
    """Record whether the overlay is ready on this machine, and why not (LESSONS 1, 19).

    Best effort: a status that cannot be written never hides the reason the
    caller is already raising.
    """
    try:
        _write_folder(overlay, root, owner, [(STATUS_FILE, _status(overlay, reason))])
    except (OSError, ValueError):
        pass


def install_built(
    overlay, built: bytes, *, root: str = OVERLAY_ROOT, owner: Tuple[int, int] = (0, 0),
) -> str:
    """Write the built file root-owned ``0644`` into its root-only folder; its path back."""
    _write_folder(overlay, root, owner, [
        (BUILT_NAME, built.decode("utf-8")), (STATUS_FILE, _status(overlay, None)),
    ])
    return "{}/{}/{}".format(root, overlay.key, BUILT_NAME)


def read_status(overlay, *, root: str = OVERLAY_ROOT) -> dict:
    """The recorded status, ``{}`` when none was written or it cannot be read."""
    try:
        with open("{}/{}/{}".format(root, overlay.key, STATUS_FILE), encoding="utf-8") as handle:
            value = json.loads(handle.read(4096))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def main(
    argv: Sequence[str], *, root: str = OVERLAY_ROOT,
    fetch: Callable[..., bytes] = fetch_diff,
    read_image: Callable[..., bytes] = read_image_file,
) -> int:
    """``fetch`` or ``build``, as root, from the installer. Never fails the install.

    ``fetch`` downloads the pinned diff into each overlay's folder; ``build``
    builds the file from it and the image's own copy, when the image is on
    this machine (it usually is not at a first install: the first deploy
    pulls it, and the root bridge builds the file then). Every outcome, every
    failure included, is written to the overlay's status and printed.
    """
    from .post_install_steps import within_seconds
    from .vllm_overlays import fetched_diff

    within = within_seconds(argv, ("fetch", "build"))
    if within is None:
        sys.stderr.write("usage: python -m vaelor.vllm_overlay_build fetch|build "
                         "[--within SECONDS]\n")
        return 2
    verb = argv[1]
    for overlay in OVERLAYS.values():
        reason: Optional[str] = None
        try:
            if verb == "fetch":
                diff = fetch(overlay)
                _write_folder(overlay, root, (0, 0), [(DIFF_NAME, diff.decode("utf-8"))])
            else:
                diff = fetched_diff(overlay, root)
                if diff is None:
                    raise OverlayUnavailable(
                        "The vLLM change has not been downloaded; re-run the Vaelor "
                        "installer with internet access.")
                # VD-217 (R98): the image read stays inside the update's share.
                pristine = read_image(overlay) if within == float("inf") else read_image(
                    overlay, timeout=max(1, int(min(within, 60))))
                install_built(overlay, build(pristine, diff, overlay), root=root)
        except Exception as error:  # an installer step never fails the install
            reason = str(error)[:400] or type(error).__name__
        if reason is not None:
            record_status(overlay, reason, root=root)
        print("vLLM attention patch {} ({}): {}".format(
            verb, overlay.key, "done" if reason is None else reason))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
