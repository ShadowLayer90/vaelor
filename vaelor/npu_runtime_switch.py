"""Move the NPU Assistant to its pinned serving pair, and back if the new pair does not serve (VD-219).

The rule is VD-217's: the installer and the console update leave a box in the
same state, and an update never leaves it half upgraded. For the NPU Assistant
that means FastFlowLM and its model package move **together**, and only to a
pair that then answers a real request on the neural processor; otherwise the
box is put back exactly as it was.

**How a box gets here.** One post-install step (`post_install_steps.STEPS`,
``start``) runs inside ``vaelor-refresh-models --apply`` - the command both the
installer and the console update's root broker run (LESSONS 6: one list, no
second copy in the installer). The step decides in a second and, when the box
holds another pair, starts :data:`UPDATE_UNIT` (a transient unit, bounded by
:data:`UPDATE_SECONDS`): the package is gigabytes and cannot fit the 450 s the
steps share, and the reload behind them must not wait for it. That unit runs
``update``: it stages the pair (`npu_runtime_files.stage_pair`: every byte
checked against its pin) while the old pair keeps serving, then asks the root
bridge to make the move (``flm_runtime_switch``, no arguments: the pair is this
package's own pin).

**The move** (:func:`switch_to_staged`) runs inside the bridge, under the lock
every ``flm_start``/``flm_stop`` takes, so the executor's restart watch cannot
launch FastFlowLM from a half-moved folder:

1. :data:`JOURNAL` records the move and the names on each side, then the
   old entries go to :data:`PREVIOUS_DIR` and the staged ones into place.
   Every step is a rename inside one folder that never moves (see
   `npu_runtime_files`).
2. FastFlowLM is started on the new pair with the tag, window and port it
   had, and :func:`serves` asks it for a short answer. A server that answers
   ``/v1/models`` but cannot run the model is not "serving" (LESSONS 1).
3. On success the journal says ``done`` and the previous pair is kept for a
   manual return until the next move. On failure every entry goes back -
   the new pair to its stage folder, so a retry downloads nothing - the old
   pair is started as it was, and the journal keeps the reason, which stops
   the step from retrying the same pair at every update (LESSONS 22: a
   failure the box would otherwise repeat forever).

A move cut off half way (power loss, a killed bridge) is put back by
:func:`recover` at the bridge's next FastFlowLM operation, because the
journal still says ``swapping`` or ``checking``. Both directions are
idempotent: each name is moved only from where it would be at that point.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .flm_service import FLM_PENALTIES
from .npu_runtime_files import (
    JOURNAL,
    PREVIOUS_DIR,
    RuntimeFilesError,
    STAGE_DIR,
    STAGED_TREE,
    installed_key,
    installed_runtime,
    package_problems,
    payload_entries,
    runtime_refuses,
    stage_pair,
    staged_dir,
    staged_ready,
)
from .npu_runtime_pin import ServingPair, serving_pair

#: The transient unit the post-install step starts (``systemd-run`` refuses a
#: second while one runs, which is the point).
UPDATE_UNIT = "vaelor-npu-runtime-update"
#: What the unit may take: a slow link moving ~3.5 GB, plus the move.
UPDATE_SECONDS = 3 * 3600

#: What the bridge call that makes the move may take: a stop, the renames, a
#: start and :data:`HEALTH_SECONDS` - and the same again for a return.
SWITCH_CALL_SECONDS = 600

#: How long a freshly started FastFlowLM has to answer one short request. It
#: loads the model at the first request (3 s to ``/v1/models`` on the Z2).
HEALTH_SECONDS = 180
HEALTH_POLL_SECONDS = 2.0

#: The question the health check asks; any non-empty answer passes.
HEALTH_PROMPT = "Reply with the single word OK."

#: The journal's phases. ``swapping``/``checking``/``returning`` are unfinished.
SWAPPING, CHECKING, RETURNING, DONE, RETURNED = (
    "swapping", "checking", "returning", "done", "returned")
UNFINISHED = frozenset({SWAPPING, CHECKING, RETURNING})

#: How the step's line ends when nothing is left to do (`post_install_steps.step_status`).
STEP_DONE_LINE = "NPU Assistant runtime ({}): done"
NOT_STARTED_LINE = "NPU Assistant runtime ({}): the move could not be started ({})"
#: Why nothing moves when `npu_runtime_pin.TARGET_PAIR` is None.
NO_TARGET = "no new pair is pinned in this release"

def _default_context() -> int:
    """The window FastFlowLM is checked with when it was not running before the move."""
    from .inference_context import NPU_RECOMMENDED_CONTEXT_TOKENS

    return int(NPU_RECOMMENDED_CONTEXT_TOKENS)


# --- The journal ------------------------------------------------------------------

def read_journal(root: Path) -> Dict[str, Any]:
    try:
        value = json.loads((root / JOURNAL).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def write_journal(root: Path, record: Dict[str, Any]) -> None:
    """Replace the journal in one rename, flushed first, so a crash leaves the old or the new."""
    record = dict(record, at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    temporary = root / (JOURNAL + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, root / JOURNAL)


# --- Moving entries ---------------------------------------------------------------

def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _move_forward(root: Path, tree: Path, old: Sequence[str], new: Sequence[str]) -> None:
    """Old entries to :data:`PREVIOUS_DIR`, staged ones into place; names are relative paths.

    Refuses (``OSError``) rather than overwrite anything already kept: the
    kept copy is the way back (R165 M14).
    """
    previous = root / PREVIOUS_DIR
    previous.mkdir(mode=0o755, exist_ok=True)
    for name in old:
        if _present(root / name):
            if _present(previous / name):
                raise FileExistsError("{} is already kept from an earlier move".format(name))
            (previous / name).parent.mkdir(parents=True, exist_ok=True)
            os.rename(root / name, previous / name)
    for name in new:
        if _present(tree / name):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            os.rename(tree / name, root / name)


def _move_back(root: Path, tree: Path, old: Sequence[str], new: Sequence[str]) -> None:
    """Undo :func:`_move_forward` from any point it reached; safe to run twice."""
    previous = root / PREVIOUS_DIR
    for name in new:
        here = root / name
        if _present(here) and not _present(tree / name):
            (tree / name).parent.mkdir(parents=True, exist_ok=True)
            os.rename(here, tree / name)
    for name in old:
        kept = previous / name
        if _present(kept) and not _present(root / name):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            os.rename(kept, root / name)


def unsettled(root: Path) -> bool:
    """Whether the journal says a move is half done (cheap: one small file)."""
    return read_journal(root).get("phase") in UNFINISHED


def recover(root: Path) -> str:
    """Put back a move that was cut off; ``""`` when there was none. Raises ``OSError`` if it cannot.

    Called before every FastFlowLM start while the journal is unsettled (R165
    R1: once per bridge process was not enough - a move that failed in the
    same process left the next start to launch from a half-moved folder).
    """
    journal = read_journal(root)
    if journal.get("phase") not in UNFINISHED:
        return ""
    tree = root / STAGE_DIR / str(journal.get("to") or "") / STAGED_TREE
    _move_back(root, tree, journal.get("old") or [], journal.get("new") or [])
    write_journal(root, dict(journal, phase=RETURNED, cause=TRANSIENT,
                             attempts=int(journal.get("attempts") or 0) + 1,
                             reason="the move was interrupted ({}), so it was put back".format(
                                 journal.get("phase"))))
    return "An interrupted move to {} was put back to {}.".format(
        journal.get("to"), journal.get("from"))


def pair_fingerprint(pair: ServingPair) -> str:
    """What a returned move is remembered by: the pair, its package's bytes and its probe."""
    return "{}|{}|{}".format(pair.key, pair.package.sha256, pair.package.probe_answer)


#: Why a move was put back (R172 item 1). A bad PACKAGE - its files are not
#: the manifest's, it answers the probe wrongly or with nothing - blocks that
#: exact pin until a new package or probe is pinned (`pair_fingerprint`). A
#: TRANSIENT failure - an I/O error, a timeout, a server that would not start or
#: answer, a move cut off - is tried once more at the next update, then blocks
#: until root runs ``retry``.
PACKAGE, TRANSIENT = "package", "transient"
TRANSIENT_ATTEMPTS = 2

#: The reasons `serves` and the move give for a bad package; anything else is transient.
PACKAGE_REASONS = ("it answered", "its files changed")

BLOCKED_PACKAGE = "a bad package: not tried again until a new package is pinned"
RETRY_ONCE = "a transient failure: tried once more at the next update"
BLOCKED_TRANSIENT = ("a transient failure, twice: not tried again until root runs "
                     "`python -I -m vaelor.npu_runtime_switch retry`")


def cause_of(reason: str) -> str:
    """:data:`PACKAGE` or :data:`TRANSIENT` for a failed check's reason."""
    return PACKAGE if str(reason).startswith(PACKAGE_REASONS) else TRANSIENT


def returned_state(root: Path, pair: ServingPair) -> Dict[str, str]:
    """How the last return of this exact pair stands: ``blocked`` (bool as text), why, and the verdict.

    Empty when this pin was never put back (a re-pinned package or probe under
    the same key is a new pin), or when the record is not a return.
    """
    journal = read_journal(root)
    if journal.get("phase") != RETURNED or journal.get("fingerprint") != pair_fingerprint(pair):
        return {}
    cause = journal.get("cause") or TRANSIENT
    attempts = int(journal.get("attempts") or 1)
    if cause == PACKAGE:
        verdict, blocked = BLOCKED_PACKAGE, True
    elif attempts >= TRANSIENT_ATTEMPTS:
        verdict, blocked = BLOCKED_TRANSIENT, True
    else:
        verdict, blocked = RETRY_ONCE, False
    return {"blocked": "yes" if blocked else "", "reason": str(journal.get("reason") or ""),
            "verdict": verdict, "cause": cause}


def returned_reason(root: Path, pair: ServingPair) -> str:
    """Why this exact pin may not be moved to now, with the verdict; ``""`` when it may."""
    state = returned_state(root, pair)
    if not state.get("blocked"):
        return ""
    return "{} ({})".format(state["reason"] or "it failed its check", state["verdict"])


def clear_returned(root: Path) -> bool:
    """Forget a returned move so the same pair may be tried again (root only: ``retry``).

    Only a settled return is cleared: an unfinished journal stays unfinished, so
    the next start still puts the folder back.
    """
    journal = read_journal(root)
    if journal.get("phase") != RETURNED:
        return False
    write_journal(root, dict(journal, phase="retry-allowed"))
    return True


# --- Serving check ----------------------------------------------------------------

def answer_fingerprint(answer: Any) -> str:
    """What a chat answer said, as one comparable string: its text, else its tool calls.

    ``""`` means it said nothing at all. A tool-call answer with no text reads
    ``tool:<name>,...`` (``tool:`` alone for an empty call: FastFlowLM parses a
    bare ``<tool_call>`` into no calls and no text).
    """
    choices = answer.get("choices") if isinstance(answer, dict) else None
    message = (choices[0] or {}).get("message") if isinstance(choices, list) and choices else None
    if not isinstance(message, dict):
        return ""
    text = str(message.get("content") or "").strip()
    if text:
        return text
    calls = message.get("tool_calls")
    if isinstance(calls, list) and calls:
        names = [str(((call or {}).get("function") or {}).get("name") or "")
                 for call in calls if isinstance(call, dict)]
        return "tool:" + ",".join(names)
    usage = answer.get("usage") if isinstance(answer, dict) else None
    produced = int((usage or {}).get("completion_tokens") or 0) if isinstance(usage, dict) else 0
    return "tool:" if produced else ""


def loopback_url(port: Any, path: str) -> str:
    """``http://127.0.0.1:<port><path>``: the only place the health check ever connects.

    The host is fixed and the port must be a whole number in range, so no
    endpoint, hosted kind or key can be passed in (VD-207 / VD-219: this module
    posts chat completions only to the bridge's own FastFlowLM).
    """
    if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
        raise ValueError("the FastFlowLM health check takes a loopback port number only")
    return "http://127.0.0.1:{}{}".format(port, path)


def ask(port: int, tag: str, prompt: str, *, timeout: float, max_tokens: int = 32) -> Any:
    """One non-streamed chat request at temperature 0; the parsed answer."""
    body = json.dumps({
        "model": tag, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": False, **FLM_PENALTIES,
    }).encode("utf-8")
    request = urllib.request.Request(loopback_url(port, "/v1/chat/completions"),
                                     data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read(1 << 20).decode("utf-8", "replace"))


def serves(port: int, tag: str, *, probe_prompt: str = "", probe_answer: str = "",
           seconds: float = HEALTH_SECONDS,
           clock: Callable[[], float] = time.monotonic,
           sleep: Callable[[float], None] = time.sleep) -> str:
    """``""`` when FastFlowLM on ``port`` serves the expected model for ``tag``; else why not.

    With a recorded probe (`npu_runtime_pin.Package.probe_prompt`), the answer
    must be the package's own, exactly, at temperature 0: a server answering
    with another model is not serving this package (VD-219 window 2: the stock
    model answered "OK" where the fine-tune answers with a tool call). Without
    one, any answer that says something passes.
    """
    deadline = clock() + seconds
    last = "it did not answer"
    while clock() < deadline:
        try:
            with urllib.request.urlopen(loopback_url(port, "/v1/models"),
                                        timeout=5) as response:
                if response.status == 200:
                    break
        except (OSError, urllib.error.URLError) as error:
            last = "it did not answer ({})".format(str(error)[:120])
        sleep(HEALTH_POLL_SECONDS)
    else:
        return last
    try:
        answer = ask(port, tag, probe_prompt or HEALTH_PROMPT, timeout=max(5.0, deadline - clock()))
    except (OSError, urllib.error.URLError, ValueError) as error:
        return "it did not complete a short answer ({})".format(str(error)[:160])
    said = answer_fingerprint(answer)
    if probe_prompt and said != probe_answer:
        return "it answered {!r}, not this package's recorded answer {!r}".format(
            said[:80], probe_answer[:80])
    if not said:
        return "it answered with nothing"
    return ""


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


# --- The move ---------------------------------------------------------------------

def switch_to_staged(root: Path, pair: ServingPair, launcher: Any, *,
                     check: Optional[Callable[[int, str], str]] = None,
                     restore_check: Optional[Callable[[int, str], str]] = None,
                     critical: Callable[[], Any] = contextlib.nullcontext,
                     pick_port: Callable[[], int] = free_port) -> Dict[str, Any]:
    """Move FastFlowLM's folder to the staged ``pair``, prove it serves, else put it back.

    ``launcher`` is the bridge's FastFlowLM process (``status``/``start``/``stop``).
    ``critical`` is the lock its starts and stops take: held only while
    FastFlowLM is stopped, the folder renamed and a process started - never
    across the hashing or the health check (R165 R7: the bridge's lock froze
    every other verb for minutes). The bridge refuses ``flm_start``/``flm_stop``
    from anyone else while a move runs (`hardware_bridge.flm_runtime_switch`).

    **Never raises** (R165 R1). Anything that fails part way puts the old pair
    back and says so; if even that fails, the journal stays unsettled and every
    later start puts it back first, or refuses.

    The old pair is checked after a return with ``restore_check``: by default
    a plain answer check, never the new package's probe (R165 R3).
    """
    explicit = check is not None
    if check is None:
        def check(port: int, tag: str) -> str:
            return serves(port, tag, probe_prompt=pair.package.probe_prompt,
                          probe_answer=pair.package.probe_answer)
    if restore_check is None:
        restore_check = check if explicit else (lambda port, tag: serves(port, tag))
    folder = pair.package.folder
    try:
        with critical():
            recover(root)
        if installed_key(root, folder) == pair.key:
            return {"moved": True, "pair": pair.key, "note": "already on this pair"}
        refused = returned_reason(root, pair)
        if refused:
            return {"moved": False, "pair": pair.key, "reason": (
                "this pair was put back earlier because {}").format(refused),
                "cause": returned_state(root, pair).get("cause", "")}
        if not staged_ready(root, pair):
            return {"moved": False, "pair": pair.key, "reason": "the pair is not staged"}
        tree = staged_dir(root, pair) / STAGED_TREE
        refused = runtime_refuses(tree, pair.package)
        if refused:
            return {"moved": False, "pair": pair.key, "reason": refused}
        shutil.rmtree(root / PREVIOUS_DIR, ignore_errors=True)
        if _present(root / PREVIOUS_DIR):
            return {"moved": False, "pair": pair.key,
                    "reason": "the copy kept from the last move could not be cleared"}
    except (OSError, RuntimeFilesError) as error:
        return {"moved": False, "pair": pair.key,
                "reason": "the move could not begin ({})".format(str(error)[:200])}
    try:
        before = launcher.status() or {}
    except Exception as error:  # noqa: BLE001 - R172: never raises, and nothing has stopped
        return {"moved": False, "pair": pair.key, "cause": TRANSIENT,
                "reason": "FastFlowLM's state could not be read ({})".format(
                    str(error)[:160] or type(error).__name__)}
    was_running = bool(before.get("running"))
    tag = str(before.get("tag") or pair.package.tag) if was_running else pair.package.tag
    ctx = int(before.get("ctx_len") or 0) or _default_context()
    port = int(before.get("port") or 0) if was_running else 0
    port = port or pick_port()
    old, new = payload_entries(root, folder), payload_entries(tree, folder)
    from_key = installed_key(root, folder)
    earlier = read_journal(root)
    attempts = (int(earlier.get("attempts") or 0)
                if earlier.get("fingerprint") == pair_fingerprint(pair) else 0) + 1
    journal = {"from": from_key, "to": pair.key, "old": old, "new": new,
               "fingerprint": pair_fingerprint(pair), "attempts": attempts}
    moving = dict(root=root, tree=tree, old=old, new=new, journal=journal, launcher=launcher,
                  critical=critical, was_running=was_running, tag=tag, ctx=ctx, port=port,
                  restore_check=restore_check, pair=pair)
    try:
        with critical():
            launcher.stop()
            write_journal(root, dict(journal, phase=SWAPPING))
            _move_forward(root, tree, old, new)
            write_journal(root, dict(journal, phase=CHECKING))
            launcher.start(pair.package.tag, ctx_len=ctx, port=port)
    except Exception as error:  # noqa: BLE001 - R1: a failed move is put back, never raised
        return _put_back(moving, "the move failed part way ({})".format(
            str(error)[:200] or type(error).__name__))
    try:
        reason = check(port, pair.package.tag)
    except Exception as error:  # noqa: BLE001 - a check that breaks is a failed check
        reason = "it could not be checked ({})".format(str(error)[:160] or type(error).__name__)
    if not reason:
        # FastFlowLM may replace files it judges outdated with its own stock
        # model and then answer (window 2): what serves must still be the package.
        changed = package_problems(root / "models" / folder, pair.package)
        if changed:
            reason = "its files changed once FastFlowLM started ({})".format("; ".join(changed[:3]))
    if reason:
        return _put_back(moving, reason)
    try:
        with critical():
            write_journal(root, dict(journal, phase=DONE))
    except Exception as error:  # noqa: BLE001 - R172: unrecorded is not moved; put it back
        return _put_back(moving, "the finished move could not be recorded ({})".format(
            str(error)[:160] or type(error).__name__))
    if not was_running:
        try:
            with critical():
                launcher.stop()
        except Exception:  # noqa: BLE001 - moved and recorded; a stop that fails is only said
            pass
    shutil.rmtree(staged_dir(root, pair), ignore_errors=True)
    return {"moved": True, "pair": pair.key, "from": from_key, "port": port}


def _put_back(moving: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Return every entry and restart the old pair as it was; the result says how that went."""
    root, launcher, journal = moving["root"], moving["launcher"], moving["journal"]
    pair = moving["pair"]
    cause = cause_of(reason)
    result: Dict[str, Any] = {"moved": False, "pair": pair.key, "reason": reason, "cause": cause}
    def attempt(step: Callable[[], Any]) -> None:
        try:
            step()
        except Exception:  # noqa: BLE001 - the files coming back is what matters
            pass

    try:
        with moving["critical"]():
            attempt(launcher.stop)
            attempt(lambda: write_journal(root, dict(journal, phase=RETURNING, reason=reason)))
            _move_back(root, moving["tree"], moving["old"], moving["new"])
            # A journal that cannot be written still says SWAPPING/CHECKING; the
            # next start's recover() finds nothing to move and settles it.
            attempt(lambda: write_journal(root, dict(journal, phase=RETURNED, reason=reason,
                                                     cause=cause)))
            if moving["was_running"]:
                launcher.start(moving["tag"], ctx_len=moving["ctx"], port=moving["port"])
    except Exception as error:  # noqa: BLE001 - said; the journal keeps the box from starting
        result.update(back_on=installed_key(root, pair.package.folder), previous_serves=False,
                      previous_problem=(
                          "it could not be put back ({}); FastFlowLM is not started until it is"
                          .format(str(error)[:160] or type(error).__name__)))
        return result
    restored = ""
    if moving["was_running"]:
        try:
            restored = moving["restore_check"](moving["port"], moving["tag"])
        except Exception as error:  # noqa: BLE001 - said, never raised
            restored = str(error)[:160] or type(error).__name__
    result.update(back_on=installed_key(root, pair.package.folder),
                  previous_serves=(not restored) if moving["was_running"] else None,
                  previous_problem=restored)
    return result


# --- The post-install step and its unit -------------------------------------------

def _has_npu() -> bool:
    import glob

    from .flm_service import NPU_SYSFS_CLASS_GLOB

    return bool(glob.glob(NPU_SYSFS_CLASS_GLOB))


def step_line(root: Path, target: Optional[ServingPair],
              start_unit: Callable[[List[str]], Any], *, has_npu: Callable[[], bool] = _has_npu) -> str:
    """What the post-install step does, as its one line (`post_install_steps.step_status`).

    With no target pinned (`npu_runtime_pin.TARGET_PAIR` is ``None``) nothing is
    moved, whatever the box holds (R165 R2: a box with no model, or an extra
    model folder, read as "not the current pair" and started a 3.4 GB download
    that could only be refused).
    """
    if target is None:
        return STEP_DONE_LINE.format(NO_TARGET)
    if not has_npu() or not (root / "flm").exists():
        return STEP_DONE_LINE.format("not installed on this machine")
    if not installed_runtime(root):
        return ("NPU Assistant runtime ({}): not moved - FastFlowLM of an unknown version is "
                "installed; run the installer again").format(target.key)
    if installed_key(root, target.package.folder) == target.key:
        return STEP_DONE_LINE.format(target.key)
    refused = returned_reason(root, target)
    if refused:
        return ("NPU Assistant runtime ({}): not moved - the last attempt was put back because "
                "{}; it serves {}").format(target.key, refused, installed_key(root))
    retry = returned_state(root, target)
    try:
        done = start_unit(update_unit_argv())
    except (OSError, subprocess.SubprocessError) as error:
        return NOT_STARTED_LINE.format(target.key, str(error)[:160])
    if getattr(done, "returncode", 0) != 0:
        said = str(getattr(done, "stderr", "") or "").strip()[:160] or "systemd would not start the unit"
        if "already" in said:
            return "NPU Assistant runtime ({}): already moving in the background ({})".format(
                target.key, UPDATE_UNIT)
        return NOT_STARTED_LINE.format(target.key, said)
    return ("NPU Assistant runtime ({}): downloading in the background ({}){}; {} keeps "
            "serving until the new pair has answered on the NPU").format(
                target.key, UPDATE_UNIT,
                ", trying again after {}".format(retry["reason"]) if retry else "",
                installed_key(root))


def update_unit_argv(python: str = sys.executable) -> List[str]:
    """The fixed command the step hands systemd: nothing a client chooses."""
    return ["systemd-run", "--unit", UPDATE_UNIT, "--collect", "--no-block", "--quiet",
            "--property=RuntimeMaxSec={}".format(UPDATE_SECONDS),
            python, "-I", "-m", "vaelor.npu_runtime_switch", "update"]


def _start_unit(argv: List[str]) -> Any:
    return subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False,
                          stdin=subprocess.DEVNULL)


def run_update(root: Path, target: Optional[ServingPair], ask_bridge: Callable[[], Dict[str, Any]],
               stage: Callable[[Path, ServingPair], str] = stage_pair) -> str:
    """The unit's body: stage while the old pair serves, then have the bridge move. One line."""
    if target is None:
        return STEP_DONE_LINE.format(NO_TARGET)
    if installed_key(root, target.package.folder) == target.key:
        return STEP_DONE_LINE.format(target.key)
    refused = returned_reason(root, target)
    if refused:
        return "NPU Assistant runtime ({}): not moved - put back earlier because {}".format(
            target.key, refused)
    try:
        stage(root, target)
    except (RuntimeFilesError, OSError) as error:
        return "NPU Assistant runtime ({}): not moved - {}".format(target.key, str(error)[:300])
    try:
        result = ask_bridge() or {}
    except Exception as error:  # noqa: BLE001 - an old or absent bridge is said, not raised
        return ("NPU Assistant runtime ({}): staged, not moved - the hardware bridge did not "
                "make the move ({})").format(target.key, str(error)[:200])
    if result.get("moved"):
        return STEP_DONE_LINE.format(target.key)
    verdict = returned_state(root, target).get("verdict") or ""
    return "NPU Assistant runtime ({}): not moved - {}{}; it serves {}".format(
        target.key, result.get("reason") or "the bridge said nothing more",
        " ({})".format(verdict) if verdict else "", installed_key(root))


def _ask_bridge() -> Dict[str, Any]:
    from .hardware_bridge_client import HardwareBridgeClient

    return HardwareBridgeClient(timeout=SWITCH_CALL_SECONDS).flm_runtime_switch()


def _journal_line(line: str) -> None:
    try:
        import syslog

        syslog.openlog("vaelor-npu-runtime")
        syslog.syslog(syslog.LOG_INFO, line)
    except Exception:  # noqa: BLE001 - the printed line stands without it
        pass


def main(argv: Sequence[str]) -> int:
    """``start`` (the post-install step), ``update`` (its unit) or ``retry``, as root. Never fails an install.

    ``retry`` forgets a pair the box put back after it failed its check, then
    runs ``update``: the one way to try that same pair again (R165 R6).
    """
    from . import npu_runtime_pin
    from .flm_service import FLM_MODEL_PATH
    from .post_install_steps import is_root

    verb = list(argv[1:])
    if verb not in (["start"], ["update"], ["retry"]):
        sys.stderr.write("usage: python -I -m vaelor.npu_runtime_switch start|update|retry\n")
        return 2
    if not is_root():
        sys.stderr.write("The NPU runtime move writes a root-only folder; run it as root.\n")
        return 2
    root = Path(FLM_MODEL_PATH)
    target = npu_runtime_pin.TARGET_PAIR
    if verb == ["start"]:
        line = step_line(root, target, _start_unit)
    else:
        if verb == ["retry"]:
            clear_returned(root)
        line = run_update(root, target, _ask_bridge)
    print(line)
    _journal_line(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
