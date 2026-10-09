"""What a Performance marker calls the serving line it happened on (B9).

A marker names its line (ACC-200): a machine, or the whole model a split
serves. On a mixed placement one machine carries two lines - llama.cpp serving
AI Chat on the controller, and the controller's share of a vLLM split - so
"Stopped serving (This Vaelor controller)" did not say which of the two
stopped, when only llama.cpp had. A line is now named by what serves on it:
the engine, then the machine ("llama.cpp on This Vaelor controller"), or the
engine and "whole model" for a split.

**Default chosen by Claude (B9), not the owner:** the engine's own name is the
word, because it is what differs between the two lines on one machine and is
what the rest of the dashboard already says; the deployment's name was the
alternative and is not used, since llama.cpp's AI Chat model has no deployment
row to name. An engine the rows do not name, or two in one bucket (a switch),
leaves the machine alone, as before - never a guess.

The words are the backend's (the browser holds none). Pure: the caller hands
in the range's folded buckets (`performance_dashboard_range.fold_rows`).
"""

from __future__ import annotations

from typing import Any, Mapping

from .model_usage import ENGINE_LLAMACPP, ENGINE_VLLM

#: How each engine is named on a marker.
ENGINE_WORDS = {ENGINE_VLLM: "vLLM", ENGINE_LLAMACPP: "llama.cpp"}
#: A line one engine serves on one machine.
LINE_ON_MACHINE = "{engine} on {machine}"
#: The whole model a split serves.
LINE_WHOLE_MODEL = "{engine}, {whole}"


def line_name(machine: str, engine: str, *, whole: bool) -> str:
    """The marker's name for a line: ``machine`` is what the marker said before."""
    word = ENGINE_WORDS.get(str(engine or ""), "")
    if not word:
        return machine
    if whole:
        return LINE_WHOLE_MODEL.format(engine=word, whole=machine)
    return LINE_ON_MACHINE.format(engine=word, machine=machine)


#: How many buckets from a marker's own a bucket may be and still name its
#: engine (review round 1). A poller writes a row every few seconds while a
#: line serves, so its first or last row lands in the transition's bucket or
#: the next; a bucket further off says what served at another time.
MAX_BUCKETS_AWAY = 2

#: Which side of the moment a marker's engine is read from: a start is named
#: by what serves FROM it, a stop by what served UP TO it.
_LATER, _EARLIER = "started", "stopped"


def _one_engine(bucket: Mapping[str, Any]) -> str:
    engines = {name for name in (bucket.get("_engines") or ()) if name}
    return next(iter(engines)) if len(engines) == 1 else ""


def engine_near(buckets: Mapping[int, Mapping[str, Any]], index: int, *, kind: str = "") -> str:
    """The one engine a marker at bucket ``index`` is named by, or ``""``.

    A start (``kind`` ``started``) reads the nearest bucket at or after it; a
    stop (``stopped``) the nearest at or before it. Any other marker names an
    engine only when the nearest buckets on both sides agree - two sides that
    disagree are not settled by a tie-break (LESSONS 5). Nothing further than
    :data:`MAX_BUCKETS_AWAY` names anything, and a bucket that recorded two
    engines (a switch) or none names nothing: the marker then names the
    machine alone.
    """
    near = [at for at in buckets if abs(at - index) <= MAX_BUCKETS_AWAY]
    later = min((at for at in near if at >= index), default=None)
    earlier = max((at for at in near if at <= index), default=None)
    if kind == _LATER:
        return "" if later is None else _one_engine(buckets[later])
    if kind == _EARLIER:
        return "" if earlier is None else _one_engine(buckets[earlier])
    sides = {_one_engine(buckets[at]) for at in (later, earlier) if at is not None}
    return next(iter(sides)) if len(sides) == 1 else ""
