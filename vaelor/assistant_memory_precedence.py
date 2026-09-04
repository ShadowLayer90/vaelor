"""Live-source precedence for durable Assistant memory (VD-052 / VD-062, #97).

A durable fact that has a LIVE SOURCE must be *derived* from that source at read
time, never *recalled* from the store. Durable memory holds only the
non-derivable semantic residue; anything a live tool can answer is the tool's to
answer.

This is the ``raspberry_pi.detect_product`` rule (#211) applied to the memory
store: a live HAT ID EEPROM outranks the SunFounder-written ``.variant`` file,
because the file is a recollection and the EEPROM is the board stating its own
identity now. Here the ``.variant`` file is a ``device_fact`` / ``project_fact``
carrying a ``live_source`` binding, and the EEPROM is the live tool read.

The module is pure and has no store or tool dependency: it is handed a memory
dict and the live-context snapshot and returns a disposition. Both read paths
consult it — ``AssistantMemoryStore.context_for`` (which can derive) and the
deterministic ``grounded_memory_answer`` (which cannot, and so must refuse).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


#: No live source: the fact is non-derivable residue and may be served verbatim.
SETTLED = "settled"
#: A live source *and* a trustworthy live value: serve the live value, not the
#: recalled one. The live read is authoritative whether it agrees or disagrees.
DERIVED = "derived"
#: A live source but no trustworthy live value at read time: surface annotated
#: "verify live" and NEVER as a settled fact.
VERIFY_LIVE = "verify-live"

#: Flags the down-cycle reconciler writes onto a memory. Read here so a sweep's
#: verdict is consumed rather than discarded (LESSONS 11).
CONFIRMED = "confirmed"
STALE = "stale"
UNKNOWN = "unknown"
RECONCILE_FLAGS = frozenset({CONFIRMED, STALE, UNKNOWN})


def live_source_of(memory: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """The ``{"tool", "selector"}`` binding on a memory, or None.

    Accepts a nested dict (as ``_memory`` emits it) OR the raw JSON string a
    column holds before decoding. Handling the undecoded string matters for
    #97's own invariant: if a binding that is really present read as "no live
    source" (because it arrived as a str), ``servable_as_settled`` would return
    True and the fact would be served stale — a value written in one module and
    misread in another, the exact class this work guards against. A malformed
    binding is treated as absent rather than trusted.
    """
    if not isinstance(memory, dict):
        return None
    binding = memory.get("live_source")
    if isinstance(binding, str):
        try:
            binding = json.loads(binding)
        except (TypeError, ValueError):
            return None
    if not isinstance(binding, dict):
        return None
    tool = str(binding.get("tool", "")).strip()
    selector = str(binding.get("selector", "")).strip()
    if not tool or not selector:
        return None
    return {"tool": tool, "selector": selector}


def has_live_source(memory: Dict[str, Any]) -> bool:
    """True when this memory is derivable from a live tool and must not settle."""
    return live_source_of(memory) is not None


def _selector_value(selector: str, live_context: Optional[Dict[str, Any]]) -> Optional[str]:
    """Follow a dotted ``a.b.c`` selector into the live-context snapshot."""
    if not isinstance(live_context, dict):
        return None
    node: Any = live_context
    for step in selector.split("."):
        if isinstance(node, dict) and step in node:
            node = node[step]
        else:
            return None
    if node is None or isinstance(node, (dict, list)):
        return None
    text = str(node).strip()
    return text or None


def agrees(content: str, live_value: str) -> bool:
    """Whether a stored fact still describes the live value.

    Deliberately loose: stored facts are prose ("The dashboard runs on port
    8081") and live values are terse ("8081"). Agreement is containment either
    way, case-folded. Used only to distinguish `confirmed` from `stale`; the
    read-path guard does not depend on it, because a live source never settles.
    """
    left = " ".join(str(content).lower().split())
    right = " ".join(str(live_value).lower().split())
    if not right:
        return False
    return right in left or left in right


def resolve(
    memory: Dict[str, Any], live_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Classify one memory for a read path.

    Returns ``{disposition, verify_live, live_value, reason}``. ``verify_live``
    is the single boolean a caller needs to know it must not present the memory
    as settled.
    """
    source = live_source_of(memory)
    if source is None:
        return {
            "disposition": SETTLED,
            "verify_live": False,
            "live_value": None,
            "reason": "no live source",
        }
    flag = memory.get("reconcile_flag")
    live_value = _selector_value(source["selector"], live_context)
    if live_value is not None and flag != STALE:
        return {
            "disposition": DERIVED,
            "verify_live": False,
            "live_value": live_value,
            "reason": "derived from {}.{}".format(source["tool"], source["selector"]),
        }
    # A live source with nothing trustworthy to derive from right now: the
    # reconciler may already have flagged it, but either way it is never settled.
    reason = (
        "reconciler flagged stale"
        if flag == STALE
        else "no live read for {}.{}".format(source["tool"], source["selector"])
    )
    return {
        "disposition": VERIFY_LIVE,
        "verify_live": True,
        "live_value": None,
        "reason": reason,
    }


def decode_live_source(value: Any) -> Optional[Dict[str, str]]:
    """Parse a stored JSON live-source binding back to a dict, or None."""
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return live_source_of({"live_source": parsed})


def for_context(
    item: Dict[str, Any], live_context: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Shape one selected memory for the model, applying live-source precedence.

    A fact with no live source is emitted unchanged. A fact that has one is
    NEVER emitted as settled: it is either derived from the live read (and the
    stored text marked as history) or marked "verify live". This is the
    detect_product rule (#211) at the memory read path — the recalled value does
    not outrank the live source.
    """
    emitted = {
        "type": item["memory_type"],
        "scope": item["scope"],
        "content": str(item["content"])[:700],
        "source": item["source"],
        "confidence": item["confidence"],
        "live_source": item.get("live_source"),
    }
    verdict = resolve(item, live_context)
    if verdict["disposition"] == SETTLED:
        return emitted
    emitted["precedence"] = verdict["disposition"]
    emitted["reconcile_flag"] = item.get("reconcile_flag")
    if verdict["disposition"] == DERIVED:
        emitted["live_value"] = verdict["live_value"]
        emitted["note"] = (
            "Live value now: {}. The stored text is history, not the current "
            "reading ({}).".format(verdict["live_value"], verdict["reason"])
        )
    else:
        emitted["verify_live"] = True
        emitted["note"] = (
            "Verify live before relying on this — it has a live source and must "
            "not be served as settled ({}).".format(verdict["reason"])
        )
    return emitted


def servable_as_settled(memory: Dict[str, Any]) -> bool:
    """Whether a no-model path may serve this memory verbatim as a settled fact.

    The deterministic ``grounded_memory_answer`` cannot derive a live value, so
    its only honest move for a live-source memory is to decline. Mirrors
    ``detect_product`` refusing to report the ``.variant`` file's guess while a
    HAT EEPROM is present (#211): the recollection does not win over a live
    source, even when the live read is not in hand.
    """
    return not has_live_source(memory)
