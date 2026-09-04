"""What each recommended model is bad at, in the owner's words.

VD-071: *"recommend the 4B model and notate it's shortcomings"*. This is where
the notation lives, keyed by model, **derived from the measured record rather
than retyped beside it**.

The first version hard-coded the 4B's figures into a React component. Two
things were wrong with that. It duplicated numbers `model_footprint.py` already
owns - the exact habit that put a superseded model's timings into three ledger
rows - and it rendered those figures for whichever model was installed,
including the 1.7B that VD-071 deliberately keeps as an alternative, where
every one of them is wrong: ~10 s per answer rather than ~20, 3,157 MB rather
than 5,542, a 6 s start rather than 37.

**A model with no measured entry gets no shortcomings list**, not a generic
one. Saying nothing is the honest output when nobody has measured it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .model_footprint import MIB, footprint

#: Behavioural weaknesses that are not derivable from a footprint row. Each is
#: a measured claim, with the measurement that established it.
_BEHAVIOUR: Dict[str, List[str]] = {
    "qwen3-4b-instruct-2507-q4-0": [
        # VD-076. The model the Pi actually ships had no entry here for one
        # release, so `shortcomings()` returned None for it and the owner
        # running the recommended model saw an empty disclosure - while the
        # owner running the *superseded* alternative got the full list. The
        # module's rule ("a model with no measured entry gets no shortcomings
        # list") did not cover this: the model is measured, it was the
        # behaviour key that was missing.
        "It was chosen for how quickly it answers rather than for the highest "
        "score. On the same 200-question harness it is about 2.6 points less "
        "accurate than the best 4B measured on this appliance.",
        # **Removed for the second time, 2026-08-11, and the reason is worth
        # keeping.** Alpha 41 removed a fast-wake sentence because
        # `--slot-save-path` was rendered and nothing called save or restore.
        # Alpha 46 wired both calls (#152) and restored the sentence on the
        # strength of unit tests. The first live test on the appliance then
        # found the mechanism has never functioned there: llama-server logged
        # a restore for `assistant-recent-0.kvslot` against a file that does
        # not exist, `/var/lib/vaelor/kv-cache` is empty, and the mapping
        # table holds no rows (#158). The guards failed safe - no wrong prefix
        # was ever served - but an owner reading this list was being told a
        # protection was running when it was not.
        #
        # Twice now the same shape: the code existed, the tests passed, and
        # nobody had watched it work on the hardware. **The sentence returns
        # only after the mechanism is observed working on the appliance**, not
        # after it is implemented, and not after it is green. #158 carries
        # that, and #156 - the answer timeout that cancels every model reply -
        # very likely has to land first, since a cancelled turn may leave no
        # state worth saving.
    ],
    "qwen3-4b-q4": [
        # VD-070: reversibility 90.5% -> 81.0% against the model it replaced,
        # while diagnosis went 48.2% -> 92.9%. On an appliance this is the
        # regression that matters: it fixes things the heavy way.
        "It sometimes fixes things the heavy way - restarting a service where a "
        "smaller change would do, such as restarting a container instead of "
        "correcting a health-check port. Read what it proposes before "
        "approving it.",
    ],
    "qwen3-1.7b-q4": [
        # 48.2% on diagnosis against the 4B's 92.9%, same harness, same
        # hardware. Faster and lighter, and wrong more than half the time.
        "It diagnoses poorly - it was right on about half the appliance "
        "problems the 4B got right. It is here because it is small and quick, "
        "not because it is better.",
    ],
}


def model_facts(
    model_name: str, platform: str, context_tokens: int
) -> Dict[str, Any]:
    """What the screen needs to know about the deployed model.

    Resolved from the connection's own model name, so there is no record
    shared between processes and nothing to keep in step. VD-069 delivered
    this through a file the executor wrote and the control plane read; that
    file existed for the lifecycle, and the lifecycle is now llama-server's
    own `--sleep-idle-seconds` (VD-073).

    Returns an empty `shortcomings` list and no periods when the model cannot
    be identified or has not been measured at this window - saying nothing is
    the honest output, and it is what `shortcomings` already does.
    """
    from .model_footprint import footprint, identify_by_file
    from .model_service_compose import SLEEP_IDLE_SECONDS

    model_id = identify_by_file(str(model_name or "").split("/")[-1]) or ""
    detail = shortcomings(model_id, platform, context_tokens) if model_id else None
    record = footprint(model_id, platform, context_tokens, 1) if model_id else None
    return {
        "model": model_id,
        "shortcomings": (detail or {}).get("items") or [],
        # Stated only when both are known. A period without a wake time is a
        # promise with the cost left out.
        "sleep_idle_seconds": SLEEP_IDLE_SECONDS if record else None,
        "cold_start_seconds": (record or {}).get("cold_start_seconds"),
    }


def shortcomings(
    model_id: str, platform: str, context_tokens: int, slots: int = 1
) -> Optional[Dict[str, Any]]:
    """What to tell the owner about this model, or None if nobody measured it.

    Returns `{"model": str, "items": [str]}`. The cost lines are generated from
    the footprint record so they cannot drift from it; the behavioural lines
    come from the table above.
    """
    record = footprint(model_id, platform, context_tokens, slots)
    behaviour = _BEHAVIOUR.get(model_id)
    if record is None or behaviour is None:
        return None
    items = list(behaviour)
    seconds = record.get("seconds_per_answer")
    if seconds:
        items.append(
            "It takes about {:.0f} seconds to answer.".format(seconds)
        )
    resident = record.get("steady_state_bytes")
    if resident:
        # GiB, because the divisor is binary. `tests/test_byte_units.py`
        # caught the first version pairing 1024^3 with "GB" - a 7.4%
        # overstatement of the kind VD-047 records, in owner-facing copy.
        #
        # #146: the sentence names its basis now. A tester watched Home rise
        # ~2.7 GB while this line said 3.5 GiB, a gigabyte apart with nothing
        # saying why. Both are correct - this is the container's own memory
        # charge, the figure its limit is enforced against; the machine-level
        # rise on Home is smaller because shared and reclaimable pages are
        # not all new. Naming the basis is what stops the two readings being
        # read as a contradiction.
        items.append(
            "It holds about {:.1f} GiB of its own memory allowance while "
            "running, which is why it does not stay running. The machine-wide "
            "rise on Home reads lower, because part of that allowance is "
            "memory the system already shares.".format(resident / (1024 * MIB))
        )
    cold = record.get("cold_start_seconds")
    if cold:
        items.append(
            "Starting it takes about {:.0f} seconds.".format(cold)
        )
    return {"model": model_id, "items": items}
