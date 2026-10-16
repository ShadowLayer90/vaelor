"""The draft models a cluster deployment may speculate with, pinned, and the one lookahead rule.

VD-213 (owner decision 2026-10-12): the catalog's Qwen3-30B-A3B serves with
RedHat's DFlash drafter and a per-load lookahead, measured on the Z2 on vLLM
0.27. This module is the one home of two things every reader shares:

* :data:`DRAFTERS`, the pinned table. A drafter is named by its KEY and never
  by a repo a client typed: the repo, the commit it is pulled at, its weight
  size, its licence, the target repos it drafts for and the images it was run
  on are the table's. Adding a drafter, or re-pinning one, is a reviewed code
  change, exactly as an image is (`vllm_images`).
* :func:`require_schedule`, the rule for the lookahead table
  (``num_speculative_tokens_per_batch_size``): the deploy payload's check, the
  renderer's check on both sides of the root bridge, and the fit preview's
  form all ask it, so a schedule cannot pass one and be refused by another
  (LESSONS 6).

**The schedule's shape is Vaelor's rule, not vLLM's.** vLLM 0.27 reads each
entry as ``(range_start, range_end, num_speculative_tokens)`` over an inclusive
range of running requests and validates none of it (``vllm/config/
speculative.py`` at tag v0.27.0, LESSONS 15). So Vaelor refuses everything it
has not measured the meaning of: ranges must start at 1, run contiguously and
end at the deployment's request cap, so every load the server can reach has
exactly one lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

#: The most lookahead tokens one step may draft: 15, the largest lookahead
#: VD-213 ran (fixed 15 measured below the no-drafter baseline at every load,
#: so the bound is a ceiling on what is offered, not a recommendation).
MAX_DRAFT_TOKENS = 15

#: The most entries a schedule may carry: one per request at the highest cap.
MAX_SCHEDULE_ENTRIES = 64

Schedule = Tuple[Tuple[int, int, int], ...]


@dataclass(frozen=True)
class Drafter:
    """One pinned draft model and what it may be used with."""

    key: str
    #: What the deploy form calls it.
    label: str
    #: The ``vllm serve --speculative-config.method`` it runs as.
    method: str
    repo: str
    #: The commit the drafter is pulled and served at, so every machine runs
    #: the bytes that were measured.
    revision: str
    #: The size of its weight file (``model.safetensors``), read off the Hub
    #: tree at :attr:`revision`; the fit keeps this much on every replica.
    weight_bytes: int
    licence: str
    #: The target repos it drafts for. The first is the one measured.
    targets: Tuple[str, ...]
    #: The pinned images (`vllm_images` keys) whose vLLM runs it.
    images: Tuple[str, ...]
    max_tokens: int = MAX_DRAFT_TOKENS


DFLASH_QWEN3_30B = Drafter(
    key="dflash-qwen3-30b-a3b-2507",
    label="DFlash (RedHat) for Qwen3-30B-A3B",
    method="dflash",
    repo="RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.dflash",
    # VD-213: pinned at this commit, Apache-2.0; ``model.safetensors`` is
    # 1,362,042,120 bytes (the whole repo 1,363,257,696).
    revision="27694eafe9100abfba4916ca7f86947d14b012c6",
    weight_bytes=1_362_042_120,
    licence="Apache-2.0",
    # It drafts for Qwen3-30B-A3B-Instruct-2507 and its quantizations; the
    # catalog's 4-bit build is the one VD-213 measured.
    targets=(
        "cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
    ),
    images=("vllm-0.27-rocm10",),
)

#: Every drafter, by the key a deployment records.
DRAFTERS: Dict[str, Drafter] = {DFLASH_QWEN3_30B.key: DFLASH_QWEN3_30B}


def drafter_profile(key: Any) -> Drafter:
    """The pinned drafter a key names, or a plain refusal."""
    if not isinstance(key, str) or key not in DRAFTERS:
        raise ValueError(
            "That is not a draft model this Vaelor version serves with. "
            "Choose one the deploy form lists, or turn the draft model off."
        )
    return DRAFTERS[key]


def drafters_for(repo: Any, image_key: Optional[str] = None) -> Tuple[Drafter, ...]:
    """The drafters that draft for ``repo``, and run on ``image_key`` when one is named."""
    name = str(repo or "").strip()
    return tuple(
        drafter for drafter in DRAFTERS.values()
        if name in drafter.targets and (image_key is None or image_key in drafter.images)
    )


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_schedule(value: Any, cap: Any, *, max_tokens: int = MAX_DRAFT_TOKENS) -> Schedule:
    """The lookahead table as a tuple, or a plain refusal of what is wrong with it.

    ``value`` is a list of ``[first, last, tokens]`` entries over the number of
    requests running at once; ``cap`` is the deployment's request cap. The
    ranges are inclusive, start at 1, follow each other with no gap or overlap
    and end at ``cap``; each lookahead is 1 to ``max_tokens``.
    """
    if not _whole(cap) or cap < 1:
        raise ValueError(
            "A draft model needs a request cap per machine, so its lookahead "
            "table can cover every load. Set Max requests per machine."
        )
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("The draft model's lookahead table must list at least one range.")
    if len(value) > MAX_SCHEDULE_ENTRIES:
        raise ValueError(
            "The draft model's lookahead table may list at most {} ranges, one "
            "per request at the highest cap.".format(MAX_SCHEDULE_ENTRIES)
        )
    entries = []
    expected = 1
    for entry in value:
        if (
            not isinstance(entry, (list, tuple)) or len(entry) != 3
            or not all(_whole(part) for part in entry)
        ):
            raise ValueError(
                "Each lookahead entry is three whole numbers: the first and "
                "last request count it covers, and the tokens drafted."
            )
        first, last, tokens = entry
        if first != expected or last < first:
            raise ValueError(
                "The lookahead ranges must start at 1 request and follow each "
                "other with no gap or overlap."
            )
        if not 1 <= tokens <= max_tokens:
            raise ValueError(
                "A lookahead must draft 1 to {} tokens.".format(max_tokens)
            )
        entries.append((first, last, tokens))
        expected = last + 1
    if expected - 1 != cap:
        raise ValueError(
            "The lookahead ranges must end at the request cap ({} requests), "
            "so every load the machine can reach has one.".format(cap)
        )
    return tuple(entries)


def fit_schedule(schedule: Schedule, cap: int) -> Schedule:
    """A default schedule trimmed or stretched to ``cap``: the ranges below it, the last ending at it.

    Only for a catalog default the owner left untouched while changing the
    cap: the measured lookahead per load is kept, and the table covers
    exactly 1..``cap`` (:func:`require_schedule`). An owner's own table is
    never reshaped; it is checked as sent.
    """
    kept = [list(entry) for entry in schedule if entry[0] <= cap]
    kept[-1][1] = cap
    return tuple((first, last, tokens) for first, last, tokens in kept)


def max_lookahead(schedule: Schedule) -> int:
    """``num_speculative_tokens``: vLLM's most draft tokens per step, the table's largest."""
    return max(tokens for _first, _last, tokens in schedule)


def varies(schedule: Schedule) -> bool:
    """Whether the table drafts a different number of tokens at different loads.

    VD-213: on vLLM 0.27's older model runner such a table forces piecewise
    GPU graphs (one user fell to 57 tokens a second); the newer runner
    keeps full graphs, so a varying table requires it.
    """
    return len({tokens for _first, _last, tokens in schedule}) > 1


def schedule_word(schedule: Schedule) -> str:
    """The table as vLLM's dotted flag value: ``[[1,1,4],[2,8,7]]``, no spaces, no quotes.

    VD-213: vLLM 0.27's own argument parser was checked in the pinned image to
    read this spelling into the same list the JSON form gives, so no brace or
    quote reaches a unit's ``ExecStart``.
    """
    return "[" + ",".join(
        "[{},{},{}]".format(first, last, tokens) for first, last, tokens in schedule
    ) + "]"
