"""Which model each tier runs, decided from the measured capability table.

**Every row here names a model that is installed on the machine it was
measured on.** Three rows previously did not: ``gemma3-3b-FLM``,
``gemma3-8b-FLM`` and ``gemma3-1b-FLM`` carried real numbers filed under a
``gemma3`` family that was never benchmarked because it is not present. The
measurements were of ``llama3.2:3b``, ``llama3.1:8b`` and ``llama3.2:1b``, and
they are back under those names. A measurement attributed to a model that does
not exist is a defect in the one property this module exists to guarantee.

**These rates are not probabilities.** FLM at temperature 0 is neither random
nor stateless: identical requests differ *within* a server run, but the whole
sequence repeats exactly across restarts — 12/12 position-identical replies
between the 8,192 and 131,072 runs, on all four models measured at both. So
``schema_valid_rate`` is "the fraction of the first twelve requests to a fresh
server that validated", it is reproducible rather than probable, and a binomial
confidence interval does not apply to it. Only request indices 0–11 have ever
been measured; behaviour at index 500 in an always-on process is not
established by anything here.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from .inference_context import AGENT_PROMPT_TOKENS


MIB = 1024 ** 2

#: The model tags ``flm-real`` itself accepts, read from the live inventory of
#: the measured machine on 2026-08-07. This is the set a launch has to hit: FLM
#: knows nothing about Vaelor's or Lemonade's spelling of a model name, and a
#: name it does not recognise comes back as *"model not found"* on a model that
#: is installed.
#:
#: ``qwen3.5:4b`` was installed and converted (q4nx) on the Z2 for the VD-108
#: model-selection run and served live on 2026-08-18; the other nine are the
#: 2026-08-07 inventory. The addition is dated separately on purpose — these
#: names were read off the machine at two different times, and lumping the tenth
#: in with the nine would misdate when it was confirmed present.
INSTALLED_FLM_MODELS = frozenset({
    "deepseek-r1:8b",
    "gemma3:4b",
    "gemma4-it:e4b",
    "gpt-oss:20b",
    "llama3.1:8b",
    "llama3.2:1b",
    "llama3.2:3b",
    "medgemma:4b",
    "nanbeige4.1:3b",
    # Installed/converted on the Z2 for VD-108, served live 2026-08-18.
    "qwen3.5:4b",
})

#: Provenance for the ``qwen3.5-4b-FLM`` row, kept separate from
#: :data:`CAPABILITY_PROBE` because it is a *different measurement* and this
#: module exists to stop two bases being read as one. The nine rows below carry
#: numbers from the 2026-08-07 n=12 direct-``flm-real`` schema probe
#: (``CAPABILITY_PROBE``); qwen3.5's numbers come from the VD-108 model-selection
#: run — the v11 364-item reasoning eval, measured on the **Z2 XDNA2 NPU via
#: FastFlowLM** on the real q4nx artifact on 2026-08-18, thinking-off. They are
#: different machines-in-time, different sample sets (12 requests vs 364 items),
#: and different metrics (JSON-schema validity of one intent vs reasoning
#: accuracy over a benchmark). Do not average, compare, or fold the two.
#:
#: The one axis they both bear on is structured output: qwen3.5 emitted a
#: parseable answer on **364/364** items (0 unparseable) on this run, which is
#: the ``schema_valid_rate`` analog measured on the newer, larger, on-NPU basis —
#: not a twelfth-request figure from ``CAPABILITY_PROBE`` and not to be quoted as
#: one. ``medgemma`` and ``phi4-mini`` were dropped in the same run precisely for
#: failing this (201/320 and similar unparseable), so the metric discriminates.
QWEN35_NPU_MEASUREMENT: Dict[str, Any] = {
    "eval": "v11 (364 items)",
    "measured_on": "2026-08-18, Z2 XDNA2 NPU via FastFlowLM (real q4nx artifact)",
    "reasoning_accuracy": "349/364 (95.88%), v11 strict, thinking-off",
    "structured_output": "364/364 parseable, 0 unparseable",
    "confabulation": 0,
    "overreach": 0,
    "reasoning_chars": 0,
    "distinct_from": (
        "CAPABILITY_PROBE, the 2026-08-07 n=12 direct-flm-real schema probe the "
        "other rows use. Different machine-in-time, sample set and metric; the "
        "two bases must not be blended."
    ),
    "decision": "VD-108",
}

#: Whether each NPU model can produce schema-valid structured output at the real
#: ~4.7 k agent prompt, measured. This is a *capability* table, not a speed
#: table, and it is the primary selection criterion for the assistant tier: a
#: faster model that cannot answer in the required shape cannot do the job at
#: all. ``schema_valid_rate`` is the fraction of runs whose JSON validated.
#:
#: Every row carries three things it previously did not:
#:
#: * ``flm_tag`` — the name ``flm-real`` answers to. The key is Vaelor's
#:   identifier and they are different strings; see :func:`flm_model_tag`.
#: * ``resident_bytes`` — measured RSS at ``--ctx-len 8192``. On an always-on
#:   tier sharing one memory pool with the GPU tier this is the axis that
#:   matters most, and it was absent, so the selector could not use it.
#: * a corrected rate. The whole table is the n=12 direct-``flm-real`` run of
#:   2026-08-07. The rates it replaces came from n=3 probes that additionally
#:   capped completions below what the models needed, which turned truncation
#:   into apparent incapability on two models that in fact pass.
#:
#: No empty response was observed on any model in that run. ``gpt-oss-20b-FLM``
#: previously recorded 0.8, and that figure was taken *through Lemonade*; on the
#: direct path it is 0.0. Its ``hardware_errors`` flag stays set for the
#: intermittent ``qds_device::wait()`` stall, which is a separate observation
#: and did reproduce.
NPU_MODEL_CAPABILITY: Dict[str, Dict[str, Any]] = {
    "gemma3-4b-FLM": {
        # VD-068's pick, superseded by VD-108. On the v11 364-item eval it
        # scored 75.00% with the worst honesty profile of the field (13
        # confabulations, 9 overreaches), which is why it lost the reopened
        # selection; its schema_valid_rate here is still the 2026-08-07 probe's.
        "flm_tag": "gemma3:4b",
        "schema_valid_rate": 1.0,
        "empty_response_rate": 0.0,
        "native_tool_calls": False,
        "hardware_errors": False,
        "parameters": "4B",
        "resident_bytes": 5567 * MIB,
    },
    "qwen3.5-4b-FLM": {
        # VD-108: the Z2 NPU Assistant model, chosen by the owner on reasoning
        # accuracy. Every number in this row is from QWEN35_NPU_MEASUREMENT — the
        # v11 364-item eval measured on the Z2 NPU (q4nx via FLM) on 2026-08-18,
        # NOT the 2026-08-07 n=12 probe the rows around it use. `schema_valid_rate`
        # here is 364/364 answers parseable (0 unparseable) on that run: the
        # structured-output analog on a larger, newer, on-NPU basis, not a
        # CAPABILITY_PROBE twelfth-request figure. See QWEN35_NPU_MEASUREMENT for
        # why the two bases are kept apart. `resident_bytes` is FLM VmRSS from the
        # Z2 speed run (~6,383 MiB). The reasoning-accuracy fields carry the
        # measurement that justifies VD-108's owner pin; they are this model's
        # own NPU numbers and are deliberately not filed on any other row (the
        # runners-up were scored on the 5080 proxy, a different basis).
        "flm_tag": "qwen3.5:4b",
        "schema_valid_rate": 1.0,
        "reasoning_accuracy": 349 / 364,
        "reasoning_accuracy_correct": 349,
        "reasoning_accuracy_total": 364,
        "empty_response_rate": 0.0,
        "native_tool_calls": False,
        "hardware_errors": False,
        "parameters": "4B",
        "resident_bytes": 6383 * MIB,
        "measurement": QWEN35_NPU_MEASUREMENT,
    },
    "medgemma-4b-FLM": {
        # Absent from the table entirely until now, despite holding 12/12 at the
        # agent prompt in this run and 100% in the earlier one. It matches the
        # incumbent on every measured axis and needs fewer completion tokens per
        # answer (294–399 against 437–536). It is a medical-domain fine-tune and
        # its general appliance-reasoning quality has never been evaluated —
        # only its JSON structure — which is the one piece of work that could
        # legitimately change the assistant model.
        "flm_tag": "medgemma:4b",
        "schema_valid_rate": 1.0,
        "empty_response_rate": 0.0,
        "native_tool_calls": False,
        "hardware_errors": False,
        "parameters": "4B",
        "resident_bytes": 5567 * MIB,
    },
    "gemma4-it-e4b": {
        # The only FLM model that returned tool calls at all. Its recorded 0%
        # was an artifact of the harness, not a property of the model: it emits
        # exactly 560 tokens every time and the probe capped it at 500. At the
        # 1536 its caller allows, it is 12/12 with one distinct reply.
        "flm_tag": "gemma4-it:e4b",
        "schema_valid_rate": 1.0,
        "empty_response_rate": 0.0,
        "native_tool_calls": True,
        "hardware_errors": False,
        "parameters": "4B",
        "resident_bytes": 8662 * MIB,
    },
    "gpt-oss-20b-FLM": {
        # Also 12/12 when driven directly. The 0% and the 80% empty rate this
        # row used to carry were both measured through Lemonade under a
        # 500-token cap against the 688 tokens this model emits every time.
        "flm_tag": "gpt-oss:20b",
        "schema_valid_rate": 1.0,
        "empty_response_rate": 0.0,
        "native_tool_calls": False,
        # The intermittent `qds_device::wait() unexpected command state` stall
        # (VD-001). Reproduced; not superseded by this run.
        "hardware_errors": True,
        "parameters": "20B",
        "resident_bytes": 16582 * MIB,
    },
    "llama3.1-8b-FLM": {
        # Filed as `gemma3-8b-FLM` at 0.333 until 2026-08-07. The 0.333 was this
        # model's, at n=3 and a 500-token cap; the name was not.
        "flm_tag": "llama3.1:8b",
        "schema_valid_rate": 0.917,
        "empty_response_rate": 0.0,
        "native_tool_calls": False,
        "hardware_errors": False,
        "parameters": "8B",
        "resident_bytes": 7149 * MIB,
    },
    "llama3.2-3b-FLM": {
        # Filed as `gemma3-3b-FLM` at 0.667 until 2026-08-07.
        "flm_tag": "llama3.2:3b",
        "schema_valid_rate": 0.917,
        "empty_response_rate": 0.0,
        "native_tool_calls": False,
        "hardware_errors": False,
        "parameters": "3B",
        "resident_bytes": 3997 * MIB,
    },
    "deepseek-r1-8b": {
        "flm_tag": "deepseek-r1:8b",
        "schema_valid_rate": 0.667,
        "empty_response_rate": 0.0,
        "native_tool_calls": False,
        "hardware_errors": False,
        "parameters": "8B",
        "resident_bytes": 7202 * MIB,
    },
    "nanbeige4.1-3b-FLM": {
        "flm_tag": "nanbeige4.1:3b",
        "schema_valid_rate": 0.083,
        "empty_response_rate": 0.0,
        "native_tool_calls": False,
        "hardware_errors": False,
        "parameters": "3B",
        "resident_bytes": 3944 * MIB,
    },
    "llama3.2-1b-FLM": {
        # Filed as `gemma3-1b-FLM` at 0.0 until 2026-08-07. The 0.0 reproduced
        # at n=12: this model does not answer in the required shape at the agent
        # prompt, whatever budget it is given.
        "flm_tag": "llama3.2:1b",
        "schema_valid_rate": 0.0,
        "empty_response_rate": 0.0,
        "native_tool_calls": False,
        "hardware_errors": False,
        "parameters": "1B",
        "resident_bytes": 1841 * MIB,
    },
}

#: The prompt the capability table was measured against, and how to read it.
#:
#: ``description`` used to say "the appliance's own agent system prompt". It is
#: not one: it is a synthetic, deterministic 13,038-character stand-in the
#: benchmark harness generates, matched by length to what
#: :data:`~vaelor.inference_context.AGENT_PROMPT_TOKENS` read at the time,
#: 4,668. Under the Gemma tokenizers it measures 5,472–5,527 tokens rather than
#: 4,668, so the Gemma rows answered a ~17% longer prompt than that stated.
#:
#: **The real standing prompt is 1,344 tokens** (VD-075, measured with the
#: deployed model's own /tokenize). So this probe is roughly 3.5x the prompt
#: any request actually carries, and every rate below was taken against the
#: larger one. That makes the rows conservative rather than wrong - a model
#: that held its schema over 4,668 tokens will hold it over 1,344 - but they
#: are not a measurement of the shipped workload and should not be quoted as
#: one.
#:
#: ``samples`` and ``max_tokens`` are here because without them a rate is not
#: interpretable: the same ``1.0`` was previously carried from n=3 at a
#: 500-token cap, and at that cap this model measures 0.75. The budget, not the
#: model, decides the rate below roughly 600 completion tokens.
CAPABILITY_PROBE = {
    # **The literal, deliberately not `AGENT_PROMPT_TOKENS`.** This records what
    # a benchmark that has already been run actually sent, and it sent a
    # stand-in sized to the then-current 4,668. Tracking the constant would
    # have relabelled those completed runs as 1,344-token probes the moment the
    # constant was measured (VD-075) - rewriting history to match a number
    # nobody re-ran. If the probe is regenerated at the real prompt size, change
    # this and the description together, and re-run the rows below.
    "prompt_tokens": 4668,
    "description": "the 4,668-token agent-prompt stand-in the benchmarks generate",
    "samples": 12,
    "max_tokens": 1536,
    "context_tokens": 8192,
    "request_indices": "0-11",
    "measured_on": "2026-08-07, standalone flm-real, Lemonade out of the path",
    "reproducibility": (
        "FLM at temperature 0 is deterministic across restarts but not across "
        "request index: the same twelve requests to a fresh server return the "
        "same twelve answers, and identical requests within one run differ. "
        "These rates are reproducible, not probable, so binomial confidence "
        "intervals do not apply and only request indices 0-11 are measured."
    ),
}

#: Minimum schema-valid rate a model must reach to be trusted with structured
#: agent work. Anything below this returns unusable output some of the time,
#: which on an agent path is a failed run, not a degraded one.
#:
#: **It no longer discriminates.** Four of the nine measured models reach 1.0,
#: so meeting the gate is necessary and not sufficient, and something else has
#: to choose. That something is :data:`SELECTION_CRITERIA`, stated rather than
#: left to whatever order the rows happen to sort in.
MINIMUM_SCHEMA_VALID_RATE = 1.0

#: How the assistant model is chosen, in order. Each rung exists because the one
#: above it stopped separating candidates.
#:
#: ``resident-footprint`` is second on purpose. While the assistant tier is
#: resident it shares one memory pool with the GPU tier, so among models that can
#: all do the job the smallest one leaves the most room for the other tier — 5.6 GB
#: against 8.7 GB and 16.6 GB across the four that reach the gate. It was not in
#: the table at all, which is why the tie used to fall straight through to the
#: model name.
#:
#: ``name`` is still last, and it is still not a merit criterion. It is here so
#: that two rows identical on every measured axis produce a stable answer rather
#: than an arbitrary one — and :func:`recommended_npu_model` reports it as the
#: rung that decided, so a selection resting on it is visible instead of
#: incidental. Today it does decide: ``gemma3-4b-FLM`` and ``medgemma-4b-FLM``
#: are identical on capability, footprint, empty responses and hardware errors.
SELECTION_CRITERIA: Tuple[str, ...] = (
    "capability",
    "resident-footprint",
    "empty-responses",
    "hardware-errors",
    "name",
)

#: The owner's pin for the assistant tier, above the mechanical rungs.
#:
#: VD-108 reopened the choice and picked ``qwen3.5-4b-FLM`` on **reasoning
#: accuracy** — an axis none of :data:`SELECTION_CRITERIA` encodes. The rungs
#: rank on structured-output capability, resident footprint, empty responses,
#: hardware errors and name; the owner weighed 95.88% v11 accuracy (0
#: confabulation, 0 overreach) against a ~3 s latency cost and chose accuracy.
#:
#: This is recorded as an explicit, named pin rather than reverse-engineered into
#: a mechanical accuracy criterion, for two reasons this module cares about:
#:
#: * It is an **owner decision**, a value judgment (accuracy over latency), not a
#:   derived optimum. Encoding it as "highest reasoning_accuracy wins" would make
#:   any future model that measures 96% auto-win, which is not what was decided.
#: * The accuracy numbers behind it sit on **different bases** — qwen3.5 on the Z2
#:   NPU, the runners-up on the 5080 GGUF proxy — so a ranking axis over them
#:   would blend provenances, the exact failure this module exists to prevent.
#:
#: The pin is not a bypass of capability. :func:`recommended_npu_model` applies it
#: only when the pinned model is present in the table *and* clears the
#: schema-valid gate; a pin can never deploy a model that cannot answer in the
#: required shape, and on a candidate table that does not contain it (a synthetic
#: test table, or a re-measurement that drops it below the gate) the mechanical
#: rungs decide as before. The deciding rung is reported as ``owner-pin`` so a
#: selection resting on it is visible, exactly as a name-broke-the-tie one is.
ASSISTANT_MODEL_PIN: Dict[str, str] = {
    "model": "qwen3.5-4b-FLM",
    "decision": "VD-108",
    "axis": "reasoning accuracy",
    "basis": (
        "349/364 (95.88%) on the v11 eval, measured on the Z2 NPU (q4nx via FLM) "
        "2026-08-18, thinking-off, 0 confabulation, 0 overreach — against the "
        "superseded gemma3-4b-FLM's 75.00%"
    ),
}

#: The GPU tier's measured configuration.
CHAT_TIER_MODEL = "gpt-oss-20b-GGUF"

#: FLM exposes no usable native tool calling, so Vaelor keeps orchestrating
#: through structured JSON intent that it executes itself. This is a fact about
#: the tier, recorded so no code path starts assuming otherwise. Re-confirmed by
#: the 2026-08-07 run, in which ``response_format`` was inert on all three of
#: ``none``, ``json_object`` and ``json_schema``.
NPU_NATIVE_TOOL_CALLING = False


def flm_model_tag(
    model: str,
    candidates: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Translate a Vaelor model identifier into the tag ``flm-real`` accepts.

    These are different strings for the same model — ``gemma3-4b-FLM`` is what
    the selector returns and what a Lemonade listing calls it; ``gemma3:4b`` is
    what FLM answers to. Under VD-001 and VD-002 Vaelor launches ``flm-real``
    itself, so ``npu_tier_plan()["model"]`` cannot be handed to it as-is, and
    the failure mode when it is has no diagnostic value at all: *"model not
    found"* on a model that is installed.

    The mapping is a lookup, not a rewrite rule. Stripping ``-FLM`` and
    replacing the last hyphen with a colon happens to produce the right answer
    for all nine installed models, and that is exactly what makes it dangerous:
    it would keep producing confident answers for the tenth. Every tag here is a
    name read off the machine.

    A name that is already an FLM tag passes through, so an operator who pins
    ``gemma3:4b`` directly gets what they asked for. Anything else returns
    ``known`` false and no tag: guessing is what this function exists to stop.
    """
    table = dict(candidates or NPU_MODEL_CAPABILITY)
    name = str(model or "").strip()
    facts = table.get(name) or {}
    tag = str(facts.get("flm_tag") or "")
    if tag:
        source = "capability table"
    elif ":" in name:
        tag, source = name, "already an flm-real tag"
    else:
        source = ""
    installed = tag in INSTALLED_FLM_MODELS
    if not tag:
        return {
            "model": name,
            "tag": "",
            "known": False,
            "installed": False,
            "source": "",
            "reason": (
                "{} has no known flm-real tag. Vaelor's identifiers and FLM's "
                "tags are different strings, and there is no rule to derive one "
                "from the other, so nothing is guessed here: a wrong tag comes "
                "back as \"model not found\" on a model that is installed."
            ).format(name or "This model"),
        }
    return {
        "model": name,
        "tag": tag,
        "known": True,
        "installed": installed,
        "source": source,
        "reason": "" if installed else (
            "{} translates to the flm-real tag {}, which was not in the model "
            "inventory read from the measured machine. It may still be "
            "installed here; nothing has checked this one."
        ).format(name, tag),
    }


def npu_model_capability(
    model: str,
    candidates: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Whether one named model is measured fit for structured agent work.

    The threshold is applied here rather than at each call site because it was
    already applied once, in :func:`recommended_npu_model`, and a guard
    implemented twice is how the KV-cache defect happened: two independent
    copies of one rule, with the same half wrong in both.

    ``usable`` answers "does any measurement rule this model out", which is a
    different question from "has this model been measured". A model absent from
    the table is ``usable`` with ``measured`` false: there is no evidence for
    it and none against it, and refusing to emit a plan for a model an operator
    pinned would restore the 131,072-token preallocation that the context flag
    exists to prevent — a worse outcome than deploying on the operator's word.
    A model *in* the table below the threshold is refused outright, because
    there the evidence exists and it is negative.
    """
    table = dict(candidates or NPU_MODEL_CAPABILITY)
    name = str(model or "")
    facts = table.get(name)
    if facts is None:
        return {
            "model": name,
            "usable": True,
            "measured": False,
            "schema_valid_rate": None,
            "resident_bytes": None,
            "reason": (
                "{} has not been measured against {}, so nothing here supports "
                "it and nothing rules it out. It is deployed on the operator's "
                "word rather than on evidence."
            ).format(name or "This model", CAPABILITY_PROBE["description"]),
        }
    rate = float(facts.get("schema_valid_rate", 0.0))
    usable = rate >= MINIMUM_SCHEMA_VALID_RATE
    return {
        "model": name,
        "usable": usable,
        "measured": True,
        "schema_valid_rate": rate,
        "resident_bytes": _resident_bytes(facts),
        "reason": "" if usable else (
            "{} returned schema-valid output on {:.0%} of {} runs at {}, short "
            "of the {:.0%} structured agent work needs. Output that is unusable "
            "some of the time is a failed agent run, not a degraded one."
        ).format(
            name, rate, CAPABILITY_PROBE["samples"],
            CAPABILITY_PROBE["description"], MINIMUM_SCHEMA_VALID_RATE,
        ),
    }


def _resident_bytes(facts: Mapping[str, Any]) -> Optional[int]:
    """Measured RSS for one model, or ``None`` when nothing measured it.

    ``None`` is not zero. A model whose footprint was never recorded must not
    win a smallest-footprint tie-break by default, which is what returning 0
    would do — the same distinction ``verify_accelerator_in_use`` draws between
    "holding nothing" and "could not be read".
    """
    value = facts.get("resident_bytes")
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _ranking_key(name: str, facts: Mapping[str, Any]) -> Tuple[Any, ...]:
    """One entry's position under :data:`SELECTION_CRITERIA`, in that order."""
    footprint = _resident_bytes(facts)
    return (
        -float(facts.get("schema_valid_rate", 0.0)),
        float("inf") if footprint is None else float(footprint),
        float(facts.get("empty_response_rate", 0.0)),
        bool(facts.get("hardware_errors")),
        str(name),
    )


def deciding_criterion(best: Tuple[Any, ...], runner_up: Tuple[Any, ...]) -> str:
    """Which rung actually separated the winner from the next candidate.

    Reported rather than assumed. "Chosen on capability" was printed while four
    models were tied on capability and the sort order was doing the work; a
    selection that rests on the model's name should say so in the same breath
    as the selection.
    """
    for criterion, first, second in zip(SELECTION_CRITERIA, best, runner_up):
        if first != second:
            return criterion
    return SELECTION_CRITERIA[-1]


def _pinned_model(
    pin: Optional[Mapping[str, str]], table: Mapping[str, Mapping[str, Any]]
) -> str:
    """The owner-pinned model, but only when it is present and passes the gate.

    A pin is a decision to override the mechanical rungs, not to override
    capability: a model that cannot answer in the required shape cannot be
    deployed by fiat. So the pin applies only when its model is in this
    candidate table *and* clears :data:`MINIMUM_SCHEMA_VALID_RATE`. On a
    synthetic table that does not contain it, or a re-measurement that drops it
    below the gate, this returns ``""`` and the mechanical order decides —
    which is why the tie-break tests, driven on tables without the pinned model,
    are unaffected by the pin.
    """
    if not pin:
        return ""
    name = str(pin.get("model") or "")
    facts = table.get(name)
    if facts is None:
        return ""
    if float(facts.get("schema_valid_rate", 0.0)) < MINIMUM_SCHEMA_VALID_RATE:
        return ""
    return name


#: Sentinel so ``pin`` distinguishes "caller said nothing, use the module pin"
#: from ``pin=None`` ("caller disabled the pin"). Bound late — reading
#: :data:`ASSISTANT_MODEL_PIN` inside the call rather than capturing it as a
#: default keeps the constant patchable and keeps the two meanings separable.
_DEFAULT_PIN = object()


def recommended_npu_model(
    candidates: Optional[Mapping[str, Mapping[str, Any]]] = None,
    *,
    pin: Any = _DEFAULT_PIN,
) -> Dict[str, Any]:
    """Pick the assistant-tier model: owner pin first, then the mechanical rungs.

    The mechanical order is :data:`SELECTION_CRITERIA`: capability gates, then the
    smallest resident footprint wins (the assistant tier shares one memory pool
    with the GPU tier while resident), then empty responses, hardware errors, and
    finally the model name — reported, not narrated, so a selection resting on the
    name is visible rather than incidental.

    Above all of that sits one owner pin. VD-108 chose ``qwen3.5-4b-FLM`` on
    reasoning accuracy — 95.88% on the v11 eval measured on the Z2 NPU, against the
    superseded ``gemma3-4b-FLM``'s 75.00% — an axis no rung encodes. The pin is
    :data:`ASSISTANT_MODEL_PIN` and it is applied through :func:`_pinned_model`,
    which honours it only when the pinned model is in the table and clears the
    schema-valid gate. When it applies, ``selection_criterion`` is ``owner-pin``
    and the reason names both the decision and the model the mechanical rungs
    would otherwise have picked, so the override is auditable. When it does not —
    a candidate table without the pinned model, or a re-measurement that drops it
    below the gate — the mechanical rungs decide exactly as before.
    """
    if pin is _DEFAULT_PIN:
        pin = ASSISTANT_MODEL_PIN
    table = dict(candidates or NPU_MODEL_CAPABILITY)
    ranked = sorted(table.items(), key=lambda item: _ranking_key(item[0], item[1]))
    if not ranked:
        return {
            "model": "", "usable": False, "rejected": [], "tied_at_gate": [],
            "selection_criterion": "", "tie_break_order": list(SELECTION_CRITERIA),
            "owner_pin": None,
            "reason": "No NPU model has been measured on this appliance.",
        }
    # The mechanical winner is ranked[0]; the owner pin can lift another model
    # above it, but only one that has cleared the same gate.
    mech_name = ranked[0][0]
    pinned = _pinned_model(pin, table)
    name = pinned or mech_name
    facts = table[name]
    rejected = [
        {
            "model": other,
            "schema_valid_rate": float(other_facts.get("schema_valid_rate", 0.0)),
            "empty_response_rate": float(other_facts.get("empty_response_rate", 0.0)),
            "hardware_errors": bool(other_facts.get("hardware_errors")),
            "resident_bytes": _resident_bytes(other_facts),
        }
        for other, other_facts in ranked
        if other != name
    ]
    rate = float(facts.get("schema_valid_rate", 0.0))
    usable = rate >= MINIMUM_SCHEMA_VALID_RATE
    footprint = _resident_bytes(facts)
    tied = [
        other for other, other_facts in ranked
        if float(other_facts.get("schema_valid_rate", 0.0)) >= MINIMUM_SCHEMA_VALID_RATE
    ]
    best_key = _ranking_key(name, facts)
    if pinned:
        criterion = "owner-pin"
    elif len(ranked) > 1:
        criterion = deciding_criterion(best_key, _ranking_key(*ranked[1]))
    else:
        criterion = "capability"
    # Models this one cannot be told apart from on any measured axis. Only these
    # are what the name rung is actually breaking a tie between — the wider
    # `tied_at_gate` set includes models the footprint rung already separated.
    indistinguishable = [
        other for other, other_facts in ranked
        if other != name and _ranking_key(other, other_facts)[:-1] == best_key[:-1]
    ]
    if not usable:
        reason = (
            "No measured NPU model reaches the {:.0%} schema-valid rate "
            "structured agent work needs; the best is {} at {:.0%}."
        ).format(MINIMUM_SCHEMA_VALID_RATE, name, rate)
    elif pinned:
        reason = _pin_reason(name, mech_name, tied, len(ranked), pin)
    else:
        reason = _selection_reason(
            name, rate, footprint, tied, criterion, len(ranked), indistinguishable,
        )
    return {
        "model": name,
        "usable": usable,
        "schema_valid_rate": rate,
        "resident_bytes": footprint,
        "flm_tag": flm_model_tag(name, table)["tag"],
        # The rung that actually separated this model from the next one, not
        # the rung it was hoped to win on. ``owner-pin`` when a decision, not a
        # measurement, chose it.
        "selection_criterion": criterion,
        "tie_break_order": list(SELECTION_CRITERIA),
        "tied_at_gate": tied,
        "indistinguishable": indistinguishable,
        "native_tool_calls": bool(facts.get("native_tool_calls")),
        # The pin that decided, or None when the mechanical rungs did. Carried so
        # a reader can see the decision behind an ``owner-pin`` without parsing
        # the reason string.
        "owner_pin": dict(pin) if pinned else None,
        "rejected": rejected,
        "reason": reason,
    }


def _pin_reason(
    name: str,
    mech_name: str,
    tied: list,
    total: int,
    pin: Mapping[str, str],
) -> str:
    """Say the owner overrode the mechanical rungs, and on what basis.

    Names the model the rungs would have picked, so the override is legible as an
    override rather than presented as a derived result.
    """
    gate = (
        "{} of {} measured models clear the schema-valid gate. "
    ).format(len(tied), total)
    if mech_name and mech_name != name:
        gate += (
            "The mechanical rungs (footprint, then name) would pick {}. "
        ).format(mech_name)
    return gate + (
        "{} pinned {} on {}: {}. This is a decision, not a measurement rung, and "
        "is reported as owner-pin so it is not mistaken for one."
    ).format(
        # The pin is authoritative for its own axis; the fallback is a single
        # word so it does not restate ASSISTANT_MODEL_PIN's "reasoning accuracy"
        # literal (LESSONS 6 / VD-090: one sentence, one place in a module).
        pin.get("decision", "The owner"), name,
        pin.get("axis") or "accuracy", pin.get("basis", ""),
    )


def _selection_reason(
    name: str,
    rate: float,
    footprint: Optional[int],
    tied: list,
    criterion: str,
    total: int,
    indistinguishable: list,
) -> str:
    """Say which rung chose this model, and what the rungs above it settled."""
    gate = (
        "{} of {} measured models reach the {:.0%} schema-valid gate at {}, "
        "and {} is one of them."
    ).format(len(tied), total, rate, CAPABILITY_PROBE["description"], name)
    if len(tied) <= 1:
        return "{} It is the only one that does.".format(gate)
    gate += " Capability therefore did not choose between them."
    if criterion == "resident-footprint" and footprint is not None:
        return gate + (
            " {} was chosen on the smallest resident footprint, {:.1f} GiB — the "
            "axis that matters while the assistant tier is resident, because it "
            "shares one memory pool with the GPU tier."
        ).format(name, footprint / (1024 ** 3))
    if criterion == "name":
        return gate + (
            " Nor did footprint, empty responses or hardware errors: {} is "
            "indistinguishable from {} on every one of them, so the model name "
            "broke the tie. That is a stable answer rather than a merit one, "
            "and it is reported here instead of being left to the sort order."
        ).format(name, ", ".join(indistinguishable) or "another model")
    return gate + " {} was chosen on {}.".format(name, criterion)
