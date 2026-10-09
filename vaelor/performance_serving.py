"""The serving section of the Performance snapshot: live gauges, or why there are none.

Split out of :mod:`vaelor.performance_snapshot` (VD-147, slice S1a) with no
behaviour change, so the snapshot assembly keeps room under the line ceiling
and the serving words have one home the dashboard can read too.

This module owns three things:

* the words the snapshot uses for what the fleet's GPU cluster record says
  (``CLUSTER_*``) and why an unloaded record is unloaded (``UNLOAD_*``);
* every sentence that explains why there is no live serving reading; and
* :func:`serving_section` and :func:`uncollected_signals`, which turn the newest
  stored serving sample and that record into the block the tab renders.

The honesty rules are the snapshot's: only a fresh sample is live (ACC-049), a
gauge the engine did not emit is ``None`` and never ``0``, and a reason is
always a sentence chosen from what was read - the record, or the sample's age.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Mapping, Optional

from .serving_metrics import GOODPUT_BUDGET_SECONDS
from .serving_metrics import SERVING_POLL_SECONDS

#: How old the newest serving sample may be and still be shown as the engine's
#: live state (ACC-049). The poller writes every ``SERVING_POLL_SECONDS`` while
#: the engine answers and stops writing the moment it does not, so three
#: intervals tolerates two slow or missed scrapes; past this the sample is the
#: last thing a stopped, unloaded or crashed engine said, not what it says now.
SERVING_STALE_AFTER_SECONDS = 3 * SERVING_POLL_SECONDS

#: How far in the FUTURE a sample's time may sit and still count as fresh: a
#: little clock jitter between the store and this process, no more. A stamp
#: further ahead than this is not a reading from "now" but a wrong clock or a
#: wrong unit (milliseconds read as seconds), and its age cannot be trusted.
SERVING_FUTURE_TOLERANCE_SECONDS = 5.0

def _finite(value: Any) -> Optional[float]:
    """A reading as a finite float, or ``None`` — booleans are not readings."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _human_window(window_seconds: int) -> str:
    """A short, human phrase for a window length: \"15 min\", \"1 h\", \"24 h\".

    Used only in the lifetime-fallback note, so a reader sees the span the
    empty window covered without a raw seconds count.
    """
    seconds = max(1, int(window_seconds))
    if seconds < 60:
        return "%d s" % seconds
    if seconds < 3600:
        return "%d min" % max(1, round(seconds / 60))
    hours = round(seconds / 3600)
    if hours < 48:
        return "%d h" % hours
    return "%d d" % round(seconds / 86400)


#: Why the serving section has no reading: the GPU engine is not serving, its
#: metrics endpoint is off or unreachable, or nothing has been scraped yet. One
#: sentence, so the "not collected" state reads the same everywhere.
#: The flat-gauge key prefix the poller encodes per-model RED under. A key is
#: ``model:<model_name>:<field>``; the model name is recovered by stripping this
#: prefix and splitting the final ``:field`` off with one rsplit, so a name that
#: itself carries ``/`` or ``-`` (a Hugging Face id) survives the round trip.
_MODEL_KEY_PREFIX = "model:"

SERVING_NOT_COLLECTED = (
    "The GPU serving engine is not exposing live metrics right now — it is not "
    "serving, its /metrics endpoint is unreachable, or nothing has been scraped "
    "yet — so there are no live serving gauges to show."
)


#: What the fleet's GPU cluster record says about serving, as the snapshot
#: reports it (``serving.cluster_state``; :data:`CLUSTER_UNKNOWN` when it could
#: not be read). This module owns the five words; the record's own states come from
#: ``gpu_cluster_mode`` and are mapped onto them in ``performance_snapshot_source``.
CLUSTER_SERVING = "serving"
CLUSTER_UNLOADED = "unloaded"
CLUSTER_LOADING = "loading"
CLUSTER_NONE = "none"
#: The record could not be read: whether a model is being served is not known.
#: Distinct from :data:`CLUSTER_NONE`, which is a record that was read and
#: holds no GPU cluster deployment.
CLUSTER_UNKNOWN = "unknown"

#: The reason when the fleet's record could not be read and no live sample
#: says otherwise. It claims nothing the record did not support: not "not
#: serving", only that Vaelor cannot tell (VD-147, DG5).
SERVING_RECORD_UNREADABLE = (
    "Vaelor could not read the cluster's deployment record, so whether a model "
    "is being served is not known, and there are no live serving gauges to show."
)

#: The not-collected reason while the fleet records a healthy GPU cluster
#: deployment. The card's badge then reads "Serving - no live metrics" (from
#: that same record), so this sentence speaks about the METRICS scrape only and
#: never says the engine is not serving - the badge and the text would contradict.
SERVING_CLUSTER_NOT_SCRAPED = (
    "A GPU cluster deployment is recorded as serving, but no reading from its "
    "/metrics endpoint has been stored, so there are no live serving gauges to show."
)

#: The record-owned reasons while the cluster model is not in memory. They come
#: from the deployment record, never from the absence of a sample, so they hold
#: even when a last sample is still inside the freshness bound (a sample while
#: the record says unloaded is not live). Idle versus manual follows the lease
#: rule the unload itself writes: an idle unload keeps AI Chat's lease on the
#: cluster credential so the next request wakes it; a manual unload parks it.
SERVING_SCALED_TO_ZERO = "Scaled to zero to free the GPU; it loads on the next request."
SERVING_UNLOADED_BY_HAND = "Unloaded by hand; load it from Cluster > Deployments."
SERVING_UNLOADED = "The cluster model is unloaded, and why could not be read, so there are no live serving gauges."
SERVING_LOADING = "Loading the model; live gauges appear once it answers."
#: Why an unloaded record is unloaded (``serving.cluster_cause``).
UNLOAD_IDLE = "idle"
UNLOAD_MANUAL = "manual"
_RECORD_REASONS = {
    (CLUSTER_UNLOADED, UNLOAD_IDLE): SERVING_SCALED_TO_ZERO,
    (CLUSTER_UNLOADED, UNLOAD_MANUAL): SERVING_UNLOADED_BY_HAND,
}
#: Why there are no gauges when nothing has been scraped, by what the record
#: says. Any other state (a record read and holding no GPU cluster) keeps the
#: general sentence.
_NO_SAMPLE_REASONS = {
    CLUSTER_SERVING: SERVING_CLUSTER_NOT_SCRAPED,
    CLUSTER_UNKNOWN: SERVING_RECORD_UNREADABLE,
}


#: What a serving figure covers when ONE model is split across machines
#: (``distributed``): there is a single engine, read on the lead machine, and
#: its figures are the whole model's. There is no per-machine breakdown to
#: show, and none is implied. ``{count}`` machines, read on ``{lead}``.
SERVING_SPLIT_SCOPE = (
    "One model is split across {count} machines, so these figures are the whole "
    "model's, read on {lead}. There is no per-machine breakdown for a split model."
)
#: The same, when the record does not say which machines.
SERVING_SPLIT_SCOPE_UNNAMED = (
    "One model is split across several machines, so these figures are the whole "
    "model's. There is no per-machine breakdown for a split model."
)
#: The same, when the record names the lead but not a count of machines.
SERVING_SPLIT_SCOPE_LEAD_ONLY = (
    "One model is split across machines, so these figures are the whole model's, "
    "read on {lead}. There is no per-machine breakdown for a split model."
)
#: The record's word for that layout (`cluster_gpu_sizing.VERDICT_DISTRIBUTED`).
PLACEMENT_SPLIT = "distributed"


def serving_scope_note(cluster: Mapping[str, Any]) -> str:
    """The sentence saying what the serving figures cover, or ``""``.

    Empty for one full copy per machine (the per-replica coverage note already
    speaks for that layout), for single-machine serving and for nothing served.
    """
    if cluster.get("placement") != PLACEMENT_SPLIT:
        return ""
    lead, count = str(cluster.get("lead_name") or ""), cluster.get("machines")
    if lead and isinstance(count, int) and count > 1:
        return SERVING_SPLIT_SCOPE.format(count=count, lead=lead)
    if lead:
        # The record names the lead but not how many machines (review nit).
        return SERVING_SPLIT_SCOPE_LEAD_ONLY.format(lead=lead)
    return SERVING_SPLIT_SCOPE_UNNAMED


def _serving_not_live(
    sampled_at: Optional[float], age: Optional[float],
    cluster: Mapping[str, Any], *, future: bool = False,
) -> Dict[str, Any]:
    """The honest "not live" serving block for a sample too old, undated or future-dated.

    Carries the sample's time and age so the screen and the Assistant can say
    how old the last reading is, and no gauges at all: an old reading shown as
    current is exactly the defect (ACC-049). With a healthy cluster deployment
    on record the stale sentence is scoped to the metrics scrape, for the same
    no-contradiction reason as :data:`SERVING_CLUSTER_NOT_SCRAPED`.
    """
    if future:
        reason = (
            "The newest GPU serving reading is stamped in the future, so its age "
            "cannot be trusted and it is not shown as live."
        )
    elif age is None:
        reason = (
            "The newest GPU serving reading carries no readable time, so it "
            "cannot be shown as live."
        )
    elif cluster.get("state") == CLUSTER_SERVING:
        reason = (
            "A GPU cluster deployment is recorded as serving, but the last reading "
            "from its /metrics endpoint is " + _human_window(age) + " old, so the "
            "gauges are not shown as live: the metrics scrape has stopped reaching it."
        )
    else:
        reason = (
            "The GPU serving engine's last reading is " + _human_window(age)
            + " old, so it is not shown as live: the engine has stopped, unloaded "
            "or crashed, or its /metrics endpoint no longer answers the scrape."
        )
    return {
        "collected": False,
        "stale": True,
        "cluster_state": cluster.get("state"),
        "reason": reason,
        "sampled_at": sampled_at,
        "age_seconds": None if age is None or future else round(age),
        "metrics": {},
        "models": [],
        "coverage_note": "",
    }


def _coverage_note(metrics: Mapping[str, Any]) -> str:
    """A plain sentence when the aggregate covers fewer replicas than exist (ACC-053).

    Also when every replica answered but some had just restarted: a restarted
    replica's throughput for that interval is not known, so the totals are
    partial too (review S-10).
    """
    read = metrics.get("replicas_read")
    total = metrics.get("replicas_total")
    restarted = metrics.get("replicas_restarted")
    if read is not None and total is not None and read < total:
        return (
            f"These figures cover {round(read)} of {round(total)} replicas; the others "
            "could not be read at the last scrape, so totals and queue depths are partial."
        )
    if restarted and total:
        return (
            f"{round(restarted)} of {round(total)} replicas restarted since the previous "
            "reading, so their output is not in these throughput figures."
        )
    return ""


def serving_section(
    serving: Optional[Mapping[str, Any]], now: Optional[float] = None,
    cluster: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """:func:`_serving_block`, with the deployment's layout and what the figures cover.

    ``placement`` is the record's word for how the deployment is laid out
    (``replicated``, ``distributed`` or ``""``), and ``scope_note`` the owner
    sentence for a split model (:func:`serving_scope_note`) - present in every
    state, so an unloaded or stale split deployment says the same thing.
    """
    block = _serving_block(serving, now, cluster)
    record = cluster if isinstance(cluster, Mapping) else {}
    block["placement"] = str(record.get("placement") or "")
    block["scope_note"] = serving_scope_note(record)
    # How many machines the deployment spans, from the record: a split model is
    # one engine (one replica) across several machines (pass-3 review).
    machines = record.get("machines")
    block["machines"] = int(machines) if isinstance(machines, int) and not isinstance(machines, bool) else None
    return block


def _serving_block(
    serving: Optional[Mapping[str, Any]], now: Optional[float] = None,
    cluster: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The live serving gauges the E′ scrape collected, or the honest empty state.

    ``serving`` is the newest stored serving sample (from
    :func:`vaelor.serving_store.latest_serving`), a mapping of the normalized
    gauge keys, or ``None``/empty when the engine is not serving or nothing has
    been scraped. When present, each gauge is read through :func:`_finite`, so a
    field the engine did not emit is ``None`` here — never a fabricated ``0`` —
    and the block is marked ``collected``. When absent, the block is the explicit
    "not collected" state with a truthful reason and no numbers at all.

    The five llama.cpp gauges (decode and prefill rate, running and waiting
    requests, and the per-decode batch size) are always keyed, ``None`` when the
    sample omits one. The deeper vLLM-cluster signals (``kv_cache_fraction``,
    ``ttft_seconds``, ``tpot_seconds`` and the ``request_success_total`` counter)
    are keyed only when the sample carries them, so a single-node llama.cpp
    sample surfaces none of them and :func:`uncollected_signals` still reports
    them as not collected on that engine.

    **Only a fresh sample is live (ACC-049).** The sample's ``sampled_at`` (which
    :func:`vaelor.serving_store.latest_serving` stamps from the row's own time)
    must be within :data:`SERVING_STALE_AFTER_SECONDS` of ``now``; an older or
    undated one yields the not-live block with its age, so neither the card nor
    the Assistant repeats the last figures of an engine that has stopped. So
    does one stamped more than :data:`SERVING_FUTURE_TOLERANCE_SECONDS` ahead.

    ``cluster`` is the fleet's GPU cluster record as ``{state, cause}`` (``state``
    one of :data:`CLUSTER_SERVING`, :data:`CLUSTER_UNLOADED`,
    :data:`CLUSTER_LOADING`, :data:`CLUSTER_NONE`), or ``None`` when it could not
    be read, which is :data:`CLUSTER_UNKNOWN`: with no sample the reason is
    :data:`SERVING_RECORD_UNREADABLE`, never a sentence saying nothing serves.
    It is echoed as ``cluster_state``. Unloaded or loading answers from the
    record whatever the sample says; serving scopes every not-collected reason
    to the metrics scrape, so no badge is contradicted by its sentence.
    """
    # No record at all (the store is unwired, locked or unreadable) is
    # "unknown", never the silence of a record that was read and empty.
    cluster = dict(cluster) if cluster else {"state": CLUSTER_UNKNOWN}
    state = cluster.get("state")
    if state in (CLUSTER_UNLOADED, CLUSTER_LOADING):
        reason = (
            SERVING_LOADING if state == CLUSTER_LOADING
            else _RECORD_REASONS.get((state, cluster.get("cause")), SERVING_UNLOADED)
        )
        return {
            "collected": False, "stale": False, "cluster_state": state,
            "cluster_cause": cluster.get("cause") or "", "reason": reason,
            "sampled_at": None, "age_seconds": None, "metrics": {}, "models": [],
            "coverage_note": "",
        }
    if not serving:
        return {
            "collected": False, "stale": False, "cluster_state": state,
            "reason": _NO_SAMPLE_REASONS.get(state, SERVING_NOT_COLLECTED),
            "sampled_at": None, "age_seconds": None, "metrics": {}, "models": [],
            "coverage_note": "",
        }
    sampled_at = _finite(serving.get("sampled_at"))
    clock = time.time() if now is None else float(now)
    raw_age = None if sampled_at is None else clock - sampled_at
    if raw_age is not None and raw_age < -SERVING_FUTURE_TOLERANCE_SECONDS:
        return _serving_not_live(sampled_at, None, cluster, future=True)
    age = None if raw_age is None else max(0.0, raw_age)
    if age is None or age > SERVING_STALE_AFTER_SECONDS:
        return _serving_not_live(sampled_at, age, cluster)
    metrics: Dict[str, Any] = {
        "decode_tokens_per_second": _finite(serving.get("decode_tokens_per_second")),
        "prompt_tokens_per_second": _finite(serving.get("prompt_tokens_per_second")),
        "requests_processing": _finite(serving.get("requests_processing")),
        "requests_deferred": _finite(serving.get("requests_deferred")),
        "busy_slots_per_decode": _finite(serving.get("busy_slots_per_decode")),
    }
    # The deep vLLM-cluster signals live beside the five llama.cpp gauges, but on
    # a different contract: each is ADDED only when the serving sample actually
    # carries it, so a llama.cpp sample that emits none of them leaves no key
    # here (never a None placeholder for a value that engine cannot produce).
    # Their presence is what lets uncollected_signals tell the truth per engine.
    for deep_key in (
        "generation_throughput_tokens_per_second",
        "prompt_throughput_tokens_per_second",
        "replicas_read",
        "replicas_total",
        "replicas_restarted",
        "kv_cache_fraction",
        "ttft_seconds",
        "tpot_seconds",
        "request_success_total",
        "prefix_cache_hit_rate",
        "preemptions_total",
        "goodput_ratio",
        "goodput_budget_seconds",
    ):
        deep_value = _finite(serving.get(deep_key))
        if deep_value is not None:
            metrics[deep_key] = deep_value
    # Per-model RED arrives as encoded flat gauges (model:<name>:<field>); decode
    # them into a models list and keep them OUT of metrics, which carries scalar
    # whole-engine gauges only, so a per-model figure is not shown twice.
    models = _decode_models(serving)
    return {
        "collected": True, "stale": False, "cluster_state": state,
        "reason": "", "sampled_at": sampled_at,
        "age_seconds": round(age), "metrics": metrics, "models": models,
        "coverage_note": _coverage_note(metrics),
    }


def _decode_models(serving: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The per-model RED rows decoded from the ``model:<name>:<field>`` gauges.

    Each row is ``{model, requests_total, errors_total, error_rate,
    avg_e2e_seconds}``; a model is kept only when its request count is present,
    ``error_rate`` is ``0.0`` on zero requests rather than a divide, and
    ``avg_e2e_seconds`` is ``None`` when the engine reported no e2e observations.
    Rows are sorted by request count, busiest first.
    """
    totals: Dict[str, Dict[str, Any]] = {}
    for raw_key, raw_value in serving.items():
        if not isinstance(raw_key, str) or not raw_key.startswith(_MODEL_KEY_PREFIX):
            continue
        model_name, _, field = raw_key[len(_MODEL_KEY_PREFIX):].rpartition(":")
        if not model_name or not field:
            continue
        value = _finite(raw_value)
        if value is None:
            continue
        totals.setdefault(model_name, {})[field] = value
    models: List[Dict[str, Any]] = []
    for model_name, fields in totals.items():
        requests = fields.get("requests_total")
        if requests is None:
            continue
        errors = fields.get("errors_total", 0.0)
        models.append({
            "model": model_name,
            "requests_total": requests,
            "errors_total": errors,
            "error_rate": (errors / requests) if requests > 0 else 0.0,
            "avg_e2e_seconds": fields.get("avg_e2e_seconds"),
        })
    models.sort(key=lambda entry: entry["requests_total"], reverse=True)
    return models


def uncollected_signals(serving: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The deep serving signals not yet present in this sample, each with its unlock.

    Truthful per signal, never a fabricated number and never self-contradicting:
    a signal is listed here if, and only if, the ``serving`` sample does NOT carry
    the reading that proves it. ``serving`` is the collected serving section (from
    :func:`serving_section`).

    On the single-node **llama.cpp** engine (Mode A), and with no ``serving`` at
    all, the sample carries none of these, so every one is listed against the vLLM
    cluster serving that emits them. On a **vLLM** cluster (Mode B) the controller
    scrape feeds the readings in as they appear, and each drops out of this list
    the moment its proof is in the sample - when all of them are, the list is empty
    and the footer says the scrape now feeds them all.
    """
    metrics = serving.get("metrics") if isinstance(serving, Mapping) else None
    metrics = metrics or {}
    models = serving.get("models") if isinstance(serving, Mapping) else None
    # Each scalar deep signal, mapped to the serving metric key whose presence in
    # the sample proves it was collected. per_model_red is proven differently: by
    # a non-empty decoded models list, not a scalar gauge.
    proof_key = {
        "ttft": "ttft_seconds",
        "tpot": "tpot_seconds",
        "kv_cache": "kv_cache_fraction",
        "preemptions": "preemptions_total",
        "prefix_cache": "prefix_cache_hit_rate",
        "goodput": "goodput_ratio",
    }
    per_model_collected = bool(models)

    def collected(signal_id: str) -> bool:
        if signal_id == "per_model_red":
            return per_model_collected
        key = proof_key.get(signal_id)
        return key is not None and key in metrics

    budget = round(GOODPUT_BUDGET_SECONDS)
    candidates = [
        ("per_model_red", "Per-model / per-deployment RED",
         "Per-model RED reads the vLLM engine's per-model_name series; the gateway "
         "request log has no model column of its own."),
        ("ttft", "Time to first token (TTFT)",
         "TTFT is the vLLM time-to-first-token histogram, read from the cluster's "
         "/metrics."),
        ("tpot", "Time per output token (TPOT)",
         "TPOT is the vLLM inter-token-latency histogram, read from the cluster's "
         "/metrics."),
        ("kv_cache", "KV-cache utilisation",
         "KV-cache occupancy is the vLLM kv_cache_usage gauge, read from the "
         "cluster's /metrics."),
        ("preemptions", "Preemptions",
         "Preemption counts are the vLLM num_preemptions counter, read from the "
         "cluster's /metrics."),
        ("prefix_cache", "Prefix-cache hit rate",
         "Prefix-cache reuse is the vLLM prefix-cache hit and query counters, read "
         "from the cluster's /metrics."),
        ("goodput", "Goodput (requests within the latency budget)",
         "Goodput is the share of requests finishing within the %d s budget, from "
         "the vLLM end-to-end latency histogram." % budget),
    ]
    signals = [
        {"id": signal_id, "label": label, "reason": reason}
        for signal_id, label, reason in candidates
        if not collected(signal_id)
    ]
    if not signals:
        next_step = (
            "The vLLM cluster's Prometheus /metrics scrape now feeds every deep "
            "serving signal - TTFT, TPOT, KV-cache, prefix-cache hit rate, "
            "preemptions, goodput and per-model RED - so none remain uncollected."
        )
    else:
        remaining = ", ".join(signal["label"] for signal in signals)
        next_step = (
            "Collected from the vLLM cluster's Prometheus /metrics scrape once GPU "
            "clustering serves the model there. Still uncollected in this sample: "
            + remaining + "."
        )
    return {
        "collected": not signals,
        "next_step": next_step,
        "signals": signals,
    }
