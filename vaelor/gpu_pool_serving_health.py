"""Whether a cluster deployment is serving NOW, and the one word the card shows.

ACC-055 (High). A model split across machines (the ``distributed`` intent) was
health-checked only while it deployed: once the row read ``healthy`` nothing
looked at it again, so stopping one half left it ``healthy`` for ever, and the
Endpoints card's "Serving" was fixed text. The replicated intent already had a
watch (`gpu_pool_replicas.ReplicaHealth`); this module gives the split one and
owns the card's reading for both.

Runs in two processes:

* the **workload executor** - :class:`SplitHealth` is driven by the 30 s mode
  watch (`gpu_cluster_mode_watch`), which writes its reading onto the row as
  ``units.health`` and, after :data:`SPLIT_DOWN_PASSES` failed checks in a row,
  rewrites the row ``failed`` with a sentence naming the part that stopped;
* the **control plane** - :func:`serving_reading` turns the stored row into the
  word and sentence ``GET /cluster/serving`` hands the card. Derived from what
  the watch MEASURED and wrote; a row nobody has checked says so.

**What "serving" means for a split.** Every unit the row owns is running (the
lead's server and each worker's Ray process, read with ``systemctl show`` over
each node's own transport - the controller's bridge, a worker's SSH) AND the
lead's vLLM ``/health`` answers. The unit reading is the one that sees a
stopped half: with pipeline parallelism the lead can keep answering ``/health``
for a while after a worker's Ray process is gone.

**Unknown is not down (owner rule, 2026-09-28).** A part that could not be ASKED
(its machine unreachable, SSH or the broker failing) is not a part that
stopped. While the lead's API still answers, such a pass reads
:data:`DEGRADED` - written to ``units.degraded_reason``, the field the fleet
display reads - and counts nothing towards the teardown. Only a MEASURED stop,
or an API that does not answer, counts; :data:`SPLIT_DOWN_PASSES` of those in a
row mark the row failed.

**Cost per pass (S10).** One ``joined_node`` (one broker resolve) and one
transport per machine, every unit on that machine read over it, and one
``/health`` probe bounded by :data:`PROBE_TIMEOUT_SECONDS`.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, Optional

from .gpu_pool_replicas import (
    REPLICA_ALIVE, REPLICA_DOWN, REPLICA_DOWN_PASSES, REPLICA_UNREACHABLE,
    replica_entries,
)
from .gpu_pool_startup import UNREACHABLE, describe_unit_state, unit_alive
from .gpu_pool_units import (
    VLLM_ENGINE, is_replicated, serving_units, unit_role,
)

#: Failed checks in a row before a split row is marked ``failed``: the same
#: count, on the same 30 s watch, that fails a replicated row with no replica
#: answering - one rule for "stopped serving", whatever the intent.
SPLIT_DOWN_PASSES = REPLICA_DOWN_PASSES

#: How long the lead's ``/health`` may take on one watch pass.
PROBE_TIMEOUT_SECONDS = 3.0

#: The words :func:`serving_reading` answers, and so the card's vocabulary.
SERVING = "serving"
DEGRADED = "degraded"
PARTLY_SERVING = "partly-serving"
NOT_ANSWERING = "not-answering"
NOT_CHECKED = "not-checked"
PAUSED = "paused"
STARTING = "starting"
FAILED = "failed"
SERVING_WORDS = (
    SERVING, DEGRADED, PARTLY_SERVING, NOT_ANSWERING, NOT_CHECKED, PAUSED,
    STARTING, FAILED,
)

#: What a degraded split says: which machines could not be asked. Written to
#: ``units.degraded_reason`` (the fleet's degraded display) and cleared again
#: when they answer - unless a forced removal owns that field (``lost_nodes``).
SPLIT_DEGRADED = (
    "Vaelor could not reach {nodes} to check its part of this model. The model "
    "still answers, but it may be serving at reduced capacity; it is shown as "
    "degraded until that part can be checked again."
)

#: One part's last state, as :class:`SplitHealth` writes it.
PART_RUNNING = "running"
PART_STOPPED = "stopped"
PART_UNREACHABLE = "unreachable"

#: The failure a split row is marked with. ``parts`` is :func:`describe_parts`.
SPLIT_DOWN = (
    "The model split across machines stopped serving: it failed {passes} "
    "health checks in a row, 30 seconds apart. Last reading: {parts}."
)

#: What a serving split's row says from its first failed check until the watch
#: fails it (W4-D6): the last reading and the window. The window is not
#: shortened - the same checks ride a worker through a reboot (ACC-187).
_SPLIT_CHECKING_WINDOW = (
    "This split did not answer its last health check ({detail}). Vaelor checks "
    "every 30 seconds and stops it after {passes} failed checks in a row (about "
    "two minutes)"
)
SPLIT_CHECKING = _SPLIT_CHECKING_WINDOW + "; this was {count} of {passes}."

#: The same sentence when the row's count cannot be read (review A5): the
#: window is still said, and no number is invented for the count.
SPLIT_CHECKING_UNCOUNTED = _SPLIT_CHECKING_WINDOW + "."


def down_passes(health: Any) -> int:
    """The failed checks in a row a split's ``units.health`` records; 0 when unreadable.

    The watch writes a non-negative integer. Anything else - a hand-edited or
    corrupted row - reads as no count rather than raising, because this is
    read for every row of the Deployments list and on every watch pass, and
    one bad row must not take either down (review A5).
    """
    value = health.get("down_passes") if isinstance(health, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def split_checking_note(units: Mapping[str, Any], state: str) -> str:
    """The row's sentence while a serving split is failing its checks, else ``""``."""
    health = (units or {}).get("health")
    if (
        str(state) != "healthy" or not isinstance(health, Mapping)
        or (units or {}).get("engine") != VLLM_ENGINE or is_replicated({"units": units})
        or health.get("state") != NOT_ANSWERING
    ):
        return ""
    count = down_passes(health)
    # A NOT_ANSWERING reading always counts at least one failed check; zero
    # here means the count could not be read.
    return (SPLIT_CHECKING if count else SPLIT_CHECKING_UNCOUNTED).format(
        detail=str(health.get("detail", "") or "no part answered"),
        passes=SPLIT_DOWN_PASSES, count=count,
    )


#: The sentences the card shows beside a word that is not "serving".
_DETAILS = {
    NOT_CHECKED: (
        "Vaelor has not checked this model since it was deployed; the first "
        "check runs within a minute."
    ),
    PAUSED: (
        "The model is unloaded to free the GPU. Vaelor could not read whether "
        "it was unloaded after sitting idle or by hand; loading it from "
        "Cluster > Deployments brings it back."
    ),
    STARTING: "The model is loading. It serves once every part has started.",
}


#: The paused sentence by who unloaded it (`gpu_serving_target.deployment_unload_cause`).
_PAUSED_DETAILS = {
    "unloaded-idle": (
        "The model was unloaded after sitting idle to free the GPU; the next "
        "request that needs it loads it again."
    ),
    "unloaded-manual": (
        "The model was unloaded by hand; load it from Cluster > Deployments."
    ),
}


def health_url(endpoint: str) -> str:
    """vLLM's ``/health`` on the lead, from the row's ``.../v1`` endpoint."""
    base = str(endpoint or "").rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base + "/health"


def describe_parts(parts: List[Mapping[str, Any]], api: str) -> str:
    """``the server on Z2 (running); the Ray worker on ZBook (stopped); the API (not answering)``."""
    roles = {"server": "the model server", "ray-worker": "the Ray worker"}
    pieces = [
        "{} on {} ({})".format(
            roles.get(str(part.get("role", "")), "the part"),
            part.get("node") or part.get("node_id") or "an unknown machine",
            part.get("state", PART_UNREACHABLE),
        )
        for part in parts
    ]
    pieces.append("its API ({})".format(api))
    return "; ".join(pieces)


class SplitHealth:
    """The mode watch's reading of a healthy ``distributed`` row.

    ``node_reader(node_id)`` returns ``(name, read)`` for one machine - its
    display name and a reader of one unit's systemd properties over ONE
    transport - raising when the machine cannot be resolved; ``read`` raises
    when the machine cannot be asked. ``probe`` is an HTTP probe answering
    ``True``/``False``/``UNREACHABLE``. The count of failed checks lives ON THE
    ROW (``units.health.down_passes``), so an executor restart neither forgets
    a failing split nor invents one.
    """

    def __init__(
        self, *, node_reader: Callable[[str], Any],
        probe: Callable[[str], Optional[bool]],
        passes: int = SPLIT_DOWN_PASSES,
        clock: Callable[[], float] = time.time,
    ):
        self._node_reader = node_reader
        self._probe = probe
        self._passes = max(1, int(passes))
        self._clock = clock

    @classmethod
    def for_operations(cls, ops: Any) -> Optional["SplitHealth"]:
        """Built over `GpuPoolOperations`' own resolver and transports, or ``None``."""
        runtime = getattr(ops, "runtime", None)
        joined = getattr(ops, "joined_node", None)
        transport = getattr(ops, "_transport", None)
        if None in (runtime, joined, transport):
            return None

        def node_reader(node_id: str) -> Any:
            node = joined(node_id, include_credential=True)  # one resolve per machine
            carrier = transport(node)  # one transport per machine
            return (
                str(node.get("name") or node_id),
                lambda unit: runtime.unit_state(carrier, unit),
            )

        return cls(node_reader=node_reader, probe=bounded_probe)

    def observe(self, record: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """``{health, failure}`` for a healthy split vLLM row, else ``None``."""
        units = record.get("units") or {}
        if (
            str(record.get("state", "")) != "healthy"
            or units.get("engine") != VLLM_ENGINE
            or is_replicated(record)
        ):
            return None
        parts = self._read_parts(record)
        answer = self._probe(health_url(str(record.get("endpoint", ""))))
        # The replica watch's words for what an HTTP probe found.
        api = (
            REPLICA_ALIVE if answer is True
            else REPLICA_UNREACHABLE if answer is UNREACHABLE else REPLICA_DOWN
        )
        stopped = any(part["state"] == PART_STOPPED for part in parts)
        unknown = [part for part in parts if part["state"] == PART_UNREACHABLE]
        down_now = stopped or answer is not True
        if not down_now and not unknown:
            state = SERVING
        elif not down_now:
            state = DEGRADED  # could not be asked, still answering: not down
        else:
            state = NOT_ANSWERING
        previous = units.get("health") if isinstance(units.get("health"), Mapping) else {}
        down = down_passes(previous) + 1 if down_now else 0
        health = {
            "state": state,
            "checked_at": int(self._clock()),
            "down_passes": down,
            "api": api,
            "parts": parts,
            "detail": "" if state == SERVING else describe_parts(parts, api),
        }
        failure = ""
        if down >= self._passes:
            failure = SPLIT_DOWN.format(passes=self._passes, parts=health["detail"])
        degraded = ""
        if state == DEGRADED:
            degraded = SPLIT_DEGRADED.format(nodes=", ".join(
                sorted({str(part["node"]) for part in unknown})))
        return {"health": health, "failure": failure, "degraded_reason": degraded}

    def _read_parts(self, record: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Every unit the row owns, read machine by machine over one transport each."""
        by_node: Dict[str, List[str]] = {}
        for node_id, unit in serving_units(record):
            by_node.setdefault(node_id, []).append(unit)
        parts: List[Dict[str, Any]] = []
        for node_id, units in by_node.items():
            try:
                name, read = self._node_reader(node_id)
            except Exception as error:  # noqa: BLE001 - an unresolvable machine is unknown
                name, read, problem = node_id, None, str(error)[:200]
            for unit in units:
                part = {"node_id": node_id, "node": name, "role": unit_role(unit), "unit": unit}
                if read is None:
                    part.update(state=PART_UNREACHABLE, detail=problem)
                    parts.append(part)
                    continue
                try:
                    properties = read(unit)
                except Exception as error:  # noqa: BLE001 - an unaskable node is unknown
                    part.update(state=PART_UNREACHABLE, detail=str(error)[:200])
                else:
                    alive = unit_alive(properties)
                    part.update(
                        state=PART_RUNNING if alive else PART_STOPPED,
                        detail=describe_unit_state(properties),
                    )
                parts.append(part)
        return parts


def bounded_probe(url: str) -> Optional[bool]:
    """The deploy's probe semantics, bounded to :data:`PROBE_TIMEOUT_SECONDS`."""
    try:
        with urllib.request.urlopen(url, timeout=PROBE_TIMEOUT_SECONDS) as response:
            return response.status == 200
    except urllib.error.HTTPError:
        return False
    except (OSError, urllib.error.URLError):
        return UNREACHABLE


def serving_reading(record: Mapping[str, Any], unload_cause: str = "") -> Dict[str, str]:
    """``{state, detail}``: the card's one word for a cluster row, and why.

    From the row the watch writes, never assumed: a healthy replicated row
    counts its replicas' last ``alive`` readings; a healthy split row reads its
    ``units.health``; a row no check has reached is :data:`NOT_CHECKED`, not
    "serving".
    """
    state = str(record.get("state", "") or "")
    units = record.get("units") or {}
    if state == "failed":
        return {"state": FAILED, "detail": str(units.get("failure", "") or "")}
    if state == "unloaded":
        return {"state": PAUSED, "detail": _PAUSED_DETAILS.get(unload_cause, _DETAILS[PAUSED])}
    if state == "deploying":
        return {"state": STARTING, "detail": _DETAILS[STARTING]}
    if state != "healthy":
        return {"state": NOT_CHECKED, "detail": _DETAILS[NOT_CHECKED]}
    if is_replicated(record):
        entries = replica_entries(record)
        alive = sum(1 for entry in entries if entry.get("alive"))
        if entries and alive == len(entries):
            return {"state": SERVING, "detail": ""}
        if not entries or not any("alive" in entry for entry in entries):
            return {"state": NOT_CHECKED, "detail": _DETAILS[NOT_CHECKED]}
        word = PARTLY_SERVING if alive else NOT_ANSWERING
        return {"state": word, "detail": (
            "{} of {} copies of the model are answering.".format(alive, len(entries))
        )}
    health = units.get("health") if isinstance(units.get("health"), Mapping) else None
    if not health:
        return {"state": NOT_CHECKED, "detail": _DETAILS[NOT_CHECKED]}
    if health.get("state") == SERVING:
        return {"state": SERVING, "detail": ""}
    if health.get("state") == DEGRADED:
        return {"state": DEGRADED, "detail": str(
            units.get("degraded_reason") or health.get("detail") or "")}
    return {"state": NOT_ANSWERING, "detail": (
        "The model split across machines is not answering: {}.".format(
            health.get("detail") or "no part could be read")
    )}
