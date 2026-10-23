"""At a join: stop and remove model serving an earlier controller left on the machine.

Owner 2026-10-22 (VD-133 1.5.5 rc5): the Z2 was wiped and installed fresh, and
the ZBook was re-added as it was. The join worked, and the ZBook kept serving
the wiped controller's model - ``vaelor-vllm-gpu-model-server`` and its gate,
38.2 GB of its 47.2 GB GPU pool - while this controller had no record of
either: the card read "Idle", the serve dialog said the model would not fit,
and nothing in the console could stop them. Decided: a join removes model
serving left on the machine by an earlier controller - Vaelor's own serving
services, never the owner's apps - and keeps the model files on disk.

**Who runs this:** the workload executor, inside `ClusterOperations.join_node`,
over the join's own SSH transport to the worker. Nothing here runs on the
controller's root bridge.

**What counts as left there is decided from THIS controller's records,
keyed on the joining node's id** (:func:`owned_units`). A unit is this
controller's when a deployment record places work on that node id and derives
that unit there (`gpu_pool_units.serving_units`; a pooled CPU record's own
unit names; a deploy's recorded pulls), or when this controller's model
library has a row for that node naming the pull unit. Everything else of
Vaelor's serving shape on the machine is removed. A machine enrolled afresh
has a new node id, so nothing on it is this controller's; a Re-add of this
controller's own worker keeps its id, and its serving stays. The id rather
than the machine's name or address, because a record can manage only what it
can reach through that id - serving that a forced removal left under an old id
is beyond every record's reach, which is the defect.

**Nothing is spelled here.** The unit shapes are `gpu_pool_units`'s and
`pooled_runtime`'s, the stops are the runtimes' own (`GpuPoolRuntime.stop_unit`,
which takes a gate's key file with it, `stop_pull_unit`, `clear_ray_plane`,
`PooledRuntime.stop_unit`), run through `gpu_pool_removal.stop_placements`, and
a container is Vaelor's by `vaelor_containers.vaelor_component` - the rule the
uninstaller's `remove_vaelor_containers` holds - narrowed to the serving and
pull families. Every command is on the worker's SSH allowlist as it stands
(LESSONS 18): ``systemctl`` (``list-units``, ``list-unit-files``, ``show``,
``disable --now``, ``stop``, ``daemon-reload``, ``reset-failed``), ``rm -f`` of
a unit file and a gate's key file, ``nft delete table`` of a split's own table,
and ``docker ps`` / ``docker rm -f``. Model files are never touched.

Never raises (LESSONS 1): a removal that fails, or a machine whose services
could not be listed, is named in the result and said in the join's message.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Set

from .gpu_pool_removal import stop_placements
from .gpu_pool_units import (
    MANAGED_NAME_PREFIXES, PULL_UNIT_PATTERN, RAY_WORKER_ROLE, SERVING_UNIT_PATTERN,
    describe_left_behind, serving_units, unit_deployment, unit_role, unstopped,
)
from .gpu_ray_plane import RayPlane
from .pooled_runtime import POOLED_UNIT_PATTERN
from .vaelor_containers import COMPOSE_PROJECT_LABEL, VAELOR_OWNER_LABEL, vaelor_component

LOGGER = logging.getLogger(__name__)

#: The key the join's result carries this report under.
LEFT_SERVING = "left_serving"

#: The two reads of a worker's services: what systemd has loaded (a unit whose
#: file is gone but which still runs is here) and what unit files exist (a
#: disabled one that is not loaded is there). The uninstaller globs the files;
#: the SSH policy refuses a glob in a systemctl argument, so the whole list is
#: read and matched here instead.
UNIT_READS = (
    ("systemctl", "list-units", "--all", "--type=service", "--no-legend", "--plain"),
    ("systemctl", "list-unit-files", "--type=service", "--no-legend"),
)
#: Every container, one per line: ``name|owner label|compose project``. Each
#: label is read on its own, as the uninstaller's `remove_vaelor_containers`
#: reads them: ``{{.Labels}}`` joins every label with commas, so a label VALUE
#: holding ",io.vaelor.owner=..." would read as Vaelor's own label. In
#: ``docker ps`` ``.Labels`` is that joined STRING, so ``index .Labels`` fails
#: ("cannot index slice/array with type string", found live on Docker 29.1.3,
#: 2026-10-22); ``.Label "key"`` prints one label's value, or nothing.
CONTAINER_READ = (
    "docker", "ps", "-a", "--format",
    '{{{{.Names}}}}|{{{{.Label "{}"}}}}|{{{{.Label "{}"}}}}'.format(
        VAELOR_OWNER_LABEL, COMPOSE_PROJECT_LABEL),
)
_NO_VALUE = "<no value>"

#: The container families removed: model servers, Ray workers, gates and
#: model downloads. Speed-up builds and custom-agent gates are not model serving.
_SERVING_COMPONENT = "gpu-serving"


def _kind(unit: str) -> str:
    """``serving``, ``pull``, ``pooled`` or ``""`` for any other unit."""
    for kind, pattern in (("serving", SERVING_UNIT_PATTERN), ("pull", PULL_UNIT_PATTERN),
                          ("pooled", POOLED_UNIT_PATTERN)):
        if re.fullmatch(pattern, unit):
            return kind
    return ""


def owned_units(store: Any, node_id: str) -> Set[str]:
    """Every serving and pull unit this controller's records put on ``node_id``."""
    node_id = str(node_id)
    owned: Set[str] = set()
    for record in store.list_pooled_deployments():
        units = record.get("units") if isinstance(record.get("units"), Mapping) else {}
        for pull in units.get("pulls") or []:
            if isinstance(pull, Mapping) and str(pull.get("node_id", "")) == node_id:
                owned.add(str(pull.get("unit", "")))
        owned.update(unit for where, unit in serving_units(record) if where == node_id)
        if node_id in {str(item) for item in record.get("node_ids") or []}:
            # A pooled CPU record keeps the names its runtime wrote.
            owned.update(str(unit) for unit in [units.get("root"), *(units.get("workers") or [])]
                         if unit)
    for row in store.list_model_cache():
        if str(row.get("node_id", "")) == node_id and row.get("unit"):
            owned.add(str(row["unit"]))
    return owned


def _listed_units(transport: Any) -> Optional[Set[str]]:
    """Vaelor serving and pull units on the machine, or ``None`` when neither read answered."""
    found: Set[str] = set()
    answered = False
    for argv in UNIT_READS:
        try:
            text = str(transport.run(list(argv), timeout=60))
        except Exception as error:  # noqa: BLE001 - the other read may answer
            LOGGER.warning("Could not read the machine's services (%s): %s", argv[1], error)
            continue
        answered = True
        for line in text.splitlines():
            words = line.replace("\u25cf", " ").split()
            if words and _kind(words[0]):
                found.add(words[0])
    return found if answered else None


def _labels(owner: str, project: str) -> Dict[str, str]:
    """The two labels the container rule reads, without the ones it does not carry."""
    read = {VAELOR_OWNER_LABEL: owner.strip(), COMPOSE_PROJECT_LABEL: project.strip()}
    return {key: value for key, value in read.items() if value and value != _NO_VALUE}


def _left_containers(transport: Any, owned: Set[str]) -> List[str]:
    """Vaelor serving containers on the machine this controller does not own."""
    names = []
    for line in str(transport.run(list(CONTAINER_READ), sudo=True, timeout=60)).splitlines():
        name, _, rest = line.strip().partition("|")
        owner, _, project = rest.partition("|")
        if (name and name.startswith(MANAGED_NAME_PREFIXES)
                and vaelor_component(_labels(owner, project), name) == _SERVING_COMPONENT
                and name + ".service" not in owned):
            names.append(name)
    return sorted(names)


def remove_left_serving(operations: Any, node: Mapping[str, Any], transport: Any) -> Dict[str, Any]:
    """Stop and remove what an earlier controller left serving on ``node``; never raises.

    ``{"removed": [unit or container, ...], "failed": [unstopped entries],
    "unread": "" or why the machine could not be checked}``.
    """
    report: Dict[str, Any] = {"removed": [], "failed": [], "unread": ""}
    node_id, name = str(node.get("id", "")), str(node.get("name") or node.get("id") or "")
    try:
        owned = owned_units(operations.store, node_id)
        listed = _listed_units(transport)
        if listed is None:
            report["unread"] = "its services could not be listed"
            return report
        left = sorted(unit for unit in listed if unit not in owned)
        serving = [unit for unit in left if _kind(unit) == "serving"]
        # Planes first: `stop_placements` stops last-listed first, so a split's
        # firewall is cleared after the Ray worker it fences (ACC-163).
        planes = sorted({unit_deployment(unit) for unit in serving
                         if unit_role(unit) == RAY_WORKER_ROLE})
        placements = [(transport, node, RayPlane(plane)) for plane in planes]
        placements += [(transport, node, unit) for unit in serving]
        report["failed"] += stop_placements(operations.gpu_operations, placements)
        for unit in left:
            stop = {"pull": operations.gpu_runtime.stop_pull_unit,
                    "pooled": operations.pooled_runtime.stop_unit}.get(_kind(unit))
            if stop is None:
                continue
            try:
                stop(transport, unit)
            except Exception as error:  # noqa: BLE001 - named, never masked
                LOGGER.warning("Left serving %s on %s would not stop: %s", unit, name, error)
                report["failed"].append(unstopped(node_id, name, unit, error))
        failed = {entry.get("unit") for entry in report["failed"]}
        report["removed"] = [unit for unit in left if unit not in failed]
        try:
            containers = _left_containers(transport, owned)
            if containers:
                transport.run(["docker", "rm", "-f", *containers], sudo=True, timeout=120)
                report["removed"] += containers
        except Exception as error:  # noqa: BLE001 - named, never masked
            LOGGER.warning("Could not remove leftover serving containers on %s: %s", name, error)
            report["failed"].append(unstopped(node_id, name, "its serving containers", error))
    except Exception as error:  # noqa: BLE001 - the join stands; this is said
        LOGGER.exception("Could not check %s for serving an earlier controller left", name)
        report["unread"] = str(error)[:300] or type(error).__name__
    return report


def left_serving_message(name: str, report: Optional[Mapping[str, Any]]) -> str:
    """The join message's sentences for this step, or ``""`` when it found nothing."""
    if not isinstance(report, Mapping):
        return ""
    sentences = []
    if report.get("removed"):
        sentences.append(
            "Stopped and removed model serving left on {} that this controller has no "
            "deployment for there: {}; its downloaded model files were kept.".format(
                name, ", ".join(report["removed"])))
    if report.get("failed"):
        sentences.append("{} Stop it on {} itself, or remove the machine and add it again.".format(
            describe_left_behind(report["failed"]), name))
    if report.get("unread"):
        sentences.append(
            "{} was not checked for model serving left there that this controller has no "
            "deployment for ({}); if its GPU is still busy, remove the machine and add it "
            "again.".format(name, report["unread"]))
    return " ".join(sentences)
