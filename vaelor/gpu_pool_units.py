"""What a vLLM GPU deployment runs on each node, derived from its record - once.

Every name the GPU serving tier writes into a node's systemd has one home here
- the serving container and unit per role, the model-pull oneshot per repo -
and so does the answer to "which serving units does THIS deployment own, and on
which node": :func:`serving_units` reads a stored ``pooled_deployments`` record
and says. Both are one definition because the defect they close was two.

**The record used to carry a ``started`` list, written only with the healthy
or the failed record.** A deploy that DIED - the executor killed between
`start_distributed_server` and the health probe - left a ``deploying`` row
whose units said only ``{"engine": "vllm"}``. The mode watch then returned the
appliance to Mode A with nothing stopped, the GPU failure-watch relaunched
llama.cpp into the aperture ``vaelor-vllm-<name>-server.service`` was still
loading into (``WantedBy=multi-user.target``, ``Restart=on-failure``: a double
residency that survived reboot), and `remove` on that row read the absent list,
stopped nothing, and deleted the only record naming the units - while the row's
operator note promised that removal would clear them. The units were never
unknowable: a record's name and its node list determine them exactly, because
`gpu_pool_operations.deploy` guarantees the lead is ``node_ids[0]`` and the
runtime's start functions spell their names from the deployment name alone. So
the names are FUNCTIONS of the record, the starts call the same functions, and
nothing is written down that could disagree with them.

**The derivation is mode-aware (VD-129).** A ``distributed`` record runs the
server on ``node_ids[0]`` and a Ray worker on every other node; a
``replicated`` record runs one server on EVERY node (the throughput intent:
one whole copy of the model per machine behind a balancer on the controller)
and, beside each WORKER's server, a gate - the nginx container that is the
only thing on that machine listening on the LAN, because vLLM's own key was
live-probed to leave ``/invocations``, ``/tokenize`` and ``/metrics`` open
(the VD-129 amendment). The record says which under ``units["mode"]`` -
written on the ``deploying`` row and every row after it - and a record with
NO mode (written before the word existed) derives the UNION of both shapes,
because a stop of a unit systemd never loaded is the runtime's no-op and a
mode-less row must be able to orphan nothing. The balancer is not a unit - it
is a container the root bridge runs - so `gpu_pool_operations.stop_units_for`
stops it beside the units; its name is spelled here for the same reason the
units' are.

Named here but NOT derived per record: the model-pull oneshot. Its name is per
REPO, shared with the model library's own pulls of that repo on the same node,
and it holds no GPU - a deploy's failure record lists the pulls it left running
(``units["pulls"]``) because only the deploy knows which it started.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .cluster_gpu_sizing import VERDICT_DISTRIBUTED, VERDICT_REPLICATED
from .cluster_placement import CONTROLLER_PLACEMENT_ID

#: The deployment-name rule every managed serving name embeds: short, lowercase,
#: hyphenatable. Written once, because :func:`deployment_name` enforces it and
#: :data:`SERVING_UNIT_PATTERN` matches it - a rule and its matcher spelled
#: twice are the two things that drift.
_NAME_RULE = r"[a-z0-9][a-z0-9-]{0,38}"

#: What ``units["engine"]`` says on every vLLM record - ``deploying`` included,
#: so a stuck deploy is offered the GPU removal path, never the CPU one. Owned
#: here with the other words a record carries; `gpu_pool_operations` and
#: `gpu_pool_replicas` import it, and `cluster_manager` reads it.
VLLM_ENGINE = "vllm"
#: The single-machine engine's word, beside the cluster engine's: this module
#: owns both, and the usage ledger, the speed verdict and the Performance
#: payloads import them (VD-147).
LLAMACPP_ENGINE = "llama.cpp"

#: The three things this tier runs as units. A distributed deployment runs the
#: lead's merged Ray head and vLLM server as ONE container
#: (`GpuPoolRuntime.start_distributed_server`) and a Ray worker on every other
#: participant; a replicated one runs a server on every node and, on every
#: WORKER, a gate (`GpuPoolRuntime.start_gate`, VD-129): the product's nginx
#: image on that worker's cluster address, requiring the cluster key and
#: forwarding only ``/v1/`` and ``/health`` to the replica on loopback.
SERVER_ROLE = "server"
RAY_WORKER_ROLE = "ray-worker"
GATE_ROLE = "gate"
_ROLES = (SERVER_ROLE, RAY_WORKER_ROLE, GATE_ROLE)

#: The balancer a replicated deployment runs on the controller (VD-129): a
#: second nginx container beside the LLM Server proxy, named in the serving
#: family so the root bridge's ``docker stop``/``rm`` shapes reach it. It is a
#: container and never a unit, so :data:`_ROLES` does not list it and
#: `serving_units` never derives it.
BALANCER_ROLE = "balancer"

#: What every managed serving container and unit begins with, and what every
#: pull container and unit begins with.
_SERVING_PREFIX = "vaelor-vllm-"
_PULL_PREFIX = "vaelor-model-pull-"
#: Both, for a reader that must recognise any name this module makes - the
#: on-device Assistant's reply judge refuses a reply that only lists them.
MANAGED_NAME_PREFIXES = (_SERVING_PREFIX, _PULL_PREFIX)

#: Where a WORKER's gate reads its nginx config from (VD-129): a root-owned
#: ``0600`` file per gate unit, carrying the cluster key, written with
#: ``install -m 0600 /dev/stdin`` (never ``tee``, whose file is ``0644``) and
#: mounted read-only into the gate container, so the key is on no argv, in no
#: unit file and in no ``docker inspect`` outside root. `GpuPoolRuntime.stop_unit`
#: removes it with the unit. The root bridge's argv policy spells the same
#: directory itself (a policy may not import the code it polices);
#: `tests/test_gpu_pool_units.py` ties the two spellings.
GATE_CONFIG_ROOT = "/etc/vaelor/vllm"
#: Its mode, 0755 (VD-194 P0, decided): the secrets are the 0600 files inside,
#: and the gate deploy and the worker profile both assert this one value, so
#: neither resets what the other wrote (LESSONS 6).
GATE_CONFIG_ROOT_MODE = "0755"

#: The docker binary the units invoke, by absolute path. A systemd ``ExecStart``
#: has no ``PATH`` search worth trusting, and the sibling unit author
#: `pooled_runtime` writes absolute paths for the same reason. The *transport*
#: calls keep the bare ``docker`` token - the spelling the allowlist matches.
DOCKER = "/usr/bin/docker"

#: The ordering both unit kinds declare: docker must be up, and the network too,
#: before a container that pulls an image or joins a Ray cluster is started.
#: Written once so the serving unit and the pull oneshot cannot drift apart.
UNIT_AFTER = "After=network-online.target docker.service"

#: How long ``docker stop`` may drain a serving container before docker kills it.
#: Freeing a multi-GB model off the GPU is not instant.
STOP_GRACE_SECONDS = 30

#: The exact shape a managed serving unit name takes, assembled from the SAME
#: pieces :func:`unit_name` assembles, so `GpuPoolRuntime.stop_unit` can never
#: disable an unrelated unit and can never refuse one this module named. The
#: role is a named group so :func:`unit_role` can read it back off a unit the
#: stop is handed - the gate's config file goes with the gate and nothing
#: else. Both patterns are STRINGS compiled at their use site - the tree's
#: convention for a rule no table instruments - never a module-level
#: ``re.Pattern``.
SERVING_UNIT_PATTERN = (
    re.escape(_SERVING_PREFIX) + _NAME_RULE
    + "-(?P<role>" + "|".join(re.escape(role) for role in _ROLES) + r")\.service"
)
PULL_UNIT_PATTERN = re.escape(_PULL_PREFIX) + r"[a-z0-9][a-z0-9-]{0,63}\.service"


def deployment_name(value: Any) -> str:
    """The one home for the deployment-name rule.

    Both the operations entry point and the runtime validate the name through
    this, so the rule and its message cannot drift. It is a module function
    rather than a runtime method so the operations layer's call survives a
    mocked runtime.
    """
    name = str(value).strip().lower()
    if not re.fullmatch(_NAME_RULE, name):
        raise ValueError("Use a short lowercase GPU deployment name.")
    return name


def container_name(name: str, role: str) -> str:
    """``vaelor-vllm-<name>-<role>``: the docker container a serving unit runs.

    Spelling only. The runtime validates ``name`` through
    :func:`deployment_name` before it writes anything, and a stored record's
    name was validated when the record was written; a name that was never
    validated yields a unit :func:`require_serving_unit` refuses at the stop.
    """
    if role not in _ROLES:
        raise ValueError("The GPU serving role is not one this tier runs.")
    return f"{_SERVING_PREFIX}{name}-{role}"


def unit_name(name: str, role: str) -> str:
    """The systemd unit supervising :func:`container_name`: the same name, ``.service``."""
    return container_name(name, role) + ".service"


def unit_role(unit: Any) -> str:
    """The role a managed serving unit runs, or a refusal for any other name.

    The ONE match against :data:`SERVING_UNIT_PATTERN`: :func:`require_serving_unit`
    is this with the name handed back, and `GpuPoolRuntime.stop_unit` asks it
    which unit is a gate, so the config file only a gate ever had is removed
    with that unit and never looked for beside another.
    """
    match = re.fullmatch(SERVING_UNIT_PATTERN, str(unit))
    if match is None:
        raise ValueError("The GPU service name is invalid.")
    return match.group("role")


def require_serving_unit(unit: Any) -> str:
    """``unit``, or a refusal: only the managed serving shape may be stopped."""
    unit_role(unit)
    return str(unit)


def balancer_name(name: str) -> str:
    """``vaelor-vllm-<name>-balancer``: the replicated deployment's balancer container."""
    return f"{_SERVING_PREFIX}{name}-{BALANCER_ROLE}"


def gate_config_path(unit: str) -> str:
    """The ``0600`` nginx config a worker's gate is launched with, by its unit.

    Keyed on the GATE unit and refused for any other, so the path can only
    ever name a gate's file under :data:`GATE_CONFIG_ROOT` - the runtime
    writes it before the unit and `stop_unit` removes it after, and neither
    can be asked for a server's or a Ray worker's.
    """
    if unit_role(unit) != GATE_ROLE:
        raise ValueError("Only a GPU gate unit has a config file.")
    return f"{GATE_CONFIG_ROOT}/{unit}.conf"


def record_mode(record: Mapping[str, Any]) -> str:
    """``units["mode"]`` of a stored record: ``distributed``, ``replicated`` or ``""``."""
    units = record.get("units") or {}
    return str(units.get("mode", "") or "") if isinstance(units, Mapping) else ""


def is_replicated(record: Mapping[str, Any]) -> bool:
    """Whether a stored record is the throughput intent's replicated shape."""
    return record_mode(record) == VERDICT_REPLICATED


def serving_units(record: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """``[(node_id, unit)]`` a stored record owns, in the order they were started.

    Mode-aware (VD-129), read off the record's ``name``, ``node_ids`` and
    ``units["mode"]`` and nothing else, so it answers for a ``deploying`` row
    exactly as for a healthy one:

    * ``distributed`` - the server on ``node_ids[0]`` (`gpu_pool_operations.deploy`
      writes the lead first, `_lead_first`, and starts it first) and a Ray
      worker on each node after it;
    * ``replicated`` - one server on EVERY node, the controller's first, and
      a gate beside the server on every node after it (the VD-129 amendment:
      a worker's replica is reachable only through its gate, so the gate is
      stopped wherever the replica is - removal, the abandoned pass, every
      leave - without a list);
    * no mode (a row written before the word existed) - the UNION of the two:
      the server on every node, and a gate and a Ray worker on every node
      after the first. A stop of a unit systemd never loaded is the runtime's
      no-op, so the union costs a read per unit and can orphan nothing.

    Empty for a record naming no nodes.
    """
    node_ids = [
        str(node_id) for node_id in (record.get("node_ids") or [])
        if str(node_id)
    ]
    if not node_ids:
        return []
    name = str(record.get("name", "") or "")
    mode = record_mode(record)
    units: List[Tuple[str, str]] = [(node_ids[0], unit_name(name, SERVER_ROLE))]
    if node_ids[0] != CONTROLLER_PLACEMENT_ID:
        # A worker lead's API is behind its keyed gate (ACC-162), whatever
        # the mode; a gate that was never started costs one no-op stop.
        units.append((node_ids[0], unit_name(name, GATE_ROLE)))
    for node_id in node_ids[1:]:
        if mode != VERDICT_DISTRIBUTED:
            units.append((node_id, unit_name(name, SERVER_ROLE)))
            units.append((node_id, unit_name(name, GATE_ROLE)))
        if mode != VERDICT_REPLICATED:
            units.append((node_id, unit_name(name, RAY_WORKER_ROLE)))
    return units


def remove_blockers(
    record: Mapping[str, Any], failures: Iterable[Mapping[str, Any]],
    departed: Iterable[str] = (),
) -> List[Dict[str, str]]:
    """The :func:`unstopped` entries `remove` must refuse to drop the row over.

    Every one of them for a ``distributed`` record: its units are one process
    group, and a Ray worker left on an unreachable node is half of a cluster
    the operator will otherwise never find again. For a ``replicated`` record
    only the CONTROLLER's (VD-129, VD-127 cleanup item 14): each replica is
    its own server, a worker that left the cluster or is unplugged cannot be
    probed or stopped and must not hold the record hostage - its gate is on
    the same machine and goes the same way - while a unit on this controller
    is always reachable from here, so a stop that failed here is a fault
    worth refusing on rather than a machine that went away. Keyed on the node
    id, so a gate is treated exactly as the replica beside it.
    """
    entries = [dict(entry) for entry in failures]
    # ``departed``: node ids that are neither this controller nor an enrolled
    # machine any more (`cluster_placement.departed_node_ids`) - a worker a
    # forced removal took away (owner decision 2026-09-28). Nothing can ever
    # stop a unit there again, and that removal already told the owner what
    # may still run on it, so it must not hold the record hostage. Derived
    # from the cluster store, never from the record's own display marker,
    # which a stale writer can overwrite (review R2).
    gone = {str(node_id) for node_id in departed}
    entries = [entry for entry in entries if entry.get("node_id") not in gone]
    if is_replicated(record):
        return controller_unstopped(entries)
    return entries


def lead_first(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The chosen nodes with the controller first, if it is one of them (D2).

    Order carries meaning: `gpu_pool_operations.deploy` writes ``node_ids`` in
    this order and starts ``nodes[0]`` as the server, and `serving_units`
    trusts that order - which is why the rule lives beside the derivation it
    guarantees. The fit engine orders by free memory, which is the right rule
    for CHOOSING nodes and the wrong one for choosing which publishes the API.
    This machine leads whenever it takes part: its API is fronted by Vaelor's
    own auth proxy (or the replicated deployment's balancer, VD-129), and its
    lead is the one the mode switch moves AI Chat onto. A worker lead is
    served too - its API on loopback behind its own keyed gate (ACC-162) - but
    only when the controller is not one of the chosen nodes.
    """
    controller = [
        node for node in nodes
        if str(node.get("id", "")) == CONTROLLER_PLACEMENT_ID
    ]
    if not controller:
        return list(nodes)
    return controller + [node for node in nodes if node not in controller]


def unstopped(node_id: str, node_name: str, unit: str, error: Any) -> Dict[str, str]:
    """One unit a stop could not stop - the shape a record and a note carry.

    Node AND unit, because those are the two facts an operator needs to clear
    it by hand; the error, because "could not be stopped" without the reason
    is the swallowed failure this exists to replace.
    """
    return {
        "node_id": str(node_id or ""), "node": str(node_name or ""),
        "unit": str(unit or ""), "error": str(error),
    }


def controller_unstopped(
    entries: Iterable[Mapping[str, Any]],
) -> List[Dict[str, str]]:
    """The :func:`unstopped` entries that sit on THIS controller's node.

    The one rule for the one exception to "always restore Mode A" (VD-127): a
    worker's unstopped unit holds a worker's GPU and llama.cpp may come back
    beside it, but a unit still resident on the controller shares the aperture
    llama.cpp would be relaunched into. The switch reads this to decide whether
    to restore the lease, and the watch reads it to write the note that says
    it did not - one predicate, so the two cannot disagree about which units
    kept AI Chat unassigned. Keyed on the node id the placement layer gives
    the controller, never on a node name.
    """
    # A split's firewall and slice left behind (`plane_left`) holds no GPU,
    # so it never keeps AI Chat unassigned (ACC-187 review 4).
    return [
        dict(entry) for entry in entries if entry.get("kind") != RAY_PLANE_KIND
        if str(entry.get("node_id", "")) == CONTROLLER_PLACEMENT_ID
    ]


def _where(entry: Mapping[str, Any]) -> str:
    """The machine an unstopped entry names, in words."""
    return str(entry.get("node") or entry.get("node_id") or "an unknown node")


def describe_unstopped(entries: Iterable[Mapping[str, Any]]) -> str:
    """``unit on node (error); ...`` - what is still to be stopped, by name."""
    parts = []
    for entry in entries:
        where = _where(entry)
        what = entry.get("unit") or "its units"
        parts.append(f"{what} on {where} ({entry.get('error', '')})")
    return "; ".join(parts)


def describe_left_behind(entries: Iterable[Mapping[str, Any]]) -> str:
    """Plain sentences for what a stop left behind: units, and split firewalls."""
    units, planes = [], []
    for entry in entries:
        (planes if entry.get("kind") == RAY_PLANE_KIND else units).append(entry)
    sentences = []
    if units:
        sentences.append("Could not stop {}.".format(describe_unstopped(units)))
    for entry in planes:
        sentences.append("Could not clear {} on {} ({}).".format(
            RAY_PLANE_LEFT, _where(entry), entry.get("error", "")))
    return " ".join(sentences)


def plane_left(node_id: str, node_name: str, error: Any) -> Dict[str, str]:
    """One machine whose split firewall and slice a stop could not clear."""
    return {**unstopped(node_id, node_name, RAY_PLANE_LEFT, error), "kind": RAY_PLANE_KIND}


# --- The model-pull oneshot, named per repo ---------------------------------


def repo_slug(model: str) -> str:
    """A DNS/unit-safe slug of a validated ``org/name`` repo.

    Used for the pull unit name, the container name and the progress file.
    Lower-cased and reduced to ``[a-z0-9-]`` so ``Qwen/Qwen3-8B`` becomes
    ``qwen-qwen3-8b`` - one value all three derive from, so they cannot drift.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", str(model).lower()).strip("-")
    return slug[:64] or "model"


def pull_container_name(model: str) -> str:
    return f"{_PULL_PREFIX}{repo_slug(model)}"


def pull_unit_name(model: str) -> str:
    return pull_container_name(model) + ".service"


def require_pull_unit(unit: Any) -> str:
    """``unit``, or a refusal: only the managed pull shape may be polled or stopped."""
    if not re.fullmatch(PULL_UNIT_PATTERN, str(unit)):
        raise ValueError("The model pull service name is invalid.")
    return str(unit)


# --- The unit bodies the tier writes, beside the names they carry -----------


#: A split's Ray unit gives up after this many failed starts in this window and
#: stays ``failed`` (systemd's ``start-limit-hit``), rather than retrying a
#: permanent failure - Docker on the wrong cgroup driver - every few seconds
#: for ever (ACC-187 review 3).
RAY_START_LIMIT_SECONDS = 600
RAY_START_LIMIT_BURST = 5

#: What a stop names in place of a unit when a split's firewall and slice on
#: a machine could not be cleared (ACC-187 review 2/3), and the marker that
#: tells such an entry apart from a unit that would not stop.
RAY_PLANE_LEFT = "the split's firewall and slice"
RAY_PLANE_KIND = "ray-plane"


def unit_text(
    *, description: str, container_name: str, exec_start: str,
    unlimited_restarts: bool = False, start_pre: str = "", ray_slice: str = "",
    slice_checks: Tuple[str, str] = ("", ""),
) -> str:
    """Render the container unit that supervises one ``docker run``.

    Housed beside the names (VD-129) because a unit body is the other thing
    this tier writes into a node's systemd, and the runtime that emits the
    ``ExecStart`` was at its line ceiling.

    ``unlimited_restarts`` adds ``StartLimitIntervalSec=0`` (VD-129): a
    replica is one of N behind a balancer, so systemd must keep relaunching
    it rather than give up after its start-rate limit.

    **These units run as ROOT** - no ``User=``/``Group=`` and none of the
    ``NoNewPrivileges``/``ProtectSystem=strict``/``PrivateTmp`` hardening the
    host-process units carried. Deliberate, not an omission: ``docker run``
    must reach the docker socket, and that hardening breaks it. **The
    CONTAINER is the sandbox and docker is the privilege boundary** - how
    Vaelor's root hardware bridge launches the single-node GPU container
    (`gpu_rocmfpx_service`).

    The container runs in the FOREGROUND (no ``-d``) so systemd supervises
    the real process and ``Restart=on-failure`` means what it says; a
    tolerated ``ExecStartPre=-docker rm -f`` clears a stale container name
    left by a crash, and ``ExecStop`` drains the model off the GPU before
    docker kills it. Assembled line by line rather than from a multi-line
    template, so no duplicated blob is shared with another unit author.
    """
    lines = [
        "[Unit]",
        f"Description={description}",
        UNIT_AFTER,
        "Wants=network-online.target",
        "Requires=docker.service",
        # A split's Ray unit is bound to its slice (ACC-187): the slice is
        # started first, and a slice that stops stops the unit. BindsTo= only
        # carries a stop - the next start reloads the fence (ExecStartPre)
        # against the slice's current cgroup, and re-checks it.
        *([f"BindsTo={ray_slice}", f"After={ray_slice}",
           f"StartLimitIntervalSec={RAY_START_LIMIT_SECONDS}",
           f"StartLimitBurst={RAY_START_LIMIT_BURST}"] if ray_slice else []),
        *(["StartLimitIntervalSec=0"] if unlimited_restarts else []),
        "",
        "[Service]",
        "Type=simple",
        f"ExecStartPre=-{DOCKER} rm -f {container_name}",
        # A split's Ray unit loads its fence first, and does not start if it
        # cannot (ACC-163): after a reboot Ray never comes up unfenced.
        *([f"ExecStartPre={start_pre}"] if start_pre else []),
        # A split's Ray unit re-checks at EVERY start that Docker uses the
        # systemd driver, and after it that the container is in its slice
        # (ACC-187 review 2); a failed check fails the unit, the fence stays.
        *([f"ExecStartPre={slice_checks[0]}"] if slice_checks[0] else []),
        f"ExecStart={exec_start}",
        *([f"ExecStartPost={slice_checks[1]}"] if slice_checks[1] else []),
        f"ExecStop={DOCKER} stop --time {STOP_GRACE_SECONDS} {container_name}",
        # A split's Ray container goes even when its START failed - a failed
        # slice check, say - when systemd runs no ExecStop and only signals the
        # docker client, which `ray start --block` may outlive (review 3).
        *([f"ExecStopPost=-{DOCKER} rm -f {container_name}"] if ray_slice else []),
        "Restart=on-failure",
        "RestartSec=5",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ]
    return "\n".join(lines)


def pull_unit_text(
    *, description: str, container_name: str, exec_start: str,
) -> str:
    """Render the oneshot pull unit.

    Built line by line for the same reason `unit_text` is, and root for the
    same reason: the payload is a ``docker run``, which needs the docker
    socket. ``Type=oneshot`` - the pull runs once and exits. The tolerated
    ``ExecStartPre=-… rm -f`` clears a container left by a dockerd restart or
    an OOM kill, which would otherwise block every future pull of that repo.
    """
    lines = [
        "[Unit]",
        f"Description={description}",
        UNIT_AFTER,
        "Wants=network-online.target",
        "Requires=docker.service",
        "",
        "[Service]",
        "Type=oneshot",
        f"ExecStartPre=-{DOCKER} rm -f {container_name}",
        f"ExecStart={exec_start}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ]
    return "\n".join(lines)
