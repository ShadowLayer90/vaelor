"""Routes for the two machine settings: the GPU memory pool and the cluster link.

VD-161 and VD-162. One read, one write:

* ``GET /host-settings`` - what each machine's GPU memory pool is and may be
  set to, this controller's own network links, the chosen cluster link, and
  whether each enrolled worker holds an address on that link's network. A
  read: it opens this machine's own files and the stored worker inventory, and
  reaches no other machine.
* ``POST /host-settings/cluster-link`` - record the owner's choice of link, or
  clear it. Administrator-only, confirmed, audited. It changes no machine: a
  split deployment started afterwards binds the chosen link.

**Changing a pool is a job, not a route here.** Setting or removing a machine's
GPU memory pool, and restarting a worker, are privileged changes to a host, so
they go through ``POST /jobs`` like every other one (``host.gpu-memory.apply``,
``cluster.node.gpu-memory``, ``cluster.node.reboot``), each with its own typed
confirmation (`gpu_memory_pool_nodes`). Restarting this controller is the
existing power action.

**A worker's card is a stored reading, and says how old it is.** The controller
is read live on every request. A worker is read over SSH only at enrolment, at
a Recheck and by its own jobs, so its card carries ``checked_at`` - when the
reading was taken, by a probe that answered or a pool job (`_read_at`) - and,
past :data:`STALE_AFTER_SECONDS` or after a check that failed, a
``stale_reason``. The console then asks for a Recheck before a change is
reviewed, instead of reviewing one against a reading nobody can vouch for
(review S7).

**Only what a card draws is served** (review S4): `gpu_memory_pool.card_view`
of the status, never a machine's files.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from flask import g, request

from . import cluster_link, cluster_link_shared_card, gpu_memory_pool, gpu_node_facts
from .api_common import ApiContext, payload as _payload
from .api_machine import platform_driver
from .cluster_capacity import gpu_facts_from_accelerators
from .cluster_gpu_sizing import VERDICT_DISTRIBUTED
from .cluster_placement import CONTROLLER_PLACEMENT_ID

#: How old a worker's stored reading may be before a change asks for a Recheck
#: first. An upper bound on "recent enough to review against", chosen, not
#: measured: a job re-reads the machine before it writes either way.
STALE_AFTER_SECONDS = 900

#: What the link panel says about a split model that is already deployed: the
#: choice is read when a split is DEPLOYED, and a Load re-serves on the binding
#: the record wrote, so an existing one does not move.
_RUNNING_SPLIT_NOTE = (
    "{} is already deployed and keeps the link it was deployed on. Remove it "
    "and deploy it again to move it to a different link."
)


def _worker_link(node: Dict[str, Any], chosen: Any) -> Dict[str, Any]:
    """Whether one worker holds an address on the chosen link's network."""
    record = {
        "node_id": node["id"], "name": node.get("name") or node["id"],
        "on_link": None, "address": "", "interface": "", "reason": "",
    }
    if not chosen or not chosen.get("valid"):
        return record
    inventory = node.get("inventory") or {}
    if "links" not in inventory:
        record["reason"] = (
            "Recheck this machine to see whether it is on the cluster link; "
            "its network links were last read before this setting existed."
        )
        return record
    links = inventory["links"]
    if not isinstance(links, list):
        # Read, and the read failed: not the same as "has no address".
        record["reason"] = (
            "This machine's network addresses could not be read at its last "
            "check, so whether it is on the cluster link is not known. "
            "Recheck it."
        )
        return record
    binding = cluster_link.binding_in_network(links, chosen["network"])
    if binding is None:
        record["on_link"] = False
        record["reason"] = (
            "This machine had no IPv4 address on the cluster link's network at "
            "its last check. A split model cannot include it until it has one."
        )
        return record
    record.update({"on_link": True, "address": binding[0], "interface": binding[1]})
    return record


def _stamp(value: Any) -> Optional[int]:
    return int(value) if isinstance(value, (int, float)) and value > 0 else None


def _read_at(node: Dict[str, Any]) -> Optional[int]:
    """When a worker's pool reading was TAKEN: its last successful probe, or its last pool job.

    The probe's reading is dated by the node's ``last_seen``, which only a
    probe that answered stamps - enrolment (`ClusterStore.add_node`) and every
    Recheck or capacity re-read that reached the machine (`_reprobe_node`).
    It is not dated by ``inventory.checked_at``: enrolment never wrote that
    stamp, so a worker read seconds earlier at enrolment showed "not read
    yet" and asked for a Recheck (v1.5 cold install, 2026-10-09); and a check
    that FAILED writes it too (ACC-094), which made the old reading it kept
    look freshly read.
    """
    inventory = node.get("inventory") or {}
    stamps = [
        stamp for stamp in (
            _stamp(node.get("last_seen")), _stamp(inventory.get("gpu_memory_pool_checked_at")),
        ) if stamp is not None
    ]
    return max(stamps) if stamps else None


#: How a worker's pool card asks for a Recheck before a change is reviewed.
_RECHECK_BEFORE_CHANGE = "Recheck it before changing its GPU memory pool."


def _stale_reason(node: Dict[str, Any], name: str, read_at: Optional[int], now: float) -> str:
    """Why a supported worker's reading may not be reviewed against, or ``""``.

    A check after the reading that could not reach the machine says so, in its
    own recorded words, rather than letting the older reading pass for current.
    """
    inventory = node.get("inventory") or {}
    attempted = _stamp(inventory.get("checked_at"))
    if inventory.get("reachable") is False and attempted is not None and (
        read_at is None or attempted > read_at
    ):
        why = str(inventory.get("unreachable_reason") or "").strip()
        return "This reading of {} is older than its last check, which failed{} {}".format(
            name, ": " + why if why else ".", _RECHECK_BEFORE_CHANGE,
        )
    if read_at is None or now - read_at > STALE_AFTER_SECONDS:
        return "This reading of {} is not recent. {}".format(name, _RECHECK_BEFORE_CHANGE)
    return ""


def _enrolled_interface(address: str) -> str:
    """The interface carrying ``address`` on this machine (`local_cluster_interface`), or ``""``."""
    return str(gpu_node_facts.local_cluster_interface(address).get("name") or "")


def register_host_settings_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    require_auth = context.require_auth

    def _probe() -> Dict[str, Any]:
        """This machine's own readers; a test injects ``host_settings_probe``."""
        injected = callbacks.get("host_settings_probe") or {}
        return {
            "pool_facts": injected.get("pool_facts", gpu_memory_pool.read_local_facts),
            "links": injected.get("links", cluster_link.discover_links),
            "now": injected.get("now", time.time),
            # B8: the interface carrying the controller's advertised address,
            # read live the way cluster_manager.summary reads it.
            "enrolled_interface": injected.get("enrolled_interface", _enrolled_interface),
        }

    def _controller_gpu() -> Dict[str, Any]:
        try:
            accelerators = platform_driver(callbacks).accelerators()
        except (AttributeError, NotImplementedError, OSError):
            accelerators = []
        return gpu_facts_from_accelerators(accelerators)

    def _store():
        manager = callbacks.get("cluster_manager")
        return getattr(manager, "store", None)

    def _worker_machine(node: Dict[str, Any], now: float) -> Dict[str, Any]:
        inventory = node.get("inventory") or {}
        stored = inventory.get("gpu_memory_pool")
        status = (
            stored if isinstance(stored, dict) and "supported" in stored
            # Never read (enrolled before the setting), or the read returned
            # nothing: say so, from the same rule that reads a live machine.
            else gpu_memory_pool.pool_status(None, inventory.get("gpu"))
        )
        read_at = _read_at(node)
        name = node.get("name") or node["id"]
        return {
            "node_id": node["id"],
            "name": name,
            "role": "worker",
            "checked_at": read_at,
            # A machine with no setting has nothing to go stale.
            "stale_reason": (
                _stale_reason(node, name, read_at, now) if status.get("supported") else ""
            ),
            "pool": gpu_memory_pool.card_view(status),
        }

    def _surface() -> Dict[str, Any]:
        probe = _probe()
        store = _store()
        now = float(probe["now"]())
        nodes: List[Dict[str, Any]] = store.list_nodes() if store is not None else []
        machines = [{
            "node_id": CONTROLLER_PLACEMENT_ID,
            "name": gpu_memory_pool.CONTROLLER_NAME,
            "role": "controller",
            # Read live, on this request: never stale.
            "checked_at": None,
            "stale_reason": "",
            "pool": gpu_memory_pool.card_view(
                gpu_memory_pool.pool_status(probe["pool_facts"](), _controller_gpu())
            ),
        }]
        machines.extend(_worker_machine(node, now) for node in nodes)
        links = probe["links"]()
        if store is None:
            link_surface: Dict[str, Any] = {
                "available": False,
                "reason": "Fleet management is unavailable, so no cluster link can be chosen.",
                "links": links, "chosen": None, "machines": [], "notes": [],
            }
        else:
            # B8: each link says whether it is this controller's shared LAN
            # card, by prepare_split's own FULL rule - the advertised address,
            # or the interface that carries it (review round 1: a LAN card
            # holding a second address is fenced as shared too).
            advertised = str((store.controller() or {}).get("advertise_address", "") or "")
            links = cluster_link_shared_card.with_shared_notes(
                links, advertise_address=advertised,
                enrolled_interface=probe["enrolled_interface"](advertised) if advertised else "",
            )
            chosen = cluster_link.chosen_link(cluster_link.ClusterLinkSetting(store), links)
            # Only a split (capacity) deployment binds a link; a replicated
            # one, Mode A and an empty cluster have nothing to say here.
            notes = [
                _RUNNING_SPLIT_NOTE.format(deployment["name"])
                for deployment in store.list_pooled_deployments()
                if (deployment.get("units") or {}).get("mode") == VERDICT_DISTRIBUTED
            ]
            link_surface = {
                "available": True, "reason": "", "links": links, "chosen": chosen,
                "machines": [_worker_link(node, chosen) for node in nodes],
                "notes": notes,
            }
        return {"gpu_memory_pool": {"machines": machines}, "cluster_link": link_surface}

    @blueprint.get("/host-settings")
    @require_auth("operator")
    def host_settings():
        return _payload(_surface())

    @blueprint.post("/host-settings/cluster-link")
    @require_auth("administrator", csrf=True)
    def host_settings_cluster_link():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            body = {}
        if body.get("confirm") != cluster_link.SET_CONFIRMATION:
            return _payload(
                error={
                    "code": "cluster_link_confirmation_required",
                    "message": "Review and confirm the cluster link change first.",
                },
                status=400,
            )
        store = _store()
        if store is None:
            return _payload(
                error={
                    "code": "cluster_unavailable",
                    "message": "Fleet management is not running, so the cluster link cannot be stored.",
                },
                status=503,
            )
        setting = cluster_link.ClusterLinkSetting(store)
        name = str(body.get("interface", "") or "").strip()
        try:
            if name:
                name = setting.choose(name, _probe()["links"]())["name"]
            else:
                setting.clear()
        except ValueError as error:
            security.audit(
                g.auth_session.username, "cluster.link.set", "failure",
                target=name[:32] or "none", remote_addr=request.remote_addr or "",
            )
            return _payload(
                error={"code": "cluster_link_rejected", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "cluster.link.set", "success",
            target=name or "none", remote_addr=request.remote_addr or "",
        )
        return _payload(_surface())
