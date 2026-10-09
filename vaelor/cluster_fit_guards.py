"""What a ``/cluster/fit`` preview refuses exactly as the deploy would.

The fit engine (`cluster_gpu_sizing`) answers one question: does this model fit
these GPUs. Two other things decide whether a cluster deploy can start at all,
and the deploy refused them only after the owner pressed Serve, while the
preview had already said "fits" (live test of build f768b46):

* **The GPU serving mode switch is held** (ACC-188). One cluster deployment
  holds the switch at a time (`gpu_cluster_mode.MODE_B_HELD`); a second one is
  refused by `enter`. The preview reads the same mode record and says so with
  the same sentence.
* **A split's chosen cluster link cannot be used** (ACC-189). Either link
  carries a split since the socket-owner fence (owner 2026-10-01, ACC-187),
  so neither the Ethernet nor a shared card is refused any more; what the
  deploy still refuses is a chosen link that is gone or has no address
  (`cluster_link.link_for_split`), and the preview says the same. A link
  setting that cannot be read at all is said as such, never taken as "none".

The engine's verdict is never rewritten: a model that fits still fits. The
refusal rides beside it as ``refusal`` (``code`` and the plain ``message``),
and a surface must not offer Serve while one is present (VD-168).

It also widens the fit's live refresh (ACC-190): with no machine selected the
preview sized the whole fleet from each worker's join-time snapshot, so a worker
whose GPU had been freed read as holding the model it no longer held.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional

from . import cluster_link
from .cluster_link_recommendation import split_link_for
from .cluster_gpu_sizing import VERDICT_DISTRIBUTED, VERDICT_REPLICATED
from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .gpu_cluster_mode import MODE_B_HELD
from .gpu_cluster_mode_state import ClusterModeStore
from .gpu_serving_target import MODE_CLUSTER

LOGGER = logging.getLogger(__name__)

#: The two refusal codes the preview can carry, beside the engine's verdict.
MODE_SWITCH_HELD = "mode_switch_held"
SPLIT_LINK_REFUSED = "split_link_refused"

#: When the chosen cluster link could not be read at all.
LINK_UNREAD = (
    "Vaelor could not read the cluster link setting, so it cannot say whether "
    "this split can start. Check Cluster > Setup and try again."
)

#: What the form says before Serve when THIS deploy would take the GPU serving
#: mode switch (W4-D4): this controller takes part and the switch is free. One
#: home for the words; the form shows them only when they ride on the fit.
MODE_SWITCH_HEADING = "This turns GPU clustering on"
MODE_SWITCH_NOTICE = (
    "Enabling GPU clustering on this controller stops its AI Chat model, moves "
    "AI Chat and the LLM Server to the cluster, and reverts when the cluster is "
    "removed."
)

#: What the held-switch sentence calls a deployment the form has not named yet.
_UNNAMED = "this deployment"


def refresh_scope(store: Any, node_ids: Optional[List[str]]) -> Optional[List[str]]:
    """The workers the fit re-probes: the selected ones, else every enrolled one.

    ``capacity_ledger`` re-probes only the ids it is given (and only when their
    snapshot is stale), so "nothing selected" used to mean "nothing refreshed".
    An unreadable store keeps the old scope; the refresh is best effort either way.
    """
    if node_ids:
        return node_ids
    try:
        enrolled = [str(node["id"]) for node in store.list_nodes()]
    except (AttributeError, KeyError, TypeError, OSError) as error:
        LOGGER.info("Fit refresh scope: the enrolled machines could not be listed: %s", error)
        return node_ids
    return enrolled or node_ids


def mode_switch_refusal(mode_state: Any, deployment: Any) -> Optional[Dict[str, str]]:
    """The deploy's held-switch refusal, or ``None`` when the switch is free.

    The deployment that holds the switch may deploy over itself: that is the
    one name `enter` would not be asked about.
    """
    if getattr(mode_state, "mode", "") != MODE_CLUSTER:
        return None
    holder = str(getattr(mode_state, "deployment_name", "") or "")
    wanted = str(deployment or "").strip()
    if not holder or holder == wanted:
        return None
    return {"code": MODE_SWITCH_HELD, "message": MODE_B_HELD.format(holder, wanted or _UNNAMED)}


def mode_switch_notice(mode_state: Any, deployment: Any, node_ids: Any) -> Optional[Dict[str, str]]:
    """The "turns GPU clustering on" notice when this deploy would take the switch.

    Only when this controller is selected and the switch is free: held by
    another deployment is a refusal (said beside the verdict), held by this one
    is a redeploy over itself, and a cluster of workers alone never touches it.
    A record that could not be read is not proof the switch is held, so the
    warning is kept.
    """
    if CONTROLLER_PLACEMENT_ID not in [str(node_id) for node_id in (node_ids or [])]:
        return None
    if mode_state is not None and getattr(mode_state, "mode", "") == MODE_CLUSTER:
        return None
    return {"heading": MODE_SWITCH_HEADING, "message": MODE_SWITCH_NOTICE}


def fit_pins(body: Mapping[str, Any]) -> tuple:
    """``(per-node pins, one pin for every machine)`` from a fit body.

    The deploy's own fields (``interfaces``, ``interface``); only whether a
    pin is present matters to the preview - the deploy validates the names.
    """
    node_pins = body.get("interfaces")
    return (node_pins if isinstance(node_pins, dict) else {},
            str(body.get("interface", "") or "").strip())


def apply_fit_guards(
    decision: Dict[str, Any], store: Any, *, deployment: Any = None,
    read_mode: Optional[Callable[[], Any]] = None,
    read_link: Optional[Callable[[Any], Optional[Mapping[str, Any]]]] = None,
    split_link: Optional[str] = None, pins: Any = ({}, ""),
    node_ids: Any = None,
) -> Dict[str, Any]:
    """``decision`` with the deploy's own refusal beside the verdict, if any.

    ``split_link`` is the owner's link choice for a split (`parse_split_link`):
    Ethernet never reads the chosen link, the cluster link is refused when
    none is chosen, and nothing sent reads the Setup choice as before.
    ``pins`` is ``(per-node pins, one pin for every machine)`` as the deploy
    reads them: beside the cluster link they are refused in the deploy's words.
    ``node_ids`` are the selected machines: with this controller among them and
    the switch free, ``mode_switch`` says the deploy turns GPU clustering on.
    """
    verdict = decision.get("verdict")
    if verdict not in (VERDICT_DISTRIBUTED, VERDICT_REPLICATED) or decision.get("refusal"):
        return decision
    try:
        mode_state = (read_mode or (lambda: ClusterModeStore().read()))()
    except (OSError, ValueError) as error:
        LOGGER.info("Fit preview: the mode switch record could not be read: %s", error)
        mode_state = None
    refusal = mode_switch_refusal(mode_state, deployment)
    if refusal is None and verdict == VERDICT_DISTRIBUTED:
        try:
            split_link_for(split_link, lambda: (read_link or cluster_link.link_for_split)(store), *pins)
        except ValueError as error:
            refusal = {"code": SPLIT_LINK_REFUSED, "message": str(error)}
        except (AttributeError, KeyError, TypeError, OSError) as error:
            # Not read is not "no link": the preview says it could not tell.
            LOGGER.info("Fit preview: the cluster link could not be read: %s", error)
            refusal = {"code": SPLIT_LINK_REFUSED, "message": LINK_UNREAD}
    if refusal is not None:
        return {**decision, "refusal": refusal}
    notice = mode_switch_notice(mode_state, deployment, node_ids)
    return {**decision, "mode_switch": notice} if notice else decision
