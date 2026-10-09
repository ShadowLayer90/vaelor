"""Whether the Assistant is actually being served on the neural processor.

``platforms/accelerators.discover_npus`` is pure hardware discovery. It can see
that a neural processor exists and whether a userspace runtime is installed, but
it has no knowledge of what Vaelor has *deployed* onto the device -- so on a Z2
serving the Assistant on the NPU via flm-real, the System screen's "Used by"
still read "...no Vaelor inference backend targets this device". That is true of
the silicon and false of the machine: the Assistant was on it the whole time.

This layer combines the hardware record with the running deployment state before
the ``/api/v2/system/machine`` payload carries it to the ComputePanel, so the
operator reads what is actually using the device. It lives outside
``accelerators.py`` deliberately: that module is at the 1,000-line ceiling and is
kept to hardware discovery, and the signal here -- the active ``deployment-agent``
credential and the flm serving preconditions -- is a control-plane concern the
hardware layer must not learn.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from . import model_reachability
from .provider_runtime import managed_local_connection

#: What the record says when the Assistant is deployed on this NPU but its
#: model server is not answering. Plain words: the owner reads this sentence
#: under "Used by", followed by the probe's own reason when it has one.
NOT_ANSWERING_REASON = (
    "The Assistant's model server on this neural processor is not answering."
)

#: A raw endpoint address inside a probe's reason. Stripped from owner copy:
#: a loopback URL names nothing the owner can act on.
_ADDRESS = re.compile(r"https?://[^\s]*[^\s.,;:)]")


def _down_reason(probe: Dict[str, Any]) -> str:
    detail = str(
        probe.get("model_availability_reason") or probe.get("detail") or ""
    ).strip()
    detail = _ADDRESS.sub("its local address", detail)
    return "{} {}".format(NOT_ANSWERING_REASON, detail).strip()


def annotate_npu_serving(
    neural_accelerators: List[Dict[str, Any]],
    callbacks: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return the records with the first NPU marked when the Assistant serves it.

    The hardware-discovery fields are never mutated: the serving state is *added*
    as ``serving_assistant``/``serving_model``, so a reader still has the untouched
    ``runtime_detected``/``reason`` beneath it and the panel decides which to show.
    A machine with no neural accelerator -- every Raspberry Pi -- gets its list
    back unchanged and never reaches the credential broker, so its behaviour is
    exactly what it was.
    """
    records = [dict(record) for record in neural_accelerators or []]
    if not records:
        return records
    deployed = _deployed_lease(callbacks)
    if deployed is None:
        return records
    pinned, lease = deployed
    # Deployed is not serving (ACC-100). The System screen said "Serving the
    # Assistant" off the lease and the files alone while flm-real was down and
    # the Assistant tab, which probes, said unreachable. The same probe the
    # Assistant's status uses decides it here, so the two screens agree.
    try:
        probe = model_reachability.probe_connection(lease)
    except (AttributeError, OSError, TypeError, ValueError) as error:
        probe = {"reachable": False, "detail": str(error)}
    # Only the primary NPU is the one flm-real binds to; a second device
    # stays honestly idle rather than inheriting the first one's model.
    if probe.get("reachable") and probe.get("offering_models") is not False:
        records[0]["serving_assistant"] = True
        records[0]["serving_model"] = pinned
    else:
        records[0]["assistant_down"] = True
        records[0]["assistant_model"] = pinned
        records[0]["assistant_down_reason"] = _down_reason(probe)
    return records


def _deployed_lease(callbacks: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """``(tag, lease)`` when the Assistant is deployed on the NPU, else ``None``.

    Deployed, not serving: whether the server answers is the caller's probe.

    The bridge-free signal is the active ``deployment-agent`` credential -- the
    very lease the executor's restart-on-boot reconcile keys on (VD-001): a
    managed-local loopback lease whose pinned model equals the NPU tier's
    ``flm_tag``. A box that never deployed the Assistant has no such lease and
    reads honestly as idle; an Assistant on a hosted provider, or on the GPU via
    llama.cpp (which pins no model on its lease), is not this and is not claimed.
    The flm preconditions (binary, sysfs device, installed model) are checked
    last, so a lease left behind after any of the three went away reads as idle
    rather than as a live server -- the configured-but-not-servable case must not
    over-claim, exactly as the Assistant status distinguishes configured from
    reachable.

    Fails closed: ``/system/machine`` must never 500 because the credential
    broker is busy or down, so an unanswerable question reads as idle, never as
    serving.
    """
    broker = callbacks.get("credential_broker")
    if broker is None:
        return None
    from .model_connection import resolve_model_connection

    try:
        lease = resolve_model_connection(broker)
    except Exception:
        return None
    if not lease or not managed_local_connection(lease):
        return None
    pinned = str(lease.get("model") or "").strip()
    if not pinned:
        return None

    from .flm_service import discover_npu_serving
    from .inference_tuning import npu_tier_plan

    if pinned != str(npu_tier_plan().get("flm_tag") or ""):
        return None
    if not discover_npu_serving(pinned).get("available"):
        return None
    return pinned, dict(lease)
