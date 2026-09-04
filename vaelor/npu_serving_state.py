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

from typing import Any, Dict, List, Optional

from .provider_runtime import managed_local_connection


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
    served = _served_model(callbacks)
    if served:
        # Only the primary NPU is the one flm-real binds to; a second device
        # stays honestly idle rather than inheriting the first one's model.
        records[0]["serving_assistant"] = True
        records[0]["serving_model"] = served
    return records


def _served_model(callbacks: Dict[str, Any]) -> Optional[str]:
    """The served model tag when the Assistant is deployed on the NPU, else ``None``.

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
    return pinned
