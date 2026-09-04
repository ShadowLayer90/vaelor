"""What a model server actually got, read after it answers.

Deployment sequencing — write the file, validate it, pull the image, start the
container — is a different job from asking what came up, and this module is the
second one. Both readings it takes are of the same shape and were found the
same way: llama.cpp starts, ``/health`` answers 200, and the thing that went
wrong is visible only in a number nobody read.

* **The accelerator.** With no accelerator library directory on
  ``LD_LIBRARY_PATH`` the HIP backend cannot resolve, llama.cpp loads the CPU
  backend, and the server serves at 198 tok/s prefill instead of 1835 with no
  error anywhere. The evidence is how much accelerator memory the model is
  holding.
* **The context window.** A slot context above the model's training context is
  *capped*, not refused, and the overshoot is paid in resident bytes no request
  can use. It can also go the other way: with no slot count stated, the runtime
  gives *each* of its slots the whole requested window, and the only symptom is
  an out-of-memory kill minutes later. Both are read here, and both are
  announced — this module used to ask whether the window was capped, which is
  half the question.

**The baseline is why this module owns the ordering rather than the executor.**
The accelerator reading is a delta, and a delta is only worth what its baseline
is worth. Read the baseline while the model being replaced is still resident
and the new model appears to hold nothing at all — which is a *redeploy*
reporting a broken GPU on a machine whose GPU is fine, the module's own purpose
inverted. So the baseline is taken with the previous container stopped, and
:func:`~vaelor.accelerator_runtime.verify_accelerator_in_use` refuses to call a
zero delta a fallback while the adapter is visibly holding gigabytes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .accelerator_runtime import (
    ANNOUNCED_CONTEXT_STATES,
    reported_gpu_memory_bytes,
    verify_accelerator_in_use,
    verify_context_from_server,
)


def accelerator_baseline(
    accelerators: Optional[Sequence[Mapping[str, Any]]],
) -> Optional[int]:
    """What the accelerator holds with no managed model of ours resident.

    ``None`` is preserved rather than turned into zero: an adapter that reports
    nothing gives no baseline, and a baseline of zero would claim every byte the
    next reading sees.
    """
    return reported_gpu_memory_bytes(accelerators)


#: Established degradations, which are the only ones announced. A reading that
#: could not be taken says nothing, because a banner for an unanswered question
#: is the same defect as a banner for a failure that is not there.
#:
#: ``not-offloaded`` belongs here for the same reason ``cpu-fallback`` does. An
#: accelerator that was requested and then not granted costs the identical 9x,
#: and it was previously invisible twice over: the plan's own fallback reason
#: sat in a different field, and the acceleration block said the model *"runs
#: on the CPU by configuration, so no accelerator was expected"*.
ANNOUNCED_ACCELERATION_STATES = ("cpu-fallback", "not-offloaded")


def verify_started_model(
    *,
    backend: str,
    accelerators: Optional[Sequence[Mapping[str, Any]]],
    baseline_bytes: Optional[int],
    base_url: str,
    requested_context: Any,
    requested_slots: Any = 1,
    announce: Optional[Callable[[str], None]] = None,
    requested_backend: Optional[str] = None,
    gpu_layers_requested: Any = 0,
    declined_reason: str = "",
    props_reader: Optional[Callable[[str, float], Optional[Any]]] = None,
) -> Dict[str, Any]:
    """Read back what the server that just started is actually running.

    ``backend`` is what was launched; ``requested_backend`` and
    ``gpu_layers_requested`` are what was asked for. The gap between them is a
    reading in its own right — the deploy that prompted this passed the
    *granted* backend alone, so a request for ROCm that was refused before the
    container started looked identical to a CPU appliance.
    """
    acceleration = verify_accelerator_in_use(
        backend,
        accelerators,
        baseline_bytes=baseline_bytes,
        requested_backend=requested_backend,
        gpu_layers_requested=gpu_layers_requested,
        declined_reason=declined_reason,
    )
    context_built = verify_context_from_server(
        base_url,
        requested_context,
        requested_slots=requested_slots,
        # Passed through for the same reason the reader below it is injectable:
        # a check that can only be exercised against a live socket is a check
        # whose failing arm nobody ever runs.
        fetch=props_reader,
    )
    if announce is not None:
        if acceleration["state"] in ANNOUNCED_ACCELERATION_STATES:
            announce(acceleration["detail"])
        # Every degraded context state, not `capped` alone. Asking only whether
        # the window was capped is what let a window multiplied by four pass
        # this line in silence on its way to an OOM kill.
        if context_built["state"] in ANNOUNCED_CONTEXT_STATES:
            announce(context_built["detail"])
    return {"acceleration": acceleration, "context_built": context_built}
