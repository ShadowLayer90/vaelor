"""Is a model resident in memory on this appliance RIGHT NOW - measured, per engine.

The memory down-cycle (VD-052, VD-101) may sweep only while no inference
engine holds a model, and until today nothing could tell it: the tool status
it read carried fields no engine filled (ACC-109), and the Pi's llama.cpp
reports ``loaded_models: []`` from ``/v1/models`` even while serving. Owner
decision (2026-09-28): the down-cycle treats UNKNOWN residency as busy until a
measured reading exists. :func:`appliance_residency` is that reading - the one
owner of the question "is a model resident now" - and each engine answers from
something that was measured, with the basis stated beside it:

* **NPU (FastFlowLM)** - ``flm serve`` loads its model at start and holds it
  while it runs, so the hardware bridge's ``flm_status`` (the flm-real process
  it supervises) is the reading;
* **GPU llama.cpp (Mode A)** - the bridge's serving container is launched
  without an idle sleep, so ``gpu_status`` (``docker inspect`` of that
  container) is the reading;
* **cluster vLLM (Mode B)** - the deployment row the 30 s mode watch keeps
  measured (startup verified, replicas probed, a split's parts checked):
  ``healthy``/``deploying`` hold the GPU, ``unloaded`` (scaled to zero) does not;
* **llama.cpp in a compose project (the Pi's Assistant, a GPU AI Chat on a
  stock GGUF)** - these sleep after ``--sleep-idle-seconds``; the engine's own
  ``GET /props`` reports ``is_sleeping``. A build that does not report it reads
  ``unknown``, never "not resident".

Anything that cannot be read is :data:`UNKNOWN`, and :attr:`Residency.busy` is
true for resident OR unknown - including a broker that cannot say which lease
is active, and a local-model lease that is not on loopback (a model this
appliance cannot measure). Only the hosted provider is positively elsewhere.
Runs in the control plane, where the down-cycle scheduler reads it
(`ControlPlaneRuntime._inference_tier_loaded`, VD-134); every source is a read
the control plane already makes.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence
from urllib.parse import urlsplit

from .credential_broker_client import BROKER_NO_ANSWER, BROKER_UNAVAILABLE, CredentialError
from .hosted_providers import OFF_MACHINE_KINDS

#: The three answers, and the engines that give them.
RESIDENT = "resident"
NOT_RESIDENT = "not-resident"
UNKNOWN = "unknown"

ENGINE_NPU = "npu-fastflowlm"
ENGINE_GPU = "gpu-llama.cpp"
ENGINE_CLUSTER = "cluster-vllm"
ENGINE_LLAMA = "llama.cpp"

#: The lease purposes whose loopback endpoints are this appliance's own models.
LOCAL_MODEL_PURPOSES = ("deployment-agent", "ai-chat")

#: The basis a reading about one lease states.
LEASE_BASIS = "the {} lease"

#: How long one ``/props`` read may take: a local server answers at once.
PROPS_TIMEOUT_SECONDS = 2.0

#: Cluster record states that hold the controller's GPU, and one that does not.
_HOLDING = ("healthy", "deploying")
_RELEASED = "unloaded"


class Reading(NamedTuple):
    """One engine's answer: which engine, resident or not, and how we know."""

    engine: str
    state: str
    basis: str
    detail: str = ""

    def as_dict(self) -> Dict[str, str]:
        return self._asdict()


class Residency(NamedTuple):
    """Every engine's reading, and what the down-cycle does with them."""

    readings: List[Reading]

    @property
    def resident(self) -> bool:
        return any(reading.state == RESIDENT for reading in self.readings)

    @property
    def unknown(self) -> bool:
        return any(reading.state == UNKNOWN for reading in self.readings)

    @property
    def busy(self) -> bool:
        """What the down-cycle guard asks: resident, or not known to be idle."""
        return self.resident or self.unknown

    def as_dict(self) -> Dict[str, Any]:
        return {
            "engines": [reading.as_dict() for reading in self.readings],
            "resident": self.resident, "unknown": self.unknown, "busy": self.busy,
        }


def _process_reading(engine: str, read: Callable[[], Any], basis: str) -> Reading:
    """A bridge status verb's ``running`` as a reading; unanswered is unknown."""
    try:
        status = read() or {}
    except Exception as error:  # noqa: BLE001 - an unanswered bridge is not "idle"
        return Reading(engine, UNKNOWN, basis, "the hardware bridge did not answer: {}".format(
            type(error).__name__))
    if not isinstance(status, Mapping) or "running" not in status:
        return Reading(engine, UNKNOWN, basis, "the status carried no running flag")
    return Reading(engine, RESIDENT if status.get("running") else NOT_RESIDENT, basis)


def npu_residency(flm_status: Callable[[], Any]) -> Reading:
    return _process_reading(
        ENGINE_NPU, flm_status,
        "flm-real holds its model while it runs (hardware bridge flm_status)",
    )


def gpu_residency(gpu_status: Callable[[], Any]) -> Reading:
    return _process_reading(
        ENGINE_GPU, gpu_status,
        "the GPU serving container never sleeps (hardware bridge gpu_status)",
    )


def cluster_residency(
    mode_state: Any, deployment: Optional[Mapping[str, Any]],
) -> Optional[Reading]:
    """The controller's Mode B deployment as a reading; ``None`` outside Mode B."""
    from .gpu_serving_target import MODE_CLUSTER

    if getattr(mode_state, "mode", "") != MODE_CLUSTER:
        return None
    basis = "the cluster deployment row the mode watch keeps measured"
    state = str((deployment or {}).get("state", ""))
    if state in _HOLDING:
        return Reading(ENGINE_CLUSTER, RESIDENT, basis, state)
    if state == _RELEASED:
        return Reading(ENGINE_CLUSTER, NOT_RESIDENT, basis, state)
    return Reading(ENGINE_CLUSTER, UNKNOWN, basis, state or "no deployment row")


def default_http_get(url: str) -> str:
    """A loopback GET with a short timeout; raises on any failure."""
    with urllib.request.urlopen(url, timeout=PROPS_TIMEOUT_SECONDS) as response:
        return response.read(256 * 1024).decode("utf-8", errors="replace")


def llama_residency(base_url: str, http_get: Callable[[str], str]) -> Reading:
    """A llama.cpp server's ``/props`` ``is_sleeping`` as a reading.

    ``/props`` does not wake a sleeping server, and ``is_sleeping`` is the
    engine's own flag for "the weights are unloaded". Missing, unreadable, or
    not a boolean: :data:`UNKNOWN`.
    """
    basis = "llama.cpp /props is_sleeping at {}".format(_origin(base_url))
    try:
        props = json.loads(http_get(_origin(base_url) + "/props"))
    except (OSError, ValueError, urllib.error.URLError) as error:
        return Reading(ENGINE_LLAMA, UNKNOWN, basis, type(error).__name__)
    sleeping = props.get("is_sleeping") if isinstance(props, Mapping) else None
    if not isinstance(sleeping, bool):
        return Reading(ENGINE_LLAMA, UNKNOWN, basis, "this build does not report is_sleeping")
    return Reading(ENGINE_LLAMA, NOT_RESIDENT if sleeping else RESIDENT, basis)


def _origin(base_url: str) -> str:
    parts = urlsplit(str(base_url or ""))
    return "{}://{}".format(parts.scheme or "http", parts.netloc)


def _loopback_port(base_url: str) -> Optional[int]:
    parts = urlsplit(str(base_url or ""))
    if parts.hostname not in ("127.0.0.1", "localhost", "::1"):
        return None
    try:
        return int(parts.port) if parts.port else None
    except ValueError:
        return None


def _port_of(status: Any) -> Optional[int]:
    try:
        return int((status or {}).get("port") or 0) or None
    except (AttributeError, TypeError, ValueError):
        return None


def appliance_residency(
    *, bridge: Any, broker: Any, mode_store: Any, cluster_store: Any,
    http_get: Callable[[str], str] = default_http_get,
    purposes: Sequence[str] = LOCAL_MODEL_PURPOSES,
) -> Residency:
    """Every engine on this appliance, measured: the one owner of "resident now".

    The NPU and the GPU container are read from the hardware bridge; Mode B
    from the deployment row the mode file names; and every OTHER loopback
    model endpoint a local lease points at (the Pi's Assistant, a stock-GGUF
    AI Chat) from its own ``/props``. A lease whose port is the NPU's, the GPU
    container's or the cluster's is already measured and is not asked twice.
    """
    readings: List[Reading] = []
    flm: Any = None
    gpu: Any = None
    try:
        flm = bridge.flm_status()
    except Exception:  # noqa: BLE001 - reported as unknown below
        flm = None
    readings.append(npu_residency(lambda: _raise_if_none(flm)))
    try:
        gpu = bridge.gpu_status()
    except Exception:  # noqa: BLE001 - reported as unknown below
        gpu = None
    readings.append(gpu_residency(lambda: _raise_if_none(gpu)))
    measured_ports = {_port_of(flm), _port_of(gpu)} - {None}
    try:
        state = mode_store.read()
        deployment = (
            cluster_store.get_pooled_deployment(state.deployment_name)
            if getattr(state, "deployment_name", "") else None
        )
    except Exception:  # noqa: BLE001 - an unread mode is not "no cluster"
        readings.append(Reading(ENGINE_CLUSTER, UNKNOWN, "the GPU serving mode file", "unreadable"))
        state = None
    else:
        cluster = cluster_residency(state, deployment)
        if cluster is not None:
            readings.append(cluster)
            port = int(getattr(state, "cluster_port", 0) or 0)
            if port:
                measured_ports |= {port, port + 1}
    asked = set()
    for purpose in purposes:
        lease = _lease(broker, purpose)
        if isinstance(lease, Reading):
            readings.append(lease)
            continue
        if lease is None or str(lease.get("provider", "")) in OFF_MACHINE_KINDS:
            continue  # no lease, or a hosted provider: nothing held here
        base_url = str(lease.get("base_url", "") or "")
        port = _loopback_port(base_url)
        if port is None:
            readings.append(Reading(
                ENGINE_LLAMA, UNKNOWN, LEASE_BASIS.format(purpose),
                "its model is not on this machine's loopback, so it is not measured here",
            ))
            continue
        if port in measured_ports or port in asked:
            continue
        asked.add(port)
        readings.append(llama_residency(base_url, http_get))
    return Residency(readings)


def _lease(broker: Any, purpose: str) -> Any:
    """The active lease, ``None`` for no lease, or an UNKNOWN reading (review S9).

    "No credential is active" is a fact; a broker that could not answer is not,
    and reads as unknown - never as "nothing local to ask".
    """
    basis = LEASE_BASIS.format(purpose)
    try:
        return broker.resolve_active(purpose) or None
    except CredentialError as error:
        if str(error) in (BROKER_UNAVAILABLE, BROKER_NO_ANSWER):
            return Reading(ENGINE_LLAMA, UNKNOWN, basis, "the credential broker did not answer")
        return None
    except LookupError:
        return None
    except Exception as error:  # noqa: BLE001 - an unanswered broker is not "idle"
        return Reading(ENGINE_LLAMA, UNKNOWN, basis, type(error).__name__)


def _raise_if_none(status: Any) -> Any:
    if status is None:
        raise ConnectionError("no answer")
    return status
