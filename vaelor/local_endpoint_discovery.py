"""Find an OpenAI-compatible model server already running on this machine.

Vaelor could only ever see a local model it had started itself.

The cause, established rather than assumed: the single path that registers a
local model is ``executor._deploy_model``, which stands up its own llama.cpp
container and then registers the credential it just created via
``activate_managed_local``. Nothing anywhere probes for an endpoint Vaelor does
not own — not gated on a machine class, not a probe that fails, simply absent.
On a Raspberry Pi the gap is invisible, because Vaelor deploys the model and
therefore always owns it. On any machine where something else is already
serving models, a working endpoint answering in about a second showed up as
``MODEL REQUIRED`` and an empty picker.

This is independent of how Vaelor comes to run models itself. Whatever the
appliance ends up owning, an owner who already has an OpenAI-compatible server
on the box should have it found and offered.

Three rules, and the first is the important one:

**Offer, never adopt.** Finding a server is not consent to send someone's
questions through it. Discovery produces a *candidate* an owner can accept,
carrying what it is, where it is and what it holds. Nothing here activates
anything.

**Loopback only.** An endpoint on ``127.0.0.1`` is on the machine the user
already trusts. Vaelor does not scan the network looking for models; that is
someone else's machine and possibly someone else's data.

**Say when nothing was found.** "Vaelor looked on this machine and found no
local AI endpoint" is a different message from silence, and both are different
from ``MODEL REQUIRED`` with no account of what is required or how.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Callable, Dict, List, Optional, Sequence


#: Ports probed on loopback, with what is normally behind each. A list rather
#: than a scan: probing every port on the machine is intrusive, slow, and would
#: knock on services that are not model servers.
CANDIDATE_PORTS = (
    (13305, "Lemonade"),
    (8000, "Lemonade"),
    (1234, "LM Studio"),
    (11434, "Ollama"),
    (8080, "llama.cpp server"),
    (34011, "Vaelor managed local model"),
)

#: Only loopback. Both families, because a host may resolve either way.
LOOPBACK_HOSTS = ("127.0.0.1", "[::1]")

#: Per-probe budget. A port with nothing behind it refuses immediately; one
#: with something slow behind it is not worth blocking a settings page for.
PROBE_TIMEOUT_SECONDS = 1.5

#: Ceiling on the whole sweep, however many probes it takes.
#:
#: The first version ran twelve probes serially at 1.5 s each: eighteen seconds
#: worst case. A refused connection returns instantly, so on a healthy
#: developer machine that never showed - but a firewalled or containerised host
#: *drops* rather than refuses, and pays the full timeout on every port. Probes
#: now run concurrently and the sweep as a whole is bounded, so the cost is one
#: probe timeout rather than the sum of them.
SWEEP_BUDGET_SECONDS = 3.0

#: How long a completed sweep is served before another is started. Endpoints do
#: not come and go second by second, and this is read on a settings page paint.
CACHE_SECONDS = 60.0

#: Paths tried for a model listing, in order. ``/api/v1`` is Lemonade's own
#: namespace and carries the per-model engine that ``/v1`` does not.
MODEL_PATHS = ("/api/v1/models", "/v1/models")

#: Where Lemonade reports which engine each loaded model is on. This is the
#: fact the two-tier design needs: without it Vaelor sees two models and has no
#: reason to send the Assistant to the NPU one and AI Chat to the GPU one.
HEALTH_PATHS = ("/api/v1/health", "/health")

#: Engine names normalised from whatever the server calls them.
ENGINES = {"npu": "npu", "gpu": "gpu", "cpu": "cpu", "igpu": "gpu", "dgpu": "gpu"}


def _fetch(url: str, timeout: float) -> Optional[Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read(512 * 1024).decode("utf-8", "replace"))
    except (OSError, ValueError, urllib.error.URLError):
        return None


def _listening(host: str, port: int, timeout: float) -> bool:
    """Whether anything holds this port, without sending it a request."""
    family = socket.AF_INET6 if host.startswith("[") else socket.AF_INET
    address = host.strip("[]")
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.settimeout(timeout)
            return probe.connect_ex((address, port)) == 0
    except OSError:
        return False


def _model_names(listing: Any) -> List[str]:
    entries = []
    if isinstance(listing, dict):
        entries = listing.get("data") or listing.get("models") or []
    elif isinstance(listing, list):
        entries = listing
    names = []
    for entry in entries:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict):
            name = entry.get("id") or entry.get("model") or entry.get("name")
            if name:
                names.append(str(name))
    return names[:200]


def model_engines(health: Any) -> Dict[str, str]:
    """Which engine each loaded model is running on, from a health report.

    Returns only what the server actually said. A model the report does not
    mention is absent from the result rather than guessed at — routing the
    Assistant to a tier on an assumption is how the two-tier design would
    quietly become a one-tier one.
    """
    found: Dict[str, str] = {}
    candidates: Sequence[Any] = ()
    if isinstance(health, dict):
        for key in ("models", "loaded_models", "data", "model_loaded"):
            value = health.get(key)
            if isinstance(value, (list, tuple)):
                candidates = value
                break
            if isinstance(value, dict):
                candidates = [
                    {"id": name, **(facts if isinstance(facts, dict) else {})}
                    for name, facts in value.items()
                ]
                break
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("id") or entry.get("model") or entry.get("name") or "")
        device = str(entry.get("device") or entry.get("engine") or "").lower()
        if name and device in ENGINES:
            found[name] = ENGINES[device]
    return found


def probe_endpoint(
    host: str,
    port: int,
    label: str = "",
    *,
    timeout: float = PROBE_TIMEOUT_SECONDS,
    fetch: Callable[[str, float], Optional[Any]] = _fetch,
    is_listening: Callable[[str, int, float], bool] = _listening,
) -> Optional[Dict[str, Any]]:
    """Describe one loopback endpoint, or ``None`` if there is nothing usable.

    A port that is open but does not answer a model listing is not reported: a
    database or a web app is not an AI endpoint, and offering one as a model
    connection would be worse than finding nothing.
    """
    if not is_listening(host, port, timeout):
        return None
    base = "http://{}:{}".format(host, port)
    models: List[str] = []
    api_root = ""
    for path in MODEL_PATHS:
        listing = fetch(base + path, timeout)
        names = _model_names(listing)
        if names:
            models = names
            api_root = base + path.rsplit("/", 1)[0]
            break
    if not models:
        return None
    engines: Dict[str, str] = {}
    for path in HEALTH_PATHS:
        engines = model_engines(fetch(base + path, timeout))
        if engines:
            break
    return {
        "base_url": api_root,
        "host": host,
        "port": port,
        "label": label or "Local model server",
        "provider": "openai-compatible",
        "local": True,
        "models": models,
        # Which engine each *loaded* model is on. Empty when the server does
        # not say, which is a different fact from "they are all on the CPU".
        "model_engines": engines,
        "engines_known": bool(engines),
        "adopted": False,
        "reason": (
            "Found on this machine at {}. Vaelor has not connected to it: "
            "accept it to route questions through it."
        ).format(base),
    }


def discover(
    ports: Sequence[Any] = CANDIDATE_PORTS,
    hosts: Sequence[str] = LOOPBACK_HOSTS,
    *,
    timeout: float = PROBE_TIMEOUT_SECONDS,
    probe: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
    budget: float = SWEEP_BUDGET_SECONDS,
) -> Dict[str, Any]:
    """Endpoints available on this machine, as offers rather than connections.

    Probes run concurrently and the sweep is bounded by ``budget``, so an
    unreachable port costs one timeout rather than adding one. Still not a
    scan: it is the same short allowlist, on loopback only.

    This is the blocking form, and nothing on a request path should call it.
    Use :func:`discover_cached`.
    """
    look = probe or probe_endpoint
    targets = []
    for entry in ports:
        port, label = entry if isinstance(entry, (list, tuple)) else (entry, "")
        for host in hosts:
            targets.append((host, int(port), label))

    def probe_one(target):
        host, port, label = target
        try:
            return look(host, port, label, timeout=timeout)
        except (OSError, ValueError):
            return None

    results: List[Optional[Dict[str, Any]]] = [None] * len(targets)
    if targets:
        # Deliberately not a `with` block. `ThreadPoolExecutor.__exit__` calls
        # `shutdown(wait=True)`, which blocks until every *running* probe
        # returns - so the budget below would bound nothing and the sweep would
        # still take as long as its slowest port. Found by the test for it.
        pool = ThreadPoolExecutor(max_workers=min(len(targets), 16))
        try:
            futures = {pool.submit(probe_one, item): index
                       for index, item in enumerate(targets)}
            try:
                for future in as_completed(futures, timeout=budget):
                    results[futures[future]] = future.result()
            except FuturesTimeout:
                # Whatever finished inside the budget is the answer. A port
                # that has not replied by now is not one an owner is waiting
                # on, and abandoning it beats making them wait.
                pass
        finally:
            # Queued probes are cancelled; one already in flight is left to
            # finish on its own socket timeout rather than held onto here.
            pool.shutdown(wait=False, cancel_futures=True)

    found: List[Dict[str, Any]] = []
    seen = set()
    # Iterated in target order, not completion order, so the result does not
    # depend on which probe happened to answer first.
    for result in results:
        if not result:
            continue
        key = (result["port"], tuple(sorted(result["models"])))
        if key in seen:
            # The same server answering on both loopback families.
            continue
        seen.add(key)
        found.append(result)
    if not found:
        return {
            "endpoints": [],
            "searched": ["{}:{}".format(host, entry[0] if isinstance(entry, (list, tuple)) else entry)
                         for entry in ports for host in hosts[:1]],
            "reason": (
                "Vaelor looked on this machine and found no local AI endpoint. "
                "It checks loopback only, so a model server on another machine "
                "has to be added by hand."
            ),
        }
    return {
        "endpoints": found,
        "searched": [],
        "reason": "",
    }


_CACHE_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {}
_REFRESHING = False


def _never_scanned() -> Dict[str, Any]:
    return {
        "endpoints": [],
        "searched": [],
        "scanning": True,
        "reason": (
            "Vaelor is checking this machine for a local AI endpoint. This "
            "page will show any it finds."
        ),
    }


def reset_cache() -> None:
    """Drop the memoised sweep (tests, and after an endpoint is accepted)."""
    global _REFRESHING
    with _CACHE_LOCK:
        _CACHE.clear()
        _REFRESHING = False


def discover_cached(
    *,
    now: Optional[Callable[[], float]] = None,
    ttl: float = CACHE_SECONDS,
    scan: Optional[Callable[[], Dict[str, Any]]] = None,
    background: bool = True,
) -> Dict[str, Any]:
    """The last completed sweep, refreshing out of band. Never blocks.

    This exists because the first version called :func:`discover` straight from
    the ``/ai-chat/setup`` handler. Twelve serial probes at 1.5 s is an
    eighteen-second page-load endpoint on any host that drops rather than
    refuses, and the browser was right to abort it.

    A caller that has never scanned gets an immediate "checking" answer rather
    than a wait, and the real result on the next paint. The settings page
    renders either way.
    """
    global _REFRESHING
    clock = now or time.monotonic
    moment = clock()
    run = scan or discover
    with _CACHE_LOCK:
        cached = _CACHE.get("result")
        age = moment - float(_CACHE.get("refreshed_at", 0.0))
        fresh = cached is not None and 0 <= age < ttl
        if fresh:
            return dict(cached)
        start_refresh = not _REFRESHING
        if start_refresh:
            _REFRESHING = True

    def refresh() -> None:
        global _REFRESHING
        try:
            result = run()
        except Exception:  # pragma: no cover - a sweep must never crash a thread
            result = {
                "endpoints": [], "searched": [],
                "reason": "The local endpoint scan did not complete.",
            }
        with _CACHE_LOCK:
            _CACHE["result"] = result
            _CACHE["refreshed_at"] = clock()
            _REFRESHING = False

    if start_refresh:
        if background:
            thread = threading.Thread(
                target=refresh, name="vaelor-endpoint-discovery", daemon=True
            )
            thread.start()
        else:
            # Only for callers that want the sweep inline, such as a test or a
            # deliberate on-demand refresh. Never the request path.
            refresh()
            with _CACHE_LOCK:
                return dict(_CACHE.get("result") or _never_scanned())
    with _CACHE_LOCK:
        cached = _CACHE.get("result")
    # A stale result is still a true statement about this machine a minute ago,
    # and it beats showing nothing while the refresh runs.
    return dict(cached) if cached is not None else _never_scanned()


def tier_assignments(endpoint: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Which discovered model belongs to which tier, where the server said.

    The Assistant wants the NPU-resident model and AI Chat wants the GPU one.
    That is only knowable because the health report names the engine per model;
    where it does not, this returns no assignment rather than picking one.
    """
    engines = dict((endpoint or {}).get("model_engines") or {})
    if not engines:
        return {
            "assistant": None,
            "chat": None,
            "reason": (
                "This server does not report which engine each model runs on, "
                "so Vaelor cannot tell its NPU model from its GPU one."
            ),
        }
    npu = [name for name, device in engines.items() if device == "npu"]
    gpu = [name for name, device in engines.items() if device == "gpu"]
    return {
        "assistant": npu[0] if npu else None,
        "chat": gpu[0] if gpu else None,
        "reason": "",
        "npu_models": npu,
        "gpu_models": gpu,
    }
