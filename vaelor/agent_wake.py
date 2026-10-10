"""A deployed agent's side of two questions about its model: is it up, and wake it.

Both run inside the loopback agent runtime (:mod:`vaelor.agent_server`), which
reads only its 0600 config and holds no broker socket or database.

**Is the model up (ACC-072).** The runtime's ``/health`` answered ``ok`` whatever
its model was doing, and the console read Serving from that. :class:`ModelProbe`
asks the backing endpoint's ``GET /models`` with the agent's own model key - the
cheapest request every OpenAI-compatible server answers - bounded and cached,
so ``/health`` can say whether the model answered and the console can say
"model not answering" instead of Serving.

**Wake it (owner decision 2026-09-28).** A cluster model scaled to zero after
sitting idle counts as available and wakes on demand, but the agent posted
straight to the stopped balancer and its first chat failed with a connection
error. :class:`WakeClient` asks the control plane (``POST /api/v2/agents/wake``,
authenticated by the agent's per-deployment bearer, the one its memory calls
carry) what its RECORDED model deployment is doing; the control plane enqueues
the same warm load the inference gateway's wake enqueues for an idle unload and
never for a manual one. The runtime then answers its client with an honest
OpenAI-shaped 503 and ``Retry-After`` - "the model is waking" or "the model was
unloaded by hand" - instead of a connection error. Only a request that already
passed the agent's keyed gate reaches the runtime, so only a valid agent key
can cause a wake.
"""

from __future__ import annotations

import json
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Mapping, Optional

#: The model probe is bounded and cached, so ``/health`` - polled by the console
#: and the reconcile - never multiplies requests to the model.
MODEL_PROBE_TIMEOUT_SECONDS = 3
MODEL_PROBE_CACHE_SECONDS = 15.0

#: How the model probe describes itself in the backing server's logs. Purely
#: descriptive: no code - here or anywhere - may treat a request differently
#: because of a header a caller controls. The probe needs no exclusion from
#: usage metering anyway: it calls the recorded deployment's own loopback
#: endpoint directly (``agent_backing.resolve_recorded_backing``), never the
#: LLM Server gate or the inference gateway where requests are metered.
MODEL_PROBE_USER_AGENT = "vaelor-agent-model-probe/1"

#: A wake request is small and must not hold a client's request open.
WAKE_TIMEOUT_SECONDS = 4
MAX_WAKE_RESPONSE_BYTES = 16 * 1024


class ModelProbe:
    """``GET <base_url>/models`` with the agent's model key, cached. Never raises."""

    def __init__(
        self, base_url: str, api_key: str = "", *,
        opener: Optional[Callable[..., Any]] = None,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ):
        self._url = str(base_url or "").rstrip("/") + "/models"
        self._api_key = str(api_key or "")
        self._opener = opener or urllib.request.urlopen
        self._clock = clock
        self._wall = wall
        self._lock = threading.Lock()
        self._cached: Optional[Dict[str, Any]] = None
        self._cached_at = 0.0

    def status(self) -> Dict[str, Any]:
        """``{reachable, detail, checked_at}`` - the last probe, at most 15 s old."""
        with self._lock:
            now = self._clock()
            if self._cached is not None and now - self._cached_at < MODEL_PROBE_CACHE_SECONDS:
                return dict(self._cached)
            self._cached = self._probe()
            self._cached_at = now
            return dict(self._cached)

    def _probe(self) -> Dict[str, Any]:
        headers = {"Accept": "application/json", "User-Agent": MODEL_PROBE_USER_AGENT}
        if self._api_key:
            headers["Authorization"] = "Bearer {}".format(self._api_key)
        request = urllib.request.Request(self._url, headers=headers, method="GET")
        try:
            with self._opener(request, timeout=MODEL_PROBE_TIMEOUT_SECONDS) as response:
                code = int(getattr(response, "status", 200))
        except urllib.error.HTTPError as error:
            code = int(getattr(error, "code", 0) or 0)
        except (OSError, ValueError, urllib.error.URLError):
            return self._result(False, "The model endpoint could not be reached.")
        if 200 <= code < 300:
            return self._result(True, "")
        # /health is unauthenticated: no upstream status code, no body - only
        # that the model did not accept the check (S8).
        return self._result(False, "The model endpoint did not accept the check.")

    def _result(self, reachable: bool, detail: str) -> Dict[str, Any]:
        return {"reachable": reachable, "detail": detail, "checked_at": self._wall()}


def _endpoint_ok(endpoint: str) -> bool:
    parts = urllib.parse.urlsplit(endpoint)
    return (
        parts.scheme in ("http", "https") and bool(parts.hostname)
        and not parts.username and not parts.password
    )


class WakeClient:
    """Ask the control plane to wake this agent's recorded model. Never raises."""

    def __init__(
        self, endpoint: str, token: str, *,
        opener: Optional[Callable[..., Any]] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
    ):
        self._endpoint = endpoint
        self._token = token
        self._opener = opener or urllib.request.urlopen
        self._ssl_context = ssl_context

    def ask(self) -> Optional[Dict[str, Any]]:
        """The control plane's answer (``state``, ``retry_after``, ``detail``), or ``None``."""
        request = urllib.request.Request(
            self._endpoint, data=b"{}", method="POST",
            headers={
                "Content-Type": "application/json", "Accept": "application/json",
                "Authorization": "Bearer {}".format(self._token),
            },
        )
        extra = {} if self._ssl_context is None else {"context": self._ssl_context}
        try:
            with self._opener(request, timeout=WAKE_TIMEOUT_SECONDS, **extra) as response:
                body = json.loads(response.read(MAX_WAKE_RESPONSE_BYTES).decode("utf-8"))
        except Exception:  # noqa: BLE001 - no answer: the caller keeps its own reason
            return None
        data = body.get("data") if isinstance(body, Mapping) else None
        return dict(data) if isinstance(data, Mapping) else None


def build_wake_client(section: Any) -> Optional[WakeClient]:
    """A wake client from the config's ``wake`` block, or ``None`` when unusable.

    The block is the agent's memory block with the wake endpoint in place of the
    memory one (same bearer, same pinned certificate when the endpoint is
    https), so the two can never disagree about how the control plane is
    reached.
    """
    if not isinstance(section, Mapping):
        return None
    endpoint = str(section.get("endpoint") or "").strip()
    token = str(section.get("token") or "").strip()
    if not endpoint or not token or not _endpoint_ok(endpoint):
        return None
    context = None
    ca_pem = str(section.get("ca_pem") or "").strip()
    if ca_pem and urllib.parse.urlsplit(endpoint).scheme == "https":
        # The memory client's pin on the controller's household authority, the
        # name checked as the deploy said (VD-212) - one TLS setup for both
        # calls to the control plane. Imported here: agent_server imports
        # this module.
        from .agent_server import pin_section_checks_hostname, pinned_certificate_context

        try:
            context = pinned_certificate_context(ca_pem, pin_section_checks_hostname(section))
        except (ssl.SSLError, ValueError):
            return None
    return WakeClient(endpoint, token, ssl_context=context)


def wake_endpoint(memory_endpoint: str) -> str:
    """The wake URL beside a memory base URL (``.../agents/memory`` -> ``.../agents/wake``)."""
    base = str(memory_endpoint or "").rstrip("/")
    if base.endswith("/memory"):
        return base[: -len("/memory")] + "/wake"
    return ""


#: The runtime's answers when a wake was asked, by the control plane's state.
WAKING_REASON = (
    "The agent's model is loading after sitting idle. Retry after the time "
    "given in Retry-After."
)
UNLOADED_REASON = (
    "The agent's model was unloaded by hand. It must be loaded again from "
    "Cluster > Deployments before this agent can answer."
)


def wake_answer(answer: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The 503 error body for a wake answer, or ``None`` to keep the caller's own.

    ``unloaded-idle`` and ``starting`` answer ``model_waking`` with the
    control plane's retry hint; ``unloaded-manual`` answers ``model_unloaded``
    and no retry hint (retrying cannot help). Anything else - serving,
    unknown, no answer - leaves the caller's unreachable answer standing.
    """
    if not answer:
        return None
    state = str(answer.get("state") or "")
    if state in ("unloaded-idle", "starting"):
        try:
            retry = max(1, int(answer.get("retry_after") or 30))
        except (TypeError, ValueError):
            retry = 30
        return {"error": {
            "type": "model_waking", "message": WAKING_REASON, "retry_after": retry,
        }}
    if state == "unloaded-manual":
        return {"error": {"type": "model_unloaded", "message": UNLOADED_REASON}}
    return None
