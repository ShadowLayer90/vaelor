"""Authenticated OpenAI-compatible gateway for the active cluster inference service."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from flask import Blueprint, Response, jsonify, request, stream_with_context

from .inference_client import remote_inference_budget
from .inference_metrics import is_failure
from .provider_runtime import managed_local_connection
from .session_affinity import SESSION_HEADER, only_header_value, session_headers


MAX_REQUEST_BYTES = 1024 * 1024
FORWARDED_RESPONSE_HEADERS = {"content-type", "cache-control"}

#: How many trailing raw upstream bytes to keep while streaming so the final SSE
#: ``usage`` event can be read in the ``finally`` without ever buffering or
#: delaying the client stream (B3). vLLM's usage chunk plus the ``[DONE]``
#: sentinel are well inside this; the bounded tail keeps latency and memory flat.
USAGE_TAIL_BYTES = 32 * 1024

#: The Retry-After a scaled-to-zero cluster model's wake answers with (G3b,
#: CN-4): a warm reload is seconds, so a short, standard hint the client obeys.
GATEWAY_WAKE_RETRY_AFTER = "30"
GATEWAY_WAKE_MESSAGE = (
    "The cluster model was scaled to zero on idle and is being woken; "
    "retry shortly."
)

#: The most of an upstream error body relayed to the client. vLLM's error JSON
#: (a context-length refusal, a bad parameter) is a few hundred bytes; the cap
#: only stops a misbehaving upstream from streaming an unbounded error page.
UPSTREAM_ERROR_BODY_BYTES = 64 * 1024

#: Upstream statuses that mean the GATEWAY's own lease credential was refused.
#: The caller authenticated to the gateway successfully, so relaying a 401/403
#: would blame their token for the gateway's misconfiguration: these become a
#: 502 the gateway owns, counted as a failure.
_UPSTREAM_AUTH_STATUSES = frozenset({401, 403})

#: Header on a model listing answered from the lease while the cluster model is
#: scaled to zero or still loading, so a client that cares can tell the listing
#: names a model that is not in memory yet (ACC-057).
MODEL_LOADED_HEADER = "X-Vaelor-Model-Loaded"


def upstream_answer_seconds() -> int:
    """How long a chat completion may take upstream before the gateway gives up.

    The cluster model is a connected endpoint, so it gets the budget every
    connected endpoint gets: `inference_client.remote_inference_budget`, 240 s
    unless the owner sets ``VAELOR_INFERENCE_TIMEOUT_SECONDS`` (bounded to
    30-900 s), read per request. It used to be a fixed 120 s, stated nowhere,
    and a long answer from the 30B outlasted it (review A12). It bounds the
    wait for the upstream to start answering - all of a non-streamed answer -
    and each read of a stream; the LLM Server gate in front allows an hour.
    """
    return remote_inference_budget()


def _as_int(value) -> int:
    """A token count coerced to a non-negative int, 0 for anything unreadable.

    The upstream ``usage`` object should carry integers, but a null or a garbled
    field must degrade to "0 tokens" rather than raise into the metering path.
    """
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def merge_stream_usage_option(parsed, body):
    """The upstream body with ``stream_options.include_usage`` on for a stream (B4).

    Only a streaming request is ever changed, and only when the client left
    ``include_usage`` unset: an explicit client ``false`` is honoured untouched
    (that stream then honestly meters 0 tokens), and any sibling
    ``stream_options`` field is preserved by merging into the object rather than
    replacing it. A non-streaming request returns the original bytes unchanged,
    so the client's own request contract is never altered - the injection is on
    the upstream copy alone.
    """
    if not isinstance(parsed, dict) or not parsed.get("stream"):
        return body
    options = parsed.get("stream_options")
    if not isinstance(options, dict):
        options = {}
    if "include_usage" in options:
        return body
    merged = dict(parsed)
    merged["stream_options"] = {**options, "include_usage": True}
    return json.dumps(merged).encode("utf-8")


def sniff_stream_usage(tail):
    """The token usage from the final SSE ``usage`` event in a raw byte tail (B3).

    The bytes are a bounded rolling tail of the upstream stream, so the first
    event may be a fragment; the tail is decoded once with errors ignored (a
    UTF-8 codepoint split across a read boundary is simply dropped) and split on
    the SSE blank-line event separator. The last ``data:`` event whose JSON
    carries a non-null ``usage`` object wins - vLLM emits usage in the closing
    event, then a ``data: [DONE]`` sentinel that is skipped here. A stream that
    never carried a usage object (an ``include_usage:false`` client, or an engine
    that omitted it) yields an empty dict, which the caller records as 0 tokens.
    """
    if not tail:
        return {}
    text = tail.decode("utf-8", "ignore")
    usage = {}
    for event in text.split("\n\n"):
        for line in event.splitlines():
            stripped = line.strip()
            if not stripped.startswith("data:"):
                continue
            payload = stripped[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                parsed = json.loads(payload)
            except (ValueError, json.JSONDecodeError):
                continue
            candidate = parsed.get("usage") if isinstance(parsed, dict) else None
            if isinstance(candidate, dict):
                usage = candidate
    return usage


def _relayable_json(body: bytes) -> bool:
    """Whether an upstream error body is complete JSON that may be relayed as-is.

    A body at or over the read cap was cut short, and a cut JSON document no
    longer parses - it must not go out labelled ``application/json``.
    """
    if not body or len(body) > UPSTREAM_ERROR_BODY_BYTES:
        return False
    try:
        json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return False
    return True


def upstream_error_response(error):
    """The model's own error, relayed with its own status (ACC-056).

    ``urlopen`` raises an ``HTTPError`` for any upstream 4xx/5xx; it used to be
    caught with everything else and turned into a 503 "model unavailable",
    losing the model's explanation (a prompt over the context window, a bad
    parameter) and counting the caller's mistake as the model failing.

    What is relayed is deliberately narrow, because the gateway faces the LAN:

    * a **4xx whose body is complete JSON** - the model's own refusal, the
      value ACC-056 exists for - goes back verbatim with its status;
    * a **5xx**, or any body that is not complete JSON (an nginx HTML error
      page, a vLLM exception text naming paths and addresses, a JSON document
      cut at the read cap), is wrapped in the gateway's own error envelope
      carrying only the upstream status. Its text never reaches the client;
    * an upstream **401/403** refuses the GATEWAY's credential, not the
      caller's, so it becomes a 502 in the gateway's own words.

    Returns ``(response, status)``; the status is what the RED log records.
    """
    try:
        # One byte over the cap is read so a body that was cut can be told
        # from one that merely filled it.
        body = error.read(UPSTREAM_ERROR_BODY_BYTES + 1) or b""
    except (OSError, ValueError):
        body = b""
    finally:
        try:
            error.close()
        except (OSError, ValueError):
            pass
    status = int(getattr(error, "code", 0) or 0)
    if status < 400:
        # urlopen follows redirects and raises only for errors, so a sub-400
        # HTTPError is a malformed answer: the gateway owns it as a bad gateway.
        status = 502
    if status in _UPSTREAM_AUTH_STATUSES:
        return jsonify({"error": {
            "message": (
                "The cluster model refused the gateway's own credential; the "
                "gateway's lease for it needs repairing."
            ),
            "type": "upstream_auth_error",
        }}), 502
    if is_failure(status) or not _relayable_json(body):
        return jsonify({"error": {
            "message": "The cluster model answered HTTP {} without a detail the gateway can pass on.".format(status),
            "type": "upstream_error",
            "upstream_status": status,
        }}), status
    return Response(body, status=status, content_type="application/json"), status


def create_inference_gateway_blueprint(
    tokens, broker, metrics, tracer=None, cluster_wake=None, meter=None,
    cluster_resting=None,
):
    """The ``/inference/v1`` blueprint.

    ``cluster_wake`` is the lease-resolution seam that enqueues a warm load for
    an idle-unloaded cluster model and names it; only a chat completion - real
    inference - calls it. ``cluster_resting`` is the same check WITHOUT the
    enqueue: it names the deployment when the lease's cluster model is scaled
    to zero or loading. The model listing uses it to answer from the lease
    instead of waking the model, because a client that polls ``/models`` (most
    chat UIs do) would otherwise reload a model the idle watch then unloads
    again, in a loop, since a listing is not traffic the watch can see
    (ACC-057).
    """
    blueprint = Blueprint(
        "inference_gateway", __name__, url_prefix="/inference/v1"
    )

    def emit_span(parsed, *, status, duration_ms, streaming,
                  prompt_tokens=0, completion_tokens=0, response_bytes=0):
        """Ship one OTLP span for a recorded request, never affecting the response.

        A no-op when no tracer is wired or tracing is disabled. The tracer already
        swallows every failure and fires the POST off-thread; this extra guard
        means even constructing the call cannot raise into the request path. The
        model on the span is the served/requested model name - the RED metric
        store has no model column, so it is read from the request here.
        """
        if tracer is None:
            return
        try:
            tracer.emit(
                model=str((parsed or {}).get("model") or ""),
                status=status, duration_ms=duration_ms, streaming=streaming,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                response_bytes=response_bytes,
            )
        except Exception:  # noqa: BLE001 - tracing must never affect the response
            pass

    def meter_usage(identity, deployment, status, prompt_tokens, completion_tokens):
        """Fold one served request into the durable usage meter, best-effort (C2/C3).

        Called only on the 2xx success seams and never on the error seam, so a
        failed upstream is not counted. Tokens are whatever was parsed/sniffed (0
        when none was present); the request itself is still counted. The call is
        wrapped so a meter fault degrades to "not counted" and can never fail the
        inference, and the meter opens its own connection here - after the
        upstream forward has finished - so no store lock is ever held across the
        urlopen.
        """
        if not status or status >= 400:
            return
        from .credential_use import note_credential_use

        # The lease's cluster credential answered: a use of it (ACC-107).
        note_credential_use(deployment, broker=broker)
        if meter is None or identity is None:
            return
        try:
            meter.record(
                key_id=str(identity.get("id") or ""),
                deployment=deployment,
                label=str(identity.get("label") or ""),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception:  # noqa: BLE001 - metering must never affect the response
            pass

    def authenticate():
        header = request.headers.get("Authorization", "")
        return tokens.authenticate(
            header[7:] if header.startswith("Bearer ") else "", "inference"
        )

    def lease():
        return broker.resolve_active("cluster-inference")

    def waking_response():
        """A 503 telling the client the idle-unloaded model is waking (CN-4).

        Returned in place of the generic upstream-unavailable 503 when the
        cluster-inference lease names a deployment in state ``unloaded`` (the
        scaled-to-zero rest state, which the seam wakes) or ``deploying`` (a
        load already in flight, which it reports without a second wake). A
        crashed or removed deployment it leaves alone: ``None`` then, so the
        caller falls through to its own 503.
        """
        if cluster_wake is None or not cluster_wake():
            return None
        return jsonify({"error": {
            "message": GATEWAY_WAKE_MESSAGE,
            "type": "model_waking",
        }}), 503, {"Retry-After": GATEWAY_WAKE_RETRY_AFTER}

    def upstream_request(profile, path: str, *, body: bytes | None = None):
        headers = {"Accept": request.headers.get("Accept", "application/json")}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if profile.get("api_key"):
            headers["Authorization"] = "Bearer {}".format(profile["api_key"])
        # VD-157: a client that names its conversation keeps it on one replica.
        # Judged by the one rule before it goes back on the wire - a header
        # sent twice is none, as the balancer reads it - and forwarded only to
        # a model this appliance serves, the rule AI Chat keeps.
        if managed_local_connection(profile):
            headers.update(session_headers(
                only_header_value(request.headers.getlist(SESSION_HEADER))
            ))
        return urllib.request.Request(
            profile["base_url"].rstrip("/") + path,
            data=body,
            headers=headers,
            method="POST" if body is not None else "GET",
        )

    def authorized():
        """Resolve the caller's identity, running ``authenticate`` exactly once (C1).

        ``authenticate`` writes ``last_used_at`` on every call, so it must not be
        invoked a second time to meter the request. This returns the identity so
        a caller threads it through instead of re-authenticating, and a
        ``(response, status)`` denial when the token is missing or revoked.
        """
        identity = authenticate()
        if identity is None:
            return None, (jsonify({
                "error": {
                    "message": "Invalid or revoked API token.",
                    "type": "authentication_error",
                }
            }), 401)
        return identity, None

    @blueprint.get("/openapi.json")
    def openapi():
        _identity, denied = authorized()
        if denied:
            return denied
        return jsonify({
            "openapi": "3.1.0",
            "info": {"title": "Vaelor Inference Gateway", "version": "1.0.0"},
            "paths": {
                "/inference/v1/models": {"get": {"summary": "List active models"}},
                "/inference/v1/chat/completions": {
                    "post": {"summary": "Create a chat completion"}
                },
            },
        })

    def unavailable(error):
        return jsonify({
            "error": {
                "message": "The active cluster model is unavailable.",
                "type": "upstream_error",
                "detail": str(error)[:200],
            }
        }), 503

    def resting_listing(profile):
        """The model list answered from the lease while the model is not loaded.

        ``None`` unless the lease's cluster deployment is scaled to zero or
        loading AND the lease names its model - then the one model the gateway
        serves is listed, marked not loaded, and nothing is woken (ACC-057).
        """
        if cluster_resting is None:
            return None
        try:
            resting = cluster_resting()
        except Exception:  # noqa: BLE001 - an unreadable state lists nothing
            resting = ""
        model = str((profile or {}).get("model") or "")
        if not resting or not model:
            return None
        return jsonify({
            "object": "list",
            "data": [{"id": model, "object": "model", "owned_by": "vaelor"}],
        }), 200, {MODEL_LOADED_HEADER: "false"}

    @blueprint.get("/models")
    def models():
        _identity, denied = authorized()
        if denied:
            return denied
        try:
            profile = lease()
        except Exception as error:
            return unavailable(error)
        try:
            with urllib.request.urlopen(upstream_request(profile, "/models"), timeout=10) as response:
                return Response(
                    response.read(), response.status,
                    content_type=response.headers.get("Content-Type", "application/json"),
                )
        except urllib.error.HTTPError as error:
            # A gate or balancer in front of a stopped replica answers 5xx:
            # that is the resting model too, listed from the lease, not woken.
            if is_failure(error.code):
                listed = resting_listing(profile)
                if listed is not None:
                    error.close()
                    return listed
            return upstream_error_response(error)
        except Exception as error:
            listed = resting_listing(profile)
            if listed is not None:
                return listed
            return unavailable(error)

    @blueprint.post("/chat/completions")
    def chat_completions():
        started = time.monotonic()
        identity, denied = authorized()
        if denied:
            return denied
        if request.content_length and request.content_length > MAX_REQUEST_BYTES:
            return jsonify({"error": {
                "message": "The request exceeds the 1 MB gateway limit.",
                "type": "invalid_request_error",
            }}), 413
        body = request.get_data(cache=False)
        try:
            parsed = json.loads(body)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("messages"), list):
                raise ValueError
        except (ValueError, json.JSONDecodeError):
            return jsonify({"error": {
                "message": "Provide a JSON chat-completions request with messages.",
                "type": "invalid_request_error",
            }}), 400
        # Resolve the lease ONCE (B2): the same profile is forwarded to and metered
        # against, so a scale-to-zero/failover flip between two resolves cannot
        # split the request from its accounting, and the broker is hit once.
        try:
            profile = lease()
            upstream = urllib.request.urlopen(
                upstream_request(
                    profile, "/chat/completions",
                    body=merge_stream_usage_option(parsed, body),
                ),
                # Review A12: the request budget, not a fixed 120 s that a
                # long non-streamed answer outlasts (`upstream_answer_seconds`).
                timeout=upstream_answer_seconds(),
            )
        except urllib.error.HTTPError as error:
            # The model ANSWERED, with an error status (ACC-056). A 5xx may be
            # a gate in front of a scaled-to-zero replica, so the wake seam is
            # still asked first; otherwise the model's own error and status go
            # back to the client and into the RED log as they are - a 4xx is
            # the caller's error, not a failure of the model.
            elapsed_ms = (time.monotonic() - started) * 1000
            streaming = bool(parsed.get("stream"))
            woken = waking_response() if is_failure(error.code) else None
            if woken is not None:
                error.close()
                status = 503
                response = woken
            else:
                response = upstream_error_response(error)
                status = response[1]
            metrics.record(status=status, duration_ms=elapsed_ms, streaming=streaming)
            emit_span(parsed, status=status, duration_ms=elapsed_ms, streaming=streaming)
            return response
        except Exception as error:
            elapsed_ms = (time.monotonic() - started) * 1000
            # The error seam (C2): recorded for the RED log and the span, but never
            # metered - a request that produced no upstream completion is not usage.
            metrics.record(
                status=503,
                duration_ms=elapsed_ms,
                streaming=bool(parsed.get("stream")),
            )
            emit_span(
                parsed, status=503, duration_ms=elapsed_ms,
                streaming=bool(parsed.get("stream")),
            )
            woken = waking_response()
            if woken is not None:
                return woken
            return jsonify({"error": {
                "message": "The active cluster model is unavailable.",
                "type": "upstream_error",
                "detail": str(error)[:200],
            }}), 503
        # The deployment dimension is the lease credential id (B1): server-chosen,
        # bounded to one per real deployment, never a client-supplied model string.
        deployment = str(profile.get("credential_id") or "")
        headers = {
            key: value for key, value in upstream.headers.items()
            if key.lower() in FORWARDED_RESPONSE_HEADERS
        }
        if parsed.get("stream"):
            def generate():
                response_bytes = 0
                usage_tail = b""
                try:
                    while chunk := upstream.read(8192):
                        response_bytes += len(chunk)
                        # Every chunk is yielded through as it arrives; only a
                        # bounded raw tail is retained for the usage parse (B3).
                        usage_tail = (usage_tail + chunk)[-USAGE_TAIL_BYTES:]
                        yield chunk
                finally:
                    upstream.close()
                    elapsed_ms = (time.monotonic() - started) * 1000
                    usage = sniff_stream_usage(usage_tail)
                    prompt_tokens = _as_int(usage.get("prompt_tokens"))
                    completion_tokens = _as_int(usage.get("completion_tokens"))
                    metrics.record(
                        status=upstream.status,
                        duration_ms=elapsed_ms,
                        streaming=True,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        response_bytes=response_bytes,
                    )
                    emit_span(
                        parsed, status=upstream.status, duration_ms=elapsed_ms,
                        streaming=True, prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens, response_bytes=response_bytes,
                    )
                    # Terminal seam (C2): one request is metered here on the 2xx
                    # stream close. A mid-stream client disconnect still lands in
                    # this finally, so it records 1 request / 0 tokens - intended,
                    # not a defect, since the request did reach the upstream.
                    meter_usage(
                        identity, deployment, upstream.status,
                        prompt_tokens, completion_tokens,
                    )
            return Response(
                stream_with_context(generate()),
                status=upstream.status,
                headers=headers,
            )
        try:
            content = upstream.read()
            usage = {}
            try:
                usage = json.loads(content).get("usage", {}) or {}
            except (AttributeError, json.JSONDecodeError):
                pass
            elapsed_ms = (time.monotonic() - started) * 1000
            prompt_tokens = _as_int(usage.get("prompt_tokens"))
            completion_tokens = _as_int(usage.get("completion_tokens"))
            metrics.record(
                status=upstream.status,
                duration_ms=elapsed_ms,
                streaming=False,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                response_bytes=len(content),
            )
            emit_span(
                parsed, status=upstream.status, duration_ms=elapsed_ms,
                streaming=False, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens, response_bytes=len(content),
            )
            # Terminal seam (C2): non-stream success. Tokens only when a usage
            # object was actually parsed; the request is counted either way.
            meter_usage(
                identity, deployment, upstream.status,
                prompt_tokens, completion_tokens,
            )
            return Response(content, status=upstream.status, headers=headers)
        finally:
            upstream.close()

    return blueprint


#: What the Settings gateway card may say about the cluster model behind the
#: gateway (ACC-097). One word per situation, each with the owner's sentence;
#: the console maps the word to its badge and shows the sentence as written.
GATEWAY_STATE_SENTENCES = {
    "reachable": "The cluster model is answering requests through this gateway.",
    "asleep": (
        "The cluster model is asleep after sitting idle. The next request "
        "through this gateway wakes it; that first answer waits while the "
        "model loads."
    ),
    "unloaded": (
        "The cluster model was unloaded by hand. A request through this "
        "gateway loads it again, or load it from Cluster > Deployments."
    ),
    "unloaded-unknown": (
        "The cluster model is unloaded, and why could not be read. A request "
        "through this gateway loads it again, or load it from Cluster > "
        "Deployments."
    ),
    "starting": (
        "The cluster model is loading. Requests are answered once it is ready."
    ),
    "not-answering": (
        "The cluster model is not answering right now. Check it under "
        "Cluster > Deployments."
    ),
    "no-model": (
        "No cluster model is set up behind this gateway yet. Deploy one from "
        "Cluster > Deployments."
    ),
    "unknown": "Vaelor could not read which cluster model this gateway uses.",
}


def cluster_model_rest_state(broker, mode_store, cluster_store) -> dict:
    """Whether the cluster model is resting on purpose - READ ONLY, wakes nothing.

    ``{"state": "unloaded", "reason": <cause>}`` for a scaled-to-zero model,
    where the cause is the ONE answer every surface reads
    (`gpu_serving_target.deployment_unload_cause`, VD-136): ``unloaded-idle``,
    ``unloaded-manual``, or ``""`` when the AI Chat lease that tells them apart
    could not be read - never guessed either way. ``{"state": "deploying"}``
    while one loads; ``{}`` when clustering is off or the record cannot be read
    (the live probe then decides).
    """
    from .gpu_cluster_mode import DEPLOYING_STATE, UNLOADED_STATE
    from .gpu_serving_target import deployment_unload_cause, gpu_cluster_mode_active

    try:
        mode_state = mode_store.read()
        name = str(getattr(mode_state, "deployment_name", "") or "")
        if not name or not gpu_cluster_mode_active(mode_state):
            return {}
        record = cluster_store.get_pooled_deployment(name) or {}
    except Exception:  # noqa: BLE001 - absence-ok: the live probe still decides
        return {}
    state = str(record.get("state", "") or "")
    if state == UNLOADED_STATE:
        return {"state": state, "reason": deployment_unload_cause(broker, mode_state, name)}
    if state == DEPLOYING_STATE:
        return {"state": state}
    return {}


def inference_gateway_status(broker, metrics, usage=None, rest_state=None):
    """What the Settings gateway card shows: the target, whether it answers, and its use.

    ACC-097: a scaled-to-zero cluster model read "No active cluster model,
    Unavailable" beside a raw ``Connection refused``, because the probe below
    ran against a model that was resting on purpose and the lease's failure
    text was shown verbatim. The resting state is now read first
    (``rest_state``, :func:`cluster_model_rest_state`) and named, and no raw
    error text reaches the owner.

    ``metrics_24h`` is the GATEWAY'S OWN traffic only, from its roll-up - AI
    Chat records into the same store under its own source and is not external
    API traffic (ACC-046), and the roll-up is complete however far the
    per-request detail has been pruned (ACC-047). ``external_24h`` adds the LLM
    Server's door and the model's own token count
    (:func:`vaelor.usage_collection.external_usage`, ACC-048); ``usage`` is the
    control plane's :class:`~vaelor.usage_collection.UsageCollection`.
    """
    from .gpu_serving_target import no_active_credential
    from .usage_collection import EXTERNAL_WINDOW_SECONDS, external_usage
    from .usage_rollup import SOURCE_GATEWAY

    healthy = False
    target = None
    try:
        profile = broker.resolve_active("cluster-inference")
    except Exception as error:  # noqa: BLE001 - the failure is the answer, named below
        state = "no-model" if no_active_credential(error) else "unknown"
        profile = None
    if profile is not None:
        target = {
            "credential_id": profile["credential_id"],
            "label": profile["label"],
            "base_url": profile["base_url"],
        }
        rest = {}
        if rest_state is not None:
            try:
                rest = rest_state() or {}
            except Exception:  # noqa: BLE001 - absence-ok: the live probe still decides
                rest = {}
        if rest.get("state") == "unloaded":
            state = {"unloaded-idle": "asleep", "unloaded-manual": "unloaded"}.get(
                str(rest.get("reason") or ""), "unloaded-unknown"
            )
        elif rest.get("state") == "deploying":
            state = "starting"
        else:
            try:
                with urllib.request.urlopen(
                    profile["base_url"].rstrip("/") + "/models", timeout=5
                ) as response:
                    healthy = response.status == 200
            except Exception:  # noqa: BLE001 - not answering is the reading
                healthy = False
            state = "reachable" if healthy else "not-answering"
    return {
        "healthy": healthy,
        "state": state,
        "target": target,
        "message": GATEWAY_STATE_SENTENCES[state],
        "metrics_24h": metrics.window_totals(
            EXTERNAL_WINDOW_SECONDS, sources=(SOURCE_GATEWAY,),
        ),
        "external_24h": external_usage(metrics, usage),
        "endpoints": {
            "models": "/inference/v1/models",
            "chat_completions": "/inference/v1/chat/completions",
            "openapi": "/inference/v1/openapi.json",
        },
    }


def gateway_status_reader(runtime):
    """The zero-argument ``inference_gateway_status`` the control plane wires.

    Kept here so the control plane's share is one line: every collaborator is
    read off ``runtime`` when Settings asks, never captured at start-up.
    """
    from .gpu_cluster_mode import ClusterModeStore

    return lambda: inference_gateway_status(
        runtime.credential_broker, runtime.inference_metrics,
        usage=getattr(runtime, "usage", None),
        rest_state=lambda: cluster_model_rest_state(
            runtime.credential_broker, ClusterModeStore(), runtime.cluster.store,
        ),
    )
