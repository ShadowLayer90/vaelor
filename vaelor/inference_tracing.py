"""Stdlib OTLP/HTTP trace emitter for the inference gateway (VD-128, Phase E').

**Why the standard library, not the OpenTelemetry SDK.** A per-request trace to
Arize Phoenix needs only to POST one small OTLP/HTTP document; the OTel Python SDK
is a heavy dependency, and a ``--force-reinstall --no-deps`` wheel deploy would
skip a newly-declared dep and silently honest-degrade the feature (LESSONS: wheel
--no-deps skips new deps). So the span is built and POSTed here with
:mod:`urllib.request` and :mod:`struct` alone - no dependency to forget to
install, and the OTLP wire shape is small enough to construct by hand.

**Protobuf, not JSON.** OTLP/HTTP defines both a protobuf and a JSON encoding, but
Phoenix's collector accepts only ``application/x-protobuf`` - a live POST of the
JSON form to ``/v1/traces`` is refused ``415 Unsupported content type``. So the
one span is serialized to the OTLP ``ExportTraceServiceRequest`` protobuf by a
tiny hand-rolled encoder (:func:`encode_traces_protobuf`) - the schema is fixed
and small, so a targeted encoder is far lighter than pulling protobuf + the SDK,
and it is verified against a real Phoenix (the span lands in the default project).

**The one hard rule: tracing is NEVER allowed to touch the inference response.**
A span is emitted only when tracing is configured AND enabled; otherwise every
call here is a no-op. When it is enabled, the POST is fire-and-forget on a daemon
thread with a short timeout, and every failure - the endpoint down, a timeout, a
non-2xx, a malformed reply - is swallowed. The gateway calls :meth:`TraceEmitter.emit`
right after it records the request, and nothing that happens here can raise into
the request path or delay the bytes already returned to the caller.

**What "configured+enabled" means.** The emitter holds a ``config`` callable that
returns the Phoenix OTLP endpoint URL when tracing is on, or an empty string when
it is off. The callable is read afresh on each request, so an admin enabling or
disabling Phoenix (which flips the persisted flag :mod:`vaelor.phoenix_state`
governs) takes effect on the next request with no gateway restart. An empty or
unreadable config is the disabled state, so a missing Phoenix is simply silent.
"""

from __future__ import annotations

import secrets
import struct
import threading
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional

#: The span every inference request produces, and the instrumentation scope it is
#: reported under. One name so Phoenix groups them and the gateway cannot spell a
#: second.
SPAN_NAME = "llm.chat"
INSTRUMENTATION_SCOPE = "vaelor.inference_tracing"

#: The resource ``service.name`` all spans carry, so Phoenix files them under one
#: service rather than an anonymous default.
SERVICE_NAME = "vaelor-inference-gateway"

#: OTLP span kind CLIENT (the gateway is the client of the upstream model) and the
#: status codes, as the OTLP/JSON proto enums. Unset(0)/Ok(1)/Error(2).
_SPAN_KIND_CLIENT = 3
_STATUS_OK = 1
_STATUS_ERROR = 2

#: How long the trace POST may take before it is abandoned. Short, because the
#: request path must never wait on the collector; the POST runs off-thread anyway,
#: but a bounded timeout stops a hung Phoenix from leaking daemon threads.
DEFAULT_TIMEOUT_SECONDS = 2.0

#: An HTTP status at or above this marks the span Error. The gateway's span is a
#: CLIENT span (the gateway calling the model), and OpenTelemetry's HTTP
#: convention marks a client span Error on any 4xx or 5xx - so a request the
#: model rejected is still a red span in Phoenix. The RED error rate is a
#: different question (did the serving side fail?) and counts 5xx only, by
#: ``inference_metrics.is_failure``; the 4xx it leaves out is its client-error
#: count, which is where this span's extra Error cases show up.
_ERROR_STATUS = 400


def _attribute(key: str, value: Any) -> Dict[str, Any]:
    """One OTLP key/value attribute, typed by the Python value.

    OTLP/JSON tags each value with its type (``stringValue``/``intValue``/
    ``boolValue``); int64 is JSON-encoded as a STRING per proto3 JSON, so an
    integer is stringified here rather than left a JSON number.
    """
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    return {"key": key, "value": {"stringValue": str(value)}}


def build_span(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    response_bytes: int,
    streaming: bool,
    status: int,
    start_ns: int,
    end_ns: int,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
) -> Dict[str, Any]:
    """One OTLP span for a single inference request, as a plain JSON-able dict.

    The trace/span ids are hex per OTLP/JSON (16 bytes / 8 bytes), random when the
    caller passes none, so each request is its own trace. The attributes are the
    facts the gateway already has - the served model, the token counts, the bytes
    streamed back, whether it streamed, and the HTTP status - under the ``llm.*``
    and ``http.*`` keys Phoenix understands. The span status is Error for a
    ``>= 400`` status so a failed request reads as a failed span.
    """
    return {
        "traceId": trace_id or secrets.token_hex(16),
        "spanId": span_id or secrets.token_hex(8),
        "name": SPAN_NAME,
        "kind": _SPAN_KIND_CLIENT,
        "startTimeUnixNano": str(int(start_ns)),
        "endTimeUnixNano": str(int(end_ns)),
        "attributes": [
            # OpenInference span kind: Phoenix files a span in its LLM view by
            # this attribute, not the OTLP kind (which its collector leaves
            # "unknown"). Without it a chat request shows as a generic span.
            _attribute("openinference.span.kind", "LLM"),
            _attribute("llm.model", str(model or "")),
            _attribute("llm.prompt_tokens", max(0, int(prompt_tokens))),
            _attribute("llm.completion_tokens", max(0, int(completion_tokens))),
            _attribute("llm.response_bytes", max(0, int(response_bytes))),
            _attribute("llm.streaming", bool(streaming)),
            _attribute("http.status", int(status)),
        ],
        "status": {
            "code": _STATUS_ERROR if int(status) >= _ERROR_STATUS else _STATUS_OK
        },
    }


# --- OTLP protobuf wire encoding (stdlib) -------------------------------------
# Just enough of the protobuf wire format to serialize ONE trace's
# `ExportTraceServiceRequest`. Field numbers are the OTLP trace.proto ones; a
# wrong number is a span Phoenix silently drops, so they are verified against a
# live Phoenix (the span appears in the default project). Wire types: 0 varint,
# 1 fixed64, 2 length-delimited.


def _pb_varint(value: int) -> bytes:
    out = bytearray()
    value = int(value)
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _pb_len(field: int, data: bytes) -> bytes:
    """A length-delimited field (string, bytes, or embedded message)."""
    return _pb_varint((field << 3) | 2) + _pb_varint(len(data)) + data


def _pb_string(field: int, text: str) -> bytes:
    return _pb_len(field, text.encode("utf-8"))


def _pb_varint_field(field: int, value: int) -> bytes:
    return _pb_varint((field << 3) | 0) + _pb_varint(value)


def _pb_fixed64(field: int, value: int) -> bytes:
    return _pb_varint((field << 3) | 1) + struct.pack("<Q", int(value) & 0xFFFFFFFFFFFFFFFF)


def _pb_any_value(value: Dict[str, Any]) -> bytes:
    """One OTLP ``AnyValue`` from a span attribute's typed JSON value.

    The span dict carries ``{"boolValue": …}``/``{"intValue": "n"}``/
    ``{"stringValue": …}`` (proto3-JSON shapes); this maps each to the matching
    ``AnyValue`` field (string=1, bool=2, int=3). int64 arrived as a string.
    """
    if "boolValue" in value:
        return _pb_varint_field(2, 1 if value["boolValue"] else 0)
    if "intValue" in value:
        return _pb_varint_field(3, int(value["intValue"]))
    return _pb_string(1, str(value.get("stringValue", "")))


def _pb_key_value(attribute: Dict[str, Any]) -> bytes:
    """One OTLP ``KeyValue`` (key=1, value=2)."""
    return _pb_string(1, str(attribute["key"])) + _pb_len(2, _pb_any_value(attribute["value"]))


def _pb_span(span: Dict[str, Any]) -> bytes:
    parts: List[bytes] = [
        _pb_len(1, bytes.fromhex(span["traceId"])),   # trace_id
        _pb_len(2, bytes.fromhex(span["spanId"])),    # span_id
        _pb_string(5, str(span["name"])),             # name
        _pb_varint_field(6, int(span.get("kind", 0))),  # kind
        _pb_fixed64(7, int(span["startTimeUnixNano"])),  # start_time_unix_nano
        _pb_fixed64(8, int(span["endTimeUnixNano"])),    # end_time_unix_nano
    ]
    for attribute in span.get("attributes", []):
        parts.append(_pb_len(9, _pb_key_value(attribute)))  # attributes
    code = int((span.get("status") or {}).get("code", 0))
    if code:
        parts.append(_pb_len(15, _pb_varint_field(3, code)))  # status{code=3}
    return b"".join(parts)


def encode_traces_protobuf(
    span: Dict[str, Any], *, service_name: str = SERVICE_NAME,
    scope: str = INSTRUMENTATION_SCOPE,
) -> bytes:
    """Serialize one span as the OTLP ``ExportTraceServiceRequest`` protobuf body.

    The envelope Phoenix's ``/v1/traces`` collector accepts: one ``ResourceSpans``
    carrying ``service.name``, one ``ScopeSpans`` naming this instrumentation, one
    span. Built by hand rather than by the OTel SDK for the reason in the module
    docstring, and returned as the raw bytes the POST sends verbatim.
    """
    scope_spans = _pb_len(1, _pb_string(1, scope)) + _pb_len(2, _pb_span(span))
    resource = _pb_len(1, _pb_key_value(_attribute("service.name", service_name)))
    resource_spans = _pb_len(1, resource) + _pb_len(2, scope_spans)
    return _pb_len(1, resource_spans)


def _http_poster(timeout: float) -> Callable[[str, bytes], None]:
    """A poster that POSTs the OTLP protobuf to ``endpoint`` over HTTP with a timeout.

    A closure so the timeout is bound once. It may raise (a down collector, a
    timeout, a bad URL); :meth:`TraceEmitter.emit` runs it behind a swallow, so a
    raise here never reaches the gateway. The content type is
    ``application/x-protobuf`` because Phoenix's collector refuses JSON.
    """

    def post(endpoint: str, body: bytes) -> None:
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/x-protobuf"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()  # drain so the connection can be reused/closed cleanly

    return post


def _daemon_spawn(run: Callable[[], None]) -> None:
    """Run ``run`` on a throwaway daemon thread, so the POST never blocks the caller.

    Daemon so a hung POST cannot keep the process alive at shutdown. A default
    seam a test replaces with a synchronous runner to assert on the POST without
    a thread.
    """
    threading.Thread(target=run, name="vaelor-trace-emit", daemon=True).start()


class TraceEmitter:
    """Emit one OTLP span per inference request to Phoenix, non-fatally.

    Constructed once and shared by the gateway. ``config`` is a callable returning
    the Phoenix OTLP endpoint URL when tracing is on, or ``""`` when off - read
    afresh on each request so an admin toggle takes effect immediately. The POST
    is fire-and-forget (``spawn``) with every failure swallowed, so a tracing
    problem can neither raise into nor slow the inference response. ``poster`` and
    ``spawn`` are injectable seams so the emitter is unit-testable with no network
    and no thread.
    """

    def __init__(
        self,
        config: Callable[[], str],
        *,
        poster: Optional[Callable[[str, bytes], None]] = None,
        spawn: Optional[Callable[[Callable[[], None]], None]] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        service_name: str = SERVICE_NAME,
    ):
        self._config = config
        self._poster = poster or _http_poster(timeout)
        self._spawn = spawn or _daemon_spawn
        self._service_name = service_name

    def endpoint(self) -> str:
        """The configured OTLP endpoint, or ``""`` when tracing is off/unreadable.

        Fail-safe: a config callable that raises is treated as disabled, so a
        broken config silences tracing rather than breaking a request.
        """
        try:
            return str(self._config() or "").strip()
        except Exception:  # noqa: BLE001 - an unreadable config is the disabled state
            return ""

    @property
    def enabled(self) -> bool:
        """Whether a span would be emitted right now (a non-empty endpoint)."""
        return bool(self.endpoint())

    def emit(
        self,
        *,
        model: str,
        status: int,
        duration_ms: float,
        streaming: bool,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        response_bytes: int = 0,
    ) -> bool:
        """Emit a span for one request, or do nothing when tracing is off.

        Returns whether a span was dispatched (a test signal; the gateway ignores
        it). A no-op when the endpoint is empty. The span's window is the recorded
        duration, ending now, so Phoenix shows the same latency the RED store did.
        Everything past the enabled gate is wrapped: building or dispatching the
        span can never raise into the gateway.
        """
        endpoint = self.endpoint()
        if not endpoint:
            return False
        try:
            end_ns = time.time_ns()
            start_ns = end_ns - int(max(0.0, float(duration_ms)) * 1_000_000)
            span = build_span(
                model=model, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens, response_bytes=response_bytes,
                streaming=streaming, status=status,
                start_ns=start_ns, end_ns=end_ns,
            )
            body = encode_traces_protobuf(span, service_name=self._service_name)
            self._spawn(lambda: self._safe_post(endpoint, body))
            return True
        except Exception:  # noqa: BLE001 - tracing must never affect the response
            return False

    def _safe_post(self, endpoint: str, body: bytes) -> None:
        """POST the span, swallowing every failure.

        Runs off the request thread. A collector that is down, slow, or rejecting
        the body drops this span best-effort; the request has already returned.
        """
        try:
            self._poster(endpoint, body)
        except Exception:  # noqa: BLE001 - a dropped span is the honest failure mode
            pass
