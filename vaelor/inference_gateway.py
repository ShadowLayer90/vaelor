"""Authenticated OpenAI-compatible gateway for the active cluster inference service."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from flask import Blueprint, Response, jsonify, request, stream_with_context


MAX_REQUEST_BYTES = 1024 * 1024
FORWARDED_RESPONSE_HEADERS = {"content-type", "cache-control"}


def create_inference_gateway_blueprint(tokens, broker, metrics):
    blueprint = Blueprint(
        "inference_gateway", __name__, url_prefix="/inference/v1"
    )

    def authenticate():
        header = request.headers.get("Authorization", "")
        return tokens.authenticate(
            header[7:] if header.startswith("Bearer ") else "", "inference"
        )

    def lease():
        return broker.resolve_active("cluster-inference")

    def upstream_request(path: str, *, body: bytes | None = None):
        profile = lease()
        headers = {"Accept": request.headers.get("Accept", "application/json")}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if profile.get("api_key"):
            headers["Authorization"] = "Bearer {}".format(profile["api_key"])
        return urllib.request.Request(
            profile["base_url"].rstrip("/") + path,
            data=body,
            headers=headers,
            method="POST" if body is not None else "GET",
        )

    def authorized():
        if authenticate() is None:
            return jsonify({
                "error": {
                    "message": "Invalid or revoked API token.",
                    "type": "authentication_error",
                }
            }), 401
        return None

    @blueprint.get("/openapi.json")
    def openapi():
        denied = authorized()
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

    @blueprint.get("/models")
    def models():
        denied = authorized()
        if denied:
            return denied
        try:
            with urllib.request.urlopen(upstream_request("/models"), timeout=10) as response:
                return Response(
                    response.read(), response.status,
                    content_type=response.headers.get("Content-Type", "application/json"),
                )
        except Exception as error:
            return jsonify({
                "error": {
                    "message": "The active cluster model is unavailable.",
                    "type": "upstream_error",
                    "detail": str(error)[:200],
                }
            }), 503

    @blueprint.post("/chat/completions")
    def chat_completions():
        started = time.monotonic()
        denied = authorized()
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
        try:
            upstream = urllib.request.urlopen(
                upstream_request("/chat/completions", body=body), timeout=120
            )
        except Exception as error:
            metrics.record(
                status=503,
                duration_ms=(time.monotonic() - started) * 1000,
                streaming=bool(parsed.get("stream")),
            )
            return jsonify({"error": {
                "message": "The active cluster model is unavailable.",
                "type": "upstream_error",
                "detail": str(error)[:200],
            }}), 503
        headers = {
            key: value for key, value in upstream.headers.items()
            if key.lower() in FORWARDED_RESPONSE_HEADERS
        }
        if parsed.get("stream"):
            def generate():
                response_bytes = 0
                try:
                    while chunk := upstream.read(8192):
                        response_bytes += len(chunk)
                        yield chunk
                finally:
                    upstream.close()
                    metrics.record(
                        status=upstream.status,
                        duration_ms=(time.monotonic() - started) * 1000,
                        streaming=True,
                        response_bytes=response_bytes,
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
                usage = json.loads(content).get("usage", {})
            except (AttributeError, json.JSONDecodeError):
                pass
            metrics.record(
                status=upstream.status,
                duration_ms=(time.monotonic() - started) * 1000,
                streaming=False,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                response_bytes=len(content),
            )
            return Response(content, status=upstream.status, headers=headers)
        finally:
            upstream.close()

    return blueprint


def inference_gateway_status(broker, metrics):
    try:
        profile = broker.resolve_active("cluster-inference")
        with urllib.request.urlopen(
            profile["base_url"].rstrip("/") + "/models", timeout=5
        ) as response:
            healthy = response.status == 200
        target = {
            "credential_id": profile["credential_id"],
            "label": profile["label"],
            "base_url": profile["base_url"],
        }
        message = "The active cluster inference target is reachable."
    except Exception as error:
        healthy = False
        target = None
        message = str(error)[:200]
    return {
        "healthy": healthy,
        "target": target,
        "message": message,
        "metrics_24h": metrics.snapshot(),
        "endpoints": {
            "models": "/inference/v1/models",
            "chat_completions": "/inference/v1/chat/completions",
            "openapi": "/inference/v1/openapi.json",
        },
    }
