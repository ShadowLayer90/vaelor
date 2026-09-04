"""Out-of-band alert delivery: email and webhook, with injectable transports.

When an automation trigger fires, the control plane already creates a durable
assistant task and records the run. Nothing, historically, told a human. This
module is the missing half: pure, side-effect-contained delivery of a fired
alert to the channels an administrator configured.

The design goal is honesty under partial failure. A single channel failing must
not stop the others, and delivery must never raise back into the trigger
evaluator - a broken SMTP relay cannot be allowed to lose the task. Every entry
point returns a structured result; the transports are seams so unit tests never
open a socket.
"""

from __future__ import annotations

import json
import smtplib
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Any, Callable, Dict, List, Optional


#: How a channel is secured on the wire. ``none`` is offered because a
#: loopback relay on the appliance itself needs no transport security; the two
#: encrypted modes cover a real mail provider.
EMAIL_SECURITIES = ("starttls", "ssl", "none")
DEFAULT_TIMEOUT_SECONDS = 15


@dataclass
class DeliveryResult:
    """One channel's outcome, safe to persist and to show an administrator."""

    channel_id: str
    kind: str
    delivered: bool
    error: str = ""
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "kind": self.kind,
            "delivered": self.delivered,
            "error": self.error,
            "detail": self.detail,
        }


#: An email transport: given the resolved settings and a built message, it
#: sends and returns nothing, raising on failure. Injected in tests so no
#: socket is opened.
EmailSender = Callable[["EmailEnvelope"], None]
#: A webhook transport: given a request, it returns ``(status_code, body)`` and
#: raises only on a transport error, not on an HTTP status.
WebhookTransport = Callable[["WebhookRequest"], "WebhookResponse"]


@dataclass
class EmailEnvelope:
    host: str
    port: int
    security: str
    username: str
    password: str
    message: EmailMessage
    timeout: int = DEFAULT_TIMEOUT_SECONDS


@dataclass
class WebhookRequest:
    url: str
    body: bytes
    headers: Dict[str, str]
    timeout: int = DEFAULT_TIMEOUT_SECONDS


@dataclass
class WebhookResponse:
    status: int
    body: str = ""


@dataclass
class Senders:
    """The two transport seams, defaulting to the real stdlib implementations."""

    email: EmailSender = field(default=None)  # type: ignore[assignment]
    webhook: WebhookTransport = field(default=None)  # type: ignore[assignment]

    def resolved_email(self) -> EmailSender:
        return self.email or smtp_send

    def resolved_webhook(self) -> WebhookTransport:
        return self.webhook or http_post


def _alert_lines(alert: Dict[str, Any]) -> List[str]:
    """Human-readable summary lines shared by both channels' text bodies."""
    return [
        "Trigger: {}".format(alert.get("trigger_name", "")),
        "Signal: {} {} {}".format(
            alert.get("source", ""),
            alert.get("operator", ""),
            alert.get("threshold", ""),
        ),
        "Observed value: {}".format(alert.get("observed_value", "")),
        "Diagnostic task: {}".format(alert.get("task_id", "") or "not recorded"),
        "Fired at (epoch seconds): {}".format(alert.get("timestamp", "")),
    ]


def build_alert_message(channel: Dict[str, Any], alert: Dict[str, Any]) -> EmailMessage:
    """Compose the notification email for one email channel."""
    message = EmailMessage()
    message["Subject"] = "Vaelor alert: {}".format(alert.get("trigger_name", "appliance"))
    message["From"] = channel.get("from_address", "")
    message["To"] = channel.get("to_address", "")
    message.set_content(
        "An appliance alert rule fired and launched a read-only diagnostic.\n\n"
        + "\n".join(_alert_lines(alert))
        + "\n\nThis message was sent by Vaelor because you configured this "
        "delivery channel. It reports; it does not ask you to act on a link."
    )
    return message


def build_webhook_body(alert: Dict[str, Any]) -> bytes:
    """The JSON payload posted to a webhook channel."""
    return json.dumps(
        {
            "trigger_name": alert.get("trigger_name", ""),
            "source": alert.get("source", ""),
            "operator": alert.get("operator", ""),
            "threshold": alert.get("threshold"),
            "observed_value": alert.get("observed_value"),
            "timestamp": alert.get("timestamp"),
            "task_id": alert.get("task_id", ""),
        },
        separators=(",", ":"),
    ).encode("utf-8")


def smtp_send(envelope: EmailEnvelope) -> None:
    """Real SMTP transport. STARTTLS, implicit SSL, or plain per ``security``."""
    context = ssl.create_default_context()
    if envelope.security == "ssl":
        with smtplib.SMTP_SSL(
            envelope.host, envelope.port, timeout=envelope.timeout, context=context
        ) as client:
            if envelope.username:
                client.login(envelope.username, envelope.password)
            client.send_message(envelope.message)
        return
    with smtplib.SMTP(envelope.host, envelope.port, timeout=envelope.timeout) as client:
        client.ehlo()
        if envelope.security == "starttls":
            client.starttls(context=context)
            client.ehlo()
        if envelope.username:
            client.login(envelope.username, envelope.password)
        client.send_message(envelope.message)


def http_post(request: WebhookRequest) -> WebhookResponse:
    """Real webhook transport. Returns the status; only raises on transport."""
    urllib_request = urllib.request.Request(
        request.url, data=request.body, headers=request.headers, method="POST"
    )
    with urllib.request.urlopen(urllib_request, timeout=request.timeout) as response:
        raw = response.read(65536)
        return WebhookResponse(
            status=int(getattr(response, "status", 0) or 0),
            body=raw.decode("utf-8", "replace"),
        )


def deliver_email(
    channel: Dict[str, Any],
    secret: Optional[str],
    alert: Dict[str, Any],
    sender: Optional[EmailSender] = None,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> DeliveryResult:
    """Deliver one alert over one email channel, capturing every failure."""
    channel_id = str(channel.get("id", ""))
    result = DeliveryResult(channel_id, "email", False)
    host = str(channel.get("smtp_host", "")).strip()
    if not host:
        result.error = "no_smtp_host"
        return result
    security = str(channel.get("security", "starttls")).strip().lower()
    if security not in EMAIL_SECURITIES:
        result.error = "unsupported_security: {}".format(security)
        return result
    envelope = EmailEnvelope(
        host=host,
        port=int(channel.get("smtp_port") or (465 if security == "ssl" else 587)),
        security=security,
        username=str(channel.get("username", "")),
        password=str(secret or ""),
        message=build_alert_message(channel, alert),
        timeout=timeout,
    )
    try:
        (sender or smtp_send)(envelope)
    except (smtplib.SMTPException, OSError, ValueError) as error:
        result.error = "smtp_failed: {}".format(str(error)[:200])
        return result
    result.delivered = True
    result.detail = "Sent to {}".format(channel.get("to_address", ""))
    return result


def deliver_webhook(
    channel: Dict[str, Any],
    secret: Optional[str],
    alert: Dict[str, Any],
    transport: Optional[WebhookTransport] = None,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> DeliveryResult:
    """Deliver one alert over one webhook channel, capturing every failure."""
    channel_id = str(channel.get("id", ""))
    result = DeliveryResult(channel_id, "webhook", False)
    url = str(channel.get("url", "")).strip()
    if not url.startswith(("http://", "https://")):
        result.error = "invalid_url"
        return result
    headers = {"Content-Type": "application/json", "User-Agent": "Vaelor-Alert/1"}
    header_name = str(channel.get("auth_header", "")).strip()
    if header_name and secret:
        headers[header_name] = str(secret)
    request = WebhookRequest(
        url=url, body=build_webhook_body(alert), headers=headers, timeout=timeout
    )
    try:
        response = (transport or http_post)(request)
    except (urllib.error.URLError, OSError, ValueError) as error:
        result.error = "request_failed: {}".format(str(error)[:200])
        return result
    if not 200 <= int(response.status) < 300:
        result.error = "http_status: {}".format(response.status)
        return result
    result.delivered = True
    result.detail = "HTTP {}".format(response.status)
    return result


def deliver_alert(
    channels: List[Dict[str, Any]],
    alert: Dict[str, Any],
    resolve_secret: Callable[[str], Optional[str]],
    senders: Optional[Senders] = None,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> List[DeliveryResult]:
    """Fan a fired alert out to every enabled channel, one result per channel.

    ``resolve_secret`` maps a channel's stored broker purpose to its secret
    (SMTP password or webhook token). It, the secret lookup, and each channel's
    own delivery are each isolated: one channel's failure - a missing secret, a
    raised transport, a non-2xx status - is recorded and the fan-out continues.
    Nothing here raises; the caller (the trigger evaluator) gets results, never
    an exception.
    """
    senders = senders or Senders()
    results: List[DeliveryResult] = []
    for channel in channels:
        channel_id = str(channel.get("id", ""))
        kind = str(channel.get("kind", ""))
        if not channel.get("enabled", True):
            continue
        try:
            secret = _resolve_channel_secret(channel, resolve_secret)
        except _SecretUnavailable as error:
            results.append(DeliveryResult(channel_id, kind, False, error=str(error)))
            continue
        try:
            if kind == "email":
                results.append(
                    deliver_email(
                        channel, secret, alert,
                        senders.resolved_email(), timeout=timeout,
                    )
                )
            elif kind == "webhook":
                results.append(
                    deliver_webhook(
                        channel, secret, alert,
                        senders.resolved_webhook(), timeout=timeout,
                    )
                )
            else:
                results.append(
                    DeliveryResult(channel_id, kind, False, error="unsupported_channel")
                )
        except Exception as error:  # noqa: BLE001 - delivery must never escape
            results.append(
                DeliveryResult(
                    channel_id, kind, False,
                    error="delivery_error: {}".format(str(error)[:200]),
                )
            )
    return results


class _SecretUnavailable(Exception):
    """A channel needs a secret and the broker could not supply it."""


def _resolve_channel_secret(
    channel: Dict[str, Any], resolve_secret: Callable[[str], Optional[str]]
) -> Optional[str]:
    purpose = str(channel.get("secret_purpose", "")).strip()
    # A channel that never had a secret stored (an open webhook, or an SMTP
    # relay that needs no login) is delivered without one rather than blocked.
    # ``has_secret`` is only ever explicitly False for such a channel; absent it
    # (e.g. a caller passing a bare dict) a set purpose is still resolved.
    if not purpose or channel.get("has_secret") is False:
        return None
    try:
        secret = resolve_secret(purpose)
    except Exception as error:  # noqa: BLE001 - broker failure is a channel failure
        raise _SecretUnavailable(
            "secret_unavailable: {}".format(str(error)[:160])
        ) from error
    if not secret:
        raise _SecretUnavailable("secret_missing")
    return secret


def summarize_results(results: List[DeliveryResult]) -> str:
    """A compact, persistable one-line summary of a fan-out."""
    if not results:
        return "no channels configured"
    return " · ".join(
        "{} {}".format(
            item.kind,
            "delivered" if item.delivered else "failed ({})".format(item.error),
        )
        for item in results
    )[:480]
