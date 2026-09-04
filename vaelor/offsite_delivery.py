"""Push a finished backup archive to a configured off-site destination.

The scheduler in :mod:`vaelor.backup_schedule` writes an encrypted ``.vaelor``
archive to ``export_root`` and then, if an off-site target is configured, asks
this module to copy that archive somewhere durable off the box. Two backends
ship: an S3-compatible object PUT (AWS Signature Version 4, which MinIO, Ceph,
Backblaze B2's S3 API, and AWS itself all accept) and a generic HTTPS PUT to a
webhook-style endpoint. Adding a third backend is adding one entry to
``_BACKENDS`` and one request builder; nothing else changes.

Three seams keep this testable and honest:

* **Transport.** Every network call goes through an injected ``transport``
  callable. Production wires :func:`http_put_transport` (stdlib ``urllib``);
  a test wires a fake that records the request and returns a status, so no test
  opens a socket.
* **Clock.** SigV4 stamps a request time. Production reads the wall clock; a
  test injects a fixed ``now`` so the signature is deterministic.
* **No secret in the config.** Destination credentials are resolved by the
  caller (through the credential broker) and handed in as ``credentials``. This
  module never reads a secret from the on-disk config and never logs one.

Every path returns a structured :class:`dict` (``ok``/``backend``/``status``/
``detail``) rather than raising for a rejected upload, so a failed push is a
recorded fact, not a crash. Only a malformed *configuration* raises
:class:`OffsiteError`, because that is a programming or setup error the caller
must fix, not a runtime delivery outcome.
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 60
_MAX_DETAIL = 300
#: Transport contract: (method, url, body, headers, timeout) -> (status, detail).
#: A status of ``0`` means the destination could not be reached at all; any
#: other value is an HTTP status the destination actually returned.
Transport = Callable[[str, str, bytes, Dict[str, str], float], Tuple[int, str]]


class OffsiteError(ValueError):
    """A malformed off-site *configuration*. Never a delivery outcome."""


def http_put_transport(
    method: str,
    url: str,
    body: bytes,
    headers: Dict[str, str],
    timeout: float,
) -> Tuple[int, str]:
    """The production transport: a single bounded ``urllib`` request.

    A non-2xx response is a delivery outcome, not an exception, so an
    ``HTTPError`` is unwrapped to its status. A connection that never completed
    (DNS, refused, timeout, TLS) is reported as status ``0`` with a redacted
    reason; the exception itself is swallowed so a network blip cannot escape as
    a crash inside the scheduler loop.
    """
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), ""
    except urllib.error.HTTPError as error:
        try:
            detail = error.read(_MAX_DETAIL).decode("utf-8", "replace")
        except OSError:
            detail = ""
        return int(error.code), detail
    except (urllib.error.URLError, TimeoutError, OSError):
        # Deliberately opaque: the message can carry the endpoint host, and this
        # string is stored in the run history the UI shows.
        return 0, "The off-site destination could not be reached."


def _clean_str(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def normalize_offsite_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate and normalize a stored off-site target. No secrets live here.

    Returns ``{}`` for an unconfigured target so callers can treat "no
    off-site" as a first-class, non-error state.
    """
    if not config:
        return {}
    if not isinstance(config, dict):
        raise OffsiteError("The off-site target configuration is invalid.")
    backend = _clean_str(config.get("backend"), 32).lower()
    if not backend:
        return {}
    if backend not in _BACKENDS:
        raise OffsiteError("This off-site backend is not supported.")
    endpoint = _clean_str(config.get("endpoint"), 500).rstrip("/")
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.hostname or parsed.query or parsed.fragment:
        raise OffsiteError("Enter a complete HTTPS endpoint URL with no query text.")
    try:
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise OffsiteError("The off-site endpoint port is invalid.")
    except ValueError as error:
        raise OffsiteError("The off-site endpoint port is invalid.") from error
    normalized = {
        "backend": backend,
        "endpoint": endpoint,
        "prefix": _clean_str(config.get("prefix"), 200).strip("/"),
        "credential_purpose": _clean_str(config.get("credential_purpose"), 80),
    }
    if backend == "s3":
        bucket = _clean_str(config.get("bucket"), 128)
        if not bucket:
            raise OffsiteError("An S3-compatible target needs a bucket name.")
        normalized["bucket"] = bucket
        normalized["region"] = _clean_str(config.get("region"), 64) or "us-east-1"
    return normalized


def _credentials_dict(credentials: Any) -> Dict[str, str]:
    """Accept either a broker lease dict or a raw JSON/opaque token string."""
    if isinstance(credentials, dict):
        token = credentials.get("token")
        if isinstance(token, str) and token.strip():
            return _credentials_dict(token)
        return {str(key): str(value) for key, value in credentials.items()}
    text = str(credentials or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return {"token": text}
    if isinstance(parsed, dict):
        return {str(key): str(value) for key, value in parsed.items()}
    return {"token": text}


def _object_name(prefix: str, archive_name: str) -> str:
    safe = Path(str(archive_name)).name
    if not safe or safe in {".", ".."}:
        raise OffsiteError("The archive name is invalid for off-site delivery.")
    return f"{prefix}/{safe}".strip("/") if prefix else safe


def _redact(url: str) -> str:
    """Drop any query string before a URL reaches the run history."""
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, "", "")
    )


def _sigv4_headers(
    *,
    method: str,
    url: str,
    body: bytes,
    region: str,
    credentials: Dict[str, str],
    now: datetime.datetime,
) -> Dict[str, str]:
    access_key = credentials.get("access_key_id") or credentials.get("access_key")
    secret_key = credentials.get("secret_access_key") or credentials.get("secret_key")
    if not access_key or not secret_key:
        raise OffsiteError("The S3 credential needs an access key id and secret.")
    parsed = urllib.parse.urlsplit(url)
    host = parsed.netloc
    # The URL path is ALREADY percent-encoded (the key was quoted when the URL
    # was built). Re-quoting here would turn each '%' into '%25', so the signed
    # canonical URI would diverge from the path actually sent and S3 would
    # compute a different signature - a 403 on every key with an encodable
    # character (a space or punctuation in the admin's prefix). Sign the path
    # exactly as it goes on the wire.
    canonical_uri = parsed.path or "/"
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(body).hexdigest()
    headers = {
        "host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
    }
    session_token = credentials.get("session_token") or credentials.get("security_token")
    if session_token:
        headers["x-amz-security-token"] = session_token
    signed_headers = ";".join(sorted(headers))
    canonical_headers = "".join(
        f"{key}:{headers[key]}\n" for key in sorted(headers)
    )
    canonical_request = "\n".join(
        [method, canonical_uri, "", canonical_headers, signed_headers, payload_hash]
    )
    scope = f"{date_stamp}/{region}/s3/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )

    def _sign(key: bytes, message: str) -> bytes:
        return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()

    signing_key = _sign(
        _sign(
            _sign(_sign(f"AWS4{secret_key}".encode("utf-8"), date_stamp), region),
            "s3",
        ),
        "aws4_request",
    )
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    headers["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return headers


def _s3_request(
    config: Dict[str, Any], name: str, body: bytes, credentials: Dict[str, str],
    now: datetime.datetime,
) -> Tuple[str, Dict[str, str]]:
    key = _object_name(config.get("prefix", ""), name)
    url = f"{config['endpoint']}/{config['bucket']}/{urllib.parse.quote(key, safe='/~')}"
    headers = _sigv4_headers(
        method="PUT", url=url, body=body, region=config["region"],
        credentials=credentials, now=now,
    )
    headers["Content-Type"] = "application/octet-stream"
    return url, headers


def _webhook_request(
    config: Dict[str, Any], name: str, body: bytes, credentials: Dict[str, str],
    now: datetime.datetime,
) -> Tuple[str, Dict[str, str]]:
    key = _object_name(config.get("prefix", ""), name)
    url = f"{config['endpoint']}/{urllib.parse.quote(key, safe='/~')}"
    headers = {"Content-Type": "application/octet-stream"}
    token = credentials.get("token") or credentials.get("bearer")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return url, headers


_BACKENDS: Dict[str, Any] = {"s3": _s3_request, "webhook": _webhook_request}


def deliver_archive(
    archive_path: Path | str,
    config: Optional[Dict[str, Any]],
    credentials: Any = None,
    *,
    now: Optional[float] = None,
    transport: Transport = http_put_transport,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Deliver one archive to the configured off-site target.

    Returns a structured result. ``ok`` is true only on a 2xx response. A
    rejected upload, an unreachable host, or a missing credential secret all
    come back as ``ok=False`` with a ``detail``; the local archive is never
    touched here, so a failed push always leaves the on-disk copy intact.
    """
    normalized = normalize_offsite_config(config)
    if not normalized:
        return {"ok": False, "backend": "", "status": 0, "detail": "No off-site target is configured.", "url": ""}
    backend = normalized["backend"]
    source = Path(archive_path)
    try:
        size = source.stat().st_size
    except OSError:
        return {"ok": False, "backend": backend, "status": 0, "detail": "The archive to deliver is missing.", "url": ""}
    if size > MAX_ARCHIVE_BYTES:
        return {"ok": False, "backend": backend, "status": 0, "detail": "The archive exceeds the off-site size limit.", "url": ""}
    body = source.read_bytes()
    stamp = datetime.datetime.fromtimestamp(
        now if now is not None else time.time(),
        tz=datetime.timezone.utc,
    )
    try:
        url, headers = _BACKENDS[backend](
            normalized, source.name, body, _credentials_dict(credentials), stamp
        )
    except OffsiteError as error:
        return {"ok": False, "backend": backend, "status": 0, "detail": str(error), "url": ""}
    headers.setdefault("Content-Length", str(len(body)))
    status, detail = transport("PUT", url, body, headers, timeout)
    ok = 200 <= status < 300
    return {
        "ok": ok,
        "backend": backend,
        "status": status,
        "detail": (detail or ("Uploaded." if ok else f"HTTP {status}"))[:_MAX_DETAIL],
        "url": _redact(url),
    }
