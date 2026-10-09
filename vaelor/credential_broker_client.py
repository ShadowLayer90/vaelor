"""JSON-over-Unix-socket client for the credential broker, and its wire contract.

The client is the leaf of the broker's three modules: :mod:`vaelor.credential_broker`
(the encrypted vault) and :mod:`vaelor.credential_broker_server` (the socket
server) both sit above it, so the protocol vocabulary the two sides must agree
on - the error type a verb answers with, the one-line frame's size limit and the
sentence for a frame over it - is declared here, where every importer can reach
it without a cycle, and re-exported by the vault module under its old spelling.
Both sides read the same names, so a client that refuses a frame the server
would also refuse is one rule, not two.

The client never offers a plaintext ``get``: a secret only ever leaves the
broker as a purpose-bound lease through ``resolve``.
"""

from __future__ import annotations

import json
import socket
from typing import Any, Optional

from .runtime_paths import env_value, run_path


MAX_REQUEST_BYTES = 16384


class CredentialError(ValueError):
    """Safe error suitable for returning through the local broker protocol."""


#: A frame over the one-line limit, refused by the client before it is sent and
#: by the server before it is parsed - one sentence, both sides.
REQUEST_TOO_LARGE = "Credential broker request is too large."

#: The socket could not be reached or the exchange broke: the broker is down.
BROKER_UNAVAILABLE = "Credential broker is unavailable."

#: The socket answered, but with nothing a client can read - an empty frame or
#: one that is not JSON. The first live instance (VD-127, 2026-09-05) was a
#: server-side ``FileNotFoundError`` inside ``vault.list`` that escaped the
#: handler and closed the connection with no body; a ``JSONDecodeError`` on
#: ``''`` then surfaced as an HTTP 500 on the AI Chat setup page. Named apart
#: from :data:`BROKER_UNAVAILABLE` because the two are different facts about
#: the broker, and both are a `CredentialError` so every route degrades the
#: same way it does for a broker that is down.
BROKER_NO_ANSWER = "Credential broker returned no answer."

#: The broker's answer for a credential id it does not hold - the one sentence
#: the vault raises, so a caller can tell "deleted" from "the broker did not
#: answer" (review S1: the two have opposite remedies).
CREDENTIAL_NOT_FOUND = "Credential was not found."


#: What a verb that raised anything but a `CredentialError` answers with
#: (`credential_broker_server`; the traceback goes to the journal). The first
#: live one (VD-127) was a WAL sidecar that vanished mid-``list``: an
#: unexpected failure inside the broker, as often transient as not.
VERB_FAILED = "The credential broker could not complete the request."

#: A reply that says only that the broker refused, with no reason of its own.
BROKER_REJECTED = "Credential broker rejected the request."


def credential_not_found(error: BaseException) -> bool:
    """Whether ``error`` is the broker saying it holds no such credential."""
    return isinstance(error, CredentialError) and str(error) == CREDENTIAL_NOT_FOUND


def broker_did_not_answer(error: BaseException) -> bool:
    """Whether ``error`` means the broker gave no answer, so a retry may get one.

    The socket could not be reached, the exchange broke, the frame was
    unreadable, or a verb failed inside the broker on something it did not
    expect - :data:`BROKER_UNAVAILABLE`, :data:`BROKER_NO_ANSWER`,
    :data:`VERB_FAILED`, or an error that never became a `CredentialError` at
    all. :data:`VERB_FAILED` is the server's catch-all for any unexpected
    exception, so it promises nothing about the next attempt (review round 1).
    Every other `CredentialError` is the broker's own ANSWER - a provider that
    cannot be used for the purpose, a secret that would not decrypt, an
    explicit rejection - and asking again gets the same one (review A6,
    LESSONS 5).
    """
    if not isinstance(error, CredentialError):
        return True
    return str(error) in (BROKER_UNAVAILABLE, BROKER_NO_ANSWER, VERB_FAILED)


class CredentialBrokerClient:
    """Small JSON-over-Unix-socket client. It never offers a plaintext get."""

    def __init__(self, socket_path: Optional[str] = None, timeout_seconds: int = 60):
        self.socket_path = socket_path or env_value(
            "VAELOR_CREDENTIAL_BROKER_SOCKET", "PM_CREDENTIAL_BROKER_SOCKET",
            run_path("credentiald.sock"),
        )
        self.timeout_seconds = timeout_seconds

    def _request(self, operation: str, **payload) -> Any:
        request_body = json.dumps(
            {"operation": operation, "payload": payload}, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        if len(request_body) > MAX_REQUEST_BYTES:
            raise CredentialError(REQUEST_TOO_LARGE)
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout_seconds)
        try:
            connection.connect(self.socket_path)
            connection.sendall(request_body)
            response = b""
            while b"\n" not in response and len(response) <= MAX_REQUEST_BYTES:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                response += chunk
        except OSError as error:
            raise CredentialError(BROKER_UNAVAILABLE) from error
        finally:
            connection.close()
        decoded = _decode_reply(response)
        if not decoded.get("ok"):
            raise CredentialError(decoded.get("error", BROKER_REJECTED))
        return decoded.get("data")

    def capabilities(self):
        return self._request("capabilities")

    def list(self, actor=None):
        return self._request("list", actor=actor)

    def put(self, provider, label, secret_value, credential_id=None, owner=""):
        return self._request(
            "put", provider=provider, label=label, secret=secret_value,
            credential_id=credential_id, owner=owner,
        )

    def delete(self, credential_id):
        return self._request("delete", credential_id=credential_id)

    def test(self, credential_id):
        return self._request("test", credential_id=credential_id)

    def activate(self, credential_id, purpose):
        return self._request(
            "activate", credential_id=credential_id, purpose=purpose
        )

    def deactivate(self, purpose):
        return self._request("deactivate", purpose=purpose)

    def models(self, credential_id):
        return self._request("models", credential_id=credential_id)

    def select_model(self, credential_id, model):
        return self._request(
            "select_model", credential_id=credential_id, model=model
        )

    def resolve_active(self, purpose):
        return self._request("resolve_active", purpose=purpose)

    def resolve(self, credential_id, purpose, actor=""):
        return self._request(
            "resolve", credential_id=credential_id, purpose=purpose, actor=actor
        )

    def mint(self, endpoint_id, label=""):
        return self._request("mint", endpoint_id=endpoint_id, label=label)

    def rotate(self, credential_id, endpoint_id=""):
        return self._request(
            "rotate", credential_id=credential_id, endpoint_id=endpoint_id
        )

    def revoke(self, credential_id, endpoint_id=""):
        return self._request(
            "revoke", credential_id=credential_id, endpoint_id=endpoint_id
        )

    def endpoint_keys(self, endpoint_id):
        # The active plaintext key set for a gate, over the in-group socket. Raises
        # CredentialError (BROKER_UNAVAILABLE / BROKER_NO_ANSWER) when the broker
        # cannot answer, which the gate renderer treats as "leave the gate as is".
        data = self._request("endpoint_keys", endpoint_id=endpoint_id)
        return list((data or {}).get("keys", []))

    def record_use(self, credential_id, used_at=None):
        """Report that Vaelor used this credential at ``used_at`` (default: now).

        The broker moves the stamp forward only (`vaelor.credential_use`).
        """
        return self._request(
            "record_use", credential_id=credential_id, used_at=used_at,
        )

    def import_key(self, endpoint_id, key, label=""):
        # Migration only: import the pre-F3b-ii key so an existing client survives.
        return self._request(
            "import_key", endpoint_id=endpoint_id, key=key, label=label
        )


def _decode_reply(response: bytes) -> dict:
    """The broker's one-line JSON object, or :data:`BROKER_NO_ANSWER`.

    An empty frame, a frame that is not UTF-8 JSON, and a frame that is JSON
    but not an object are all the same fact to a caller - nothing was answered
    - and none of them may escape as the parser's own exception.
    """
    try:
        decoded = json.loads(response.split(b"\n", 1)[0].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise CredentialError(BROKER_NO_ANSWER) from error
    if not isinstance(decoded, dict):
        raise CredentialError(BROKER_NO_ANSWER)
    return decoded
