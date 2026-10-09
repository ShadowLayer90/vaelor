"""The broker's connection tests: does a stored credential actually work.

Split from :mod:`vaelor.credential_broker` to keep the vault under the 1,000-line
production ceiling. Everything here runs INSIDE the broker daemon, on a secret
the vault has just decrypted; nothing here stores, logs or returns a secret.
Every answer is ``{"ok": bool, "message": str}`` in owner-readable words.

Each provider kind is tested by doing the thing Vaelor uses it for, as small as
possible: a hosted key asks its provider who it is, a compatible server lists
its models and answers a two-token chat, and a cluster node's SSH sign-in logs
in against the host key recorded at enrolment and disconnects (ACC-107: the SSH
kind had no branch at all, so Test fell through to a lookup of a test URL the
kind does not have and answered with the broker's generic failure sentence).
"""

from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Mapping, Optional

from .credential_broker_client import CredentialError
from .credential_profiles import validate_ssh_profile
from .hosted_providers import HOSTED_KINDS, probe as probe_hosted


#: The largest secret the vault stores. Declared here, beside the profile
#: validation that enforces it on a compatible server's key, and imported by the
#: vault (`vaelor.credential_broker`) so the two limits cannot drift apart.
MAX_SECRET_BYTES = 8192

#: How long the SSH sign-in test waits for the node at each step. Short on
#: purpose: an owner pressed a button and is waiting on the answer.
SSH_TEST_TIMEOUT_SECONDS = 12


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def validate_compatible_profile(secret_value: str) -> Dict[str, str]:
    """Validate and normalize an encrypted compatible-endpoint profile."""
    try:
        profile = json.loads(secret_value)
    except (TypeError, json.JSONDecodeError) as error:
        raise CredentialError("The compatible server profile is invalid.") from error
    if not isinstance(profile, dict):
        raise CredentialError("The compatible server profile is invalid.")

    base_url = str(profile.get("base_url", "")).strip().rstrip("/")
    model = str(profile.get("model", "")).strip()
    api_key = str(profile.get("api_key", "")).strip()
    if len(base_url) > 500 or len(model) > 200 or len(api_key) > MAX_SECRET_BYTES:
        raise CredentialError("The compatible server profile is too large.")

    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise CredentialError("Enter a complete HTTP or HTTPS server URL without credentials or query text.")
    try:
        port = parsed.port
    except ValueError as error:
        raise CredentialError("The compatible server port is invalid.") from error
    if port is not None and not 1 <= port <= 65535:
        raise CredentialError("The compatible server port is invalid.")

    try:
        resolved = {
            ipaddress.ip_address(item[4][0])
            for item in socket.getaddrinfo(parsed.hostname, port or (443 if parsed.scheme == "https" else 80))
        }
    except (OSError, ValueError) as error:
        raise CredentialError("The compatible server address could not be resolved.") from error
    if not resolved or any(
        not (address.is_private or address.is_loopback)
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
        for address in resolved
    ):
        raise CredentialError("For safety, compatible servers must use a private LAN or loopback address.")

    return {"base_url": base_url, "model": model, "api_key": api_key}


def compatible_request(
    profile: Dict[str, str],
    path: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 20,
) -> Dict[str, Any]:
    """One JSON request to a compatible server, never following a redirect."""
    headers = {
        "Accept": "application/json",
        "User-Agent": "Vaelor-Credential-Broker/1",
    }
    if profile["api_key"]:
        headers["Authorization"] = "Bearer {}".format(profile["api_key"])
    data = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(
        "{}{}".format(profile["base_url"], path),
        data=data,
        headers=headers,
        method=method,
    )
    opener = urllib.request.build_opener(_NoRedirect)
    with opener.open(request, timeout=timeout) as response:
        if not 200 <= response.status < 300:
            raise CredentialError("The compatible server rejected the connection test.")
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise CredentialError("The compatible server response was too large.")
    try:
        result = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CredentialError("The compatible server did not return valid JSON.") from error
    if not isinstance(result, dict):
        raise CredentialError("The compatible server returned an invalid response.")
    return result


def _ssh_sign_in(profile: Mapping[str, Any]) -> None:
    """Sign in to the node over the pinned transport, then disconnect."""
    from .ssh_transport import SshTransport

    SshTransport(dict(profile), timeout=SSH_TEST_TIMEOUT_SECONDS).verify_login()


def _named_in_mro(error: BaseException, name: str) -> bool:
    return any(kind.__name__ == name for kind in type(error).__mro__)


def probe_ssh_sign_in(
    secret_value: str,
    sign_in: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Log in to a cluster node with its stored SSH sign-in, and say what happened.

    The same pinned, password-authenticated connection every cluster command
    uses (`vaelor.ssh_transport.SshTransport`), opened and closed with no
    command run. A changed host key, a refused password and an unreachable
    node are three different answers, because they have three different fixes.
    ``sign_in`` is injectable for tests; production uses the real transport.
    """
    profile = validate_ssh_profile(secret_value, CredentialError)
    where = "{} port {}".format(profile["host"], profile["port"])
    try:
        (sign_in or _ssh_sign_in)(profile)
    except Exception as error:  # noqa: BLE001 - every failure is an answer, never a crash
        if _named_in_mro(error, "SshTransportError"):
            return {"ok": False, "message": str(error)[:240]}
        if _named_in_mro(error, "AuthenticationException"):
            return {
                "ok": False,
                "message": "{} refused the saved user name and password.".format(where),
            }
        if isinstance(error, (OSError, TimeoutError)) or _named_in_mro(
            error, "SSHException"
        ):
            return {
                "ok": False,
                "message": "{} could not be reached over SSH from this appliance.".format(where),
            }
        return {"ok": False, "message": "The SSH sign-in to {} did not complete.".format(where)}
    return {
        "ok": True,
        "message": "Signed in to {} as {}; the host key matches the one recorded when the node was added.".format(
            where, profile["username"]
        ),
    }


def _test_compatible(secret_value: str) -> Dict[str, Any]:
    profile = validate_compatible_profile(secret_value)
    try:
        model_data = compatible_request(profile, "/models")
        models = model_data.get("data", [])
        if not isinstance(models, list) or not models:
            return {"ok": False, "message": "The server did not report any loaded models."}
        selected_model = profile["model"] or str(models[0].get("id", "")).strip()
        if not selected_model:
            return {"ok": False, "message": "The server did not report a usable model ID."}
        if profile["model"] and not any(
            isinstance(item, dict) and item.get("id") == selected_model
            for item in models
        ):
            return {"ok": False, "message": "The selected model is not loaded on this server."}
        chat = compatible_request(
            profile,
            "/chat/completions",
            payload={
                "model": selected_model,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "temperature": 0,
                "max_tokens": 2,
            },
            timeout=45,
        )
        if not isinstance(chat.get("choices"), list):
            return {"ok": False, "message": "The chat-completions response was not compatible."}
        return {
            "ok": True,
            "message": "Connected to {}. Models and chat completions are working.".format(selected_model),
        }
    except urllib.error.HTTPError as error:
        if error.code in {401, 403}:
            return {"ok": False, "message": "The server rejected the API key."}
        return {"ok": False, "message": "The server returned HTTP {}.".format(error.code)}
    except (urllib.error.URLError, TimeoutError):
        return {"ok": False, "message": "The compatible server could not be reached from this Vaelor node."}


def probe_provider(
    provider: str, secret_value: str, policy: Mapping[str, Any]
) -> Dict[str, Any]:
    """Test one decrypted credential of ``provider`` against ``policy``.

    ``policy`` is the provider's row from the broker's ``PROVIDERS`` table; a
    hosted provider's row names the URL, header and prefix its test uses. A
    kind with no test of its own says so rather than raising.
    """
    if provider == "application-secret":
        return {
            "ok": True,
            "message": "The encrypted application secret is available to approved deployments.",
        }
    if provider == "openai-compatible":
        return _test_compatible(secret_value)
    if provider == "ssh":
        return probe_ssh_sign_in(secret_value)
    if provider in HOSTED_KINDS:
        # VD-206: through the checked, pinned transport, never `urlopen`.
        return probe_hosted(provider, secret_value)
    if not policy.get("test_url"):
        return {"ok": False, "message": "Vaelor has no connection test for this kind of credential."}
    request = urllib.request.Request(
        policy["test_url"],
        headers={
            policy["header"]: "{}{}".format(policy["prefix"], secret_value),
            "Accept": "application/json",
            "User-Agent": "Vaelor-Credential-Broker/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            ok = 200 <= response.status < 300
        return {
            "ok": ok,
            "message": "Provider accepted this credential." if ok else "Provider rejected this credential.",
        }
    except urllib.error.HTTPError as error:
        if error.code in {401, 403}:
            return {"ok": False, "message": "Provider rejected this credential."}
        return {"ok": False, "message": "Provider returned HTTP {}.".format(error.code)}
    except urllib.error.URLError:
        return {"ok": False, "message": "Provider could not be reached from this device."}
