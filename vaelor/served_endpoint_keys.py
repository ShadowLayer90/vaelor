"""Inbound served-endpoint API keys: the broker-minted ``vsk_`` server keys.

Split from :mod:`vaelor.credential_broker` to keep the vault under the 1,000-line
production ceiling and to give the most security-sensitive key path one home.

A served-endpoint key is INBOUND: Vaelor mints it and a LAN client presents it to
reach a gated endpoint (the LLM-Server proxy, a cluster balancer, an agent door).
It is the mirror image of an ``openai-compatible`` credential - the OUTBOUND
secret Vaelor presents to somebody else's server. The two must never be
interchangeable: handing an inbound door key to an outbound caller, or presenting
a foreign secret as our own gate key, would each cross the boundary. So this
provider carries its own per-endpoint purpose family
(:data:`SERVED_ENDPOINT_PURPOSE_PREFIX`) and is refused by the generic ``put``
create path - it is created only through :func:`mint`.

The plaintext key is returned by :func:`mint` and nowhere else. There is no
read-back: ``list``/metadata and every HTTP route stay fingerprint+last4 only,
and ``resolve`` hands the plaintext only to an in-group local gate-rendering
service. "One-time reveal" is therefore a property of the mint HTTP response,
not of the stored secret.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from contextlib import closing
from typing import Any, Dict

from .assistant_console_places import ENDPOINTS_PLACE
from .credential_broker_client import CredentialError


#: Provider id for an inbound, broker-minted endpoint key.
SERVED_ENDPOINT_PROVIDER = "served-endpoint"

#: Every minted endpoint key carries this prefix so a leaked token is
#: recognisable as a Vaelor server key at a glance (the convention ``sk_`` and
#: ``ghp_`` use). Single-sourced here rather than imported from
#: ``llm_server_state`` so the broker stays the one owner of the key format.
VSK_PREFIX = "vsk_"

#: The longest endpoint id a key may bind. Endpoint ids are short registry
#: names (``llm-server``, ``cluster-serving``, ``agent:<id>``).
MAX_ENDPOINT_ID = 120

#: An upper bound on an IMPORTED key's length (:func:`import_key`), defence in
#: depth so the one-off migration cannot store an absurd value; a Vaelor key is
#: about fifty bytes.
MAX_IMPORTED_KEY = 256

#: A served-endpoint credential is resolvable ONLY under its own per-endpoint
#: purpose ``served-endpoint:<endpoint_id>``. A bounded prefix keeps the
#: purpose->provider gate tight without a fixed name per endpoint - endpoints
#: are enumerated at runtime - exactly as the alert-channel family does.
SERVED_ENDPOINT_PURPOSE_PREFIX = "served-endpoint:"

#: The PROVIDERS descriptor the vault merges in. ``capabilities`` hides this id
#: and ``put`` refuses it, so it is never offered to the generic create route.
PROVIDER_DESCRIPTOR = {
    "name": "Served endpoint key",
    "auth": "managed_inbound_key",
}

#: ``put`` refuses this provider: a served-endpoint key stored from a request
#: would be user-supplied and unbound, the weak key mint exists to prevent.
PUT_REJECTED = (
    "Served-endpoint keys are minted by Vaelor, not stored from a request."
)

#: The vault's ``delete`` refuses this provider (ACC-111). A served-endpoint key
#: is retired by :func:`revoke`, which keeps the row as an audit record; a
#: generic delete from a connections list erased it instead and pulled the key
#: out from under a running gate. Owner-facing: it names where keys are managed.
DELETE_REFUSED = (
    "Keys that apps and agents use to reach this appliance are revoked, not "
    "deleted, so each one's record is kept. Rotate or revoke it under "
    + ENDPOINTS_PLACE + "."
)


def served_key_fingerprint(key: str) -> str:
    """The stored per-key fingerprint of a served-endpoint secret.

    The last eight hex of the key's SHA-256, uppercased - the value a row's
    ``fingerprint`` column holds and ``list()`` echoes. Single-sourced here (the
    broker owns the key format) so mint/import (:func:`_store_key`),
    :func:`rotate`, and the executor's applied-binding marker all derive the SAME
    fingerprint from a plaintext key. That equality is what lets the convergence
    poll compare a fingerprint set read from ``list()`` (no decrypt) against a
    marker written from the just-applied plaintext without either basis drifting.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[-8:].upper()


def _endpoint_has_any_row(vault, endpoint_id: str) -> bool:
    """Whether ANY served-endpoint row - revoked included - is bound to ``endpoint_id``.

    :func:`import_key`'s once-per-lifetime guard counts rows directly rather than
    through the active-only :func:`endpoint_keys`: revoking the sole key must not
    reopen the door to a second, caller-chosen import. A direct COUNT also decrypts
    nothing and writes no plaintext-read audit row.
    """
    clean_endpoint = str(endpoint_id).strip()
    with closing(vault._connect()) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM credentials WHERE provider = ? AND owner = ?",
            (SERVED_ENDPOINT_PROVIDER, clean_endpoint),
        ).fetchone()[0]
    return int(count) > 0


def is_served_endpoint_purpose(purpose: str) -> bool:
    """Whether ``purpose`` names a served-endpoint lease, within a length bound."""
    return (
        purpose.startswith(SERVED_ENDPOINT_PURPOSE_PREFIX)
        and len(purpose) > len(SERVED_ENDPOINT_PURPOSE_PREFIX)
        and len(purpose) <= len(SERVED_ENDPOINT_PURPOSE_PREFIX) + MAX_ENDPOINT_ID
    )


def metadata_view(row: Any) -> Dict[str, Any]:
    """The key-free fields to fold into a served-endpoint credential's metadata.

    Its binding (``endpoint_id``, held in the ``owner`` column) and its real
    revocation state (``revoked`` plus the ``revoked_at`` stamp) - never the
    plaintext. The stamp is read defensively so a row selected before the
    ``revoked_at`` column existed still renders as live rather than raising.
    """
    revoked_at = row["revoked_at"] if "revoked_at" in row.keys() else None
    return {
        "endpoint_id": row["owner"],
        "revoked": revoked_at is not None,
        "revoked_at": revoked_at,
    }


def _served_endpoint_row(vault, credential_id: str, endpoint_id: str) -> Any:
    """Fetch a served-endpoint row for a lifecycle verb, enforcing the boundary.

    Refuses a credential that is absent or of any OTHER provider (rotate and
    revoke are served-endpoint verbs only), and - when an ``endpoint_id`` is
    given - refuses one whose ``owner`` binding names a different endpoint, the
    same S2 binding the resolve path enforces so a key is only ever managed
    under its own endpoint. Returns the row with the columns rotate/revoke need.
    """
    clean_id = str(credential_id).strip()
    with closing(vault._connect()) as connection:
        row = connection.execute(
            """
            SELECT id, provider, owner, version, nonce, ciphertext, revoked_at
            FROM credentials WHERE id = ?
            """,
            (clean_id,),
        ).fetchone()
    if row is None:
        raise CredentialError("The served-endpoint credential could not be located.")
    if row["provider"] != SERVED_ENDPOINT_PROVIDER:
        raise CredentialError("This lifecycle verb is only for served-endpoint keys.")
    clean_endpoint = str(endpoint_id).strip()
    if clean_endpoint and clean_endpoint != str(row["owner"]).strip():
        raise CredentialError("This key belongs to a different endpoint.")
    return row


def revoke(vault, credential_id: str, endpoint_id: str = "") -> Dict[str, Any]:
    """Mark a served-endpoint key revoked, keeping the row for audit.

    The row is never deleted: a revoked key stays as an audit record and simply
    stops being active - :func:`endpoint_keys` (and, later, the gate) skip it.
    Revoke is idempotent: re-revoking an already-revoked key keeps its original
    stamp rather than raising, so a repeated ``DELETE`` is not an error.
    """
    row = _served_endpoint_row(vault, credential_id, endpoint_id)
    revoked_at = row["revoked_at"]
    if revoked_at is None:
        revoked_at = int(time.time())
        with closing(vault._connect()) as connection:
            connection.execute(
                "UPDATE credentials SET revoked_at = ? WHERE id = ?",
                (revoked_at, row["id"]),
            )
            connection.commit()
        vault._audit("credential.revoke", row["id"], SERVED_ENDPOINT_PROVIDER)
    return {
        "credential_id": row["id"],
        "endpoint_id": row["owner"],
        "revoked": True,
        "revoked_at": revoked_at,
    }


def rotate(vault, credential_id: str, endpoint_id: str = "") -> Dict[str, Any]:
    """Replace a served-endpoint key's secret in place, revealing the new one ONCE.

    A fresh ``vsk_`` key is generated and re-encrypted at a BUMPED version, so
    the vault's AAD ``(id, provider, version)`` binds the new ciphertext to the
    new version; the row keeps its ``id`` and ``owner`` (endpoint) binding while
    its ciphertext, nonce, fingerprint, last4, version and ``updated_at`` are
    replaced. A revoked key cannot be rotated. The plaintext ``key`` is in this
    return value and nowhere else - the same one-time reveal as :func:`mint`.
    """
    row = _served_endpoint_row(vault, credential_id, endpoint_id)
    if row["revoked_at"] is not None:
        raise CredentialError("A revoked endpoint key cannot be rotated.")
    key = VSK_PREFIX + secrets.token_urlsafe(32)
    secret_bytes = key.encode("utf-8")
    now = int(time.time())
    new_version = int(row["version"]) + 1
    nonce = os.urandom(12)
    ciphertext = vault._cipher.encrypt(
        nonce,
        secret_bytes,
        vault._associated_data(row["id"], SERVED_ENDPOINT_PROVIDER, new_version),
    )
    fingerprint = served_key_fingerprint(key)
    key_last4 = key[-4:]
    with closing(vault._connect()) as connection:
        connection.execute(
            """
            UPDATE credentials
            SET nonce = ?, ciphertext = ?, fingerprint = ?, version = ?,
                last4 = ?, updated_at = ?
            WHERE id = ?
            """,
            (nonce, ciphertext, fingerprint, new_version, key_last4, now, row["id"]),
        )
        connection.commit()
    vault._audit("credential.rotate", row["id"], SERVED_ENDPOINT_PROVIDER)
    return {
        "credential_id": row["id"],
        "endpoint_id": row["owner"],
        "key_fingerprint": fingerprint,
        "last4": key_last4,
        "version": new_version,
        "revoked": False,
        "updated_at": now,
        "key": key,
    }


def endpoint_keys(vault, endpoint_id: str) -> list:
    """The ACTIVE (non-revoked) plaintext keys bound to ``endpoint_id``.

    THE BOUNDARY (design section 2, review resequencing): this is the only
    bulk-plaintext path, so it is deliberately narrow. It is NOT an HTTP route
    (no route returns a plaintext key set) and NOT a broker socket verb: today
    the LLM-Server gate is still rendered from the state record, so no
    out-of-process gate renderer needs the set over the socket. It is a direct
    in-process :class:`CredentialVault` method, reached only inside the broker
    daemon. When F3b-ii wires an in-group gate renderer (the executor) to source
    its key set from the broker over the socket, a socket verb is added THEN,
    beside the existing ``resolve`` plaintext lease, not before it is needed.
    """
    clean_endpoint = str(endpoint_id).strip()
    with closing(vault._connect()) as connection:
        rows = connection.execute(
            """
            SELECT id, provider, nonce, ciphertext, version
            FROM credentials
            WHERE provider = ? AND owner = ? AND revoked_at IS NULL
            ORDER BY created_at
            """,
            (SERVED_ENDPOINT_PROVIDER, clean_endpoint),
        ).fetchall()
    keys = [vault._decrypt(row) for row in rows]
    vault._audit("credential.endpoint-keys", clean_endpoint, SERVED_ENDPOINT_PROVIDER)
    return keys


def _clean_endpoint(endpoint_id: str) -> str:
    """A validated endpoint id, or raise: short registry names only."""
    clean = str(endpoint_id).strip()
    if not clean or len(clean) > MAX_ENDPOINT_ID:
        raise CredentialError(
            "A served-endpoint key needs an endpoint id of up to 120 characters."
        )
    return clean


def _store_key(
    vault, endpoint_id: str, label: str, key: str, audit_action: str
) -> Dict[str, Any]:
    """Encrypt and INSERT one served-endpoint key row, returning its profile.

    The single write path behind both :func:`mint` (which GENERATES the key) and
    :func:`import_key` (which is HANDED one): the two differ only in where ``key``
    comes from and in the audit verb they name, so the cipher, INSERT and audit
    live once here rather than in two near-identical copies. The row starts at
    version 1 with its ``owner`` column holding the endpoint binding, the shape
    ``put`` gives a credential. The plaintext ``key`` is echoed back so a caller
    that must reveal it once can (mint does; import_key drops it).
    """
    secret_bytes = key.encode("utf-8")
    now = int(time.time())
    item_id = "cred_{}".format(secrets.token_hex(12))
    nonce = os.urandom(12)
    ciphertext = vault._cipher.encrypt(
        nonce,
        secret_bytes,
        vault._associated_data(item_id, SERVED_ENDPOINT_PROVIDER, 1),
    )
    fingerprint = served_key_fingerprint(key)
    key_last4 = key[-4:]
    with closing(vault._connect()) as connection:
        connection.execute(
            """
            INSERT INTO credentials (
                id, provider, label, fingerprint, nonce, ciphertext,
                version, owner, created_at, updated_at, last_used_at,
                last_test_status, last4
            ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, NULL, NULL, ?)
            """,
            (
                item_id, SERVED_ENDPOINT_PROVIDER, label, fingerprint,
                nonce, ciphertext, endpoint_id, now, now, key_last4,
            ),
        )
        connection.commit()
    vault._audit(audit_action, item_id, SERVED_ENDPOINT_PROVIDER)
    return {
        "credential_id": item_id,
        "endpoint_id": endpoint_id,
        "label": label,
        "key_fingerprint": fingerprint,
        "last4": key_last4,
        "revoked": False,
        "created_at": now,
        "key": key,
    }


def mint(vault, endpoint_id: str, label: str = "") -> Dict[str, Any]:
    """Generate and store a fresh inbound endpoint key, revealing it ONCE.

    The broker generates the key itself and stores it through the vault's own
    encrypt path (this is the vault's ``mint`` method, split here only for the
    line ceiling; it uses the same cipher, connection and audit as ``put``).
    Each call makes a NEW credential - several live keys per endpoint are
    allowed - so mint never echoes or rotates an existing stored secret. The
    plaintext ``key`` is in this return value and nowhere else.
    """
    clean_endpoint = _clean_endpoint(endpoint_id)
    clean_label = str(label).strip()[:80] or clean_endpoint
    key = VSK_PREFIX + secrets.token_urlsafe(32)
    return _store_key(vault, clean_endpoint, clean_label, key, "credential.mint")


def import_key(
    vault, endpoint_id: str, key: str, label: str = ""
) -> Dict[str, Any]:
    """Store a GIVEN key value as an endpoint's FIRST active key (migration only).

    The single exception to mint's generate-only rule: it stores a value the
    caller supplies rather than one the broker generates. It exists for ONE job -
    importing the LLM Server's pre-F3b-ii ``state.json`` key into the broker so an
    already-configured external client keeps working across the move (design B5).
    Importing that key exposes nothing new, because it was always-revealed before
    the move; that is what justifies the exception.

    It is bounded so it can never become a back door. It is created ONLY through
    this function - the generic ``put`` still refuses ``served-endpoint`` and no
    HTTP route reaches it - and it refuses once per endpoint lifetime: it is
    refused if any key - active OR revoked - already exists for the endpoint (a
    direct row count, not the active-only :func:`endpoint_keys`). So revoking the
    sole key can NOT reopen the door to a second, caller-chosen import; it can only
    ever ESTABLISH a first key, never ADD a chosen key to a populated endpoint, and
    a repeated migration attempt raises rather than duplicating (the idempotence
    the migration leans on). Unlike ``mint`` it does NOT return the plaintext: the
    caller already holds the value it passed in.
    """
    clean_endpoint = _clean_endpoint(endpoint_id)
    clean_key = str(key).strip()
    if not clean_key or len(clean_key) > MAX_IMPORTED_KEY:
        raise CredentialError("An imported endpoint key must be a non-empty value.")
    if _endpoint_has_any_row(vault, clean_endpoint):
        raise CredentialError(
            "This endpoint already has a key (active or revoked), so an import is "
            "refused."
        )
    clean_label = str(label).strip()[:80] or clean_endpoint
    profile = _store_key(
        vault, clean_endpoint, clean_label, clean_key, "credential.import"
    )
    profile.pop("key", None)
    return profile
