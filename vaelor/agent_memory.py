"""Control-plane store for a deployed agent's private recent-interaction memory (F5a).

A cluster agent is given a short recollection of its recent turns, but the
loopback InfluxDB it is stored in carries no authentication and is never
reachable from the agent process. So every read and write is mediated here, on
the control plane, and the whole design turns on ONE rule:

    **the row is tagged and filtered by the RESOLVED ``agent_id`` the caller
    passes in, and never by any value taken from the record payload.**

The route in front of this store resolves an agent's bearer token to its
deployment identity and hands THAT identity down as ``agent_id``. This module
never inspects a record for an id, so one agent cannot address another agent's
partition even by crafting a record that names it: the only id that ever reaches
the tag is the argument, mirroring how the telemetry ingest path stamps the
authenticated node and ignores the body's own node tag.

The store is deliberately small and deny-by-default:

* Isolation is structural. :meth:`AgentMemoryStore.record` writes with the tag
  ``{"agent_id": <resolved>}`` and :meth:`AgentMemoryStore.recall` filters on the
  same resolved id through :meth:`vaelor.database.Database.get_tagged`, whose tag
  value is injection-guarded. An empty or malformed ``agent_id`` is refused here
  rather than allowed to become an unfiltered read or write.
* Everything is bounded. At most :data:`MAX_RECORDS_PER_CALL` records per call,
  at most :data:`MAX_FIELDS_PER_RECORD` fields per record, each field key held to
  an identifier shape and each value truncated to :data:`MAX_FIELD_CHARS`, so a
  single call cannot grow the series without limit.

The per-agent memory token is ingest-style and hash-only. :func:`mint_memory_key`
issues a ``vak_`` bearer once in the clear and returns its sha256 hash for
storage; the plaintext is never persisted or logged, and the deployment store
resolves a presented token by constant-time hash comparison. This secret
authenticates the AGENT TO the control plane, the opposite direction to the
inbound served key that authenticates external clients to the agent.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from typing import Any, Dict, List, Mapping, Optional

from .telemetry_ingest import valid_node_id

#: The one measurement every agent-memory row is written to and read from. Held
#: as a single token, so it is not a re-spellable word-set, and shared by the
#: record and recall paths so the name they write and read can never drift.
MEASUREMENT = "agent_memory"

#: The tag key the resolved agent identity is stamped on. This is the ONLY tag
#: the store ever writes or filters by; isolation is exactly this key carrying
#: the resolved id and nothing from a payload.
AGENT_ID_TAG = "agent_id"

#: The prefix marking a minted per-agent memory token. Distinct from the inbound
#: ``vsk_`` served-key prefix on purpose: this is the agent-to-control-plane
#: direction, a separate secret with its own lifecycle.
MEMORY_KEY_PREFIX = "vak_"

#: How many records one :meth:`AgentMemoryStore.record` call may store. Anything
#: beyond this is discarded rather than written, so a single authenticated call
#: cannot flood the store.
MAX_RECORDS_PER_CALL = 32

#: How many fields one record may carry. Extra keys past this are dropped.
MAX_FIELDS_PER_RECORD = 16

#: The character ceiling each stored field value is truncated to (roughly 1 KB).
#: A memory row is a compact interaction summary, not a transcript.
MAX_FIELD_CHARS = 1024

#: The default number of recent rows :meth:`AgentMemoryStore.recall` returns when
#: a caller names no limit, and the hard ceiling it is clamped to regardless of
#: what a caller asks for.
DEFAULT_RECALL_LIMIT = 8
MAX_RECALL_LIMIT = 200

#: A stored field key must be a short lower-case identifier. Validating the key
#: shape keeps the write to a small, predictable column set; a key outside this
#: shape is dropped, so a caller cannot mint an arbitrarily named series.
_FIELD_KEY_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}")


class AgentMemoryError(ValueError):
    """A refused agent-memory operation, safe to surface to the caller."""


class AgentMemoryUnavailable(RuntimeError):
    """The store behind agent memory cannot serve a call right now (a 503).

    Distinct from :class:`AgentMemoryError`, which is the caller's fault: this is
    the control plane's own storage being off, starting, failed or refusing the
    write, and it must reach the agent as "unavailable" rather than as a 200
    that stored nothing.
    """


class RetainedTelemetryDatabase:
    """The control plane's telemetry store, resolved afresh on every memory call.

    Agent memory lives in the same InfluxDB the telemetry history uses, and that
    store has four retention states (off, starting, failed, running) owned by
    ``vaelor.control_plane._telemetry_source``. ``source`` is that resolver: it
    returns the ready :class:`vaelor.database.Database`, ``None`` when retention
    is switched off, or raises ``TelemetryStoreError`` while starting or after a
    failed start. Each of the not-running states becomes
    :class:`AgentMemoryUnavailable` here, so a memory call against a store that
    is not there is refused out loud instead of reading as "no memories yet".
    """

    def __init__(self, source: Any):
        self._source = source

    def _database(self) -> Any:
        try:
            database = self._source()
        except AgentMemoryUnavailable:
            raise
        except Exception as error:  # the retention state's own reason, passed on
            raise AgentMemoryUnavailable(str(error) or type(error).__name__) from error
        if database is None:
            raise AgentMemoryUnavailable(
                "Telemetry retention is switched off, so agent memory has "
                "nowhere to be kept."
            )
        is_ready = getattr(database, "is_ready", None)
        if callable(is_ready) and not is_ready():
            raise AgentMemoryUnavailable(
                "The telemetry database that holds agent memory is not answering."
            )
        return database

    def set_tagged(self, measurement: str, tags: Mapping[str, str], fields: Mapping[str, Any]):
        return self._database().set_tagged(measurement, dict(tags), dict(fields))

    def get_tagged(self, measurement: str, tag_key: str, tag_value: str, limit: int = 100):
        return self._database().get_tagged(measurement, tag_key, tag_value, limit)


def _first_certificate(pem: str) -> str:
    """The first PEM certificate block in ``pem``, or an empty string.

    The serving certificate file may carry a chain; only its leaf is pinned, so
    trusting the file can never mean trusting everything an intermediate signed.
    """
    begin = "-----BEGIN CERTIFICATE-----"
    end = "-----END CERTIFICATE-----"
    start = pem.find(begin)
    if start < 0:
        return ""
    stop = pem.find(end, start)
    if stop < 0:
        return ""
    return pem[start:stop + len(end)] + "\n"


#: One PEM certificate block, header to footer.
_PEM_BLOCK = re.compile(r"-----BEGIN CERTIFICATE-----.+?-----END CERTIFICATE-----", re.S)


def _every_certificate(pem: str) -> str:
    """Every PEM certificate block in ``pem``, in order, or an empty string."""
    found = _PEM_BLOCK.findall(pem)
    return "".join(block + "\n" for block in found)


def controller_certificate_pem() -> str:
    """What a deployed agent pins for the control plane, or ``""`` if there is nothing.

    The one answer workers use too (`cluster_worker_profile.controller_trust`,
    LESSONS 6). With the household authority's trust bundle (VD-212), EVERY
    certificate in it: the root, and while an existing install migrates the
    certificate still served - pinning the first only would drop the root, or
    pin a certificate the authority is about to retire. Before the authority
    has run, the served certificate's leaf only, as before: that file may carry
    a chain, and trusting it must never mean trusting everything an
    intermediate signed. The deploy runs in the workload executor, so both are
    files that process reads. Certificates are public; no key is touched.
    """
    from .cluster_worker_profile import TRUST_BUNDLE, controller_trust

    trust = controller_trust()
    if not trust.path:
        return ""
    try:
        with open(trust.path, encoding="ascii") as handle:
            text = handle.read()
    except (OSError, UnicodeDecodeError):
        return ""
    return _every_certificate(text) if trust.kind == TRUST_BUNDLE else _first_certificate(text)


def pin_checks_hostname(pem: str) -> bool:
    """Whether an agent pinning ``pem`` also checks the controller's name (VD-212).

    Only when the set is authorities alone - the household root after the
    migration - so the certificate the controller serves can only be the
    root's own issue, which carries every private address the controller
    holds. Any end-entity certificate in the set (the installer's pre-VD-212
    self-signed one during the migration, an owner's custom certificate, or a
    CA-issued leaf served before the authority ran) may be the one served,
    and its names were chosen without this controller's cluster address in
    mind: it stays an exact pin of that one key, with no name check, as before
    VD-212 (review R17). The agent is re-rendered when the set changes (it is
    in the surface digest). An unreadable or empty set checks the name: the
    stricter answer (LESSONS 1).
    """
    try:
        from cryptography import x509

        certificates = x509.load_pem_x509_certificates(pem.encode("ascii"))
    except Exception:  # noqa: BLE001 - unreadable: the stricter answer
        return True
    for certificate in certificates:
        try:
            authority = certificate.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
        except x509.ExtensionNotFound:
            authority = False
        if not authority:
            return False
    return True


def pin_digest(pem: str) -> str:
    """A digest of exactly the certificates an agent pins, for its surface digest (VD-212).

    The sha256 over each certificate's DER sha256, in order: it changes when the
    household root changes, when it first appears, and when the migration drops
    the retired console certificate - each a reason to re-render the agent's
    pin. ``""`` when nothing is pinned.
    """
    import ssl

    blocks = _PEM_BLOCK.findall(pem or "")
    if not blocks:
        return ""
    digest = hashlib.sha256()
    for block in blocks:
        try:
            der = ssl.PEM_cert_to_DER_cert(block + "\n")
        except ValueError:
            der = block.encode("ascii", "replace")
        digest.update(hashlib.sha256(der).digest())
    return digest.hexdigest()


def memory_client_config(advertise_address: str, token: str, *,
                         certificate_pem: Optional[str] = None) -> Dict[str, str]:
    """The ``config['memory']`` block a deployed agent reads (F5b).

    Built from the SAME LAN advertise address the inbound gate binds and the
    control plane's own port (``VAELOR_PORT``, default 34001). The control plane
    serves TLS on that port whenever it holds a serving certificate, so the
    endpoint is ``https`` and the block carries that certificate for the agent
    to pin; plain ``http`` only when there is no certificate at all (a
    development box). A plain-HTTP call to the TLS port is refused on every
    attempt, which is how agent memory never worked (ACC-132). The token that
    pairs with this endpoint lives only in the 0600 config.
    """
    from .runtime_paths import env_value

    try:
        port = int(env_value("VAELOR_PORT", "PM_DASHBOARD_PORT", "34001"))
    except (TypeError, ValueError):
        port = 34001
    pem = controller_certificate_pem() if certificate_pem is None else certificate_pem
    scheme = "https" if pem else "http"
    block = {
        "endpoint": "{}://{}:{}/api/v2/agents/memory".format(scheme, advertise_address, port),
        "token": token,
    }
    if pem:
        block["ca_pem"] = pem
        # VD-212: the agent checks the advertise address against the
        # controller's certificate, except while it pins the pre-VD-212
        # self-signed one (`pin_checks_hostname`). Written explicitly, so the
        # agent never guesses; a config without it predates VD-212.
        block["check_hostname"] = pin_checks_hostname(pem)
    return block


def mint_memory_key() -> Dict[str, str]:
    """Issue a fresh per-agent memory token, returning its plaintext and hash.

    The plaintext is handed to the deploy path exactly once and belongs only in
    the agent's 0600 config; the sha256 hash is what the deployment row stores.
    The plaintext is never persisted here, so a stored hash can never be turned
    back into the token it authenticates.
    """
    token = "{}{}".format(MEMORY_KEY_PREFIX, secrets.token_urlsafe(32))
    return {"token": token, "hash": memory_key_hash(token)}


def memory_key_hash(token: str) -> str:
    """The sha256 hex of a memory token - the only form ever kept at rest."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _resolved_agent_id(agent_id: Any) -> str:
    """The resolved agent id, validated, or raise - never a silent widening.

    ``agent_id`` is always the token-resolved deployment identity the route
    derives; validating it here is the belt to the route's braces, so an empty
    or malformed id becomes a refusal instead of a read or write with no tag
    filter at all.
    """
    if not valid_node_id(agent_id):
        raise AgentMemoryError(
            "An agent memory identity is required and must be a resolved id."
        )
    return agent_id


class AgentMemoryStore:
    """Record and recall one agent's private memory rows, isolated by resolved id.

    The injected ``database`` is a :class:`vaelor.database.Database` (or a fake
    with the same ``set_tagged``/``get_tagged`` surface, which is what the tests
    use), so the store carries no InfluxDB coupling of its own.
    """

    def __init__(self, database: Any):
        self.database = database

    def record(self, agent_id: str, records: Any) -> int:
        """Append bounded memory rows for ONE agent, returning how many were stored.

        Each row is tagged with the resolved ``agent_id`` argument - never a value
        read from a record - so a record that names another agent changes nothing
        about which partition it lands in. Malformed or empty rows are skipped
        rather than failing the call. A store that refuses EVERY usable row raises
        :class:`AgentMemoryUnavailable` instead of answering "stored 0", because a
        write that silently kept nothing reads to the agent exactly like success.
        """
        agent = _resolved_agent_id(agent_id)
        if not isinstance(records, list):
            raise AgentMemoryError("Memory records must be supplied as a list.")
        stored = 0
        attempted = 0
        for record in records[:MAX_RECORDS_PER_CALL]:
            fields = self._bounded_fields(record)
            if not fields:
                continue
            attempted += 1
            ok, _ = self.database.set_tagged(MEASUREMENT, {AGENT_ID_TAG: agent}, fields)
            if ok:
                stored += 1
        if attempted and not stored:
            raise AgentMemoryUnavailable("The memory store refused every row it was given.")
        return stored

    def recall(self, agent_id: str, limit: int = DEFAULT_RECALL_LIMIT) -> List[Dict[str, Any]]:
        """The most recent rows for ONE agent, newest first, bounded by ``limit``.

        Reads only the partition the resolved ``agent_id`` tags, through the
        injection-guarded tagged read, so an agent sees its own rows and no other
        agent's. ``limit`` is clamped to :data:`MAX_RECALL_LIMIT`.
        """
        agent = _resolved_agent_id(agent_id)
        bounded = max(1, min(int(limit), MAX_RECALL_LIMIT))
        return self.database.get_tagged(MEASUREMENT, AGENT_ID_TAG, agent, bounded)

    @staticmethod
    def _bounded_fields(record: Any) -> Dict[str, str]:
        """Coerce one record into a bounded, string-typed, identifier-keyed dict.

        A field named like the isolation tag is dropped so a payload can never
        smuggle an id into the write, a key outside the identifier shape is
        dropped, and each value is stringified and truncated. A record with no
        usable field yields an empty dict and is not written.
        """
        if not isinstance(record, Mapping):
            return {}
        fields: Dict[str, str] = {}
        for key, value in record.items():
            if len(fields) >= MAX_FIELDS_PER_RECORD:
                break
            field_key = str(key).strip()
            if field_key == AGENT_ID_TAG:
                continue
            if not _FIELD_KEY_PATTERN.fullmatch(field_key):
                continue
            fields[field_key] = str(value)[:MAX_FIELD_CHARS]
        return fields
