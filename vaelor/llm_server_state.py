"""Persisted enable-state and API key for the outbound LLM Server (M1).

The **LLM Server** exposes Vaelor's own single-node GPU AI-Chat model - the
``llama-server`` ROCmFPX fork - as an OpenAI-compatible API on the LAN, so an
external OpenAI client can use it. It is the OPPOSITE direction of an
``openai-compatible`` *client* credential (Vaelor reaching OUT to someone else's
server): here Vaelor is the server. So the exposure is a **server-side setting
plus a generated secret**, not a client credential, and it deliberately does not
pass through the credential broker's ``openai-compatible`` validator (which, by
design, forbids public addresses for the inbound-client case).

This module is the single home of that state: a small JSON record
``{"enabled": bool}`` under the state root (the KEY(S) live in the credential
broker as ``served-endpoint`` creds bound to ``llm-server`` as of F3b-ii), plus
the pure derivation that turns ``(enabled, active-key-set)`` into the PROXY's
binding. Two processes touch it -
the control plane writes it (the enable/disable/rotate routes) and the workload
executor reads it (to start/stop the proxy in front of the loopback model, and to
honour it across a reboot) - so it is written group-readable to the shared jobs
group, the same way the job queue is shared between those two accounts.

**The model itself is NEVER keyed or LAN-bound.** It is served loopback-only by
:mod:`vaelor.gpu_rocmfpx_service`, and the LAN exposure is a Vaelor-controlled
nginx auth proxy (:mod:`vaelor.llm_server_proxy`) - the only trustworthy,
engine-agnostic gate, because the ROCmFP4 fork's ``llama-server`` does not enforce
an api key. So :func:`serve_binding` derives the PROXY's ``(listen_host,
api_key)``: a LAN listen host is returned ONLY when a key is present, so even a
corrupt "enabled but key-less" record can never produce a keyless LAN proxy (which
:func:`vaelor.llm_server_proxy._require_keys` also refuses at the launch boundary).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

from .credential_broker_client import CredentialError
from .runtime_paths import data_path, jobs_group_id, state_path
from .served_endpoint_keys import SERVED_ENDPOINT_PROVIDER, served_key_fingerprint


#: Server keys are prefixed so a leaked value is recognisable as a Vaelor LLM
#: Server key at a glance, and never confused with a client bearer token.
LLM_SERVER_KEY_PREFIX = "vsk_"

#: Entropy for the generated key. 32 URL-safe bytes is ~256 bits, the same
#: strength the session and VNC tokens use (:mod:`vaelor.security`).
KEY_ENTROPY_BYTES = 32

#: The proxy's listen host when the LLM Server is disabled: loopback (moot, since
#: a disabled server runs no proxy at all - it is the fail-safe binding the marker
#: uses to mean "no LAN exposure").
LOOPBACK_HOST = "127.0.0.1"

#: The proxy's listen host when the LLM Server is enabled: all interfaces, so the
#: gate is reachable across the LAN. The model stays loopback-only behind it, and
#: internal AI Chat keeps reaching the model directly on loopback (keyless), so the
#: AI-Chat credential's ``base_url`` is unaffected.
LAN_BIND_HOST = "0.0.0.0"

LOGGER = logging.getLogger(__name__)

#: The persisted record's default location, under the Vaelor state root.
STATE_FILE = state_path("llm-server/state.json")

#: The job the control plane enqueues so the executor brings the proxy to the
#: persisted state at once, rather than on the next 30 s reconcile.
LLM_SERVER_APPLY_JOB = "llm_server.apply"


def enqueue_apply(job_store: Any, username: str, action: str):
    """Enqueue one ``llm_server.apply`` job; the job's id, or ``None``.

    The one enqueue every LLM Server change goes through - the toggle and
    rotate routes and the generic endpoint-key routes - so a key minted,
    rotated or revoked reaches the running gate as promptly as a toggle does.
    **Never fatal, whatever the store raises.** It runs AFTER the broker has
    already minted or rotated a key, so an exception here (a locked SQLite
    queue raises ``sqlite3.OperationalError``, not ``ValueError``) would turn
    the response into a 500 and lose the one-time plaintext - and, for a
    rotate, the old key is already dead. The failure is logged with its type
    and message - safe, because the only payload handed to the store is
    ``{"action": ...}``, so no key can be in either - and the caller answers
    ``pending`` while the executor's reconcile converges the gate from the
    same state.
    """
    if job_store is None:
        return None
    try:
        job = job_store.create(LLM_SERVER_APPLY_JOB, username, {"action": action})
    except Exception as error:  # noqa: BLE001 - the reveal must still be returned
        LOGGER.warning(
            "The LLM Server apply job (%s) could not be queued; the executor's "
            "reconcile will converge the gate instead: %s: %s",
            action, type(error).__name__, error,
        )
        return None
    return (job or {}).get("id")

#: The applied-binding marker's file name, in the executor-owned models root.
#: One spelling for the two readers that share the marker - the Mode A
#: failure-watch (`ExecutorLlmServerMixin`, over its own ``models_root``) and
#: the Mode B reconcile (`ClusterModeSwitch`, over :data:`APPLIED_BINDING_FILE`)
#: - so the gate one converged is the gate the other compares against.
APPLIED_BINDING_FILENAME = ".llm-server-applied.json"

#: The marker's production path: the executor's default models root.
APPLIED_BINDING_FILE = data_path("models/" + APPLIED_BINDING_FILENAME)


@dataclass(frozen=True)
class LlmServerSettings:
    """The LLM Server's persisted state: whether it is on (F3b-ii).

    Immutable so a read cannot be mutated in place by a caller. The KEY(S) moved
    to the credential broker in F3b-ii; the record now carries only ``enabled``.
    ``legacy_api_key`` is the pre-F3b-ii plaintext key still on disk on a box that
    has not migrated yet - :func:`migrate_legacy_key` imports it into the broker
    and :meth:`LlmServerStore.forget_legacy_key` then drops it. It is NOT a live
    binding input: the active key SET always comes from the broker, so an
    ``enabled`` record with an empty broker set produces NO LAN bind.
    """

    enabled: bool = False
    legacy_api_key: str = ""


def generate_key() -> str:
    """A fresh, strong, prefixed LLM Server key."""
    return LLM_SERVER_KEY_PREFIX + secrets.token_urlsafe(KEY_ENTROPY_BYTES)


def _coerce(raw: Any) -> LlmServerSettings:
    """A stored record turned into settings, failing safe to disabled.

    ``enabled`` is read verbatim now (F3b-ii): the no-keyless-door invariant moved
    from an enabled<->key coupling in the record to :func:`serve_binding`, which
    binds the LAN only when the BROKER's key set is non-empty. Any ``api_key`` on
    disk is a pre-F3b-ii legacy value surfaced for migration, never a live input.
    """
    if not isinstance(raw, Mapping):
        return LlmServerSettings()
    return LlmServerSettings(
        enabled=bool(raw.get("enabled")),
        legacy_api_key=str(raw.get("api_key") or "").strip(),
    )


def serve_binding(
    enabled: bool, api_keys: Sequence[str]
) -> Tuple[str, List[str]]:
    """The ``(listen_host, key_set)`` the auth PROXY should run with (F3b-ii).

    Derived from ``(enabled, active-key-set-from-broker)``. THE no-keyless-door
    invariant: a LAN listen host is returned ONLY when the server is enabled AND
    the key set is non-empty. "Enabled with an EMPTY key set" is loopback and no
    keys, which the controller reads as "run no proxy" - the same refusal
    :func:`vaelor.llm_server_proxy._require_keys` makes at the launch boundary.
    Empty/blank entries are dropped so a stray "" cannot masquerade as a key.
    """
    keys = [str(key) for key in (api_keys or []) if str(key).strip()]
    if enabled and keys:
        return LAN_BIND_HOST, keys
    return LOOPBACK_HOST, []


def external_base_url(address: str, port: int, *, enabled: bool) -> str:
    """The OpenAI-compatible base URL an external client would use, or ``""``.

    Empty when the server is disabled (there is nothing to advertise) or the
    inputs are incomplete, so the surface never invents a URL that does not
    answer. ``address`` is the appliance's LAN address, resolved by the caller.
    """
    clean = str(address or "").strip()
    if not enabled or not clean or int(port or 0) <= 0:
        return ""
    return "http://{}:{}/v1".format(clean, int(port))


def harden_for_jobs_group(path: Path) -> None:
    """Make a record group-readable by the shared jobs group, best effort.

    The executor account (``vaelor-workloads``, group ``vaelor-jobs``) must be
    able to read what the control-plane account (``vaelor``) writes, exactly as
    they share the job queue - and the GPU cluster mode record
    (`gpu_cluster_mode.ClusterModeStore`) crosses the same two accounts in the
    other direction, so it hardens through this one function rather than a
    copy. Every step is optional and swallowed: the mode/group calls are absent
    or unprivileged on a developer box (and on Windows, where ``grp`` does not
    exist), and a failure there must not stop a toggle.
    """
    try:
        os.chmod(path, 0o640)
    except OSError:
        pass
    try:
        import grp
    except ImportError:
        return
    gid = jobs_group_id(grp)
    if gid is None:
        return
    try:
        os.chown(path, -1, gid)
    except OSError:
        pass


class LlmServerStore:
    """Read and write the LLM Server's persisted state.

    Injectable ``path`` so tests drive it against a tmp file with no state root.
    Reads fail safe to a disabled record; writes are atomic (temp file then
    ``os.replace``) and hardened to the shared jobs group.
    """

    def __init__(self, path: str = STATE_FILE):
        self._path = Path(path)

    def read(self) -> LlmServerSettings:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return LlmServerSettings()
        return _coerce(raw)

    def _write(self, settings: LlmServerSettings) -> LlmServerSettings:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        record = {"enabled": settings.enabled}
        # The legacy plaintext key is written back ONLY while it still needs
        # migrating (not yet imported into the broker). Once migrate_legacy_key
        # has imported it, forget_legacy_key drops it and the record is the plain
        # {"enabled": bool} the broker-backed feature means.
        if settings.legacy_api_key:
            record["api_key"] = settings.legacy_api_key
        payload = json.dumps(record, separators=(",", ":"))
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(payload + "\n", encoding="utf-8")
        harden_for_jobs_group(tmp)
        os.replace(tmp, self._path)
        harden_for_jobs_group(self._path)
        return settings

    def write_enabled(self, enabled: bool) -> LlmServerSettings:
        """Persist the on/off flag, PRESERVING any not-yet-migrated legacy key.

        The keys live in the broker now; this record carries only the flag. A
        legacy key still on disk (a pre-F3b-ii box whose migration has not run -
        e.g. the broker was unreachable) is written back untouched, so a disable
        can never strand the one secret an external client is using. Once
        :func:`migrate_legacy_key` imports it, :meth:`forget_legacy_key` drops it.
        """
        current = self.read()
        return self._write(LlmServerSettings(
            enabled=bool(enabled), legacy_api_key=current.legacy_api_key,
        ))

    def forget_legacy_key(self) -> LlmServerSettings:
        """Drop the legacy plaintext key, keeping the flag (post-import cleanup)."""
        current = self.read()
        return self._write(LlmServerSettings(enabled=current.enabled))


def key_fingerprint(api_keys) -> str:
    """A short, non-reversible fingerprint of a key SET, or ``""`` when it is empty.

    Lets the applied-binding marker record WHICH keys the running proxy carries
    without persisting any plaintext (F3b-ii feeds this per-key FINGERPRINTS, not
    plaintext keys - see :func:`binding_marker`). A SET-hash: taken over the
    sorted, de-duplicated inputs, so it is independent of order and moves on any
    add, rotate or revoke - the reconcile then sees the drift and re-keys the gate.
    A single string (or one-item set) is accepted and hashes the same as that one
    item alone, so a single-key box's marker round-trips.
    """
    if isinstance(api_keys, str):
        keys = [api_keys] if api_keys else []
    else:
        keys = [str(key) for key in api_keys if str(key)]
    if not keys:
        return ""
    canonical = "\n".join(sorted(set(keys)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class AppliedBinding:
    """What the auth proxy was last converged to: listen host + key fingerprint.

    The fingerprint is the SET-hash of the endpoint's per-key fingerprints
    (:func:`key_fingerprint` over :func:`~vaelor.served_endpoint_keys.served_key_fingerprint`
    values, F3b-ii), so a one-key gate and a multi-key gate both reduce to one
    comparable field and any add/rotate/revoke shows as drift. The default is the DISABLED
    binding (loopback, no key), so a missing marker on a disabled box reads as
    "already converged" (no proxy should be running) while an enabled state with no
    marker reads as drift and forces one convergence.
    """

    host: str = LOOPBACK_HOST
    key_fingerprint: str = ""


def binding_marker(enabled: bool, api_keys: Sequence[str]) -> AppliedBinding:
    """The applied-binding marker the DESIRED ``(enabled, key-set)`` maps to.

    ``api_keys`` here is the endpoint's per-key FINGERPRINTS, never plaintext
    (F3b-ii): the DESIRED marker builds them straight from the broker's
    fingerprint-only ``list()`` (no decrypt, no per-poll audit) and the APPLIED
    marker is written from :func:`~vaelor.served_endpoint_keys.served_key_fingerprint`
    over the just-applied keys, so both sides share ONE basis and any
    add/rotate/revoke still moves the hash. Derived through :func:`serve_binding` so
    the marker and the proxy launch can never disagree about what "enabled" means:
    the same listen host, and a SET-hash (:func:`key_fingerprint`) of the same set.
    Comparing this against the persisted :class:`AppliedBinding` is how the
    failure-watch reconcile tells a converged proxy from a drifted one.
    """
    host, keys = serve_binding(enabled, api_keys)
    return AppliedBinding(host=host, key_fingerprint=key_fingerprint(keys))


def applied_marker(enabled: bool, api_keys: Sequence[str]) -> AppliedBinding:
    """The marker to COMMIT after the gate was just applied with plaintext keys.

    The per-key fingerprints are taken with
    :func:`~vaelor.served_endpoint_keys.served_key_fingerprint` - the value the
    broker's ``list()`` reports for each key - so the committed marker equals
    the DESIRED one the next poll builds from :func:`active_key_fingerprints`
    without decrypting anything. Both reconciles (Mode A's executor mixin and
    Mode B's mode switch) commit through this one derivation.
    """
    return binding_marker(enabled, [
        served_key_fingerprint(str(key)) for key in (api_keys or [])
        if str(key).strip()
    ])


class AppliedBindingStore:
    """Read and write the applied-binding marker (FINDING A).

    Persisted so the reconcile can compare the proxy's converged binding against
    the desired state even after an executor restart (the proxy, launched by the
    root bridge, outlives the executor). The record holds only a host and a key
    FINGERPRINT - never the plaintext key - so it needs no stronger perms than the
    state record. Reads fail safe to the disabled binding; writes are atomic and
    BEST EFFORT: the marker lives in the executor-owned models root (always
    writable in production), and a write that cannot land (a tmp/absent root in a
    unit test) simply leaves the marker unchanged rather than breaking a launch.
    """

    def __init__(self, path: str):
        self._path = Path(path)

    def read(self) -> AppliedBinding:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return AppliedBinding()
        if not isinstance(raw, Mapping):
            return AppliedBinding()
        return AppliedBinding(
            host=str(raw.get("host") or LOOPBACK_HOST),
            key_fingerprint=str(raw.get("key_fingerprint") or ""),
        )

    def write(self, binding: AppliedBinding) -> AppliedBinding:
        # Only write when the parent directory already exists: in production the
        # models root always does, so a missed write means a genuinely unusable
        # root, not a routine condition; a unit test with a throwaway root simply
        # skips it. Never create the parent (that would litter a dev box with a
        # bogus models tree) and never raise (a marker is a hint, not a barrier).
        if not self._path.parent.is_dir():
            return binding
        payload = json.dumps(
            {"host": binding.host, "key_fingerprint": binding.key_fingerprint},
            separators=(",", ":"),
        )
        try:
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(payload + "\n", encoding="utf-8")
            harden_for_jobs_group(tmp)
            os.replace(tmp, self._path)
            harden_for_jobs_group(self._path)
        except OSError:
            pass
        return binding


#: The endpoint id the LLM Server's keys are bound to in the credential broker.
LLM_SERVER_ENDPOINT_ID = "llm-server"

#: The label a minted or imported LLM Server key carries in the broker.
DEFAULT_KEY_LABEL = "LLM Server key"


@dataclass(frozen=True)
class EnableOutcome:
    """The result of :func:`enable`: the new state, and any one-time reveal.

    ``minted_key`` carries the plaintext of a NEWLY minted first key, revealed
    exactly once to the caller; it is ``""`` when the server already had keys
    (the idempotent case), so the surface shows nothing new and an existing
    external client is undisturbed.
    """

    settings: LlmServerSettings
    minted_key: str = ""


def active_keys(broker) -> List[str]:
    """The LLM Server's active broker keys.

    Raises :class:`~vaelor.credential_broker_client.CredentialError`
    (``BROKER_UNAVAILABLE`` / ``BROKER_NO_ANSWER``) when the broker cannot answer;
    the gate renderer treats that as "leave the running gate exactly as it is"
    rather than as a definitive empty set (the broker-down fail-safe).
    """
    return list(broker.endpoint_keys(LLM_SERVER_ENDPOINT_ID))


def llm_server_rows(broker) -> List[Mapping[str, Any]]:
    """The LLM Server's ACTIVE (non-revoked) served-endpoint rows, fingerprint-only.

    Filtered from the broker's ``list()`` - the same filter the
    ``_keys_for_surface`` GET already applies (served-endpoint provider, this
    endpoint, not revoked) - so both the read surface and the convergence poll read
    ONE definition. ``list()`` never decrypts a key and writes no plaintext-read
    audit, so this is safe on every ~30 s reconcile pass. Propagates
    :class:`~vaelor.credential_broker_client.CredentialError` when the broker cannot
    answer (the caller decides the fail-safe).
    """
    rows = broker.list()
    return [
        row
        for row in (rows or [])
        if row.get("provider") == SERVED_ENDPOINT_PROVIDER
        and str(row.get("endpoint_id") or "") == LLM_SERVER_ENDPOINT_ID
        and not row.get("revoked")
    ]


def active_key_fingerprints(broker) -> List[str]:
    """The sorted set of the LLM Server's active per-key FINGERPRINTS (drift signal).

    The convergence check needs only whether the key set changed, not the keys
    themselves, so this reads the fingerprint-only :func:`llm_server_rows` and never
    decrypts. Sorted and de-duplicated so it matches the marker's set basis. Raises
    :class:`~vaelor.credential_broker_client.CredentialError` on broker-down, which
    the executor turns into "leave the running gate as it is".
    """
    return row_fingerprints(llm_server_rows(broker))


def row_fingerprints(rows) -> List[str]:
    """The sorted, de-duplicated, NON-EMPTY per-key fingerprints of ``rows``.

    The one reading of a listing's key set, shared by the executor's drift poll
    (:func:`active_key_fingerprints`) and the console's status surface, so the
    two cannot disagree about whether a row without a fingerprint is a key: it
    is not, exactly as :func:`serve_binding` drops a blank key.
    """
    return sorted({
        str(row.get("fingerprint") or "").strip() for row in (rows or [])
    } - {""})


def migrate_legacy_key(store, broker) -> None:
    """Import a pre-F3b-ii ``state.json`` key into the broker, once (design B5).

    On the first enable/apply/read after upgrade a box may still carry the old
    plaintext ``api_key``. If the broker holds no ``llm-server`` key yet, that
    exact value is imported so an already-configured external client keeps
    working; then the plaintext is dropped from the record. Idempotent and safe:

    * broker unreachable -> return, keeping the legacy key on disk to retry later
      (never lose the one secret a client is using);
    * the legacy key is ALREADY an active broker key -> just drop the on-disk
      copy (a re-run after a write that could not land);
    * some OTHER key exists but not the legacy one -> leave the record untouched
      (do not clobber; in this phase only migration/enable ever mints a key, so
      this only happens transiently and re-resolves once the import lands).
    """
    settings = store.read()
    legacy = settings.legacy_api_key
    if not legacy:
        return
    try:
        existing = list(broker.endpoint_keys(LLM_SERVER_ENDPOINT_ID))
    except CredentialError:
        return
    if legacy not in existing:
        if existing:
            return
        try:
            broker.import_key(LLM_SERVER_ENDPOINT_ID, legacy, DEFAULT_KEY_LABEL)
        except CredentialError:
            return
    try:
        store.forget_legacy_key()
    except OSError:
        pass


def enable(store, broker) -> EnableOutcome:
    """Turn the LLM Server on, minting a FIRST key only when none exists (item 2).

    Idempotent: a box that already has active broker keys keeps them, so an
    external client stays working and nothing is revealed; only an empty key set
    mints a first key, whose plaintext is revealed ONCE in the outcome. The
    pre-F3b-ii key is migrated FIRST, so re-enabling never mints over the key an
    existing client already holds. Propagates ``CredentialError`` when the broker
    is unreachable (nothing is armed).
    """
    migrate_legacy_key(store, broker)
    minted = ""
    if not active_keys(broker):
        minted = str(broker.mint(LLM_SERVER_ENDPOINT_ID, DEFAULT_KEY_LABEL).get("key", ""))
    settings = store.write_enabled(True)
    return EnableOutcome(settings=settings, minted_key=minted)


def disable(store, broker) -> LlmServerSettings:
    """Turn the LLM Server off WITHOUT destroying its keys (items 2 and 6).

    Disable never revokes the keys: they stay in the broker so re-enabling, or a
    cluster ``leave``, restores the SAME set an external client already holds
    rather than minting a fresh one. The legacy key is migrated first so the flag
    write cannot strand it.
    """
    migrate_legacy_key(store, broker)
    return store.write_enabled(False)
