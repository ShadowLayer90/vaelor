"""The cluster manager's worker-telemetry half: ingest keys, install, remove, heal.

Extracted from `cluster_manager` (which was at 904 lines) so the Recheck, removal
and self-heal fixes of the 2026-09-28 sweep could land without pushing that
module past the ceiling. `ClusterManager` inherits `WorkerTelemetryMixin`; the
methods read the same ``store``, ``broker``, ``transport_factory``,
``last_sample_time`` and ``ingest_status`` attributes the manager sets.

**Which process runs this.** Everything here runs in the CONTROL PLANE: the
operator routes (install, remove, Recheck, reconcile) and the 15-minute
`worker_telemetry_reconcile_scheduler` both call it there, and so does the
telemetry ingest route whose per-node status (`telemetry_ingest_status`) the
liveness check reads - one process, so that in-memory status is the same object
the reconcile consults.

**Every change to a worker's agent is said out loud (ACC-121).** A Recheck and
the self-heal used to re-key or reinstall an agent and throw the result away,
while the Activity page promised every audited change. `RECONCILE_CHANGES` names
the outcomes that changed something on the worker or rotated its key, and
`audit_reconcile` is the one place both the routes and the scheduler record
them, so the two entry points cannot audit differently.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from typing import Any, Callable, Dict, Optional

from .cluster_store import NODE_NOT_FOUND
from .gpu_vendor_status import NOT_APPLICABLE, STATUS_FIELD, known_code
from .platforms.graphics_software import integrated_amd_gpu

LOGGER = logging.getLogger(__name__)

#: The actor the automatic 15-minute repair is recorded under in the audit
#: trail. Plain words, because the Activity page prints it in the User column.
AUTOMATIC_REPAIR_ACTOR = "Vaelor (automatic repair)"

#: How old a worker's hardware snapshot may be before the automatic repair pass
#: re-reads it (ACC-194): under the sweep's 15 minutes, so every sweep does.
CAPACITY_REFRESH_SECONDS = 600

#: The audit action every telemetry reconcile outcome is recorded under.
RECONCILE_AUDIT_ACTION = "cluster.telemetry.reconcile"

#: Reconcile outcomes that changed the worker or its ingest key, each with the
#: sentence an owner reads for it. `healthy` and `warming` changed nothing and
#: are not audited; `absent` (no agent, none expected) and `clock-skew` (a
#: clock no reinstall can fix) are reported but change nothing either.
RECONCILE_CHANGES: Dict[str, str] = {
    "restarted": "The telemetry agent was stopped, so Vaelor started it again.",
    "reshipped-ca": (
        "The telemetry agent trusted an old controller certificate, so Vaelor "
        "sent it the current one and restarted it."
    ),
    "reprovisioned-stale": (
        "The telemetry agent was running but its readings were not arriving, "
        "so Vaelor reinstalled it with a new reporting key."
    ),
    "restarted-backlog": (
        "The telemetry agent was stuck resending readings too old to accept, "
        "so Vaelor restarted it."
    ),
    "reinstalled-absent": (
        "The telemetry agent was missing from the machine, so Vaelor "
        "reinstalled it with a new reporting key."
    ),
    "reprovisioned-drift": (
        "The telemetry agent on the machine was not the version this "
        "controller installs, so Vaelor reinstalled it with a new reporting key."
    ),
    "restarted-sampler": (
        "The GPU sampler on the machine had stopped, so Vaelor started it again."
    ),
}

#: Where a node's drift loop-guard record is kept on its inventory, and the
#: fields kept (pass-2 review SC-B).
DRIFT_RECORD_KEY = "telemetry_repair"
DRIFT_RECORD_FIELDS = ("fingerprint", "attempts", "last_attempt_at", "reported", "error")

#: The reconcile's answer for a JOINED worker whose agent needs reinstalling
#: (review S10): the worker profile job owns telegraf.conf and the agent there,
#: so the reconcile queues that job instead of re-keying the agent itself -
#: one owner of the file (LESSONS 6).
PROFILE_QUEUED = "profile-update-queued"
_REINSTALLS = ("reprovisioned-drift", "reprovisioned-stale", "reinstalled-absent")

#: What the worker telemetry routes answer for a joined worker (owner,
#: 2026-10-05): its telemetry is part of its worker software, laid down and
#: kept up to date by this controller, and comes off when it leaves the cluster.
TELEMETRY_STAYS_WITH_WORKER = (
    "Telemetry is part of a worker's software: it comes with the machine and comes off when the "
    "machine is removed from the cluster. It cannot be removed on its own."
)
TELEMETRY_IS_PROFILE = (
    "This worker's telemetry is part of its worker software, which this controller lays down "
    "and keeps up to date. Recheck the machine to update it."
)

#: Outcomes that changed nothing but that the owner still needs to read.
RECONCILE_NOTES: Dict[str, str] = {
    PROFILE_QUEUED: (
        "The telemetry agent needs reinstalling, so Vaelor queued an update of this "
        "machine's worker software, which reinstalls it."
    ),
    "clock-skew": (
        "The telemetry agent is running, but the machine's clock is too far "
        "from the controller's, so its readings are refused. Reinstalling "
        "cannot fix a clock: turn on network time (NTP) on the machine."
    ),
    "warming": (
        "The telemetry agent started recently; its first readings are still "
        "on the way."
    ),
    "absent": "Telemetry is not set up on this machine.",
    "drift-unresolved": (
        "Vaelor reinstalled the telemetry agent, but what is on the machine "
        "still differs from what this controller installs, so the update could "
        "not be applied. It tries again later, or when you press Recheck."
    ),
    "sampler-restarts-stopped": (
        "The GPU sampler on the machine keeps stopping. Vaelor started it again "
        "three times and has stopped trying on its own; press Recheck to try again."
    ),
    "drift-stopped": (
        "Vaelor reinstalled the telemetry agent several times, but what is on "
        "the machine still differs from what this controller installs. It has "
        "stopped trying on its own; press Recheck to try again."
    ),
}

#: Notes that are also recorded in the audit trail, the first time a pass
#: reports each (review S-5): an update that could not be applied is something
#: the owner must be able to find later, not only read on a Recheck.
AUDITED_NOTES = frozenset({"drift-unresolved", "drift-stopped", "sampler-restarts-stopped"})


def reconcile_sentence(result: Optional[Dict[str, Any]]) -> str:
    """The owner-readable sentence for a reconcile outcome, or ``""``."""
    action = str((result or {}).get("action", "") or "")
    sentence = RECONCILE_CHANGES.get(action) or RECONCILE_NOTES.get(action, "")
    error = str((result or {}).get("error", "") or "")
    if action in AUDITED_NOTES and error:
        # The command's own error from the reinstall that did not take.
        sentence = "{} The last attempt failed: {}".format(sentence, error)
    # A pending drift finding reported beside another repair (review S-1).
    pending = str((result or {}).get("drift", "") or "")
    if pending in RECONCILE_NOTES:
        sentence = "{} {}".format(sentence, RECONCILE_NOTES[pending]).strip()
    return sentence


def repair_note(record: Optional[Dict[str, Any]]) -> str:
    """The Fleet card's sentence for a worker whose update could not be applied, or ``""``.

    ``record`` is the drift loop guard's memory for the node; it says which
    note the last pass reported (review S-5).
    """
    reported = str((record or {}).get("reported", "") or "")
    if reported not in AUDITED_NOTES:
        return ""
    return reconcile_sentence({"action": reported, "error": (record or {}).get("error", "")})


def audit_reconcile(
    audit: Optional[Callable[..., Any]], actor: str, node_id: str,
    result: Optional[Dict[str, Any]] = None, error: Optional[BaseException] = None,
    remote_addr: str = "",
) -> None:
    """Record one reconcile outcome in the audit trail when it changed anything.

    ``audit`` is a `SecurityStore.audit`-shaped callable. A failure is always
    recorded (a reconcile can fail mid-reprovision, after the key was re-minted),
    a state-changing outcome is recorded under its own action word, and a
    no-change outcome is not, to keep the trail readable. Never raises: an audit
    store that cannot be written must not undo a repair that already happened,
    but it is logged rather than swallowed.
    """
    if audit is None:
        return
    if error is not None:
        outcome, details = "failure", {"reason": str(error)[:300]}
    else:
        outcome = str((result or {}).get("action", "") or "")
        if outcome in AUDITED_NOTES:
            if not (result or {}).get("newly"):
                return
            details = {"summary": reconcile_sentence(result)}
        elif outcome not in RECONCILE_CHANGES:
            return
        else:
            details = {"summary": reconcile_sentence(result)}
    try:
        audit(
            actor, RECONCILE_AUDIT_ACTION, outcome, target=str(node_id),
            remote_addr=remote_addr, details=details,
        )
    except Exception:  # noqa: BLE001 - the repair stands; say the record failed
        LOGGER.warning("could not record the telemetry repair for node %s", node_id)


def _database_last_sample_time(node_id: str):
    """Epoch seconds of the newest telemetry row the controller holds for a node.

    Reads this process's telemetry retention source, the same one the ingest
    write appends to. Raises LookupError when retention is switched off, so a
    reconcile treats liveness as unknown and never churns an agent it cannot
    prove is stale.
    """
    from .control_plane import _telemetry_source
    from .telemetry_store import HISTORY_MEASUREMENT

    database = _telemetry_source()
    if database is None:
        raise LookupError("telemetry retention is switched off")
    return database.last_sample_time(HISTORY_MEASUREMENT, node=node_id)


def _database_latest_sample(node_id: str):
    """The newest telemetry row the controller holds for a node, or None.

    Raises LookupError when retention is switched off, exactly as
    :func:`_database_last_sample_time` does, so "no row" and "could not ask"
    stay different answers.
    """
    from .control_plane import _telemetry_source
    from .telemetry_store import HISTORY_MEASUREMENT

    database = _telemetry_source()
    if database is None:
        raise LookupError("telemetry retention is switched off")
    return database.latest_sample(HISTORY_MEASUREMENT, node=node_id)


def _default_ingest_status(node_id: str) -> Dict[str, Any]:
    from .telemetry_ingest_status import ingest_status

    return ingest_status(node_id)


class WorkerTelemetryMixin:
    """Ingest keys and the telemetry agent's lifecycle on enrolled workers."""

    #: Set by `ClusterManager.__init__`; declared here so the mixin reads as
    #: self-contained. `ingest_status` is `telemetry_ingest_status.ingest_status`
    #: unless a test injects a fake of the same shape.
    last_sample_time: Callable[[str], Any]
    ingest_status: Callable[[str], Dict[str, Any]]
    #: The newest stored row for a node (`_database_latest_sample` unless a
    #: test sets one on the instance).
    latest_sample: Callable[[str], Any] = staticmethod(_database_latest_sample)

    def _drift_guard(self) -> Dict[str, Dict[str, Any]]:
        """Per-node memory of the last reinstall-for-drift (the loop guard)."""
        guard = getattr(self, "_telemetry_drift", None)
        if guard is None:
            guard = self._telemetry_drift = {}
        return guard

    def _drift_guard_for(self, node: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """The loop guard, with this node's record read back from its stored inventory.

        The record is kept on the node (pass-2 review SC-B), so a control-plane
        restart neither allows three more reinstalls (each one a new reporting
        key) nor loses the Fleet card's sentence.
        """
        guard = self._drift_guard()
        stored = (node.get("inventory") or {}).get(DRIFT_RECORD_KEY)
        if node.get("id") not in guard and isinstance(stored, dict) and stored:
            guard[node["id"]] = dict(stored)
        return guard

    def _save_drift_record(self, node_id: str) -> None:
        """Write this node's loop-guard record onto its stored inventory (or clear it)."""
        node = self.store.get_node(node_id) if hasattr(self.store, "get_node") else getattr(self.store, "node", None)
        if node is None:
            return
        inventory = dict(node.get("inventory") or {})
        record = self._drift_guard().get(node_id)
        if record:
            inventory[DRIFT_RECORD_KEY] = {key: record.get(key) for key in DRIFT_RECORD_FIELDS}
        elif DRIFT_RECORD_KEY in inventory:
            inventory.pop(DRIFT_RECORD_KEY)
        else:
            return
        self.store.update_node(node_id, inventory=inventory)

    def wants_gpu_sampler(self, node: Dict[str, Any]) -> bool:
        """Whether this worker should run the GPU vendor sampler (VD-147).

        **The worker's own emitter is the single answer**: its newest row says
        whether graphics readings apply to the machine (any status but "none
        apply" means yes). Only before a first row carries a status - a worker
        just enrolled, or one still on an agent from before these readings -
        does this fall back to what enrolment discovered about its GPU. So a
        worker without an AMD integrated GPU is never asked for a sampler, and
        never put into a reinstall loop over one.
        """
        try:
            row = self.latest_sample(node["id"])
        except Exception:  # noqa: BLE001 - an unreadable store is "no row yet"
            row = None
        code = known_code(row.get(STATUS_FIELD)) if isinstance(row, dict) else None
        if code is not None:
            return code != NOT_APPLICABLE
        inventory = node.get("inventory") or {}
        gpu = inventory.get("gpu") if isinstance(inventory.get("gpu"), dict) else {}
        return integrated_amd_gpu({
            "unified_memory": gpu.get("unified_memory"),
            "vendor": inventory.get("gpu_vendor"),
        })

    def _refresh_gpu_limits(self, node: Dict[str, Any], transport) -> None:
        """Store the GPU's own temperature limits the sampler reported, if any.

        Read with the ``cat`` the reconcile already uses (no allowlist is
        widened), parsed and bounded by `gpu_vendor_sample.parse_limits`, and
        kept on the node's inventory as ``gpu_temperature_limits``. Limits are
        static, so this rides on the repair pass rather than on every sample.
        Best-effort: a worker with no sampler simply has none.
        """
        import json

        from .gpu_vendor_sample import SAMPLE_PATH, parse_limits
        from .ssh_transport import SshTransportError

        try:
            raw = json.loads(str(transport.run(["cat", SAMPLE_PATH])))
        except (SshTransportError, ValueError):
            return
        limits = parse_limits(raw.get("limits") if isinstance(raw, dict) else None)
        inventory = self._stored_inventory(node)
        if inventory.get("gpu_temperature_limits", {}) == limits:
            return
        inventory["gpu_temperature_limits"] = limits
        self.store.update_node(node["id"], inventory=inventory)

    def _stored_inventory(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """The node's inventory as the store holds it NOW (pass-3 review SC-B).

        A refresh that wrote back the copy read before the repair pass replaced
        the whole inventory and dropped what the pass had saved on it - the
        loop-guard record among it. Each refresh merges into a fresh read.
        """
        reader = getattr(self.store, "get_node", None)
        current = reader(node["id"]) if callable(reader) else None
        return dict(((current or node).get("inventory")) or {})

    def _refresh_gpu_identity(self, node: Dict[str, Any], transport) -> Dict[str, Any]:
        """Fill in a worker's GPU vendor and sensor label when its enrolment predates them.

        Read BEFORE the repair decides whether the worker should run a GPU
        sampler (review S-6): a worker enrolled by an older controller has no
        ``gpu_vendor`` and its older agent reports no status, so without this a
        controller-only upgrade took two passes, two reinstalls and two new
        reporting keys to converge. Best-effort; returns the node as it now is.
        """
        inventory = dict(node.get("inventory") or {})
        if inventory.get("gpu_vendor") and "gpu_temperature_label" in inventory:
            return node
        reader = getattr(transport, "gpu_identity", None)
        found = reader() if callable(reader) else {}
        changed = False
        for key, value in (("gpu_vendor", found.get("vendor")),
                           ("gpu_temperature_label", found.get("temperature_label"))):
            if isinstance(value, str) and value and inventory.get(key) != value:
                inventory[key] = value
                changed = True
        if not changed:
            return node
        self.store.update_node(node["id"], inventory=inventory)
        return {**node, "inventory": inventory}

    def _refresh_machine_class(self, node: Dict[str, Any], transport) -> None:
        """Fill in a worker's machine class when its enrolment predates it."""
        inventory = self._stored_inventory(node)
        if inventory.get("machine_class"):
            return
        reader = getattr(transport, "machine_class", None)
        found = reader() if callable(reader) else ""
        if found:
            inventory["machine_class"] = found
            self.store.update_node(node["id"], inventory=inventory)

    def _last_report_time(self, node_id: str):
        """When the controller last heard from a node's agent, in controller time.

        Two readings: the moment the ingest route last ACCEPTED a post from it
        (the controller's clock, lost on a restart) and the row time the store
        holds (stamped by the WORKER's clock). The receipt decides whenever
        there is one - `telemetry_ingest_status.reporting_reading`, the rule
        the reporting verdict uses - so a worker whose clock runs behind is not
        silent (ACC-126) and one whose clock runs ahead is not fresh; the row
        time is read only when nothing has been received since a restart.
        """
        from .telemetry_ingest_status import reporting_reading

        stored = self.last_sample_time(node_id)
        try:
            received = (self.ingest_status(node_id) or {}).get("last_accepted_at")
        except Exception:  # noqa: BLE001 - an unreadable status is no receipt
            received = None
        return reporting_reading(received, stored)

    def _refusal_in_force(self, node_id: str) -> str:
        """The ingest refusal that still describes this node, or ``""``.

        `telemetry_ingest_status.refusal_in_force` is the one rule, shared with
        the Fleet card: only the NEWEST outcome counts (a worker whose next post
        was accepted is no longer refused) and only while it is recent (a worker
        refused and then silent is a silent worker, repaired like one - review
        SC3). An unreadable status is no refusal.
        """
        try:
            from .telemetry_ingest_status import refusal_in_force

            return refusal_in_force(self.ingest_status(node_id) or {})
        except Exception:  # noqa: BLE001 - unknown is not refused
            return ""

    def _clock_refused(self, node_id: str) -> bool:
        """True while this node's posts are being refused for its CLOCK."""
        from .telemetry_ingest_status import CLOCK_SKEW_REFUSAL

        return self._refusal_in_force(node_id) == CLOCK_SKEW_REFUSAL

    def _backlog_refused(self, node_id: str) -> bool:
        """True while this node's agent is stuck resending a too-old backlog."""
        from .telemetry_ingest_status import BACKLOG_REFUSAL

        return self._refusal_in_force(node_id) == BACKLOG_REFUSAL

    @staticmethod
    def _ingest_key_hash(key: str) -> str:
        """The sha256 hex of an ingest key - the only form the store ever holds."""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _stored_key_hash(self, node_id: str) -> str:
        for stored_id, stored_hash in self.store.ingest_key_hashes():
            if stored_id == node_id:
                return str(stored_hash)
        return ""

    def mint_ingest_key(self, node_id: str) -> str:
        """Issue a telemetry ingest key for a node, storing ONLY its hash.

        Returns the plaintext key exactly once: the caller hands it to the
        worker, and it is never stored in the clear, logged, or recoverable from
        the store. Re-minting replaces the stored hash, so an old key stops
        working the moment a new one is issued. Raises when the node is unknown.
        """
        node = self.store.get_node(node_id)
        if node is None:
            raise ValueError(NODE_NOT_FOUND)
        key = "vnk_{}".format(secrets.token_urlsafe(32))
        self.store.set_ingest_key_hash(node_id, self._ingest_key_hash(key))
        return key

    def node_for_ingest_key(self, presented_key: str) -> Optional[str]:
        """The node a presented ingest key authenticates to, or ``None``.

        Compared in CONSTANT time against every stored hash, even after a match,
        so the time taken does not reveal which node matched. An empty or
        non-string key matches nothing.
        """
        if not isinstance(presented_key, str) or not presented_key:
            return None
        presented_hash = self._ingest_key_hash(presented_key)
        matched: Optional[str] = None
        for node_id, stored_hash in self.store.ingest_key_hashes():
            if hmac.compare_digest(presented_hash, str(stored_hash)):
                matched = node_id
        return matched

    def clear_ingest_key(self, node_id: str) -> bool:
        """Revoke a node's telemetry ingest key by clearing its stored hash.

        `store.ingest_key_hashes` omits an empty hash, so the old key
        authenticates to nothing the moment this runs, while the node stays
        enrolled.
        """
        node = self.store.get_node(node_id)
        if node is None:
            raise ValueError(NODE_NOT_FOUND)
        return self.store.set_ingest_key_hash(node_id, "")

    def telemetry_provisioned(self, node_id: str) -> bool:
        """Whether Vaelor holds a reporting key for this node's agent."""
        return bool(self._stored_key_hash(node_id))

    def _worker_transport(self, node: Dict[str, Any]):
        """A pinned SSH transport to an enrolled worker, from its stored profile."""
        profile = self.broker.resolve(node["credential_id"], "cluster-node")
        return self.transport_factory(profile)

    def _controller_ingest_url(self) -> str:
        """The keyed ingest URL a worker posts telemetry to (one derivation, shared
        with the profile job: `cluster_worker_profile.controller_ingest_url`)."""
        from .cluster_worker_profile import controller_ingest_url

        return controller_ingest_url(self.store)

    def install_worker_telemetry(self, node_id: str) -> Dict[str, str]:
        """Install the Telegraf telemetry agent onto one enrolled worker (E2b).

        Mints a fresh ingest key, then ships and enables the whole agent stack
        over the worker's pinned SSH transport. **A failed install leaves the key
        exactly as it found it (ACC-089).** It used to keep the new key, so the
        card offered Remove telemetry for an agent that never started and no
        Install to try again; the previous hash - empty for a first install - is
        put back before the error is raised.
        """
        from .cluster_worker_profile import controller_ca_source
        from .worker_telemetry_runtime import WorkerTelemetryRuntime

        node = self.store.get_node(node_id, include_credential=True)
        if node is None:
            raise ValueError(NODE_NOT_FOUND)
        architecture = str((node.get("inventory") or {}).get("architecture", ""))
        url = self._controller_ingest_url()
        previous = self._stored_key_hash(node_id)
        key = self.mint_ingest_key(node_id)
        controller_ca = controller_ca_source()
        try:
            transport = self._worker_transport(node)
            return WorkerTelemetryRuntime().install(
                transport, node_id=node_id, architecture=architecture,
                ingest_url=url, ingest_key=key, controller_ca_source=controller_ca,
                gpu_sampler=self.wants_gpu_sampler(node),
            )
        except BaseException:
            self.store.set_ingest_key_hash(node_id, previous)
            raise

    def telemetry_owned_by_profile(self, node: Dict[str, Any]) -> bool:
        """Whether this worker's agent belongs to its worker profile (joined, with a job queue)."""
        return bool(getattr(self, "job_store", None) is not None
                    and (node.get("labels") or {}).get("swarm_node_id"))

    def _refresh_capacity(self, node_id: str) -> None:
        """Re-read a worker's hardware on the automatic pass, as Recheck does (ACC-194).

        A worker's card - its GPU memory pool above all - came only from its
        enrolment and from Recheck, so a pool resized since (30.4 to 44 GiB on
        the lab pair) read the old size until someone pressed Recheck, while
        Machine settings already showed the new one. The capacity half of
        Recheck now runs on the repair pass too, gated by the fit's own
        freshness rule so it costs at most one probe per sweep and never
        touches the telemetry agent. Best effort: a failure keeps the snapshot.
        """
        refresh = getattr(self, "refresh_nodes_for_fit", None)
        if not callable(refresh):
            return
        try:
            refresh([node_id], freshness_seconds=CAPACITY_REFRESH_SECONDS)
        except Exception as error:  # noqa: BLE001 - reported, the repair goes on
            LOGGER.warning("could not re-read the hardware of node %s: %s", node_id, error)

    def reconcile_worker_telemetry(self, node_id: str, recheck: bool = False) -> Dict[str, str]:
        """Bring back a down, missing, stale-CA or reporting-stale agent.

        Called by the operator reconcile route, by Recheck, and by the 15-minute
        `worker_telemetry_reconcile_scheduler` in `control_plane_runtime` (this
        docstring used to say there was no periodic caller; there is). An
        unreachable worker raises through the transport and is reported, never
        evicted.

        ``recheck`` is the owner pressing Recheck: it retries a reinstall for
        drift that the loop guard would otherwise leave until its window passes.

        A MISSING agent is reinstalled only when Vaelor holds a key for it,
        i.e. the owner installed telemetry and something removed it since; a
        worker the owner never set up is left alone. A running agent whose
        newest post was refused for its CLOCK is not reinstalled - that cannot
        fix a clock, and doing it every 15 minutes was the loop ACC-126 found.
        """
        from .cluster_worker_profile import controller_ca_source
        from .worker_telemetry_runtime import WorkerTelemetryRuntime

        if not recheck:
            # Recheck re-reads the hardware itself before it gets here.
            self._refresh_capacity(node_id)
        node = self.store.get_node(node_id, include_credential=True)
        if node is None:
            raise ValueError(NODE_NOT_FOUND)
        joined = self.telemetry_owned_by_profile(node)
        # Review S10: on a joined worker the profile job owns the agent, its CA
        # and telegraf.conf; this pass may restart a stopped agent but never
        # re-keys or reinstalls it - it queues that job instead.
        controller_ca = None if joined else controller_ca_source()
        queued: Dict[str, bool] = {}

        def reprovision():
            if not joined:
                return self.install_worker_telemetry(node_id)
            from .worker_profile_job import RECHECK_REASON, SWEEP_REASON, queue_profile_job

            queued["yes"] = bool(queue_profile_job(self.job_store, node_id,
                                                   RECHECK_REASON if recheck else SWEEP_REASON))
            return {}

        transport = self._worker_transport(node)
        try:
            node = self._refresh_gpu_identity(node, transport)
        except Exception as error:  # noqa: BLE001 - reported, the repair goes on
            LOGGER.warning("could not refresh the GPU identity of node %s: %s", node_id, error)
        try:
            result = WorkerTelemetryRuntime().reconcile(
                transport,
                controller_ca_source=controller_ca,
                node_id=node_id,
                last_sample_time=self._last_report_time,
                reprovision=reprovision,
                heal_absent=self.telemetry_provisioned(node_id),
                clock_refused=lambda: self._clock_refused(node_id),
                backlog_refused=lambda: self._backlog_refused(node_id),
                gpu_sampler=self.wants_gpu_sampler(node),
                drift_guard=self._drift_guard_for(node),
                retry_drift=recheck,
            )
        finally:
            # Saved even when the reinstall raised (pass-3 review SC-B): the
            # attempt it counted is what stops the next pass from looping.
            try:
                self._save_drift_record(node_id)
            except Exception as error:  # noqa: BLE001 - reported, the repair stands
                LOGGER.warning("could not keep the repair record of node %s: %s", node_id, error)
        if queued.get("yes") and str(result.get("action")) in _REINSTALLS:
            # Nothing was reinstalled here; the queued profile job does it.
            result = {**result, "action": PROFILE_QUEUED}
        # Two static facts the per-node GPU temperature bands need (VD-147),
        # refreshed on this same pass; neither may fail the repair itself.
        for refresh in (self._refresh_gpu_limits, self._refresh_machine_class):
            try:
                refresh(node, transport)
            except Exception as error:  # noqa: BLE001 - reported, the repair stands
                LOGGER.warning(
                    "could not refresh GPU limit facts for node %s: %s", node_id, error
                )
        return result
