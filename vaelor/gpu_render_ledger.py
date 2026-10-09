"""What a cluster GPU deployment's units were rendered from, and whether this release renders them differently.

W4-D1 (2026-10-01), the VD-081 class one tier over. An upgrade replaced the code
that renders a vLLM deployment's systemd units and left every running unit at
the previous release's text: ``ROCPROFILER_REGISTER_ENABLED=0`` (VD-165) was
installed and inert on both machines of a replicated deployment, and the idle
engine went on spinning a core at 100 %. The post-upgrade refresh only knew the
Mode A Assistant compose; nothing knew what a pooled GPU deployment's units said.

**The record now says what its units were rendered FROM, and a digest of what
they said.** Every serving unit the runtime installs - a replica, a split's lead
and Ray workers, a worker's gate - is noted here at the moment it is installed
(`note`, called from the runtime's own install path), with its kind, the typed
values the one template rendered it from (VD-143) and a SHA-256 of the text. A
Load or a deploy wraps its serve in :func:`recording` and stamps the notes into
the row (:data:`RENDERS_FIELD`). The agent tier's ``surface_digest`` is the same
shape: what the running thing was started with, against what a relaunch would
start now.

**The comparison re-renders through the same template, never a second one.**
:func:`render_drift` renders each recorded unit's values with the code that is
installed now and compares the digest. Every byte of the text counts - an
environment line, a flag, a slice check, the description - because nothing is
normalised away before hashing. A gate's config is part of its digest, rendered
with a fixed placeholder in place of the cluster key, so a key rotation is not a
render change and no key is ever hashed into the row.

**A row that recorded nothing was served by a release from before this ledger**
and is reported as needing a refresh: what its units say is unknown, and the
release that served it is older than the one installed. That is the honest
answer for the live defect, not a guess.

What this does NOT cover, and why that is acceptable: the replica balancer and
a split's slice/firewall files. The balancer's config is rendered root-side and
converged from the row by the mode reconcile after every restart of the bridge,
which the installer performs; a split's plane is re-written by every Load. The
controller's bridge renders an MoE model's entry-program mount from its own
installed path, which no client value reaches; both sides of the comparison
here render the client's, so a release that moved only that path is not seen.
Files a unit mounts rather than states - that entry program's content, a
split's Ray token, nftables ruleset and slice unit - are outside the digest
for the same reason; a split's are re-written by every Load.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
from typing import Any, Dict, Iterator, List, Mapping, Optional

from .gpu_cluster_mode import HEALTHY_STATE, UNLOADED_STATE

#: Where a vLLM row keeps the render ledger, in its ``units`` blob.
RENDERS_FIELD = "renders"

#: Where a row keeps what an upgrade refresh could not do (`refresh_note`).
REFRESH_NOTE_FIELD = "refresh_note"

#: A worker gate's kind in the ledger. The runtime's managed-unit kinds name the
#: vLLM containers (VD-143); the gate is the one serving unit that is not one.
GATE_KIND = "gate"

#: The key a gate's config is rendered with for its digest: a fixed word, so the
#: digest follows the template and never the cluster key (which rotates, and
#: must not be hashed into a row).
_PLACEHOLDER_KEY = "render-ledger-placeholder"

#: The kinds a ledger entry may name. A model-library pull is a oneshot that
#: serves nothing, so it is not a serving render and is never noted.
_SERVING_KINDS = ("replica", "distributed-lead", "ray-worker", GATE_KIND)

_LOCAL = threading.local()

#: The step that applies a release to a vLLM row, by the row's state: a
#: serving row is unloaded and loaded, an unloaded one only loaded. One home
#: for the refresh note here and the split fence note (`cluster_manager`).
NEXT_STEP = {HEALTHY_STATE: "Unload, then Load it", UNLOADED_STATE: "Load it"}


@contextlib.contextmanager
def recording() -> Iterator[List[Any]]:
    """Collect every serving unit installed on this thread until the block ends.

    Thread-local, because the executor's job thread is the only one that
    installs units for a job and the mode watch runs beside it; nested blocks
    restore the outer journal.
    """
    previous = getattr(_LOCAL, "journal", None)
    journal: List[Any] = []
    _LOCAL.journal = journal
    try:
        yield journal
    finally:
        _LOCAL.journal = previous


def note(transport: Any, kind: str, values: Mapping[str, Any]) -> None:
    """Record one installed serving unit, when a :func:`recording` is open.

    Called by the runtime after the unit is installed, never before, so a
    unit that failed to install is not recorded as running. Outside a
    recording (a model-library pull, a test that drives the runtime alone)
    this does nothing.
    """
    journal = getattr(_LOCAL, "journal", None)
    if journal is None or str(kind) not in _SERVING_KINDS:
        return
    plain = _plain(values)
    try:
        unit, digest = render_digest(str(kind), plain)
    except (AttributeError, KeyError, TypeError, ValueError):
        # The unit is installed and serving; a ledger that cannot describe it
        # must not fail that. Left unrecorded, the next upgrade refreshes it.
        return
    journal.append((transport, str(kind), plain, unit, digest))


def entries(journal: List[Any], transports: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The journal as row entries, each naming the node its transport reached.

    Matched by identity against the serve's own ``{node id: transport}``, so a
    node is named by the record's own id, never by an address. A unit noted
    twice on one node (a retried start) keeps its last render.
    """
    node_of = {id(transport): str(node_id) for node_id, transport in transports.items()}
    kept: Dict[tuple, Dict[str, Any]] = {}
    for transport, kind, values, unit, digest in journal:
        node = node_of.get(id(transport), "")
        kept[(node, unit)] = {
            "node": node, "unit": unit, "kind": kind, "values": values,
            "digest": digest,
        }
    return list(kept.values())


def render_digest(kind: str, values: Mapping[str, Any]) -> tuple:
    """``(unit name, SHA-256 hex)`` of what the installed code renders for these values.

    A replica, a split's lead or a Ray worker is rendered by
    `GpuPoolRuntime.render_managed_unit`, THE template both writers use; a gate
    by `GpuPoolRuntime.render_gate` with its config, the key a placeholder.
    Raises ``ValueError`` when the template no longer takes these values.
    """
    from .gpu_pool_runtime import GpuPoolRuntime  # the runtime imports this module

    if kind not in _SERVING_KINDS:
        # A model pull, or a word no serving unit carries: not a render this
        # ledger compares, so never answered with a digest of something else.
        raise ValueError("A {} unit is not a serving render.".format(kind))
    runtime = GpuPoolRuntime()
    if not isinstance(values, Mapping):
        raise ValueError("A recorded unit's values are not named values.")
    if kind == GATE_KIND:
        unit, text, config = runtime.render_gate(
            name=values.get("name"), listen_host=values.get("listen_host"),
            port=values.get("port"), api_key=_PLACEHOLDER_KEY,
        )
        body = "{}\0{}\0{}".format(unit, text, config)
    else:
        unit, text = runtime.render_managed_unit(kind, dict(values))
        body = "{}\0{}".format(unit, text)
    return unit, hashlib.sha256(body.encode("utf-8")).hexdigest()


def render_drift(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Whether an upgrade must re-render this vLLM row, and why, in plain words.

    ``{"refresh": bool, "reason": str, "changed": [{"node", "unit"}]}``. Only a
    ``healthy`` row is ever refreshed: an unloaded one renders from the
    installed code on its next Load, and a row deploying or failed is not
    serving anything an upgrade could re-render.
    """
    units = record.get("units") or {}
    state = str(record.get("state", "") or "")
    if state == UNLOADED_STATE:
        return _verdict(False, "unloaded; it renders from this release on its next Load")
    if state != HEALTHY_STATE:
        return _verdict(False, "{}, not serving; left as it is".format(state or "unknown"))
    recorded = [entry for entry in units.get(RENDERS_FIELD) or [] if isinstance(entry, dict)]
    if not recorded:
        return _verdict(True, (
            "served by a release that did not record what its units say, so "
            "this release renders them again"
        ))
    changed = []
    for entry in recorded:
        try:
            _unit, digest = render_digest(str(entry.get("kind", "")), entry.get("values") or {})
        except (AttributeError, KeyError, TypeError, ValueError):
            digest = ""
        if digest != entry.get("digest"):
            changed.append({"node": str(entry.get("node", "")), "unit": str(entry.get("unit", ""))})
    if not changed:
        return _verdict(False, "its units are already what this release renders")
    return _verdict(True, "this release renders {} differently".format(", ".join(
        "{} on {}".format(item["unit"], item["node"] or "a machine with no id")
        for item in changed
    )), changed)


def refresh_note(units: Mapping[str, Any], state: str) -> str:
    """The owner's sentence for a refresh that did not finish, ``""`` when none.

    Worded from the row's state, like the split fence note: the step that
    applies this release's units from where the row now stands.
    """
    note_record = (units or {}).get(REFRESH_NOTE_FIELD)
    if not isinstance(note_record, dict):
        return ""
    reason = str(note_record.get("reason", "") or "")
    detail = str(note_record.get("detail", "") or "").strip()
    step = NEXT_STEP.get(str(state), "")
    if not step:
        return ""
    if reason == "unreachable":
        machines = ", ".join(str(name) for name in note_record.get("machines") or [])
        return ("Upgraded but not yet refreshed on {}, which could not be reached: "
                "{} to apply this release's serving settings.").format(
                    machines or "a machine", step)
    if reason == "load-failed":
        return ("The upgrade stopped this model to apply this release's serving "
                "settings and could not start it again: {} {} to retry.").format(
                    _sentence(detail), step)
    # Refused before anything stopped, or stopped and put back: the detail is
    # the operation's own sentence, which already says what to do.
    return "Upgraded but not yet refreshed: {}".format(_sentence(detail) or "see the job.")


def _sentence(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _verdict(refresh: bool, reason: str, changed: Optional[List[Dict[str, str]]] = None):
    return {"refresh": refresh, "reason": reason, "changed": list(changed or [])}


def _plain(values: Mapping[str, Any]) -> Dict[str, Any]:
    """The values as the row will store them, so a digest taken now and one
    taken from the stored row later are taken from the same thing."""
    return json.loads(json.dumps(dict(values), default=str))
