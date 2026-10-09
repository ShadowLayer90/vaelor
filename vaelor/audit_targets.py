"""What an audit row's target is, in words, resolved where its type is known.

W6-D1: the Security audit trail's Target column showed raw identifiers - a hex
trigger id, ``conv_``/``mem_``/``cred_`` ids and job UUIDs - and "Approved an
operation" never said which operation. The frontend could only have guessed a
type from an id's prefix, and a prefix is not a type (LESSONS 6: two mechanisms
answering "what is this?"). The *action* is what knows the target's type: the
code that writes ``assistant.memory.delete`` writes a memory id. So the type
comes from ``audit_target_kinds``, keyed by action, and the name from the store
that owns that kind of thing.

W7-D1: every action is declared in ``audit_target_kinds``, and an action that
is not is ``unknown`` - its raw id goes behind the disclosure with a generic
word. The first version showed any undeclared action's target as is, which put
container, key, agent, skill and machine ids on screen.

Every row gets ``target_view``: ``{"kind", "label", "known"}``. ``label`` is
never empty. A target that cannot be named says so ("a chat that no longer
exists"); the raw id stays in ``target`` for the details disclosure. An
operation also carries ``job_type``, and the console names it through its one
owner of job words (``jobLabels``) rather than a second copy here.

Nothing here reveals a secret: a key is named by its label and the last four
characters of its fingerprint, which is a digest of the key, never the key.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from .audit_target_kinds import (
    FIXED_TARGETS,
    KIND_WORDS,
    NAMED_TARGETS,
    NO_TARGET_ACTIONS,
    REFUSED_RESULTS,
    RESOLVED_KINDS,
    TARGET_KINDS,
    UNDECLARED_LABEL,
    gone_phrase,
    unread_phrase,
)
from .operation_projection import LEDGER_JOBS, parse_operation_id

#: What a target of each kind is called when it can no longer be read, where
#: the generic "<a word> that no longer exists" would say less than it knows.
CHAT_GONE = "a chat that no longer exists"
GONE: Dict[str, str] = {
    "operation": "an operation that is no longer recorded",
    "assistant_chat": CHAT_GONE,
    "ai_chat": CHAT_GONE,
    "alert_rule": "an alert rule that has since been deleted",
}

#: Said for a row that names no single item (a sign-in, a backup run).
NO_TARGET = "Not about one item"

MEMORY_PREVIEW_WORDS = 6


def _first_words(text: str, count: int = MEMORY_PREVIEW_WORDS) -> str:
    words = str(text or "").split()
    preview = " ".join(words[:count])
    return preview + ("…" if len(words) > count else "")


class _Unreadable:
    """A store read that failed, kept apart from a read that found nothing."""


UNREADABLE = _Unreadable()

#: What a target of each kind is called when its store did not answer (W7-3:
#: a locked database is not a deleted chat - LESSONS 8).
NOT_READ: Dict[str, str] = {
    "operation": "an operation whose record could not be read just now",
    "ai_chat": "an AI Chat that could not be read just now",
}

#: The kind without its private label, for a viewer who may not read the item
#: itself (W7-1, LESSONS 18). An operator reads the trail; that does not make
#: another user's chat title or an administrator's memory theirs to read.
WITHHELD: Dict[str, str] = {
    "assistant_chat": "Chat (another user's)",
    "ai_chat": "AI Chat (another user's)",
    "agent": "Agent (another user's)",
    "automation": "Automation (another user's)",
    "memory": KIND_WORDS["memory"],
    "key": KIND_WORDS["key"],
    "alert_rule": KIND_WORDS["alert_rule"],
}

#: How a named item of each newly resolved kind reads (W7-D1).
LABEL_FORMS: Dict[str, str] = {
    "app": "App: {}",
    "agent": "Agent: {}",
    "automation": "Automation: {}",
    "skill": "Skill: {}",
    "mcp_server": "MCP server: {}",
    "node": "Machine: {}",
    "backup": "Cluster backup of {}",
}

#: The ``details`` fields a row may keep when its kind is withheld from the
#: viewer (W8-1). An allowlist, so a field written later is redacted until
#: someone decides it is safe: the first scope covered the label and left
#: every row's ``details`` and raw ``target`` going out as written, and an
#: operator read a deleted alert rule's name from the JSON (LESSONS 18).
SAFE_WITHHELD_DETAILS: Dict[str, tuple] = {
    "assistant_chat": (),
    "ai_chat": (),
    "agent": (),
    "automation": (),
    "memory": (),
    # The job a key change ran as is an operation, which Activity shows.
    "key": ("job_id",),
    "alert_rule": (),
}

ADMINISTRATOR = "administrator"
#: Kinds listed to their author only (an administrator reads all automations).
OWN_ONLY = ("assistant_chat", "ai_chat", "agent")
#: Kinds whose names only an administrator reads.
ADMINISTRATOR_ONLY = ("memory", "key")


def _safe(read: Callable[[], Any], default: Any) -> Any:
    """``read()``; ``default`` when it found nothing; UNREADABLE when it failed."""
    try:
        found = read()
    except Exception:  # noqa: BLE001  # absence-ok: a failed read is reported as UNREADABLE, never as gone
        return UNREADABLE
    return default if found is None else found


def _named(found: Any, field: str = "name") -> Any:
    """A record's ``field``, None when there is no record, UNREADABLE as is."""
    if found is UNREADABLE:
        return UNREADABLE
    if isinstance(found, dict) and str(found.get(field) or "").strip():
        return str(found[field]).strip()
    return None


def describe_targets(
    events: List[Dict[str, Any]],
    callbacks: Dict[str, Any],
    viewer: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Add ``target_view`` to every event, naming only what ``viewer`` may read.

    ``viewer`` is ``{"username", "role"}`` of the signed-in account. W7-1: the
    first version read every store with no scope, and with the ROW's actor for
    AI Chat, so an operator read the administrator's chat titles and memory
    text here while every route that owns them refused (LESSONS 18). Each kind
    uses the scope its own listing route applies: conversations and custom
    agents are the viewer's own; memories and keys are administrator-only; an
    alert rule or an automation is the viewer's own, or any for an
    administrator; operations, apps, skills, MCP servers, machines and cluster
    backups are anyone's who may read the trail (operator and up), as their
    own lists are.
    """
    username = str(viewer.get("username") or "")
    administrator = viewer.get("role") == ADMINISTRATOR
    names: Dict[str, Dict[str, Any]] = {kind: {} for kind in RESOLVED_KINDS}
    withheld: Dict[str, set] = {kind: set() for kind in RESOLVED_KINDS}

    def kind_of(event: Dict[str, Any]) -> Any:
        return TARGET_KINDS.get(str(event.get("action", "")))

    wanted: Dict[str, set] = {}
    for event in events:
        kind, item = kind_of(event), str(event.get("target") or "")
        if kind not in RESOLVED_KINDS or not item:
            continue
        own = str(event.get("actor") or "") == username
        if kind in OWN_ONLY and not own:
            withheld[kind].add(item)
        elif kind in ADMINISTRATOR_ONLY and not administrator:
            withheld[kind].add(item)
        else:
            wanted.setdefault(kind, set()).add(item)

    _assistant_names(callbacks, username, wanted, names)
    _operation_names(callbacks, events, kind_of, names)
    _key_names(callbacks, wanted, names)
    _alert_rule_names(callbacks, username, administrator, wanted, names, withheld)
    _ai_chat_names(callbacks, username, wanted, names)
    _store_names(callbacks, username, administrator, wanted, names, withheld)

    for event in events:
        event["target_view"] = _view(event, names, withheld, administrator, username)
        if _withheld(event, withheld, username):
            _redact(event)
    return events


def _withheld(event: Dict[str, Any], withheld: Dict[str, set], username: str) -> bool:
    """Whether this viewer is kept from the row's item: one answer, read by
    the label (``_view``) and by the payload redaction alike (W8-1)."""
    kind = TARGET_KINDS.get(str(event.get("action", "")))
    target = str(event.get("target") or "")
    if kind not in RESOLVED_KINDS or not target:
        return False
    own = str(event.get("actor") or "") == username
    return target in withheld.get(kind, ()) and not (kind == "alert_rule" and own)


def _redact(event: Dict[str, Any]) -> None:
    """Drop the raw target and every detail not listed safe for the kind."""
    kind = TARGET_KINDS.get(str(event.get("action", "")))
    safe = SAFE_WITHHELD_DETAILS.get(kind, ())
    details = event.get("details")
    event["details"] = {
        field: value for field, value in details.items() if field in safe
    } if isinstance(details, dict) else {}
    event["target"] = ""


def _assistant_names(callbacks, username, wanted, names) -> None:
    assistant = callbacks.get("assistant_memory")
    if assistant is None or not (wanted.get("assistant_chat") or wanted.get("memory")):
        return
    found = _safe(lambda: assistant.audit_names(
        username, wanted.get("assistant_chat", ()), wanted.get("memory", ())), {})
    for kind, form in (("assistant_chat", "Chat: {}"), ("memory", "Memory: {}")):
        for item in wanted.get(kind, ()):
            if found is UNREADABLE:
                names[kind][item] = UNREADABLE
            elif item in found:
                text = found[item] if kind == "assistant_chat" else _first_words(found[item])
                names[kind][item] = form.format(text)


def _operation_names(callbacks, events, kind_of, names) -> None:
    jobs = callbacks.get("job_store")
    job_types: Dict[str, Any] = {}  # W7-4: one read per job id per request
    for event in events:
        if kind_of(event) != "operation":
            continue
        item = str(event.get("target") or "")
        job_id = item
        if str(event.get("action", "")).startswith("operation."):
            # operation.* writes an operation id; its one parser says which
            # ledger, and only the jobs ledger is a job id (W7-5).
            parsed = _safe(lambda: parse_operation_id(item), ("", ""))
            ledger, job_id = parsed if parsed is not UNREADABLE else ("", "")
            if ledger != LEDGER_JOBS:
                continue
        if jobs is None or not job_id:
            continue
        if job_id not in job_types:
            job_types[job_id] = _safe(lambda: jobs.get(job_id), None)
        job = job_types[job_id]
        if job is UNREADABLE:
            names["operation"][item] = UNREADABLE
        elif isinstance(job, dict) and job.get("type"):
            names["operation"][item] = str(job["type"])


def _key_names(callbacks, wanted, names) -> None:
    broker = callbacks.get("credential_broker")
    if broker is None or not wanted.get("key"):
        return
    records = _safe(broker.list, [])
    if records is UNREADABLE:
        for item in wanted["key"]:
            names["key"][item] = UNREADABLE
        return
    for record in records or []:
        if isinstance(record, dict) and record.get("id") in wanted["key"]:
            suffix = str(record.get("fingerprint") or "")[-4:].upper()
            names["key"][record["id"]] = "Key: {}{}".format(
                record.get("label") or "Unnamed key",
                " …{}".format(suffix) if suffix else "",
            )


def _alert_rule_names(callbacks, username, administrator, wanted, names, withheld) -> None:
    automations = callbacks.get("automations")
    for item in wanted.get("alert_rule", ()):
        if automations is None:
            continue
        rule = _safe(lambda: automations.get_trigger(
            item, None if administrator else username), None)
        if rule is UNREADABLE:
            names["alert_rule"][item] = UNREADABLE
        elif isinstance(rule, dict) and rule.get("name"):
            names["alert_rule"][item] = "Alert rule: {}".format(rule["name"])
        elif not administrator:
            # Someone else's rule, or one since deleted: an operator sees the
            # kind; only the rule's own author reads its recorded name.
            withheld["alert_rule"].add(item)


def _ai_chat_names(callbacks, username, wanted, names) -> None:
    rag = callbacks.get("rag_chat")
    for item in wanted.get("ai_chat", ()):
        if rag is None:
            continue
        # The VIEWER's scope, never the row's actor (W7-1).
        row = _read_ai_chat(rag, username, item)
        if row is not None:
            names["ai_chat"][item] = row


def _store_names(callbacks, username, administrator, wanted, names, withheld) -> None:
    """Apps, agents, automations, skills, MCP servers, machines (W7-D1).

    Each store is read once per request, and in the scope its own list route
    applies. A store that is not wired names nothing, and the row reads as
    its kind's gone phrase or word - never as the raw id.
    """
    def by_id(kind: str, read: Callable[[str], Any], field: str = "name") -> None:
        for item in wanted.get(kind, ()):
            found = _named(_safe(lambda: read(item), None), field)
            if found is not None:
                names[kind][item] = found

    inventory = callbacks.get("workload_inventory")
    if inventory is not None and wanted.get("app"):
        # W8-3: a light name read of only these ids, never ``list_all`` (which
        # inspects every container, reads every model and discovers the GPU).
        # Apps are listed to anyone who may read the trail, so no scope.
        # ``app_names`` answers None for a Docker it could not read: unread,
        # never "gone" (LESSONS 8).
        found = _safe(lambda: inventory.app_names(wanted["app"]), UNREADABLE)
        if not isinstance(found, dict):
            found = UNREADABLE
        for item in wanted["app"]:
            name = UNREADABLE if found is UNREADABLE else found.get(item)
            if name is UNREADABLE or (isinstance(name, str) and name.strip()):
                names["app"][item] = name

    agents = callbacks.get("custom_agents")
    if agents is not None:
        by_id("agent", lambda item: agents.get(item, username))
    automations = callbacks.get("automations")
    if automations is not None:
        for item in wanted.get("automation", ()):
            found = _named(_safe(lambda: automations.get(
                item, None if administrator else username), None))
            if found is not None:
                names["automation"][item] = found
            elif not administrator:
                withheld["automation"].add(item)
    skills = callbacks.get("skills_library")
    if skills is not None:
        by_id("skill", skills.get)
    catalog = callbacks.get("mcp_catalog")
    if catalog is not None:
        by_id("mcp_server", catalog.get)
    manager = callbacks.get("cluster_manager")
    store = getattr(manager, "store", None)
    if store is not None and wanted.get("node"):
        listing = _safe(lambda: store.list_nodes(limit=200), [])
        nodes = {} if listing is UNREADABLE else {
            str(node.get("id")): node for node in listing if isinstance(node, dict)
        }
        for item in wanted["node"]:
            # The name only: a machine's address belongs in no Target cell.
            found = UNREADABLE if listing is UNREADABLE else _named(nodes.get(item))
            if found is not None:
                names["node"][item] = found
    backups = callbacks.get("cluster_backups")
    if backups is not None:
        by_id("backup", lambda item: _backup(backups, item), "service_name")


def _backup(backups: Any, item: str) -> Any:
    """A cluster backup's record; None for one the store says is not there.

    ``ClusterBackupStore.get`` answers "not found" with ``ValueError`` - an
    answer, not a store that failed to read (LESSONS 8).
    """
    try:
        return backups.get(item)
    except ValueError:
        return None


def _read_ai_chat(rag: Any, username: str, item: str) -> Any:
    """"AI Chat: <title>", None when it is not there, UNREADABLE when unread.

    ``ensure_conversation`` raises its own ``RagChatError`` for a conversation
    that is not this user's or not there - an answer - and anything else for a
    store that did not answer (W7-3).
    """
    from .rag_chat import RagChatError

    try:
        row = rag.ensure_conversation(username, item)
    except RagChatError:
        return None
    except Exception:  # noqa: BLE001  # absence-ok: reported as UNREADABLE, never as gone
        return UNREADABLE
    if isinstance(row, dict) and row.get("title"):
        return "AI Chat: {}".format(row["title"])
    return None


def _view(
    event: Dict[str, Any],
    names: Dict[str, Dict[str, Any]],
    withheld: Dict[str, set],
    administrator: bool,
    username: str,
) -> Dict[str, Any]:
    action = str(event.get("action", ""))
    target = str(event.get("target") or "")
    kind = TARGET_KINDS.get(action)
    if not target or action in NO_TARGET_ACTIONS:
        return {"kind": kind or "none", "label": NO_TARGET, "known": False}
    if action in NAMED_TARGETS:
        return {"kind": "named", "label": target, "known": True}
    if action in FIXED_TARGETS:
        return {"kind": "fixed", "label": FIXED_TARGETS[action], "known": True}
    if kind is None:
        # W7-D1: an action nobody declared is not assumed readable.
        return {"kind": "unknown", "label": UNDECLARED_LABEL, "known": False}
    if kind not in RESOLVED_KINDS:
        return {"kind": kind, "label": KIND_WORDS[kind], "known": True}
    own = str(event.get("actor") or "") == username
    if _withheld(event, withheld, username):
        return {"kind": kind, "label": WITHHELD[kind], "known": True}
    found = names.get(kind, {}).get(target)
    if kind == "backup":
        # What the row recorded when it was written wins: a deleted backup's
        # record is gone, so the service recorded beside it names it (W7-D1).
        recorded = (event.get("details") or {}).get("service")
        if isinstance(recorded, str) and recorded.strip():
            found = recorded.strip()
    if found is UNREADABLE:
        return {"kind": kind, "label": NOT_READ.get(kind) or unread_phrase(kind), "known": False}
    if kind == "alert_rule" and not found and (administrator or own):
        recorded = (event.get("details") or {}).get("name")
        if isinstance(recorded, str) and recorded.strip():
            found = "Alert rule: {} (since deleted)".format(recorded.strip())
    if found is None:
        if str(event.get("result") or "") in REFUSED_RESULTS:
            # A refused request may name something that never existed.
            return {"kind": kind, "label": KIND_WORDS[kind], "known": False}
        return {"kind": kind, "label": GONE.get(kind) or gone_phrase(kind), "known": False}
    if kind == "operation":
        # The console names the type through jobLabels, its one owner.
        return {"kind": kind, "label": "Operation", "job_type": found, "known": True}
    form = LABEL_FORMS.get(kind)
    return {"kind": kind, "label": form.format(found) if form else found, "known": True}
