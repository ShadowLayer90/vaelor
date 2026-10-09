"""What every audit action's target is: a kind, a readable name, a fixed thing, or nothing.

W7-D1: ``audit_targets`` treated any action outside its kinds table as "its
target is already a readable name" and showed the target as is. For a large
share of the actions the backend writes that was false: a 64-character
container id under "Saved an app configuration", ``cred_`` ids under endpoint
key changes, ``custom_``, ``skill_``, ``mcpsrv_``, ``node_`` and hex ids, and the
slug ``deployment-copilot``. A default of "readable" is the absence of a
decision presented as one (LESSONS 6/8).

So every action is declared here, in exactly one of four tables, and an action
in none of them is ``unknown``: its target goes behind the disclosure with a
generic word, never as the visible label. ``tests/test_audit_target_kinds.py``
fails when an action in ``frontend/src/lib/auditActions.generated.json`` (the
backend's audit actions, read from its syntax tree) is in none of the tables,
so a new action cannot slip through.

* :data:`TARGET_KINDS` - the target is an id of a kind; ``audit_targets`` names
  it from that kind's store where one is wired (:data:`RESOLVED_KINDS`) and
  otherwise by the kind's word (:data:`KIND_WORDS`).
* :data:`NAMED_TARGETS` - the target the code writes is already a name an owner
  reads (an account, a model, a network link, a tool).
* :data:`FIXED_TARGETS` - the target is a fixed internal slug (``gpu-ai-chat``,
  ``local-host``) or carries an address; the row shows what it stands for.
* :data:`NO_TARGET_ACTIONS` - the action is not about one item.
"""

from __future__ import annotations

from typing import Dict, FrozenSet

#: The kind of each id-carrying target, by the action that writes it.
TARGET_KINDS: Dict[str, str] = {
    # Operations (job ids, and operation ids through their own parser).
    "job.create": "operation",
    "job.cancel": "operation",
    "job.retry": "operation",
    "operation.cancel": "operation",
    "operation.retry": "operation",
    "operation.dismiss": "operation",
    "appliance.factory_reset.approve": "operation",
    "appliance.portable_state.approve": "operation",
    "appliance.remove_vaelor.approve": "operation",
    "assistant.agent_operation.approve": "operation",
    "upgrade.create": "operation",
    # Conversations and memories.
    "assistant.chat": "assistant_chat",
    "assistant.conversation.update": "assistant_chat",
    "assistant.conversation.delete": "assistant_chat",
    "ai_chat.message": "ai_chat",
    "ai_chat.message.regenerate": "ai_chat",
    "ai_chat.conversation.fork": "ai_chat",
    "ai_chat.conversation.delete": "ai_chat",
    "ai_chat.agent.proposal": "ai_chat",
    "assistant.memory.create": "memory",
    "assistant.memory.update": "memory",
    "assistant.memory.delete": "memory",
    # Keys: stored provider keys and endpoint keys are one store and one rule.
    "credential.activate": "key",
    "credential.delete": "key",
    "credential.rotate": "key",
    "credential.store": "key",
    "credential.test": "key",
    "credential.model.select": "key",
    "ai_chat.connection.activate": "key",
    "endpoint.key.mint": "key",
    "endpoint.key.rotate": "key",
    "endpoint.key.revoke": "key",
    # Alert rules (audited under the Assistant's trigger store).
    "assistant.trigger.create": "alert_rule",
    "assistant.trigger.update": "alert_rule",
    "assistant.trigger.delete": "alert_rule",
    # Apps: the target is the container id.
    "app.config.update": "app",
    "app.console.exec": "app",
    "app.credentials.read": "app",
    "app.diagnostic": "app",
    "app.file.update": "app",
    "app.fs.delete": "app",
    "app.fs.download": "app",
    "app.fs.mkdir": "app",
    "app.fs.upload": "app",
    "app.remote_desktop.session": "app",
    # Custom agents and what is attached to them.
    "assistant.agent.create": "agent",
    "assistant.agent.update": "agent",
    "assistant.agent.delete": "agent",
    "assistant.agent.rollback": "agent",
    "assistant.connector.approve": "agent",
    "assistant.connector.execute": "agent",
    "assistant.connector.revoke": "agent",
    "assistant.connector.test": "agent",
    "assistant.automation.create": "automation",
    "assistant.automation.update": "automation",
    "assistant.automation.delete": "automation",
    "skills.library.register": "skill",
    "skills.library.update": "skill",
    "skills.library.remove": "skill",
    "mcp.catalog.health": "mcp_server",
    "mcp.catalog.register": "mcp_server",
    "mcp.catalog.update": "mcp_server",
    "mcp.catalog.remove": "mcp_server",
    # Machines and their backups.
    "cluster.node.enroll": "node",
    "cluster.node.remove": "node",
    "cluster.node.architecture-eviction": "node",
    "cluster.telemetry.install": "node",
    "cluster.telemetry.reconcile": "node",
    "cluster.telemetry.remove": "node",
    "cluster.backup.delete": "backup",
    # Kinds named by their word alone: nothing an owner reads names them.
    "admin.agent_api_token.create": "api_token",
    "admin.agent_api_token.delete": "api_token",
    "admin.agent_api_token.revoke": "api_token",
    "admin.session.revoke": "session",
    "appliance.backup.restore_stage": "plan",
    "appliance.factory_reset.stage": "plan",
    "appliance.portable_state.stage": "plan",
    "appliance.remove_vaelor.stage": "plan",
    "application.draft.approve": "app_draft",
    "application.draft.configure": "app_draft",
    "application.draft.create": "app_draft",
    "application.draft.discard": "app_draft",
    "application.draft.research": "app_draft",
    "application.draft.research.capable": "app_draft",
    "application.draft.validate": "app_draft",
    "ai_chat.collection.create": "knowledge_collection",
    "ai_chat.collection.delete": "knowledge_collection",
    "ai_chat.document.delete": "knowledge_document",
    "ai_chat.document.ingest": "knowledge_document",
    "assistant.agent_write.approve": "knowledge_document",
    "assistant.alert_channel.create": "alert_channel",
    "assistant.alert_channel.delete": "alert_channel",
    "assistant.alert_channel.test": "alert_channel",
    "assistant.alert_channel.update": "alert_channel",
    "assistant.delegate": "assistant_task",
    "assistant.task.approval": "assistant_task",
    "assistant.task.binding.replace": "assistant_task",
    "assistant.task.comment": "assistant_task",
    "assistant.task.create": "assistant_task",
    "assistant.task.handoff": "assistant_task",
    "assistant.task.retry": "assistant_task",
    "assistant.task.transition": "assistant_task",
    "assistant.skill.delete": "assistant_skill",
    "assistant.skill.propose": "assistant_skill",
    "assistant.skill.review": "assistant_skill",
    "assistant.skill.update": "assistant_skill",
    "checkpoint.delete": "checkpoint",
    "checkpoint.restore.request": "checkpoint",
    "checkpoint.verify": "checkpoint",
    "integrations.connection.create": "integration_connection",
    "integrations.connection.revoke": "integration_connection",
    "integrations.connection.test": "integration_connection",
    "integrations.grant.create": "integration_grant",
    "integrations.grant.revoke": "integration_grant",
    "mcp.grant.attach": "mcp_grant",
    "mcp.grant.revoke": "mcp_grant",
    "skills.attach": "skill_attachment",
    "skills.detach": "skill_attachment",
    "workload_act.agent.grant": "agent_permission",
    "workload_act.agent.revoke": "agent_permission",
}

#: The word for each kind: the label when the item is not named (withheld, a
#: refused request, or a kind no store names) and the noun of its gone phrase.
KIND_WORDS: Dict[str, str] = {
    "operation": "Operation",
    "assistant_chat": "Chat",
    "ai_chat": "AI Chat conversation",
    "memory": "Memory",
    "key": "Key",
    "alert_rule": "Alert rule",
    "app": "App",
    "agent": "Agent",
    "automation": "Automation",
    "skill": "Skill",
    "mcp_server": "MCP server",
    "node": "Machine",
    "backup": "Cluster backup",
    "api_token": "Agent API token",
    "session": "Signed-in session",
    "plan": "Staged recovery plan",
    "app_draft": "App draft",
    "knowledge_collection": "Knowledge collection",
    "knowledge_document": "Knowledge document",
    "alert_channel": "Alert channel",
    "assistant_task": "Assistant task",
    "assistant_skill": "Assistant skill",
    "checkpoint": "Restore point",
    "integration_connection": "Connected integration",
    "integration_grant": "Integration grant",
    "mcp_grant": "MCP server grant",
    "skill_attachment": "Skill attachment",
    "agent_permission": "Agent workload permission",
}

#: The kinds ``audit_targets`` names from a store; every other kind shows its word.
RESOLVED_KINDS: FrozenSet[str] = frozenset({
    "operation", "assistant_chat", "ai_chat", "memory", "key", "alert_rule",
    "app", "agent", "automation", "skill", "mcp_server", "node", "backup",
})

#: The target the code writes is already a name an owner reads.
NAMED_TARGETS: FrozenSet[str] = frozenset({
    "admin.user.create",
    "admin.user.delete",
    "admin.user.update",
    "auth.bootstrap",
    "auth.totp.disable",
    "auth.totp.enable",
    "auth.totp.setup",
    "workload_act.operator.grant",
    "workload_act.operator.revoke",
    "assistant.model.calibrate",
    "assistant.tool.run",
    "cluster.link.set",
    "cluster.retained_volume.delete",
    "cluster.service.diagnostic",
    "cluster.serving.rotate-key",
    "device.model.choose",
    "mcp.external.run",
    "mcp.request",
    "model.install_release",
    "skills.library.import_github",
})

#: Each fixed thing written once, whichever actions stand for it.
_REMOTE_LOGIN = "Remote login"
_BROWSER_DESKTOP = "Browser desktop"
_KVM = "KVM keyboard and mouse"
_LLM_SERVER = "The LLM Server endpoint"
_PHOENIX = "Trace collector (Phoenix)"
_APPLIANCE = "The appliance itself"
_WEB_RESEARCH = "Web research"

#: The target is a fixed internal slug, or carries a machine's address; the row
#: says what it stands for. An address belongs in the From column only.
FIXED_TARGETS: Dict[str, str] = {
    "agent.plan": "Deployment plan",
    "assistant.preference.update": "Assistant intelligence setting",
    "cluster.performance.profile": "GPU serving profile",
    "cluster.ssh.inspect": "A machine's SSH sign-in",
    "fan.case.update": "Case fan",
    "fan.cpu.update": "CPU fan",
    "host.rdp.disable": _REMOTE_LOGIN,
    "host.rdp.enable": _REMOTE_LOGIN,
    "host.remote_desktop.session": _BROWSER_DESKTOP,
    "host.remote_desktop.session_end": _BROWSER_DESKTOP,
    "kvm.control.acquire": _KVM,
    "kvm.control.release": _KVM,
    "lighting.update": "Case LED lighting",
    "llm_server.disable": _LLM_SERVER,
    "llm_server.enable": _LLM_SERVER,
    "llm_server.rotate_key": _LLM_SERVER,
    "phoenix.disable": _PHOENIX,
    "phoenix.enable": _PHOENIX,
    "power.reboot": _APPLIANCE,
    "power.restart_service": _APPLIANCE,
    "power.shutdown": _APPLIANCE,
    "system.display.update": "Front display",
    "system.network.test": "Internet connectivity",
    "web-research.plan": _WEB_RESEARCH,
    "web-research.queue": _WEB_RESEARCH,
}

#: The action is not about one item.
NO_TARGET_ACTIONS: FrozenSet[str] = frozenset({
    "appliance.backup.offsite",
    "appliance.backup.offsite_push",
    "appliance.backup.passphrase",
    "appliance.backup.run",
    "appliance.backup.schedule",
    "appliance.factory_reset.cancel",
    "appliance.portable_state.cancel",
    "appliance.portable_state.export",
    "appliance.remove_vaelor.cancel",
    "assistant.tasks.archive_finished",
    "auth.login",
    "auth.logout",
    "auth.mfa",
})

#: The label of a row whose action no table declares (W7-D1): the raw id is
#: behind the disclosure, never the visible label.
UNDECLARED_LABEL = "An item this row does not name"

#: Results that mean the request was refused or failed; an unnamed target on
#: such a row may never have existed, so it gets its kind's word, not "gone".
REFUSED_RESULTS: FrozenSet[str] = frozenset({"failure", "rejected", "refused", "denied", "noop"})


def gone_phrase(kind: str) -> str:
    """"a <word> that no longer exists", lower-cased, with its article."""
    word = KIND_WORDS[kind]
    acronym = word[:2].isupper()
    noun = word if acronym else word[0].lower() + word[1:]
    # An acronym is read letter by letter: "an MCP server", "an AI Chat".
    article = "an" if noun[0].lower() in ("aefhilmnorsx" if acronym else "aeiou") else "a"
    return "{} {} that no longer exists".format(article, noun)


def unread_phrase(kind: str) -> str:
    """"a <word> that could not be read just now" (W7-3, LESSONS 8)."""
    return gone_phrase(kind).replace("that no longer exists", "that could not be read just now")
