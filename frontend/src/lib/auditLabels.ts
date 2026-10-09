/**
 * Readable names for audit-event slugs.
 *
 * The activity tables used to render `event.action.replaceAll(".", " ")` under
 * a `text-transform: capitalize` rule, so `ai_chat.message` reached the reader
 * as "Ai Chat Message" and `host.rdp.enable` as "Host Rdp Enable". Those are
 * transliterations of an internal identifier, not names of anything a person
 * did. Every event now resolves to a phrase written for a reader.
 *
 * Resolution order:
 *   1. an exact phrase for the events an operator actually sees;
 *   2. a composed "<verb> <subject>" phrase built from the slug's own parts;
 *   3. a sentence-cased fallback that at least expands the known abbreviations.
 */

const exactLabels: Record<string, string> = {
  "auth.login": "Signed in",
  "auth.logout": "Signed out",
  "auth.bootstrap": "Created the first administrator",
  // Neutral (review round 2): past rows were real removals and new ones are the
  // route's refusal (VD-194, owner 2026-10-05); the Result column says which.
  "cluster.telemetry.remove": "Remove worker telemetry",
  // W8 review: the writer records this only for a wrong code (api_auth_routes.py).
  "auth.mfa": "Refused a two-factor code",
  "auth.totp.setup": "Started two-factor setup",
  "auth.totp.enable": "Enabled two-factor authentication",
  "auth.totp.disable": "Disabled two-factor authentication",
  "totp.key": "Read a two-factor secret",

  "admin.user.create": "Created a user account",
  "admin.user.update": "Updated a user account",
  "admin.user.delete": "Deleted a user account",
  "admin.session.revoke": "Signed another session out",
  "admin.agent_api_token.create": "Issued an agent API token",
  "admin.agent_api_token.revoke": "Revoked an agent API token",

  "power.reboot": "Rebooted this appliance",
  "power.shutdown": "Shut this appliance down",
  // #208: this restarts `vaelor-control-plane.service`, not the hardware
  // bridge. "hardware service" misnamed it and read as a second, broader thing
  // — the same wording Overview.tsx deliberately avoids ("Restart service").
  "power.restart_service": "Restarted the Vaelor control plane",

  "job.create": "Approved an operation",
  "job.cancel": "Cancelled an operation",
  "job.retry": "Retried an operation",
  "jobs.recent": "Reviewed recent operations",
  "operation.cancel": "Cancelled an operation",
  "operation.retry": "Retried an operation",

  "ai_chat.message": "Sent an AI Chat message",
  "ai_chat.message.regenerate": "Regenerated an AI Chat answer",
  "ai_chat.conversation.fork": "Branched an AI Chat conversation",
  "ai_chat.conversation.delete": "Deleted an AI Chat conversation",
  "ai_chat.collection.create": "Created a knowledge collection",
  "ai_chat.collection.delete": "Deleted a knowledge collection",
  "ai_chat.document.ingest": "Added a knowledge document",
  "ai_chat.document.delete": "Removed a knowledge document",
  "ai_chat.connection.activate": "Switched the AI Chat connection",
  "ai_chat.agent.proposal": "Received an AI Chat agent proposal",
  "knowledge.document": "Opened a knowledge document",

  "assistant.chat": "Asked the Assistant",
  "assistant.delegate": "Started an appliance check",
  "assistant.tool.run": "Ran an Assistant tool",
  "assistant.agent_write.approve": "Approved an Assistant change",
  // W6-D1: these rows are written by the alert-rule routes; "trigger" is the
  // store's word, "alert rule" is the owner's, on every screen that shows one.
  "assistant.trigger.create": "Created an alert rule",
  "assistant.trigger.update": "Turned an alert rule on or off",
  "assistant.trigger.delete": "Deleted an alert rule",
  "assistant.tasks.archive_finished": "Archived finished appliance checks",
  "agent.plan": "Prepared an Assistant plan",

  "device.model.choose": "Chose the enclosure model",
  "fan.cpu.update": "Changed the CPU fan policy",
  "fan.case.update": "Changed the case fan policy",
  "lighting.update": "Changed the case lighting",
  "system.display.update": "Changed the display settings",
  "system.network.test": "Tested internet connectivity",
  "system.update.apply": "Applied operating-system updates",
  "host.docker.install": "Installed Docker and Compose",
  "host.memory.optimize": "Applied a memory profile",
  "host.rdp.enable": "Enabled Remote Desktop",
  "host.rdp.disable": "Disabled Remote Desktop",
  "host.vnc.enable": "Enabled VNC access",
  "host.remote_desktop.session": "Opened a Remote Desktop session",
  "app.remote_desktop.session": "Opened an app desktop session",
  "app.config.update": "Saved an app configuration",
  "app.diagnostic": "Ran an app diagnostic",

  "compose.draft": "Drafted a Docker stack",
  "compose.import": "Imported a Docker stack",
  "compose.remove": "Removed a Docker stack",
  "managed.remove": "Removed a managed workload",
  "model.deploy": "Started a local AI model",
  "model.remove": "Removed a local AI model",
  "application.research": "Researched an application",
  // W6-3: three the regex scan could not see (a hyphenated first segment, and
  // an action passed as a module constant). Written on every worker Recheck.
  "web-research.plan": "Reviewed a web research service change",
  "web-research.queue": "Queued a web research service change",
  // W8 review: one row per check that changed something - a repair, a note, or a
  // failure. The words say what was done; the Result column says how it went.
  "cluster.telemetry.reconcile": "Checked a worker's telemetry",
  // W6-3: written through a local variable, which the regex scan could not see.
  "application.draft.research": "Researched an application draft",
  "application.draft.research.capable": "Researched an application draft with the larger model",

  "checkpoint.verify": "Verified a recovery checkpoint",
  "checkpoint.restore": "Restored a recovery checkpoint",
  "checkpoint.restore.request": "Requested a checkpoint restore",
  "checkpoint.delete": "Deleted a recovery checkpoint",
  "recovery.checkpoints": "Reviewed recovery checkpoints",

  "kvm.control.acquire": "Took keyboard and mouse control",
  "kvm.control.release": "Released keyboard and mouse control",

  "cluster.initialize": "Initialized the cluster",
  "cluster.node.enroll": "Enrolled a worker",
  "cluster.node.join": "Joined a worker to the cluster",
  "cluster.node.remove": "Removed a worker",
  "cluster.link.set": "Changed the cluster link",
  "cluster.node.availability": "Changed worker availability",
  "cluster.node.architecture-eviction":
    "Drained and removed a worker of the wrong processor architecture",
  "cluster.ssh.inspect": "Inspected a worker over SSH",
  // W5-D7: the actions the backend writes that no verb above composed; held
  // against vaelor/ by auditLabelsCoverage.test.ts.
  "cluster.serving.rotate-key": "Rotated the cluster serving key",
  "cluster.performance.profile": "Captured a serving performance profile",
  "llm_server.rotate_key": "Rotated an LLM Server key",
  "skills.library.register": "Registered a skill",
  "skills.library.import_github": "Imported a skill from GitHub",
  "skills.attach": "Attached a skill to an agent",
  "skills.detach": "Detached a skill from an agent",
  "mcp.catalog.register": "Registered an MCP server",
  "mcp.catalog.health": "Checked an MCP server",
  "mcp.grant.attach": "Gave an agent an MCP server",
  "mcp.request": "Answered an MCP request",
  "app.console.exec": "Ran a command in an app",
  "app.credentials.read": "Read an app's credentials",
  "app.fs.download": "Downloaded an app file",
  "app.fs.mkdir": "Created an app folder",
  "app.fs.upload": "Uploaded an app file",
  "appliance.backup.offsite": "Changed the off-site backup",
  "appliance.backup.offsite_push": "Sent a backup off-site",
  "appliance.backup.passphrase": "Set the backup passphrase",
  "appliance.backup.restore_stage": "Prepared a backup restore",
  "appliance.backup.schedule": "Changed the backup schedule",
  "assistant.model.calibrate": "Calibrated the Assistant model",
  "assistant.task.binding.replace": "Changed the model an appliance check runs on",
  "credential.deactivate": "Deactivated a connection",
  "credential.endpoint-keys": "Read an endpoint's keys",
  "credential.lease": "Lent a connection to a service",
  "credential.list": "Listed connections",
  "host.remote_desktop.session_end": "Ended a Remote Desktop session",
  "model.install_release": "Installed a released model",
  "operation.dismiss": "Dismissed an operation",
  "workload_act.agent.grant": "Let an agent act on a workload",
  "workload_act.operator.grant": "Let an operator act on a workload",
  "appliance.portable_state.export": "Exported the portable state",
  "appliance.portable_state.stage": "Prepared a portable state import",
  "appliance.portable_state.approve": "Approved a portable state import",
  "appliance.portable_state.cancel": "Cancelled a portable state import",
  "upgrade.create": "Started a Vaelor update",
  "llm_server.apply": "Applied LLM Server settings",
};

/** Trailing slug segments that describe what the operator did. */
const verbs: Record<string, string> = {
  activate: "Activated",
  apply: "Applied",
  approval: "Reviewed an approval for",
  approve: "Approved",
  archive: "Archived",
  availability: "Changed the availability of",
  backup: "Backed up",
  cancel: "Cancelled",
  comment: "Commented on",
  configure: "Configured",
  create: "Created",
  delegate: "Delegated",
  delete: "Deleted",
  deploy: "Deployed",
  diagnostic: "Ran a diagnostic on",
  disable: "Disabled",
  discard: "Discarded",
  enable: "Enabled",
  enroll: "Enrolled",
  execute: "Ran",
  export: "Exported",
  fork: "Branched",
  handoff: "Handed off",
  import: "Imported",
  ingest: "Added",
  initialize: "Initialized",
  inspect: "Inspected",
  install: "Installed",
  join: "Joined",
  load: "Loaded",
  manage: "Managed",
  mint: "Issued",
  optimize: "Tuned",
  propose: "Proposed",
  pull: "Downloaded",
  regenerate: "Regenerated",
  remove: "Removed",
  restore: "Restored",
  retry: "Retried",
  review: "Reviewed",
  revoke: "Revoked",
  rollback: "Rolled back",
  rotate: "Rotated",
  run: "Ran",
  select: "Selected",
  session: "Opened a session for",
  stage: "Prepared",
  store: "Stored",
  test: "Tested",
  transition: "Moved",
  unload: "Unloaded",
  update: "Updated",
  validate: "Validated",
  verify: "Verified",
};

/**
 * W8-D1: the owner's noun for each whole subject a verb rule may name. A verb
 * rule counts as named only through this table: "skills.library.remove" read
 * "Removed skills library" because the slug's own words were joined and the
 * coverage test called that named (LESSONS 2/5).
 */
const subjects: Record<string, string> = {
  "admin.agent_api_token": "an agent API token",
  "app.file": "an app's configuration file",
  "app.fs": "an app file",
  "appliance.backup": "a backup of this appliance",
  "appliance.factory_reset": "a factory reset",
  "appliance.remove_vaelor": "the removal of Vaelor",
  "application.draft": "an application draft",
  "assistant.agent": "an Assistant custom agent",
  "assistant.agent_operation": "an agent's proposed operation",
  "assistant.alert_channel": "an alert channel",
  "assistant.automation": "an automation",
  "assistant.connector": "an Assistant connector",
  "assistant.conversation": "an Assistant chat",
  "assistant.memory": "an Assistant memory",
  "assistant.preference": "an Assistant preference",
  "assistant.skill": "an Assistant skill",
  "assistant.task": "an Assistant task",
  "cluster.backup": "a cluster backup",
  "cluster.retained_volume": "a kept cluster app volume",
  "cluster.service": "a cluster service",
  "cluster.telemetry": "worker telemetry",
  credential: "a connection",
  "credential.model": "a connection's model",
  "endpoint.key": "an endpoint key",
  "integrations.connection": "an integration connection",
  "integrations.grant": "an integration grant",
  llm_server: "the LLM Server",
  "mcp.catalog": "an MCP server",
  "mcp.external": "an external MCP tool",
  "mcp.grant": "an agent's MCP server access",
  phoenix: "request tracing (Phoenix)",
  "skills.library": "a skill",
  "workload_act.agent": "an agent's workload access",
  "workload_act.operator": "an operator's workload access",
};

/** Slug words whose plain-English name is not the word itself. */
const subjectWords: Record<string, string> = {
  admin: "administration",
  agent_api_token: "agent API token",
  ai: "AI",
  ai_chat: "AI Chat",
  app: "app",
  factory_reset: "factory reset",
  gpu: "GPU model",
  host: "appliance",
  kvm: "KVM",
  llm: "AI service",
  llm_server: "LLM Server",
  mfa: "two-factor sign-in",
  portable_state: "portable state",
  rdp: "Remote Desktop",
  remote_desktop: "Remote Desktop",
  ssh: "SSH",
  totp: "two-factor authentication",
  vnc: "VNC",
};

function subjectPhrase(parts: string[]): string {
  return parts
    .map((part) => subjectWords[part] ?? part.replaceAll("_", " "))
    .join(" ")
    .trim();
}

function sentenceCase(value: string): string {
  const text = value.trim();
  if (!text) return "";
  return text[0].toUpperCase() + text.slice(1);
}

/**
 * Results that mean the action happened as its label says (W8 review). Any
 * other result - `failure`, `attempt`, a reconcile's own words - is shown
 * through `attemptLabels`, so a failed row never reads in the past tense of
 * success ("Signed in", "Repaired..."). LESSONS 5.
 */
export const DONE_RESULTS: ReadonlySet<string> = new Set(["success", "ok", "completed"]);

/**
 * Results that mean the action was STARTED, not finished (ffb8234 review F2):
 * a reboot "accepted" before the host goes down, research "queued" for later.
 * They are not failures, and they are not done either, so a row carrying one
 * reads through `startedLabels` ("Started a reboot"), never as "Rebooted".
 */
export const STARTED_RESULTS: ReadonlySet<string> = new Set(["accepted", "queued", "sent"]);

/** Results that are not a failure: done, or started. */
export const SUCCESS_RESULTS: ReadonlySet<string> = new Set([...DONE_RESULTS, ...STARTED_RESULTS]);

/** Actions whose words assert no outcome, so they need no attempt form. */
export const OUTCOME_NEUTRAL_ACTIONS: ReadonlySet<string> = new Set([
  "cluster.telemetry.reconcile",
  // An imperative, not a past tense: true of the old successful removals and
  // of today's refusals alike, with the Result column saying which.
  "cluster.telemetry.remove",
  "mcp.catalog.health",
]);

/**
 * The words for an action whose row records that it was started, not finished
 * (ffb8234 review F2). Every action whose writer records a started result
 * needs one; auditLabelsCoverage.test.ts holds that against the writers.
 */
const startedLabels: Record<string, string> = {
  "application.draft.research": "Queued research on an application draft",
  "application.draft.research.capable": "Queued research on an application draft with the larger model",
  "model.install_release": "Started installing a released model",
  "power.reboot": "Started a reboot of this appliance",
  "power.restart_service": "Started a restart of the Vaelor control plane",
  "power.shutdown": "Started shutting this appliance down",
  "web-research.queue": "Queued a web research service change",
};

/** Whether `action` has started-wording for a started result. */
export function auditActionHasStartedLabel(action: string): boolean {
  return Boolean(startedLabels[(action ?? "").trim()]);
}

/**
 * The words for an action whose row did not succeed. Every action whose writer
 * records a non-success outcome needs one (auditLabelsCoverage.test.ts reads
 * those outcomes from tools/audit_actions.py).
 */
const attemptLabels: Record<string, string> = {
  "ai_chat.message": "Tried to send an AI Chat message",
  "app.console.exec": "Tried to run a command in an app",
  "app.fs.delete": "Tried to delete an app file",
  "app.fs.mkdir": "Tried to create an app folder",
  "app.fs.upload": "Tried to upload an app file",
  "appliance.backup.offsite_push": "Tried to send a backup off-site",
  "appliance.backup.run": "Tried to back up this appliance",
  "application.draft.approve": "Tried to approve an application draft",
  "application.draft.configure": "Tried to configure an application draft",
  "application.draft.create": "Tried to create an application draft",
  "application.draft.discard": "Tried to discard an application draft",
  "application.draft.research": "Tried to research an application draft",
  "application.draft.research.capable": "Tried to research an application draft with the larger model",
  "application.draft.validate": "Tried to validate an application draft",
  "assistant.alert_channel.test": "Tried to test an alert channel",
  "assistant.chat": "Tried to ask the Assistant",
  "assistant.memory.create": "Tried to create an Assistant memory",
  "auth.login": "Refused a sign-in",
  "auth.mfa": "Refused a two-factor code",
  "cluster.link.set": "Tried to change the cluster link",
  "cluster.node.architecture-eviction": "Tried to remove a worker of the wrong processor architecture",
  "cluster.node.enroll": "Tried to enrol a worker",
  "cluster.performance.profile": "Tried to capture a serving performance profile",
  "cluster.telemetry.install": "Tried to install worker telemetry",
  "credential.store": "Tried to store a connection",
  "credential.test": "Tried to test a connection",
  "endpoint.key.mint": "Tried to issue an endpoint key",
  "endpoint.key.revoke": "Tried to revoke an endpoint key",
  "endpoint.key.rotate": "Tried to rotate an endpoint key",
  "host.rdp.enable": "Tried to enable Remote Desktop",
  "integrations.connection.test": "Tried to test an integration connection",
  "operation.cancel": "Tried to cancel an operation",
  "operation.dismiss": "Tried to dismiss an operation",
  "operation.retry": "Tried to retry an operation",
  "power.reboot": "Tried to reboot this appliance",
  "power.restart_service": "Tried to restart the Vaelor control plane",
  "power.shutdown": "Tried to shut this appliance down",
  "system.network.test": "Tried to test internet connectivity",
  "upgrade.create": "Tried to start a Vaelor update",
  "assistant.agent_operation.approve": "Tried to approve an agent's proposed operation",
  "assistant.tool.run": "Tried to run an Assistant tool",
  "llm_server.enable": "Tried to enable the LLM Server",
  "mcp.catalog.register": "Tried to register an MCP server",
  "mcp.catalog.remove": "Tried to remove an MCP server",
  "mcp.catalog.update": "Tried to update an MCP server",
  "mcp.external.run": "Tried to run an external MCP tool",
  "mcp.grant.attach": "Tried to give an agent an MCP server",
  "mcp.grant.revoke": "Tried to revoke an agent's MCP server access",
  "skills.attach": "Tried to attach a skill to an agent",
  "skills.detach": "Tried to detach a skill from an agent",
  "skills.library.import_github": "Tried to import a skill from GitHub",
  "skills.library.register": "Tried to register a skill",
  "skills.library.remove": "Tried to remove a skill",
  "skills.library.update": "Tried to update a skill",
  "workload_act.agent.revoke": "Found no workload access to revoke from an agent",
  "workload_act.operator.revoke": "Found no workload access to revoke from an operator",
};

/**
 * Whether `action` has a written phrase, rather than reaching the fallback
 * (W5-D7): an exact label, or a known verb over a subject declared in
 * `subjects`. A subject spelled from the slug's own words does not count
 * (W8-D1, LESSONS 2/5).
 */
export function auditLabelIsNamed(action: string): boolean {
  const slug = (action ?? "").trim();
  if (exactLabels[slug]) return true;
  const parts = slug.split(".").filter(Boolean);
  return parts.length > 1
    && Boolean(verbs[parts[parts.length - 1]])
    && Boolean(subjects[parts.slice(0, -1).join(".")]);
}

/**
 * A readable label for one audit-event action slug. Unknown slugs still get a
 * useful phrase rather than a title-cased identifier, so a new server event
 * never regresses the activity tables to machine text.
 */
export function auditActionLabel(action: string, result?: string): string {
  const slug = (action ?? "").trim();
  if (!slug) return "Unrecorded event";
  if (result !== undefined && STARTED_RESULTS.has(result) && startedLabels[slug]) {
    return startedLabels[slug];
  }
  if (result !== undefined && !SUCCESS_RESULTS.has(result) && attemptLabels[slug]) {
    return attemptLabels[slug];
  }
  const exact = exactLabels[slug];
  if (exact) return exact;

  const parts = slug.split(".").filter(Boolean);
  const verb = parts.length > 1 ? verbs[parts[parts.length - 1]] : undefined;
  if (verb) {
    const subject = subjects[parts.slice(0, -1).join(".")] ?? subjectPhrase(parts.slice(0, -1));
    return subject ? `${verb} ${subject}` : verb;
  }
  return sentenceCase(subjectPhrase(parts));
}

/** How an outcome chip is coloured, by the states sheet: done is green, failed red, refused amber, the rest grey or blue. */
export type AuditResultTone = "success" | "failure" | "warning" | "info" | "neutral";

/*
 * The outcome words, one per result a writer records (tools/audit_actions.py
 * lists the literal ones; the backup and telemetry-reconcile writers record a
 * computed value, from backup_schedule.py and cluster_worker_telemetry.py).
 */
const RESULT_WORDS: Record<string, { label: string; tone: AuditResultTone }> = {
  // Started, not finished: a reboot accepted before the host goes down,
  // research queued for later, a message handed on.
  accepted: { label: "Started", tone: "info" },
  queued: { label: "Queued", tone: "info" },
  sent: { label: "Sent", tone: "info" },
  failure: { label: "Failed", tone: "failure" },
  error: { label: "Failed", tone: "failure" },
  failed: { label: "Failed", tone: "failure" },
  // Vaelor declined the request (a review gate, a policy, an LLM Server with
  // no key): not a fault, and not done.
  rejected: { label: "Refused", tone: "warning" },
  refused: { label: "Refused", tone: "warning" },
  denied: { label: "Refused", tone: "warning" },
  // App console and file actions record that they were attempted, not how
  // they ended.
  attempt: { label: "Attempted", tone: "neutral" },
  noop: { label: "Nothing to change", tone: "neutral" },
  // An off-site copy with no off-site destination configured.
  skipped: { label: "Skipped", tone: "neutral" },
  // Telemetry reconcile: what Vaelor did to the worker, in the words of its
  // sentence in RECONCILE_CHANGES. It records the action, not that the fault
  // is gone (the next pass can record drift-unresolved), so never green.
  restarted: { label: "Restarted", tone: "info" },
  "restarted-backlog": { label: "Restarted", tone: "info" },
  "restarted-sampler": { label: "Restarted", tone: "info" },
  "reshipped-ca": { label: "Certificate re-sent", tone: "info" },
  "reinstalled-absent": { label: "Reinstalled", tone: "info" },
  "reprovisioned-stale": { label: "Reinstalled", tone: "info" },
  "reprovisioned-drift": { label: "Reinstalled", tone: "info" },
  // ...and these are a repair that did not take.
  "drift-unresolved": { label: "Not fixed", tone: "warning" },
  "drift-stopped": { label: "Stopped retrying", tone: "warning" },
  "sampler-restarts-stopped": { label: "Stopped retrying", tone: "warning" },
};

/**
 * The operator's word for an audit event's outcome, and its tone. The tables
 * once printed the result slug itself, which uppercase styling disguised and
 * the redesign's sentence-case chips exposed (LESSONS 5). Writers record many
 * results besides "success" and "failure" (see RESULT_WORDS); a value none of
 * them records says so rather than echoing the slug.
 */
export function auditResultView(result: string): { label: string; tone: AuditResultTone } {
  if (DONE_RESULTS.has(result)) return { label: "Succeeded", tone: "success" };
  return RESULT_WORDS[result] ?? { label: "Outcome not recognised", tone: "neutral" };
}
