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
  "auth.mfa": "Completed two-factor sign-in",
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
  "cluster.node.availability": "Changed worker availability",
  "cluster.node.architecture-eviction":
    "Drained and removed a worker of the wrong processor architecture",
  "cluster.ssh.inspect": "Inspected a worker over SSH",
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
  manage: "Managed",
  optimize: "Tuned",
  propose: "Proposed",
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
  update: "Updated",
  validate: "Validated",
  verify: "Verified",
};

/** Slug words whose plain-English name is not the word itself. */
const subjectWords: Record<string, string> = {
  admin: "administration",
  agent_api_token: "agent API token",
  ai: "AI",
  ai_chat: "AI Chat",
  app: "app",
  factory_reset: "factory reset",
  host: "appliance",
  kvm: "KVM",
  llm: "AI service",
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
 * A readable label for one audit-event action slug. Unknown slugs still get a
 * useful phrase rather than a title-cased identifier, so a new server event
 * never regresses the activity tables to machine text.
 */
export function auditActionLabel(action: string): string {
  const slug = (action ?? "").trim();
  if (!slug) return "Unrecorded event";
  const exact = exactLabels[slug];
  if (exact) return exact;

  const parts = slug.split(".").filter(Boolean);
  const verb = parts.length > 1 ? verbs[parts[parts.length - 1]] : undefined;
  if (verb) {
    const subject = subjectPhrase(parts.slice(0, -1));
    return subject ? `${verb} ${subject}` : verb;
  }
  return sentenceCase(subjectPhrase(parts));
}
