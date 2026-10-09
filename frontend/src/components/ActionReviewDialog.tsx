import { useMemo, useRef } from "react";
import { AppsDialog, AppsFacts } from "./appsKit";
import { Button } from "./ui";
import { Icon, ICON_SIZE } from "./Icon";
import type { AssistantEvidence } from "./agentTypes";
import { useMachineProfile } from "../hooks/useMachineProfile";
import { jobLabel } from "../lib/jobPresentation";
import { machineNoun } from "../lib/machine";
import { evidenceSourceLabel } from "../lib/evidenceSourceLabels";
import { formatQuantity, formatTypedGib } from "../lib/format";
import { modelSurfaceName, runtimeModeName } from "../lib/modelModes";

export interface ProposedJob {
  type: string;
  payload: Record<string, unknown>;
}

const sensitive = /password|passwd|token|secret|api.?key|credential/i;

const jobCopy: Record<string, { title: string; change: string; checks: string[]; approval?: string; readOnly?: boolean }> = {
  "compose.install": {
    title: "Install a managed application",
    change: "Create a managed Compose project, download its container image, and start it with Vaelor safety limits.",
    checks: ["Docker and Compose availability", "Storage and memory capacity", "Port conflicts", "Container health after startup"],
  },
  "compose.validate": {
    title: "Validate a Docker stack",
    change: "Read and validate the selected Compose configuration. No container will start during validation.",
    checks: ["Compose syntax", "Unsafe host access", "CPU and memory limits", "Port conflicts"],
  },
  "compose.deploy": {
    title: "Deploy a Docker stack",
    change: "Download the declared images and start the already-validated Compose project.",
    checks: ["Validated project state", "Image availability", "Port conflicts", "Container health after startup"],
  },
  "compose.start": {
    title: "Start this application",
    change: "Start the selected managed application with its saved configuration and verify its containers become healthy.",
    checks: ["Managed project identity", "Port availability", "Container startup", "Application health"],
  },
  "compose.stop": {
    title: "Stop this application",
    change: "Stop the selected application. Its managed configuration and persistent data remain available for restart.",
    checks: ["Managed project identity", "Dependent services", "Graceful container stop", "Persistent data retention"],
  },
  "compose.restart": {
    title: "Restart this application",
    change: "Recreate any service whose saved configuration changed, restart the selected application, and verify it returns to a healthy state.",
    checks: ["Managed project identity", "Current health", "Graceful restart", "Application health after restart"],
  },
  "compose.update": {
    title: "Update this application",
    change: "Capture the running image set, acquire the configured updates, recreate the managed application, and automatically restore the previous images if health verification fails. A checkpoint is still recommended for persistent data recovery.",
    checks: ["Managed project identity", "Immutable rollback image capture", "Image acquisition", "Application health after recreation"],
    approval: "Approve update",
  },
  "compose.backup": {
    title: "Create an application checkpoint",
    change: "Capture the selected application's managed configuration and recovery data, then verify the checkpoint before listing it for restore.",
    checks: ["Managed project identity", "Available backup storage", "Checkpoint contents", "Recovery record verification"],
    approval: "Create checkpoint",
  },
  "compose.import": {
    title: "Import and deploy a Docker stack",
    change: "Validate the supplied Compose YAML, save it as a managed project, download its images, and start only after every safety check passes.",
    checks: ["Compose syntax and normalized configuration", "Unsafe host access and secret scan", "CPU and memory limits", "Port conflicts and container health"],
  },
  "model.download": {
    title: "Download a local AI model",
    change: "Download the selected model into Vaelor-managed storage and verify the resulting file.",
    checks: ["Available storage", "Model size and format", "Download source", "File verification"],
    approval: "Approve download",
  },
  "model.inspect": {
    title: "Check model compatibility",
    change: "Read public model metadata, list exact GGUF files, and compare verified file sizes with this node. No model is downloaded or started.",
    checks: ["Exact Hugging Face repository", "GGUF file metadata", "Available storage", "RAM fit with operating-system reserve"],
    approval: "Run compatibility check",
    readOnly: true,
  },
  "model.deploy": {
    title: "Start a local AI model",
    change: "Create a managed local model service and wait for its private health endpoint.",
    checks: ["RAM fit", "Model file verification", "Private port availability", "Inference health check"],
    approval: "Approve model start",
  },
  "host.docker.install": {
    title: "Install Docker and Compose",
    change: "Use this operating system’s supported package manager to install Docker Engine and Compose, then enable the service.",
    checks: ["Supported operating system", "Package manager readiness", "Service account access", "Docker service health"],
  },
  "host.memory.optimize": {
    title: "Apply a Linux memory profile",
    change: "Set the reviewed Linux swappiness value now and save the same policy for future boots.",
    checks: ["Administrator approval", "Supported profile", "Active kernel value", "Persistent sysctl configuration"],
  },
  "host.gpu-memory.apply": {
    title: "Change this controller's GPU memory pool",
    change: "Write the GPU memory pool setting on this controller and rebuild its boot image. Nothing restarts; the pool changes size the next time you restart this controller.",
    checks: ["Administrator approval", "A GPU that shares system memory", "A size inside this machine's range", "Boot image rebuilt, or the old setting put back"],
    approval: "Approve change",
  },
  "cluster.node.gpu-memory": {
    title: "Change a worker's GPU memory pool",
    change: "Write the GPU memory pool setting on this worker and rebuild its boot image. Nothing restarts; the pool changes size the next time you restart that worker.",
    checks: ["Administrator approval", "A GPU that shares system memory", "A size inside that machine's range", "Setting read back from the worker"],
    approval: "Approve change",
  },
  "host.web-research.manage": {
    title: "Manage guarded web research",
    change: "Install, repair, or remove Vaelor's private public-source discovery service using the exact reviewed action.",
    checks: ["Administrator approval", "Digest-pinned service image", "Loopback-only network binding", "Human-visible readiness probe"],
  },
};

/**
 * What each payload field is, in the owner's words (W6 retest, LESSONS 6): the
 * review printed the wire key ("profile", "inspection job id"). A field not
 * named here is shown by its key with its separators as spaces, and the
 * guard (reviewDialogWords.test.tsx) holds the fields the callers send.
 */
const SCOPE_FIELDS: Record<string, string> = {
  project: "App",
  profile: "Memory profile",
  template: "App template",
  port: "Port",
  repo: "Model repository",
  file: "Model file",
  path: "Model file",
  size_bytes: "Size",
  action: "Change",
  mode: "How it runs",
  surface: "Used by",
  endpoint: "Address",
  content: "Compose file",
  size_gib: "Pool size",
};

/**
 * Ids and typed confirmations: the reader approves what they stand for, never
 * the token itself (the AppsActionReview board). Named ones, plus any key that
 * is an id (`draft_id`, `node_id`) or an acknowledgement (`data_loss_ack`).
 */
const HIDDEN_FIELDS = new Set(["confirm", "confirmation", "id"]);

function hiddenField(key: string) {
  return HIDDEN_FIELDS.has(key) || /_id$/.test(key) || /_ack$/.test(key);
}

/**
 * FE-W7-5: what an enum value in a payload is called on the screen that sent
 * it. The review printed the wire word ("set", "install", "balanced",
 * "ai-chat"); a value not named here is shown as recorded.
 */
const ACTION_WORDS: Record<string, Record<string, string>> = {
  "host.gpu-memory.apply": { set: "Set a new pool size", revert: "Remove the pool setting" },
  "cluster.node.gpu-memory": { set: "Set a new pool size", revert: "Remove the pool setting" },
  "host.web-research.manage": { install: "Install the search service", repair: "Repair the search service", remove: "Remove the search service" },
};

/** One payload value in the owner's words and units (FE-W7-5), or null when only the generic rules apply. */
function reviewValue(jobType: string, key: string, value: unknown): string | null {
  // W7-D5: a model file reads as its catalog button reads it, and a pool size
  // the owner typed in GiB shows its GB beside it - both through lib/format.ts.
  if (key === "size_bytes" && typeof value === "number") {
    return formatQuantity(value, jobType.startsWith("model.") ? "model" : "capacity");
  }
  if (key === "size_gib" && typeof value === "number") return formatTypedGib(value);
  if (key === "port" && value === 0) return "Chosen automatically";
  if (typeof value !== "string") return null;
  if (key === "action") return ACTION_WORDS[jobType]?.[value] ?? null;
  // The mode and surface words have one owner, lib/modelModes.ts (LESSONS 6).
  if (key === "mode") return runtimeModeName(value);
  if (key === "surface") return modelSurfaceName(value);
  return null;
}

function displayValue(jobType: string, key: string, value: unknown) {
  const worded = reviewValue(jobType, key, value);
  if (worded !== null) return worded;
  if (key === "content" && typeof value === "string") {
    return `${value.length.toLocaleString()} characters of Compose YAML`;
  }
  if (sensitive.test(key)) return "Stored securely · value hidden";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value)) return value.join(", ");
  if (value && typeof value === "object") return "Structured configuration";
  return String(value ?? "Not specified");
}

interface ActionReviewDialogProps {
  job: ProposedJob | null;
  /**
   * The caller's own names for payload values ({profile: "AI low latency"}):
   * the review printed the wire value ("ai_latency") because only the caller
   * knows what it is called on its screen.
   */
  valueLabels?: Record<string, string>;
  summary: string;
  evidence?: AssistantEvidence[];
  suggestedActions?: string[];
  busy: boolean;
  /** Why the approval was refused (VD-189): shown inside this dialog, never on the inert page beneath. */
  error?: string;
  onCancel: () => void;
  onApprove: () => void;
}

/**
 * A review dialog with no proposed job renders nothing — and must *do* nothing.
 *
 * Several call sites keep this component mounted permanently with `job={null}`
 * and hand it a job only when the reader opens a review (WebResearchSetup is
 * one). The body's `useMachineProfile()` is a network discovery, and hooks run
 * before an early `return null`, so a dialog that was never opened still asked
 * the appliance what machine it is — on every mount, for a surface with nothing
 * on screen. Worse, that request outlived the mount: its answer arrived after
 * the reader had moved on, and in the test suite the straggler landed inside
 * whichever test happened to be running and was charged to that test's mock.
 *
 * Gating the whole body behind `job` here means a closed dialog costs nothing.
 */
export function ActionReviewDialog(props: ActionReviewDialogProps) {
  if (!props.job) return null;
  return <ActionReviewBody {...props} job={props.job} />;
}

function ActionReviewBody({
  job,
  summary,
  evidence = [],
  suggestedActions = [],
  busy,
  error,
  valueLabels = {},
  onCancel,
  onApprove,
}: ActionReviewDialogProps & { job: ProposedJob }) {
  // The AppsActionReview board: Cancel takes focus first, so Enter never approves by accident.
  const cancelRef = useRef<HTMLButtonElement>(null);
  /*
   * Task #76. The footer of every approval dialog — inspect, download, deploy,
   * restart — read "The job is validated again **on the Pi**". On an HP Z2
   * Mini G1a there is no Pi: the workstation validates its own job. An
   * approval dialog is the worst place in the product to be wrong about
   * *where work runs*, because that is most of what the reader is approving.
   *
   * Read here rather than threaded in from the six call sites. A prop is
   * something a caller can get wrong, and "which machine is this" is not a
   * caller's fact — it is discovery's. `machineNoun` degrades to "machine",
   * which is true on a Pi as well, so the sentence is never false while the
   * answer is still outstanding (VD-005).
   */
  const machine = useMachineProfile();
  const noun = machineNoun(machine?.machine_class ?? "generic");
  const copy = useMemo(
    () => jobCopy[job.type] ?? {
      title: jobLabel(job.type),
      change: "Run the proposed Vaelor operation with its server-side safety policy.",
      checks: ["Authorization", "Input validation", "Live preflight checks", "Audited result"],
    },
    [job],
  );

  // `node_id` is an internal id; the review names the machine in its summary.
  const scope = Object.entries(job.payload).filter(([key]) => !hiddenField(key));
  return (
    <AppsDialog
      busy={busy}
      className="action-review-dialog"
      describedBy="action-review-description"
      error={error || undefined}
      eyebrow={copy.readOnly ? "Read-only check · no download" : "Approval required · nothing has run"}
      footer={(
        <>
          <Button disabled={busy} onClick={onCancel} ref={cancelRef}>Cancel</Button>
          <Button disabled={busy} onClick={onApprove} variant="primary">{busy ? "Submitting…" : copy.approval ?? "Approve and run"}</Button>
        </>
      )}
      footerStart={(
        <span className="action-review__shield">
          <Icon aria-hidden="true" name="shield" size={ICON_SIZE.inline} />
          The job is validated again on this {noun} and recorded in Activity.
        </span>
      )}
      initialFocusRef={cancelRef}
      onClose={onCancel}
      title={copy.title}
      titleId="action-review-title"
    >
      <p className="action-review__change" id="action-review-description">{copy.change}</p>
      <section className="action-review__section">
        <h3>Why Vaelor proposed this</h3>
        <p>{summary}</p>
      </section>
      <div className="action-review__grid">
        <section className="action-review__panel">
          <h3>Preflight plan</h3>
          <ol className="action-review__checks">
            {copy.checks.map((check) => (
              <li key={check}><Icon aria-hidden="true" name="done" size={ICON_SIZE.inline} /><span>{check}</span></li>
            ))}
          </ol>
        </section>
        <section className="action-review__panel">
          <h3>Change scope</h3>
          <AppsFacts
            className="action-review__scope"
            rows={[
              { key: "operation", label: "Operation", value: jobLabel(job.type) },
              ...scope.map(([key, value]) => ({
                key,
                label: SCOPE_FIELDS[key] ?? key.replaceAll("_", " "),
                value: valueLabels[key] ?? displayValue(job.type, key, value),
              })),
            ]}
          />
        </section>
      </div>
      {evidence.length > 0 && (
        <details className="action-review__evidence">
          <summary>Evidence used · {evidence.length} source{evidence.length === 1 ? "" : "s"}</summary>
          <ul>
            {evidence.map((item) => (
              <li key={`${item.source}-${item.summary}`}>
                <strong title={item.source}>{evidenceSourceLabel(item.source)}</strong>
                <span>{item.summary}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
      {suggestedActions.length > 0 && (
        <section className="action-review__section">
          <h3>After approval</h3>
          <ul className="action-review__after">{suggestedActions.map((item) => <li key={item}>{item}</li>)}</ul>
        </section>
      )}
    </AppsDialog>
  );
}
