import { Button } from "./ui";
import { useMemo, useRef } from "react";
import { Icon } from "./Icon";
import type { AssistantEvidence } from "./agentTypes";
import { ModalShell } from "./ModalShell";
import { useMachineProfile } from "../hooks/useMachineProfile";
import { machineNoun } from "../lib/machine";

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
    change: "Start the selected managed application and verify its containers become healthy.",
    checks: ["Managed project identity", "Port availability", "Container startup", "Application health"],
  },
  "compose.stop": {
    title: "Stop this application",
    change: "Stop the selected application. Its managed configuration and persistent data remain available for restart.",
    checks: ["Managed project identity", "Dependent services", "Graceful container stop", "Persistent data retention"],
  },
  "compose.restart": {
    title: "Restart this application",
    change: "Restart the selected application and verify it returns to a healthy state.",
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
  "host.web-research.manage": {
    title: "Manage guarded web research",
    change: "Install, repair, or remove Vaelor's private public-source discovery service using the exact reviewed action.",
    checks: ["Administrator approval", "Digest-pinned service image", "Loopback-only network binding", "Human-visible readiness probe"],
  },
};

function displayValue(key: string, value: unknown) {
  if (key === "content" && typeof value === "string") {
    return `${value.length.toLocaleString()} characters of Compose YAML`;
  }
  if (sensitive.test(key)) return "Stored securely · value hidden";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value)) return value.join(", ");
  if (value && typeof value === "object") return "Structured configuration";
  return String(value ?? "Not specified");
}

export function ActionReviewDialog({
  job,
  summary,
  evidence = [],
  suggestedActions = [],
  busy,
  onCancel,
  onApprove,
}: {
  job: ProposedJob | null;
  summary: string;
  evidence?: AssistantEvidence[];
  suggestedActions?: string[];
  busy: boolean;
  onCancel: () => void;
  onApprove: () => void;
}) {
  const closeRef = useRef<HTMLButtonElement>(null);
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
    () => job ? jobCopy[job.type] ?? {
      title: job.type.replaceAll(".", " "),
      change: "Run the proposed Vaelor operation with its server-side safety policy.",
      checks: ["Authorization", "Input validation", "Live preflight checks", "Audited result"],
    } : null,
    [job],
  );


  if (!job || !copy) return null;
  const scope = Object.entries(job.payload).filter(([key]) => !["confirm"].includes(key));
  return (
    <ModalShell
      className="action-review-dialog"
      describedBy="action-review-description"
      initialFocusRef={closeRef}
      labelledBy="action-review-title"
      onClose={() => { if (!busy) onCancel(); }}
      size="standard"
    >
        <header>
          <span><Icon name="shield" size={22} /></span>
          <div><small>{copy.readOnly ? "Read-only check · no download" : "Approval required · nothing has run"}</small><h2 id="action-review-title">{copy.title}</h2></div>
          <Button aria-label="Close action review" className="icon-button" disabled={busy} onClick={onCancel} ref={closeRef} type="button">×</Button>
        </header>
        <p id="action-review-description">{copy.change}</p>
        <section>
          <h3>Why Vaelor proposed this</h3>
          <p>{summary}</p>
        </section>
        <div className="action-review-dialog__grid">
          <section>
            <h3>Preflight plan</h3>
            <ol>{copy.checks.map((check) => <li key={check}><Icon name="shield" size={14} /><span>{check}</span></li>)}</ol>
          </section>
          <section>
            <h3>Change scope</h3>
            <dl>
              <div><dt>Operation</dt><dd>{job.type.replaceAll(".", " ")}</dd></div>
              {scope.map(([key, value]) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{displayValue(key, value)}</dd></div>)}
            </dl>
          </section>
        </div>
        {evidence.length > 0 && <details><summary>Evidence used · {evidence.length} source{evidence.length === 1 ? "" : "s"}</summary><ul>{evidence.map((item) => <li key={`${item.source}-${item.summary}`}><strong>{item.source.replaceAll(".", " ")}</strong><span>{item.summary}</span></li>)}</ul></details>}
        {suggestedActions.length > 0 && <section><h3>After approval</h3><ul>{suggestedActions.map((item) => <li key={item}>{item}</li>)}</ul></section>}
        <footer>
          <p><Icon name="shield" size={14} /> The job is validated again on this {noun} and recorded in Activity.</p>
          <div>
            <Button variant="quiet" disabled={busy} onClick={onCancel} type="button">Cancel</Button>
            <Button variant="primary" disabled={busy} onClick={onApprove} type="button">{busy ? "Submitting…" : copy.approval ?? "Approve and run"}</Button>
          </div>
        </footer>
    </ModalShell>
  );
}
