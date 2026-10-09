import { useState } from "react";
import { apiRequest } from "../lib/api";
import { ActionReviewDialog } from "./ActionReviewDialog";
import { AppsDialog, AppsInset } from "./appsKit";
import { Button, Input, Textarea } from "./ui";
import type { WorkloadJob } from "./workloads-types";
import "../styles/apps-install.css";

/*
 * The guarded Docker composer (VD-200, the AppsComposeImport board): name the
 * project, paste a Compose file, review the exact action, approve. Review
 * deployment opens the shared action review; Approve and run queues the import.
 */

// Shown as the Compose editor's placeholder, not pre-filled content, so the
// caret starts a clean document instead of landing inside example text.
const COMPOSE_EXAMPLE = "services:\n  app:\n    image: nginx:stable-alpine\n    restart: unless-stopped\n    ports:\n      - \"8088:80\"\n    mem_limit: 256m\n    cpus: \"1.0\"\n";

export function composeProjectProblem(value: string) {
  const valid = /^[A-Za-z0-9][A-Za-z0-9_-]{1,47}$/.test(value);
  return value === "" || valid ? "" : (
    "Use 2-48 characters, starting with a letter or number; only letters, numbers, hyphens, and underscores are allowed."
  );
}

export function ComposeImportDialog({
  csrfToken,
  onClose,
  onQueued,
}: {
  csrfToken: string;
  onClose: () => void;
  /** The import was accepted: the page adds the job and says what happens next. */
  onQueued: (job: WorkloadJob) => void;
}) {
  const [project, setProject] = useState("");
  // Starts empty so typing begins a clean document; the example is the placeholder.
  const [compose, setCompose] = useState("");
  const [error, setError] = useState("");
  const [review, setReview] = useState(false);
  const [busy, setBusy] = useState(false);
  const projectProblem = composeProjectProblem(project);
  const projectName = project.trim().toLowerCase();

  const importCompose = async () => {
    setBusy(true);
    setError("");
    try {
      const job = await apiRequest<WorkloadJob>("/jobs", {
        method: "POST",
        body: JSON.stringify({ type: "compose.import", payload: { project: projectName, content: compose } }),
      }, csrfToken);
      onQueued(job);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The custom stack could not be queued.");
    } finally {
      setBusy(false);
    }
  };

  const reviewReason = !project
    ? "Name the project so Vaelor can manage it."
    : projectProblem
      ? projectProblem
      : !compose.trim()
        ? "Paste the Compose file you want Vaelor to validate."
        : undefined;

  return (
    <>
      <AppsDialog
        busy={busy}
        className="apps-compose"
        error={error ? <><strong>Action needed</strong> {error}</> : undefined}
        eyebrow="Guarded Docker composer"
        footer={<>
          <Button disabled={busy} onClick={onClose}>Close</Button>
          <Button busy={busy} disabledReason={reviewReason} onClick={() => setReview(true)} variant="primary">{busy ? "Preparing…" : "Review deployment"}</Button>
        </>}
        onClose={onClose}
        title="Import a custom stack"
        titleId="custom-compose-title"
      >
        <p>Published images and CPU/memory limits are required. Privileged mode, host namespaces, devices, Docker socket access, unsafe bind mounts, and inline secrets are blocked.</p>
        <div className="apps-compose__project">
          <Input
            aria-describedby={projectProblem ? "custom-project-error" : undefined}
            aria-invalid={Boolean(projectProblem)}
            label="Project name"
            maxLength={48}
            onChange={(event) => { setProject(event.target.value); setError(""); }}
            placeholder="example-stack"
            value={project}
          />
          {projectProblem && <small className="apps-field-error" id="custom-project-error" role="alert">{projectProblem}</small>}
        </div>
        <Textarea
          className="apps-compose__editor"
          label="Compose YAML"
          maxLength={65536}
          onChange={(event) => { setCompose(event.target.value); setError(""); }}
          placeholder={COMPOSE_EXAMPLE}
          rows={8}
          spellCheck={false}
          value={compose}
        />
        <AppsInset
          detail="Docker normalizes the file first; Vaelor then evaluates the actual effective configuration."
          icon="shield"
          title="Validated before anything starts"
        />
      </AppsDialog>
      <ActionReviewDialog
        busy={busy}
        evidence={[
          { source: "workloads.capabilities", summary: "Docker and Compose readiness are checked again before deployment." },
          { source: "compose.policy", summary: "Host access, inline secrets, missing resource limits, and port conflicts are blocked." },
        ]}
        job={review ? { type: "compose.import", payload: { project: projectName, content: compose } } : null}
        onApprove={() => {
          setReview(false);
          void importCompose();
        }}
        onCancel={() => setReview(false)}
        summary={`Import ${projectName} as a Vaelor-managed Docker project.`}
        suggestedActions={[
          "Watch validation and image download progress in Recent setup activity.",
          "Open Manage to inspect health, logs, configuration, and removal controls.",
        ]}
      />
    </>
  );
}
