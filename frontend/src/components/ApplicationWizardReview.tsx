import { useRef, type ReactNode } from "react";
import { AppsDialog, AppsFacts, AppsProgress } from "./appsKit";
import { type ComposeDraft, type DeploymentProgress, shortDigest } from "./applicationDeploymentModel";
import { StepHeading } from "./ApplicationWizardRequest";
import { Icon, ICON_SIZE } from "./Icon";
import { StatusPill } from "./StatusPill";
import { Button } from "./ui";
import type { StatusTone } from "./ui/status";

/*
 * Steps 4 and 5 of the custom-application wizard and its final approval (the
 * AppsWizardDeploy board): the redacted Compose with its validation, the
 * "Approve this exact draft?" dialog over the wizard, and how the deploy ends.
 */

export const VALIDATION_ERRORS_REASON = "Resolve validation errors before approval.";

const VALIDATION_TONES: Record<ComposeDraft["validation"][number]["level"], StatusTone> = {
  pass: "success",
  warning: "warning",
  error: "danger",
};

export function ReviewStep({ draft }: { draft: ComposeDraft }) {
  return (
    <section aria-labelledby="application-review-heading" className="apps-wizard__section">
      <StepHeading
        id="application-review-heading"
        pill={<code className="apps-wizard__digest" title={draft.manifestDigest}>{shortDigest(draft.manifestDigest)}</code>}
        step="04 / Review"
        title="Inspect the immutable deployment"
      />
      <div className="apps-wizard__review-grid">
        <div className="apps-wizard__review-column">
          <h4 className="apps-wizard__subhead">Redacted Compose</h4>
          <pre aria-label="Redacted Compose draft" className="apps-wizard__compose" tabIndex={0}><code>{draft.redactedCompose}</code></pre>
          <p className="apps-wizard__muted">Credential values are injected by the broker at execution time and cannot appear in this preview.</p>
        </div>
        <div className="apps-wizard__review-column">
          <h4 className="apps-wizard__subhead">Validation</h4>
          <ul className="apps-wizard__validation">
            {draft.validation.map((item, index) => (
              <li data-level={item.level} key={`${item.message}${index}`}>
                <StatusPill label={item.level} tone={VALIDATION_TONES[item.level]} />
                <span>{item.message}</span>
              </li>
            ))}
          </ul>
          <AppsFacts
            rows={[
              { label: "Draft", value: draft.id, mono: true },
              { label: "Created", value: new Date(draft.createdAt).toLocaleString() },
              { label: "Images", value: `${draft.images.length} digest-pinned` },
            ]}
          />
        </div>
      </div>
    </section>
  );
}

const DEPLOY_PILLS: Record<DeploymentProgress["state"], { label: string; tone: StatusTone }> = {
  queued: { label: "Queued", tone: "info" },
  running: { label: "Running", tone: "info" },
  healthy: { label: "Healthy", tone: "success" },
  failed: { label: "Failed", tone: "danger" },
  cancelled: { label: "Cancelled", tone: "neutral" },
  rolling_back: { label: "Rolling back", tone: "info" },
  rolled_back: { label: "Rolled back", tone: "neutral" },
};

const CHECK_PILLS: Record<"pending" | "fail", { label: string; tone: StatusTone }> = {
  pending: { label: "Pending", tone: "neutral" },
  fail: { label: "Failed", tone: "danger" },
};

export function DeployStep({ deployment }: { deployment: DeploymentProgress | null | undefined }) {
  const state = deployment?.state ?? "queued";
  const pill = DEPLOY_PILLS[state];
  const progress = deployment?.progress ?? 0;
  return (
    <section aria-labelledby="application-progress-heading" className="apps-wizard__section">
      <StepHeading
        id="application-progress-heading"
        pill={<StatusPill label={pill.label} tone={pill.tone} />}
        step="05 / Deploy"
        title="Deployment and health"
      />
      <AppsProgress fraction={progress / 100} label="Deployment progress" />
      <p className={state === "failed" ? "apps-wizard__deploy-message apps-wizard__deploy-message--failed" : "apps-wizard__deploy-message"} role="status">
        {deployment?.message ?? "The approved deployment is waiting for an executor."}
      </p>
      {deployment?.healthChecks?.length ? (
        <ul aria-label="Health checks" className="apps-wizard__health">
          {deployment.healthChecks.map((check) => {
            const checkPill = check.status === "pass" ? null : CHECK_PILLS[check.status];
            return (
              <li data-status={check.status} key={check.name}>
                <strong>{check.name}</strong>
                <span>{check.detail || check.status}</span>
                {checkPill && <StatusPill label={checkPill.label} tone={checkPill.tone} />}
              </li>
            );
          })}
        </ul>
      ) : null}
      {deployment?.openUrl && <a className="apps-wizard__endpoint" href={deployment.openUrl} rel="noreferrer" target="_blank">Open verified endpoint</a>}
    </section>
  );
}

/**
 * The final authorization, its own small dialog over the wizard. Go back takes
 * focus first, and a refusal shows inside it (VD-189) — the page under it is
 * inert.
 */
export function ApprovalDialog({
  busy,
  draft,
  error,
  onApprove,
  onClose,
}: {
  busy: boolean;
  draft: ComposeDraft;
  error?: ReactNode;
  onApprove: () => void;
  onClose: () => void;
}) {
  const goBack = useRef<HTMLButtonElement>(null);
  return (
    <AppsDialog
      busy={busy}
      describedBy="deployment-approval-description"
      error={error}
      eyebrow="Final authorization"
      footer={<>
        <Button disabled={busy} onClick={onClose} ref={goBack}>Go back</Button>
        <Button busy={busy} onClick={onApprove} variant="primary">{busy ? "Queuing deployment…" : "Approve and deploy"}</Button>
      </>}
      initialFocusRef={goBack}
      onClose={onClose}
      size="narrow"
      title="Approve this exact draft?"
      titleId="deployment-approval-heading"
    >
      <p className="apps-wizard__approval-lead" id="deployment-approval-description">Vaelor will deploy only manifest <code>{shortDigest(draft.manifestDigest)}</code>. Any edit or research change creates a new draft and requires another approval.</p>
      <ul className="apps-wizard__check-list apps-wizard__check-list--done">
        {["Images are bound to reviewed digests.", "Compose validation has no blocking errors.", "Secrets remain in the credential broker.", "Health checks and rollback are audited."].map((item) => (
          <li key={item}><Icon aria-hidden="true" name="done" size={ICON_SIZE.inline} /><span>{item}</span></li>
        ))}
      </ul>
    </AppsDialog>
  );
}
