import type { FormEvent, ReactNode } from "react";
import { AppsKv } from "./appsKit";
import { type ApplicationIntent, imageArchitectures, type ResearchReport, WORKFLOW_STEPS } from "./applicationDeploymentModel";
import { Icon, ICON_SIZE } from "./Icon";
import { StatusPill } from "./StatusPill";
import { Button, Textarea } from "./ui";

/*
 * The custom-application wizard's chrome and its first step (the
 * AppsWizardResearch board, "1 · Request"): the step bar that replaces the
 * numbered list, the guarded-research line that stays under it on every step,
 * the current-request card, and the request form with what Vaelor detected.
 */

/** The five steps; done steps carry a tick, the current one is orange. */
export function WizardStepBar({ current }: { current: number }) {
  return (
    <ol aria-label="Deployment progress" className="apps-wizard__steps">
      {WORKFLOW_STEPS.map((label, index) => {
        const step = index + 1;
        const state = step < current ? "done" : step === current ? "current" : "next";
        return (
          <li aria-current={state === "current" ? "step" : undefined} className={"apps-wizard__step apps-wizard__step--" + state} data-complete={state === "done"} key={label}>
            <span aria-hidden="true" className="apps-wizard__step-mark">{state === "done" ? <Icon name="done" size={ICON_SIZE.inline} /> : step}</span>
            <span>{label}</span>
            {state === "done" && <span className="sr-only">, done</span>}
          </li>
        );
      })}
    </ol>
  );
}

export function GuardedResearchLine() {
  return (
    <div className="apps-wizard__guarded" role="note">
      <Icon aria-hidden="true" name="shield" size={ICON_SIZE.inline} />
      <span><strong>Guarded internet research.</strong> Public HTTPS sources are evidence only. Vaelor never gives the model direct network, shell, Docker, or credential access.</span>
    </div>
  );
}

/** A step's heading: "02 / Research" over the title, a status pill at the right. */
export function StepHeading({ id, pill, step, title }: { id: string; pill?: ReactNode; step: string; title: string }) {
  return (
    <div className="apps-wizard__heading">
      <div>
        <span className="apps-eyebrow">{step}</span>
        <h3 id={id}>{title}</h3>
      </div>
      {pill}
    </div>
  );
}

/** The request this wizard is working on, with the way back to the previous step. */
export function CurrentRequest({ action, intent, research, showFacts = true }: { action?: ReactNode; intent: ApplicationIntent; research: ResearchReport | null; showFacts?: boolean }) {
  const complete = research?.status === "complete" ? research : null;
  return (
    <aside aria-label="Current deployment request" className="apps-wizard__current">
      <div className="apps-wizard__current-head">
        <div className="apps-wizard__current-text">
          <small>Current request</small>
          <strong>{intent.application}</strong>
          <span>{complete ? complete.compatibilitySummary : intent.summary}</span>
        </div>
        {action}
      </div>
      {complete && showFacts && (
        <AppsKv
          cells={[
            { label: "License", value: complete.license || "Not stated" },
            { label: "Architecture", value: imageArchitectures(complete) || "Not stated" },
            { label: "Evidence", value: `${complete.sources.length} sources` },
          ]}
          label="Request facts"
        />
      )}
    </aside>
  );
}

export function RequestStep({
  disabled,
  error,
  formId,
  intent,
  loading,
  onChange,
  onRetry,
  onSubmit,
  request,
  research,
  researchWorking,
}: {
  disabled: boolean;
  /** An intent-stage failure (a 422 "not detected" or a 502): the recovery panel shows under the form. */
  error: boolean;
  formId: string;
  intent: ApplicationIntent | null;
  loading: boolean;
  onChange: (value: string) => void;
  onRetry: () => void;
  onSubmit: (event: FormEvent) => void;
  request: string;
  research: ResearchReport | null;
  researchWorking: boolean;
}) {
  const stage = researchWorking ? "Research started automatically" : research ? "Request researched" : "Preparing guarded research";
  return (
    <section aria-labelledby="application-request-heading" className="apps-wizard__section">
      <StepHeading id="application-request-heading" step="01 / Request" title="What should Vaelor deploy?" />
      <form className="application-deployment__request-form apps-wizard__request-form" id={formId} onSubmit={onSubmit}>
        <Textarea disabled={disabled || loading} id="application-request" label="Describe the application and how you expect to use it" onChange={(event) => onChange(event.target.value)} placeholder="Example: Deploy a private Home Assistant server with its data on NVMe and expose port 8123 on my LAN." rows={3} value={request} />
      </form>
      {intent && (
        <div className="apps-wizard__intent">
          <div className="apps-wizard__intent-head">
            <small>Detected application{intent.refinement?.modelUsed ? " · interpreted by selected model" : " · built-in parser"}</small>
            <StatusPill label={stage} tone={research && !researchWorking ? "neutral" : "info"} />
          </div>
          <strong className="apps-wizard__intent-name">{intent.application}</strong>
          <p>{intent.summary}</p>
          {intent.researchCapability && (
            <div className="apps-wizard__intent-capability">
              <small>Research intelligence</small>
              <span className="apps-wizard__intent-model">{intent.researchCapability.label}</span>
              <p>{intent.researchCapability.summary}</p>
              {intent.researchCapability.strongerModelRecommended && intent.researchCapability.recommendation ? <p>{intent.researchCapability.recommendation}</p> : null}
              <details className="apps-wizard__disclosure">
                <summary>Current limitations</summary>
                <ul className="apps-wizard__dot-list">{intent.researchCapability.limitations.map((item) => <li key={item}>{item}</li>)}</ul>
              </details>
            </div>
          )}
        </div>
      )}
      {error && (
        // #247b: an intent-stage failure (a 422 "not detected" or a 502
        // workflow error) gets the same affordance the research stage has: edit
        // the request and try again, instead of a wall.
        <div aria-label="Request recovery" className="apps-wizard__inset" role="group">
          <p><strong>What to do next:</strong> Edit your request above to be more specific - for example, name the exact application or paste its official website - then try again. Vaelor did not create or change anything.</p>
          <div><Button disabled={disabled || !request.trim()} onClick={onRetry}>Try again</Button></div>
        </div>
      )}
    </section>
  );
}
