import type { ReactNode } from "react";
import { AppsKv, AppsProgress } from "./appsKit";
import {
  modelTierLabel,
  needsImageEvidence,
  researchBlockers,
  researchPhaseLabels,
  type ResearchReport,
  shortDigest,
} from "./applicationDeploymentModel";
import { StepHeading } from "./ApplicationWizardRequest";
import { Icon, ICON_SIZE } from "./Icon";
import { StatusPill } from "./StatusPill";
import { Button, Textarea } from "./ui";
import type { StatusTone } from "./ui/status";

/*
 * Step 2 of the custom-application wizard (the AppsWizardResearch board):
 * research running, needing input, stopped, not supported, and complete with
 * its facts, verified images and paged sources; plus the research-model card
 * and its escalation states.
 */

export const SOURCE_PAGE_SIZE = 4;
export const LARGER_MODEL_UNAVAILABLE = "The larger model is unavailable.";

const RESEARCH_CHECKS = [
  "Official documentation and current installation guidance",
  "Container images and ARM64 support",
  "Memory, storage, network ports, and persistent data",
  "Safe defaults, health checks, and recovery requirements",
];

function researchPill(research: ResearchReport | null, needsInput: boolean): { label: string; tone: StatusTone } | null {
  if (!research) return null;
  if (needsInput) return { label: "Needs your input", tone: "warning" };
  if (research.status === "failed") return null;
  if (research.status !== "complete" || research.compatibility === "pending") return { label: "Checking", tone: "info" };
  if (research.compatibility === "compatible") return { label: "Compatible", tone: "success" };
  if (research.compatibility === "conditional") return { label: "Compatible with conditions", tone: "warning" };
  return { label: "Not supported", tone: "danger" };
}

export interface EscalationControl {
  busy: boolean;
  disabled: boolean;
  onEscalate?: () => void;
}

function LargerModelButton({ control, reason, unavailable }: { control: EscalationControl; reason?: string; unavailable: boolean }) {
  return (
    <Button
      busy={control.busy}
      disabled={control.disabled}
      disabledReason={unavailable ? reason || LARGER_MODEL_UNAVAILABLE : undefined}
      onClick={control.onEscalate}
    >
      Try the larger model
    </Button>
  );
}

/**
 * The tier that produced the CURRENT manifest, plus the manual escalation. The
 * label is always the tier that was actually used — never a graphics-model
 * claim when the assistant ran.
 */
export function ResearchModelPanel({
  control,
  hasDraft,
  research,
  strongerRecommended,
}: {
  control: EscalationControl;
  hasDraft: boolean;
  research: ResearchReport;
  strongerRecommended: boolean;
}) {
  const tier = research.modelTier;
  if (!tier) return null;
  const onAssistant = tier !== "gpu/ai-chat";
  const canEscalate = !hasDraft && onAssistant && Boolean(control.onEscalate);
  const note = research.escalatedToCapable && onAssistant
    ? "Escalated to the graphics model. Its richer result will replace this plan shortly."
    : canEscalate && strongerRecommended
      ? "A richer plan may be possible on the graphics model."
      : null;
  return (
    <aside aria-label="Research model" className="apps-wizard__current">
      <div className="apps-wizard__current-head">
        <div className="apps-wizard__current-text">
          <small>Research model</small>
          <strong>{modelTierLabel(tier)}</strong>
          {note && <span>{note}</span>}
        </div>
        {canEscalate && <LargerModelButton control={control} reason={research.capableUnavailableReason} unavailable={research.capableAvailable === false} />}
      </div>
    </aside>
  );
}

/** Advanced recovery: the operator names official HTTPS sources (at most eight). */
export function SourceRecovery({
  busy,
  disabled,
  invalid,
  onChange,
  onRetry,
  urlCount,
  value,
}: {
  busy: boolean;
  disabled: boolean;
  invalid: boolean;
  onChange: (value: string) => void;
  onRetry: () => void;
  urlCount: number;
  value: string;
}) {
  const reason = urlCount === 0
    ? "Add at least one HTTPS source."
    : invalid ? "Enter no more than eight valid HTTPS source URLs, one per line." : undefined;
  return (
    <details className="apps-wizard__disclosure apps-wizard__sources-recovery">
      <summary>Advanced recovery: add official sources</summary>
      <p>Most people do not need this. Use it only when automatic research could not identify the application or its official documentation.</p>
      <Textarea aria-invalid={invalid || undefined} hint="Enter up to eight public HTTPS pages, one per line. Do not enter passwords, tokens, or private links." id="application-source-urls" label="Official documentation URLs" onChange={(event) => onChange(event.target.value)} placeholder="https://project.example/docs/install" rows={4} value={value} />
      <div className="apps-wizard__inline-actions">
        <span className="apps-wizard__count">{urlCount} / 8 sources</span>
        <Button busy={busy} disabled={disabled} disabledReason={busy || disabled ? undefined : reason} onClick={onRetry}>{busy ? "Checking safely…" : "Retry with these sources"}</Button>
      </div>
    </details>
  );
}

function ResearchProgress({ research }: { research: ResearchReport }) {
  const label = researchPhaseLabels[research.phase || ""] || "Working";
  const progress = research.progress ?? 0;
  return (
    <div className="apps-wizard__progress">
      <div className="apps-wizard__progress-row"><span>{label}</span><strong>{progress}%</strong></div>
      <AppsProgress fraction={progress / 100} label="Research progress" />
      <small>You can close this window. Vaelor saves this operation and resumes here when you return.</small>
    </div>
  );
}

export function ResearchStep({
  advancedRecovery,
  clarifying,
  escalation,
  multiService,
  needsInput,
  onCancel,
  onRetry,
  research,
  researchRecovery,
  retrying,
  setSourcePage,
  sourcePage,
  starting,
}: {
  advancedRecovery: ReactNode;
  clarifying: { context: string; questions: string[] } | null;
  escalation: EscalationControl;
  multiService: boolean;
  needsInput: boolean;
  onCancel?: () => void;
  onRetry?: () => void;
  research: ResearchReport | null;
  researchRecovery?: ReactNode;
  retrying: boolean;
  setSourcePage: (page: number) => void;
  sourcePage: number;
  starting: boolean;
}) {
  const pill = researchPill(research, needsInput);
  return (
    <section aria-labelledby="application-research-heading" className="apps-wizard__section">
      <StepHeading
        id="application-research-heading"
        pill={pill && <StatusPill label={pill.label} tone={pill.tone} />}
        step="02 / Research"
        title="Verify compatibility and sources"
      />
      {!research && (
        <div className="apps-wizard__inset" role="status" aria-live="polite">
          <h4>{starting ? "Researching in the background" : "Automatic research needs help"}</h4>
          <p>{starting ? "Vaelor is finding trustworthy documentation and checking whether the application can run safely on this device." : "Vaelor could not identify enough reliable information automatically. You can retry the request, or use the advanced recovery option."}</p>
          {starting && (
            <ul aria-label="Research checks in progress" className="apps-wizard__check-list">
              {RESEARCH_CHECKS.map((item) => <li key={item}><Icon aria-hidden="true" name="search" size={ICON_SIZE.inline} /><span>{item}</span></li>)}
            </ul>
          )}
        </div>
      )}
      {!research && !starting && advancedRecovery}
      {research && research.status !== "complete" && (
        <ResearchPending
          advancedRecovery={advancedRecovery}
          clarifying={clarifying}
          escalation={escalation}
          needsInput={needsInput}
          onCancel={onCancel}
          onRetry={onRetry}
          research={research}
          researchRecovery={researchRecovery}
          retrying={retrying}
        />
      )}
      {research?.status === "complete" && (
        <ResearchComplete
          advancedRecovery={advancedRecovery}
          multiService={multiService}
          research={research}
          researchRecovery={researchRecovery}
          setSourcePage={setSourcePage}
          sourcePage={sourcePage}
        />
      )}
    </section>
  );
}

function ResearchPending({
  advancedRecovery,
  clarifying,
  escalation,
  needsInput,
  onCancel,
  onRetry,
  research,
  researchRecovery,
  retrying,
}: {
  advancedRecovery: ReactNode;
  clarifying: { context: string; questions: string[] } | null;
  escalation: EscalationControl;
  needsInput: boolean;
  onCancel?: () => void;
  onRetry?: () => void;
  research: ResearchReport;
  researchRecovery?: ReactNode;
  retrying: boolean;
}) {
  const running = research.status === "queued" || research.status === "running";
  const stopped = research.status === "failed" || needsInput;
  const blocker = researchBlockers[research.blockerLayer || ""];
  const title = research.status === "failed"
    ? (needsInput ? researchPhaseLabels.needs_input : blocker?.title || "Automatic research could not be completed")
    : (researchPhaseLabels[research.phase || ""] || "Research is continuing automatically");
  const showsSourceRecovery = needsInput || !research.blockerLayer || ["interpretation", "discovery", "acquisition", "synthesis"].includes(research.blockerLayer);
  return (
    <>
      <div className="apps-wizard__inset" role="status" aria-live="polite">
        <h4>{title}</h4>
        {!clarifying && <p>{research.error || "Vaelor is checking trusted sources, container support, device fit, networking, storage, and safe operating defaults. You may close this window; the operation is saved and resumes when you return."}</p>}
        {clarifying && (
          // #247x: a distinct "needs your input" block — concrete, grounded
          // questions the operator answers, not a dead-end and not a guessed
          // answer. Edit request and Advanced recovery are the ways to respond.
          <div aria-label="Vaelor needs more information" className="apps-wizard__clarify" role="group">
            <strong>Vaelor needs a bit more to proceed</strong>
            {clarifying.context && <p>{clarifying.context}</p>}
            <ol>{clarifying.questions.map((question) => <li key={question}>{question}</li>)}</ol>
            <p>Answer above with “Edit request”, or add the official source under Advanced recovery, then research again.</p>
          </div>
        )}
        {!clarifying && stopped && blocker && <p className="apps-wizard__next"><strong>{needsInput ? "Research is waiting for your input:" : "What to do next:"}</strong> {blocker.recovery}</p>}
        {running && <ResearchProgress research={research} />}
        <div className="apps-wizard__inline-actions">
          {running && onCancel && <Button disabled={retrying} onClick={onCancel}>Cancel research</Button>}
          {stopped && onRetry && <Button disabled={retrying} onClick={onRetry}>{retrying ? "Retrying…" : "Retry automatic research"}</Button>}
          {/* The same graphics-model escalation the research-model card offers,
              reachable when discovery HARD-FAILED (no manifest, so no card).
              Disabled honestly when no graphics lease is live; never offered
              when the pass that just failed was already the graphics model. */}
          {stopped && escalation.onEscalate && research.modelTier !== "gpu/ai-chat" && (
            <LargerModelButton control={escalation} reason={research.capableUnavailableReason} unavailable={research.capableAvailable === false} />
          )}
        </div>
      </div>
      {stopped && researchRecovery}
      {stopped && showsSourceRecovery && advancedRecovery}
    </>
  );
}

function ResearchComplete({
  advancedRecovery,
  multiService,
  research,
  researchRecovery,
  setSourcePage,
  sourcePage,
}: {
  advancedRecovery: ReactNode;
  multiService: boolean;
  research: ResearchReport;
  researchRecovery?: ReactNode;
  setSourcePage: (page: number) => void;
  sourcePage: number;
}) {
  const pages = Math.max(1, Math.ceil(research.sources.length / SOURCE_PAGE_SIZE));
  const visible = research.sources.slice(sourcePage * SOURCE_PAGE_SIZE, (sourcePage + 1) * SOURCE_PAGE_SIZE);
  return (
    <>
      <p className="apps-wizard__summary">{research.compatibilitySummary}</p>
      {research.compatibility === "unsupported" && (needsImageEvidence(research) ? (
        // The verdict failed only because no official container image could be
        // verified from the sources Vaelor had — not because the app does not
        // fit. Point at the real, actionable recovery on this screen.
        <div aria-label="Verified container image needed" className="apps-wizard__inset apps-wizard__inset--danger" role="group">
          <h4>Vaelor stopped before making changes</h4>
          <p>No official container image could be verified from the available sources, so there is nothing safe to deploy yet. Set up guarded web research so Vaelor can find the official image, or add the official image source under Advanced recovery. Nothing was downloaded or installed.</p>
          {researchRecovery}
          {advancedRecovery}
        </div>
      ) : (
        <div className="apps-wizard__inset apps-wizard__inset--danger" role="alert">
          <h4>Vaelor stopped before making changes</h4>
          <p>Configuration is unavailable because Vaelor could not confirm this application fits this appliance. The compatibility finding above explains why. Nothing was downloaded or installed.</p>
        </div>
      ))}
      <AppsKv
        cells={[
          { label: "License", value: research.license || "Not stated" },
          { label: "Minimum memory", value: research.minimumMemoryMb ? `${research.minimumMemoryMb} MiB` : "Not stated" },
          { label: "Minimum storage", value: research.minimumStorageGb ? `${research.minimumStorageGb} GB` : "Not stated" },
          { label: "Evidence", value: `${research.sources.length} cited sources` },
        ]}
        label="Research facts"
      />
      <h4 className="apps-wizard__subhead">Verified container images{multiService ? ` (${research.images.length} services)` : ""}</h4>
      {research.images.length === 0 ? <p className="apps-wizard__muted">No container image was verified.</p> : (
      <div className="apps-wizard__table-wrap">
        <table className="apps-wizard__table">
          <thead><tr>{multiService && <th>Service</th>}<th>Image</th><th>Immutable digest</th><th>Architectures</th><th>Verification</th></tr></thead>
          <tbody>
            {research.images.map((image) => (
              <tr key={`${image.image}${image.digest}`}>
                {multiService && <td>{image.service || "—"}</td>}
                <td>{image.image}</td>
                <td><code title={image.digest}>{shortDigest(image.digest)}</code></td>
                <td>{image.architectures.join(", ")}</td>
                <td><StatusPill label={image.verified ? "Verified" : "Unverified"} tone={image.verified ? "success" : "warning"} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      )}
      <div className="apps-wizard__sources">
        <div className="apps-wizard__subhead-row"><h4 className="apps-wizard__subhead">Sources and supported claims</h4><span>Page {sourcePage + 1} of {pages}</span></div>
        {research.sources.length === 0
          ? <p>No attributable sources were returned. Deployment cannot proceed.</p>
          : (
            <div className="apps-wizard__source-grid">
              {visible.map((source) => (
                <article className="apps-wizard__source" key={source.id}>
                  <a href={source.url} rel="noreferrer" target="_blank">{source.title}</a>
                  <small>{source.publisher} · retrieved {new Date(source.retrievedAt).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })}</small>
                  <ul className="apps-wizard__dot-list">{source.supports.map((claim) => <li key={claim}>{claim}</li>)}</ul>
                </article>
              ))}
            </div>
          )}
        <nav aria-label="Research source pages" className="apps-wizard__inline-actions">
          <Button disabled={sourcePage === 0} onClick={() => setSourcePage(Math.max(0, sourcePage - 1))}>Previous</Button>
          <Button disabled={sourcePage + 1 >= pages} onClick={() => setSourcePage(Math.min(pages - 1, sourcePage + 1))}>Next</Button>
        </nav>
      </div>
    </>
  );
}
