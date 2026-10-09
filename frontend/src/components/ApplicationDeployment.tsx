import { FormEvent, type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import "../styles/apps-wizard.css";
import { AppsDialog } from "./appsKit";
import {
  type ApplicationIntent,
  type AsyncState,
  clarifyingQuestions,
  type ComposeDraft,
  type DeploymentConfiguration,
  type DeploymentProgress,
  type ResearchReport,
  researchCanPoll,
  researchNeedsInput,
  type WorkflowPhase,
  workflowPhaseFor,
  workflowStep,
} from "./applicationDeploymentModel";
import { ConfigureLocked, ConfigureStep, useDeploymentConfiguration } from "./ApplicationWizardConfigure";
import { CurrentRequest, GuardedResearchLine, RequestStep, StepHeading, WizardStepBar } from "./ApplicationWizardRequest";
import { ResearchModelPanel, ResearchStep, SourceRecovery } from "./ApplicationWizardResearch";
import { ApprovalDialog, DeployStep, ReviewStep, VALIDATION_ERRORS_REASON } from "./ApplicationWizardReview";
import { StatusPill } from "./StatusPill";
import { Button, Notice } from "./ui";

export {
  CLARIFYING_QUESTIONS_MARKER,
  defaultDeploymentName,
  MAX_DEPLOYMENT_NAME,
  PRIVILEGED_HOST_MOUNTS,
  UNVERIFIED_IMAGE_MARKER,
} from "./applicationDeploymentModel";
export type {
  ApplicationIntent,
  AsyncState,
  ComposeDraft,
  DeploymentConfiguration,
  DeploymentProgress,
  ImageFact,
  PrivilegedHostMountSource,
  ResearchModelTier,
  ResearchReport,
  ResearchSource,
  WorkflowPhase,
} from "./applicationDeploymentModel";

export interface ApplicationDeploymentProps {
  initialRequest?: string;
  intent?: ApplicationIntent | null;
  research?: ResearchReport | null;
  draft?: ComposeDraft | null;
  deployment?: DeploymentProgress | null;
  resumeResearch?: { jobId: string; draftId: string; requestSummary?: string };
  /** Standalone stories may auto-start. The production container owns this transition. */
  autoStartResearch?: boolean;
  /**
   * Whether the underlying server draft can still be configured. A draft can be
   * configured exactly once; once a validated compose exists (e.g. a finalized
   * draft reopened from the resume banner) the Configure step is presented
   * read-only, not as an editable form whose only outcome is a
   * "can no longer be configured" error. Defaults to configurable.
   */
  configurable?: boolean;
  disabled?: boolean;
  /**
   * D4d. Whether a Swarm cluster with at least one joined worker is active. When
   * it is, a reviewed (approvable) draft offers "Deploy to cluster" beside the
   * single-node install — the multi-service cluster deploy of the SAME reviewed,
   * digest-pinned draft. Absent/false leaves only the single-node path, which is
   * never changed by this.
   */
  clusteringActive?: boolean;
  /** Opens the cluster placement flow for the reviewed draft (D4d). */
  onDeployToCluster?: (draft: ComposeDraft) => void;
  onClose?: () => void;
  onDirtyChange?: (dirty: boolean) => void;
  onClassify: (request: string) => Promise<ApplicationIntent>;
  onStartResearch: (intent: ApplicationIntent, sourceUrls: string[]) => Promise<ResearchReport>;
  onRefreshResearch?: (researchId: string) => Promise<ResearchReport>;
  onCancelResearch?: (researchId: string) => Promise<ResearchReport>;
  onRetryResearch?: (researchId: string) => Promise<ResearchReport>;
  onReplaceActiveResearch?: () => Promise<void>;
  /**
   * Manually re-run research on the capable graphics model (A). Enqueues the
   * capable pass on the current draft and returns a pending report the wizard
   * polls to completion. Honestly degrades: when no graphics lease is live the
   * control is disabled with its reason, and a late 409 keeps the assistant's
   * result while flipping `capableAvailable` off — never a generic error.
   */
  onEscalateToCapable?: (researchId: string) => Promise<ResearchReport>;
  researchRecovery?: ReactNode;
  onGenerateDraft: (configuration: DeploymentConfiguration) => Promise<ComposeDraft>;
  onStoreSecret?: (name: string, value: string) => Promise<string>;
  onApprove: (draft: ComposeDraft) => Promise<void>;
  onCancelDeployment?: () => Promise<void>;
  onManage?: () => void;
  onRollback?: () => Promise<void>;
  onIntentChange?: (intent: ApplicationIntent) => void;
  onResearchChange?: (research: ResearchReport) => void;
  onDraftChange?: (draft: ComposeDraft) => void;
}

/** Only an administrator may deploy; the wizard says so beside each held action. */
export const ADMINISTRATOR_ONLY_REASON = "Only an administrator can deploy a custom application.";
const REQUEST_FORM_ID = "application-request-form";
const CONFIGURE_FORM_ID = "application-configure-form";

/**
 * The custom-application wizard (VD-200, the AppsWizardResearch and
 * AppsWizardDeploy boards): five steps in ONE dialog — Request, Research,
 * Configure, Review, Deploy — with the step's action in the dialog footer.
 * "Back to Apps and AI" is the header's close action.
 */
export function ApplicationDeployment(props: ApplicationDeploymentProps) {
  const [request, setRequest] = useState(props.initialRequest ?? "");
  const [intent, setIntent] = useState<ApplicationIntent | null>(props.intent ?? null);
  const [research, setResearch] = useState<ResearchReport | null>(props.research ?? null);
  const [draft, setDraft] = useState<ComposeDraft | null>(props.draft ?? null);
  const [phase, setPhase] = useState<WorkflowPhase>(workflowPhaseFor(props.research, props.draft));
  const [asyncState, setAsyncState] = useState<AsyncState>("idle");
  const [error, setError] = useState("");
  const [sourcePage, setSourcePage] = useState(0);
  const [sourceUrlText, setSourceUrlText] = useState("");
  const [showApproval, setShowApproval] = useState(false);
  const config = useDeploymentConfiguration(intent, research);
  // A failed graphics-model escalation must not erase a verified plan the
  // assistant already produced. When escalating from a working report we
  // snapshot it here; if the capable pass comes back failed/needs_input we
  // restore the snapshot and show `escalationNote` instead of the failure.
  // Only ever a report we actually held - never fabricated.
  const escalationSnapshotRef = useRef<ResearchReport | null>(null);
  const [escalationNote, setEscalationNote] = useState("");
  /**
   * Whether the reader put themselves on this step.
   *
   * The two effects below re-derive the workflow phase from props, so **any**
   * prop update - a job poll, a restore resolving late, a parent re-render -
   * recomputed the step and moved the reader to it. Click "Edit request" on a
   * slow network and the next poll returned you to Research with the form you
   * were typing into gone. It surfaced as a suite that failed about one run in
   * three; on an appliance it is an edit that cannot be started.
   *
   * The rule is the same one the lighting draft needed: a background update may
   * **seed** state the reader has not engaged with, and must not **overwrite**
   * state they have. So props still drive the phase right up until the reader
   * navigates, and stop the moment they do - until the reader's own next
   * action resumes the workflow and hands the wheel back.
   */
  const readerChosePhase = useRef(false);
  const autoResearchIntentRef = useRef<string | null>(null);
  const refreshInFlightRef = useRef(false);
  /**
   * The full request the reader last submitted for classification - what
   * "Edit request" restores. It is deliberately NOT the derived
   * `intent.application`, which is a lossy app-name summary (e.g. "Vaultwarden
   * self hosted password manager"); re-researching that bare noun phrase failed
   * the intent gate ("Ask to deploy…", #247v). Seed it from the original
   * `initialRequest` and refresh it every time the reader classifies.
   */
  const submittedRequestRef = useRef(props.initialRequest ?? "");

  // A background `props.intent` update seeds the identity panels, but must never
  // overwrite the editable request with the derived app-name summary (#247v).
  useEffect(() => { if (props.intent !== undefined) { setIntent(props.intent); } }, [props.intent]);
  useEffect(() => { if (props.research !== undefined) { setResearch(props.research); if (!readerChosePhase.current) setPhase(workflowPhaseFor(props.research, props.draft ?? draft)); } }, [props.draft, props.research, draft]);
  useEffect(() => { if (props.draft !== undefined) { setDraft(props.draft); if (!readerChosePhase.current) setPhase(workflowPhaseFor(props.research ?? research, props.draft)); } }, [props.draft, props.research, research]);
  useEffect(() => {
    props.onDirtyChange?.(config.dirty);
    return () => props.onDirtyChange?.(false);
  }, [config.dirty, props.onDirtyChange]);

  useEffect(() => {
    if (props.autoStartResearch === false || !intent || research || asyncState === "loading" || autoResearchIntentRef.current === intent.id) return;
    autoResearchIntentRef.current = intent.id;
    setPhase("research");
    void run(() => props.onStartResearch(intent, []), (result) => {
      setResearch(result);
      props.onResearchChange?.(result);
      if (result.status === "complete") setPhase(result.compatibility === "unsupported" ? "research" : "configure");
    });
  }, [asyncState, intent, props.autoStartResearch, props.onResearchChange, props.onStartResearch, research]);

  useEffect(() => {
    if (!research || !researchCanPoll(research) || !props.onRefreshResearch) return;
    const researchId = research.id;
    const timer = window.setTimeout(async () => {
      if (refreshInFlightRef.current) return;
      refreshInFlightRef.current = true;
      try {
        const result = await props.onRefreshResearch!(researchId);
        applyResearchResult(result);
      } catch (caught) {
        setAsyncState("error");
        setError(caught instanceof Error ? caught.message : "Vaelor could not check research progress.");
      } finally {
        refreshInFlightRef.current = false;
      }
    }, 2000);
    return () => window.clearTimeout(timer);
  }, [props.onRefreshResearch, props.onResearchChange, research]);

  /** A step the reader chose. Props stop re-deriving the phase until they act. */
  const holdPhase = (next: WorkflowPhase) => {
    readerChosePhase.current = true;
    setPhase(next);
  };
  /**
   * "Edit request" restores the reader's ORIGINAL full request, not the derived
   * app-name summary (#247v). If no genuine original was carried (a durable
   * resume surfaces only the summary), the current field is left untouched.
   */
  const editRequest = () => {
    if (submittedRequestRef.current.trim()) setRequest(submittedRequestRef.current);
    holdPhase("request");
  };
  // A deployment the container restored or queued owns the last step, even when
  // the derived phase is still Review (the draft exists too).
  const shownPhase: WorkflowPhase = props.deployment && phase === "review" ? "deploy" : phase;
  const currentStep = workflowStep(shownPhase);
  // A finalized draft (its validated compose already generated) can no longer be
  // configured; the Configure step is shown read-only rather than as an editable
  // form whose "Generate" only errors. Defaults to configurable.
  const configurable = props.configurable !== false;
  const hasValidationErrors = draft?.validation.some((item) => item.level === "error") ?? false;
  const runningResearch = research?.status === "queued" || research?.status === "running";
  const sourceUrls = useMemo(() => sourceUrlText.split(/\r?\n/).map((value) => value.trim()).filter(Boolean), [sourceUrlText]);
  const invalidSourceUrls = sourceUrls.length > 8 || sourceUrls.some((value) => {
    try { return new URL(value).protocol !== "https:"; }
    catch { return true; }
  });

  async function run<T>(operation: () => Promise<T>, onSuccess: (result: T) => void) {
    setAsyncState("loading");
    setError("");
    try { onSuccess(await operation()); setAsyncState("idle"); }
    catch (caught) { setAsyncState("error"); setError(caught instanceof Error ? caught.message : "The operation could not be completed."); }
  }

  /**
   * Apply a fresh research result, protecting a verified plan from a failed
   * escalation. When a graphics-model re-run was started from a working report
   * (a snapshot exists) and it comes back failed or needs_input, the assistant's
   * plan is still safe on the draft — so restore the snapshot and surface an
   * honest, non-destructive note rather than replacing it with the failure. A
   * completed result clears the snapshot; a still-running one keeps it.
   */
  function applyResearchResult(result: ResearchReport) {
    const snapshot = escalationSnapshotRef.current;
    if (snapshot && (result.status === "failed" || result.status === "needs_input")) {
      escalationSnapshotRef.current = null;
      setEscalationNote("The larger model couldn't improve the plan; keeping the assistant's result.");
      setResearch(snapshot); props.onResearchChange?.(snapshot);
      setPhase(snapshot.compatibility === "unsupported" ? "research" : "configure");
      return;
    }
    if (result.status === "complete") escalationSnapshotRef.current = null;
    setResearch(result); props.onResearchChange?.(result);
    if (result.status === "complete") setPhase(result.compatibility === "unsupported" ? "research" : "configure");
  }

  function clearResearchArtifacts() {
    escalationSnapshotRef.current = null; setEscalationNote("");
    setResearch(null); setDraft(null); setSourcePage(0); setSourceUrlText(""); config.reset(); setShowApproval(false); setPhase(intent ? "research" : "request"); setError("");
  }

  async function classify(event?: FormEvent) {
    // The reader has started work again, so the workflow may resume deriving
    // the step from what the appliance reports.
    readerChosePhase.current = false;
    event?.preventDefault();
    if (!request.trim()) { setError("Describe the application you want to deploy."); return; }
    submittedRequestRef.current = request.trim();
    setAsyncState("loading"); setError("");
    try { await props.onReplaceActiveResearch?.(); } catch (caught) { setAsyncState("error"); setError(caught instanceof Error ? caught.message : "The previous research operation could not be cancelled."); return; }
    // A new request starts a new server-owned workflow. Never leave the prior
    // application's evidence or approval state visible while the replacement
    // draft is being classified and researched.
    setIntent(null);
    setResearch(null);
    setDraft(null);
    setSourcePage(0);
    config.reset();
    setPhase("request");
    autoResearchIntentRef.current = null;
    try {
      const result = await props.onClassify(request.trim());
      autoResearchIntentRef.current = result.id;
      setIntent(result);
      props.onIntentChange?.(result);
      setPhase("research");
      const researchResult = await props.onStartResearch(result, []);
      setResearch(researchResult);
      props.onResearchChange?.(researchResult);
      if (researchResult.status === "complete") {
        setPhase(researchResult.compatibility === "unsupported" ? "research" : "configure");
      }
      setAsyncState("idle");
    } catch (caught) {
      setAsyncState("error");
      setError(caught instanceof Error ? caught.message : "Vaelor could not research this application.");
    }
  }

  function startResearch() {
    readerChosePhase.current = false;
    if (!intent) return;
    if (invalidSourceUrls) { setError("Enter no more than eight valid HTTPS source URLs, one per line."); return; }
    const enteredSourceUrls = sourceUrls; clearResearchArtifacts();
    setPhase("research");
    void run(() => props.onStartResearch(intent, enteredSourceUrls), (result) => {
      setResearch(result); props.onResearchChange?.(result);
      if (result.status === "complete") setPhase(result.compatibility === "unsupported" ? "research" : "configure");
    });
  }

  function retryResearch() {
    readerChosePhase.current = false;
    if (!research || !props.onRetryResearch) return;
    const researchId = research.id; clearResearchArtifacts();
    void run(() => props.onRetryResearch!(researchId), (result) => {
      setResearch(result); props.onResearchChange?.(result);
      if (result.status === "complete") setPhase(result.compatibility === "unsupported" ? "research" : "configure");
    });
  }

  function escalateToCapable() {
    // A deliberate re-run on the graphics model; the reader has acted again.
    readerChosePhase.current = false;
    if (!research || !props.onEscalateToCapable) return;
    const researchId = research.id;
    // Snapshot a WORKING plan so a failed capable pass can restore it (FIX 3b).
    // Escalating from an already-failed state carries no snapshot.
    escalationSnapshotRef.current = research.status === "complete" ? research : null;
    setEscalationNote("");
    setPhase("research");
    void run(() => props.onEscalateToCapable!(researchId), (result) => {
      // A refused escalation hands back the SAME report with the larger model
      // marked unavailable (no new job). Keep the reader on Research, where the
      // card now says why the button is held, instead of moving them on.
      if (result.id === researchId && result.status === "complete") {
        escalationSnapshotRef.current = null;
        setResearch(result); props.onResearchChange?.(result);
        holdPhase("research");
        return;
      }
      applyResearchResult(result);
    });
  }

  async function prepareDraft() {
    readerChosePhase.current = false;
    if (!intent || !research) return;
    if (config.blockedReasons.length) { setError(config.blockedReasons[0]); return; }
    const enteredSecrets = Object.entries(config.secretValues).filter(([, value]) => value.trim());
    if (enteredSecrets.length && !props.onStoreSecret) { setError("Secure application-secret storage is unavailable."); return; }
    setAsyncState("loading");
    setError("");
    try {
      const stored = await Promise.all(enteredSecrets.map(async ([secretName, value]) => [secretName, await props.onStoreSecret!(secretName, value)] as const));
      const nextReferences = { ...config.secretReferences, ...Object.fromEntries(stored) };
      config.setSecretReferences(nextReferences);
      config.clearSecretValues();
      const result = await props.onGenerateDraft(config.build(request, intent.id, research.id, nextReferences));
      setDraft(result); props.onDraftChange?.(result); config.markClean(); setPhase("review");
      setAsyncState("idle");
    } catch (caught) {
      setAsyncState("error");
      setError(caught instanceof Error ? caught.message : "The application draft could not be prepared.");
    }
  }

  function generateDraft(event: FormEvent) {
    event.preventDefault();
    void prepareDraft();
  }

  function approve() {
    if (!draft) return;
    void run(() => props.onApprove(draft), () => { setShowApproval(false); setPhase("deploy"); });
  }

  const loading = asyncState === "loading";
  const researchStarting = phase === "research" && loading && !research;
  const needsInput = researchNeedsInput(research);
  const clarifying = clarifyingQuestions(research?.error);
  const adminReason = props.disabled ? ADMINISTRATOR_ONLY_REASON : undefined;
  const escalation = {
    busy: phase === "research" && loading,
    disabled: Boolean(props.disabled),
    onEscalate: props.onEscalateToCapable ? escalateToCapable : undefined,
  };
  const advancedRecovery = (
    <SourceRecovery
      busy={loading}
      disabled={Boolean(props.disabled) || invalidSourceUrls || sourceUrls.length === 0}
      invalid={invalidSourceUrls}
      onChange={setSourceUrlText}
      onRetry={startResearch}
      urlCount={sourceUrls.length}
      value={sourceUrlText}
    />
  );
  const closeButton = props.onClose ? <Button onClick={props.onClose}>Close</Button> : null;

  let body: ReactNode = null;
  let footer: ReactNode = closeButton;
  if (shownPhase === "request") {
    const label = loading ? (phase === "request" ? "Understanding request…" : "Researching automatically…") : intent ? "Research again" : "Research and prepare plan";
    body = (
      <RequestStep
        disabled={Boolean(props.disabled)}
        error={asyncState === "error" && Boolean(error)}
        formId={REQUEST_FORM_ID}
        intent={intent}
        loading={loading}
        onChange={setRequest}
        onRetry={() => void classify()}
        onSubmit={classify}
        request={request}
        research={research}
        researchWorking={researchStarting || runningResearch}
      />
    );
    footer = (
      <span className="apps-wizard__footer-action">
        <Button busy={loading} disabledReason={loading ? undefined : adminReason ?? (!request.trim() ? "Describe the application you want to deploy." : undefined)} form={REQUEST_FORM_ID} type="submit" variant="primary">{label}</Button>
        {intent && !loading && request.trim() && !props.disabled && <span className="apps-wizard__footer-note">{Math.round(intent.confidence * 100)}% understood</span>}
      </span>
    );
  } else if (shownPhase === "research" && intent) {
    body = (
      <ResearchStep
        advancedRecovery={advancedRecovery}
        clarifying={clarifying}
        escalation={escalation}
        multiService={config.multiService}
        needsInput={needsInput}
        onCancel={research && props.onCancelResearch ? () => void run(() => props.onCancelResearch!(research.id), (result) => { setResearch(result); props.onResearchChange?.(result); }) : undefined}
        onRetry={props.onRetryResearch ? retryResearch : undefined}
        research={research}
        researchRecovery={props.researchRecovery}
        retrying={loading}
        setSourcePage={setSourcePage}
        sourcePage={sourcePage}
        starting={researchStarting}
      />
    );
    if (research?.status === "complete") {
      const reason = research.compatibility === "unsupported"
        ? "Research did not confirm a compatible image."
        : !research.sources.length ? "Research returned no attributable sources." : undefined;
      footer = <Button disabledReason={reason} onClick={() => holdPhase("configure")} variant="primary">Configure deployment</Button>;
    }
  } else if (shownPhase === "configure" && research?.status === "complete") {
    if (configurable) {
      body = (
        <section aria-labelledby="application-configure-heading" className="apps-wizard__section">
          <StepHeading id="application-configure-heading" pill={<StatusPill label="Secrets by reference only" tone="neutral" />} step="03 / Configure" title="Set safe appliance limits" />
          <ConfigureStep config={config} formId={CONFIGURE_FORM_ID} onSubmit={generateDraft} research={research} />
        </section>
      );
      const held = adminReason ? [adminReason] : config.blockedReasons;
      footer = (
        <Button busy={loading} disabledReason={loading || !held.length ? undefined : held.join(" ")} form={CONFIGURE_FORM_ID} type="submit" variant="primary">
          {loading ? "Building draft…" : draft ? "Regenerate validated draft" : "Generate validated Compose draft"}
        </Button>
      );
    } else if (!draft) {
      body = (
        <section aria-labelledby="application-configure-heading" className="apps-wizard__section">
          <StepHeading id="application-configure-heading" pill={<StatusPill label="Draft finalized" tone="neutral" />} step="03 / Configure" title="Configuration is locked" />
          <ConfigureLocked />
        </section>
      );
    }
  } else if (shownPhase === "review" && draft) {
    body = <ReviewStep draft={draft} />;
    const approvalReason = adminReason ?? (hasValidationErrors ? VALIDATION_ERRORS_REASON : undefined);
    footer = <>
      {props.clusteringActive && props.onDeployToCluster && <Button disabledReason={approvalReason} onClick={() => props.onDeployToCluster!(draft)}>Deploy to cluster</Button>}
      <Button disabledReason={approvalReason} onClick={() => { setError(""); setShowApproval(true); }} variant="primary">Review approval</Button>
    </>;
  } else if (shownPhase === "deploy") {
    body = <DeployStep deployment={props.deployment} />;
    const state = props.deployment?.state;
    const actions = <>
      {props.onCancelDeployment && (state === "queued" || state === "running") && <Button busy={loading} onClick={() => void run(props.onCancelDeployment!, () => undefined)}>{loading ? "Requesting cancellation…" : "Cancel deployment"}</Button>}
      {props.onRollback && state === "failed" && <Button busy={loading} onClick={() => void run(props.onRollback!, () => undefined)} variant="danger">{loading ? "Starting rollback…" : "Review and start rollback"}</Button>}
      {props.onManage && state === "healthy" && <Button onClick={props.onManage} variant="primary">Manage this application</Button>}
    </>;
    const hasAction = (props.onCancelDeployment && (state === "queued" || state === "running")) || (props.onRollback && state === "failed") || (props.onManage && state === "healthy");
    footer = hasAction ? actions : closeButton;
  }

  const researchTier = research?.status === "complete" ? research.modelTier : undefined;
  const asideAction = shownPhase === "research"
    ? <Button onClick={editRequest}>Edit request</Button>
    : shownPhase === "configure"
      ? <Button onClick={() => holdPhase("research")}>Review research</Button>
      : shownPhase === "review" && configurable
        ? <Button onClick={() => holdPhase("configure")}>Edit configuration</Button>
        : undefined;

  return (
    <AppsDialog
      className="apps-wizard"
      closeLabel="Back to Apps and AI"
      eyebrow={`Custom application · step ${currentStep} of 5`}
      footer={footer}
      onClose={props.onClose ?? (() => undefined)}
      size="wide"
      title="Deploy from a verified plan"
      titleId="application-deployment-title"
    >
      <div className="apps-wizard__chrome">
        <WizardStepBar current={currentStep} />
        <GuardedResearchLine />
      </div>
      {/* The wizard's own refusal sits under the step bar, so the step it is
          about stays named above it; the approval dialog shows its own. */}
      {error && !showApproval && <Notice severity="danger"><strong>Action needed</strong> {error}</Notice>}
      {intent && shownPhase !== "request" && shownPhase !== "deploy" && <CurrentRequest action={asideAction} intent={intent} research={research} showFacts={shownPhase === "research"} />}
      {escalationNote && <div className="apps-wizard__note" role="status" aria-live="polite"><strong>Larger model.</strong> {escalationNote}</div>}
      {/* The research-model card belongs to the Research step (the board); Configure
          and Review reach it through "Review research". */}
      {research && researchTier && shownPhase === "research" && (
        <ResearchModelPanel
          control={escalation}
          hasDraft={Boolean(draft)}
          research={research}
          strongerRecommended={Boolean(intent?.researchCapability?.strongerModelRecommended)}
        />
      )}
      {body}
      {showApproval && draft && (
        <ApprovalDialog
          busy={loading}
          draft={draft}
          error={error || undefined}
          onApprove={approve}
          onClose={() => { if (!loading) setShowApproval(false); }}
        />
      )}
    </AppsDialog>
  );
}
