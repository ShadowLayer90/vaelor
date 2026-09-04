import { FormEvent, type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import "../styles/application-deployment.css";
import { ModalShell } from "./ModalShell";
import { Button, Checkbox, Input, Select, Textarea } from "./ui";

export type WorkflowPhase = "request" | "research" | "configure" | "review" | "deploy";
export type AsyncState = "idle" | "loading" | "error";

export interface ApplicationIntent {
  id: string;
  application: string;
  summary: string;
  confidence: number;
  questions?: string[];
  researchCapability?: {
    label: string;
    summary: string;
    limitations: string[];
    strongerModelRecommended: boolean;
    recommendation?: string | null;
  };
  refinement?: { source: string; modelUsed: boolean };
}

export interface ResearchSource {
  id: string;
  title: string;
  url: string;
  publisher: string;
  retrievedAt: string;
  supports: string[];
}

export interface ImageFact {
  image: string;
  digest: string;
  architectures: string[];
  verified: boolean;
}

export interface ResearchReport {
  id: string;
  status: "queued" | "running" | "complete" | "failed" | "needs_input";
  phase?: string;
  blockerLayer?: string;
  progress?: number;
  compatibility: "pending" | "compatible" | "conditional" | "unsupported";
  compatibilitySummary: string;
  images: ImageFact[];
  ports: Array<{ container: number; protocol: "tcp" | "udp"; purpose: string }>;
  volumes: Array<{ target: string; purpose: string; required: boolean }>;
  environment: Array<{ name: string; description: string; secret: boolean; required: boolean; defaultValue?: string }>;
  license?: string;
  minimumMemoryMb?: number;
  minimumStorageGb?: number;
  sources: ResearchSource[];
  /**
   * Distinct Compose service names the verified manifest pins an image for.
   * Operator-added published ports and privileged host mounts must name one of
   * these (the backend refuses a service it did not pin an image for), so the
   * Configure step drives its service pickers from this list.
   */
  services?: string[];
  error?: string;
}

/**
 * The only host paths the backend will bind-mount into an application, mirrored
 * from `PRIVILEGED_HOST_MOUNTS` in `vaelor/compose_policy.py`. That constant is
 * the single source of truth shared by the guided-config assembler and the
 * executor backstop; this mirror only decides what the operator may opt into in
 * the UI. Keep the two in sync — a `tests/test_compose_policy.py` pins the
 * backend surface, and the mount is refused server-side if this drifts wider.
 * Each entry is mounted read-only and only with explicit per-mount consent.
 */
export const PRIVILEGED_HOST_MOUNTS = [
  {
    source: "/var/run/docker.sock",
    readOnly: true,
    title: "Docker socket",
    // The consent label MUST make the blast radius explicit: mounting the
    // Docker socket hands the container root-equivalent control of the host.
    consentLabel:
      "Give this container read-only access to the Docker socket (/var/run/docker.sock). "
      + "This grants it root-equivalent control of this host — it can inspect, start, stop, "
      + "and replace every container on the appliance. Leave this off unless the application "
      + "genuinely manages Docker and you accept that risk.",
  },
] as const;

export type PrivilegedHostMountSource = (typeof PRIVILEGED_HOST_MOUNTS)[number]["source"];

export interface DeploymentConfiguration {
  request: string;
  intentId: string;
  researchId: string;
  name: string;
  ports: Record<string, number>;
  memoryMb: number;
  storageGb: number;
  settings: Record<string, string>;
  secretReferences: Record<string, string>;
  replaceExisting: boolean;
  /**
   * Host ports the operator chose to publish for a service the researched plan
   * left unexposed. Sent to the backend as `configuration.add_ports`; each names
   * a pinned-image service and a container target port, with an optional host
   * port (defaults to the target) and protocol (defaults to tcp).
   */
  addPorts: Array<{ service: string; target: number; published?: number; protocol: "tcp" | "udp" }>;
  /**
   * Allowlisted privileged host mounts the operator explicitly consented to.
   * Sent as `configuration.host_mounts`. Only ever populated when the operator
   * ticks the consent box, and only with an allowlisted source.
   */
  hostMounts: Array<{ service: string; source: PrivilegedHostMountSource; consent: true }>;
}

const emptyPortRow = (service: string) => ({
  id: Math.random().toString(36).slice(2),
  service,
  target: "",
  published: "",
  protocol: "tcp" as "tcp" | "udp",
});
type PortRow = ReturnType<typeof emptyPortRow>;

function portFieldValid(value: string) {
  const trimmed = value.trim();
  if (!/^\d+$/.test(trimmed)) return false;
  const port = Number(trimmed);
  return port >= 1 && port <= 65535;
}

export interface ComposeDraft {
  id: string;
  manifestDigest: string;
  redactedCompose: string;
  validation: Array<{ level: "pass" | "warning" | "error"; message: string }>;
  images: ImageFact[];
  createdAt: string;
}

export interface DeploymentProgress {
  state: "queued" | "running" | "healthy" | "failed" | "cancelled" | "rolling_back" | "rolled_back";
  progress: number;
  message: string;
  healthChecks?: Array<{ name: string; status: "pending" | "pass" | "fail"; detail?: string }>;
  openUrl?: string;
}

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
  onClose?: () => void;
  onDirtyChange?: (dirty: boolean) => void;
  onClassify: (request: string) => Promise<ApplicationIntent>;
  onStartResearch: (intent: ApplicationIntent, sourceUrls: string[]) => Promise<ResearchReport>;
  onRefreshResearch?: (researchId: string) => Promise<ResearchReport>;
  onCancelResearch?: (researchId: string) => Promise<ResearchReport>;
  onRetryResearch?: (researchId: string) => Promise<ResearchReport>;
  onReplaceActiveResearch?: () => Promise<void>;
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

const sourcePageSize = 4;

/**
 * #247q: a completed research can report "not supported" for two very different
 * reasons, and the container collapses both (`unknown` and `unsupported`) to
 * `compatibility === "unsupported"`. One is a genuine fit problem — a reviewed
 * application Vaelor will not substitute a community image for. The other, and
 * the common one for anything typed in, is that no registry source proved a
 * digest-pinned image; that is recoverable by setting up guarded web research or
 * adding an official source. `application_research_service.py` emits this exact
 * phrase for the recoverable case, so we offer recovery only when it actually
 * helps and never mislabel a real incompatibility as "set up web research". The
 * wording is asserted on both sides so drift fails a test.
 */
export const UNVERIFIED_IMAGE_MARKER = "could be verified from the available sources";

/**
 * #247x: research can stop not because the app is incompatible, but because
 * Vaelor is unsure and needs the operator to answer a concrete question — which
 * of two images is official, which edition, or the official source. The backend
 * emits that state as a research failure whose message leads with this marker
 * and joins the grounded questions with ` ||| ` (one source of truth lives in
 * `application_research_service.py`; a test asserts the phrase on both sides so
 * drift fails). This is a distinct "needs your input" state — never a fabricated
 * answer — and distinct from the #247q "no image proven" recovery above.
 */
export const CLARIFYING_QUESTIONS_MARKER = "Vaelor needs a bit more to proceed";

function clarifyingQuestions(text: string | null | undefined): { context: string; questions: string[] } | null {
  if (!text || !text.includes(CLARIFYING_QUESTIONS_MARKER)) return null;
  const segment = text.slice(text.indexOf(CLARIFYING_QUESTIONS_MARKER));
  const parts = segment.split(" ||| ").map((part) => part.trim()).filter(Boolean);
  if (parts.length < 2) return null;
  const context = parts[0].slice(CLARIFYING_QUESTIONS_MARKER.length).replace(/^[\s.:—-]+/, "").trim();
  return { context, questions: parts.slice(1) };
}

function needsImageEvidence(research: ResearchReport | null | undefined) {
  return Boolean(
    research
    && research.status === "complete"
    && research.compatibility === "unsupported"
    && research.images.length === 0
    && research.compatibilitySummary.includes(UNVERIFIED_IMAGE_MARKER),
  );
}

function shortDigest(value: string) {
  return value.length > 24 ? `${value.slice(0, 18)}…${value.slice(-8)}` : value;
}

//: The executor's Compose project name must match /^[a-z0-9][a-z0-9_-]{1,47}$/
//: - two to 48 characters. Seeding the field with the whole slugified request
//: overran that: a verbose "Deploy IT-Tools, the self-hosted ... on my LAN"
//: became a 150-character name that passed research, configure, review and
//: approval and only failed at deploy with the bare "The Compose project name
//: is invalid." A deployment name is the user's to set; the default is a short,
//: valid seed derived from the identified application, never the request.
export const MAX_DEPLOYMENT_NAME = 48;
export function defaultDeploymentName(application: string | undefined): string {
  const slug = String(application ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+/, "")
    .slice(0, MAX_DEPLOYMENT_NAME)
    .replace(/-+$/, "");
  return slug.length >= 2 ? slug : "managed-app";
}

function workflowStep(phase: WorkflowPhase) {
  return ["request", "research", "configure", "review", "deploy"].indexOf(phase) + 1;
}

const researchPhaseLabels: Record<string, string> = {
  queued: "Waiting to start",
  interpreting: "Understanding your request",
  searching: "Finding trustworthy sources",
  acquiring: "Retrieving public evidence",
  synthesizing: "Comparing the evidence",
  validating: "Checking compatibility and safety",
  needs_input: "Waiting for your decision",
  ready_for_review: "Ready for your review",
};

const researchBlockers: Record<string, { title: string; recovery: string }> = {
  // #247g: recovery advice must point only at controls that exist here. "Choose
  // a stronger model" was unactionable (one local model is installed and it is
  // already the top one) and linked nowhere, so it is replaced with the real
  // recovery paths on this screen: a clearer name, the guarded web-research
  // setup shown below, and adding official sources under Advanced recovery.
  interpretation: { title: "The selected model could not understand the request reliably", recovery: "Try a clearer or more exact application name, then retry. You can also add the official documentation URL under Advanced recovery below. No app facts were guessed." },
  discovery: { title: "Vaelor could not find trustworthy sources", recovery: "Set up guarded web research below, or add the official documentation URL under Advanced recovery, then retry." },
  acquisition: { title: "Vaelor could not retrieve enough public evidence", recovery: "Retry when the source is reachable, or add another official source under Advanced recovery." },
  synthesis: { title: "The selected model could not compare the captured evidence", recovery: "Retry, or add the official documentation URL under Advanced recovery so Vaelor has clearer evidence to compare. The downloaded evidence remains isolated and no app was installed." },
  compatibility: { title: "This application does not fit the current node", recovery: "Review the compatibility evidence or choose another node. Vaelor did not partially install it." },
  policy: { title: "The request did not pass Vaelor's safety policy", recovery: "Correct the highlighted request or configuration, then retry." },
  authorization: { title: "The exact change still needs approval", recovery: "Review the prepared plan and approve it when you are ready." },
  execution: { title: "The approved installation did not finish", recovery: "Review the failed step and logs before retrying or rolling back." },
  verification: { title: "Installation finished but the app is not healthy yet", recovery: "Review health checks, ports, and logs before deciding whether to retry or roll back." },
};

function workflowPhaseFor(research: ResearchReport | null | undefined, draft: ComposeDraft | null | undefined): WorkflowPhase {
  if (draft) return "review";
  if (!research) return "request";
  if (research.status === "complete") return research.compatibility === "unsupported" ? "research" : "configure";
  return "research";
}

function researchNeedsInput(research: ResearchReport | null | undefined) {
  return research?.status === "needs_input" || research?.phase === "needs_input";
}

function researchCanPoll(research: ResearchReport | null | undefined) {
  return Boolean(research && (research.status === "queued" || research.status === "running") && research.phase !== "needs_input" && research.phase !== "ready_for_review");
}

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
  const [name, setName] = useState("");
  const [memoryMb, setMemoryMb] = useState(1024);
  const [storageGb, setStorageGb] = useState(5);
  const [ports, setPorts] = useState<Record<string, number>>({});
  const [settings, setSettings] = useState<Record<string, string>>({});
  const [secretReferences, setSecretReferences] = useState<Record<string, string>>({});
  const [secretValues, setSecretValues] = useState<Record<string, string>>({});
  const [replaceExisting, setReplaceExisting] = useState(false);
  const [addPorts, setAddPorts] = useState<PortRow[]>([]);
  const [dockerSocketConsent, setDockerSocketConsent] = useState(false);
  const [dockerSocketService, setDockerSocketService] = useState("");
  const [showApproval, setShowApproval] = useState(false);
  const [configurationDirty, setConfigurationDirty] = useState(false);
  const approvalCloseRef = useRef<HTMLButtonElement>(null);
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
  // The editable field is the reader's own words - `initialRequest` or what they
  // typed - and stays that until they submit again.
  useEffect(() => { if (props.intent !== undefined) { setIntent(props.intent); } }, [props.intent]);
  useEffect(() => { if (props.research !== undefined) { setResearch(props.research); if (!readerChosePhase.current) setPhase(workflowPhaseFor(props.research, props.draft ?? draft)); } }, [props.draft, props.research, draft]);
  useEffect(() => { if (props.draft !== undefined) { setDraft(props.draft); if (!readerChosePhase.current) setPhase(workflowPhaseFor(props.research ?? research, props.draft)); } }, [props.draft, props.research, research]);
  useEffect(() => {
    props.onDirtyChange?.(configurationDirty);
    return () => props.onDirtyChange?.(false);
  }, [configurationDirty, props.onDirtyChange]);

  useEffect(() => {
    if (!research || research.status !== "complete") return;
    setName((current) => current || defaultDeploymentName(intent?.application));
    setMemoryMb((current) => Math.max(current, research.minimumMemoryMb ?? 0));
    setStorageGb((current) => Math.max(current, research.minimumStorageGb ?? 0));
    setPorts((current) => Object.keys(current).length ? current : Object.fromEntries(research.ports.map((port) => [`${port.container}/${port.protocol}`, port.container])));
    setSettings((current) => Object.keys(current).length ? current : Object.fromEntries(research.environment.filter((item) => !item.secret).map((item) => [item.name, item.defaultValue ?? ""])));
    setSecretReferences((current) => Object.keys(current).length ? current : Object.fromEntries(research.environment.filter((item) => item.secret).map((item) => [item.name, ""])));
    // When the plan pins exactly one service, the advanced port and host-mount
    // pickers have only one possible target, so seed it rather than force a
    // redundant choice. Multi-service plans leave the operator to pick.
    const services = (research.services ?? []).filter(Boolean);
    if (services.length === 1) setDockerSocketService((current) => current || services[0]);
  }, [intent, research]);

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
        setResearch(result);
        props.onResearchChange?.(result);
        if (result.status === "complete") setPhase(result.compatibility === "unsupported" ? "research" : "configure");
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
   * app-name summary (#247v). Re-researching a bare summary phrase failed the
   * intent gate; the submitted text always passes it. If no genuine original was
   * carried (a durable resume surfaces only the summary), the current field is
   * left untouched rather than blanked.
   */
  const editRequest = () => {
    if (submittedRequestRef.current.trim()) setRequest(submittedRequestRef.current);
    holdPhase("request");
  };
  const currentStep = workflowStep(phase);
  // A finalized draft (its validated compose already generated) can no longer be
  // configured; the Configure step is shown read-only rather than as an editable
  // form whose "Generate" only errors. Defaults to configurable for standalone
  // use where no server draft backs the wizard.
  const configurable = props.configurable !== false;
  const sourcePages = Math.max(1, Math.ceil((research?.sources.length ?? 0) / sourcePageSize));
  const visibleSources = research?.sources.slice(sourcePage * sourcePageSize, (sourcePage + 1) * sourcePageSize) ?? [];
  const hasValidationErrors = draft?.validation.some((item) => item.level === "error") ?? false;
  const requiredSecretsMissing = research?.environment.some((item) => item.secret && item.required && !secretValues[item.name]?.trim() && !secretReferences[item.name]?.trim()) ?? false;
  const availableServices = useMemo(
    () => (research?.services ?? []).filter((value): value is string => Boolean(value)),
    [research?.services],
  );
  // Each operator-added port must name a pinned-image service and a valid
  // container port; a host port and protocol are optional. Rows with any of
  // these unmet block the draft rather than sending the backend a value it will
  // reject, and the offending field carries the message inline.
  const portRowErrors = addPorts.map((row) => ({
    service: !row.service,
    target: !portFieldValid(row.target),
    published: row.published.trim() !== "" && !portFieldValid(row.published),
  }));
  const portRowsInvalid = portRowErrors.some((issue) => issue.service || issue.target || issue.published);
  // Consent without a service to attach the mount to cannot be sent.
  const consentMissingService = dockerSocketConsent && !dockerSocketService;
  const advancedConfigInvalid = portRowsInvalid || consentMissingService;
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

  function clearResearchArtifacts() {
    setResearch(null); setDraft(null); setSourcePage(0); setSourceUrlText(""); setName(""); setMemoryMb(1024); setStorageGb(5); setPorts({}); setSettings({}); setSecretReferences({}); setSecretValues({}); setReplaceExisting(false); setAddPorts([]); setDockerSocketConsent(false); setDockerSocketService(""); setShowApproval(false); setConfigurationDirty(false); setPhase(intent ? "research" : "request"); setError("");
  }

  async function classify(event?: FormEvent) {
    // The reader has started work again, so the workflow may resume
    // deriving the step from what the appliance reports.
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
    setName("");
    setMemoryMb(1024);
    setStorageGb(5);
    setPorts({});
    setSettings({});
    setSecretReferences({});
    setSecretValues({});
    setReplaceExisting(false);
    setAddPorts([]);
    setDockerSocketConsent(false);
    setDockerSocketService("");
    setConfigurationDirty(false);
    setPhase("request");
    autoResearchIntentRef.current = null;
    setAsyncState("loading");
    setError("");
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
    // The reader has started work again, so the workflow may resume
    // deriving the step from what the appliance reports.
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
    // The reader has started work again, so the workflow may resume
    // deriving the step from what the appliance reports.
    readerChosePhase.current = false;
    if (!research || !props.onRetryResearch) return;
    const researchId = research.id; clearResearchArtifacts();
    void run(() => props.onRetryResearch!(researchId), (result) => {
      setResearch(result); props.onResearchChange?.(result);
      if (result.status === "complete") setPhase(result.compatibility === "unsupported" ? "research" : "configure");
    });
  }

  async function prepareDraft() {
    // The reader has started work again, so the workflow may resume
    // deriving the step from what the appliance reports.
    readerChosePhase.current = false;
    if (!intent || !research) return;
    if (requiredSecretsMissing) { setError("Enter each required application secret."); return; }
    if (portRowsInvalid) { setError("Each published port needs a service and a container port between 1 and 65535."); return; }
    if (consentMissingService) { setError("Choose the service that should receive the Docker socket mount."); return; }
    const enteredSecrets = Object.entries(secretValues).filter(([, value]) => value.trim());
    if (enteredSecrets.length && !props.onStoreSecret) { setError("Secure application-secret storage is unavailable."); return; }
    setAsyncState("loading");
    setError("");
    try {
      const stored = await Promise.all(enteredSecrets.map(async ([secretName, value]) => [secretName, await props.onStoreSecret!(secretName, value)] as const));
      const nextReferences = { ...secretReferences, ...Object.fromEntries(stored) };
      setSecretReferences(nextReferences);
      setSecretValues({});
      const configuredAddPorts = addPorts.map((row) => ({
        service: row.service,
        target: Number(row.target),
        ...(row.published.trim() ? { published: Number(row.published) } : {}),
        protocol: row.protocol,
      }));
      const configuredHostMounts: DeploymentConfiguration["hostMounts"] =
        dockerSocketConsent && dockerSocketService
          ? [{ service: dockerSocketService, source: "/var/run/docker.sock", consent: true }]
          : [];
      const configuration: DeploymentConfiguration = { request, intentId: intent.id, researchId: research.id, name, ports, memoryMb, storageGb, settings, secretReferences: nextReferences, replaceExisting, addPorts: configuredAddPorts, hostMounts: configuredHostMounts };
      const result = await props.onGenerateDraft(configuration);
      setDraft(result); props.onDraftChange?.(result); setConfigurationDirty(false); setPhase("review");
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

  const compatibilityLabel = research?.compatibility === "pending" ? "Checking" : research?.compatibility === "compatible" ? "Compatible" : research?.compatibility === "conditional" ? "Compatible with conditions" : "Not supported";
  const researchStarting = phase === "research" && asyncState === "loading" && !research;
  const researchWorking = researchStarting || runningResearch;
  const needsInput = researchNeedsInput(research);
  const clarifying = clarifyingQuestions(research?.error);
  const advancedSourceRecovery = (
    <details className="application-deployment__advanced-sources">
      <summary>Advanced recovery: add official sources</summary>
      <p>Most people do not need this. Use it only when automatic research could not identify the application or its official documentation.</p>
      <Textarea aria-invalid={invalidSourceUrls || undefined} id="application-source-urls" label={<>Official documentation URLs <small>Enter up to eight public HTTPS pages, one per line. Do not enter passwords, tokens, or private links.</small></>} onChange={(event) => setSourceUrlText(event.target.value)} placeholder="https://project.example/docs/install" rows={4} value={sourceUrlText} />
      <div className="application-deployment__research-actions"><span>{sourceUrls.length} / 8 sources</span><Button type="button" onClick={startResearch} disabled={props.disabled || asyncState === "loading" || invalidSourceUrls || sourceUrls.length === 0}>{asyncState === "loading" ? "Checking safely…" : "Retry with these sources"}</Button></div>
    </details>
  );

  return (
    <section className="application-deployment" aria-labelledby="application-deployment-title">
      <header className="application-deployment__header">
        <div>
          <p className="application-deployment__eyebrow">Applications / researched deployment</p>
          <h1 id="application-deployment-title">Deploy from a verified plan</h1>
          <p>Vaelor researches public documentation, checks this appliance, and prepares an immutable Compose draft before anything runs.</p>
        </div>
        <div className="application-deployment__header-actions">
          {props.onClose && <Button type="button" onClick={props.onClose}>Back to Workloads</Button>}
          <ol className="application-deployment__steps" aria-label="Deployment progress">
            {["Request", "Research", "Configure", "Review", "Deploy"].map((label, index) => <li key={label} aria-current={index + 1 === currentStep ? "step" : undefined} data-complete={index + 1 < currentStep}>{index + 1}<span>{label}</span></li>)}
          </ol>
        </div>
      </header>

      <div className="application-deployment__notice" role="status" aria-live="polite">
        <strong>Guarded internet research</strong>

        <span>Public HTTPS sources are evidence only. Vaelor never gives the model direct network, shell, Docker, or credential access.</span>
      </div>
      {intent && phase !== "request" && (
        <aside className="application-deployment__current" aria-label="Current deployment request">
          <div>
            <small>Current request</small>
            <strong>{intent.application}</strong>
            <span>{research?.status === "complete" ? research.compatibilitySummary : intent.summary}</span>
          </div>
          {research?.status === "complete" && (
            <dl>
              <div><dt>License</dt><dd>{research.license || "Not stated"}</dd></div>
              <div><dt>Architecture</dt><dd>{research.images.flatMap((image) => image.architectures).filter((value, index, values) => values.indexOf(value) === index).join(", ") || "Not stated"}</dd></div>
              <div><dt>Evidence</dt><dd>{research.sources.length} sources</dd></div>
            </dl>
          )}
          {phase === "research" && <Button type="button" onClick={editRequest}>Edit request</Button>}
          {phase === "configure" && <Button type="button" onClick={() => holdPhase("research")}>Review research</Button>}
          {phase === "review" && configurable && <Button type="button" onClick={() => holdPhase("configure")}>Edit configuration</Button>}
        </aside>
      )}
      {error && <div className="application-deployment__error" role="alert"><strong>Action needed</strong><span>{error}</span></div>}

      {phase === "request" && <section className="application-deployment__panel" aria-labelledby="application-request-heading">
        <div className="application-deployment__section-heading"><div><span>01 / Request</span><h2 id="application-request-heading">What should Vaelor deploy?</h2></div>{intent && <span className="application-deployment__status">{Math.round(intent.confidence * 100)}% understood</span>}</div>
        <form onSubmit={classify} className="application-deployment__request-form">
          <Textarea disabled={props.disabled || asyncState === "loading"} id="application-request" label="Describe the application and how you expect to use it" onChange={(event) => setRequest(event.target.value)} placeholder="Example: Deploy a private Home Assistant server with its data on NVMe and expose port 8123 on my LAN." rows={4} value={request} />
          <Button type="submit" variant="primary" disabled={props.disabled || asyncState === "loading" || !request.trim()}>{asyncState === "loading" ? (phase === "request" ? "Understanding request…" : "Researching automatically…") : intent ? "Research again" : "Research and prepare plan"}</Button>
        </form>
        {intent && <div className="application-deployment__intent"><div><small>Detected application{intent.refinement?.modelUsed ? " · interpreted by selected model" : " · built-in parser"}</small><strong>{intent.application}</strong><p>{intent.summary}</p></div><span>{researchWorking ? "Research started automatically" : research ? "Request researched" : "Preparing guarded research"}</span>{intent.researchCapability && <div><small>Research intelligence</small><strong>{intent.researchCapability.label}</strong><p>{intent.researchCapability.summary}</p>{intent.researchCapability.strongerModelRecommended && intent.researchCapability.recommendation ? <p>{intent.researchCapability.recommendation}</p> : null}<details><summary>Current limitations</summary><ul>{intent.researchCapability.limitations.map((item) => <li key={item}>{item}</li>)}</ul></details></div>}</div>}
        {asyncState === "error" && error && (
          // #247b: an intent-stage failure (a 422 "not detected" or a 502
          // workflow error) used to leave only the generic banner, while the
          // research stage got a rich recovery panel. Give the request stage an
          // equivalent affordance so the reader can edit and retry instead of
          // hitting a wall.
          <div className="application-deployment__recovery" role="group" aria-label="Request recovery">
            <p><strong>What to do next:</strong> Edit your request above to be more specific — for example, name the exact application or paste its official website — then try again. Vaelor did not create or change anything.</p>
            <Button type="button" onClick={() => void classify()} disabled={props.disabled || !request.trim()}>Try again</Button>
          </div>
        )}
      </section>}
      {props.deployment?.openUrl && <a href={props.deployment.openUrl} target="_blank" rel="noreferrer">Open verified endpoint</a>}

      {intent && phase === "research" && <section className="application-deployment__panel" aria-labelledby="application-research-heading">
        <div className="application-deployment__section-heading"><div><span>02 / Research</span><h2 id="application-research-heading">Verify compatibility and sources</h2></div>{research && <span className={`application-deployment__compatibility application-deployment__compatibility--${research.compatibility}`}>{compatibilityLabel}</span>}</div>
        {!research && <div className="application-deployment__research-start"><div className="application-deployment__empty" role="status" aria-live="polite"><h3>{researchStarting ? "Researching in the background" : "Automatic research needs help"}</h3><p>{researchStarting ? "Vaelor is finding trustworthy documentation and checking whether the application can run safely on this device." : "Vaelor could not identify enough reliable information automatically. You can retry the request, or use the advanced recovery option."}</p>{researchStarting && <ul className="application-deployment__research-checks" aria-label="Research checks in progress"><li>Official documentation and current installation guidance</li><li>Container images and ARM64 support</li><li>Memory, storage, network ports, and persistent data</li><li>Safe defaults, health checks, and recovery requirements</li></ul>}</div>{!researchStarting && advancedSourceRecovery}</div>}
        {research && research.status !== "complete" && (
          <div className="application-deployment__research-state">
            <div className="application-deployment__empty" role="status" aria-live="polite">
              <h3>{research.status === "failed" ? (researchBlockers[research.blockerLayer || ""]?.title || "Automatic research could not be completed") : (researchPhaseLabels[research.phase || ""] || "Research is continuing automatically")}</h3>
              <p>{research.error || "Vaelor is checking trusted sources, container support, device fit, networking, storage, and safe operating defaults. You may close this window; the operation is saved and resumes when you return."}</p>
              {runningResearch && <div className="application-deployment__research-progress"><div><span>{researchPhaseLabels[research.phase || ""] || "Working"}</span><strong>{research.progress ?? 0}%</strong></div><progress aria-label="Research progress" value={research.progress ?? 0} max="100">{research.progress ?? 0}%</progress><small>You can close this window. Vaelor saves this operation and resumes here when you return.</small></div>}
              {runningResearch && props.onCancelResearch && <Button type="button" onClick={() => void run(() => props.onCancelResearch!(research.id), (result) => { setResearch(result); props.onResearchChange?.(result); })} disabled={asyncState === "loading"}>Cancel research</Button>}
              {(research.status === "failed" || needsInput) && props.onRetryResearch && (
                <Button type="button" onClick={retryResearch} disabled={asyncState === "loading"}>
                  {asyncState === "loading" ? "Retrying…" : "Retry automatic research"}
                </Button>
              )}
            </div>
            {(research.status === "failed" || needsInput) && (
              <div className="application-deployment__recovery">
                {clarifying ? (
                  // #247x: a distinct "needs your input" block — concrete,
                  // grounded questions the operator answers, not a dead-end and
                  // not a guessed answer. Edit request (in the header aside) and
                  // Advanced recovery remain the ways to respond.
                  <div className="application-deployment__clarify" role="group" aria-label="Vaelor needs more information">
                    <strong>Vaelor needs a bit more to proceed</strong>
                    {clarifying.context && <p>{clarifying.context}</p>}
                    <ol className="application-deployment__clarify-questions">
                      {clarifying.questions.map((question) => <li key={question}>{question}</li>)}
                    </ol>
                    <p>Answer above with “Edit request”, or add the official source under Advanced recovery, then research again.</p>
                  </div>
                ) : (
                  research.blockerLayer && researchBlockers[research.blockerLayer] && <p><strong>{needsInput ? "Research is waiting for your input:" : "What to do next:"}</strong> {researchBlockers[research.blockerLayer].recovery}</p>
                )}
                {props.researchRecovery}
                {(needsInput || !research.blockerLayer || ["interpretation", "discovery", "acquisition", "synthesis"].includes(research.blockerLayer)) && advancedSourceRecovery}
              </div>
            )}
          </div>
        )}
        {research?.status === "complete" && <>
          <p className="application-deployment__compatibility-copy">{research.compatibilitySummary}</p>
          {research.compatibility === "unsupported" && (needsImageEvidence(research) ? (
            // The verdict failed only because no official container image could
            // be verified from the sources Vaelor had — not because the app does
            // not fit. Point at the real, actionable recovery on this screen.
            <div className="application-deployment__unsupported" role="group" aria-label="Verified container image needed">
              <strong>Vaelor stopped before making changes</strong>
              <p>No official container image could be verified from the available sources, so there is nothing safe to deploy yet. Set up guarded web research so Vaelor can find the official image, or add the official image source under Advanced recovery. Nothing was downloaded or installed.</p>
              {props.researchRecovery}
              {advancedSourceRecovery}
            </div>
          ) : (
            <div className="application-deployment__unsupported" role="alert"><strong>Vaelor stopped before making changes</strong><p>Configuration is unavailable because Vaelor could not confirm this application fits this appliance. The compatibility finding above explains why. Nothing was downloaded or installed.</p></div>
          ))}
          <div className="application-deployment__fact-grid">
            <article><small>License</small><strong>{research.license || "Not stated"}</strong></article>
            <article><small>Minimum memory</small><strong>{research.minimumMemoryMb ? `${research.minimumMemoryMb} MB` : "Not stated"}</strong></article>
            <article><small>Minimum storage</small><strong>{research.minimumStorageGb ? `${research.minimumStorageGb} GB` : "Not stated"}</strong></article>
            <article><small>Evidence</small><strong>{research.sources.length} cited sources</strong></article>
          </div>
          <h3>Verified container images</h3>
          <div className="application-deployment__table-wrap"><table><thead><tr><th>Image</th><th>Immutable digest</th><th>Architectures</th><th>Verification</th></tr></thead><tbody>{research.images.map((image) => <tr key={`${image.image}${image.digest}`}><td>{image.image}</td><td><code title={image.digest}>{shortDigest(image.digest)}</code></td><td>{image.architectures.join(", ")}</td><td>{image.verified ? "Verified" : "Unverified"}</td></tr>)}</tbody></table></div>
          <div className="application-deployment__sources"><div className="application-deployment__sources-heading"><h3>Sources and supported claims</h3><span>Page {sourcePage + 1} of {sourcePages}</span></div>{visibleSources.map((source) => <article key={source.id}><div><a href={source.url} target="_blank" rel="noreferrer">{source.title}</a><small>{source.publisher} · retrieved {new Date(source.retrievedAt).toLocaleDateString()}</small></div><ul>{source.supports.map((claim) => <li key={claim}>{claim}</li>)}</ul></article>)}{research.sources.length === 0 && <p>No attributable sources were returned. Deployment cannot proceed.</p>}<nav aria-label="Research source pages"><Button type="button" onClick={() => setSourcePage((page) => Math.max(0, page - 1))} disabled={sourcePage === 0}>Previous</Button><Button type="button" onClick={() => setSourcePage((page) => Math.min(sourcePages - 1, page + 1))} disabled={sourcePage + 1 >= sourcePages}>Next</Button></nav></div>
          {phase === "research" && <Button type="button" variant="primary" onClick={() => holdPhase("configure")} disabled={research.compatibility === "unsupported" || !research.sources.length}>Configure deployment</Button>}
        </>}
      </section>}

      {research?.status === "complete" && phase !== "request" && phase !== "research" && !configurable && !draft && <section className="application-deployment__panel" aria-labelledby="application-configure-heading">
        <div className="application-deployment__section-heading"><div><span>03 / Configure</span><h2 id="application-configure-heading">Configuration is locked</h2></div><span className="application-deployment__status">Draft finalized</span></div>
        <div className="application-deployment__unsupported" role="status"><strong>This plan has already been configured</strong><p>Its immutable Compose draft was generated, so its configuration can no longer be edited. Review and approve it, or discard the saved research to configure this application again from scratch.</p></div>
      </section>}

      {research?.status === "complete" && phase !== "request" && phase !== "research" && configurable && <section className="application-deployment__panel" aria-labelledby="application-configure-heading">
        <div className="application-deployment__section-heading"><div><span>03 / Configure</span><h2 id="application-configure-heading">Set safe appliance limits</h2></div><span className="application-deployment__status">Secrets by reference only</span></div>
        <form className="application-deployment__configuration" onSubmit={generateDraft}>
          <fieldset><legend>Identity and resources</legend><Input id="application-name" label="Deployment name" maxLength={MAX_DEPLOYMENT_NAME} onChange={(event) => { setName(event.target.value); setConfigurationDirty(true); }} pattern="[a-z0-9][a-z0-9-]{1,47}" required value={name} /><Input id="application-memory" label="Memory limit (MB)" min={research.minimumMemoryMb ?? 128} onChange={(event) => { setMemoryMb(event.currentTarget.valueAsNumber); setConfigurationDirty(true); }} required step="128" type="number" value={memoryMb} /><Input id="application-storage" label="Reserved storage (GB)" min={research.minimumStorageGb ?? 1} onChange={(event) => { setStorageGb(event.currentTarget.valueAsNumber); setConfigurationDirty(true); }} required type="number" value={storageGb} /></fieldset>
          {!!research.ports.length && <fieldset><legend>Network ports</legend>{research.ports.map((port) => { const key = `${port.container}/${port.protocol}`; return <Input id={`application-port-${key.replaceAll("/", "-")}`} key={key} label={port.purpose + " - container " + key} max="65535" min="1" onChange={(event) => { setPorts((current) => ({ ...current, [key]: event.currentTarget.valueAsNumber })); setConfigurationDirty(true); }} required type="number" value={ports[key] ?? port.container} />; })}</fieldset>}
          {!!research.environment.filter((item) => !item.secret).length && <fieldset><legend>Application settings</legend>{research.environment.filter((item) => !item.secret).map((item) => <Input id={`application-setting-${item.name}`} key={item.name} label={<> {item.name}<small>{item.description}</small></>} onChange={(event) => { setSettings((current) => ({ ...current, [item.name]: event.target.value })); setConfigurationDirty(true); }} required={item.required} value={settings[item.name] ?? ""} />)}</fieldset>}
          {!!research.environment.filter((item) => item.secret).length && <fieldset><legend>Application secrets</legend><p>Enter each value once. Vaelor stores it immediately in the encrypted broker; only a purpose-bound reference enters the deployment draft.</p>{research.environment.filter((item) => item.secret).map((item) => <Input autoComplete="new-password" id={`application-secret-${item.name}`} key={item.name} label={<> {item.name}<small>{item.description}{secretReferences[item.name] ? " - Stored securely; enter a new value only to replace it." : ""}</small></>} onChange={(event) => { setSecretValues((current) => ({ ...current, [item.name]: event.target.value })); setConfigurationDirty(true); }} placeholder={secretReferences[item.name] ? "Stored securely" : "Enter secret value"} required={item.required && !secretReferences[item.name]} type="password" value={secretValues[item.name] ?? ""} />)}</fieldset>}
          {availableServices.length > 0 && <fieldset className="application-deployment__advanced-config"><legend>Publish an extra port (advanced)</legend><p>Most deployments need nothing here. Use it to reach a service the researched plan left with no published port. Vaelor only pins a host port onto the already-verified image.</p>{addPorts.map((row, index) => <div className="application-deployment__advanced-row" key={row.id}>
            <Select id={`application-add-port-service-${row.id}`} label="Service" error={portRowErrors[index]?.service ? "Choose a service." : undefined} value={row.service} onChange={(event) => { const value = event.target.value; setAddPorts((current) => current.map((item) => item.id === row.id ? { ...item, service: value } : item)); setConfigurationDirty(true); }}><option value="">Choose a service</option>{availableServices.map((service) => <option key={service} value={service}>{service}</option>)}</Select>
            <Input id={`application-add-port-target-${row.id}`} label="Container port" error={portRowErrors[index]?.target ? "1 to 65535." : undefined} inputMode="numeric" max="65535" min="1" type="number" value={row.target} onChange={(event) => { const value = event.target.value; setAddPorts((current) => current.map((item) => item.id === row.id ? { ...item, target: value } : item)); setConfigurationDirty(true); }} />
            <Input id={`application-add-port-published-${row.id}`} label={<>Host port <small>Optional; defaults to the container port.</small></>} error={portRowErrors[index]?.published ? "1 to 65535." : undefined} inputMode="numeric" max="65535" min="1" type="number" value={row.published} onChange={(event) => { const value = event.target.value; setAddPorts((current) => current.map((item) => item.id === row.id ? { ...item, published: value } : item)); setConfigurationDirty(true); }} />
            <Select id={`application-add-port-protocol-${row.id}`} label="Protocol" value={row.protocol} onChange={(event) => { const value = event.target.value === "udp" ? "udp" : "tcp"; setAddPorts((current) => current.map((item) => item.id === row.id ? { ...item, protocol: value } : item)); setConfigurationDirty(true); }}><option value="tcp">TCP</option><option value="udp">UDP</option></Select>
            <Button type="button" onClick={() => { setAddPorts((current) => current.filter((item) => item.id !== row.id)); setConfigurationDirty(true); }}>Remove</Button>
          </div>)}<Button type="button" onClick={() => { setAddPorts((current) => [...current, emptyPortRow(availableServices.length === 1 ? availableServices[0] : "")]); setConfigurationDirty(true); }}>Add published port</Button></fieldset>}
          {availableServices.length > 0 && <fieldset className="application-deployment__advanced-config application-deployment__host-access"><legend>Privileged host access (advanced)</legend><p>Off by default. Only the Docker socket is allowed, and only read-only. Nothing is mounted unless you tick the box below.</p>
            {availableServices.length > 1 && <Select id="application-host-mount-service" label="Service" error={consentMissingService ? "Choose a service." : undefined} value={dockerSocketService} onChange={(event) => { setDockerSocketService(event.target.value); setConfigurationDirty(true); }}><option value="">Choose a service</option>{availableServices.map((service) => <option key={service} value={service}>{service}</option>)}</Select>}
            <Checkbox checked={dockerSocketConsent} id="application-host-mount-docker" label={<>{PRIVILEGED_HOST_MOUNTS[0].title} <small>{PRIVILEGED_HOST_MOUNTS[0].consentLabel}</small></>} onChange={(event) => { setDockerSocketConsent(event.target.checked); setConfigurationDirty(true); }} /></fieldset>}
          <div className="application-deployment__config-summary"><div><strong>{research.volumes.length}</strong><span>managed volumes</span></div><div><strong>{research.ports.length + addPorts.length}</strong><span>published ports</span></div><div><strong>{research.environment.length}</strong><span>configuration keys</span></div></div>
          <div className="application-deployment__replace"><Checkbox checked={replaceExisting} id="application-replace-existing" label={<>Allow replacement if this application is already managed <small>The current configuration is backed up first. Leave this off for a new installation.</small></>} onChange={(event) => { setReplaceExisting(event.target.checked); setConfigurationDirty(true); }} /></div>
          <Button type="submit" variant="primary" disabled={props.disabled || asyncState === "loading" || requiredSecretsMissing || advancedConfigInvalid}>{asyncState === "loading" ? "Building draft…" : draft ? "Regenerate validated draft" : "Generate validated Compose draft"}</Button>
        </form>
      </section>}

      {draft && (phase === "review" || phase === "deploy") && <section className="application-deployment__panel" aria-labelledby="application-review-heading">
        <div className="application-deployment__section-heading"><div><span>04 / Review</span><h2 id="application-review-heading">Inspect the immutable deployment</h2></div><code title={draft.manifestDigest}>{shortDigest(draft.manifestDigest)}</code></div>
        <div className="application-deployment__review-grid"><div><h3>Redacted Compose</h3><pre aria-label="Redacted Compose draft"><code>{draft.redactedCompose}</code></pre><p>Credential values are injected by the broker at execution time and cannot appear in this preview.</p></div><div><h3>Validation</h3><ul className="application-deployment__validation">{draft.validation.map((item, index) => <li key={`${item.message}${index}`} data-level={item.level}><strong>{item.level}</strong><span>{item.message}</span></li>)}</ul><dl><div><dt>Draft</dt><dd>{draft.id}</dd></div><div><dt>Created</dt><dd>{new Date(draft.createdAt).toLocaleString()}</dd></div><div><dt>Images</dt><dd>{draft.images.length} digest-pinned</dd></div></dl><Button type="button" variant="primary" onClick={() => setShowApproval(true)} disabled={props.disabled || hasValidationErrors || phase === "deploy"}>Review approval</Button>{hasValidationErrors && <p role="alert">Resolve validation errors before approval.</p>}</div></div>
      </section>}

      {(phase === "deploy" || props.deployment) && <section className="application-deployment__panel" aria-labelledby="application-progress-heading"><div className="application-deployment__section-heading"><div><span>05 / Deploy</span><h2 id="application-progress-heading">Deployment and health</h2></div><span className="application-deployment__status">{props.deployment?.state.replaceAll("_", " ") ?? "Queued"}</span></div><progress aria-label="Deployment progress" value={props.deployment?.progress ?? 0} max="100">{props.deployment?.progress ?? 0}%</progress><p role="status">{props.deployment?.message ?? "The approved deployment is waiting for an executor."}</p>{props.deployment?.healthChecks?.length ? <ul className="application-deployment__health">{props.deployment.healthChecks.map((check) => <li key={check.name} data-status={check.status}><strong>{check.name}</strong><span>{check.detail || check.status}</span></li>)}</ul> : null}<div className="application-deployment__progress-actions">{props.onCancelDeployment && (props.deployment?.state === "queued" || props.deployment?.state === "running") && <Button type="button" onClick={() => void run(props.onCancelDeployment!, () => undefined)} disabled={asyncState === "loading"}>{asyncState === "loading" ? "Requesting cancellation…" : "Cancel deployment"}</Button>}{props.onManage && props.deployment?.state === "healthy" && <Button type="button" onClick={props.onManage}>Manage this application</Button>}{props.onRollback && props.deployment?.state === "failed" && <Button type="button" variant="danger" onClick={() => void run(props.onRollback!, () => undefined)} disabled={asyncState === "loading"}>{asyncState === "loading" ? "Starting rollback…" : "Review and start rollback"}</Button>}</div></section>}

      {showApproval && draft && <ModalShell
        className="application-deployment__approval"
        describedBy="deployment-approval-description"
        initialFocusRef={approvalCloseRef}
        labelledBy="deployment-approval-heading"
        onClose={() => { if (asyncState !== "loading") setShowApproval(false); }}
      >
        <header><div><span>Final authorization</span><h2 id="deployment-approval-heading">Approve this exact draft?</h2></div><Button ref={approvalCloseRef} type="button" variant="quiet" onClick={() => setShowApproval(false)} aria-label="Close approval dialog" disabled={asyncState === "loading"}>×</Button></header>
        <p id="deployment-approval-description">Vaelor will deploy only manifest <code>{shortDigest(draft.manifestDigest)}</code>. Any edit or research change creates a new draft and requires another approval.</p>
        <ul><li>Images are bound to reviewed digests.</li><li>Compose validation has no blocking errors.</li><li>Secrets remain in the credential broker.</li><li>Health checks and rollback are audited.</li></ul>
        <footer><Button type="button" onClick={() => setShowApproval(false)} disabled={asyncState === "loading"}>Go back</Button><Button type="button" variant="primary" onClick={approve} disabled={asyncState === "loading"}>{asyncState === "loading" ? "Queuing deployment…" : "Approve and deploy"}</Button></footer>
      </ModalShell>}
    </section>
  );
}
