/*
 * The custom-application wizard's data shapes and pure helpers (VD-200 split of
 * ApplicationDeployment.tsx, the AppsWizardResearch and AppsWizardDeploy
 * boards). Nothing here renders; ApplicationDeployment re-exports every public
 * name so callers keep importing from it.
 */

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
  /** The Compose service this image is pinned for, when the manifest names one. */
  service?: string;
}

/**
 * Which model interpreted the evidence that produced the CURRENT manifest.
 * `gpu/ai-chat` is the 27B graphics model; `npu/deployment-agent` is the 4B
 * assistant. The tier shown must always be the tier that produced the manifest
 * on screen — never a claim the graphics model was used when it was not.
 */
export type ResearchModelTier = "gpu/ai-chat" | "npu/deployment-agent";

export interface ResearchReport {
  id: string;
  status: "queued" | "running" | "complete" | "failed" | "needs_input";
  phase?: string;
  blockerLayer?: string;
  progress?: number;
  compatibility: "pending" | "compatible" | "conditional" | "unsupported";
  compatibilitySummary: string;
  images: ImageFact[];
  ports: Array<{ container: number; protocol: "tcp" | "udp"; purpose: string; service?: string }>;
  volumes: Array<{ target: string; purpose: string; required: boolean; service?: string }>;
  environment: Array<{ name: string; description: string; secret: boolean; required: boolean; defaultValue?: string; service?: string }>;
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
  /**
   * The model that produced THIS manifest (A). Absent on a still-running or
   * legacy result, in which case no tier is claimed. `escalatedToCapable` means
   * the assistant's pass queued a graphics-model follow-up whose richer result
   * will replace this manifest shortly; `capableAvailable` is whether a graphics
   * (GPU AI-Chat) lease is live right now, which gates the manual escalation.
   */
  modelTier?: ResearchModelTier;
  capableAvailable?: boolean;
  /** VD-207: why the larger model cannot be used now (AI Chat's model is hosted and needs approval). */
  capableUnavailableReason?: string;
  escalatedToCapable?: boolean;
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

export function clarifyingQuestions(text: string | null | undefined): { context: string; questions: string[] } | null {
  if (!text || !text.includes(CLARIFYING_QUESTIONS_MARKER)) return null;
  const segment = text.slice(text.indexOf(CLARIFYING_QUESTIONS_MARKER));
  const parts = segment.split(" ||| ").map((part) => part.trim()).filter(Boolean);
  if (parts.length < 2) return null;
  const context = parts[0].slice(CLARIFYING_QUESTIONS_MARKER.length).replace(/^[\s.:—-]+/, "").trim();
  return { context, questions: parts.slice(1) };
}

export function needsImageEvidence(research: ResearchReport | null | undefined) {
  return Boolean(
    research
    && research.status === "complete"
    && research.compatibility === "unsupported"
    && research.images.length === 0
    && research.compatibilitySummary.includes(UNVERIFIED_IMAGE_MARKER),
  );
}

export function shortDigest(value: string) {
  return value.length > 24 ? `${value.slice(0, 18)}…${value.slice(-8)}` : value;
}

/**
 * The per-service edit key for a researched port. Two services in one manifest
 * can expose the same container port (a web and an api both on 8080), so the key
 * carries the service to keep their edits distinct. A single-service manifest
 * (no `service`) keeps the historic `container/protocol` key, so the existing
 * configure round-trip and its tests are unchanged.
 */
export function portConfigKey(port: { container: number; protocol: string; service?: string }) {
  return port.service ? `${port.service}:${port.container}/${port.protocol}` : `${port.container}/${port.protocol}`;
}

/**
 * Group manifest rows by their Compose service, preserving first-seen order, so
 * a multi-service plan renders each service's ports/settings/secrets together
 * and legibly. Rows without a service collapse into one unlabelled group, which
 * keeps a single-service plan rendering exactly as before.
 */
export function groupByService<T extends { service?: string }>(rows: T[]): Array<[string, T[]]> {
  const groups = new Map<string, T[]>();
  for (const row of rows) {
    const key = row.service ?? "";
    const bucket = groups.get(key);
    if (bucket) bucket.push(row);
    else groups.set(key, [row]);
  }
  return [...groups.entries()];
}

/** Human tier label — the tier that produced the manifest currently on screen. */
export function modelTierLabel(tier: ResearchModelTier): string {
  return tier === "gpu/ai-chat" ? "Answered with the graphics model" : "Answered with the assistant";
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

export const WORKFLOW_STEPS = ["Request", "Research", "Configure", "Review", "Deploy"] as const;

export function workflowStep(phase: WorkflowPhase) {
  return ["request", "research", "configure", "review", "deploy"].indexOf(phase) + 1;
}

export const researchPhaseLabels: Record<string, string> = {
  queued: "Waiting to start",
  interpreting: "Understanding your request",
  searching: "Finding trustworthy sources",
  acquiring: "Retrieving public evidence",
  synthesizing: "Comparing the evidence",
  validating: "Checking compatibility and safety",
  needs_input: "Waiting for your decision",
  ready_for_review: "Ready for your review",
};

export const researchBlockers: Record<string, { title: string; recovery: string }> = {
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

export function workflowPhaseFor(research: ResearchReport | null | undefined, draft: ComposeDraft | null | undefined): WorkflowPhase {
  if (draft) return "review";
  if (!research) return "request";
  if (research.status === "complete") return research.compatibility === "unsupported" ? "research" : "configure";
  return "research";
}

export function researchNeedsInput(research: ResearchReport | null | undefined) {
  return research?.status === "needs_input" || research?.phase === "needs_input";
}

export function researchCanPoll(research: ResearchReport | null | undefined) {
  return Boolean(research && (research.status === "queued" || research.status === "running") && research.phase !== "needs_input" && research.phase !== "ready_for_review");
}

/** Unique architectures across the verified images, in first-seen order. */
export function imageArchitectures(research: ResearchReport) {
  return research.images.flatMap((image) => image.architectures).filter((value, index, values) => values.indexOf(value) === index).join(", ");
}
