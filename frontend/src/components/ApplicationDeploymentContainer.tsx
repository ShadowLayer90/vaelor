import { useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import { canonicalOperationState, jobIsReady, jobIsTerminal, jobNeedsAttention } from "../lib/jobPresentation";
import type { Session } from "../types";
import {
  ApplicationDeployment,
  type ApplicationIntent,
  type ComposeDraft,
  type DeploymentConfiguration,
  type DeploymentProgress,
  type ResearchReport,
} from "./ApplicationDeployment";
import { WebResearchSetup } from "./WebResearchSetup";

interface ServerIntent {
  application_query: string;
  confidence: "low" | "medium" | "high";
  missing_inputs?: string[];
  refinement?: {
    source: string;
    interpretation: string;
    model_used: boolean;
    follow_up_questions: string[];
  };
}

interface ServerManifest {
  application: { id: string; name: string; summary: string; license?: string };
  compatibility: { status: "verified" | "conditional" | "unsupported" | "unknown"; architectures: string[]; reason: string };
  images: Array<{ service?: string; repository: string; digest: string; architectures: string[]; source_url: string }>;
  ports: Array<{ name: string; protocol: "tcp" | "udp"; target: number; published: number; required: boolean }>;
  volumes: Array<{ name: string; mount_path: string; required: boolean }>;
  variables: Array<{ name: string; description: string; secret: boolean; required: boolean; default?: string }>;
  resources: { memory_bytes: number; storage_bytes: number; cpu_cores: number };
  sources: Array<{ url: string; title: string; kind: string; sha256: string; verified: boolean }>;
}

interface ServerDraft {
  id: string;
  state: string;
  intent: ServerIntent;
  manifest_digest: string | null;
  compose_digest: string | null;
  manifest: ServerManifest | null;
  compose: Record<string, unknown> | null;
  validation: { policy: string; checks?: string[]; warnings?: string[] } | null;
  created_at: number;
  updated_at: number;
  research_capability?: {
    selected_intelligence: { tier: string; label: string; description: string };
    summary: string;
    limitations: string[];
    follow_up_questions: string[];
    stronger_model_recommended: boolean;
    recommendation?: string | null;
  };
}

interface ServerJob {
  id: string;
  type: string;
  state: string;
  operation_state?: string;
  attention?: boolean;
  retryable?: boolean;
  readiness?: string;
  liveness?: string;
  progress: number;
  message: string;
  phase?: string;
  blocker_layer?: string;
  result?: Record<string, unknown>;
}

interface ApprovalResult {
  draft: ServerDraft;
  proposed_job: { type: string; payload: Record<string, unknown> };
}



function pendingResearch(job: ServerJob, message?: string): ResearchReport {
  const operation = canonicalOperationState(job);
  const terminalFailure = jobNeedsAttention(job) || ["rejected", "cancelled"].includes(operation);
  return {
    id: job.id,
    status: operation === "needs_approval" ? "needs_input" : jobIsTerminal(job) && !jobIsReady(job) ? "failed" : operation === "queued" ? "queued" : "running",
    phase: job.phase,
    blockerLayer: job.blocker_layer || (typeof job.result?.blocker_layer === "string" ? job.result.blocker_layer : undefined),
    progress: job.progress,
    compatibility: "pending",
    compatibilitySummary: message || job.message || "Vaelor is researching this application.",
    images: [], ports: [], volumes: [], environment: [], sources: [],
    error: operation === "needs_approval" || terminalFailure ? (message || job.message) : undefined,
  };
}

function intentFromDraft(draft: ServerDraft): ApplicationIntent {
  const confidence = draft.intent.confidence === "high" ? .9 : draft.intent.confidence === "medium" ? .65 : .4;
  return {
    id: draft.id,
    application: draft.intent.application_query,
    summary: draft.intent.refinement?.interpretation || `Vaelor identified ${draft.intent.application_query} as a container application that requires verified deployment research.`,
    confidence,
    questions: draft.intent.refinement?.follow_up_questions?.length
      ? draft.intent.refinement.follow_up_questions
      : draft.research_capability?.follow_up_questions ?? [],
    refinement: draft.intent.refinement ? {
      source: draft.intent.refinement.source,
      modelUsed: draft.intent.refinement.model_used,
    } : undefined,
    researchCapability: draft.research_capability ? {
      label: draft.research_capability.selected_intelligence.label,
      summary: draft.research_capability.summary,
      limitations: draft.research_capability.limitations,
      strongerModelRecommended: draft.research_capability.stronger_model_recommended,
      recommendation: draft.research_capability.recommendation,
    } : undefined,
  };
}

export function mapApplicationCompatibility(status: ServerManifest["compatibility"]["status"]): ResearchReport["compatibility"] {
  return status === "verified" ? "compatible" : status === "conditional" ? "conditional" : "unsupported";
}

function reportFromDraft(draft: ServerDraft): ResearchReport {
  if (!draft.manifest) throw new Error("Application research completed without a verified manifest.");
  const manifest = draft.manifest;
  const status = manifest.compatibility.status;
  return {
    id: draft.id,
    status: "complete",
    compatibility: mapApplicationCompatibility(status),
    compatibilitySummary: manifest.compatibility.reason,
    images: manifest.images.map((image) => ({
      image: image.repository,
      digest: image.digest,
      architectures: image.architectures,
      verified: Boolean(image.source_url),
    })),
    ports: manifest.ports.map((port) => ({ container: port.target, protocol: port.protocol, purpose: port.name })),
    // The pinned-image service names the operator may attach an added port or a
    // privileged host mount to. The verified manifest stores one per image.
    services: Array.from(new Set(manifest.images.map((image) => image.service).filter((value): value is string => Boolean(value)))),
    volumes: manifest.volumes.map((volume) => ({ target: volume.mount_path, purpose: volume.name, required: volume.required })),
    environment: manifest.variables.map((variable) => ({
      name: variable.name,
      description: variable.description,
      secret: variable.secret,
      required: variable.required,
      defaultValue: variable.default,
    })),
    license: manifest.application.license,
    minimumMemoryMb: Math.ceil(manifest.resources.memory_bytes / 1024 / 1024),
    minimumStorageGb: Math.max(1, Math.ceil(manifest.resources.storage_bytes / 1024 / 1024 / 1024)),
    sources: manifest.sources.map((source, index) => ({
      id: `${index}-${source.sha256}`,
      title: source.title,
      url: source.url,
      publisher: (() => { try { return new URL(source.url).hostname; } catch { return source.kind; } })(),
      retrievedAt: new Date(draft.updated_at * 1000).toISOString(),
      supports: [source.verified ? `Verified ${source.kind.replaceAll("_", " ")} evidence` : `${source.kind.replaceAll("_", " ")} evidence`],
    })),
  };
}

function redactedCompose(compose: Record<string, unknown>, manifest: ServerManifest): string {
  const secretNames = new Set(manifest.variables.filter((item) => item.secret).map((item) => item.name));
  function redact(value: unknown, key = ""): unknown {
    if (secretNames.has(key) || /credential|secret|password|token|api.?key/i.test(key)) return "<managed credential reference>";
    if (Array.isArray(value)) return value.map((item) => redact(item));
    if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([childKey, childValue]) => [childKey, redact(childValue, childKey)]));
    return value;
  }
  return JSON.stringify(redact(compose), null, 2);
}

/**
 * The immutable review draft for a restored server draft, or `null` when the
 * draft has not yet been generated into a validated Compose.
 *
 * A resumed draft that already carries a validated compose is no longer
 * configurable — `POST /configure` refuses it. Reconstructing its review draft
 * here lets a resume land on Review (or the active Deploy), never on an editable
 * Configure step whose only outcome is a "can no longer be configured" error.
 */
function reviewDraftFromServer(draft: ServerDraft): ComposeDraft | null {
  return draft.compose && draft.validation && draft.manifest && draft.manifest_digest
    ? composeDraftFromServer(draft)
    : null;
}

function composeDraftFromServer(draft: ServerDraft): ComposeDraft {
  if (!draft.compose || !draft.manifest || !draft.manifest_digest) throw new Error("The server did not return a complete immutable draft.");
  const checks = draft.validation?.checks ?? [];
  const warnings = draft.validation?.warnings ?? [];
  return {
    id: draft.id,
    manifestDigest: draft.manifest_digest,
    redactedCompose: redactedCompose(draft.compose, draft.manifest),
    validation: [
      ...checks.map((message) => ({ level: "pass" as const, message })),
      ...warnings.map((message) => ({ level: "warning" as const, message })),
    ],
    images: draft.manifest.images.map((image) => ({ image: image.repository, digest: image.digest, architectures: image.architectures, verified: true })),
    createdAt: new Date(draft.updated_at * 1000).toISOString(),
  };
}

function wait(milliseconds: number) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

export function ApplicationDeploymentContainer({ session, onJobQueued, initialRequest = "", autoClassify = false, onClose, onDirtyChange, onManaged, resumeResearch, startFresh = false }: { session: Session; onJobQueued?: (job: ServerJob) => void; initialRequest?: string; autoClassify?: boolean; onClose?: () => void; onDirtyChange?: (dirty: boolean) => void; onManaged?: () => void; resumeResearch?: { jobId: string; draftId: string; requestSummary?: string }; startFresh?: boolean }) {
  const [serverDraft, setServerDraft] = useState<ServerDraft | null>(null);
  const [intent, setIntent] = useState<ApplicationIntent | null>(null);
  const [research, setResearch] = useState<ResearchReport | null>(null);
  const [draft, setDraft] = useState<ComposeDraft | null>(null);
  const [deployment, setDeployment] = useState<DeploymentProgress | null>(null);
  const [deploymentJobId, setDeploymentJobId] = useState("");
  const serverDraftRef = useRef<ServerDraft | null>(null);
  const researchDraftIdsRef = useRef<Record<string, string>>({});
  const researchJobIdRef = useRef("");
  const initialRequestStarted = useRef(false);
  const resumeKey = `vaelor.application-research.${session.user.username}`;
  const deploymentResumeKey = `vaelor.application-deployment.${session.user.username}`;
  useEffect(() => {
    if (!deploymentJobId) return;
    let cancelled = false;
    const poll = async () => {
      while (!cancelled) {
        try {
          const job = await apiRequest<ServerJob>(`/jobs/${deploymentJobId}`);
          if (cancelled) return;
          const state: DeploymentProgress["state"] = jobIsReady(job)
            ? "healthy" : canonicalOperationState(job) === "cancelled" ? "cancelled" : canonicalOperationState(job) === "failed" || canonicalOperationState(job) === "rejected" ? "failed" : "running";
          const verifiedServices = Array.isArray((job.result?.health as { services?: unknown } | undefined)?.services)
            ? ((job.result?.health as { services: unknown[] }).services).map(String)
            : [];
          const openUrl = typeof job.result?.open_url === "string" ? job.result.open_url : undefined;
          setDeployment({ state, progress: job.progress, message: job.message, openUrl, healthChecks: state === "healthy" ? [{ name: "Managed application", status: "pass", detail: verifiedServices.length ? `Running and healthy: ${verifiedServices.join(", ")}.` : "The executor completed its service checks." }] : state === "failed" ? [{ name: "Managed application", status: "fail", detail: job.message }] : state === "cancelled" ? [{ name: "Managed application", status: "pending", detail: "Deployment was cancelled. The reviewed draft remains available for a new approval." }] : [{ name: "Managed application", status: "pending", detail: job.message }] });
          if (jobIsTerminal(job)) {
            window.localStorage.removeItem(deploymentResumeKey);
            return;
          }
        } catch (error) {
          if (!cancelled) setDeployment({ state: "failed", progress: 0, message: error instanceof Error ? error.message : "Deployment status could not be refreshed." });
          return;
        }
        await wait(2000);
      }
    };
    void poll();
    return () => { cancelled = true; };
  }, [deploymentJobId, deploymentResumeKey]);

  useEffect(() => {
    // "Describe a custom application" starts a NEW request. The wizard must open
    // blank on the Request step, never silently reopen the last saved draft the
    // container would otherwise restore from localStorage. Clear both resume keys
    // and skip restoration entirely.
    if (startFresh) { window.localStorage.removeItem(resumeKey); window.localStorage.removeItem(deploymentResumeKey); return; }
    if (resumeResearch?.jobId && resumeResearch.draftId) window.localStorage.setItem(resumeKey, JSON.stringify(resumeResearch));
    const saved = window.localStorage.getItem(resumeKey);
    if (!saved) return;
    let operation: { jobId: string; draftId: string };
    try { operation = JSON.parse(saved) as { jobId: string; draftId: string }; }
    catch { window.localStorage.removeItem(resumeKey); return; }
    if (!operation.jobId || !operation.draftId) { window.localStorage.removeItem(resumeKey); return; }
    researchDraftIdsRef.current[operation.jobId] = operation.draftId;
    researchJobIdRef.current = operation.jobId;
    void Promise.all([
      apiRequest<ServerJob>(`/jobs/${operation.jobId}`),
      apiRequest<ServerDraft>(`/applications/drafts/${operation.draftId}`),
    ]).then(([job, restoredDraft]) => {
      serverDraftRef.current = restoredDraft;
      setServerDraft(restoredDraft);
      setIntent(intentFromDraft(restoredDraft));
      // A draft whose compose + validation already exist is finalized, and it
      // reopens on a READ-ONLY Review whatever the research job now reports.
      // This was gated on jobIsReady(job), but the research job can be
      // superseded, failed, or non-deterministically re-run after the draft was
      // validated - none of which unmakes a validated compose. That gate landed
      // a finalized draft back on step-2 "Automatic research could not be
      // completed" with retry blocked; the presence of compose+validation is
      // the condition, not the job's state (mirrors the deployment-resume path).
      const reviewDraft = reviewDraftFromServer(restoredDraft);
      if (reviewDraft) {
        setResearch(reportFromDraft(restoredDraft));
        setDraft(reviewDraft);
        window.localStorage.removeItem(resumeKey);
      } else if (jobIsReady(job) && restoredDraft.manifest) {
        setResearch(reportFromDraft(restoredDraft));
        window.localStorage.removeItem(resumeKey);
      } else {
        setResearch(pendingResearch(job));
        if (jobIsTerminal(job) && canonicalOperationState(job) !== "needs_approval") window.localStorage.removeItem(resumeKey);
      }
    }).catch(() => window.localStorage.removeItem(resumeKey));
  }, [resumeKey, startFresh, deploymentResumeKey]);

  useEffect(() => {
    if (startFresh) return;
    const saved = window.localStorage.getItem(deploymentResumeKey);
    if (!saved) return;
    let operation: { jobId: string; draftId: string };
    try { operation = JSON.parse(saved) as { jobId: string; draftId: string }; }
    catch { window.localStorage.removeItem(deploymentResumeKey); return; }
    if (!operation.jobId || !operation.draftId) { window.localStorage.removeItem(deploymentResumeKey); return; }
    setDeploymentJobId(operation.jobId);
    void apiRequest<ServerDraft>(`/applications/drafts/${operation.draftId}`).then((restoredDraft) => {
      serverDraftRef.current = restoredDraft;
      setServerDraft(restoredDraft);
      setIntent(intentFromDraft(restoredDraft));
      if (restoredDraft.manifest) setResearch(reportFromDraft(restoredDraft));
      const reviewDraft = reviewDraftFromServer(restoredDraft);
      if (reviewDraft) setDraft(reviewDraft);
    }).catch(() => {
      // The job poll still presents the durable executor result. A missing draft
      // is reported by that workflow rather than creating another deployment.
    });
  }, [deploymentResumeKey, startFresh]);

  async function createApplicationDraft(request: string) {
    const result = await apiRequest<ServerDraft>("/applications/drafts", { method: "POST", body: JSON.stringify({ message: request }) }, session.csrf_token);
    serverDraftRef.current = result;
    setServerDraft(result);
    const mapped = intentFromDraft(result);
    setIntent(mapped);
    return { result, mapped };
  }

  async function classify(request: string) {
    const { mapped } = await createApplicationDraft(request);
    return mapped;
  }

  useEffect(() => {
    if (!autoClassify || !initialRequest.trim() || initialRequestStarted.current) return;
    initialRequestStarted.current = true;
    void createApplicationDraft(initialRequest)
      .then(({ result, mapped }) => researchServerDraft(result, mapped, []))
      .catch(() => {
        // ApplicationDeployment keeps the request editable so the operator can
        // correct a failed or ambiguous first-pass classification.
      });
  }, [autoClassify, initialRequest]);

  async function researchServerDraft(targetDraft: ServerDraft, _intent: ApplicationIntent, sourceUrls: string[]) {
    const researchJob = await apiRequest<ServerJob>("/jobs", {
      method: "POST",
      body: JSON.stringify({ type: "application.research", payload: { draft_id: targetDraft.id, source_urls: sourceUrls } }),
    }, session.csrf_token);
    onJobQueued?.(researchJob);
    researchDraftIdsRef.current[researchJob.id] = targetDraft.id;
    researchJobIdRef.current = researchJob.id;
    if ((jobIsReady(researchJob)) && canonicalOperationState(researchJob) !== "needs_approval") {
      const result = await apiRequest<ServerDraft>(`/applications/drafts/${targetDraft.id}`);
      serverDraftRef.current = result;
      setServerDraft(result);
      const completed = reportFromDraft(result);
      setResearch(completed);
      return completed;
    }
    window.localStorage.setItem(resumeKey, JSON.stringify({ jobId: researchJob.id, draftId: targetDraft.id }));
    const mapped = pendingResearch(researchJob);
    setResearch(mapped);
    return mapped;
  }

  async function refreshResearch(jobId: string) {
    const job = await apiRequest<ServerJob>(`/jobs/${jobId}`);
    if (!jobIsTerminal(job)) return pendingResearch(job);
    const draftId = researchDraftIdsRef.current[jobId];
    if (!jobIsReady(job)) {
      if (canonicalOperationState(job) === "needs_approval" && draftId) { window.localStorage.setItem(resumeKey, JSON.stringify({ jobId, draftId })); } else { window.localStorage.removeItem(resumeKey); }
      return pendingResearch(job, job.message || "Application research stopped before completion.");
    }
    if (!draftId) throw new Error("The saved research operation no longer identifies its application draft.");
    const result = await apiRequest<ServerDraft>(`/applications/drafts/${draftId}`);
    serverDraftRef.current = result;
    setServerDraft(result);
    window.localStorage.removeItem(resumeKey);
    return reportFromDraft(result);
  }
  async function retryResearch(researchId: string) {
    const draftId = researchDraftIdsRef.current[researchId] || serverDraftRef.current?.id || serverDraft?.id;
    if (!draftId) {
      throw new Error("The saved request is unavailable. Return to the request step and try again.");
    }
    const child = await apiRequest<ServerJob>(`/jobs/${researchId}/retry`, { method: "POST", body: "{}" }, session.csrf_token);
    researchDraftIdsRef.current[child.id] = draftId;
    researchJobIdRef.current = child.id;
    window.localStorage.setItem(resumeKey, JSON.stringify({ jobId: child.id, draftId }));
    onJobQueued?.(child);
    const mapped = pendingResearch(child);
    setResearch(mapped);
    return mapped;
  }

  async function replaceActiveResearch() {
    const activeId = researchJobIdRef.current || (research && (research.status === "queued" || research.status === "running") ? research.id : "");
    if (!activeId) { window.localStorage.removeItem(resumeKey); return; }
    if (research?.status !== "queued" && research?.status !== "running") {
      researchJobIdRef.current = ""; window.localStorage.removeItem(resumeKey); return;
    }
    try {
      await apiRequest<ServerJob>("/jobs/" + activeId + "/cancel", { method: "POST", body: "{}" }, session.csrf_token);
    } catch (error) {
      const message = error instanceof Error ? error.message.toLowerCase() : "";
      if (!message.includes("already finished")) throw error;
    }
    researchJobIdRef.current = ""; window.localStorage.removeItem(resumeKey);
  }

  async function cancelResearch(jobId: string) {
    const job = await apiRequest<ServerJob>(`/jobs/${jobId}/cancel`, { method: "POST", body: "{}" }, session.csrf_token);
    researchJobIdRef.current = ""; window.localStorage.removeItem(resumeKey);
    return pendingResearch(job, "Cancellation requested. Vaelor will preserve the request but will not deploy anything.");
  }

  async function researchApplication(currentIntent: ApplicationIntent, sourceUrls: string[]) {
    const targetDraft = serverDraftRef.current ?? serverDraft;
    if (!targetDraft) throw new Error("Create an application draft before starting research.");
    return researchServerDraft(targetDraft, currentIntent, sourceUrls);
  }

  async function generateDraft(configuration: DeploymentConfiguration) {
    if (!serverDraft?.manifest) throw new Error("Complete verified research before configuring this application.");
    const manifest = serverDraft.manifest;
    const ports = Object.fromEntries(manifest.ports.map((port) => [`${port.name}`, configuration.ports[`${port.target}/${port.protocol}`] ?? port.published]));
    const variables: Record<string, unknown> = { ...configuration.settings };
    for (const [name, credentialId] of Object.entries(configuration.secretReferences)) {
      if (credentialId) variables[name] = { credential_id: credentialId };
    }
    const configured = await apiRequest<ServerDraft>(`/applications/drafts/${serverDraft.id}/configure`, {
      method: "POST", body: JSON.stringify({ configuration: {
        ports,
        variables,
        resources: {
          memory_bytes: Math.round(configuration.memoryMb * 1024 * 1024),
          storage_bytes: Math.round(configuration.storageGb * 1024 * 1024 * 1024),
        },
        replace_existing: configuration.replaceExisting,
        // Only sent when the operator added them. `build_compose` accepts these
        // exact keys (vaelor/application_deployments.py); host_mounts is refused
        // server-side unless the source is allowlisted and consent is true.
        ...(configuration.addPorts.length ? { add_ports: configuration.addPorts } : {}),
        ...(configuration.hostMounts.length ? { host_mounts: configuration.hostMounts } : {}),
      } }),
    }, session.csrf_token);
    const validated = await apiRequest<ServerDraft>(`/applications/drafts/${serverDraft.id}/validate`, { method: "POST", body: "{}" }, session.csrf_token);
    setServerDraft(validated);
    const mapped = composeDraftFromServer({ ...configured, ...validated });
    setDraft(mapped);
    return mapped;
  }

  async function approveDeployment(reviewed: ComposeDraft) {
    if (!serverDraft) throw new Error("The server-owned application draft is unavailable.");
    const approval = await apiRequest<ApprovalResult>(`/applications/drafts/${serverDraft.id}/approve`, {
      method: "POST", body: JSON.stringify({ manifest_digest: reviewed.manifestDigest }),
    }, session.csrf_token);
    const job = await apiRequest<ServerJob>("/jobs", { method: "POST", body: JSON.stringify(approval.proposed_job) }, session.csrf_token);
    const initialState: DeploymentProgress["state"] = jobIsReady(job)
      ? "healthy" : canonicalOperationState(job) === "cancelled" ? "cancelled" : canonicalOperationState(job) === "failed" || canonicalOperationState(job) === "rejected" ? "failed" : canonicalOperationState(job) === "running" ? "running" : "queued";
    setDeployment({ state: initialState, progress: job.progress, message: job.message || "Approved deployment queued." });
    setDeploymentJobId(job.id);
    if (jobIsTerminal(job)) window.localStorage.removeItem(deploymentResumeKey);
    else window.localStorage.setItem(deploymentResumeKey, JSON.stringify({ jobId: job.id, draftId: serverDraft.id }));
    onJobQueued?.(job);
  }

  async function cancelDeployment() {
    if (!deploymentJobId) return;
    const job = await apiRequest<ServerJob>(`/jobs/${deploymentJobId}/cancel`, { method: "POST", body: "{}" }, session.csrf_token);
    setDeployment({ state: canonicalOperationState(job) === "cancelled" ? "cancelled" : "running", progress: job.progress, message: job.message || "Cancellation requested." });
  }

  async function storeSecret(name: string, value: string) {
    const result = await apiRequest<{ id: string }>("/credentials", {
      method: "POST",
      body: JSON.stringify({
        provider: "application-secret",
        label: `${intent?.application ?? "application"}/${name}`,
        secret: value,
      }),
    }, session.csrf_token);
    if (!result.id) throw new Error(`Vaelor did not return a secure reference for ${name}.`);
    return result.id;
  }

  // A draft can be configured exactly once: `POST /configure` refuses a draft
  // that is past `configuration` or already carries a generated compose. Telling
  // the wizard this keeps it from presenting an editable Configure step (and its
  // dead-end "Generate" button) for a finalized draft, e.g. one reopened from the
  // resume banner. A draft not yet created is configurable by default.
  const configurable = !serverDraft || (serverDraft.state === "configuration" && !serverDraft.compose);

  return <ApplicationDeployment
    initialRequest={initialRequest || resumeResearch?.requestSummary || ""}
    intent={intent}
    research={research}
    draft={draft}
    deployment={deployment}
    autoStartResearch={false}
    configurable={configurable}
    resumeResearch={resumeResearch}
    disabled={session.user.role !== "administrator"}
    onClose={onClose}
    onDirtyChange={onDirtyChange}
    onClassify={classify}
    onStartResearch={researchApplication}
    onRefreshResearch={refreshResearch}
    onCancelResearch={cancelResearch}
    onRetryResearch={retryResearch}
    onReplaceActiveResearch={replaceActiveResearch}
    researchRecovery={<WebResearchSetup
      session={session}
      onResearch={() => {
        if (research) void retryResearch(research.id).catch(() => undefined);
      }}
    />}
    onGenerateDraft={generateDraft}
    onStoreSecret={storeSecret}
    onApprove={approveDeployment}
    onCancelDeployment={cancelDeployment}
    onManage={onManaged}
  />;
}
