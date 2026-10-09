import { useCallback, useEffect, useState } from "react";
import type { Session } from "../types";
import { ApiError, apiRequest } from "../lib/api";
import { useGpuServingMode } from "../hooks/useGpuServingMode";
import { CustomAgentManager } from "./CustomAgentManager";
import { Notice } from "./ui";
import type { AgentProfile } from "./agentTypes";
import type { GpuServingMode } from "../lib/gpuServingMode";

/**
 * F1: the Cluster "Agents & tools" authoring surface for inference agents.
 *
 * An inference agent is a read-only custom-agent definition that a cluster model
 * backs and the fleet deploys - it is built HERE, where it is deployed, not on
 * the Assistant. It reuses the Assistant's {@link CustomAgentManager} in its
 * `surface="inference"` mode, so the create form offers no acting permissions
 * and lists only inference agents (`/assistant/profiles?surface=inference`).
 *
 * Authoring is administrator-only, the same envelope the cluster deploy runs
 * under. A non-administrator gets no list here, and the note says so rather
 * than promising one: `/assistant/profiles` returns only the definitions the
 * SIGNED-IN account built (`CustomAgentStore.list(actor)`), so no other account
 * can read an administrator's inference agents through it. What a
 * non-administrator can see instead is where agents that are RUNNING appear -
 * the Deployments tab reads `/cluster/agents`, which needs an operator account.
 */

const NON_ADMIN_NOTE_HEADING = "Administrators build inference agents";
const NON_ADMIN_NOTE =
  "Inference agents are built, changed and removed by an administrator, and each one is listed only to the administrator who built it, so none can be shown to this account.";
const INFERENCE_READ_FAILED = "The inference agents could not be read.";
const OPERATOR_RUNNING_NOTE =
  "Agents running on the cluster are listed on the Deployments tab, under Agents.";
const VIEWER_RUNNING_NOTE =
  "Agents running on the cluster are listed on the Deployments tab, which needs an operator or administrator account.";

/**
 * The model an inference agent answers with, said from the cluster serving
 * reading - never the Assistant model, which inference agents do not use
 * (ACC-079).
 */
export function inferenceModelNotice(serving: GpuServingMode): string {
  if (!serving.known) {
    return "Vaelor could not read whether a cluster model is serving, so it cannot say what inference agents would answer with.";
  }
  if (!serving.deployment) return "No cluster model is serving, so inference agents cannot be deployed yet.";
  const model = serving.model && serving.model !== serving.deployment
    ? `${serving.model} (${serving.deployment})`
    : serving.deployment;
  return `Inference agents answer with the cluster model ${model}.`;
}

export function ClusterInferenceAgentsPanel({ session }: { session: Session }) {
  const isAdmin = session.user.role === "administrator";
  // A cluster inference agent is backed by the serving cluster model, so the
  // manager's authoring gate reads that signal rather than the Assistant model.
  const serving = useGpuServingMode(session.user.role);
  const running = Boolean(serving.known && serving.deployment);
  const paused = serving.known && !running ? serving.pausedCluster : undefined;
  // Owner decision (sweep 2026-09-28): a cluster model that scaled to zero
  // wakes on the next request, so it backs authoring - with a note that the
  // first run wakes it. One unloaded by hand does not wake on a request, so it
  // does not open authoring; the note says how to load it.
  const asleep = paused?.wakesOnRequest === true;
  const clusterModelServing = running || asleep;
  // A deployed agent's first request wakes an idle-unloaded model (answered
  // with a retry hint), as do an AI Chat message and Load (w2-agents).
  const clusterModelNote = asleep
    ? `The cluster model ${paused?.deployment ?? ""} is asleep after sitting idle. You can build agents on it now. A deployed agent's first request wakes it and is answered with a retry hint; an AI Chat message or Load in Cluster > Deployments also wakes it. Loading may take a few minutes.`
    : paused?.wakesOnRequest === false
      ? `The cluster model ${paused.deployment} was unloaded by hand and does not wake on a request. Load it from Cluster > Deployments before building agents on it.`
      : paused
        ? `The cluster model ${paused.deployment} is unloaded, and Vaelor could not tell whether a request will wake it. Load it from Cluster > Deployments before building agents on it.`
        : "";
  const [profiles, setProfiles] = useState<AgentProfile[]>([]);
  const [error, setError] = useState("");

  // Read the inference agents only. The surface param filters out the builtins
  // and every assistant-surface definition on the backend, so the two kinds of
  // agent never mix here. Only an administrator's read is rendered, so only an
  // administrator makes it.
  const load = useCallback(() => {
    if (!isAdmin) return undefined;
    const controller = new AbortController();
    apiRequest<AgentProfile[]>("/assistant/profiles?surface=inference", { signal: controller.signal })
      .then((rows) => {
        if (controller.signal.aborted) return;
        setProfiles(Array.isArray(rows) ? rows : []);
        setError("");
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        // LESSONS 24: the server's sentence is shown; anything else is a
        // programming error, logged with its trace and never worded as one.
        if (!(reason instanceof ApiError)) console.error("Reading the inference agents failed", reason);
        setError(reason instanceof ApiError && reason.message
          ? `${INFERENCE_READ_FAILED} ${reason.message}`
          : INFERENCE_READ_FAILED);
      });
    return () => controller.abort();
  }, [isAdmin]);

  useEffect(() => load(), [load]);

  return (
    <div className="cl-stack cl-inference">
      {error && (
        <Notice severity="danger" heading="Inference agents unavailable">{error}</Notice>
      )}
      {isAdmin ? (
        <CustomAgentManager
          clusterModelNote={clusterModelNote || inferenceModelNotice(serving)}
          clusterModelServing={clusterModelServing}
          csrfToken={session.csrf_token}
          modelLabel=""
          modelReady={false}
          onChanged={() => { void load(); }}
          profiles={profiles}
          surface="inference"
          tasks={[]}
        />
      ) : (
        <Notice severity="info" heading={NON_ADMIN_NOTE_HEADING}>
          {NON_ADMIN_NOTE}{" "}
          {session.user.role === "operator" ? OPERATOR_RUNNING_NOTE : VIEWER_RUNNING_NOTE}
        </Notice>
      )}
    </div>
  );
}
