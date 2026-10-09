import { useEffect } from "react";
import { LLM_SERVER_ENDPOINT_ID, useEndpoints } from "../lib/endpoints";
import { llmServerBadge, llmServerRuntimeNote } from "../lib/llmServerStatus";
import { StatusPill } from "./StatusPill";
import { DEPLOYMENTS_MODELS_HREF } from "../lib/clusterSections";

/**
 * The compact LLM Server summary embedded in AI Chat (F3c).
 *
 * It replaces the full single-key panel that used to live here: managing the
 * endpoint's keys now belongs in Cluster -> Deployments, beside the model
 * deployments they expose, and this leaves only the truthful state and a link
 * there. It still reports the
 * surface's own `target_kind` up to its host, because the AI Chat connection
 * card resolves the serving mode from it — dropping that report would leave the
 * card unable to say which engine is answering.
 */
export function LlmServerSummary({
  onTargetKind,
}: {
  onTargetKind?: (kind: string | undefined) => void;
}) {
  const { endpoints, loading, error } = useEndpoints();
  const endpoint = endpoints.find((item) => item.id === LLM_SERVER_ENDPOINT_ID);
  const kind = endpoint?.target_kind;

  useEffect(() => {
    onTargetKind?.(kind);
  }, [kind, onTargetKind]);

  const pill = error
    ? { label: "Error", tone: "danger" as const }
    : loading && !endpoint
      ? { label: "Checking…", tone: "info" as const }
      : endpoint
        ? llmServerBadge(endpoint)
        : { label: "No GPU model", tone: "warning" as const };

  // The backend's sentence wins whenever the door is not serving: it says why
  // (paused, starting, not running, unverified) rather than restating the flag.
  const note = endpoint ? llmServerRuntimeNote(endpoint) : "";
  const line = error
    ? "The LLM Server status could not be read."
    : loading && !endpoint
      ? "Checking whether a model is exposed on your LAN."
      : note
        ? note
        : !endpoint?.available
          // B3: the backend's lead, the same sentence the Endpoints card shows.
          ? endpoint?.unavailableNote ?? ""
          : endpoint.enabled
            ? "Exposing this model on your LAN as an OpenAI-compatible API, protected by API keys."
            : "Ready to expose this model on your LAN. It is disabled for now.";

  return (
    <section className="llm-server-summary" aria-label="LLM Server">
      <div className="llm-server-summary__top">
        <div className="llm-server-summary__heading">
          <span className="ai-chat-eyebrow">Inbound API</span>
          <strong>LLM Server</strong>
        </div>
        <StatusPill label={pill.label} tone={pill.tone} />
      </div>
      <p className="llm-server-summary__line">{line}</p>
      <a className="llm-server-summary__link" href={DEPLOYMENTS_MODELS_HREF}>
        Manage endpoints and API keys in Cluster &rarr; Deployments &rarr; Models
      </a>
    </section>
  );
}
