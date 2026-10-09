import type { AgentStatus } from "../components/agentTypes";

/**
 * The one answer to "is the Assistant's model answering?" for every screen.
 *
 * Ask read `reachable === false` and said the model was not answering, while
 * Routines read `configured` alone and showed the same model ready with Run
 * enabled (ACC-136). Both now read this projection.
 *
 * - `configured`: a model is chosen for the appliance and the viewer has not
 *   opted out with "basic". Authoring and the Ask pill's "no model" wording
 *   key on this.
 * - `answering`: configured AND the server's readiness check came back
 *   reachable AND the server has not said it offers no model. Anything that
 *   sends work to the model (Run, test runs, schedules) keys on this. A
 *   missing or failed check is not answering: unknown is never ready.
 * - `notAnsweringReason`: plain-English reason, non-empty exactly when a
 *   configured model is not answering.
 * - `notOffered`: the server's `model_availability_reason`, non-empty exactly
 *   when the server is reachable but says it offers no model
 *   (`offering_models === false`, ACC-099). Ask's "No model available" pill and
 *   notice come from it; it is also the `notAnsweringReason` then.
 */
export interface AssistantModelReadiness {
  configured: boolean;
  answering: boolean;
  notAnsweringReason: string;
  notOffered: string;
}

export const MODEL_NOT_OFFERED_REASON =
  "The model server is reachable but is not offering any model.";

export const MODEL_NOT_CONFIRMED_REASON =
  "Vaelor has not confirmed that the selected AI model is answering.";

export function assistantModelReadiness(
  status: AgentStatus | null | undefined,
  intelligenceChoice: string | null | undefined,
): AssistantModelReadiness {
  const configured = Boolean(status?.configured && intelligenceChoice !== "basic");
  if (!configured) return { configured: false, answering: false, notAnsweringReason: "", notOffered: "" };
  if (status?.reachable === true && status.offering_models === false) {
    const notOffered = status.model_availability_reason || MODEL_NOT_OFFERED_REASON;
    return { configured: true, answering: false, notAnsweringReason: notOffered, notOffered };
  }
  if (status?.reachable === true) return { configured: true, answering: true, notAnsweringReason: "", notOffered: "" };
  const reason = status?.reachable === false
    ? (status.unreachable_reason || "The selected AI model could not be reached.")
    : MODEL_NOT_CONFIRMED_REASON;
  return { configured: true, answering: false, notAnsweringReason: reason, notOffered: "" };
}
