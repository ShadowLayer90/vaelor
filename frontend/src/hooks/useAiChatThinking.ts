import { useCallback, useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import type { AiChatThinkingStep, AiChatThinkingView } from "../components/aiChatTypes";

/** What the toolbar's thinking control needs: the reading, the step, and a setter. */
export interface AiChatThinkingState {
  /** `null` until `/ai-chat/thinking` answers for this connection and model. */
  view: AiChatThinkingView | null;
  /** The step a send carries; "" when the control is hidden. */
  step: AiChatThinkingStep | "";
  /** Saving the owner's choice failed; the control says so beside itself. */
  error: string;
  choose: (step: AiChatThinkingStep) => void;
}

/**
 * VD-209: whether the active connection's model can think, and the step the
 * owner chose for that connection.
 *
 * The appliance decides capability (from the provider's own model listing, or
 * its one family table) and remembers the step per connection. This hook only
 * asks, and it asks again whenever the connection or the model changes, so a
 * model that cannot think never inherits another model's control. Anything
 * unreadable leaves the control hidden: it never draws steps nobody confirmed.
 */
export function useAiChatThinking(csrfToken: string, connectionId: string, model: string): AiChatThinkingState {
  const [view, setView] = useState<AiChatThinkingView | null>(null);
  const [step, setStep] = useState<AiChatThinkingStep | "">("");
  const [error, setError] = useState("");

  useEffect(() => {
    setView(null); setStep(""); setError("");
    if (!connectionId || !model) return;
    let current = true;
    apiRequest<AiChatThinkingView>(`/ai-chat/thinking?model=${encodeURIComponent(model)}`)
      .then((reading) => {
        if (!current || !reading || typeof reading !== "object") return;
        setView(reading);
        setStep(reading.available && reading.steps.includes(reading.step) ? reading.step : "");
      })
      .catch(() => { /* hidden: an unread capability is never shown as one */ });
    return () => { current = false; };
  }, [connectionId, model]);

  const choose = useCallback((next: AiChatThinkingStep) => {
    if (!view?.available || !view.steps.includes(next)) return;
    const previous = step;
    setStep(next); setError("");
    apiRequest(
      "/ai-chat/thinking",
      { method: "PATCH", body: JSON.stringify({ credential_id: view.credential_id, step: next }) },
      csrfToken,
    ).catch((failure: unknown) => {
      setStep(previous);
      setError(failure instanceof Error && failure.message ? failure.message : "The thinking step was not saved.");
    });
  }, [csrfToken, step, view]);

  return { view, step, error, choose };
}
