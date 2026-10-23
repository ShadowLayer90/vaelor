import { useCallback, useState } from "react";
import { blamedModel } from "../components/AiChatModelPicker";

/**
 * Models the server could not run, and what it said about each.
 *
 * The model list is bare ids, so a model that is not loaded looks like any
 * other; what the appliance said cannot run is shown against the model until it
 * answers. Moved out of `AiChat` (VD-221 review round 1, item 13) so the page
 * has room under the 1,000-line ceiling without packing lines.
 */
export function useAiChatModelFailures() {
  const [modelFailures, setModelFailures] = useState<Record<string, string>>({});

  /**
   * One failure, reported one way.
   *
   * Every failed question leaves the same three things behind: the failed
   * exchange in the transcript, the question back in the composer, and a
   * banner. What the banner could not do until now was say which model failed,
   * so a picker full of models the appliance cannot run gave the same sentence
   * whichever one produced it.
   */
  const reportFailure = useCallback((error: unknown, model: string, failure: string) => {
    const blamed = blamedModel(error, model);
    if (!blamed) return failure;
    setModelFailures((current) => ({ ...current, [blamed]: failure }));
    return `${blamed} could not run this request. ${failure}`;
  }, []);

  /** It answered, so whatever it did last time is no longer true of it. */
  const clearModelFailure = useCallback((model: string) => {
    if (!model) return;
    setModelFailures((current) => {
      if (!(model in current)) return current;
      const next = { ...current };
      delete next[model];
      return next;
    });
  }, []);

  return { modelFailures, reportFailure, clearModelFailure };
}
