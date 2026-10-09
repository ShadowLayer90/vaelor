const SOURCE_LABELS: Record<string, string> = {
  "built-in-capability": "Policy-verified catalog",
  "built-in-planner": "Built-in planner",
};

/**
 * Who wrote an answer, from what was stored with it (ACC-103).
 *
 * A model answer used to be labelled with the tier selected NOW, so changing
 * the Assistant's model relabelled every earlier answer. The tier is recorded
 * with the answer (`model_label`); an answer saved before that was recorded
 * says it came from a model without naming one.
 */
export function assistantSourceLabel(source: string, modelLabel?: string) {
  if (source === "connected-model") {
    return modelLabel || "Answered by a model";
  }
  return SOURCE_LABELS[source] || source.replaceAll("-", " ");
}
