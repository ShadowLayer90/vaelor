/**
 * The words for a model's runtime mode and the surface it serves: one owner
 * (LESSONS 6). ActionReviewDialog, the "Use model" panel and the Assistant
 * setup each spelled them before.
 *
 * The mode ids and names are the backend's: a registered copy of
 * vaelor/copilot_setup.py `_RUNTIME_MODES` (sent as `runtime_modes`), held to
 * it by src/lib/modelModes.test.ts, which also fails on a spelling anywhere
 * else in the frontend. The surface ids are the `surface` a `model.deploy`
 * carries (vaelor/model_catalog.py `deploy_surface`).
 */

export type RuntimeMode = "efficient" | "balanced" | "quality";

/** In the backend's order: least memory first. */
export const RUNTIME_MODES: readonly RuntimeMode[] = ["efficient", "balanced", "quality"];

export const RUNTIME_MODE_NAMES: Readonly<Record<RuntimeMode, string>> = {
  efficient: "Memory saver",
  balanced: "Balanced",
  quality: "Long context",
};

export type ModelSurface = "ai-chat" | "assistant";

export const MODEL_SURFACE_NAMES: Readonly<Record<ModelSurface, string>> = {
  "ai-chat": "AI Chat",
  assistant: "Assistant",
};

const own = <K extends string>(table: Readonly<Record<K, string>>, value: string): string | null =>
  Object.prototype.hasOwnProperty.call(table, value) ? table[value as K] : null;

/** A runtime mode's name, or null for a value this table does not own. */
export function runtimeModeName(mode: string): string | null {
  return own(RUNTIME_MODE_NAMES, mode);
}

/** A model surface's name, or null for a value this table does not own. */
export function modelSurfaceName(surface: string | undefined): string | null {
  return surface === undefined ? null : own(MODEL_SURFACE_NAMES, surface);
}
