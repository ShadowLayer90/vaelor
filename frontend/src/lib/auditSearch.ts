import type { AuditEvent } from "../types";
import { auditDisplayedText } from "./auditTargets";

const normalize = (value: string) =>
  value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

/**
 * Whether a row matches the search, by the words the row shows (FE-W7-2).
 *
 * It searched the action slug, the raw target id and the backend label, so
 * "Download AI model" (the words on screen) missed while a hidden job UUID
 * matched. It now reads `auditDisplayedText`, the same text the cells render;
 * a raw id is found only by opening its disclosure, never by the search.
 */
export function auditMatches(event: AuditEvent, query: string) {
  const term = normalize(query);
  if (!term) return true;
  return normalize(auditDisplayedText(event)).includes(term);
}
