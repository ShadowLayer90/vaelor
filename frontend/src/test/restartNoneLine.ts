import { waitFor } from "@testing-library/react";
import { expect } from "vitest";

/**
 * Why the restart dialog's no-jobs line says what it says (VD-133 1.5.5 #5).
 * Found live on the Z2, 2026-10-21: with a 30B model serving AI Chat on both
 * cluster machines, Reboot device said "Nothing Vaelor is running would be cut
 * off." The restart-impact read counts jobs only.
 */
export const RESTART_NONE_WHY = "LESSONS 5 and 8 / VD-133 1.5.5 #5, found live 2026-10-21: the restart-impact read "
  + "counts Vaelor JOBS only, so with no job running the dialog may say only that no job is running - never that "
  + "nothing is. What stops (models, apps, AI Chat) is the caller's description. Do not reword this line back into "
  + "\"nothing is running\", and do not loosen the test to match.";

/**
 * The dialog's no-jobs line, found by its place rather than its words, so a
 * reworded line fails on the text assertion with the reason above instead of
 * a bare "unable to find".
 */
export async function restartNoneLine(dialog: HTMLElement): Promise<string> {
  let text = "";
  await waitFor(() => {
    const line = dialog.querySelector(".restart-impact__none");
    expect(line, `${RESTART_NONE_WHY}\n  (the dialog showed no "no job is running" line)`).not.toBeNull();
    text = line?.textContent ?? "";
  });
  return text;
}
