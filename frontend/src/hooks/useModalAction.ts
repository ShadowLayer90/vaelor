import { useCallback, useEffect, useRef, useState } from "react";

/** Said when a refusal carries no message of its own, so an alert is never empty. */
export const MODAL_ACTION_FALLBACK = "That did not go through, and no reason was given. Nothing was changed.";

/**
 * Run a modal's own action and keep its refusal IN the modal (W4d-D16, D18).
 *
 * A dialog that handed its save to the page let the page show the refusal,
 * on the page - under the open dialog, where `useDialogFocus` has made
 * everything inert and aria-hidden. The owner saw a dialog that did nothing.
 * Run the action through `run` and pass `error` to the dialog (`ModalShell`'s
 * or `ConfirmDialog`'s `error`). `run` resolves `true` when the action
 * succeeded, so the caller closes the dialog only then.
 */
export function useModalAction() {
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  const run = useCallback(async (action: () => unknown): Promise<boolean> => {
    setBusy(true);
    setError("");
    try {
      await action();
      return true;
    } catch (caught) {
      const message = caught instanceof Error ? caught.message.trim() : String(caught ?? "").trim();
      if (mounted.current) setError(message || MODAL_ACTION_FALLBACK);
      return false;
    } finally {
      if (mounted.current) setBusy(false);
    }
  }, []);

  const clear = useCallback(() => setError(""), []);
  return { error, busy, run, clear, setError };
}
