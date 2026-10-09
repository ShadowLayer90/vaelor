/**
 * Which remembered cluster operation the owner has dismissed (ACC-210).
 *
 * The Cluster page keeps its current operation across page loads so its
 * progress can be followed. Its dialog opened again on every load, so a
 * failed operation, which has no Done, came back until "Continue in
 * background" was pressed again. The dismissal is kept per operation id: a
 * new operation still opens its dialog. Storage that is blocked only loses
 * the memory, never the page.
 */
export function dismissedOperationKey(operationKey: string): string {
  return `${operationKey}.dismissed`;
}

export function readDismissedOperation(operationKey: string): string {
  try {
    return window.localStorage.getItem(dismissedOperationKey(operationKey)) ?? "";
  } catch {
    return "";
  }
}

export function writeDismissedOperation(operationKey: string, id: string | null): void {
  try {
    if (id) window.localStorage.setItem(dismissedOperationKey(operationKey), id);
    else window.localStorage.removeItem(dismissedOperationKey(operationKey));
  } catch {
    // Nothing to do: the dialog is still closed on this page.
  }
}
