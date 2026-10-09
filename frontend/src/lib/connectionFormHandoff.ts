/**
 * The one way into the form that adds a model connection for AI Chat.
 *
 * VD-200 found AI Chat's empty picker sending people to "add an external
 * provider in Details" - a form that does not exist - and Settings → Connections
 * sending them to "Workloads → Improve the assistant", a page since renamed and
 * a destination that set up the Assistant, not AI Chat. The add form really
 * lives in Apps and AI's assistant setup dialog (`CopilotSetup`), so this opens
 * that dialog with the form showing, the same handoff shape the planner uses
 * (`workloadHandoff.ts`): a session flag, then a navigation Apps and AI reads on
 * mount. Only an administrator can add a connection (`POST /credentials`).
 */
export const CONNECTION_FORM_FLAG = "vaelor.open-ai-chat-connection-form";

/** Where the form is, in words, for anyone who cannot use the link. */
export const CONNECTION_FORM_PATH = "Apps and AI → Set up assistant → Connect a model for AI Chat";

export function openAiChatConnectionForm() {
  try {
    window.sessionStorage.setItem(CONNECTION_FORM_FLAG, "1");
  } catch {
    // Storage refused: the navigation still lands on Apps and AI.
  }
  window.dispatchEvent(new CustomEvent("pironman:navigate", { detail: "workloads" }));
}

/**
 * Whether Apps and AI was opened to show the form. Reading is pure (a state
 * initialiser may run twice); the reader clears it once it has acted.
 */
export function connectionFormRequested(): boolean {
  try {
    return window.sessionStorage.getItem(CONNECTION_FORM_FLAG) === "1";
  } catch {
    return false;
  }
}

export function clearConnectionFormRequest() {
  try {
    window.sessionStorage.removeItem(CONNECTION_FORM_FLAG);
  } catch {
    // Nothing to clear when storage is refused.
  }
}
