import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import type { AiChatConnection, AiChatConversation } from "../components/aiChatTypes";

/** `GET /ai-chat/web-search`: whether the toggle can be offered. Read without searching. */
export interface AiChatWebSearchView {
  available: boolean;
  state: string;
  reason: string;
  administrator?: boolean;
  /** Whether this account may turn it on: administrators only (owner, VD-221 item 14). */
  may_turn_on?: boolean;
  /** Why it may not, when it may not. */
  role_reason?: string;
}

/** What the composer's Web search control needs. */
export interface AiChatWebSearchState {
  /** `null` until the appliance has said whether web research is ready. */
  view: AiChatWebSearchView | null;
  /** Ready, and this account may turn it on: the toggle works. */
  usable: boolean;
  /** Whether the next send searches the web. Off by default, per conversation. */
  on: boolean;
  /** The hosted service the pages go to with the question, or "" (VD-206). */
  destination: string;
  /** Saving the choice to this conversation failed; said beside the control. */
  error: string;
  toggle: () => void;
  /** Read availability again (after web research was set up). */
  recheck: () => void;
}

/**
 * VD-221: AI Chat's Web search toggle.
 *
 * **Off by default, remembered per conversation, on the appliance.** A new chat
 * and a temporary chat start off; turning it on in a saved chat is written to
 * that chat (`PATCH /ai-chat/conversations/<id>`), so reopening it - in this
 * browser or another - shows what it was left at. A new chat that is sent with
 * it on keeps it: the appliance records the choice on the chat it creates.
 * Per conversation rather than per account because what leaves the box should
 * be decided where the question is asked: one chat about the news must not turn
 * search on for every other chat. The send itself carries the choice, so a
 * stale record can never make the appliance search.
 */
export function useAiChatWebSearch({
  csrfToken,
  conversationId,
  temporary,
  record,
  connection,
}: {
  csrfToken: string;
  conversationId: string;
  temporary: boolean;
  record?: AiChatConversation;
  connection?: AiChatConnection | null;
}): AiChatWebSearchState {
  const [view, setView] = useState<AiChatWebSearchView | null>(null);
  const [on, setOn] = useState(false);
  const [error, setError] = useState("");
  const recordRef = useRef(record);
  recordRef.current = record;
  const previousId = useRef(conversationId);
  // The chat whose stored choice has been applied: once per chat, so a list
  // read that lands before a PATCH cannot flip the owner's click back.
  const appliedFor = useRef("");

  const recheck = useCallback(() => {
    apiRequest<AiChatWebSearchView>("/ai-chat/web-search")
      .then((reading) => { if (reading && typeof reading === "object") setView(reading); })
      .catch(() => setView({ available: false, state: "unknown", reason: "Web search could not be checked on this appliance." }));
  }, []);

  useEffect(() => { recheck(); }, [recheck]);

  useEffect(() => {
    const previous = previousId.current;
    previousId.current = conversationId;
    setError("");
    appliedFor.current = "";
    // A temporary chat has no id, so it falls through to off below.
    const stored = recordRef.current;
    if (stored && stored.id === conversationId) {
      appliedFor.current = conversationId;
      setOn(stored.web_search === true);
      return;
    }
    // The chat this browser just created by sending: it keeps what was sent.
    if (previous === "" && conversationId) { appliedFor.current = conversationId; return; }
    setOn(false);
  }, [conversationId, temporary]);

  // A chat opened from the address bar: its record arrives after its id.
  useEffect(() => {
    if (temporary || !record || record.id !== conversationId || appliedFor.current === conversationId) return;
    appliedFor.current = conversationId;
    setOn(record.web_search === true);
  }, [conversationId, record, temporary]);

  const usable = Boolean(view?.available && view.may_turn_on !== false);

  const toggle = useCallback(() => {
    if (!usable) return;
    const next = !on;
    setOn(next); setError("");
    appliedFor.current = conversationId; // the owner's click outranks a record still on its way
    if (!conversationId || temporary) return;
    apiRequest(
      `/ai-chat/conversations/${conversationId}`,
      { method: "PATCH", body: JSON.stringify({ web_search: next }) },
      csrfToken,
    ).catch((failure: unknown) => {
      setOn(!next);
      setError(failure instanceof Error && failure.message ? failure.message : "The Web search choice was not saved.");
    });
  }, [conversationId, csrfToken, on, temporary, usable]);

  // Where the pages go: decided by the backend's `connection_locality` (`local`
  // is false), the same rule its "sent to" record reads (LESSONS 6).
  const destination = connection?.local === false
    ? connection.provider_label || connection.label || connection.provider
    : "";
  return { view, usable, on: on && usable, destination, error, toggle, recheck };
}
