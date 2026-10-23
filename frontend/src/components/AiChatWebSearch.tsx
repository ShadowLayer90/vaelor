import { useState } from "react";
import type { AiChatWebSearchState } from "../hooks/useAiChatWebSearch";
import type { Session } from "../types";
import type { AiChatMessage } from "./aiChatTypes";
import { Icon } from "./Icon";
import { Button } from "./ui";
import { WebResearchSetup } from "./WebResearchSetup";

/**
 * VD-221: the Web search toggle beside Knowledge, and the line under the
 * composer that says what it does.
 *
 * Off by default. On, the line says what leaves the appliance: the question,
 * as a search, through the box's guarded search - and, for a model that runs
 * elsewhere, the pages it reads go there with the question. Only an
 * administrator may turn it on (owner, VD-221 item 14); anyone else sees it
 * unusable with that reason. When web research is not ready, pressing it opens
 * the reason and, for an administrator, the setup row Apps already uses
 * (`WebResearchSetup`), read without a search query. While a send is in flight
 * it cannot be changed: the send already carries the choice.
 */
export function AiChatWebSearchToggle({ webSearch, noteId, expanded, onPress, busy }: {
  webSearch: AiChatWebSearchState;
  noteId: string;
  /** Whether the unusable toggle's explanation is open. */
  expanded: boolean;
  onPress: () => void;
  /** A send is in flight. */
  busy: boolean;
}) {
  const usable = webSearch.usable;
  const read = Boolean(webSearch.view);
  return (
    <Button
      aria-controls={usable ? undefined : noteId}
      aria-describedby={webSearch.on ? noteId : undefined}
      aria-expanded={read && !usable ? expanded : undefined}
      aria-pressed={usable ? webSearch.on : undefined}
      className={[
        "ai-chat-websearch",
        webSearch.on ? "is-on" : "",
        read && !usable ? "is-unavailable" : "",
      ].filter(Boolean).join(" ")}
      disabled={!read || (busy && usable)}
      onClick={onPress}
      title={usable ? "Search the web for this chat's answers" : "Web search is not available to you yet"}
      type="button"
    >
      <Icon name="search" size={16} />
      Web search
    </Button>
  );
}

/**
 * VD-222 item 4: weather questions with a place are read from Open-Meteo
 * (geocoding-api.open-meteo.com and api.open-meteo.com), a destination the
 * owner is told about here, where Web search is turned on.
 */
export const OPEN_METEO = "Open-Meteo";

/** The line under the composer: what Web search does now, or why it cannot. */
export function AiChatWebSearchNote({ webSearch, noteId, session, explaining, onClose }: {
  webSearch: AiChatWebSearchState;
  noteId: string;
  session: Session;
  explaining: boolean;
  onClose: () => void;
}) {
  if (webSearch.error) {
    return <p className="ai-chat-websearch__note is-error" id={noteId} role="alert">{webSearch.error}</p>;
  }
  if (webSearch.on) {
    return (
      <p className="ai-chat-websearch__note" id={noteId}>
        Web search is on: your question goes to public search engines through this
        appliance&rsquo;s guarded search, and answers cite the pages read. A weather
        question that names a place is read from {OPEN_METEO} rather than searched;
        Open-Meteo is sent only the words read as the place, never the question.
        {webSearch.destination
          ? ` Those pages are sent to ${webSearch.destination} with your question.`
          : ""}
      </p>
    );
  }
  const view = webSearch.view;
  if (!explaining || !view || webSearch.usable) return null;
  const roleReason = view.may_turn_on === false
    ? view.role_reason || "Only an administrator can turn on Web search."
    : "";
  return (
    <div className="ai-chat-websearch__setup" id={noteId} role="region" aria-label="About Web search">
      <p className="ai-chat-websearch__note">
        {roleReason || `Web search is unavailable: ${view.reason || "guarded web research is not ready."}`}{" "}
        {roleReason
          ? "Nothing is searched from this account."
          : "It uses this appliance’s guarded web research. Nothing is searched until it is set up and you turn Web search on."}
      </p>
      {!roleReason && (
        <WebResearchSetup
          onResearch={webSearch.recheck}
          quickStatus
          readyActionLabel="Use it for Web search"
          session={session}
        />
      )}
      <Button className="ai-chat-ghost" onClick={() => { webSearch.recheck(); onClose(); }} type="button" variant="quiet">
        Close
      </Button>
    </div>
  );
}

/** Whether the unusable toggle's explanation is open; the toggle opens it. */
export function useWebSearchExplanation(webSearch?: AiChatWebSearchState) {
  const [explaining, setExplaining] = useState(false);
  const toggle = () => {
    if (!webSearch) return;
    if (webSearch.view && !webSearch.usable) setExplaining((open) => !open);
    else webSearch.toggle();
  };
  return { explaining, close: () => setExplaining(false), toggle };
}

/** VD-221: a stored answer's web search, as its byline says it. */
export type AiChatWebSearchRecord = {
  outcome?: string;
  query?: string;
  sources?: number;
  results?: number;
  sent_to?: string;
};

/** The byline fact for a turn's web search, or "" when none ran. */
export function webSearchLabel(metadata?: AiChatMessage["metadata"]): string {
  const record = metadata?.web_search;
  if (!record || typeof record !== "object") return "";
  const sentTo = record.sent_to ? ` · sent to ${record.sent_to}` : "";
  if (record.outcome === "found") {
    const count = record.sources ?? 0;
    return `Searched the web · ${count} page${count === 1 ? "" : "s"}${sentTo}`;
  }
  if (record.outcome === "weather") return `Weather from ${OPEN_METEO}`;
  if (record.outcome === "weather_ask") return `${OPEN_METEO} found more than one place`;
  if (record.outcome === "weather_where") return "Weather needs a town or city";
  if (record.outcome === "no_results") return "Web search found nothing relevant";
  if (record.outcome === "failed") return "Web search could not run";
  if (record.outcome === "unavailable") return "Web search unavailable";
  return "";
}

/** A citation's page address, when it is one AI Chat may link to: https and well formed. */
export function citationLink(url?: string): string {
  if (!url || !/^https:\/\/\S+$/i.test(url)) return "";
  try {
    return new URL(url).protocol === "https:" ? url : "";
  } catch {
    return "";
  }
}

/** The site a web citation came from, for the line under its title. */
export function citationSite(url?: string): string {
  const link = citationLink(url);
  return link ? new URL(link).hostname : "";
}
