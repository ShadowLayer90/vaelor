import { useRef, useState } from "react";
import { useFollowScroll } from "../hooks/useFollowScroll";
import type { AiChatCitation, AiChatMessage } from "./aiChatTypes";
import { modelDisplayName } from "../lib/modelIdentity";
import { performanceLine } from "../lib/performanceLine";
import { AiChatMarkdown } from "./AiChatMarkdown";
import { Icon } from "./Icon";
import { Button, Textarea } from "./ui";

/**
 * What a search over the user's knowledge actually did.
 *
 * `searched: false` means no collection was selected, so nothing was looked
 * up. `searched: true` with `passages: 0` means the documents were searched
 * and matched nothing - previously indistinguishable from the first case,
 * because the answer came back clean and uncited either way.
 */
export interface AiChatRetrieval {
  searched?: boolean;
  collections?: number;
  passages?: number;
}

/**
 * A stored answer also records which model produced it.
 *
 * `failed` marks the notice AI Chat writes in place of an answer that never
 * arrived. It is the reason the question above it still needs a retry, and it
 * is set by the client that wrote the notice rather than inferred from the
 * text, so a model that happens to say "this request failed" is not mistaken
 * for one.
 */
export type AiChatThreadMessage = AiChatMessage & {
  retrieval?: AiChatRetrieval;
  model?: string;
  failed?: boolean;
};

/**
 * Which turn is a question still waiting for an answer, if any.
 *
 * Two shapes reach this, and the retry control used to be reachable in
 * neither. Stop leaves the optimistic user turn last, and it has no server id
 * because it was never written. A failed send leaves the same user turn with a
 * failure notice appended after it, so the last message is an assistant turn.
 * Only the final exchange qualifies: an earlier question that failed has since
 * been answered or abandoned, and offering to re-run it would rewrite history.
 */
export function unansweredQuestionIndex(messages: AiChatThreadMessage[]): number {
  const last = messages.length - 1;
  if (last < 0) return -1;
  if (messages[last].role === "user") return last;
  if (messages[last].failed && messages[last - 1]?.role === "user") return last - 1;
  return -1;
}

function plural(count: number, noun: string) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

/*
 * Retrieval hands the answer every passage it found; the answer decides which
 * of them it stands on, and says so with an [S1] marker. Showing the whole
 * retrieval as "2 cited sources" put a citation disclosure under "I don't have
 * any information about that", which invites the reader to check a claim the
 * answer never made. Models emit the marker in ASCII or in the fullwidth
 * brackets they were trained on, so both are read.
 */
const CITATION_MARKER = /[[［【]\s*S\s*(\d+)\s*[\]］】]/gi;

export function citedSources(
  content: string, citations?: AiChatCitation[],
): AiChatCitation[] {
  if (!citations?.length) return [];
  const referenced = new Set<number>();
  for (const [, ordinal] of String(content ?? "").matchAll(CITATION_MARKER)) {
    referenced.add(Number(ordinal));
  }
  return citations.filter((_, index) => referenced.has(index + 1));
}

export function retrievalLabel(retrieval?: AiChatRetrieval): string {
  if (!retrieval || retrieval.searched === undefined) return "";
  if (!retrieval.searched) return "No knowledge selected · answered from model knowledge only";
  const collections = plural(retrieval.collections ?? 0, "collection");
  return retrieval.passages
    ? `Searched ${collections} · ${plural(retrieval.passages, "matching passage")}`
    : `Searched ${collections} · no matching passages`;
}

export function AiChatThread({
  messages,
  busy,
  generating,
  resumed,
  model,
  canModify,
  onRegenerate,
  onRetry,
  onFork,
  onNotice,
  onSuggestion,
  onStop,
}: {
  messages: AiChatThreadMessage[];
  /** A request is in flight somewhere, so no turn here may be re-run. */
  busy: boolean;
  /**
   * The request in flight *for this conversation*, if any, and the model it
   * was sent with.
   *
   * `busy` alone drove the indicator, and both are the viewer's state rather
   * than the request's: with a reply in flight in one chat, opening another
   * announced "Generating with <whatever the picker now says>" in a
   * conversation that had asked nothing.
   */
  generating: { model: string } | null;
  /**
   * A question restored from the store whose answer the appliance is still
   * writing, or one it never delivered. A reload mid-answer used to leave the
   * question alone in the transcript with no reply, no spinner and no error.
   */
  resumed?: { awaiting: boolean; lost: boolean };
  model: string;
  canModify: boolean;
  onRegenerate: (message: AiChatThreadMessage, content?: string) => Promise<void>;
  /**
   * Ask the unanswered question again. Separate from `onRegenerate` because a
   * turn that was never saved has no id to regenerate against: the caller
   * decides between re-running a stored turn and re-sending an optimistic one.
   */
  onRetry: (message: AiChatThreadMessage) => Promise<void>;
  onFork: (message: AiChatThreadMessage) => Promise<void>;
  onNotice: (message: string) => void;
  onSuggestion: (message: string) => void;
  onStop?: () => void;
}) {
  const [editingId, setEditingId] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const threadRef = useRef<HTMLDivElement>(null);
  const { following, scrollToBottom } = useFollowScroll(threadRef, [
    messages.length,
    messages.at(-1)?.content,
    busy,
  ], {
    /*
     * Nothing to follow before the first message. Pinning the empty state to
     * its bottom scrolled the opening starter prompts up out of sight, so a
     * new user arrived at AI Chat already looking past the suggestions meant
     * to get them started.
     */
    enabled: messages.length > 0,
  });

  const copy = async (content: string) => {
    await navigator.clipboard.writeText(content);
    onNotice("Message copied.");
  };

  const retryIndex = unansweredQuestionIndex(messages);
  /*
   * An answer is on its way to the last turn — either generating live in this
   * conversation, or being written server-side while the resume-poll waits it
   * out after a dropped send. Until it genuinely lands, the trailing turn is
   * not finished, so it must not wear a finished turn's clothes: no Copy /
   * Regenerate / Branch / Retry and no "Searched …" footer, which together read
   * as a completed answer. Presenting the still-arriving turn as final is what
   * let a reader act on a partial — the Retry beside an unanswered question
   * fired a second request while the first was still running ("the model is
   * busy answering another request"). The in-progress affordance below says
   * what is actually happening; the controls return the moment it completes.
   */
  const answerPending = Boolean(generating) || Boolean(resumed?.awaiting);

  return (
    <>
    <div className="ai-chat-messages" role="log" aria-live="polite" ref={threadRef}>
      {messages.length ? messages.map((message, index) => {
        const key = message.id ?? `${message.role}-${index}`;
        const isEditing = message.id === editingId;
        /*
         * The byline names the model that produced THIS answer. It used to
         * render the model picker's current value, so changing the dropdown
         * silently relabelled every past reply - and the relabelling survived
         * a reload, because it was never the stored value being shown.
         */
        const byline = message.role === "assistant"
          ? modelDisplayName(message.model || model)
          : "Prompt";
        const retrieval = message.role === "assistant" ? retrievalLabel(message.retrieval) : "";
        const performance = message.role === "assistant" ? performanceLine(message.metadata?.performance) : "";
        const cited = message.role === "assistant"
          ? citedSources(message.content, message.citations)
          : [];
        /*
         * A question whose answer never arrived is the one turn that most
         * needs a retry, and it was the one turn with no action on it. The
         * gate used to also demand a server id and an editable conversation,
         * which excluded both states the app actually produces: Stop leaves an
         * unsaved turn, and a first failed send has no conversation yet.
         */
        // The trailing turn while an answer is still arriving. Its final-state
        // controls and footer are withheld until the turn actually completes.
        const pendingTail = answerPending && index === messages.length - 1;
        const canRetry = index === retryIndex && !pendingTail;
        return (
          <article className={`ai-chat-turn ai-chat-turn--${message.role}`} key={key}>
            <div className="ai-chat-turn__node" aria-hidden="true">
              {message.role === "user" ? "U" : <Icon name="memory" size={15} />}
            </div>
            <div className="ai-chat-turn__body">
              <header>
                <strong>{message.role === "user" ? "You" : "Vaelor AI"}</strong>
                <span>{byline}</span>
              </header>
              {isEditing ? (
                <div className="ai-chat-turn__edit">
                  <Textarea
                    autoFocus
                    label={<span className="sr-only">Edit message</span>}
                    maxLength={8000}
                    onChange={(event) => setDraft(event.target.value)}
                    rows={5}
                    value={draft}
                  />
                  <div>
                    <Button onClick={() => setEditingId(null)} type="button" variant="quiet">Cancel</Button>
                    <Button
                      disabled={!draft.trim() || busy}
                      onClick={() => void onRegenerate(message, draft).then(() => setEditingId(null))}
                      type="button"
                      variant="primary"
                    >
                      Save and regenerate
                    </Button>
                  </div>
                </div>
              ) : message.role === "assistant" ? (
                <AiChatMarkdown content={message.content} />
              ) : <p>{message.content}</p>}
              {retrieval && !pendingTail && (
                <p className="ai-chat-turn__retrieval">
                  <small>{retrieval}</small>
                </p>
              )}
              {performance && !pendingTail && (
                <p className="ai-chat-turn__performance">
                  <small>{performance}</small>
                </p>
              )}
              {cited.length && !pendingTail ? (
                <details className="ai-chat-citations">
                  <summary>{plural(cited.length, "cited source")}</summary>
                  {cited.map((citation) => (
                    <blockquote key={citation.id}>
                      <strong>{citation.document} · chunk {citation.chunk}</strong>
                      <span>{citation.collection}</span>
                      <p>{citation.excerpt}</p>
                    </blockquote>
                  ))}
                </details>
              ) : null}
              {!isEditing && !pendingTail && (
                <div className="ai-chat-turn__actions">
                  <Button onClick={() => void copy(message.content)} type="button" variant="quiet">Copy</Button>
                  {canModify && message.id && message.role === "user" && (
                    <Button onClick={() => { setEditingId(message.id!); setDraft(message.content); }} type="button" variant="quiet">Edit</Button>
                  )}
                  {canRetry && (
                    <Button disabled={busy} onClick={() => void onRetry(message)} type="button" variant="quiet">Retry answer</Button>
                  )}
                  {canModify && message.id && message.role === "assistant" && (
                    <Button disabled={busy} onClick={() => void onRegenerate(message)} type="button" variant="quiet">Regenerate</Button>
                  )}
                  {canModify && message.id && (
                    <Button disabled={busy} onClick={() => void onFork(message)} type="button" variant="quiet">Branch here</Button>
                  )}
                </div>
              )}
            </div>
          </article>
        );
      }) : (
        <div className="ai-chat-empty">
          <div className="ai-chat-empty__intro">
            <span className="ai-chat-empty__glyph"><Icon name="memory" size={28} /></span>
            <div>
              <p className="page-eyebrow">General AI + cited knowledge</p>
              <h2>What are you working on?</h2>
              <p>Ask directly, or start with one of these common workflows.</p>
            </div>
          </div>
          <div className="ai-chat-empty__starters">
            <Button onClick={() => onSuggestion("Explain this technical concept in plain language: ")} type="button" variant="quiet">
              <span>01</span><strong>Explain something</strong><small>Turn a technical topic into a clear answer.</small>
            </Button>
            <Button onClick={() => onSuggestion("Summarize the key points and action items from my selected knowledge: ")} type="button" variant="quiet">
              <span>02</span><strong>Summarize knowledge</strong><small>Use an attached runbook, log, or document.</small>
            </Button>
            <Button onClick={() => onSuggestion("Help me plan this deployment step by step: ")} type="button" variant="quiet">
              <span>03</span><strong>Plan a deployment</strong><small>Map prerequisites, risks, and next actions.</small>
            </Button>
            <Button onClick={() => onSuggestion("Compare these options and recommend one for this Vaelor node: ")} type="button" variant="quiet">
              <span>04</span><strong>Compare options</strong><small>Evaluate tradeoffs for this appliance.</small>
            </Button>
          </div>
        </div>
      )}
      {!generating && resumed?.awaiting && (
        <p className="assistant-resumed-wait" role="status">
          Your last question is still being answered on this appliance. The reply appears
          here as soon as it lands — you do not need to ask again.
        </p>
      )}
      {!generating && resumed?.lost && (
        <p className="assistant-resumed-wait assistant-resumed-wait--lost" role="status">
          No answer arrived for your last question, and Vaelor has stopped waiting for it.
          Nothing was changed. Ask it again when you are ready.
        </p>
      )}
      {generating && (
        <div className="assistant-thinking ai-chat-thinking">
          <span className="assistant-thinking__dot" /><span className="assistant-thinking__dot" /><span className="assistant-thinking__dot" /><strong>Generating with {modelDisplayName(generating.model || model)}</strong>
          {onStop && (
            <Button onClick={onStop} type="button" variant="quiet">Stop</Button>
          )}
        </div>
      )}
    </div>
      {/* Outside the live region: toggling this on scroll must not be
          announced as new conversation content. */}
      {!following && messages.length > 0 && (
        <div className="ai-chat-messages__resume">
          <Button onClick={() => scrollToBottom("smooth")} type="button" variant="quiet">
            <Icon name="chevron" /> Jump to latest
          </Button>
        </div>
      )}
    </>
  );
}
