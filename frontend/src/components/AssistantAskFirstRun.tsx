import type { CopilotSetupData } from "./CopilotSetup";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import { Button, Card } from "./ui";

/**
 * First-time Assistant setup (VD-200, the AssistFirstRun board): two choices
 * and one sentence about whose model is whose.
 *
 * VD-049 / VD-201 item 2: there is no outside-model choice here. The Assistant
 * runs Vaelor's own model or built-in basic mode; a model the owner connects
 * is AI Chat's.
 */
export function AssistantAskFirstRun({
  busy,
  onChoose,
  setupData,
}: {
  busy: boolean;
  onChoose: (choice: "basic" | "local") => void;
  setupData: CopilotSetupData;
}) {
  const recommendation = setupData.recommendation;
  const canInstall = recommendation.can_install;
  /*
   * Why the recommended model cannot be installed, from the two facts the
   * server sends. Storage is the one it names; anything else is said as what
   * it is - this machine cannot take it - rather than guessed at.
   */
  const blockedPill = recommendation.storage_ok === false ? "More space needed" : "Cannot install here";
  const blockedReason = recommendation.storage_ok === false
    ? "This machine does not have the free storage the model needs. Built-in basic mode still works."
    : "This machine cannot install the recommended model. Built-in basic mode still works.";
  return (
    <section aria-labelledby="assistant-first-run-title" className="as-first-run ui-card">
      <div className="as-first-run__intro">
        <span aria-hidden="true" className="ui-row__icon ui-row__icon--accent as-first-run__icon"><Icon name="bolt" size={20} /></span>
        <div>
          <span className="as-label">First-time assistant setup</span>
          <h2 id="assistant-first-run-title">How smart should Vaelor be?</h2>
          <p>Choose once now. You can change this later without losing chats or appliance settings.</p>
        </div>
      </div>
      <div className="as-first-run__choices">
        <article className={canInstall ? "as-choice as-choice--recommended" : "as-choice"}>
          <div className="as-choice__top">
            <span className="as-choice__flag">Recommended</span>
            <span aria-hidden="true" className="ui-row__icon ui-row__icon--accent"><Icon name="npu" size={16} /></span>
          </div>
          <h3>Install {recommendation.primary.name}</h3>
          {/*
            * The size of the model named beside it, from the catalog entry's
            * own note - derived from the byte count the fit check divides by,
            * so the name and the size cannot part company. It once read "about
            * 1.1 GB" whatever was recommended.
            */}
          <p>Private local answers · {recommendation.primary.size_note} · reviewed before download</p>
          <div className="as-choice__action">
            <Button
              disabled={busy || !canInstall}
              onClick={() => onChoose("local")}
              type="button"
              variant="primary"
            >
              Install {recommendation.primary.name}
            </Button>
            {!canInstall && <StatusPill label={blockedPill} tone="warning" />}
          </div>
          {!canInstall && <p className="as-small as-muted">{blockedReason}</p>}
        </article>
        <article className="as-choice">
          <div className="as-choice__top">
            <span className="as-choice__flag as-choice__flag--plain">No download</span>
            <span aria-hidden="true" className="ui-row__icon"><Icon name="cpu" size={16} /></span>
          </div>
          <h3>Use built-in basic mode</h3>
          <p>Code-based live appliance answers and diagnostics; broader questions stay limited</p>
          <div className="as-choice__action">
            <Button disabled={busy} onClick={() => onChoose("basic")} type="button">Use built-in basic mode</Button>
          </div>
        </article>
      </div>
      <p className="as-small as-muted">The Assistant runs Vaelor&apos;s own model or built-in basic mode. A model you connect yourself is for AI Chat.</p>
    </section>
  );
}

/** While the Assistant's setup is being read: grey lines, said once, never a guess. */
export function AssistantAskFirstRunLoading({ labelId }: { labelId?: string } = {}) {
  return (
    <Card className="as-first-run-loading">
      <span className="as-label" id={labelId}>Loading the choices</span>
      <div className="ui-loading" role="status">
        <span className="sr-only">Loading the choices</span>
        <span aria-hidden="true" className="ui-skeleton" style={{ width: "40%" }} />
        <span aria-hidden="true" className="ui-skeleton" style={{ width: "70%" }} />
        <span aria-hidden="true" className="ui-skeleton" />
      </div>
    </Card>
  );
}

/**
 * "What do you want to know?" with the questions this machine can answer, as
 * buttons that put the question in the box (the AssistFirstRun board). The
 * reader still sends it, so nothing is asked on their behalf.
 */
export function AssistantAskSuggestions({
  onPick,
  prompts,
}: {
  onPick: (prompt: string) => void;
  prompts: string[];
}) {
  return (
    <section aria-labelledby="assistant-chat-title" className="as-welcome ui-card">
      <span aria-hidden="true" className="ui-row__icon"><Icon name="chat" size={16} /></span>
      <div className="as-welcome__body">
        <h2 id="assistant-chat-title">What do you want to know?</h2>
        {/*
          * These are the only invitation the machine-reading tools get. The
          * Assistant can read both compute engines and the suggestions were
          * built around the enclosure, so on a workstation nothing on screen
          * led anywhere near them.
          */}
        <div className="as-welcome__chips">
          {prompts.map((prompt) => (
            <Button key={prompt} onClick={() => onPick(prompt)} type="button">{prompt}</Button>
          ))}
        </div>
      </div>
    </section>
  );
}
