/**
 * What the local model is bad at, and that it sleeps when nobody is using it.
 *
 * **VD-071** - the model is recommended *with* its shortcomings, noted rather
 * than buried. That obligation is the reason this component still exists.
 *
 * **VD-073** - llama-server unloads the model's weights after fifteen minutes
 * of quiet and reloads them on the next question, which on the Pi returns
 * about 88% of its memory. VD-069 built a five-state machine, a shared record
 * between two processes, a sweeper thread and a chat guard to do that, and
 * this file rendered the states. All of it is gone: the container never stops,
 * so there is nothing to report - the first question after a sleep simply
 * takes about twenty seconds and is answered.
 *
 * What survives is VD-069's actual obligation, which was never the state
 * machine: **the period is stated rather than discovered.** An owner who waits
 * twenty seconds should already know why.
 */
import type { AssistantModelFacts } from "./agentTypes";

/**
 * The sleep period and the wake, as whole minutes and seconds, or `null` when
 * the server did not send both - the model then does not sleep, or nobody
 * measured its wake, and neither is a sentence to state.
 */
function sleepFacts(model?: AssistantModelFacts): { idleMinutes: number; wake: number } | null {
  const idleMinutes = model?.sleep_idle_seconds ? Math.round(model.sleep_idle_seconds / 60) : 0;
  // Rounded, because the record stores a measurement (37.0, 21.2) and a
  // sentence promising "21.2 seconds" claims a precision the next wake will
  // not honour.
  const wake = Math.round(model?.cold_start_seconds ?? 0);
  return idleMinutes > 0 && wake > 0 ? { idleMinutes, wake } : null;
}

function sleepSentence({ idleMinutes, wake }: { idleMinutes: number; wake: number }) {
  return `The local model unloads itself after ${idleMinutes} minutes without a question, to give the memory back `
    + `to the rest of the appliance. The next question loads it again and takes about ${wake} seconds; the ones `
    + "after that are normal speed.";
}

/**
 * The same sentence as one quiet line on Ask (owner, 2026-10-07: "Keep on
 * Ask"). The drawer keeps it too; the wait happens at the composer, so the
 * reason for it is stated there, not only behind Change intelligence.
 */
export function AssistantModelSleepNote(props: { model?: AssistantModelFacts }) {
  const facts = sleepFacts(props.model);
  return facts ? <p className="as-sleep-note">{sleepSentence(facts)}</p> : null;
}

export function AssistantModelState(props: { model?: AssistantModelFacts }) {
  const shortcomings = props.model?.shortcomings ?? [];
  const facts = sleepFacts(props.model);

  if (!shortcomings.length && !facts) return null;

  return (
    <div className="as-model-state" role="group" aria-label="Local model">
      {/*
        * Stated, not hidden. Rendered only when the server sent both numbers,
        * because a sentence about a duration nobody measured is the thing
        * VD-073's own row warns against.
        */}
      {facts && <p className="as-note">{sleepSentence(facts)}</p>}

      {shortcomings.length > 0 && (
        <details className="as-disclosure">
          <summary>What this model is not good at</summary>
          <ul>
            {shortcomings.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
