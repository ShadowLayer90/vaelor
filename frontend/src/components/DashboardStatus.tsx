import type { ReactNode } from "react";
import { StatusPill } from "./StatusPill";
import { fillTime, toneOf, type DashboardNode, type EngineBlock, type Freshness } from "../lib/performanceDashboard";

/** A freshness mark: the backend's label in the backend's tone, as the boards' outline pill. */
export function FreshnessBadge({ freshness }: { freshness: Freshness | undefined }) {
  if (!freshness?.label) return null;
  const tone = toneOf(freshness.tone);
  return <StatusPill className={`perf-badge perf-badge--${tone}`} label={freshness.label} tone={tone} />;
}

/** "Not reporting since 14:02": the backend's template with the local clock time, else its label. */
function nodeState(freshness: Freshness): string {
  return freshness.since_template ? fillTime(freshness.since_template, freshness.since) : freshness.label;
}

/**
 * The status card (VD-200, the ClusterPerformance board): the engine's pill
 * and the backend's sentences, each machine with its freshness pill, the
 * store's note, and - on the right - the Why the shell passes in. Every word
 * and every tone is the backend's; this component only places them.
 */
export function DashboardStatus({ engine, nodes, hostNote, why }: {
  engine: EngineBlock | undefined; nodes: DashboardNode[]; hostNote?: string; why?: ReactNode;
}) {
  const sentences = [engine?.reason, engine?.coverage_note, engine?.scope_note, engine?.routing_note]
    .filter((sentence): sentence is string => Boolean(sentence && sentence.trim()));
  const notes = [...nodes.flatMap((node) => [node.freshness?.reason, node.gpu_status_reason, node.clock_note]), hostNote]
    .filter((sentence): sentence is string => Boolean(sentence && sentence.trim()));
  return (
    <section className="ui-card perf-strip" aria-label="Serving status">
      <div className="perf-strip__main">
        <div className="perf-strip__lead">
          {engine?.badge && (
            <StatusPill className={`perf-badge perf-badge--${toneOf(engine.badge_tone)}`} label={engine.badge} tone={toneOf(engine.badge_tone)} />
          )}
          {sentences.length > 0 && <p className="perf-strip__text">{sentences.join(" ")}</p>}
        </div>
        {nodes.length > 0 && (
          <ul className="perf-strip__nodes" aria-label="Machines">
            {nodes.map((node, index) => (
              <li key={index}>
                <span className="perf-strip__node-name">{node.name}</span>
                {node.freshness && (
                  <StatusPill
                    className={`perf-strip__node-state perf-strip__node-state--${toneOf(node.freshness.tone)}`}
                    label={nodeState(node.freshness)}
                    tone={toneOf(node.freshness.tone)}
                  />
                )}
              </li>
            ))}
          </ul>
        )}
        {notes.length > 0 && <p className="perf-strip__notes">{[...new Set(notes)].join(" ")}</p>}
      </div>
      {why}
    </section>
  );
}
