import { useMemo, useState } from "react";
import type { Metrics, Session, TelemetrySample } from "../types";
import type { MachineProfile } from "../lib/machine";
import { sampleTimeMs } from "../lib/connectionState";
import { timeAgo } from "../lib/format";
import { MemoryOptimizer } from "./MemoryOptimizer";
import { ModalShell } from "./ModalShell";
import { overviewCards, type OverviewCard } from "./overviewCards";
import { Button, UnavailableValue } from "./ui";

/** How many bars a reading's trend draws: one per sample, the SystemCompute board's twelve. */
export const READING_BARS = 12;

/**
 * The time the drawn bars really cover, from the samples' own times - never a
 * window they do not reach. The board's "the last 30 seconds" is what twelve
 * samples 2.5 s apart come to; a gap (a hidden tab, a missed poll) says its
 * longer span, and too few samples say so.
 */
export function readingSpan(history: TelemetrySample[]): string {
  // A poll that returned the same sample again is not a second reading.
  const times = [...new Set(history.slice(-READING_BARS).map((sample) => sample.sampled_at).filter((at) => at > 0))];
  if (times.length === 0) return "no readings yet";
  if (times.length === 1) return "one reading";
  const seconds = Math.round((sampleTimeMs(times[times.length - 1]) - sampleTimeMs(times[0])) / 1000);
  if (seconds < 90) return `the last ${seconds} second${seconds === 1 ? "" : "s"}`;
  const minutes = Math.round(seconds / 60);
  return `the last ${minutes} minutes`;
}

/** "33.9 GB of 45.6 GB" -> ["33.9", "GB of 45.6 GB"]; "61°C" -> ["61", "°C"]. */
function splitReading(value: string): [string, string] {
  const match = value.match(/^(-?[\d.,]+)\s*(.*)$/);
  return match ? [match[1], match[2]] : [value, ""];
}

/**
 * The newest READING_BARS samples as fractions of the reading's scale, always
 * READING_BARS slots wide: `scale` when the reading has a natural one (a
 * percentage, degrees), otherwise a fifth above the largest sample shown. A
 * slot with no sample yet, or a sample that was not read, is an empty bar,
 * never a zero (LESSONS 1).
 */
export function readingBars(values: Array<number | null>, scale: number | null): Array<number | null> {
  const shown = values.slice(-READING_BARS);
  const measured = shown.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (!measured.length) return [];
  const top = scale ?? Math.max(...measured) * 1.2;
  const bars = shown.map((value) => (value === null || !Number.isFinite(value) ? null : top > 0 ? Math.max(0, value) / top : 0));
  return [...Array<null>(READING_BARS - bars.length).fill(null), ...bars];
}

function ReadingCard({ card }: { card: OverviewCard }) {
  const [number, unit] = splitReading(card.value);
  // Every series overviewCards draws is a percentage or degrees Celsius except
  // network throughput, which has no natural ceiling.
  const bars = card.unavailableReason ? [] : readingBars(card.values, card.label.startsWith("Network") ? null : 100);
  return (
    <article aria-label={card.label} className="ui-card reading-card" data-tone={card.tone}>
      <div className="ui-stat-tile__top"><span>{card.label}</span><span>{card.detail}</span></div>
      {card.unavailableReason
        ? <div className="ui-stat-tile__value--unread"><UnavailableValue label={`${card.label} unavailable`} reason={card.unavailableReason} /></div>
        : (
          // Drawn as a large number and a small unit; read out whole.
          <div className="reading-card__value">
            <span aria-hidden="true" className="ui-stat-tile__value">{number}</span>
            {unit && <span aria-hidden="true" className="reading-card__unit">{unit}</span>}
            <span className="sr-only">{card.value}</span>
          </div>
        )}
      {bars.length > 0 && (
        <div aria-hidden="true" className="ui-trend">
          {bars.map((bar, index) => (
            <i
              className={index === bars.length - 1 && bar !== null ? "is-latest" : bar === null ? "is-empty" : undefined}
              key={index}
              style={{ height: bar === null ? undefined : `${Math.max(6, Math.min(100, bar * 100))}%` }}
            />
          ))}
        </div>
      )}
      {card.actionLabel && card.onAction && (
        <div className="reading-card__action">
          <Button
            aria-label={card.actionDescription ? `${card.actionLabel} — ${card.actionDescription}` : card.actionLabel}
            onClick={card.onAction}
            title={card.actionDescription ?? card.actionLabel}
          >
            {card.actionLabel}
          </Button>
        </div>
      )}
    </article>
  );
}

/**
 * System › Compute › Live readings (VD-200, the SystemCompute board): the six
 * readings Home's telemetry wall used to show, each with its recent trend.
 * Tune memory sits on the Memory reading. Pausing and refreshing live
 * telemetry is the top bar's pill.
 */
export function SystemLiveReadings({
  history,
  machine,
  metrics,
  session,
  stale,
}: {
  history: TelemetrySample[];
  machine: MachineProfile;
  metrics: Metrics;
  session: Session;
  stale: boolean;
}) {
  const [showMemoryOptimizer, setShowMemoryOptimizer] = useState(false);
  const newest = history.at(-1)?.sampled_at ?? 0;
  const cards = useMemo(
    () => overviewCards({ metrics, history, machine, onTuneMemory: () => setShowMemoryOptimizer(true) }),
    [history, metrics, machine],
  );
  return (
    <section aria-labelledby="live-readings-heading" className={stale ? "live-readings live-readings--stale" : "live-readings"}>
      <div className="live-readings__heading">
        {/* Each bar is one reading, taken every 2.5 s while this page is open; the age is the newest one's. */}
        <h2 id="live-readings-heading">Live readings <span>· {readingSpan(history)}{newest > 0 && `, ${stale ? "last updated" : "updated"} ${timeAgo(sampleTimeMs(newest))}`}</span></h2>
        {/* The colours group the readings; they are not a severity scale. */}
        <span>Colours group the readings; they do not mean healthy or unhealthy.</span>
      </div>
      <div className="live-readings__grid" data-cards={cards.length}>
        {cards.map((card) => <ReadingCard card={card} key={card.label} />)}
      </div>
      {showMemoryOptimizer && <ModalShell labelledBy="memory-optimizer-title" onClose={() => setShowMemoryOptimizer(false)}>
        <MemoryOptimizer onClose={() => setShowMemoryOptimizer(false)} session={session} />
      </ModalShell>}
    </section>
  );
}
