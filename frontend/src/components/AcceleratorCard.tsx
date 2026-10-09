import { useState } from "react";
import { Icon, ICON_SIZE, type IconName } from "./Icon";
import { Button, UnavailableValue } from "./ui";

/**
 * One compute engine, with `[?]` for every field it does not measure.
 *
 * The rule this card exists to enforce: **no reading is shown for hardware
 * that did not produce it.** The neural accelerator reports no power at all —
 * `xrt-smi` prints `Estimated Power: N/A` — so its power row is an
 * unavailable value carrying that sentence, never a zero and never an em dash
 * that reads as a measurement. Mesa is genuinely unreadable headless. The
 * firmware note names the file that was read. Every reason here comes from the
 * driver; none is written at the call site.
 */
export interface EngineFact {
  label: string;
  value: string | null;
  /** Required whenever `value` is `null`. Shown on the `[?]` itself. */
  reason?: string;
  /** Extra prose under the row, for a fact that needs explaining. */
  note?: string;
  /** The longer reason behind the note, kept behind a "Why" disclosure so the cell stays short. */
  why?: string;
}

/**
 * A part of the machine as a row of System › Compute › Hardware (VD-200, the
 * SystemCompute board): its icon, its name and a one-line summary; opened in
 * place, every fact as a key/value grid. The name is the heading and the
 * heading's button opens it (the accordion pattern), so a screen reader hears
 * "Graphics, collapsed" and the summary is read after it. The details stay in
 * the page while closed, hidden, so nothing is fetched or lost on opening.
 */
export function AcceleratorCard({
  icon,
  title,
  subtitle,
  summary,
  facts,
  children,
  defaultOpen = false,
}: {
  icon: IconName;
  title: string;
  subtitle?: string | null;
  /** The one line under the name; the parts that matter at a glance. */
  summary?: string;
  facts: EngineFact[];
  children?: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const slug = title.replace(/\W+/g, "-").toLowerCase();
  const detailsId = `engine-${slug}-details`;
  return (
    <section className={open ? "hw-row hw-row--open" : "hw-row"} aria-labelledby={`engine-${slug}`}>
      <div className="hw-row__head">
        <span aria-hidden="true" className="ui-row__icon"><Icon name={icon} size={ICON_SIZE.nav} /></span>
        <div className="hw-row__text">
          <h3 id={`engine-${slug}`}>
            <Button
              aria-controls={detailsId}
              aria-expanded={open}
              className="hw-row__toggle"
              onClick={() => setOpen((value) => !value)}
              variant="quiet"
            >
              {title}
            </Button>
          </h3>
          {(summary || subtitle) && <p className="hw-row__summary">{summary || subtitle}</p>}
        </div>
        <span aria-hidden="true" className="hw-row__chevron"><Icon name="chevron" size={ICON_SIZE.inline} /></span>
      </div>
      <div className="hw-row__details" hidden={!open} id={detailsId}>
        {subtitle && summary && subtitle !== summary && !summary.startsWith(subtitle) && <p className="hw-row__subtitle">{subtitle}</p>}
        <FactGrid facts={facts} />
        {children}
      </div>
    </section>
  );
}

/**
 * A part's facts as the board's key/value grid. A value that was not read
 * says "Not read", keeps its [?] name and tooltip, and prints its reason
 * under it; the Hardware rows and the Accelerators cards both draw this.
 */
export function FactGrid({ facts }: { facts: EngineFact[] }) {
  return (
    <dl className="engine-facts">
      {facts.map((fact) => (
        <div key={fact.label}>
          <dt>{fact.label}</dt>
          <dd>
            {fact.value === null
              ? <UnavailableValue
                className="engine-facts__unread"
                mark="Not read"
                label={`${fact.label} unavailable`}
                reason={fact.reason ?? "This device does not report that value"}
              />
              : fact.value}
            {/*
              * The reason is printed, not only hovered: a tooltip is a poor
              * place for the only copy of a fact - it cannot be read on a
              * touch screen and disappears the moment the pointer moves.
              */}
            {fact.value === null && (
              <small>{fact.reason ?? "This device does not report that value"}</small>
            )}
            {/* Both, when there are both: why there is no reading, and what the reading means. */}
            {fact.note && <small>{fact.note}</small>}
            {fact.why && (
              <details className="engine-facts__why">
                <summary>Why</summary>
                <small>{fact.why}</small>
              </details>
            )}
          </dd>
        </div>
      ))}
    </dl>
  );
}
