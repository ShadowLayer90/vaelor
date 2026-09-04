import { Icon, type IconName } from "./Icon";
import { UnavailableValue } from "./ui";

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
}

export function AcceleratorCard({
  icon,
  title,
  subtitle,
  facts,
  children,
}: {
  icon: IconName;
  title: string;
  subtitle?: string | null;
  facts: EngineFact[];
  children?: React.ReactNode;
}) {
  return (
    <section className="engine-card" aria-labelledby={`engine-${title.replace(/\W+/g, "-").toLowerCase()}`}>
      <div className="panel-heading">
        <div>
          <h3 id={`engine-${title.replace(/\W+/g, "-").toLowerCase()}`}>{title}</h3>
          {subtitle && <p>{subtitle}</p>}
        </div>
        <Icon name={icon} />
      </div>
      <dl className="engine-facts">
        {facts.map((fact) => (
          <div key={fact.label}>
            <dt>{fact.label}</dt>
            <dd>
              {fact.value === null
                ? <UnavailableValue
                  label={`${fact.label} unavailable`}
                  reason={fact.reason ?? "This device does not report that value"}
                />
                : fact.value}
              {/*
                * The reason is printed, not only hovered. Four `?` marks on
                * this panel told a sighted reader nothing at all, and a
                * tooltip is a poor place for the only copy of a fact — it
                * cannot be read on a touch screen, cannot be selected, and
                * disappears the moment the pointer moves.
                */}
              {fact.value === null && (
                <small>{fact.reason ?? "This device does not report that value"}</small>
              )}
              {/*
                * Both, when there are both. "Why is there no reading" and "why
                * is this pool near-empty" are different questions, and an
                * unreadable value does not retire the explanation of the thing
                * it would have measured.
                */}
              {fact.note && <small>{fact.note}</small>}
            </dd>
          </div>
        ))}
      </dl>
      {children}
    </section>
  );
}
