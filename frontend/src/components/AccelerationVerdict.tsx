import {
  accelerationVerdict,
  basisNote,
  contextVerdict,
  type AccelerationReading,
  type ContextReading,
} from "../lib/acceleration";
import { StatusPill } from "./StatusPill";

/**
 * What the model server actually got, on a screen.
 *
 * Both readings answer "verify what you got, not that the process answers",
 * and until now neither had anywhere to be read: the only path to the owner
 * was a job-progress line at 95% that scrolled away.
 *
 * The three-state discipline is the whole design here. Each reading gets one
 * pill whose tone comes from the reading's own state — so `unknown` renders as
 * neutral rather than green, and `unattributed` renders as information rather
 * than a fault. The appliance's own `detail` sentence is printed verbatim
 * underneath, because it names the library that failed to resolve and quotes
 * the measurement behind the claim, and re-writing that here would be a second
 * place for the wording to drift from the reading.
 */

function Reading({
  detail,
  note,
  reference,
  title,
  verdict,
}: {
  detail: string;
  note?: string;
  reference?: Record<string, unknown> | null;
  title: string;
  verdict: { tone: Parameters<typeof StatusPill>[0]["tone"]; headline: string; state: string };
}) {
  return (
    <div className="acceleration-reading" data-state={verdict.state}>
      <div className="acceleration-reading__heading">
        <span>{title}</span>
        <StatusPill label={verdict.headline} tone={verdict.tone} />
      </div>
      {detail && <p>{detail}</p>}
      {note && <p className="acceleration-reading__basis">{note}</p>}
      {/*
        * A reference figure, and labelled as one. The backend renamed this key
        * from `measurement` to `reference_measurement` precisely because that
        * word, printed beside a deployment, reads as a reading of *this*
        * deployment — which it never was.
        */}
      {reference ? (
        <p className="acceleration-reading__reference">
          Reference measurement from a previous run on this hardware, shown as evidence for how
          this failure behaves. It is not a reading of the server beside it.
        </p>
      ) : null}
    </div>
  );
}

export function AccelerationVerdict({
  acceleration,
  context,
  title = "What this model server got",
  headingLevel = "h3",
}: {
  acceleration?: AccelerationReading | null;
  context?: ContextReading | null;
  title?: string;
  headingLevel?: "h3" | "h4";
}) {
  const accelerated = accelerationVerdict(acceleration);
  const built = contextVerdict(context);
  if (!accelerated && !built) return null;
  /* The level follows the host outline (#150): in workloads-jobs the parent
     heads with h2, so this is h3; ComputePanel nests it under an
     AcceleratorCard whose own title is h3, so there it must be h4 — one
     fixed level would read as a missing or duplicated section to a
     screen-reader outline in one of the two hosts. */
  const Heading = headingLevel;
  return (
    <section aria-label={title} className="acceleration-verdict">
      <Heading>{title}</Heading>
      {accelerated && acceleration && (
        <Reading
          detail={acceleration.detail}
          note={basisNote(acceleration)}
          reference={acceleration.reference_measurement}
          title="Accelerator"
          verdict={accelerated}
        />
      )}
      {built && context && (
        <Reading
          detail={context.detail}
          reference={context.reference_measurement}
          title="Context window"
          verdict={built}
        />
      )}
    </section>
  );
}
