import { useEffect, useMemo, useState } from "react";
import "../styles/apps-setup.css";
import {
  canonicalOperationState,
  jobIsReady,
  jobIsTerminal,
  jobLabel,
  jobNeedsAttention,
  jobStateLabel,
  jobSummary,
} from "../lib/jobPresentation";
import { AppsDialog, AppsProgress } from "./appsKit";
import { AppsBanner } from "./appsSetupParts";
import { StatusPill } from "./StatusPill";
import { Button } from "./ui";
import { joinClassNames } from "./ui/field";
import type { UpdateJob } from "./UpdateJobStatus";

/*
 * Setting up the on-device Assistant (the lower half of the
 * AppsAssistantSetup board): the NPU model install's live timeline. Shared by
 * every surface that starts the install (`useNpuInstall`), so it assumes
 * nothing about the page that opened it.
 *
 * A lost status read is said (LESSONS 8 / VD-189), and the last bar read is
 * drawn grey and aged in words rather than frozen as if it were current.
 */

const clampProgress = (value: number) => Math.max(0, Math.min(100, Math.round(value || 0)));

/** How old the last reading is, in the words the board uses ("1 min old"). */
function readingAge(since: number, now: number) {
  const minutes = Math.max(1, Math.round((now - since) / 60000));
  return `${minutes} min old`;
}

/** Each step's dot: the step's own state, so a failed step is red in a running list. */
function stepTone(state: string) {
  const value = (state ?? "").toLowerCase();
  if (["failed", "rejected"].includes(value)) return "danger";
  if (["completed", "healthy"].includes(value)) return "success";
  if (["cancelled", "superseded"].includes(value)) return "neutral";
  return "info";
}

export function NpuInstallDialog({
  job,
  onDismiss,
  onRetryRead,
  readLost,
}: {
  job: UpdateJob;
  readLost: boolean;
  onRetryRead: () => void;
  onDismiss: () => void;
}) {
  const terminal = jobIsTerminal(job);
  const failed = terminal && (jobNeedsAttention(job) || canonicalOperationState(job) === "rejected");
  const succeeded = terminal && jobIsReady(job) && !failed;
  const progress = clampProgress(succeeded ? 100 : job.progress);
  const title = jobLabel(job.type);
  // Newest first, so the current step is in view without scrolling.
  const steps = useMemo(() => [...(job.events ?? [])].reverse(), [job.events]);

  // When this job last changed: the age of what is shown once reading stops.
  const [lastRead, setLastRead] = useState(() => Date.now());
  useEffect(() => { setLastRead(Date.now()); }, [job]);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!readLost) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 30000);
    return () => window.clearInterval(timer);
  }, [readLost]);

  const pill = readLost && !terminal
    ? <StatusPill label={`${progress}% · ${readingAge(lastRead, now)}`} reading="stale" />
    : <StatusPill label={jobStateLabel(job).replace(/^./, (letter) => letter.toUpperCase())} tone={failed ? "danger" : succeeded ? "success" : "info"} />;

  return (
    <AppsDialog
      eyebrow="On-device Assistant"
      // One footer button for the one thing it does: Close while the install
      // runs on, Dismiss once it has ended - never both side by side.
      footer={<Button onClick={onDismiss}>{terminal ? "Dismiss" : "Close"}</Button>}
      onClose={onDismiss}
      title="Setting up the on-device Assistant"
      titleId="npu-install-title"
    >
      {readLost && !terminal && (
        <AppsBanner
          action={(
            <>
              <Button onClick={onRetryRead}>Retry reading</Button>
              <a className="apps-link" href="#/activity">Open Activity</a>
            </>
          )}
          tone="warning"
        >
          Vaelor could not read this install's state for about a minute, so the progress shown may be out of date. The install keeps running on the appliance.
        </AppsBanner>
      )}
      {!terminal && !readLost && (
        <p>Vaelor is preparing the on-device model (downloading it first if it is not already on the machine), verifying it, and starting the Assistant on the neural processor. This keeps running if you close this - reopen the Assistant setup to check on it.</p>
      )}
      <section
        aria-busy={!terminal || undefined}
        aria-label={title}
        className={joinClassNames("apps-npu-job", failed && "apps-npu-job--failed")}
      >
        <div className="apps-npu-job__head">
          <strong>{title}</strong>
          {pill}
        </div>
        {!failed && (
          <div className="apps-npu-job__bar">
            <AppsProgress fraction={progress / 100} label={`${title} progress`} stale={readLost && !terminal} />
            {!terminal && !readLost && <span className="apps-npu-job__percent">{progress}%</span>}
          </div>
        )}
        {failed && <p className="apps-npu-job__error" role="alert">{jobSummary(job)}</p>}
        {succeeded && <p className="apps-fineprint">The Assistant answers on the neural processor.</p>}
        {!terminal && !readLost && steps.length > 0 && (
          <ol className="apps-npu-job__steps">
            {steps.map((step, index) => (
              <li key={step.id ?? `${step.state}-${index}`}>
                <span aria-hidden="true" className={`apps-npu-job__dot apps-npu-job__dot--${stepTone(step.state)}`} />
                <span className="apps-npu-job__step">{step.message || jobStateLabel({ state: step.state })}</span>
                <span className="apps-npu-job__step-percent">{clampProgress(step.progress)}%</span>
              </li>
            ))}
          </ol>
        )}
      </section>
    </AppsDialog>
  );
}
