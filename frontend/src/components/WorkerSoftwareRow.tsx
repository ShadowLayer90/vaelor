import { useEffect, useRef, useState } from "react";
import "../styles/cluster-fleet.css";
import { Button } from "./ui";
import { StatusPill } from "./StatusPill";
import type { StatusTone } from "./ui/status";
import type { ClusterMode } from "../lib/clusterMode";

/** One component's reading, as `worker_profile_state` words it (VD-194 P1). */
export interface WorkerSoftwareComponent {
  id: string;
  label: string;
  status: "matches" | "differs" | "unread" | "not-applicable";
  word: string;
  why: string;
  note: string;
  appliance_owned: boolean;
  /** What was read, where a component reports a value (amd-smi's version); "" otherwise. */
  measured: string;
  /** The whole sentence this row draws: the status, then each reason as a sentence. */
  words: string;
}

/** One labelled profile code; the row keeps them under a disclosure (owner check 2026-10-04). */
export interface ProfileCode {
  id: string;
  label: string;
  value: string;
}

/** One appliance unit the probe found; `state_words` is the backend's (VD-173). */
export interface WorkerSoftwareUnit {
  name: string;
  present: boolean;
  running: boolean;
  disabled: boolean;
  reload_needed: boolean;
  state_words: string;
}

/** Words for whoever may press Recheck, and for everyone else (VD-189). */
export interface RoleWords {
  administrator: string;
  other: string;
}

/** One step of the backend's age wording: below `below` seconds, the age / `unit` fills `{n}`. */
export interface AgeStep {
  below: number | null;
  unit: number;
  words: string;
}

/**
 * A worker's software against its controller's profile (VD-194 P1). Every
 * word is the backend's (`worker_profile_state.worker_software_view`, VD-173);
 * this row only lays them out and ages the reading on the box's clock.
 */
export interface WorkerSoftware {
  node_id: string;
  state: string;
  label: string;
  tone: StatusTone;
  exit?: string;
  exit_words?: RoleWords;
  sentence: string;
  checked_at: number | null;
  checked: string;
  /** The box's clock when the view was served (round 2 F4). */
  served_at?: number;
  /** The backend's age wording, which its own `checked` was written from. */
  age_words_table?: AgeStep[];
  stale: boolean;
  stale_sentence: string;
  /** How old a reading may be before the row says so (the backend's window). */
  stale_after_seconds?: number;
  old_reading_words?: RoleWords;
  /** What an undated reading reads as: never the state's own label or tone. */
  undated?: { label: string; tone: StatusTone; sentence: RoleWords; checked: string };
  /** Whether the worker matches what this controller expects, in one plain line. */
  profile_summary: string;
  /** The profile codes behind that line, each labelled, for the disclosure. */
  profile_codes?: { summary: string; rows: ProfileCode[] };
  components?: WorkerSoftwareComponent[];
  appliance?: { reading: string; sentence: string; units?: WorkerSoftwareUnit[] };
  attempt_note: string;
  /**
   * The newest worker-profile update (VD-194 P2): queued, running, or how the
   * last one ended, in the job's own words. "" when there has been none.
   */
  update_note?: string;
  /**
   * What a current reading reads as once the card has held it past the window
   * (round 3 R3-4): the backend's own old-reading pill. Null for other states.
   */
  when_old?: { label: string; tone: StatusTone; exit_words: RoleWords } | null;
}

/**
 * Why Review conversion is offered but cannot be pressed (owner, 2026-10-06:
 * keep the button, disabled with its reason, until conversion is built). The
 * words are the backend's `NOT_CONVERTIBLE_YET`, which every full-appliance
 * state's exit already carries.
 */
export const CONVERSION_UNAVAILABLE = "Converting this machine to the worker profile is not available yet";

/** The states in which the full appliance is on the worker: the ones a conversion is for. */
function applianceOnMachine(state: string): boolean {
  return state.startsWith("full-appliance");
}

/** How often the row re-reads its own age while no new payload arrives (round 2 F4). */
const AGE_TICK_MS = 60_000;

/**
 * G9 (VD-194, LESSONS 8): the row never claims anything about a worker
 * without the time it was read.
 */
function dated(software: WorkerSoftware): boolean {
  return typeof software.checked_at === "number" && software.checked_at > 0;
}

/** The backend's table applied to an age; null when the table is missing or does not fit. */
function ageWords(table: AgeStep[] | undefined, seconds: number): string | null {
  const step = table?.find((row) => row.below === null || seconds < row.below);
  if (!step) return null;
  return step.unit ? step.words.replace("{n}", String(Math.floor(seconds / step.unit))) : step.words;
}

/**
 * The reading's age on the box's clock (round 2 F4): how old it was when it
 * was served, plus how long the browser has held it since - so a browser whose
 * clock is wrong cannot make it older or younger. Null for a server that does
 * not send `served_at`; the row then shows the backend's words as they came.
 */
function useReadingAge(software: WorkerSoftware): number | null {
  const received = useRef<{ payload: WorkerSoftware; at: number } | null>(null);
  if (received.current?.payload !== software) received.current = { payload: software, at: Date.now() };
  const [, setTick] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(() => setTick((tick) => tick + 1), AGE_TICK_MS);
    return () => window.clearInterval(timer);
  }, []);
  // Round 3 R3-5: a served_at that cannot date the reading (missing, not
  // positive, or older than the reading itself) leaves the backend's words.
  const served = software.served_at;
  if (!dated(software) || typeof served !== "number" || served <= 0 || served < (software.checked_at as number)) {
    return null;
  }
  const held = Math.max(0, (Date.now() - received.current.at) / 1000);
  return served - (software.checked_at as number) + held;
}

export function WorkerSoftwareRow({ software, mode, administrator, variant = "full" }: {
  software: WorkerSoftware;
  mode: ClusterMode;
  /** Whether the viewer may press Recheck (it is administrator-only). */
  administrator: boolean;
  /**
   * `summary` is the collapsed machine card's row (pill, age and the one
   * plain line); `full` is the opened card's, with the sentence, and in
   * Advanced the component rows, the services found and the profile codes.
   */
  variant?: "summary" | "full";
}) {
  const advanced = mode === "advanced";
  const full = variant === "full";
  const age = useReadingAge(software);
  const role = administrator ? "administrator" : "other";
  const description = software.exit_words ? software.exit_words[role] : software.exit;
  const title = full
    ? <span className="cf-software__title">Worker software</span>
    : <span className="cf-label">Worker software</span>;
  if (!dated(software)) {
    const undated = software.undated;
    return (
      <section className="cf-software" aria-label="Worker software">
        <div className="cf-software__head">
          {title}
          {undated && <StatusPill tone={undated.tone} label={undated.label} description={description} />}
          {undated && <span className="cf-note">{undated.checked}</span>}
        </div>
        {undated && <p className="cf-software__sentence">{undated.sentence[role]}</p>}
        {software.attempt_note && <p className="cf-note">{software.attempt_note}</p>}
        {software.update_note && <p className="cf-software__update">{software.update_note}</p>}
      </section>
    );
  }
  const staleAfter = software.stale_after_seconds;
  // Round 4 N5: the backend's flag, or the row's own age - either says old.
  // served_at is whole seconds, so at 900.x s the row alone would round under.
  const old = software.stale || (age !== null && typeof staleAfter === "number" && age > staleAfter);
  const checked = (age === null ? null : ageWords(software.age_words_table, age)) ?? software.checked;
  const oldWords = software.old_reading_words ? software.old_reading_words[role] : software.stale_sentence;
  // Round 3 R3-4: once the row's own age is past the window, the pill reads
  // the backend's old-reading words, never "Up to date" beside "reading is old".
  const aged = old && !software.stale && software.when_old ? software.when_old : null;
  const pillLabel = aged ? aged.label : software.label;
  const pillTone = aged ? aged.tone : software.tone;
  const pillDescription = aged ? aged.exit_words[role] : description;
  const components = software.components ?? [];
  const units = (software.appliance?.units ?? []).filter((unit) => unit.present);
  const codes = software.profile_codes?.rows ?? [];
  return (
    <section className="cf-software" aria-label="Worker software">
      <div className="cf-software__head">
        {title}
        <StatusPill tone={pillTone} label={pillLabel} description={pillDescription} />
        <span className="cf-note">{checked}</span>
        {old && oldWords && <span className="cf-note cf-software__stale">{oldWords}</span>}
      </div>
      {full && software.sentence && <p className="cf-software__sentence">{software.sentence}</p>}
      {software.profile_summary && <p className="cf-note">{software.profile_summary}</p>}
      {software.attempt_note && <p className="cf-note">{software.attempt_note}</p>}
      {software.update_note && <p className="cf-software__update">{software.update_note}</p>}
      {full && advanced && components.length > 0 && (
        <dl className="cf-software__components">
          {components.map((component) => (
            <div key={component.id} className="cf-software__component">
              <dt>{component.label}</dt>
              <dd className={`cf-software__status--${component.status}`}>
                {component.measured && <span className="cf-software__measured">{component.measured} </span>}
                <span>{component.words}</span>
              </dd>
            </div>
          ))}
        </dl>
      )}
      {full && advanced && units.length > 0 && (
        <p className="cf-note">
          {"Appliance services found: "}
          {units.map((unit, index) => (
            <span key={unit.name}>
              {index > 0 && " · "}
              <span>{`${unit.name}: ${unit.state_words}`}</span>
            </span>
          ))}
        </p>
      )}
      {full && advanced && codes.length > 0 && (
        <details className="cf-software__codes">
          <summary>{software.profile_codes?.summary}</summary>
          <dl>
            {codes.map((code) => (
              <div key={code.id}>
                <dt>{code.label}</dt>
                <dd>{code.value}</dd>
              </div>
            ))}
          </dl>
        </details>
      )}
      {/* Owner, 2026-10-06: conversion is kept on the row, offered and
          disabled with its reason, until converting a machine is built. */}
      {full && applianceOnMachine(software.state) && (
        <div className="cf-software__convert">
          <Button disabled disabledReason={CONVERSION_UNAVAILABLE}>Review conversion</Button>
        </div>
      )}
    </section>
  );
}
