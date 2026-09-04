import { Icon, type IconName } from "./Icon";
import { StatusPill } from "./StatusPill";

/**
 * The five-rung remote-console ladder, rendered (task #54).
 *
 * `remote_console` used to be one thing with three states — `ready`,
 * `not_configured`, `unavailable` — and three genuinely different conditions
 * were collapsed into the last two. Measured on the Z2 Mini G1a on 2026-08-07:
 * the HP Remote System Controller is not fitted, and AMD DASH is enabled in
 * firmware while TCP 623/664 are silently swallowed by the network
 * controller's management engine. Advertised, present, unreachable and
 * unprovisioned are four separate facts, and squeezing them into "Needs setup"
 * promises a path Vaelor can walk when the next step is a person pressing F3 at
 * POST.
 *
 * Every row therefore carries `actor` — who can move it up — and `actionable`,
 * which the backend derives from the state rather than reporting separately.
 * Nothing here renders a control: a row is informational until its own state
 * says otherwise, and the one place that decides is `capability()` in
 * `vaelor/console_capabilities.py`.
 */
export type ConsoleRung = "absent" | "advertised" | "present" | "reachable" | "provisioned";

export type ConsoleActor =
  | "vaelor" | "operator-remote" | "operator-onsite" | "hardware" | "none";

export interface ConsoleLadderRow {
  id: string;
  title: string;
  state: ConsoleRung;
  rung: number;
  actor: ConsoleActor;
  actor_label: string;
  detail: string;
  actionable: boolean;
  reachable?: boolean | null;
}

const ROW_ICONS: Record<string, IconName> = {
  "console.video": "hdmi",
  "console.input": "usb",
  "power.oob": "atx",
};

/**
 * Deliberately not "Needs setup".
 *
 * `advertised` and `present` are the two rungs the old model had no word for,
 * and both of them read as an unfinished checklist under the old labels. They
 * are states of the world, not steps in a flow.
 */
export function ladderStateLabel(state: ConsoleRung): string {
  return {
    absent: "Not available on this machine",
    advertised: "Advertised in firmware, never contacted",
    present: "Present, not provisioned",
    reachable: "Answering, awaiting credentials",
    provisioned: "Ready",
  }[state] ?? "Unknown";
}

/** Whether both halves of a screen-and-keyboard session are at rung 4. */
export function consoleSessionAvailable(rows: ConsoleLadderRow[] | undefined): boolean {
  const console_ = (rows ?? []).filter((row) => row.id.startsWith("console."));
  return console_.length > 0 && console_.every((row) => row.actionable);
}

export function ConsoleLadder({ rows }: { rows?: ConsoleLadderRow[] }) {
  return (
    <section className="data-panel" aria-labelledby="console-ladder-heading">
      <div className="panel-heading">
        <div>
          <h2 id="console-ladder-heading">Remote access on this machine</h2>
          <p>What discovery established, and who can change it</p>
        </div>
        <Icon name="shield" />
      </div>
      <div className="readiness-list">
        {rows?.length
          ? rows.map((row) => (
            <div className="kvm-health" key={row.id}>
              <span>
                <Icon name={ROW_ICONS[row.id] ?? "shield"} />
                <small>{row.title}</small>
              </span>
              <strong>{ladderStateLabel(row.state)}</strong>
              {/*
                * Classed, because the rule that places this grid targets
                * `> span` and there are two of them: the detail landed in the
                * title's cell and printed on top of it. A positional selector
                * would work and would break again the moment a third element
                * is added, so each cell names itself.
                */}
              <span className="kvm-health__detail">{row.detail}</span>
              <StatusPill
                label={row.actor_label}
                status={row.actionable ? "healthy" : "neutral"}
              />
            </div>
          ))
          : (
            <div className="readiness-row">
              <Icon name="shield" size={17} />
              <span>Remote access</span>
              <strong>Checking…</strong>
            </div>
          )}
      </div>
    </section>
  );
}
