import { Icon, type IconName } from "./Icon";
import { StatusPill } from "./StatusPill";
import { LoadingLines } from "./ui";

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

/**
 * Remote console's "Remote access on this machine" (VD-200, the Console
 * board): a row per rung-bearing capability - what discovery established, in
 * words, and a pill naming who can move it up. Nothing here renders a control.
 */
export function ConsoleLadder({ rows }: { rows?: ConsoleLadderRow[] }) {
  return (
    <section aria-labelledby="console-ladder-heading" className="card ui-card sys-card">
      <header className="ui-card__header">
        <div className="ui-card__titles">
          <h2 id="console-ladder-heading">Remote access on this machine</h2>
        </div>
        <div className="ui-card__actions"><span>What discovery found, and who can change it</span></div>
      </header>
      <div className="ui-card__body ui-card__body--flush">
        {rows?.length
          ? rows.map((row) => (
            <div className="ui-row ladder-row" key={row.id}>
              <span aria-hidden="true" className="ui-row__icon"><Icon name={ROW_ICONS[row.id] ?? "shield"} size={18} /></span>
              <div className="ui-row__text">
                <div className="ui-row__title">{row.title}</div>
                {/*
                  * The state and the detail each have their own element: a
                  * positional rule once printed the detail on top of the title.
                  */}
                <div className="ui-row__detail">
                  <span className="ladder-row__state">{ladderStateLabel(row.state)}</span>
                  {" · "}
                  <span className="kvm-health__detail">{row.detail}</span>
                </div>
              </div>
              <div className="ui-row__trailing">
                <StatusPill label={row.actor_label} tone={row.actionable ? "success" : "neutral"} />
              </div>
            </div>
          ))
          : <LoadingLines label="Checking remote access" />}
      </div>
    </section>
  );
}
