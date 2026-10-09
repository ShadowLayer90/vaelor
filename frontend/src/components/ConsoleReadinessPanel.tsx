import { Icon, type IconName } from "./Icon";
import { StatusPill } from "./StatusPill";
import { ListRow, UnavailableValue } from "./ui";

export interface ConsoleCapabilities {
  video: { capture_detected: boolean; stream_ready: boolean };
  hid: { controller_available: boolean; configured: boolean; input_ready?: boolean };
  atx: { detected: boolean; actions: string[] };
  console_ready: boolean;
  readiness?: { state: "unavailable" | "not_configured" | "ready"; reason: string };
}

export interface ReadinessRow {
  icon: IconName;
  label: string;
  state: string;
  ready: boolean;
}

/**
 * The three readiness rows, from discovery.
 *
 * Home rendered `HDMI capture / USB HID / ATX control` from a hardcoded array
 * whose value cell was the literal string "Pending", with no fetch of
 * `/kvm/capabilities` anywhere on the page. A commissioned KVM read "Pending";
 * a machine that can never host one read "Pending". It looked like live
 * discovery and was a picture of one.
 */
export function consoleReadinessRows(capability: ConsoleCapabilities): ReadinessRow[] {
  return [
    {
      icon: "hdmi",
      label: "HDMI capture",
      ready: capability.video.capture_detected,
      state: capability.video.stream_ready
        ? "Streaming"
        : capability.video.capture_detected ? "Detected" : "Not connected",
    },
    {
      icon: "usb",
      label: "USB HID",
      ready: capability.hid.controller_available,
      state: capability.hid.input_ready ?? capability.hid.configured
        ? "Commissioned"
        : capability.hid.controller_available ? "Detected" : "Not connected",
    },
    {
      icon: "atx",
      label: "ATX control",
      ready: capability.atx.detected,
      state: capability.atx.detected
        ? `${capability.atx.actions.length} action${capability.atx.actions.length === 1 ? "" : "s"}`
        : "Not connected",
    },
  ];
}

export function consoleStatusLabel(capability: ConsoleCapabilities): string {
  if (capability.readiness?.state === "ready" || capability.console_ready) return "Ready";
  return capability.video.capture_detected || capability.hid.controller_available
    ? "Partly commissioned"
    : "Not commissioned";
}

/**
 * The KVM hardware path, on Remote console's Physical KVM view (VD-200, the
 * ConsoleKvm board): one row per half, read from `/kvm/capabilities`. On a
 * machine that will never have a capture card fitted the rows say "Not
 * connected" for good, which is a commissioning checklist rather than live
 * status - so it sits with the checklist, not on the console's first view.
 * It takes the capability the workspace already fetched rather than asking
 * again.
 */
export function ConsoleReadinessPanel({
  capability,
  failed = false,
}: {
  capability: ConsoleCapabilities | null;
  failed?: boolean;
}) {
  const unknownReason = "The remote-console hardware check did not respond, so this state is unknown";
  const label = capability ? consoleStatusLabel(capability) : failed ? "Status unknown" : "Checking hardware";
  const rows = capability
    ? consoleReadinessRows(capability)
    : ([["hdmi", "HDMI capture"], ["usb", "USB HID"], ["atx", "ATX control"]] as const)
      .map(([icon, rowLabel]) => ({ icon, label: rowLabel, state: "", ready: false }));

  return (
    <section
      aria-labelledby="console-readiness-heading"
      className={failed && !capability ? "card ui-card sys-card sys-card--refused" : "card ui-card sys-card"}
    >
      <header className="ui-card__header">
        <div className="ui-card__titles">
          <h2 id="console-readiness-heading">KVM hardware path</h2>
          <p>What discovery found on this machine</p>
        </div>
        <div className="ui-card__actions">
          <StatusPill
            label={label}
            reading={capability ? undefined : "unread"}
            tone={capability && label === "Ready" ? "success" : "neutral"}
          />
        </div>
      </header>
      <div className="ui-card__body ui-card__body--flush">
        {rows.map((row) => (
          <ListRow
            icon={<Icon name={row.icon} size={18} />}
            key={row.label}
            title={row.label}
            trailing={capability
              ? <span className="sys-mono">{row.state}</span>
              : failed
                ? <UnavailableValue label={`${row.label} state unknown`} mark="—" reason={unknownReason} />
                : <span>Checking…</span>}
          />
        ))}
        {/* Print the reason, not only in a tooltip: a tooltip is unreachable on touch. */}
        {!capability && failed && <p className="sys-refusal sys-refusal--last">{unknownReason}</p>}
      </div>
    </section>
  );
}
