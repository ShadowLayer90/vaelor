import { Icon, type IconName } from "./Icon";
import { StatusPill } from "./StatusPill";
import { UnavailableValue } from "./ui";

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
 * The KVM hardware path, where commissioning is the subject.
 *
 * This panel used to sit on Home and occupy a third of the lower grid. Its
 * three rows are honest — they are read from `/kvm/capabilities` — but on a
 * machine that will never have a capture card fitted they say "Not connected"
 * forever, which is a commissioning checklist rather than live status. It has
 * moved here whole, on both machine classes, and takes its own `capability`
 * from the workspace that was already fetching it rather than issuing a second
 * request for the same answer.
 */
export function ConsoleReadinessPanel({
  capability,
  failed = false,
}: {
  capability: ConsoleCapabilities | null;
  failed?: boolean;
}) {
  const unknownReason = "The remote-console hardware check did not respond, so this state is unknown";

  return (
    <section className="data-panel" aria-labelledby="console-readiness-heading">
      <div className="panel-heading">
        <div>
          <h2 id="console-readiness-heading">KVM hardware path</h2>
          <p>What discovery found on this machine</p>
        </div>
        <StatusPill
          status={capability && consoleStatusLabel(capability) === "Ready" ? "healthy" : "neutral"}
          label={capability ? consoleStatusLabel(capability) : failed ? "Status unknown" : "Checking hardware"}
        />
      </div>
      <div className="readiness-list">
        {capability
          ? consoleReadinessRows(capability).map((row) => (
            <div className="readiness-row" key={row.label}>
              <Icon name={row.icon} size={17} />
              <span>{row.label}</span>
              <strong>{row.state}</strong>
            </div>
          ))
          : ([["hdmi", "HDMI capture"], ["usb", "USB HID"], ["atx", "ATX control"]] as const).map(([icon, label]) => (
            <div className="readiness-row" key={label}>
              <Icon name={icon} size={17} />
              <span>{label}</span>
              <strong>
                {failed
                  ? (
                    <>
                      {/* Print the reason beside the mark, not only in its
                          tooltip: a tooltip is unreachable on touch and to
                          assistive tech, the same way AcceleratorCard does it. */}
                      <UnavailableValue label={`${label} state unknown`} reason={unknownReason} />
                      <small>{unknownReason}</small>
                    </>
                  )
                  : "Checking…"}
              </strong>
            </div>
          ))}
      </div>
    </section>
  );
}
