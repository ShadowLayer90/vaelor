import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import { Button } from "./ui";
import { consoleSessionAvailable, type ConsoleLadderRow } from "./ConsoleLadder";
import type { RemoteSessionState } from "./remoteSessionState";
import { sessionStateLabel } from "./remoteSessionState";

export interface KvmCapabilities {
  video: {
    capture_detected: boolean;
    capture_devices: Array<{ name: string; device: string }>;
    stream_ready: boolean;
    stream_configured?: boolean;
  };
  hid: {
    controller_available: boolean;
    configured: boolean;
    input_ready?: boolean;
    controllers: string[];
  };
  atx: { detected: boolean; actions: string[] };
  virtual_media: { available: boolean; reason: string };
  console_ready: boolean;
  readiness?: {
    state: "unavailable" | "not_configured" | "ready";
    ready: boolean;
    video_ready: boolean;
    input_ready: boolean;
    reason: string;
  };
  ladder?: ConsoleLadderRow[];
  commissioning: Array<{ id: string; complete: boolean; title: string; detail: string }>;
  control: { owner?: string | null; expires_at?: number | null };
}

/**
 * The physical-KVM ladder needs its own labels, and this is not a cosmetic
 * split.
 *
 * `sessionStateLabel` renders `not_configured` as "Needs setup", which is
 * truthful for the browser desktop above it — there is an "Install browser
 * desktop" button, and Vaelor really can walk that path. It is not truthful
 * here. Every remaining step on the physical-KVM path happens at the machine:
 * plugging in capture hardware, commissioning a USB gadget, connecting ATX
 * leads. On a workstation whose out-of-band management would be provisioned by
 * pressing F3 at POST, "Needs setup" promises a path the software does not
 * have.
 *
 * The five-rung `absent`/`advertised`/`present`/`reachable`/`provisioned`
 * ladder with its remediation `actor` now lives beside this, in
 * `ConsoleLadder`. These two labels are what the stage's own summary reads,
 * and they stay because they are the honest short form of the same fact.
 */
export function physicalKvmStateLabel(state: RemoteSessionState) {
  if (state === "not_configured") return "Needs setup at the machine";
  return sessionStateLabel(state);
}

export function physicalKvmState(
  capability: KvmCapabilities | null,
  setupRequested: boolean,
): RemoteSessionState {
  if (setupRequested) return "configuring";
  if (!capability) return "unavailable";
  if (capability.readiness?.ready ?? capability.console_ready) return "ready";
  return capability.video.capture_detected || capability.hid.controller_available
    ? "not_configured"
    : "unavailable";
}

export function PhysicalKvmStage({
  busy,
  canControl,
  capability,
  onControl,
  state,
  username,
}: {
  busy: string;
  canControl: boolean;
  capability: KvmCapabilities | null;
  onControl: (action: "acquire" | "release") => void;
  state: RemoteSessionState;
  username: string;
}) {
  /*
   * The one gate on the control button, and it is the ladder rather than a
   * second reading of the same snapshot. A row that reports anything below
   * `provisioned` while a Connect control still renders is exactly the
   * mismatch the ladder exists to remove, so the button is derived from the
   * rows the screen is showing. Until the ladder arrives the answer is "no
   * session", which is the fail-safe direction.
   */
  const sessionAvailable = consoleSessionAvailable(capability?.ladder);
  const owned = capability?.control.owner === username;

  return (
    <section
      className={`kvm-stage ${sessionAvailable ? "kvm-stage--ready" : ""}`}
      aria-labelledby="physical-kvm-title"
    >
      <div className="kvm-stage__screen">
        <span className="kvm-scanline" />
        <div className="kvm-stage__content">
          <Icon name="hdmi" size={48} />
          <small>PHYSICAL KVM VIDEO</small>
          <strong id="physical-kvm-title">
            {
              capability?.video.stream_ready
                ? capability.video.capture_devices[0]?.name || "Protected video stream ready"
                : capability?.video.capture_detected
                  ? "Capture device needs streaming setup"
                  : "No capture adapter detected"
            }
          </strong>
          <p>
            {sessionAvailable
              ? "Video and isolated input are ready for a protected control session."
              : capability?.video.capture_detected
                ? "The adapter is visible, but physical KVM stays locked until its protected stream and isolated USB input bridge are commissioned."
                : capability && !capability.hid.controller_available
                  /*
                   * Do not advise buying capture hardware here. Physical KVM
                   * needs both halves, and the keyboard half emulates a USB
                   * device — which requires a USB device controller this
                   * machine does not have. Discovery found no controller at
                   * all, so a capture adapter bought on the strength of that
                   * sentence would move the state from "unavailable" to
                   * "needs setup" and then stop, with no remaining step that
                   * could ever complete.
                   */
                  ? "Physical KVM needs a capture device and a USB device controller that can emulate a keyboard. Neither was found on this machine, and no capture hardware adds a USB device controller — so this stays unavailable here."
                  : "Physical KVM remains optional. Add supported HDMI capture and isolated USB control hardware when you need pre-boot access."}
          </p>
          {!sessionAvailable && (
            <a className="ui-button ui-button--quiet" href="#physical-kvm-setup">View hardware checklist</a>
          )}
        </div>
      </div>
      <aside className="kvm-stage__controls" aria-label="Physical KVM readiness">
        <div className="kvm-health">
          <span><Icon name="hdmi" /><small>Video input</small></span>
          <strong>
            {
              capability?.video.stream_ready
                ? "Protected capture stream ready"
                : capability?.video.capture_detected
                  ? "Video device detected; stream setup required"
                  : "No capture adapter detected"
            }
          </strong>
          <StatusPill
            label={capability?.video.stream_ready ? "Ready" : capability?.video.capture_detected ? "Needs setup" : "Not connected"}
            status={capability?.video.stream_ready ? "healthy" : "neutral"}
          />
        </div>
        <div className="kvm-health">
          <span><Icon name="usb" /><small>Keyboard and mouse</small></span>
          <strong>
            {capability?.hid.input_ready
              ? "Isolated USB input bridge ready"
              : capability?.hid.configured
                ? "USB HID found; input bridge required"
              : capability?.hid.controller_available
                ? "USB controller found; setup required"
                : "No compatible USB controller"}
          </strong>
          <StatusPill
            label={capability?.hid.input_ready ? "Ready" : capability?.hid.controller_available ? "Needs setup" : "Unavailable"}
            status={capability?.hid.input_ready ? "healthy" : "neutral"}
          />
        </div>
        <div className="kvm-health">
          <span><Icon name="atx" /><small>Target power</small></span>
          <strong>{capability?.atx.detected ? "ATX controls commissioned" : "Optional ATX leads not detected"}</strong>
          <StatusPill label={capability?.atx.detected ? "Ready" : "Not connected"} status={capability?.atx.detected ? "healthy" : "neutral"} />
        </div>
        <div className="kvm-owner">
          <small>PHYSICAL KVM STATE</small>
          <strong>{physicalKvmStateLabel(state)}</strong>
          <span>{capability?.readiness?.reason || (sessionAvailable ? "Video and isolated keyboard/mouse input are ready." : "Control unlocks after video and USB input pass discovery.")}</span>
          <small>CONTROL SESSION</small>
          <strong>{capability?.control.owner ? `Controlled by ${capability.control.owner}` : "View only"}</strong>
          <span>{owned ? "Keyboard and mouse control is active for this session." : "Request control to prove keyboard and mouse input ownership."}</span>
          {sessionAvailable && canControl && (
            <Button variant="primary"
              disabled={Boolean(busy)}
              onClick={() => onControl(owned ? "release" : "acquire")}
              type="button"
            >
              {owned ? "Release control" : "Request control"}
            </Button>
          )}
        </div>
      </aside>
    </section>
  );
}
