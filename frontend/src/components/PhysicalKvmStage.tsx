import { Icon, ICON_SIZE } from "./Icon";
import { StatusPill } from "./StatusPill";
import { SectionCard } from "./systemUi";
import { Button, Notice } from "./ui";
import { consoleSessionAvailable, type ConsoleLadderRow } from "./ConsoleLadder";
import type { RemoteSessionState } from "./remoteSessionState";
import { sessionStateLabel } from "./remoteSessionState";

/** The checklist card's id on the Physical KVM view. */
export const PHYSICAL_KVM_CHECKLIST_ID = "physical-kvm-setup";

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
 * truthful for the browser desktop — there is an "Install browser desktop"
 * button, and Vaelor really can walk that path. It is not truthful here.
 * Every remaining step on the physical-KVM path happens at the machine:
 * plugging in capture hardware, commissioning a USB gadget, connecting ATX
 * leads.
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

/** The card's title: what the capture side is, in words. */
function videoTitle(capability: KvmCapabilities | null) {
  return capability?.video.stream_ready
    ? capability.video.capture_devices[0]?.name || "Protected video stream ready"
    : capability?.video.capture_detected
      ? "Capture device needs streaming setup"
      : "No capture adapter detected";
}

/** The sentence inside the dashed screen. */
function screenSentence(capability: KvmCapabilities | null, sessionAvailable: boolean) {
  if (sessionAvailable) return "Video and isolated input are ready for a protected control session.";
  if (capability?.video.capture_detected) {
    return "The adapter is visible, but physical KVM stays locked until its protected stream and isolated USB input bridge are commissioned.";
  }
  if (capability && !capability.hid.controller_available) {
    /*
     * Do not advise buying capture hardware here. Physical KVM needs both
     * halves, and the keyboard half emulates a USB device - which requires a
     * USB device controller this machine does not have. A capture adapter
     * bought on the strength of that advice would move the state from
     * "unavailable" to "needs setup" and then stop for good.
     */
    return "Physical KVM needs a capture device and a USB device controller that can emulate a keyboard. Neither was found on this machine, and no capture hardware adds a USB device controller — so this stays unavailable here.";
  }
  return "Physical KVM remains optional. Add supported HDMI capture and isolated USB control hardware when you need pre-boot access.";
}

interface HardwareRow {
  key: string;
  label: string;
  value: string;
  pill: { label: string; tone: "success" | "neutral" };
}

function hardwareRows(capability: KvmCapabilities | null): HardwareRow[] {
  return [
    {
      key: "video",
      label: "Video input",
      value: capability?.video.stream_ready
        ? "Protected capture stream ready"
        : capability?.video.capture_detected
          ? "Video device detected; stream setup required"
          : "No capture adapter detected",
      pill: {
        label: capability?.video.stream_ready ? "Ready" : capability?.video.capture_detected ? "Needs setup" : "Not connected",
        tone: capability?.video.stream_ready ? "success" : "neutral",
      },
    },
    {
      key: "input",
      label: "Keyboard and mouse",
      value: capability?.hid.input_ready
        ? "Isolated USB input bridge ready"
        : capability?.hid.configured
          ? "USB HID found; input bridge required"
          : capability?.hid.controller_available
            ? "USB controller found; setup required"
            : "No compatible USB controller",
      pill: {
        label: capability?.hid.input_ready ? "Ready" : capability?.hid.controller_available ? "Needs setup" : "Unavailable",
        tone: capability?.hid.input_ready ? "success" : "neutral",
      },
    },
    {
      key: "power",
      label: "Target power",
      value: capability?.atx.detected ? "ATX controls commissioned" : "Optional ATX leads not detected",
      pill: {
        label: capability?.atx.detected ? "Ready" : "Not connected",
        tone: capability?.atx.detected ? "success" : "neutral",
      },
    },
  ];
}

/**
 * Remote console's Physical KVM video card (VD-200, the Console board): what
 * the capture side is, the three halves of the hardware path, and the door to
 * the hardware checklist, which is its own view now.
 */
export function KvmVideoCard({
  capability,
  onOpenChecklist,
  state,
}: {
  capability: KvmCapabilities | null;
  onOpenChecklist: () => void;
  state: RemoteSessionState;
}) {
  const sessionAvailable = consoleSessionAvailable(capability?.ladder);
  const pill = sessionAvailable
    ? { label: "Ready", tone: "success" as const }
    : state === "unavailable" ? { label: "Optional", tone: "neutral" as const } : { label: physicalKvmStateLabel(state), tone: "neutral" as const };
  return (
    <section
      aria-labelledby="physical-kvm-title"
      className={sessionAvailable ? "card ui-card sys-card kvm-card kvm-card--ready" : "card ui-card sys-card kvm-card"}
    >
      <header className="ui-card__header">
        <div className="ui-card__titles">
          <span className="sys-eyebrow">Physical KVM video</span>
          <h2 id="physical-kvm-title">{videoTitle(capability)}</h2>
        </div>
        <div className="ui-card__actions"><StatusPill label={pill.label} tone={pill.tone} /></div>
      </header>
      <div className="ui-card__body sys-stack">
        <div className="kvm-screen">
          <Icon name="display" size={ICON_SIZE.standalone} />
          <p>{screenSentence(capability, sessionAvailable)}</p>
        </div>
        <div className="kvm-rows">
          {hardwareRows(capability).map((row) => (
            <div className="kvm-row" key={row.key} title={row.value}>
              <span>{row.label}</span>
              <StatusPill description={row.value} label={row.pill.label} tone={row.pill.tone} />
            </div>
          ))}
        </div>
        <div>
          <Button onClick={onOpenChecklist} type="button" variant="secondary">View hardware checklist</Button>
        </div>
      </div>
    </section>
  );
}

/**
 * The Physical KVM view's first card (VD-200, the ConsoleKvm board): the
 * screen, each half of the hardware path in words, and the control session.
 * The control button is derived from the ladder the screen is showing, never
 * from a second reading of the same snapshot: until the ladder arrives the
 * answer is "no session", the fail-safe direction.
 */
export function PhysicalKvmStage({
  busy,
  canControl,
  capability,
  onControl,
  outcome,
  state,
  username,
}: {
  busy: string;
  canControl: boolean;
  capability: KvmCapabilities | null;
  onControl: (action: "acquire" | "release") => void;
  /** The last control request's result, said in this card. */
  outcome?: { text: string; refused: boolean } | null;
  state: RemoteSessionState;
  username: string;
}) {
  const sessionAvailable = consoleSessionAvailable(capability?.ladder);
  const owned = capability?.control.owner === username;
  const pill = sessionAvailable
    ? { label: "Ready", tone: "success" as const }
    : { label: physicalKvmStateLabel(state), tone: "neutral" as const };
  return (
    <section
      aria-labelledby="physical-kvm-title"
      className={sessionAvailable ? "card ui-card sys-card cns-stage cns-stage--ready" : "card ui-card sys-card cns-stage"}
    >
      <header className="ui-card__header">
        <div className="ui-card__titles">
          <span className="sys-eyebrow">Physical KVM video</span>
          <h2 id="physical-kvm-title">{videoTitle(capability)}</h2>
        </div>
        <div className="ui-card__actions"><StatusPill label={pill.label} tone={pill.tone} /></div>
      </header>
      <div className="cns-stage__grid">
        <div className="cns-stage__screen">
          <div className="kvm-screen">
            <Icon name="console" size={ICON_SIZE.standalone} />
            <p>{screenSentence(capability, sessionAvailable)}</p>
          </div>
        </div>
        <aside aria-label="Physical KVM readiness" className="cns-stage__controls">
          {hardwareRows(capability).map((row) => (
            <div className="cns-health" key={row.key}>
              <div>
                <small>{row.label}</small>
                <strong>{row.value}</strong>
              </div>
              <StatusPill label={row.pill.label} tone={row.pill.tone} />
            </div>
          ))}
          <div className="cns-owner">
            <span className="sys-eyebrow">Control session</span>
            {sessionAvailable && <span className="cns-owner__reason">{capability?.readiness?.reason || "Video and isolated keyboard/mouse input are ready."}</span>}
            <strong>{capability?.control.owner ? `Controlled by ${capability.control.owner}` : "View only"}</strong>
            <span>
              {!sessionAvailable
                ? capability?.readiness?.reason || "Control unlocks after video and USB input pass discovery."
                : owned ? "Keyboard and mouse control is active for this session." : "Request control to prove keyboard and mouse input ownership."}
            </span>
            {outcome && <Notice severity={outcome.refused ? "danger" : "info"}>{outcome.text}</Notice>}
            {sessionAvailable && canControl && (
              <div>
                <Button disabled={Boolean(busy)} onClick={() => onControl(owned ? "release" : "acquire")} type="button" variant="primary">
                  {owned ? "Release control" : "Request control"}
                </Button>
              </div>
            )}
          </div>
        </aside>
      </div>
    </section>
  );
}

const commissioningFallback = [
  {
    id: "capture", complete: false,
    title: "Connect a supported USB HDMI capture adapter",
    detail: "Vaelor will detect it automatically; no Linux commands are required.",
  },
  {
    id: "hid", complete: false,
    title: "Commission isolated keyboard and mouse emulation",
    detail: "Vaelor will verify the USB device controller before enabling input.",
  },
  {
    id: "atx", complete: false,
    title: "Connect isolated ATX power leads",
    detail: "Optional; enables audited target power and reset controls.",
  },
];

/** The Physical KVM view's setup checklist (the ConsoleKvm board), from discovery. */
export function KvmChecklistCard({ capability, failed = false }: {
  capability: KvmCapabilities | null;
  /** `/kvm/capabilities` did not answer: the count is unknown, not zero. */
  failed?: boolean;
}) {
  const steps = capability?.commissioning?.length ? capability.commissioning : commissioningFallback;
  const done = steps.filter((step) => step.complete).length;
  /*
   * LESSONS 8 / S-Y9: with no discovery answer, "0 of 3 done" is a count the
   * observer produced, not the machine. Say it was not read.
   */
  const progress = capability === null
    ? <StatusPill label={failed ? "Not read" : "Checking"} reading="unread" tone="neutral" />
    : <span>{done} of {steps.length} done</span>;
  return (
    <SectionCard
      actions={progress}
      className="kvm-checklist"
      description="Hardware discovery checks each requirement automatically."
      flush
      id={PHYSICAL_KVM_CHECKLIST_ID}
      title="Physical KVM setup"
      titleId="physical-kvm-setup-title"
    >
      <ol className="kvm-steps">
        {steps.map((step, index) => (
          <li className={step.complete ? "kvm-step kvm-step--done" : "kvm-step"} key={step.id}>
            <span aria-hidden="true" className="kvm-step__mark">{step.complete ? <Icon name="done" size={ICON_SIZE.inline} /> : index + 1}</span>
            <div>
              <strong>{step.title}</strong>
              <small>{step.detail}</small>
            </div>
            {step.complete
              ? <span className="kvm-step__tag kvm-step__tag--done">Done</span>
              : step.id === "atx" ? <span className="kvm-step__tag">Optional</span> : null}
          </li>
        ))}
      </ol>
      <div className="kvm-protected">
        <Icon name="shield" size={16} />
        <p><strong>Protected by discovery.</strong> Streaming and input stay disabled until real capture and isolated HID hardware pass verification.</p>
      </div>
    </SectionCard>
  );
}
