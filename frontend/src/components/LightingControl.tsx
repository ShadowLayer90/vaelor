import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import { isMulticolourStyle, normalizeHex } from "../lib/colorValue";
import type { Session } from "../types";
import { ColorField } from "./ColorField";
import { Icon } from "./Icon";
import type { SectionStatus } from "./FanControl";
import { StatusPill } from "./StatusPill";
import { KvGrid, SectionCard, useVisiblePoll } from "./systemUi";
import { Button, Notice } from "./ui";
import { useMachineProfile } from "../hooks/useMachineProfile";
import { unknownMachine } from "../lib/machine";

interface LightingStyle {
  id: string;
  name: string;
}

interface LightingState {
  detected: boolean;
  hardware: string;
  led_count: number;
  enabled: boolean;
  color: string;
  brightness: number;
  speed: number;
  style: string;
  styles: LightingStyle[];
}

const colorPresets = [
  // The preset is the enclosure brand's orange; the name only makes sense on
  // one, so it is named by its colour instead.
  { name: "Ember orange", value: "#ff6a00" },
  { name: "Ice blue", value: "#00d9ff" },
  { name: "Signal green", value: "#35e06f" },
  { name: "Electric violet", value: "#8b5cff" },
  { name: "Hot pink", value: "#ff3b8d" },
  { name: "Warm white", value: "#ffd7a8" },
];

/**
 * System › Case lighting (VD-200, the SystemLighting board): the live preview
 * with what the enclosure holds, the lights and their colour beside the effect
 * and its sliders, and one save bar. Nothing reaches the lights until Save.
 */
export function LightingControl({
  onStatus,
  reloadToken = 0,
  session,
}: {
  /** The page header's pill ("4 RGB lights ready"), reported up to the System page. */
  onStatus?: (status: SectionStatus | null) => void;
  /** Bumped by the System page's Reload. */
  reloadToken?: number;
  session: Session;
}) {
  const [state, setState] = useState<LightingState | null>(null);
  const [enabled, setEnabled] = useState(true);
  const [color, setColor] = useState("#ff6a00");
  const [brightness, setBrightness] = useState(70);
  const [speed, setSpeed] = useState(50);
  const [lightingStyle, setLightingStyle] = useState("breathing");
  const [busy, setBusy] = useState(false);
  const [colorValid, setColorValid] = useState(true);
  /*
   * #154: a valid colour typed into ColorField but not yet blurred lives in the
   * field's own draft and never reaches `color` (which commits on blur, #149).
   * That value is unsaved work the appliance does not hold, so the field reports
   * it here. The save strip counts it as a change, and Save flushes it - without
   * this, a typed-but-unblurred colour read "All changes saved" and the first
   * Save sent the stale value, so a second click was needed.
   */
  const [pendingColor, setPendingColor] = useState<string | null>(null);
  /*
   * The same pending value, mirrored in a ref so `refresh` (a stable callback)
   * can read it. It gates seeding as a *live* divergence, not a permanent one:
   * a draft standing when the appliance answers protects itself, but a draft
   * typed and then withdrawn lifts the gate again so the real settings seed.
   * Using the monotonic `touched` for this instead stuck the gate on a
   * withdrawn draft and let a Save write the hard-coded defaults over the
   * enclosure's state (#154 regression).
   */
  const pendingRef = useRef<string | null>(null);
  const notePending = (pending: string | null) => {
    pendingRef.current = pending;
    setPendingColor(pending);
  };
  const [message, setMessage] = useState("");
  const [messageSeverity, setMessageSeverity] = useState<"success" | "danger">("success");
  const initialized = useRef(false);
  /** Set the moment the reader changes any field, and never cleared. */
  const touched = useRef(false);
  /**
   * Wraps a draft setter so every reader-driven change records itself in one
   * place. Marking each call site by hand is how one of them gets missed, and
   * the field that got missed is the one whose edit a poll eats.
   */
  const edited = <T,>(setter: (value: T) => void) => (value: T) => {
    touched.current = true;
    setter(value);
  };
  const previewRef = useRef<HTMLDivElement>(null);
  const machine = useMachineProfile();
  const lighting = (machine ?? unknownMachine).capabilities.case_lighting;

  const refresh = useCallback(async () => {
    const next = await apiRequest<LightingState>("/lighting");
    setState(next);
    /*
     * Seed the draft from the appliance only while the reader has not started
     * one of their own.
     *
     * `initialized` alone was not enough. The first `/lighting` response can
     * land *after* the console is interactive — the panel renders as soon as
     * discovery reports a lighting controller, which is a different request —
     * so a colour typed in that window was silently overwritten by the poll
     * and the edit disappeared under the reader's hands. It showed up here as
     * two intermittently failing colour tests; on an appliance it is an edit
     * that vanishes if the network is slow.
     */
    if (!initialized.current && !touched.current && !pendingRef.current) {
      setEnabled(next.enabled);
      setColor(next.color);
      setBrightness(next.brightness);
      setSpeed(next.speed);
      setLightingStyle(next.style);
      initialized.current = true;
    }
  }, []);

  useEffect(() => {
    if (!lighting.available) return;
    void refresh().catch(() => undefined);
  }, [lighting.available, refresh]);
  useVisiblePoll(() => void refresh().catch(() => undefined), 5_000, lighting.available);

  useEffect(() => {
    if (reloadToken > 0 && lighting.available) void refresh().catch(() => undefined);
  }, [lighting.available, refresh, reloadToken]);

  const apply = async () => {
    // Flush any valid colour the reader typed but has not blurred, so the first
    // Save carries it rather than the last committed value (#154).
    const colorToSave = pendingColor ?? color;
    setBusy(true);
    setMessage("");
    try {
      const next = await apiRequest<LightingState>(
        "/lighting",
        {
          method: "PATCH",
          body: JSON.stringify({ enabled, color: colorToSave, brightness, speed, style: lightingStyle }),
        },
        session.csrf_token,
      );
      setState(next);
      setColor(colorToSave);
      notePending(null);
      setMessageSeverity("success");
      setMessage(enabled ? "Case lighting updated." : "Case lighting switched off.");
    } catch (error) {
      setMessageSeverity("danger");
      setMessage(error instanceof Error ? error.message : "Case lighting could not be updated.");
    } finally {
      setBusy(false);
    }
  };

  const canControl = session.user.role !== "viewer";
  const colorIgnored = isMulticolourStyle(lightingStyle);
  /*
   * A multicolour effect is a reason to explain the colour, never a reason to
   * lock it. The colour is stored and sent with every save whatever the effect
   * is, so disabling the row while the copy said the value was "saved for
   * later" left no way to set the value the screen claimed to be keeping —
   * short of switching to Solid, saving, switching back and saving again.
   */
  const colorDisabled = !canControl || !enabled;
  // Every reason the colour row can go dead gets an explanation.
  const colorDisabledReason = !canControl
    ? "Operator access is required to change case lighting."
    : !enabled
      ? "Switch the case lights on to choose a colour."
      : undefined;
  const styleName = state?.styles.find((item) => item.id === lightingStyle)?.name ?? lightingStyle;
  /*
   * The same three-state rule the save strip below now follows, applied to the
   * value itself. `color` starts at a hard-coded `#ff6a00` and the field
   * presents it exactly as it presents a colour that came off the enclosure —
   * so before `/lighting` answers, four inputs and a swatch state a reading
   * that was never taken. The panel stays usable, because a reader's early
   * edit is deliberately protected here; what changes is that it stops
   * claiming the starting value is this enclosure's colour.
   */
  const colorNote = colorDisabledReason
    ?? (!state
      ? "Vaelor has not read this enclosure's colour yet, so the value below is a "
        + "starting point rather than a reading. Anything you set here is saved as a new colour."
      : colorIgnored
        ? `${styleName} cycles through its own colours, so this colour is not on the lights right now. `
          + "Set it here and it is used as soon as you choose a single-colour effect."
        : undefined);
  // Compared against the last saved state so the user can see, and undo,
  // pending edits instead of losing them silently on navigation. `pendingColor`
  // folds in a valid colour typed but not blurred, so the strip stops claiming
  // that value is already saved (#154).
  const effectiveColor = pendingColor ?? color;
  const dirty = Boolean(state) && (
    enabled !== state!.enabled
    || normalizeHex(effectiveColor) !== normalizeHex(state!.color)
    || brightness !== state!.brightness
    || speed !== state!.speed
    || lightingStyle !== state!.style
  );

  /*
   * An invalid colour draft lives inside ColorField and never reaches `color`,
   * so `setColor(state.color)` is a no-op for it: the draft, its error and the
   * Save block all survived a Revert. Remounting the field on this token
   * restores every draft from the committed colour and clears the errors with
   * it, and Revert stays available while the colour is invalid even when
   * nothing else is dirty — otherwise a bad value with no other edit had no
   * exit but a page reload.
   */
  const [colorRevision, setColorRevision] = useState(0);

  /*
   * Revert means "put back what the appliance has". Until the appliance has
   * said what it has, there is nothing to put back — and this used to return
   * silently, so a reader with an invalid colour and a slow `/lighting` had
   * their one exit render enabled and do nothing. A control that cannot act
   * says so on itself; it does not accept the click and discard it.
   */
  const revert = () => {
    if (!state) return;
    setEnabled(state.enabled);
    setColor(state.color);
    setBrightness(state.brightness);
    setSpeed(state.speed);
    setLightingStyle(state.style);
    setColorRevision((current) => current + 1);
    setColorValid(true);
    notePending(null);
    setMessage("");
  };

  // An explicit-save form must not lose work to a stray navigation or reload.
  useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  useEffect(() => {
    const preview = previewRef.current;
    if (preview) {
      preview.style.setProperty("--lighting-color", color);
      preview.style.setProperty("--lighting-brightness", `${enabled ? brightness : 0}%`);
      preview.style.setProperty("--lighting-opacity", String(enabled ? brightness / 100 : 0));
      preview.style.setProperty("--lighting-speed", `${Math.max(0.7, 4.5 - speed * 0.035)}s`);
    }
    /*
     * `lighting.available` is a dependency because the preview element does
     * not exist until discovery confirms there are lights: without it the
     * first run found no node and no later run was scheduled, so the console
     * mounted with none of its custom properties set.
     */
  }, [color, brightness, speed, enabled, lighting.available]);


  const ledCount = state?.detected ? state.led_count : null;
  /*
   * The page header's pill. `led_count: 4` is a server-side literal, so it is
   * stated only once the controller has answered as detected.
   */
  const status: SectionStatus = !lighting.available
    ? { label: machine ? "No lighting controller" : "Detecting lighting", tone: "neutral" }
    : ledCount !== null
      ? { label: `${ledCount} RGB lights ready`, tone: "success" }
      : { label: "Detecting lighting", tone: "neutral", reading: "unread" };
  const statusKey = `${status.label}|${status.tone}|${status.reading ?? ""}`;
  useEffect(() => {
    onStatus?.(status);
    // The key carries every field the pill shows.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onStatus, statusKey]);
  useEffect(() => () => onStatus?.(null), [onStatus]);

  /*
   * Not one control here was gated on anything once: on a machine with no
   * LEDs the page rendered a live preview, a swatch, three effects, two
   * sliders and a Save that audited `lighting.update` as a success.
   */
  if (!lighting.available) {
    return (
      <div className="sys-section">
        <SectionCard
          actions={<StatusPill label={machine ? "No lighting controller" : "Detecting lighting"} tone="neutral" />}
          title="Case lighting"
          titleId="lighting-title"
        >
          <div className="sys-absent">
            <span aria-hidden="true" className="sys-absent__dash">—</span>
            <p>{machine ? lighting.reason : "Checking what lighting hardware this machine has."}</p>
            {machine && <p>There are no lights here to set a colour, effect or brightness on, so no controls are offered.</p>}
          </div>
        </SectionCard>
      </div>
    );
  }

  const savedStyleName = state ? state.styles.find((item) => item.id === state.style)?.name ?? state.style : null;
  const changes = state ? [
    enabled !== state.enabled,
    normalizeHex(effectiveColor) !== normalizeHex(state.color),
    brightness !== state.brightness,
    speed !== state.speed,
    lightingStyle !== state.style,
  ].filter(Boolean).length : 0;

  return (
    <div className="sys-section sys-lighting">
      <section aria-label="Live preview" className="card ui-card sys-hero">
        <div aria-hidden="true" className={`sys-lights sys-lights--${lightingStyle}`} ref={previewRef}>
          <div className="sys-lights__case"><span /><span /><span /><span /></div>
        </div>
        <div className="sys-hero__reading">
          <small>Live preview</small>
          <strong>{enabled ? styleName : "Lights off"}</strong>
          {/* The chosen colour is shown here as well as in the hex field, so
              it is never readable in only one place. */}
          <p>{brightness}% brightness · <span className="sys-mono">{color.toUpperCase()}</span>{colorIgnored ? " · multicolour effect" : ""}</p>
        </div>
        <KvGrid
          className="sys-hero__facts"
          items={[
            {
              label: "Case lights",
              value: !state ? "Not read" : `${state.enabled ? "On" : "Off"}${ledCount !== null ? ` · ${ledCount} lights` : ""}`,
            },
            { label: "Controller", value: state?.hardware || "Not read" },
            {
              label: "Saved on the enclosure",
              value: !state ? "Not read" : state.enabled ? `${savedStyleName} · ${state.brightness}%` : "Lights off",
            },
          ]}
          label="Case lighting on the enclosure"
        />
      </section>

      <div className="sys-grid-2">
        <SectionCard
          actions={(
            <Button
              aria-checked={enabled}
              aria-label="Case lights"
              className="sys-switch"
              disabled={!canControl}
              onClick={() => { touched.current = true; setEnabled((current) => !current); }}
              role="switch"
              variant="quiet"
            >
              <span aria-hidden="true" className="sys-switch__track"><i /></span>
              {enabled ? "On" : "Off"}
            </Button>
          )}
          description={`Switch ${ledCount !== null ? `all ${ledCount}` : "the"} RGB lights on or off.`}
          title="Lights and colour"
          titleId="lighting-title"
        >
          <fieldset
            aria-describedby={colorNote ? "lighting-color-note" : undefined}
            className="sys-fieldset"
            disabled={colorDisabled}
          >
            <legend className="sys-field__label">Colour</legend>
            {/* One visible explanation, associated with the group. */}
            {colorNote && <p className="sys-note" id="lighting-color-note">{colorNote}</p>}
            <ColorField
              disabled={colorDisabled}
              key={colorRevision}
              onChange={edited(setColor)}
              // A valid draft protects itself from the seeding poll while it
              // stands, and lifts that protection when withdrawn (#154).
              onPendingChange={notePending}
              onValidityChange={setColorValid}
              value={color}
            />
            <span className="sys-field__label">Presets</span>
            <div className="sys-presets">
              {colorPresets.map((preset) => (
                /* An ancestor `fieldset[disabled]` alone leaves `disabled ===
                   false` on the button itself, so the state is stated here. */
                <Button
                  aria-pressed={color === preset.value}
                  className="sys-preset"
                  disabled={colorDisabled}
                  key={preset.value}
                  onClick={() => edited(setColor)(preset.value)}
                  type="button"
                  variant="secondary"
                >
                  <span aria-hidden="true" className="sys-preset__dot" ref={(element) => element?.style.setProperty("background", preset.value)} />
                  {preset.name}
                </Button>
              ))}
            </div>
          </fieldset>
        </SectionCard>

        <SectionCard description="How the lights move." title="Effect">
          <div className="sys-stack">
            <fieldset className="sys-fieldset" disabled={!canControl || !enabled}>
              <legend className="sr-only">Effect</legend>
              <div className="sys-effects">
                {state?.styles.map((item) => (
                  <Button
                    aria-pressed={lightingStyle === item.id}
                    className="sys-effect"
                    key={item.id}
                    onClick={() => edited(setLightingStyle)(item.id)}
                    type="button"
                    variant="secondary"
                  >
                    {item.name}
                  </Button>
                ))}
              </div>
            </fieldset>
            <label className="sys-slider">
              <span><span>Brightness</span><output>{brightness}%</output></span>
              <input
                aria-label="RGB brightness"
                aria-valuetext={`${brightness} percent`}
                className="lighting-slider"
                disabled={!canControl || !enabled}
                max="100"
                min="0"
                onChange={(event) => edited(setBrightness)(Number(event.target.value))}
                type="range"
                value={brightness}
              />
            </label>
            <label className="sys-slider">
              <span><span>Animation speed</span><output>{speed}%</output></span>
              <input
                aria-label="RGB animation speed"
                aria-valuetext={`${speed} percent`}
                className="lighting-slider"
                disabled={!canControl || !enabled || lightingStyle === "solid"}
                max="100"
                min="0"
                onChange={(event) => edited(setSpeed)(Number(event.target.value))}
                type="range"
                value={speed}
              />
            </label>
            <p className="sys-muted">Solid does not animate, so speed is locked while Solid is chosen.</p>
          </div>
        </SectionCard>
      </div>

      <section aria-labelledby="lighting-save-title" className="card ui-card sys-savebar">
        <div className="sys-savebar__text">
          <span className="sys-eyebrow">Case lighting · {changes === 0 ? "no changes" : `${changes} change${changes === 1 ? "" : "s"}`}</span>
          {/*
            * Three states, not two: before the first `/lighting` answer this
            * once claimed "All changes saved" about settings Vaelor had not read.
            */}
          <h2 aria-live="polite" id="lighting-save-title">
            {!state ? "Reading the current settings" : dirty ? "Unsaved changes" : "All changes saved"}
          </h2>
          <p>{messageSeverity === "danger" && message
            ? "Unsaved changes stay in the form, so you can try again."
            : "Nothing reaches the lights until you save. Leaving the page asks first."}</p>
        </div>
        <div className="sys-savebar__actions">
          <Button
            disabled={!canControl || busy || !state || (!dirty && colorValid)}
            disabledReason={!state ? "Vaelor has not read this enclosure's current lighting yet." : undefined}
            onClick={revert}
            type="button"
            variant="secondary"
          >
            Revert
          </Button>
          <Button
                        disabled={!canControl || busy || !dirty || !colorValid}
            disabledReason={!canControl
              ? "Operator access is required to change case lighting."
              : !colorValid && dirty ? "Correct the colour value before saving." : undefined}
            onClick={() => void apply()}
            variant="primary"
          >
            {busy ? "Saving…" : canControl ? "Save lighting" : "Operator access required"}
          </Button>
        </div>
        {message && <Notice className="sys-savebar__outcome" severity={messageSeverity}>{message}</Notice>}
      </section>
    </div>
  );
}
