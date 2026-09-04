import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import { isMulticolourStyle, normalizeHex } from "../lib/colorValue";
import type { Session } from "../types";
import { ColorField } from "./ColorField";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import { Button, Notice, UnavailableValue } from "./ui";
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

export function LightingControl({ session }: { session: Session }) {
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
    void refresh();
    const interval = window.setInterval(() => void refresh(), 5_000);
    return () => window.clearInterval(interval);
  }, [lighting.available, refresh]);

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

  /*
   * Not one control here was gated on anything. On a machine with no LEDs the
   * page rendered a live animated preview, a colour swatch, three effects, two
   * sliders and a Save that returned 200 and audited `lighting.update` as a
   * success against target `ws2812`. `led_count: 4` is a server-side literal,
   * so "the four built-in lights" was asserted on every host.
   */
  if (!lighting.available) {
    return (
      <section className="lighting-section" aria-labelledby="lighting-title">
        <div className="section-heading lighting-section__heading">
          <div>
            <span className="page-eyebrow">System / Case lighting</span>
            <h2 id="lighting-title">Case lighting</h2>
            <p>{machine ? lighting.reason : "Checking what lighting hardware this machine has."}</p>
          </div>
          <StatusPill
            label={machine ? "No lighting controller" : "Detecting lighting"}
            status="neutral"
          />
        </div>
        <p className="lighting-unavailable">
          <UnavailableValue label="Case lighting unavailable" reason={lighting.reason ?? "Not reported by this device"} />
          <span>There are no lights here to set a colour, effect or brightness on, so no controls are offered.</span>
        </p>
      </section>
    );
  }

  return (
    <section className="lighting-section" aria-labelledby="lighting-title">
      <div className="section-heading lighting-section__heading">
        <div>
          <span className="page-eyebrow">System / Case lighting</span>
          <h2 id="lighting-title">Case lighting</h2>
          <p>Choose how the {state?.led_count ?? "built-in"} case lights look. Changes are applied when you select Save lighting.</p>
        </div>
        <StatusPill
          label={state?.detected ? `${state.led_count} RGB lights ready` : "Detecting lighting"}
          status={state?.detected ? "healthy" : "neutral"}
        />
      </div>

      {message && <Notice severity={messageSeverity}><Icon name="bolt" />{message}</Notice>}

      <div className="lighting-console">
        <div className={`lighting-preview lighting-preview--${lightingStyle}`} ref={previewRef}>
          <div className="lighting-preview__case" aria-hidden="true">
            <span /><span /><span /><span />
          </div>
          <div className="lighting-preview__copy">
            <small>Live preview</small>
            <strong>{enabled ? state?.styles.find((item) => item.id === lightingStyle)?.name ?? lightingStyle : "Lights off"}</strong>
            {/* The chosen colour is shown here as well as in the hex field, so
                it is never readable in only one place. */}
            <span>{brightness}% brightness · {color.toUpperCase()}{colorIgnored ? " · multicolour effect" : ""}</span>
          </div>
        </div>

        <div className="lighting-controls">
          <div className="lighting-power-row">
            <div><strong>Case lights</strong><span>Switch {state?.led_count ? `all ${state.led_count}` : "the"} RGB lights on or off.</span></div>
            <Button
              aria-checked={enabled}
              className="switch-control"
              disabled={!canControl}
              onClick={() => { touched.current = true; setEnabled((current) => !current); }}
              role="switch"
            >
              <span />
              {enabled ? "On" : "Off"}
            </Button>
          </div>

          <fieldset
            aria-describedby={colorNote ? "lighting-color-note" : undefined}
            disabled={colorDisabled}
          >
            <legend>Color</legend>
            {/* One visible explanation, associated with the group, rather than
                repeating itself under all four inputs. */}
            {colorNote && (
              <p className="lighting-color-note" id="lighting-color-note">{colorNote}</p>
            )}
            <div className="color-picker-row">
              <ColorField
                disabled={colorDisabled}
                key={colorRevision}
                onChange={edited(setColor)}
                // A valid draft the reader typed protects itself from the
                // seeding poll while it stands, and lifts that protection when
                // it is withdrawn - unlike the monotonic `touched`, which a
                // withdrawn draft used to stick (#154 regression).
                onPendingChange={notePending}
                onValidityChange={setColorValid}
                value={color}
              />
              <div className="color-presets">
                {colorPresets.map((preset) => (
                  /* An ancestor `fieldset[disabled]` alone leaves
                     `disabled === false` and no `aria-disabled` on the button
                     itself, so the state is stated here rather than inferred. */
                  <Button aria-label={preset.name} aria-pressed={color === preset.value} className="color-preset" disabled={colorDisabled} key={preset.value} onClick={() => edited(setColor)(preset.value)} title={preset.name} type="button"><span aria-hidden="true" ref={(element) => element?.style.setProperty("background", preset.value)} /></Button>
                ))}
              </div>
            </div>
          </fieldset>

          <fieldset disabled={!canControl || !enabled}>
            <legend>Effect</legend>
            <div className="lighting-effects">
              {state?.styles.map((item) => (
                <Button aria-pressed={lightingStyle === item.id} className="lighting-effect" key={item.id} onClick={() => edited(setLightingStyle)(item.id)} type="button">
                  <span className={`effect-glyph effect-glyph--${item.id}`} />
                  {item.name}
                </Button>
              ))}
            </div>
          </fieldset>

          <div className="lighting-sliders">
            <label>
              <span><strong>Brightness</strong><output>{brightness}%</output></span>
              <input
                aria-label="RGB brightness"
                aria-valuetext={`${brightness} percent`} className="lighting-slider" disabled={!canControl || !enabled}
                max="100"
                min="0"
                onChange={(event) => edited(setBrightness)(Number(event.target.value))}
                type="range"
                value={brightness}
              />
            </label>
            <label>
              <span><strong>Animation speed</strong><output>{speed}%</output></span>
              <input
                aria-label="RGB animation speed"
                aria-valuetext={`${speed} percent`} className="lighting-slider" disabled={!canControl || !enabled || lightingStyle === "solid"}
                max="100"
                min="0"
                onChange={(event) => edited(setSpeed)(Number(event.target.value))}
                type="range"
                value={speed}
              />
            </label>
          </div>

          <div className="lighting-actions">
            {/*
              * Three states, not two. `dirty` was `Boolean(state) && …`, so
              * before the first `/lighting` response it was false and this line
              * claimed "All changes saved" — a settled verdict about settings
              * Vaelor had not yet read, on a panel the reader can already type
              * into. Not knowing is its own answer and says so.
              */}
            <p aria-live="polite" className="lighting-actions__state">
              {!state ? "Reading the current settings" : dirty ? "Unsaved changes" : "All changes saved"}
            </p>
            <div className="lighting-actions__buttons">
              <Button
                disabled={!canControl || busy || !state || (!dirty && colorValid)}
                disabledReason={!state ? "Vaelor has not read this enclosure's current lighting yet." : undefined}
                onClick={revert}
                type="button"
                variant="quiet"
              >
                Revert
              </Button>
              <Button className="lighting-apply" variant="primary" disabled={!canControl || busy || !dirty || !colorValid} disabledReason={!colorValid && dirty ? "Correct the colour value before saving." : undefined} onClick={() => void apply()}>
                {busy ? "Saving…" : canControl ? "Save lighting" : "Operator access required"}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
