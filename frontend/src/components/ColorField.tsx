import { useEffect, useId, useState } from "react";
import {
  channelError,
  clampChannel,
  hexError,
  hexToRgb,
  normalizeHex,
  rgbToHex,
  type Rgb,
} from "../lib/colorValue";
import { Input } from "./ui";

/**
 * Editable colour entry: a hex field, three channel fields, a visible swatch
 * that opens the native picker, and a live preview — all kept in step.
 *
 * Alpha 11 exposed no numeric entry at all; the only custom-colour affordance
 * was a 6x4px `opacity: 0` input, so a specific brand colour could not be
 * typed and the current value was never shown beside the control that set it.
 *
 * Fields hold their own draft text so a half-typed value ("#ff6") is not
 * rewritten under the cursor. The committed colour only changes once the draft
 * parses.
 */
export function ColorField({
  value,
  onChange,
  onValidityChange,
  onPendingChange,
  disabled = false,
  disabledReason,
}: {
  value: string;
  onChange: (hex: string) => void;
  /** Lets the owning form refuse to save while a field is showing an error. */
  onValidityChange?: (valid: boolean) => void;
  /**
   * Reports a valid colour the reader has typed but not yet committed (blur or
   * Enter). The committed-if-flushed hex, or null when the drafts match the
   * committed value or are not yet a complete, valid colour. See #154 below.
   */
  onPendingChange?: (pending: string | null) => void;
  disabled?: boolean;
  disabledReason?: string;
}) {
  const pickerId = useId().replaceAll(":", "") + "-picker";
  const rgb = hexToRgb(value) ?? { r: 0, g: 0, b: 0 };
  const [hexDraft, setHexDraft] = useState(value.toUpperCase());
  const [channelDrafts, setChannelDrafts] = useState<Record<keyof Rgb, string>>({
    r: String(rgb.r),
    g: String(rgb.g),
    b: String(rgb.b),
  });

  // Presets, the picker and a reload all change the colour from outside; adopt
  // it unless the user is mid-edit with an equivalent value.
  useEffect(() => {
    const next = hexToRgb(value);
    if (!next) return;
    if (normalizeHex(hexDraft) !== normalizeHex(value)) setHexDraft(value.toUpperCase());
    setChannelDrafts((current) => {
      const committed = { r: String(next.r), g: String(next.g), b: String(next.b) };
      const unchanged = (["r", "g", "b"] as const).every(
        (key) => clampChannel(Number(current[key])) === next[key] && channelError(current[key]) === "",
      );
      return unchanged ? current : committed;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  /*
   * #149: committing on every keystroke half-applied entries that were still
   * being typed. Typing 999 into Red passed through the momentarily-valid 99,
   * so Hex flipped to #63… and the live preview showed a colour never chosen
   * — before any error appeared. Typed fields commit on blur or Enter; the
   * native picker still commits per change, because each change there is a
   * complete pick.
   */
  const commitHex = (raw: string) => {
    const normalized = normalizeHex(raw);
    // Only a changed value commits: blur alone must not count as an edit,
    // or tabbing through the field before the first /lighting response
    // marks the form touched and freezes the seeding on the defaults.
    if (normalized && normalized !== normalizeHex(value)) onChange(normalized);
  };

  const commitChannel = (key: keyof Rgb, raw: string) => {
    if (channelError(raw)) return;
    const next = clampChannel(Number(raw));
    if (next !== rgb[key]) onChange(rgbToHex({ ...rgb, [key]: next }));
  };

  // Validation waits for blur. Reporting on every keystroke means "#f", "#ff",
  // "#ff6"... each raise an assertive alert while the user is simply still
  // typing a valid value.
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const markTouched = (key: string) => setTouched((current) => ({ ...current, [key]: true }));
  const problemFor = (key: string, problem: string) => (touched[key] ? problem : "") || undefined;

  const hexProblem = hexError(hexDraft);
  const channelLabels: Array<[keyof Rgb, string]> = [["r", "Red"], ["g", "Green"], ["b", "Blue"]];
  const anyProblem = Boolean(hexProblem)
    || channelLabels.some(([key]) => Boolean(channelError(channelDrafts[key])));

  // Report validity upward so the form cannot silently save the last good
  // colour while a field is visibly showing an error.
  useEffect(() => {
    onValidityChange?.(!anyProblem);
  }, [anyProblem, onValidityChange]);

  /*
   * #154: a valid value the reader typed but has not blurred is real work the
   * backend does not hold yet. It must not be committed to `color` here (that
   * is #149 - an intermediate keystroke would repaint hex and preview), but the
   * owning form has to count it as unsaved and flush it on Save. Report the
   * committed-if-flushed hex up; null when the drafts equal the committed
   * colour, so a blurred-and-saved value stops looking pending.
   */
  const committed = normalizeHex(value);
  const draftedHex = normalizeHex(hexDraft);
  const channelsClean = channelLabels.every(([key]) => channelError(channelDrafts[key]) === "");
  const draftedChannels = channelsClean
    ? rgbToHex({
        r: clampChannel(Number(channelDrafts.r)),
        g: clampChannel(Number(channelDrafts.g)),
        b: clampChannel(Number(channelDrafts.b)),
      })
    : null;
  const pendingHex = draftedHex && draftedHex !== committed
    ? draftedHex
    : draftedChannels && committed && draftedChannels !== committed
      ? draftedChannels
      : null;
  useEffect(() => {
    onPendingChange?.(pendingHex);
  }, [pendingHex, onPendingChange]);

  return (
    <div className="color-field">
      <div className="color-field__entry">
        <label className="color-field__swatch" htmlFor={pickerId}>
          <span
            aria-hidden="true"
            className="color-field__swatch-chip"
            ref={(element) => element?.style.setProperty("background", value)}
          />
          <span className="color-field__swatch-text">Pick a colour</span>
          <input
            className="color-field__picker"
            disabled={disabled}
            id={pickerId}
            onChange={(event) => commitHex(event.target.value)}
            type="color"
            value={normalizeHex(value) ?? "#000000"}
          />
        </label>

        <Input
          className="color-field__hex"
          disabled={disabled}
          disabledReason={disabledReason}
          error={problemFor("hex", hexProblem)}
          inputMode="text"
          label="Hex"
          maxLength={7}
          onBlur={() => { markTouched("hex"); commitHex(hexDraft); }}
          onChange={(event) => setHexDraft(event.target.value)}
          onKeyDown={(event) => { if (event.key === "Enter") { markTouched("hex"); commitHex(hexDraft); } }}
          spellCheck={false}
          value={hexDraft}
        />

        {channelLabels.map(([key, label]) => {
          const problem = channelError(channelDrafts[key]);
          return (
            <Input
              className="color-field__channel"
              disabled={disabled}
              disabledReason={disabledReason}
              error={problemFor(key, problem)}
              inputMode="numeric"
              key={key}
              label={label}
              /*
               * Deliberately type="text": a number input sanitises unparseable
               * entries to "", which hides what the user typed and makes the
               * "whole numbers only" message unreachable.
               */
              onBlur={() => { markTouched(key); commitChannel(key, channelDrafts[key]); }}
              onChange={(event) => setChannelDrafts((current) => ({ ...current, [key]: event.target.value }))}
              onKeyDown={(event) => { if (event.key === "Enter") { markTouched(key); commitChannel(key, channelDrafts[key]); } }}
              type="text"
              value={channelDrafts[key]}
            />
          );
        })}
      </div>
    </div>
  );
}
