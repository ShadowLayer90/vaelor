import { Button, Checkbox, Input, Select } from "./ui";
import { formatQuantity } from "../lib/format";
import {
  normaliseAdvanced,
  type AdvancedEdits,
  type AdvancedResult,
  type GpuServingAdvanced,
} from "../lib/gpuServeAdvanced";

/**
 * The cluster deploy form's collapsed Advanced section (VD-213): the request
 * cap, the prompt tokens per step, the draft model with its lookahead alone
 * and busy, and vLLM's newer model runner.
 *
 * It opens on the defaults the server will apply (`result.shown`). `edits`
 * holds only the fields the owner changed; empty means untouched, and the form
 * then sends none of the four settings, so the server applies the model's own.
 * "Reset to recommended" clears the edits. What is sent, and what is wrong, is
 * `advancedServeFields`' to decide; this only draws it, so the screen and the
 * payload cannot disagree.
 */
export function GpuServeAdvanced({
  advanced: offered,
  edits,
  result,
  onChange,
}: {
  advanced: GpuServingAdvanced;
  edits: AdvancedEdits | null;
  result: AdvancedResult;
  onChange: (edits: AdvancedEdits | null) => void;
}) {
  const advanced = normaliseAdvanced(offered);
  const { shown, errors, splitReason } = result;
  const set = (patch: AdvancedEdits) => onChange({ ...(edits ?? {}), ...patch });
  const drafters = splitReason ? [] : advanced.drafters;
  const drafter = drafters.find((entry) => entry.key === shown.drafterKey);
  const cap = shown.maxNumSeqs.trim();
  const runner = advanced.new_runner;
  const noDrafter = splitReason || (drafters.length ? "" : "No draft model is offered for this model and vLLM version.");
  return (
    <details className="cd-advanced">
      <summary>Advanced — requests per machine, draft model, model runner</summary>
      <div className="cd-advanced__body">
        {advanced.note && <p className="cd-note">{advanced.note}</p>}
        <div className="cd-grid">
          <Input label="Max requests per machine" inputMode="numeric" value={shown.maxNumSeqs}
            error={errors.maxNumSeqs}
            hint={`How many requests one machine answers at once; the rest wait their turn. Lower keeps each answer quick and leaves room for a draft model; higher serves more people at once, each a little slower.${result.defaults.maxNumSeqs === "" ? " Blank uses vLLM's own limit." : ""}`}
            onChange={(event) => set({ maxNumSeqs: event.target.value })} />
          <Input label="Prompt tokens per step" inputMode="numeric" value={shown.maxBatchedTokens}
            error={errors.maxBatchedTokens}
            hint={`How much of a waiting prompt a machine reads in one step. Higher starts long prompts sooner; lower keeps answers already streaming steady.${result.defaults.maxBatchedTokens === "" ? " Blank uses vLLM's own value." : ""}`}
            onChange={(event) => set({ maxBatchedTokens: event.target.value })} />
          <Select label="Draft model" value={shown.drafterKey}
            error={errors.drafterKey}
            disabled={Boolean(noDrafter)}
            disabledReason={noDrafter || undefined}
            hint={noDrafter
              ? undefined
              : `A small model guesses the next few words and the main model checks them in one step, so answers come faster; under load the first word comes slightly later.${drafter ? ` Downloads ${formatQuantity(drafter.weight_bytes, "model")} to every machine serving the model.` : ""}`}
            onChange={(event) => set({ drafterKey: event.target.value })}>
            <option value="">Off</option>
            {drafters.map((entry) => (
              <option key={entry.key} value={entry.key}>{entry.label}</option>
            ))}
            {shown.drafterKey && !drafter && <option value={shown.drafterKey}>Not offered</option>}
          </Select>
          {drafter && (
            <Input label="Lookahead when alone" inputMode="numeric" value={shown.alone}
              error={errors.alone}
              hint={`Words guessed ahead while one request runs (1 to ${drafter.max_tokens}). Further ahead pays off when the guesses land and wastes work when they miss.`}
              onChange={(event) => set({ alone: event.target.value })} />
          )}
          {drafter && result.busyShown && (
            <Input label="Lookahead when busy" inputMode="numeric" value={shown.busy}
              error={errors.busy}
              hint={`Words guessed ahead while 2${cap ? ` to ${cap}` : " or more"} requests run. A busy machine has less spare time to check guesses.`}
              onChange={(event) => set({ busy: event.target.value })} />
          )}
        </div>
        {runner.offered || splitReason ? (
          <Checkbox label="Use vLLM's newer model runner"
            checked={!splitReason && (result.runnerForced || shown.newRunner)}
            disabled={Boolean(splitReason) || result.runnerForced}
            disabledReason={splitReason || (result.runnerForced
              ? "On while the lookahead differs alone and busy: the older runner can only do that with slower GPU graphs."
              : undefined)}
            hint={splitReason || result.runnerForced ? undefined : runner.note || undefined}
            onChange={(event) => set({ newRunner: event.target.checked })} />
        ) : (
          runner.note && <p className="cd-note">{runner.note}</p>
        )}
        <div className="cl-actions">
          <Button variant="quiet" onClick={() => onChange(null)}>Reset to recommended</Button>
        </div>
      </div>
    </details>
  );
}
