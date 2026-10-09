import { StatusPill } from "./StatusPill";
import { Button, Notice } from "./ui";
import { formatBytes } from "../lib/format";
import type { StatusTone } from "./ui/status";
import type {
  GpuFitDecision,
  GpuFitDistributedPlacement,
  GpuFitOption,
  GpuFitSinglePlacement,
  GpuFitVerdict,
} from "./fleetTypes";

/**
 * The GPU fit/sizing decision (`POST /api/v2/cluster/fit`) in one plain
 * sentence a non-technical owner can read at a glance.
 *
 * Pure and presentational on purpose: it takes a `GpuFitDecision` and renders
 * it, and fetches nothing. Phase 3/4's deploy modals decide *when* a fit is
 * asked for; this is what they show once it comes back, so the same words
 * describe a model's placement wherever the question is put.
 *
 * Deliberately plain: it shows the verdict as ok or a warning, the backend's
 * own one-sentence summary, and — only when a model does not fit — how far
 * short it is and the ways out in ordinary words. It never surfaces the
 * engine's internals (tensor/pipeline parallelism, link speeds, shard byte
 * counts, GTT); those belong to the sizing engine, not to the person deciding
 * whether to deploy.
 *
 * **Every refusal is the engine's own sentence, printed verbatim** (VD-127 D6
 * and D10). `single_refused` carries "This model fits one machine (X). Serve it
 * as the AI Chat model on that machine instead of clustering." and
 * `unsupported_intent` carries the engine's "… is not available yet" sentence
 * for what the chosen intent cannot do; a `single`/`distributed`/`wont_fit`
 * summary may carry the reclaim clause ("…plus N GiB reclaimed by stopping the
 * AI Chat model", VD-127 D1). None of them is re-worded here — a second
 * refusal written in the console is how a preview and a deploy come to
 * disagree.
 *
 * **So is the `replicated` placement's sentence (VD-129).** It states what
 * the cluster gains and does not gain — room for at least K full-context
 * requests per machine (a floor from the KV budget, so "at least" and never
 * "about"), the budget held at about N machines' worth of concurrency, and
 * that no answer gets faster — from the engine's own budget arithmetic and the
 * measurement on the pair. The console prints it and adds no line of its own,
 * because a paraphrase of a measured claim is a claim nobody measured.
 */

const VERDICT_LABELS: Record<GpuFitVerdict, string> = {
  single: "Fits on one machine",
  distributed: "Fits across machines",
  replicated: "Replicated across machines",
  wont_fit: "Won't fit",
  single_refused: "Serve this on one machine instead",
  unsupported_intent: "Not available yet",
};

/**
 * Which verdicts are a placement and which are a refusal. Stated as data rather
 * than as `verdict !== "wont_fit"` scattered across callers, so a verdict added
 * to the engine cannot silently read as "it fits" anywhere.
 */
const PLACED: ReadonlySet<GpuFitVerdict> = new Set<GpuFitVerdict>([
  "single", "distributed", "replicated",
]);

export function fitIsServable(verdict: GpuFitVerdict | undefined): boolean {
  return Boolean(verdict && PLACED.has(verdict));
}

const VERDICT_TONES: Record<GpuFitVerdict, StatusTone> = {
  single: "success",
  distributed: "success",
  replicated: "success",
  wont_fit: "warning",
  single_refused: "warning",
  unsupported_intent: "neutral",
};

/** The ways out of a "won't fit", each in one ordinary phrase. */
const OPTION_LABELS: Record<GpuFitOption["kind"], string> = {
  shorter_context: "set a shorter Max context",
  raise_gtt_ceiling: "raise the GPU memory pool",
  smaller_quantization: "pick a smaller model",
  add_node: "add another machine",
};

function isDistributed(
  placement: GpuFitSinglePlacement | GpuFitDistributedPlacement | undefined,
): placement is GpuFitDistributedPlacement {
  return Boolean(placement) && "parallelism" in (placement as GpuFitDistributedPlacement);
}

export function FitDecision({
  decision,
  onServeAsAiChatModel,
  refusalId,
}: {
  decision: GpuFitDecision;
  /** The refusal's id, so a disabled Serve can point at the one sentence. */
  refusalId?: string;
  /**
   * Where a `single_refused` verdict sends the owner: the single-node AI Chat
   * model install. Omitted when the surface has nowhere to send them, in which
   * case the sentence still says what to do.
   */
  onServeAsAiChatModel?: () => void;
}) {
  const wontFit = decision.verdict === "wont_fit";
  const refusedForOneMachine = decision.verdict === "single_refused";
  const options = (decision.options ?? [])
    .map((option) => OPTION_LABELS[option.kind])
    .filter(Boolean);
  return (
    <section className="cd-fit__decision" aria-label="Whether this model fits">
      <div>
        <StatusPill
          label={VERDICT_LABELS[decision.verdict] ?? "Fit decision"}
          tone={VERDICT_TONES[decision.verdict] ?? "neutral"}
        />
      </div>

      <p className="cd-fit__summary">{decision.summary}</p>

      {wontFit && decision.gpu_memory_pool?.would_fit && (
        <p className="cd-note">{decision.gpu_memory_pool.sentence}</p>
      )}

      {decision.verdict === "single" && decision.placement && !isDistributed(decision.placement) && (
        <p className="cd-note">
          Runs on {decision.placement.name}
          {decision.placement.headroom_bytes
            ? `, with ${formatBytes(decision.placement.headroom_bytes)} to spare.`
            : "."}
        </p>
      )}

      {decision.verdict === "distributed" && isDistributed(decision.placement) && (
        <p className="cd-fit__summary">
          Split across {decision.placement.node_count} machines.
        </p>
      )}

      {/* What the deploy would refuse before starting (VD-168), in its own
          words; the verdict above is kept, and the form offers no Serve. */}
      {decision.refusal?.message && (
        <Notice severity="warning" id={refusalId}>{decision.refusal.message}</Notice>
      )}

      {refusedForOneMachine && onServeAsAiChatModel && (
        <div className="cl-actions">
          <Button variant="secondary" type="button" onClick={onServeAsAiChatModel}>
            Install it as the AI Chat model
          </Button>
        </div>
      )}

      {wontFit && (
        <p className="cd-note">
          {typeof decision.deficit_bytes === "number"
            ? `${formatBytes(decision.deficit_bytes)} short.`
            : "Not enough GPU memory across the fleet."}
          {options.length ? ` You can ${options.join(", or ")}.` : ""}
        </p>
      )}
    </section>
  );
}
