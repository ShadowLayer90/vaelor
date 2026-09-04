import type { StatusTone } from "./ui/status";
import { StatusPill } from "./StatusPill";

/**
 * What a GPU AI-Chat deploy actually got, on a screen.
 *
 * The GPU AI-Chat tier serves ANY GGUF a user brings, not just the one measured
 * ROCmFP4 build. That freedom is only honest if the screen says which one this
 * is: the recommended build is a tuned FP4 recipe with speculative decoding, and
 * everything else is a standard model that runs but is larger and slower. The
 * backend states both facts on the deploy result — `optimized` (true only for
 * the measured FP4 build) and the fit verdict (`fit_mode` plus a ready-to-show
 * `fit_reason` sentence) — so this component renders them without re-deriving a
 * word of it. The `fit_reason` string is printed verbatim so the copy has one
 * source of truth in the backend, exactly as `AccelerationVerdict` prints the
 * appliance's own `detail`.
 *
 * It reuses `AccelerationVerdict`'s `.acceleration-*` classes and its
 * labelled-reading-plus-pill shape, because this is the same kind of surface: a
 * reading whose severity is carried only by the pill's tone, never a banner.
 */

export interface GpuChatTierResult {
  /** True ONLY for the recommended ROCmFP4 build; false for any other GGUF. */
  optimized?: boolean;
  /** Which fork recipe launched. Present only on a GPU AI-Chat deploy. */
  recipe?: "rocmfp4" | "generic" | string;
  /** Whether every layer, some layers, or none were offloaded to the GPU. */
  fit_mode?: "gpu" | "partial" | "cpu" | string;
  /** A ready-to-show human sentence from the backend. Printed verbatim. */
  fit_reason?: string;
}

/**
 * The three fit verdicts, each with the headline and the pill tone the task
 * fixes. A partial offload is information, not a fault, and a CPU-only run is an
 * honest "does not fit the GPU budget" rather than a failure — so neither is
 * painted as a degradation.
 */
const FIT_HEADLINES: Record<string, { headline: string; tone: StatusTone }> = {
  gpu: { headline: "Running fully on the GPU", tone: "success" },
  partial: { headline: "Partly on the GPU, the rest on the CPU", tone: "info" },
  cpu: { headline: "Running on the CPU", tone: "neutral" },
};

/**
 * Whether a deploy result carries the GPU AI-Chat descriptors at all. Only a
 * deploy that went down the GPU fork sets `recipe`; a plain Assistant/compose
 * deploy leaves these absent, and this component renders nothing for it.
 */
export function isGpuChatTier(result: GpuChatTierResult | null | undefined): boolean {
  return Boolean(result && (result.recipe === "rocmfp4" || result.recipe === "generic"));
}

export function GpuChatTier({
  result,
  title = "How this AI Chat model runs",
  headingLevel = "h3",
}: {
  result: GpuChatTierResult;
  title?: string;
  headingLevel?: "h3" | "h4";
}) {
  if (!isGpuChatTier(result)) return null;
  const Heading = headingLevel;
  const optimized = result.optimized === true;
  // An unrecognised fit_mode still prints its reason, under a neutral pill —
  // "not established", never invented into one of the three verdicts.
  const fit = result.fit_mode
    ? FIT_HEADLINES[result.fit_mode] ?? { headline: "Reported", tone: "neutral" as StatusTone }
    : undefined;
  return (
    <section aria-label={title} className="acceleration-verdict">
      <Heading>{title}</Heading>
      <div className="acceleration-reading" data-state={optimized ? "optimized" : "standard"}>
        <div className="acceleration-reading__heading">
          <span>Model build</span>
          <StatusPill
            label={optimized ? "Optimized build" : "Standard model"}
            tone={optimized ? "success" : "info"}
          />
        </div>
        <p>
          {optimized
            ? "This is the recommended build — a tuned FP4 recipe with speculative decoding, measured on this machine's GPU."
            : "This is a standard, unoptimized model. It runs, but it is larger and less optimized than the recommended build, which uses a tuned FP4 recipe with speculative decoding — so expect it to be slower."}
        </p>
      </div>
      {fit ? (
        <div className="acceleration-reading" data-state={result.fit_mode}>
          <div className="acceleration-reading__heading">
            <span>Where it runs</span>
            <StatusPill label={fit.headline} tone={fit.tone} />
          </div>
          {/* The backend's own sentence, verbatim: one source of truth for the
              reason a model landed fully on the GPU, split across both, or on
              the CPU. */}
          {result.fit_reason ? <p>{result.fit_reason}</p> : null}
        </div>
      ) : null}
    </section>
  );
}
