import { useState } from "react";
import { Button, Notice, UnavailableValue } from "./ui";
import { formatBytes, formatPercent, formatTemperature } from "../lib/format";
import { gpuTemperatureLabel } from "../lib/gpuTemperature";
import { StatusPill } from "./StatusPill";
import { PerfCard } from "./PerformanceDiagnostics";
import { ConfirmDialog } from "./ConfirmDialog";
import type { Session } from "../types";
import {
  runGpuKernelProfile,
  runServingProfile,
  type KernelProfileOutcome,
  type CpuProfileResult,
  type EbpfProfileResult,
  type GpuKernelCategory,
  type GpuKernelResult,
  type GpuPowerKind,
  type GpuProfileResult,
  type ProfileCapture,
  type ServingProfile,
  type SpecDecodingResult,
} from "../lib/clusterPerformance";

/**
 * The on-demand serving profile (VD-128, administrator only): the action and
 * the result of each capture - one line each on the card, the whole result
 * under its Details (VD-200, the ClusterPerformanceAdvanced board).
 */

/** The CPU capture: the hottest symbols and their self-overhead, or the note. */
function CpuCaptureResult({ result }: { result: CpuProfileResult }) {
  if (result.symbols.length === 0) {
    return <p className="perf-empty">{result.note ?? "perf recorded no attributable symbols."}</p>;
  }
  return (
    <ol className="perf-profile__symbols">
      {result.symbols.map((symbol, index) => (
        <li key={`${symbol.symbol}-${index}`}>
          <span className="perf-profile__symbol-pct">{symbol.overhead_percent.toFixed(1)}%</span>
          <span className="perf-profile__symbol-name">{symbol.symbol}</span>
          {symbol.shared_object && <span className="perf-muted">{symbol.shared_object}</span>}
        </li>
      ))}
    </ol>
  );
}

/** One GPU snapshot field: its value, or nothing when the snapshot lacked it. */
function GpuField({ label, value, render }: { label: string; value: number | undefined; render: (value: number) => string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>
        {value === undefined
          ? <UnavailableValue label={`${label} unavailable`} reason="The GPU snapshot did not carry this field." />
          : render(value)}
      </dd>
    </div>
  );
}

/**
 * What the snapshot's power figure is called, by the kind the backend read.
 * Package power (the whole chip) is never under one of these: it has its own
 * line below, so it cannot be mistaken for the GPU's draw.
 */
const POWER_LABEL: Record<GpuPowerKind, string> = {
  "graphics-engine": "GPU power (graphics engine)",
  gpu: "GPU power",
  unidentified: "Power (part not identified)",
};

/** The GPU snapshot: power, temperature (with its sensor), clock, occupancy and the serving process's share. */
function GpuCaptureResult({ result }: { result: GpuProfileResult }) {
  const snapshot = result.snapshot;
  const process = snapshot.process;
  const empty = Object.keys(snapshot).length === 0;
  if (empty) {
    return <p className="perf-empty">{result.note ?? "amd-smi returned no readable fields."}</p>;
  }
  const powerLabel = snapshot.power_kind ? POWER_LABEL[snapshot.power_kind] : "GPU power";
  const temperatureLabel = gpuTemperatureLabel(snapshot.gpu_temperature_sensor);
  return (
    <>
      <dl className="perf-kv">
        <GpuField label={powerLabel} value={snapshot.power_watts} render={(value) => `${value.toFixed(1)} W`} />
        <GpuField label={temperatureLabel} value={snapshot.gpu_temperature_c} render={(value) => formatTemperature(value)} />
        <GpuField label="GFX clock" value={snapshot.gfx_clock_mhz} render={(value) => `${Math.round(value)} MHz`} />
        <GpuField label="GFX busy" value={snapshot.gfx_activity_percent} render={formatPercent} />
      </dl>
      {snapshot.package_power_watts !== undefined && (
        <p className="perf-muted">
          Package power (whole chip): {snapshot.package_power_watts.toFixed(1)} W. This is the processor and
          graphics together, not the GPU alone.
        </p>
      )}
      {result.note && <p className="perf-muted">{result.note}</p>}
      {process && (
        <p className="perf-muted">
          Serving process (PID {process.pid}
          {process.name ? ` · ${process.name}` : ""}):
          {process.gtt !== undefined ? ` ${formatBytes(process.gtt)} GTT` : " GTT not reported"}
          {process.compute_percent !== undefined ? `, ${formatPercent(process.compute_percent)} compute` : ""}
        </p>
      )}
    </>
  );
}

/** Blocked time from microseconds: sub-millisecond as µs, then ms, then seconds. */
function formatMicros(us: number): string {
  if (us < 1000) return `${Math.round(us)} \u00b5s`;
  const ms = us / 1000;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

/**
 * The eBPF capture: OFF-CPU blocking time by serving thread - the complement to
 * the CPU capture's on-CPU symbols. A heading with the window and the total
 * blocked time, then the threads that blocked most (blocked time, then the thread
 * comm). An honest empty state when nothing blocked off-CPU in the window.
 */
function EbpfCaptureResult({ result }: { result: EbpfProfileResult }) {
  if (result.rows.length === 0) {
    return <p className="perf-empty">{result.note ?? "Captured, but the serving process did not block off-CPU during the window."}</p>;
  }
  return (
    <>
      <p className="perf-muted">
        Off-CPU blocking {"\u00b7"} {result.window_seconds} s {"\u00b7"} {formatMicros(result.total_us)} blocked across serving threads
      </p>
      <ol className="perf-profile__offcpu">
        {result.rows.map((row, index) => (
          <li key={`${row.label}-${index}`}>
            <span className="perf-profile__offcpu-us">{formatMicros(row.value_us)}</span>
            <span className="perf-profile__offcpu-comm">{row.label}</span>
          </li>
        ))}
      </ol>
    </>
  );
}

/** Shorten a long ISA kernel name for the row; the full name rides in a title. */
function truncateKernel(name: string): string {
  return name.length > 32 ? `${name.slice(0, 31)}\u2026` : name;
}

/** The lead sentence: where the GPU spent most of its time, in plain English,
 * from the dominant category. Factual - no judgement beyond the share itself. */
function kernelLead(top: GpuKernelCategory): string {
  const pct = Math.round(top.pct);
  if (top.pct >= 90) return `Nearly all GPU time (${pct}%) is ${top.label.toLowerCase()}.`;
  if (top.pct >= 50) return `Most GPU time (${pct}%) is ${top.label.toLowerCase()}.`;
  return `The largest share (${pct}%) is ${top.label}.`;
}

/**
 * The GPU kernel trace, led by a PLAIN-ENGLISH category breakdown: a one-line
 * summary of where the GPU spent its time, then one row per category (its
 * share, label and a one-line explanation). The raw ISA kernels stay available
 * for experts under a collapsed "Individual kernels" expander - truncated
 * names with the full name in a title. An honest empty state when the window
 * observed no kernels - never a fabricated trace.
 */
function GpuKernelCaptureResult({ result }: { result: GpuKernelResult }) {
  if (result.rows.length === 0) {
    return <p className="perf-empty">{result.note ?? "Captured, but no GPU kernels ran during the window."}</p>;
  }
  const categories = result.categories ?? [];
  return (
    <>
      <p className="perf-muted">
        GPU kernels {"\u00b7"} {result.window_seconds} s {"\u00b7"} {formatMicros(result.total_us)} device time across {result.rows.length} kernels
      </p>
      {categories.length > 0 ? (
        <ol className="perf-profile__kernel-cats">
          {categories.map((cat) => (
            <li key={cat.id}>
              <span className="perf-profile__kernel-cat-pct">{cat.pct}%</span>
              <span className="perf-profile__kernel-cat-text">
                <span className="perf-profile__kernel-cat-label">{cat.label}</span>
                <span className="perf-profile__kernel-cat-desc">{cat.description}</span>
              </span>
            </li>
          ))}
        </ol>
      ) : null}
      <details className="perf-profile__kernel-raw">
        <summary>Individual kernels</summary>
        <ol className="perf-profile__kernels">
          {result.rows.map((row, index) => {
            const pct = result.total_us > 0 ? (row.total_us / result.total_us) * 100 : 0;
            return (
              <li key={`${row.name}-${index}`}>
                <span className="perf-profile__kernel-us">{formatMicros(row.total_us)}</span>
                <span className="perf-profile__kernel-name" title={row.name}>{truncateKernel(row.name)}</span>
                <span className="perf-muted">{Math.round(pct)}%</span>
              </li>
            );
          })}
        </ol>
      </details>
    </>
  );
}

/** A share from 0..1 as a whole percent. */
function share(value: number): string {
  return formatPercent(value * 100);
}

/** The draft model's acceptance in one plain sentence (VD-213). */
function specLead(result: SpecDecodingResult): string {
  const kept = result.acceptance_rate !== undefined ? ` · ${share(result.acceptance_rate)} of guessed tokens kept` : "";
  return `${result.accepted_per_step.toFixed(1)} tokens accepted per step, ${result.mean_acceptance_length.toFixed(1)} written per step${kept}`;
}

/**
 * The draft model's acceptance over the window (VD-213): what it guessed and
 * kept, the tokens accepted per step, and how often each guess position is
 * right. Read from vLLM's own counters at the window's start and end; a window
 * with no guesses never reaches here (the capture says why instead).
 */
function SpecDecodingCaptureResult({ result }: { result: SpecDecodingResult }) {
  return (
    <>
      <p className="perf-muted">
        Draft model {"·"} {result.window_seconds} s {"·"} {result.drafts} guessing steps
      </p>
      <dl className="perf-kv">
        <div><dt>Tokens guessed</dt><dd>{result.drafted_tokens}</dd></div>
        <div><dt>Tokens accepted</dt><dd>{result.accepted_tokens}</dd></div>
        <div><dt>Tokens accepted per step</dt><dd>{result.accepted_per_step.toFixed(2)}</dd></div>
        <div><dt>Tokens written per step</dt><dd>{result.mean_acceptance_length.toFixed(2)}</dd></div>
      </dl>
      <p className="perf-muted">
        Each step the model checks the draft model&apos;s guesses and keeps the ones it agrees with, then writes one
        token of its own, so a step writes the accepted tokens plus one.
      </p>
      {result.positions.length > 0 ? (
        <>
          <p className="perf-muted">How often each guess is right</p>
          <ol aria-label="How often each guess is right">
            {result.positions.map((slot) => (
              <li key={slot.position}>
                <span className="perf-profile__symbol-pct">{share(slot.rate)}</span>
                <span>Guess {slot.position}</span>
              </li>
            ))}
          </ol>
        </>
      ) : null}
      {result.note && <p className="perf-muted">{result.note}</p>}
    </>
  );
}

/** The capture's result in one line, as the board shows it; the whole result is under Details. */
function captureSummary(capture: ProfileCapture): string {
  if (!capture.available || !capture.result) return capture.reason;
  if (capture.id === "cpu") {
    const cpu = capture.result as CpuProfileResult;
    return cpu.symbols.length
      ? cpu.symbols.slice(0, 3).map((symbol) => `${symbol.overhead_percent.toFixed(1)}% ${symbol.symbol}`).join(" · ")
      : cpu.note ?? "perf recorded no attributable symbols.";
  }
  if (capture.id === "ebpf") {
    const ebpf = capture.result as EbpfProfileResult;
    return ebpf.rows.length
      ? `${formatMicros(ebpf.total_us)} blocked off-CPU in ${ebpf.window_seconds} s, most in ${ebpf.rows[0].label}`
      : ebpf.note ?? "Captured, but the serving process did not block off-CPU during the window.";
  }
  if (capture.id === "speculative_decoding") return specLead(capture.result as SpecDecodingResult);
  if (capture.id === "gpu_kernel") {
    const kernels = capture.result as GpuKernelResult;
    const top = kernels.categories?.[0];
    if (!kernels.rows.length) return kernels.note ?? "Captured, but no GPU kernels ran during the window.";
    return top ? kernelLead(top) : `${formatMicros(kernels.total_us)} of GPU time in ${kernels.window_seconds} s`;
  }
  const snapshot = (capture.result as GpuProfileResult).snapshot;
  const parts = [
    snapshot.power_watts !== undefined ? `${snapshot.power_kind ? POWER_LABEL[snapshot.power_kind] : "GPU power"} ${snapshot.power_watts.toFixed(1)} W` : "",
    snapshot.gpu_temperature_c !== undefined ? formatTemperature(snapshot.gpu_temperature_c) : "",
    snapshot.gfx_clock_mhz !== undefined ? `GFX clock ${Math.round(snapshot.gfx_clock_mhz)} MHz` : "",
    snapshot.gfx_activity_percent !== undefined ? `GFX busy ${formatPercent(snapshot.gfx_activity_percent)}` : "",
  ].filter(Boolean);
  return parts.length ? parts.join(" · ") : (capture.result as GpuProfileResult).note ?? "amd-smi returned no readable fields.";
}

/** Whether a capture has more to show than its one line: an empty result is said whole in that line. */
function hasDetail(capture: ProfileCapture): boolean {
  if (!capture.available || !capture.result) return false;
  if (capture.id === "cpu") return (capture.result as CpuProfileResult).symbols.length > 0;
  if (capture.id === "ebpf") return (capture.result as EbpfProfileResult).rows.length > 0;
  if (capture.id === "gpu_kernel") return (capture.result as GpuKernelResult).rows.length > 0;
  if (capture.id === "speculative_decoding") return true;
  return Object.keys((capture.result as GpuProfileResult).snapshot).length > 0;
}

/**
 * What the GPU kernel trace's reader left out (VD-214): old traces it could not
 * clear (and did not read), and traces it refused to read, each with its
 * reason - said in plain lines, never dropped in silence.
 */
function traceReaderLines(capture: ProfileCapture): string[] {
  if (capture.id !== "gpu_kernel" || !capture.result) return [];
  const result = capture.result as GpuKernelResult;
  const lines: string[] = [];
  const kept = result.stale_traces_kept ?? 0;
  if (kept > 0) {
    lines.push(`${kept} old trace file${kept === 1 ? " was" : "s were"} kept: the bridge could not remove ${kept === 1 ? "it" : "them"}, and this capture did not read ${kept === 1 ? "it" : "them"}.`);
  }
  for (const refused of result.traces_refused ?? []) {
    lines.push(`Trace ${refused.name} ${refused.reason}.`);
  }
  return lines;
}

/** One capture row: its name, its result in a line and its pill; the whole result under Details. */
function CaptureRow({ capture }: { capture: ProfileCapture }) {
  return (
    <section className="perf-capture" aria-label={capture.label}>
      <div className="perf-capture__row">
        <div className="perf-capture__text">
          <strong>{capture.label}</strong>
          <p className="perf-muted">{captureSummary(capture)}</p>
          {traceReaderLines(capture).map((line) => <p className="perf-muted" key={line}>{line}</p>)}
        </div>
        <StatusPill label={capture.available ? "Captured" : "Not available"} tone={capture.available ? "success" : "neutral"} />
      </div>
      {capture.result && hasDetail(capture) ? (
        <details className="perf-capture__details">
          <summary>Details</summary>
          {capture.id === "cpu"
            ? <CpuCaptureResult result={capture.result as CpuProfileResult} />
            : capture.id === "ebpf"
              ? <EbpfCaptureResult result={capture.result as EbpfProfileResult} />
              : capture.id === "gpu_kernel"
                ? <GpuKernelCaptureResult result={capture.result as GpuKernelResult} />
                : capture.id === "speculative_decoding"
                  ? <SpecDecodingCaptureResult result={capture.result as SpecDecodingResult} />
                  : <GpuCaptureResult result={capture.result as GpuProfileResult} />}
        </details>
      ) : null}
    </section>
  );
}

/**
 * How long GPU tracing may stay on, in whole minutes rounded up:
 * `gpu_kernel_profile.TRACING_DEADLINE_MINUTES` (590 s, rounded up). Said before the owner starts it; `tests/test_gpu_kernel_profile_console.py`
 * ties this copy to the producer (LESSONS 6).
 */
export const KERNEL_TRACING_LIMIT_MINUTES = 10;

/** What the owner is told before a GPU kernel profile restarts the model (VD-214). */
export const KERNEL_PROFILE_WARNING =
  "This restarts this controller's copy of the model twice: once with GPU tracing on, and again normally after the " +
  "capture. Each restart takes about two minutes. Other machines in the cluster keep answering; a model on this " +
  "machine alone is offline for that time.";
export const KERNEL_PROFILE_NOTE =
  `While tracing is on, one CPU core runs at 100%. Tracing switches itself off after at most ${KERNEL_TRACING_LIMIT_MINUTES} ` +
  "minutes, even if this page is closed.";

/** The restarts' outcome in one plain line (VD-214). */
function kernelOutcomeLine(outcome: KernelProfileOutcome): string {
  const tracing = outcome.tracing_confirmed ? "Captured with GPU tracing on. " : "";
  return `${tracing}${outcome.revert_note}`;
}

/**
 * The on-demand profile action (VD-128, administrator only; VD-200, the
 * ClusterPerformanceAdvanced board). One click runs a single bounded profile
 * of the GPU serving process through the root bridge and shows each capture's
 * real result where it ran, and the honest reason where it could not - never
 * a fabricated flame graph. The GPU kernel profile (VD-214) is the second,
 * disruptive action: it restarts this controller's copy with GPU tracing on,
 * so it asks first and says what it costs.
 */
export function ProfilePanel({ session, className }: { session: Session; className?: string }) {
  const [profile, setProfile] = useState<ServingProfile | null>(null);
  const [running, setRunning] = useState<"" | "profile" | "kernel">("");
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState("");

  const start = (kind: "profile" | "kernel") => {
    setConfirming(false);
    setRunning(kind);
    setError("");
    (kind === "kernel" ? runGpuKernelProfile : runServingProfile)(session.csrf_token)
      .then((result) => setProfile(result))
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "The profile could not be run."))
      .finally(() => setRunning(""));
  };

  return (
    <PerfCard
      className={["perf-profile", className].filter(Boolean).join(" ")}
      label="On-demand profile"
      subtitle={"On-demand profile · administrators"}
      title="Profile the serving process"
      trailing={(
        <Button disabled={running !== ""} onClick={() => start("profile")} type="button" variant="secondary">
          {running === "profile" ? "Profiling…" : "Run a profile"}
        </Button>
      )}
    >
      <p className="perf-muted">
        A one-off, few-second capture of what the GPU serving process is doing; real where the tools can run on this
        box, with the reason stated where one cannot.
      </p>
      <div className="perf-capture__row">
        <p className="perf-muted">
          {running === "kernel"
            ? "Restarting the model with GPU tracing on, capturing, then restarting it normally. This takes a few minutes."
            : "Per-kernel GPU time on vLLM 0.27 needs the model restarted with GPU tracing on, for testing."}
        </p>
        <Button disabled={running !== ""} onClick={() => setConfirming(true)} type="button" variant="secondary">
          {running === "kernel" ? "Profiling GPU kernels…" : "GPU kernel profile (restarts the model)"}
        </Button>
      </div>
      {error && <Notice severity="warning">{error}</Notice>}
      {profile?.kernel_profile && <p className="perf-muted">{kernelOutcomeLine(profile.kernel_profile)}</p>}
      {profile && (
        <div className="perf-captures">
          {profile.captures.map((capture) => <CaptureRow capture={capture} key={capture.id} />)}
        </div>
      )}
      <ConfirmDialog
        busy={false}
        confirmLabel="Restart and profile"
        description={KERNEL_PROFILE_WARNING}
        note={KERNEL_PROFILE_NOTE}
        onCancel={() => setConfirming(false)}
        onConfirm={() => start("kernel")}
        open={confirming}
        title="Run a GPU kernel profile?"
      />
    </PerfCard>
  );
}
