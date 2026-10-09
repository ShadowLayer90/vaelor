import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import { formatQuantity } from "../lib/format";
import { jobIsReady, jobIsTerminal } from "../lib/jobPresentation";
import type { Session } from "../types";
import { Icon } from "./Icon";
import { ActionReviewDialog, type ProposedJob } from "./ActionReviewDialog";
import { KvGrid } from "./systemUi";
import { Button, LoadingLines, Notice, UnavailableValue } from "./ui";

interface MemoryStatus {
  total_bytes: number;
  available_bytes: number;
  used_percent: number;
  swap_total_bytes: number;
  swap_free_bytes: number;
  swappiness: number | null;
  zswap_enabled: boolean;
  zram_devices: string[];
  compressed_swap: boolean;
  pressure: { some_avg10: number | null; full_avg10: number | null };
  profiles: Array<{
    id: "balanced" | "ai-latency" | "capacity";
    name: string;
    swappiness: number;
    description: string;
  }>;
}

/** The fields of a job this panel reads while it waits for the policy to apply. */
interface MemoryJob {
  id: string;
  type: string;
  state: string;
  message?: string;
}

/** How long the panel follows the job before handing the owner to Activity. */
const JOB_POLL_INTERVAL_MS = 1500;
const JOB_POLL_ATTEMPTS = 40;

export function MemoryOptimizer({
  onClose,
  session,
}: {
  onClose: () => void;
  session: Session;
}) {
  const [status, setStatus] = useState<MemoryStatus | null>(null);
  const [profile, setProfile] = useState<MemoryStatus["profiles"][number]["id"]>("balanced");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  /** VD-189: a failure is an alert, never an info notice that reads like progress. */
  const [failure, setFailure] = useState("");
  const [pendingJob, setPendingJob] = useState<ProposedJob | null>(null);
  /** VD-189: a refused job stays in the review stacked over this panel; the panel under it is inert. */
  const [reviewError, setReviewError] = useState("");

  const mounted = useRef(true);
  // Set on every mount: a development double-mount (StrictMode) runs the
  // cleanup once, and a flag only ever cleared left the panel loading forever.
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);

  const load = useCallback(async () => {
    try {
      const result = await apiRequest<MemoryStatus>("/system/memory", { cache: "no-store" });
      if (!mounted.current) return null;
      setStatus(result);
      const current = result.profiles.find((item) => item.swappiness === result.swappiness);
      if (current) setProfile(current.id);
      return result;
    } catch (error) {
      if (mounted.current) setFailure(error instanceof Error ? error.message : "Memory policy could not be loaded.");
      return null;
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const review = () => {
    setPendingJob({
      type: "host.memory.optimize",
      payload: { profile, confirm: "apply-memory-profile" },
    });
  };

  const apply = async () => {
    if (!pendingJob) return;
    setBusy(true);
    setNotice(""); setFailure(""); setReviewError("");
    let queued = false;
    try {
      let job = await apiRequest<MemoryJob>("/jobs", {
        method: "POST",
        body: JSON.stringify({
          type: "host.memory.optimize",
          payload: pendingJob.payload,
        }),
      }, session.csrf_token);
      queued = true;
      setNotice("Applying the memory policy…");
      setPendingJob(null);
      // W4d-D21: the panel read the policy once, on open, and kept showing the
      // old swappiness after the job had applied the new one. Follow the job,
      // then read the policy back from the machine.
      for (let attempt = 0; attempt < JOB_POLL_ATTEMPTS && !jobIsTerminal(job); attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, JOB_POLL_INTERVAL_MS));
        if (!mounted.current) return;
        job = await apiRequest<MemoryJob>(`/jobs/${encodeURIComponent(job.id)}`, { cache: "no-store" });
      }
      if (!mounted.current) return;
      if (!jobIsTerminal(job)) {
        setNotice("The memory policy is still being applied. Activity shows when it finishes.");
      } else if (jobIsReady(job)) {
        const after = await load();
        setNotice(after
          ? `Memory policy applied without a reboot. Linux swappiness is now ${after.swappiness ?? "unknown"}.`
          : "Memory policy applied without a reboot.");
      } else {
        setNotice("");
        setFailure(job.message || "The memory policy could not be applied.");
      }
    } catch (error) {
      const text = error instanceof Error ? error.message : "The memory policy could not be queued.";
      if (queued) { setNotice(""); setFailure(text); } else setReviewError(text);
    } finally {
      setBusy(false);
    }
  };

  /*
   * The DialogsSystem board's memory optimizer. It renders inside the
   * ModalShell that Tune memory opens (SystemLiveReadings), so it draws the
   * dialog's header, body and footer itself.
   */
  const compressed = status?.zswap_enabled
    ? "zswap detected"
    : status?.zram_devices.length
      ? `${status.zram_devices.length} zram device detected`
      : "Compressed swap is not enabled; Vaelor will not add it automatically.";
  return (
    <section className="sys-dialog-content" aria-labelledby="memory-optimizer-title">
      <header className="sys-dialog__header">
        <div className="sys-dialog__titles">
          <span className="sys-eyebrow">System memory</span>
          <h2 id="memory-optimizer-title">Memory optimizer</h2>
          <p>Choose how Linux balances active applications, filesystem cache, and swap. Local AI is sized separately from the selected model.</p>
        </div>
        <Button variant="quiet" onClick={onClose}>Close</Button>
      </header>
      <div className="sys-dialog__body">
        {status ? <>
          <KvGrid
            className="sys-kv--large"
            items={[
              { label: "Available RAM", value: formatQuantity(status.available_bytes, "free"), detail: `${status.used_percent}% in use` },
              {
                label: "Swap",
                value: status.swap_total_bytes ? formatQuantity(status.swap_total_bytes - status.swap_free_bytes, "used") : "Not configured",
                detail: status.compressed_swap ? "Compressed swap active" : status.swap_total_bytes ? "Swap on disk, not compressed" : "No swap",
              },
              /*
               * `?? 0` here once rendered "0.0%" on any kernel without PSI,
               * where `system_inventory.py` reports `some_avg10: None`. "Nothing
               * is waiting" and "this kernel does not measure that" differ.
               */
              {
                label: "Memory pressure",
                value: status.pressure.some_avg10 === null
                  ? <UnavailableValue label="Memory pressure unavailable" reason="This kernel does not publish /proc/pressure/memory, so waiting time is not measured" />
                  : `${status.pressure.some_avg10.toFixed(1)}%`,
                detail: "10-second waiting average",
              },
              { label: "Current policy", value: status.swappiness ?? "Unknown", detail: "Linux swappiness" },
            ]}
            label="Memory now"
          />
          <div className="sys-choice-grid sys-choice-grid--3">
            {status.profiles.map((item) => (
              <Button aria-pressed={profile === item.id} className="sys-choice sys-choice--profile" key={item.id} onClick={() => setProfile(item.id)} type="button" variant="secondary">
                <Icon name={item.id === "ai-latency" ? "cpu" : item.id === "capacity" ? "database" : "memory"} size={18} />
                <strong>{item.name}</strong>
                <small>{item.description}</small>
                <span className="sys-mono">Swappiness {item.swappiness}</span>
              </Button>
            ))}
          </div>
          <div className="sys-callout">
            <Icon name="shield" size={18} />
            <p><strong>What Vaelor changes.</strong> Only the reviewed Linux swappiness profile. Docker workloads keep hard memory limits, and llama.cpp keeps a separate OS reserve based on live available RAM and model size.</p>
          </div>
        </> : !failure && <LoadingLines label="Reading Linux memory policy…" />}
        {notice && <Notice severity={notice.startsWith("Memory policy applied") ? "success" : "info"}>{notice}</Notice>}
        {failure && <Notice severity="danger">{failure}</Notice>}
      </div>
      {status && (
        <footer className="sys-dialog__footer sys-dialog__footer--split">
          <span className="sys-muted">{compressed}</span>
          <Button variant="primary" disabled={busy} disabledReason={session.user.role === "administrator" ? undefined : "Administrator access is required to change a Linux memory policy."} onClick={review}>{busy ? "Preparing…" : "Review memory plan"}</Button>
        </footer>
      )}
      <ActionReviewDialog
        busy={busy}
        error={reviewError}
        job={pendingJob}
        onApprove={() => void apply()}
        onCancel={() => { if (!busy) { setReviewError(""); setPendingJob(null); } }}
        summary={`Apply the ${status?.profiles.find((item) => item.id === profile)?.name ?? profile} policy. This changes only Linux swappiness and does not reboot this machine.`}
        valueLabels={{ profile: status?.profiles.find((item) => item.id === profile)?.name ?? profile }}
      />
    </section>
  );
}
