import { useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import { formatQuantity } from "../lib/format";
import type { Session } from "../types";
import { Icon } from "./Icon";
import { ActionReviewDialog, type ProposedJob } from "./ActionReviewDialog";
import { Button, Notice, UnavailableValue } from "./ui";

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
  const [pendingJob, setPendingJob] = useState<ProposedJob | null>(null);

  useEffect(() => {
    void apiRequest<MemoryStatus>("/system/memory")
      .then((result) => {
        setStatus(result);
        const current = result.profiles.find((item) => item.swappiness === result.swappiness);
        if (current) setProfile(current.id);
      })
      .catch((error) => setNotice(error instanceof Error ? error.message : "Memory policy could not be loaded."));
  }, []);

  const review = () => {
    setPendingJob({
      type: "host.memory.optimize",
      payload: { profile, confirm: "apply-memory-profile" },
    });
  };

  const apply = async () => {
    if (!pendingJob) return;
    setBusy(true);
    setNotice("");
    try {
      await apiRequest("/jobs", {
        method: "POST",
        body: JSON.stringify({
          type: "host.memory.optimize",
          payload: pendingJob.payload,
        }),
      }, session.csrf_token);
      setNotice("Memory policy queued. It applies without a reboot and will appear in Activity.");
      setPendingJob(null);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The memory policy could not be queued.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="memory-optimizer" aria-labelledby="memory-optimizer-title">
      <div className="panel-heading">
        <div>
          <span className="page-eyebrow">System memory</span>
          <h2 id="memory-optimizer-title">Memory optimizer</h2>
          <p>Choose how Linux balances active applications, filesystem cache, and swap. Local AI is sized separately from the selected model.</p>
        </div>
        <Button variant="quiet" onClick={onClose}>Close</Button>
      </div>
      {status ? <>
        <div className="memory-optimizer__facts">
          <span><small>Available RAM</small><strong>{formatQuantity(status.available_bytes, "free")}</strong><em>{status.used_percent}% in use</em></span>
          <span><small>Swap</small><strong>{status.swap_total_bytes ? formatQuantity(status.swap_total_bytes - status.swap_free_bytes, "used") : "Not configured"}</strong><em>{status.compressed_swap ? "Compressed swap active" : "Disk or no swap"}</em></span>
          {/*
            * `?? 0` here rendered "0.0%" — a claim of no memory pressure at
            * all — on any kernel without PSI, where `system_inventory.py`
            * deliberately reports `some_avg10: None`. "Nothing is waiting on
            * memory" and "this kernel does not measure that" are different
            * facts, and the second is the one a reader needs before trusting
            * the first.
            */}
          <span><small>Memory pressure</small><strong>{status.pressure.some_avg10 === null
            ? <UnavailableValue label="Memory pressure unavailable" reason="This kernel does not publish /proc/pressure/memory, so waiting time is not measured" />
            : `${status.pressure.some_avg10.toFixed(1)}%`}</strong><em>10-second waiting average</em></span>
          <span><small>Current policy</small><strong>{status.swappiness ?? "Unknown"}</strong><em>Linux swappiness</em></span>
        </div>
        <div className="memory-policy-grid">
          {status.profiles.map((item) => (
            <Button aria-pressed={profile === item.id} className="memory-policy-option" key={item.id} onClick={() => setProfile(item.id)} type="button">
              <span><Icon name={item.id === "ai-latency" ? "cpu" : item.id === "capacity" ? "database" : "memory"} /></span>
              <strong>{item.name}</strong>
              <p>{item.description}</p>
              <small>Swappiness {item.swappiness}</small>
            </Button>
          ))}
        </div>
        <div className="memory-optimizer__guidance">
          <Icon name="shield" />
          <div>
            <strong>What Vaelor changes</strong>
            <p>Only the reviewed Linux swappiness profile. Docker workloads keep hard memory limits, and llama.cpp keeps a separate OS reserve based on live available RAM and model size.</p>
          </div>
        </div>
        <div className="memory-optimizer__actions">
          <span>{status.zswap_enabled ? "zswap detected" : status.zram_devices.length ? `${status.zram_devices.length} zram device detected` : "Compressed swap is not enabled; Vaelor will not add it automatically."}</span>
          <Button variant="primary" disabled={busy} disabledReason={session.user.role === "administrator" ? undefined : "Administrator access is required to change a Linux memory policy."} onClick={review}>{busy ? "Preparing…" : "Review memory plan"}</Button>
        </div>
      </> : <div className="page-loading" role="status">Reading Linux memory policy…</div>}
      {notice && <Notice severity="info">{notice}</Notice>}
      <ActionReviewDialog
        busy={busy}
        job={pendingJob}
        onApprove={() => void apply()}
        onCancel={() => !busy && setPendingJob(null)}
        summary={`Apply the ${status?.profiles.find((item) => item.id === profile)?.name ?? profile} policy. This changes only Linux swappiness and does not reboot this machine.`}
      />
    </section>
  );
}
