import { useCallback, useEffect, useState } from "react";
import { useMachineProfile } from "../hooks/useMachineProfile";
import { apiRequest } from "../lib/api";
import { canonicalOperationState, jobIsReady, jobIsTerminal, jobNeedsAttention } from "../lib/jobPresentation";
import { retentionNote, type RetentionStatus } from "../lib/retention";
import { mediaSlotRows, reserveNote, storageReading } from "../lib/storage";
import type { Session } from "../types";
import type { SectionStatus } from "./FanControl";
import type { StorageSummary } from "./Sidebar";
import { NetworkCard, StorageCard, type Connectivity, type NetworkInterface, type StorageDevice, type StorageVolume } from "./SystemHardwareCards";
import { FrontDisplayCard, ServiceLogDialog, ServicesCard, serviceView, type DisplayState, type InventoryService } from "./SystemServicesCard";
import { SoftwareUpdatesCard, type PackageUpdates } from "./SystemSoftwareUpdates";
import { SystemUpdatePanel } from "./SystemUpdatePanel";
import type { UpdateJob } from "./UpdateJobStatus";
import { useVisiblePoll } from "./systemUi";
import { Notice } from "./ui";

export { serviceView } from "./SystemServicesCard";

interface Inventory {
  storage: {
    devices: StorageDevice[];
    volumes: StorageVolume[];
    media_counts: Record<string, number>;
    media_presence: Record<string, { present: boolean; count: number }>;
    nvme_temperatures: Record<string, number>;
    smartctl_available: boolean;
  };
  network: {
    interfaces: NetworkInterface[];
    dns: string[];
    routes: Array<Record<string, unknown>>;
  };
  services: InventoryService[];
  updates: PackageUpdates;
}

type Outcome = { text: string; refused: boolean } | null;

/**
 * System › Hardware and services (VD-200, the System and SystemHardwarePi
 * boards): Storage beside Network, Vaelor's services (beside the front display
 * on a Pi), then Update Vaelor beside the operating system's updates. It reads
 * `/system/inventory` and `/system/display` and hands each card its part; each
 * card says its own outcome, where its button is.
 */
export function SystemInventoryPanel({
  onStatus,
  reloadToken = 0,
  session,
}: {
  /** The page header's pill ("Attention needed"), reported up to the System page. */
  onStatus?: (status: SectionStatus | null) => void;
  /** Bumped by the System page's Reload. */
  reloadToken?: number;
  session: Session;
}) {
  const [inventory, setInventory] = useState<Inventory | null>(null);
  const [display, setDisplay] = useState<DisplayState | null>(null);
  /** The inventory read failed: said once, at the top of the tab. */
  const [readFailure, setReadFailure] = useState("");
  const [connectivity, setConnectivity] = useState<Connectivity | null>(null);
  /** When the check last ran, so a repeat press visibly changes something. */
  const [checkedAt, setCheckedAt] = useState(0);
  const [networkRefusal, setNetworkRefusal] = useState("");
  const [logs, setLogs] = useState<{ service: string; output: string } | null>(null);
  const [logsRefusal, setLogsRefusal] = useState("");
  const [busy, setBusy] = useState("");
  const [displayOutcome, setDisplayOutcome] = useState<Outcome>(null);
  const [updateOutcome, setUpdateOutcome] = useState<Outcome>(null);
  const [updateJob, setUpdateJob] = useState<UpdateJob | null>(null);
  /** Hides the finished live-status panel once the reader has acknowledged it. */
  const [dismissedJobId, setDismissedJobId] = useState("");
  const [retention, setRetention] = useState<RetentionStatus | null>(null);
  const machine = useMachineProfile();
  const canControl = session.user.role !== "viewer";

  // Retention (#186) is a soft add: a control plane that predates the route,
  // or a failed read, adds no line rather than a wrong one.
  useEffect(() => {
    let cancelled = false;
    void apiRequest<RetentionStatus>("/telemetry/retention")
      .then((next) => { if (!cancelled) setRetention(next); })
      .catch(() => undefined);
    return () => { cancelled = true; };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [nextInventory, nextDisplay] = await Promise.all([
        apiRequest<Inventory>("/system/inventory"),
        apiRequest<DisplayState>("/system/display"),
      ]);
      setInventory(nextInventory);
      setDisplay(nextDisplay);
      setReadFailure("");
    } catch (error) {
      setReadFailure(error instanceof Error ? error.message : "System details are unavailable.");
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);
  useVisiblePoll(() => void refresh(), 15_000);

  useEffect(() => {
    if (reloadToken > 0) void refresh();
  }, [refresh, reloadToken]);

  const testNetwork = async () => {
    setBusy("network");
    setNetworkRefusal("");
    try {
      setConnectivity(await apiRequest<Connectivity>(
        "/system/network/test", { method: "POST", body: "{}" }, session.csrf_token,
      ));
      setCheckedAt(Date.now());
    } catch (error) {
      setNetworkRefusal(error instanceof Error ? error.message : "Connection test failed.");
    } finally {
      setBusy("");
    }
  };

  const updateDisplay = async (patch: Partial<DisplayState>) => {
    setBusy("display");
    setDisplayOutcome(null);
    try {
      const next = await apiRequest<DisplayState>(
        "/system/display",
        { method: "PATCH", body: JSON.stringify(patch) },
        session.csrf_token,
      );
      setDisplay(next);
      setDisplayOutcome({ text: "Front display settings applied.", refused: false });
    } catch (error) {
      setDisplayOutcome({ text: error instanceof Error ? error.message : "Display settings could not be changed.", refused: true });
    } finally {
      setBusy("");
    }
  };

  const openLogs = async (service: string) => {
    setBusy(`logs-${service}`);
    setLogsRefusal("");
    try {
      setLogs(await apiRequest<{ service: string; output: string }>(
        `/system/services/${encodeURIComponent(service)}/logs?lines=120`,
      ));
    } catch (error) {
      setLogsRefusal(error instanceof Error ? error.message : "Service logs are unavailable.");
    } finally {
      setBusy("");
    }
  };

  const runUpdate = async (action: "stage" | "apply") => {
    setBusy(`update-${action}`);
    setUpdateOutcome(null);
    setDismissedJobId("");
    try {
      let job = await apiRequest<UpdateJob>(
        "/jobs",
        {
          method: "POST",
          body: JSON.stringify({
            type: `system.update.${action}`,
            payload: action === "apply" ? { confirm: "apply-updates" } : {},
          }),
        },
        session.csrf_token,
      );
      setUpdateJob(job);
      for (let attempt = 0; attempt < 1200; attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 3000));
        job = await apiRequest<UpdateJob>(`/jobs/${encodeURIComponent(job.id)}`);
        setUpdateJob(job);
        if (jobIsTerminal(job)) break;
      }
      if (jobIsReady(job)) {
        // The job's own line: it names any update the install held back
        // rather than calling a partial install complete (W4d-D31).
        setUpdateOutcome({
          text: action === "stage"
            ? "Updates downloaded. Review and install them when you are ready."
            : job.message || "System updates installed.",
          refused: false,
        });
        await refresh();
      } else if (jobNeedsAttention(job) || canonicalOperationState(job) === "rejected") {
        throw new Error(job.message || "The update job failed.");
      }
    } catch (error) {
      setUpdateOutcome({ text: error instanceof Error ? error.message : "The update could not be completed.", refused: true });
    } finally {
      setBusy("");
    }
  };

  const loading = inventory === null && !readFailure;
  const services = inventory?.services ?? [];
  const servicesHealthy = inventory !== null && services.every((service) => serviceView(service).settled);
  // The storage facts the storage panel carried (the design guide's placement
  // map): the filesystem's reserve, the media slots and the retention warning.
  const storageSummary = (inventory?.storage ?? null) as StorageSummary | null;
  const notes = [reserveNote(storageReading(storageSummary, {})), retentionNote(retention)].filter(Boolean);
  const isAppliance = machine?.machine_class === "pi-appliance";
  // A workstation board draws no front display; a Pi always has the card,
  // and so does any machine whose display answered as detected.
  const showDisplay = isAppliance || Boolean(display?.detected);

  /*
   * LESSONS 1 / S-Y3: a failed re-read keeps the last inventory on screen,
   * so the pill must stop calling it healthy - it is the previous answer,
   * grey, and says so.
   */
  const stale = inventory !== null && Boolean(readFailure);
  const status: SectionStatus = inventory === null
    ? { label: readFailure ? "Not read" : "Checking hardware", tone: "neutral", reading: "unread" }
    : stale
      ? { label: "Old reading", tone: "neutral", reading: "stale" }
      : servicesHealthy ? { label: "Services healthy", tone: "success" } : { label: "Attention needed", tone: "warning" };
  const statusKey = `${status.label}|${status.tone}|${status.reading ?? ""}`;
  useEffect(() => {
    onStatus?.(status);
    // The key carries every field the pill shows.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onStatus, statusKey]);
  useEffect(() => () => onStatus?.(null), [onStatus]);

  const updateBusy = busy === "update-stage" ? "stage" : busy === "update-apply" ? "apply" : "";

  return (
    <div className="sys-section sys-hardware" id="system-health">
      {readFailure && (
        <Notice severity="danger">
          {stale ? `${readFailure} The services and storage below are from the last reading that answered.` : readFailure}
        </Notice>
      )}
      <div className="sys-grid-2">
        <StorageCard
          devices={inventory?.storage?.devices ?? []}
          loading={loading}
          mediaSlots={mediaSlotRows(storageSummary)}
          notes={notes}
          temperatures={inventory?.storage?.nvme_temperatures ?? {}}
          volumes={inventory?.storage?.volumes ?? []}
        />
        <NetworkCard
          busy={busy === "network"}
          canTest={canControl}
          checkedAt={checkedAt}
          connectivity={connectivity}
          interfaces={inventory?.network?.interfaces ?? []}
          loading={loading}
          onTest={() => void testNetwork()}
          refusal={networkRefusal}
        />
      </div>

      <div className={showDisplay ? "sys-grid-2" : "sys-grid-1"}>
        <ServicesCard
          busyService={busy.startsWith("logs-") ? busy.slice(5) : ""}
          canViewLogs={canControl}
          loading={loading}
          logsRefusal={logsRefusal}
          onViewLogs={(id) => void openLogs(id)}
          services={services}
          stale={stale}
        />
        {showDisplay && (
          <FrontDisplayCard
            busy={busy === "display"}
            canControl={canControl}
            display={display}
            loading={display === null && !readFailure}
            onChange={(patch) => void updateDisplay(patch)}
            outcome={displayOutcome}
          />
        )}
      </div>

      <div className="sys-grid-2">
        <SystemUpdatePanel session={session} />
        <SoftwareUpdatesCard
          busy={updateBusy}
          canStage={canControl}
          isAdministrator={session.user.role === "administrator"}
          job={updateJob && updateJob.id !== dismissedJobId ? updateJob : null}
          loading={loading}
          onDismissJob={() => setDismissedJobId(updateJob?.id ?? "")}
          onRun={(action) => void runUpdate(action)}
          outcome={updateOutcome}
          updates={inventory?.updates}
        />
      </div>

      {logs && <ServiceLogDialog logs={logs} onClose={() => setLogs(null)} />}
    </div>
  );
}
