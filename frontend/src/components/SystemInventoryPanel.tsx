import { useCallback, useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import { formatQuantity, timeAgo } from "../lib/format";
import { canonicalOperationState, jobIsReady, jobIsTerminal, jobNeedsAttention } from "../lib/jobPresentation";
import type { Session } from "../types";
import { Icon } from "./Icon";
import { PaginatedItems } from "./PaginatedItems";
import { StatusPill } from "./StatusPill";
import { Button, Checkbox, Notice, Select } from "./ui";

interface StorageDevice {
  name: string;
  path?: string;
  type?: string;
  size?: number;
  model?: string;
  tran?: string;
  mountpoints?: Array<string | null>;
  children?: StorageDevice[];
}

interface StorageVolume {
  id: string;
  device_id: string;
  name: string;
  path: string;
  model: string;
  kind: "nvme" | "usb" | "microsd" | "sata" | "other";
  mountpoint: string;
  total_bytes: number;
  used_bytes: number;
  free_bytes: number;
  used_percent: number;
  removable: boolean;
}

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
    interfaces: Array<{
      name: string;
      state: string;
      addresses: Array<{ family: string; address: string; prefix: number }>;
      rx_bytes: number;
      tx_bytes: number;
      rx_errors: number;
      tx_errors: number;
    }>;
    dns: string[];
    routes: Array<Record<string, unknown>>;
  };
  services: Array<{
    id: string;
    unit: string;
    available: boolean;
    active: string;
    substate: string;
    enabled: string;
    restarts: number;
    exit_status: number;
  }>;
  updates: {
    available: boolean;
    count: number;
    packages: string[];
    details: Array<{
      name: string;
      installed: string;
      candidate: string;
      source: string;
      classification: string;
    }>;
    staged: boolean;
    staged_archive_count: number;
    staged_size_bytes: number;
    estimated_install_minutes: number;
    reboot_likelihood: "unlikely" | "possible" | "likely";
    reboot_required: boolean;
    last_action?: "stage" | "apply";
    last_completed_at?: number;
  };
}

interface DisplayState {
  detected: boolean;
  enabled: boolean;
  rotation: number;
  sleep_timeout: number;
  pages: string[];
  disk?: string;
  network_interface?: string;
  hardware?: string;
  bus?: string;
}

interface Connectivity {
  dns: boolean;
  internet: boolean;
  /** `null` where the probe could not time a round trip. */
  latency_ms: number | null;
}

interface UpdateJob {
  id: string;
  type: string;
  state: string;
  operation_state?: string;
  attention?: boolean;
  retryable?: boolean;
  readiness?: string;
  liveness?: string;
  progress: number;
  message: string;
}

const flattenDevices = (devices: StorageDevice[]): StorageDevice[] =>
  devices.flatMap((device) => [device, ...flattenDevices(device.children ?? [])]);

export function SystemInventoryPanel({ session }: { session: Session }) {
  const [inventory, setInventory] = useState<Inventory | null>(null);
  const [display, setDisplay] = useState<DisplayState | null>(null);
  const [connectivity, setConnectivity] = useState<Connectivity | null>(null);
  /** When the check last ran, so a repeat press visibly changes something. */
  const [checkedAt, setCheckedAt] = useState(0);
  const [logs, setLogs] = useState<{ service: string; output: string } | null>(null);
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState("");
  const [updateJob, setUpdateJob] = useState<UpdateJob | null>(null);
  const [confirmApply, setConfirmApply] = useState(false);
  const [expandedStorageId, setExpandedStorageId] = useState("");
  const canControl = session.user.role !== "viewer";

  const refresh = useCallback(async () => {
    try {
      const [nextInventory, nextDisplay] = await Promise.all([
        apiRequest<Inventory>("/system/inventory"),
        apiRequest<DisplayState>("/system/display"),
      ]);
      setInventory(nextInventory);
      setDisplay(nextDisplay);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "System details are unavailable.");
    }
  }, []);

  useEffect(() => {
    void refresh();
    const interval = window.setInterval(() => void refresh(), 15_000);
    return () => window.clearInterval(interval);
  }, [refresh]);

  const testNetwork = async () => {
    setBusy("network");
    setMessage("");
    try {
      setConnectivity(await apiRequest<Connectivity>(
        "/system/network/test", { method: "POST", body: "{}" }, session.csrf_token,
      ));
      setCheckedAt(Date.now());
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Connection test failed.");
    } finally {
      setBusy("");
    }
  };

  const updateDisplay = async (patch: Partial<DisplayState>) => {
    setBusy("display");
    setMessage("");
    try {
      const next = await apiRequest<DisplayState>(
        "/system/display",
        { method: "PATCH", body: JSON.stringify(patch) },
        session.csrf_token,
      );
      setDisplay(next);
      setMessage("Front display settings applied.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Display settings could not be changed.");
    } finally {
      setBusy("");
    }
  };

  const openLogs = async (service: string) => {
    setBusy(`logs-${service}`);
    setMessage("");
    try {
      setLogs(await apiRequest<{ service: string; output: string }>(
        `/system/services/${encodeURIComponent(service)}/logs?lines=120`,
      ));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Service logs are unavailable.");
    } finally {
      setBusy("");
    }
  };

  const runUpdate = async (action: "stage" | "apply") => {
    setBusy(`update-${action}`);
    setMessage("");
    setConfirmApply(false);
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
        setMessage(action === "stage"
          ? "Updates downloaded. Review and install them when you are ready."
          : "System updates installed successfully.");
        await refresh();
      } else if (jobNeedsAttention(job) || canonicalOperationState(job) === "rejected") {
        throw new Error(job.message || "The update job failed.");
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "The update could not be completed.");
    } finally {
      setBusy("");
    }
  };

  const devices = flattenDevices(inventory?.storage?.devices ?? [])
    .filter((device) => device.type === "disk");
  const volumes = inventory?.storage?.volumes ?? [];
  const storageGroups = Object.values(volumes.reduce<Record<string, StorageVolume[]>>((groups, volume) => {
    const key = volume.device_id || volume.path || volume.id;
    (groups[key] ??= []).push(volume);
    return groups;
  }, {}));
  const networkInterfaces = inventory?.network?.interfaces ?? [];
  const updates = inventory?.updates;
  const inventoryLoading = inventory === null;
  const services = inventory?.services ?? [];
  const servicesHealthy = inventory !== null
    && services.every((service) => !service.available || service.active === "active");
  const mediumLabel = (kind: string) => ({
    nvme: "NVMe",
    usb: "USB storage",
    microsd: "microSD",
    sata: "SATA storage",
    other: "Storage",
  }[kind] ?? "Storage");

  return (
    <section className="system-inventory" id="system-health" aria-labelledby="system-health-title">
      <div className="section-heading">
        <div>
          {/*
            * The rail says "System", the tab chip says "Hardware & services",
            * and this heading used to say "Appliance health center" - a third
            * name for the same page, and one word away from the sidebar's
            * "Appliance health" panel, which reports something else entirely.
            * The breadcrumb names the section and the heading names the tab.
            */}
          <span className="page-eyebrow">System</span>
          <h2 id="system-health-title">Hardware & services</h2>
          <p>Understand storage, networking, Vaelor services, updates, and the front display without opening a terminal.</p>
        </div>
        <StatusPill
          label={inventoryLoading ? "Checking hardware" : servicesHealthy ? "Services healthy" : "Attention needed"}
          status={inventoryLoading ? "neutral" : servicesHealthy ? "healthy" : "degraded"}
        />
      </div>

      {message && <Notice severity="info"><Icon name="activity" />{message}</Notice>}

      <div className="system-inventory__grid">
        <article className="data-panel system-card">
          <div className="panel-heading">
            <div><h3>Storage</h3><p>Physical storage devices and available space.</p></div>
            <Icon name="nvme" />
          </div>
          <div className="system-card__list">
            {storageGroups.length ? storageGroups.map((group) => {
              const primary = [...group].sort((left, right) => right.total_bytes - left.total_bytes)[0];
              const storageId = primary.device_id || primary.path || primary.id;
              const detailsId = `storage-details-${storageId.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
              const expanded = expandedStorageId === storageId;
              const temperature = Object.entries(inventory?.storage?.nvme_temperatures ?? {})
                .find(([key]) => key.includes(storageId) || key.includes(primary.name))?.[1];
              const mountCount = new Set(group.map((volume) => volume.mountpoint).filter(Boolean)).size;
              return (
                <div className={`storage-device${expanded ? " storage-device--expanded" : ""}`} key={storageId}>
                  <Button
                    aria-controls={detailsId}
                    aria-expanded={expanded}
                    className="storage-device__trigger"
                    onClick={() => setExpandedStorageId(expanded ? "" : storageId)}
                    variant="quiet"
                  >
                    <span className="storage-device__icon"><Icon name={primary.kind === "nvme" ? "nvme" : "database"} /></span>
                    <span className="storage-device__summary">
                      <strong>{mediumLabel(primary.kind)} · {primary.model || primary.name}</strong>
                      <small>{formatQuantity(primary.total_bytes, "capacity")} total · {formatQuantity(primary.free_bytes, "free")} free · {mountCount} mount{mountCount === 1 ? "" : "s"}</small>
                    </span>
                    <Icon className="storage-device__chevron" name="chevron" size={16} />
                  </Button>
                  {expanded && (
                    <div className="storage-device__details" id={detailsId}>
                      <div className="storage-device__mounts">
                        <strong>Mounted volumes</strong>
                        {group.map((volume) => (
                          <div className="storage-volume-row" key={volume.id}>
                            <span>{volume.mountpoint || "Not mounted"}</span>
                            <small>{volume.used_percent.toFixed(0)}% used · {formatQuantity(volume.free_bytes, "free")} free of {formatQuantity(volume.total_bytes, "capacity")}</small>
                          </div>
                        ))}
                      </div>
                      <dl className="storage-device__facts">
                        <div><dt>Device</dt><dd>{primary.device_id || primary.path || primary.name}</dd></div>
                        <div><dt>Connection</dt><dd>{mediumLabel(primary.kind)}</dd></div>
                        <div><dt>Drive temperature</dt><dd>{temperature === undefined ? "Not reported" : `${temperature}°C`}</dd></div>
                      </dl>
                    </div>
                  )}
                </div>
              );
            }) : devices.length ? devices.map((device) => (
              <div className="system-device-row" key={device.path ?? device.name}>
                <span><Icon name={device.tran === "nvme" ? "nvme" : "database"} /></span>
                <div>
                  <strong>{device.model?.trim() || device.name}</strong>
                  <small>{device.path ?? device.name} · {formatQuantity(device.size, "capacity")}</small>
                </div>
              </div>
            )) : <p className="system-card__empty">{inventoryLoading ? "Loading storage devices…" : "No block-device details were returned."}</p>}
          </div>
        </article>

        <article className="data-panel system-card">
          <div className="panel-heading">
            <div><h3>Network</h3><p>Interface traffic and a safe outbound check.</p></div>
            <Icon name="network" />
          </div>
          <div className="system-card__list">
            {networkInterfaces.filter((item) => item.name !== "lo").map((item) => (
              <div className="network-row" key={item.name}>
                <span className={`network-state network-state--${item.state}`} />
                <div>
                  <strong>{item.name} <small>{item.state}</small></strong>
                  <small>{item.addresses.map((address) => address.address).join(", ") || "No address"}</small>
                  <small>Received {formatQuantity(item.rx_bytes, "transfer")} · Sent {formatQuantity(item.tx_bytes, "transfer")} · {item.rx_errors + item.tx_errors} errors</small>
                </div>
              </div>
            ))}
            {inventoryLoading && <p className="system-card__empty">Loading network interfaces…</p>}
            {/* Without this the panel resolved to nothing but its own heading,
                which reads as broken rather than as "none found". */}
            {!inventoryLoading && !networkInterfaces.filter((item) => item.name !== "lo").length && (
              <p className="system-card__empty">
                No network interfaces were reported. The connection test below still works,
                and the sidebar shows current internet status.
              </p>
            )}
          </div>
          {/*
            * The outcome sits above the control, not below it.
            *
            * A tester pressed this three times and reported no spinner, no
            * result and no change — while Activity recorded all three runs. The
            * result was rendering: it was rendering *underneath* the last
            * element of a card whose interface list is long on any Docker host,
            * so on an ordinary window it landed below the fold. Pressing a
            * button and being shown nothing is the defect this release exists
            * to remove, and it does not stop being that because the markup was
            * technically present.
            *
            * The timestamp is the other half. Without it a second press
            * re-rendered a byte-identical pill, so even a reader looking
            * straight at the result could not tell the check had run again.
            */}
          {connectivity && (
            <div className="connectivity-result" role="status" aria-live="polite">
              <StatusPill label={connectivity.internet ? "Internet reachable" : "Internet unavailable"} status={connectivity.internet ? "healthy" : "degraded"} />
              <span>
                DNS {connectivity.dns ? "working" : "failed"}
                {connectivity.latency_ms === null ? "" : ` · round trip to the internet ${connectivity.latency_ms} ms`}
                {checkedAt ? ` · checked ${timeAgo(checkedAt)}` : ""}
              </span>
            </div>
          )}
          <Button disabled={!canControl || busy === "network"} onClick={() => void testNetwork()}>
            {busy === "network" ? "Testing…" : connectivity ? "Test again" : "Test internet connection"}
          </Button>
        </article>

        <article className="data-panel system-card system-card--services">
          <div className="panel-heading">
            <div><h3>Vaelor services</h3><p>Only control-plane-owned services are exposed here.</p></div>
            <Icon name="settings" />
          </div>
          <div className="service-table">
            {(inventory?.services ?? []).map((service) => (
              <div className="service-row" key={service.id}>
                <div>
                  <strong>{service.id.replaceAll("-", " ")}</strong>
                  <small>{service.available ? `${service.substate} · ${service.restarts} restarts` : "Not installed"}</small>
                </div>
                <StatusPill label={service.available ? service.active : "Optional"} status={service.active === "active" ? "healthy" : "neutral"} />
                <Button variant="quiet" disabled={!canControl || busy === `logs-${service.id}`} onClick={() => void openLogs(service.id)}>
                  View logs
                </Button>
              </div>
            ))}
            {inventoryLoading && <p className="system-card__empty">Checking appliance services…</p>}
          </div>
          <div className="system-card__footer">
            <span>Software updates</span>
            <strong>
              {inventoryLoading
                ? "Checking…"
                : updates?.reboot_required
                ? "Restart required"
                : updates?.count
                  ? `${updates.count} available`
                  : "Up to date"}
            </strong>
          </div>
          <div className="system-update-controls">
            {inventoryLoading ? <small>Checking operating-system updates…</small> : updates?.packages?.length ? (
              <details>
                <summary>Review {updates.count} package updates</summary>
                <div className="system-update-impact">
                  <span><small>Downloaded</small><strong>{updates.staged ? formatQuantity(updates.staged_size_bytes, "capacity") : "Not staged"}</strong></span>
                  <span><small>Estimated install</small><strong>About {updates.estimated_install_minutes} min</strong></span>
                  <span><small>Restart likelihood</small><strong>{updates.reboot_likelihood}</strong></span>
                </div>
                <p className="system-update-explainer">Package scripts may restart affected desktop or background services. Vaelor keeps the device online unless it explicitly reports that a full restart is required.</p>
                <div className="system-update-packages">
                  <PaginatedItems items={updates.details ?? []} label="Package updates" pageSize={10} render={(item) => (
                    <span className="system-update-package" key={item.name}>
                      <strong>{item.name}</strong>
                      <small>{item.installed} → {item.candidate}</small>
                      <em>{item.classification}</em>
                    </span>
                  )} />
                </div>
              </details>
            ) : <small>No pending operating-system packages were reported.</small>}
            {updateJob && busy.startsWith("update-") && (
              <div className="update-progress" role="status">
                <span>{updateJob.message || "Preparing update job"}</span>
                <strong>{updateJob.progress}%</strong>
              </div>
            )}
            <div className="system-update-actions">
              <Button
                disabled={!canControl || !updates?.available || busy.startsWith("update-")}
                onClick={() => void runUpdate("stage")}
              >
                {busy === "update-stage" ? "Downloading…" : updates?.staged ? "Download again" : "Download updates"}
              </Button>
              {updates?.staged && !confirmApply && (
                <Button
                  disabled={session.user.role !== "administrator" || busy.startsWith("update-")}
                  onClick={() => setConfirmApply(true)}
                  variant="primary"
                >
                  Install staged updates
                </Button>
              )}
            </div>
            {updates?.staged && session.user.role !== "administrator" && (
              <small>Administrator access is required to install operating-system updates.</small>
            )}
            {confirmApply && (
              <div className="update-confirm" role="alert">
                <strong>Install the downloaded updates now?</strong>
                <p>Services may restart. Vaelor will tell you if the device itself needs a restart.</p>
                <div>
                  <Button variant="primary" onClick={() => void runUpdate("apply")}>Yes, install updates</Button>
                  <Button variant="quiet" onClick={() => setConfirmApply(false)}>Not now</Button>
                </div>
              </div>
            )}
          </div>
        </article>

        <article className="data-panel system-card">
          <div className="panel-heading">
            {/*
              * Three controls sit on one screen under two different save
              * models: Cooling and Case lighting stage a change and wait for
              * Apply or Save, and this one writes as soon as you change it.
              * Nothing said so, so changing "Sleep after" produced no
              * confirmation, no button and no banner — and a reader had no way
              * to know whether they still owed the page a press. The card
              * states its own model rather than leaving it to be discovered by
              * reloading.
              */}
            <div><h3>Front OLED display</h3><p>Choose when the front status screen is active. Changes here save as soon as you make them — there is nothing to apply.</p></div>
            <Icon name="oled" />
          </div>
          {display?.detected ? (
            <div className="display-controls">
              <div className="switch-row">
                <span><strong>Display power</strong><small>Show live system information</small></span>
                <Checkbox
                  checked={display.enabled}
                  disabled={!canControl || busy === "display"}
                  disabledReason={!canControl ? "Operator access is required to change display power." : undefined}
                  id="display-power"
                  label="Enabled"
                  onChange={(event) => void updateDisplay({ enabled: event.target.checked })}
                />
              </div>
              <Select
                label="Screen orientation"
                disabled={!canControl || busy === "display"}
                onChange={(event) => void updateDisplay({ rotation: Number(event.target.value) })}
                value={display.rotation}
              >
                <option value={0}>Normal</option>
                  <option value={180}>Rotated 180°</option>
                </Select>
              <Select
                label="Sleep after"
                disabled={!canControl || busy === "display"}
                onChange={(event) => void updateDisplay({ sleep_timeout: Number(event.target.value) })}
                value={display.sleep_timeout}
              >
                <option value={0}>Never</option>
                  <option value={10}>10 seconds</option>
                  <option value={30}>30 seconds</option>
                  <option value={60}>1 minute</option>
                  <option value={300}>5 minutes</option>
                  <option value={600}>10 minutes</option>
                  <option value={1800}>30 minutes</option>
                  <option value={3600}>1 hour</option>
                </Select>
              <small className="display-meta">{display.hardware || "Front OLED"} · {display.bus || "I²C"} · showing {display.pages.length || "default"} page{display.pages.length === 1 ? "" : "s"} · {display.network_interface || "automatic network"}</small>
            </div>
          ) : <p className="system-card__empty">{inventoryLoading ? "Checking display controls…" : "OLED controls were not detected in this installation."}</p>}
        </article>
      </div>

      {logs && (
        <div className="log-drawer" role="dialog" aria-modal="true" aria-labelledby="service-log-title">
          <div className="log-drawer__backdrop" onClick={() => setLogs(null)} />
          <section>
            <div className="panel-heading">
              <div><span className="page-eyebrow">Read-only service log</span><h3 id="service-log-title">{logs.service.replaceAll("-", " ")}</h3></div>
              <Button variant="quiet" onClick={() => setLogs(null)}>Close</Button>
            </div>
            <pre>{logs.output || "No recent log entries."}</pre>
          </section>
        </div>
      )}
    </section>
  );
}
