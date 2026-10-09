import { useRef } from "react";
import { StatusPill } from "./StatusPill";
import { KvGrid, SectionCard, SystemDialog } from "./systemUi";
import { Button, Checkbox, LoadingLines, Notice, Select } from "./ui";
import type { StatusTone } from "./ui/status";

/*
 * System › Hardware and services: Vaelor's own services and the Pi appliance's
 * front display (VD-200, the System and SystemHardwarePi boards), and the
 * read-only service log (the DialogsSystem board).
 */

/**
 * One row of `SystemInventory.services()` in `vaelor/system_inventory.py`.
 * The state fields are systemd's own words from `systemctl show`, and all of
 * them are absent when the host has no `systemctl` to ask.
 */
export interface InventoryService {
  id: string;
  unit: string;
  available: boolean;
  active?: string;
  substate?: string;
  enabled?: string;
  restarts?: number;
  exit_status?: number;
}

/**
 * ACC-131. The pill used to be `active` or grey, so a Vaelor service that had
 * FAILED wore the same grey pill as one that is not installed. Every
 * `ActiveState` systemd can write gets its own plain word here, and anything
 * else - including the inventory's own `unknown` when the read failed - says
 * it could not be read rather than borrowing another state's look.
 */
const SERVICE_ACTIVE_STATES: Record<string, { label: string; tone: StatusTone }> = {
  active: { label: "Running", tone: "success" },
  activating: { label: "Starting", tone: "info" },
  reloading: { label: "Reloading", tone: "info" },
  refreshing: { label: "Refreshing", tone: "info" },
  deactivating: { label: "Stopping", tone: "warning" },
  inactive: { label: "Stopped", tone: "warning" },
  maintenance: { label: "Maintenance", tone: "warning" },
  failed: { label: "Failed", tone: "danger" },
};

export interface ServiceView {
  label: string;
  tone: StatusTone;
  detail: string;
  /** Running, or simply not installed: nothing here needs the owner. */
  settled: boolean;
}

export function serviceView(service: InventoryService): ServiceView {
  const restarts = typeof service.restarts === "number" ? `${service.restarts} restarts` : "restarts not reported";
  const known = service.active ? SERVICE_ACTIVE_STATES[service.active] : undefined;
  if (known && known.tone === "danger") {
    const exit = typeof service.exit_status === "number" && service.exit_status !== 0 ? `Exit status ${service.exit_status}` : "Stopped after an error";
    return { ...known, detail: `${exit} · ${restarts}`, settled: false };
  }
  if (!service.available && service.active === "inactive") {
    return { label: "Optional", tone: "neutral", detail: "Not installed", settled: true };
  }
  if (!service.available || !known) {
    return { label: "Unknown", tone: "neutral", detail: "Service state could not be read", settled: false };
  }
  return { ...known, detail: `${service.substate ?? "State"} · ${restarts}`, settled: known.tone === "success" };
}

/**
 * The name a reader knows a service by. The ids are the logical keys of
 * `VAELOR_LINUX_SERVICES` (vaelor/platform_drivers.py); one this table does not
 * know is shown as its id in words rather than hidden.
 */
const SERVICE_NAMES: Record<string, string> = {
  "control-plane": "Control plane",
  "workload-executor": "Workload executor",
  "workload-broker": "Workload broker",
  "credential-broker": "Credential broker",
  "vnc-gateway": "VNC gateway",
  "vnc-tls": "VNC TLS proxy",
};

export function serviceName(id: string): string {
  const known = SERVICE_NAMES[id];
  if (known) return known;
  const words = id.replaceAll("-", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** The header pill: how many are running, or what needs the owner. */
export function servicesSummary(services: InventoryService[]): { label: string; tone: StatusTone } {
  const views = services.map(serviceView);
  const failed = views.filter((view) => view.tone === "danger").length;
  if (failed) return { label: `${failed} failed`, tone: "danger" };
  const unsettled = views.filter((view) => !view.settled).length;
  if (unsettled) return { label: `${unsettled} need${unsettled === 1 ? "s" : ""} attention`, tone: "warning" };
  const running = views.filter((view) => view.tone === "success").length;
  return { label: running === views.length ? `All ${running} running` : `${running} running`, tone: "success" };
}

export function ServicesCard({
  busyService,
  canViewLogs,
  loading,
  logsRefusal,
  onViewLogs,
  services,
  stale = false,
}: {
  busyService: string;
  canViewLogs: boolean;
  loading: boolean;
  /** Why the last log could not be read, said in this card. */
  logsRefusal: string;
  onViewLogs: (id: string) => void;
  services: InventoryService[];
  /** The last read failed and these rows are the previous answer: nothing here is green (S-Y3). */
  stale?: boolean;
}) {
  const summary = servicesSummary(services);
  const reading = stale ? "stale" as const : undefined;
  return (
    <SectionCard
      actions={loading
        ? <StatusPill label="Checking" reading="unread" tone="neutral" />
        : services.length > 0 && <StatusPill label={stale ? `Old · ${summary.label}` : summary.label} reading={reading} tone={summary.tone} />}
      className="sys-services"
      description="Only control-plane-owned services are listed."
      flush
      footer={<span className="sys-muted">Viewers see the list; View logs needs operator access.</span>}
      title="Vaelor services"
    >
      {logsRefusal && <p className="sys-refusal" role="alert">{logsRefusal}</p>}
      <div className="sys-service-list">
        {services.map((service) => {
          const view = serviceView(service);
          return (
            <div className="sys-service" data-service-id={service.id} key={service.id}>
              <div className="sys-service__text">
                <strong>{serviceName(service.id)}</strong>
                <small>{view.detail}</small>
              </div>
              <StatusPill label={view.label} reading={reading} tone={view.tone} />
              <Button
                disabled={!canViewLogs || busyService === service.id}
                onClick={() => onViewLogs(service.id)}
                variant="quiet"
              >
                View logs
              </Button>
            </div>
          );
        })}
      </div>
      {loading && <LoadingLines label="Checking appliance services…" lines={3} />}
    </SectionCard>
  );
}

export function ServiceLogDialog({
  logs,
  onClose,
}: {
  logs: { service: string; output: string };
  onClose: () => void;
}) {
  const closeRef = useRef<HTMLButtonElement>(null);
  return (
    <SystemDialog
      eyebrow="Read-only service log · last 120 lines"
      onClose={onClose}
      size="wide"
      title={serviceName(logs.service)}
      titleId="service-log-title"
      initialFocusRef={closeRef}
      footer={<span className="sys-muted">Nothing here can change the service.</span>}
    >
      <pre className="sys-log">{logs.output || "No recent log entries."}</pre>
    </SystemDialog>
  );
}

export interface DisplayState {
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

export function FrontDisplayCard({
  busy,
  canControl,
  display,
  loading,
  onChange,
  outcome,
}: {
  busy: boolean;
  canControl: boolean;
  display: DisplayState | null;
  loading: boolean;
  onChange: (patch: Partial<DisplayState>) => void;
  /** The last change's result, in this card: applied, or refused. */
  outcome: { text: string; refused: boolean } | null;
}) {
  return (
    <SectionCard
      actions={display?.detected
        ? <StatusPill label={display.enabled ? "On" : "Off"} tone={display.enabled ? "success" : "neutral"} />
        : !loading && <StatusPill label="No display" tone="neutral" />}
      className="sys-oled"
      /*
       * Cooling and Case lighting stage a change and wait for Apply or Save;
       * this one writes as soon as you change it, and says so.
       */
      description="Choose when the front status screen is active. Changes here save as soon as you make them — there is nothing to apply."
      title="Front OLED display"
    >
      {display?.detected ? (
        <div className="sys-stack">
          <div className="sys-switch-row">
            <span><strong>Display power</strong><small>Show live system information</small></span>
            <Checkbox
              checked={display.enabled}
              className="sys-toggle"
              disabled={!canControl || busy}
              disabledReason={!canControl ? "Operator access is required to change display power." : undefined}
              id="display-power"
              label="Enabled"
              onChange={(event) => onChange({ enabled: event.target.checked })}
            />
          </div>
          <div className="sys-grid-fields">
            <Select
              disabled={!canControl || busy}
              label="Screen orientation"
              onChange={(event) => onChange({ rotation: Number(event.target.value) })}
              value={display.rotation}
            >
              <option value={0}>Normal</option>
              <option value={180}>Rotated 180°</option>
            </Select>
            <Select
              disabled={!canControl || busy}
              label="Sleep after"
              onChange={(event) => onChange({ sleep_timeout: Number(event.target.value) })}
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
          </div>
          <KvGrid
            items={[
              { label: "Hardware", value: `${display.hardware || "Front OLED"} · ${display.bus || "I²C"}`, mono: true },
              { label: "Showing", value: `${display.pages.length || "default"} page${display.pages.length === 1 ? "" : "s"}`, mono: true },
              { label: "Network shown", value: display.network_interface || "automatic network", mono: true },
            ]}
            label="Front display"
          />
          {outcome && <Notice severity={outcome.refused ? "danger" : "success"}>{outcome.text}</Notice>}
        </div>
      ) : loading
        ? <LoadingLines label="Checking display controls…" />
        : <p className="sys-muted">OLED controls were not detected in this installation.</p>}
    </SectionCard>
  );
}
