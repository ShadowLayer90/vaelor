import { useState, type ChangeEvent } from "react";
import { useModalAction } from "../hooks/useModalAction";
import { apiRequest } from "../lib/api";
import { bytesIn, formatQuantity, typedFromBytes } from "../lib/format";
import type { Session } from "../types";
import { CodeBlock, KeyValues } from "./ClusterPrimitives";
import { ClusterDialog } from "./ClusterDialog";
import { StatusPill } from "./StatusPill";
import { Button, Input, Select, TabSet } from "./ui";
import "../styles/cluster-deployments.css";

export interface ClusterServiceSummary {
  id?: string;
  name?: string;
  image?: string;
  replicas?: string;
  /**
   * The honest backend state the summary derives from Swarm's real update +
   * placement facts (D3), one of healthy/deploying/updating/rolled-back/failed/
   * unavailable, with a one-line `reason`. The Deployments row renders these
   * through the shared status tones instead of a replica-string regex. Absent on
   * a row the backend could not enrich (a bulk-inspect miss), which the row
   * shows as an honest unknown rather than a guess.
   */
  state?: string;
  reason?: string;
}

interface ClusterServiceDetails {
  name: string;
  image: string;
  desired_replicas: number;
  updated_at: string;
  resources: {
    limits: { MemoryBytes?: number };
    reservations: { MemoryBytes?: number };
  };
  update_policy: {
    parallelism: number;
    failure_action: string;
    order: string;
  };
  labels: Record<string, string>;
  constraints: string[];
  mounts: Array<{
    type: string;
    source: string;
    target: string;
    read_only: boolean;
  }>;
  ports: Array<{
    published: number;
    target: number;
    protocol: string;
    mode: string;
  }>;
  tasks: Array<{
    id: string;
    name: string;
    node: string;
    desired: string;
    current: string;
    error: string;
  }>;
}

interface ClusterServiceLogs {
  output: string;
  lines: number;
}

interface ClusterDiagnostic {
  node_name: string;
  tool: string;
  data: Record<string, unknown> | null;
  output: string;
}

interface ClusterBackup {
  id: string;
  service_name: string;
  volume: string;
  size_bytes: number;
  sha256: string;
  reason: string;
  created_at: number;
}

interface Props {
  service: ClusterServiceSummary;
  session: Session;
  onReview: (
    action: string,
    payload: Record<string, unknown>,
  ) => Promise<void>;
  onNotice: (message: string, refused?: boolean) => void;
}

type LabelOp = "==" | "!=";
interface LabelConstraint {
  key: string;
  op: LabelOp;
  value: string;
}

interface ServiceSettings {
  replicas: string;
  memoryLimitMib: string;
  memoryReservationMib: string;
  updateParallelism: string;
  updateOrder: "start-first" | "stop-first";
  cpuLimit: string;
  labels: LabelConstraint[];
}

const isManaged = (name = "") => name.startsWith("vaelor-");

// Mirrors the backend's `cluster_service_reconfigure` label validation so a bad
// constraint is caught before review.
const LABEL_TOKEN = /^[A-Za-z0-9._-]{1,63}$/;
// vaelor.*/pironman.* are managed at join; a `node.`/`node.labels.` key would
// double-namespace (Vaelor adds the prefix). Enter the bare label key.
const RESERVED_LABEL = /^(vaelor|pironman|node)\./i;
const labelRowComplete = (row: LabelConstraint) => Boolean(row.key || row.value);
const labelRowValid = (row: LabelConstraint) =>
  LABEL_TOKEN.test(row.key.trim()) &&
  LABEL_TOKEN.test(row.value.trim()) &&
  (row.op === "==" || row.op === "!=") &&
  !RESERVED_LABEL.test(row.key.trim());

const memoryMib = (bytes = 0, fallback: number) =>
  String(bytes > 0 ? Math.max(1, typedFromBytes(bytes, "MiB")) : fallback);

export function ClusterServiceManager({
  service,
  session,
  onReview,
  onNotice,
}: Props) {
  const [open, setOpen] = useState(false);
  // VD-200 (owner decision 13): the manager's four sections are tabs; an
  // operator, who cannot change settings, opens on Network and storage.
  const [tab, setTab] = useState(session.user.role === "administrator" ? "settings" : "network");
  const [loading, setLoading] = useState(false);
  const [details, setDetails] = useState<ClusterServiceDetails | null>(null);
  const [logs, setLogs] = useState<ClusterServiceLogs | null>(null);
  const [diagnostic, setDiagnostic] = useState<ClusterDiagnostic | null>(null);
  // W4d-D18: a refusal from inside the dialog is shown inside the dialog
  // (ModalShell's `error`). The page notice renders under the modal, where the
  // owner cannot see it.
  const action = useModalAction();
  const [backups, setBackups] = useState<ClusterBackup[]>([]);
  const [deleteConfirmation, setDeleteConfirmation] = useState("");
  const [settings, setSettings] = useState<ServiceSettings>({
    replicas: "1",
    memoryLimitMib: "512",
    memoryReservationMib: "256",
    updateParallelism: "1",
    updateOrder: "stop-first",
    cpuLimit: "2.0",
    labels: [],
  });
  const name = String(service.name ?? "");

  const load = async () => {
    setLoading(true);
    setLogs(null);
    setDiagnostic(null);
    action.clear();
    try {
      const [serviceDetails, serviceBackups] = await Promise.all([
        apiRequest<ClusterServiceDetails>(
          `/cluster/services/${encodeURIComponent(name)}`,
        ),
        apiRequest<ClusterBackup[]>(
          `/cluster/backups?service_name=${encodeURIComponent(name)}`,
        ),
      ]);
      setDetails(serviceDetails);
      setBackups(serviceBackups);
      const limit = serviceDetails.resources?.limits?.MemoryBytes ?? 0;
      const reservation = (
        serviceDetails.resources?.reservations?.MemoryBytes ?? 0
      );
      setSettings({
        replicas: String(serviceDetails.desired_replicas),
        memoryLimitMib: memoryMib(limit, 512),
        memoryReservationMib: memoryMib(
          reservation,
          Math.min(256, Math.max(64, typedFromBytes((limit || bytesIn(512, "MiB")) / 2, "MiB"))),
        ),
        updateParallelism: String(
          Math.max(1, serviceDetails.update_policy?.parallelism ?? 1),
        ),
        updateOrder: serviceDetails.update_policy?.order === "start-first"
          ? "start-first"
          : "stop-first",
        cpuLimit: "2.0",
        labels: [],
      });
      setOpen(true);
    } catch (error) {
      onNotice(error instanceof Error ? error.message : "Service details are unavailable.", true);
    } finally {
      setLoading(false);
    }
  };

  const runDiagnostic = async (tool: "stats" | "processes" | "health") => {
    setLoading(true);
    setLogs(null);
    setDiagnostic(null);
    action.clear();
    try {
      setDiagnostic(await apiRequest<ClusterDiagnostic>(
        `/cluster/services/${encodeURIComponent(name)}/diagnostics`,
        {
          method: "POST",
          body: JSON.stringify({ tool }),
        },
        session.csrf_token,
      ));
    } catch (error) {
      action.setError(error instanceof Error ? error.message : "The diagnostic is unavailable.");
    } finally {
      setLoading(false);
    }
  };

  const loadLogs = async () => {
    setLoading(true);
    setLogs(null);
    setDiagnostic(null);
    action.clear();
    try {
      setLogs(await apiRequest<ClusterServiceLogs>(
        `/cluster/services/${encodeURIComponent(name)}/logs?lines=200`,
      ));
    } catch (error) {
      action.setError(error instanceof Error ? error.message : "Service logs are unavailable.");
    } finally {
      setLoading(false);
    }
  };

  const review = async (action: "restart" | "refresh" | "rollback") => {
    setOpen(false);
    await onReview(`${action}-service`, { service_name: name });
  };

  const numericSettings = {
    replicas: Number(settings.replicas),
    memoryLimitMib: Number(settings.memoryLimitMib),
    memoryReservationMib: Number(settings.memoryReservationMib),
    updateParallelism: Number(settings.updateParallelism),
    cpuLimit: Number(settings.cpuLimit),
  };
  const filledLabels = settings.labels.filter(labelRowComplete);
  const labelsValid = filledLabels.every(labelRowValid);
  const settingsValid = (
    Number.isInteger(numericSettings.replicas)
    && numericSettings.replicas >= 1
    && numericSettings.replicas <= 32
    && Number.isInteger(numericSettings.memoryLimitMib)
    // 16 MiB floor, identical to the backend plan/operations/driver (D2
    // follow-up a): a 64 MiB app (it-tools) must be reconfigurable, so the
    // client cannot invalidate the form before the reconcile runs.
    && numericSettings.memoryLimitMib >= 16
    && numericSettings.memoryLimitMib <= 131072
    && Number.isInteger(numericSettings.memoryReservationMib)
    && numericSettings.memoryReservationMib >= 16
    && numericSettings.memoryReservationMib <= numericSettings.memoryLimitMib
    && Number.isInteger(numericSettings.updateParallelism)
    && numericSettings.updateParallelism >= 1
    && numericSettings.updateParallelism <= Math.min(numericSettings.replicas, 8)
    && numericSettings.cpuLimit >= 0.25
    && numericSettings.cpuLimit <= 64
    && labelsValid
  );

  // The field's value is read when the event fires, never inside the updater:
  // React may run an updater later, after it has written the committed value
  // back into the input (a dialog's opening focus does that), and an updater
  // reading the input then keeps the old value and drops what was typed.
  const setField = <K extends Exclude<keyof ServiceSettings, "labels">>(field: K) =>
    (event: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
      const value = event.target.value as ServiceSettings[K];
      setSettings((current) => ({ ...current, [field]: value }));
    };

  const updateLabel = (index: number, patch: Partial<LabelConstraint>) =>
    setSettings((current) => ({
      ...current,
      labels: current.labels.map((row, position) =>
        position === index ? { ...row, ...patch } : row,
      ),
    }));

  const deleteBackup = async (backupId: string) => {
    setLoading(true);
    action.clear();
    try {
      await apiRequest(`/cluster/backups/${encodeURIComponent(backupId)}`, {
        method: "DELETE",
        body: JSON.stringify({ confirmation: deleteConfirmation }),
      }, session.csrf_token);
      setBackups((current) => current.filter((item) => item.id !== backupId));
      setDeleteConfirmation("");
      onNotice("Cluster backup deleted.");
    } catch (error) {
      action.setError(error instanceof Error ? error.message : "The backup could not be deleted.");
    } finally {
      setLoading(false);
    }
  };

  const isAdministrator = session.user.role === "administrator";
  const replicasHealthy = /^(\d+)\/\1$/.test(String(service.replicas ?? ""));
  const tabs = [
    ...(isAdministrator ? [{ id: "settings", label: "Settings" }] : []),
    { id: "network", label: "Network and storage" },
    { id: "diagnostics", label: "Diagnostics" },
    { id: "backups", label: "Backups" },
  ];
  const shownTool = logs ? "logs" : diagnostic?.tool ?? "";
  const toolButton = (tool: "stats" | "processes" | "health" | "logs", label: string) => (
    <Button
      aria-pressed={shownTool === tool}
      className={shownTool === tool ? "cl-chip is-on" : "cl-chip"}
      disabled={loading}
      key={tool}
      onClick={() => void (tool === "logs" ? loadLogs() : runDiagnostic(tool))}
    >
      {label}
    </Button>
  );

  return (
    <>
      {isManaged(name) && (
        <Button disabled={loading} onClick={() => void load()}>
          {loading && !open ? "Opening…" : "Manage"}
        </Button>
      )}

      {open && details && (
        <ClusterDialog
          aside={(
            <StatusPill
              label={`${service.replicas || "Starting"} replicas`}
              status={replicasHealthy ? "healthy" : "degraded"}
            />
          )}
          error={action.error}
          eyebrow="Cluster service"
          footer={isAdministrator ? (
            <>
              <Button onClick={() => void review("restart")}>Review restart</Button>
              <Button onClick={() => void review("refresh")}>Review image refresh</Button>
              <Button onClick={() => void review("rollback")}>Review rollback</Button>
              <Button
                className="cl-danger-outline"
                onClick={async () => {
                  setOpen(false);
                  await onReview("remove-service", { service_name: name });
                }}
              >
                Review removal
              </Button>
            </>
          ) : undefined}
          onClose={() => !loading && setOpen(false)}
          subtitle={details.image}
          title={details.name}
          titleId="cluster-service-title"
          size="wide"
        >
          <KeyValues
            items={[
              { label: "Desired replicas", value: details.desired_replicas, mono: false },
              { label: "Published ports", value: details.ports.length || "Private only", mono: false },
              { label: "Persistent mounts", value: details.mounts.length, mono: false },
              { label: "Updated", value: details.updated_at || "Not reported", mono: false },
            ]}
            label="Service facts"
          />

          <TabSet
            className="cl-svc-tabs"
            items={tabs}
            label="Service sections"
            onSelect={setTab}
            selectedId={tab}
          >
            {tab === "settings" && isAdministrator && (
              <>
                <p className="cl-meta">
                  Bounded capacity and rolling-update controls. Raw environment variables remain protected.
                </p>
                <div className="cl-svc-fields">
                  <Input hint="1-32 running copies" id="cluster-replicas" inputMode="numeric" label="Replicas" max={32} min={1} onChange={setField("replicas")} type="number" value={settings.replicas} />
                  <Input hint="Hard container limit" id="cluster-memory-limit" inputMode="numeric" label="Memory limit per replica (MiB)" max={131072} min={16} onChange={setField("memoryLimitMib")} type="number" value={settings.memoryLimitMib} />
                  <Input hint="Scheduling commitment" id="cluster-memory-reservation" inputMode="numeric" label="Memory reservation (MiB)" max={numericSettings.memoryLimitMib || 16} min={16} onChange={setField("memoryReservationMib")} type="number" value={settings.memoryReservationMib} />
                  <Input hint="Maximum 8" id="cluster-update-parallelism" inputMode="numeric" label="Tasks updated together" max={Math.min(numericSettings.replicas || 1, 8)} min={1} onChange={setField("updateParallelism")} type="number" value={settings.updateParallelism} />
                  <Select hint="Start-first needs temporary spare capacity" id="cluster-update-order" label="Replacement order" onChange={setField("updateOrder")} value={settings.updateOrder}><option value="stop-first">Stop old task first</option><option value="start-first">Start replacement first</option></Select>
                  <Input hint="0.25–64 cores, reserves a quarter core" id="cluster-cpu-limit" inputMode="decimal" label="CPU limit per replica (cores)" onChange={setField("cpuLimit")} value={settings.cpuLimit} />
                </div>
                <fieldset className="cl-stack cl-svc-labels">
                  <legend className="sr-only">Node label constraints</legend>
                  <p className="cl-meta">
                    <span className="cl-strong">Node label constraints</span>
                    {" · against existing node labels. Vaelor-managed labels cannot be used, and the service keeps its placement pin."}
                  </p>
                  {settings.labels.map((row, index) => (
                    <div key={index} className="cl-svc-label-row">
                      <Input id={`cluster-label-key-${index}`} label={`Label key ${index + 1}`} onChange={(event) => updateLabel(index, { key: event.target.value })} value={row.key} />
                      <Select id={`cluster-label-op-${index}`} label={`Match ${index + 1}`} onChange={(event) => updateLabel(index, { op: event.target.value as LabelOp })} value={row.op}><option value="==">is</option><option value="!=">is not</option></Select>
                      <Input id={`cluster-label-value-${index}`} label={`Label value ${index + 1}`} onChange={(event) => updateLabel(index, { value: event.target.value })} value={row.value} />
                      <Button variant="quiet" onClick={() => setSettings((current) => ({ ...current, labels: current.labels.filter((_, position) => position !== index) }))}>Remove</Button>
                    </div>
                  ))}
                </fieldset>
                {!settingsValid && (
                  <p className="form-error cl-bad-text" role="alert">
                    Check replica, memory, CPU, label, and rolling-update limits.
                  </p>
                )}
                <div className="cl-actions">
                  <Button variant="quiet" onClick={() => setSettings((current) => ({ ...current, labels: [...current.labels, { key: "", op: "==", value: "" }] }))}>Add label constraint</Button>
                  <span className="cl-spacer" />
                  <Button
                    variant="primary"
                    disabled={!settingsValid}
                    onClick={async () => {
                      setOpen(false);
                      await onReview("configure-service", {
                        service_name: name,
                        replicas: numericSettings.replicas,
                        memory_limit_mib: numericSettings.memoryLimitMib,
                        memory_reservation_mib: numericSettings.memoryReservationMib,
                        update_parallelism: numericSettings.updateParallelism,
                        update_order: settings.updateOrder,
                        cpu_limit: numericSettings.cpuLimit,
                        label_constraints: filledLabels.map((row) => ({
                          key: row.key.trim(),
                          op: row.op,
                          value: row.value.trim(),
                        })),
                      });
                    }}
                  >
                    Review settings
                  </Button>
                </div>
              </>
            )}

            {tab === "network" && (
              details.ports.length || details.mounts.length ? (
                <ul aria-label="Network and storage" className="cl-svc-list">
                  {details.ports.map((port) => (
                    <li key={`port-${port.published}-${port.target}`}>
                      <strong>{port.published} → {port.target}</strong>
                      <span>{port.protocol} · {port.mode}</span>
                    </li>
                  ))}
                  {details.mounts.map((mount) => (
                    <li key={`mount-${mount.source}-${mount.target}`}>
                      <strong className="cl-svc-mono">{mount.source}</strong>
                      <span>{mount.target} · {mount.type}{mount.read_only ? " · read only" : ""}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="cl-meta">This service publishes no ports and has no persistent mounts.</p>
              )
            )}

            {tab === "diagnostics" && (
              <>
                <p className="cl-meta">App-scoped checks only — this does not expose a worker shell.</p>
                <div aria-label="Diagnostics" className="cl-chips" role="group">
                  {toolButton("stats", "Resource use")}
                  {toolButton("processes", "Processes")}
                  {toolButton("health", "Health")}
                  {toolButton("logs", "Load logs")}
                </div>
                {details.tasks.length ? (
                  <ul aria-label="Tasks" className="cl-svc-list">
                    {details.tasks.map((task) => (
                      <li key={task.id}>
                        <span>{task.node || "Pending placement"}</span>
                        <div>
                          <strong>{task.current || task.desired}</strong>
                          {task.error && <span>{` ${task.error}`}</span>}
                        </div>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="cl-meta">Swarm reports no tasks for this service.</p>
                )}
                {logs && (
                  <CodeBlock className="cl-svc-output">
                    {logs.output || `No output in the last ${logs.lines} lines.`}
                  </CodeBlock>
                )}
                {diagnostic && (
                  <CodeBlock className="cl-svc-output">
                    {`${diagnostic.tool} · ${diagnostic.node_name}\n`}
                    {diagnostic.data
                      ? JSON.stringify(diagnostic.data, null, 2)
                      : diagnostic.output}
                  </CodeBlock>
                )}
              </>
            )}

            {tab === "backups" && (
              <>
                <div className="cl-svc-section-head">
                  <h3>Recovery backups</h3>
                  {isAdministrator && (
                    <Button
                      onClick={async () => {
                        setOpen(false);
                        await onReview("backup-service", {
                          service_name: name,
                        });
                      }}
                    >
                      Review new backup
                    </Button>
                  )}
                </div>
                <p className="cl-meta">Verified copies of this worker-local named volume.</p>
                {backups.length ? (
                  <>
                    <ul aria-label="Recovery backups" className="cl-svc-backups">
                      {backups.map((backup) => (
                        <li key={backup.id}>
                          <div className="cl-rows__text">
                            <strong>{new Date(backup.created_at * 1000).toLocaleString()}</strong>
                            <span>
                              {formatQuantity(backup.size_bytes, "checkpoint")} · {backup.reason} · <code>{backup.sha256.slice(0, 16)}…</code>
                            </span>
                          </div>
                          {isAdministrator && (
                            <div className="cl-actions">
                              <Button
                                onClick={async () => {
                                  setOpen(false);
                                  await onReview("restore-service", {
                                    service_name: name,
                                    backup_id: backup.id,
                                  });
                                }}
                              >
                                Review restore
                              </Button>
                              <Button
                                className="cl-danger-outline"
                                disabled={loading || deleteConfirmation !== name}
                                onClick={() => void deleteBackup(backup.id)}
                              >
                                Delete
                              </Button>
                            </div>
                          )}
                        </li>
                      ))}
                    </ul>
                    {isAdministrator && (
                      <Input id="cluster-delete-confirmation" label={`Type ${name} to enable backup deletion`} onChange={(event) => setDeleteConfirmation(event.target.value)} value={deleteConfirmation} />
                    )}
                  </>
                ) : (
                  <p className="cl-meta">No recovery backup has been created for this service.</p>
                )}
              </>
            )}
          </TabSet>
        </ClusterDialog>
      )}
    </>
  );
}
