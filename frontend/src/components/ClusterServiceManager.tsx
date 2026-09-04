import { useState } from "react";
import { apiRequest } from "../lib/api";
import { formatQuantity } from "../lib/format";
import type { Session } from "../types";
import { ModalShell } from "./ModalShell";
import { StatusPill } from "./StatusPill";
import { Button, Input, Select } from "./ui";

export interface ClusterServiceSummary {
  id?: string;
  name?: string;
  image?: string;
  replicas?: string;
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
  onNotice: (message: string) => void;
}

interface ServiceSettings {
  replicas: string;
  memoryLimitMib: string;
  memoryReservationMib: string;
  updateParallelism: string;
  updateOrder: "start-first" | "stop-first";
}

const isManaged = (name = "") => name.startsWith("vaelor-");

const memoryMib = (bytes = 0, fallback: number) =>
  String(bytes > 0 ? Math.max(1, Math.round(bytes / 1024 ** 2)) : fallback);

export function ClusterServiceManager({
  service,
  session,
  onReview,
  onNotice,
}: Props) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [details, setDetails] = useState<ClusterServiceDetails | null>(null);
  const [logs, setLogs] = useState<ClusterServiceLogs | null>(null);
  const [diagnostic, setDiagnostic] = useState<ClusterDiagnostic | null>(null);
  const [backups, setBackups] = useState<ClusterBackup[]>([]);
  const [deleteConfirmation, setDeleteConfirmation] = useState("");
  const [settings, setSettings] = useState<ServiceSettings>({
    replicas: "1",
    memoryLimitMib: "512",
    memoryReservationMib: "256",
    updateParallelism: "1",
    updateOrder: "stop-first",
  });
  const name = String(service.name ?? "");

  const load = async () => {
    setLoading(true);
    setLogs(null);
    setDiagnostic(null);
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
          Math.min(256, Math.max(64, Math.round((limit || 512 * 1024 ** 2) / 2 / 1024 ** 2))),
        ),
        updateParallelism: String(
          Math.max(1, serviceDetails.update_policy?.parallelism ?? 1),
        ),
        updateOrder: serviceDetails.update_policy?.order === "start-first"
          ? "start-first"
          : "stop-first",
      });
      setOpen(true);
    } catch (error) {
      onNotice(error instanceof Error ? error.message : "Service details are unavailable.");
    } finally {
      setLoading(false);
    }
  };

  const runDiagnostic = async (tool: "stats" | "processes" | "health") => {
    setLoading(true);
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
      onNotice(error instanceof Error ? error.message : "The diagnostic is unavailable.");
    } finally {
      setLoading(false);
    }
  };

  const loadLogs = async () => {
    setLoading(true);
    try {
      setLogs(await apiRequest<ClusterServiceLogs>(
        `/cluster/services/${encodeURIComponent(name)}/logs?lines=200`,
      ));
    } catch (error) {
      onNotice(error instanceof Error ? error.message : "Service logs are unavailable.");
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
  };
  const settingsValid = (
    Number.isInteger(numericSettings.replicas)
    && numericSettings.replicas >= 1
    && numericSettings.replicas <= 32
    && Number.isInteger(numericSettings.memoryLimitMib)
    && numericSettings.memoryLimitMib >= 128
    && numericSettings.memoryLimitMib <= 131072
    && Number.isInteger(numericSettings.memoryReservationMib)
    && numericSettings.memoryReservationMib >= 64
    && numericSettings.memoryReservationMib <= numericSettings.memoryLimitMib
    && Number.isInteger(numericSettings.updateParallelism)
    && numericSettings.updateParallelism >= 1
    && numericSettings.updateParallelism <= Math.min(numericSettings.replicas, 8)
  );

  const deleteBackup = async (backupId: string) => {
    setLoading(true);
    try {
      await apiRequest(`/cluster/backups/${encodeURIComponent(backupId)}`, {
        method: "DELETE",
        body: JSON.stringify({ confirmation: deleteConfirmation }),
      }, session.csrf_token);
      setBackups((current) => current.filter((item) => item.id !== backupId));
      setDeleteConfirmation("");
      onNotice("Cluster backup deleted.");
    } catch (error) {
      onNotice(error instanceof Error ? error.message : "The backup could not be deleted.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <article className="fleet-service">
        <div className="fleet-service__identity">
          <strong>{name}</strong>
          <span>{service.image}</span>
        </div>
        <StatusPill
          label={service.replicas || "Starting"}
          status={String(service.replicas ?? "").match(/^(\d+)\/\1$/) ? "healthy" : "degraded"}
        />
        {isManaged(name) && (
          <Button disabled={loading} onClick={() => void load()}>
            {loading ? "Opening…" : "Manage"}
          </Button>
        )}
      </article>

      {open && details && (
        <ModalShell labelledBy="cluster-service-title" onClose={() => !loading && setOpen(false)}>
          <section className="cluster-service-manager">
            <header>
              <div>
                <span className="eyebrow">CLUSTER SERVICE</span>
                <h2 id="cluster-service-title">{details.name}</h2>
                <p>{details.image}</p>
              </div>
              <StatusPill
                label={`${service.replicas || "Starting"} replicas`}
                status={String(service.replicas ?? "").match(/^(\d+)\/\1$/) ? "healthy" : "degraded"}
              />
            </header>

            <dl className="cluster-service-manager__facts">
              <div><dt>Desired replicas</dt><dd>{details.desired_replicas}</dd></div>
              <div><dt>Published ports</dt><dd>{details.ports.length || "Private only"}</dd></div>
              <div><dt>Persistent mounts</dt><dd>{details.mounts.length}</dd></div>
              <div><dt>Updated</dt><dd>{details.updated_at || "Not reported"}</dd></div>
            </dl>

            {session.user.role === "administrator" && (
              <section>
                <div className="cluster-service-manager__section-heading">
                  <div>
                    <h3>Service settings</h3>
                    <p>
                      Bounded capacity and rolling-update controls. Raw
                      environment variables remain protected.
                    </p>
                  </div>
                </div>
                <div className="cluster-service-manager__settings">
                  <Input hint="1-32 running copies" id="cluster-replicas" inputMode="numeric" label="Replicas" max={32} min={1} onChange={(event) => setSettings((current) => ({ ...current, replicas: event.target.value }))} type="number" value={settings.replicas} />
                  <Input hint="Hard container limit" id="cluster-memory-limit" inputMode="numeric" label="Memory limit per replica (MiB)" max={131072} min={128} onChange={(event) => setSettings((current) => ({ ...current, memoryLimitMib: event.target.value }))} type="number" value={settings.memoryLimitMib} />
                  <Input hint="Scheduling commitment" id="cluster-memory-reservation" inputMode="numeric" label="Memory reservation (MiB)" max={numericSettings.memoryLimitMib || 128} min={64} onChange={(event) => setSettings((current) => ({ ...current, memoryReservationMib: event.target.value }))} type="number" value={settings.memoryReservationMib} />
                  <Input hint="Maximum 8" id="cluster-update-parallelism" inputMode="numeric" label="Tasks updated together" max={Math.min(numericSettings.replicas || 1, 8)} min={1} onChange={(event) => setSettings((current) => ({ ...current, updateParallelism: event.target.value }))} type="number" value={settings.updateParallelism} />
                  <Select hint="Start-first needs temporary spare capacity" id="cluster-update-order" label="Replacement order" onChange={(event) => setSettings((current) => ({ ...current, updateOrder: event.target.value as ServiceSettings["updateOrder"] }))} value={settings.updateOrder}><option value="stop-first">Stop old task first</option><option value="start-first">Start replacement first</option></Select>
                </div>
                {!settingsValid && (
                  <p className="form-error" role="alert">
                    Check replica, memory, and rolling-update limits.
                  </p>
                )}
                <Button

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
                    });
                  }}
                >
                  Review settings
                </Button>
              </section>
            )}

            {!!details.ports.length && (
              <section>
                <h3>Network</h3>
                <div className="cluster-service-manager__rows">
                  {details.ports.map((port) => (
                    <div key={`${port.published}-${port.target}`}>
                      <strong>{port.published} → {port.target}</strong>
                      <span>{port.protocol} · {port.mode}</span>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {!!details.mounts.length && (
              <section>
                <h3>Storage</h3>
                <div className="cluster-service-manager__rows">
                  {details.mounts.map((mount) => (
                    <div key={`${mount.source}-${mount.target}`}>
                      <strong>{mount.source}</strong>
                      <span>{mount.target} · {mount.type}{mount.read_only ? " · read only" : ""}</span>
                    </div>
                  ))}
                </div>
              </section>
            )}

            <section>
              <div className="cluster-service-manager__section-heading">
                <div>
                  <h3>Tasks and diagnostics console</h3>
                  <p>App-scoped checks only—this does not expose a worker shell.</p>
                </div>
                <div className="cluster-service-manager__diagnostic-actions">
                  <Button variant="quiet" disabled={loading} onClick={() => void runDiagnostic("stats")}>Resource use</Button>
                  <Button variant="quiet" disabled={loading} onClick={() => void runDiagnostic("processes")}>Processes</Button>
                  <Button variant="quiet" disabled={loading} onClick={() => void runDiagnostic("health")}>Health</Button>
                  <Button variant="quiet" disabled={loading} onClick={() => void loadLogs()}>Load logs</Button>
                </div>
              </div>
              <div className="cluster-service-manager__tasks">
                {details.tasks.map((task) => (
                  <div key={task.id}>
                    <span>{task.node || "Pending placement"}</span>
                    <strong>{task.current || task.desired}</strong>
                    {task.error && <small>{task.error}</small>}
                  </div>
                ))}
              </div>
              {logs && (
                <pre className="cluster-service-manager__logs">
                  {logs.output || `No output in the last ${logs.lines} lines.`}
                </pre>
              )}
              {diagnostic && (
                <div className="cluster-service-manager__diagnostic">
                  <span>{diagnostic.tool} · {diagnostic.node_name}</span>
                  <pre className="cluster-service-manager__logs">
                    {diagnostic.data
                      ? JSON.stringify(diagnostic.data, null, 2)
                      : diagnostic.output}
                  </pre>
                </div>
              )}
            </section>

            <section>
              <div className="cluster-service-manager__section-heading">
                <div>
                  <h3>Recovery backups</h3>
                  <p>Verified copies of this worker-local named volume.</p>
                </div>
                {session.user.role === "administrator" && (
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
              {backups.length ? (
                <>
                  <div className="cluster-service-manager__backups">
                    {backups.map((backup) => (
                      <article key={backup.id}>
                        <div>
                          <strong>{new Date(backup.created_at * 1000).toLocaleString()}</strong>
                          <span>
                            {formatQuantity(backup.size_bytes, "checkpoint")} · {backup.reason}
                          </span>
                          <code>{backup.sha256.slice(0, 16)}…</code>
                        </div>
                        {session.user.role === "administrator" && (
                          <div>
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
                            <Button variant="danger"

                              disabled={loading || deleteConfirmation !== name}
                              onClick={() => void deleteBackup(backup.id)}
                            >
                              Delete
                            </Button>
                          </div>
                        )}
                      </article>
                    ))}
                  </div>
                  {session.user.role === "administrator" && (
                    <Input className="cluster-service-manager__delete-confirm" id="cluster-delete-confirmation" label={`Type ${name} to enable backup deletion`} onChange={(event) => setDeleteConfirmation(event.target.value)} value={deleteConfirmation} />
                  )}
                </>
              ) : (
                <p className="cluster-service-manager__empty">
                  No recovery backup has been created for this service.
                </p>
              )}
            </section>

            {session.user.role === "administrator" && (
              <footer>
                <Button onClick={() => void review("restart")}>Review restart</Button>
                <Button onClick={() => void review("refresh")}>Review image refresh</Button>
                <Button onClick={() => void review("rollback")}>Review rollback</Button>
                <Button variant="danger"

                  onClick={async () => {
                    setOpen(false);
                    await onReview("remove-service", { service_name: name });
                  }}
                >
                  Review removal
                </Button>
              </footer>
            )}
          </section>
        </ModalShell>
      )}
    </>
  );
}
