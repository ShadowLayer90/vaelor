import { useEffect, useMemo, useState } from "react";
import { Button, Input, Notice, Select } from "./ui";
import { ClusterDialog } from "./ClusterDialog";
import "../styles/cluster-dialogs.css";
import { apiRequest } from "../lib/api";
import { formatMemory } from "./fleetTypes";
import type { AppTemplate, FleetNode } from "./fleetTypes";

/**
 * The cluster app-deploy modal: pick a catalog app, choose where its replicas
 * land (run-once on the best-fit machine, spread one per machine, or pinned to
 * one), and review a resource-aware placement plan.
 *
 * Placement intent is the core of D1. Run-once needs no machine pick — the
 * controller finds the best fit. Spread and pin take a replica count; pin takes
 * a machine. A stateful app (one that keeps data on its machine) cannot spread,
 * so that option is disabled with the reason. The app's environment and any
 * password are set server-side and never collected here. Every validation gate
 * precedes the review button.
 */

type Intent = "run-once" | "spread" | "pin";
type LabelOp = "==" | "!=";
interface LabelConstraint {
  key: string;
  op: LabelOp;
  value: string;
}

interface Props {
  joinedWorkers: FleetNode[];
  busy: boolean;
  /** Why the plan review was refused (VD-189): shown in this dialog, never on the inert page beneath. */
  error?: string;
  onClose: () => void;
  onReview: (nodeId: string | undefined, payload: Record<string, unknown>) => void;
}

// Mirrors the backend's `cluster_service_reconfigure` validation so a bad value
// is caught before review, not after the plan refuses it.
const LABEL_TOKEN = /^[A-Za-z0-9._-]{1,63}$/;
// vaelor.*/pironman.* are managed at join; a `node.`/`node.labels.` key would
// double-namespace (Vaelor adds the prefix). Enter the bare label key.
const RESERVED_LABEL = /^(vaelor|pironman|node)\./i;

function labelRowComplete(row: LabelConstraint): boolean {
  return Boolean(row.key || row.value);
}

function labelRowValid(row: LabelConstraint): boolean {
  return (
    LABEL_TOKEN.test(row.key.trim()) &&
    LABEL_TOKEN.test(row.value.trim()) &&
    (row.op === "==" || row.op === "!=") &&
    !RESERVED_LABEL.test(row.key.trim())
  );
}

export function DeployAppModal({ joinedWorkers, busy, error, onClose, onReview }: Props) {
  const [templates, setTemplates] = useState<AppTemplate[]>([]);
  const [form, setForm] = useState({
    nodeId: "",
    templateId: "",
    name: "",
    port: "",
    intent: "run-once" as Intent,
    // Empty until the owner types a count: the starting count depends on the
    // app and the placement (see `replicasText`).
    replicas: "",
  });
  const [cpuLimit, setCpuLimit] = useState("2.0");
  const [labels, setLabels] = useState<LabelConstraint[]>([]);

  useEffect(() => {
    void apiRequest<AppTemplate[]>("/apps/catalog")
      .then(setTemplates)
      .catch(() => setTemplates([]));
  }, []);

  const template = useMemo(
    () => templates.find((item) => item.id === form.templateId),
    [templates, form.templateId],
  );
  const stateful = Boolean(template?.stateful);
  // W4d-D19: a pin puts every copy on one machine, sharing the app's one data
  // volume, so a data-keeping app starts at one replica; spread starts at two.
  const replicasText = form.replicas || (form.intent === "pin" && stateful ? "1" : "2");
  const replicas = Number(replicasText);
  const sharedVolumeWarning = form.intent === "pin" && stateful && replicas > 1;

  // Spread over per-task local volumes would split a stateful app's data, so
  // that intent is refused for it (the backend reconcile refuses it too).
  const spreadBlocked = form.intent === "spread" && stateful;

  const filledLabels = labels.filter(labelRowComplete);
  const cpuValue = Number(cpuLimit);
  const cpuValid = cpuValue >= 0.25 && cpuValue <= 64;
  const labelsValid = filledLabels.every(labelRowValid);

  const canReview = Boolean(
    form.templateId &&
      form.name &&
      Number(form.port) >= 1024 &&
      !spreadBlocked &&
      cpuValid &&
      labelsValid &&
      (form.intent !== "pin" || form.nodeId) &&
      (form.intent !== "spread" || (replicas >= 2 && replicas <= 32)) &&
      (form.intent !== "pin" || (replicas >= 1 && replicas <= 32)),
  );

  const updateLabel = (index: number, patch: Partial<LabelConstraint>) =>
    setLabels((current) =>
      current.map((row, position) => (position === index ? { ...row, ...patch } : row)),
    );

  const readBack = (() => {
    if (form.intent === "run-once") {
      return "One replica on the machine with the most free memory.";
    }
    if (form.intent === "spread") {
      if (stateful) return "This app keeps data on its machine, so it cannot be spread.";
      return `${replicas || 0} replicas, one on each of ${replicas || 0} machines.`;
    }
    const picked = joinedWorkers.find((node) => node.id === form.nodeId);
    return picked
      ? `${replicas || 0} replica${replicas === 1 ? "" : "s"} on ${picked.name}.`
      : "Pick a machine to pin this app to.";
  })();

  // Why Review placement is off: the first unmet gate, said beside the button.
  const reviewReason = busy || canReview
    ? undefined
    : !form.templateId
      ? "Choose a catalog app."
      : !form.name
        ? "Name the deployment."
        : !(Number(form.port) >= 1024)
          ? "Choose a published port of 1024 or above."
          : spreadBlocked
            ? "This app keeps data on its machine, so it cannot be spread."
            : !cpuValid
              ? "Choose a CPU limit from 0.25 to 64 cores."
              : !labelsValid
                ? "Fix the label constraints under Advanced."
                : form.intent === "pin" && !form.nodeId
                  ? "Pick a machine to pin this app to."
                  : form.intent === "spread"
                    ? "Spread takes 2 to 32 replicas."
                    : "Choose 1 to 32 replicas.";

  return (
    <ClusterDialog
      className="cd-dialog"
      error={error}
      eyebrow="Resource-aware placement"
      footer={(
        <>
          <Button variant="secondary" onClick={onClose}>Cancel</Button>
          <Button
            variant="primary"
            disabled={busy || !canReview}
            disabledReason={reviewReason}
            onClick={() =>
              onReview(form.intent === "pin" ? form.nodeId : undefined, {
                template_id: form.templateId,
                name: form.name,
                port: Number(form.port),
                intent: form.intent,
                ...(form.intent === "run-once" ? {} : { replicas }),
                cpu_limit: cpuValue,
                label_constraints: filledLabels.map((row) => ({
                  key: row.key.trim(),
                  op: row.op,
                  value: row.value.trim(),
                })),
              })
            }
          >
            Review placement
          </Button>
        </>
      )}
      onClose={() => !busy && onClose()}
      title="Deploy an app to the cluster"
      titleId="cluster-app-title"
    >
      <p>Catalog images use bounded memory and reviewed ports. Choose where the app runs; Vaelor sets its environment for you.</p>
      <div className="cd-grid">
        <Select
          label="Application"
          value={form.templateId}
          onChange={(event) => {
            const next = templates.find((item) => item.id === event.target.value);
            setForm((current) => ({
              ...current,
              templateId: event.target.value,
              name: event.target.value,
              port: next ? String(next.offered_port ?? next.default_port) : "",
              // A stateful app cannot spread; fall back to run-once.
              intent: next?.stateful && current.intent === "spread" ? "run-once" : current.intent,
            }));
          }}
        >
          <option value="">Choose a catalog app</option>
          {templates.map((item) => (
            <option key={item.id} value={item.id}>
              {item.name} · {item.memory}
            </option>
          ))}
        </Select>
        <Select
          label="Placement"
          value={form.intent}
          onChange={(event) => setForm({ ...form, intent: event.target.value as Intent })}
        >
          <option value="run-once">Run once — best machine</option>
          <option value="spread" disabled={stateful}>
            {stateful ? "Spread — unavailable (keeps data)" : "Spread — one per machine"}
          </option>
          <option value="pin">Pin — to one machine</option>
        </Select>
        {(form.intent === "spread" || form.intent === "pin") && (
          <Input
            label="Replicas"
            inputMode="numeric"
            value={replicasText}
            onChange={(event) => setForm({ ...form, replicas: event.target.value })}
          />
        )}
        {form.intent === "pin" && (
          <Select
            label="Machine"
            value={form.nodeId}
            onChange={(event) => setForm({ ...form, nodeId: event.target.value })}
          >
            <option value="">Choose a machine</option>
            {joinedWorkers.map((node) => (
              <option key={node.id} value={node.id}>
                {node.name} · {formatMemory(node.inventory.memory_bytes)}
              </option>
            ))}
          </Select>
        )}
        <Input
          label="Deployment name"
          value={form.name}
          onChange={(event) => setForm({ ...form, name: event.target.value })}
        />
        <Input
          label="Published port"
          inputMode="numeric"
          value={form.port}
          onChange={(event) => setForm({ ...form, port: event.target.value })}
        />
      </div>
      <details className="cd-advanced">
        <summary>Advanced — CPU limit and label constraints</summary>
        <div className="cd-advanced__body">
          <Input
            label="CPU limit (cores)"
            inputMode="decimal"
            value={cpuLimit}
            onChange={(event) => setCpuLimit(event.target.value)}
          />
          {!cpuValid && (
            <p className="cd-bad-text" role="alert">Choose a CPU limit from 0.25 to 64 cores.</p>
          )}
          <fieldset className="cd-labels">
            <legend>Node label constraints · restrict placement to machines whose existing labels match. Vaelor-managed labels cannot be used.</legend>
            {labels.map((row, index) => (
              <div key={index} className="cd-label-row">
                <Input
                  label={`Label key ${index + 1}`}
                  placeholder="label key"
                  value={row.key}
                  onChange={(event) => updateLabel(index, { key: event.target.value })}
                />
                <Select
                  label={`Match ${index + 1}`}
                  value={row.op}
                  onChange={(event) => updateLabel(index, { op: event.target.value as LabelOp })}
                >
                  <option value="==">is</option>
                  <option value="!=">is not</option>
                </Select>
                <Input
                  label={`Label value ${index + 1}`}
                  placeholder="value"
                  value={row.value}
                  onChange={(event) => updateLabel(index, { value: event.target.value })}
                />
                <Button
                  aria-label={`Remove label constraint ${index + 1}`}
                  variant="quiet"
                  onClick={() => setLabels((current) => current.filter((_, position) => position !== index))}
                >
                  Remove
                </Button>
              </div>
            ))}
            <div className="cl-actions">
              <Button
                variant="quiet"
                onClick={() => setLabels((current) => [...current, { key: "", op: "==", value: "" }])}
              >
                Add label constraint
              </Button>
            </div>
            {!labelsValid && (
              <p className="cd-bad-text" role="alert">Each constraint needs a valid, non-managed key and value.</p>
            )}
          </fieldset>
        </div>
      </details>
      <p className="cd-fit__summary" role="status">{readBack}</p>
      {sharedVolumeWarning && (
        <Notice severity="warning">
          All {replicas} copies would share one data volume on this machine. An app that keeps a single database file, as most do, can corrupt it with more than one writer; use 1 replica unless this app supports several.
        </Notice>
      )}
      <p className="cd-note">Vaelor sets this app's environment, including any password, automatically — you never enter it here.</p>
    </ClusterDialog>
  );
}
