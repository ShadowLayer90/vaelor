import { useMemo, useState } from "react";
import "../styles/apps-wizard.css";
import { AppsDialog } from "./appsKit";
import { Button, Input, Notice, Select } from "./ui";
import { formatMemory } from "./fleetTypes";
import type { FleetNode } from "./fleetTypes";
import type { PlacementInput, PlacementService } from "../lib/researchedApp";

/**
 * The cluster placement step for an APPROVED researched app (D4d), mirroring the
 * catalog `DeployAppModal`'s controls but per manifest service: a stateful
 * service (one that keeps a named volume) pins to a data node the operator picks,
 * and every stateless service follows one app-level intent (run once on the best
 * machine, or spread one per machine). Advanced, per service: a memory limit that
 * defaults to the manifest figure — with the honest "512 MB default, set a limit"
 * note when research determined no footprint — a CPU limit, and node label
 * constraints. It builds the `placements` map the deploy plan and job consume and
 * never special-cases a specific app; the real fit (and any arithmetic refusal)
 * comes from the reviewed plan the caller previews next.
 */

type AppIntent = "run-once" | "spread";
type LabelOp = "==" | "!=";
interface LabelConstraint {
  key: string;
  op: LabelOp;
  value: string;
}

// Mirrors `cluster_service_reconfigure`/`DeployAppModal` so a bad constraint is
// caught before the plan refuses it.
const LABEL_TOKEN = /^[A-Za-z0-9._-]{1,63}$/;
const RESERVED_LABEL = /^(vaelor|pironman|node)\./i;
// The compose memory bound the backstop enforces: 64 MiB – 256 GiB.
const MIN_MEMORY_MIB = 64;
const MAX_MEMORY_MIB = 262144;

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

interface Props {
  appName: string;
  services: PlacementService[];
  joinedWorkers: FleetNode[];
  /**
   * The head controller as a placement target when it is an eligible one, or
   * null. It carries no `swarm_node_id` label (it is the Swarm manager, not one
   * of its members) so it is not in `joinedWorkers`, yet the capacity ledger
   * counts it and the deploy labels it on demand — so a stateful service must be
   * able to pin to it, notably on a single-controller cluster where it is the
   * only machine with room. Its `id` is `CONTROLLER_PLACEMENT_ID`, the exact id
   * the backend pin fit matches (the ledger `node_id`).
   */
  controllerNode?: FleetNode | null;
  busy: boolean;
  /** A backend refusal (e.g. an unfittable service) from the preview, shown inline. */
  error?: string;
  onClose: () => void;
  onReview: (placements: Record<string, PlacementInput>) => void;
}

export function ClusterAppPlacementModal({
  appName,
  services,
  joinedWorkers,
  controllerNode,
  busy,
  error,
  onClose,
  onReview,
}: Props) {
  const [appIntent, setAppIntent] = useState<AppIntent>("run-once");
  const [pinNodes, setPinNodes] = useState<Record<string, string>>({});
  const [memory, setMemory] = useState<Record<string, string>>(() =>
    Object.fromEntries(services.map((service) => [service.key, String(service.defaultMemoryMib)])),
  );
  const [cpu, setCpu] = useState<Record<string, string>>({});
  const [labels, setLabels] = useState<Record<string, LabelConstraint[]>>({});

  const statefulServices = useMemo(() => services.filter((service) => service.stateful), [services]);
  const statelessServices = useMemo(() => services.filter((service) => !service.stateful), [services]);

  // The machines a stateful service can pin to: every joined worker, plus the
  // head controller when it is an eligible target (labelled so). Each target's
  // value is the node `id` the backend pin fit matches — for the controller that
  // is `CONTROLLER_PLACEMENT_ID` — so the payload names a machine the deploy
  // accepts. The controller leads so it is offered first.
  const pinTargets = useMemo(() => {
    const targets = joinedWorkers.map((node) => ({
      id: node.id,
      label: `${node.name} · ${formatMemory(node.inventory.memory_bytes)}`,
    }));
    if (controllerNode) {
      targets.unshift({
        id: controllerNode.id,
        label: `${controllerNode.name} (controller) · ${formatMemory(controllerNode.inventory.memory_bytes)}`,
      });
    }
    return targets;
  }, [joinedWorkers, controllerNode]);

  const setPin = (key: string, nodeId: string) =>
    setPinNodes((current) => ({ ...current, [key]: nodeId }));
  const setMem = (key: string, value: string) =>
    setMemory((current) => ({ ...current, [key]: value }));
  const setCpuValue = (key: string, value: string) =>
    setCpu((current) => ({ ...current, [key]: value }));
  const serviceLabels = (key: string) => labels[key] ?? [];
  const setServiceLabels = (key: string, next: LabelConstraint[]) =>
    setLabels((current) => ({ ...current, [key]: next }));

  const memoryValid = (key: string) => {
    const raw = (memory[key] ?? "").trim();
    if (raw === "") return true; // falls back to the manifest figure
    const value = Number(raw);
    return Number.isInteger(value) && value >= MIN_MEMORY_MIB && value <= MAX_MEMORY_MIB;
  };
  const cpuValid = (key: string) => {
    const raw = (cpu[key] ?? "").trim();
    if (raw === "") return true; // falls back to the manifest/app default
    const value = Number(raw);
    return value > 0 && value <= 256;
  };
  const labelsValid = (key: string) =>
    serviceLabels(key).filter(labelRowComplete).every(labelRowValid);

  const everyPinChosen = statefulServices.every((service) => pinNodes[service.key]);
  const everyMemoryValid = services.every((service) => memoryValid(service.key));
  const everyCpuValid = services.every((service) => cpuValid(service.key));
  const everyLabelsValid = services.every((service) => labelsValid(service.key));

  const canReview = Boolean(
    services.length &&
      everyPinChosen &&
      everyMemoryValid &&
      everyCpuValid &&
      everyLabelsValid,
  );

  const buildPlacements = (): Record<string, PlacementInput> => {
    const placements: Record<string, PlacementInput> = {};
    for (const service of services) {
      const entry: PlacementInput = service.stateful
        ? { pin_node: pinNodes[service.key] }
        : { intent: appIntent };
      // Send a memory override only when the operator changed it from the
      // manifest figure — an unchanged value keeps the backend's default and its
      // honest "defaulted" flag.
      const memRaw = (memory[service.key] ?? "").trim();
      if (memRaw !== "" && Number(memRaw) !== service.defaultMemoryMib) {
        entry.memory_mib = Number(memRaw);
      }
      const cpuRaw = (cpu[service.key] ?? "").trim();
      if (cpuRaw !== "") entry.cpu = Number(cpuRaw);
      const filledLabels = serviceLabels(service.key).filter(labelRowComplete);
      if (filledLabels.length) {
        entry.label_constraints = filledLabels.map((row) => ({
          key: row.key.trim(),
          op: row.op,
          value: row.value.trim(),
        }));
      }
      placements[service.key] = entry;
    }
    return placements;
  };

  const updateLabel = (key: string, index: number, patch: Partial<LabelConstraint>) =>
    setServiceLabels(
      key,
      serviceLabels(key).map((row, position) => (position === index ? { ...row, ...patch } : row)),
    );

  // Why Review placement is held, in the board's words ("Choose a data node for
  // db, and fix the memory limit above.").
  const unpinned = statefulServices.filter((service) => !pinNodes[service.key]).map((service) => service.key);
  const fixes = [
    !everyMemoryValid && "the memory limit",
    !everyCpuValid && "the CPU limit",
    !everyLabelsValid && "the label constraints",
  ].filter((value): value is string => Boolean(value));
  const heldParts = [
    unpinned.length ? `choose a data node for ${unpinned.join(" and ")}` : "",
    fixes.length ? `fix ${fixes.join(" and ")} above` : "",
  ].filter(Boolean);
  const heldReason = !canReview
    ? heldParts.length
      ? heldParts.join(", and ").replace(/^./, (first) => first.toUpperCase()) + "."
      : "Research named no services to place."
    : undefined;

  return (
    <AppsDialog
      busy={busy}
      className="apps-placement"
      eyebrow="Cluster placement"
      footer={<>
        <Button disabled={busy} onClick={onClose} variant="quiet">Cancel</Button>
        <Button
          busy={busy}
          disabledReason={busy ? undefined : heldReason}
          onClick={() => onReview(buildPlacements())}
          variant="primary"
        >
          Review placement
        </Button>
      </>}
      onClose={onClose}
      title={`Place ${appName} across the cluster`}
      titleId="cluster-app-placement-title"
    >
      <p>
        This deploys the reviewed, digest-pinned app as one Swarm service per manifest service on a
        shared private network. Choose where each service runs; nothing is deployed until you review
        the plan.
      </p>

      <div className="apps-placement__grid">
        {statelessServices.length > 0 && (
          <Select
            id="cluster-app-placement-intent"
            label="Placement for the app's services"
            onChange={(event) => setAppIntent(event.target.value as AppIntent)}
            value={appIntent}
          >
            <option value="run-once">Run once - best machine</option>
            <option value="spread">Spread - one per machine</option>
          </Select>
        )}
        {statefulServices.map((service) => (
          <Select
            id={`cluster-app-placement-pin-${service.key}`}
            key={`pin-${service.key}`}
            label={`Data node for ${service.key}`}
            onChange={(event) => setPin(service.key, event.target.value)}
            value={pinNodes[service.key] ?? ""}
          >
            <option value="">Choose a machine</option>
            {pinTargets.map((target) => (
              <option key={target.id} value={target.id}>{target.label}</option>
            ))}
          </Select>
        ))}
      </div>

      <Notice severity="info" standing>
        {statefulServices.length > 0
          ? `${statefulServices.length} service${statefulServices.length === 1 ? "" : "s"} keep${statefulServices.length === 1 ? "s" : ""} data and pin${statefulServices.length === 1 ? "s" : ""} to the machine you choose; the rest follow the placement above.`
          : "No service keeps persistent data, so every service follows the placement above."}
      </Notice>

      <details className="apps-wizard__disclosure apps-placement__advanced">
        <summary>Advanced - per-service memory, CPU, and label constraints</summary>
        {services.map((service) => (
          <fieldset className="apps-placement__service" key={`advanced-${service.key}`}>
            <legend>{service.key}{service.stateful ? " (keeps data)" : ""}</legend>
            <div className="apps-placement__grid">
              <Input
                error={memoryValid(service.key) ? undefined : "Choose a memory limit from 64 to 262144 MiB."}
                id={`cluster-app-placement-memory-${service.key}`}
                inputMode="numeric"
                label="Memory limit (MiB)"
                onChange={(event) => setMem(service.key, event.target.value)}
                value={memory[service.key] ?? ""}
              />
              <Input
                error={cpuValid(service.key) ? undefined : "Choose a CPU limit above 0 and up to 256 cores."}
                id={`cluster-app-placement-cpu-${service.key}`}
                inputMode="decimal"
                label="CPU limit (cores)"
                onChange={(event) => setCpuValue(service.key, event.target.value)}
                placeholder="app default"
                value={cpu[service.key] ?? ""}
              />
            </div>
            {service.memoryDefaulted && (
              <p className="apps-wizard__muted">
                Using the 512 MB default - research did not determine this service's footprint. Set a
                limit if it needs more or less.
              </p>
            )}
            <p className="apps-wizard__muted">Restrict placement to machines whose existing labels match. Vaelor-managed labels cannot be used.</p>
            {serviceLabels(service.key).map((row, index) => (
              <div className="apps-placement__label-row" key={index}>
                <input
                  aria-label={`${service.key} label key ${index + 1}`}
                  className="ui-control"
                  onChange={(event) => updateLabel(service.key, index, { key: event.target.value })}
                  placeholder="label key"
                  value={row.key}
                />
                <select
                  aria-label={`${service.key} label operator ${index + 1}`}
                  className="ui-control"
                  onChange={(event) => updateLabel(service.key, index, { op: event.target.value as LabelOp })}
                  value={row.op}
                >
                  <option value="==">is</option>
                  <option value="!=">is not</option>
                </select>
                <input
                  aria-label={`${service.key} label value ${index + 1}`}
                  className="ui-control"
                  onChange={(event) => updateLabel(service.key, index, { value: event.target.value })}
                  placeholder="value"
                  value={row.value}
                />
                <Button
                  onClick={() =>
                    setServiceLabels(
                      service.key,
                      serviceLabels(service.key).filter((_, position) => position !== index),
                    )
                  }
                  variant="quiet"
                >
                  Remove
                </Button>
              </div>
            ))}
            <div>
              <Button
                onClick={() =>
                  setServiceLabels(service.key, [...serviceLabels(service.key), { key: "", op: "==", value: "" }])
                }
                variant="quiet"
              >
                Add label constraint
              </Button>
            </div>
            {!labelsValid(service.key) && (
              <p className="apps-placement__field-error" role="alert">Each constraint needs a valid, non-managed key and value.</p>
            )}
          </fieldset>
        ))}
      </details>

      {/* The preview's refusal (e.g. a service no machine can fit) sits where the
          board draws it: under the choices it is about, inside the dialog. */}
      {error && <Notice severity="danger">{error}</Notice>}
    </AppsDialog>
  );
}
