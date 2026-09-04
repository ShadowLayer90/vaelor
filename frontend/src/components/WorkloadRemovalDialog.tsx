import { useEffect, useRef, useState } from "react";
import "../styles/workload-removal.css";
import { ModalShell } from "./ModalShell";
import { Button, Input } from "./ui";

export interface RemovalDependency {
  kind: string;
  id: string;
  name: string;
  relationship: string;
  active: boolean;
  blocking: boolean;
}

export interface RemovalResource {
  kind: "app" | "model" | "runtime";
  id: string;
  name: string;
  display_identity?: string;
  project?: string;
  path?: string;
}

export type DependencyStrategy = "resolve" | "cascade";

export interface WorkloadRemovalPlan {
  resource: RemovalResource;
  dependencies: RemovalDependency[];
  affected_resources: RemovalDependency[];
  blocked: boolean;
  plan_digest: string;
  confirmation: string;
  display_identity?: string;
  defaults: { dependency_strategy?: DependencyStrategy | null; retain_data: boolean; create_backup: boolean };
  requirements: {
    dependency_strategy_required?: boolean;
    dependency_strategies?: DependencyStrategy[];
    cascade_required?: boolean;
    retain_data_supported: boolean;
    backup_supported: boolean;
  };
  disclosures: string[];
}

export interface WorkloadRemovalOptions {
  dependency_strategy: DependencyStrategy | null;
  cascade?: boolean;
  retain_data: boolean;
  create_backup: boolean;
  confirmation: string;
}

export function WorkloadRemovalDialog({
  plan,
  busy,
  error,
  onCancel,
  onConfirm,
}: {
  plan: WorkloadRemovalPlan | null;
  busy: boolean;
  error?: string;
  onCancel: () => void;
  onConfirm: (options: WorkloadRemovalOptions) => void;
}) {
  const [options, setOptions] = useState<WorkloadRemovalOptions>({ dependency_strategy: null, cascade: false, retain_data: true, create_backup: true, confirmation: "" });
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!plan) return;
    setOptions({
      dependency_strategy: null,
      cascade: false,
      retain_data: plan.requirements.retain_data_supported
        ? plan.defaults.retain_data
        : false,
      create_backup: plan.defaults.create_backup,
      confirmation: "",
    });
  }, [plan]);

  const displayIdentity = plan?.resource.display_identity ?? plan?.display_identity ?? plan?.resource.name ?? "";
  if (!plan) return null;
  const strategyMissing = options.dependency_strategy === null;
  const strategyExecutable = plan.blocked ? options.dependency_strategy === "cascade" : options.dependency_strategy === "resolve";
  const strategyInvalid = !strategyMissing && !strategyExecutable;
  const canSubmit = !busy && strategyExecutable && options.confirmation === displayIdentity;
  const activeDependencies = plan.dependencies.filter((item) => item.active);

  return <ModalShell
    backdropClassName="modal-shell--opaque"
    className="workload-removal"
    describedBy="workload-removal-description"
    initialFocusRef={closeRef}
    labelledBy="workload-removal-title"
    onClose={() => { if (!busy) onCancel(); }}
    size="wide"
  >
      <header>
        <div><span>Dependency-aware removal</span><h2 id="workload-removal-title">Remove {displayIdentity}?</h2><p id="workload-removal-description">Vaelor checked the live workload graph. Review everything that will stop or remain before approval.</p></div>
          <Button ref={closeRef} aria-label="Close removal plan" disabled={busy} onClick={onCancel} variant="quiet">×</Button>
      </header>

      {error && <div className="workload-removal__error" role="alert">{error}</div>}
      <section aria-labelledby="removal-impact-title">
          <div className="workload-removal__heading"><h3 id="removal-impact-title">Dependency impact</h3><span>{activeDependencies.length} active · {plan.dependencies.length} total</span></div>
          {plan.dependencies.length ? <ul className="workload-removal__dependencies">{plan.dependencies.map((item) => <li key={`${item.kind}-${item.id}-${item.relationship}`} data-blocking={item.blocking}><div><strong>{item.name}</strong><small>{item.kind} · {item.relationship}</small></div><span>{item.active ? "Active" : "Inactive"}{item.blocking ? " · blocks removal" : ""}</span></li>)}</ul> : <p className="workload-removal__clear">No managed service, model, Assistant, AI Chat, or agent currently depends on this resource.</p>}
      </section>

          {!!plan.affected_resources.length && <section aria-labelledby="removal-affected-title"><div className="workload-removal__heading"><h3 id="removal-affected-title">Removed with this resource</h3><span>{plan.affected_resources.length} related resources</span></div><ul className="workload-removal__dependencies">{plan.affected_resources.map((item) => <li key={`${item.kind}-${item.id}-${item.relationship}`}><div><strong>{item.name}</strong><small>{item.kind} · {item.relationship}</small></div><span>{item.active ? "Running" : "Stopped"}</span></li>)}</ul></section>}

      <form onSubmit={(event) => { event.preventDefault(); if (canSubmit) onConfirm(options); }}>
        <fieldset><legend>1. Resolve active dependencies</legend>
          <label className="workload-removal__choice"><input className="ui-control--radio" type="radio" name="dependency-strategy" checked={options.dependency_strategy === "resolve"} onChange={() => setOptions((current) => ({ ...current, dependency_strategy: "resolve", cascade: false }))} /><span><strong>{plan.blocked ? "Resolve dependencies separately" : "No active dependencies"}</strong><small>{plan.blocked ? "Cancel this removal, resolve the active model or connection, then review a fresh dependency report." : "Nothing currently blocks this resource; keep the removal scoped to it."}</small></span></label>
          <label className="workload-removal__choice"><input className="ui-control--radio" type="radio" name="dependency-strategy" checked={options.dependency_strategy === "cascade"} disabled={!plan.blocked} onChange={() => setOptions((current) => ({ ...current, dependency_strategy: "cascade", cascade: true }))} /><span><strong>Deactivate dependents and remove together</strong><small>Explicit cascade: Vaelor deactivates only the listed managed relationships before removing this resource.</small></span></label>
          {strategyMissing && <p className="workload-removal__blocked" role="status">Choose how Vaelor should handle dependencies before approving removal.</p>}{strategyInvalid && <p className="workload-removal__blocked" role="status">Choose cascade removal for this blocked plan, or resolve its dependencies before approving.</p>}
        </fieldset>

        <fieldset><legend>2. Application data</legend>
          <label className="workload-removal__choice"><input className="ui-control--radio" type="radio" name="data" checked={options.retain_data} disabled={!plan.requirements.retain_data_supported} onChange={() => setOptions((current) => ({ ...current, retain_data: true }))} /><span><strong>Retain persistent data</strong><small>{plan.requirements.retain_data_supported ? "Keep managed volumes so they can be reattached later." : "This resource is the data being removed, so retaining it would cancel removal."}</small></span></label>
          <label className="workload-removal__choice"><input className="ui-control--radio" type="radio" name="data" checked={!options.retain_data} onChange={() => setOptions((current) => ({ ...current, retain_data: false }))} /><span><strong>Delete persistent data</strong><small>This permanently removes the data covered by the server plan after the recovery step.</small></span></label>
        </fieldset>

        <fieldset><legend>3. Recovery protection</legend>
          <label className="workload-removal__choice"><input className="ui-control--checkbox" type="checkbox" checked={options.create_backup} disabled={!plan.requirements.backup_supported} onChange={(event) => setOptions((current) => ({ ...current, create_backup: event.target.checked }))} /><span><strong>Create a verified backup or checkpoint first</strong><small>The job result records its identifier and whether retained data can be restored.</small></span></label>
          {!options.create_backup && plan.requirements.backup_supported && <p className="workload-removal__blocked">You are choosing removal without a new recovery point. Existing checkpoints are unchanged.</p>}
        </fieldset>

        <section className="workload-removal__disclosures" aria-labelledby="removal-disclosures-title"><h3 id="removal-disclosures-title">What the executor guarantees</h3><ul>{plan.disclosures.map((item) => <li key={item}>{item}</li>)}</ul></section>
        <div className="workload-removal__confirmation">
          <Input
            autoComplete="off"
            label={<>Type <code>{displayIdentity}</code> to approve this exact plan</>}
            onChange={(event) => setOptions((current) => ({ ...current, confirmation: event.target.value }))}
            value={options.confirmation}
          />
        </div>
        <footer>
          <Button disabled={busy} onClick={onCancel}>Cancel</Button>
            <Button disabled={!canSubmit} type="submit" variant="danger">{busy ? "Removing safely…" : options.cascade ? "Approve cascade removal" : "Approve removal"}</Button>
        </footer>
        {strategyInvalid && <p className="workload-removal__blocked" role="status">This dependency strategy cannot execute the reviewed plan.</p>}
      </form>
  </ModalShell>;
}
