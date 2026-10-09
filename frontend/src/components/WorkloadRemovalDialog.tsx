import { useRef, useState } from "react";
import "../styles/workload-removal.css";
import { AppsDialog, AppsInset } from "./appsKit";
import { StatusPill } from "./StatusPill";
import { Button, Input, LoadingLines } from "./ui";

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

/**
 * The removal's first step (the ManageRemoval board): it opens the moment
 * Review removal is pressed, so the click has a visible result while the plan
 * loads. Cancel, Escape and the backdrop all hand the page back at once
 * (VLR-071): the plan request cannot be aborted and may never answer.
 */
export function RemovalCheckingDialog({ name, onCancel }: { name: string; onCancel: () => void }) {
  const cancelRef = useRef<HTMLButtonElement>(null);
  return (
    <AppsDialog
      eyebrow="Dependency-aware removal"
      footer={<Button onClick={onCancel} ref={cancelRef} type="button">Cancel</Button>}
      initialFocusRef={cancelRef}
      onClose={onCancel}
      size="standard"
      title={`Checking what depends on ${name}`}
      titleId="removal-loading-title"
    >
      <p role="status">Vaelor is reading the live workload graph so it can show exactly what would stop or remain.</p>
      <LoadingLines label="Reading the workload graph" />
    </AppsDialog>
  );
}

/** A dependency's pill: an active blocker red, an active dependent green, the rest grey. */
function dependencyPill(item: RemovalDependency) {
  if (item.active && item.blocking) return <StatusPill label="Active · blocks removal" tone="danger" />;
  if (item.active) return <StatusPill label="Active" tone="success" />;
  return <StatusPill label={item.blocking ? "Inactive · blocks removal" : "Inactive"} tone="neutral" />;
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
  // The plan's defaults are read once, when this plan's dialog mounts: the
  // caller keys the dialog by `plan_digest`, so a different plan is a fresh
  // dialog. Re-applying them in an effect on every new `plan` object reset the
  // owner's choices and typed confirmation whenever the same plan was handed
  // over again - and, landing after the first input, could leave Approve
  // disabled with the form visibly filled in.
  const [options, setOptions] = useState<WorkloadRemovalOptions>(() => ({
    dependency_strategy: null,
    cascade: false,
    retain_data: plan
      ? (plan.requirements.retain_data_supported ? plan.defaults.retain_data : false)
      : true,
    create_backup: plan ? plan.defaults.create_backup : true,
    confirmation: "",
  }));
  // Cancel takes focus first: the safe answer is the one a stray Enter gives.
  const cancelRef = useRef<HTMLButtonElement>(null);

  const displayIdentity = plan?.resource.display_identity ?? plan?.display_identity ?? plan?.resource.name ?? "";
  if (!plan) return null;
  const strategyMissing = options.dependency_strategy === null;
  const strategyExecutable = plan.blocked ? options.dependency_strategy === "cascade" : options.dependency_strategy === "resolve";
  const strategyInvalid = !strategyMissing && !strategyExecutable;
  const typed = options.confirmation === displayIdentity;
  const canSubmit = !busy && strategyExecutable && typed;
  const activeDependencies = plan.dependencies.filter((item) => item.active);
  // The reason sits under the disabled Approve; the strategy's own line says the rest.
  const approveReason = busy || !strategyExecutable ? undefined : !typed ? `Type ${displayIdentity} to approve.` : undefined;

  return (
    <AppsDialog
      busy={busy}
      className="workload-removal"
      describedBy="workload-removal-description"
      error={error || undefined}
      eyebrow="Dependency-aware removal"
      eyebrowTone="danger"
      footer={<>
        <Button disabled={busy} onClick={onCancel} ref={cancelRef}>Cancel</Button>
        <Button
          busy={busy}
          className="manage-solid-danger"
          disabled={!canSubmit}
          disabledReason={approveReason}
          form="workload-removal-form"
          type="submit"
          variant="danger"
        >
          {busy ? "Removing safely…" : options.cascade ? "Approve cascade removal" : "Approve removal"}
        </Button>
      </>}
      initialFocusRef={cancelRef}
      onClose={onCancel}
      role="alertdialog"
      size="standard"
      title={`Remove ${displayIdentity}?`}
      titleId="workload-removal-title"
    >
      <p id="workload-removal-description">Vaelor checked the live workload graph. Review everything that will stop or remain before approval.</p>
      <section aria-labelledby="removal-impact-title" className="workload-removal__group">
        <div className="workload-removal__heading"><h3 id="removal-impact-title">Dependency impact</h3><span>{activeDependencies.length} active · {plan.dependencies.length} total</span></div>
        {plan.dependencies.length ? (
          <ul className="workload-removal__dependencies">
            {plan.dependencies.map((item) => (
              <li data-blocking={item.blocking && item.active} key={`${item.kind}-${item.id}-${item.relationship}`}>
                <div><strong>{item.name}</strong><small>{item.kind} · {item.relationship}</small></div>
                {dependencyPill(item)}
              </li>
            ))}
          </ul>
        ) : <p className="workload-removal__clear">No managed service, model, Assistant, AI Chat, or agent currently depends on this resource.</p>}
      </section>

      {!!plan.affected_resources.length && (
        <section aria-labelledby="removal-affected-title" className="workload-removal__group">
          <div className="workload-removal__heading"><h3 id="removal-affected-title">Removed with this resource</h3><span>{plan.affected_resources.length} related resources</span></div>
          <ul className="workload-removal__dependencies">
            {plan.affected_resources.map((item) => (
              <li key={`${item.kind}-${item.id}-${item.relationship}`}>
                <div><strong>{item.name}</strong><small>{item.kind} · {item.relationship}</small></div>
                <StatusPill label={item.active ? "Running" : "Stopped"} tone={item.active ? "success" : "neutral"} />
              </li>
            ))}
          </ul>
        </section>
      )}

      <form className="workload-removal__form" id="workload-removal-form" onSubmit={(event) => { event.preventDefault(); if (canSubmit) onConfirm(options); }}>
        <fieldset><legend>1. Resolve active dependencies</legend>
          <label className="workload-removal__choice"><input className="ui-control--radio" type="radio" name="dependency-strategy" checked={options.dependency_strategy === "resolve"} onChange={() => setOptions((current) => ({ ...current, dependency_strategy: "resolve", cascade: false }))} /><span><strong>{plan.blocked ? "Resolve dependencies separately" : "No active dependencies"}</strong><small>{plan.blocked ? "Cancel this removal, resolve the active model or connection, then review a fresh dependency report." : "Nothing currently blocks this resource; keep the removal scoped to it."}</small></span></label>
          {plan.blocked && <label className="workload-removal__choice"><input className="ui-control--radio" type="radio" name="dependency-strategy" checked={options.dependency_strategy === "cascade"} onChange={() => setOptions((current) => ({ ...current, dependency_strategy: "cascade", cascade: true }))} /><span><strong>Deactivate dependents and remove together</strong><small>Explicit cascade: Vaelor deactivates only the listed managed relationships before removing this resource.</small></span></label>}
          {strategyMissing && <p className="workload-removal__blocked" role="status">Choose how Vaelor should handle dependencies before approving removal.</p>}
          {strategyInvalid && <p className="workload-removal__blocked" role="status">Choose cascade removal for this blocked plan, or resolve its dependencies before approving.</p>}
        </fieldset>

        <fieldset><legend>2. Application data</legend>
          <label className="workload-removal__choice"><input className="ui-control--radio" type="radio" name="data" checked={options.retain_data} disabled={!plan.requirements.retain_data_supported} onChange={() => setOptions((current) => ({ ...current, retain_data: true }))} /><span><strong>Retain persistent data</strong><small>{plan.requirements.retain_data_supported ? "Keep managed volumes so they can be reattached later." : "This resource is the data being removed, so retaining it would cancel removal."}</small></span></label>
          <label className="workload-removal__choice"><input className="ui-control--radio" type="radio" name="data" checked={!options.retain_data} onChange={() => setOptions((current) => ({ ...current, retain_data: false }))} /><span><strong>Delete persistent data</strong><small>This permanently removes the data covered by the server plan after the recovery step.</small></span></label>
        </fieldset>

        <fieldset><legend>3. Recovery protection</legend>
          <label className="workload-removal__choice"><input className="ui-control--checkbox" type="checkbox" checked={options.create_backup} disabled={!plan.requirements.backup_supported} onChange={(event) => setOptions((current) => ({ ...current, create_backup: event.target.checked }))} /><span><strong>Create a verified backup or checkpoint first</strong><small>The job result records its identifier and whether retained data can be restored.</small></span></label>
          {!options.create_backup && plan.requirements.backup_supported && <p className="workload-removal__warning">You are choosing removal without a new recovery point. Existing checkpoints are unchanged.</p>}
        </fieldset>

        {plan.disclosures.length > 0 && (
          <AppsInset className="workload-removal__disclosures" title="What the executor guarantees">
            <ul>{plan.disclosures.map((item) => <li key={item}>{item}</li>)}</ul>
          </AppsInset>
        )}
        <div className="workload-removal__confirmation">
          <Input
            autoComplete="off"
            label={<>Type <code>{displayIdentity}</code> to approve this exact plan</>}
            onChange={(event) => setOptions((current) => ({ ...current, confirmation: event.target.value }))}
            value={options.confirmation}
          />
        </div>
        {strategyInvalid && <p className="workload-removal__blocked" role="status">This dependency strategy cannot execute the reviewed plan.</p>}
      </form>
    </AppsDialog>
  );
}
