import { Button, Checkbox, Input, Select } from "./ui";
import type { Dispatch, SetStateAction } from "react";
import { formatQuantity } from "../lib/format";
import { ModalShell } from "./ModalShell";
import type { FleetNode, InferenceRuntimes, LlmForm } from "./fleetTypes";
import { formatMemory } from "./fleetTypes";

interface FleetLlmModalProps {
  busy: boolean;
  eligibleWorkers: FleetNode[];
  form: LlmForm;
  inferenceRuntimes: InferenceRuntimes;
  selectableTargets: FleetNode[];
  setForm: Dispatch<SetStateAction<LlmForm>>;
  onClose: () => void;
  onReview: (nodeId: string | undefined, payload: Record<string, unknown>) => void;
  onToggleWorker: (nodeId: string) => void;
}

export function FleetLlmModal({
  busy,
  eligibleWorkers,
  form,
  inferenceRuntimes,
  selectableTargets,
  setForm,
  onClose,
  onReview,
  onToggleWorker,
}: FleetLlmModalProps) {
  const hasValidPlacement = form.deploymentMode === "single"
    ? Boolean(form.nodeId)
    : form.deploymentMode === "pooled"
      ? [2, 4, 8].includes(form.nodeIds.length) && Boolean(form.pooledModel)
      : form.nodeIds.length >= 2 && form.nodeIds.length <= 8;
  const pooledModel = inferenceRuntimes.pooled?.models.find(
    (model) => model.id === form.pooledModel,
  );

  return (
    <ModalShell labelledBy="cluster-llm-title" onClose={onClose}>
      <section className="fleet-enroll">
        <header>
          <span className="page-eyebrow">MANAGED INFERENCE</span>
          <h2 id="cluster-llm-title">Configure a cluster LLM server</h2>
          <p>Choose where independent model replicas run, then provide the exact details from its reviewed Hugging Face listing.</p>
        </header>
        <div className="fleet-enroll__grid">
          <fieldset className="fleet-placement">
            <legend>Deployment mode</legend>
            <div className="fleet-placement__modes">
              <label className={form.deploymentMode === "single" ? "is-selected" : ""}>
                <input
                  className="fleet-placement-mode-input" checked={form.deploymentMode === "single"}
                  name="llm-deployment-mode"
                  onChange={() => setForm({ ...form, deploymentMode: "single" })}
                  type="radio"
                  value="single"
                />
                <span><strong>Single node</strong><small>Run one model copy on this controller or an active worker.</small></span>
              </label>
              <label className={form.deploymentMode === "replicated" ? "is-selected" : ""}>
                <input
                  className="fleet-placement-mode-input" checked={form.deploymentMode === "replicated"}
                  name="llm-deployment-mode"
                  onChange={() => setForm({ ...form, deploymentMode: "replicated" })}
                  type="radio"
                  value="replicated"
                />
                <span><strong>Replicated service</strong><small>One independent copy per selected node for throughput and failover.</small></span>
              </label>
              <label className={form.deploymentMode === "pooled" ? "is-selected" : ""}>
                <input
                  className="fleet-placement-mode-input" checked={form.deploymentMode === "pooled"}
                  name="llm-deployment-mode"
                  onChange={() => setForm({ ...form, deploymentMode: "pooled" })}
                  type="radio"
                  value="pooled"
                />
                <span><strong>Pooled memory</strong><small>Split one converted model across exactly 2, 4, or 8 matching-architecture nodes.</small></span>
              </label>
            </div>
          </fieldset>
          {form.deploymentMode === "single" ? (
            <div className="fleet-placement__workers">
              <Select className="fleet-placement-select" label="Compute target" value={form.nodeId} onChange={(event) => setForm({ ...form, nodeId: event.target.value })}>
                <option value="">Choose this controller or an active worker</option>
                {selectableTargets.map((node) => (
                  <option key={node.id} value={node.id}>{node.name} · {formatMemory(node.inventory.memory_bytes)}</option>
                ))}
              </Select>
            </div>
          ) : (
            <fieldset className="fleet-placement fleet-placement__workers">
              <legend>{form.deploymentMode === "pooled" ? "Active workers" : "Active compute nodes"}</legend>
              <p>{form.deploymentMode === "pooled"
                ? "Select exactly 2, 4, or 8 nodes in order. The first selected node is the root and stores the complete model."
                : "Select 2–8 nodes. The controller can participate; every selected node stores its own model copy."}</p>
              <div className="fleet-worker-list">
                {selectableTargets.map((node) => (
                  <label key={node.id}>
                    <input checked={form.nodeIds.includes(node.id)} className="fleet-placement-worker-input" onChange={() => onToggleWorker(node.id)} type="checkbox" />
                    <span><strong>{node.name}</strong><small>{formatMemory(node.inventory.memory_bytes)} memory · {node.inventory.architecture ?? "Unknown architecture"}</small></span>
                  </label>
                ))}
              </div>
              <output aria-live="polite">{form.deploymentMode === "pooled"
                ? `${form.nodeIds.length} selected · ${form.nodeIds.length ? `root: ${eligibleWorkers.find((node) => node.id === form.nodeIds[0])?.name ?? "unknown"}` : "select the root first"}`
                : `${form.nodeIds.length} of 2 minimum nodes selected`}</output>
            </fieldset>
          )}
          <Input label="Deployment name" value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} />
          {form.deploymentMode === "pooled" ? (
            <>
              <Select label="Converted model preset" value={form.pooledModel} onChange={(event) => setForm({ ...form, pooledModel: event.target.value })}>
                <option value="">Choose a verified model</option>
                {inferenceRuntimes.pooled?.models.map((model) => (
                  <option key={model.id} value={model.id}>{model.name} · {formatQuantity(model.download_bytes, "model")}</option>
                ))}
              </Select>
              <aside className="fleet-pooled-warning">
                <strong>No replica failover</strong>
                <p>Every selected node participates in each token. One lost node stops this model. Use private wired Ethernet; the worker protocol is not encrypted.</p>
                {pooledModel && <small>{pooledModel.name} · {pooledModel.kv_heads} KV heads · up to {pooledModel.max_sequence_length.toLocaleString()} tokens</small>}
              </aside>
            </>
          ) : (
            <>
              <Input label="Hugging Face repository" placeholder="owner/model-GGUF" value={form.repository} onChange={(event) => setForm({ ...form, repository: event.target.value })} />
              <Input label="GGUF file" placeholder="model.Q4_K_M.gguf" value={form.file} onChange={(event) => setForm({ ...form, file: event.target.value })} />
              <Input label="Exact file size (GB)" inputMode="decimal" placeholder="4.2" value={form.sizeGb} onChange={(event) => setForm({ ...form, sizeGb: event.target.value })} />
              <Input label="API port" inputMode="numeric" value={form.port} onChange={(event) => setForm({ ...form, port: event.target.value })} />
            </>
          )}
          <div className="fleet-enroll__check">
            <Checkbox
              checked={form.useForAssistant}
              label="Use for Vaelor Assistant after its API passes"
              onChange={(event) => setForm({ ...form, useForAssistant: event.target.checked })}
            />
          </div>
        </div>
        <div className="fleet-enroll__footer">
          <Button variant="quiet" onClick={onClose}>Cancel</Button>
          <Button variant="primary"

            disabled={busy || !hasValidPlacement || (form.deploymentMode !== "pooled" && (!form.repository || !form.file || !(Number(form.sizeGb) > 0)))}
            onClick={() => onReview(form.deploymentMode === "single" ? form.nodeId : undefined, {
              deployment_mode: form.deploymentMode,
              ...(form.deploymentMode !== "single" ? { node_ids: form.nodeIds } : {}),
              ...(form.deploymentMode === "pooled" ? { pooled_model: form.pooledModel } : {
                model_file: form.file,
                model_repo: form.repository,
                model_size_bytes: Math.round(Number(form.sizeGb) * 1024 ** 3),
                port: Number(form.port),
              }),
              name: form.name,
              use_for_assistant: form.useForAssistant,
            })}
          >
            Review fit and deployment
          </Button>
        </div>
      </section>
    </ModalShell>
  );
}
