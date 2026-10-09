import { Button, Input, Notice, Select } from "./ui";
import { useState, type Dispatch, type SetStateAction } from "react";
import { bytesFromTyped, formatQuantity } from "../lib/format";
import { OptionCard } from "./ClusterPrimitives";
import { ClusterDialog } from "./ClusterDialog";
import { GpuServeForm } from "./GpuServeForm";
import type { FleetNode, InferenceRuntimes, LlmForm } from "./fleetTypes";
import { formatMemory } from "./fleetTypes";
import type { Session } from "../types";
import "../styles/cluster-dialogs.css";

interface FleetLlmModalProps {
  busy: boolean;
  /** Why the plan review was refused (VD-189): shown in this dialog, never on the inert page beneath. */
  error?: string;
  session: Session;
  eligibleWorkers: FleetNode[];
  /**
   * GPU-capable, eligible participants — the serve targets for the GPU mode.
   * The head controller is one of them when its own GPU was discovered.
   */
  gpuNodes: FleetNode[];
  form: LlmForm;
  inferenceRuntimes: InferenceRuntimes;
  selectableTargets: FleetNode[];
  setForm: Dispatch<SetStateAction<LlmForm>>;
  onClose: () => void;
  onReview: (nodeId: string | undefined, payload: Record<string, unknown>) => void;
  onServeGpu: (payload: Record<string, unknown>) => void;
  onToggleWorker: (nodeId: string) => void;
  /**
   * Where a `single_refused` fit sends the owner: the single-node AI Chat model
   * install, which is how a one-box model is served (VD-127 D6).
   */
  onServeAsAiChatModel?: () => void;
}

type DeploymentMode = LlmForm["deploymentMode"];

/** The four deployment modes, in the board's order (ClusterDialogsServe). */
const MODES: ReadonlyArray<readonly [DeploymentMode, string, string]> = [
  ["gpu", "GPU serving", "Serve on the GPU cluster with vLLM, sized by the fit engine."],
  ["single", "Single node", "Run one model copy on this controller or an active worker."],
  ["replicated", "Replicated service", "One independent copy per selected node for throughput and failover."],
  ["pooled", "Pooled memory", "Split one converted model across exactly 2, 4, or 8 matching-architecture nodes."],
];

/**
 * Configure a cluster LLM server (VD-200 boards ClusterDialogsServe and
 * ClusterDialogsServe2). GPU serving is the wide dialog whose Serve is gated by
 * the fit decision (GpuServeForm); the three CPU modes are the standard-width
 * form that keeps the reviewed plan step ("Review fit and deployment").
 */
export function FleetLlmModal({
  busy,
  error,
  session,
  eligibleWorkers,
  gpuNodes,
  form,
  inferenceRuntimes,
  selectableTargets,
  setForm,
  onClose,
  onReview,
  onServeGpu,
  onToggleWorker,
  onServeAsAiChatModel,
}: FleetLlmModalProps) {
  // The dialog's footer element: GpuServeForm portals Cancel and Serve into it.
  const [footerSlot, setFooterSlot] = useState<HTMLDivElement | null>(null);
  const isGpu = form.deploymentMode === "gpu";
  const hasValidPlacement = form.deploymentMode === "single"
    ? Boolean(form.nodeId)
    : form.deploymentMode === "pooled"
      ? [2, 4, 8].includes(form.nodeIds.length) && Boolean(form.pooledModel)
      : form.nodeIds.length >= 2 && form.nodeIds.length <= 8;
  const pooledModel = inferenceRuntimes.pooled?.models.find(
    (model) => model.id === form.pooledModel,
  );
  const detailsMissing = form.deploymentMode !== "pooled"
    && (!form.repository || !form.file || !(Number(form.sizeGb) > 0));
  // Why "Review fit and deployment" is off, said beside it (the board's line).
  const reviewReason = busy
    ? undefined
    : !hasValidPlacement
      ? form.deploymentMode === "single"
        ? "Choose a compute target."
        : form.deploymentMode === "pooled"
          ? "Select 2, 4 or 8 nodes and a converted model preset."
          : "Select 2 to 8 nodes."
      : detailsMissing
        ? "Enter the repository, the GGUF file and its exact size."
        : undefined;

  const cpuFooter = (
    <>
      <Button variant="secondary" onClick={onClose}>Cancel</Button>
      <Button
        variant="primary"
        disabled={busy || !hasValidPlacement || detailsMissing}
        disabledReason={reviewReason}
        onClick={() => onReview(form.deploymentMode === "single" ? form.nodeId : undefined, {
          deployment_mode: form.deploymentMode,
          ...(form.deploymentMode !== "single" ? { node_ids: form.nodeIds } : {}),
          ...(form.deploymentMode === "pooled" ? { pooled_model: form.pooledModel } : {
            model_file: form.file,
            model_repo: form.repository,
            model_size_bytes: bytesFromTyped(form.sizeGb, "GiB") ?? 0,
            port: Number(form.port),
          }),
          name: form.name,
        })}
      >
        Review fit and deployment
      </Button>
    </>
  );

  return (
    <ClusterDialog
      className="cd-dialog"
      error={error}
      eyebrow="Managed inference"
      footer={isGpu ? <div className="cd-footer-slot" ref={setFooterSlot} /> : cpuFooter}
      onClose={onClose}
      size={isGpu ? "wide" : "standard"}
      title="Configure a cluster LLM server"
      titleId="cluster-llm-title"
    >
      <p>{isGpu
        ? "Serve a model on the GPU cluster with vLLM. Pick a curated model or paste a Hugging Face link; the fit decision tells you whether it runs on one machine, across several, or not at all."
        : "Choose where independent model replicas run, then provide the exact details from its reviewed Hugging Face listing."}</p>
      <fieldset className="cd-group">
        <legend className="cd-label">Deployment mode</legend>
        <div className={isGpu ? "cd-options cd-options--four" : "cd-options cd-options--four cd-options--compact"}>
          {MODES.map(([value, title, detail]) => (
            <OptionCard
              key={value}
              checked={form.deploymentMode === value}
              // The standard-width CPU form keeps all four choices, titles only.
              detail={isGpu ? detail : undefined}
              name="llm-deployment-mode"
              onChange={() => setForm({ ...form, deploymentMode: value })}
              title={title}
              type="radio"
              value={value}
            />
          ))}
        </div>
      </fieldset>
      {isGpu ? (
        <GpuServeForm
          session={session}
          gpuNodes={gpuNodes}
          busy={busy}
          name={form.gpuName}
          onName={(value) => setForm({ ...form, gpuName: value })}
          onServe={onServeGpu}
          onClose={onClose}
          onServeAsAiChatModel={onServeAsAiChatModel}
          footerSlot={footerSlot}
        />
      ) : (
        <>
          {form.deploymentMode === "single" ? (
            <Select label="Compute target" value={form.nodeId} onChange={(event) => setForm({ ...form, nodeId: event.target.value })}>
              <option value="">Choose this controller or an active worker</option>
              {selectableTargets.map((node) => (
                <option key={node.id} value={node.id}>{node.name} · {formatMemory(node.inventory.memory_bytes)}</option>
              ))}
            </Select>
          ) : (
            <fieldset className="cd-group">
              <legend className="cd-label">{form.deploymentMode === "pooled" ? "Active workers" : "Active compute nodes"}</legend>
              <p className="cd-note">{form.deploymentMode === "pooled"
                ? "Select exactly 2, 4, or 8 nodes in order. The first selected node is the root and stores the complete model."
                : "Select 2–8 nodes. The controller can participate; every selected node stores its own model copy."}</p>
              <div className="cd-options">
                {selectableTargets.map((node) => (
                  <OptionCard
                    key={node.id}
                    checked={form.nodeIds.includes(node.id)}
                    detail={`${formatMemory(node.inventory.memory_bytes)} memory · ${node.inventory.architecture ?? "Unknown architecture"}`}
                    onChange={() => onToggleWorker(node.id)}
                    title={node.name}
                    type="checkbox"
                  />
                ))}
              </div>
              <output aria-live="polite" className="cd-note">{form.deploymentMode === "pooled"
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
              <Notice severity="warning" standing heading="No replica failover">
                Every selected node participates in each token. One lost node stops this model. Use private wired Ethernet; the worker protocol is not encrypted.
                {pooledModel && ` ${pooledModel.name} · ${pooledModel.kv_heads} KV heads · up to ${pooledModel.max_sequence_length.toLocaleString()} tokens.`}
              </Notice>
            </>
          ) : (
            <>
              <Input label="Hugging Face repository" placeholder="owner/model-GGUF" value={form.repository} onChange={(event) => setForm({ ...form, repository: event.target.value })} />
              <Input label="GGUF file" placeholder="model.Q4_K_M.gguf" value={form.file} onChange={(event) => setForm({ ...form, file: event.target.value })} />
              <div className="cd-grid">
                <Input label="Exact file size (GiB)" inputMode="decimal" placeholder="4.2" value={form.sizeGb} onChange={(event) => setForm({ ...form, sizeGb: event.target.value })} />
                <Input label="API port" inputMode="numeric" value={form.port} onChange={(event) => setForm({ ...form, port: event.target.value })} />
              </div>
            </>
          )}
        </>
      )}
    </ClusterDialog>
  );
}
