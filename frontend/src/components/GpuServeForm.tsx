import { useEffect, useId, useMemo, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Button, Input, Notice, SegmentedControl, Select } from "./ui";
import { OptionCard } from "./ClusterPrimitives";
import { FitDecision, fitIsServable } from "./FitDecision";
import { GpuSplitChoice } from "./GpuSplitChoice";
import { apiRequest } from "../lib/api";
import {
  GPU_MODEL_CATALOG,
  RECOMMENDED_CATALOG_ID,
  catalogModel,
  type GpuModelSpecFields,
} from "../lib/gpuCatalog";
import {
  GPU_MEMORY_UTILIZATION_DEFAULTS,
  INTENT_CAPACITY,
  INTENT_THROUGHPUT,
  WORKER_LED_LAN_NOTICE,
  type GpuClusterIntent,
} from "../lib/gpuServingMode";
import {
  CONTROLLER_PLACEMENT_ID,
  formatClusterLink,
  formatMemory,
  type FleetNode,
  type GpuFitDecision,
  type GpuServingChoices,
  type GpuSplitLink,
  type GpuSplitMode,
} from "./fleetTypes";
import { linkSummary as describeLink, readHostSettings, splitLinkLine, type HostSettings } from "../lib/hostSettings";
import type { Session } from "../types";
import { bytesIn } from "../lib/format";

/**
 * The GPU (vLLM) serve sub-form: pick a curated model (Easy) or paste a Hugging
 * Face link and set the geometry yourself (Advanced), say what clustering is
 * FOR, see the honest fit decision the moment anything changes, choose the GPU
 * nodes to serve on, and Serve — gated on a verdict that is actually a
 * placement.
 *
 * It owns the live fit call (`POST /cluster/fit`) because the fit is a
 * stateful, async read that the pure `FleetLlmModal` shell should not carry.
 * The one thing it hands upward is the finished deploy payload, through
 * `onServe`; `FleetCenter` queues the `cluster.llm.deploy` (mode `gpu`) job and
 * renders progress in the same operation modal every cluster change uses.
 *
 * **The typed deployment name is NOT owned here.** It lives in the parent's
 * `LlmForm` and arrives as a prop, because this sub-form is unmounted and
 * remounted whenever the Deployment-mode radios leave and re-enter "GPU
 * serving" — and a `useState("gpu-model")` seed re-ran on every one of those,
 * silently throwing away a name the owner had typed. A value the user typed
 * must outlive any remount of the control that collected it. For the same
 * reason the name is no longer folded into the *model spec* that keys the fit
 * request: a deployment name is not a model fact, and letting it key the fit
 * cancelled and re-issued a `/cluster/fit` POST on every keystroke.
 *
 * Every terminal state is truthful: no GPU node available, a fit still being
 * checked, a refusal (`wont_fit`, `single_refused`, `unsupported_intent`), a
 * spec the engine rejects, or a cluster it cannot reach each render a real
 * message, and Serve stays disabled until the engine returns a real
 * `single`/`distributed`/`replicated` placement and at least one node is
 * selected.
 *
 * **The fit is asked with the launch fraction (VD-129).** The GPU memory use
 * field rides in the fit request as `gpu_memory_utilization`, exactly as the
 * deploy sends it, because under the throughput intent the engine's fit rule
 * is per node at that fraction — a preview taken at one fraction and a deploy
 * launched at another would be two different questions (VD-B3b-1). Its
 * default follows the intent (0.80 for throughput, 0.90 for capacity) until
 * the operator types a value, which then outlives any intent change.
 */

type ServePath = "easy" | "advanced";

/**
 * What the *model spec* is called in a fit request on the Advanced path. It is
 * a constant rather than the typed deployment name: the two are different
 * facts, and coupling them made every keystroke in the name field cancel the
 * in-flight fit. Nothing renders it — `FitDecision` prints the engine's own
 * summary — so a fixed, honest label is enough.
 */
const CUSTOM_MODEL_LABEL = "Custom model";

/** How long the typed name must rest before the fit is asked again with it. */
const FIT_NAME_SETTLE_MS = 400;

/**
 * Why Serve is off when the engine answered with a verdict this console does
 * not know. A newer control plane can add one; the honest reading is that the
 * console is behind, not that the deployment details are incomplete.
 */
export const UNRECOGNISED_FIT_REASON =
  "Fit result not recognised; refresh or check for a console update.";

interface GpuServeFormProps {
  session: Session;
  /**
   * The valid serve targets: every GPU-capable, eligible participant. VD-125:
   * that includes the head controller when its own GPU was discovered — it is
   * an ordinary node here, identified by `CONTROLLER_PLACEMENT_ID` and given no
   * forced selection and no separate validation.
   */
  gpuNodes: FleetNode[];
  busy: boolean;
  /** The typed deployment name, owned by the parent so a remount cannot lose it. */
  name: string;
  onName: (value: string) => void;
  onServe: (payload: Record<string, unknown>) => void;
  onClose: () => void;
  /** Where a `single_refused` verdict sends the owner instead of clustering. */
  onServeAsAiChatModel?: () => void;
  /**
   * The dialog footer Cancel and Serve are portalled into (FleetLlmModal's
   * ClusterDialog). Null until that element mounts; omitted, they render inline.
   */
  footerSlot?: HTMLElement | null;
}


/** A positive number parsed from a text field, or null when blank/invalid. */
function positive(value: string): number | null {
  const parsed = Number(value.trim());
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

interface AdvancedGeometry {
  hidden_layers: string;
  attention_heads: string;
  kv_heads: string;
  head_dim: string;
  context_length: string;
  quantization: string;
  parameter_billions: string;
  weight_gb: string;
}

const EMPTY_GEOMETRY: AdvancedGeometry = {
  hidden_layers: "",
  attention_heads: "",
  kv_heads: "",
  head_dim: "",
  context_length: "",
  quantization: "",
  parameter_billions: "",
  weight_gb: "",
};

/**
 * What clustering is being asked for (VD-127 D10, VD-129). Capacity splits one
 * model that no single machine can hold; throughput runs one whole copy per
 * machine behind a balancer on this controller. What either cannot do for the
 * selection in hand is the *engine's* to say — through the preview's own
 * verdict and sentence, never through a control disabled here with a reason
 * the deploy would not repeat.
 *
 * The split is said as what it is FOR (owner decision 2026-09-30): capacity,
 * not speed. Measured across the pair, a split model answered a little slower
 * than one machine; the way to more speed is a copy on each machine.
 */
const INTENTS: ReadonlyArray<{
  id: GpuClusterIntent;
  title: string;
  detail: string;
}> = [
  {
    id: INTENT_CAPACITY,
    title: "Run a model bigger than one machine",
    detail: "Splits one model across the selected machines, so a model or a context too big for one machine can run. For capacity, not speed.",
  },
  {
    id: INTENT_THROUGHPUT,
    title: "Serve the same model with more throughput",
    detail: "Runs one copy of the model on each selected machine behind one endpoint, so more requests at once keep their response time. No answer gets faster.",
  },
];

/** Build the sizing spec `POST /cluster/fit` (and the link deploy path) needs
 *  from the advanced fields, or null when a required field is missing. The
 *  engine re-validates it, so a subtly bad geometry still surfaces as a real
 *  fit error rather than a silent guess. */
function advancedSpec(geometry: AdvancedGeometry): GpuModelSpecFields | null {
  const hidden_layers = positive(geometry.hidden_layers);
  const attention_heads = positive(geometry.attention_heads);
  const kv_heads = positive(geometry.kv_heads);
  const head_dim = positive(geometry.head_dim);
  const context_length = positive(geometry.context_length);
  if (!hidden_layers || !attention_heads || !kv_heads || !head_dim || !context_length) {
    return null;
  }
  const weightGib = positive(geometry.weight_gb);
  const parameter_billions = positive(geometry.parameter_billions) ?? 0;
  return {
    name: CUSTOM_MODEL_LABEL,
    weight_bytes: weightGib ? bytesIn(weightGib, "GiB") : 0,
    hidden_layers,
    attention_heads,
    kv_heads,
    head_dim,
    context_length,
    quantization: geometry.quantization.trim(),
    parameter_billions,
  };
}

/** Drop a zero weight so the engine estimates from parameters instead of being
 *  told the model weighs nothing. */
function fitBody(spec: GpuModelSpecFields): Record<string, unknown> {
  const { weight_bytes, ...rest } = spec;
  return weight_bytes > 0 ? { ...rest, weight_bytes } : { ...rest };
}

/**
 * Why the cluster link card cannot be chosen: the settings' own reason where
 * they give one (a chosen link that is gone, links that could not be read).
 */
function clusterLinkAbsent(link: HostSettings["cluster_link"] | null): string {
  if (!link) return "Link settings could not be read";
  if (!link.available) return link.reason || "Link settings could not be read";
  return link.chosen?.reason || "None chosen in Cluster > Setup";
}

export function GpuServeForm({
  session,
  gpuNodes,
  busy,
  name,
  onName,
  onServe,
  onClose,
  onServeAsAiChatModel,
  footerSlot,
}: GpuServeFormProps) {
  const [path, setPath] = useState<ServePath>("easy");
  // W4-D3: the fit preview is told the deployment's name, so a refusal names
  // it - settled rather than per keystroke, so typing does not cancel fit
  // after fit (the reason the name was once left out entirely).
  const [fitName, setFitName] = useState(name.trim());
  useEffect(() => {
    const timer = window.setTimeout(() => setFitName(name.trim()), FIT_NAME_SETTLE_MS);
    return () => window.clearTimeout(timer);
  }, [name]);
  const [catalogId, setCatalogId] = useState<string>(GPU_MODEL_CATALOG[0]?.id ?? "");
  const [link, setLink] = useState("");
  const [geometry, setGeometry] = useState<AdvancedGeometry>(EMPTY_GEOMETRY);
  // What clustering is for, and the Max context, open on the chosen curated
  // model's own layout where it has one (`gpuCatalog` `defaults`, the
  // backend's `catalog_defaults`): the recommended model opens as one copy
  // per machine at a 32,768-token context, which is the layout that was
  // decided for it. Both are null until the owner chooses, and a choice the
  // owner made is kept whatever model is picked afterwards.
  const [pickedIntent, setPickedIntent] = useState<GpuClusterIntent | null>(null);
  const [typedMaxModelLen, setTypedMaxModelLen] = useState<string | null>(null);
  const layout = path === "easy" ? catalogModel(catalogId)?.defaults : undefined;
  const intent: GpuClusterIntent = pickedIntent ?? layout?.intent ?? INTENT_CAPACITY;
  const maxModelLen = typedMaxModelLen ?? (layout ? String(layout.max_model_len) : "");
  const [port, setPort] = useState("8000");
  // Blank by default: each machine records, when it joins the cluster, which of
  // its network connections carries the address the cluster reaches it on, and
  // the deploy uses that. A typed value pins one connection across every
  // machine instead, for the rare setup where the recorded one is not wanted.
  const [iface, setIface] = useState("");
  // The owner's chosen cluster link (Machine settings), read once; null when
  // it could not be read, and the enrolled links are shown as before.
  const [clusterLink, setClusterLink] = useState<HostSettings["cluster_link"] | null>(null);
  useEffect(() => {
    let live = true;
    void Promise.resolve()
      .then(() => apiRequest<unknown>("/host-settings"))
      .then((value) => { if (live) setClusterLink(readHostSettings(value)?.cluster_link ?? null); })
      .catch(() => undefined);
    return () => { live = false; };
  }, []);
  // Null until the operator types: the field then shows the intent's own
  // default and follows an intent change; a typed value is theirs and stays.
  const [typedGpuMemUtil, setTypedGpuMemUtil] = useState<string | null>(null);
  const gpuMemUtil = typedGpuMemUtil ?? GPU_MEMORY_UTILIZATION_DEFAULTS[intent];
  // G3b scale-to-zero: an optional idle window typed in MINUTES. Blank keeps
  // the deployment always on; a value auto-unloads the GPU after that long
  // idle, and the next request reloads it warm.
  const [idleTimeout, setIdleTimeout] = useState("");
  // Owner decision 2026-09-29: a cluster model answers without thinking unless
  // this is ticked; offered only for a model whose template has the switch.
  const [thinking, setThinking] = useState(false);
  // The serving choices the fit offers for this model (`vllm_serve_options`):
  // the vLLM image (the default until the owner picks another), multi-token
  // prediction (off unless asked) and text-only (on for a model that reads
  // images). Each is sent only where the model allows it.
  // Each choice is kept WITH the model source it was made for, so a different
  // model never inherits the last one's image, prediction setting or offered
  // choices - not even for the one render before an effect could clear them.
  const [imagePick, setImagePick] = useState<{ source: string; key: string } | null>(null);
  const [mtpPick, setMtpPick] = useState<{ source: string; tokens: string } | null>(null);
  const [textOnlyPick, setTextOnlyPick] = useState<{ source: string; value: boolean } | null>(null);
  // The choices the last fit offered for a model. Kept while a new fit is in
  // flight or has failed, so the controls stay on screen and a choice the fit
  // refused can be taken back.
  const [servingFor, setServingFor] = useState<{ source: string; choices: GpuServingChoices } | null>(null);
  const [nodeIds, setNodeIds] = useState<string[]>([]);
  // VD-167 (owner 2026-10-01): how a model split across machines is split,
  // pipeline unless the owner chooses tensor-parallel, and the link it rides,
  // Ethernet unless the owner chooses the cluster link - for either mode. The
  // form never picks either for the owner, a recommendation included.
  const [splitMode, setSplitMode] = useState<GpuSplitMode>("pipeline");
  const [splitLinkChoice, setSplitLinkChoice] = useState<GpuSplitLink>("ethernet");
  // Whether the last answered fit split the model. Kept while a new fit is in
  // flight, so the split choices do not vanish under the owner's pointer.
  const [splitShown, setSplitShown] = useState(false);
  const formIds = useId().replaceAll(":", "");

  const [fit, setFit] = useState<GpuFitDecision | null>(null);
  const [fitError, setFitError] = useState("");
  const [fitLoading, setFitLoading] = useState(false);
  // B6: the question the fit on screen answers. Serve waits until it is the
  // question the form would ask now - a name still settling, or a fit asked
  // under an older key, is not this deployment's answer.
  const [fitFor, setFitFor] = useState("");

  // The sizing spec is the catalog model's geometry (Easy) or the assembled
  // advanced fields. Deliberately independent of the deployment name.
  const spec = useMemo<GpuModelSpecFields | null>(() => {
    if (path === "easy") return catalogModel(catalogId)?.spec ?? null;
    return advancedSpec(geometry);
  }, [path, catalogId, geometry]);

  // The link word the fit and the deploy still carry on the wire. It decides
  // nothing: a model split across machines is always a pipeline (owner
  // decision 2026-09-30). The form used to offer a box that, unticked, sent
  // "co-located" and made the engine split every layer across the machines —
  // the slow way, measured. The box is gone; the real link is shown below.
  const fleetLink = "cross-node";
  // The launch fraction as the wire carries it: absent when the field is blank
  // or not a number, so the backend applies its own per-intent default — the
  // same one the field shows — rather than being sent a zero. One value for
  // the fit and the deploy, so the two cannot be asked at different fractions.
  const launchUtilization = positive(gpuMemUtil);
  const utilizationField = launchUtilization
    ? { gpu_memory_utilization: launchUtilization }
    : {};
  // The launch context the same way: absent when blank, so the backend sizes
  // and launches at the model's own context — the field's hint — and one
  // value for the fit and the deploy, because the fit sizes the KV cache at
  // the context the deploy launches with (VD-129, one derivation).
  const launchContext = positive(maxModelLen);
  const contextField = launchContext ? { max_model_len: launchContext } : {};
  // G3b scale-to-zero: the idle window in MINUTES here, sent as seconds; absent
  // when blank so the deployment stays always-on. A value below the backend's
  // documented two-minute floor is clamped up there, not here.
  const idleMinutes = positive(idleTimeout);
  const idleField = idleMinutes ? { idle_timeout: Math.round(idleMinutes * 60) } : {};
  // The fit body the route reads: a `model` spec (the key `cluster_fit` looks
  // for), the link, the operator's `intent`, the launch fraction and context
  // (VD-129), and — so the preview matches the deploy (VD-B3b-1) — the
  // selected nodes. Sorting the ids keeps the key stable regardless of the
  // order they were picked in, and any selection, model, intent, fraction or
  // context change re-runs the fit.
  // The model source rides along so the fit can say whether this model has a
  // thinking switch (`vaelor/model_thinking`), about the repo the deploy serves.
  const modelSource = path === "easy" ? catalogId : link.trim();
  const serving = servingFor?.source === modelSource ? servingFor.choices : null;
  const mtpTokens = mtpPick?.source === modelSource ? mtpPick.tokens : "0";
  const vllmImage = imagePick?.source === modelSource ? imagePick.key : null;
  // Text-only is on for a model that reads images unless ticked off for THIS model.
  const textOnly = textOnlyPick?.source === modelSource ? textOnlyPick.value : true;
  // Multi-token prediction changes what one copy needs (its draft layer and
  // state), so the preview is asked with it exactly as the deploy sends it -
  // under ONE gate: the model offers it, and each machine serves its own copy.
  const draftTokens = Number(mtpTokens) > 0 ? Number(mtpTokens) : 0;
  const mtpOffered = Boolean(serving?.mtp.available) && intent === INTENT_THROUGHPUT;
  const mtpField = mtpOffered && draftTokens ? { mtp_tokens: draftTokens } : {};
  // The image rides in the fit too, when it is not the default: the image
  // decides how a hybrid model's cache is kept, so the preview must size the
  // cache of the image the deploy will launch.
  const imageField = serving && vllmImage && vllmImage !== serving.default_image
    ? { vllm_image: vllmImage } : {};
  // The split's choices exist only while the preview splits the model: a
  // choice the owner can no longer see is never sent (review). Nothing chosen
  // is a pipeline, so `split_mode` is sent only for tensor.
  const showSplit = intent === INTENT_CAPACITY && splitShown;
  // The cluster link is offered when Cluster > Setup holds a usable one, and
  // a cluster-link choice without one is neither drawn nor sent as chosen.
  const chosenLink = clusterLink?.available && clusterLink.chosen?.valid ? clusterLink.chosen : null;
  const effectiveLink: GpuSplitLink = splitLinkChoice === "cluster-link" && !chosenLink ? "ethernet" : splitLinkChoice;
  const splitFields = showSplit
    ? { split_link: effectiveLink, ...(splitMode === "tensor" ? { split_mode: splitMode } : {}) }
    : {};
  // The cluster link sets each machine's interface for the split, so a pin
  // beside it is neither offered nor sent (the deploy refuses the pair, and
  // the preview, given the pin, refuses it in the same words).
  const pinBlocked = showSplit && effectiveLink === "cluster-link";
  const pinField = !pinBlocked && iface.trim() ? { interface: iface.trim() } : {};
  const fitKey = spec
    ? JSON.stringify({
      model: fitBody(spec),
      ...(modelSource ? { model_source: modelSource } : {}),
      ...(fitName ? { name: fitName } : {}),
      link: fleetLink,
      intent,
      ...splitFields,
      ...pinField,
      ...utilizationField,
      ...contextField,
      ...mtpField,
      ...imageField,
      node_ids: [...nodeIds].sort(),
    })
    : "";

  const nodeCount = gpuNodes.length;
  useEffect(() => {
    // No spec, or nowhere to serve, means no fit question to ask.
    if (!fitKey || !nodeCount) {
      setFit(null);
      setFitError("");
      setFitLoading(false);
      setSplitShown(false);
      return;
    }
    let cancelled = false;
    // Clear the previous model's verdict up front so a stale "fits" can never
    // gate Serve while the new fit is still in flight (VD-B3b-3).
    setFit(null);
    setFitLoading(true);
    setFitError("");
    void apiRequest<GpuFitDecision>(
      "/cluster/fit",
      { method: "POST", body: fitKey },
      session.csrf_token,
    )
      .then((decision) => {
        if (cancelled) return;
        setFit(decision);
        setFitFor(fitKey);
        const split = decision.verdict === "distributed";
        setSplitShown(split);
        // A model that stops splitting forgets the tensor choice, so one that
        // splits again opens on the default rather than a hidden Tensor.
        if (!split) setSplitMode("pipeline");
        setServingFor(decision.serving ? { source: modelSource, choices: decision.serving } : null);
        setFitLoading(false);
      })
      .catch((error) => {
        if (cancelled) return;
        setFit(null);
        setFitError(error instanceof Error ? error.message : "The fit decision is unavailable.");
        setFitLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // `modelSource` rides in `fitKey`, so the choices are filed under the
    // model this fit was asked about.
  }, [fitKey, nodeCount, session.csrf_token]);

  const toggleNode = (id: string) => {
    setNodeIds((current) =>
      current.includes(id) ? current.filter((entry) => entry !== id) : [...current, id]);
  };

  const chosenImage = vllmImage ?? serving?.default_image ?? null;
  const imageNote = serving?.images.find((image) => image.key === chosenImage)?.kernel_note ?? "";
  // Multi-token prediction is offered on one copy per machine only; the
  // backend refuses it across a model split, so the form says so first.
  const mtpReason = !serving
    ? "Waiting for the fit decision."
    : !serving.mtp.available
      ? serving.mtp.detail
      : intent !== INTENT_THROUGHPUT
        ? "Offered only when each machine serves its own copy."
        : undefined;
  const servingFields = {
    ...imageField,
    ...mtpField,
    ...(serving?.text_only.applies ? { text_only: textOnly } : {}),
  };
  const sourceReady = path === "easy" ? Boolean(catalogId) : link.trim().length > 0;
  const fits = fitIsServable(fit?.verdict);
  // D2: the controller always leads when it participates, which is what makes
  // the model API loopback-only behind the LLM Server's keyed proxy — and what
  // makes this a MODE SWITCH rather than one more deployment.
  const controllerSelected = nodeIds.includes(CONTROLLER_PLACEMENT_ID);
  // VD-129: a throughput selection without this controller is refused by the
  // engine (`unsupported_intent`, its own sentence) before any endpoint could
  // exist, so the LAN notice would warn about a deployment that cannot happen.
  // The refusal is rendered above; the notice is the capacity intent's alone.
  const workerLed = intent !== INTENT_THROUGHPUT && nodeIds.length > 0 && !controllerSelected;
  // VD-125: the REAL link each selected machine brings, derived from its own
  // captured fact — never the static "2.5G / Wi-Fi" the checkbox used to claim.
  // A machine whose speed or medium could not be read says so honestly, and one
  // whose fact predates the capture reads "link speed unknown".
  const selectedNodes = useMemo(
    () => gpuNodes.filter((node) => nodeIds.includes(node.id)),
    [gpuNodes, nodeIds],
  );
  // A split's traffic goes over the cluster link when the owner chooses it in
  // this form (ACC-208, VD-167); otherwise over each machine's enrolled link.
  const splitLink = showSplit && selectedNodes.length && effectiveLink === "cluster-link"
    ? splitLinkLine(clusterLink, selectedNodes) : null;
  const linkSummary = splitLink ?? selectedNodes
    .map((node) => `${node.name} ${formatClusterLink(node.inventory.cluster_interface)}`)
    .join(" · ");
  // What the deploy would refuse before starting (VD-168): no Serve while present.
  const refusal = fit?.refusal?.message ?? "";
  const fitPending = fitLoading || fitName !== name.trim() || fitFor !== fitKey;
  const canServe =
    !busy
    && sourceReady
    && Boolean(name.trim())
    && nodeIds.length >= 1
    && !fitPending
    && fits
    && !refusal
    && (positive(port) ?? 0) > 0;
  const refusalId = `${formIds}-refusal`;
  const splitReason = fit?.verdict === "distributed" && fit.placement && "parallelism_reason" in fit.placement
    ? fit.placement.parallelism_reason : "";

  const submit = () => {
    if (!canServe || !spec) return;
    onServe({
      node_ids: nodeIds,
      name: name.trim(),
      model_source: path === "easy" ? catalogId : link.trim(),
      // A pasted link's geometry rides as `model_spec`; strip a zero weight so
      // the backend estimates from parameters rather than rejecting a
      // weight_bytes of 0 (`build_model_spec` requires a positive weight).
      ...(path === "advanced" ? { model_spec: fitBody(spec) } : {}),
      port: Number(port),
      ...pinField,
      // The fraction and the context the preview above was asked with (VD-129).
      ...utilizationField,
      ...contextField,
      ...idleField,
      // Sent only where the model has the switch; the backend defaults it off.
      ...(fit?.thinking?.switch ? { thinking } : {}),
      // The image, multi-token prediction and text-only, where offered.
      ...servingFields,
      ...splitFields,
      link: fleetLink,
      // The same word the preview asked with, so `plan_gpu_fit` cannot reach a
      // different verdict for the deploy than the one shown above the button.
      intent,
      // `use_for_assistant` is deliberately absent (VD-127 D4): the GPU path
      // refuses it up front, because a teardown's credential cascade would
      // leave the on-device Assistant unassigned for good. Sending it would be
      // asking for a refusal after the model has already been pulled.
    });
  };

  // Cancel and Serve sit in the dialog's fixed footer (ClusterDialog): the
  // modal hands its footer element down, and they are portalled into it. A
  // form rendered on its own (no slot given) keeps them inline beneath it.
  const placeFooter = (actions: ReactNode) => footerSlot === undefined
    ? <div className="cl-actions cl-actions--end">{actions}</div>
    : footerSlot ? createPortal(actions, footerSlot) : null;

  if (!gpuNodes.length) {
    return (
      <div className="cl-stack gpu-serve">
        <p role="status">
          No GPU-capable node is available in the fleet, so a vLLM model cannot
          be served yet. This controller counts when its own GPU is discovered;
          otherwise enrol a worker with a discovered GPU, then try again.
        </p>
        {placeFooter(<Button variant="secondary" onClick={onClose}>Close</Button>)}
      </div>
    );
  }

  return (
    <div className="cl-stack gpu-serve">
      <SegmentedControl
        label="How to choose the model"
        onChange={setPath}
        options={[
          { value: "easy", label: "Easy — pick a curated model" },
          { value: "advanced", label: "Advanced — paste a Hugging Face link" },
        ]}
        value={path}
      />

      {path === "easy" ? (
        <Select
          label="Curated model"
          value={catalogId}
          onChange={(event) => setCatalogId(event.target.value)}
        >
          {GPU_MODEL_CATALOG.map((model) => (
            <option key={model.id} value={model.id}>
              {model.spec.name} · {model.repo}
              {model.id === RECOMMENDED_CATALOG_ID ? " (recommended)" : ""}
            </option>
          ))}
        </Select>
      ) : (
        <>
          <Input
            label="Hugging Face link or org/name"
            placeholder="https://huggingface.co/org/model  or  org/model"
            value={link}
            onChange={(event) => setLink(event.target.value)}
          />
          <div className="cd-grid">
            <Input label="Hidden layers" inputMode="numeric" value={geometry.hidden_layers}
              onChange={(event) => setGeometry({ ...geometry, hidden_layers: event.target.value })} />
            <Input label="Attention heads" inputMode="numeric" value={geometry.attention_heads}
              onChange={(event) => setGeometry({ ...geometry, attention_heads: event.target.value })} />
            <Input label="KV heads" inputMode="numeric" value={geometry.kv_heads}
              onChange={(event) => setGeometry({ ...geometry, kv_heads: event.target.value })} />
            <Input label="Head dimension" inputMode="numeric" value={geometry.head_dim}
              onChange={(event) => setGeometry({ ...geometry, head_dim: event.target.value })} />
            <Input label="Context length" inputMode="numeric" value={geometry.context_length}
              onChange={(event) => setGeometry({ ...geometry, context_length: event.target.value })} />
            <Input label="Quantization" placeholder="f16, q4_k_m…" value={geometry.quantization}
              onChange={(event) => setGeometry({ ...geometry, quantization: event.target.value })} />
            <Input label="Parameters (billions)" inputMode="decimal" value={geometry.parameter_billions}
              onChange={(event) => setGeometry({ ...geometry, parameter_billions: event.target.value })} />
            <Input label="Weight size (GiB, optional)" inputMode="decimal" value={geometry.weight_gb}
              hint="Leave blank to estimate from parameters and quantization."
              onChange={(event) => setGeometry({ ...geometry, weight_gb: event.target.value })} />
          </div>
        </>
      )}

      <fieldset className="cd-group gpu-serve__intent">
        <legend className="cd-label">What is clustering for?</legend>
        <div className="cd-options">
          {INTENTS.map((option) => (
            <OptionCard key={option.id} checked={intent === option.id} detail={option.detail}
              name="gpu-cluster-intent" onChange={() => setPickedIntent(option.id)}
              title={option.title} type="radio" value={option.id} />
          ))}
        </div>
      </fieldset>

      {/* The fit decision, and - while it splits the model - the split's own
          choices, in one box (the board's fit panel). */}
      <div className="cd-box">
        <div aria-live="polite" className="cd-fit">
          {fitLoading && <p className="cd-note">Checking whether this model fits…</p>}
          {!fitLoading && fitError && (
            <p className="cd-bad-text" role="alert">{fitError}</p>
          )}
          {!fitLoading && !fitError && fit && (
            <FitDecision decision={fit} onServeAsAiChatModel={onServeAsAiChatModel} refusalId={refusalId} />
          )}
          {!fitLoading && !fitError && !fit && (
            <p className="cd-note">
              {path === "advanced"
                ? "Enter the model geometry to see whether it fits."
                : "Choose a model to see whether it fits."}
            </p>
          )}
        </div>
        {showSplit && (
          <GpuSplitChoice
            mode={splitMode} onMode={setSplitMode}
            link={effectiveLink} onLink={setSplitLinkChoice}
            reason={splitReason}
            recommendation={splitMode === "tensor" ? fit?.link_recommendation ?? null : null}
            clusterLink={chosenLink
              ? describeLink({ name: chosenLink.name, kind: chosenLink.kind ?? "unknown", speed_mbps: chosenLink.speed_mbps ?? null })
              : null}
            clusterLinkAbsent={clusterLinkAbsent(clusterLink)}
            summary={linkSummary ? `Link between machines: ${linkSummary}` : ""}
          />
        )}
      </div>

      <fieldset className="cd-group">
        <legend className="cd-label">GPU nodes · select the machines to serve on; Vaelor works out how the model is placed on them.</legend>
        <div className="cd-options">
          {gpuNodes.map((node) => (
            <OptionCard key={node.id} checked={nodeIds.includes(node.id)}
              detail={`${formatMemory(node.inventory.memory_bytes)} memory · ${node.inventory.architecture ?? "Unknown architecture"}`}
              onChange={() => toggleNode(node.id)} title={node.name} type="checkbox" />
          ))}
        </div>
        <output aria-live="polite" className="cd-note">{nodeIds.length} of {gpuNodes.length} GPU node(s) selected</output>
      </fieldset>

      <div className="cd-grid gpu-serve__tuning">
        <Input label="Deployment name" value={name} onChange={(event) => onName(event.target.value)} />
        <Input label="API port" inputMode="numeric" value={port} onChange={(event) => setPort(event.target.value)} />
        <Input label="Cluster interface (optional)" value={iface}
          disabled={pinBlocked}
          disabledReason={pinBlocked ? "Not used while the split rides the cluster link, which sets each machine's interface." : undefined}
          hint={pinBlocked ? undefined : "Leave blank to use the connection each machine recorded when it joined."}
          onChange={(event) => setIface(event.target.value)} />
        <Input label="GPU memory use (0.10–0.95)" inputMode="decimal" value={gpuMemUtil}
          hint={`Default ${GPU_MEMORY_UTILIZATION_DEFAULTS[INTENT_THROUGHPUT]} with a copy on each machine, ${GPU_MEMORY_UTILIZATION_DEFAULTS[INTENT_CAPACITY]} when one model is split. Set at most 0.95: vLLM's start-up briefly holds more than its share.`}
          onChange={(event) => setTypedGpuMemUtil(event.target.value)} />
        <Input label="Max context (optional)" inputMode="numeric" value={maxModelLen}
          hint="Defaults to the model's full context length."
          onChange={(event) => setTypedMaxModelLen(event.target.value)} />
        <Input label="Auto-unload when idle (minutes, optional)" inputMode="numeric" value={idleTimeout}
          hint="Reclaim the GPU after this many idle minutes; the next request reloads it warm. Blank keeps it always on (minimum 2)."
          onChange={(event) => setIdleTimeout(event.target.value)} />
      </div>
      {/* The REAL per-node link, derived from the selected machines' own
          captured facts (VD-125); honest "unknown" when a fact can't be read. */}
      {linkSummary && !showSplit && (
        <p className="cd-note" role="status">Link between machines: {linkSummary}</p>
      )}
      {serving && (
        <div className="cd-grid">
          <Select label="vLLM version" value={chosenImage ?? ""}
            hint="A version a machine does not have yet is downloaded first (about 27 GB). A deployment keeps the version it was made with."
            onChange={(event) => setImagePick({ source: modelSource, key: event.target.value })}>
            {serving.images.map((image) => (
              <option key={image.key} value={image.key}>{image.label}</option>
            ))}
          </Select>
          <Select label="Multi-token prediction" value={serving.mtp.available ? mtpTokens : "0"}
            disabled={Boolean(mtpReason)}
            disabledReason={mtpReason}
            hint={mtpReason ? undefined : serving.mtp.detail}
            onChange={(event) => setMtpPick({ source: modelSource, tokens: event.target.value })}>
            <option value="0">Off</option>
            {Array.from({ length: serving.mtp.max_tokens }, (_, index) => index + 1).map((count) => (
              <option key={count} value={String(count)}>
                {count} extra {count === 1 ? "word" : "words"} per step
                {count === serving.mtp.suggested_tokens ? " (suggested)" : ""}
              </option>
            ))}
          </Select>
        </div>
      )}
      {imageNote && <p className="cd-note" role="status">{imageNote}</p>}
      {/* Why a hybrid model will be started without its cache settings,
          when the fit could not read a fact they are decided from. */}
      {serving?.cache_note && <p className="cd-note">{serving.cache_note}</p>}
      {serving?.text_only.applies && (
        <OptionCard checked={textOnly} detail={serving.text_only.detail}
          onChange={(event) => setTextOnlyPick({ source: modelSource, value: event.target.checked })}
          title="Serve text only (skip the image reader)" type="checkbox" />
      )}
      {/* Thinking is offered only for a model whose template has the switch.
          A model without one (the recommended Instruct model among them) is
          shown no control at all, only the plain sentence that says so: a
          ticked box there would promise a behaviour the model cannot give. */}
      {fit?.thinking?.switch && (
        <OptionCard checked={thinking} detail={fit.thinking.detail}
          onChange={(event) => setThinking(event.target.checked)}
          title="Let the model think before answering — slower, sometimes better" type="checkbox" />
      )}
      {fit?.thinking && !fit.thinking.switch && <p className="cd-note">{fit.thinking.detail}</p>}

      {/* The consequences of the selection, said before the button rather than
          discovered after it. W4-D4: the mode switch only when the fit says
          THIS deploy takes it, in the backend's words. */}
      {fit?.mode_switch && (
        <Notice severity="warning" heading={fit.mode_switch.heading}>
          {fit.mode_switch.message}
        </Notice>
      )}
      {workerLed && (
        <Notice severity="info" heading="A worker leads this cluster">
          {WORKER_LED_LAN_NOTICE}
        </Notice>
      )}

      {placeFooter(
        <>
          <Button variant="secondary" onClick={onClose}>Cancel</Button>
          <Button
            variant="primary"
            disabled={!canServe}
            // A refusal is shown once, in the fit box; the button points at it.
            aria-describedby={refusal ? refusalId : undefined}
            disabledReason={canServe ? undefined : serveBlockedReason()}
            onClick={submit}
          >
            Serve on GPU
          </Button>
        </>,
      )}
    </div>
  );

  /** Why Serve is off: the first unmet gate, in order. */
  function serveBlockedReason(): string {
    if (!sourceReady) return "Choose a curated model or paste a Hugging Face link.";
    if (!nodeIds.length) return "Select at least one GPU node.";
    if (fitPending && !fitError) return "Waiting for the fit decision.";
    // The deploy's own refusal (VD-168) is in the fit box.
    if (refusal) return "See why above.";
    if (fit?.verdict === "wont_fit") return "This model won't fit on the fleet's GPUs.";
    if (fit?.verdict === "single_refused") return "This model fits one machine, so serve it there as the AI Chat model.";
    // The engine's own sentence for what the intent cannot do here (VD-129:
    // "select the controller too"), so the button and the pill agree.
    if (fit?.verdict === "unsupported_intent") return fit.summary;
    // Every refusal the engine can send is named above, so a fit that is
    // present yet not servable is a verdict this console has no words for.
    // `fitIsServable` fails closed on it; the reason must say so.
    if (fit && !fits) return UNRECOGNISED_FIT_REASON;
    if (!fit) return "A fit decision is needed before serving.";
    return "Complete the deployment details.";
  }
}
