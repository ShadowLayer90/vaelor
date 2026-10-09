/**
 * The curated GPU serving catalog, mirrored for the Easy serve picker.
 *
 * The backend's `GPU_MODEL_CATALOG` (`vaelor/gpu_model_catalog.py`) is the
 * source of truth: an Easy serve submits a catalog *id* as `model_source`, and
 * the backend resolves the weights repo and sizing geometry from that id. But
 * `POST /cluster/fit` sizes a *spec dict* — it does not resolve catalog ids — so
 * the Easy path must send the same geometry the backend catalog holds to get a
 * fit decision. This module carries that geometry, byte-for-byte with
 * `gpu_model_catalog.GPU_MODEL_CATALOG`, so the picker label, the fit spec,
 * and the deployed model never disagree. A drift here is a wrong fit preview,
 * so the values are held identical to the Python catalog by
 * `tests/test_gpu_catalog_frontend_parity.py`.
 */

/** The transformer geometry `build_model_spec` validates, plus the display
 *  name. These are exactly the `_SPEC_FIELDS` a catalog entry contributes. */
export interface GpuModelSpecFields {
  name: string;
  weight_bytes: number;
  hidden_layers: number;
  attention_heads: number;
  kv_heads: number;
  head_dim: number;
  context_length: number;
  quantization: string;
  parameter_billions: number;
}

export interface GpuCatalogModel {
  /** The catalog id submitted as `model_source` for an Easy serve. */
  id: string;
  /** The Hugging Face weights repo the backend pulls (shown for context). */
  repo: string;
  /** The sizing geometry sent to `POST /cluster/fit`. */
  spec: GpuModelSpecFields;
  /**
   * The layout the deploy form opens on for this entry, where one was decided
   * for it (the backend's `catalog_defaults`): what clustering is for, and
   * the Max context. An entry without it leaves the form's own defaults.
   */
  defaults?: { intent: "capacity" | "throughput"; max_model_len: number };
}

/**
 * The preset the picker opens on and marks as recommended: the backend's
 * `RECOMMENDED_CATALOG_ID` (owner decision 2026-09-30, from the vLLM 0.27
 * benchmark on the pair). It is the first entry, as it is in the backend
 * catalog; `tests/test_gpu_catalog_frontend_parity.py` holds both.
 */
export const RECOMMENDED_CATALOG_ID = "qwen3_30b_a3b_instruct_2507_w4a16";

export const GPU_MODEL_CATALOG: readonly GpuCatalogModel[] = [
  {
    id: "qwen3_30b_a3b_instruct_2507_w4a16",
    repo: "cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
    spec: {
      name: "Qwen 3 30B-A3B Instruct (4-bit)",
      weight_bytes: 18_094_507_352,
      hidden_layers: 48,
      attention_heads: 32,
      kv_heads: 4,
      head_dim: 128,
      context_length: 262144,
      quantization: "w4a16",
      parameter_billions: 30.5,
    },
    defaults: { intent: "throughput", max_model_len: 32768 },
  },
  {
    id: "qwen3_8b_fp16",
    repo: "Qwen/Qwen3-8B",
    spec: {
      name: "Qwen 3 8B (FP16)",
      weight_bytes: 16_381_516_776,
      hidden_layers: 36,
      attention_heads: 32,
      kv_heads: 8,
      head_dim: 128,
      context_length: 32768,
      quantization: "f16",
      parameter_billions: 8.2,
    },
  },
  {
    id: "qwen3_14b_bf16",
    repo: "Qwen/Qwen3-14B",
    spec: {
      name: "Qwen 3 14B (BF16)",
      weight_bytes: 29_536_665_640,
      hidden_layers: 40,
      attention_heads: 40,
      kv_heads: 8,
      head_dim: 128,
      context_length: 40960,
      quantization: "bf16",
      parameter_billions: 14.8,
    },
  },
  {
    id: "qwen3_32b_fp8",
    repo: "Qwen/Qwen3-32B-FP8",
    spec: {
      name: "Qwen 3 32B (FP8)",
      weight_bytes: 34_322_567_640,
      hidden_layers: 64,
      attention_heads: 64,
      kv_heads: 8,
      head_dim: 128,
      context_length: 40960,
      quantization: "fp8",
      parameter_billions: 32.8,
    },
  },
  {
    id: "qwen3_32b_fp16",
    repo: "Qwen/Qwen3-32B",
    spec: {
      name: "Qwen 3 32B (FP16)",
      weight_bytes: 65_524_328_560,
      hidden_layers: 64,
      attention_heads: 64,
      kv_heads: 8,
      head_dim: 128,
      context_length: 32768,
      quantization: "f16",
      parameter_billions: 32.5,
    },
  },
  {
    id: "mistral_small_24b_bf16",
    repo: "unsloth/Mistral-Small-24B-Instruct-2501",
    spec: {
      name: "Mistral Small 24B (BF16)",
      weight_bytes: 47_144_848_872,
      hidden_layers: 40,
      attention_heads: 32,
      kv_heads: 8,
      head_dim: 128,
      context_length: 32768,
      quantization: "bf16",
      parameter_billions: 23.6,
    },
  },
];

export function catalogModel(id: string): GpuCatalogModel | undefined {
  return GPU_MODEL_CATALOG.find((model) => model.id === id);
}
