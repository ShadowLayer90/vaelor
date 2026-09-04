import type { AccelerationReading, ContextReading } from "../lib/acceleration";

export interface WorkloadCapabilities {
  docker: {
    installed: boolean;
    compose: boolean;
    compose_version: string | null;
    // Whether Docker's storage area is present. Absent (older backend, or the
    // fallback capabilities dict), "ok", or "unknown" are all treated as
    // healthy by the UI; only "storage_missing" — a data-root confirmed gone —
    // means a container cannot start even though Docker is installed. See
    // docker_health.py.
    runtime?: string;
    runtime_reason?: string;
    installation?: { available: boolean; method: string | null; reason: string };
  };
  os?: {
    id: string;
    name: string;
    support_level: "verified" | "compatible" | "limited" | "unknown";
    support_label: string;
  };
  runtime?: { architecture?: string };
  roots?: { workloads?: string; models?: string };
  job_types: string[];
}

export interface ApplicationFeatures {
  research: boolean;
  drafts: boolean;
  deploy: boolean;
}

export interface WorkloadJob {
  id: string;
  type: string;
  state: string;
  operation_state?: string;
  attention?: boolean;
  retryable?: boolean;
  readiness?: string;
  liveness?: string;
  resource_liveness?: string;
  phase?: string;
  progress: number;
  created_at: number;
  message: string;
  attempt: number;
  retry_of?: string | null;
  payload?: Record<string, unknown>;
  result?: {
    candidates?: Array<{
      id: string;
      gated: boolean;
      files: Array<{
        name: string;
        size_bytes: number;
        fits_hardware?: boolean;
        fit_reason?: string;
      }>;
    }>;
    path?: string;
    file?: string;
    size_bytes?: number;
    endpoint?: string;
    matching_files?: number;
    /**
     * What the model server that just started actually got.
     *
     * Both readings are taken after `/health` answers, because the failures
     * they catch are invisible to it: llama.cpp loads the CPU backend when it
     * cannot resolve its accelerator library, and caps a slot context above
     * the model's training context — starting, answering, and saying so only
     * in its own log. Until now they arrived as a progress message at 95%
     * that scrolled away.
     */
    acceleration?: AccelerationReading | null;
    context_built?: ContextReading | null;
    /**
     * GPU AI-Chat tier descriptors, set only when a `model.deploy` went down
     * the GPU fork (`surface: "ai-chat"` on a capable box). `optimized` is true
     * ONLY for the recommended ROCmFP4 build; `recipe` names the fork recipe;
     * `fit_mode`/`fit_reason` say whether every layer, some, or none reached the
     * GPU, with a ready-to-show sentence. Absent on a plain Assistant deploy.
     */
    optimized?: boolean;
    recipe?: "rocmfp4" | "generic";
    fit_mode?: "gpu" | "partial" | "cpu";
    fit_reason?: string;
  };
}

export interface AgentStatus {
  name: string;
  configured: boolean;
  provider: string;
  model?: string | null;
  approval_required: boolean;
  tools: string[];
}

export interface AgentPlan {
  agent: string;
  source: string;
  summary: string;
  rationale: string;
  warnings: string[];
  checklist: string[];
  proposed_job: { type: string; payload: Record<string, unknown> } | null;
  approval_required: boolean;
  conversation_id?: string;
  application_intent?: {
    type: "application.deploy";
    application_query: string;
  } | null;
}

export type InstallFlow = "catalog" | "model" | "copilot" | "custom" | "planner" | "researched" | null;
