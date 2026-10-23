import type { PerformanceSummary } from "../lib/performanceLine";
import type { AnswerCutOff } from "../lib/aiChatExport";
import type { AiChatWebSearchRecord } from "./AiChatWebSearch";

export interface AiChatConnection {
  id: string;
  label: string;
  provider: string;
  /** The service's name for display ("Anthropic"), from `describe_connections`. */
  provider_label?: string;
  active_for: string[];
  selected_model?: string;
  /**
   * Whether this endpoint is served by this appliance at all; `null` when the
   * record carries no address to judge by (`local_source: "unknown"`).
   */
  local?: boolean | null;
  /**
   * How that was decided, from `vaelor.chat_connections.connection_locality`:
   * `vaelor-managed` (a credential Vaelor minted for its own deploy),
   * `loopback-address` (on this machine, but not one Vaelor deployed), or
   * `remote`, `vaelor-cluster` (the GPU cluster Vaelor deployed: prompts go to
   * its machines on the owner's network, `local` stays null), or `unknown`.
   * Only the first proves the appliance is serving the model itself.
   */
  local_source?: "vaelor-managed" | "loopback-address" | "remote" | "vaelor-cluster" | "unknown" | string;
  /** The one-sentence reason behind `local`, written by the backend. */
  local_reason?: string;
}

export interface AiChatCollection {
  id: string;
  name: string;
  description: string;
  document_count: number;
  chunk_count: number;
  size_bytes: number;
}

export interface AiChatDocument {
  id: string;
  collection_id: string;
  name: string;
  media_type: string;
  size_bytes: number;
  chunk_count: number;
  created_at: number;
}

export interface AiChatCitation {
  id: string;
  collection: string;
  document: string;
  chunk: number;
  excerpt: string;
  /** A web page's address, on a citation Web search supplied (VD-221). */
  url?: string;
}

export interface AiChatAgentGrantOperation {
  id: string;
  name: string;
  access: "read" | "write";
  risk: string;
}

export interface AiChatAgentGrant {
  grant_id: string;
  app_instance_id: string;
  app_name: string;
  manifest_version: string;
  manifest_digest: string;
  operations: AiChatAgentGrantOperation[];
}

export interface AiChatAgentProposal {
  profile_id: string;
  profile_name: string;
  profile_version: number;
  task: string;
  capabilities: string[];
  integrations: string[];
  app_grants: AiChatAgentGrant[];
  approval_required: true;
}

export interface AiChatAgent {
  id: string;
  name: string;
  description: string;
  custom?: boolean;
  version?: number;
  enabled?: boolean;
  operational?: boolean;
}

export interface AiChatMessage {
  id?: number;
  role: "user" | "assistant";
  content: string;
  citations?: AiChatCitation[];
  created_at?: number;
  metadata?: {
    source?: string;
    approval_required?: boolean;
    proposed_agent_task?: AiChatAgentProposal;
    /** Compact per-answer timing (total, TTFT, prefill/decode tok/s). */
    performance?: PerformanceSummary;
    /** A stored failure notice, not an answer (`rag_chat.MESSAGE_METADATA_KEYS`). */
    failed?: boolean;
    /** The model the failed request went to. */
    failed_model?: string;
    /** The thinking summary the provider returned (VD-209 item 4), if any. */
    thinking?: AiChatThinking;
    /**
     * VD-221: the web search behind this answer, or - on a reply Vaelor wrote
     * instead of a guess - which state Web search was in ("off", "unavailable").
     */
    web_search?: AiChatWebSearchRecord | string;
    /**
     * VD-206: why this answer stopped before its end - the model's
     * answer-length limit, or the 12,000 characters AI Chat keeps
     * (`ANSWER_CUT_OFF_CAUSES`). Absent on a finished answer.
     */
    cut_off?: AnswerCutOff;
  };
}

/**
 * VD-209: the four steps of AI Chat's thinking control. The backend's own
 * words (`vaelor.chat_thinking.THINKING_STEPS`), copied deliberately; if they
 * ever differ, the Python list is right.
 */
export type AiChatThinkingStep = "off" | "low" | "medium" | "high";

/** A turn's thinking: model output, shown as plain text and never as HTML. */
export interface AiChatThinking {
  summary: string;
  /** Measured from the provider's stream; absent when nothing could be measured. */
  seconds?: number;
  step?: AiChatThinkingStep;
  /** The summary was longer than AI Chat keeps (12,000 characters). */
  truncated?: boolean;
}

/** `GET /ai-chat/thinking`: whether the chosen model can think, and how. */
export interface AiChatThinkingView {
  credential_id: string;
  model: string;
  /** False hides the control: the model does not think, or nothing could say. */
  available: boolean;
  /** The steps this model takes, in order; a model that cannot stop thinking has no "off". */
  steps: AiChatThinkingStep[];
  /** The step a send carries: the remembered one if the model takes it, else the default. */
  step: AiChatThinkingStep;
  /** Whether the provider returns a readable summary on AI Chat's path. */
  summary: boolean;
  /** Why the control is hidden when that is "could not read", else "". */
  reason: string;
}

export interface AiChatConversation {
  id: string;
  title: string;
  /** The model that last answered in this chat; "" when none has. */
  model: string;
  collections: string[];
  updated_at: number;
  archived: boolean;
  /** VD-221: whether this chat's Web search toggle was left on. Older chats: absent, off. */
  web_search?: boolean;
}

/** VD-210: what clustering means for the picker - which row is the cluster, which are held. */
export interface AiChatClustering {
  /** Connection id to the backend's sentence for why AI Chat cannot take it now. */
  refusals: Record<string, string>;
  /** The cluster's own connection, grouped under "Cluster"; "" when not clustered. */
  clusterId: string;
}

export interface AiChatSetup {
  connections: AiChatConnection[];
  active_connection: AiChatConnection | null;
  /**
   * The backend's one sentence for why the connection cannot be changed right
   * now (a GPU cluster holds it, VD-127), or "" / absent when it can. The
   * picker disables itself on it before a click; the activate route answers
   * the same sentence if one lands anyway.
   */
  assignment_refusal?: string;
  /**
   * VD-210: the connections AI Chat cannot take right now, each with the
   * backend's sentence - this machine's own GPU model while a cluster holds
   * the GPU. Every other connection stays choosable.
   */
  connection_refusals?: Record<string, string>;
  /** The connection the GPU cluster serves through (the mode file's), or "". */
  cluster_credential_id?: string;
  preference: { model: string; collection_ids: string[] };
  collections: AiChatCollection[];
  limits: { document_bytes: number; collections: number };
  retrieval: string;
  /**
   * VD-220: the custom-agent runner's own verdict on AI Chat's model
   * (`ai_chat_model_lease.custom_agent_readiness`): its state, whether it
   * blocks New agent, runs and schedules, its sentence, and the model's label.
   */
  custom_agents?: { state: string; blocks: boolean; sentence: string; label: string };
}
