import type { PerformanceSummary } from "../lib/performanceLine";

export interface AssistantModelFacts {
  /** The catalog id of the deployed model, or "" when it is unrecognised. */
  model: string;
  /** Measured weaknesses, per model. Empty when nobody has measured it. */
  shortcomings: string[];
  /** Seconds of quiet before llama-server unloads the weights (VD-073). */
  sleep_idle_seconds?: number | null;
  /** Measured seconds for the first question after an unload. */
  cold_start_seconds?: number | null;
}

export interface AgentProfile {
  id: string;
  name: string;
  description: string;
  scopes: string[];
  operational?: boolean;
  model_required?: boolean;
  model_ready?: boolean;
  tools?: string[];
  custom?: boolean;
  version?: number;
  enabled?: boolean;
  instructions?: string;
  permissions?: string[];
  read_collection_ids?: string[];
  write_collection_id?: string;
  web_access?: { enabled: boolean; allowed_domains: string[] };
  connectors?: AgentConnector[];
}

export interface AgentConnectorOperation {
  id: string;
  description: string;
  method: "GET" | "HEAD" | "POST" | "PUT" | "PATCH" | "DELETE";
  path: string;
  input_location: "query" | "json";
  request_schema: Record<string, unknown>;
  response_schema: Record<string, unknown>;
  timeout_seconds: number;
  max_response_bytes: number;
  rate_limit_per_minute: number;
  approval: "not_required" | "required";
}

export interface AgentConnector {
  id: string;
  name: string;
  base_origin: string;
  credential_ref: string;
  auth: "none" | "bearer" | "x-api-key";
  operations: AgentConnectorOperation[];
}

export interface AgentWriteProposal {
  type: "knowledge.document";
  collection_id: string;
  collection_name?: string;
  name: string;
  content: string;
  media_type: string;
  executed?: boolean;
  document_id?: string;
  approved_by?: string;
}

export interface AgentTask {
  id: string;
  actor: string;
  assigned_to?: string;
  title: string;
  description: string;
  kind: "temporary" | "durable";
  profile: string;
  profile_version?: number;
  approval_required?: boolean;
  approval_context?: {
    profile_name?: string;
    profile_version?: number;
    capabilities?: string[];
    integrations?: string[];
  };
  state: string;
  result: {
    answer?: string;
    summary?: string;
    findings?: string[];
    recommendations?: string[];
    next_actions?: string[];
    evidence?: AssistantEvidence[];
    sources?: AssistantEvidence[];
    warnings?: string[];
    errors?: string[];
    /** Specific questions the run puts back to the user when it could not verify
     * an answer or the task was ambiguous (#247w). Paired with outcome
     * "needs_input"; the run asks rather than guessing. */
    clarifications?: string[];
    /** "delivered" | "no_result" | "needs_input" — what the run produced,
     * independent of whether it ran. */
    outcome?: string;
    outcome_reason?: string;
    executed_changes?: boolean;
    /** True when the run fell back to built-in diagnostics without the model. */
    degraded?: boolean;
    /**
     * Present when the user may re-run this SAME task on the more capable GPU
     * model (shared with AI Chat). Set only when the run underperformed - it
     * failed, delivered degraded, produced no result, or an eligible tool loop
     * fell back - AND a GPU model is available. Absent on a clean success.
     */
    escalation_available?: {
      model_tier: "capable";
      model: string;
      surface: string;
    };
    capability_audit?: {
      execution_binding?: {
        mode?: string;
        provider?: string;
        model?: string;
        status?: string;
      };
      [key: string]: unknown;
    };
    knowledge_sources?: Array<{
      collection: string;
      document: string;
      chunk: number;
    }>;
    proposed_writes?: AgentWriteProposal[];
  };
  error: string;
  updated_at: number;
}

export interface HandoffTarget {
  username: string;
  role: string;
}

export interface AssistantSkill {
  id: string;
  name: string;
  description: string;
  content: string;
  version: number;
  status: "active" | "proposed" | "rejected";
  provenance: string;
  use_count?: number;
  last_used_at?: number | null;
  model_ready?: boolean;
  validation: { reviewed: boolean; secret_scan: string };
}

/**
 * What an unattended run of this rule's pinned definition may do.
 *
 * Creating the rule is the approval for every run it will ever make — no run
 * stops for a further one — so the screen has to be able to say what is being
 * approved. The server computes this from the pinned definition version, not
 * the agent's current one.
 */
export interface CapabilityDisclosure {
  agent: string;
  definition_version: number;
  pinned_definition_available: boolean;
  reads: string[];
  web_access: string;
  integrations: string[];
  writes: string;
}

export interface Automation {
  id: string;
  name: string;
  prompt: string;
  profile: string;
  kind: string;
  schedule_text: string;
  next_run_at?: number | null;
  enabled: boolean;
  capability_disclosure?: CapabilityDisclosure;
}

export interface Trigger {
  id: string;
  name: string;
  prompt: string;
  profile: string;
  source: string;
  operator: string;
  threshold: number;
  cooldown_seconds: number;
  enabled: boolean;
  last_value?: number | null;
  last_triggered_at?: number | null;
  capability_disclosure?: CapabilityDisclosure;
}

/**
 * A place a fired alert is delivered. The secret (SMTP password or webhook
 * token) is never carried here — only `has_secret` says whether one is stored.
 */
export interface AlertChannel {
  id: string;
  kind: "email" | "webhook";
  name: string;
  enabled: boolean;
  smtp_host: string;
  smtp_port: number;
  security: string;
  from_address: string;
  to_address: string;
  username: string;
  url: string;
  auth_header: string;
  has_secret: boolean;
  last_delivery_status: string;
  last_delivery_error: string;
  last_delivery_at?: number | null;
}

export interface AssistantEvidence {
  source: string;
  summary: string;
}

export interface ApplicationResearchIntent {
  type: "application.deploy";
  application_query: string;
  delivery_method: "auto" | "container";
  confidence: "low" | "medium" | "high";
  research_required: boolean;
  missing_inputs: string[];
}

/**
 * A screen an answer named, with the hash route that actually reaches it.
 *
 * `navigation_steps` in `vaelor/assistant_answer_presentation.py` computes
 * these from a fixed table of destinations — they are never written by the
 * model — which is why an answer that says "ask it in AI Chat" can be turned
 * into a link rather than left as a sentence the reader has to act on.
 */
export interface AssistantNavigationStep {
  label: string;
  route: string;
  control: string;
}

export interface ChatMessage {
  id?: number;
  role: "user" | "assistant";
  content: string;
  created_at?: number;
  metadata?: {
    source?: string;
    evidence?: AssistantEvidence[];
    suggested_actions?: string[];
    next_steps?: AssistantNavigationStep[];
    proposed_job?: { type: string; payload: Record<string, unknown> } | null;
    application_intent?: ApplicationResearchIntent | null;
    proposed_agent_task?: AgentRunProposal | null;
    /** Compact per-answer timing (total, TTFT, prefill/decode tok/s). */
    performance?: PerformanceSummary;
    /**
     * A response the reader stopped. Terminal, and never a failure: it is
     * written by this client into the transcript so that a stopped answer has
     * a visible end, rather than leaving a question sitting alone.
     */
    stopped?: boolean;
  };
}

export interface AssistantAnswer {
  answer: string;
  conversation_id: string;
  source: string;
  evidence: AssistantEvidence[];
  suggested_actions: string[];
  next_steps?: AssistantNavigationStep[];
  proposed_job?: { type: string; payload: Record<string, unknown> } | null;
  application_intent?: ApplicationResearchIntent | null;
  proposed_agent_task?: AgentRunProposal | null;
  approval_required: boolean;
  /** Compact per-answer timing when the model answered; absent otherwise. */
  performance?: PerformanceSummary;
}

export interface AgentRunProposal {
  profile_id: string;
  profile_name: string;
  profile_version: number;
  task: string;
  capabilities: string[];
  integrations?: string[];
}

export interface AgentStatus {
  configured: boolean;
  /** Whether the endpoint answered a probe, as opposed to merely being set. */
  reachable?: boolean;
  unreachable_reason?: string;
  /**
   * What is measured about the model that is actually deployed - its
   * shortcomings (VD-071) and the idle-unload periods llama-server is
   * configured with (VD-073). Absent when Vaelor cannot identify the model,
   * which is the honest answer rather than a generic one.
   */
  model_facts?: AssistantModelFacts;
  endpoint?: string;
  provider: string;
  model?: string | null;
  capability?: {
    tier: "rules" | "basic-local" | "capable-local" | "advanced-local" | "connected" | "frontier";
    label: string;
    description: string;
    limitations: string[];
  };
}

export interface AssistantConversation {
  id: string;
  title: string;
  summary: string;
  created_at: number;
  updated_at: number;
  archived: boolean;
  message_count: number;
}
