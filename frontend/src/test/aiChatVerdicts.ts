/*
 * `/ai-chat/setup` as the appliance answers it for the custom-agent screens:
 * `custom_agents` is `ai_chat_model_lease.custom_agent_readiness`'s verdict,
 * with the runner's own sentences (AGENT_UNAVAILABLE). VD-220, kb R212.
 */

export const NO_MODEL_SENTENCE = "Custom agents run on AI Chat's model, and AI Chat has no model chosen. "
  + "Choose one in AI Chat, then run this agent again.";

export const CLUSTER_NOT_SERVING_SENTENCE = "Custom agents run on AI Chat's model, and that is the GPU cluster's "
  + "model, which is not serving right now: it is still loading, or it was unloaded in Cluster > Deployments. "
  + "Load it there, or run this agent again once it has finished loading.";

/** AI Chat on a model: the verdict carries its label. */
export const aiChatSetupReady = {
  collections: [],
  active_connection: { id: "c1", label: "Vaelor GPU", provider: "openai-compatible", active_for: ["ai-chat"] },
  custom_agents: { state: "ready", blocks: false, sentence: "", label: "qwen3.8-27b · Vaelor GPU" },
};

/** No AI Chat model chosen: the one verdict that blocks. */
export const aiChatSetupNone = {
  collections: [],
  active_connection: null,
  custom_agents: { state: "none", blocks: true, sentence: NO_MODEL_SENTENCE, label: "" },
};

/**
 * Mode B while the cluster is loading or was unloaded by hand:
 * `gpu_cluster_mode._degrade_ai_chat` clears the AI Chat lease, so there is no
 * active connection - and the runner says the cluster is not serving, which
 * blocks nothing (the server accepts schedules and runs report why).
 */
export const aiChatSetupClusterNotServing = {
  collections: [],
  active_connection: null,
  custom_agents: { state: "cluster-not-serving", blocks: false, sentence: CLUSTER_NOT_SERVING_SENTENCE, label: "" },
};
