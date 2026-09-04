export interface AiChatConnection {
  id: string;
  label: string;
  provider: string;
  active_for: string[];
  selected_model?: string;
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
  };
}

export interface AiChatConversation {
  id: string;
  title: string;
  model: string;
  collections: string[];
  updated_at: number;
  archived: boolean;
}

export interface AiChatSetup {
  connections: AiChatConnection[];
  active_connection: AiChatConnection | null;
  preference: { model: string; collection_ids: string[] };
  collections: AiChatCollection[];
  limits: { document_bytes: number; collections: number };
  retrieval: string;
}
