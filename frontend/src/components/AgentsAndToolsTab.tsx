import { useState } from "react";
import type { Session } from "../types";
import { TabSet } from "./ui";
import { ClusterInferenceAgentsPanel } from "./ClusterInferenceAgentsPanel";
import { McpCatalogPanel } from "./McpCatalogPanel";
import { SkillsLibraryPanel } from "./SkillsLibraryPanel";
import "../styles/cluster-agents.css";

/**
 * The Cluster page's "Agents & tools" tab (F6a, F6b, F1): the section that
 * gathers what a fleet agent is built from.
 *
 * It holds three registries - the inference agents deployed on the cluster, the
 * MCP catalog (the Model Context Protocol servers whose tools an agent may call)
 * and the skills library (the reusable skills a deployed agent or model may draw
 * on). They are sub-tabs rather than stacked: each is a full registry with its
 * own card list and dialogs, so one visible at a time keeps the section from
 * becoming a long scroll. VD-200 (the ClusterAgents boards): the sub-tab strip is
 * drawn as a segmented control at the right of the section's heading; it stays a
 * real tablist, so the keyboard and a screen reader still treat it as tabs.
 */
type AgentsToolsView = "inference" | "mcp" | "skills";

export function AgentsAndToolsTab({ session }: { session: Session }) {
  const [view, setView] = useState<AgentsToolsView>("inference");
  return (
    <div className="cl-agents">
      <div className="cl-agents__intro">
        <h2>What a fleet agent is built from</h2>
        <p>
          Build the read-only inference agents you deploy on the cluster, and register the tools
          and skills they draw on: a Model Context Protocol server to expose its tools, or a skill
          that describes a capability an agent assumes.
        </p>
      </div>
      <TabSet
        className="cl-agents__tabs"
        items={[
          { id: "inference", label: "Inference agents" },
          { id: "mcp", label: "MCP servers" },
          { id: "skills", label: "Skills" },
        ]}
        label="Agents and tools registries"
        listClassName="cl-agents__seg"
        onSelect={(id) => setView(id as AgentsToolsView)}
        panelClassName="cl-agents__panel"
        selectedId={view}
      >
        {view === "inference" && <ClusterInferenceAgentsPanel session={session} />}
        {view === "mcp" && <McpCatalogPanel session={session} />}
        {view === "skills" && <SkillsLibraryPanel session={session} />}
      </TabSet>
    </div>
  );
}
