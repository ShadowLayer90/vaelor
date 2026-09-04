import type { ReactNode } from "react";
import type { Role } from "../types";
import { AssistantNavigationTabs, type AssistantTab } from "./AssistantNavigationTabs";

interface AgentCenterTabsProps {
  active: AssistantTab;
  role: Role;
  onChange: (tab: AssistantTab) => void;
  askPanel: ReactNode;
  historyPanel: ReactNode;
  routinesPanel: ReactNode;
}

export function AgentCenterTabs({
  active,
  role,
  onChange,
  askPanel,
  historyPanel,
  routinesPanel,
}: AgentCenterTabsProps) {
  /*
   * Routines is administrator-only, so selecting it for anybody else would
   * leave the ARIA tablist reporting a selection that is not in it. History is
   * not gated: it is the same evidence every reader already had on Ask.
   */
  const resolved: AssistantTab = active === "routines" && role !== "administrator" ? "ask" : active;
  return (
    <section className="agent-center">
      <AssistantNavigationTabs
        active={resolved}
        onChange={onChange}
        role={role}
      />

      {resolved === "routines" && routinesPanel}
      {resolved === "history" && historyPanel}
      {resolved === "ask" && askPanel}
    </section>
  );
}
