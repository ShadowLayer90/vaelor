import { createContext, useContext, type ReactNode } from "react";
import { TopbarPageActions, usePagePlace } from "../lib/topbarSlot";

/*
 * Where an Assistant tab puts the board's top-bar items (VD-200: the model pill
 * and the tab's one primary action - Change intelligence, New agent, New
 * schedule, Reload): the console's top bar, through the shell's page slot.
 * The items stay in the component whose state they act on - New agent opens
 * the editor that CustomAgentManager owns - instead of being lifted into the
 * page container.
 *
 * Outside the Assistant (Cluster mounts CustomAgentManager and the automations
 * panel too) the items render where they are written, and the panels say
 * nothing to the breadcrumb.
 */
const InAssistantPage = createContext(false);

export function AssistantBarProvider({ children }: { children: ReactNode }) {
  return <InAssistantPage.Provider value>{children}</InAssistantPage.Provider>;
}

/** True when this component is on the Assistant page. */
export function useInAssistantBar(): boolean {
  return useContext(InAssistantPage);
}

export function AssistantBarActions({ children }: { children: ReactNode }) {
  return useContext(InAssistantPage) ? <TopbarPageActions>{children}</TopbarPageActions> : <>{children}</>;
}

/**
 * The breadcrumb's parts after "Assistant" for the shown tab and view
 * ("Ask about this machine / Skills", "Routines / Agents"), said only on the
 * Assistant page.
 */
export function useAssistantPlace(parts: readonly string[]): void {
  usePagePlace(useContext(InAssistantPage) ? parts : null);
}
