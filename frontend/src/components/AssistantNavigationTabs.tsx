import type { KeyboardEvent } from "react";
import type { Role } from "../types";
import { Button } from "./ui";

export type AssistantTab = "ask" | "routines" | "history";

/**
 * Three tabs, cut on tense: what I am asking now, what runs without me, what
 * has already run.
 *
 * The six that were here — Ask Vaelor, Troubleshoot, Agents, Memory, Skills,
 * Schedules — were cut along implementation seams. The two that replaced them
 * were cut on "conversation vs administration", which is a seam too: it forced
 * Ask to carry a live chat and a forty-five-row audit archive at once, so the
 * answer the screen exists for got 146px against the archive's 1,883, and it
 * forced Agents to carry authoring, scheduling and credentials together.
 *
 * A third tab is the simplification, not a cost. History is not administration
 * — every reader has one — so it is not administrator-gated; authoring agents
 * and their unattended runs is, and stays behind Routines.
 *
 * `memory` used to be listed here with no `adminOnly` flag, while every
 * endpoint `MemoryCenter` calls requires an administrator: an operator saw the
 * tab and it failed on load, every time. Memory is now its own `#/memory`
 * route, reached from an administrator-only link on the composer, because it is
 * appliance-wide and shared with AI Chat rather than the Assistant's own.
 */
const tabs: ReadonlyArray<{
  id: AssistantTab;
  label: string;
  /** The top bar breadcrumb's words for the tab (VD-200, the Assist boards). */
  place: string;
  adminOnly?: boolean;
}> = [
  { id: "ask", label: "Ask", place: "Ask about this machine" },
  { id: "routines", label: "Routines", place: "Routines", adminOnly: true },
  { id: "history", label: "History", place: "History" },
];

/**
 * Each tab's place in the breadcrumb ("Assistant / Ask about this machine"),
 * from the same table as the strip, so the two cannot drift.
 */
export const ASSISTANT_TAB_PLACES: Readonly<Record<AssistantTab, string>> = Object.fromEntries(
  tabs.map((tab) => [tab.id, tab.place]),
) as Record<AssistantTab, string>;

/**
 * The canonical `TabSet` primitive owns a single panel and swaps its children;
 * this tab strip does not. Its three panels are separate, separately-owned
 * components (`AgentAssistantPanel`, `AssistantHistoryPanel`,
 * `CustomAgentsPanel`) that its consumer renders as siblings after the strip,
 * each already exposing its own `role="tabpanel"` id (`#ask-panel`,
 * `#history-panel`, `#routines-panel`) while it is the one shown. Wrapping those in `TabSet`'s own panel
 * would nest one tabpanel inside another and break the `aria-controls` link to
 * the real ids. So the ARIA tabs pattern is implemented here directly — the
 * same roving `tabIndex` and Arrow/Home/End behaviour `TabSet` provides — and
 * `aria-controls` continues to name the real panel each tab governs.
 */
export function AssistantNavigationTabs({
  active,
  onChange,
  role,
}: {
  active: AssistantTab;
  onChange: (tab: AssistantTab) => void;
  role: Role;
}) {
  const visibleTabs = tabs.filter((tab) => !tab.adminOnly || role === "administrator");

  const focusTab = (id: AssistantTab) => {
    onChange(id);
    // The button may not yet carry tabIndex 0 in this render, so focus it on the
    // next frame once the roving index has followed the selection.
    const focusNow = () => document.getElementById("assistant-tab-" + id)?.focus();
    focusNow();
    queueMicrotask(focusNow);
  };

  const onTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>, id: AssistantTab) => {
    const index = visibleTabs.findIndex((tab) => tab.id === id);
    if (index === -1) return;
    const last = visibleTabs.length - 1;
    let nextIndex: number | null = null;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") nextIndex = (index + 1) % visibleTabs.length;
    else if (event.key === "ArrowLeft" || event.key === "ArrowUp") nextIndex = (index - 1 + visibleTabs.length) % visibleTabs.length;
    else if (event.key === "Home") nextIndex = 0;
    else if (event.key === "End") nextIndex = last;
    if (nextIndex === null) return;
    event.preventDefault();
    focusTab(visibleTabs[nextIndex].id);
  };

  return (
    <div className="as-tabs" role="tablist" aria-label="Vaelor assistant workspaces">
      {visibleTabs.map((tab) => {
        const selected = active === tab.id;
        return (
          <Button
            // Only the selected tab's panel is rendered; naming an absent
            // panel points a screen reader at nothing (UX-A4).
            aria-controls={selected ? tab.id + "-panel" : undefined}
            aria-selected={selected}
            className={selected ? "as-tab as-tab--active" : "as-tab"}
            id={"assistant-tab-" + tab.id}
            key={tab.id}
            onClick={() => onChange(tab.id)}
            onKeyDown={(event) => onTabKeyDown(event, tab.id)}
            role="tab"
            tabIndex={selected ? 0 : -1}
            type="button"
            variant="quiet"
          >
            {tab.label}
          </Button>
        );
      })}
    </div>
  );
}
