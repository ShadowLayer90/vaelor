import type { KeyboardEvent } from "react";
import type { Role } from "../types";
import { Icon, type IconName } from "./Icon";
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
  icon: IconName;
  adminOnly?: boolean;
}> = [
  { id: "ask", label: "Ask", icon: "cpu" },
  { id: "routines", label: "Routines", icon: "memory", adminOnly: true },
  { id: "history", label: "History", icon: "activity" },
];

/**
 * The canonical `TabSet` primitive owns a single panel and swaps its children;
 * this tab strip does not. Its three panels are separate, separately-owned
 * components (`AgentAssistantPanel`, `AssistantHistoryPanel`,
 * `CustomAgentsPanel`) that its consumer renders as siblings after the strip,
 * each already exposing its own `role="tabpanel"` id (`#ask-panel`,
 * `#history-panel`, `#routines-panel`). Wrapping those in `TabSet`'s own panel
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
    <div className="assistant-tabs" role="tablist" aria-label="Vaelor assistant workspaces">
      {visibleTabs.map((tab) => {
        const selected = active === tab.id;
        return (
          <Button
            aria-controls={tab.id + "-panel"}
            aria-selected={selected}
            className={selected ? "assistant-tab assistant-tab--active" : "assistant-tab"}
            id={"assistant-tab-" + tab.id}
            key={tab.id}
            onClick={() => onChange(tab.id)}
            onKeyDown={(event) => onTabKeyDown(event, tab.id)}
            role="tab"
            tabIndex={selected ? 0 : -1}
            type="button"
            variant="quiet"
          >
            <Icon name={tab.icon} size={18} /> {tab.label}
          </Button>
        );
      })}
    </div>
  );
}
