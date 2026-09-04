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

export function AssistantNavigationTabs({
  active,
  onChange,
  role,
}: {
  active: AssistantTab;
  onChange: (tab: AssistantTab) => void;
  role: Role;
}) {
  return (
    <div className="assistant-tabs" role="tablist" aria-label="Vaelor assistant workspaces">
      {tabs.filter((tab) => !tab.adminOnly || role === "administrator").map((tab) => (
        <Button
          aria-controls={tab.id + "-panel"}
          aria-selected={active === tab.id}
          className={active === tab.id ? "assistant-tab--active" : undefined}
          key={tab.id}
          onClick={() => onChange(tab.id)}
          role="tab"
          type="button"
          variant="quiet"
        >
          <Icon name={tab.icon} size={18} /> {tab.label}
        </Button>
      ))}
    </div>
  );
}
