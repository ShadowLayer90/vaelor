import type { ReactNode } from "react";
import { Icon } from "./Icon";
import type { AgentStatus } from "./agentTypes";

/**
 * Which intelligence answers on Ask (VD-200, the AssistIntelligence board's
 * "Active intelligence" card). It sits in the Change-intelligence drawer now:
 * the page's own top line already names the model in its pill, so a second
 * card restating it above every conversation was the board's first cut.
 *
 * `pill` is the same honest state the page's pill shows (Checking… while the
 * status is unread, never a guess); `children` are the model's own facts
 * (AssistantModelState) inside the card.
 */
export function AssistantEngineSummary({
  children,
  pill,
  status,
}: {
  children?: ReactNode;
  pill?: ReactNode;
  status: AgentStatus | null;
}) {
  const loading = status === null;
  return (
    <section aria-busy={loading || undefined} aria-labelledby="assistant-engine-title" className="as-engine ui-card">
      <div className="as-engine__top">
        <span aria-hidden="true" className={loading ? "ui-row__icon" : "ui-row__icon ui-row__icon--accent"}>
          <Icon name={loading || status.configured ? "npu" : "cpu"} size={16} />
        </span>
        <div className="as-engine__titles">
          <span className="as-label">Active intelligence</span>
          <strong id="assistant-engine-title">{loading ? "Checking active intelligence" : status.configured ? status.provider : "Built-in appliance help"}</strong>
          <span className="as-small as-muted">
            {loading
              ? "Reading the current Assistant connection from this node…"
              : status.configured
                ? `${status.model || "Auto-detected model"} · live hardware context included`
                : "Basic answers and live diagnostics work now. Add a model for broader reasoning."}
          </span>
        </div>
        {!loading && pill}
      </div>
      {status?.capability && (
        <div className={`as-box as-engine__capability as-engine__capability--${status.capability.tier}`}>
          <strong>{status.capability.label}</strong>
          <span className="as-small as-muted">{status.capability.description}</span>
          {status.capability.limitations.length > 0 && (
            <span className="as-small as-muted">{status.capability.limitations.join(" ")}</span>
          )}
        </div>
      )}
      {children}
    </section>
  );
}
