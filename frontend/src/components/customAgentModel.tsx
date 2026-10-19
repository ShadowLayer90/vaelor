import { useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import { destinationName } from "../lib/destinations";
import { hashForPage } from "../lib/navigation";
import type { AgentTask } from "./agentTypes";
import type { AiChatSetup } from "./aiChatTypes";
import { StatusPill } from "./StatusPill";
import { Notice } from "./ui";

/*
 * VD-220 (owner): a custom agent runs on AI Chat's model, wherever AI Chat
 * points, never the Assistant's NPU model. With no AI Chat model the run fails
 * at once with the backend's sentence (`ai_chat_model_lease.AGENT_UNAVAILABLE`),
 * which this module shows plainly with the way to AI Chat beside it.
 */

/** How every one of the backend's unavailable sentences begins (`_AGENT`). */
export const AI_CHAT_UNAVAILABLE_LEAD = "Custom agents run on AI Chat's model, and ";
/** The runner's error code in front of that sentence (`AgentTaskError`). */
const UNAVAILABLE_CODE = /^ai_chat_model_unavailable:\s*/;

/** Whether a run belongs to a custom agent (the store mints `custom_<hex>` ids). */
export function isCustomAgentTask(task: Pick<AgentTask, "profile">): boolean {
  return task.profile.startsWith("custom_");
}

/**
 * The backend's "AI Chat's model is unavailable" sentence for a failed run,
 * without the error code in front of it, or "" when the run failed otherwise.
 * The busy sentence arrives with no code; the others carry one.
 */
export function aiChatUnavailableSentence(error: string | undefined): string {
  const sentence = String(error ?? "").trim().replace(UNAVAILABLE_CODE, "");
  return sentence.startsWith(AI_CHAT_UNAVAILABLE_LEAD) ? sentence : "";
}

/** A run's error as the owner reads it: the sentence alone when it is AI Chat's. */
export function runErrorText(error: string | undefined): string {
  return aiChatUnavailableSentence(error) || String(error ?? "");
}

/**
 * The runner's own verdict on AI Chat's model, sent on `/ai-chat/setup` as
 * `custom_agents` (`ai_chat_model_lease.custom_agent_readiness`, VD-220). The
 * console never infers a state from the lease's shape (kb R212).
 */
export type CustomAgentVerdict = NonNullable<AiChatSetup["custom_agents"]>;

/**
 * What the agent screens know about AI Chat's model. Not read is not none
 * (LESSONS 8): only the backend's blocking verdict holds custom agents back;
 * a cluster that is loading or unloaded says so and blocks nothing.
 */
export type AiChatModelRead =
  | { state: "reading" }
  | { state: "unread" }
  | { state: "ready"; label: string; short: string }
  | { state: "blocked"; sentence: string }
  | { state: "notice"; sentence: string; unreadable: boolean };

/** AI Chat's model, read once from `/ai-chat/setup`; a failed read is "unread", never blocking. */
export function useAiChatModel(): AiChatModelRead {
  const [model, setModel] = useState<AiChatModelRead>({ state: "reading" });
  useEffect(() => {
    let live = true;
    void apiRequest<Partial<AiChatSetup>>("/ai-chat/setup")
      .then((setup) => { if (live) setModel(aiChatModelFromSetup(setup)); })
      .catch(() => { if (live) setModel({ state: "unread" }); });
    return () => { live = false; };
  }, []);
  return model;
}

/** Why a custom agent cannot be created, run or scheduled, or undefined when it can. */
export function aiChatModelBlocks(model: AiChatModelRead): string | undefined {
  return model.state === "blocked" ? model.sentence : undefined;
}

/** The backend's verdict from `/ai-chat/setup`, or "unread" when the read did not carry one. */
export function aiChatModelFromSetup(setup: Partial<AiChatSetup> | null | undefined): AiChatModelRead {
  const verdict = setup?.custom_agents;
  if (!verdict || typeof verdict.state !== "string") return { state: "unread" };
  if (verdict.state === "ready") {
    const label = String(verdict.label ?? "");
    return { state: "ready", label, short: label.split(" · ")[0] ?? "" };
  }
  const sentence = String(verdict.sentence ?? "");
  return verdict.blocks
    ? { state: "blocked", sentence }
    : { state: "notice", sentence, unreadable: verdict.state === "unreadable" };
}

export function GoToAiChat() {
  return (
    <a className="ui-button ui-button--secondary" href={hashForPage("ai-chat")}>
      Go to {destinationName("ai-chat")}
    </a>
  );
}

/**
 * One line on a custom agent's card: where it runs, and the model when known.
 * When it cannot take a run, the page's banner says why once.
 */
export function RunsOnAiChatModel({ model }: { model: AiChatModelRead }) {
  return (
    <span className="ar-meta">
      {model.state === "ready" && model.label ? `Runs on AI Chat's model (${model.label})`
        : model.state === "blocked" || model.state === "notice" ? "Runs on AI Chat's model · not available right now"
          : "Runs on AI Chat's model"}
    </span>
  );
}

/** A run that failed because AI Chat's model could not take it, said plainly. */
export function AiChatUnavailableNotice({ sentence }: { sentence: string }) {
  return (
    <Notice severity="danger">
      <span className="ar-notice-row">
        <span>{sentence}</span>
        <GoToAiChat />
      </span>
    </Notice>
  );
}

/** The Agents view's top-bar pill: AI Chat's model, which custom agents run on. */
export function AiChatModelPill({ model }: { model: AiChatModelRead }) {
  if (model.state === "ready") return <StatusPill label={model.short || "AI Chat's model"} status="healthy" />;
  if (model.state === "blocked") return <StatusPill label="AI Chat model needed" status="degraded" />;
  if (model.state === "notice" && !model.unreadable) return <StatusPill label="AI Chat model not serving" status="degraded" />;
  return model.state === "reading"
    ? <StatusPill label="Checking…" status="neutral" />
    : <StatusPill label="Not read" reading="unread" status="neutral" />;
}

/** The Agents view's banner: where custom agents run, and what holds them back - in the backend's words. */
export function AiChatModelBanner({ model }: { model: AiChatModelRead }) {
  if (model.state === "reading") return <Notice severity="info">Checking AI Chat's model…</Notice>;
  if (model.state === "unread") {
    return (
      <Notice severity="warning" standing>
        AI Chat's model could not be read just now. Custom agents run on it, so New agent and runs stay
        on; a run that cannot reach it says why.
      </Notice>
    );
  }
  if (model.state === "blocked") {
    return (
      <Notice severity="warning" standing>
        <span className="ar-notice-row">
          <span>{model.sentence} New agent and runs stay off until then.</span>
          <GoToAiChat />
        </span>
      </Notice>
    );
  }
  if (model.state === "notice") return <Notice severity="warning" standing>{model.sentence}</Notice>;
  return (
    <Notice severity="success">
      {model.label ? `Custom agents run on AI Chat's model (${model.label}).` : "Custom agents run on AI Chat's model."}
    </Notice>
  );
}
