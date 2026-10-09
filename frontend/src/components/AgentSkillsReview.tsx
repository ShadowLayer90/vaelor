import { useEffect, useMemo, useState } from "react";
import {
  DEFAULT_SKILL_LIMITS,
  previewAgentSkills,
  type SkillGuidanceRow,
  type SkillLimits,
  type SkillsPreview,
} from "../lib/clusterAgents";
import { Notice } from "./ui";

/**
 * What attaching skills to a deployed cluster agent actually does (ACC-078,
 * ACC-138): the read scopes each skill grants, and exactly how much of its
 * guidance is sent - whole, shortened, or not at all and why. The account is
 * the controller's own (`POST /cluster/agents/skills-preview`, the same
 * derivation the deploy runs), so this review and the deploy cannot disagree.
 */

/** How long the skill choice must settle before it is previewed. */
const PREVIEW_DEBOUNCE_MS = 250;

export interface SkillsPreviewState {
  preview: SkillsPreview | null;
  loading: boolean;
  error: string;
}

const IDLE: SkillsPreviewState = { preview: null, loading: false, error: "" };

/**
 * Preview the chosen skills. Debounced so ticking several boxes asks once, and
 * abortable so a superseded choice never lands its answer over a newer one.
 */
export function useSkillsPreview(skillIds: string[], csrfToken?: string): SkillsPreviewState {
  const key = skillIds.join("\n");
  const [state, setState] = useState<SkillsPreviewState>(IDLE);

  useEffect(() => {
    const ids = key ? key.split("\n") : [];
    if (!ids.length) {
      setState(IDLE);
      return;
    }
    if (!csrfToken) {
      setState({ preview: null, loading: false, error: "This console session cannot ask the controller for the skills check." });
      return;
    }
    const controller = new AbortController();
    setState((current) => ({ ...current, loading: true, error: "" }));
    const timer = window.setTimeout(() => {
      previewAgentSkills(ids, csrfToken, controller.signal)
        .then((preview) => {
          if (!controller.signal.aborted) setState({ preview, loading: false, error: "" });
        })
        .catch((reason: unknown) => {
          if (controller.signal.aborted) return;
          setState({
            preview: null,
            loading: false,
            error: reason instanceof Error && reason.message ? reason.message : "The request did not complete.",
          });
        });
    }, PREVIEW_DEBOUNCE_MS);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [csrfToken, key]);

  return state;
}

/** A read scope in plain words: "cluster:read" reads as "cluster". */
export function scopeWords(scope: string): string {
  return scope.replace(/:read$/, "");
}

/** The configure step's promise, stated with the limits the deploy applies. */
export function skillsAttachCopy(limits: SkillLimits = DEFAULT_SKILL_LIMITS): string {
  return (
    "Attaching a skill adds its read scopes to what the agent can read, and "
    + "sends the agent the skill's name, description and up to "
    + `${limits.instructions_per_skill.toLocaleString("en-US")} characters of its instructions. `
    + `All attached skills share ${limits.block.toLocaleString("en-US")} characters, and at most `
    + `${limits.skills} are sent; a skill that does not fit is not sent and grants no read scope.`
  );
}

function guidanceText(row: SkillGuidanceRow): string {
  if (row.status === "sent") return "Sent in full";
  if (row.status === "shortened") {
    return `Shortened: sent ${row.sent_chars.toLocaleString("en-US")} of ${row.total_chars.toLocaleString("en-US")} characters`;
  }
  return row.reason ? `Not sent: ${row.reason}` : "Not sent";
}

/**
 * The review step's "Skills attached" block. `names` maps a skill id to the
 * library's name, used when the preview row carries none - an id is never
 * shown to the owner.
 */
export function AgentSkillsReview({
  names,
  state,
}: {
  names: Record<string, string>;
  state: SkillsPreviewState;
}) {
  const { preview, loading, error } = state;
  const nameOf = useMemo(
    () => (skillId: string, fallback = "") => fallback || names[skillId] || "A skill no longer in the library",
    [names],
  );
  const attached = Object.keys(names);

  return (
    <div className="cl-panel">
      <strong>Skills attached</strong>
      {attached.length === 0 ? (
        <p>No skills attached.</p>
      ) : (
        <>
          <p>{skillsAttachCopy(preview?.limits)}</p>
          {loading && !preview && <p role="status">Checking what these skills send to the agent...</p>}
          {error && (
            <p role="alert">{`Vaelor could not check what these skills send (${error.replace(/\.$/, "")}). The deploy still applies the limits above.`}</p>
          )}
          {preview ? (
            <>
              <ul className="cd-list">
                {preview.guidance.map((row) => (
                  <li key={row.skill_id}>{`${nameOf(row.skill_id, row.name)} - ${guidanceText(row)}`}</li>
                ))}
              </ul>
              <p>
                {`Read access this agent will get from its skills: ${
                  preview.scopes.length ? preview.scopes.map(scopeWords).join(", ") : "none"
                }.`}
              </p>
              {preview.refusals.map((refusal) => (
                <Notice
                  heading={`${nameOf(refusal.skill_id)} cannot be attached`}
                  key={refusal.skill_id}
                  severity="danger"
                >
                  {refusal.reason}
                </Notice>
              ))}
            </>
          ) : !loading && !error ? (
            <ul className="cd-list">
              {attached.map((skillId) => <li key={skillId}>{nameOf(skillId)}</li>)}
            </ul>
          ) : null}
        </>
      )}
    </div>
  );
}
