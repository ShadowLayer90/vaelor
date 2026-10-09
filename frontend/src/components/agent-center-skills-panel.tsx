import { useCallback, useEffect, useRef, type FormEvent } from "react";
import type { AssistantSkill } from "./agentTypes";
import { ConfirmDialog } from "./ConfirmDialog";
import { SkillEditorDialog, type SkillDraft } from "./SkillEditorDialog";
import { StatusPill } from "./StatusPill";
import { Button, Input, Notice, Textarea } from "./ui";
import { timeAgo } from "../lib/format";

export const isBuiltinSkill = (provenance: string) => ["vaelor-builtin", "pironman-builtin"].includes(provenance);

interface AgentCenterSkillsPanelProps {
  busy: boolean;
  deletingSkill: AssistantSkill | null;
  /** The open editor's or delete confirmation's refusal (VD-189), shown inside it. */
  dialogError?: string;
  editingSkill: AssistantSkill | null;
  modelReady: boolean;
  newSkillId: string;
  notice: string;
  skillContent: string;
  skillDescription: string;
  skillName: string;
  skills: AssistantSkill[];
  onDeleteSkill: () => void;
  onEditSkill: (skill: AssistantSkill) => void;
  onProposeSkill: (event: FormEvent) => void;
  onReviewSkill: (skill: AssistantSkill, decision: "active" | "rejected") => void;
  onSaveSkill: (draft: SkillDraft) => void;
  onSetDeletingSkill: (skill: AssistantSkill | null) => void;
  onSetEditingSkill: (skill: AssistantSkill | null) => void;
  onSetSkillContent: (value: string) => void;
  onSetSkillDescription: (value: string) => void;
  onSetSkillName: (value: string) => void;
}

/** A skill's state in the operator's words, not the wire's slug (LESSONS 5). */
// The board writes these in lower case, as a state word beside the name rather than a heading.
const skillStatusWords: Record<string, string> = { active: "active", proposed: "proposed", rejected: "rejected" };

export function AgentCenterSkillsPanel({
  busy,
  deletingSkill,
  dialogError = "",
  editingSkill,
  modelReady,
  newSkillId,
  notice,
  skillContent,
  skillDescription,
  skillName,
  skills,
  onDeleteSkill,
  onEditSkill,
  onProposeSkill,
  onReviewSkill,
  onSaveSkill,
  onSetDeletingSkill,
  onSetEditingSkill,
  onSetSkillContent,
  onSetSkillDescription,
  onSetSkillName,
}: AgentCenterSkillsPanelProps) {
  const newSkillRef = useRef<HTMLElement | null>(null);
  const revealNewSkill = useCallback(() => {
    newSkillRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    newSkillRef.current?.focus({ preventScroll: true });
  }, []);
  useEffect(() => {
    if (!newSkillId || !newSkillRef.current) return;
    revealNewSkill();
  }, [newSkillId, revealNewSkill, skills]);
  const activeCount = skills.filter((item) => item.status === "active").length;
  // The reason beside every control the missing model turns off. The gate is
  // "a model is set up" - the server's own approve check (`ready`) - so the
  // words say that, not the board's "answering" (VD-200 assist review: the
  // reason must be true of the gate it explains).
  const needsModel = modelReady ? undefined : "Needs a model to be set up";
  return (
    /*
     * A view inside Ask rather than a destination of its own. The moment a
     * reader wants to write a skill is the moment an answer came out wrong,
     * and that answer is one "Hide" away - it was two tabs away.
     */
    <section aria-labelledby="assistant-skills-title" className="as-skills ui-card" id="assistant-skills">
      <header className="as-skills__head">
        <div>
          <span className="as-label">Reviewed skills</span>
          <h2 id="assistant-skills-title">Teach Vaelor a better answer</h2>
          <p>Short reviewed skills the assistant may follow. Proposed skills stay inactive until their safety scan and instructions are approved.</p>
        </div>
        <StatusPill status={modelReady ? "healthy" : "degraded"} label={modelReady ? `${activeCount} active` : "Model required"} />
      </header>
      {!modelReady && <div className="as-skills__warn"><Notice severity="warning">Select a local or connected model before skills can run.</Notice></div>}
      <div className="as-skills__body">
        <div className="as-skills__propose">
          <form className="as-skills__form" onSubmit={onProposeSkill}>
            <h3>Propose a skill</h3>
            <Input id="skill-name" label="Skill name" maxLength={100} onChange={(event) => onSetSkillName(event.target.value)} value={skillName} />
            <Input id="skill-description" label="When should it be used?" maxLength={300} onChange={(event) => onSetSkillDescription(event.target.value)} value={skillDescription} />
            <Textarea id="skill-content" label="Reviewed instructions" maxLength={8000} onChange={(event) => onSetSkillContent(event.target.value)} rows={4} value={skillContent} />
            <div className="as-skills__submit">
              <Button
                disabled={!modelReady || busy || !skillName.trim() || !skillDescription.trim() || !skillContent.trim()}
                disabledReason={needsModel}
                type="submit"
                variant="primary"
              >
                Propose skill
              </Button>
            </div>
          </form>
          {/*
            * `href="#skill-<id>"` was not an in-page anchor here: the app routes
            * on the hash, so the browser rewrote the location, the router
            * resolved an unknown path, and "Review proposal" dropped the reader
            * on Home while the proposal sat further down this very list.
            * Scrolling and focusing the card directly is what the link was
            * always meant to do.
            */}
          {notice && (
            <Notice severity="info">
              <span className="as-skills__notice">
                <span>{notice}</span>
                {newSkillId && <Button className="as-btn-ghost" onClick={revealNewSkill} type="button">Review proposal</Button>}
              </span>
            </Notice>
          )}
        </div>
        <div className="as-skills__list">
          {skills.map((skill) => {
            const builtin = isBuiltinSkill(skill.provenance);
            return (
              <article
                className={skill.status === "proposed" ? "as-skill as-skill--proposed" : "as-skill"}
                id={`skill-${skill.id}`}
                key={skill.id}
                ref={skill.id === newSkillId ? newSkillRef : undefined}
                tabIndex={skill.id === newSkillId ? -1 : undefined}
              >
                <div className="as-skill__top">
                  <div>
                    <span className="as-small as-muted">v{skill.version} · {skill.provenance}</span>
                    <h3>{skill.name}</h3>
                  </div>
                  <StatusPill status={skill.status === "active" ? "healthy" : skill.status === "rejected" ? "critical" : "degraded"} label={skillStatusWords[skill.status] ?? "Status not recognised"} />
                </div>
                <p>{skill.description}</p>
                {skill.status === "active" && (
                  <span className="as-small as-muted">
                    Runtime matched {skill.use_count ?? 0} time{skill.use_count === 1 ? "" : "s"}
                    {skill.last_used_at ? ` · last used ${timeAgo(skill.last_used_at * 1000)}` : " · waiting for a relevant request"}
                  </span>
                )}
                <details className="as-skill__instructions"><summary>View instructions</summary><p>{skill.content}</p></details>
                <div className="as-skill__actions">
                  {skill.status === "proposed" && (
                    <>
                      <Button disabled={!modelReady || busy} disabledReason={needsModel} onClick={() => onReviewSkill(skill, "active")} type="button" variant="primary">Approve skill</Button>
                      <Button disabled={busy} onClick={() => onReviewSkill(skill, "rejected")} type="button">Reject</Button>
                    </>
                  )}
                  {!builtin && skill.status !== "proposed" && (
                    <>
                      <Button disabled={busy} onClick={() => onEditSkill(skill)} type="button">Edit</Button>
                      <Button className="as-btn-danger" disabled={busy} onClick={() => onSetDeletingSkill(skill)} type="button">Delete</Button>
                    </>
                  )}
                </div>
                {builtin && <span className="as-small as-muted">Built-in skills cannot be edited or deleted.</span>}
              </article>
            );
          })}
        </div>
      </div>
      <SkillEditorDialog busy={busy} error={dialogError} onCancel={() => onSetEditingSkill(null)} onSave={onSaveSkill} skill={editingSkill} />
      <ConfirmDialog
        busy={busy}
        confirmLabel="Delete skill"
        error={dialogError}
        description={deletingSkill ? `Delete "${deletingSkill.name}" and its revision history? This cannot be undone.` : ""}
        irreversible
        onCancel={() => onSetDeletingSkill(null)}
        onConfirm={onDeleteSkill}
        open={Boolean(deletingSkill)}
        title="Delete custom skill?"
      />
    </section>
  );
}
