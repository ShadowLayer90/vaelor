import { useCallback, useEffect, useRef, type FormEvent } from "react";
import type { AssistantSkill } from "./agentTypes";
import { ConfirmDialog } from "./ConfirmDialog";
import { Icon } from "./Icon";
import { SkillEditorDialog, type SkillDraft } from "./SkillEditorDialog";
import { StatusPill } from "./StatusPill";
import { Button, Input, Notice, Textarea } from "./ui";
import { timeAgo } from "../lib/format";

export const isBuiltinSkill = (provenance: string) => ["vaelor-builtin", "pironman-builtin"].includes(provenance);

interface AgentCenterSkillsPanelProps {
  busy: boolean;
  deletingSkill: AssistantSkill | null;
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

export function AgentCenterSkillsPanel({
  busy,
  deletingSkill,
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
  return (
    /*
     * A disclosure inside Ask rather than a destination of its own. The moment
     * a reader wants to write a skill is the moment an answer came out
     * wrong, and that answer is on this screen — it was two tabs away.
     */
    <section aria-labelledby="assistant-skills-title" className="skills-center" id="assistant-skills">
      <div className="section-heading">
        <div>
          <span className="page-eyebrow">Reviewed skills</span>
          <h2 id="assistant-skills-title">Teach Vaelor a better answer</h2>
          <p>Short reviewed skills the assistant may follow. Proposed skills stay inactive until their safety scan and instructions are approved.</p>
        </div>
        <StatusPill status={modelReady ? "healthy" : "degraded"} label={modelReady ? `${skills.filter((item) => item.status === "active").length} active` : "Model required"} />
      </div>
      {!modelReady && <Notice severity="warning"><Icon name="cpu" />Select a local or connected model before skills can run.</Notice>}
      <form className="skill-proposal" onSubmit={onProposeSkill}>
        <Input id="skill-name" label="Skill name" maxLength={100} onChange={(event) => onSetSkillName(event.target.value)} value={skillName} />
        <Input id="skill-description" label="When should it be used?" maxLength={300} onChange={(event) => onSetSkillDescription(event.target.value)} value={skillDescription} />
<Textarea className="skill-proposal__content" id="skill-content" label="Reviewed instructions" maxLength={8000} onChange={(event) => onSetSkillContent(event.target.value)} rows={4} value={skillContent} />
        <Button disabled={!modelReady || busy || !skillName.trim() || !skillDescription.trim() || !skillContent.trim()} type="submit" variant="primary">Propose skill</Button>
      </form>
      {/*
        * `href="#skill-<id>"` was not an in-page anchor here: the app routes on
        * the hash, so the browser rewrote the location, the router resolved an
        * unknown path, and "Review proposal" dropped the reader on Home while
        * the proposal sat further down this very list. Scrolling and focusing
        * the card directly is what the link was always meant to do.
        */}
      {notice && <Notice severity="info">{notice}{newSkillId && <Button onClick={revealNewSkill} type="button" variant="quiet">Review proposal</Button>}</Notice>}
      <div className="skill-list">
        {skills.map((skill) => (
          <article className="skill-card" id={`skill-${skill.id}`} key={skill.id} ref={skill.id === newSkillId ? newSkillRef : undefined} tabIndex={skill.id === newSkillId ? -1 : undefined}>
            <div className="agent-task-card__header"><div><small>v{skill.version} · {skill.provenance}</small><h2>{skill.name}</h2></div><StatusPill status={skill.status === "active" ? "healthy" : skill.status === "rejected" ? "critical" : "degraded"} label={skill.status} /></div>
            <p>{skill.description}</p>
            {skill.status === "active" && (
              <small>
                Runtime matched {skill.use_count ?? 0} time{skill.use_count === 1 ? "" : "s"}
                {skill.last_used_at ? ` · last used ${timeAgo(skill.last_used_at * 1000)}` : " · waiting for a relevant request"}
              </small>
            )}
            <details><summary>View instructions</summary><p>{skill.content}</p></details>
            {skill.status === "proposed" && <div className="agent-task-card__actions"><Button disabled={!modelReady || busy} onClick={() => onReviewSkill(skill, "active")} type="button" variant="primary">Approve skill</Button><Button disabled={busy} onClick={() => onReviewSkill(skill, "rejected")} type="button" variant="quiet">Reject</Button></div>}
            {!isBuiltinSkill(skill.provenance) && skill.status !== "proposed" && (
              <div className="agent-task-card__actions">
                <Button disabled={busy} onClick={() => onEditSkill(skill)} type="button" variant="quiet">Edit</Button>
                <Button className="danger-text" disabled={busy} onClick={() => onSetDeletingSkill(skill)} type="button" variant="danger">Delete</Button>
              </div>
            )}
          </article>
        ))}
      </div>
      <SkillEditorDialog busy={busy} onCancel={() => onSetEditingSkill(null)} onSave={onSaveSkill} skill={editingSkill} />
      <ConfirmDialog
        busy={busy}
        confirmLabel="Delete skill"
        description={deletingSkill ? `Delete "${deletingSkill.name}" and its revision history? This cannot be undone.` : ""}
        onCancel={() => onSetDeletingSkill(null)}
        onConfirm={onDeleteSkill}
        open={Boolean(deletingSkill)}
        title="Delete custom skill?"
      />
    </section>
  );
}
