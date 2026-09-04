import { useEffect, useRef, useState } from "react";
import type { AssistantSkill } from "./agentTypes";
import { Button, Input, Textarea } from "./ui";

export interface SkillDraft {
  name: string;
  description: string;
  content: string;
}

export function SkillEditorDialog({
  skill,
  busy,
  onCancel,
  onSave,
}: {
  skill: AssistantSkill | null;
  busy: boolean;
  onCancel: () => void;
  onSave: (draft: SkillDraft) => void;
}) {
  const [draft, setDraft] = useState<SkillDraft>({
    name: "",
    description: "",
    content: "",
  });
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  const nameRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!skill) return;
    setDraft({
      name: skill.name,
      description: skill.description,
      content: skill.content,
    });
    setConfirmDiscard(false);
    nameRef.current?.focus();
  }, [skill]);

  useEffect(() => {
    if (!skill) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) {
        const dirty = draft.name !== skill.name
          || draft.description !== skill.description
          || draft.content !== skill.content;
        if (dirty) setConfirmDiscard(true);
        else onCancel();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [busy, draft, onCancel, skill]);

  if (!skill) return null;
  const valid = draft.name.trim() && draft.description.trim() && draft.content.trim();
  const dirty = draft.name !== skill.name
    || draft.description !== skill.description
    || draft.content !== skill.content;
  const requestClose = () => {
    if (busy) return;
    if (dirty) setConfirmDiscard(true);
    else onCancel();
  };
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={requestClose}>
      <form
        aria-labelledby="skill-editor-title"
        aria-modal="true"
        className="dialog dialog--wide"
        onMouseDown={(event) => event.stopPropagation()}
        onSubmit={(event) => {
          event.preventDefault();
          onSave(draft);
        }}
        role="dialog"
      >
        <h2 id="skill-editor-title">Revise {skill.name}</h2>
        <p>Saving creates version {skill.version + 1}. The revision stays inactive until you review and approve it again.</p>
        <Input
          disabled={busy}
          label="Skill name"
          maxLength={100}
          onChange={(event) => setDraft({ ...draft, name: event.target.value })}
          ref={nameRef}
          value={draft.name}
        />
        <Input
          disabled={busy}
          label="When should it be used?"
          maxLength={300}
          onChange={(event) => setDraft({ ...draft, description: event.target.value })}
          value={draft.description}
        />
        <Textarea
          disabled={busy}
          label="Reviewed instructions"
          maxLength={8000}
          onChange={(event) => setDraft({ ...draft, content: event.target.value })}
          rows={7}
          value={draft.content}
        />
        <div className="dialog__actions">
          {confirmDiscard && (
            <div role="alert">
              <strong>Discard unsaved skill changes?</strong>
              <span>Your edits have not been saved as a new version.</span>
              <Button onClick={() => setConfirmDiscard(false)} variant="quiet">Keep editing</Button>
              <Button onClick={onCancel} variant="danger">Discard changes</Button>
            </div>
          )}
          <Button disabled={busy} onClick={requestClose} variant="quiet">
            Cancel
          </Button>
          <Button disabled={busy || !valid} type="submit" variant="primary">
            {busy ? "Saving..." : "Save new version"}
          </Button>
        </div>
      </form>
    </div>
  );
}
