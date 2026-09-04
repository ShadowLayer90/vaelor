import { useEffect, useRef, useState } from "react";
import { ConfirmDialog } from "./ConfirmDialog";
import { ModalShell } from "./ModalShell";
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
  }, [skill]);

  if (!skill) return null;
  const valid = draft.name.trim() && draft.description.trim() && draft.content.trim();
  const dirty = draft.name !== skill.name
    || draft.description !== skill.description
    || draft.content !== skill.content;
  // Backdrop click and Escape both route here through ModalShell's onClose, and
  // the Tab focus trap comes with it. A dirty revision asks before it is lost.
  const requestClose = () => {
    if (busy) return;
    if (dirty) setConfirmDiscard(true);
    else onCancel();
  };
  return (
    <ModalShell
      className="skill-editor"
      initialFocusRef={nameRef}
      labelledBy="skill-editor-title"
      onClose={requestClose}
      size="standard"
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          onSave(draft);
        }}
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
          <Button disabled={busy} onClick={requestClose} variant="quiet">
            Cancel
          </Button>
          <Button disabled={busy || !valid} type="submit" variant="primary">
            {busy ? "Saving..." : "Save new version"}
          </Button>
        </div>
      </form>
      <ConfirmDialog
        busy={busy}
        confirmLabel="Discard changes"
        description="Your edits have not been saved as a new version."
        onCancel={() => setConfirmDiscard(false)}
        onConfirm={onCancel}
        open={confirmDiscard}
        title="Discard unsaved skill changes?"
      />
    </ModalShell>
  );
}
