import { useCallback, useState, type ReactNode } from "react";
import { ApiError } from "../lib/api";
import {
  deleteSkill,
  parseSkillEntries,
  registerSkill,
  updateSkill,
  useSkillsLibrary,
  type Skill,
  type SkillInput,
} from "../lib/skillsLibrary";
import type { Session } from "../types";
import { useModalAction } from "../hooks/useModalAction";
import { OptionCard } from "./ClusterPrimitives";
import { ClusterDialog } from "./ClusterDialog";
import { Icon } from "./Icon";
import { ToolTags } from "./McpCatalogPanel";
import { ClusterConfirm } from "./ClusterConfirm";
import { SkillsGithubImportModal } from "./SkillsGithubImportModal";
import { StatusPill } from "./StatusPill";
import { Button, EmptyState, Input, LoadingLines, Notice, Select, Textarea } from "./ui";
import "../styles/cluster-agents.css";

/**
 * Skills library (F6b): the "Agents & tools" surface, beside the MCP catalog,
 * that lists the skill manifests a deployed agent or model may draw on, and lets
 * an administrator register, edit, enable and remove them.
 *
 * Each skill shows its kind, whether it is enabled, and - read only - the
 * capability grants and broker credential ids it assumes. A skill never shows a
 * secret: `credentials` are ids only. Attaching a skill to a target is a later
 * phase; this surface is the library itself.
 *
 * Reading the library is an operator capability; every change is an
 * administrator one, so a non-administrator sees the library in full but the
 * register action and per-row controls are withheld.
 *
 * VD-200 (the ClusterAgentsSkills and ClusterDialogsTools boards): the skills
 * are cards in a three-up grid, and the editor and the removal are the Cluster
 * page's own dialogs.
 */

/** The editor's own field state, kept as strings until it is submitted. */
interface EditorForm {
  name: string;
  kind: string;
  description: string;
  instructions: string;
  grants: string;
  credentials: string;
  enabled: boolean;
}

const EMPTY_FORM: EditorForm = {
  name: "",
  kind: "skill",
  description: "",
  instructions: "",
  grants: "",
  credentials: "",
  enabled: true,
};

const ADMIN_ONLY_NOTE =
  "You can view the library. Registering, editing, enabling or removing a skill needs an administrator account.";
const UNCONFIGURED_NOTE =
  "No skills library is configured on this appliance yet. Once it is, the registered skills and the capabilities they assume appear here.";
const KIND_LOCKED_NOTE = "A skill's kind cannot be changed after it is registered.";

function toEditorForm(skill: Skill): EditorForm {
  return {
    name: skill.name,
    kind: skill.kind || "skill",
    description: skill.description,
    instructions: skill.instructions,
    grants: skill.grants.join(", "),
    credentials: skill.credentials.join(", "),
    enabled: skill.enabled,
  };
}

export function SkillsLibraryPanel({ session }: { session: Session }) {
  const { library, loading, error, reload } = useSkillsLibrary();
  const isAdmin = session.user.role === "administrator";
  const [busy, setBusy] = useState("");
  const [actionError, setActionError] = useState("");
  const [editorOpen, setEditorOpen] = useState(false);
  // W4d-D16: the editor's own refusal, shown inside it (the dialog's `error`).
  // It used to land in `actionError`, on the inert page under the modal.
  const [editorError, setEditorError] = useState("");
  const [editing, setEditing] = useState<Skill | null>(null);
  const [form, setForm] = useState<EditorForm>(EMPTY_FORM);
  const [removeTarget, setRemoveTarget] = useState<Skill | null>(null);
  const [importOpen, setImportOpen] = useState(false);

  const fail = useCallback((cause: unknown, fallback: string) => {
    if (!(cause instanceof ApiError)) console.error(fallback, cause);
    setActionError(cause instanceof ApiError ? cause.message : fallback);
  }, []);

  const closeEditor = useCallback(() => {
    setEditorError("");
    setEditorOpen(false);
    setEditing(null);
    setForm(EMPTY_FORM);
  }, []);

  const openRegister = useCallback(() => {
    setEditing(null);
    setForm(EMPTY_FORM);
    setActionError("");
    setEditorOpen(true);
  }, []);

  const openImport = useCallback(() => {
    setActionError("");
    setImportOpen(true);
  }, []);

  const openEdit = useCallback((skill: Skill) => {
    setEditing(skill);
    setForm(toEditorForm(skill));
    setActionError("");
    setEditorOpen(true);
  }, []);

  const submitEditor = useCallback(async () => {
    setBusy("save");
    setEditorError("");
    try {
      const grants = parseSkillEntries(form.grants);
      const credentials = parseSkillEntries(form.credentials);
      if (editing) {
        // A skill's kind is immutable after enrolment, so it is never sent in an
        // edit; every other mutable field rides along.
        const patch: Partial<SkillInput> = {
          name: form.name.trim(),
          description: form.description.trim(),
          instructions: form.instructions,
          grants,
          credentials,
          enabled: form.enabled,
        };
        await updateSkill(editing.id, patch, session.csrf_token);
      } else {
        const input: SkillInput = {
          name: form.name.trim(),
          kind: form.kind,
          description: form.description.trim(),
          instructions: form.instructions,
          grants,
          credentials,
          enabled: form.enabled,
        };
        await registerSkill(input, session.csrf_token);
      }
      closeEditor();
      reload();
    } catch (cause) {
      const fallback = editing ? "The skill could not be updated." : "The skill could not be registered.";
      setEditorError(cause instanceof Error && cause.message.trim() ? cause.message : fallback);
    } finally {
      setBusy("");
    }
  }, [closeEditor, editing, form, reload, session.csrf_token]);

  const toggle = useCallback(async (skill: Skill) => {
    setBusy("toggle:" + skill.id);
    setActionError("");
    try {
      await updateSkill(skill.id, { enabled: !skill.enabled }, session.csrf_token);
      reload();
    } catch (cause) {
      fail(cause, "That change could not be applied.");
    } finally {
      setBusy("");
    }
  }, [fail, reload, session.csrf_token]);

  // R-F2 (LESSONS 19, VD-189): a refused removal stays in its dialog. It used
  // to go through `fail`, onto the page under the open, inert-making dialog.
  const removal = useModalAction();
  const confirmRemove = useCallback(async () => {
    if (!removeTarget) return;
    const target = removeTarget;
    setBusy("remove:" + target.id);
    const removed = await removal.run(() => deleteSkill(target.id, session.csrf_token));
    setBusy("");
    if (!removed) return;
    setRemoveTarget(null);
    reload();
  }, [reload, removal, removeTarget, session.csrf_token]);

  const editorInvalid = !form.name.trim() || !form.description.trim();
  const skills = library.skills;

  let body: ReactNode;
  if (loading && !skills.length) {
    body = (
      <div className="ui-card cl-tool__loading">
        <p aria-hidden="true">Reading the skills library...</p>
        <LoadingLines label="Reading the skills library" />
      </div>
    );
  } else if (error && !skills.length) {
    body = <Notice severity="danger">{error}</Notice>;
  } else if (!library.configured && !skills.length) {
    body = <Notice severity="info">{library.note || UNCONFIGURED_NOTE}</Notice>;
  } else if (!skills.length) {
    body = (
      <div className="ui-card cl-tool__empty">
        <EmptyState
          icon={<Icon name="file" />}
          text={isAdmin
            ? "Register a skill to describe a capability an agent can draw on."
            : "An administrator can register a skill to describe a capability an agent can draw on."}
          title="No skills registered"
        />
      </div>
    );
  } else {
    body = (
      <>
        {library.note && <Notice severity="info">{library.note}</Notice>}
        <div className="cl-grid cl-grid--three cl-tool__grid">
          {skills.map((skill) => (
            <article aria-label={skill.name || "Skill"} className="ui-card cl-tool" key={skill.id}>
              <div className="cl-tool__top">
                <div className="cl-tool__name">
                  <strong>{skill.name || "Unnamed skill"}</strong>
                  <small>{skill.kind || "skill"}</small>
                </div>
                <div className="cl-tool__pills">
                  <StatusPill label={skill.enabled ? "Enabled" : "Disabled"} tone={skill.enabled ? "success" : "neutral"} />
                </div>
              </div>

              {skill.description && <p className="cl-tool__description">{skill.description}</p>}

              <div className="cl-tool__section">
                <div className="cl-tool__section-head">
                  <h4>Grants</h4>
                  <span>{skill.grants.length}</span>
                </div>
                <ToolTags empty="No grants - this skill assumes no scopes or permissions." items={skill.grants} />
              </div>

              <div className="cl-tool__section">
                <div className="cl-tool__section-head">
                  <h4>Credentials</h4>
                  <span>{skill.credentials.length}</span>
                </div>
                <ToolTags empty="No credentials - this skill references no stored credential." items={skill.credentials} />
              </div>

              {isAdmin && (
                <div className="cl-actions cl-tool__actions">
                  <Button disabled={busy !== ""} onClick={() => void toggle(skill)}>
                    {busy === "toggle:" + skill.id
                      ? (skill.enabled ? "Disabling..." : "Enabling...")
                      : (skill.enabled ? "Disable" : "Enable")}
                  </Button>
                  <Button disabled={busy !== ""} onClick={() => openEdit(skill)} variant="quiet">
                    Edit
                  </Button>
                  <Button
                    aria-label={"Remove " + (skill.name || "skill")}
                    className="cl-danger-outline"
                    disabled={busy !== ""}
                    onClick={() => setRemoveTarget(skill)}
                  >
                    Remove
                  </Button>
                </div>
              )}
            </article>
          ))}
        </div>
      </>
    );
  }

  return (
    <section aria-labelledby="skills-library-title" className="cl-stack cl-tools">
      <div className="cl-tools__head">
        <div className="cl-tools__head-text">
          <h3 id="skills-library-title">Skills library</h3>
          <p>Reusable skills a deployed agent or model may draw on. Each names the capabilities it assumes.</p>
        </div>
        {isAdmin && (
          <div className="cl-actions">
            <Button disabled={busy !== ""} onClick={openImport}>
              <Icon name="download" />Import from GitHub
            </Button>
            <Button disabled={busy !== ""} onClick={openRegister} variant="primary">
              <Icon name="add" />Register skill
            </Button>
          </div>
        )}
      </div>

      {actionError && <Notice severity="danger">{actionError}</Notice>}
      {!isAdmin && <Notice severity="info">{ADMIN_ONLY_NOTE}</Notice>}

      {body}

      {editorOpen && (
        <ClusterDialog
          error={editorError}
          eyebrow="Skills library"
          footer={<>
            <Button disabled={busy !== ""} onClick={closeEditor}>Cancel</Button>
            <Button
              disabled={busy !== ""}
              disabledReason={editorInvalid ? "Enter a name and a description to save the skill." : undefined}
              form="skill-editor-form"
              type="submit"
              variant="primary"
            >
              {busy === "save"
                ? (editing ? "Saving..." : "Registering...")
                : (editing ? "Save changes" : "Register skill")}
            </Button>
          </>}
          onClose={() => { if (busy === "") closeEditor(); }}
          title={editing ? "Edit skill" : "Register skill"}
          titleId="skill-editor-title"
        >
          <p className="cl-meta">
            Attaching a skill records what it assumes; for a deployed cluster agent its read scopes are
            added to what the agent can read, and its name and description to the agent's guidance. It
            stays gated read-only - no acting permission or credential is conferred.
          </p>
          <form
            className="cl-tool-form"
            id="skill-editor-form"
            onSubmit={(event) => {
              event.preventDefault();
              if (!editorInvalid) void submitEditor();
            }}
          >
            <Input
              id="skill-name"
              label="Name"
              maxLength={80}
              onChange={(event) => setForm((current) => ({ ...current, name: event.target.value }))}
              placeholder="Example: cluster-reader"
              value={form.name}
            />
            <div className="cl-tool-form__kind">
              <Select
                disabled={Boolean(editing)}
                disabledReason={editing ? KIND_LOCKED_NOTE : undefined}
                hint={editing ? undefined : "A skill is a set of instructions; a plugin adds a callable capability."}
                id="skill-kind"
                label="Kind"
                onChange={(event) => setForm((current) => ({ ...current, kind: event.target.value }))}
                value={form.kind}
              >
                <option value="skill">Skill</option>
                <option value="plugin">Plugin</option>
              </Select>
            </div>
            <Textarea
              id="skill-description"
              label="Description"
              maxLength={2000}
              onChange={(event) => setForm((current) => ({ ...current, description: event.target.value }))}
              placeholder="Reads the current fleet performance metrics for a node."
              rows={2}
              value={form.description}
            />
            <Textarea
              id="skill-instructions"
              label="Instructions (the skill's how-to, given to the agent)"
              maxLength={50000}
              onChange={(event) => setForm((current) => ({ ...current, instructions: event.target.value }))}
              placeholder="How the agent should carry out this skill."
              rows={2}
              value={form.instructions}
            />
            <div className="cl-tool-form__pair">
              <Input
                id="skill-grants"
                label="Grants"
                maxLength={2000}
                onChange={(event) => setForm((current) => ({ ...current, grants: event.target.value }))}
                placeholder="cluster:read, system:read"
                value={form.grants}
              />
              <Input
                id="skill-credentials"
                label="Credentials"
                maxLength={2000}
                onChange={(event) => setForm((current) => ({ ...current, credentials: event.target.value }))}
                placeholder="Example: cred_metrics"
                value={form.credentials}
              />
            </div>
            <p className="cl-meta">
              Separate names with commas or spaces. A grant must be a known scope or permission; a
              credential is a broker id, never a secret value.
            </p>
            <OptionCard
              checked={form.enabled}
              onChange={(event) => setForm((current) => ({ ...current, enabled: event.target.checked }))}
              title="Enable this skill now"
              type="checkbox"
            />
          </form>
        </ClusterDialog>
      )}

      {importOpen && isAdmin && (
        <SkillsGithubImportModal
          csrfToken={session.csrf_token}
          onClose={() => setImportOpen(false)}
          onImported={reload}
        />
      )}

      <ClusterConfirm
        busy={Boolean(removeTarget && busy === "remove:" + removeTarget.id)}
        confirmLabel="Remove skill"
        description={removeTarget
          ? (removeTarget.name || "This skill") + " will be removed from the library and can no longer be attached to an agent or model. Any stored credential it referenced is left untouched."
          : ""}
        error={removal.error}
        eyebrow="Remove"
        onCancel={() => {
          if (busy !== "") return;
          removal.clear();
          setRemoveTarget(null);
        }}
        onConfirm={() => void confirmRemove()}
        open={Boolean(removeTarget)}
        title="Remove this skill?"
      />
    </section>
  );
}
