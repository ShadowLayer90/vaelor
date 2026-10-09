import { useCallback, useMemo, useRef, useState } from "react";
import { ApiError } from "../lib/api";
import {
  importSkillsFromGithub,
  parseGithubRepo,
  registerSkill,
  type ImportedSkill,
} from "../lib/skillsLibrary";
import { ClusterDialog } from "./ClusterDialog";
import { Button, Checkbox, Input, Notice } from "./ui";
import "../styles/cluster-agents.css";

/**
 * Import from GitHub (F6b): preview a public (or token-reached private) GitHub
 * repository's SKILL.md files and register the chosen ones as cluster skills.
 *
 * The preview never registers anything; the administrator reviews the found
 * files, deselects any they do not want, and only "Import selected" writes them
 * through the ordinary register route (one call per skill, so each is screened).
 * The appliance screens every found file against those same register rules
 * before import: a file the library would refuse arrives with its reasons, is
 * not preselected, cannot be selected, and says why. A refusal that still
 * happens at import (the library changed in between) is reported as a problem,
 * never as a quiet "skipped" under a success tone.
 * An imported skill arrives with no grants and the kind "skill" - a SKILL.md
 * declares none - and grants can be added afterward with the per-row Edit.
 *
 * The optional access token is used only for this one fetch: it is a password
 * field, is never written to storage, and is dropped when the modal closes.
 *
 * VD-200 (the ClusterDialogsImport board): the Cluster page's wide dialog. The
 * three fields sit on one row, Fetch again stays in the body once a preview is
 * shown, and the footer carries the step's own action.
 */

/** How many characters of an instructions body the row preview shows. */
const PREVIEW_LIMIT = 200;

/** One skill's fate after the batch: registered, or refused with the library's reason. */
interface ImportOutcome {
  path: string;
  name: string;
  status: "imported" | "refused";
  reason: string;
}

/** A found file the library's register rules would accept. */
function importable(skill: ImportedSkill): boolean {
  return skill.refusals.length === 0;
}

/** The one-line account of a finished batch, and how serious it is. */
function outcomeSummary(imported: number, refused: number): {
  severity: "success" | "warning" | "danger";
  text: string;
} {
  const skillsWord = (count: number) => count + (count === 1 ? " skill" : " skills");
  if (refused === 0) return { severity: "success", text: "Imported " + skillsWord(imported) + "." };
  if (imported === 0) {
    return {
      severity: "danger",
      text: "Nothing was imported: the library refused " + (refused === 1 ? "the selected skill." : "all " + refused + " selected skills."),
    };
  }
  return {
    severity: "warning",
    text: "Imported " + skillsWord(imported) + "; the library refused " + refused + ".",
  };
}

/** Collapse whitespace and truncate an instructions body for the row preview. */
function instructionsPreview(instructions: string): string {
  const collapsed = instructions.replace(/\s+/g, " ").trim();
  return collapsed.length > PREVIEW_LIMIT
    ? collapsed.slice(0, PREVIEW_LIMIT) + "…"
    : collapsed;
}

export function SkillsGithubImportModal({
  csrfToken,
  onClose,
  onImported,
}: {
  csrfToken: string;
  onClose: () => void;
  onImported: () => void;
}) {
  const [repository, setRepository] = useState("");
  const [refInput, setRefInput] = useState("");
  const [token, setToken] = useState("");
  const [fetching, setFetching] = useState(false);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState("");
  const [found, setFound] = useState<ImportedSkill[] | null>(null);
  const [repoWarnings, setRepoWarnings] = useState<string[]>([]);
  const [resolvedRef, setResolvedRef] = useState("");
  const [listingComplete, setListingComplete] = useState(true);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [summary, setSummary] = useState<ImportOutcome[] | null>(null);
  const repositoryRef = useRef<HTMLInputElement>(null);

  const busy = fetching || importing;

  const requestClose = useCallback(() => {
    if (busy) return;
    // The token lived only in this component's state; clearing it here is the
    // whole of "never stored" - nothing was written anywhere else.
    setToken("");
    onClose();
  }, [busy, onClose]);

  const selectedSkills = useMemo(
    () => (found ?? []).filter((skill) => importable(skill) && selected[skill.path]),
    [found, selected],
  );

  const runFetch = useCallback(async () => {
    const parsed = parseGithubRepo(repository);
    if (!parsed) {
      setError(
        "Enter a GitHub repository as owner/repo or a https://github.com/owner/repo URL.",
      );
      setFound(null);
      setSummary(null);
      return;
    }
    setFetching(true);
    setError("");
    setSummary(null);
    try {
      const result = await importSkillsFromGithub(
        {
          owner: parsed.owner,
          repo: parsed.repo,
          ref: refInput.trim() || undefined,
          token: token.trim() || undefined,
        },
        csrfToken,
      );
      setFound(result.skills);
      setRepoWarnings(result.warnings);
      setResolvedRef(result.ref);
      setListingComplete(result.listingComplete);
      // Only a file the library would accept starts selected; a refused one
      // shows its reasons instead of failing later at import.
      const preselected: Record<string, boolean> = {};
      for (const skill of result.skills) preselected[skill.path] = importable(skill);
      setSelected(preselected);
    } catch (cause) {
      if (!(cause instanceof ApiError)) console.error("The repository could not be read.", cause);
      setError(cause instanceof ApiError ? cause.message : "The repository could not be read.");
      setFound(null);
    } finally {
      setFetching(false);
    }
  }, [csrfToken, refInput, repository, token]);

  const runImport = useCallback(async () => {
    setImporting(true);
    setError("");
    const outcomes: ImportOutcome[] = [];
    // Sequential on purpose: each register is screened server-side, and one
    // refusal (say, a name registered since the preview) must not abort the batch.
    for (const skill of selectedSkills) {
      try {
        await registerSkill(
          {
            name: skill.name,
            kind: "skill",
            description: skill.description,
            instructions: skill.instructions,
            grants: [],
            credentials: [],
            enabled: true,
          },
          csrfToken,
        );
        outcomes.push({ path: skill.path, name: skill.name, status: "imported", reason: "" });
      } catch (cause) {
        outcomes.push({
          path: skill.path,
          name: skill.name,
          status: "refused",
          reason: cause instanceof ApiError ? cause.message : "The skill could not be registered.",
        });
      }
    }
    setImporting(false);
    setSummary(outcomes);
    onImported();
  }, [csrfToken, onImported, selectedSkills]);

  const importedCount = summary?.filter((item) => item.status === "imported").length ?? 0;
  const refusedCount = summary?.filter((item) => item.status === "refused").length ?? 0;
  const foundCount = found?.length ?? 0;
  const blockedCount = (found ?? []).filter((skill) => !importable(skill)).length;
  const selectedCount = selectedSkills.length;
  const finished = outcomeSummary(importedCount, refusedCount);

  const fetchLabel = fetching ? "Fetching skills…" : found === null ? "Fetch skills" : "Fetch again";
  const footer = summary ? (
    <Button onClick={requestClose} variant="primary">Close</Button>
  ) : found === null ? (
    <>
      <Button disabled={busy} onClick={requestClose}>Cancel</Button>
      <Button busy={fetching} disabled={busy} form="skills-import-form" type="submit" variant="primary">{fetchLabel}</Button>
    </>
  ) : (
    <>
      <Button disabled={busy} onClick={requestClose}>Cancel</Button>
      <Button
        busy={importing}
        disabled={busy || selectedCount === 0}
        disabledReason={selectedCount === 0
          ? (foundCount > 0 && blockedCount === foundCount
            ? "None of these files can be imported; each one says why."
            : "Select at least one skill to import.")
          : undefined}
        onClick={() => void runImport()}
        variant="primary"
      >
        {importing ? "Importing…" : "Import selected" + (selectedCount ? " (" + selectedCount + ")" : "")}
      </Button>
    </>
  );

  return (
    <ClusterDialog
      className="cl-import"
      error={error}
      eyebrow="Skills library"
      footer={footer}
      initialFocusRef={summary ? undefined : repositoryRef}
      onClose={requestClose}
      size="wide"
      title="Import skills from GitHub"
      titleId="skills-import-title"
    >
      {!summary && (
        <>
          <p>
            Preview a repository laid out as <code>skills/&lt;name&gt;/SKILL.md</code> and register the
            files you choose. Imported skills arrive with no grants and the kind &ldquo;skill&rdquo;; add
            grants afterward with Edit. Nothing is registered until you choose Import selected.
          </p>
          <form
            className="cl-import__form"
            id="skills-import-form"
            onSubmit={(event) => {
              event.preventDefault();
              if (!busy) void runFetch();
            }}
          >
            <div className="cl-import__fields">
              <Input
                hint="An owner/repo pair, or a https://github.com/owner/repo URL."
                id="skills-import-repository"
                label="Repository"
                onChange={(event) => setRepository(event.target.value)}
                placeholder="Example: example-org/skills"
                ref={repositoryRef}
                value={repository}
              />
              <Input
                hint="Leave empty to use the repository's default branch."
                id="skills-import-ref"
                label="Branch or ref (optional)"
                onChange={(event) => setRefInput(event.target.value)}
                placeholder="Example: main"
                value={refInput}
              />
              <Input
                hint="For a private repository only. Used for this fetch alone and never stored."
                id="skills-import-token"
                label="Access token (optional)"
                onChange={(event) => setToken(event.target.value)}
                type="password"
                value={token}
              />
            </div>
            {found !== null && (
              <div className="cl-actions cl-actions--end">
                <Button busy={fetching} disabled={busy} type="submit">{fetchLabel}</Button>
              </div>
            )}
          </form>
        </>
      )}

      {found !== null && !summary && (
        <div className="cl-import__results">
          {repoWarnings.length > 0 && (
            <Notice severity="warning">
              <ul className="cl-import__inline-list">
                {repoWarnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </Notice>
          )}

          {foundCount === 0 ? (
            listingComplete ? (
              <Notice severity="info">
                No SKILL.md files were found in {resolvedRef ? "the " + resolvedRef + " ref of " : ""}
                this repository.
              </Notice>
            ) : (
              <Notice severity="warning">
                No SKILL.md files were found in the part of this repository that could be listed.
                The rest may still hold some.
              </Notice>
            )
          ) : (
            <>
              <div className="cl-import__summary">
                <span>
                  Found {foundCount} SKILL.md {foundCount === 1 ? "file" : "files"}
                  {resolvedRef ? " on " + resolvedRef : ""}.
                  {blockedCount > 0 &&
                    " " + blockedCount + (blockedCount === 1 ? " cannot" : " of them cannot") +
                    " be imported; each one says why."}
                </span>
                <span>{selectedCount} selected</span>
              </div>
              <ul className="cl-import__list">
                {found.map((skill) => {
                  const chosen = importable(skill) && Boolean(selected[skill.path]);
                  return (
                    <li className="cl-import__item" key={skill.path}>
                      <div className={"cl-import__pick" + (chosen ? " is-on" : "") + (importable(skill) ? "" : " is-disabled")}>
                        <Checkbox
                          checked={chosen}
                          disabled={!importable(skill)}
                          id={"skills-import-pick-" + skill.path}
                          label={skill.name || "Unnamed skill"}
                          onChange={(event) =>
                            setSelected((current) => ({
                              ...current,
                              [skill.path]: event.target.checked,
                            }))
                          }
                        />
                      </div>
                      {skill.description && <p>{skill.description}</p>}
                      <code className="cl-import__path">{skill.path}</code>
                      {skill.instructions ? (
                        <div className="cl-import__preview">
                          <span>{instructionsPreview(skill.instructions)}</span>
                          <details className="cl-import__full">
                            <summary>Full instructions</summary>
                            <pre>{skill.instructions}</pre>
                          </details>
                        </div>
                      ) : (
                        <p className="cl-import__preview">This skill has no instructions body.</p>
                      )}
                      {skill.warnings.length > 0 && (
                        <ul className="cl-import__warnings">
                          {skill.warnings.map((warning) => (
                            <li key={warning}>{warning}</li>
                          ))}
                        </ul>
                      )}
                      {!importable(skill) && (
                        <p className="cl-warn-text cl-import__refusal">{"Cannot be imported: " + skill.refusals.join(" ")}</p>
                      )}
                    </li>
                  );
                })}
              </ul>
            </>
          )}
        </div>
      )}

      {summary && (
        <div className="cl-import__results">
          <Notice severity={finished.severity}>{finished.text}</Notice>
          {refusedCount > 0 && (
            <ul className="cl-import__refusals">
              {summary
                .filter((item) => item.status === "refused")
                .map((item) => (
                  <li key={item.path}>
                    {(item.name || item.path) + ": " + item.reason}
                  </li>
                ))}
            </ul>
          )}
        </div>
      )}
    </ClusterDialog>
  );
}
