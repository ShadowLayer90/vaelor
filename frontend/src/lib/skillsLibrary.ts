import { useCallback, useEffect, useState } from "react";
import { apiRequest } from "./api";

/**
 * Skills library (F6b): the typed data behind the "Agents & tools" fleet
 * section, beside the MCP catalog. A skill manifest names the capabilities a
 * deployed agent or model may assume - the scope/permission grants it needs and
 * the broker credential ids it draws on - and, for an administrator, is
 * registered, edited, enabled or removed here.
 *
 * The honesty rule the backend enforces is carried here: a skill row never holds
 * a secret. `credentials` are the ids of stored credentials, never their values,
 * and `grants` are scope/permission names - both read defensively, never
 * assumed. Attaching a skill to a target is a later phase (F6c); this surface is
 * the library only.
 */

/** One registered skill, exactly as the library returns it. Never a secret. */
export interface Skill {
  id: string;
  name: string;
  description: string;
  instructions: string;
  kind: string;
  enabled: boolean;
  grants: string[];
  credentials: string[];
  created_at: number | null;
  updated_at: number | null;
}

/**
 * The library surface: whether a skills library is configured on this appliance
 * at all, the registered skills, and an optional note the backend uses to
 * explain what a manifest means in its own words.
 */
export interface SkillsLibrary {
  configured: boolean;
  skills: Skill[];
  note: string;
}

/** The fields an administrator supplies to register or edit a skill. */
export interface SkillInput {
  name: string;
  kind: string;
  description: string;
  instructions: string;
  grants: string[];
  credentials: string[];
  enabled: boolean;
}

/** The wire shape of `GET /skills/library`. */
interface SkillsLibrarySurface {
  configured?: boolean;
  skills?: unknown;
  note?: string;
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function asNumberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asStringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

/**
 * Coerce one skill object from any of the routes into the {@link Skill} the
 * surface renders. The list, create and update routes all return a skill in the
 * same shape; a missing or wrong-typed field degrades to an empty default rather
 * than throwing, so a slightly older backend never blanks the whole surface.
 */
function normalizeSkill(value: unknown): Skill {
  const record = (value ?? {}) as Record<string, unknown>;
  return {
    id: asString(record.id),
    name: asString(record.name),
    description: asString(record.description),
    instructions: asString(record.instructions),
    kind: asString(record.kind),
    enabled: record.enabled === true,
    grants: asStringList(record.grants),
    credentials: asStringList(record.credentials),
    created_at: asNumberOrNull(record.created_at),
    updated_at: asNumberOrNull(record.updated_at),
  };
}

/** Read the library. Reports `configured: false` with no skills when unset. */
export async function listSkills(signal?: AbortSignal): Promise<SkillsLibrary> {
  const surface = await apiRequest<SkillsLibrarySurface>("/skills/library", { cache: "no-store", signal });
  return {
    configured: surface.configured === true,
    skills: Array.isArray(surface.skills) ? surface.skills.map(normalizeSkill) : [],
    note: typeof surface.note === "string" ? surface.note : "",
  };
}

/** Register a new skill (administrator). Returns the created skill. */
export async function registerSkill(input: SkillInput, csrfToken: string): Promise<Skill> {
  const skill = await apiRequest<unknown>(
    "/skills/library",
    { method: "POST", body: JSON.stringify(input), cache: "no-store" },
    csrfToken,
  );
  return normalizeSkill(skill);
}

/**
 * Apply a partial update to a skill (administrator) - a single `{ enabled }` for
 * the row toggle, or a set of edited fields. Returns the updated skill.
 */
export async function updateSkill(
  id: string,
  patch: Partial<SkillInput>,
  csrfToken: string,
): Promise<Skill> {
  const skill = await apiRequest<unknown>(
    `/skills/library/${encodeURIComponent(id)}`,
    { method: "POST", body: JSON.stringify(patch), cache: "no-store" },
    csrfToken,
  );
  return normalizeSkill(skill);
}

/** Remove a skill from the library (administrator). */
export async function deleteSkill(id: string, csrfToken: string): Promise<void> {
  await apiRequest<unknown>(
    `/skills/library/${encodeURIComponent(id)}`,
    { method: "DELETE", cache: "no-store" },
    csrfToken,
  );
}

/** One SKILL.md previewed from a GitHub repository, before it is registered. */
export interface ImportedSkill {
  path: string;
  name: string;
  description: string;
  instructions: string;
  /** Non-blocking notes from what was read (ignored frontmatter, files left behind). */
  warnings: string[];
  /**
   * Every reason the library's register rules would refuse this file, worked
   * out by the appliance before import. Non-empty means it cannot be imported.
   */
  refusals: string[];
}

/** The preview a repository import returns: the found skills and any warnings. */
export interface GithubImportResult {
  skills: ImportedSkill[];
  warnings: string[];
  ref: string;
  /**
   * True only when the appliance says it listed the whole repository. Anything
   * else (false, or a reply without the field) is treated as a partial listing,
   * so an empty result is never presented as "nothing is there".
   */
  listingComplete: boolean;
}

/** What an administrator supplies to preview a repository's SKILL.md files. */
export interface GithubImportInput {
  owner: string;
  repo: string;
  ref?: string;
  token?: string;
}

/**
 * Parse a repository reference an administrator typed - either an `owner/repo`
 * pair or a full `https://github.com/owner/repo` URL - into its owner and repo.
 * Returns `null` when the value is not a GitHub repository (a non-github host, a
 * missing segment, or a segment carrying characters GitHub does not allow), so
 * the caller can reject it clearly before any request reaches the network.
 */
export function parseGithubRepo(value: string): { owner: string; repo: string } | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const hosted = trimmed.match(/^(?:https?:\/\/)?(?:www\.)?github\.com\/(.+)$/i);
  let remainder: string;
  if (hosted) {
    remainder = hosted[1];
  } else if (trimmed.includes("://") || trimmed.split("/")[0].includes(".")) {
    // A URL or host-qualified reference that is not github.com: reject clearly.
    return null;
  } else {
    remainder = trimmed;
  }
  const segments = remainder.replace(/\.git$/i, "").split("/").filter(Boolean);
  if (segments.length < 2) return null;
  const [owner, repo] = segments;
  const allowed = /^[A-Za-z0-9._-]+$/;
  if (!allowed.test(owner) || !allowed.test(repo)) return null;
  return { owner, repo };
}

function normalizeImportedSkill(value: unknown): ImportedSkill {
  const record = (value ?? {}) as Record<string, unknown>;
  return {
    path: asString(record.path),
    name: asString(record.name),
    description: asString(record.description),
    instructions: asString(record.instructions),
    warnings: asStringList(record.warnings),
    refusals: asStringList(record.refusals),
  };
}

function normalizeImportResult(value: unknown): GithubImportResult {
  const record = (value ?? {}) as Record<string, unknown>;
  return {
    skills: Array.isArray(record.skills) ? record.skills.map(normalizeImportedSkill) : [],
    warnings: asStringList(record.warnings),
    ref: asString(record.ref),
    listingComplete: record.listing_complete === true,
  };
}

/**
 * Preview a GitHub repository's SKILL.md files (administrator). This never
 * registers anything; it returns the parsed candidates for the administrator to
 * review and then register through {@link registerSkill}. The optional token is
 * used for this one fetch only and is never persisted by the client.
 */
export async function importSkillsFromGithub(
  input: GithubImportInput,
  csrfToken: string,
): Promise<GithubImportResult> {
  const surface = await apiRequest<unknown>(
    "/skills/library/import/github",
    { method: "POST", body: JSON.stringify(input), cache: "no-store" },
    csrfToken,
  );
  return normalizeImportResult(surface);
}

/**
 * Split a free-text entry ("read_file, list_dir search") into a de-duplicated
 * list, tolerating commas, spaces and newlines as separators. It is the same
 * tokeniser the MCP catalog uses for its approved-tool list, reused here for a
 * skill's grants and credential ids so the two surfaces parse identically.
 */
export { parseApprovedTools as parseSkillEntries } from "./mcpCatalog";

export interface SkillsLibraryState {
  library: SkillsLibrary;
  loading: boolean;
  error: string;
  reload: () => void;
}

/**
 * Read the skills library, exposing the same three states the sibling admin
 * surfaces use: a first load, a read library, and a failure carrying the
 * appliance's own message. The request is abortable so leaving the tab never
 * lands a stale answer, and `reload` re-reads after a register, edit or removal.
 */
export function useSkillsLibrary(): SkillsLibraryState {
  const [library, setLibrary] = useState<SkillsLibrary>({ configured: false, skills: [], note: "" });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reloadKey, setReloadKey] = useState(0);
  const reload = useCallback(() => setReloadKey((value) => value + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    listSkills(controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return;
        setLibrary(result);
        setError("");
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        setError(reason instanceof Error ? reason.message : "The skills library could not be read.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [reloadKey]);

  return { library, loading, error, reload };
}
