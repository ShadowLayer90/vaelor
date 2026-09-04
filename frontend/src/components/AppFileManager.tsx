import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiRequest, downloadApiRequest } from "../lib/api";
import { formatQuantity } from "../lib/format";
import type { Role } from "../types";
import { ConfirmDialog } from "./ConfirmDialog";
import { Icon } from "./Icon";
import { Button, Input, Notice, OperationFeedback, type OperationState } from "./ui";

/** One entry in a directory listing: `d` = folder, `f` = file. */
interface FsEntry {
  name: string;
  type: "d" | "f";
  size: number;
}

interface FsListing {
  roots: string[];
  path: string;
  entries: FsEntry[];
}

/** Server-enforced per-file upload cap; pre-checked here for a friendlier error. */
const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;

/** Join a directory to a child name without minting a double slash at the root. */
function joinPath(base: string, name: string): string {
  return base.endsWith("/") ? base + name : `${base}/${name}`;
}

/**
 * The root that contains `path`, and the segments of `path` below it. The
 * breadcrumb never offers a crumb above this root, which is how "never navigate
 * above the current root" is enforced: the leftmost crumb is the root itself.
 */
function locate(roots: string[], path: string): { root: string; segments: string[] } {
  const root = roots.find((candidate) => path === candidate || path.startsWith(candidate.endsWith("/") ? candidate : `${candidate}/`)) ?? roots[0] ?? "";
  const relative = path.slice(root.length).replace(/^\/+/, "");
  return { root, segments: relative ? relative.split("/") : [] };
}

/** Folders first, then files, each alphabetised case-insensitively. */
function ordered(entries: FsEntry[]): FsEntry[] {
  return [...entries].sort((a, b) => {
    if (a.type !== b.type) return a.type === "d" ? -1 : 1;
    return a.name.localeCompare(b.name, undefined, { sensitivity: "base" });
  });
}

export function AppFileManager({
  appId,
  role,
  csrfToken,
}: {
  appId: string;
  role: Role;
  csrfToken: string;
}) {
  const isAdmin = role === "administrator";
  const [roots, setRoots] = useState<string[]>([]);
  const [path, setPath] = useState("");
  const [entries, setEntries] = useState<FsEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [noticeState, setNoticeState] = useState<OperationState>("idle");
  const [busy, setBusy] = useState(false);
  const [uploadPercent, setUploadPercent] = useState<number | null>(null);
  const [creatingFolder, setCreatingFolder] = useState(false);
  const [folderName, setFolderName] = useState("");
  const [pendingDelete, setPendingDelete] = useState<FsEntry | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const setFeedback = (message: string, state: OperationState = "success") => {
    setNotice(message);
    setNoticeState(message ? state : "idle");
  };

  // A directory the listing reported as loaded; `undefined` requests the
  // server's default root. `null` means nothing loaded yet (first render).
  const list = useCallback(async (target?: string) => {
    setLoading(true);
    setError("");
    try {
      const query = target !== undefined ? `?path=${encodeURIComponent(target)}` : "";
      const data = await apiRequest<FsListing>(`/managed/apps/${appId}/fs/list${query}`);
      setRoots(data.roots ?? []);
      setPath(data.path ?? "");
      setEntries(data.entries ?? []);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "This app's files could not be listed.");
    } finally {
      setLoading(false);
    }
  }, [appId]);

  useEffect(() => {
    // The backend gates every /fs/* route on administrator, so a non-admin
    // request would only 403. Mirror the Console tab: never fetch, and show a
    // calm access panel instead of an error.
    if (!isAdmin) {
      setLoading(false);
      return;
    }
    void list();
  }, [isAdmin, list]);

  const { root, segments } = useMemo(() => locate(roots, path), [roots, path]);
  const sorted = useMemo(() => ordered(entries), [entries]);

  const openFolder = (name: string) => {
    setFeedback("");
    setCreatingFolder(false);
    void list(joinPath(path, name));
  };

  const navigateTo = (target: string) => {
    setFeedback("");
    setCreatingFolder(false);
    void list(target);
  };

  const download = async (entry: FsEntry) => {
    setFeedback("");
    try {
      await downloadApiRequest(
        `/managed/apps/${appId}/fs/download?path=${encodeURIComponent(joinPath(path, entry.name))}`,
        entry.name,
      );
    } catch (caught) {
      setFeedback(caught instanceof Error ? caught.message : "That file could not be downloaded.", "error");
    }
  };

  const createFolder = async () => {
    const name = folderName.trim();
    if (!name || busy) return;
    setBusy(true);
    setFeedback("");
    try {
      await apiRequest(
        `/managed/apps/${appId}/fs/mkdir`,
        { method: "POST", body: JSON.stringify({ path, name }) },
        csrfToken,
      );
      setCreatingFolder(false);
      setFolderName("");
      setFeedback(`Folder "${name}" created.`);
      await list(path);
    } catch (caught) {
      setFeedback(caught instanceof Error ? caught.message : "The folder could not be created.", "error");
    } finally {
      setBusy(false);
    }
  };

  const confirmDelete = async () => {
    if (!pendingDelete || busy) return;
    const entry = pendingDelete;
    setBusy(true);
    setFeedback("");
    try {
      await apiRequest(
        `/managed/apps/${appId}/fs/delete`,
        { method: "POST", body: JSON.stringify({ path: joinPath(path, entry.name) }) },
        csrfToken,
      );
      setPendingDelete(null);
      setFeedback(`${entry.type === "d" ? "Folder" : "File"} "${entry.name}" deleted.`);
      await list(path);
    } catch (caught) {
      setFeedback(caught instanceof Error ? caught.message : "That item could not be deleted.", "error");
      setPendingDelete(null);
    } finally {
      setBusy(false);
    }
  };

  /**
   * Upload runs on XMLHttpRequest rather than the shared fetch helper so the bar
   * can show real byte progress; it still sends the CSRF header and credentials
   * and parses the standard `{ error: { message } }` envelope on failure.
   */
  const upload = (file: File) =>
    new Promise<void>((resolve, reject) => {
      const form = new FormData();
      form.append("path", path);
      form.append("file", file);
      const request = new XMLHttpRequest();
      request.open("POST", `/api/v2/managed/apps/${appId}/fs/upload`);
      request.withCredentials = true;
      request.setRequestHeader("Accept", "application/json");
      request.setRequestHeader("X-CSRF-Token", csrfToken);
      request.upload.onprogress = (event) => {
        if (event.lengthComputable) setUploadPercent(Math.round((event.loaded / event.total) * 100));
      };
      const fail = (fallback: string) => {
        let message = fallback;
        try {
          message = JSON.parse(request.responseText)?.error?.message ?? fallback;
        } catch {
          // Keep the fallback for a non-JSON proxy error.
        }
        reject(new Error(message));
      };
      request.onload = () => {
        if (request.status >= 200 && request.status < 300) resolve();
        else fail("The file could not be uploaded.");
      };
      request.onerror = () => fail("The upload did not reach the appliance. Try again.");
      request.send(form);
    });

  const onFileChosen = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    // Allow re-choosing the same file after an error by clearing the input.
    event.target.value = "";
    if (!file) return;
    setFeedback("");
    if (file.size > MAX_UPLOAD_BYTES) {
      setFeedback(`"${file.name}" is ${formatQuantity(file.size, "used")}, over the 100 MB per-file limit.`, "error");
      return;
    }
    setBusy(true);
    setUploadPercent(0);
    try {
      await upload(file);
      setFeedback(`"${file.name}" uploaded.`);
      await list(path);
    } catch (caught) {
      setFeedback(caught instanceof Error ? caught.message : "The file could not be uploaded.", "error");
    } finally {
      setBusy(false);
      setUploadPercent(null);
    }
  };

  const noStorage = !loading && !error && roots.length === 0;

  return (
    <section className="app-file-manager" aria-labelledby="app-file-manager-title">
      <div className="tool-explainer">
        <Icon name="database" />
        <span>
          <strong id="app-file-manager-title">Files</strong>
          <small>Browse this app&apos;s stored files. Upload, download, create folders, and delete items. Deleting is permanent.</small>
        </span>
      </div>

      {!isAdmin ? (
        <div className="app-file-manager__empty" role="status">
          <Icon name="lock" size={30} />
          <p>Administrator access is required to manage this app&apos;s files.</p>
        </div>
      ) : (
      <>
      {error && (
        <Notice severity="danger">
          <span>{error}</span>
          <Button onClick={() => void list(path || undefined)}>Try again</Button>
        </Notice>
      )}

      {loading && roots.length === 0 && !error && <p className="app-file-manager__note" role="status">Loading files…</p>}

      {noStorage ? (
        <div className="app-file-manager__empty" role="status">
          <Icon name="folder" size={30} />
          <p>This app has no browsable storage.</p>
        </div>
      ) : roots.length > 0 && (
        <>
          {roots.length > 1 && (
            <div className="app-file-manager__roots" role="tablist" aria-label="Storage locations">
              {roots.map((candidate) => (
                <Button
                  key={candidate}
                  role="tab"
                  aria-selected={candidate === root}
                  variant={candidate === root ? "primary" : "quiet"}
                  disabled={busy}
                  onClick={() => navigateTo(candidate)}
                >
                  {candidate}
                </Button>
              ))}
            </div>
          )}

          <div className="app-file-manager__toolbar">
            <nav className="app-file-manager__breadcrumb" aria-label="Current folder">
              <Button variant="quiet" disabled={busy} onClick={() => navigateTo(root)}>{root || "/"}</Button>
              {segments.map((segment, index) => {
                const target = joinPath(root, segments.slice(0, index + 1).join("/"));
                const isCurrent = index === segments.length - 1;
                return (
                  <span className="app-file-manager__crumb" key={target}>
                    <Icon name="chevron" size={14} />
                    <Button
                      variant="quiet"
                      disabled={busy || isCurrent}
                      aria-current={isCurrent ? "page" : undefined}
                      onClick={() => navigateTo(target)}
                    >
                      {segment}
                    </Button>
                  </span>
                );
              })}
            </nav>
            <div className="app-file-manager__actions">
              <Button
                disabled={busy}
                onClick={() => { setCreatingFolder((value) => !value); setFolderName(""); }}
              >
                New folder
              </Button>
              <Button
                disabled={busy}
                onClick={() => fileInputRef.current?.click()}
              >
                {uploadPercent === null ? "Upload" : `Uploading… ${uploadPercent}%`}
              </Button>
              <Button disabled={busy} onClick={() => void list(path)}>Refresh</Button>
              <input
                ref={fileInputRef}
                className="sr-only"
                type="file"
                tabIndex={-1}
                aria-hidden="true"
                disabled={busy}
                onChange={(event) => void onFileChosen(event)}
              />
            </div>
          </div>

          {creatingFolder && (
            <div className="app-file-manager__new-folder">
              <Input
                label="New folder name"
                value={folderName}
                disabled={busy}
                autoFocus
                onChange={(event) => setFolderName(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") void createFolder();
                  if (event.key === "Escape") { setCreatingFolder(false); setFolderName(""); }
                }}
              />
              <Button variant="primary" disabled={busy || !folderName.trim()} onClick={() => void createFolder()}>Create</Button>
              <Button variant="quiet" disabled={busy} onClick={() => { setCreatingFolder(false); setFolderName(""); }}>Cancel</Button>
            </div>
          )}

          <ul className="app-file-manager__list" aria-busy={loading}>
            {loading ? (
              <li className="app-file-manager__note">Loading files…</li>
            ) : sorted.length === 0 ? (
              <li className="app-file-manager__note">This folder is empty.</li>
            ) : (
              sorted.map((entry) => (
                <li className="app-file-manager__row" key={`${entry.type}:${entry.name}`} data-type={entry.type}>
                  {entry.type === "d" ? (
                    <Button className="app-file-manager__name" variant="quiet" disabled={busy} onClick={() => openFolder(entry.name)}>
                      <Icon name="folder" size={18} />
                      <span className="app-file-manager__label">{entry.name}</span>
                    </Button>
                  ) : (
                    <Button className="app-file-manager__name" variant="quiet" disabled={busy} onClick={() => void download(entry)}>
                      <Icon name="file" size={18} />
                      <span className="app-file-manager__label">{entry.name}</span>
                      <span className="app-file-manager__size">{formatQuantity(entry.size, "used")}</span>
                    </Button>
                  )}
                  <div className="app-file-manager__row-actions">
                    {entry.type === "f" && (
                      <Button variant="quiet" disabled={busy} aria-label={`Download ${entry.name}`} onClick={() => void download(entry)}>
                        <Icon name="download" size={16} />
                      </Button>
                    )}
                    <Button
                      variant="danger"
                      disabled={busy}
                      aria-label={`Delete ${entry.name}`}
                      onClick={() => setPendingDelete(entry)}
                    >
                      Delete
                    </Button>
                  </div>
                </li>
              ))
            )}
          </ul>
        </>
      )}

      {notice && <OperationFeedback className="managed-operation-feedback" message={notice} state={noticeState} />}

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        title={pendingDelete ? `Delete ${pendingDelete.name}?` : ""}
        description={
          pendingDelete?.type === "d"
            ? `The folder "${pendingDelete.name}" and everything inside it will be permanently deleted. This cannot be undone.`
            : `"${pendingDelete?.name ?? ""}" will be permanently deleted. This cannot be undone.`
        }
        confirmLabel="Delete permanently"
        busy={busy}
        onCancel={() => { if (!busy) setPendingDelete(null); }}
        onConfirm={() => void confirmDelete()}
      />
      </>
      )}
    </section>
  );
}
