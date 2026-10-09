import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiRequest, downloadApiRequest } from "../lib/api";
import { bytesIn, formatQuantity } from "../lib/format";
import type { Role } from "../types";
import "../styles/apps-manage.css";
import "../styles/apps-manage-tools.css";
import { AppsDialog, AppsInset } from "./appsKit";
import { Icon, ICON_SIZE } from "./Icon";
import { Button, Input, LoadingLines, type OperationState } from "./ui";
import { TabNotice } from "./WorkloadConfigurationTabs";

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
const MAX_UPLOAD_BYTES = bytesIn(100, "MiB");

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
  // A failed delete is said inside its dialog (the ManageFiles board), which stays open.
  const [deleteError, setDeleteError] = useState("");
  const cancelDeleteRef = useRef<HTMLButtonElement>(null);
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
    setDeleteError("");
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
      setDeleteError(caught instanceof Error && caught.message ? caught.message : "That item could not be deleted.");
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

  const closeDelete = () => { if (!busy) { setPendingDelete(null); setDeleteError(""); } };

  return (
    <section className="manage-tool" aria-labelledby="manage-files-title">
      <AppsInset
        detail="Browse this app's stored files. Upload, download, create folders, and delete items. Deleting is permanent."
        icon="database"
        title={<span id="manage-files-title">Files</span>}
      />

      {!isAdmin ? (
        <AppsInset detail="Administrator access is required to manage this app's files." icon="lock" />
      ) : (
      <>
      {error && (
        <div className="manage-banner manage-banner--danger" role="alert">
          <span>{error}</span>
          <Button onClick={() => void list(path || undefined)}>Try again</Button>
        </div>
      )}

      {loading && roots.length === 0 && !error && <div className="manage-tool__loading"><p className="manage-tool__quiet" role="status">Loading files…</p><LoadingLines label="Loading files" lines={1} /></div>}

      {noStorage ? (
        <p className="manage-tool__quiet" role="status">This app has no browsable storage.</p>
      ) : roots.length > 0 && (
        <>
          {roots.length > 1 && (
            <div className="manage-roots" role="tablist" aria-label="Storage locations">
              {roots.map((candidate) => (
                <Button
                  aria-selected={candidate === root}
                  className={candidate === root ? "manage-roots__option is-on" : "manage-roots__option"}
                  disabled={busy}
                  key={candidate}
                  onClick={() => navigateTo(candidate)}
                  role="tab"
                  variant="quiet"
                >
                  {candidate}
                </Button>
              ))}
            </div>
          )}

          <div className="manage-files__toolbar">
            <nav className="manage-files__breadcrumb" aria-label="Current folder">
              <Button variant="quiet" disabled={busy || segments.length === 0} aria-current={segments.length === 0 ? "page" : undefined} onClick={() => navigateTo(root)}>{root || "/"}</Button>
              {segments.map((segment, index) => {
                const target = joinPath(root, segments.slice(0, index + 1).join("/"));
                const isCurrent = index === segments.length - 1;
                return (
                  <span className="manage-files__crumb" key={target}>
                    <Icon name="chevron" size={ICON_SIZE.inline} />
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
            <div className="manage-files__actions">
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
            <div className="manage-files__new-folder">
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

          <ul className="manage-files__list" aria-busy={loading}>
            {loading ? (
              <li className="manage-files__note">Loading files…</li>
            ) : sorted.length === 0 ? (
              <li className="manage-files__note">This folder is empty.</li>
            ) : (
              sorted.map((entry) => (
                <li className="manage-files__row" key={`${entry.type}:${entry.name}`} data-type={entry.type}>
                  <Button
                    className="manage-files__name"
                    disabled={busy}
                    onClick={() => (entry.type === "d" ? openFolder(entry.name) : void download(entry))}
                    variant="quiet"
                  >
                    <Icon name={entry.type === "d" ? "folder" : "file"} size={ICON_SIZE.inline} />
                    <span className="manage-files__label">{entry.name}</span>
                  </Button>
                  {entry.type === "f" && <span className="manage-files__size">{formatQuantity(entry.size, "used")}</span>}
                  <div className="manage-files__row-actions">
                    {entry.type === "f" && (
                      <Button variant="quiet" disabled={busy} aria-label={`Download ${entry.name}`} onClick={() => void download(entry)}>
                        Download
                      </Button>
                    )}
                    <Button
                      variant="danger"
                      disabled={busy}
                      aria-label={`Delete ${entry.name}`}
                      onClick={() => { setDeleteError(""); setPendingDelete(entry); }}
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

      <TabNotice message={notice ? { text: notice, state: noticeState } : null} />

      {pendingDelete && (
        <AppsDialog
          busy={busy}
          className="manage-delete-dialog"
          error={deleteError || undefined}
          eyebrow="Can't be undone"
          eyebrowTone="danger"
          footer={<>
            <Button disabled={busy} onClick={closeDelete} ref={cancelDeleteRef}>Cancel</Button>
            <Button busy={busy} className="manage-solid-danger" onClick={() => void confirmDelete()} variant="danger">Delete permanently</Button>
          </>}
          initialFocusRef={cancelDeleteRef}
          onClose={closeDelete}
          role="alertdialog"
          size="narrow"
          title={`Delete ${pendingDelete.name}?`}
          titleId="app-file-delete-title"
        >
          <p>{pendingDelete.type === "d"
            ? `The folder "${pendingDelete.name}" and everything inside it will be permanently deleted. This cannot be undone.`
            : `"${pendingDelete.name}" will be permanently deleted. This cannot be undone.`}</p>
        </AppsDialog>
      )}
      </>
      )}
    </section>
  );
}
