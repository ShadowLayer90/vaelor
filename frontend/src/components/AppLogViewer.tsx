import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import { Button, Checkbox, Input, Notice, Select } from "./ui";

type Severity = "all" | "error" | "warning" | "info";

function severityOf(line: string): Exclude<Severity, "all"> {
  if (/\b(error|fatal|panic|critical|exception)\b/i.test(line)) return "error";
  if (/\b(warn|warning|degraded)\b/i.test(line)) return "warning";
  return "info";
}

function isRestartBoundary(line: string): boolean {
  return /\b(container|service|server|application)\b.*\b(started|starting|restarted|stopped)\b/i.test(line);
}

function wallClock(line: string): string | null {
  const match = line.match(/\b\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+Z?\b/);
  if (!match) return null;
  const parsed = new Date(match[0]);
  return Number.isNaN(parsed.getTime()) ? null : parsed.toLocaleString();
}

// Docker's `--timestamps` prepends an RFC3339Nano instant to every line. Split
// it off so it shows in the time column and not inside the message, and fall
// back to any timestamp the container wrote itself when there is no docker
// prefix (older captures, or lines pasted in from elsewhere).
const DOCKER_TIMESTAMP = /^(\d{4}-\d{2}-\d{2}T[\d:.]+(?:Z|[+-]\d{2}:?\d{2}))\s+(.*)$/s;

function splitEntry(line: string): { clock: string | null; body: string } {
  const leading = line.match(DOCKER_TIMESTAMP);
  if (leading) {
    const parsed = new Date(leading[1]);
    if (!Number.isNaN(parsed.getTime())) return { clock: parsed.toLocaleString(), body: leading[2] };
  }
  return { clock: wallClock(line), body: line };
}

export function AppLogViewer({ appId, appName }: { appId: string; appName: string }) {
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [paused, setPaused] = useState(false);
  const [follow, setFollow] = useState(true);
  const [wrap, setWrap] = useState(true);
  const [query, setQuery] = useState("");
  const [severity, setSeverity] = useState<Severity>("all");
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);
  const viewportRef = useRef<HTMLDivElement>(null);

  const refresh = useCallback(async () => {
    try {
      const data = await apiRequest<{ output: string }>(`/managed/apps/${appId}/logs`);
      setOutput(data.output || "");
      setError("");
      setUpdatedAt(new Date());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Recent logs could not be loaded.");
    } finally {
      setLoading(false);
    }
  }, [appId]);

  useEffect(() => {
    setLoading(true);
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (paused || !follow) return;
    const interval = window.setInterval(() => void refresh(), 3000);
    return () => window.clearInterval(interval);
  }, [follow, paused, refresh]);

  const lines = useMemo(() => output.split(/\r?\n/).filter(Boolean).filter((line) => {
    if (severity !== "all" && severityOf(line) !== severity) return false;
    return !query.trim() || line.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase());
  }), [output, query, severity]);

  useEffect(() => {
    if (follow && !paused && viewportRef.current) viewportRef.current.scrollTop = viewportRef.current.scrollHeight;
  }, [follow, lines, paused]);

  const download = () => {
    const url = URL.createObjectURL(new Blob([output], { type: "text/plain;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `${appName.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase() || "application"}-logs.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="app-log-viewer" aria-labelledby="app-log-viewer-title">
      <div className="app-log-viewer__heading">
        <div><h3 id="app-log-viewer-title">Application logs</h3><p>{updatedAt ? `Last checked ${updatedAt.toLocaleTimeString()}` : "Waiting for the first response"}</p></div>
        <div className="app-log-viewer__actions">
          <Button aria-pressed={paused} onClick={() => setPaused((value) => !value)}>{paused ? "Resume updates" : "Pause updates"}</Button>
          <Button onClick={() => void refresh()}>Refresh now</Button>
          <Button disabled={!output} onClick={download}>Download logs</Button>
        </div>
      </div>
      <div className="app-log-viewer__filters">
        <Input label="Search logs" onChange={(event) => setQuery(event.target.value)} placeholder="Message or timestamp" type="search" value={query} />
        <Select label="Severity" onChange={(event) => setSeverity(event.target.value as Severity)} value={severity}>
          <option value="all">All entries</option><option value="error">Errors</option><option value="warning">Warnings</option><option value="info">Information</option>
        </Select>
        <Checkbox checked={follow} id="app-log-follow" label="Follow latest" onChange={(event) => setFollow(event.target.checked)} />
        <Checkbox checked={wrap} id="app-log-wrap" label="Wrap long lines" onChange={(event) => setWrap(event.target.checked)} />
      </div>
      {error && <Notice severity="danger"><span>{error}</span><Button onClick={() => void refresh()}>Try again</Button></Notice>}
      <div aria-busy={loading} aria-live="polite" className={`app-log-viewer__viewport${wrap ? " is-wrapped" : ""}`} ref={viewportRef} role="log">
        {loading && !output ? <p>Loading recent logs…</p> : lines.length ? lines.map((line, index) => {
          const { clock, body } = splitEntry(line);
          return <div className={isRestartBoundary(line) ? "app-log-line is-boundary" : "app-log-line"} data-severity={severityOf(line)} key={`${index}-${line}`}><span>{clock || "Time not reported"}</span><code>{body}</code></div>;
        }) : <p>{output ? "No log entries match these filters." : "No recent log entries were reported."}</p>}
      </div>
      <small>{paused ? "Updates are paused; the current view is retained." : follow ? "Following the latest entries every three seconds." : "Automatic follow is off; use Refresh now when ready."}</small>
    </section>
  );
}
