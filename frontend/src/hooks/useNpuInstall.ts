import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import { jobIsTerminal } from "../lib/jobPresentation";
import type { UpdateJob } from "../components/UpdateJobStatus";
import type { Session } from "../types";

/**
 * Drives the on-device (NPU) Assistant model install and exposes its live job.
 *
 * The install is a multi-minute, multi-GB job. This posts it, then POLLS its
 * status so a caller can render `<UpdateJobStatus>` instead of the install
 * running silently. Shared so BOTH the Workloads "Set up assistant" flow and the
 * first-run assistant panel start the same install and show the same status -
 * rather than one of them navigating away and dropping the intent (the button
 * that says "Set up the on-device Assistant" must actually set it up).
 *
 * Resilience and safety, matching the reviewed Workloads behaviour: a single
 * transient `/jobs/<id>` read failure does not kill the view (the install keeps
 * running server-side); a second `start` while one is in flight re-shows the
 * running job rather than racing a second install on the one NPU device; and the
 * poll loop stops writing state once the component unmounts.
 */
export function useNpuInstall(
  session: Session,
  onNotice?: (message: string) => void,
): { job: UpdateJob | null; start: (tag: string) => void; dismiss: () => void } {
  const [job, setJob] = useState<UpdateJob | null>(null);
  const [dismissedId, setDismissedId] = useState("");
  const activeRef = useRef(false);
  const mountedRef = useRef(true);
  const latestJobId = useRef("");
  const noticeRef = useRef(onNotice);
  useEffect(() => { noticeRef.current = onNotice; }, [onNotice]);
  useEffect(() => () => { mountedRef.current = false; }, []);

  // Poll one install job to completion, updating `job` as it advances. Shared by
  // `start` (which posts a new job first) and the mount effect (which re-attaches
  // to a job started in ANOTHER surface). Returns the last job seen, or null if
  // the status could never be read.
  const pollJob = useCallback(async (jobId: string): Promise<UpdateJob | null> => {
    let current: UpdateJob | null = null;
    let consecutiveErrors = 0;
    for (let attempt = 0; attempt < 1200; attempt += 1) {
      // Stop as soon as the component is gone: keep polling after unmount and
      // this issues GETs for up to an hour against a view nobody is watching.
      if (!mountedRef.current) break;
      try {
        current = await apiRequest<UpdateJob>(`/jobs/${encodeURIComponent(jobId)}`);
        consecutiveErrors = 0;
        latestJobId.current = current.id;
        if (mountedRef.current) setJob(current);
        if (jobIsTerminal(current)) break;
      } catch {
        // The install job keeps running server-side; a transient read of its
        // STATUS is not a failure of the install. Give up only after a
        // sustained outage (~1 min of continuous failures).
        consecutiveErrors += 1;
        if (consecutiveErrors >= 20) break;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 3000));
    }
    return current;
  }, []);

  const start = useCallback((tag: string) => {
    // Re-show the in-progress install rather than starting a second one.
    setDismissedId("");
    if (activeRef.current) return;
    activeRef.current = true;
    void (async () => {
      let created: { job_id: string };
      try {
        created = await apiRequest<{ job_id: string }>(
          "/copilot/install-npu-model",
          { method: "POST", body: JSON.stringify({ tag }) },
          session.csrf_token,
        );
      } catch (error) {
        activeRef.current = false;
        if (mountedRef.current) {
          noticeRef.current?.(error instanceof Error ? error.message : "The on-device model install could not be started.");
        }
        return;
      }
      const current = await pollJob(created.job_id);
      activeRef.current = false;
      if (!current && mountedRef.current) {
        noticeRef.current?.("The on-device model is installing in the background. Open the Assistant setup to check its status.");
      }
    })();
  }, [session.csrf_token, pollJob]);

  // Re-attach to an install already in flight, started in another surface
  // (Workloads <-> first-run panel mount separate hook instances). On mount,
  // look for the most recent non-terminal `model.install_release` job and adopt
  // it, so its live status shows here too rather than the install running
  // invisibly. A GET failure or no jobs is a no-op - `start` remains the path in.
  useEffect(() => {
    void (async () => {
      if (activeRef.current) return;
      let jobs: UpdateJob[];
      try {
        jobs = await apiRequest<UpdateJob[]>("/jobs?limit=50");
      } catch {
        return;
      }
      if (!mountedRef.current || activeRef.current) return;
      // The endpoint returns a list, but treat anything else as "no jobs" rather
      // than throwing - a missing/odd response must not crash the mounting view.
      if (!Array.isArray(jobs)) return;
      const inflight = jobs.find(
        (it) => it.type === "model.install_release" && !jobIsTerminal(it),
      );
      if (!inflight) return;
      activeRef.current = true;
      setJob(inflight);
      latestJobId.current = inflight.id;
      await pollJob(inflight.id);
      activeRef.current = false;
    })();
    // Adopt once on mount; `pollJob` is stable (no deps).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const dismiss = useCallback(() => {
    if (latestJobId.current) setDismissedId(latestJobId.current);
  }, []);

  const visible = job && job.id !== dismissedId ? job : null;
  return { job: visible, start, dismiss };
}
