import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { OperationProjection } from "../lib/operationOwner";
import { makeOperationFixture } from "../test/operationProjectionFixture";
import { OperationOwner, operationHasStalled, operationIsWaiting } from "./OperationOwner";

const apiRequest = vi.hoisted(() => vi.fn());
vi.mock("../lib/api", () => ({ apiRequest }));

beforeEach(() => {
  apiRequest.mockReset();
  apiRequest.mockResolvedValue({ events: [], operation_id: "jobs:job-42", schema: "vaelor.operation.v1" });
});

const activeOperation = (overrides: Partial<OperationProjection> = {}): OperationProjection => makeOperationFixture({
  operation_id: "jobs:job-42",
  operation_key: "jobs/job-42",
  ledger: "jobs",
  source_id: "job-42",
  owner: { route: "/workloads/applications", resource: "/workloads/applications/demo" },
  type: "compose.deploy",
  state: "running",
  phase: "Validate the reviewed Compose plan",
  progress: { value: 42, percent: 42, determinate: true, label: "42%" },
  message: "Checking the application health endpoint.",
  recoverable_error: { message: "The selected port is already in use.", code: "port_conflict", recoverable: true },
  correction: {
    available: true,
    message: "Choose a different port and continue this operation.",
    fields: { port: { label: "Application port", value: "8080", hint: "Use the port reserved for this application." } },
  },
  permissions: { cancel: true, retry: true, resume: true, discard: true },
  result_summary: { available: true, data: { summary: "The application is ready for review." } },
  result: { summary: "The application is ready for review.", open_url: "/workloads/applications/demo" },
  artifacts: [{ id: "evidence", name: "compose evidence", url: "/api/v2/jobs/job-42/evidence", kind: "application/json" }],
  endpoint: "/api/v2/operations/jobs:job-42",
  audit_link: "/api/v2/operations/jobs:job-42/audit",
  /*
   * Relative, not fixed. This fixture describes an operation that is running
   * *now*, and a card only claims progress for an operation that has reported
   * recently (`operationHasStalled`). Hard-coded August timestamps made every
   * running fixture read as abandoned the moment the stall check existed —
   * the fixture was describing a state it did not mean.
   */
  timestamps: {
    created_at: new Date(Date.now() - 74_000).toISOString(),
    started_at: new Date(Date.now() - 72_000).toISOString(),
    updated_at: new Date(Date.now() - 60_000).toISOString(),
    completed_at: null,
  },
  revision: 4,
  ...overrides,
});

describe("OperationOwner", () => {
  it("presents the exported projection step, determinate progress, timing, and controller actions", async () => {
    const cancel = vi.fn(async () => null);
    const retry = vi.fn(async () => null);
    const resume = vi.fn(async () => null);
    render(<OperationOwner operation={activeOperation()} controller={{ cancel, retry, resume, hasSavedState: true, connection: "reconnecting" }} title="Deploy application" />);

    expect(screen.getByText("Validate the reviewed Compose plan")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Operation progress" })).toHaveValue(42);
    expect(screen.getByText(/Started/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cancel operation" }));
    expect(cancel).toHaveBeenCalledOnce();
    await waitFor(() => expect(screen.getByRole("button", { name: "Retry operation" })).not.toBeDisabled());
    fireEvent.click(screen.getByRole("button", { name: "Retry operation" }));
    expect(retry).toHaveBeenCalledOnce();
    await waitFor(() => expect(screen.getByRole("button", { name: "Resume operation" })).not.toBeDisabled());
    fireEvent.click(screen.getByRole("button", { name: "Resume operation" }));
    expect(resume).toHaveBeenCalledOnce();
    expect(screen.queryByRole("button", { name: "Done" })).not.toBeInTheDocument();
  });

  it("keeps correction in place and submits the edited values through the correction callback", () => {
    const onCorrection = vi.fn();
    render(<OperationOwner operation={activeOperation()} onCorrection={onCorrection} />);

    const port = screen.getByLabelText("Application port");
    fireEvent.change(port, { target: { value: "9090" } });
    fireEvent.click(screen.getByRole("button", { name: "Apply correction" }));
    expect(onCorrection).toHaveBeenCalledWith({ port: "9090" });
    expect(screen.getAllByText("Choose a different port and continue this operation.").length).toBeGreaterThan(0);
  });

  it("requires a distinct confirmation before discarding a saved draft", () => {
    const discardSavedState = vi.fn(async () => null);
    render(<OperationOwner operation={activeOperation()} controller={{ discardSavedState }} />);

    fireEvent.click(screen.getByRole("button", { name: "Discard saved draft" }));
    expect(discardSavedState).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Confirm discard saved draft" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Confirm discard saved draft" }));
    expect(discardSavedState).toHaveBeenCalledOnce();
  });

  it("shows results and evidence links while keeping history read-only and honest about progress", () => {
    render(<OperationOwner mode="history" operation={activeOperation({ progress: { value: null, percent: null, determinate: false, label: "In progress" } })} controller={{ done: vi.fn(async () => null) }} title="Deployment history" />);

    expect(screen.getByRole("progressbar", { name: "Operation progress" })).not.toHaveAttribute("value");
    expect(screen.getByText("In progress")).toBeInTheDocument();
    expect(screen.getByText("Application port")).toBeInTheDocument();
    expect(screen.getByText("8080")).toBeInTheDocument();
    expect(screen.getByText("The application is ready for review.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open installed app" })).toHaveAttribute("href", "/workloads/applications/demo");
    expect(screen.getByRole("button", { name: "View Activity evidence" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Done" })).not.toBeInTheDocument();
    expect(screen.getByRole("article", { name: "Deployment history" })).toHaveAttribute("data-mode", "history");
  });

  it("opens readable Activity evidence in a modal without navigating to raw JSON", async () => {
    apiRequest.mockResolvedValue({
      operation_id: "jobs:job-42",
      schema: "vaelor.operation.v1",
      events: [{
        id: 417,
        action: "job.create",
        actor: "admin",
        created_at: 1785711509,
        details: { deduplicated: false, display_identity: "nginx-welcome", resource_id: "nginx-welcome", type: "compose.backup" },
        remote_addr: "192.168.0.72",
        result: "success",
        target: "job-42",
      }],
    });
    render(<OperationOwner operation={activeOperation()} />);

    fireEvent.click(screen.getByRole("button", { name: "View Activity evidence" }));

    expect(await screen.findByRole("dialog", { name: "Activity evidence" })).toBeInTheDocument();
    expect(apiRequest).toHaveBeenCalledWith("/operations/jobs:job-42/audit", { cache: "no-store" });
    expect(screen.getByRole("heading", { name: "Job Create" })).toBeInTheDocument();
    expect(screen.getByText("Succeeded")).toBeInTheDocument();
    expect(screen.getAllByText("Nginx Welcome", { selector: "dd" })).toHaveLength(2);
    expect(screen.getByText("Compose Backup")).toBeInTheDocument();
    expect(screen.queryByText(/\{"data"/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Done" }));
    expect(screen.queryByRole("dialog", { name: "Activity evidence" })).not.toBeInTheDocument();
  });

  it("offers Done for a terminal operation and keeps it client-owned", () => {
    const done = vi.fn(async () => activeOperation({ state: "completed" }));
    render(<OperationOwner operation={activeOperation({ state: "completed", progress: { value: 100, percent: 100, determinate: true, label: "Complete" } })} controller={{ done }} />);
    fireEvent.click(screen.getByRole("button", { name: "Done" }));
    expect(done).toHaveBeenCalledOnce();
  });

  it("dismisses the finished operation through onDone after acknowledging it", async () => {
    // Regression: a finished model download's "Done" ran controller.done (which
    // only forgets the resume key) but left the operation on screen, because the
    // owner tracks it in its own state. onDone is how the owner clears that.
    const done = vi.fn(async () => activeOperation({ state: "completed" }));
    const onDone = vi.fn();
    render(<OperationOwner onDone={onDone} operation={activeOperation({ state: "completed", progress: { value: 100, percent: 100, determinate: true, label: "Complete" } })} controller={{ done }} />);
    fireEvent.click(screen.getByRole("button", { name: "Done" }));
    await waitFor(() => expect(onDone).toHaveBeenCalledOnce());
    expect(done).toHaveBeenCalledOnce();
  });

  it("keeps opaque result data out of the plain-language summary", () => {
    render(<OperationOwner operation={activeOperation({
      state: "completed",
      phase: "final_health_check",
      result: {},
      result_summary: { available: true, data: { internal_id: "job-42", exit_code: 0 } },
    })} />);

    expect(screen.getByText("Final Health Check")).toBeInTheDocument();
    expect(screen.getByText("The operation completed. Technical details are available below.")).toBeInTheDocument();
    expect(screen.queryByText(/\{"internal_id"/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("Technical result details"));
    expect(screen.getByText(/"internal_id": "job-42"/)).toBeInTheDocument();
  });

  it("describes a failed operation one way: no progress, no completion claim, no raw code (#148)", () => {
    const failure = "The selected model did not return a usable structured response. Retry once or shorten the task.";
    render(<OperationOwner mode="history" operation={activeOperation({
      state: "failed",
      message: failure,
      recoverable_error: { message: failure, code: "operation_failed", recoverable: true },
      correction: { available: false, message: failure, fields: {} },
      progress: { value: null, percent: null, determinate: false, label: "In progress" },
      result: {},
      // The backend builds `message` and `result_summary` from the same
      // record, so the identical sentence arriving under `data.message` is
      // the live payload shape, not a contrivance.
      result_summary: { available: true, data: { message: failure, attempts: 1 } },
    })} />);

    // The attention notice owns the sentence; the feedback strip, the
    // correction copy, and the Result section must not repeat it.
    expect(screen.getAllByText(failure)).toHaveLength(1);
    expect(screen.getByText("The operation did not complete.")).toBeInTheDocument();
    expect(screen.queryByText(/The operation completed/)).not.toBeInTheDocument();
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
    expect(screen.queryByText("In progress")).not.toBeInTheDocument();
    expect(screen.getByText("No further progress — this operation has ended.")).toBeInTheDocument();
    expect(screen.queryByText("operation_failed")).not.toBeInTheDocument();
  });

  it("uses the human dismissal message and canonical terminal vocabulary", () => {
    render(<OperationOwner operation={activeOperation({
      state: "superseded",
      phase: "needs_input",
      result: {},
      result_summary: { available: true, data: { code: "research_discarded", dismissal: "Saved application research discarded." } },
    })} />);

    expect(screen.getByText("Superseded")).toBeInTheDocument();
    expect(screen.getByText("Waiting for your input")).toBeInTheDocument();
    expect(screen.getByText("Saved application research discarded.")).toBeInTheDocument();
    expect(screen.queryByText("research_discarded")).not.toBeInTheDocument();
    expect(screen.getByText(/Support reference/)).toBeInTheDocument();
  });

  it("stops claiming progress for an operation that went quiet", () => {
    /*
     * Seen on the appliance: a row whose last event was eight days earlier
     * still swept an indeterminate progress bar and read "Another local
     * specialist is running." A moving bar is the strongest claim a screen
     * can make that work is happening now (#161).
     *
     * The threshold is deliberately an hour, not minutes: a large image pull
     * onto an SD card legitimately reports nothing while it works, and
     * accusing a healthy step of having stopped is the same defect pointed
     * the other way.
     */
    const stale = new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString();
    const operation = makeOperationFixture({
      state: "running",
      timestamps: { created_at: stale, started_at: stale, updated_at: stale, completed_at: null },
    });
    expect(operationHasStalled(operation)).toBe(true);

    render(<OperationOwner mode="history" operation={operation} />);
    expect(screen.queryByRole("progressbar", { name: "Operation progress" })).toBeNull();
    expect(screen.getByText(/Stopped reporting/)).toBeInTheDocument();
  });

  it("leaves a working operation alone", () => {
    expect(operationHasStalled(makeOperationFixture({ state: "running" }))).toBe(false);
    const slow = makeOperationFixture({
      state: "running",
      timestamps: {
        created_at: new Date(Date.now() - 3_600_000).toISOString(),
        started_at: new Date(Date.now() - 3_600_000).toISOString(),
        updated_at: new Date(Date.now() - 59 * 60 * 1000).toISOString(),
        completed_at: null,
      },
    });
    expect(operationHasStalled(slow)).toBe(false);
  });

  it("never calls a finished operation stalled, however old it is", () => {
    const ancient = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString();
    for (const state of ["completed", "failed", "cancelled"] as const) {
      expect(operationHasStalled(makeOperationFixture({
        state,
        timestamps: { created_at: ancient, started_at: ancient, updated_at: ancient, completed_at: ancient },
      }))).toBe(false);
    }
  });

  it("does not invent a stall when there is no timestamp at all", () => {
    expect(operationHasStalled(makeOperationFixture({
      state: "running",
      timestamps: { created_at: null, started_at: null, updated_at: null, completed_at: null },
    }))).toBe(false);
  });

  it("never accuses an operation that is correctly waiting", () => {
    /*
     * needs_approval waits for a human by design; paused waits because
     * someone asked it to. Telling an owner their pending approval "stopped
     * reporting" is #52's family — a status that contradicts the state — and
     * the first version of this check did exactly that after one hour.
     */
    const stale = new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString();
    for (const state of ["needs_approval", "paused", "queued", "ready", "waiting", "draft"] as const) {
      const operation = makeOperationFixture({
        state,
        timestamps: { created_at: stale, started_at: stale, updated_at: stale, completed_at: null },
      });
      expect(operationHasStalled(operation)).toBe(false);
    }
  });

  it("draws no sweeping bar for an operation that is waiting", () => {
    /*
     * The first version of this fix stopped calling these stalled and left
     * them animating, so a paused operation swept under the words "In
     * progress" — #52's contradiction rebuilt one component over. A bar is a
     * claim that work is happening now.
     */
    const stale = new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString();
    for (const state of ["needs_approval", "paused", "queued", "ready", "waiting"] as const) {
      const { container, unmount } = render(
        <OperationOwner operation={makeOperationFixture({
          state,
          progress: { value: null, percent: null, determinate: false, label: "In progress" },
          timestamps: { created_at: stale, started_at: stale, updated_at: stale, completed_at: null },
        })} />,
      );
      expect(container.querySelector("progress")).toBeNull();
      expect(container.textContent).toContain("Waiting");
      expect(container.textContent).not.toContain("Stopped reporting");
      expect(container.textContent).not.toContain("may have been interrupted");
      unmount();
    }
  });

  it("renders the server's staleness verdict instead of recomputing it (#180)", () => {
    /*
     * The verdict is server-owned now (`vaelor/operation_projection.py`
     * `staleness`). When the projection carries it, the card must render it and
     * not fall back to its own clock and threshold — two mechanisms answering
     * "has this gone quiet?" is how the `ready` case drifted (#201). The proof
     * is a fresh `updated_at` (a client computation would say "not stalled")
     * beside a server verdict of `stale: true`: the server wins.
     */
    const now = new Date().toISOString();
    const serverStalled = makeOperationFixture({
      state: "running",
      timestamps: { created_at: now, started_at: now, updated_at: now, completed_at: null },
      staleness: { reporting: true, stale: true, silent_seconds: 7200, stale_after_seconds: 3600 },
    });
    expect(operationHasStalled(serverStalled)).toBe(true);
    render(<OperationOwner mode="history" operation={serverStalled} />);
    expect(screen.getByText(/Stopped reporting/)).toBeInTheDocument();
  });

  it("defers to a server verdict of not-stale even when its own clock would accuse (#180)", () => {
    const old = new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString();
    const serverFresh = makeOperationFixture({
      state: "running",
      timestamps: { created_at: old, started_at: old, updated_at: old, completed_at: null },
      staleness: { reporting: true, stale: false, silent_seconds: 30, stale_after_seconds: 3600 },
    });
    expect(operationHasStalled(serverFresh)).toBe(false);
  });

  it("still draws a bar for an operation that is actually running", () => {
    const now = new Date().toISOString();
    const { container } = render(
      <OperationOwner operation={makeOperationFixture({
        state: "running",
        progress: { value: null, percent: null, determinate: false, label: "In progress" },
        timestamps: { created_at: now, started_at: now, updated_at: now, completed_at: null },
      })} />,
    );
    expect(container.querySelector("progress")).not.toBeNull();
  });

  it("agrees with the backend about which states are waiting", () => {
    /*
     * `vaelor/operation_projection.py:_progress` gives exactly this set no
     * determinate progress. Two vocabularies for one question is the failure
     * behind #52 and #74, and keeping `ready` on the reporting side while the
     * backend had it here was that failure starting again.
     */
    for (const state of ["queued", "draft", "ready", "waiting", "needs_approval", "paused"]) {
      expect(operationIsWaiting(state)).toBe(true);
    }
    for (const state of ["running", "completed", "failed", "healthy"]) {
      expect(operationIsWaiting(state)).toBe(false);
    }
  });
});
