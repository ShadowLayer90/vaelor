// The page the dashboard layout contract measures (scripts/dashboard-layout-contract.mjs).
// Test-only, never built: `vite build` reads index.html alone. It renders the real Cluster page
// (FleetCenter) on its Performance tab inside the app shell's own markup, fed by what the backend
// builders produce: the dashboard's two layers from src/test/dashboardPayloads.json (kept
// byte-equal to the builders by tests/test_dashboard_payload_contract.py), and Diagnostics from
// the golden /cluster/performance payloads (src/test/cluster_performance_golden.json),
// the scenario matching each state, so the strip's Why headline is the product's own.
import { TOPBAR_PAGE_SLOT_ID } from "../../src/lib/topbarSlot";
import { createRoot } from "react-dom/client";
import { FleetCenter } from "../../src/components/FleetCenter";
import payloads from "../../src/test/dashboardPayloads.json";
import golden from "../../src/test/cluster_performance_golden.json";
import "../../src/styles.css";

const params = new URLSearchParams(location.search);
const state = params.get("s") ?? "replicated";
window.localStorage.setItem("vaelor.cluster.mode.admin", params.get("mode") === "advanced" ? "advanced" : "easy");
window.localStorage.removeItem("vaelor.performance.range");

/** Which golden Diagnostics payload goes with each dashboard state. */
const DIAGNOSTICS: Record<string, string> = {
  replicated: "cluster_replicated_serving@1h",
  one_of_two: "cluster_replicated_one_of_two_read@15m",
  split: "cluster_split_serving@15m",
  controller_led_split: "cluster_split_serving@15m",
  single_machine: "single_machine_serving@1h",
  unloaded_idle: "cluster_unloaded_idle@15m",
  unloaded_by_hand: "cluster_unloaded_by_hand@15m",
  unloaded_cause_unknown: "cluster_unloaded_cause_unknown@15m",
  loading: "cluster_loading@15m",
  nothing_served: "nothing_served@15m",
  record_unreadable: "cluster_record_unreadable@15m",
  serving_scrape_stale: "serving_reading_stale@15m",
};
const layers = (payloads as Record<string, { now: unknown; range: unknown }>)[state];
const diagnostics = (golden as Record<string, Record<string, unknown>>)[DIAGNOSTICS[state] ?? "cluster_replicated_serving@1h"];

const json = (data: unknown) => new Response(JSON.stringify({ ok: true, data }), { status: 200, headers: { "Content-Type": "application/json" } });
window.fetch = ((input: RequestInfo | URL) => {
  const url = new URL(String(input), location.origin);
  const path = url.pathname.replace("/api/v2", "");
  if (path === "/cluster/performance/dashboard/now") return Promise.resolve(json(layers.now));
  if (path === "/cluster/performance/dashboard") return Promise.resolve(json(layers.range));
  if (path === "/cluster/performance") {
    // The route echoes the window it was asked for.
    return Promise.resolve(json({ ...diagnostics, requested_window: url.searchParams.get("window") ?? diagnostics.requested_window }));
  }
  if (path === "/cluster") {
    return Promise.resolve(json({
      controller: { initialized: true, driver: "docker-swarm", role: "head-controller", advertise_address: "" },
      runtime: { available: true, initialized: true, control_available: true, engine: "ready", nodes: [], services: [] },
      enrolled_nodes: [], requirements: { ports: [], network: "" },
    }));
  }
  if (path === "/apps/catalog") return Promise.resolve(json([]));
  return Promise.resolve(json({}));
}) as typeof fetch;

const admin = { user: { username: "admin", role: "administrator" as const }, csrf_token: "t", expires_at: 9_999_999_999 };

createRoot(document.getElementById("root")!).render(
  <div className="app-shell">
    <aside className="sidebar" aria-label="Navigation (stand-in)" />
    <div className="workspace">
      <header className="topbar"><div className="topbar__identity"><strong>Vaelor</strong></div>
        {/* The shell's page slot, as Overview draws it: a page's pill and actions render only here. */}
        <div className="topbar__actions"><div className="topbar__page" id={TOPBAR_PAGE_SLOT_ID} style={{ display: "contents" }} /></div></header>
      <main className="main" id="main-content">
        <FleetCenter session={admin} />
      </main>
    </div>
  </div>,
);
