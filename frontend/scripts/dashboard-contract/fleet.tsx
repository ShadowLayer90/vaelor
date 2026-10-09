// The page the Fleet machine-tile clauses of the layout contract measure
// (scripts/dashboard-layout-contract.mjs, fleetPass). Test-only, never built.
// It renders the real Cluster page (FleetCenter) on its Fleet tab, in
// Advanced, inside the app shell's own markup, with one enrolled worker whose
// newest telemetry row carries every reading a machine tile shows. Query:
// `sensor` is the power sensor word the backend sends ("graphics engine",
// "board", or "none" for no word) and `long=1` gives the worker a long name.
// `deploy=1` adds one split model row with both fence notes and a long LAN
// endpoint, for the Deployments row clauses (deploymentsPass); the words in it
// are fixture words, not the backend's. `endpoint=long` gives that row an IPv6
// endpoint, long enough to take the name's room at a desktop width.
// `live=1` (W4-D2) makes that row a SERVING one (Unload and Remove) and fills
// the endpoint cards (under the Models filter) the way a live box does - an enabled
// LLM Server with two keys and its address, and the cluster serving card -
// the state the Deployments tab overflowed in at 491-506 px - and keeps two
// removed apps' data volumes, one with a long name, for the Retained data panel.
// `cluster=activity` and `cluster=agents` open the last two tabs, for the tab
// strip clauses (B2, tabStripPass). `hostlink=1` gives the Setup tab's
// machine settings two links, the LAN card carrying a shared-card note, for
// the confirmation dialog clauses (review round 1, dialogPass); with `pool=1`
// it also gives both machines a GPU memory pool card (W5-D4, pool_alignment).
// `page=activity` renders the top-level Activity page (#/activity) instead of
// the Cluster page: the Security audit trail lives in both, and only the
// Cluster tab panel's own `word-break` wraps it, so a wrap measured there alone
// could not fail (LESSONS 2, FE-W7-1/3 verification).
// `nav=1` renders the real navigation (Sidebar) instead of the stand-in rail,
// for the phone navigation bar's label clause (W7-D3, nav_labels).
// `page=home-facts` renders Home's real headline facts in Home's hero markup,
// for the facts_whole_words clause (W8-D2: "Administrator" broke mid-word).
// `software=1` (VD-194 P1, FE-4) gives the worker a populated "Worker software"
// row - a full appliance with all twelve services and component readings with
// long reasons - for the machine_card and type_floor clauses. Its words stand
// in for `worker_profile_state.worker_software_view`'s; they are not the
// backend's.
import { TOPBAR_PAGE_SLOT_ID } from "../../src/lib/topbarSlot";
import { createRoot } from "react-dom/client";
import { ActivityCenter } from "../../src/components/ActivityCenter";
import { FleetCenter } from "../../src/components/FleetCenter";
import { Sidebar } from "../../src/components/Sidebar";
import { SystemStripFacts } from "../../src/components/SystemStripFacts";
import "../../src/styles.css";

const params = new URLSearchParams(location.search);
const sensor = params.get("sensor") ?? "graphics engine";
const name = params.get("long") === "1" ? "Workstation in the rack by the window" : "ZBook";
window.localStorage.setItem("vaelor.cluster.mode.admin", "advanced");

const now = Math.floor(Date.now() / 1000);
const inventory = { architecture: "x86_64", cpu_count: 16, memory_bytes: 65.3e9, os: "Ubuntu 24.04", docker: true, reachable: true };
const gpu = { present: true, reason: "", gfx_target_version: "gfx1151", device_count: 1, vram_total_bytes: 536870912,
  vram_used_bytes: 0, gtt_total_bytes: 47.2e9, gtt_used_bytes: 0, system_ram_bytes: 128e9, addressable_bytes: 47.2e9 };
const node = (node_id: string, nodeName: string, role: string) => ({
  node_id, name: nodeName, role, state: role === "worker" ? "ready" : "controller", schedulable: true, state_reason: "",
  capacity: { cpu: 16, cpu_threads: 32, memory_bytes: 65.3e9, gpu },
  reserved: { memory_bytes: 0, gpu_memory_bytes: 0, from: [] }, free: { memory_bytes: 65.3e9, gpu_memory_bytes: 44e9 },
});
const latest: Record<string, number> = {
  processor_load: 5, memory_percent: 70, gpu_busy_percent: 0, gpu_gtt_used_bytes: 39.1e9, gpu_gtt_total_bytes: 47.2e9,
  cpu_temperature_c: 61, gpu_temperature_c: 47, gpu_gfx_temperature_c: 52, gpu_power_watts: 2.04,
};
const history = {
  scope: "node", node: "node_worker", requested_window: "15m", window_seconds: 900, bucket_seconds: 30,
  sample_interval_seconds: 1, available: true, reason: "", retention: null,
  series: Object.fromEntries(Object.entries(latest).map(([key, value]) => [key, {
    points: Array.from({ length: 20 }, (_, index) => ({ t: `t${index}`, v: value * (0.9 + 0.01 * index) })),
  }])),
  last_sample_at: now - 2, last_sample_age_seconds: 2, reporting: true, reporting_window_seconds: 60,
  latest: { t: now - 2, values: latest, gpu_temperature_sensor: "graphics engine",
    gpu_power_sensor: sensor === "none" ? null : sensor, gpu_readings_note: "", implausible_note: "" },
  ingest: { received_at: now - 2, received_age_seconds: 2, clock_offset_seconds: 0.1, clock_refused: false,
    clock_refused_at: null, reason: "" },
};
const routes: Record<string, unknown> = {
  "/cluster": {
    controller: { initialized: true, driver: "docker-swarm", role: "head-controller", advertise_address: "" },
    runtime: { available: true, initialized: true, control_available: true, engine: "ready", nodes: [], services: [] },
    enrolled_nodes: [{ id: "node_worker", name, host: "192.0.2.12", port: 22, role: "worker", state: "joined",
      host_key_fingerprint: "SHA256:w", labels: { swarm_node_id: "swarm_worker" }, runtime_state: "ready",
      runtime: { status: "Ready", availability: "Active" }, telemetry_provisioned: true, inventory }],
    pooled_deployments: params.get("deploy") === "1" ? [{
      name: "qwen3-30b-split", state: params.get("live") === "1" ? "healthy" : "unloaded", model_id: "Qwen/Qwen3-30B-A3B-Instruct-2507",
      node_ids: ["controller", "swarm_worker"], engine: "vllm",
      endpoint: params.get("endpoint") === "long" ? "http://[fd00:1234:5678:9abc::200]:26200/v1" : "http://192.0.2.200:26200/v1",
      vllm_version: "vLLM 0.27", units: { engine: "vllm", mode: "distributed" },
      cleanup_note: "Fixture clean-up note, long enough to run onto a second line on a phone-width row.",
      fence_note: "Fixture fence note, also long enough to wrap on a phone-width row.",
    }] : [],
    requirements: { ports: [], network: "" },
  },
  "/cluster/capacity": { nodes: [node("controller", "This Vaelor controller", "head-controller"), node("swarm_worker", name, "worker")],
    unattributed_reservations: [], notes: {} },
  "/telemetry/current": { metrics: { cpu_percent: 3, memory_percent: 60, memory_used: 39e9 } },
  // The Activity tab (B2's tab-strip clauses open it): an empty audit trail
  // and no alert profiles or rules.
  "/audit": [],
  "/assistant/profiles": [],
  "/assistant/automations": { schedules: [], triggers: [] },
  // W4d-D20's Retained data panel: none kept unless `live=1`, which keeps two.
  "/cluster/retained-volumes": [],
};
// `audit=long` (FE-W7-1/3): Security audit rows whose words are long - a chat
// title, a URL in a memory label, an IPv6 From address and one unbroken
// 216-character token - for the audit_wrap clause.
if (params.get("audit") === "long") {
  const row = (id: number, action: string, target: string, label: string, from: string) => ({
    id, created_at: now - id * 60, actor: "admin", action, target, result: "success", remote_addr: from, details: {},
    target_view: { kind: "memory", label, known: true } });
  routes["/audit"] = [
    row(1, "assistant.chat", "conv_0123456789abcdef01234567",
      "Chat: Questions about the kitchen renovation budget, the contractor quotes and the permit timeline for spring",
      "192.0.2.44"),
    row(2, "assistant.memory.create", "mem_0123456789abcdef01234567",
      "Memory: https://example.com/very/long/path/to/the/owner/documentation/page/that/never/breaks/naturally?query=1",
      "2001:db8:85a3:1234:5678:8a2e:370:7334"),
    row(3, "assistant.memory.update", "mem_fedcba9876543210fedcba98", "Memory: " + "x".repeat(216), ""),
    // The compressed IPv6 form and a label that is one long URL and nothing else.
    row(4, "assistant.memory.update", "mem_00112233445566778899aabb",
      "https://docs.example.com/owner/guides/appliance/remote-access/certificates/troubleshooting#fingerprint",
      "2001:db8:85a3::8a2e:370:7334"),
  ];
}
if (params.get("live") === "1") {
  routes["/cluster/retained-volumes"] = [
    { id: "rv1", service_name: "vaelor-app-uptime-kuma-for-the-whole-house", volume: "vaelor-app-uptime-kuma-for-the-whole-house-uptime-data",
      node_id: "swarm_worker", node_name: name, removed_at: now - 3600, backups: 0 },
    { id: "rv2", service_name: "vaelor-app-notes", volume: "vaelor-app-notes-data", node_id: "swarm_worker", node_name: name,
      removed_at: now - 86_400, backups: 2 },
  ];
  const key = (id: string, label: string, last4: string) => ({ credential_id: id, label, key_fingerprint: `SHA256:${id}fingerprint`,
    last4, created_at: now - 86_400, last_used_at: now - 60, requests: 1234 });
  routes["/llm-server"] = { enabled: true, state: "running", available: true, unavailable_reason: "", target_kind: "cluster",
    model: "Qwen/Qwen3-30B-A3B-Instruct-2507", model_known: true, port: 11434, base_url: "http://192.0.2.200:11434/v1",
    keys: [key("k1", "Laptop at the kitchen table", "a1b2"), key("k2", "Home Assistant on the shelf", "c3d4")],
    runtime: { state: "running", reason: "", detail: "" }, usage: { state: "counting", detail: "", refused_24h: 3 } };
  routes["/cluster/serving"] = { present: true, endpoint: { id: "cluster-serving", label: "Cluster serving",
    name: "qwen3-30b-split", model: "Qwen/Qwen3-30B-A3B-Instruct-2507", base_url: "http://127.0.0.1:8000/v1",
    internal: true, replicated: false, key_present: false, key_fingerprint: "", state: "healthy",
    serving: { state: "serving", detail: "Serving on both machines." } } };
}
if (params.get("hostlink") === "1") {
  const link = (name: string, kind: string, address: string, network: string, note: string) => ({
    name, kind, state: "up", speed_mbps: kind === "wired" ? 2500 : 40000, mtu: 1500, address, network,
    usable: true, reason: "", shared_note: note });
  // `pool=1` (W5-D4): both machines' GPU memory pool cards, side by side on a
  // desktop - the controller read live, the worker read a minute ago.
  const poolOf = () => ({ supported: true, reason: "", can_change: true, blocked_reason: "", ram_bytes: 65.3e9,
    current_bytes: 47.2e9, default_bytes: 32.6e9, system_left_bytes: 18.1e9, override: { state: "vaelor", bytes: 47.2e9 },
    min_gib: 31, max_gib: 45, restart_pending: false, restart_reason: "" });
  const machines = params.get("pool") === "1" ? [
    { node_id: "controller", name: "This controller", role: "controller", checked_at: null, stale_reason: "", pool: poolOf() },
    { node_id: "node_worker", name: `${name} worker`, role: "worker", checked_at: now - 58, stale_reason: "", pool: poolOf() },
  ] : [];
  routes["/host-settings"] = { gpu_memory_pool: { machines }, cluster_link: { available: true, reason: "",
    chosen: null, machines: [], notes: [], links: [
      // As long as the backend's shared-card sentence (cluster_link_shared_card).
      link("enp1s0", "wired", "192.0.2.10", "192.0.2.0/24", "Fixture note: enp1s0 is the network card other "
        + "machines reach this controller on (192.0.2.10), so it is shared with your LAN. A split on it is fenced "
        + "as a shared network card: only the split's own connections are guarded, the LAN still reaches this "
        + "controller through enp1s0, and the split shares the card's bandwidth with everything else on it."),
      link("thunderbolt0", "thunderbolt", "198.51.100.1", "198.51.100.0/30", ""),
    ] } };
}
if (params.get("software") === "1") {
  const units = ["appliance-recovery", "appliance-upgrade", "application-research", "control-plane",
    "credential-broker", "hardware-bridge", "host-desktop", "system-update", "vnc-gateway", "vnc-tls-proxy",
    "workload-broker", "workload-executor"].map((unit) => ({ name: `vaelor-${unit}.service`, present: true,
    running: true, disabled: false, reload_needed: true, state_words: "running, reload pending" }));
  const reading = (id: string, label: string, status: string, word: string, why = "", measured = "") => ({
    id, label, status, word, why, note: "", appliance_owned: id === "state-dir", measured,
    words: `${word[0].toUpperCase()}${word.slice(1)}.${why ? ` ${why}` : ""}` });
  const nodes = (routes["/cluster"] as { enrolled_nodes: Record<string, unknown>[] }).enrolled_nodes;
  nodes[0].worker_software = {
    node_id: "node_worker", state: "full-appliance-installed", label: "Full appliance installed", tone: "warning",
    exit: "Fixture: converting is not available yet.", sentence: "Fixture: full Vaelor appliance still installed on this machine (12 services found).",
    checked_at: now - 240, checked: "checked 4 min ago", stale: false, stale_sentence: "", stale_after_seconds: 900,
    old_reading_words: { administrator: "reading is old - Recheck", other: "reading is old - an administrator can Recheck" },
    undated: { label: "Not checked", tone: "neutral", checked: "never checked",
      sentence: { administrator: "Not checked yet.", other: "Not checked yet; an administrator can check it." } },
    exit_words: { administrator: "Fixture: converting is not available yet.", other: "Fixture: converting is not available yet." }, served_at: now,
    age_words_table: [{ below: 60, unit: 0, words: "checked just now" }, { below: 7200, unit: 60, words: "checked {n} min ago" },
      { below: 172800, unit: 3600, words: "checked {n} h ago" }, { below: null, unit: 86400, words: "checked {n} days ago" }],
    profile_summary: "Beside the appliance, these differ from this controller's profile: Vaelor data folder, AMD amd-smi.",
    profile_codes: { summary: "Profile codes", rows: [
      { id: "measured", label: "Read from this machine", value: "87be4219" },
      { id: "expected", label: "This controller expects for this machine", value: "39776cfd" },
      { id: "release", label: "This release, for any worker, before per-machine settings", value: "d0bb750d" },
    ] },
    components: [
      reading("python3", "Python 3", "matches", "matches"),
      reading("state-dir", "Vaelor data folder", "differs", "differs", "Owned by root:vaelor, expected root:root. "
        + "Mode 1770, expected 0755. Has an extra access list. The appliance set this; Vaelor does not change it yet."),
      reading("telegraf", "Telemetry agent (Telegraf)", "unread", "unread", "This controller has no verified "
        + "Telegraf tarball staged, so the installed binary cannot be compared."),
      reading("amd-smi", "AMD amd-smi", "differs", "differs", "Its apt preferences file is not on this machine.",
        "Version 7.14.1-0 installed, the version the profile pins."),
    ],
    appliance: { reading: "running", sentence: "", units },
    attempt_note: "The last check (2 min ago) failed: Vaelor could not reach the machine over SSH (timed out).",
  };
}
const json = (data: unknown) => new Response(JSON.stringify({ ok: true, data }), { status: 200, headers: { "Content-Type": "application/json" } });
window.fetch = ((input: RequestInfo | URL) => {
  const path = new URL(String(input), location.origin).pathname.replace("/api/v2", "");
  if (path.startsWith("/telemetry/history")) return Promise.resolve(json(history));
  return Promise.resolve(json(routes[path] ?? {}));
}) as typeof fetch;

const admin = { user: { username: "admin", role: "administrator" as const }, csrf_token: "t", expires_at: 9_999_999_999 };
// Activity's audit trail is its own tab since VD-200 (decision 2); `page=activity` opens it by its address.
if (params.get("page") === "activity") window.history.replaceState(null, "", "#/activity/audit");

createRoot(document.getElementById("root")!).render(
  <div className="app-shell">
    {params.get("nav") === "1" ? (
      <Sidebar activePage={params.get("page") === "activity" ? "activity" : "fleet"} connection="live" connectivity={null}
        health={{ status: "healthy", reasons: [], sampled_at: now }} metrics={{}} onNavigate={() => undefined}
        storage={null} user={admin.user} />
    ) : <aside className="sidebar" aria-label="Navigation (stand-in)" />}
    <div className="workspace">
      <header className="topbar"><div className="topbar__identity"><strong>Vaelor</strong></div>
        {/* The shell's page slot, as Overview draws it: a page's pill and actions render only here. */}
        <div className="topbar__actions"><div className="topbar__page" id={TOPBAR_PAGE_SLOT_ID} style={{ display: "contents" }} /></div></header>
      <main className="main" id="main-content">
        {params.get("page") === "activity" ? <ActivityCenter session={admin} />
          : params.get("page") === "home-facts" ? (
            <section className="device-hero" aria-label="System summary">
              <div className="device-hero__content">
                <SystemStripFacts device={{ name: "Fixture", id: "fixture", version: "1.0.0b1", peripherals: [] }}
                  metrics={{ boot_time: now - 93_600 }} noun="workstation" role={admin.user.role} />
              </div>
            </section>
          ) : <FleetCenter session={admin} />}
      </main>
    </div>
  </div>,
);
