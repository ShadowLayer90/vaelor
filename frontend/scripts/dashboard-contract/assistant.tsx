// The page the Assistant clauses of the layout contract measure
// (scripts/dashboard-contract/assistant-pass.mjs, W5-S4). Test-only, never
// built. It renders the real Assistant page (AgentCenter) inside the app
// shell's own markup, with a connected Assistant model, one saved chat (so Ask
// shows its More menu and "Delete chat") and one custom agent (so Routines
// shows a card with its actions). The words here are fixture words.
import { TOPBAR_PAGE_SLOT_ID } from "../../src/lib/topbarSlot";
import { createRoot } from "react-dom/client";
import { AgentCenter } from "../../src/components/AgentCenter";
import "../../src/styles.css";

const now = new Date().toISOString();
const agent = { id: "custom_stock", name: "Stock watch", description: "Fixture agent that tracks an index", scopes: [],
  custom: true, operational: true, enabled: true, version: 1 };
const routes: Record<string, unknown> = {
  "/agent/status": { configured: true, reachable: true, provider: "managed-local", model: "Qwen3" },
  "/copilot/setup": { hardware: {}, recommendation: { can_install: true, primary: { name: "Qwen3 1.7B" } }, providers: [],
    credential_storage_ready: true },
  "/assistant/preferences": { intelligence_choice: "" },
  "/assistant/automations": { schedules: [], runs: [], triggers: [] },
  "/assistant/status": { memories: 0 },
  "/assistant/profiles": [agent],
  "/assistant/custom-agents": [agent],
  "/assistant/conversations": [{ id: "conv_fixture", title: "Fixture chat", archived: false, created_at: now, updated_at: now }],
  "/assistant/conversations/conv_fixture/messages": [
    { role: "user", content: "Fixture question?", created_at: now },
    { role: "assistant", content: "Fixture answer.", created_at: now },
  ],
  "/cluster": { pooled_deployments: [] },
  "/ai-chat/setup": { collections: [] },
};
const json = (data: unknown) => new Response(JSON.stringify({ ok: true, data }), { status: 200, headers: { "Content-Type": "application/json" } });
window.fetch = ((input: RequestInfo | URL) => {
  const path = new URL(String(input), location.origin).pathname.replace("/api/v2", "");
  return Promise.resolve(json(routes[path] ?? []));
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
        <AgentCenter session={admin} />
      </main>
    </div>
  </div>,
);
