"""Where on the console each control lives, named once (adversarial review of VD-205).

Assistant sentences named console places in a dozen hand-written spellings -
"Home", "System > Cooling", "Workloads", "Fleet", "Assistant setup" - while the
console calls two of those pages "Apps and AI" and "Cluster"
(`frontend/src/lib/destinations.ts`), and the link table that turns a named
place into a route (`assistant_answer_presentation._DESTINATIONS`) kept its own
copy. Each place is one constant here; the sentences and the link table both
read it, so a page that is renamed or moved changes in one line (LESSONS 6).

Each sentence names the *control* as well as the page: "Restart and Shut down",
"the Cooling card", "Recovery checkpoints".
"""

from __future__ import annotations

from typing import Tuple

#: The console's page names, as `frontend/src/lib/destinations.ts` gives them.
HOME = "Home"
SYSTEM = "System"
APPS = "Apps and AI"
CLUSTER = "Cluster"
ACTIVITY = "Activity"
SETTINGS = "Settings"

#: Where the detected processor and operating system are shown, as a sentence
#: opens with it.
DETECTED_HARDWARE_SCREEN = "The {} screen".format(HOME)

#: Power: restart and shut down.
#: **VD-200 moves Power from Home to the System page header**; at that merge
#: this becomes "System, under Power" and nothing else changes.
#: The heading over those controls on the page.
POWER_HEADING = "Power controls"
POWER_PLACE = "Home, under {}".format(POWER_HEADING)
POWER_CONTROL = "Restart and Shut down"
#: The panels on the System page, as their headings read.
LIGHTING_PANEL = "Case lighting"
HARDWARE_PANEL = "Hardware and services"

COOLING_PLACE = "System > Cooling"
COOLING_CONTROL = "the Cooling card"
LIGHTING_PLACE = "System > Case lighting"
HARDWARE_PLACE = "System > {}".format(HARDWARE_PANEL)
RECOVERY_PLACE = "Activity, under Recovery checkpoints"
RECOVERY_CONTROL = "Recovery checkpoints"
INSTALL_PLACE = "Apps and AI > Install"
ASSISTANT_SETUP = "Assistant setup"
#: The console's own features, named as their screens name them.
AI_CHAT = "AI Chat"
LLM_SERVER = "LLM Server"
REMOTE_CONSOLE = "Remote console"
INFERENCE_GATEWAY = "Inference gateway"
CONNECTIONS_PLACE = "Settings > Connections"
PERFORMANCE_PLACE = "Cluster > Performance"
#: Where the LLM Server is turned on and given keys: the Models filter on the
#: Cluster page's Deployments tab, which holds the served endpoints and their
#: API keys.
ENDPOINTS_CONTROL = "Models"
ENDPOINTS_PLACE = "Cluster > Deployments > {}".format(ENDPOINTS_CONTROL)
#: The route to that filter, as `DEPLOYMENTS_MODELS_HREF` in
#: `frontend/src/lib/clusterSections.ts` writes it: under the tab's default
#: "All" filter the endpoints and their keys are not drawn (VD-200 review S-D1).
ENDPOINTS_ROUTE = "?cluster=deployments&deployments=models#/fleet"

#: Every place a sentence can name, with the route that reaches it and the
#: control the link is labelled with. Most specific first, so "System >
#: Cooling" is never reduced to a bare "System". Read by
#: `assistant_answer_presentation` to emit next steps.
DESTINATIONS: Tuple[Tuple[str, str, str, bool], ...] = (
    (ENDPOINTS_PLACE, ENDPOINTS_ROUTE, ENDPOINTS_CONTROL, True),
    (LIGHTING_PLACE, "#/system", LIGHTING_PANEL, True),
    (COOLING_PLACE, "#/system", "Cooling", True),
    (HARDWARE_PLACE, "#/system", HARDWARE_PANEL, True),
    (INSTALL_PLACE, "#/workloads", "Install", True),
    (PERFORMANCE_PLACE, "#/fleet", "Performance", True),
    (CONNECTIONS_PLACE, "#/admin", "Connections", True),
    (ASSISTANT_SETUP, "#/assistant", ASSISTANT_SETUP, True),
    ("Assistant > Agents", "#/assistant/agents", "Run history", True),
    (REMOTE_CONSOLE, "#/kvm", REMOTE_CONSOLE, True),
    # Every answer that sends a user to AI Chat names it; it gets a real link.
    (AI_CHAT, "#/ai-chat", AI_CHAT, True),
    ("app manager", "#/workloads", "App manager", True),
    (APPS, "#/workloads", APPS, False),
    # The route's older names, still written by a model or an older sentence.
    ("Workloads > Install", "#/workloads", "Install", True),
    ("Workloads", "#/workloads", APPS, False),
    (ACTIVITY, "#/activity", ACTIVITY, False),
    (CLUSTER, "#/fleet", CLUSTER, False),
    ("Fleet", "#/fleet", CLUSTER, False),
    ("Admin", "#/admin", SETTINGS, False),
    (HOME, "#/", HOME, False),
    ("Overview", "#/", HOME, False),
)

#: Every console phrase of two or more words a question can use for the
#: console itself rather than for a machine: the features, every place and
#: link label above. A one-word page is in `PAGE_NAMES`, matched only as a page. A worker an
#: owner named "AI", "Server" or "Chat" is never found inside "AI Chat" or "the
#: LLM Server" (adversarial review B-6, round 2): the machine-name resolver
#: removes these from a question before it looks for a one-word name.
#: Longest first, so a phrase inside a longer one never wins; spellings that
#: differ only in case are one phrase (the matcher ignores case, and a second
#: spelling was a branch no test could tell apart). Ties sort by text, so the
#: order is the same in every process.
_CONSOLE_PHRASE_SET = (
    {AI_CHAT, LLM_SERVER, REMOTE_CONSOLE, ASSISTANT_SETUP, POWER_CONTROL, COOLING_CONTROL,
     RECOVERY_CONTROL, POWER_PLACE, POWER_HEADING, RECOVERY_PLACE, INFERENCE_GATEWAY}
    | {text for place, _route, label, _strict in DESTINATIONS for text in (place, label)
       if " " in text or ">" in text})
CONSOLE_PHRASES: Tuple[str, ...] = tuple(sorted(
    {text.lower(): text for text in sorted(_CONSOLE_PHRASE_SET)}.values(),
    key=lambda text: (-len(text), text)))

#: The console's one-word page names; one stands for the page, not a machine,
#: when the question says so ("the Home page", "System > ...").
PAGE_NAMES: Tuple[str, ...] = tuple(sorted(
    {HOME, SYSTEM, CLUSTER, ACTIVITY, SETTINGS}
    | {text for place, _route, label, _strict in DESTINATIONS for text in (place, label)
       if " " not in text and ">" not in text}))
