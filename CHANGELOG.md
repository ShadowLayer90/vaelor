# Change log

## 1.0 Beta 1

First public beta. Vaelor is a Flask + React control plane for an edge-compute
appliance (developed on an HP Z2 Mini with a Strix Halo NPU and GPU): it reports
live hardware and service status, answers questions about the machine from a
local on-device model, deploys reviewed apps and local models, and manages the
box's accounts, backups, and recovery.

New in this release:

- **Metrics export.** A Prometheus/OpenMetrics endpoint at
  `GET /api/v2/metrics` exposes CPU, GPU, NPU, memory, storage, fan, service,
  application, and job metrics for scraping, with low-cardinality labels and
  deduplicated series.
- **Alert delivery.** When an alert rule crosses a threshold it can now reach a
  person out of band by email or webhook, not just as an on-box diagnostic.
  Setup is guided: pick your email provider (Gmail, Outlook/365, Yahoo, iCloud,
  SendGrid) and the server, port, and encryption fill in automatically - you
  supply only your address and an app password. Secrets live in the encrypted
  credential broker, and you can send a test before relying on a channel.
- **Scheduled and off-site backups.** Take an encrypted snapshot of the whole
  appliance by hand or on a timer, keep a bounded set with retention, and
  optionally copy each backup off the machine to S3-compatible cloud storage or
  an HTTPS endpoint. Restore reuses the portable-state import path behind a typed
  confirmation. Setup is guided with plain, numbered steps.
- **Assistant and reliability fixes.** Broad "what should I check?" questions
  answer from live readings; multi-part questions are fully answered; readings
  render the degree symbol correctly; the GPU and NPU model tiers self-heal on
  failure; and the appliance's own cluster services are recognised as healthy
  during a managed-model deploy.

## 2.1.0 Alpha 101

Two follow-ups from the a100 pass.

Setting up a custom application can now publish a web port for an app whose
description declares none — so you can actually reach it — and, for an app that
genuinely manages Docker (such as a container log viewer), you can grant it
access to the Docker socket through a clearly-labelled, off-by-default consent
checkbox that spells out the risk in plain words. Both live in the Configure
step of the guided setup.

The Assistant now answers open "what should I check?"-style questions about this
machine — "what should I look at?", "anything I should keep an eye on?" — with a
real health summary drawn from the machine's own readings, instead of sending
you to AI Chat, which cannot see this machine. Genuine general-knowledge
questions it has no evidence for still go where they belong.

## 2.1.0 Alpha 100

Following the live end-to-end campaign, this release makes the Assistant and
custom Agents actually answer, and closes several deploy and reliability gaps.

The Assistant now answers broad questions like "are there any issues I need to
know about?" instead of returning an internal, unusable message. The on-device
model was being handed its question as a raw data structure, which it copied
back instead of answering; it is now asked in plain language and its reply is
tidied into sentences before you see it. A custom agent you grant "storage and
hardware facts" can now actually report your disk usage and free space — the
readings it was granted were being dropped before the model could read them, and
are now put first so they survive. Building and running a custom agent works end
to end: activating an agent no longer needlessly bumps its version, an empty task
now explains itself instead of doing nothing, and deleting an agent no longer
leaves a stale "test run" note behind.

Custom Applications are easier to get running. A saved research request you can
neither finish nor clear is no longer a dead end: a stuck draft can now be
discarded (a genuinely running app is still protected), and the "resume" banner
only appears for research you can actually resume. An app whose description
declares no web port can be given one so you can reach it, and an app that needs
access to the Docker socket (such as a container log viewer) can be granted it
through an explicit, clearly-labelled privileged opt-in. When the on-device
model is busy, setting up an app now falls back to the built-in reader in a few
seconds instead of waiting up to twenty. A freshly installed appliance now makes
sure its container runtime is fully ready before the first app deploy, so the
first install no longer fails until a manual restart. And a retried chat message
can no longer be answered twice: each message carries a one-time key the
appliance uses to ignore an accidental resend.

## 2.1.0 Alpha 99

A live end-to-end test of every screen turned up a batch of real defects, now
fixed. Deploying a custom application no longer fails at the first step when the
on-device model is busy — it falls back to the built-in request reader and
carries on. AI Chat answers are no longer cut off in the middle of a sentence:
long answers now get the room they need. Asking the Assistant to resume paused
monitoring no longer prints internal machine data instead of a reply, and the
Assistant now reports every fan the machine reads rather than just one. The
System screen now shows processor load instead of a permanent "no samples", and
the background-paused live telemetry now says so plainly and offers a Refresh
button. When a request is slow or drops, the app no longer claims the appliance
is offline, and a reply that is still arriving is no longer shown as if it were
finished.

## 2.1.0 Alpha 98

The graphics-processor AI model now actually runs on the graphics processor. The
control plane looked for the GPU's math libraries at one fixed location that a
recent system update stopped using, so the AI Chat model silently fell back to
the CPU and the System screen reported "the GPU library did not load", even
though the libraries were present elsewhere. Vaelor now finds the GPU runtime
wherever it is installed, and the availability check and the live server agree
on whether it is there, so the screen tells the truth. A freshly installed
appliance also provisions that GPU runtime itself now, from AMD's signed
software channel, instead of assuming another component supplied it — so a clean
machine can run the GPU AI-Chat model with no manual setup. If the GPU runtime
cannot be provisioned, the on-device (neural-processor) Assistant is unaffected.

## 2.1.0 Alpha 97

Asking the Assistant or AI Chat a question no longer risks being answered twice.
When the connection between the browser and the appliance dropped after the
question had already been sent — for example a proxy timing out a long-running
answer — the app's automatic one-time retry could re-send the same question, so
the appliance wrote the question a second time and ran the model a second time
for one ask. The retry now re-sends only when the request could not have reached
the appliance yet, and otherwise leaves the existing "still being answered"
recovery to pick the answer up, so a single question stays a single answer. The
quick retry that recovers the first action after an update is unchanged.

## 2.1.0 Alpha 96

Installing the on-device model from the setup screen now finishes on the neural
processor every time, including on a machine that has never run one before. In
Alpha 95 the one-click install downloaded the model and its runtime correctly
but the final start-up step could pick the wrong server or ask the processor for
a model it had not been given yet — so on a freshly installed appliance the
install could report a chat failure even though the model itself was in place.
The start-up step now serves the exact model that was just installed and checks
it against that same model, so a clean appliance goes from one click to a
working on-device Assistant with no terminal steps. Re-running the install over
a model that is already serving now works too, rather than failing because the
running model server still held its own program file open.

## 2.1.0 Alpha 95

A machine with a neural processor now offers its fine-tuned on-device model as
the recommended Assistant, and installs it from the setup screen with no
terminal steps. Before this, a fresh appliance recommended a large general model
run on the processor or graphics, because the on-device model — which is faster
and never leaves the machine — was not something the setup screen could offer or
install on its own. Now it is the top choice on hardware that can run it: one
click downloads the model together with the exact runtime it needs, checks both
against a known fingerprint, and starts serving it, while the previous
recommendation stays available as an alternative.

## 2.1.0 Alpha 94

Two things behind the scenes so a freshly installed appliance is ready to run
its on-device models with no terminal steps. The installer that provisions the
graphics AI engine on a clean machine was looking for its files in the wrong
place inside the downloaded package, so a clean install finished with the
graphics model unavailable; it now finds them wherever the package puts them.
And the appliance gained the machinery to install a fine-tuned on-device neural
model from a verified release — it downloads the model and the exact runtime it
needs together, checks them against a known fingerprint before trusting them,
and unpacks them where the model server can find them, all without a terminal.

## 2.1.0 Alpha 93

The first thing you do right after the appliance updates no longer fails with
a scary "the local control service is unavailable." That message came from the
browser reusing a network connection the control plane had already closed
during its restart: the very first request landed on a dead connection and was
reported as an outage, even though a second attempt always worked. The app now
quietly retries that first request on a fresh connection, so the setup step you
click right after a deploy — starting a custom-application research, for one —
just works. A genuine outage still reports itself, and nothing is retried when
a request is deliberately cancelled or times out.

## 2.1.0 Alpha 92

Deploying a custom application no longer turns the whole request into the
deployment name. When you described an application in a sentence, the setup
step quietly used that entire sentence as the name; it survived research,
configuration, review, and the final approval, then failed at the last step
with an obscure "Compose project name is invalid," and the stalled attempt
could not be edited to recover. The name is now seeded as a short, valid
default drawn from the identified application, the field is capped to what
the system accepts, and it remains yours to change — so a deployment cannot
be built around a name that only breaks at the end.

## 2.1.0 Alpha 91

The assistant now answers two hardware questions it used to leave half-answered
on this class of machine. Asking how hard the graphics processor is working, or
how hot it is running, now returns the actual readings — utilisation, temperature
and memory use — instead of just naming the card; the numbers were already being
measured, but the answer that came back had dropped them. And asking for the CPU
fan speed now reports the real RPM read from this workstation's own sensors,
rather than saying no reading was available. Both are read live from the machine
each time they are asked, and neither is guessed when a sensor is genuinely
silent.

## 2.1.0 Alpha 90

The built-in on-device assistant model can now hold a longer conversation with
its tools. Its working memory was capped at a size that a multi-step research
run could outgrow: once an agent had gathered a few pages or data feeds, the
combined request crossed the limit and the model rejected it outright, which
showed up as an empty or half-finished answer on exactly the tasks that needed
several steps. The window is now doubled, which comfortably covers a full
multi-step run, so those agent tasks complete on the built-in model instead of
stalling. It still uses only a fraction of the machine's memory, and the
"retry on the more capable model" escalation remains for the occasional task
that is genuinely too large for the built-in model.

## 2.1.0 Alpha 89

Custom agents gained two things. First, a data tool: an agent can now fetch a
data feed or API directly by web address, not only read ordinary web pages —
because many sites (sports scores, for one) hide their real data behind a
JavaScript feed a plain page-reader cannot see. Every fetch still goes through
the same guarded, public-only broker with the same protections as the existing
page reader. Second, a way out when the small on-device model struggles: if an
agent run on the built-in model fails, comes back unformatted, or never uses its
tools, the run now offers a "retry on the more capable model" action that runs
the same task on the graphics-card model (the one that powers AI Chat), which
handles tools more reliably. The built-in model stays the default; the escalation
is yours to choose, and it changes only which model runs — never what the agent
is allowed to do.

## 2.1.0 Alpha 88

Custom research agents are now real tool-using agents, and they reach for their
tools by default: for anything current or factual they gather evidence with a
tool before answering instead of guessing from memory, and if the model tries to
answer cold it is nudged once to go check first. Instead of Vaelor guessing
what to look up and handing the model a fixed pile of search results, the agent's
model now decides for itself when to search the web and which pages to open, does
it, reads what comes back, and searches again if it needs more — a genuine
back-and-forth, the way a person researching would work. The agent drives its own
tools first, and the sources it shows you are the pages it actually opened; only
if the model can't drive the tools does Vaelor fall back to the previous one-shot
search, so it is never worse than before. Every tool call still
runs through the same guarded path (only the tools the agent was granted, only
read-only ones, web search still restricted to allowed sites, pages still checked
for provenance), and if the model ever can't drive the tools the agent quietly
falls back to the previous one-shot search so it is never worse than before. The
agent is also now told today's date, so it stops mistaking last week for a date
in the future.

## 2.1.0 Alpha 83

Custom research agents now search the web the way a person would. Until now, when
an agent needed to look something up, Vaelor built the search from the plain words
of your request — so "list the final scores of last week's games" led the search
with "final" and came back with dictionary definitions of that word instead of any
scores. Now the on-device model, which understands what you are actually asking,
writes the search query itself — leading with the real subject — and when the first
results are clearly the wrong topic it looks at them, rewrites the query once, and
searches again. If it still cannot find anything relevant it says so plainly and
asks a clarifying question, exactly as before — it never fills the gap with a
guess. The old word-based search is kept as a safety net, so a search is never
worse than it used to be, only better.

## 2.1.0 Alpha 82

Custom agents no longer work blind. Until now, when you asked a custom agent to
do something, the on-device model was handed little more than the agent's
description and your raw request — no memory of what this appliance has learned,
no picture of what a finished answer should look like. A small model asked to
work with that little context fails in ways that look random. Vaelor now assembles
a working context for each agent run: the appliance's relevant saved facts, the
lessons it has recorded from earlier application work, and a clear description of
what a complete, well-formed answer contains — all clearly marked as reference
material the agent may use but never take orders from. The recorded lessons feed
an agent for the first time, so the appliance genuinely improves as it accumulates
history rather than starting cold every time.

Custom Application deploy recognition is now taught to the model as a skill rather
than pinned to a hand-kept list of trigger words, so it recognizes far more ways
of asking to run an app — "I'd love my own Nextcloud for the family photos", "a
self-hosted password manager" — while still refusing plain questions and chatter,
and still routing every real request through the same research, review, and
approval steps. Two smaller fixes: asking to edit a deploy request now brings back
exactly what you typed instead of a shortened summary, and a request like "run a
Minecraft server so my kids can play together" is now summarized as just
"Minecraft".

## 2.1.0 Alpha 81

The local-AI catalog now offers three more models for a machine with a graphics
accelerator, so there is a real choice beyond the one built-in graphics model and
the small ones: a 30-billion and a 20-billion mixture-of-experts model (each runs
far lighter than its headline size because only a few billion parameters are
active at once) and a dense 14-billion model. Vaelor can now actually run a
standard model file on the graphics processor as the AI-Chat model, not only the
one special-format model it shipped with, and it guarantees just one chat model
uses the graphics processor at a time — installing a new one retires the previous
one first, so two never fight over graphics memory. A mixture-of-experts model is
now offered on a machine whose memory fits its real footprint, instead of being
hidden by its larger headline size.

Custom Application research asks you to clarify instead of guessing when it is
unsure. If it finds two different publishers both claiming to be the app you
named, or the public sources are too thin to be sure, it now stops and asks a
specific question — which is the official one, which edition, or the official
image link — rather than picking one and proceeding.

## 2.1.0 Alpha 80

Custom research agents now tell the truth about what they found, and ask when they
are unsure instead of guessing. A web-research agent could answer confidently with
a source it never actually read — the on-device model filling a gap from memory
and attaching a plausible-looking link — because the search returned off-topic
pages and nothing scored them. Now the agent shapes its search around the subject
of your request, keeps only results that are actually on topic, and the sources it
shows are exactly the pages it read — never one the model invented, even if the
search service is unavailable. When the agent cannot confirm an answer from
public sources, or your request is ambiguous, it stops and asks you specific
questions ("I found several unrelated results — do you have the official release
page?") rather than presenting a guess as a checked answer; its run is clearly
marked "needs input" everywhere it appears. An agent that can answer from this
machine's own readings (for example, checking fan health) still answers directly.

## 2.1.0 Alpha 79

Custom Application research no longer gives up when the small on-device model has
an off moment. The research reasoning runs on the appliance's compact assistant
model, which occasionally returns a malformed answer; a single such answer used
to end the whole request with "the selected model could not choose safe research
sources". Vaelor now retries a stumbling answer a few times with progressively
simpler instructions, reads the useful part out of a messy reply, and — if the
model still cannot choose sources — falls back to a safe automatic pick of only
well-known registries, code hosts, and the application's own publisher site
(never an arbitrary search result), so research keeps moving instead of
dead-ending. When it takes that automatic path it says so, rather than presenting
a weaker result as a full check. The change never widens what counts as a trusted
source and never sends a request outside the known hosts.

## 2.1.0 Alpha 78

Custom Application research can now verify an official container image for an app
Vaelor was never taught, so the feature works for the app a user actually asks
for rather than only a short built-in list. Before, research could describe an
app but could not prove a real, digest-pinned image for it — it only had the
app's homepage or code repository, never the registry entry that carries the
image's checksum — so every app outside the built-in list stopped at "no image
could be verified". Vaelor now resolves the app's official image and asks the
public container registry directly (Docker Hub and the GitHub Container Registry)
for the exact image digest that matches this machine's processor, then proves the
deployment against that. Nothing about specific apps is hardcoded. The registry
lookup is locked to those two known registries, never sends a request anywhere
else, and still accepts only a checksum the registry itself attests — a bad or
missing checksum is reported as unverified, not guessed. The exact image and its
full digest are shown for you to approve before anything is installed.

## 2.1.0 Alpha 77

The System screen now tells the truth about the graphics-processor AI. It could
read "Accelerator: not established" while the graphics model was in fact serving
and AI Chat was answering on it, because the health check looked at a redacted
copy of the connection that carried neither the model name nor its address, so it
never reached the running server — while AI Chat used the full connection and
worked. The check now consults the same signal AI Chat does, so an established
graphics model reads as established; the neural-processor display is unchanged.

AI Chat no longer refuses a question about your own uploaded documents. When a
knowledge collection is active it searches the documents first and answers from
them, and only falls back to "ask the Assistant — I can't read this machine" when
the documents genuinely do not cover the question. So "according to my documents,
what is the maintenance window?" is answered from the file instead of deflected,
while a bare live-telemetry question with nothing in the documents still points
you to the Assistant rather than inventing a reading.

Custom Application research reports its real reason for stopping. A machine that
could not verify an official container image used to say it needed an "ARM64"
image regardless of the actual hardware; it now names the real architecture, and
when the true problem is simply that no official image source was found it points
you to set up guarded web research or add the source, instead of mislabeling it
as a hardware incompatibility.

Guarded web research now installs and starts itself the moment a custom
application or an internet-researching agent needs it, instead of dead-ending
behind a manual setup step, so those features work on first use. If it cannot be
set up — or comes up but returns nothing — it says so plainly and points to the
repair, rather than promising forever that it is "coming up".

## 2.1.0 Alpha 76

Setting up the Assistant and choosing a local AI model no longer confuse the two
accelerators. The "Set up assistant" screen recommends the neural-processor
Assistant model, and the model catalog recommends the graphics-processor chat
model, each on its own hardware — a single shared recommendation used to let one
clobber the other. The catalog copy is now true on any machine: it no longer
claims a Raspberry-Pi-measured "quickest to answer" or "the Pi's measured best"
on a workstation, and it names the graphics model's format (ROCmFP4) rather than
an engine's internal name. The "recommended" and "runs on the graphics processor"
badges no longer overlap on a narrow card.

The Assistant and AI Chat now show a reply as soon as it lands even if the
connection drops mid-answer, instead of leaving it invisible until you refresh.

Custom Application ("Assisted Research") works through a request end to end
instead of dead-ending. A media request like "stream my movies to my TV" is
recognised instead of rejected, a question phrased as "can I host…?" declines
cleanly instead of erroring, the single local model being busy asks you to wait a
moment rather than crashing, request summaries read as sentences instead of
mangled word-bags, and the recovery advice points only to controls that exist.

## 2.1.0 Alpha 75

The installer now provisions the AMD accelerator AI stacks so a fresh Strix Halo
machine is turnkey, matching what the two-accelerator workstation was set up with
by hand. On a host where an AMD accelerator is actually bound (and never on a
Raspberry Pi or a machine without one), it installs the lemonade-server snap -
which carries the neural-processor model runtime and the graphics-processor
libraries the control plane serves from - and it downloads the graphics-processor
inference engine, a third-party prebuilt, verifying it against a pinned checksum
and installing it only if the checksum matches. Both steps are optional and never
fail the install: a machine that cannot fetch them reports those models as
unavailable with a reason rather than refusing to come up. The neural-processor
Assistant and the graphics-processor AI Chat then work on a clean install the way
they do on the workstation.

## 2.1.0 Alpha 74

AI Chat now names the model it is actually using. After the graphics-processor
model was deployed, the "This chat uses …" line under the model picker could
still show a model remembered from before - a name the picker's own dropdown no
longer offered - because the remembered choice was never checked against what
the connection now serves. The dropdown quietly displayed the first real model
while the caption showed the ghost, so the two disagreed. The remembered model
is now reconciled against the live list: if it is no longer offered, both the
caption and the model actually used fall back to the available one the dropdown
shows. A remembered model that is still offered, and a past conversation's own
recorded model, are unchanged.

## 2.1.0 Alpha 73

Makes the GPU model actually start. Live installs surfaced two things Alpha 71
could not have known without running on the two-accelerator machine. First, the
GPU server was being started by the workload executor, whose security sandbox
hides the graphics processor from anything it launches (and makes its log
directory read-only), so the model could never reach the GPU. It is now started
by the same root hardware-service that runs the neural-processor model, which
has the graphics-processor access it needs - the executor's sandbox is left
untouched. Second, the routing fix from Alpha 72 is corrected so a
neural-processor Assistant deploy, which carries no model file, is no longer
pushed through the file lookup meant only for the graphics-processor model.

## 2.1.0 Alpha 72

Fixes the GPU model deploy on a machine that also has a neural processor. In
Alpha 71 the deploy first asked "does this machine serve its Assistant on the
neural processor?", and on the two-accelerator workstation the answer is yes -
so installing the GPU model was captured by that path and quietly re-served the
neural-processor Assistant's own small model instead of starting the 27B on the
graphics processor. The graphics-processor model is now recognised from its
format and routed to the GPU before that question is asked, so it lands where it
belongs and the neural-processor Assistant is left alone. A machine without a
neural processor was never affected.

## 2.1.0 Alpha 71

AI Chat can now run on the graphics processor, on a large model the catalog
recommends only where it will actually run. On a machine with the right GPU (a
Strix Halo / Radeon 8060S), the Local AI catalog now recommends Qwen 3.8 27B in
ROCmFP4 form - a 13.6 GiB model measured at up to ~37 tokens/second on that
hardware - in place of the "this machine could run a larger model, search
Hugging Face" note it used to show there. Installing it serves the model on the
GPU through the ROCmFPX llama.cpp fork (the only engine that reads its FP4
tensors) as a separate, supervised host process, and points AI Chat at it -
while the neural-processor Assistant keeps its own model untouched, so the two
accelerators run two independent models. The GPU server self-recovers after a
reboot the same way the Assistant does.

This is gated to stay honest: the 27B is recommended only when the GPU, the fork
and its libraries are all actually present, never on a Raspberry Pi or a machine
without that GPU, and it is kept out of the ordinary memory-tier ladder so it is
never offered to hardware that cannot load it. A deploy reads GPU memory before
and after the model loads and reports whether the model is really resident,
because a fork that cannot find its libraries answers on the CPU while looking
healthy - the deploy says which actually happened rather than assuming a running
server is an accelerated one.

## 2.1.0 Alpha 70

The Assistant now answers a question about a stretch of time from the history it
keeps, instead of denying it holds one. Alpha 68 gave this machine a seven-day
telemetry record, but the Assistant's answer path still read only the newest
thirty rows - about half a minute - so "how much has the CPU temperature changed
over the last hour" was met with "this appliance holds no record covering that
time" while the hour sat in the database. The Assistant now reads the window the
question names: it turns "the last hour", "the last 7 days", "yesterday" and the
like into a time range, queries that range, and answers with the actual change
over it. When the record genuinely does not reach that far back - a fresh boot,
retention switched off, or a store that cannot be read - it still says so, and
only then. Those windowed figures are downsampled averages rather than raw
readings, and the answer now says as much, so a brief spike that falls between
two averaged buckets is not quietly presented as the whole story. It also no
longer promises "more is retained" for a window the record already covers in
full. Finally, the "Ask Vaelor" header names the model that backs its answers -
the connected LLM - in place of a generic "Evidence-backed" badge, and says
"No model connected" plainly when there is none.

## 2.1.0 Alpha 69

Two fixes from a rendered-UI review of this two-accelerator machine. First,
deploying the neural-processor Assistant no longer destroys the GPU AI-Chat
model. The old logic assumed one shared local model - a single Raspberry Pi -
and so, whenever the Assistant was deployed, it deleted every other managed
local-model credential and moved AI Chat onto the Assistant's model. On a
machine that runs the Assistant on the neural processor and AI Chat on a
separate GPU model, that wiped the GPU model's credential and pointed AI Chat at
the wrong engine. The two models are now told apart by their connection, so
deploying one leaves the other, and AI Chat's assignment, untouched; a
single-model appliance behaves exactly as before. Second, the System screen's
neural-processor panel now says "Serving the Assistant" when the Assistant is
actually running on the processor, instead of reporting that nothing uses it -
the hardware view could not see what the control plane had deployed, and now the
two are combined before the screen is drawn.

## 2.1.0 Alpha 68

The Assistant can now look back over days of this machine's history, and the
x86 workstation finally records that history at all. On a HAT-less workstation
the telemetry writer had been reading the bare hardware bridge, which returns
nothing without a Pironman board, so the machine kept a 7-day retention policy
over an empty database and the Assistant could truthfully only ever see the
current moment. The writer now records the same live readings the dashboard
shows - CPU, memory, temperatures, storage, network and accelerators, drawn
from the system itself - so history accumulates toward the full seven days. A
Raspberry Pi is unaffected: its board's own readings still take precedence. And
the Assistant gained a way to ask for a span of time - "the last 24 hours", "the
last 7 days" - returning a downsampled trend (bucketed, capped) instead of only
the most recent hundred-odd samples, so questions about how the machine has
behaved over a week can actually be answered.

## 2.1.0 Alpha 67

The Assistant's local model and AI Chat's local model no longer fight over one
slot. This appliance runs two independent local models on different accelerators
— the Assistant on the neural processor and AI Chat on the GPU — but Vaelor was
letting only one local generation run at a time across both, a rule meant for a
single-model Raspberry Pi. Worse, if that one shared slot was ever left held,
both models were locked out until a restart, and asking the Assistant anything
that needs the model returned an opaque server error. Now each local model has
its own slot, so the two run at the same time and a stall on one can never
starve the other; when a model really is busy, the answer is a plain “busy, try
again” rather than a 500. Diagnostic logging was added so a stuck slot names
what is holding it.

## 2.1.0 Alpha 66

Custom agents and scheduled routines now work as soon as the Assistant model is
installed — with no per-user setup step. The Assistant workshop, the "run this
without being asked" schedules, skills, and the built-in system-health check
were showing "Model required" and marked unavailable on an appliance whose model
was installed, connected, and answering. The cause was that readiness keyed on a
per-user "intelligence choice" that only gets recorded for whoever first set up
or deployed the model, so anyone else — including an administrator on a machine
where the deploy ran under the internal account — saw a working model as absent.
Readiness now reflects the model the appliance actually has connected; the only
per-user setting that turns it off is the deliberate "basic" (no-model) choice,
and a machine with no model installed still correctly asks for one.

## 2.1.0 Alpha 65

Describing a custom application now turns on by itself once the Assistant model
is installed and working — there is no setting to flip. Before, that whole
workflow (research, planning, then install) was hidden behind environment
switches that were off by default and had no control anywhere in the app, so
the card told administrators to "enable it in the application settings" when no
such setting existed. Now Vaelor gates the workflow on the one thing it truly
needs — a working Assistant model connection — and enables research, planning
and install together when that is present. The approval before every install
and the operator/administrator requirement are unchanged; those are the real
guards. When no model is installed yet, the card says so plainly.

## 2.1.0 Alpha 64

The Assistant on the neural processor now comes back on its own after a restart.
Its model server runs as a child of the hardware bridge, so restarting the
appliance's services (or rebooting) stopped it and nothing started it again —
the Assistant stayed down until someone re-ran a deploy by hand, which every
deploy quietly needed. The workload service now checks at start-up whether this
machine serves the Assistant on the neural processor and, if the model server is
not answering, relaunches it on the same port for the same model, so the saved
connection keeps working with no manual step. It stays out of the way otherwise:
a machine without the neural processor does nothing, a server already answering
is left alone, and the check can never hold up ordinary work. A safeguard also
stops two launches from racing into two model servers on one processor.

## 2.1.0 Alpha 63

The Assistant now answers from the neural-processor model cleanly. After Alpha 62
put Qwen3.5-4B on the neural processor, a live test showed the Assistant still
rambled and was labelled "basic" — because the model server was told to run a
specific model but the appliance recorded no model name for it, and everything
that read that blank fell back to the first model in the server's catalog (a
different, reasoning model whose thinking leaked into every answer). The deploy
now records the exact model the server is running, so the Assistant answers from
Qwen3.5-4B with no thinking leak and is correctly shown as a capable local model.
Neural-processor path only; the Raspberry Pi is unaffected.

## 2.1.0 Alpha 62

Alpha 61 built the machinery to run the Assistant on the neural processor, but a
live test on the Z2 showed it never actually routed there, and the setup screen
still offered the wrong model. This fixes all three, verified on the hardware.
The Raspberry Pi is unaffected.

- **The Assistant deploy now actually reaches the neural processor.** The
  service that decides where to run the model runs in a sandbox that hides the
  `/dev` device node, so the check for "is there a neural processor" always came
  back no and the model stayed on the CPU. The check now reads the device's
  system-class entry, which the sandbox leaves visible, so the decision matches
  the hardware — without opening up the sandbox.
- **Moving the Assistant to the neural processor cleans up the old CPU server
  even when its files are gone.** The retirement is scoped to the Assistant's
  own container by exact project label; the GPU chat server is a different
  project and is never touched.
- **The Assistant setup screen now recommends the neural-processor model on this
  hardware,** instead of offering a Raspberry-Pi-style download. A machine with
  no neural processor still gets the download recommendation, unchanged.

## 2.1.0 Alpha 61

The HP Z2 Mini's Assistant now runs on the neural processor. Alpha 60 chose
Qwen3.5-4B for the neural processor and pointed the appliance at it, but there
was no machinery to actually launch it there — the Assistant was still answering
on the CPU. This builds that machinery. None of it touches the Raspberry Pi,
which has no neural processor.

- **Vaelor now launches and supervises the neural-processor model server itself.**
  It starts the server for the selected model, waits until it is genuinely
  answering, and only then retires the old CPU server — so if the neural
  processor server does not come up for any reason, the Assistant keeps
  answering on the CPU rather than being left with nothing. The server is
  health-checked and restarted if it fails, and bound to the machine's own
  loopback only.
- **The "neural processor" tier is no longer reported as permanently
  unavailable.** Whether it can serve is now discovered from the hardware — the
  server binary, the neural-processor device, and the installed model — instead
  of a hard-coded "no". When any of those is missing, the reason says which.

## 2.1.0 Alpha 60

The HP Z2 Mini's on-device Assistant now runs a better model, and two power
readings on its System screen now show the right number from the right sensor.
None of this touches the Raspberry Pi — the model and power choices here are
made only on hardware that has the Z2's neural processor and GPU.

- **The Z2's Assistant now runs Qwen3.5-4B on the neural processor.** It was
  chosen the hard way: every candidate was scored for accuracy on a 364-item
  evaluation and for speed on the actual neural processor, then the two were
  weighed together. Qwen3.5 answered 95.88% correctly on the neural processor
  with no invented facts, and it replaces the model the appliance carried
  before. (Decision VD-108.)
- **The Graphics card now shows the GPU's own power, not the whole chip's.** On
  this integrated GPU the card was reading the package sensor — about 29 W while
  the GPU sat idle — when the GPU itself was drawing 0.03 W. It now reads the
  GPU's own channel on integrated graphics, and keeps the package sensor for a
  discrete card, where that sensor is the GPU. When the reading is unavailable
  it is left blank rather than back-filled with the package number.
- **The Processor card's package power now comes from the same measurement as
  the GPU and neural-processor power,** so the three figures agree instead of
  being sampled by different tools at different instants. A cross-check line
  says whether the two available power sources agree, or by how much they
  differ when they do not.

## 2.1.0 Alpha 59

Five readings on the x86 workstation that showed "?" or a wrong value are now
correct. A clean install of the workstation surfaced them; each had been fixed
in the backend already, but the System screen was still showing a hard-coded
placeholder instead of the value the appliance had measured.

- **The workstation keeps its telemetry history without a Pironman enclosure.**
  The check for whether history was switched on read a setting that only the
  Raspberry Pi enclosure provides, so on a workstation it retried for thirty
  seconds and then reported that the hardware bridge might not have started —
  leaving history switched off on a machine whose store was healthy. A machine
  with no enclosure now keeps its own seven-day history, and the Pi's enclosure
  setting is unchanged.
- **The processor now reads "16 cores · 32 threads"** instead of showing the
  core count labelled as threads.
- **The neural-processor activity no longer claims amd-smi is not installed
  when it is.** Where two builds of amd-smi are present, the appliance now uses
  the one that actually publishes the processor's activity and power.
- **Package power now shows the watts it reads**, rather than explaining that it
  cannot read them.
- **Memory ECC states "None" honestly** on a machine that reports no
  error-correcting memory, instead of an ambiguous "not reported".

## 2.1.0 Alpha 58

Three defects found by testing the appliance the way an owner uses it, then
fixed and re-tested.

- **The Assistant no longer claims it has only seconds of history when it has
  days.** Asked "what has the CPU temperature been the last few hours", it read
  the most recent samples and told you that was everything the appliance had
  retained — while the store actually held over three days. It now says it read
  a recent window and that more is kept, instead of mistaking the window for the
  whole record.
- **A factory reset now erases the whole assistant, not most of it.** Custom
  agents, their action grants, their connector history and the AI Chat
  knowledge store were being left behind, so a reset (which an owner uses to
  hand the appliance on) kept the previous owner's agents. The reset now clears
  the entire assistant store, and a test fails if any assistant record it writes
  is not covered.
- **AI Chat stays truthful and available when the local model is busy.** The
  appliance runs one local model that the Assistant and AI Chat share, and it
  answers one request at a time. Overlapping requests used to pile up until the
  connection dropped for everyone and the screen blamed the network. A request
  that arrives while the model is answering is now told, plainly, that the model
  is busy and to try again — and it stops piling on.

## 2.1.0 Alpha 57

Faster loading, driven by the optimization gate (#206): a front-end review found
that the interface shipped as one large file and re-downloaded on every visit.

- **The dashboard loads about two-thirds less code up front.** The interface is
  now split so a screen you have not opened yet is not downloaded until you go to
  it, and the parts that rarely change (the framework itself) are kept in a
  separate file the browser can reuse. Initial download drops from ~842 kB to
  ~305 kB.
- **The browser now keeps the loaded interface instead of re-fetching it.**
  Each built file has a fingerprinted name, so it is served with a "cache
  forever" instruction and a return visit reuses it from disk. The one file that
  points at the others is always re-checked, so a new release is still picked up
  immediately and never strands you on an old version.
- **Opening a screen for the first time stays instant** — the not-yet-opened
  screens are quietly fetched in the background once the page is idle.

## 2.1.0 Alpha 56

A visual-consistency and plain-language pass over the whole v2 interface, driven
by three UI reviews, implemented against the current tree, checked by a separate
reviewer that did not write it, and reworked from that reviewer's findings before
it shipped.

- **Every page title is now the same size.** The Cluster and Apps headings were
  noticeably larger than the rest; all nine pages now share one heading size, and
  on the System page a section title no longer renders larger than the page title
  above it.
- **The Apps page lines up with every other page.** Its content column was inset
  and centred; it now uses the same width as the rest of the app, so switching to
  it no longer shifts the layout.
- **The Home status row no longer starts with a stray divider**, and status
  labels read the same everywhere (a single set of type sizes, radii, and spacing
  tokens now backs the interface instead of dozens of one-off values).
- **Plainer wording.** The Activity page drops internal phrases like "server-owned
  operations projection"; AI Chat drops "cited retrieval"; a config file path and
  environment-variable name no longer appear in an on-screen message; and the
  page-refresh control is called "Reload" everywhere. The delete-a-chat warning
  keeps its honest note that a fast-wake snapshot's data can linger on disk.
- **Faint disabled controls and near-invisible keyboard-focus outlines are
  fixed**, and text that had shrunk below the legible floor was raised back to it.
- **Telemetry retention reads cleanly.** On an appliance carrying the legacy
  30-day setting, that value is corrected to the applied 7 days, so the app no
  longer shows the configuration disagreeing with itself.

## 2.1.0 Alpha 55

A no-holds-barred adversarial pass on the Raspberry Pi — fresh testers trying to
break every feature — drove this release. Each fix below was reproduced live,
then re-broken by a second reviewer and reworked before it shipped.

- **Cancelling an app install no longer bricks that app.** A cancelled install
  used to leave hidden state that made every later install of the same app fail;
  the half-finished install is now rolled back cleanly.
- **Nonsense resource limits are refused.** An app asking for a trillion GB of
  memory used to deploy "healthy"; limits are now sanity-checked before Docker
  ever sees them.
- **A raw compose import can't overwrite the AI model's runtime** or silently
  replace a managed app without approval, and a crashed import no longer blocks
  future imports of that app.
- **The Assistant answers "what OS is installed?" and "how many CPU cores?"**
  with the real facts, and no longer answers a world question ("how many cores
  does an M4 have?") with this machine's numbers. A saved note can no longer act
  as an instruction to the model.
- **The status page stops carrying the bench box's benchmark numbers** on a
  CPU-only Pi, the raw storage API no longer counts one disk four times, and the
  telemetry retention period (7 days) is now shown in the app.
- **The control plane is served by a production web server** instead of a
  development one, with the version-disclosing header removed, sized so live
  dashboard streams don't starve ordinary requests.
- **The AI model's answer-length calibration succeeds on the Pi** instead of
  always timing out and falling back to a conservative default.
- Clearer messages where controls were disabled or errors were raw (assisted
  research, operation cancel, audit rows, invalid list limits).

## 2.1.0 Alpha 54

Alpha 51 taught the appliance to say what its memory shows. This release makes
it honest about *itself* — what machine it is, what it can run, and what version
it is on — after a live pass on the Raspberry Pi caught it reporting a different,
larger machine's identity as its own.

- **The Pi stops describing itself as the bench box.** The inference status read
  a configuration measured on another machine: a neural processor and a GPU this
  appliance does not have, two models kept resident that it never loads, and a
  memory headroom that came out *negative*. It now describes the machine you are
  actually on — one model on the CPU, sized to the memory that exists — while the
  two-accelerator layout is still used on hardware that has both.
- **"What version am I on?" gets an answer.** The Assistant used to push that to
  AI Chat as if it were general knowledge, while its own identity path could
  answer it. One place now decides what counts as a question about this machine,
  so the Assistant and its scope check can no longer disagree — and "what release
  is the next Ubuntu?" is still not answered with Vaelor's version.
- **The model picker no longer offers models this machine cannot run.** Setup
  listed accelerator-only models beside the one that actually runs here; it now
  shows only what this hardware can serve.
- **The status page names the model it is running.** The processor's engine
  reported "no model" while a healthy model answered every question; it now
  reflects the deployed model and its health.

## 2.1.0 Alpha 51

Alpha 50 gave the appliance a memory. This release lets it say what the memory
shows, instead of inviting you to ask a question it could not answer.

- **Ask how a reading has been moving and you get an answer.** "Show me the
  temperature trend over the last hour" now says which way it went, by how
  much, and — the part that matters — **how long the samples it used actually
  cover**. Thirty samples a second apart are thirty seconds, not an hour, and
  it says so rather than letting you assume.
- **It no longer offers something it cannot do.** The previous release replied
  "ask for a trend and I can answer from those", and then answered a request
  for a trend with the current reading. That invitation is gone; a reply either
  states a trend or says plainly what it holds.
- **An answer about the CPU is not given to a question about something else.**
  Asked whether the fan had sped up, or whether a backup ran, it would have
  reported the CPU's temperature. It now says which readings it keeps and
  declines the rest — while still answering "did the CPU get warmer during the
  backup", where the backup is just when, not what.
- **The browser desktop no longer opens on Ubuntu's setup wizard.** A rebuilt
  desktop account met "Welcome! — choose your language" before it met your
  desktop.
- **An old internal account name is gone**, along with the code that still
  looked for it.

## 2.1.0 Alpha 50

Ask the appliance what happened overnight and it said it keeps no record. It
was switched on to keep one, and had been keeping none since 30 July.

- **Telemetry history is recorded again, and the Assistant can read it.** The
  part that creates the store and starts the recorder was only ever reached
  from an old start-up path this release does not use, so nothing was written
  and nothing could be read. It now starts with the control plane. History
  begins accumulating from this upgrade; the samples from 20–30 July are in
  the old store and are not carried over.
- **Retention is 7 days.** Thirty was never a choice anybody made — it was a
  default sitting in a file — and this is the release where it would have
  taken effect for the first time.
- **"No history is retained" now means what it says.** Four different
  situations used to produce that one sentence: retention genuinely switched
  off, the appliance still starting up, the store unreachable, and a
  configuration that could not be read at all. They read differently now, and
  the one you can act on says what to do about it.
- **A trend is no longer reported backwards.** Samples were handed to the
  Assistant newest-first while being labelled oldest-to-newest.

## 2.1.0 Alpha 49

The Assistant was not broken in any way you could see. It answered
confidently, and it answered from readings it had never actually been given.

- **Questions are answered from the readings they asked for.** Deciding what
  to fetch and deciding what to show were two separate lists that had drifted,
  so a reading could be taken off the hardware and thrown away before anything
  read it. "What is my ping?" fetched the network status and then showed the
  model something else entirely. Eight of twenty-one ordinary questions lost a
  reading that way; now one does, and it over-fetches rather than under-tells.
- **General knowledge goes to AI Chat again.** One shared word had been enough
  to make "who delivered the Gettysburg address" appliance business.
- **A finished operation stops claiming to be running.** One response reported
  a single active operation beside nineteen rows saying they were live.
- **The desktop stopped reporting success it had not checked.** Turning off
  the screen lock accepted the instruction and changed nothing, and four of
  five failures read as success. Every setting is now read back.
- **Assorted wrong answers.** The Assistant reported the wrong version,
  described hardware the machine does not have, answered "what GPU is
  installed" with a paragraph about nothing being slow, and refused questions
  it held the answer to.

## 2.1.0 Alpha 48

The browser desktop had locked itself behind a password that does not exist,
and the button for getting out of it did nothing.

- **The browser desktop no longer locks itself out of reach.** The account it
  runs as has no password — deliberately, so nobody can log into it directly —
  but GNOME still locked the screen on idle and then asked for one. A
  connected, healthy session, permanently unusable. Commissioning now turns
  the lock off for that account. Nothing is weakened: the session is reachable
  only through Vaelor's own sign-in, and a second lock inside it protected
  nothing.
- **You can end a desktop session that has gone wrong.** "Close session" only
  stopped the browser watching — the desktop kept running, so reopening
  landed you back in the same broken session with no way out but SSH. There
  are now two controls that say what they do: **Stop viewing** leaves it
  running, and **End desktop** ends it on the appliance so the next one starts
  clean. Ending it also revokes the viewing links, so a URL left open in
  another tab cannot reach it.
- **A failure while ending a session is now visible.** The first attempt
  reported its errors on the page *behind* the session window, which looked
  exactly like a button doing nothing — and that is how it was found.

## 2.1.0 Alpha 47

Everything here was found by testing Alpha 46 on the appliance itself. The
Assistant was not broken in a way anyone could see — it was answering
confidently, and answering the wrong thing.

- **The Assistant can use its model again.** It was giving the model 15
  seconds to reply on hardware that needs 45 to 85, so every single answer
  was cancelled and quietly replaced with whatever the appliance could read
  from its own sensors. Asked which version of Vaelor it runs, it replied
  with a list of running services. It now waits as long as AI Chat already
  did on the same machine — where the same model answers correctly in about
  25 seconds.
- **When the model does not answer, the reply says so.** That fact used to
  be available only by expanding a panel, so a substituted answer looked
  like an answer. It is now the first line.
- **Asked about the past, it stops handing you the present.** "What was the
  temperature ten minutes ago" returned the current reading with nothing
  saying the past had not been read. It now answers from the samples this
  appliance retains, or says plainly that no record covers that time. A
  question about the Vaelor version reaches the version, and a question that
  also asks what hardware the machine has gets both halves answered rather
  than only the first — and if one half could not be read, the reply says so
  instead of reporting itself as a complete answer. This is the path you
  meet when the model is unavailable or slow; when it answers, the model's
  reply is still preferred.
- **A stalled job stops pretending to work.** A progress bar was still
  sweeping for an operation whose last event was eight days earlier. An
  operation silent for more than an hour now says when it was last heard
  from and stops animating — an hour, so that a slow image download is
  never accused of having stopped. A job waiting for your approval, or one
  you paused, is never described this way at all: it is waiting, not silent.
- **Remote access is safer to change, and safer to fail.** The password you
  set was being written to the system journal in clear text, because it
  travelled as a command-line argument through the privilege helper; it now
  goes over a pipe and never appears in a log or in `ps`. **If you have set
  an RDP password on this appliance before now, change it and clear the
  journal — the old one is recorded there and no update can remove it.** If
  configuring remote access fails, the service is started again rather than
  left switched off; previously a rejected password could lock you out of
  the machine you were fixing it from. The password then in force is
  whichever one got written before the failure, so if the error arrives late
  in the process, use the password you just typed. Installing the browser
  desktop no longer fails on a fresh appliance, where the check for a
  leftover session treated "there is no session" as an error. And a failure
  to *read back* the settings now says that is what happened, instead of
  reporting that every one of them failed to save.
- **Three smaller corrections.** The "Remote access" card printed each row's
  description on top of its own heading. Browsing the model catalogue
  silently saved your clicks as Assistant conversations — including into a
  conversation you already had open. Paused telemetry claimed to be fifteen
  seconds old no matter how long you were away.

**Not fixed, and worth knowing.** The Assistant's prompt cache is still
switched off by its own arithmetic: the engine needs to store about 288 MiB
for one prompt and the cache is bounded at 217 MiB, so it skips caching and
every question pays a full prefill. A fix was written for this release and
withdrawn during review — it set the bound 25 KiB below the size it had to
hold, which would have reserved 71 MiB more and still skipped, on a machine
that has been killed for memory three times. Sizing it properly needs a
measurement of the largest prompt this window can produce, which nobody has
taken yet. It is tracked as #157.

**How this release was checked.** Two reviewers who did not write it went
through it before it shipped, and between them they found nine things wrong
with the fixes above — including two that had rebuilt the very defect they
were written to remove, and one where the Assistant would have been given
*less* time than AI Chat on hardware both had measured. All of it is
corrected here. The full account is in `DECISIONS.md` under VD-086 and
VD-087.

## 2.1.0 Alpha 46

The fast wake the Pi's model was chosen for finally runs, and six of the
project's own safety checks were rebuilt so they can actually fail.

- **Returning to a recent conversation after a quiet period is fast now.**
  The appliance's model unloads itself when idle, and until this release the
  first answer afterwards re-read everything from scratch — about 95
  seconds, worse than a cold start, despite the fix for it being measured
  and decided a fortnight ago. Vaelor now saves each conversation's working
  state to disk as it answers and restores it when you come back, taking
  that first answer to about 20 seconds. The three most recent conversations
  are kept, within a fixed 366 MB. A restore happens only when it can be
  proved current: if the machine's facts changed — an upgrade, new hardware —
  or the conversation moved on, Vaelor takes the slow path rather than
  answer from a stale picture, and a brand-new conversation still pays the
  slow first answer. The model card now says exactly that, no more.
- **Six safety checks that could never fail, now can.** Independent review
  found guards written in earlier alphas that would stay green if the thing
  they guarded was deleted — a test that re-implemented the rule it was
  checking (with a hole the real rule didn't have), an index verified
  against itself, an installer check that couldn't tell a command from a
  comment or notice the restart running before the install, and a cluster
  health verdict whose wiring could be removed without a single failure.
  Each was rebuilt and then proven by reintroducing its defect and watching
  it fail. The constant-sharing guard also learned the defect's live shape —
  one function mixing two machines' measured numbers in its body — and the
  citation index now reads the training corpus, which quietly cited seven
  decisions the report called uncited.
- **Reviewed before it ships, and corrected.** Two independent reviewers who
  did not write this release attacked it before deploy, and everything they
  found is fixed in it: a half-dead model engine or two simultaneous
  questions can no longer turn a saved-state hiccup into a failed chat turn;
  the Assistant no longer swears "no history is stored" on an appliance that
  retains telemetry samples — asked about the past it now answers from them,
  and says a record is missing only when it is; a half-typed draft no longer
  follows one account into another's session on the same browser tab;
  "already installed" claims now require the file's byte length to match the
  verified artifact, not just its name; the delete dialog says exactly what
  deleting a conversation does and does not erase; and an upgrade
  re-requests the memory profile the owner asked for rather than the one an
  old estimate once walked it down to.

## 2.1.0 Alpha 45

The remaining live findings from the Alpha 41 review, closing the Pi's
ready-to-fix list.

- **Anyone can read this appliance at their own font size.** The desktop
  density is now a percentage of the browser's default rather than a hard
  13px, so raising the browser's font size works — pixel-identical for
  everyone at the ordinary setting.
- **The rail is made of links.** Destinations middle-click, open in new
  tabs, and copy as links, exactly like the content cards already did. The
  Settings tabs answer arrow keys, and every action button names its
  account, so Disable and Delete cannot be confused by ear.
- **A part-written form survives leaving the page.** The account form keeps
  its username and role, and the Assistant keeps a half-typed question —
  never a passphrase — instead of discarding them without warning.
- **The Assistant is told its own version, and told to answer all of every
  question.** Asked which Vaelor it runs, it answered about services; asked
  about 3 a.m., it presented the current reading as the past; asked a
  two-part question, it dropped half. The machine facts now carry the
  version, and the rules require every part answered, absences stated, and
  the present never dressed as history.
- **One model is current, everywhere that says so.** Only the newest deploy
  of the serving model claims "Active in Assistant"; Manage's in-use row
  offers its runtime settings instead of "Use model"; the setup card says
  when a model is already installed or serving instead of offering its
  download; one file has one size in one unit on every screen; and the
  running memory cost appears where the choice is made, not only after it.
- **Files on the appliance for Vaelor's own evaluation say so** instead of
  offering a "Use model" button into an unsupported configuration, and the
  4B's card no longer claims a reasoning advantage its 0.11 GiB size
  difference does not support.


## 2.1.0 Alpha 44

Two decisions taken with the owner on 2026-08-11, both closing gaps found by
independent review of earlier alphas.

- **A model served to the cluster is sized from its measured footprint,
  never from its file size.** The old arithmetic capped the shipped model
  below the memory its process demonstrably uses, so a worker would have
  restarted it five times and given up. The limit now comes from the same
  measured table the single-node deploy uses — deriving the same verified
  figure — the service pins the context window that measurement was taken
  at, and a model nobody has measured is refused with instructions to
  measure it, on the approval screen as well as at deploy time, so the
  number you approve is the number the service gets.
- **An upgrade preserves the memory profile actually in effect, not the one
  last mentioned.** Each deploy now leaves its effective profile in the job
  record, and the post-upgrade refresh prefers that recorded fact. The
  previous recovery scanned old requests and could resurrect a choice made
  long ago over the setting the machine was actually running — observed on
  this appliance, where it tightened the container limit by 768 MiB.

## 2.1.0 Alpha 43

Fixes for the remaining live findings from the Alpha 41 review, corrected by
independent reviewers before commit — including one wiring fault those
reviewers caught that no test had.

- **Cluster now reads Docker the way the rest of Vaelor does.** The Cluster
  screen said "the container runtime is not reachable on this appliance"
  while five Vaelor-managed containers ran beside it: the screen was asking
  Docker directly from a service that deliberately holds no such permission.
  Its readings now go through the workload broker, which holds it. When a
  reading genuinely fails, the message names the read and the process that
  asked — never the machine — and the next step is a control the owner has.
- **A failed operation is described one way.** One Activity card said Failed
  in informational blue, "In progress" beside a partly-filled progress bar,
  and "The operation completed." — with the same error sentence printed four
  times and a raw error code. The state pill's colour now agrees with its
  word, an ended operation carries no progress bar, the result line agrees
  with the state, and the error appears once.
- **Validation errors look like errors on every form.** On three of four
  forms an error rendered identically to the help text beside it, because
  each screen's own prose styling out-ranked the shared error style; that is
  fixed once, in the shared style. The account form names the rule actually
  broken, an over-long name is refused instead of silently cut at 64, each
  cooling level names its own fault, an emptied cooling field no longer
  becomes a literal 0, and typing 999 into a colour channel can no longer
  half-apply as 99 and repaint the preview with a colour never chosen.
- **AI Chat refuses immediately when the connected server offers no model.**
  It used to send the question anyway, hang, and blame the appliance —
  "The appliance took too long to respond" — for a condition it already
  knew. The refusal now names the server and says what to do on it.
- **A connection test result says when it ran.** Settings showed a green
  "success" indefinitely, however stale. A pass now reads "Passed 3 weeks
  ago", a failure is drawn as one, and rotating a secret clears a result
  that was measured on the old secret.

## 2.1.0 Alpha 42

- **The Remote console now offers an address a client can actually reach.** It
  could show one of the appliance's internal container addresses instead of its
  address on your network - behind a green "remote access available" badge - so
  copying it into a remote desktop client would never connect. Which address it
  picked depended on how many apps were running, so it came and went.
- **Setup history shows when things actually happened.** Every entry read "just
  now", and the exact date behind them was tens of thousands of years in the
  future, so there was no way to tell a change made this morning from one made
  last month.

## 2.1.0 Alpha 41

Everything in this release is a correction to Alpha 40, found by reviewers who
had not written it. Alpha 40 was not deployed.

- **A workstation is no longer given a Raspberry Pi's context window.** The
  measurement that set the Pi's window was reaching every machine, so lowering
  it for the Pi silently halved the window on a workstation with a graphics
  card and 128 GB of memory - while that machine went on reserving the memory
  it no longer used.
- **Installing an upgrade no longer changes the memory profile you chose.** If
  you had set the assistant to the smallest profile on a memory-tight machine,
  an upgrade quietly moved it back to the recommended one, doubling its context
  and raising its memory limit. Your choice is now carried across the upgrade.
- **An upgrade only refreshes the assistant's own deployment.** If you had
  installed another app that runs a local model, the upgrade could take that
  app's model and port and apply them to the assistant.
- **A deploy that cannot measure free memory now says so.** It previously
  reported that the model did not fit the machine - on a machine that was
  running that exact model at that exact size while it said it. Nothing is
  changed when this happens, and retrying a moment later is the fix.
- **The assistant model this appliance ships now tells you what it is weaker
  at**, as the other models already did. It was chosen for answering quickly
  after a quiet period rather than for the highest score, and that trade is now
  stated on the setup screen.
- **Memory figures record how they were counted.** Two different ways of
  measuring a running model differ by about 1.7 GB, and mixing them is how a
  configuration that demonstrably runs gets refused.
- **A model served to a cluster now gets the same memory bounds as a local
  one.** It was inheriting the inference engine's own defaults - a prompt cache
  larger than a worker's entire memory, and four copies of the context where
  one was budgeted - underneath a hard limit.

## 2.1.0 Alpha 40

- **An upgrade now re-applies its settings to assistants that are already
  running.** Previously an install replaced the code that decides how the local
  assistant is served and left the running one exactly as it was, so
  improvements sat unused until somebody redeployed the model by hand. On this
  appliance that meant three tested fixes were installed and inert, and the only
  sign was memory slowly climbing.
- **The appliance says when it has lost the network address it was clustered
  on.** After a router change, the machine went on advertising an address it no
  longer had; the only symptom an owner saw was a machine that would not shut
  down cleanly. It now reports the mismatch and says what to do about it -
  clustering needs a fixed address, so give the machine a reserved one and
  re-initialise.
- **The assistant is no longer offered tools for hardware this machine does not
  have.** A Raspberry Pi has no neural processor, and the assistant's own notes
  said so - and then listed a tool for reading that processor's activity. Being
  offered a tool is an invitation to use it, and using it means reporting on a
  device that is not there. The two halves now agree.
- **An upgrade restarts every Vaelor service, and the installer proves it.**
  Restarting only some leaves the rest running the previous release from memory
  while every version reading says the upgrade landed - and the one most likely
  to be missed is the service that performs deployments.
- Fixed: the version number had not moved in six releases, so a rebuilt package
  could install over itself with nothing to tell the two apart.

## 2.1.0 Alpha 35

- **The local assistant no longer creeps up on the memory it is allowed.** The
  inference engine was permitted to keep remembered pieces of past questions in
  a store larger than the whole appliance, so it never had a reason to let any
  of them go. Memory rose with every question asked until the limit arrived.
  It is now bounded to a share of what the machine actually has, and settles
  after about twenty questions instead of climbing indefinitely. It is also
  slightly *faster* that way, so nothing was traded for it.
- **The settings the recommended model needs now include that bound**, beside
  the context size and the saved-notes path, so a newly set up appliance
  inherits the configuration that was tested rather than the engine's own
  defaults. The appliance still takes the smaller of what the model asks for
  and what the machine can spare, so a smaller box is never handed a setting
  measured on a larger one.
- **The inference engine is pinned to an exact build.** It was previously
  tracked by a moving label, and it had already moved - the appliance was
  running a different build from the one every published measurement was taken
  on, and nothing recorded which. Upgrading it is now a deliberate change
  rather than something that happens on the next download.
- **The memory reserved for the assistant is now set from a measurement of
  this model on this machine**, rather than from an estimate that ran about a
  fifth low. The estimate is what had been getting the assistant stopped under
  sustained use.
- Fixed: a deployed assistant kept its original settings when the appliance was
  upgraded, so improvements sat unused until the model was redeployed by hand.

## 2.1.0 Alpha 29

- **The Pi is now recommended a different local model, chosen on the wait an
  owner actually experiences.** Every earlier comparison timed questions asked
  back to back, which is not how anyone uses an appliance. Asked the way people
  really do - one question, a pause, another - the same question took 95
  seconds rather than the 19 the old measurements suggested, because the model
  unloads while idle and has to re-read the standing notes about your machine
  before it can answer.
- **Qwen3 4B Instruct replaces Qwen3 4B as the recommendation**, in a file
  format that suits this processor: about 20 seconds to answer after a quiet
  period instead of 95. It is a little less accurate than the alternative that
  was considered, and answers far more quickly; it also stopped inventing
  answers to questions it should decline, which the previous build did five
  times as often.
- **The settings a model needs now travel with the model.** The context size,
  the file format and the saved-notes path used to be knowledge somebody had to
  remember. A newly set up appliance now inherits all three, so a fresh install
  behaves like a tuned one.
- **"Long context" can no longer ask for more than a model was measured to
  handle.** The larger setting is what took the appliance down in an earlier
  release; choosing a *smaller* one is still honoured.

## 2.1.0 Alpha 28

- **Fixed: the local AI server could be allowed more memory than the appliance
  had to give, which stopped the whole machine.** Vaelor checked that the
  model's measured requirement fitted, then added a safety margin on top and
  never re-checked the total. On the Pi that issued a limit 848 MB beyond what
  the machine could spare, and everything else — the web interface, remote
  access, the other apps — was squeezed until it stopped responding. It needed
  a power cycle.
- **Being too generous here is worse than being too strict.** With a limit
  below what the machine can spare, an over-hungry model is stopped and
  restarted and the appliance keeps working. Above it, nothing stops the model
  and the appliance goes down with it.
- The screen now reports the margin the model actually got rather than the one
  intended, so a configuration that only just fits is visible instead of having
  to be worked out.

## 2.1.0 Alpha 27

- **Fixed: the local AI server was given less memory than it was measured to
  need, and the system killed it.** Vaelor has measured figures for how much
  each model really uses, and shows them to you before you install — "fits,
  from a measured footprint for this model on this platform". The step that
  actually created the container was not looking them up, and used a rough
  estimate instead. The estimate was about 20% low, in the direction that gets
  a model killed mid-answer. On the appliance this happened after 36 questions
  in a row.

## 2.1.0 Alpha 26

- **The Manage tab now loads.** Listing your apps and models was re-reading
  every AI model file on disk from end to end to compute a checksum nobody on
  that screen looked at - about 7 GB over the memory card, 65 to 100 seconds,
  on every single load. No browser ever waited that long, so the list never
  arrived. The checksum is still taken where it matters, when you remove a
  model and Vaelor has to prove the file it deletes is the file you reviewed.

## 2.1.0 Alpha 25

- **Fixed: the Assistant status page returned an error for anyone whose model
  could not be reached.** Choosing a model and being able to reach one are two
  different things, and one line added last release assumed the second. Live on
  the appliance for twenty minutes.
- **Fixed: reinstalling a model shrank its memory.** The setup read how much
  memory was free at a moment when the model it was about to replace was still
  holding all of its own — so it saw a small machine, and gave the new model
  half the working space. On the Pi that dropped the Assistant below the size
  of its own instructions. It now accounts for the memory that is about to be
  released, and it reads what the old model is actually using rather than what
  it was allowed.
- **A model whose working space is too small for the Assistant now says so.**
  The previous message was "built the context window it asked for", which was
  true about the server and silent about the thing that mattered.
- **Screens no longer report a failed check as a finding.** If the app-setup
  check has not answered, it now says it is still checking rather than "Docker
  is not ready" — and it no longer offers to install Docker over the top of a
  working one. If the installed-apps list cannot be read, it says that, instead
  of "No apps installed yet" on an appliance full of them.

## 2.1.0 Alpha 24

- **The Assistant now gives its memory back on its own, and about 4,200 lines
  of machinery for doing that by hand are gone.** The model server we already
  ship can unload an idle model and reload it on the next question — it always
  could, on a single setting nobody had read. Vaelor now sets it: fifteen
  minutes of quiet and the weights are released. Measured on the Pi, that
  returns about 1.2 GB on the small model and about 5 GB on the recommended
  one.
- **Nothing stops, and nothing has to be started again.** The first question
  after a quiet spell takes about twenty seconds and is answered. There is no
  "starting" screen any more, no waiting for a container, no request that fails
  because the model was asleep — those states existed to manage a wait that,
  on this path, does not happen.
- Deleted with it: the background sweeper, the shared status file the two
  services passed between them, the model lifecycle job, the chat and agent
  waits, and every state the screen used to render for them. The card now says
  one thing — that the model rests when idle and what the first question after
  that costs — and says nothing at all when nobody has measured your model.
- The measured weaknesses of the installed model still appear, unchanged. They
  are about the model, not about when it is running.

## 2.1.0 Alpha 23

- A wedged model is now reclaimed after a restart of the background worker, not
  only before one. The card said "Not answering" while the part that actually
  frees the memory still saw "starting" — so after every upgrade, reboot or
  service restart, a model that had stopped responding held its 5.4 GiB
  indefinitely. Both now read the same state.
- The "starting in about 37 seconds" promise is gone from the sentence you read
  when nobody has measured your model at your context size. Only the hidden
  field had been fixed; the visible text still quoted the 4B's figure.
- Removing an app no longer waits behind the model's idle timer. The lock added
  last release was taken for every removal, so deleting Grafana serialised
  against the model housekeeping and could make the Assistant briefly report its
  own state as unknown.
- Replaced tests that checked the agent-run behaviour by searching their own
  source text. Six ways of breaking that behaviour passed those tests,
  including reverting the previous release's fix entirely. They drive the code
  now, and all six fail.
- Corrected the memory figure wherever it appears, and two comments that
  contradicted the code beside them — both of which the previous release
  claimed to have already fixed.

## 2.1.0 Alpha 22

- **The on-demand model never actually worked, on any machine, until this
  release.** The background worker decides whether the model is awake by asking
  it — and the thing it used to ask with was only ever given to the web
  interface, never to the worker. So the answer was always "no". The model
  would start, serve perfectly, and the appliance would insist it was still
  waking up: every Assistant question answered with "starting, about 37
  seconds" and never an answer, and the memory never given back. The worker now
  reads the model's address out of the file that published it, so there is no
  longer anything to forget to connect.
- Stopped a model that is up but silent from being unreclaimable. It read as
  "still starting" forever, and the idle timer only ever considered models that
  were either healthy or crash-looping — so the one case where getting 5.4 GiB
  back matters most could not happen.
- Fixed the agent wait reading the wrong engine. It was checked before the
  setting that decides which model a task uses, and against an account
  preference rather than the connection, so five of six combinations were
  wrong: it skipped runs that needed it and delayed runs that did not.
- Gave a running agent a heartbeat. Marking the model in use once at the start
  is not enough for a twenty-minute run; it was stopped at fifteen.
- An unmeasured model is no longer promised the 4B's 37 seconds.
- Model removal now takes the same lock as deployment, so it cannot race the
  idle timer against the same containers.
- Corrected the memory figure everywhere it appears: 5,542 MiB is 5.4 GiB.

## 2.1.0 Alpha 21

- Connected the agent wait that Alpha 20 said it had connected. The code was
  written, tested and described in the release notes, and nothing constructed
  the agent runner with it, so an agent run on a Pi with the model stopped
  still went at a dead endpoint and started nothing. It waits now, and a test
  asserts the wiring rather than the mechanism.
- Stopped every non-Raspberry-Pi permanently accusing itself of a fault. The
  platform gate added in Alpha 20 stopped anything writing the model's state on
  those machines, so the card read "Vaelor does not know what the local model
  is doing — this is a fault in the appliance's own reporting" forever, on a
  working machine. The state is observed and reported everywhere; only the
  stopping is confined to the Pi.
- Stopped a retry keeping a broken model in memory. Asking again after "the
  model is not answering" counted as use, which reset the fifteen-minute idle
  clock, so the one case where reclaiming 5.4 GiB matters most never happened.
- Counted AI Chat as use of the local model. On a Pi both surfaces share one
  model, and only the Assistant was recorded — so a long AI Chat conversation
  had the model stopped under it, with nothing in AI Chat able to start it
  again.
- Fixed who counts as using the local model. It was read from a per-account
  preference that is only ever set for whoever installed the model, so a second
  operator answering from the same model recorded nothing and had it stopped
  under them. It now asks the connection that is about to answer.
- Made the card quote the model you installed, everywhere. The countdown came
  from a fixed 37 seconds while the shortcomings list came from the real
  measurement, so on the smaller model the same card said 37 and 6. Both come
  from the same record now, and a machine that keeps its model resident is no
  longer told when it starts and stops.
- Stopped the countdown saying "About 0 seconds" for the seventy-four seconds
  between the estimate and the point where a start is treated as failed. It now
  says it is taking longer than usual.
- Stopped a model redeploy racing the idle timer. Both issue Docker commands
  against the same project from different threads, and a stop landing inside a
  redeploy's health check rolled back a good deployment.

## 2.1.0 Alpha 20

- Gave the Raspberry Pi's local model an on-demand life. It costs 5.4 GiB of a
  7.7 GiB machine, which is too much to hold all day for something used a few
  times, so it now starts when you ask a question and stops after fifteen
  minutes of quiet. Measured on the appliance: stopping returns **4,322 MB**,
  taking the machine from 21% free to 76%, and starting takes **35 seconds**
  against the 37 the product promises. The Assistant says which state it is in,
  how long a start takes and after how long it stops, so a 37-second silence
  reads as the appliance working rather than as a broken one.
- Stopped the idle timer killing a conversation in progress. Use was recorded
  only when a question needed the model *started*, which happens once per
  session — so the stop fired fifteen minutes after your **first** question,
  mid-answer, however much you had asked since. Every request is now recorded,
  and a long agent run keeps the model alive for as long as it runs.
- Stopped a dead model reporting itself as nearly ready. A container that was
  up but not answering read as "Starting — about 37 seconds", indefinitely,
  because the elapsed clock lived in memory that a restart cleared. There is a
  new state for it that says the model is running and not answering, and asks
  you to restart it. A crash-looping container used to render as an orderly
  "Stopping — returning memory"; it now reads as the fault it is.
- Stopped a local-model gate blocking people who do not use the local model. An
  owner on a hosted provider got an error on every chat turn, each one queueing
  a start of a 5.4 GiB model whose answer they would never see.
- Confined all of the above to the Raspberry Pi. There was no platform check at
  all, so an x86 workstation was told its always-on model starts in 37 seconds
  and stops after fifteen minutes, with shortcomings measured on a different
  model on different hardware.
- Made the "what this model is not good at" list describe the model you
  installed. It was hard-coded with the 4B's figures and shown for the smaller
  alternative too, where the answer time, the memory and the start time are all
  different. It now comes from the same measured record the sizing uses, and
  says nothing at all for a model nobody has measured.
- Fixed a docker timeout during housekeeping taking the whole job executor
  down, and moved the model's state check off the job thread, where a long
  application deploy blocked it for so long that the Assistant reported its own
  state as unknown and could not be started.

## 2.1.0 Alpha 19

- Stopped a redeploy telling you the GPU is broken when it is fine. The check
  that proves the accelerator is really in use reads how much accelerator
  memory the new model is holding, and it took its "before" reading while the
  model being replaced was still resident — so the incoming model appeared to
  hold nothing, and the deploy reported "Running on CPU — the GPU library did
  not load" at 95% on a machine where it had loaded. The reading is now taken
  with the previous container stopped, and a reading that cannot be attributed
  to this deployment is reported as unanswered rather than as a failure. The
  engines panel asks the same question without a "before" reading, and now says
  which of the two it answered instead of quietly disagreeing.
- Made the Assistant answer questions about faults. "What does the error mean",
  "what is causing the warning", "what should I do about the alert" and "what
  does bootstrap required mean" were all sent to AI Chat — which cannot read
  this machine — because the scope vocabulary had no word for something being
  wrong. Adding "this" to any of them fixed it, which is not a phrasing anyone
  owes the product. Fault and control-plane vocabulary is now in scope, and a
  sweep of 6,367 message-shaped strings confirms the widening did not start
  answering general-knowledge questions.
- Removed the fourth copy of the flash-attention rule. The deploy planner
  decided whether to enable it from the V cache alone, so a quantized K beside
  an f16 V was reported as unadjusted with flash attention off — the pairing
  measured at 1.01 tokens per second — while the container that shipped was
  resolved separately and was safe. The report and the file now come from one
  answer.
- Split model deployment out of the job executor. Deciding what to launch and
  in what order is a different job from reading back what came up, and the
  defect above lived exactly on that seam.
- Corrected the decision ledger. Two unrelated decisions carried the id
  `VD-031` after a clean merge, so looking it up returned an arbitrary one of
  them — with the wrong provenance, which is the expensive half: one is the
  owner's binding choice and the other is Claude's, overridable without
  ceremony. The cluster row keeps the id its fourteen citations use; the NPU
  row is now `VD-039`. A duplicate `VD-022` heading is folded into the row it
  belongs to, and the suite now refuses a ledger id that names two rows or a
  citation that names none.

## 2.1.0 Alpha 18

- Ported the control plane to x86 workstations, keeping the Raspberry Pi
  appliance first class. Platform behaviour now resolves through a real driver
  registry rather than one implementation with an escape hatch, and capability
  availability comes from discovery rather than from machine class — which is
  why every fitted-enclosure path is unchanged.
- Stopped the product describing hardware that is not fitted. A machine with no
  enclosure was told it had "0 enclosure fans" sharing a threshold, two white
  LEDs on GPIO 5, and a Raspberry Pi PMIC as its power source; enclosure
  mutations returned HTTP 500 instead of a reason, and "Apply cooling policy"
  reported success while doing nothing. Reboot and shutdown were unavailable
  because a systemd unit was gated on a Pironman config file and the generic
  power path was written to raise.
- Taught the appliance to see its own accelerators. The GPU and the neural
  processor are discovered, reported, and available to the Assistant as tools.
  Graphics memory is shown as the pair it actually is: on a unified-memory part
  the dedicated carve-out sits near empty while the shared aperture carries the
  model, so a single "GPU memory" figure would be false.
- Corrected four claims the code stated as facts about hardware and which were
  wrong — a reading Vaelor said could not be taken, taken three different ways,
  and a version file looked for in the wrong place. Telling someone a value is
  unavailable when it is readable is the same defect as inventing one, and it
  hides better. Notes in this area now name what was read and from where.
- Sized local inference from measurement instead of assumption. The appliance's
  own agent prompt is longer than the context window deploys used to request; a
  model cost four times its necessary memory because nothing set a context size;
  loading a second model silently evicted the first and reported success; and
  the recommended configuration sat behind an environment variable nobody sets.
- Chose the Assistant's model on whether it can do the job, not on speed. Only
  one local model returns well-formed structured output at the prompt length
  this appliance actually sends. That threshold is enforced in code rather than
  described in a comment.
- Defaulted GPU inference to ROCm on measured prefill against the current
  runtime, after a mid-measurement runtime update reversed the earlier answer.
  The figures carry the revision they were taken on, and the superseded ones are
  kept beside them so the next person can tell what has aged.

## 2.1.0 Alpha 17

- Let the model read what the run fetched. Agents reported "I do not have live
  internet access" while their fetches were succeeding: the pages were being
  discarded before the prompt, because the synthesis pass sized every model's
  context with a 700-character constant tuned for the appliance's own 1.7B. One
  real run carried 21,449 characters and the model received 819, of which the
  granted context was a snapshot holding two search-result URLs. Context is now
  sized against the model's own window minus the generation it is asking for,
  and overflow loses detail rather than structure - it finds the longest
  per-string limit that fits instead of collapsing to a snapshot.
- Stopped refusing pages for reasons that had nothing to do with the page.
  A host resolving to more than eight addresses was refused outright, so naming
  `www.espn.com` as an allowed domain still returned nothing; the cap now
  narrows the pinned address set instead, and a private address anywhere still
  refuses the fetch. The decompressed ceiling sat below an ordinary news page.
- Fixed a failure that took 145 seconds to report a rejection the server made
  instantly. No model profile had ever been measured on this appliance, so
  every connection was unmeasured and was sent a structured-output format this
  server rejects - costing a model load - and the retry was then left exposed
  long enough to be evicted mid-generation. Unmeasured endpoints now get a
  schema only when the caller declares one, a refused format is remembered only
  after the retry without it succeeds, and calibration runs at startup, which is
  when new code arrives.
- Stopped a knowledge collection turning AI Chat into a documents-only tool.
  Attaching one produced "I don't have any reliable information on the history
  of the Forth Bridge" from a model that answers it fine unattached, because the
  prompt presented the retrieved passages as the only permissible ground truth.
  Citation discipline now constrains only claims that carry a citation.
- Made the app say what it just did. "Run this check" cleared the box, said
  nothing, and parked an approval-gated item on a tab you were not looking at;
  the check toggle stayed armed while its disclosure was collapsed, so the next
  ordinary question silently became a check that went nowhere visible. A reload
  mid-answer stranded the question even though the answer had completed.
  Switching tabs reset the elapsed timer to zero. Transient banners reflowed the
  composer under the cursor, and a sentence typed where the box had just been
  was discarded without a character appearing.
- Gave the "Allowed HTTPS domains" field its width back. It rendered as a 6px
  sliver against the right edge - the control governing whether an agent may
  open a page at all - because a grid rule written for checkbox rows also
  matched text fields.
- Closed an authorization gap an adversarial review proved on a live appliance.
  An operator could not see the Routines tab, and could not read skills or
  curated memory, but the API still accepted their schedule and alert-rule
  creation — so they could install a standing rule that launches agent runs
  nobody approves. Creating, pausing, and deleting schedules and alert rules is
  now administrator-only, matching the tab that has been administrator-only
  since Alpha 8. Reading the list stays at operator, because History uses it to
  label a run nobody typed.
- Suspended unattended rules whose owner lost their access. Gating the endpoint
  does nothing about a rule already in the database, or about the account that
  made one being demoted afterwards, so ownership is now re-checked at fire
  time against the same user table the request path uses. A rule whose owner is
  no longer an enabled administrator records a blocked run and creates nothing.
- Said plainly what a scheduled or triggered run is allowed to do, and stopped
  claiming it waits for an approval it never waited for. It does not: creating
  the rule is the approval for its runs, which is exactly why creating one is
  now administrator work. Every schedule and alert rule now carries the pinned
  definition's own read scopes, research policy, and integrations, and the
  screen states that a run reads only and that anything that would change
  something becomes a proposal needing a separate human approval.

## 2.1.0 Alpha 16

- Cut the Assistant on tense instead of on plumbing. Six tabs had become two,
  which moved the density rather than reducing it: Ask carried a live chat and a
  45-row audit archive, and the answer got 146 pixels of a page the archive got
  1,883. Ask, Routines and History now hold what you are asking, what runs
  without you, and what already ran. Ask went from 51 interactive controls to
  10.
- Made the answer visible. The transcript measured zero pixels tall at 1280x720
  and 1024x768 - the reply arrived into a box with no height - because a fixed
  ceiling and two nested scrollers fought the page. It now grows with its
  content.
- Stopped appending a four-step credential wizard to the agent list. A fallback
  meant anyone with one agent got it permanently, unasked, costing about 1,400
  pixels and twenty controls of first paint.
- Stopped opening with a false alarm. The status pill claimed the model was
  missing for two seconds on every load before correcting itself.
- One answer now quotes one temperature. Telemetry read the hottest thermal
  zone while the fan controller re-read zone zero, and a single reply cited
  both. Every fact in an answer now comes from one hardware sample.
- The troubleshooter reads the symptoms you gave it. Told a machine was
  freezing with a loud fan, it reported zero fan RPM as normal and never
  noticed that a silent fan contradicts a loud one. The contradiction is now a
  finding.
- Agent runs say when the model is down instead of showing HTTP 400 and an IP
  address, and lead with what happened and what to do; the endpoint stays for
  operators, at the end.
- Numbered lists keep their numbers. A twenty-item list rendered as twenty
  items all numbered one, because a blank line ended the list.
- Whether an agent may reach the internet no longer depends on how its job was
  phrased, and "Live - updated just now" stops claiming freshness while the
  clock behind it is frozen.

## 2.1.0 Alpha 15

- Let an agent read what its own search finds. A web-research agent could search
  but never open a result: the fetch was gated on a domain allowlist, and the
  wizard's only internet grant leaves that list empty, so every outward-facing
  agent received a page of links it had no way to open. One completed run said
  so plainly - "the provided context contains only search links, not the actual
  game data". An empty allowlist now means "read the results this run's own
  guarded search returned", with every reachability guard unchanged.
- Enforced that provenance on every redirect, not just the first request. A
  vetted result could bounce an agent to any public host, whose content then
  entered the model's context labelled as trusted research; the allowlist was
  equally first-hop-only. A hop is now authorised only if it is allowlisted, is
  itself a search result, or stays on the same registrable domain - re-checked
  inside the isolated broker process, so a broker that enforced nothing cannot
  smuggle evidence past it. Fetches also report where they actually landed.
- Closed a memory boundary that this release had opened. Curated memory is
  administrator-only to list, but grounding it into answers had become
  unconditional on both surfaces, so an operator could read it verbatim - and a
  blank message returned every pinned entry, because an empty search matches
  everything. Grounding now follows the same role as reading.
- Stopped a stored memory forging a citation. Memory was concatenated into the
  prompt beside the retrieved sources, so a memory containing its own
  "Retrieved sources:" heading and an [S1] marker produced a second citation the
  reader could not check. Memory is neutralised before it reaches the prompt.
- Six Assistant tabs became two. Ask Vaelor and Troubleshoot were one question
  sent to two endpoints; Memory and Skills were both "what it knows". Ask now
  holds the question box, the run history and its filters; Agents holds the
  things you build. Memory moved to a page that names both surfaces that use it.
- Fixed appliance checks appearing to time out after a minute while still
  running, and reporting an outcome that had not happened. The cut-off was the
  browser's 60s default on a request the appliance answers in up to 240s, and
  the banner was written when the request returned rather than from the run's
  own state - so a blocked run that executed nothing reported as finished.
- AI Chat renders markdown, keeps the model that answered on each reply, says
  when a search found nothing rather than answering as if none ran, restores
  your question when a send fails, and no longer drops a reply into whichever
  conversation you switched to mid-request.
- Every disabled button in the product looked enabled: `.ui-button` had no
  disabled styling at all, only `.ui-control` did.
- Renaming a conversation saved a single character, because the field
  re-selected its contents after every keystroke.

## 2.1.0 Alpha 14

- Stopped advertising a model that never answers. Readiness was derived from
  whether the endpoint listed its models, which a server can do instantly and
  then time out on every actual request - so a green "MODEL READY" badge sat
  over a hundred per cent failure rate for days. Real inference outcomes now
  feed readiness: two consecutive failures mark the model unusable and say why,
  and one success clears it.
- Sent people somewhere that exists. A matched agent request offered to open
  the "Assistant task ledger"; there is no ledger in this product, and the
  button landed on a page that never mentions one while the run sat on another
  tab behind a collapsed disclosure. The button now says "Open this run" and
  opens the tab the run is actually on. (The first pass corrected only the
  server-side wording and missed the AI Chat button, which was the path people
  were hitting; both are fixed.)
- Said when a schedule was not created. An agent described as running "every
  morning" was created with no schedule and nothing mentioned it, so the one
  thing the user asked for silently never happened.
- Let go of the run surface. Escape did nothing while a run was in flight, so
  the only exit from a sixty-second wait was reloading the page; the run
  already continues in the background.
- Stopped asking people to retype the job they had just described, with their
  own words shown back to them as the placeholder.
- Fixed a naming rule that treated "new" as filler, turning "the New York
  Yankees" into "MLB Scores York Yankees agent".
- Gave Assistant and AI Chat different glyphs. They shared one, which is
  invisible at desktop width and fatal on short viewports where the rail drops
  its labels and the icon is all that is left.
- Kept the mobile bar's labels on screen. The longer canonical names wrapped to
  a second line, grew each item past the bar's fixed height and pushed the text
  off the bottom edge, where nothing can scroll it back.
- Made Escape close the mobile "More" sheet, which otherwise stayed open on top
  of whatever you navigated to next.
- Put an appliance check's outcome where the user is looking. A check that
  timed out cleared the form and printed its explanation on the page behind the
  open dialog, so the dialog appeared to reset itself for no reason.
- Made "Try again" try again. A retry landed in "waiting for your approval" and
  stopped there, so people watched a run that was never going to start. The
  retry now proceeds on the approval the operator just gave for the identical
  request; every run they have not already approved still waits for them.

## 2.1.0 Alpha 13

- Gave a capable model time to answer. The inference budget was capped at 60
  seconds and the agent constructor clamped any larger value back down, so a
  27B model returning structured JSON over the network was cut off mid-answer
  on every single request - the model was fine, the ceiling was not. Connected
  endpoints now get 240 seconds by default, settable with
  VAELOR_INFERENCE_TIMEOUT_SECONDS, and the browser waits longer than the
  appliance does so it can no longer abort a request that is still being
  answered.
- Removed the approval step from runs you start yourself. Typing a request and
  pressing a button, then being asked to approve your own sentence, was a
  checkpoint with nobody on the other side of it and the step everyone got
  stuck on. "Run now" runs it. Requests matched from chat still stop and wait
  for review. (Corrected: this entry also claimed schedules and triggers stop
  and wait. They never did — they have always created their run ready to
  start. See the Alpha 17 entry for what is true and what changed.)
- Made user-defined agents able to do real work. Every custom agent run on a
  test appliance failed, including one that read only local telemetry. The
  grounding guard rejected an entire answer whenever it named anything absent
  from the prompt, so an agent reporting a score was not allowed to name the
  opposing team. Unverifiable names are now refused only when the run has no
  retrieved material to cite; when sources exist the answer is kept and the
  unmatched names are surfaced as a warning.
- Stopped discarding the agent's actual answer. Result validation rebuilt a
  fixed review skeleton and dropped `answer`, so the "Answer" section of a run
  could never render and users got bullet lists instead of a reply.
- Replaced one failure message that blamed the user's model choice for every
  cause. An unreachable endpoint, a timeout, an HTTP rejection and an
  unverifiable name now read differently and name the endpoint where relevant,
  instead of all advising "choose a stronger model". Failures are also logged
  with the endpoint and exception type; previously six failed runs produced no
  server-side record at all.
- Made model status report what was observed rather than what was configured.
  "MODEL READY" and "CONNECTED MODEL" were derived from configuration alone and
  stayed green while every request through that endpoint failed. Readiness now
  reflects a cached reachability probe, and an unreachable model is named as
  such with the reason.
- Fixed run cards that never updated. Polling covered only the specialist tab,
  so an approved custom-agent run sat at "ready" while it ran and failed on the
  server, and a manual page reload was the only way to learn the outcome. Runs
  in flight now say so.
- Fixed the primary action at the end of agent creation appearing to do
  nothing. "Test before activation" opened its dialog beneath the modal that
  spawned it. Overlay stacking is now declared through layer tokens, so a
  dialog opened from a dialog is ordered by intent rather than by whichever
  number each stylesheet happened to pick.
- Stopped answering out-of-scope questions with a greeting. Keyword matching
  used bare substrings, so "tell me whether they won" matched "hey" and a
  sports question was answered with "Hi. I'm Vaelor Assistant" carrying an
  evidence badge. Matching is now word-boundary based, and such questions reach
  the existing honest "I can't answer that reliably" reply.
- Said when an appliance check ran without the AI. A run that fell back to
  built-in diagnostics was badged green "completed" with the same boilerplate
  summary as every other card, and disclosed the fallback only inside a
  collapsed section. Cards now state the degradation and lead with the one
  finding that distinguishes the run.
- Named drafted agents after their job. "every morning give me yesterday's MLB
  scores for the Yankees" produced "Every Morning Give Me Yesterday agent";
  names are now built from the distinguishing words, keeping acronyms intact.
- Opened the run history the confirmation tells the user to look in, and
  explained a version mismatch instead of showing a card headed "version 2"
  containing a run headed "version 1".
- Marked the research step Required, not Optional, for agents whose described
  job needs information from the internet - the labelling that led beginners to
  build agents that could not succeed.
- Reworked where an agent's result appears. A run was described in a dialog
  that then closed, announced by a banner pointing at a collapsed disclosure,
  and finally readable only inside a second disclosure nested in the first.
  One surface now carries the run from request through approval and execution
  to the answer, which leads at reading size with its sources beneath it.
- Gave every destination one name. The sidebar, page heading and eyebrow each
  invented their own, so seven of nine places were known by two or three names
  at once; a single registry now feeds the sidebar, headings and browser title,
  and per-route titles replace nine identical ones.
- Fixed an unknown URL silently becoming Home while the address bar kept the
  original path. Hash routes now resolve through an alias table and rewrite the
  URL, so what is on screen and what is in the address bar cannot disagree.
- Stopped the assistant proposing a change and refusing it in the same reply,
  and taught action detection the phrasings people actually use ("can you make
  the lights blue") without treating questions as requests.
- Translated the two most common Docker and systemd failures into plain
  language at the point they are recorded, so no `journalctl` invocation or
  unit path reaches a screen; the technical detail is kept for operators.
- Replaced device paths in answers with the thing the user owns ("the memory
  card"), and made one job read the same on every screen it appears on.
- Made the case-lighting colour usable while a multicolour effect is running,
  readable when disabled, and recoverable after an invalid value - Revert now
  clears the error instead of leaving Save locked forever.
- Repaired three responsive faults that the acceptance harness had reported
  green: a composer that became unreachable below 560 CSS px of height, a
  colour-grid override that addressed an element that does not exist, and a
  colour field painted under the preset row between 830 and 868 px wide.
- Strengthened that harness so it could see them: real zoom rather than a
  device-pixel-ratio change, plus occlusion, clipping and reachability checks.
  It now also reads destination names from the app's own registry instead of a
  private copy that silently went stale.
- Made the navigation rail scroll. At 150% zoom on a 720p screen it was taller
  than the viewport and could not scroll, so "Settings" was unreachable.
- Stopped AI Chat pinning its empty state to the bottom, which scrolled the
  opening suggestions out of sight before the user had done anything.
- Idempotent model downloads: double-clicking "Approve download" queued two
  multi-gigabyte transfers, and the control had no busy or disabled state.
- Cancelling the removal review no longer leaves the screen disabled with no
  way back, and disabled primary actions across Settings now say what is
  missing instead of failing silently.

## 2.1.0 Alpha 12

- Made the Assistant reveal its own conversation. The stream had no
  scroll-to-bottom logic anywhere, and sat in a fixed 640px scroller inside a
  separately scrolling page, so a question, its pending row and its answer all
  landed below the fold and the feature read as dead. A shared follow/pause
  hook now reveals new turns while the reader is at the bottom, releases while
  they read history, and resumes on return; the composer always stays reachable
  and Enter sends.
- Corrected fact selection so ordinary thermal phrasing works. "Is my pi
  running hot" previously matched "running" in the service and workload tables
  and was answered with a list of apps; word-boundary intent matching now
  routes running hot, too warm, overheating, hotter and cooler to the same live
  cooling facts.
- Stopped answering change requests with a status readout. A request to turn
  the case lights purple reported the current colour without performing,
  refusing or acknowledging it; requested changes are now either raised for
  approval or plainly declined, with the requested value preserved.
- Added real colour entry to Case lighting: hex and red/green/blue fields kept
  in step with the presets and a visible picker, replacing a 6x4 pixel
  invisible input. Lighting is an explicit-save form, so it now tracks unsaved
  changes, offers Revert, and refuses to save an invalid colour.
- Stopped showing users raw executor failures. Errno codes, absolute server
  paths, YAML parser output, Go runtime traces and Docker daemon errors are
  rewritten as what happened, why, and what to do, with the original text kept
  behind a technical-details disclosure.
- Gave Fleet one truthful state. The screen could show "Head controller",
  "Engine: Docker Swarm" and "NOT INITIALIZED" at once while never surfacing
  the runtime availability that explained it.
- Restored activity timestamps and human operation names, gave progress bars
  real progressbar semantics, distinguished a policy refusal from a breakage,
  and made disabled controls state their precondition.
- Fixed accessible naming so every navigation item is announced with the label
  the user can see, and added character counters to bounded text fields so a
  paste can no longer be truncated silently.
- Revalidated the production build, the backend and frontend suites, and the
  six-viewport responsive matrix.

## 2.1.0 Alpha 11

- Fixed Assistant transcript overflow at active-request time by separating
  pulse-dot styling from shared button wrappers, containing message widths,
  and keeping the full Stop response control visible at every viewport.
- Made non-temporary AI Chat failures durable: the conversation and prompt are
  saved before inference, the actionable failure is stored as the response,
  and the failed chat immediately appears in history for retry or review.
- Corrected Assistant answer routing so general definitions use the connected
  model instead of unrelated workload inventory, while storage answers report
  one capacity per physical device and managed-workload checks give a direct
  healthy/unhealthy verdict with an explicit host-process boundary.
- Replaced the Activity evidence raw-JSON navigation with a contained,
  accessible audit modal that presents actions, outcomes, operators, times,
  and useful details in plain language while keeping technical data optional.
- Revalidated the production build, all backend and frontend suites, and the
  six-viewport responsive matrix including AI Chat document-height and
  workspace-containment gates.

## 2.1.0 Alpha 10

- Restored the compact desktop information density users previously obtained
  only by zooming out, using native type, spacing, shell, and content-width
  tokens while preserving mobile sizing and accessible control targets.
- Recontained AI Chat so selecting history or receiving a response cannot push
  the conversation or composer outside the workspace, enlarged the writing
  area, and reorganized the toolbar for desktop, tablet, and phone layouts.
- Simplified System storage into one row per physical device, with mounted
  volumes, connection data, and available temperature evidence revealed on
  demand instead of repeating the same device for every mount.
- Expanded responsive acceptance with an explicit AI Chat containment gate and
  verified all nine primary routes at six viewport classes plus 2x rendering.

## 2.1.0 Alpha 9

- Removed the global 80% page zoom and 125% viewport compensation in favor of
  native desktop density tokens, contained workspaces, and explicit overflow,
  spill, shell-clipping, and document-height acceptance checks.
- Standardized operation states and phases for operators, moved opaque results
  behind technical details, preserved failed AI Chat turns, aligned local-model
  timeout budgets, and made catalogue and Remote Console labels match their
  actual actions and readiness.
- Made Assistant answers cover storage, managed services, and managed workloads
  in multi-clause questions, use decimal GB consistently, and avoid claiming no
  model is connected when a configured model fails.
- Verified that legacy on-device archives are non-empty but intentionally hidden
  because they lack current lineage metadata; Alpha 9 live acceptance creates
  and byte-verifies a new supported checkpoint without altering legacy files.

## 2.1.0 Alpha 8

- Reworked Assistant navigation into distinct Ask Vaelor, appliance
  troubleshooting, custom-agent, memory, skill, and schedule lanes; built-in
  troubleshooters now stay limited to appliance evidence and repair guidance.
- Fixed stale saved-research discard, the custom-application request field,
  crushed agent-card and troubleshooting-dialog controls, and a duplicate main
  landmark exposed by full responsive browser testing.
- Replaced generic AI Chat timeouts with provider-specific connection,
  rejection, malformed-response, and 45-second model-timeout errors that tell
  operators how to recover without losing their prompt.
- Expanded production-browser acceptance to exercise these exact workflows at
  phone, tablet, desktop, wide desktop, and 200% rendering sizes.

## 2.1.0 Alpha 7

- Removed six exact-fingerprinted internal engineering memories from upgraded
  appliances while preserving administrator-authored product memory and keeping
  development scratch data in its separate store.
- Repaired the flagship Assistant timeout path, same-tick install locking,
  System deep links, durable post-approval progress, installed-app catalog
  status, and unified operator-facing operation language.
- Made browser-desktop connection status depend on one-use gateway consumption,
  added explicit failure/retry behavior, and restored modal Escape, focus trap,
  and focus return semantics.
- Added short-lived GET caching and in-flight deduplication, corrected package
  version reporting, and completed the reviewed accessibility and visual polish
  fixes across KVM, lighting, workloads, forms, agent cards, and the sidebar.

## 2.1.0 Alpha 6

- Consolidated the production application into one content-hashed JavaScript
  entry bundle as well as one stylesheet, eliminating all runtime route-chunk
  fetches that browser filters could reject.
- Added real-build and release-time gates requiring exactly one deployable
  script and stylesheet while retaining complete reference validation.
- Supersedes Alpha 5 after live Chrome proved that multiple dynamically imported
  JavaScript routes could be blocked independently of their generic filenames.

## 2.1.0 Alpha 5

- Consolidated production CSS into one content-hashed stylesheet so lazy route
  rendering no longer depends on browser-sensitive dynamic CSS preload events.
- Added real-build and release-time gates that require exactly one deployable
  stylesheet and a complete generic asset/module graph.
- Supersedes Alpha 4 after live Chrome loaded its generic entry bundle but still
  rejected the Assistant route's dynamically preloaded stylesheet.

## 2.1.0 Alpha 4

- Replaced semantic production frontend chunk names with generic content-hashed
  asset names so browser privacy filters cannot block product routes such as
  Assistant based on implementation terminology.
- Added a release-build regression gate that rejects semantic or otherwise
  nonstandard deployable asset names before packaging.
- Supersedes Alpha 3 for live testing after the exact Alpha 3 package exposed
  the client-side asset-blocking failure on the commissioned Pi.

## 2.1.0 Alpha 3

- Promoted Vaelor into a fresh, product-named canonical repository with clean
  history and a deterministic allowlist-based source-integrity gate.
- Completed PRE-01 through PRE-07 UI and workflow remediation across Assistant,
  AI Chat, specialist execution, agent creation, workload management, saved
  research, memory cards, model switching, and responsive layouts.
- Added full-route responsive acceptance coverage, including compact through
  full-desktop viewports and 2x rendering, with overflow, control, request,
  server-response, and console-error guards.
- Added rollback regression coverage, corrected rollback guidance, hardened
  source-encoding checks, and fixed concurrent security-store sidecar handling.
- Restored the documented one-window `pm_dashboard` compatibility aliases in
  distributable packages and made the concurrent administrator-removal test
  synchronize authentication before exercising the mutation transaction.
- Corrected the Assistant intelligence summary's desktop action geometry and
  added a responsive gate for vertically shredded control labels.
- Retained historical PRE-00 and Alpha 2 evidence as explicitly historical
  inputs; Alpha 3 remains pending exact-package physical evidence and outside
  auditor clearance.

## 2.1.0 Alpha 2

- Consolidated product delivery into one canonical roadmap and removed the
  competing implementation plans.
- Bound the technical UX review, real-user test, supplied screenshots, live
  state, source revision, and reproducible package evidence into the PRE-00
  audit baseline without representing unresolved findings as complete.
- Added deterministic disposable fixtures for identity, workload, system,
  model-switch, checkpoint, and KVM acceptance states.
- Split the Agent Center and Workloads surfaces into smaller owned panels and
  shared durable job-state helpers, including responsive dialog corrections.
- Added immutable app-capability snapshots to agent tasks so reviewed grants
  remain attributable across execution and retry.
- This is a test candidate for PRE-01 through PRE-08 remediation. The accepted
  report findings remain open until their required automated, live, and where
  applicable physical evidence is attached and independently reviewed.

## 2.1.0 Alpha 1

- Added a fail-closed installed-app capability registry with explicit manifests,
  brokered connections, exact-version custom-agent grants, health revalidation,
  audit provenance, and approval-gated write previews.
- Added professional custom-agent app-access controls for discovery, connection
  testing, grant preview/save/revoke, compatibility, and recovery states.
- Added explicit AI Chat delegation to named custom agents as a reviewable task
  proposal; chat never infers an agent or starts work automatically.
- Added durable custom-agent app execution with immutable grant snapshots,
  read-only invocation, write previews, bounded results, and retry continuity.
- Sanitized portable imports so credentials and app connectivity must be
  re-established on the destination appliance.

## 2.0.4

- Made `vaelor` the implementation namespace and retained `pm_dashboard` only
  as a documented, one-window Python and command compatibility layer.
- Replaced the legacy entry splash with a responsive Vaelor command gateway
  that explains hardware, workload, intelligence, and fleet control domains.
- Removed the compiled upstream dashboard from distributable artifacts while
  preserving its GPL provenance and the complete current frontend source.
- Added architecture-specific dependency/license inventories and structural
  release audits for source, compatibility aliases, installers, and licenses.
- Added a secret-safe cluster service manager with live task/configuration
  inspection, bounded logs, and approval-gated rolling restart, image refresh,
  rollback, bounded replica/memory/rolling-update configuration, app-scoped
  diagnostics, verified named-volume backup/restore, and removal. Every
  cluster mutation requires administrator authorization and its
  action-specific confirmation.
- Added encrypted, portable Vaelor state export and guarded replacement import.
- Added deterministic arm64/amd64 Debian packages and a restricted multi-architecture OCI core.
- Added a local-only release builder that proves reproducible wheels and
  complete normalized source archives before generating native packages.
- Removed machine-specific certificates and obsolete legacy installer/unit
  definitions from the public source boundary.
- Kept the QEMU cluster harness developer-only and excluded it from every release artifact.
- Fixed clean Debian installation by including required directory entries.

## 1.4.x
- Change config to outside handling
- Remove deprecated code
- Remove influxdb log
