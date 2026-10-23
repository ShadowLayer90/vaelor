# Change log

## 1.5.5

A fix release. It fixes everything the 1.5.4 cold install found, stops AI Chat
inventing live facts, gives AI Chat an optional Web search, makes web research
less dependent on one free search engine, and fixes what seven rounds of fresh
installs on the hardware turned up.

### Upgrading from 1.5.4

- **Update Vaelor from the console** (**System › Hardware and services ›
  Update Vaelor**), or re-run the installer; both end in the same state.
- The web search service's settings are rewritten on every update and install
  (its secret and any search key you added are kept), so hand edits to them are
  not kept.
- InfluxDB stops writing a log line for every request and query, on install and
  on update.

### AI Chat never invents live facts

- **A tightly worded live question** (the weather in a named place, the price of
  something today) that AI Chat has no live source for gets a plain reply that
  it cannot see live data, and how to get it, instead of a confident figure
  from the model's memory. Looser questions reach the model with an instruction
  to say plainly when it cannot see live data.
- **Weather in a named place is answered from Open-Meteo**, with the source, the
  time it was read and the local time of the conditions; metric, with °F beside
  it. Only the place name is sent, never your question. When the place is
  unclear, AI Chat asks which one. Questions about the machine's own
  temperatures still go to the Assistant.

### Web search in AI Chat

- **A Web search switch beside Knowledge, off by default.** When it is on and
  web research is installed, AI Chat searches through the machine's own guarded
  search service, cites the pages it used, says plainly when they do not answer
  the question, and asks rather than guesses. Nothing leaves the machine for a
  search while it is off.
- **Only an administrator can turn it on.** Operators see the switch disabled,
  with the reason; the server refuses it for anyone else.
- **News questions search news sources from the past week only.** A question
  about older news gets only the past week.

### Web research no longer depends on one free search engine

- **The search engines were measured from the machine and the mix changed:**
  Yep, Zapmeta, Seznam, DuckDuckGo and Bing for general questions; Bing News and
  DuckDuckGo News for news. Engines that refused or blocked the machine (Mojeek
  among them) are off. Yandex is not used.
- **An optional search key you add** (Brave Search API) under
  **Settings › Connections**, administrator only. It is off until you add a key,
  and removing it always works.
- **The query planner no longer adds words like "current" or "latest"** that
  your question did not contain; they sent the engines off topic.
- **Custom-app research looks for the app on GitHub and Docker Hub first,** by
  its name, and uses the search service only as a fallback. It takes only an
  image the project itself proves: one its README names, or one under the
  project owner's own name, from a repository with real standing. Anything less
  certain is put to you; it never picks a stranger's image by download count.
  The review screen shows where the image came from and any warning. Research
  for a proven image runs even with no AI model connected.
- **Apps on Vaelor's reviewed list (Uptime Kuma, Plex, Palworld) research only
  from their pinned sources** and need no AI model. A request is matched to one
  only by the app's name as its first whole words, so "Perplexica" or "Tautulli
  for Plex" is not taken for Plex.
- New outside destinations: `api.github.com` (finding apps),
  `geocoding-api.open-meteo.com` and `api.open-meteo.com` (weather, the place
  name only; AI Chat's Web search note names Open-Meteo), and
  `api.search.brave.com` (only when you add a key).

### Fixes from the 1.5.4 cold install

- **The console certificate is no longer re-issued by controller setup.**
  Container and virtual-machine bridge addresses are no longer part of its
  identity; it is re-issued only when it lacks a name or address the machine
  has, or is near expiry, and open console pages are told 15 minutes ahead.
  After a renewal, connections opened under the old certificate close once
  idle (about 2 seconds, at most about 15 minutes); uploads, restores and AI
  Chat answers in progress finish first.
- **The trust steps can be read without the console:** the installer saves them
  in a text file on the machine and says where, and Vaelor keeps it current.
- **The GPU memory pool card tells the truth** when a restart would shrink the
  pool (for example after an uninstall removed the saved size), and offers to
  keep the current size.
- **The restart dialog lists the work that is still running,** and the restart
  must be confirmed against that list.
- **A later success of the same operation on the same target resolves an
  earlier failure**, in Activity and Apps alike; the failure stays in the
  history, marked resolved. A dismissed notice stays dismissed.
- **Vaelor's own containers are listed in their own section,** not among the
  apps it discovered.
- **The model library lists the controller and the cluster weights it holds.**
- **The first cluster deploy waits** (up to about 4 minutes) for the 4-bit
  guesser to finish building before starting the model, and the note about it
  updates as it goes.
- **AI Chat's cluster connection says it stays on the cluster's machines**
  instead of "locality unknown".
- **Uninstall:** Remote Desktop that Vaelor turned on is switched off by every
  uninstall, its login and certificate removed (Remote Desktop you set up
  yourself is left alone); a purge also removes the telemetry database, the
  Phoenix container and serving containers; a worker's purge returns it to the
  bare operating system, Docker included.
- **The installer uses a local wheel only when `--wheel` names it,** so a stray
  wheel in your home folder is never installed in place of the release.
- **The console update also refreshes the release scripts** the machine runs.
- **Home › Machines shows the controller's update status** ("Up to date",
  "Update available") in the same words as Update Vaelor.
- **Model file sizes read as GiB with GB beside them,** for example "13.6 GiB
  (14.6 GB)", and about a dozen smaller fixes to mixed units, contradictory
  states, a repeated dialog line and wording.

### First-run setup and trust

- **Trusting this Vaelor is part of first-run setup.** Right after the first
  administrator account is created, Vaelor shows how to trust its household
  authority on the computer in use (with a QR code for phones) and the
  fingerprint to compare. It can be skipped, and stays under Settings ›
  Connections › Trust this Vaelor.
- **The box's own text login screen shows the console address and the
  authority's fingerprint**, so it can be checked away from the network.
  Uninstalling removes it.
- **A fresh machine no longer falls back to the kernel's smaller GPU memory
  pool.** The installer and a worker's join save the largest safe size Vaelor
  calculates when none is saved; a size you saved is never changed, and an
  update never undoes your own Remove setting. The card says the saved size
  counts from the next restart; Vaelor still never restarts a machine itself.

### AI Chat answers

- **AI Chat shows when an answer was cut off.** An answer that stopped at the
  model's answer-length limit, or ran past the 12,000 characters AI Chat keeps,
  shows a note under it saying the end is missing, also after a reload, in a
  branch and in an export.
- **Longer answers from large local models.** A model over 8B that Vaelor runs
  itself (the cluster's 30B, the GPU's 27B) may answer with up to 2,048 tokens
  instead of 1,024, is given 240 seconds instead of 120, and is no longer told
  to keep plain answers under 120 words. Smaller models keep their own limits.
- **The Assistant and agents read a mixture-of-experts model by its full
  size.** A name like Qwen3-30B-A3B is a 30B model; it was read as 3B. AI Chat,
  the Assistant, agents and the model tiers now read every model's size the
  same way.
- **Custom agents get context in line with their model.** On a model over 8B
  that Vaelor runs, a custom agent receives far more of its sources and memory;
  small models keep what they had.
- **Vaelor records the context window it serves each model with,** so a large
  request is trimmed to fit instead of being refused for length. A machine that
  updates fills these in for Vaelor's own models within minutes of starting;
  connections you added are never changed.

### Workers and the cluster

- **A worker uninstalled on the machine stays uninstalled.** When Vaelor is
  uninstalled on a worker that is still in a cluster, the controller no longer
  puts its software back. The machine's card says so, with Re-add and Remove,
  and the controller stops the cluster placing new work there. The uninstall
  names the model serving it stopped.
- **Adding a worker clears model serving an earlier controller left on it.** A
  worker re-added to a rebuilt controller kept running the old controller's
  model and holding its GPU memory, so the new controller said a model "won't
  fit". The join plan now says it stops and removes model serving the
  controller has no deployment for on that machine, services and containers
  alike, and the join does it. Your apps and the downloaded model files stay.
- **Re-add takes effect with one press,** and reads Docker on the machine
  without needing Python (a Docker installed as a snap is recognised).
- **Automatic changes to workers are in the security audit log,** under
  "Vaelor (automatic repair)", or under you when your Recheck caused them.
- **A model service whose file is gone but which is still running is stopped,**
  not reported as already removed.

### Console

- **The restart and shut-down confirmations tell the truth.** They said nothing
  was running whenever no Vaelor job was, even while models served AI Chat; they
  now say no job is running, and what a reboot or shut down stops.
- **Activity says one thing about how an operation ended.** An operation Vaelor
  declined on purpose reads as finished, with its reason, not as both finished
  and needing attention.
- **The page behind a confirmation is usable the moment it closes;** for a
  moment afterwards, clicks were lost.
- **A full removal deletes Vaelor's release scripts directly,** and says so if
  it cannot, however the removal was started.

### Known limits

- The free search engines are not dependable on their own: two of the best
  rate-limited the machine during measurement. A search key helps most.
- GitHub allows about 60 anonymous reads an hour per machine; app research says
  so plainly when the limit is reached. For the same reason Home reads the
  update status once, when the page opens.
- A cluster started from a worker reads as "unknown" in AI Chat's model picker.
- A browser that only clicked past the certificate warning and keeps a stream
  open for more than 15 minutes after a renewal is cut off at 15 minutes.
- An imported compose file can still carry the catalog label that decides which
  apps get the file manager.

### Not verified on the hardware in this release

- Web search refused for an operator account.
- Adding and removing a real Brave Search key end to end.
- The text login screen redrawing an open prompt, and the first-run trust step
  on an iPad.
- An upload in progress while the console certificate is renewed.

## 1.5.4

An Assistant release. The on-device Assistant on the neural processor (NPU) gets
a new model, trained for the job it does, on a newer FastFlowLM. Its answers are
read whole, shown as they are written, and built from the readings each question
needs. Questions that are not about the appliance now go to AI Chat, and custom
agents run on AI Chat's model instead of the NPU.

### Upgrading from 1.5.3

- **Update Vaelor from the console** (**System › Hardware and services ›
  Update Vaelor**), or re-run the installer; both end in the same state.
- **The Assistant's new runtime and model download in the background after the
  update** (about 3.4 GB) while the current Assistant keeps answering. Vaelor
  then switches to them, which takes the Assistant offline for about ten
  seconds, and asks the new model a question whose answer was recorded when the
  model was released. If the answer does not match, or the new pair does not
  start, the machine is put back exactly as it was and the Assistant keeps
  working as in 1.5.3. The previous runtime and model are kept beside the new
  ones until the next switch.

### Upgrading from 1.5.1 or 1.5.2

- Update from the console, as from 1.5.3.
- **One caution, for this one update:** The first update from the console into this release is applied by the updater of the release it replaces, which does not take the update lock. While that one update runs, do not start another update or run the installer. Its model refresh and build steps already take the lock, and every update after it is fully protected.
- **Updating from 1.5.1:** the update page you start it from is 1.5.1's own and does not refresh by itself, so reload the page about a minute after you press Confirm update to follow the rest of the update.

### A new Assistant model, on FastFlowLM 1.0.9

- **The model was retrained for the Assistant's job.** The model 1.5.3 shipped
  was trained only on agent tasks, never on answering an owner's questions about
  the machine. The new one is a fresh fine-tune of Qwen3.5-4B on examples of the
  Assistant's own answers, each written from real machine readings and checked
  against them, together with the agent tasks and examples of handing general
  questions to AI Chat.
- **It runs on FastFlowLM 1.0.9** (1.5.3 used 1.0.2). The runtime and the model
  are pinned as one pair: the installer and the console's update install both
  or neither, check every file against the checksums pinned in Vaelor before
  switching, and refuse a model package the runtime would quietly replace with
  its own stock model.
- **Measured on the Z2 Mini's NPU**, with the questions owners actually asked
  and others the model never saw in training:
  - every one of 44 real questions, and of 20 new held-out questions, got a
    whole answer;
  - Vaelor's own NPU test (appliance, diagnostics, honesty and hand-off cases)
    passed 49 of 49, and 5 of 5 fresh honesty questions;
  - a whole answer took 12.7 seconds (median), with the first words after
    5.3 seconds. 1.5.3's model, given the same new readings, took about 31
    seconds and 6.7 seconds;
  - graded blind against the machine's readings, its answers carried about 0.2
    errors each. 1.5.3's model carried about 1.3; that grade was made earlier,
    by a different grader, so the comparison is approximate.
- **Known weak spots:** working out the processor's load from its temperature
  when the question is worded in a new way; questions about how close a reading
  is to its limit; and two new habits, suggesting a reboot "when the GPU work
  allows" while the GPU is idle, and suggesting freeing root disk space at 70%.

### Better answers from the Assistant

- **Answers appear as they are written.** If you leave or reload the page
  mid-answer, the answer finishes in the background and is saved, so it is
  there when you come back. **Stop** ends it.
- **Stop no longer wears out the NPU's server.** With FastFlowLM 1.0.2, ten
  Stops could take the Assistant down until Vaelor restarted it; Vaelor now
  lets a stopped answer run to its end out of sight instead of hanging up on
  it. Stop still clears the answer at once, but the NPU stays busy until the
  stopped answer ends (about 20 seconds, measured on 1.0.2) and the next
  question waits for it.
- **No hidden second ask.** A reply that was only an empty tool call used to be
  asked again silently, which doubled the wait. The honest answer from the
  readings is shown instead, and the timing line counts every model call.
- **Answers are read whole.** The model is asked in the reply format it was
  trained on, and Vaelor turns that into plain text: a summary, then the
  details, with sources on their own line. A reply that lists only sources, or
  copies a reading back, is asked once more.
- **Each question gets the readings it needs, kept small.** The readings the
  question names come in detail, the rest in one line each; history is thinned
  out. A typical prompt is now about 1,500 tokens, and at most about 1,850.
  Measured on the Z2 Mini with 1.5.3's model, against the readings it had
  before: the first words came after 5.6 seconds instead of 8.7, and the share
  of wrong or invented claims fell from 33% to 8.4%.
- **Clearer readings:** temperatures carry their own limits, the services line
  names every service and says which are not running, the updates line says
  plainly whether updates are waiting or a reboot is pending, and memory and
  disk figures say "X of Y GB used (P%)".
- **The Assistant answers "what time is it?", "what's the date?" and the time
  zone** from the machine's own clock.
- **Steady answers.** The Assistant asks its model at temperature 0, the
  setting every test measured (it sent 0.1 before). Every request Vaelor sends
  to FastFlowLM now sets its presence and frequency penalties to 0. FastFlowLM
  keeps those settings from one request to the next, so an answer could depend
  on the request before it; in testing, the penalties also made the model drop
  digits from numbers.

### The Assistant is for the appliance; other questions go to AI Chat

- **General questions are handed to AI Chat**, with an **Open AI Chat** button:
  for example "summarise the planet Saturn" or "what is the capital of France?". The
  model also learnt to hand such a question over itself when the fixed rules
  miss its wording.
- **Appliance questions stay with the Assistant**, including ones that were
  sent away before: the processor's speed, state or load, the graphics
  processor, "how is the computer doing", "how do I restart Jellyfin", the load
  average, and questions naming a cluster machine or an installed app. "US" in
  "US dollars" is no longer read as "us", meaning this machine.
- **When in doubt, the Assistant keeps the question.** A hand-off goes through
  only for a recognisable general-knowledge question with no appliance word,
  name or pointer in it ("this computer", "here", "we" and "our" count as
  pointers). Otherwise the Assistant shows a short notice instead: for a
  clearly appliance question it asks you to name the reading; for anything else
  it offers AI Chat. So a few general questions get the notice rather than the
  hand-off, for example "what's the temperature in Osaka?", which contains a
  reading word, or "tell me a joke".

### Custom agents run on AI Chat's model

- **Custom agents now use the model AI Chat uses,** wherever it points: this
  machine's GPU, the cluster, or a hosted service. They no longer run on the
  NPU. **If AI Chat uses a hosted service, a custom agent's prompts and the
  pages it fetches are sent to that service.**
- With no AI Chat model set up, a custom agent says so at once and does not
  run. If the cluster's model has been unloaded while idle, a run started by
  hand starts waking it and reports it unavailable for that run; a scheduled or
  alert-triggered run waits up to five minutes for the wake, then runs.
- If AI Chat's model is busy, an agent waits its turn for up to two minutes,
  then fails with a plain reason. AI Chat itself may briefly say the model is
  busy while an agent step runs.
- The agent screens say which model the agents run on and, when it is not
  ready, what to do (for example "load it in Cluster › Deployments").
- Built-in specialists, the deploy planner and custom-app research stay on the
  NPU.

The test suite, the training data and the developer tools also changed; none
of that ships.

### Tested live

- **On the Z2 Mini, through the installer** (from a development build, not
  from 1.5.3): the Assistant switched to FastFlowLM 1.0.9 and the new model by
  itself in 11 seconds, every file of the model matched its checksum, the new
  model answered the recorded question exactly, and the cluster's model kept
  serving throughout.
- The switch and the automatic return to the previous pair were also run on
  the Z2 Mini's NPU with test copies: a package FastFlowLM would replace was
  refused before anything stopped, a wrong answer put the old pair back, and
  the old pair answered again.
- The model measurements above were made on the Z2 Mini's NPU. The Stop
  change was run live with 36 Stops on one FastFlowLM 1.0.2 server, with no
  connection lost.

### Not verified in this release

- **The console update from 1.5.3 (or earlier) to 1.5.4 has not been run
  live.** In particular, the Assistant's background download and switch after
  a console update have run only on test copies, not after a real update.
- **1.5.4 has not had a cold install,** and the model's download from this
  release's files has not been run live: the Z2 Mini was given the same
  package from a local copy.
- The custom agents' move to AI Chat's model and the Assistant's new hand-off
  notices are covered by tests and installed on the Z2 Mini, but have not yet
  been checked in the console.
- How long Stop keeps the NPU busy was measured on FastFlowLM 1.0.2, not on
  1.0.9.
- Still open from 1.5.2: the ZBook worker's build of the 4-bit guesser has run
  live only once, and AI Chat does not yet trim a conversation to fit the
  context window a connection declares.

## 1.5.3

An update release: updating from the console now leaves a machine in the same
state as the installer does. In 1.5.2, a machine updated from the console
served the cluster model without its speed-ups until its owner re-ran the
installer, and a 30B deployed under 1.5.1 had to be removed and deployed again
to get them. Neither step is needed from this release on.

### Upgrading from 1.5.1 or 1.5.2

- **Update Vaelor from the console** (**System › Hardware and services ›
  Update Vaelor**). There is no installer re-run and no remove-and-redeploy
  any more.
- **One caution, for this one update:** The first update from the console into this release is applied by the updater of the release it replaces, which does not take the update lock. While that one update runs, do not start another update or run the installer. Its model refresh and build steps already take the lock, and every update after it is fully protected.
- **Updating from 1.5.1:** the update page you start it from is 1.5.1's own and does not refresh by itself, so reload the page about a minute after you press Confirm update to follow the rest of the update.
- A cluster model that is serving is reloaded once near the end of the update,
  which takes it offline for about a minute and a half. It comes back with
  every speed-up, including the 4-bit guesser; no second Unload and Load is
  needed.
- Re-running the installer still works, and ends in the same state.

### The console update does everything the installer does

- **The same build steps.** Once the new version has passed its health check,
  the update fetches vLLM's attention fix and builds it, builds Vaelor's own
  GPU routines and the 4-bit guesser, and removes pieces the release retired.
  It is the same list of steps the installer runs, run from the same place, and
  it runs before any model is reloaded, so the reload serves what was built.
  The steps share a time limit that leaves room for the reload. A step that
  does not finish leaves the model serving without that speed-up, and the
  console says so; it does not leave the update half done. On the Z2 Mini,
  run on their own rather than inside a real update, the steps took 27 seconds
  when nothing had been built yet and about a second when everything had.
- **A 30B deployed under 1.5.1 is upgraded on its next reload.** A Load,
  including the one the update runs, now gives an older deployment what a
  fresh deployment of the same model gets on the same machines: the guesser,
  Vaelor's GPU routines and the new defaults. It does so only where these were
  measured: the default vLLM 0.27 image, the 4-bit (AWQ) Qwen3-30B-A3B, one
  copy per machine, and every machine reading a gfx1151 GPU. A setting you
  chose yourself is kept. The guesser's model is downloaded during the update
  while the current model keeps serving.
- **A decline is kept honest and retried.** If the guesser will not fit in
  memory, or cannot be downloaded, the model is reloaded with everything else
  and without the guesser and what needs it, and its note says what was held
  back and why. Later updates do not keep retrying it; the next Load you start
  does.
- **A reload is no longer skipped when only the speed-ups changed.** The check
  that decides whether an update reloads a model now counts every setting a
  Load would change. This also fixes the installer, which had the same gap.
- **One update at a time.** While an update runs, a second update is refused
  at once, and the installer's model refresh waits a limited time for it and
  then stops with a message rather than running alongside it.
- **The updater restarts itself after each update,** once it is idle, so the
  next update is applied by the new version's own updater.
- **The update page follows the update to the end.** After you confirm, it
  shows each step as it happens (download and check, install, restart, build
  steps, guesser download, model reload), each ending done, skipped or held
  back with the reason, and switches to "Updated to" by itself. It also shows
  what the cluster model now runs with, or what was held back and why. If the
  update takes more than 20 minutes it says so.
- **One reload, not two.** The 4-bit guesser is now built before the model is
  reloaded, within a time limit, so the update's reload already uses it. If the
  build does not finish in time, the model loads with the full-size guesser and
  its note says so.

The test suite and the developer tools also changed; none of that ships.

### Tested live

- **The console update from the published 1.5.1 was run on the Z2 Mini and
  the ZBook Ultra,** with every speed-up piece and the guesser's files removed
  from both machines first, and a 30B deployed by 1.5.1 itself. From pressing
  Confirm update: 1.5.3 was running after 40 seconds, the build steps and the
  guesser download were done after about two minutes, and the 30B was offline
  for a minute and a half while it reloaded, once. Both machines came back
  with every speed-up, the 4-bit guesser included, built from scratch on each.
  The update page followed it to the end without a reload, and the updater
  restarted onto 1.5.3's code. The 30B then wrote about 123, 121 and 95 word
  pieces a second for one person at 65, 708 and 6,443 prompt tokens, and 167 on
  the coding test, the same as 1.5.2 freshly deployed. No GPU faults.

### Not verified in this release

- **The console update from 1.5.2 to 1.5.3 has not been run live;** only the
  update from 1.5.1 has.
- **1.5.3 has not had a cold install.**
- Still open from 1.5.2: the ZBook worker's build of the 4-bit guesser has run
  live only once, and AI Chat does not yet trim a conversation to fit the
  context window a connection declares.

## 1.5.2

A speed release for the cluster's recommended model, Qwen3-30B-A3B, on the AMD
Strix Halo GPU. For one person it now writes about two and a half to three
times as fast as in 1.5.1, long prompts are read more than twice as fast, and
accuracy on a standard coding test is unchanged. The model speed figures below
were measured on the HP Z2 Mini G1a.

### Upgrading from 1.5.1

- **Re-run the installer** (`git pull` in the clone, then
  `sudo ~/vaelor/deploy/install-vaelor.sh --unattended`). The installer
  downloads vLLM's attention fix (below) and builds the speed-up pieces on each
  machine; the console's update installs 1.5.2 but does not download that fix,
  and without it the model serves without its guesser. A cluster model that is
  serving is reloaded by the installer, which takes it offline for a minute or
  two. Each worker gets its own copies when the controller next installs its
  serving software.
- **A 30B deployed under 1.5.1 keeps most of the settings it was deployed
  with**, since a deployment serves again the way it was made (a Load only adds
  the step merging on a Strix Halo GPU). To get the guesser, Vaelor's GPU
  routines and the new defaults, remove that deployment and deploy the model
  again from the console after re-running the installer. Its first start uses
  the full-size guesser while the 4-bit copy is built in the background; Unload
  and Load it once afterwards to switch to the 4-bit copy.

### The cluster model is faster

One person, measured on the Z2 Mini, 1.5.1 against 1.5.2:

- a short coding test (the first ten HumanEval problems) as run by the Ornith
  project: about 52 to 164 tokens (word pieces) a second;
- short coding answers: about 52 to 172 tokens a second;
- long answers: about 52 to 122 tokens a second;
- the first word for a 6,400-token prompt: about 5.3 to 2.0 seconds.

Two to eight people at once also get faster. Accuracy is unchanged:
on HumanEval (164 coding problems, graded by their official tests) the model
passes 153 or 154, against 154 without any of this; a few borderline problems
flip between any two runs.

What does it:

- **A small "guesser" model.** The 30B now serves with a small draft model
  (RedHat's DFlash guesser for this model, about 1.4 GB, downloaded when you
  deploy it) that proposes the next few tokens; the big model checks them in a
  single step and keeps only the tokens it agrees with. At most eight
  requests run at once per machine, and the guesser looks further ahead when
  fewer people are using it. The model library downloads the guesser with the
  model, and Remove deletes it too unless a deployment still uses it.
- **An Advanced section in the deploy form** for the request limit, how much
  is read per step, the guesser, how far it looks ahead, and vLLM's newer
  runner (vLLM's newer way of running each step, which keeps the GPU's
  recorded replay of a whole step working with the guesser). It starts from
  the model's recommended values.
- **Long prompts stay fast with the guesser.** vLLM's attention code slowed
  down sharply on long prompts when checking guesses. The installer downloads
  an upstream vLLM fix (pull request #59448) and applies it on each machine to
  that machine's own copy of vLLM, with Vaelor's own changes on top, and uses
  the result only if its fingerprint (SHA-256) matches. Vaelor ships no vLLM
  code. A machine that cannot get the fix serves without the guesser, and the
  console says why.
- **Vaelor's own GPU routines.** The model's heaviest parts (its expert layers,
  its 4-bit dense layers and its attention) run on routines Vaelor wrote for
  Strix Halo. One of them is built on each machine inside a locked-down
  container (no network, no GPU, read-only), checked by fingerprint before
  every use; if anything is refused, the next-best routine serves and the log
  says why. They switch on only on gfx1151 (the Radeon 8060S graphics in AMD
  Strix Halo); any other GPU uses vLLM's own.
- **Faster prompt reading.** New routines for reading the prompt make long
  prompts more than twice as fast to read as in 1.5.1.
- **Step merging.** Two small GPU steps in each attention layer are now done
  as one, so each word needs fewer separate GPU jobs.
- **A 4-bit guesser, built on your machine.** The installer makes a 4-bit copy
  of the downloaded guesser on each machine, inside the same kind of
  locked-down container, so the guesser reads less memory per step. Every file
  must match its pinned fingerprint, or the full-size guesser serves instead
  with a plain reason; it never stops the model from serving. Nothing extra is
  downloaded, and Vaelor ships no model file.
- **Old copies are cleaned up.** The installer removes retired versions of
  these routines, never one that a running model or a deployment still uses.

The last batch of this work, compared with the build before it: one person's
writing steps are 6 to 8% faster, the Ornith coding test rose from 152 to 164
tokens a second, short prompts are read about 20 milliseconds sooner, and
accuracy is unchanged.

### Conversations reuse what the model already read

- **Agents, research and planning keep to one machine.** With a model served on
  two machines, Vaelor's multi-step work (the agent tool loop, custom agents,
  application research, the search planner and the Assistant) now tags its
  requests so each one returns to the machine that already read its earlier
  steps, instead of starting over on the other. A six-step tool loop went from
  12.3 to 9.4 seconds (measured on the ZBook with two copies of the model).
  The tag is a code derived from the task, never its text, and is sent only to
  models this appliance serves.
- **AI Chat's history is kept in steady blocks.** The instructions at the top
  of a chat are now the same on every turn, and older messages drop off four at
  a time, so the model can reuse what it read last turn. The next turn's first
  word came about twice as fast in testing on the ZBook (424 to 197
  milliseconds without a document collection; with one, about 1.5 to 1.4
  seconds on every other turn). AI Chat now keeps 8 to 11 recent messages
  instead of 8, and temporary chats use the same window.

### Performance tab and fixes

- **GPU timing per routine on demand.** A profile can now restart the cluster
  model with GPU tracing switched on, for a limited time only; it returns to
  normal serving by itself when the time is up or the capture ends, whichever
  comes first.
- A profile in the Performance tab shows how many of the guesser's proposals
  are kept.
- While a guesser serves, writing speed is counted in tokens, not steps (each
  step can write several).
- A profile reads only its own trace files, old ones can be cleared, and the
  trace reader refuses links and limits how much it reads and for how long.
- Found while installing this release on the test machines: the speed-ups
  stayed off after a model was loaded, because the load runs in a sandbox that
  is not allowed to read the GPU's type. Vaelor now reads it from the graphics
  core's version instead, so a load switches them on.

### Not verified in this release

- **1.5.2 has not had a cold install.** It was installed in place on the existing
  test machines, not on wiped ones, so the first-start builds that a fresh
  machine runs have not been run live.
- **The ZBook worker's build of the 4-bit guesser has run live only once.**
- The background build of the 4-bit guesser that a first start runs has not
  been run live.
- The upgrade from the published 1.5.1 has not been run: the test machines
  already held the vLLM fix and the guesser from development builds, so a first
  download of the fix, a first guesser download and the remove-and-redeploy
  step were not exercised.
- The console's own update to 1.5.2 has not been run live.
- The model speed figures are from the Z2 Mini; the tool-loop and AI Chat
  timings are from the ZBook. The ZBook Ultra runs the model more slowly.
- Known gap: AI Chat does not yet trim a conversation to fit the context
  window a connection declares, and keeping a few more messages widens that
  existing gap slightly.

## 1.5.1

A fix release after 1.5: the console's certificate now comes from a household
certificate authority, so the browser stops warning once you trust it, and the
defects found by installing 1.5 cold from GitHub on fresh machines are fixed.

### Household certificate authority

- **The console's certificate comes from a household certificate authority.**
  Each install creates its own root, which can vouch only for private
  addresses and local names, and signs the console's certificate with it. The
  installer no longer makes a self-signed certificate. At the end of the install
  it prints the root's fingerprints and one command per operating system that
  installs the root on your own devices only if the fingerprint matches.
  Vaelor's own machines trust the root automatically. See `SECURE_ACCESS.md`.
- **Renewed without a restart.** A new root service, `vaelor-tls-authority`,
  renews the console's certificate before it expires and when the machine's
  addresses change. The console and the browser desktop serve the new
  certificate from the next connection. A console update installs and starts the
  service too.
- **Existing installs switch over safely.** Workers get the root first. The
  console keeps its previous certificate until every worker the controller
  reached recently holds the root (or up to 24 hours when there is no record of
  the workers), then switches. An online worker without the root holds it up;
  `vaelor.tls_authority promote` switches anyway. See `SECURE_ACCESS.md`.
- **Keeping the root.** A keep-data uninstall keeps the root. A data purge
  removes it, and also takes it out of the machine's trust store on a controller
  or a worker. `install-vaelor.sh --import-authority FILE` restores a root
  exported earlier, so your devices stay trusted (`--replace-authority` replaces
  a different one already on the machine).
- The installer checks the console's certificate against the root after the
  health check. A mismatch is reported but does not fail the install.
- The trust commands the installer prints use the console address it reports
  (the machine's IP address), not its hostname, which other devices often cannot
  resolve.

### Cold-install fixes

Found by wiping both machines and installing the published 1.5 release from
GitHub as a first-time owner would.

- **A leftover Docker Swarm no longer confuses the cluster.** The console, joining
  a worker, app placement, capacity and the cluster manager now share one answer
  to "is this machine the controller?". A Swarm left behind without Vaelor's
  controller record shows as not set up, and **Review controller setup** adopts
  it.
- **Uninstall leaves less behind.** A data purge takes the machine out of a
  Docker Swarm only when the Swarm is Vaelor's: Vaelor installed Docker, the
  machine is an enrolled worker, or the purge is a bare-OS one. Any other Swarm
  is left alone, and the command to leave it is printed. Uninstall also stops a
  Vaelor account's processes before removing the account, and removes every
  `vaelor-*` service, drop-in and leftover enablement link, and nothing else.
- **A worker's GPU memory pool reading** is dated only by checks that actually
  reached the worker, so a **Recheck** updates it instead of leaving it "not
  read yet".
- **The NPU model folder is readable after every install path.** Fetching the
  model, the console's Install button and re-running the installer all leave it
  `0755`/`0644`, with a test that keeps the three in step, and re-running the installer repairs a
  folder an earlier release left unreadable.
- The README says that the first visit creates the administrator account.

### Licensing

- **The Assistant's NPU model names its licence: Apache-2.0**, the licence of
  Qwen3.5-4B and of FastFlowLM's NPU conversion it is built on.
  `THIRD_PARTY_NOTICES.md` lists what Vaelor changed (the fine-tuned language
  weights), and the release now carries `qwen35-4b-npu2.LICENSE.txt` beside the
  model parts.

## 1.5

The first release since 1.0 Beta 2. Vaelor grows from one appliance into a
small GPU cluster managed from the same console, AI Chat can reach hosted
services, and the whole console has been redesigned. Everything below was built
and tested, and most of it was run on an HP Z2 Mini G1a controller with an HP
ZBook Ultra G1a worker; "Not verified in this release" at the end of this
section lists what has not yet been run on that hardware.

### Upgrading from 1.0 Beta 2

- **Re-run the installer once** (`git pull` in the clone, then
  `sudo ~/vaelor/deploy/install-vaelor.sh --unattended`) instead of using the
  console update for this one upgrade. The installer now binds InfluxDB to
  localhost (it previously listened on every interface with no
  authentication), stages the worker telemetry agent, and pre-pulls the new
  trace collector; the console update installs the new release but does not
  do those three things.
- **Certificate.** A box installed before 1.5 keeps its existing self-signed
  certificate, which names the machine only in its common name. Workers need a
  certificate that also lists the controller's addresses. Before adding a
  worker, move `/opt/vaelor/tls/vaelor.crt` aside and re-run the installer to
  generate a new one; your browser will ask you to accept it again.
- The installer now downloads from the `v1.5` release by default
  (`VAELOR_RELEASE_TAG` chooses another).

### GPU clustering across machines

- **Add a machine.** Join a second Strix Halo machine from Cluster › Add
  machine over an SSH login with `sudo`. The cluster has its own section in
  the console: Setup, Fleet, Deployments, Performance, Activity, and Agents &
  tools.
- **Serving moves to vLLM when you cluster.** A single machine serves with
  llama.cpp; forming a GPU cluster switches serving to vLLM containers, and
  taking the cluster down switches back. The two never run at once, because a
  model loaded in one holds the memory the other needs.
- **Two ways to use two machines.** *More people at once:* one copy of the
  model on each machine behind a balancer the controller runs, with each
  conversation kept on one copy and a request retried once on the other copy
  if its machine stops answering. *A model bigger than one machine:* split one
  model across machines (pipeline-parallel by default, with tensor-parallel as
  an option that recommends the fastest shared link). A split needs a
  dedicated link between the machines, such as a Thunderbolt cable, and
  nftables on each; Vaelor fences that link so nothing else on your network
  can reach its ports, and refuses a split over your shared network.
- **A recommended cluster model.** Qwen3-30B-A3B-Instruct-2507 in 4-bit
  weights, one copy per machine, on vLLM 0.27. The earlier vLLM 0.22 image
  stays selectable, and a deployment keeps the image it was deployed on.
  Models answer without thinking unless you turn it on.
- **Fit before you serve.** The serve form previews whether a model fits each
  machine, and shows the same refusal the deploy would give.
- **Load, Unload, and scale to zero.** Unload a cluster model by hand, or let
  it unload itself when idle; the next request wakes it. While a model is
  unloaded the LLM Server port stays open and answers that the model is
  unloaded. A failed deployment can be started again from its record with
  Load, and an update re-renders a serving deployment whose settings changed.
- **Machine settings.** Set how much system memory each machine's GPU may use
  for models (applied after you confirm it, counted from a restart you start),
  and choose which of the controller's links a split uses.
- **A slim, controller-managed worker.** A worker runs no console and no
  Assistant of its own. The controller installs what the worker needs (the
  telemetry agent, a GPU sampler, and AMD's `amd-smi` at a pinned version),
  checks it every 15 minutes, and after an update brings it to the
  controller's version. Each worker's card shows a Worker software row. The
  full installer refuses to run on a worker; `--leave-worker-role` takes the
  worker software off when its controller is gone for good. A machine that
  already carries the full appliance is shown as such; it is not converted in
  place in this release.
- **Removing machines.** A worker that can no longer be reached can be
  removed by force after a typed confirmation, and the removal says what may
  remain on it.
- **Cluster apps.** Apps deployed to the cluster get placement, replicas, CPU
  limits, label constraints, and a re-check of placement when reconfigured. A
  researched multi-service app can be deployed to the cluster as one group.
  App data volumes can be backed up, restored (the service is always scaled
  back afterwards), and deleted from the console, and the data a removed app
  leaves behind is listed and deletable by name.

### LLM Server, API keys, and usage

- **The LLM Server** publishes the model you serve — on one machine or on the
  cluster — as an OpenAI-compatible API on port 11434 of your LAN, in Cluster ›
  Deployments.
- **Keys you control.** Mint several keys, each shown once; rotate or revoke
  them; rotate the cluster's internal serving key in place. The console warns
  that revoking the last key closes the port.
- **Usage metering** per key and per deployment, counted by the model and the
  LLM Server gate rather than trusted from a request header, with one "last
  used" time per key.
- Every key change and refusal is in the Security audit, written in words.

### Agents, MCP tools, and skills

- **MCP catalog.** Administrators curate the MCP servers agents may use; each
  agent is granted specific tools, deny-by-default.
- **Skills library.** Write skills with an instructions body, import them from
  GitHub, and attach them to agents.
- **Cluster agents.** Deploy an agent that runs on the cluster's model from an
  easy gallery or an advanced form (backing model, MCP grants, skills, review).
  A deployed agent is read-only, has its own inbound key and its own memory,
  and follows its current keys and model. Tool calling is enabled on the
  cluster's vLLM so agents can use their MCP tools.

### Observability

- **Fleet metrics.** Per-machine metrics with history for the controller and
  every worker. Workers report through a lean telemetry agent the controller
  installs, sending to a keyed endpoint over TLS pinned to the controller's
  certificate.
- **Alerts on workers.** An alert rule can watch a worker's CPU temperature
  or memory against that machine's own readings.
- **Performance dashboard.** A per-machine dashboard of measured values only:
  request health for every way in (the inference gateway, the LLM Server, and
  AI Chat), each judged on time to first word and writing speed; the serving
  engine's own throughput and queue; live in-use memory; and on-demand
  profiling (a CPU capture and a GPU snapshot), with the exact reason when a
  profiler cannot run.
- **Request tracing** to an Arize Phoenix collector the controller runs on
  loopback only; the console says how to reach it over SSH.
- **Activity and alerts** for the cluster in one feed. "Needs attention" means
  the same thing everywhere: failed, waiting for approval, or paused, and not
  since resolved.

### AI Chat

- **Hosted services.** Connect OpenAI, Anthropic (its own API), Google Gemini,
  OpenRouter, or any HTTPS endpoint. Hosted connections are for AI Chat only
  and are labelled as leaving the machine. They are reached through a pinned
  HTTPS transport: public addresses only, the checked address is the one
  connected to, no redirects, certificates verified, an overall deadline, and
  keys redacted from every log and error.
- **Thinking control.** Off, Low, Medium, or High, for every model that
  supports it, starting at Off wherever a model allows it; the model's
  thinking is shown as a collapsed summary.
- **Your choice while the cluster serves.** When a cluster is formed it takes
  AI Chat's model once; after that AI Chat can use any model — the cluster's,
  this machine's, or a hosted one — without touching the cluster.
- **Servers only when you add them.** Vaelor no longer looks for AI servers on
  its own; a server or service appears only after an operator adds it.
  Viewers never see a server address.
- On a machine whose Assistant runs on the NPU, AI Chat no longer lists the
  Assistant's NPU model. On a Raspberry Pi, where one model serves both, it
  stays.

### Assistant

- **Knows the cluster.** The Assistant answers questions about every machine
  in a cluster — what it serves, its health, per-machine history and events,
  disks, network, and fans — from the telemetry Vaelor collects, and answers
  "which model does AI Chat use?" and "is the LLM Server running?" from the
  serving state itself.
- **Keeps its own model.** The Assistant always uses the on-device model
  Vaelor installs. It never moves to the cluster's model and never uses a
  hosted service; custom-application research that needs a bigger model falls
  back only to a model on this machine or the cluster.
- **Never shows an empty or unreadable answer.** A reply from the on-device
  model with no text the owner can read, or only labels and internal ids, is
  retried once and then reported as no answer.
- Research escalation to the larger model was fixed so it actually escalates.

### Console redesign

- Every page was rebuilt to a new design: navigation grouped as Overview
  (Home, Cluster, System, Remote console), AI (Apps and AI, AI Chat,
  Assistant), and Manage (Activity, Settings); a new icon set; search across
  the console; no visible text smaller than 12 px.
- A refused action is shown inside the dialog where you asked for it.
- Update Vaelor moved to System › Hardware and services.
- Home and System › Compute show the GPU and NPU on their own cards, with
  honest states when a reading is missing.

### Security

- **Sign-in attempts are limited** per address and per account; an owner's
  known address is not locked out.
- **InfluxDB listens on localhost only** (see Upgrading).
- **The appliance certificate lists the machine's names and addresses**, so
  browsers and worker telemetry verify it.
- The audit shows the real client address, never the TLS terminator's.
- The remote-desktop service's state directories are owned by Vaelor, and the
  privileged bridge never follows a path inside them.
- A port held by something Vaelor manages is never handed to another app or
  model, and ports bound only to loopback get no Open link.
- The sudo password used to enrol a worker never reaches the worker's files or
  programs, and worker files are staged only in a private directory that is
  removed afterwards.

### Install, update, and remove

- The installer also pre-pulls the Arize Phoenix image (about 1.5 GB) and
  stages the pinned telemetry agent for workers.
- The installer and every update remove stale web-interface files left by an
  earlier version.
- "Up to date" and Reinstall are judged by the installed build, not only the
  version number.
- Remove Vaelor states exactly what it removes, including containerd's image
  store, and removes only what the installer added.

### Fixes and smaller changes

- **Performance counts every way in.** The request latency and errors card,
  its "why", and the Assistant's performance tool cover the LLM Server and AI
  Chat as well as the inference gateway, each with its own verdict against the
  budget.
- Many fixes from repeated whole-console sweeps: status words that match the
  backend, byte sizes shown with one unit per quantity, a saved app
  configuration that says when it reaches the running app, Restart service
  audited before it restarts and restarting once, and more.

### API changes

- `/api/v2/cluster/performance`: the keys `gateway`, `baseline_gateway`,
  `gateway_detail`, and `gateway.scope` are renamed `requests`,
  `baseline_requests`, `requests.doors[].coverage`, and
  `requests.latency_basis`. The old names are still served in 1.5 and are
  listed under `deprecated_keys`; they will be removed in a later release.

### Not verified in this release

- Splitting one model across machines has been proven on two machines, but the
  link fence added since, and the tensor-parallel option, have not yet been
  re-run live on two machines.
- The GPU memory pool and cluster link settings have not yet been re-checked
  live since they were finished.
- The sign-in attempt limits are covered by tests but have not yet been
  re-checked live on the appliance.
- The worker software job was run live on an existing worker. Its `amd-smi`
  install path, its behaviour beside a split deployment, and its failure paths
  are covered by tests only.
- GPU serving and clustering are verified only on the AMD Strix Halo GPU
  (gfx1151). Other GPUs are reported, not served on.
- The Raspberry Pi was not re-tested on hardware for this release.
- Adding a brand-new worker with the slim install is covered by tests only; the
  live runs updated an existing worker.

## 1.0 Beta 2

The second public beta. Everything in Beta 1, plus a wave of work driven by live
end-to-end testing on the HP Z2 Mini: a genuinely one-command install, a full
uninstaller, one-click applications, documents in AI Chat, a smarter Assistant,
and a more consistent interface.

New in this release:

- **One-command install.** `git clone` and run the installer — that is the whole
  procedure. The installer now finds or downloads the release wheel itself, so
  there is no wheel path to hunt for and no need to `cd` into the clone, and it
  provisions the on-device NPU Assistant model as part of the same run rather
  than a separate step (defer that ~3.35 GB download with `--without-npu-model`).
  It self-heals two failures that used to strand a fresh box — AMD rotating its
  ROCm point release, and a dpkg trigger race during a system update — installs
  Python if the base image lacks it, and stages its own scripts so the in-console
  uninstaller works on a wheel install too.

- **Remove Vaelor.** A complete, typed-confirmation teardown that returns the
  machine to a bare OS, from the console or the command line: it removes the
  services, accounts, state, and AI runtimes. It runs detached so it can remove
  the very control plane that launched it, and was validated by wiping and
  reinstalling a real appliance from scratch.

- **One-click applications.** A curated catalog of eleven ready-to-deploy apps,
  each with a real guided setup — reveal generated secrets, edit configuration
  files, publish a web port, apply labels. Every managed application now has an
  admin console for running commands inside its container, a file manager for its
  data volumes, and working, timestamped logs. Files you upload are readable by
  apps that do not run as root.

- **AI Chat reads your documents.** Retrieval in AI Chat now extracts text from
  PDF, Word (`.docx`), Excel (`.xlsx`/`.xls`), and PowerPoint (`.pptx`) files,
  not just plain text, so you can ask questions grounded in real documents.
  Untrusted-file parsing is hardened against XML-entity, decompression-bomb, and
  disguised-binary attacks. Legacy `.doc`/`.ppt` are declined with clear guidance
  to re-save in a modern format.

- **A smarter, better-behaved Assistant.** The on-device NPU Assistant is
  auto-enabled the moment a working model is present and re-enables itself if it
  stops. Definition and general-knowledge questions ("what is an operating
  system?") go to AI Chat, while questions about *this* machine stay on the
  Assistant. On a box with both a GPU and an NPU the two model tiers no longer
  compete over which serves AI Chat, a model already in managed storage is reused
  instead of downloaded again, whole-document retrieval was fixed, and the
  model-switch tier is now labelled clearly. Local-LLM performance metrics were
  added.

- **Web research works on a clean box.** The guarded web-research engine is now
  provisioned automatically, so a granted agent can actually reach the web
  without a separate manual container deploy.

- **Truthful networking and update flows.** A Docker bridge that is down no
  longer reports a stale gateway address as if it were live. The software-update
  flow shows a live download status, and the setup assistant's "Done" button now
  actually dismisses a finished operation.

- **A more consistent interface.** Three waves of UI polish fixed concrete bugs
  surfaced by a full interface audit and converged hand-built screens onto the
  shared design system, closing responsive-layout gaps along the way.

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
