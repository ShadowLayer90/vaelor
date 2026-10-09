# Vaelor architecture

Vaelor is a local-first control plane. The web application and public API
consume capability data; platform adapters decide how a supported host
implements those capabilities.

## Runtime layers

1. **Web and API** — the React interface and authenticated `/api/v2` routes.
2. **Control-plane services** — assistant, agents, RAG, workload lifecycle,
   fleet orchestration, recovery, audit, and inference routing.
3. **Guarded brokers** — credential, workload, update, desktop, recovery, and
   noVNC processes with narrow identities and fixed responsibilities.
4. **Capability adapters** — hardware, OS, package manager, container
   scheduler, inference, storage, remote access, telemetry, and power.
5. **Managed state** — versioned SQLite and JSON state under
   `/var/lib/vaelor`, logs under `/var/log/vaelor`, and sockets under
   `/run/vaelor`.

## Stable boundaries

`vaelor` and `VAELOR_*` are the implementation and public identities. The
`pm_dashboard` Python namespace contains only compatibility aliases during the
first Vaelor migration window. New integrations must use the Vaelor commands,
environment variables, service names, and paths.

Generic hosts must never be identified as Pironman simply because a peripheral
probe is unavailable. Unsupported controls remain absent or read-only, and a
server-side guard — not the UI — is what enforces that.

`platform_contracts.py` defines the stable structural interfaces.
`vaelor/platforms/` is the hardware-platform driver package and holds the
registry: `platforms/base.py` carries the hardware-neutral primitives,
`platforms/raspberry_pi.py` is the Raspberry Pi and Pironman enclosure driver,
`platforms/workstation.py` is the x86-64 workstation and generic-host driver,
and `platforms/accelerators.py` is read-only GPU and neural-accelerator
discovery. `platforms/__init__.py` registers each driver with a probe;
`select_hardware_platform()` runs the probes in registration order and takes
the first that claims the host, with `VAELOR_PLATFORM_DRIVER` as a test-only
override. `register_driver()` adds a driver ahead of the generic fallback.

Each driver answers `machine_class` (`pi-appliance`, `workstation`, or
`generic`), a product, a power snapshot, a memory split, a thermal policy, and
one `{available, reason}` record per capability. `reason` is user-facing prose
explaining why *this* machine cannot do it and is `null` when it can.

Three readings on that contract exist because the obvious field was wrong.
`cpu_cores` means **cores**, read from `Core(s) per socket` × `Socket(s)` or
from the kernel's CPU topology — not `os.cpu_count()`, which is threads;
`cpu_threads` carries the logical count beside it and
`cpu_cores_are_threads` flags a host where neither could be established.
`memory_split` states what the OS can see, what firmware reserved for graphics,
and whether that split is a setting an owner can change rather than a property
of the part. `power.previous_shutdown` says whether the last boot ended in a
shutdown or simply stopped, because `vcgencmd get_throttled` is cleared by a
power loss and cannot answer it.
`GET /api/v2/system/machine` serves that contract. Capability availability
comes from discovery, never from the machine class alone: an enclosure that
really is reporting on an unusual host is believed.

`platform_drivers.py` supplies the remaining default Linux implementations and
adapts the selected hardware driver. Callers receive normalized capabilities
and fixed guarded commands from those drivers; they do not infer support from
an architecture or distribution name. Reference-platform details remain inside
the driver implementation or the upstream Pironman enclosure bridge.

Host power is a driver capability, not a legacy import check. Each driver
answers `power_actions()`: the base implementation reaches systemd-logind with
a fixed argument vector, and the Raspberry Pi driver overrides it only to keep
SunFounder's sequenced shutdown, falling back to the generic path when that
helper is absent. The privileged hardware bridge holds the privilege and the
driver supplies the mechanism; the unprivileged control plane asks the bridge.
The bridge unit therefore starts on every machine and treats the enclosure as
optional.

Health thresholds are a platform fact. 70/80 °C is a Raspberry Pi constant
because the SoC soft-throttles at 80 °C; a workstation processor boosts into
the mid-nineties by design and is served 97/100 °C. The UI consumes the served
policy rather than keeping its own table.

The shared runtime registry is composed once and injected into API, workload,
remote-access, storage, telemetry, and power consumers. `linux_storage.py` and
`linux_telemetry.py` are generic host implementations; enclosure adapters may
enrich their results but do not replace the portable baseline. Unsupported
mutations report a capability reason and fail closed with HTTP 503: each
enclosure route checks the driver's capability answer before acting and also
converts a raising hardware callback into the same 503, so a control that did
nothing can never report success.

`linux_sensors.py` selects CPU temperature from a *labelled* hwmon sensor and
reports the source it used. Taking `max()` over thermal zones is a last resort
flagged as unlabelled, because on an AMD workstation the only ACPI zone reads
the board, not the processor. Telemetry omits a key entirely when it cannot be
measured; a missing sensor is never reported as zero.

The OS driver owns the managed-service catalog, while the package driver owns
both guarded update commands and normalization of update inventory. The
inference driver supplies runtime architecture and API features to workload
capability reporting. This keeps `SystemInventory`, APIs, Assistant guidance,
and UI policy independent of systemd unit names, APT output, and reference-host
labels.

Native Pironman power actions cross the existing root-owned hardware bridge
over its group-restricted Unix socket. The bridge accepts only service restart,
host reboot, and host shutdown identifiers and maps them to fixed `systemctl`
argument arrays. Arbitrary commands never cross that boundary. Hosts without
that privileged provider expose all three actions as unavailable.

## Workloads and inference

Docker Compose manages single-node applications. Docker Swarm manages
multi-node application placement, health, drain, and failover. Inference is a
separate subsystem:

- a managed local server handles a model on one node;
- replicated servers increase throughput and availability; and
- a pooled backend may divide one compatible model across exactly 2, 4, or 8
  private wired nodes.

All modes sit behind Vaelor's OpenAI-compatible inference gateway. The gateway
provides scoped API tokens, model discovery, streaming, health routing,
telemetry, and explicit failure semantics.

### GPU serving: one container per recipe, one engine at a time

GPU serving runs in containers, never as a host binary. A bare engine has to
match the host's ROCm exactly and does not survive the host being re-imaged: a
soname drift between the build's ROCm and the box's is not something a library
path can bridge. So the image carries its own ROCm and the host carries none of
the engine's requirements — only a GPU, its device nodes, and a working Docker.
The engine is launched by the root hardware bridge, because the workload
executor's sandbox hides `/dev/dri` and `/dev/kfd`; the container is the sandbox
and Docker is the privilege boundary.

Which image runs is chosen by the model's catalog engine, not by a flag: a stock
GGUF gets the mainline ROCm image driven as `llama-server`, and the FP4 model
gets the fork image whose own entrypoint owns its recipe. Two recipes, two
images, one code path that resolves between them. Because the images are a
prerequisite rather than an artifact of the deploy, the installer pre-pulls them —
on hardware that can actually run them, and only once the console is up and
healthy, so the longest download in the install never stands in front of the
health gate.

The single-node model container always binds loopback and is never passed an API
key — one engine was proven not to enforce a key at all, which would have made an
engine-level key an unauthenticated LAN endpoint. Exposing a model on the LAN is
therefore a separate Vaelor-controlled auth proxy in front of that loopback port,
and its invariant is that there is no other way in.

Across machines the engine changes. A single node serves on llama.cpp; enabling
GPU clustering moves serving to vLLM, which shards one model across nodes, and
tearing the cluster down reverts to llama.cpp, which is never deleted. The two
are a mutually exclusive mode switch rather than engines that coexist, because a
model resident on one holds the memory the other needs. vLLM is never offered for
a single node, and its image is pulled per node by the deploy that needs it
rather than placed on every machine at install.

**The clustered mode holds the same invariant.**
Every vLLM API server binds loopback on its own machine, whoever leads. A
controller-led cluster is fronted by the auth proxy. A worker lead,
like a worker replica, is fronted by a keyed nginx gate on that worker's
cluster address, and the controller reaches it only with the key. Ray's and
RCCL's sockets are a separate plane on the chosen cluster link. A worker-led
split deployed by a build older than 1.5 stays LAN-open, and the console labels
it so, until it is Loaded again. The worker-led split has not yet been run live
on two machines.

**A split's Ray plane is a second door, locked with a token and fenced on its own link.**
Without these measures, anyone who can reach a split's Ray ports can run code as root
on every machine of the split. Ray's defaults take any joining node and any remote
driver; RCCL, Gloo and the TCPStore have no authentication at all, and Gloo carries
pickled objects; and the containers run as root with the host's network and GPU. So
**a split needs a dedicated cluster link** (Cluster > Setup, such as a Thunderbolt
cable) **and nftables on every machine**, and is refused on the shared network. Every
Ray container runs with Ray's token authentication (a per-deployment token, changed at
every Load, root-only on each machine), the head has no dashboard and no monitor, and a
Vaelor-owned nftables table fences the link to the split's machines and drops Ray's
pinned port band for everyone else on every other interface. Each Ray unit loads that
table before it starts, so a reboot does not bring Ray up unfenced.

Unknown application requests use a separate researched-deployment pipeline.
The unprivileged control plane sends only an intent and up to eight public
source URLs over a Unix socket to `vaelor-application-research`. That service
is the only component in the workflow allowed outbound HTTPS. It pins DNS
answers to the connected peer, revalidates redirects, rejects non-public and
metadata addresses, bounds compressed and decoded responses, and returns
normalized untrusted evidence. A server-owned manifest and Compose draft are
then bound by SHA-256 digests. Only an administrator approval can mint the
minimal `compose.import` job reference consumed by the executor.

An agent's guarded fetch adds provenance to that reachability boundary, and
the two are separate questions. Reachability asks whether an address is public;
provenance asks whether this run was authorized to read that page. A fetch may
read an operator-allowlisted domain, a URL this run's own guarded search
returned, or a page on the same registrable domain as the hop that authorized
it - checked on every redirect, not only the first, and capped at three hops
with loop detection. The policy travels across the research socket because the
broker process is the only component that sees a redirect. Vaelor ships no
Public Suffix List, so `vaelor/research_provenance.py` approximates the
registrable domain from the last two labels plus an explicit table of
multi-label and shared-hosting suffixes; a suffix missing from that table
merges two sites into one, which is why the table, and not the fetch, is the
thing to extend. Recorded evidence names the URL the fetch actually landed on.

### Machine settings: the GPU memory pool and the cluster link

Two host settings are owner-chosen and capability-discovered rather than fixed
by the installer. The GPU memory pool - how much system memory
a shared-memory GPU may use for models - exists only on a machine whose kernel
reports the limit and whose GPU was found to share system memory. It is changed
through a dedicated root-bridge verb on the controller, and through the enrolled
SSH channel on a worker: one whole number inside bounds, one fixed file, a
boot-image rebuild, and never a restart. The cluster link is a choice among the
controller's own network links, used only when one model is split across
machines; each participant binds the address it already holds on that link's
network, and a machine without one refuses the deploy before anything starts.
Neither setting is applied by an install, a repair or an upgrade.

## The GPU cluster: one controller, slim workers

A GPU cluster has one **controller** — the machine the installer ran on, which
holds the console, the Assistant, the cluster's records, and the credential
broker — and one or more **workers**. A worker is enrolled over a
fingerprint-pinned SSH login and is driven entirely over SSH; it runs none of
Vaelor's own services. The controller lays a versioned **worker profile** on
each worker (a telemetry agent, a GPU sampler, and a pinned `amd-smi`), records
the profile's digest in a root-owned marker on the worker, re-checks it every
15 minutes, and after its own update brings each worker to the new profile. The
marker is also what makes the full installer refuse to run on a worker, so a
worker never turns back into a second appliance by accident.

Serving on the cluster has two shapes, both on vLLM containers launched through
the root bridge on the controller and through the enrolled SSH channel on a
worker:

- **Replicated** — one complete copy of the model per machine, behind an nginx
  balancer the controller runs on loopback. The balancer shares its state across
  its worker processes, keeps each conversation on one copy, fails a replica that stops
  answering, and resends a request that never reached a replica once to
  another. Throughput rises with the number of machines; a single answer does
  not get faster.
- **Split** — one model divided across machines, pipeline-parallel unless the
  owner chooses tensor-parallel, for a model too large for one machine. Only a
  split uses the dedicated cluster link and its fence (above).

The LLM Server's proxy fronts whichever loopback port serves — the single-node
model or the cluster's balancer — so the mode switch never changes how a LAN
client reaches the model. A model that is unloaded, by hand or by idle
scale-to-zero, keeps its keyed door: the proxy answers that it is unloaded, and
for an idle unload the next keyed request wakes it. Usage is counted from the
model's own responses and the proxy's gate, never from a request header, and
every key is a broker credential shown once and kept only as a fingerprint.

Cluster **agents** run as their own containers launched through the root
bridge. Their MCP tool grants are enforced at the call itself, intersected with
the tools the server is currently approved for, so a grant can only narrow. A
deployed agent is read-only, reaches no broker socket, and receives its secrets
as root-owned files mounted into its container.

## Telemetry and observability

Each worker's telemetry agent sends to one keyed endpoint on the controller,
`POST /api/v2/telemetry/ingest`, over TLS pinned to the controller's own
certificate, which is why that certificate lists the controller's addresses.
Readings land in the controller's InfluxDB, which listens on localhost only.
The Fleet view, alert rules, the Performance dashboard, and the Assistant all
read the same per-machine series, and the Performance snapshot is one derivation
served both to the tab and, as a read-only tool, to the Assistant and MCP
clients.

Per-request traces go to an Arize Phoenix collector the controller runs as a
container published on loopback only. On-demand profiling runs through the root
bridge as a bounded capture, and reports the exact reason when a profiler cannot
run on this host.

## Model connections and where prompts go

The **Assistant** always runs on the model Vaelor installs for it — on the NPU
on a Strix Halo machine — and never takes an outside model or the cluster's.
Its custom-application research may fall back to AI Chat's model only when that
model runs on this machine or the cluster.

**AI Chat** may use a local model, the cluster's model, a server on the local
network that an operator added, or a **hosted service** (OpenAI, Anthropic,
Google Gemini, OpenRouter, or a custom HTTPS endpoint). Nothing is discovered:
an AI server exists in Vaelor only after someone adds it. Hosted services are
their own connection kinds, granted to AI Chat and to nothing else, and are
reached only through a pinned HTTPS transport: the host name is resolved once,
every address must be public, the connection goes to the address that was
checked, redirects are refused, the certificate is verified against the host
name, the whole request has a deadline, and the key is redacted from every log
and error in every encoding. A connection's server address is shown to
operators and administrators, never to viewers.

## Mutation model

Inspection and planning are side-effect free. A mutation becomes a durable job
only after approval. Jobs record evidence, progress, outcome, and recovery
metadata. Privileged work is delegated to the smallest broker that can perform
it; the model never receives a shell or raw credential.

Unattended runs are the one case where no human approves the individual run.
A schedule or an alert rule creates its agent task ready to start, so creating
the rule is the approval for every run it makes. That is why creating, pausing,
and deleting one is administrator-only, why ownership is re-checked against the
user table each time a rule fires rather than trusted from the row, and why the
rule states the pinned definition's own read scopes, research policy, and
integrations before it is saved. The run itself stays inside the same mutation
model as every other: it may execute reads, and a connector write, an app
capability write, or a knowledge write becomes a preview that needs its own
separate approval before any transport happens.

## Distribution boundaries

The native `arm64` and `amd64` Debian packages install all of Vaelor's systemd
services and are the only distributions that claim host-level control. The
multi-architecture OCI image contains the same portable Python and web core,
but drops Linux capabilities, runs as an unprivileged UID, binds to loopback by
default, and does not mount host devices, systemd, package-manager state, or
the Docker socket. Capability discovery therefore hides unavailable host
actions instead of simulating them.

Portable state is a separate encrypted, versioned contract. Export snapshots
live SQLite databases, removes active sessions, verifies every declared file,
and includes only users, conversations, agents, knowledge, audit history,
update metadata, and workload definitions. Host-bound credentials, tokens,
models, volumes, backups, TLS material, sessions, and hardware identity remain
on the source node. Import is staged under the recovery broker, requires an
exact one-use approval, stops only the fixed Vaelor service set, and rolls back
partially replaced files before services restart if any step fails.
