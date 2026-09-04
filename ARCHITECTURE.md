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

The native `arm64` and `amd64` Debian packages install all eleven systemd
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
