# Supported platforms

This document describes Vaelor compatibility. It is intentionally more
specific than “the Pironman hardware can run on this OS”: some appliance
operating systems support an enclosure but cannot provide a general-purpose
Docker host, package manager, desktop, or local AI runtime.

Last reviewed: 2026-10-10 (Vaelor 1.5.1)

## Support levels

- **Verified** — the complete control plane is an intended configuration and its core paths are covered by tests.
- **Compatible** — the OS uses the supported Linux interfaces, but individual releases or desktop services may require additional validation.
- **Limited** — SunFounder provides a Pironman setup path, but the OS intentionally restricts one or more control-plane features.
- **Not validated** — detection and read-only telemetry may work, but the project does not claim functional support.

The Overview page reports the detected OS and one of these levels. Feature availability is also gated at runtime using architecture, RAM, CPU, desktop services, package manager, and hardware discovery.

## Operating-system matrix

| Operating system | Level | Hardware controls | Docker and apps | Assistant and local models | Browser desktop | Host updates |
| --- | --- | --- | --- | --- | --- | --- |
| Raspberry Pi OS 64-bit (Desktop or Lite) | Verified | Full where fitted | Full | Full when RAM permits | Desktop edition only | Full |
| Debian 64-bit on Raspberry Pi 5 | Verified | Full where fitted | Full | Full when RAM permits | When a supported desktop is installed | Full |
| Ubuntu 64-bit (Desktop or Server) | Compatible | Full where fitted | Full | Full when RAM permits | Desktop edition only | Full |
| Ubuntu 24.04 ARM64 generic host | Compatible | Capability discovery only | Full | Full when RAM permits | Desktop edition only | Full |
| Ubuntu 24.04 AMD64 generic host | Compatible (package verified) | Capability discovery only | Full | CPU; accelerator discovered and sized, offload opt-in | Desktop edition only | Full |
| Ubuntu 26.04 AMD64 workstation (HP Z2 Mini G1a) | Compatible (turnkey installer validated end to end on the appliance; GPU cluster controller) | Absent with a reason | Full | On-device: Assistant on the NPU, AI Chat on the GPU, both fetched and provisioned by the installer; vLLM when clustered | Desktop edition only | Full |
| Ubuntu 26.04 AMD64 laptop (HP ZBook Ultra G1a) | Compatible (turnkey installer validated end to end; GPU cluster worker) | Battery/AC reported; enclosure controls absent with a reason | Full | As a full appliance: Assistant on the NPU, AI Chat on the GPU. As a cluster worker: vLLM serving only, managed by the controller | Desktop edition only | Full (a worker is updated by its controller) |
| Kali Linux 64-bit | Compatible | Full where fitted | Full | Full when RAM permits | When a supported desktop is installed | Full |
| Homebridge on a supported Debian-family host | Limited | Full where fitted | Existing host workloads only | Hosted providers recommended | Host-dependent | Host-dependent |
| Home Assistant OS | Limited | SunFounder add-on path | Not a general Docker host | Hosted provider only | Not available | Managed by Home Assistant |
| Umbrel OS | Limited | Hardware integration path | Umbrel-managed apps | Hosted provider recommended | Not supported by this project | Managed by Umbrel |
| Batocera.linux | Limited | SunFounder setup path | Not supported | Not supported | Not supported | Managed by Batocera |
| Other Linux distributions | Not validated | Read-only discovery may work | Disabled until validated | Hosted provider may work | Not validated | Disabled |
| Non-Linux hosts | Unsupported | No | No | No | No | No |

### Important release notes

- Use a **64-bit (`aarch64` or `x86_64`) OS** for local models and the complete
  feature set. A 32-bit OS cannot install several modern AI dependencies.
- A clean Ubuntu 24.04 ARM64 generic host passes the Vaelor installer and
  eleven-service HTTPS pre-production gate in the developer-only QEMU lab.
  QEMU is not a supported deployment platform and does not replace acceptance
  on physical Raspberry Pi hardware. Pironman hardware controls are
  intentionally absent on that host.
- A clean Ubuntu 24.04 AMD64 generic host passes the same wheel, installer,
  eleven-service, and HTTPS gate without Docker preinstalled. The test lab is
  development infrastructure and is not part of the Vaelor release.
- Native Debian packages are architecture-labelled for `arm64` and `amd64`.
  They install the full host appliance. The OCI distribution is a restricted
  portable core: it supports the UI, Assistant, AI Chat, portable databases,
  and connected OpenAI-compatible inference, but deliberately does not control
  host Docker, systemd, updates, remote desktop, or physical hardware.
- The currently deployed Ubuntu 26.04 system is treated as **compatible**, not as a SunFounder-published verified release. Its control-plane paths are tested on the actual appliance, while release-specific desktop behavior is checked at runtime.
- The **HP Z2 Mini G1a (Strix Halo)** is the x86-64 AI-workstation appliance line.
  On it the single `install-vaelor.sh` run is validated end to end: it provisions
  the control plane, Docker, and InfluxDB, and — gated on the hardware it finds —
  the FastFlowLM NPU runtime with the on-device Assistant model (Qwen3.5-4B,
  fetched from the release), plus the gfx1151 ROCm runtime and the container
  images the GPU AI-Chat model and the LAN LLM Server serve from. A clean install
  from a wiped disk and the in-console "Remove Vaelor" teardown back to a bare OS
  were both verified on the appliance. The bare-OS teardown removes Docker and
  containerd with both image stores (`/var/lib/docker` and
  `/var/lib/containerd` - on Docker 29 the latter is where the images live, and
  the first cold teardown left 44 GB there), InfluxDB, the ROCm packages,
  amd-smi, novnc and the browser-desktop packages; it purges only the OS
  packages Vaelor recorded installing (so a Docker, InfluxDB or amd-smi that was
  already present is left alone) and keeps the shared OS tools and anything the
  installer did not add (`deploy/README.md` lists both sides). This validates
  that machine; it does not promote every Ubuntu 26.04 build to Verified.
- The **HP ZBook Ultra G1a (Strix Halo)** is the same silicon in a laptop: the
  identical `AMD Ryzen AI Max` SoC (Radeon 8060S iGPU, gfx1151, PCI `0x1002:0x1586`)
  and XDNA NPU (`0x1022:0x17f0`), so it resolves to the same **workstation**
  machine class and the same GPU/NPU code paths as the Z2 Mini — no platform-driver
  changes were needed. A clean `install-vaelor.sh` run was validated end to end on
  the laptop (2026-09-03): control plane, Docker, InfluxDB, the FastFlowLM NPU
  runtime with the Qwen3.5-4B on-device Assistant serving on the NPU, and the
  gfx1151 ROCm runtime plus the GPU AI-Chat engine, with all twelve services up
  and the console serving over HTTPS. The laptop battery/AC is reported through
  the standard power-supply telemetry; the physical enclosure controls a Pironman
  provides are absent with a reason, as on any workstation. One first-run note:
  a bare host needs `git` installed (`sudo apt install -y git`) before the
  documented `git clone` step. This validates that machine; it does not promote
  every Ubuntu 26.04 build to Verified.
- **The GPU serving step changed after the two runs above were recorded**:
  serving moved into containers, so the installer no longer fetches a bare
  `llama-server` build and pre-pulls the serving images instead. The gfx1151
  ROCm runtime is still installed, for host telemetry (AMD's ROCm build of
  `amd-smi` is the one that publishes this part's APU metrics) and the reported
  ROCm version, not for serving. **A cold clean install of both machines on
  2026-09-06** — each torn down with the documented bare-OS uninstall and
  reinstalled by the documented first-time steps — validated that change: twelve
  services active and enabled, the serving images present at their pinned
  digests, the NPU model fetched from the release, and the console answering —
  and, on the Z2 Mini, again after a reboot.
- **Vaelor 1.5** reached the Z2 Mini by upgrading it in place, with a model
  serving across the Z2 Mini and the ZBook throughout. The ZBook now runs as a
  **cluster worker** that the Z2 Mini provisions and updates, rather than as a
  full appliance.
- **The published 1.5 release was installed cold from GitHub on 2026-10-09**:
  the Z2 Mini wiped and installed with the documented installer, and the ZBook
  wiped and enrolled again as a worker from the Z2 Mini. It found six defects,
  fixed in 1.5.1. The fixes were deployed in place to the Z2 Mini, and the
  certificate switchover was verified on both machines; 1.5.1 itself has not
  had a cold install.
- The commissioned Ubuntu 26.04 Raspberry Pi completed the versioned
  Pironman-to-Vaelor 2.0.4 migration on 2026-07-30. Ten Vaelor services,
  HTTPS health, encrypted credentials, legacy aliases, and the existing Qwen
  model were verified after migration. This validates that appliance; it does
  not promote every future Ubuntu 26.04 build to the Verified support level.
  **The Raspberry Pi was not re-tested on hardware for Vaelor 1.5**; no Pi
  run is recorded for this release.
- Raspberry Pi OS Lite and Ubuntu Server do not include a graphical desktop. KVM remains available when capture hardware is fitted; browser RDP/VNC requires a desktop service.
- A NAS is a workload configuration, not a separate released Pironman enclosure
  profile. Pironman 5 and Max models can host OpenMediaVault or another NAS
  stack when its OS and storage requirements are satisfied.

## Machine classes

Every host resolves to one machine class, chosen by a driver probe in
`vaelor/platforms/`, and `GET /api/v2/system/machine` reports it alongside one
`{available, reason}` record per capability.

| Machine class | Selected when | Enclosure controls | CPU health thresholds |
| --- | --- | --- | --- |
| `pi-appliance` | The device tree names a Raspberry Pi | Present where the enclosure bridge reports the peripheral | 70 °C elevated, 80 °C critical (the SoC soft-throttles at 80 °C) |
| `workstation` | SMBIOS identifies the machine | Absent with a reason unless an enclosure is genuinely discovered | 97 °C elevated, 100 °C critical (these processors boost into the mid-nineties by design) |
| `generic` | Neither a device tree nor SMBIOS identity | Absent with a reason | As `workstation` |

### What an x86 host does and does not provide

- **Provided:** SMBIOS identity, labelled CPU temperature (`k10temp`/`Tctl` or
  `coretemp`), NVMe and network sensor temperatures, per-volume storage use,
  uptime, and unprivileged accelerator telemetry read straight from sysfs —
  temperature, power, clock, utilisation, VRAM and GTT.
- **Absent, with a reason:** case fans, case lighting, OLED, CPU-fan control,
  and battery. HP, Dell and Lenovo business machines keep the fan curve in the
  embedded controller; the in-tree `hp-wmi-sensors` interface is documented
  read-only and on the probed machine exposes no `fan*` or `pwm*` attribute at
  all. Fan control is reported as unavailable, never stubbed.
- **Never treated as a fan:** the ACPI `Processor` cooling devices. A desktop
  x86 host exposes one per thread — thirty-two on the probed machine — each a
  passive throttle state. Driving them would present a working-looking
  four-level fan control whose only effect is to throttle the CPU.
- **Host power works.** Reboot, shutdown, and control-plane restart are
  available on a generic x86 host through systemd-logind. The privileged
  hardware bridge performs them, so **no polkit rule is required or shipped**:
  it runs as root and logind authorizes it unconditionally. The unprivileged
  control-plane account deliberately cannot power the host itself; it asks the
  bridge over its Unix socket. An unprivileged service has no active local
  session, so logind falls through to `auth_admin_keep` and cannot be satisfied
  non-interactively — which is why granting that account the polkit action
  would be the wrong fix rather than the missing one.
- **Not available on any x86 host without root:** package energy counters
  (`intel-rapl` `energy_uj` is root-only since the PLATYPUS mitigations), and
  there is no equivalent of the Raspberry Pi undervoltage or throttle bitmask.
  `power.source` is `null` on these machines rather than a fabricated label.

### Accelerators

Discovery is read-only sysfs and needs no vendor tooling: `amd-smi` and ROCm
are optional enrichment and their absence changes nothing. A discovered
accelerator is not a usable one — `/dev/kfd` and `/dev/dri/render*` are owned
by the `render` and `video` groups, which are frequently empty, so readiness
reporting includes group membership and resolves numeric GIDs at runtime.

A neural processor is discovered and reported as a grantable device capability.
Vaelor's built-in inference backends target the CPU and GPU; NPU inference runs
through a separately supervised runtime (FastFlowLM). On the HP Z2 Mini this is
live — the Assistant is served on the NPU and AI Chat on the GPU — while a host
without that runtime configured gets accelerator discovery and telemetry only.
Note that `/dev/accel/*` is owned by the same `render` group as the GPU nodes,
so group membership cannot separate NPU access from GPU access; only per-device
passthrough can.

Model-fit sizing uses the accelerator's VRAM carve-out plus, on a unified part,
its GTT aperture — clamped by what the host can spare, because GTT is system
RAM. The CPU-only ladder still stops at 8B, because CPU generation above that
is too slow to be a usable assistant.

On a unified part the size of that aperture is a machine setting: the
kernel gives the GPU about half of visible memory, and an administrator may
raise it per machine, in whole GiB, up to the ceiling the fit prints. The
setting exists only where discovery finds the kernel limit and a GPU that
shares system memory; a discrete card, a Raspberry Pi and a machine whose
kernel does not report the limit have none and say why. A new size counts
from the machine's next restart, which only its owner starts. Rebuilding the
boot image needs `update-initramfs` (Debian and Ubuntu); a machine without it
is refused before anything is written.

### GPU serving prerequisites

GPU serving runs in containers, so the host needs no ROCm of its own — each
image carries the ROCm it was built against. What it does need:

- **Docker running**, with its content store initialised. The installer proves
  this before it finishes rather than assuming a started daemon is a ready one.
- **`/dev/kfd` and `/dev/dri`**, passed into the container by the root hardware
  bridge. The `render` and `video` group ids are resolved numerically at launch
  and never assumed — a container has no host group database, so a name would
  not resolve inside it.
- **A gfx1151 accelerator.** The capability gate opens on a gfx1151 GPU plus a
  usable Docker, and on nothing else: a host binary or a host ROCm library is
  reported for information only and its absence changes no verdict.

The images the installer pre-pulls (skip with `--skip-image-pull`). Sizes are
measured on the appliance; each reference is pinned by digest in the module that
launches it, so the box that installs and the box that redeploys a year later
run the same engine:

| Image | Registry | Approximate size | Pulled on | Serves |
| --- | --- | --- | --- | --- |
| `docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-10.0` | Docker Hub | 1.4 GB | a gfx1151/gfx1150 GPU | stock GGUF models (llama.cpp) |
| `ghcr.io/julianmb/q38rocm:latest` | GitHub Container Registry | 3.3 GB | a gfx1151/gfx1150 GPU | the FP4 27B (a llama.cpp fork) |
| `nginx:stable-alpine` | Docker Hub | 0.03 GB | any box with Docker | the LAN LLM Server auth proxy, and the cluster's balancer and gates |
| `arizephoenix/phoenix:version-20.9.0` | Docker Hub | 1.5 GB | any box with Docker | the loopback-only request-trace collector |

The two GPU images are gfx1151 builds, so the pre-pull reads the host's own
`gfx_target_version` from the KFD topology — the same signal the node inventory
carries — and skips them, saying so, on any other AMD part. It also skips them,
saying so, when the Docker data root has less free space than the missing images
plus 2 GB of headroom for unpacking.

### Clustering prerequisites

Serving one model across more than one GPU machine is a separate mode, enabled
from the console after install. **Install Vaelor on the controller only.** Every
other machine joins as a **GPU worker** from Cluster › Add machine and is
provisioned and kept current by the controller; it runs no console and no
Assistant of its own. The full installer refuses to run on a machine that is a
worker (`--leave-worker-role` takes the worker software off when its controller
is gone for good). A machine that already carries the full appliance can be
enrolled, and its card says so; it is not converted to the slim worker install
in place in this release.

A machine joining as a worker needs:

- **The same platform as the controller.** Clusters are architecture-homogeneous,
  and GPU serving is verified only on gfx1151 (Strix Halo). A worker of another
  architecture is drained and removed rather than kept.
- **An SSH login that is a sudoer.** The controller writes systemd units, installs
  packages, and runs containers on the worker as root over that login. The sudo
  password never reaches the worker's files or programs.
- **Docker.** The enrolment flow installs it if it is missing.
- **What the controller installs:** a telemetry agent (a pinned Telegraf build the
  controller's installer staged), a GPU sampler, and AMD's `amd-smi` at a pinned
  version from AMD's repository — not the full ROCm stack. When AMD no longer
  publishes that version, the newest available one is installed and the console
  says which version runs. The controller re-checks a worker's software every 15
  minutes and after its own update brings each worker to its version.
- **`/dev/kfd` and `/dev/dri`,** as above.
- **Registry reachability for the vLLM image, on every GPU machine, at deploy
  time.** The image is pulled per machine as part of the deploy, not at install,
  and a machine that cannot reach the registry fails the deploy up front:

  | vLLM image | Registry | Approximate size |
  | --- | --- | --- |
  | vLLM 0.27 on ROCm 10 (`rocm/vllm`, the default) | Docker Hub | 27 GB |
  | vLLM 0.22 (`ryai-vllm`, the earlier image, still selectable) | `oci-registry.ryai.dev` | 26 GB |

  The image pull reports one coarse step, not progress; the byte-accurate progress
  a cluster deploy shows is the model **weights** download, a separate step with
  its own byte counter. A deployment keeps the image it was deployed on.
- **A usable cluster interface.** A machine's cluster interface is the link that
  carries the address the cluster reaches it on, fixed at enrolment. A machine
  with no such link is recorded with the reason and refused by the GPU deploy,
  not by enrolment.
- **The controller's certificate with its addresses.** Worker telemetry verifies
  the controller's certificate. From 1.5.1 the controller's household
  certificate authority issues it with the controller's addresses, including
  on a controller upgraded from an older release (see `SECURE_ACCESS.md`).

**Splitting one model across machines** additionally needs a **dedicated link**
between them — such as a Thunderbolt cable — chosen in Cluster › Setup, and
**nftables (`nft`) on every machine of the split**. A split with no link chosen,
or with a machine's main LAN card chosen, is refused in plain words. Splitting is
pipeline-parallel by default; tensor-parallel is an owner's choice and recommends
the fastest link the machines share. *Splitting was proven on two machines before
the link fence below was added; the fenced version and the tensor-parallel option
have not yet been re-run live on two machines.* Running one copy of the model on each machine needs no
dedicated link.

### Ports

| Port | Bind | What |
| --- | --- | --- |
| 34001 | LAN | the console and `/api/v2`, over HTTPS; workers also send telemetry here, keyed |
| 8080–8099 | loopback | the single-node model server (AI Chat) |
| 11434 | LAN | the LLM Server auth proxy, `Bearer`-keyed |
| 8000–8079 | loopback on every machine | the cluster vLLM API; a worker's is reached only through a keyed gate on its cluster address |
| 6006 | loopback | the Arize Phoenix trace collector (reach it over SSH) |
| 6379–6381, 10001, 26000–26299 | the dedicated cluster link | a split's Ray, RCCL and Gloo traffic, fenced as described below |

**The cluster vLLM API binds loopback on every machine.** A controller-led
deployment is fronted by the LLM Server auth proxy. A worker's replica, and a
deployment led by a worker, are fronted by a keyed nginx gate on that worker's
cluster address, and the model answers only callers holding the cluster key. A
worker-led split deployed by a build older than 1.5 stays LAN-open until it is
Loaded again, and the console labels it so.

**A model split across machines needs a dedicated cluster link and nftables.**
Without the measures below, anyone who can reach a split's Ray ports can run code as
root on every machine of the split. A split is therefore served only over a dedicated
link between its machines, and only where every machine has nftables. On that link a
Vaelor-owned firewall table lets only the split's machines in; Ray's own ports (6379-6381,
10001, 26000-26299) answer nobody else on any other interface; and Ray requires the
deployment's own token, changed at every Load. RCCL, Gloo and the TCPStore have no
authentication of their own; the fence is their guard. No Vaelor service port is in the
fenced band.

Apart from that Vaelor-owned table for a split, the installer configures **no
firewall** — it adds no `ufw`, `iptables`, `nftables` or `firewalld` rule and removes
none, so a host firewall stays entirely the operator's. If one is enabled on a GPU
cluster member, the cluster traffic between the machines must be permitted or the
collective hangs rather than failing cleanly.

## Untested platforms

Vaelor has been run on AMD Strix Halo machines (the HP Z2 Mini G1a and HP ZBook
Ultra G1a) and on the Raspberry Pi. **No NVIDIA, Intel, other AMD GPU, or
CPU-only x86 machine has been tested.** On such a machine:

- GPU model serving and GPU clustering are not offered; the GPU serving gate opens
  only on a gfx1151 GPU with a usable Docker. AI Chat can still use a server on
  your network or a hosted service.
- Accelerator identity and telemetry are reported where the kernel exposes them
  without vendor tools, and anything that cannot be read is reported as absent,
  with a reason. Some verdicts are known to be worded for AMD hardware: for
  example, an NVIDIA or Intel GPU may be described as not found, and a GPU whose
  memory could not be read may be described as having dedicated memory.
- The installer fetches AMD's NPU runtime (FastFlowLM) and the NPU Assistant
  model whenever any neural-accelerator device is present, including a non-AMD
  one, where they will not run. Pass `--without-npu-model` to skip the model
  download on such a machine.

Reports from other hardware are welcome; see `SUPPORT.md`.

## Pironman enclosure matrix

The app can automatically detect a product from installed variant data and peripherals. If detection is inconclusive, Settings lets the user choose a model. That choice controls labels, product artwork, and feature discovery; it never fabricates telemetry for hardware that is absent. The choice is refused server-side, with HTTP 409, on a machine where no enclosure was discovered.

Supported product profiles:

- Pironman 5
- Pironman 5 Max
- Pironman 5 Mini
- Pironman 5 Pro Max

This list follows SunFounder’s published Pironman 5 series catalog. PiPower is
an optional power/UPS accessory: when its sensors are present, it enriches the
selected enclosure with voltage, current, wattage, charging, and battery data.
It does not replace the enclosure identity. Likewise, NAS describes a storage
workload, not another enclosure model. Unreleased or preview hardware is not
offered until SunFounder publishes a stable product profile and specifications.

## What runtime detection changes

- CPU and RAM determine whether local AI installation is offered and the maximum recommended model size.
- Desktop services determine whether browser remote desktop can be configured.
- Docker, Compose, and the host package manager are checked before their controls are enabled.
- When Docker is already present, setup adopts it without reinstalling it. When it is absent on a validated Debian-family host, an administrator can install Docker Engine and Compose with one approved action. Appliance operating systems are never modified by that generic installer.
- OLED, RGB, case-fan, CPU-fan, storage, power, and battery controls appear only when the selected enclosure supports them and discovery confirms the required interface.
- Voltage, throttling, and battery data are shown only when the Raspberry Pi PMIC or PiPower hardware reports them.

## Upstream references

- [SunFounder Pironman 5 setup documentation](https://docs.sunfounder.com/projects/pironman5/en/latest/pironman5/set_up/set_up_pironman5.html)
- [SunFounder Pironman 5 compatible systems](https://github.com/sunfounder/pironman5#compatible-systems)
- [Home Assistant OS setup](https://docs.sunfounder.com/projects/pironman5/en/latest/pironman5/set_up/set_up_home_assistant.html)
- [Pironman 5 documentation](https://docs.sunfounder.com/projects/pironman5/en/latest/)

These links describe upstream enclosure compatibility. This document remains the authority for which expanded Control Plane features this repository enables on each OS.
