# Supported platforms

This document describes Vaelor compatibility. It is intentionally more
specific than “the Pironman hardware can run on this OS”: some appliance
operating systems support an enclosure but cannot provide a general-purpose
Docker host, package manager, desktop, or local AI runtime.

Last reviewed: 2026-08-28

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
| Ubuntu 26.04 AMD64 workstation (HP Z2 Mini G1a) | Compatible (hardware probed, package not yet accepted) | Absent with a reason | Full | On-device: Assistant on the NPU, AI Chat on the GPU | Desktop edition only | Full |
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
- The commissioned Ubuntu 26.04 Raspberry Pi completed the versioned
  Pironman-to-Vaelor 2.0.4 migration on 2026-07-30. Ten Vaelor services,
  HTTPS health, encrypted credentials, legacy aliases, and the existing Qwen
  model were verified after migration. This validates that appliance; it does
  not promote every future Ubuntu 26.04 build to the Verified support level.
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
