"""Hardware-neutral primitives shared by every Vaelor platform driver.

Nothing in this module may assume a Raspberry Pi, a Pironman enclosure, an
accelerator, or an architecture. Platform-specific facts belong in the driver
modules that import from here.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

#: A hwmon PWM control node, e.g. ``pwm1``. Excludes ``pwm1_enable`` and the
#: other attribute files that share the prefix.
PWM_NODE = re.compile(r"pwm\d+")


Snapshot = Dict[str, Any]

#: A machine class describes the *shape* of the host, not its vendor. It is the
#: single fact the UI needs in order to decide which control surfaces are worth
#: rendering at all.
PI_APPLIANCE = "pi-appliance"
WORKSTATION = "workstation"
GENERIC = "generic"
MACHINE_CLASSES = (PI_APPLIANCE, WORKSTATION, GENERIC)

#: Every capability a driver must answer for. Answering with a bare ``False``
#: is not enough: the operator needs to know *why* their machine cannot do it.
CAPABILITY_KEYS = (
    "case_fan",
    "case_lighting",
    "oled",
    "cpu_fan",
    # Reading a fan's speed is a different capability from setting it, and
    # merging them told a workstation with three readable fans that it had
    # none. `cpu_fan` and `case_fan` remain about control; this is about
    # whether any tachometer can be read at all.
    "fan_readings",
    "battery",
    "gpu",
    "npu",
)

GENERIC_PRODUCT: Dict[str, Any] = {
    "id": "generic",
    "name": "Generic computer",
    "icon": "generic",
    "fan_count": 0,
    "nvme_slots": 0,
    "capabilities": [],
    "detected_from": "host",
    "confident": True,
}

# SMBIOS chassis types worth naming. Anything else is reported numerically
# rather than guessed at.
CHASSIS_TYPES = {
    "3": "desktop",
    "4": "low-profile desktop",
    "5": "pizza box",
    "6": "mini tower",
    "7": "tower",
    "8": "portable",
    "9": "laptop",
    "10": "notebook",
    "13": "all-in-one",
    "15": "space-saving",
    "16": "lunch box",
    "17": "main server chassis",
    "23": "rack mount chassis",
    "30": "tablet",
    "31": "convertible",
    "32": "detachable",
    "35": "mini pc",
    "36": "stick pc",
}


def text(path: Path) -> str:
    """Read a sysfs/procfs style file, tolerating absence and NUL padding."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except OSError:
        return ""


def read_number(path: Path) -> Optional[int]:
    raw = text(path)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def capability(available: bool, reason: str = "") -> Dict[str, Any]:
    """Build one capability record.

    ``reason`` is user-facing prose explaining why the capability is missing on
    *this* machine, and is ``None`` when the capability is present. A capability
    that is unavailable without a reason is a bug: the UI has nothing to show
    and the operator is left guessing.
    """
    if available:
        return {"available": True, "reason": None}
    return {"available": False, "reason": reason or "This machine does not provide this control."}


def read_os_release(path: str = "/etc/os-release") -> Dict[str, Any]:
    values: Dict[str, str] = {}
    for line in text(Path(path)).splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip("\"'")
    os_id = values.get("ID", platform.system().lower() or "unknown").lower()
    family = "debian" if os_id in {"debian", "ubuntu", "raspbian", "kali"} else os_id
    if os_id in {"raspbian", "debian"}:
        support_level = "verified"
        support_label = "Verified"
        support_note = "Full Vaelor support with the Pironman enclosure adapter on Raspberry Pi OS and Debian."
    elif os_id in {"ubuntu", "kali"}:
        support_level = "compatible"
        support_label = "Compatible"
        support_note = "Core hardware and control-plane features are compatible; release-specific desktop features are validated separately."
    elif os_id in {"homeassistant", "hassos", "batocera", "umbrel", "homebridge"}:
        support_level = "limited"
        support_label = "Limited"
        support_note = "The Pironman enclosure adapter is available, but this appliance OS does not expose every Vaelor workload, update, desktop, or agent feature."
    else:
        support_level = "unknown"
        support_label = "Not validated"
        support_note = "This operating system is detected but is not in the published Vaelor compatibility matrix."
    return {
        "id": os_id,
        "name": values.get("PRETTY_NAME") or values.get("NAME") or platform.system(),
        "version": values.get("VERSION_ID", ""),
        "family": family,
        "supported": support_level in {"verified", "compatible", "limited"},
        "support_level": support_level,
        "support_label": support_label,
        "support_note": support_note,
    }


def _cpuinfo_blocks(cpuinfo_path: str) -> List[Dict[str, str]]:
    """``/proc/cpuinfo`` as one dictionary per logical processor.

    Read per block rather than flattened, because the fields that answer "how
    many cores" are per-socket and a flattened read keeps only the first
    processor's copy of them.
    """
    blocks: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    for line in text(Path(cpuinfo_path)).splitlines():
        if ":" not in line:
            if current:
                blocks.append(current)
                current = {}
            continue
        key, value = line.split(":", 1)
        current[key.strip().lower()] = value.strip()
    if current:
        blocks.append(current)
    return blocks


def _sysfs_topology(sys_root: str) -> List[Dict[str, str]]:
    """``core_id``, ``physical_package_id`` and thread siblings per CPU.

    This is where an ARM host answers the question: a Raspberry Pi's
    ``/proc/cpuinfo`` has no ``cpu cores`` and no ``physical id`` at all, so a
    reader that only knows the x86 fields has nothing on the appliance.
    """
    records: List[Dict[str, str]] = []
    try:
        entries = sorted((Path(sys_root) / "devices/system/cpu").iterdir())
    except OSError:
        return []
    for entry in entries:
        if not re.fullmatch(r"cpu\d+", entry.name):
            continue
        topology = entry / "topology"
        records.append({
            "cpu": entry.name,
            "core_id": text(topology / "core_id").strip(),
            "package": text(topology / "physical_package_id").strip(),
            "thread_siblings": text(topology / "thread_siblings_list").strip(),
        })
    return records


def _siblings_per_core(records: List[Dict[str, str]]) -> Optional[int]:
    """Threads sharing one core, from ``thread_siblings_list``.

    ``0-1`` is one core with two threads; ``3`` is one core with one. Used only
    when the core ids themselves are missing, because counting distinct cores
    is a reading and dividing by this is an inference.
    """
    for record in records:
        raw = record.get("thread_siblings") or ""
        count = 0
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                low, _, high = part.partition("-")
                try:
                    count += int(high) - int(low) + 1
                except ValueError:
                    return None
            else:
                count += 1
        if count:
            return count
    return None


def cpu_topology(
    cpuinfo_path: str = "/proc/cpuinfo",
    sys_root: str = "/sys",
) -> Dict[str, Any]:
    """Physical cores and logical CPUs, told apart, with their source.

    ``os.cpu_count()`` counts **logical** CPUs. On an SMT part that is threads,
    and this codebase published it as ``cpu_cores`` and printed it to the
    Assistant as *"32 cores"* on a 16-core processor — in every answer about the
    machine. It survived because the appliance it was written on is a Raspberry
    Pi: 4 cores, no SMT, so the wrong field happened to hold the right number.

    Cores are **read**, not divided, wherever the machine says so:

    1. ``/proc/cpuinfo``'s ``cpu cores`` per distinct ``physical id`` — what
       ``lscpu`` prints as ``Core(s) per socket`` × ``Socket(s)``.
    2. ``/sys/devices/system/cpu/cpu*/topology`` — distinct
       ``(physical_package_id, core_id)`` pairs. This is the ARM answer, where
       the x86 cpuinfo fields do not exist.
    3. Only if neither publishes core identity: logical CPUs divided by the
       threads sharing one core. That is an inference, ``derived`` says so, and
       ``reason`` says so in prose.

    When nothing establishes it, ``cores`` is ``None``. A count that cannot be
    read is not a count to be guessed at — VD-005 applied to a number.
    """
    blocks = _cpuinfo_blocks(cpuinfo_path)
    processors = [block for block in blocks if "processor" in block]
    records = _sysfs_topology(sys_root)
    logical = len(processors) or len(records) or (os.cpu_count() or 1)
    result: Dict[str, Any] = {
        "logical": logical,
        "cores": None,
        "sockets": None,
        "threads_per_core": None,
        "source": "logical-only",
        "derived": False,
        "reason": "",
    }
    per_socket: Dict[str, int] = {}
    for index, block in enumerate(processors):
        raw = block.get("cpu cores")
        if not raw:
            continue
        try:
            count = int(raw)
        except ValueError:
            continue
        # A single-socket part often omits `physical id`. Keyed by the index
        # only when it does, so two sockets are never collapsed into one.
        per_socket[block.get("physical id", str(index) if len(processors) == 1 else "0")] = count
    if per_socket:
        cores = sum(per_socket.values())
        result.update({
            "cores": cores,
            "sockets": len(per_socket),
            "threads_per_core": round(logical / cores, 2) if cores else None,
            "source": "proc-cpuinfo",
        })
        return result
    pairs = {
        (record["package"], record["core_id"])
        for record in records
        if record.get("core_id")
    }
    if pairs:
        cores = len(pairs)
        result.update({
            "cores": cores,
            "sockets": len({package for package, _ in pairs}) or 1,
            "threads_per_core": round(logical / cores, 2) if cores else None,
            "source": "sysfs-topology",
        })
        return result
    siblings = _siblings_per_core(records)
    if siblings and siblings > 0 and logical % siblings == 0:
        result.update({
            "cores": logical // siblings,
            "threads_per_core": float(siblings),
            "source": "derived-by-division",
            "derived": True,
            "reason": (
                "Neither /proc/cpuinfo nor the kernel's CPU topology published "
                "core identities on this host, so the core count is {} logical "
                "CPUs divided by the {} threads sharing a core rather than a "
                "figure that was read."
            ).format(logical, siblings),
        })
        return result
    result["reason"] = (
        "This host publishes no core count: /proc/cpuinfo carries neither "
        "'cpu cores' nor 'physical id' and the kernel's CPU topology could not "
        "be read, so only the {} logical CPUs are known. Whether those are "
        "cores or threads is not established here."
    ).format(logical)
    return result


def dmi_identity(dmi_root: str = "/sys/class/dmi/id") -> Dict[str, Any]:
    """Read SMBIOS identity, which is how an x86 host says what it is.

    Device-tree hosts have no DMI and return empty strings here; the caller
    treats that as "no DMI identity", not as an error.
    """
    root = Path(dmi_root)
    vendor = text(root / "sys_vendor")
    product = text(root / "product_name")
    board = text(root / "board_name")
    chassis_code = text(root / "chassis_type")
    placeholders = {
        "", "to be filled by o.e.m.", "system manufacturer", "default string",
        "system product name", "none", "not specified", "not applicable",
    }
    if vendor.lower() in placeholders:
        vendor = ""
    if product.lower() in placeholders:
        product = ""
    if board.lower() in placeholders:
        board = ""
    return {
        "vendor": vendor,
        "product": product,
        "board": board,
        "chassis_type": chassis_code,
        "chassis": CHASSIS_TYPES.get(chassis_code, ""),
        "present": bool(vendor or product),
    }


def dmi_model_name(dmi: Dict[str, Any]) -> str:
    """Compose a human product name without repeating the vendor."""
    vendor = str(dmi.get("vendor") or "").strip()
    product = str(dmi.get("product") or "").strip()
    if not product:
        return vendor
    if vendor and not product.lower().startswith(vendor.lower()):
        return f"{vendor} {product}"
    return product


def board_info(
    model_path: str = "/proc/device-tree/model",
    cpuinfo_path: str = "/proc/cpuinfo",
    dmi_root: str = "/sys/class/dmi/id",
    sys_root: str = "/sys",
) -> Dict[str, Any]:
    """Identify the board across device-tree and SMBIOS hosts.

    Before the platform seam existed this read device-tree only, so every x86
    host fell back to its hostname and reported itself as an unknown computer.

    ``cpu_cores`` means **cores**. It used to mean ``os.cpu_count()``, which is
    threads, and every surface downstream printed it as cores. Where the core
    count cannot be established the field still carries the logical count —
    a number is more useful than a gap — but ``cpu_cores_are_threads`` says so,
    and nothing may print it as cores without checking that flag.
    """
    model = text(Path(model_path))
    cpuinfo = text(Path(cpuinfo_path))
    fields: Dict[str, str] = {}
    for line in cpuinfo.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields.setdefault(key.strip().lower(), value.strip())
    cpu_model = fields.get("model name") or platform.processor() or "unknown"
    revision = fields.get("revision", "")
    is_pi = (
        "raspberry pi" in model.lower()
        or "raspberry pi" in fields.get("hardware", "").lower()
    )
    dmi = dmi_identity(dmi_root)
    resolved = model or dmi_model_name(dmi) or platform.node() or "Unknown computer"
    topology = cpu_topology(cpuinfo_path, sys_root)
    return {
        "model": resolved,
        "cpu_model": cpu_model,
        "revision": revision,
        "architecture": platform.machine() or "unknown",
        "cpu_cores": topology["cores"] or topology["logical"],
        "cpu_cores_are_threads": topology["cores"] is None,
        "cpu_threads": topology["logical"],
        "cpu_sockets": topology["sockets"],
        "cpu_threads_per_core": topology["threads_per_core"],
        "cpu_topology_source": topology["source"],
        "cpu_topology_derived": topology["derived"],
        "cpu_topology_reason": topology["reason"],
        "is_raspberry_pi": is_pi,
        "vendor": dmi["vendor"],
        "chassis": dmi["chassis"],
        "firmware_identity": "device-tree" if model else "smbios" if dmi["present"] else "none",
    }


def battery_supplies(sys_root: str = "/sys") -> List[Dict[str, Any]]:
    """Batteries the kernel is reporting, read from ``/sys/class/power_supply``.

    This is the file every laptop publishes and that nothing in Vaelor read.
    Battery was hardcoded absent on every SMBIOS host — correct for a mini PC
    on mains, wrong for a laptop, and asserted either way without looking. The
    same false-absence pattern has now been wrong four times in this codebase,
    so this looks.
    """
    supplies = []
    try:
        entries = sorted((Path(sys_root) / "class" / "power_supply").iterdir())
    except OSError:
        return []
    for entry in entries:
        if text(entry / "type").strip().lower() != "battery":
            continue
        capacity = read_number(entry / "capacity")
        supplies.append({
            "name": entry.name,
            "percentage": int(capacity) if capacity is not None else None,
            "status": text(entry / "status").strip() or None,
            "voltage": read_number(entry / "voltage_now"),
            "current": read_number(entry / "current_now"),
        })
    return supplies


def controllable_fans(sys_root: str = "/sys") -> List[str]:
    """hwmon PWM nodes this host exposes, if any.

    CPU-fan control was hardcoded absent on every x86 machine with no probe at
    all. A workstation whose vendor keeps fan control in the embedded
    controller genuinely has none — but that is a thing to discover, not to
    assume, and a board that does expose ``pwm1`` was being told it did not.
    """
    from ..sensor_facts import is_prohibited

    found: List[str] = []
    try:
        nodes = sorted((Path(sys_root) / "class" / "hwmon").iterdir())
    except OSError:
        return []
    for node in nodes:
        try:
            for entry in sorted(node.iterdir()):
                # `PWM_NODE` already excludes attribute files, but the
                # prohibition is checked explicitly rather than left to depend
                # on a regex someone might later relax. `pwm1_enable` is not a
                # control node: writing 0 pins this machine's fans to maximum
                # for 90 seconds and writing 1 is refused outright.
                if is_prohibited(entry.name):
                    continue
                if PWM_NODE.fullmatch(entry.name):
                    found.append("{}/{}".format(node.name, entry.name))
        except OSError:
            continue
    return found


def thermal_policy(machine_class: str) -> Dict[str, Any]:
    """Health thresholds for CPU temperature, per machine class.

    70/80 °C is a Raspberry Pi constant: the SoC soft-throttles at 80 °C and
    the enclosure fan map tops out just below it. A desktop AMD or Intel part
    boosts into the mid-nineties by design and only throttles at Tjmax, so
    reusing the Pi numbers pins a workstation in "Attention" forever. These are
    the temperatures at which an operator should actually act.
    """
    if machine_class == PI_APPLIANCE:
        return {
            "warning_c": 70.0,
            "critical_c": 80.0,
            "rationale": "The Raspberry Pi SoC soft-throttles at 80 °C.",
        }
    return {
        "warning_c": 97.0,
        "critical_c": 100.0,
        "rationale": (
            "Desktop and workstation processors boost into the mid-nineties by "
            "design and only throttle at their junction limit."
        ),
    }


class HardwarePlatformBase:
    """Common driver behaviour: identity, OS, and feature policy.

    Subclasses own the enclosure, the power source, and the capability reasons.
    """

    machine_class = GENERIC

    def __init__(
        self,
        *,
        model_path: str = "/proc/device-tree/model",
        cpuinfo_path: str = "/proc/cpuinfo",
        dmi_root: str = "/sys/class/dmi/id",
        sys_root: str = "/sys",
        runner: Callable[..., Any] = subprocess.run,
    ):
        self._model_path = model_path
        self._cpuinfo_path = cpuinfo_path
        self._dmi_root = dmi_root
        # On the base because boot forensics are not Pi-specific: every host
        # that runs systemd can be asked how its previous boot ended, and an
        # x86 workstation loses power the same way an appliance does.
        self._runner = runner
        # Held on the base rather than on each driver because the CPU topology
        # is read from it on every host: the x86 cpuinfo fields do not exist on
        # ARM, and the kernel's topology directory is where the appliance's own
        # core count comes from.
        self._sys_root = sys_root

    # -- identity ---------------------------------------------------------
    def board(self) -> Dict[str, Any]:
        return board_info(
            self._model_path, self._cpuinfo_path, self._dmi_root, self._sys_root
        )

    def operating_system(self) -> Dict[str, Any]:
        return read_os_release()

    # -- driver surface ---------------------------------------------------
    def product(
        self, raw: Optional[Snapshot] = None, board: Optional[Snapshot] = None
    ) -> Dict[str, Any]:
        """Identify the enclosure. ``board`` is read when not supplied.

        One public name with one signature: a caller that has board facts
        passes them, and a route that does not simply calls ``product(raw)``.
        """
        return dict(GENERIC_PRODUCT)

    def resolve_product(self, raw: Optional[Snapshot] = None) -> Dict[str, Any]:
        """Identify the enclosure for a caller that has no board facts.

        Every surface that needs a product — ``/device``, ``/fans``, the
        Assistant — goes through this, so they cannot disagree about how many
        case fans the machine has.
        """
        return self.product(raw, self.board())

    def power(
        self, metrics: Optional[Snapshot], board: Snapshot
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def capabilities(
        self,
        raw: Optional[Snapshot] = None,
        metrics: Optional[Snapshot] = None,
        inventory: Optional[Snapshot] = None,
    ) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def feature_policy(
        self, board: Snapshot, inventory: Snapshot
    ) -> Dict[str, Any]:
        from ..model_sizing import feature_policy as _policy

        return _policy(board, inventory, self.accelerators())

    def accelerators(self) -> list[Dict[str, Any]]:
        return []

    def memory_split(self, inventory: Optional[Snapshot] = None) -> Dict[str, Any]:
        """How this machine's RAM is divided, and whether an owner set it.

        On the base because it is not accelerator-specific: a machine with no
        carve-out at all still has to answer *"is the figure you showed me the
        whole memory?"*, and "no reservation could be read" is that answer.
        """
        from ..memory_split import graphics_memory_split
        from .accelerators import configured_graphics_memory

        return graphics_memory_split(
            (inventory or {}).get("memory_total_bytes"),
            self.accelerators(),
            configured_graphics_memory(self._sys_root),
        )

    def previous_shutdown(self) -> Dict[str, Any]:
        """Whether the last boot ended in a shutdown or simply stopped.

        Read once per process and published on the power payload of every
        platform. An appliance that cannot say it lost power leaves its owner
        to find out by walking past a dark machine, which is how the incident
        this answers was actually discovered.
        """
        from ..boot_forensics import cached_previous_shutdown

        return cached_previous_shutdown(self._runner)

    def neural_accelerators(self) -> list[Dict[str, Any]]:
        return []

    def graphics(self) -> Dict[str, Any]:
        """Static graphics inventory. Empty on a platform with no adapter."""
        return {
            "adapters": [],
            "neural_accelerators": [],
            "rocm_version": None,
            "mesa_version": None,
        }

    def power_actions(self) -> Dict[str, Dict[str, Any]]:
        """Host power actions this platform can perform.

        Every host that runs systemd can reboot and power off; that is not a
        driver-specific fact and so it is the base behaviour. A platform
        overrides this only when it has a *better* mechanism, such as an
        enclosure that must sequence its own power rails first.
        """
        from ..host_power import systemd_power_actions

        return systemd_power_actions()

    def power_telemetry(self) -> bool:
        """Whether this platform can report any power reading at all.

        The generic host power controller used to answer this by asking
        ``shutil.which("vcgencmd")`` — a Broadcom firmware tool — on every
        machine. The platform driver is the only thing that knows.
        """
        return False

    def thermal_policy(self) -> Dict[str, Any]:
        return thermal_policy(self.machine_class)

    # -- composed payloads ------------------------------------------------
    def snapshot(
        self,
        raw: Optional[Snapshot],
        metrics: Optional[Snapshot],
        inventory: Optional[Snapshot],
    ) -> Snapshot:
        board = self.board()
        inventory = inventory or {}
        board["memory_total_bytes"] = int(
            inventory.get("memory_total_bytes", 0) or 0
        )
        product = self.product(raw, board)
        return {
            "product": product,
            "board": board,
            "os": self.operating_system(),
            # What the OS can see is not what is fitted, and the difference is
            # a setting rather than a fact about the part. Published beside the
            # board so no surface has to infer a total from `memory_total`.
            "memory_split": self.memory_split(inventory),
            "power": self.power(metrics, board),
            "feature_policy": self.feature_policy(board, inventory),
            "machine_class": self.machine_class,
            "accelerators": self.accelerators(),
            "neural_accelerators": self.neural_accelerators(),
            "graphics": self.graphics(),
            "thermal_policy": self.thermal_policy(),
        }

    def machine(
        self,
        raw: Optional[Snapshot] = None,
        metrics: Optional[Snapshot] = None,
        inventory: Optional[Snapshot] = None,
    ) -> Snapshot:
        """The ``GET /api/v2/system/machine`` contract."""
        board = self.board()
        return {
            "machine_class": self.machine_class,
            "product": self.product(raw, board),
            "capabilities": self.capabilities(raw, metrics, inventory),
        }
