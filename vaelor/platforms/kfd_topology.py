"""Which GPU family this machine has, read from the kernel's KFD topology.

The AMD compute driver publishes each node's ``gfx_target_version`` under
``/sys/class/kfd/kfd/topology/nodes/<n>/properties`` - ``110501`` for gfx1151
(the Radeon 8060S), ``0`` for the CPU node. It is the one machine fact that
says which GPU FAMILY a box has without reading a product name, and the
cluster's vLLM launch reads it (`vllm_images.is_rdna3_family`): a hybrid
model is started with a bf16 KV cache only on a family without FP8 hardware.

**One derivation, two readers.** A worker is read over SSH by a Python
program sent as text (`ssh_transport._GPU_PROBE`); the controller reads its
own sysfs in-process. Until this module only the worker probe read the
topology: the controller's accelerator inventory
(`platforms.accelerators._card_record`) never carried the value, so the
controller's GPU family was always blank in production while the test
fixtures wrote one in by hand - and a launch that needs every machine's
family could never get its settings on any cluster the controller took part
in. So the reader is ONE piece of source text, :data:`GFX_READER_SOURCE`:
the worker probe embeds it verbatim, and :func:`gfx_target_version` here is
that same text compiled. There is no second spelling to drift.

**Second source: the card's graphics-core version (VD-216, R92).** The
workload executor runs ``PrivateDevices=yes``, and amdgpu filters a GPU
node's topology file by the device cgroup, so inside that sandbox the GPU
node's ``properties`` gives ``Operation not permitted`` and the topology
reads as nothing. The card's IP discovery table
(``/sys/class/drm/cardN/device/ip_discovery/die/0/GC/0/{major,minor,revision}``)
stays readable there. It is turned into the topology's number through
:data:`GC_TARGET_VERSIONS`, the kernel's own table, and NOT through
``major*10000 + minor*100 + revision``: that formula holds for 11.5.1 but
the kernel maps 11.0.1 to 110003, 11.0.3 to 110001, 10.1.10 to 100100 and
12.1.0 to 120500. A core version missing from the table reads as nothing.
The topology is read first and wins whenever it gives a value, and the core
version is consulted ONLY when a topology node's file could not be read
(R93): a topology that is absent, or lists only zero-valued nodes, says
the compute driver has no GPU here, and a card's core version must not
invent one.

It reports what it read and never guesses: no topology, only zero-valued
nodes, a refused node with no readable core version, or a core version the
table does not hold are all ``""``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List

#: The kernel's graphics-core version -> ``gfx_target_version`` table, as
#: ``kgd2kfd_probe`` in ``drivers/gpu/drm/amd/amdkfd/kfd_device.c`` assigns
#: it (read at linux master, 2026-10-15). Copied, not computed: several
#: entries do not follow any packing of the three numbers.
GC_TARGET_VERSIONS = {
    "9.0.1": "90000", "9.1.0": "90002", "9.2.2": "90002", "9.2.1": "90004",
    "9.3.0": "90012", "9.4.0": "90006", "9.4.1": "90008", "9.4.2": "90010",
    "9.4.3": "90402", "9.4.4": "90402", "9.5.0": "90500",
    "10.1.10": "100100", "10.1.2": "100101", "10.1.1": "100102",
    "10.1.3": "100103", "10.1.4": "100103",
    "10.3.0": "100300", "10.3.2": "100301", "10.3.1": "100303",
    "10.3.4": "100302", "10.3.5": "100304", "10.3.3": "100305",
    "10.3.6": "100306", "10.3.7": "100306",
    "11.0.0": "110000", "11.0.1": "110003", "11.0.4": "110003",
    "11.0.2": "110002", "11.0.3": "110001",
    "11.5.0": "110500", "11.5.1": "110501", "11.5.2": "110502",
    "11.5.3": "110503", "11.5.4": "110504", "11.5.6": "110504",
    "11.7.0": "110700", "11.7.1": "110701",
    "12.0.0": "120000", "12.0.1": "120001", "12.1.0": "120500",
}

#: The reader, as the text both machines run. Standard library only, no name
#: from outside itself but ``glob`` and ``os``, which it imports, and the
#: table, which is written into the text: it has to stand alone inside the
#: worker probe's ``python3 -c`` program.
GFX_READER_SOURCE = (
    "def gfx_target_version(sys_root='/sys'):\n"
    "    import glob\n"
    "    import os\n"
    "    pattern = sys_root + '/class/kfd/kfd/topology/nodes/*/properties'\n"
    "    refused = False\n"
    "    for prop in sorted(glob.glob(pattern)):\n"
    "        try:\n"
    "            with open(prop) as handle:\n"
    "                for line in handle:\n"
    "                    if line.startswith('gfx_target_version'):\n"
    "                        value = line.split()[1]\n"
    "                        if value and value != '0':\n"
    "                            return value\n"
    "        except OSError:\n"
    "            refused = True\n"
    "        except IndexError:\n"
    "            continue\n"
    "    if not refused:\n"
    "        return ''\n"
    "    gc_targets = " + repr(GC_TARGET_VERSIONS) + "\n"
    "    cards = []\n"
    "    for card in glob.glob(sys_root + '/class/drm/card*'):\n"
    "        number = os.path.basename(card)[4:]\n"
    "        if number.isdigit():\n"
    "            cards.append((int(number), card))\n"
    "    for _, card in sorted(cards):\n"
    "        device = card + '/device'\n"
    "        try:\n"
    "            with open(device + '/vendor') as handle:\n"
    "                if handle.read().strip().lower() != '0x1002':\n"
    "                    continue\n"
    "            parts = []\n"
    "            for name in ('major', 'minor', 'revision'):\n"
    "                with open(device + '/ip_discovery/die/0/GC/0/' + name) as handle:\n"
    "                    parts.append(str(int(handle.read().strip())))\n"
    "        except (OSError, ValueError):\n"
    "            continue\n"
    "        value = gc_targets.get('.'.join(parts), '')\n"
    "        if value:\n"
    "            return value\n"
    "    return ''\n"
)

#: The record key the value travels under, on an accelerator record and in a
#: node's GPU facts alike.
GFX_FIELD = "gfx_target_version"

#: The PCI vendor whose compute driver publishes the topology.
_AMD_VENDOR_ID = "0x1002"

_namespace: Dict[str, Any] = {}
exec(compile(GFX_READER_SOURCE, "<kfd topology reader>", "exec"), _namespace)

#: The controller's reader: :data:`GFX_READER_SOURCE`, compiled.
gfx_target_version: Callable[..., str] = _namespace["gfx_target_version"]


def with_gfx_target_version(
    accelerators: Iterable[Any], sys_root: str = "/sys",
) -> List[Any]:
    """``accelerators`` with each AMD GPU record carrying the topology's value.

    The accelerator inventory is read from the DRM card directories, which do
    not hold the value; this adds it from the topology (or, where a sandbox
    hides the topology, the card's core version), as the worker probe's
    answer carries it beside its cards. A record that already has one keeps
    it, and a machine where neither can be read is returned as it came.
    """
    records = list(accelerators or [])
    value = None
    stamped: List[Any] = []
    for record in records:
        if (
            isinstance(record, dict) and record.get("kind") == "gpu"
            and str(record.get("vendor_id", "")).lower() == _AMD_VENDOR_ID
            and not record.get(GFX_FIELD)
        ):
            if value is None:
                value = str(gfx_target_version(str(sys_root)) or "")
            if value:
                record = {**record, GFX_FIELD: value}
        stamped.append(record)
    return stamped
