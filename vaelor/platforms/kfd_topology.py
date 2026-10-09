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

It reports what it read and never guesses: no topology, an unreadable file
or only zero-valued nodes are all ``""``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List

#: The reader, as the text both machines run. Standard library only, no name
#: from outside itself but ``glob``, which it imports: it has to stand alone
#: inside the worker probe's ``python3 -c`` program.
GFX_READER_SOURCE = (
    "def gfx_target_version(sys_root='/sys'):\n"
    "    import glob\n"
    "    pattern = sys_root + '/class/kfd/kfd/topology/nodes/*/properties'\n"
    "    for prop in sorted(glob.glob(pattern)):\n"
    "        try:\n"
    "            with open(prop) as handle:\n"
    "                for line in handle:\n"
    "                    if line.startswith('gfx_target_version'):\n"
    "                        value = line.split()[1]\n"
    "                        if value and value != '0':\n"
    "                            return value\n"
    "        except (OSError, IndexError):\n"
    "            continue\n"
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
    not hold the value; this adds it from the topology, as the worker probe's
    answer carries it beside its cards. A record that already has one keeps
    it, and a machine with no topology is returned as it came.
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
