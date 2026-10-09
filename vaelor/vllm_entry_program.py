"""The container entry program that shows vLLM its own tuned MoE tables, then execs.

**The defect this works around is vLLM's.** vLLM ships a tuned Triton
fused-MoE table for the Radeon 8060S (gfx1151) as
``E=128,N=768,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json`` in
``vllm/model_executor/layers/fused_moe/configs/``, but builds the name it LOOKS
UP from the platform's device name, which ``vllm/platforms/rocm.py`` maps for
PCI id ``0x1586`` to ``AMD_Radeon_8060S``. So ``get_moe_configs`` asks for
``...device_name=AMD_Radeon_8060S...``, finds nothing and runs the untuned
default ("Using default MoE config"). Read at tags ``v0.22.1`` (commit
``0decac0``) and ``v0.27.0`` (commit ``4bdc8a7``, which still carries both
spellings), and seen in the served image's log on the Z2.

**The fix is vLLM's own mechanism** (LESSONS 15): ``get_moe_configs`` reads
the folder ``VLLM_TUNED_CONFIG_FOLDER`` names FIRST and its packaged folder
second. A container of an MoE model (`vllm_serve_options`) is started through
:data:`ENTRY_SCRIPT` instead of straight into ``vllm`` or ``ray``. Inside the
container, before anything else, it:

1. finds the installed vLLM (``importlib.util.find_spec``) and its packaged
   tables folder;
2. asks vLLM for the device name it will look tables up by - its own
   ``get_device_name_as_file_name`` where the version has one, else the
   platform's ``get_device_name`` with spaces as underscores, which is the
   0.22.1 spelling;
3. symlinks into :data:`TUNED_FOLDER` (a folder of the container's own ``/tmp``)
   every packaged table that names this same GPU another way, under the name
   vLLM computes - never over a table vLLM already finds under that name;
4. ``os.execvp``s the command it was given, so ``vllm`` or ``ray`` is the
   container's process exactly as before.

Every step is no-op-safe: an image whose names agree, a vLLM that cannot be
found or asked, or a folder that cannot be made is reported on stderr and the
command still runs, on vLLM's defaults - the behaviour before this program.
**Nothing is written on any host**: the links live and die with the container,
and point at files inside the image. No part of vLLM travels in Vaelor.

The program is Vaelor's own and is mounted read-only into the container, like
the pull program (`gpu_pull_program`): out of the installed package by the
controller's root bridge, which first proves no account but root can change
it (VD-143), and from a content-hashed file in a worker's store, written over
the worker's SSH like the pull program.
"""

from __future__ import annotations

import hashlib

#: The variable vLLM reads (``vllm/envs.py``) for a user tuned-config folder,
#: and the folder the program fills, inside the container.
TUNED_FOLDER_VARIABLE = "VLLM_TUNED_CONFIG_FOLDER"
TUNED_FOLDER = "/tmp/vaelor-moe"
TUNED_FOLDER_ENV = "{}={}".format(TUNED_FOLDER_VARIABLE, TUNED_FOLDER)

#: Where the program is mounted in every container that runs it.
CONTAINER_ENTRY_PROGRAM = "/opt/vaelor-entry/vaelor_vllm_entry.py"

#: The interpreter that runs it: the image's own Python.
ENTRY_INTERPRETER = "python"

#: The self-contained program. Standard library only; vLLM is imported lazily
#: and every failure falls through to the exec.
ENTRY_SCRIPT = '''"""Show vLLM its packaged tuned MoE tables under the name it looks up, then exec."""
import importlib.util
import os
import sys

FOLDER = os.environ.get("VLLM_TUNED_CONFIG_FOLDER") or "/tmp/vaelor-moe"
PREFIX = "device_name="


def say(message):
    sys.stderr.write("vaelor-entry: " + message + "\\n")
    sys.stderr.flush()


def same_gpu(name):
    """A device name folded so two spellings of one GPU compare equal."""
    folded = name.lower().replace(" ", "_")
    if folded.startswith("amd_"):
        folded = folded[4:]
    if folded.endswith("_graphics"):
        folded = folded[:-9]
    return folded


def device_name():
    """The device name vLLM builds its table file names from."""
    try:
        from vllm.utils.platform_utils import get_device_name_as_file_name
        return get_device_name_as_file_name()
    except ImportError:
        from vllm.platforms import current_platform
        return current_platform.get_device_name().replace(" ", "_")


def link_tables():
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        say("vLLM is not installed here; no tuned MoE tables linked")
        return 0
    configs = os.path.join(
        os.path.dirname(spec.origin), "model_executor", "layers", "fused_moe", "configs",
    )
    device = device_name()
    linked = 0
    for name in sorted(os.listdir(configs)):
        if not name.endswith(".json"):
            continue
        fields = name[:-5].split(",")
        shipped = [field[len(PREFIX):] for field in fields if field.startswith(PREFIX)]
        if len(shipped) != 1 or shipped[0] == device:
            continue
        if same_gpu(shipped[0]) != same_gpu(device):
            continue
        wanted = name.replace(PREFIX + shipped[0], PREFIX + device, 1)
        if os.path.exists(os.path.join(configs, wanted)):
            continue
        os.makedirs(FOLDER, exist_ok=True)
        link = os.path.join(FOLDER, wanted)
        if not os.path.lexists(link):
            os.symlink(os.path.join(configs, name), link)
        linked += 1
    say("{} tuned MoE table(s) linked for {}".format(linked, device))
    return linked


def main(argv):
    if len(argv) < 2:
        say("no command to run")
        return 2
    try:
        link_tables()
    except Exception as error:
        say("tuned MoE tables not linked ({}); vLLM uses its defaults".format(error))
    os.execvp(argv[1], argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''

#: The program's digest, and the content-hashed name a worker's store keeps it
#: under - so a new version is a new file and never truncates one a starting
#: container is reading, exactly as for the pull program.
ENTRY_SCRIPT_DIGEST = hashlib.sha256(ENTRY_SCRIPT.encode("utf-8")).hexdigest()
ENTRY_SCRIPT_NAME = "vaelor_vllm_entry-{}.py".format(ENTRY_SCRIPT_DIGEST[:12])


def write_program_on_worker(transport, target: str, text: str, digest: str) -> str:
    """Put a content-hashed Vaelor program at ``target`` on a WORKER, over SSH.

    ``sha256sum`` (allowlisted) is the cheap presence-and-match probe; only a
    miss or a mismatch writes, so the common case is a read, not a ``tee``
    racing a container that is starting from the file. The pull program and
    this entry program are both shipped this way; the controller never comes
    here, because its bridge mounts both out of the installed package.
    """
    from .ssh_transport import SshTransportError

    try:
        found = str(transport.run(["sha256sum", target])).split()[0]
    except (SshTransportError, IndexError):
        found = ""
    if found != digest:
        transport.run(["tee", target], sudo=True, stdin_text=text)
    return target


def entry_mount(host_program: str) -> list:
    """The ``docker run`` option mounting the program read-only."""
    return ["-v", "{}:{}:ro".format(host_program, CONTAINER_ENTRY_PROGRAM)]


def entry_command(command: list) -> list:
    """``command`` run through the program: ``python <program> <command...>``."""
    return [ENTRY_INTERPRETER, CONTAINER_ENTRY_PROGRAM] + list(command)


if __name__ == "__main__":
    # The controller's containers run THIS file, mounted read-only out of the
    # installed package by the root bridge (VD-143). The file carries the
    # program as text, so running the file runs exactly that text.
    import sys as _sys

    _namespace = {"__name__": "vaelor_vllm_entry"}
    exec(compile(ENTRY_SCRIPT, ENTRY_SCRIPT_NAME, "exec"), _namespace)
    _sys.exit(_namespace["main"](_sys.argv))
