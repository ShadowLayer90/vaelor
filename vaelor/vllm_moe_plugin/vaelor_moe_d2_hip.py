# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Vaelor contributors.
"""Vaelor's HIP d2 expert kernels, loaded from the code object this machine built (VD-216).

Original Vaelor work. The kernels are ``vaelor_moe_d2.hip`` in this folder;
Vaelor's installer compiles that exact file (its sha256 is pinned as
:data:`SOURCE_SHA256`) as root, with the compiler inside the pinned vLLM image,
into :data:`CODE_OBJECT`, and writes :data:`RECORD` beside it: the source's and
the code object's sha256, and whether the build succeeded. The root bridge
checks both again before it mounts this folder (`vllm_moe_kernels`).

**Here, inside the server**, the record and the code object are read once,
the code object's bytes are hashed IN MEMORY and those same bytes are handed
to ``hipModuleLoadData`` - nothing is read twice, so no file can change between
the check and the load (LESSONS 18, #178). A record that is missing, says the
build failed, names another source, or does not match the bytes is a named
refusal (:func:`record_refusal`); the plugin then runs the Triton d2 kernels
exactly as before and writes one marker line saying why.

Launches go through the HIP runtime torch itself loaded (found in this
process's memory map, never a second copy), on torch's current stream, so a
GPU graph capture records them like any other kernel. Nothing is allocated:
the intermediate rows live in the call's own ``workspace13`` tail
(`vaelor_moe_kernel.d2_layout`), which `refusal` has already proved free.

It also holds the dispatch `vaelor_moe_triton.run_experts` calls
(:func:`run_hip`): load on the first call outside a graph capture, never
during one; a refused or failed load or launch turns the HIP kernels off for
the process, says so once, and leaves the call to the Triton kernels.

This file imports nothing heavy at module level (torch only inside the
dispatch), so Vaelor's tests read its rules - and drive the dispatch with
fakes - without torch or a GPU.
"""

import ctypes
import hashlib
import json
import os

from vaelor_moe_kernel import (
    FAILED, GPU_ARCH, HIDDEN, HIP_TO, INTERMEDIATE, REGISTERED, SERVED_IN_CAPTURE, STOOD_ASIDE,
    TOP_K, d2_layout, hip_down_lanes, hip_scratch_refusal, say_once,
)

#: The kernel source, the built code object and the build's record, in this folder.
SOURCE = "vaelor_moe_d2.hip"
CODE_OBJECT = "vaelor_moe_d2_hip.co"
RECORD = "vaelor_moe_d2_hip.json"

#: sha256 of :data:`SOURCE`. The build refuses any other source; so does the load.
SOURCE_SHA256 = "032ebc7ee3dcb67419e9ed1ea7bd2e0dce924491c8586653c9d8e0ce15115bde"

#: The launch shape the source fixes (its ``VG_*``/``VD_*`` defines; a test
#: reads them out of the source so the two cannot drift - LESSONS 6).
GATE_WAVES = 16
GATE_ROWS = 2
DOWN_WAVES = 8
#: The lanes-per-row variants of the down kernel the source carries.
DOWN_LANES = (2, 4)

#: The kernels' names in the code object, by (kind, expert-id type).
KERNELS = {
    ("gate_up", "torch.int32"): "vaelor_d2_gate_up_i32",
    ("gate_up", "torch.int64"): "vaelor_d2_gate_up_i64",
    ("down2", "torch.int32"): "vaelor_d2_down2_i32",
    ("down2", "torch.int64"): "vaelor_d2_down2_i64",
    ("down4", "torch.int32"): "vaelor_d2_down4_i32",
    ("down4", "torch.int64"): "vaelor_d2_down4_i64",
}


#: Largest code object accepted (a gfx1151 build is about 40 KB).
_LIMIT = 1 << 20

#: The refusals, each a plain sentence the server log and the console carry.
NO_RECORD = "no build record is in the kernel folder (the HIP kernels were not built on this machine)"
BUILD_FAILED = "the build on this machine did not succeed ({})"
OTHER_SOURCE = "the build record names a different kernel source than this release's"
SOURCE_CHANGED = "the HIP source in the kernel folder is not the pinned one"
NO_OBJECT = "the built code object is missing from the kernel folder"
OBJECT_CHANGED = "the built code object's checksum differs from the build record's"


def record_refusal(record, source, code):
    """Why the code object must not be loaded, or ``None`` (module docstring).

    ``record`` is the parsed record (or ``None``), ``source`` and ``code`` the
    bytes read from the folder (or ``None`` when absent).
    """
    if not isinstance(record, dict):
        return NO_RECORD
    if record.get("ready") is not True:
        return BUILD_FAILED.format(str(record.get("reason") or "no reason was recorded")[:200])
    if record.get("source_sha256") != SOURCE_SHA256:
        return OTHER_SOURCE
    if source is None or hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        return SOURCE_CHANGED
    if code is None:
        return NO_OBJECT
    if hashlib.sha256(code).hexdigest() != record.get("built_sha256"):
        return OBJECT_CHANGED
    return None


def read_folder(folder):
    """``(record, source bytes, code bytes)`` from ``folder``, each ``None`` when unreadable."""

    def read(name, limit):
        try:
            with open(os.path.join(folder, name), "rb") as handle:
                data = handle.read(limit + 1)
        except OSError:
            return None
        return data if len(data) <= limit else None

    raw = read(RECORD, 1 << 16)
    try:
        record = json.loads(raw.decode("utf-8")) if raw is not None else None
    except (UnicodeDecodeError, ValueError):
        record = None
    return record, read(SOURCE, 1 << 20), read(CODE_OBJECT, _LIMIT)


def hip_library_path(maps="/proc/self/maps"):
    """The HIP runtime this process already loaded (torch's), or ``None``."""
    try:
        with open(maps, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                path = line.split()[-1] if line.split() else ""
                if os.path.basename(path).startswith("libamdhip64.so"):
                    return path
    except OSError:
        return None
    return None


class _Hip:
    """The four HIP runtime calls this needs, bound with their exact argument types."""

    def __init__(self, path):
        library = ctypes.CDLL(path)
        self.load = library.hipModuleLoadData
        self.load.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self.load.restype = ctypes.c_int
        self.function = library.hipModuleGetFunction
        self.function.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        self.function.restype = ctypes.c_int
        self.launch = library.hipModuleLaunchKernel
        self.launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 7
                                + [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.POINTER(ctypes.c_void_p)])
        self.launch.restype = ctypes.c_int


class HipKernels:
    """One device's loaded code object; :meth:`run` launches both kernels for one call."""

    def __init__(self, hip, code):
        self._hip = hip
        self._image = ctypes.create_string_buffer(code, len(code))  # kept alive with the module
        module = ctypes.c_void_p()
        status = hip.load(ctypes.byref(module), ctypes.cast(self._image, ctypes.c_void_p))
        if status != 0:
            raise RuntimeError("hipModuleLoadData returned {}".format(status))
        self._module = module
        self._functions = {}
        for key, name in KERNELS.items():
            function = ctypes.c_void_p()
            status = hip.function(ctypes.byref(function), module, name.encode("ascii"))
            if status != 0:
                raise RuntimeError("hipModuleGetFunction({}) returned {}".format(name, status))
            self._functions[key] = function

    def launch(self, kind, dtype, grid, threads, stream, arguments):
        """One kernel on ``stream`` (an int ``hipStream_t``); every argument a pointer."""
        values = [ctypes.c_void_p(int(value)) for value in arguments]
        pointers = (ctypes.c_void_p * len(values))(
            *[ctypes.cast(ctypes.pointer(value), ctypes.c_void_p) for value in values])
        status = self._hip.launch(self._functions[(kind, dtype)], grid[0], grid[1], 1,
                                  threads, 1, 1, 0, ctypes.c_void_p(int(stream)), pointers, None)
        if status != 0:
            raise RuntimeError("hipModuleLaunchKernel({}) returned {}".format(kind, status))

    def run(self, output, hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids,
            inter, stream, lanes):
        """Both launches for one checked call; ``inter`` is the pairs x 768 bf16 scratch,
        ``lanes`` the down variant (:data:`DOWN_LANES`)."""
        tokens = int(hidden_states.shape[0])
        pairs = tokens * int(topk_ids.shape[1])
        dtype = str(topk_ids.dtype)
        self.launch("gate_up", dtype, (pairs, INTERMEDIATE // (GATE_WAVES * GATE_ROWS)),
                    32 * GATE_WAVES, stream,
                    (hidden_states.data_ptr(), topk_ids.data_ptr(), w1.data_ptr(),
                     w1_scale.data_ptr(), inter.data_ptr()))
        self.launch("down{}".format(lanes), dtype,
                    (tokens, HIDDEN // (DOWN_WAVES * 32 // lanes)), 32 * DOWN_WAVES, stream,
                    (inter.data_ptr(), topk_ids.data_ptr(), topk_weights.data_ptr(),
                     w2.data_ptr(), w2_scale.data_ptr(), output.data_ptr()))


def load(folder, *, library=None):
    """``(kernels, None)`` when the folder's code object passes every check and
    loads, else ``(None, reason)``. ``library`` is a test seam."""
    record, source, code = read_folder(folder)
    refused = record_refusal(record, source, code)
    if refused is not None:
        return None, refused
    path = library or hip_library_path()
    if path is None:
        return None, "the HIP runtime torch uses could not be found in this process"
    try:
        return HipKernels(_Hip(path), code), None
    except (OSError, AttributeError, RuntimeError) as error:
        return None, "the code object could not be loaded ({})".format(str(error)[:160])


# --- The dispatch `vaelor_moe_triton.run_experts` calls ------------------------

def hip_intermediate(workspace13, tokens):
    """The HIP kernels' intermediate rows (pairs x I, bf16): the same place in
    d2's workspace13 tail as the Triton d2 keeps them (`vaelor_moe_d2.d2_views`),
    taken from the tail alone - below `D2_FROM` tokens no workspace2 head is
    proved, so none is touched."""
    pairs = tokens * TOP_K
    flat = workspace13.view(-1)
    start = flat.numel() - d2_layout(tokens)[0] + tokens * HIDDEN
    return flat[start: start + pairs * INTERMEDIATE].view(pairs, INTERMEDIATE)


#: This process's HIP d2 kernels, by device: loaded, or ``None`` once refused.
_HIP = {}


def hip_kernels(device, capturing):
    """The HIP d2 kernels for ``device``, loaded on the first call outside a
    graph capture (loading a module while a stream captures is not safe), or
    ``None`` - and the one marker line saying which, and why (VD-216)."""
    key = str(device)
    if key in _HIP:
        return _HIP[key]
    if capturing:
        return None
    import torch

    folder = os.path.dirname(os.path.abspath(__file__))
    with torch.cuda.device(device):
        kernels, reason = load(folder)
    _HIP[key] = kernels
    if kernels is None:
        say_once(STOOD_ASIDE, (
            "the HIP d2 kernels stand aside: {}; so the Triton kernels serve expert calls of "
            "1 to {} tokens, as before.").format(reason, HIP_TO), key="hip-aside")
    else:
        say_once(REGISTERED, (
            "the HIP d2 kernels this machine built were checked and loaded; they serve expert "
            "calls of 1 to {} tokens on {}.").format(HIP_TO, GPU_ARCH), key="hip-loaded")
    return kernels


def run_hip(output, hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids,
            workspace13, capturing):
    """Launch the HIP d2 kernels for one checked call; ``False`` when they do
    not take it (not loaded, refused, or their scratch is not proved free), and
    the Triton kernels then serve it exactly as before. A launch that fails
    turns them off for this process and says so once; the call then runs on
    Triton, which rewrites every byte the failed launch could have touched."""
    import torch

    kernels = hip_kernels(hidden_states.device, capturing)
    if kernels is None or hip_scratch_refusal(output, hidden_states, topk_weights, topk_ids,
                                              workspace13) is not None:
        return False
    tokens = int(hidden_states.shape[0])
    inter = hip_intermediate(workspace13, tokens)
    try:
        kernels.run(output, hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids,
                    inter, torch.cuda.current_stream(hidden_states.device).cuda_stream,
                    hip_down_lanes(tokens))
    except Exception as error:  # a launch that fails never stops the server
        _HIP[str(hidden_states.device)] = None
        say_once(FAILED, (
            "a HIP d2 launch failed ({}); from now on the Triton kernels serve expert calls "
            "of 1 to {} tokens in this process.").format(error, HIP_TO), key="hip-failed")
        return False
    if capturing:
        say_once(SERVED_IN_CAPTURE, "the HIP d2 kernels were recorded into a GPU graph for the "
                                    "first time ({} tokens).".format(tokens), key="hip-captured")
    return True
