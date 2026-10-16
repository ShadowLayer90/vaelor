# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Vaelor contributors.
"""The ``vaelor_moe_kernel`` vLLM general plugin: Vaelor's 4-bit kernels for the 30B (VD-216).

vLLM loads every ``vllm.general_plugins`` entry point it can find in each of
its processes. Vaelor's root bridge mounts the folder holding this file, the
kernels and a minimal ``.dist-info`` read-only into the container of the
cluster 30B and puts it on ``PYTHONPATH`` (Vaelor's `vllm_moe_kernels`);
nothing of vLLM's is modified.

:func:`register` wraps two things, each pinned and decided on its own:

* the WNA16 experts' ``apply`` in
  ``vllm.model_executor.layers.fused_moe.experts.triton_moe``, taking
  Vaelor's path ONLY for the call it was built and measured for
  (:func:`refusal`): 4-bit weights in groups of 32 without zero points, SiLU,
  no expert map, routing weights applied after the experts, bf16, the exact
  weight and scale shapes of the 30B, the GPU it was written for (gfx1151:
  the "d2" kernel uses a gfx11 dot instruction), and 1 to
  :data:`PREFILL_TO` tokens;
* the dense 4-bit GEMM function vLLM's ``rdna_hybrid_w4a16`` custom op calls
  (:func:`register_dense`), for the 30B's qkv and o layers from
  :data:`DENSE_FROM` tokens (:func:`dense_refusal`; `vaelor_dense_w4`), and
  the HIP op it calls below that, 1 to :data:`SMALL_TO` tokens - the one-user
  writing step (:func:`small_refusal`), each pinned on its own.

Every other call - any other model, shape or GPU - goes to vLLM's own
kernel, unchanged.

**Three expert paths, chosen by the token count** (VD-216): the per-pair
kernel (`vaelor_moe_triton`) below :data:`D2_FROM` tokens, where it measured
ahead; "d2" (`vaelor_moe_d2`: int4 turned to bf16 by a magic number and
multiplied with ``v_dot2_f32_bf16``) from :data:`D2_FROM` to
:data:`MAX_TOKENS` - 40 is vLLM's largest captured step at 8 users (graph
sizes 24, 32, 40), where d2 measured 1.3-1.7x vLLM's retuned kernel on
routing shaped like the real one; and the prompt-reading path
(`vaelor_moe_prefill`) above that, up to vLLM's 8,192-token batch, measured
1.3-2.1x vLLM's kernel on the Z2 (2026-10-14). From 1 to :data:`HIP_TO`
tokens, Vaelor's HIP d2 kernels (`vaelor_moe_d2.hip`, built as root on each
machine; `vaelor_moe_d2_hip`) serve in place of the first two when this
machine's build passed every check, and the Triton kernels otherwise.
:func:`uses_d2`, :func:`uses_hip` and :func:`uses_prefill` are the one place
each boundary is decided.

**It stands aside, and says so once, plainly,** when the vLLM it finds is not
the pinned version or the wrapped method's source is not the one it was
checked against (:data:`PINNED_APPLY_SHA256`, read in the pinned image): a
different method may hand it different buffers. It also stands aside when it
cannot be loaded at all. Standing aside is vLLM's own kernel, never a failed
start.

**What it did is written to the server's log as one marker line per event**
(:data:`MARKER` then the event, :data:`EVENTS`), so the controller reads it
from the unit's journal when the copy has started (`vllm_moe_kernels.
journal_note`) and the deployment's row says what this machine actually runs:
registered, served inside a GPU graph, stood aside (and why), or failed.

This file imports nothing heavy at module level, so Vaelor's tests read its
rules without vLLM, torch or a GPU.
"""

import hashlib
import inspect
import logging

#: The version string ``vllm.__version__`` reports in the pinned 0.27 image.
PINNED_VLLM_VERSION = "0.27.1.dev5+gf46a9dfe2.d20260827"
#: sha256 of ``inspect.getsource(TritonWNA16Experts.apply)`` in that image.
PINNED_APPLY_SHA256 = "4fe208696c08c3fa9484516b5f17e9fbe0ef2b1746aaa0f9bd1cf74cbbedddc7"

#: The largest decode-sized expert call (d2's top), where d2 takes over, and
#: the largest prompt-reading call (vLLM's 8,192-token batch budget).
MAX_TOKENS = 40
D2_FROM = 5
PREFILL_TO = 8192
#: The largest call the HIP d2 kernels serve, where they measured ahead of
#: HEAD's kernels on the Z2 with cold weights (2026-10-14: 1.03-1.24x at 1-12
#: tokens, parity at 14-16, behind from 20); above it the Triton d2 serves.
HIP_TO = 12
#: The down variant (lanes per output row) by token count, measured the same day.
HIP_DOWN_LANES = ((6, 2), (HIP_TO, 4))
#: The tag that opens every line about the HIP kernels (read apart, like the dense).
HIP_MARK = "[hip]"

#: The one GPU the kernels were written and measured on (`gpu_arch`).
GPU_ARCH = "gfx1151"
ANOTHER_GPU = "not gfx1151"
#: The call's token count is outside what the kernels serve.
TOKEN_COUNT = "token count"
#: workspace2 cannot hold what the chosen kernel keeps there.
WORKSPACE_TOO_SMALL = "workspace too small"
WORKSPACE13_TOO_SMALL = "workspace13 too small"

#: d2's launch plans for gfx1151. Each row: the largest token count it serves;
#: gate+up (pairs in expert order, rows per program, packed words per step,
#: warps); down (per pair, rows per program, packed words per step, warps). A
#: per-pair down writes one bf16 row per pair, which `d2_layout` places.
#: Retuned 2026-10-14 from VD-216's GEAK findings (our sweep values, re-measured
#: on the Z2): gate+up reads a whole 2,048-value row (256 words) per step with 4
#: warps; the 5-token row - the one-user decode step - uses 16-row gate+up
#: programs and a per-pair down (+4.3% end to end, live), and 24-40 tokens a
#: 64-row down.
D2_PLANS = (
    (5, (True, 16, 256, 4), (True, 32, 32, 1)),
    (8, (True, 16, 256, 4), (False, 32, 32, 2)),
    (16, (True, 32, 256, 4), (True, 32, 32, 1)),
    (24, (True, 32, 256, 4), (True, 64, 32, 1)),
    (32, (True, 32, 256, 4), (True, 64, 32, 1)),
    (40, (True, 32, 256, 4), (True, 64, 32, 1)),
)

#: **The dense 4-bit layers** (`vaelor_dense_w4`): the 30B's qkv and o, as
#: (N, K); the token count our decode kernels start at; vLLM's module that
#: serves them, and the sha256 of each of its functions the wrapper relies on,
#: read in the pinned image (2026-10-14).
DENSE_SHAPES = {(5120, 2048): "qkv", (2048, 4096): "o"}
#: The tag that opens every dense marker line (the controller reads them apart).
DENSE_MARK = "[dense]"
DENSE_FROM = 6
DENSE_MODULE = "vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16"
PINNED_DENSE_SHA256 = {
    "_rdna_hybrid_w4a16_apply_impl": "255a599c92810e5060357ff35ce1db9c5f7586c3f11ead6b74ecdeff29699837",
    "triton_w4a16_skinny_fmt_gemm": "999ed6f4be3f2106afc5c63e74866489b6070264ca5174e07647ae48380bacef",
    "_triton_w4a16_skinny_fmt_kernel": "1ca903bb325332bbd01eac23c2d217d741736e931a7b963994e42ba9ff872add",
}
#: Below :data:`DENSE_FROM`, ``_rdna_hybrid_w4a16_apply_impl`` calls vLLM's HIP
#: op instead (1-5 tokens; vLLM's MAX_SKINNY_BATCH_SIZE), with the same weight
#: bytes as int8 [N, K/2]. Its Python wrapper, pinned like the others (read in
#: the pinned image, 2026-10-14), is what :func:`register_dense` also wraps.
SMALL_TO = 5
SMALL_MODULE = "vllm._custom_ops"
PINNED_SMALL_SHA256 = {
    "wvSplitK_int4_g": "a5107db645d24c2a22472130dc09e4f41a7587b53861c5ee166fc7f8f65611ba",
}
_INT32 = "torch.int32"
_INT8 = "torch.int8"

#: The cluster 30B's expert layer (Qwen3-30B-A3B, 4-bit, group 32).
EXPERTS = 128
HIDDEN = 2048
INTERMEDIATE = 768
TOP_K = 8
GROUP = 32
W13_SHAPE = (EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2)
W2_SHAPE = (EXPERTS, HIDDEN, INTERMEDIATE // 2)
S13_SHAPE = (EXPERTS, 2 * INTERMEDIATE, HIDDEN // GROUP)
S2_SHAPE = (EXPERTS, HIDDEN, INTERMEDIATE // GROUP)

#: Set to ``1`` for the host-side index proof before each launch. Debugging
#: only: it reads ids back to the host, so it is never on in a served unit.
CHECK_VARIABLE = "VAELOR_MOE_KERNEL_CHECK"

#: The word every status line starts its message with, and the events. The
#: controller's reader (`vllm_moe_kernels`) holds the same words; a test ties them.
MARKER = "VAELOR_MOE_KERNEL"
REGISTERED = "registered"
SERVED_EAGER = "served-eager"
SERVED_IN_CAPTURE = "served-in-capture"
STOOD_ASIDE = "stood-aside"
STOOD_ASIDE_IN_CAPTURE = "stood-aside-in-capture"
FAILED = "failed"
EVENTS = (REGISTERED, SERVED_EAGER, SERVED_IN_CAPTURE, STOOD_ASIDE,
          STOOD_ASIDE_IN_CAPTURE, FAILED)

#: What `vaelor_moe_triton.run_experts` reports a call as.
RAN_EAGER = "eager"
RAN_IN_CAPTURE = "captured"
ASIDE_IN_CAPTURE = "aside-in-capture"

_BF16 = "torch.bfloat16"
_UINT8 = "torch.uint8"
_FLOAT32 = "torch.float32"
_ID_TYPES = frozenset({"torch.int32", "torch.int64"})

#: Under vLLM's own logger, so the lines reach the container's log.
LOG = logging.getLogger("vllm.vaelor_moe_kernel")
_SAID = set()
#: Marks the wrapper, so a second ``register`` in one process wraps nothing.
_WRAPPED = "_vaelor_moe_kernel"
#: The error that turned the kernel's Python path off in this process.
_FAILED = []
#: The same, for the dense path (each path turns only itself off).
_DENSE_FAILED = []


def say_once(event, message, key=None):
    """Write one marker line for ``event``, once per process (per ``key``, when
    two paths report the same event: the expert path and the dense path)."""
    if (key or event) in _SAID:
        return
    _SAID.add(key or event)
    if str(key).startswith("dense-"):
        message = DENSE_MARK + " " + message
    if str(key).startswith("hip-"):
        message = HIP_MARK + " " + message
    LOG.info("%s %s: %s", MARKER, event, message)


def method_digest(method):
    """sha256 of a method's source text, as :data:`PINNED_APPLY_SHA256` was taken."""
    return hashlib.sha256(inspect.getsource(method).encode("utf-8")).hexdigest()


def stand_aside_reason(version, digest):
    """Why the kernel must not wrap this vLLM, or ``None`` when it may."""
    if version != PINNED_VLLM_VERSION:
        return (
            "this is vLLM {}, not the pinned {}, so vLLM's own expert kernel "
            "serves every call.".format(version, PINNED_VLLM_VERSION)
        )
    if digest != PINNED_APPLY_SHA256:
        return (
            "vLLM's expert method is not the one it was checked against (source "
            "sha256 {}..., pinned {}...), so vLLM's own expert kernel serves every "
            "call.".format(digest[:12], PINNED_APPLY_SHA256[:12])
        )
    return None


def _is(tensor, shape, dtype):
    return (
        tensor is not None and tuple(tensor.shape) == shape
        and str(tensor.dtype) == dtype and tensor.is_contiguous()
    )


def _span(tensor, elements=None):
    """``(first byte, end byte)`` of a tensor's memory: metadata only, no sync."""
    start = int(tensor.data_ptr())
    count = int(tensor.numel()) if elements is None else int(elements)
    return start, start + count * int(tensor.element_size())


def _apart(first, second):
    return first[1] <= second[0] or second[1] <= first[0]


def uses_d2(tokens):
    """Whether a call of ``tokens`` tokens is d2's: the ONE place that boundary is
    decided - `refusal` checks the workspaces the kernel this names will use, and
    `vaelor_moe_triton.run_experts` launches it (LESSONS 6)."""
    return D2_FROM <= tokens <= MAX_TOKENS


def uses_hip(tokens):
    """Whether a call of ``tokens`` tokens is the HIP d2 kernels' when they are
    loaded: the one place that boundary is decided (`refusal` proves their
    scratch, `vaelor_moe_triton.run_experts` launches them)."""
    return 1 <= tokens <= HIP_TO


def hip_down_lanes(tokens):
    """The HIP down variant (lanes per output row) for a call of ``tokens``."""
    for largest, lanes in HIP_DOWN_LANES:
        if tokens <= largest:
            return lanes
    raise ValueError("no HIP down variant covers {} tokens".format(tokens))


def uses_prefill(tokens):
    """Whether a call is the prompt-reading path's (`vaelor_moe_prefill`): the one
    place, like :func:`uses_d2`, refusal and dispatch both ask."""
    return tokens > MAX_TOKENS


def prefill_layout(tokens):
    """``(workspace13 head, workspace2 head)`` the prompt-reading path uses, in
    bf16 slots: SiLU rows at workspace13's start - the output's place, written
    only after they are read, exactly as vLLM 0.27 orders it - and gate+up
    then down rows in workspace2."""
    pairs = tokens * TOP_K
    return pairs * INTERMEDIATE, pairs * HIDDEN


def d2_plan(tokens):
    """d2's plan row for ``tokens`` (:data:`D2_FROM`..:data:`MAX_TOKENS`)."""
    for largest, gate_up, down in D2_PLANS:
        if tokens <= largest:
            return gate_up, down
    raise ValueError("no d2 plan covers {} tokens".format(tokens))


def d2_layout(tokens):
    """``(workspace13 tail, workspace2 head)`` d2 uses, in bf16 slots.

    vLLM 0.27 hands the call its output as the START of ``workspace13`` (one
    buffer, read in `modular_kernel._allocate_buffers`), so nothing d2 keeps
    while the output is written may sit there. Its permuted input (tokens x
    H), intermediate rows (pairs x I) and pair order (pairs int32) go at the
    END of ``workspace13``; a per-pair down's bf16 rows (pairs x H, exactly
    ``workspace2``) go in ``workspace2``. `refusal` proves both fit and touch
    nothing else.
    """
    pairs = tokens * TOP_K
    tail = tokens * HIDDEN + pairs * INTERMEDIATE + 2 * pairs
    return tail, pairs * HIDDEN if d2_plan(tokens)[1][0] else 0


#: The GPU architecture each device reports, read once (`gpu_arch`).
_ARCH = {}


def gpu_arch(device):
    """The device's architecture (``gfx1151``), from torch's device properties, cached."""
    key = str(device)
    if key not in _ARCH:
        try:
            import torch

            name = torch.cuda.get_device_properties(device).gcnArchName
        except Exception:  # no GPU, no torch, no such property: not the measured GPU
            name = ""
        _ARCH[key] = str(name or "").split(":")[0]
    return _ARCH[key]


def refusal(experts, output, hidden_states, w1, w2, topk_weights, topk_ids,
            activation, global_num_experts, expert_map, a1q_scale, a2_scale,
            workspace13, workspace2, apply_router_weight_on_input):
    """The first condition this expert call fails, by name, or ``None`` when it qualifies.

    One named check per condition, in order, so a test can flip each one
    alone and see it refused for exactly that reason. Read from tensor
    metadata and the layer's quantisation settings only - shapes, types,
    strides and addresses, never a value on the GPU - so it costs no
    synchronisation inside a graph capture.
    """
    config = getattr(experts, "quant_config", None)
    if config is None:
        return "no quantisation settings"
    if getattr(config, "use_int4_w4a16", False) is not True:
        return "not 4-bit weights"
    if list(getattr(experts, "block_shape", None) or ()) != [0, GROUP]:
        return "not groups of 32"
    if getattr(config, "w1_zp", None) is not None:
        return "gate+up zero points"
    if getattr(config, "w2_zp", None) is not None:
        return "down zero points"
    if getattr(config, "quant_dtype", None) is not None:
        return "quantised activations"
    if getattr(config, "gemm1_clamp_limit", None) is not None:
        return "a clamped activation"
    if getattr(experts, "_lora_context", None) is not None:
        return "LoRA"
    if getattr(activation, "name", activation) != "SILU":
        return "not SiLU"
    if expert_map is not None:
        return "an expert map"
    if a1q_scale is not None:
        return "an input scale"
    if a2_scale is not None:
        return "an intermediate scale"
    if apply_router_weight_on_input:
        return "routing weights on the input"
    if global_num_experts not in (-1, EXPERTS):
        return "another expert count"
    if hidden_states is None or len(tuple(hidden_states.shape)) != 2:
        return "input not two-dimensional"
    tokens = int(hidden_states.shape[0])
    if not 1 <= tokens <= PREFILL_TO:
        return TOKEN_COUNT
    if not _is(hidden_states, (tokens, HIDDEN), _BF16):
        return "input shape or type"
    if not _is(output, (tokens, HIDDEN), _BF16):
        return "output shape or type"
    if not _is(w1, W13_SHAPE, _UINT8):
        return "gate+up weights"
    if not _is(w2, W2_SHAPE, _UINT8):
        return "down weights"
    if not _is(getattr(config, "w1_scale", None), S13_SHAPE, _BF16):
        return "gate+up scales"
    if not _is(getattr(config, "w2_scale", None), S2_SHAPE, _BF16):
        return "down scales"
    if not _is(topk_weights, (tokens, TOP_K), _FLOAT32):
        return "routing weights shape or type"
    if (topk_ids is None or tuple(topk_ids.shape) != (tokens, TOP_K)
            or not topk_ids.is_contiguous()):
        return "expert ids shape"
    if str(topk_ids.dtype) not in _ID_TYPES:
        return "expert ids type"
    if workspace2 is None or str(workspace2.dtype) != _BF16 or not workspace2.is_contiguous():
        return "workspace"
    devices = {str(item.device) for item in (output, hidden_states, w1, w2, topk_ids,
                                              topk_weights, workspace2)}
    if len(devices) != 1:
        return "more than one device"
    if gpu_arch(hidden_states.device) != GPU_ARCH:
        return ANOTHER_GPU
    pairs = tokens * TOP_K
    others = _call_regions(output, hidden_states, topk_ids, topk_weights)
    if uses_prefill(tokens):
        return _prefill_refusal(tokens, workspace13, workspace2, devices, others)
    if not uses_d2(tokens):
        # The per-pair kernel: intermediate rows, then the pair order (int32 =
        # two bf16 slots each), at the head of workspace2.
        used = pairs * INTERMEDIATE + 2 * pairs
        if int(workspace2.numel()) < used:
            return WORKSPACE_TOO_SMALL
        return _overlap("workspace", _span(workspace2, used), others)
    refused, scratch13 = _d2_tail_refusal(tokens, workspace13, devices, others)
    head = d2_layout(tokens)[1]
    if refused is None and int(workspace2.numel()) < head:
        refused = WORKSPACE_TOO_SMALL
    if refused or not head:
        return refused
    return _overlap("workspace", _span(workspace2, head),
                    others + (("d2 scratch", scratch13),))


def _d2_tail_refusal(tokens, workspace13, devices, others):
    """``(refusal or None, span)`` of d2's scratch at the END of workspace13
    (:func:`d2_layout`): the Triton d2 kernels' and, for :func:`uses_hip`
    calls, the HIP kernels' intermediate rows (which use only this)."""
    if (workspace13 is None or str(workspace13.dtype) != _BF16
            or not workspace13.is_contiguous() or str(workspace13.device) not in devices):
        return "workspace13", None
    tail = d2_layout(tokens)[0]
    if int(workspace13.numel()) < tail:
        return WORKSPACE13_TOO_SMALL, None
    start = int(workspace13.data_ptr()) + (int(workspace13.numel()) - tail) * 2
    scratch13 = (start, start + tail * 2)
    return _overlap("workspace13", scratch13, others), scratch13


def hip_scratch_refusal(output, hidden_states, topk_weights, topk_ids, workspace13):
    """Why the HIP kernels may not take a call :func:`refusal` already passed,
    or ``None``. Below :data:`D2_FROM` tokens `refusal` proved only the
    per-pair kernel's workspace2; the HIP kernels keep their intermediate rows
    in d2's workspace13 tail, so that is proved here, by the same rule. A
    refusal leaves the call to the per-pair kernel - never to vLLM's."""
    tokens = int(hidden_states.shape[0])
    if not uses_hip(tokens):
        return TOKEN_COUNT
    devices = {str(item.device) for item in (output, hidden_states, topk_ids, topk_weights)}
    others = _call_regions(output, hidden_states, topk_ids, topk_weights)
    return _d2_tail_refusal(tokens, workspace13, devices, others)[0]


def _prefill_refusal(tokens, workspace13, workspace2, devices, others):
    """The prompt-reading path's workspace conditions (:func:`prefill_layout`)."""
    if (workspace13 is None or str(workspace13.dtype) != _BF16
            or not workspace13.is_contiguous() or str(workspace13.device) not in devices):
        return "workspace13"
    silu, rows = prefill_layout(tokens)
    if int(workspace13.numel()) < silu:
        return WORKSPACE13_TOO_SMALL
    if int(workspace2.numel()) < rows:
        return WORKSPACE_TOO_SMALL
    staged = _span(workspace2, rows)
    refused = _overlap("workspace", staged, others)
    if refused:
        return refused
    # The output may share the SiLU rows' place (vLLM's own order); nothing else may.
    return _overlap("workspace13", _span(workspace13, silu),
                    others[1:] + (("prompt rows", staged),))


def _call_regions(output, hidden_states, topk_ids, topk_weights):
    """The call's own buffers no scratch may touch, by name, as ``(name, span)``."""
    return (("output", _span(output)), ("input", _span(hidden_states)),
            ("expert ids", _span(topk_ids)), ("routing weights", _span(topk_weights)))


def _overlap(region, span, others):
    """``"<region> overlaps the <name>"`` for the first of ``others`` it touches, else ``None``."""
    for name, other in others:
        if not _apart(span, other):
            return region + " overlaps the " + name
    return None


def qualifies(*arguments, **named):
    """Whether this expert call is the one Vaelor's kernel serves (:func:`refusal`)."""
    return refusal(*arguments, **named) is None


def wrap(original, run):
    """The ``apply`` that sends a qualifying call to ``run`` and every other to ``original``."""

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
              global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2,
              expert_tokens_meta, apply_router_weight_on_input):
        refused = "off" if _FAILED else refusal(
            self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
            global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2,
            apply_router_weight_on_input)
        if refused == ANOTHER_GPU:
            say_once(STOOD_ASIDE, (
                "this GPU is {}, not the {} Vaelor's kernels were written for (d2 uses a "
                "gfx11 dot instruction), so vLLM's own expert kernel serves every call."
            ).format(gpu_arch(hidden_states.device) or "unknown", GPU_ARCH))
        if refused is None:
            tokens = int(hidden_states.shape[0])
            try:
                ran = run(output, hidden_states, w1, w2, self.quant_config.w1_scale,
                          self.quant_config.w2_scale, topk_weights, topk_ids, workspace13,
                          workspace2)
            except Exception as error:  # a kernel that cannot run never stops the server
                _FAILED.append(error)
                say_once(FAILED, (
                    "Vaelor's expert kernel raised ({}). From now on, every expert call "
                    "this vLLM process makes from Python - eager steps and any GPU graph "
                    "captured after this - uses vLLM's own kernel; GPU graphs already "
                    "captured with Vaelor's kernel keep replaying it.").format(error))
                ran = ""
            if ran == RAN_EAGER:
                say_once(SERVED_EAGER, "served its first expert call outside a GPU graph "
                                       "({} tokens).".format(tokens))
                return None
            if ran == RAN_IN_CAPTURE:
                say_once(SERVED_IN_CAPTURE, "recorded into a GPU graph for the first time "
                                            "({} tokens); every replay of it runs Vaelor's "
                                            "kernel.".format(tokens))
                return None
            if ran == ASIDE_IN_CAPTURE:
                say_once(STOOD_ASIDE_IN_CAPTURE, (
                    "a GPU graph of {0} tokens was captured with vLLM's own expert kernel: "
                    "this process never ran {0} tokens outside a graph, so Vaelor's kernels "
                    "for it may not be compiled, and compiling while a graph is being "
                    "captured is not safe.").format(tokens))
        return original(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
                        activation, global_num_experts, expert_map, a1q_scale, a2_scale,
                        workspace13, workspace2, expert_tokens_meta,
                        apply_router_weight_on_input)

    setattr(apply, _WRAPPED, True)
    apply.__wrapped__ = original
    return apply


def dense_refusal(a, b_q, scales, group_size, zp_bias, zp, counts=(DENSE_FROM, PREFILL_TO),
                  packed=(_INT32, 8)):
    """The first condition a dense call fails, by name, or ``None`` when ours serves it.

    vLLM calls the wrapped function for every call above its HIP path (above
    5 tokens, or more when the activations would overflow that kernel's
    on-chip memory) - so each condition stands alone, like :func:`refusal`.
    ``counts`` and ``packed`` (weight dtype, values per element) are the
    path's own: :func:`small_refusal` checks the HIP op's calls by the same rule.
    """
    if zp is not None:
        return "dense zero points"
    if zp_bias != 8:
        return "dense zero bias"
    if group_size != GROUP:
        return "dense group size"
    if a is None or len(tuple(a.shape)) != 2 or str(a.dtype) != _BF16 or not a.is_contiguous():
        return "dense input"
    if int(a.shape[0]) < counts[0] or int(a.shape[0]) > counts[1]:
        return "dense token count"
    if (b_q is None or len(tuple(b_q.shape)) != 2 or str(b_q.dtype) != packed[0]
            or not b_q.is_contiguous()):
        return "dense weights"
    n, k = int(b_q.shape[0]), int(b_q.shape[1]) * packed[1]
    if (n, k) not in DENSE_SHAPES or int(a.shape[1]) != k:
        return "dense shape"
    if not _is(scales, (n, k // GROUP), _BF16):
        return "dense scales"
    if len({str(a.device), str(b_q.device), str(scales.device)}) != 1:
        return "dense devices"
    if gpu_arch(a.device) != GPU_ARCH:
        return ANOTHER_GPU
    return None


def small_refusal(weight, activation, scale, group_size, zero_points, bias):
    """:func:`dense_refusal` for a call of vLLM's HIP op (``wvSplitK_int4_g``):
    1 to :data:`SMALL_TO` tokens, the weights as int8 [N, K/2], no bias."""
    if bias is not None:
        return "dense bias"
    return dense_refusal(activation, weight, scale, group_size, 8, zero_points,
                         counts=(1, SMALL_TO), packed=(_INT8, 2))


def _dense_served(refused, run, a, b_q, scales):
    """Our output for one dense call, or ``None`` for vLLM's; says what happened once."""
    if refused == ANOTHER_GPU:
        say_once(STOOD_ASIDE, "they stand aside because this GPU is {}, not {}, so "
                              "vLLM's own kernels serve the qkv and o layers.".format(
                                  gpu_arch(a.device) or "unknown", GPU_ARCH),
                 key="dense-gpu")
    if refused is not None:
        return None
    try:
        out, ran = run(a, b_q, scales)
    except Exception as error:  # never stops the server
        _DENSE_FAILED.append(error)
        say_once(FAILED, (
            "one of them raised ({}). From now on, vLLM's own kernel serves every "
            "qkv and o call this process makes from Python; GPU graphs already "
            "captured with Vaelor's keep replaying it.").format(error),
            key="dense-failed")
        out, ran = None, ""
    tokens = int(a.shape[0])
    if ran == RAN_EAGER:
        say_once(SERVED_EAGER, "they served their first call outside a GPU graph "
                               "({} tokens).".format(tokens),
                 key="dense-eager")
    if ran == RAN_IN_CAPTURE:
        say_once(SERVED_IN_CAPTURE, "they were recorded into a GPU graph for the "
                                    "first time ({} tokens).".format(tokens),
                 key="dense-captured")
    if ran == ASIDE_IN_CAPTURE:
        say_once(STOOD_ASIDE_IN_CAPTURE, (
            "writing steps of {0} tokens use vLLM's slower qkv and o kernels: their "
            "GPU graph was captured before this process had run {0} tokens outside "
            "one.").format(tokens),
            key="dense-aside-in-capture")
    return out


def wrap_dense(original, run):
    """vLLM's dense GEMM function, with a qualifying call sent to ``run`` instead."""

    def triton_w4a16_skinny_fmt_gemm(a, b_q, scales, group_size, zp_bias=8, zp=None):
        refused = "off" if _DENSE_FAILED else dense_refusal(a, b_q, scales, group_size,
                                                            zp_bias, zp)
        out = _dense_served(refused, run, a, b_q, scales)
        if out is not None:
            return out
        return original(a, b_q, scales, group_size, zp_bias=zp_bias, zp=zp)

    setattr(triton_w4a16_skinny_fmt_gemm, _WRAPPED, True)
    triton_w4a16_skinny_fmt_gemm.__wrapped__ = original
    return triton_w4a16_skinny_fmt_gemm


def wrap_small(original, run):
    """vLLM's HIP op for 1-5 tokens, with a qualifying call sent to ``run`` instead.

    ``run`` gets the int8 weights as vLLM holds them; a count its plan leaves
    to vLLM returns ``(None, "")`` and reaches ``original`` without a line.
    """

    def wvSplitK_int4_g(weight, activation, scale, cu_count, group_size, zero_points=None,
                        bias=None):
        refused = "off" if _DENSE_FAILED else small_refusal(weight, activation, scale,
                                                            group_size, zero_points, bias)
        out = _dense_served(refused, run, activation, weight, scale)
        if out is not None:
            return out
        return original(weight, activation, scale, cu_count, group_size, zero_points, bias)

    setattr(wvSplitK_int4_g, _WRAPPED, True)
    wvSplitK_int4_g.__wrapped__ = original
    return wvSplitK_int4_g


def dense_stand_aside_reason(version, digests, pinned_digests=None):
    """Why the dense wrapper must not wrap this vLLM, or ``None`` when it may."""
    if version != PINNED_VLLM_VERSION:
        return ("they stand aside because this is vLLM {}, not the pinned {}, so vLLM's own "
                "kernels serve the qkv and o layers.".format(version, PINNED_VLLM_VERSION))
    for name, pinned in sorted((pinned_digests or PINNED_DENSE_SHA256).items()):
        if digests.get(name) != pinned:
            return ("they stand aside because vLLM's {} is not the one they were "
                    "checked against (source sha256 {}..., pinned {}...).".format(
                        name, str(digests.get(name))[:12], pinned[:12]))
    return None


def register_dense(version):
    """Wrap vLLM's dense 4-bit GEMM function when it is the pinned one (module docstring)."""
    import importlib

    try:
        module = importlib.import_module(DENSE_MODULE)
        original = module.triton_w4a16_skinny_fmt_gemm
        if getattr(original, _WRAPPED, False):
            return
        digests = {}
        for name in PINNED_DENSE_SHA256:
            item = getattr(module, name)
            digests[name] = method_digest(getattr(item, "fn", item))
    except Exception as error:
        say_once(STOOD_ASIDE, "they stand aside because vLLM's code for the qkv and o "
                              "layers could not be found ({}).".format(error), key="dense-import")
        return
    reason = dense_stand_aside_reason(version, digests)
    if reason is not None:
        say_once(STOOD_ASIDE, reason, key="dense-pin")
        return
    try:
        from vaelor_dense_w4 import run_dense
    except Exception as error:
        say_once(STOOD_ASIDE, "they stand aside because they could not be loaded "
                              "({}).".format(error), key="dense-kernels")
        return
    kernel = module._triton_w4a16_skinny_fmt_kernel
    module.triton_w4a16_skinny_fmt_gemm = wrap_dense(
        original, lambda a, b_q, scales: run_dense(a, b_q, scales, kernel))
    say_once(REGISTERED, "they registered: calls of 6 to {} tokens on the 30B's qkv and o layers "
                         "will use Vaelor's kernels on {}.".format(PREFILL_TO, GPU_ARCH),
             key="dense-registered")
    register_small(lambda a, weight, scales: run_dense(a, weight, scales, kernel))


def register_small(run):
    """Wrap vLLM's HIP op for 1-5 tokens when its wrapper is the pinned one.

    Called by :func:`register_dense` once the version and the 6-8,192 path's
    pins have passed; this pin guards only this op, so a different one leaves
    the larger counts served and says which calls stand aside.
    """
    import importlib

    try:
        module = importlib.import_module(SMALL_MODULE)
        original = module.wvSplitK_int4_g
        if getattr(original, _WRAPPED, False):
            return
        digest = method_digest(original)
    except Exception as error:
        say_once(STOOD_ASIDE, "the 1-{} token kernels stand aside because vLLM's code for them "
                              "could not be found ({}), so vLLM's HIP kernel serves those "
                              "qkv and o calls.".format(SMALL_TO, error), key="dense-small-import")
        return
    pinned = PINNED_SMALL_SHA256["wvSplitK_int4_g"]
    if digest != pinned:
        say_once(STOOD_ASIDE, (
            "the 1-{} token kernels stand aside because vLLM's wvSplitK_int4_g is not the one "
            "they were checked against (source sha256 {}..., pinned {}...), so vLLM's HIP "
            "kernel serves those qkv and o calls.").format(SMALL_TO, digest[:12], pinned[:12]),
            key="dense-small-pin")
        return
    module.wvSplitK_int4_g = wrap_small(original, run)
    say_once(REGISTERED, "they registered for 1 to {} tokens too (the one-user writing step) "
                         "on {}, where their plan measured ahead of vLLM's HIP kernel.".format(
                             SMALL_TO, GPU_ARCH),
             key="dense-small-registered")


def register():
    """vLLM's entry point: wrap the experts' ``apply``, then the dense 4-bit GEMM,
    each only when this vLLM is the pinned one; either standing aside leaves
    the other as it decided."""
    _register_experts()
    try:
        import vllm
        version = str(getattr(vllm, "__version__", ""))
    except Exception:  # the expert step has already said why
        return
    register_dense(version)


def _register_experts():
    """Wrap the experts' ``apply`` when this vLLM is the pinned one."""
    try:
        import vllm
        from vllm.model_executor.layers.fused_moe.experts import triton_moe
        experts = triton_moe.TritonWNA16Experts
        original = experts.apply
    except Exception as error:  # an import that fails is a stand-aside, never a failed start
        say_once(STOOD_ASIDE, "vLLM's expert code could not be found ({}), so vLLM's own "
                              "expert kernel serves every call.".format(error))
        return
    if getattr(original, _WRAPPED, False):
        return
    try:
        digest = method_digest(original)
    except (OSError, TypeError):  # no source to read is not the pinned method
        digest = ""
    reason = stand_aside_reason(str(getattr(vllm, "__version__", "")), digest)
    if reason is not None:
        say_once(STOOD_ASIDE, reason)
        return
    try:
        from vaelor_moe_triton import run_experts
    except Exception as error:
        say_once(STOOD_ASIDE, "its kernels could not be loaded ({}), so vLLM's own expert "
                              "kernel serves every call.".format(error))
        return
    experts.apply = wrap(original, run_experts)
    say_once(REGISTERED, "expert calls of up to {} tokens on the 30B's 4-bit layers will "
                         "use Vaelor's kernels on {} (d2 from {} tokens, prompt reading above "
                         "{}); every other call uses vLLM's own.".format(
                             PREFILL_TO, GPU_ARCH, D2_FROM, MAX_TOKENS))
