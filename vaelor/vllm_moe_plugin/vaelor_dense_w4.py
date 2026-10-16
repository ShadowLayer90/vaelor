# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Vaelor contributors.
"""Vaelor's kernels for the 30B's 4-bit dense layers (qkv and o) on gfx1151 (VD-216).

Original Vaelor work. GEAK (AMD-AGI/GEAK) measured what limits vLLM 0.27's
dense 4-bit path on this GPU, and we use those findings, not its code (owner,
2026-10-14): at decode sizes its programs walk K in lock-step, so rows 1-2 KB
apart all read the same memory channels; a dot-instruction GEMV with
magic-number dequantisation suits a handful of tokens; matrix units suit more.
At prompt sizes the launch configuration collapses above about 3,000 tokens.
No line of this comes from vLLM, GEAK, Ornith/Ciru or any other project.

**The weights** are vLLM's stored form for this kernel class: int32 [N, K/8],
each word holding eight 4-bit values in the order 0,2,4,6 at bits 0-15 and
1,3,5,7 at bits 16-31, so ``(word >> 4j) & 0x000F000F`` is the pair
(value 2j, value 2j+1) - the activations need no reordering for the dot
instruction - and bf16 scales [N, K/32]; symmetric, value = n - 8.

**Decode, 1-64 tokens** (:func:`run_dense`), by :data:`DECODE_PLANS`:

* ``"mx"``: `vaelor_dense_mx`, a Gluon kernel on the matrix units that reads
  each weight once for every token of the call, builds the dequantised values
  in the matrix unit's own registers and applies each group's scale once in
  fp32 - every size but qkv at 1-3 and above 48 tokens. At 1-5 tokens it takes
  the calls vLLM sends to its HIP op (`vaelor_moe_kernel.wrap_small`): the
  one-user writing step is 5 tokens, where it measured 31.9 against 38.4 us
  (qkv) and 26.4 against 38.5 us (o) on the Z2, 2026-10-14;
* ``"vllm"``: vLLM's HIP op keeps qkv at 1-3 tokens, where it measured ahead;
* ``"wmma"``: :func:`vaelor_dense_wmma` (008abd6), faster for qkv at 49-64
  tokens: the matrix units, each 4-bit value turned into ``(n - 8) * scale``
  exactly in fp32 and rounded once to bf16 (the value vLLM's own kernel
  multiplies);
* ``"gemv"``: :func:`vaelor_dense_gemv` (008abd6), d2's dot-word arithmetic
  on the dense rows.

When Triton's Gluon dialect cannot be loaded, the 008abd6 plans
(:data:`FALLBACK_PLANS`) serve 6-64 tokens, vLLM's HIP op keeps 1-5, and the
console says so.

The Triton kernels start each program's K walk at a different step (its
column block modulo the step count), so neighbouring programs read different
memory channels. A program's start depends only on its own id, never on the
order programs run.

**Compiled before serving.** The first call outside a GPU graph for a layer -
vLLM's start-up profiling run - compiles every decode and prompt kernel that
layer can launch (:func:`compile_layer`), so none compiles on a served request.

**Prompt reading, above 64 tokens** (:data:`PROMPT_PLANS`). Prompt reading
is compute-bound, and on this GPU the bf16 GEMM (torch, hipBLASLt) runs these
two shapes at about 35 TFLOPS where every fused 4-bit kernel we have measured
runs at 20-27. So ``"dq"`` unpacks the layer's weights once per call into a
bf16 scratch (:func:`vaelor_dense_unpack`: ``(n - 8) * scale`` exact in fp32,
rounded once - the value vLLM's kernel multiplies) and multiplies by the bf16
GEMM: qkv 1.2-1.6x and o 1.2-1.9x vLLM's kernel at our launch values, layer
by layer, errors below vLLM's (Z2, 2026-10-14). The scratch is N x K bf16
(21 MB for qkv), allocated per call by :func:`run_dense` like its output, so
vLLM's start-up profiling run at the 8,192-token budget counts it.

**But the per-layer win is not the model's win on short prompts** (kb X5).
Writing the scratch costs the layers after it: in a 48-layer stack of the
real kernels the expert layer that follows ran 0.17-0.3 ms slower per layer,
and the served build (-5) read 142-token prompts 12 ms and 487-token prompts
11 ms slower than -4 (live r39). So ``"dq"`` serves only above 1,536 tokens,
where the live build measured 13-31 ms ahead of -4 (1,366-1,868 tokens) and
the stack 45-120 ms; below it ``"vllm"`` is vLLM's own kernel at -4's launch
values, exactly -4's prompt path.

**65-384 tokens (WO1, 2026-10-15).** vLLM's kernel at -4's values launches
one 128-token tile row here, so 40 programs fill the GPU's 40 compute units
once and qkv+o ran at about 11 TFLOPS. Two kernels that write no scratch
replace it: at 65-200 the Gluon decode kernel with a token grid (32-row
blocks at the 32-token decode plan; weights re-read per block, from cache),
at 201-384 the Triton matrix kernel with a token grid. In the 48-layer stack
of real kernels they read 65-358-token prompts 6-10 ms faster than -6, and
the expert layers after them did not slow (the -5 trap, kb X11); errors stay
below vLLM's. Above 384 both lost to vLLM's kernel, which keeps those counts.
**One launch per call:** prompt reading is eager (no GPU graph above 80
tokens), so each launch costs host time - a Gluon launch about 43 us against
20 us for vLLM's kernel. Run as separate 32-row launches the same work cost
85-177 us of host time per call, and end to end the server gained only half
of the stack's win; one launch with a token grid keeps the host ahead.
"""

import torch
import triton
import triton.language as tl

from vaelor_moe_d2 import _activation_words, _group_total, _word_dot
from vaelor_moe_kernel import (
    ASIDE_IN_CAPTURE, DENSE_FROM, DENSE_SHAPES, PREFILL_TO, RAN_EAGER, RAN_IN_CAPTURE,
    STOOD_ASIDE, say_once,
)

try:
    import vaelor_dense_mx
except ImportError as error:  # no Gluon in this Triton: the 008abd6 plans serve
    vaelor_dense_mx = None
    _MX_MISSING = error
else:
    _MX_MISSING = None

#: The smallest and largest decode call; above it, prompt reading (:func:`run_prompt`).
DECODE_FROM = 1
DECODE_TO = 64

#: Decode launch plans, by shape: (largest M, kernel, launch values). "mx":
#: `vaelor_dense_mx.PLAN_FIELDS` (rows per program, warps on K, start
#: stagger, int32 of shared memory held, scales staged in shared memory,
#: paired activation loads). Swept on the Z2, 2026-10-14, real layer-24
#: weights rotated through 128 MB of copies in GEAK's frozen harness; 1-5
#: tokens the same way against vLLM's HIP op, which keeps qkv at 1-3
#: (27.1 / 32.4 / 30.1 us against 31.5 / 31.4 / 31.4; "vllm": no launch values).
DECODE_PLANS = {
    "qkv": ((3, "vllm", None),
            (16, "mx", (32, 4, 3, 8192, False, False)),
            (24, "mx", (32, 4, 3, 8192, True, False)),
            (32, "mx", (32, 2, 3, 4096, True, True)),
            (48, "mx", (32, 4, 3, 8192, False, False)),
            (64, "wmma", (32, 2, 64, 2, 2))),
    "o": ((32, "mx", (32, 4, 3, 8192, True, False)),
          (40, "mx", (32, 4, 3, 8192, False, False)),
          (48, "mx", (32, 4, 3, 0, False, False)),
          (56, "mx", (32, 4, 3, 0, False, True)),
          (64, "mx", (16, 8, 3, 0, False, False))),
}

#: The 008abd6 decode plans: (largest M, kernel, (rows per program, words or
#: groups per step, tokens per program, warps, stages)). They serve when
#: Gluon is missing, which :data:`DECODE_PLANS` needs; vLLM's HIP op keeps 1-5.
FALLBACK_PLANS = {
    "qkv": ((5, "vllm", None), (16, "wmma", (32, 4, 16, 2, 2)), (32, "wmma", (32, 2, 32, 2, 2)),
            (64, "wmma", (32, 2, 64, 2, 2))),
    "o": ((5, "vllm", None), (8, "gemv", (16, 64, 2, 2, 1)), (16, "wmma", (32, 4, 16, 2, 2)),
          (32, "wmma", (32, 2, 32, 4, 2)), (64, "wmma", (32, 2, 64, 2, 2))),
}

#: Prompt-reading plans, by shape: (largest M, kernel, launch values). "mx":
#: the Gluon decode kernel in one launch, blocks of this many token rows on
#: its second grid axis, at that count's :data:`DECODE_PLANS` row - no scratch,
#: the weights re-read per block from the GPU's cache. "wmma": :func:`vaelor_dense_wmma`
#: with a token grid (rows, groups per step, tokens per program, warps,
#: stages). "dq": :func:`vaelor_dense_unpack` (rows, packed words per
#: program, warps), then the bf16 GEMM. "vllm": vLLM's own kernel (BLOCK_M,
#: BLOCK_N, BLOCK_K, warps, stages), BLOCK_K always the 32-value scale group,
#: at -4's launch values. "dq" above 1,536 tokens: alone it was ahead from 65
#: (qkv) and 257 (o), but end to end it is ahead only on long prompts (module
#: docstring). "mx" and "wmma" at 65-~500 (WO1, 2026-10-15): see the module
#: docstring; without Gluon :data:`FALLBACK_PROMPT_PLANS` serves.
PROMPT_PLANS = {
    "qkv": ((200, "mx", 32), (384, "wmma", (64, 1, 128, 2, 1)),
            (1536, "vllm", (256, 256, 32, 16, 2)), (1 << 30, "dq", (4, 128, 8))),
    "o": ((200, "mx", 32), (320, "wmma", (64, 2, 64, 4, 2)), (384, "wmma", (64, 2, 128, 4, 1)),
          (1536, "vllm", (256, 256, 32, 16, 2)), (1 << 30, "dq", (4, 128, 4))),
}

#: The -6 prompt plans, which serve when Gluon is missing ("mx" needs it).
FALLBACK_PROMPT_PLANS = {
    "qkv": ((100, "vllm", (128, 128, 32, 8, 2)), (1536, "vllm", (256, 256, 32, 16, 2)),
            (1 << 30, "dq", (4, 128, 8))),
    "o": ((256, "vllm", (128, 64, 32, 8, 2)), (1536, "vllm", (256, 256, 32, 16, 2)),
          (1 << 30, "dq", (4, 128, 4))),
}

#: A masked weight word: eight 4-bit values that all decode to zero (8 - 8).
_ZERO_WORD = tl.constexpr(-2004318072)


def decode_plan(shape, tokens):
    """``(kernel, launch values)`` for a decode call of ``tokens`` on a named shape."""
    plans = FALLBACK_PLANS if vaelor_dense_mx is None else DECODE_PLANS
    for largest, kernel, plan in plans[shape]:
        if tokens <= largest:
            return kernel, plan
    raise ValueError("no dense decode plan covers {} tokens".format(tokens))


def plan_refusals():
    """``[(layer, tokens, reason)]`` for every decode count its "mx" plan cannot serve.

    The Gluon kernel's loads are unmasked and in range only under
    `vaelor_dense_mx.plan_refusal`'s conditions, so the module refuses to load
    (and the dense kernels stand aside, `vaelor_moe_kernel.register_dense`)
    rather than launch a plan that breaks them.
    """
    found = []
    if vaelor_dense_mx is None:
        return found
    for (n, k), shape in sorted(DENSE_SHAPES.items()):
        for tokens in range(DECODE_FROM, DECODE_TO + 1):
            kernel, plan = decode_plan(shape, tokens)
            reason = vaelor_dense_mx.plan_refusal(n, k, tokens, plan) if kernel == "mx" else None
            if reason:
                found.append((shape, tokens, reason))
    return found


def prompt_plans():
    """The prompt-plan table this process serves: :data:`FALLBACK_PROMPT_PLANS` without Gluon."""
    return FALLBACK_PROMPT_PLANS if vaelor_dense_mx is None else PROMPT_PLANS


def prompt_plan(shape, tokens):
    """``(kernel, launch values)`` for a prompt-reading call of ``tokens``."""
    for largest, kernel, plan in prompt_plans()[shape]:
        if tokens <= largest:
            return kernel, plan
    raise ValueError("no dense prompt plan covers {} tokens".format(tokens))


def prompt_refusals():
    """``[(layer, block, reason)]`` for every "mx" prompt row whose token block
    the Gluon kernel cannot serve: its decode plan must be "mx" and pass
    `vaelor_dense_mx.plan_refusal` at that block size (the kernel's loads are
    unmasked; G47: a bad shape aborts the process)."""
    found = []
    if vaelor_dense_mx is None:
        return found
    for (n, k), shape in sorted(DENSE_SHAPES.items()):
        for _largest, kernel, block in PROMPT_PLANS[shape]:
            if kernel != "mx":
                continue
            decode_kernel, plan = decode_plan(shape, block)
            reason = ("its decode plan is not the Gluon kernel" if decode_kernel != "mx"
                      else vaelor_dense_mx.plan_refusal(n, k, block, plan))
            if reason:
                found.append((shape, block, reason))
    return found


if plan_refusals():
    raise ImportError("VD-216: a dense decode plan breaks its kernel's in-range loads: "
                      "{}".format(plan_refusals()))
if prompt_refusals():
    raise ImportError(
        "LESSONS 1 / VD-216 (WO1): an 'mx' prompt row's token block has no checked Gluon "
        "decode plan, so the dense kernels stand aside rather than launch unmasked loads "
        "out of range (G47: a bad shape aborts the process). Pick a block whose decode row "
        "is 'mx'; do not remove this check: {}".format(prompt_refusals()[:4]))


@triton.jit
def vaelor_dense_unpack(w_ptr, s_ptr, out_ptr, N: tl.constexpr, K: tl.constexpr,
                        BN: tl.constexpr, BW: tl.constexpr):
    """BN rows x BW packed words of the layer -> bf16 ``(n - 8) * scale``, in K order.

    Each word is read once and its eight values written as one 16-byte run:
    value j of a word sits at bit ``(j // 2) * 4 + (j % 2) * 16``.
    """
    KW: tl.constexpr = K // 8
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    cols = tl.program_id(1) * BW + tl.arange(0, BW)
    ok = (rows[:, None] < N) & (cols[None, :] < KW)
    words = tl.load(w_ptr + rows[:, None] * KW + cols[None, :], mask=ok, other=_ZERO_WORD)
    j = tl.arange(0, 8)
    nibbles = (words[:, :, None] >> ((j // 2) * 4 + (j % 2) * 16)[None, None, :]) & 15
    scales = tl.load(s_ptr + rows[:, None] * (K // 32) + (cols // 4)[None, :], mask=ok,
                     other=0.0).to(tl.float32)
    values = ((nibbles.to(tl.float32) - 8.0) * scales[:, :, None]).to(tl.bfloat16)
    tl.store(out_ptr + rows[:, None, None] * K + (cols * 8)[None, :, None] + j[None, None, :],
             values, mask=ok[:, :, None])


@triton.jit
def vaelor_dense_gemv(x_ptr, w_ptr, s_ptr, out_ptr, M, N: tl.constexpr, K: tl.constexpr,
                      BN: tl.constexpr, BW: tl.constexpr, TM: tl.constexpr):
    """BN output rows for TM tokens, by d2's dot words; staggered K walk."""
    column_block = tl.program_id(0)
    token_block = tl.program_id(1)
    KW: tl.constexpr = K // 8
    G: tl.constexpr = K // 32
    GPS: tl.constexpr = BW // 4
    STEPS: tl.constexpr = KW // BW
    rows = column_block * BN + tl.arange(0, BN)
    rows_ok = rows < N
    lanes = tl.arange(0, TM)
    acc = tl.zeros((BN, TM), tl.float32)
    first = column_block % STEPS
    for step in range(STEPS):
        kb = ((first + step) % STEPS) * BW
        cols = kb + tl.arange(0, BW)
        words = tl.load(w_ptr + rows[:, None] * KW + cols[None, :],
                        mask=rows_ok[:, None] & (cols[None, :] < KW), other=_ZERO_WORD)
        groups = kb // 4 + tl.arange(0, GPS)
        scales = tl.load(s_ptr + rows[:, None] * G + groups[None, :],
                         mask=rows_ok[:, None] & (groups[None, :] < G), other=0.0).to(tl.float32)
        for t in tl.static_range(TM):
            token = token_block * TM + t
            live = token < M
            xa, xb, xc, xd, sums = _activation_words(
                x_ptr + token.to(tl.int64) * (K // 2), kb, live, KW, BW, GPS)
            part = tl.sum(_group_total(_word_dot(words, xa, xb, xc, xd, BN, BW), sums, scales,
                                       BN, GPS), axis=1)
            acc = tl.where(lanes[None, :] == t, acc + part[:, None], acc)
    tokens = token_block * TM + lanes
    tl.store(out_ptr + tokens[None, :].to(tl.int64) * N + rows[:, None], acc.to(tl.bfloat16),
             mask=rows_ok[:, None] & (tokens[None, :] < M))


@triton.jit
def vaelor_dense_wmma(x_ptr, w_ptr, s_ptr, out_ptr, M, N: tl.constexpr, K: tl.constexpr,
                      BN: tl.constexpr, GSTEP: tl.constexpr, BM: tl.constexpr,
                      F32_DOT: tl.constexpr = False):
    """A BM x BN output tile by the matrix units, GSTEP 32-value groups per step;
    staggered K walk. F32_DOT multiplies in fp32 - the same products - for
    Triton's CPU interpreter (tests), which mis-multiplies bf16 operands."""
    column_block = tl.program_id(0)
    token_block = tl.program_id(1)
    KW: tl.constexpr = K // 8
    G: tl.constexpr = K // 32
    BK: tl.constexpr = GSTEP * 32
    STEPS: tl.constexpr = G // GSTEP
    rows = column_block * BN + tl.arange(0, BN)
    rows_ok = rows < N
    tokens = token_block * BM + tl.arange(0, BM)
    tokens_ok = tokens < M
    within = tl.arange(0, BK)
    shift = ((within % 8) // 2) * 4 + (within % 2) * 16
    acc = tl.zeros((BM, BN), tl.float32)
    first = column_block % STEPS
    for step in range(STEPS):
        k0 = ((first + step) % STEPS) * BK
        ks = k0 + within
        a = tl.load(x_ptr + tokens[:, None].to(tl.int64) * K + ks[None, :],
                    mask=tokens_ok[:, None] & (ks[None, :] < K), other=0.0)
        words = tl.load(w_ptr + rows[:, None] * KW + (ks // 8)[None, :],
                        mask=rows_ok[:, None] & (ks[None, :] < K), other=_ZERO_WORD)
        nibbles = (words >> shift[None, :]) & 15
        groups = k0 // 32 + tl.arange(0, GSTEP)
        scales = tl.load(s_ptr + rows[:, None] * G + groups[None, :],
                         mask=rows_ok[:, None] & (groups[None, :] < G), other=0.0)
        spread = tl.reshape(tl.broadcast_to(scales.to(tl.float32)[:, :, None], (BN, GSTEP, 32)),
                            (BN, BK))
        weights = ((nibbles.to(tl.float32) - 8.0) * spread).to(tl.bfloat16)
        if F32_DOT:
            acc = tl.dot(a.to(tl.float32), tl.trans(weights.to(tl.float32)), acc=acc)
        else:
            acc = tl.dot(a, tl.trans(weights), acc=acc)
    tl.store(out_ptr + tokens[:, None].to(tl.int64) * N + rows[None, :], acc.to(tl.bfloat16),
             mask=tokens_ok[:, None] & rows_ok[None, :])


def _triton_launch(kernel, plan, a, b_q, scales, out):
    """``(jit function, grid, arguments, named)`` of a 008abd6 Triton decode launch."""
    tokens, k = (int(v) for v in a.shape)
    n = int(b_q.shape[0])
    rows, step, per_program, warps, stages = plan
    grid = (triton.cdiv(n, rows), triton.cdiv(tokens, per_program))
    if kernel == "gemv":
        return vaelor_dense_gemv, grid, (a.view(torch.int32), b_q, scales, out, tokens), dict(
            N=n, K=k, BN=rows, BW=step, TM=per_program, num_warps=warps, num_stages=stages)
    return vaelor_dense_wmma, grid, (a, b_q, scales, out, tokens), dict(
        N=n, K=k, BN=rows, GSTEP=step, BM=per_program, num_warps=warps, num_stages=stages)


def run_decode(a, b_q, scales, out, shape):
    """Launch our decode kernel for one checked call into ``out``."""
    kernel, plan = decode_plan(shape, int(a.shape[0]))
    if kernel == "mx":
        vaelor_dense_mx.run_mx(a, b_q, scales, out, plan)
        return
    jit, grid, arguments, named = _triton_launch(kernel, plan, a, b_q, scales, out)
    jit[grid](*arguments, **named)


def _prompt_launch(a, b_q, scales, out, plan, tokens):
    """``(grid, arguments, named)`` of vLLM's kernel at a "vllm" prompt plan's values."""
    k = int(a.shape[1])
    n = int(b_q.shape[0])
    block_m, block_n, block_k, warps, stages = plan
    return (triton.cdiv(tokens, block_m), triton.cdiv(n, block_n)), (
        a, b_q, scales, scales, out, tokens, n, k, k // 8, k // 32), dict(
        group_size=32, ZP_BIAS=8, HAS_ZP=False, BLOCK_M=block_m, BLOCK_N=block_n,
        BLOCK_K=block_k, num_warps=warps, num_stages=stages)


def _unpack_launch(b_q, scales, scratch, plan):
    """``(grid, arguments, named)`` of :func:`vaelor_dense_unpack` into ``scratch`` [N, K]."""
    n, k = int(b_q.shape[0]), int(b_q.shape[1]) * 8
    rows, words, warps = plan
    return (triton.cdiv(n, rows), triton.cdiv(k // 8, words)), (b_q, scales, scratch), dict(
        N=n, K=k, BN=rows, BW=words, num_warps=warps)


def run_prompt(a, b_q, scales, out, shape, vllm_kernel, scratch=None):
    """One prompt-reading call into ``out``: "mx" the Gluon kernel with a token
    grid (one launch); "wmma" our Triton kernel;
    "dq" unpacks into ``scratch`` (bf16 [N, K], from :func:`run_dense`) and
    multiplies by the bf16 GEMM; "vllm" is vLLM's own kernel (``vllm_kernel``,
    pinned by its source) at our launch values."""
    tokens = int(a.shape[0])
    kernel, plan = prompt_plan(shape, tokens)
    if kernel == "mx":
        vaelor_dense_mx.run_mx(a, b_q, scales, out, decode_plan(shape, plan)[1], plan)
        return
    if kernel == "wmma":
        jit, grid, arguments, named = _triton_launch(kernel, plan, a, b_q, scales, out)
        jit[grid](*arguments, **named)
        return
    if kernel == "dq":
        grid, arguments, named = _unpack_launch(b_q, scales, scratch, plan)
        vaelor_dense_unpack[grid](*arguments, **named)
        torch.mm(a, scratch.t(), out=out)
        return
    grid, arguments, named = _prompt_launch(a, b_q, scales, out, plan, tokens)
    vllm_kernel[grid](*arguments, **named)


def compile_counts(shape):
    """One token count per (kernel, plan, tile set, multiple of 16, one) the layer can launch.

    Triton compiles a kernel per constant set and specialises the token count
    on whether it is a multiple of 16 and whether it is 1, so one of each
    covers every count. The unpacker takes no count: one per plan. An "mx"
    prompt row ("mx-prompt") is the Gluon kernel at its token block's decode
    plan with a token grid, a constant set of its own.
    """
    counts = {}
    for tokens in range(DECODE_FROM, DECODE_TO + 1):
        kernel, plan = decode_plan(shape, tokens)
        if kernel == "vllm":
            continue
        tiles = vaelor_dense_mx.tiles_for(tokens) if kernel == "mx" else ()
        counts.setdefault((kernel, plan, tiles, tokens % 16 == 0, tokens == 1), tokens)
    smallest = DECODE_TO + 1
    for largest, kernel, plan in prompt_plans()[shape]:
        top = min(largest, PREFILL_TO)
        # The row's first multiple of 16 as well (review R83): its two ends need
        # not be one, and its %16 variant then compiled on a served request.
        sixteen = -(-smallest // 16) * 16
        for tokens in (top, top - 1, smallest, smallest + 1, sixteen, sixteen + 1):
            if smallest <= tokens <= top:
                kind = {"vllm": "prompt", "mx": "mx-prompt"}.get(kernel, kernel)
                key = (kernel, plan, (), False, False) if kernel == "dq" else (
                    kind, plan, (), tokens % 16 == 0, False)
                counts.setdefault(key, tokens)
        smallest = top + 1
    return counts


def compile_layer(shape, a, b_q, scales, vllm_kernel):
    """Compile every decode and prompt kernel this layer can launch, running none.

    A compile depends only on the arguments' types, alignment and constants,
    so the leading rows of ONE empty tensor - the largest count's, allocated
    once (review r1: a tensor per count held two 64 MB tensors at once inside
    vLLM's profiling run) - stand in for the activations of every count; the
    unpacker compiles against a one-row scratch the same way.
    """
    out = torch.empty((1, int(b_q.shape[0])), dtype=a.dtype, device=a.device)
    counts = sorted(compile_counts(shape).items(), key=lambda item: item[1])
    stand_in = a.new_empty((counts[-1][1], int(a.shape[1])))
    for (kernel, plan, _tiles, _even, _one), tokens in counts:
        rows = stand_in[:tokens]
        if kernel == "mx":
            vaelor_dense_mx.compile_mx(rows, b_q, scales, out, plan)
        elif kernel == "mx-prompt":  # plan: the token block
            vaelor_dense_mx.compile_mx(rows, b_q, scales, out, decode_plan(shape, plan)[1], plan)
        elif kernel == "dq":
            grid, arguments, named = _unpack_launch(b_q, scales, out, plan)
            vaelor_dense_unpack.warmup(*arguments, grid=grid, **named)
        elif kernel == "prompt":
            grid, arguments, named = _prompt_launch(rows, b_q, scales, out, plan, tokens)
            vllm_kernel.warmup(*arguments, grid=grid, **named)
        else:
            jit, grid, arguments, named = _triton_launch(kernel, plan, rows, b_q, scales, out)
            jit.warmup(*arguments, grid=grid, **named)


#: (shape, token count) pairs already launched outside a graph capture in this
#: process, so their kernels are compiled; a capture of any other stands aside
#: (the expert path's rule, `vaelor_moe_triton.run_experts`).
_WARM = set()
#: Layers whose kernels :func:`compile_layer` compiled in this process.
_COMPILED = set()


def run_dense(a, b_q, scales, vllm_kernel):
    """One checked dense call (`vaelor_moe_kernel.dense_refusal`): ``(output, how it ran)``.

    The output is ``None`` when a graph capture of a count never run eagerly
    stood aside (:data:`ASIDE_IN_CAPTURE`), or - with no line - when the
    count's plan is vLLM's ("vllm"): the caller then uses vLLM's own. The
    weights come as vLLM's HIP op holds them (int8 [N, K/2]) at 1-5 tokens,
    the same bytes as int32 [N, K/8].
    """
    tokens = int(a.shape[0])
    if b_q.dtype == torch.int8:
        b_q = b_q.view(torch.int32)
    n, k = int(b_q.shape[0]), int(a.shape[1])
    shape = DENSE_SHAPES[(n, k)]
    if tokens <= DECODE_TO and decode_plan(shape, tokens)[0] == "vllm":
        return None, ""
    capturing = a.is_cuda and torch.cuda.is_current_stream_capturing()
    if capturing and (shape, tokens) not in _WARM:
        return None, ASIDE_IN_CAPTURE
    if _MX_MISSING is not None:
        say_once(STOOD_ASIDE, (
            "the Gluon decode kernels for 1-64 tokens stand aside because Triton's Gluon "
            "could not be loaded ({}); the 008abd6 Triton kernels serve 6-64 tokens and "
            "vLLM's HIP kernel 1-5.").format(
                _MX_MISSING), key="dense-gluon")
    if not capturing and shape not in _COMPILED:
        compile_layer(shape, a, b_q, scales, vllm_kernel)
        _COMPILED.add(shape)
    out = torch.empty((tokens, n), dtype=a.dtype, device=a.device)
    if tokens <= DECODE_TO:
        run_decode(a, b_q, scales, out, shape)
    else:
        scratch = None
        if prompt_plan(shape, tokens)[0] == "dq":
            scratch = torch.empty((n, k), dtype=a.dtype, device=a.device)
        run_prompt(a, b_q, scales, out, shape, vllm_kernel, scratch)
    if capturing:
        return out, RAN_IN_CAPTURE
    _WARM.add((shape, tokens))
    return out, RAN_EAGER
