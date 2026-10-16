# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Vaelor contributors.
"""Vaelor's prompt-reading expert (MoE) path: 41 to 8,192 tokens on gfx1151 (VD-216).

Original Vaelor work. GEAK (AMD-AGI/GEAK) measured, on vLLM 0.27's own expert
kernel at prompt sizes, what limits it on this GPU - the int4 dequantisation's
vector work, a tile table with nothing above 32 tokens, and a register spill -
and we use those findings, not its code (owner, 2026-10-14). No line of this
comes from vLLM, GEAK, Ornith/Ciru or any other project.

**The call is laid out exactly as vLLM 0.27 lays out its own** (read in the
pinned image's `TritonWNA16Experts.apply`), so the workspaces hold what they
were sized for and in the same order:

1. the route pairs are grouped by expert into blocks of :data:`BLOCK_ROWS`
   rows by vLLM's own ``moe_align_block_size`` op;
2. gate+up for every pair into ``workspace2`` (pairs x 2I);
3. vLLM's ``silu_and_mul`` op into the START of ``workspace13`` (pairs x I) -
   where the output will be written later, as in vLLM;
4. down for every pair, weighted by its routing weight, into ``workspace2``
   (pairs x H; gate+up's rows are dead by then);
5. vLLM's ``moe_sum`` op adds each token's pairs into the output.

**The grouped GEMM is ours** (:func:`vaelor_moe_prefill_gemm`): one program
per (expert block, BN output rows). The weight tile is the LEFT operand -
[BN, BK] in the checkpoint's own N-first layout - and the activations are
loaded transposed, so the product is acc^T = W . A^T with no transpose of the
dequantised tile. Each 4-bit value becomes ``(n - 8) * scale`` computed exactly
in fp32 and rounded once to bf16 - the value vLLM's kernel multiplies - so the
errors match vLLM's. Short prompts use 16-row blocks, large ones 128-row
blocks. Each plan gives the compiler a register budget through
``waves_per_eu`` - 2 waves per SIMD, or 0 for the compiler's own - the knob
that removed a register spill in GEAK's measurement.

**Compiled before serving.** The first call outside a GPU graph - vLLM's
start-up profiling run - compiles both GEMMs of every plan row
(:func:`compile_plans`), so no prompt waits on a compile.

Every load and store is masked; a pair whose expert id is outside ``[0, E)``
reads and writes nothing.
"""

import os

import torch
import triton
import triton.language as tl

from vaelor_moe_kernel import EXPERTS, GROUP, HIDDEN, INTERMEDIATE, PREFILL_TO, TOP_K

#: Launch plans by token count (measured on the Z2, 2026-10-14). Each row: the
#: largest token count it serves, then a plan for gate+up and one for down:
#: (block rows, output rows per program, values of K per step, warps, stages,
#: waves per SIMD - 0 = the compiler's -, weights as the left operand, column
#: tiles first). Gate+up at 41-256 tokens: 16-row blocks - about 3-16 pairs
#: per expert here, so a 32-row block was mostly padding - with the
#: activations as the left operand and one stage, 4-12% faster per layer than
#: the earlier rows (which were already 6-12% ahead of GEAK at 41-128, kb R41)
#: under both uniform and skewed routing. Down on 16-row blocks only to 128:
#: above it, with skewed routing (the frozen harness's Zipf model; the 30B's
#: own prompt routing is not measured), the 32-row plan stays ahead. Measured cache-honestly on the Z2, 2026-10-15 (WO1), and in the
#: 48-layer stack; errors unchanged (same arithmetic).
PREFILL_PLANS = (
    (128, (16, 32, 256, 2, 1, 0, False, False), (16, 64, 128, 4, 2, 0, False, True)),
    (256, (16, 32, 256, 2, 1, 0, False, False), (32, 32, 128, 2, 2, 2, True, True)),
    (1024, (32, 64, 128, 4, 2, 2, False, True), (32, 32, 128, 2, 2, 2, True, True)),
    (8192, (128, 64, 64, 4, 2, 2, True, True), (128, 64, 64, 4, 2, 2, True, True)),
)

#: Whether Triton runs on its CPU interpreter (tests), which takes no
#: ``waves_per_eu``.
_INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"


def prefill_plan(tokens):
    """The launch plan for a prompt-reading call of ``tokens`` tokens."""
    for largest, gate_up, down in PREFILL_PLANS:
        if tokens <= largest:
            return gate_up, down
    raise ValueError("no prompt-reading plan covers {} tokens".format(tokens))


@triton.jit
def _to_bf16_rne(value):
    """fp32 -> bf16, round to nearest even, by integer arithmetic (finite values)."""
    bits = value.to(tl.uint32, bitcast=True)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _centred(word, AT: tl.constexpr, one_at, offset_at):
    """The nibble already sitting at bit AT (8, 12 or 16), as (n - 8) * scale, exact.

    OR-ing it into the mantissa of 1.0 gives the float 1 + n * 2^(AT-23) with no
    integer-to-float conversion; one fused multiply-add with the per-word
    constants one_at = s * 2^(23-AT) and offset_at = -(2^(23-AT) + 8) * s gives
    (n - 8) * s exactly (both constants and the result are exact in fp32).
    """
    lifted = ((word & (0xF << AT)) | 0x3F800000).to(tl.float32, bitcast=True)
    return _to_bf16_rne(tl.fma(lifted, one_at, offset_at))


@triton.jit
def _exact_tile(words, scales):
    """[words, rows] packed words -> [words, rows, 2, 2, 2] bf16 values.

    ``tl.join`` adds its axis LAST, so the three value axes come out as
    (c, b, a) for value 4a + 2b + c; the caller reverses them.
    """
    low = words << 8      # values 0, 1 at bits 8, 12
    high = words >> 12    # values 5, 6, 7 at bits 8, 12, 16
    one8, off8 = scales * 32768.0, scales * -32776.0
    one12, off12 = scales * 2048.0, scales * -2056.0
    one16, off16 = scales * 128.0, scales * -136.0
    first = tl.join(tl.join(_centred(low, 8, one8, off8), _centred(low, 12, one12, off12)),
                    tl.join(_centred(words, 8, one8, off8), _centred(words, 12, one12, off12)))
    second = tl.join(tl.join(_centred(words, 16, one16, off16), _centred(high, 8, one8, off8)),
                     tl.join(_centred(high, 12, one12, off12), _centred(high, 16, one16, off16)))
    return tl.join(first, second)


@triton.jit
def vaelor_moe_prefill_gemm(a_ptr, w_ptr, s_ptr, c_ptr, sorted_ptr, block_expert_ptr,
                            padded_ptr, weights_ptr, P, EM,
                            E: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                            ROWS_PER_A: tl.constexpr, WEIGHTED: tl.constexpr,
                            GS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                            BK: tl.constexpr, W_LEFT: tl.constexpr, N_FAST: tl.constexpr,
                            F32_DOT: tl.constexpr = False):
    """One expert block of BM sorted pairs times BN output rows of that expert.

    A row is pair // ROWS_PER_A (the token for gate+up, the pair itself for
    down); C row is the pair. With WEIGHTED the pair's routing weight scales it.
    W_LEFT multiplies W . A^T (no transpose of the dequantised tile); otherwise
    A . W^T, whose result is stored along the output row (the plan picks).
    N_FAST launches a block's column tiles back to back, so its BM activation
    rows are read from memory once and reused from cache across the row. F32_DOT
    multiplies in fp32 (the same products: Triton's CPU interpreter, which
    the tests use off the GPU, mis-multiplies bf16 operands).
    """
    if N_FAST:
        column_block = tl.program_id(0)
        block = tl.program_id(1)
    else:
        block = tl.program_id(0)
        column_block = tl.program_id(1)
    padded = tl.load(padded_ptr, mask=EM > 0, other=0)
    if block * BM >= padded:
        return
    if True:
        slots = block * BM + tl.arange(0, BM)
        pairs = tl.load(sorted_ptr + slots, mask=slots < EM, other=P)
        live = (pairs >= 0) & (pairs < P)
        pairs = tl.where(live, pairs, 0).to(tl.int64)
        expert = tl.load(block_expert_ptr + block, mask=block * BM < EM, other=-1).to(tl.int64)
        expert_ok = (expert >= 0) & (expert < E)
        expert = tl.where(expert_ok, expert, 0)
        live = live & expert_ok
        rows = column_block * BN + tl.arange(0, BN)
        rows_ok = expert_ok & (rows < N)
        a_rows = pairs // ROWS_PER_A
        KW: tl.constexpr = K // 8
        G: tl.constexpr = K // GS
        WPT: tl.constexpr = BK // 8
        word_rows = tl.arange(0, WPT)
        # Tiles are [words, rows]: one packed word (8 values in K order, low
        # nibble first) and one group scale per word.
        w_tile = (w_ptr + expert * (N * KW) + rows[None, :].to(tl.int64) * KW
                  + word_rows[:, None])
        s_tile = (s_ptr + expert * (N * G) + rows[None, :].to(tl.int64) * G
                  + (word_rows[:, None] * 8) // GS)
        tile_ok = rows_ok[None, :] & (word_rows[:, None] < WPT)
        ks = tl.arange(0, BK)
        acc_t = tl.zeros((BN, BM), tl.float32)
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(a_ptr + a_rows[:, None] * K + k0 + ks[None, :],
                        mask=live[:, None] & (k0 + ks[None, :] < K), other=0.0)
            words = tl.load(w_tile + k0 // 8, mask=tile_ok, other=-2004318072).to(
                tl.uint32, bitcast=True)
            scales = tl.load(s_tile + k0 // GS, mask=tile_ok, other=0.0).to(tl.float32)
            tile = _exact_tile(words, scales)
            weights = tl.reshape(tl.permute(tile, (1, 0, 4, 3, 2)), (BN, BK))
            if F32_DOT:
                weights = weights.to(tl.float32)
                a = a.to(tl.float32)
            if W_LEFT:
                acc_t = tl.dot(weights, tl.trans(a), acc=acc_t)
            else:
                acc = tl.dot(a, tl.trans(weights), acc=acc)
        if WEIGHTED:
            routing = tl.load(weights_ptr + pairs, mask=live, other=0.0).to(tl.float32)
        else:
            routing = tl.full((BM,), 1.0, tl.float32)
        if W_LEFT:
            tl.store(c_ptr + pairs[None, :] * N + rows[:, None],
                     (acc_t * routing[None, :]).to(tl.bfloat16),
                     mask=live[None, :] & rows_ok[:, None])
        else:
            tl.store(c_ptr + pairs[:, None] * N + rows[None, :],
                     (acc * routing[:, None]).to(tl.bfloat16),
                     mask=live[:, None] & rows_ok[None, :])


def prefill_views(workspace13, workspace2, tokens):
    """``(gate+up rows, SiLU rows, down rows)``: workspace views, as vLLM 0.27 lays them out."""
    pairs = tokens * TOP_K
    flat13 = workspace13.view(-1)
    flat2 = workspace2.view(-1)
    gate_up = flat2[: pairs * 2 * INTERMEDIATE].view(pairs, 2 * INTERMEDIATE)
    silu = flat13[: pairs * INTERMEDIATE].view(pairs, INTERMEDIATE)
    down = flat2[: pairs * HIDDEN].view(tokens, TOP_K, HIDDEN)
    return gate_up, silu, down


def _gemm_launch(a, w, scales, out, sorted_ids, block_experts, padded, weights, pairs,
                 slots, n, k, rows_per_a, weighted, plan):
    """``(grid, arguments, named)`` of one :func:`vaelor_moe_prefill_gemm` launch."""
    rows, columns, step, warps, stages, waves, w_left, n_fast = plan
    extra = {} if _INTERPRETED or not waves else {"waves_per_eu": waves}
    grid = (triton.cdiv(slots, rows), triton.cdiv(n, columns))
    if n_fast:
        grid = grid[::-1]
    return grid, (a, w.view(torch.int32), scales, out, sorted_ids, block_experts, padded,
                  weights, pairs, slots), dict(
        E=EXPERTS, N=n, K=k, ROWS_PER_A=rows_per_a, WEIGHTED=weighted, GS=GROUP, BM=rows,
        BN=columns, BK=step, W_LEFT=w_left, N_FAST=n_fast, F32_DOT=_INTERPRETED,
        num_warps=warps, num_stages=stages, **extra)


def _gemm(a, w, scales, out, sorted_ids, block_experts, padded, weights, pairs, n, k,
          rows_per_a, weighted, plan):
    grid, arguments, named = _gemm_launch(a, w, scales, out, sorted_ids, block_experts, padded,
                                          weights, pairs, sorted_ids.numel(), n, k, rows_per_a,
                                          weighted, plan)
    vaelor_moe_prefill_gemm[grid](*arguments, **named)


#: Whether :func:`compile_plans` has run in this process.
_COMPILED = []


def compile_plans(hidden_states, w1, w2, w1_scale, w2_scale, blocks, views, weights):
    """Compile gate+up and down for every plan row, running none.

    Triton compiles per constant set and specialises the pair and slot counts
    on whether they are multiples of 16: vLLM's alignment pads the slots to
    pairs + experts x (block rows - 1), a multiple of 16 exactly when the pair
    count is, which is when the token count is even - so one even and one odd
    count per row cover every launch. Only the arguments' types and alignment
    matter, so this call's own tensors stand in for every count.
    """
    gate_up, silu, down = views
    for largest, first, second in PREFILL_PLANS:
        top = min(largest, PREFILL_TO)
        for tokens in (top, top - 1):
            pairs = tokens * TOP_K
            for plan, a, w, scales, out, n, k, rows_per_a, weighted in (
                    (first, hidden_states, w1, w1_scale, gate_up, 2 * INTERMEDIATE, HIDDEN,
                     TOP_K, False),
                    (second, silu, w2, w2_scale, down, HIDDEN, INTERMEDIATE, 1, True)):
                slots = pairs + EXPERTS * (plan[0] - 1)
                grid, arguments, named = _gemm_launch(a, w, scales, out, *blocks, weights, pairs,
                                                      slots, n, k, rows_per_a, weighted, plan)
                vaelor_moe_prefill_gemm.warmup(*arguments, grid=grid, **named)


def run_prefill(output, hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids,
                workspace13, workspace2):
    """One checked prompt-reading call into ``output`` (module docstring)."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    import vllm._custom_ops as vllm_ops

    tokens = int(hidden_states.shape[0])
    pairs = tokens * TOP_K
    first, second = prefill_plan(tokens)
    blocks = moe_align_block_size(topk_ids, first[0], EXPERTS, None)
    gate_up, silu, down = prefill_views(workspace13, workspace2, tokens)
    weights = topk_weights.view(-1)
    if not _COMPILED and not torch.cuda.is_current_stream_capturing():
        compile_plans(hidden_states, w1, w2, w1_scale, w2_scale, blocks, (gate_up, silu, down),
                      weights)
        _COMPILED.append(True)
    _gemm(hidden_states, w1, w1_scale, gate_up, *blocks, weights,
          pairs, 2 * INTERMEDIATE, HIDDEN, TOP_K, False, first)
    torch.ops._C.silu_and_mul(silu, gate_up)
    if second[0] != first[0]:
        blocks = moe_align_block_size(topk_ids, second[0], EXPERTS, None)
    _gemm(silu, w2, w2_scale, down, *blocks, weights, pairs,
          HIDDEN, INTERMEDIATE, 1, True, second)
    vllm_ops.moe_sum(down, output)
