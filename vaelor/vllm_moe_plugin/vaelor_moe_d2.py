# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Vaelor contributors.
"""Vaelor's "d2" expert kernels for 5-40 tokens on gfx1151 (VD-216, phase 2).

Original Vaelor work, written from first principles and from measurements on
gfx1151 (DECISIONS.md VD-216, "Above 16 tokens"); no line of it comes from
vLLM, Ornith/Ciru or any other project.

**Why it is faster than the per-pair kernel** (`vaelor_moe_triton`), which is
bound by vector-ALU work per route pair (about six instructions per weight).
Each 32-bit word of packed weights (eight nibbles) is turned into four bf16
pairs with one shift and one and-or each, using the magic number 0x4300 -
bf16 128.0, whose last mantissa bit is worth 1, so 128 + n is exact - and
multiplied with four ``v_dot2_f32_bf16`` (two products each, fp32
accumulation): about 11 instructions per 8 weights per token. The +128 offset
and the symmetric format's -8 are removed per 32-value group by subtracting
136 x the group's activation sum, and the group scale is applied once to the
fp32 group sum. Both outputs of the inline assembly are early-clobber
(``=&v``): without it the compiler may give the accumulator an input's
register, and the chained dot products read a clobbered input (measured).

The activations are permuted per 8-value block to the order the shift-and-mask
yields - (x0,x4), (x1,x5), (x2,x6), (x3,x7) - by :func:`vaelor_moe_d2_split`,
and the intermediate rows are written in that order directly.

**gfx11 only**: ``v_dot2_f32_bf16`` is an RDNA 3 instruction; the plugin
refuses any other GPU before it gets here (`vaelor_moe_kernel.refusal`).

**Buffers**: views of the call's own workspaces only (:func:`d2_views`, laid
out by `vaelor_moe_kernel.d2_layout`), so nothing is allocated and nothing
waits on the host inside a graph capture. Every load and store is masked; an
expert id outside ``[0, E)`` reads and writes nothing.
"""

import torch
import triton
import triton.language as tl

from vaelor_moe_kernel import (
    EXPERTS, HIDDEN, INTERMEDIATE, TOP_K, d2_layout, d2_plan,
)

_MASK = tl.constexpr(0x000F000F)
_MAGIC = tl.constexpr(0x43004300)
#: 0x88888888: eight nibbles that decode to zero, what a masked weight word reads.
_ZERO_WORD = tl.constexpr(-2004318072)

#: One packed word against one token's four permuted activation words.
_DOT_WORD = tl.constexpr("""
v_and_or_b32 $1, $2, $7, $8
v_dot2_f32_bf16 $0, $1, $3, 0
v_lshrrev_b32 $1, 4, $2
v_and_or_b32 $1, $1, $7, $8
v_dot2_f32_bf16 $0, $1, $4, $0
v_lshrrev_b32 $1, 8, $2
v_and_or_b32 $1, $1, $7, $8
v_dot2_f32_bf16 $0, $1, $5, $0
v_lshrrev_b32 $1, 12, $2
v_and_or_b32 $1, $1, $7, $8
v_dot2_f32_bf16 $0, $1, $6, $0
""")


@triton.jit
def _word_dot(w, xa, xb, xc, xd, BN: tl.constexpr, BW: tl.constexpr):
    """[BN, BW] packed words x one token's permuted activations -> sum of (128 + n) * x, fp32."""
    shape: tl.constexpr = (BN, BW)
    result, _scratch = tl.inline_asm_elementwise(
        _DOT_WORD, "=&v,=&v,v,v,v,v,v,v,v",
        [w, tl.broadcast_to(xa[None, :], shape), tl.broadcast_to(xb[None, :], shape),
         tl.broadcast_to(xc[None, :], shape), tl.broadcast_to(xd[None, :], shape),
         tl.full(shape, _MASK, tl.int32), tl.full(shape, _MAGIC, tl.int32)],
        dtype=(tl.float32, tl.int32), is_pure=True, pack=1)
    return result


@triton.jit
def _bf16_pair_sum(word):
    """An int32 holding two bf16 -> their fp32 sum."""
    low = (word << 16).to(tl.float32, bitcast=True)
    high = (word & -65536).to(tl.float32, bitcast=True)
    return low + high


@triton.jit
def _activation_words(row, kb, live, KW: tl.constexpr, BW: tl.constexpr, GPS: tl.constexpr):
    """The four permuted activation words for weight words kb.., and each group's sum."""
    words = kb + tl.arange(0, BW)
    ok = live & (words < KW)
    xa = tl.load(row + 4 * words, mask=ok, other=0)
    xb = tl.load(row + 4 * words + 1, mask=ok, other=0)
    xc = tl.load(row + 4 * words + 2, mask=ok, other=0)
    xd = tl.load(row + 4 * words + 3, mask=ok, other=0)
    eights = _bf16_pair_sum(xa) + _bf16_pair_sum(xb) + _bf16_pair_sum(xc) + _bf16_pair_sum(xd)
    return xa, xb, xc, xd, tl.sum(tl.reshape(eights, (GPS, 4)), axis=1)


@triton.jit
def _weight_words(w_base, rows, rows_ok, kb, KW: tl.constexpr, BW: tl.constexpr):
    cols = kb + tl.arange(0, BW)
    return tl.load(w_base + rows[:, None] * KW + cols[None, :],
                   mask=rows_ok[:, None] & (cols[None, :] < KW), other=_ZERO_WORD)


@triton.jit
def _scales(s_base, rows, rows_ok, kb, G: tl.constexpr, GPS: tl.constexpr):
    groups = kb // 4 + tl.arange(0, GPS)
    return tl.load(s_base + rows[:, None] * G + groups[None, :],
                   mask=rows_ok[:, None] & (groups[None, :] < G), other=0.0).to(tl.float32)


@triton.jit
def _group_total(products, group_sums, scales, BN: tl.constexpr, GPS: tl.constexpr):
    """Per 32-value group: the products' sum less 136 x the activations' sum, scaled."""
    part = tl.sum(tl.reshape(products, (BN, GPS, 4)), axis=2)
    return (part - 136.0 * group_sums[None, :]) * scales


@triton.jit
def _permuted_at(k):
    """Where value k sits in the per-8 permuted order (x0,x4,x1,x5,x2,x6,x3,x7)."""
    within = k % 8
    return (k - within) + 2 * (within % 4) + within // 4


@triton.jit
def vaelor_moe_d2_split(x_ptr, xp_ptr, M, K: tl.constexpr):
    """One token's activations, copied into the permuted order the dot words read."""
    token = tl.program_id(0)
    k = tl.arange(0, K)
    ok = (token < M) & (k < K)
    values = tl.load(x_ptr + token.to(tl.int64) * K + k, mask=ok, other=0.0)
    tl.store(xp_ptr + token.to(tl.int64) * K + _permuted_at(k), values, mask=ok)


@triton.jit
def vaelor_moe_d2_gate_up(xw_ptr, ids_ptr, w_ptr, s_ptr, inter_ptr, order_ptr, P,
                          E: tl.constexpr, K: tl.constexpr, I: tl.constexpr,
                          TOPK: tl.constexpr, BN: tl.constexpr, BW: tl.constexpr,
                          ORDERED: tl.constexpr):
    """Gate and up for one route pair and BN intermediate rows, then SiLU(gate) * up."""
    if ORDERED:
        slot = tl.program_id(0)
        column_block = tl.program_id(1)
        pair = tl.load(order_ptr + slot, mask=slot < P, other=-1)
    else:
        column_block = tl.program_id(0)
        pair = tl.program_id(1)
    pair_ok = (pair >= 0) & (pair < P)
    pair = tl.where(pair_ok, pair, 0)
    expert = tl.load(ids_ptr + pair, mask=pair_ok, other=-1).to(tl.int64)
    live = pair_ok & (expert >= 0) & (expert < E)
    expert = tl.where(live, expert, 0)
    KW: tl.constexpr = K // 8
    G: tl.constexpr = K // 32
    GPS: tl.constexpr = BW // 4
    rows = column_block * BN + tl.arange(0, BN)
    rows_ok = live & (rows < I)
    w_expert = w_ptr + expert * (2 * I * KW)
    s_expert = s_ptr + expert * (2 * I * G)
    x_row = xw_ptr + (pair // TOPK).to(tl.int64) * (K // 2)
    gate = tl.zeros((BN, GPS), tl.float32)
    up = tl.zeros((BN, GPS), tl.float32)
    for kb in range(0, KW, BW):
        xa, xb, xc, xd, sums = _activation_words(x_row, kb, live, KW, BW, GPS)
        gate += _group_total(_word_dot(_weight_words(w_expert, rows, rows_ok, kb, KW, BW),
                                       xa, xb, xc, xd, BN, BW),
                             sums, _scales(s_expert, rows, rows_ok, kb, G, GPS), BN, GPS)
        up += _group_total(_word_dot(_weight_words(w_expert, rows + I, rows_ok, kb, KW, BW),
                                     xa, xb, xc, xd, BN, BW),
                           sums, _scales(s_expert, rows + I, rows_ok, kb, G, GPS), BN, GPS)
    g = tl.sum(gate, axis=1)
    u = tl.sum(up, axis=1)
    hidden = g / (1.0 + tl.exp(-g)) * u
    tl.store(inter_ptr + pair.to(tl.int64) * I + _permuted_at(rows), hidden.to(tl.bfloat16),
             mask=rows_ok)


@triton.jit
def _down_rows(a_row, w_expert, s_expert, rows, rows_ok, live, K: tl.constexpr,
               BN: tl.constexpr, BW: tl.constexpr):
    """One pair's down projection for BN output rows, fp32 [BN]."""
    KW: tl.constexpr = K // 8
    G: tl.constexpr = K // 32
    GPS: tl.constexpr = BW // 4
    acc = tl.zeros((BN, GPS), tl.float32)
    for kb in range(0, KW, BW):
        xa, xb, xc, xd, sums = _activation_words(a_row, kb, live, KW, BW, GPS)
        acc += _group_total(_word_dot(_weight_words(w_expert, rows, rows_ok, kb, KW, BW),
                                      xa, xb, xc, xd, BN, BW),
                            sums, _scales(s_expert, rows, rows_ok, kb, G, GPS), BN, GPS)
    return tl.sum(acc, axis=1)


@triton.jit
def vaelor_moe_d2_down_token(aw_ptr, ids_ptr, weights_ptr, w_ptr, s_ptr, out_ptr, M,
                             E: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                             TOPK: tl.constexpr, BN: tl.constexpr, BW: tl.constexpr):
    """Down for one token and BN output rows, its experts added with their routing weights."""
    column_block = tl.program_id(0)
    token = tl.program_id(1)
    token_ok = token < M
    rows = column_block * BN + tl.arange(0, BN)
    total = tl.zeros((BN,), tl.float32)
    for j in range(TOPK):
        pair = token * TOPK + j
        expert = tl.load(ids_ptr + pair, mask=token_ok, other=-1).to(tl.int64)
        live = token_ok & (expert >= 0) & (expert < E)
        expert = tl.where(live, expert, 0)
        weight = tl.load(weights_ptr + pair, mask=live, other=0.0).to(tl.float32)
        total += weight * _down_rows(
            aw_ptr + pair.to(tl.int64) * (K // 2), w_ptr + expert * (N * (K // 8)),
            s_ptr + expert * (N * (K // 32)), rows, live & (rows < N), live, K, BN, BW)
    tl.store(out_ptr + token.to(tl.int64) * N + rows, total.to(tl.bfloat16),
             mask=token_ok & (rows < N))


@triton.jit
def vaelor_moe_d2_down_pair(aw_ptr, ids_ptr, weights_ptr, w_ptr, s_ptr, rows_ptr, order_ptr, P,
                            E: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                            BN: tl.constexpr, BW: tl.constexpr):
    """Down for one route pair (expert-ordered slots); its weighted bf16 row, summed later."""
    slot = tl.program_id(0)
    column_block = tl.program_id(1)
    pair = tl.load(order_ptr + slot, mask=slot < P, other=-1)
    pair_ok = (pair >= 0) & (pair < P)
    pair = tl.where(pair_ok, pair, 0)
    expert = tl.load(ids_ptr + pair, mask=pair_ok, other=-1).to(tl.int64)
    live = pair_ok & (expert >= 0) & (expert < E)
    expert = tl.where(live, expert, 0)
    rows = column_block * BN + tl.arange(0, BN)
    rows_ok = live & (rows < N)
    weight = tl.load(weights_ptr + pair, mask=live, other=0.0).to(tl.float32)
    row = weight * _down_rows(
        aw_ptr + pair.to(tl.int64) * (K // 2), w_ptr + expert * (N * (K // 8)),
        s_ptr + expert * (N * (K // 32)), rows, rows_ok, live, K, BN, BW)
    tl.store(rows_ptr + pair.to(tl.int64) * N + rows, row.to(tl.bfloat16), mask=rows_ok)


@triton.jit
def vaelor_moe_d2_sum(rows_ptr, ids_ptr, out_ptr, M, E: tl.constexpr, N: tl.constexpr,
                      TOPK: tl.constexpr, BN: tl.constexpr):
    """A token's output: the sum of its pairs' rows; a pair with an invalid expert adds nothing."""
    column_block = tl.program_id(0)
    token = tl.program_id(1)
    token_ok = token < M
    columns = column_block * BN + tl.arange(0, BN)
    j = tl.arange(0, TOPK)
    pairs = token * TOPK + j
    experts = tl.load(ids_ptr + pairs, mask=token_ok & (j < TOPK), other=-1)
    ok = token_ok & (experts >= 0) & (experts < E)
    values = tl.load(rows_ptr + pairs[:, None].to(tl.int64) * N + columns[None, :],
                     mask=ok[:, None] & (columns[None, :] < N), other=0.0)
    tl.store(out_ptr + token.to(tl.int64) * N + columns,
             tl.sum(values.to(tl.float32), axis=0).to(tl.bfloat16),
             mask=token_ok & (columns < N))


def d2_views(workspace13, workspace2, tokens):
    """``(permuted input, intermediate, order, pair rows)``: workspace views, never allocations.

    The first three at the END of ``workspace13`` (its start is the call's
    output), the pair rows at the head of ``workspace2`` (``None`` when the
    plan's down is per token); `vaelor_moe_kernel.d2_layout` is the sizes.
    """
    pairs = tokens * TOP_K
    tail, head = d2_layout(tokens)
    flat = workspace13.view(-1)
    base = flat.numel() - tail
    xp = flat[base: base + tokens * HIDDEN].view(tokens, HIDDEN)
    base += tokens * HIDDEN
    inter = flat[base: base + pairs * INTERMEDIATE].view(pairs, INTERMEDIATE)
    base += pairs * INTERMEDIATE
    order = flat[base: base + 2 * pairs].view(torch.int32)
    rows = workspace2.view(-1)[:head].view(pairs, HIDDEN) if head else None
    return xp, inter, order, rows


def run_d2(output, hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids,
           workspace13, workspace2, order_pairs):
    """Launch d2 for one checked call into ``output``. ``order_pairs`` is the
    per-pair kernel's stable expert-order pass (`vaelor_moe_triton`)."""
    tokens = int(hidden_states.shape[0])
    pairs = tokens * TOP_K
    (ordered, gate_rows, gate_words, gate_warps), (per_pair, down_rows, down_words,
                                                   down_warps) = d2_plan(tokens)
    xp, inter, order, rows = d2_views(workspace13, workspace2, tokens)
    ids = topk_ids.view(-1)
    weights = topk_weights.view(-1)
    vaelor_moe_d2_split[(tokens,)](hidden_states, xp, tokens, K=HIDDEN, num_warps=4)
    if ordered or per_pair:
        order_pairs[(1,)](ids, order, pairs, E=EXPERTS, PB=triton.next_power_of_2(pairs),
                          num_warps=4)
    grid = (pairs, INTERMEDIATE // gate_rows) if ordered else (INTERMEDIATE // gate_rows, pairs)
    vaelor_moe_d2_gate_up[grid](
        xp.view(torch.int32), ids, w1.view(torch.int32), w1_scale, inter, order, pairs,
        E=EXPERTS, K=HIDDEN, I=INTERMEDIATE, TOPK=TOP_K, BN=gate_rows, BW=gate_words,
        ORDERED=ordered, num_warps=gate_warps, num_stages=2)
    if not per_pair:
        vaelor_moe_d2_down_token[(HIDDEN // down_rows, tokens)](
            inter.view(torch.int32), ids, weights, w2.view(torch.int32), w2_scale, output,
            tokens, E=EXPERTS, K=INTERMEDIATE, N=HIDDEN, TOPK=TOP_K, BN=down_rows,
            BW=down_words, num_warps=down_warps, num_stages=2)
        return
    vaelor_moe_d2_down_pair[(pairs, HIDDEN // down_rows)](
        inter.view(torch.int32), ids, weights, w2.view(torch.int32), w2_scale, rows, order,
        pairs, E=EXPERTS, K=INTERMEDIATE, N=HIDDEN, BN=down_rows, BW=down_words,
        num_warps=down_warps, num_stages=2)
    vaelor_moe_d2_sum[(HIDDEN // 256, tokens)](rows, ids, output, tokens, E=EXPERTS,
                                               N=HIDDEN, TOPK=TOP_K, BN=256, num_warps=2)
