# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Vaelor contributors.
"""Vaelor's decode-sized expert (MoE) kernels for symmetric 4-bit weights (VD-216).

Original Vaelor work, written from first principles and from measurements on
gfx1151 (DECISIONS.md VD-216, "Expert-kernel prototype"); no line of it comes
from vLLM, Ornith/Ciru or any other project. It runs inside the pinned vLLM
image as part of the ``vaelor_moe_kernel`` plugin, loaded at run time from a
read-only folder; it never modifies a file of vLLM's.

**The weights it reads** are the layout vLLM's WNA16 expert path holds for the
cluster 30B (4-bit, group 32, no zero point, value = nibble - 8), N-first and
packed two to a byte, low nibble first:

* gate+up ``w13`` uint8 ``(E, 2*I, H/2)``, its scales bf16 ``(E, 2*I, H/32)``;
* down ``w2`` uint8 ``(E, H, I/2)``, its scales bf16 ``(E, H, I/32)``.

**Two launches per layer, three when the pairs are ordered.** One program per
(route pair, block of intermediate columns) computes gate and up together,
dequantising the 4-bit values in registers, summing each 32-value group in
fp32 and applying its scale once, then SiLU(gate) * up, written as bf16. One
program per (token, block of output columns) then runs the down projection for
that token's eight experts and adds them with the routing weights straight
into the layer's output. From eight tokens up the gate+up programs walk the
pairs in expert order, so pairs that share an expert re-read its rows from
cache. There is no 16-row padding and no alignment step.

**Safe inside GPU-graph capture.** Nothing here synchronises with the host or
allocates: the intermediate rows and the pair order live in the call's own
``workspace2`` (`plan_workspace`), and every grid and constant depends only
on the token count, which is fixed for a captured graph.

**Every load and store is masked**, and an expert id outside ``[0, E)`` reads
nothing and writes nothing (a GPU-side guard, VD-216's incident: a side test
that read an unchecked index faulted the GPU). The host-side index proof
(:func:`check_call`) runs only for tests and under
``VAELOR_MOE_KERNEL_CHECK=1``, never in the hot path, because it reads values
back to the host.
"""

import os

import torch
import triton
import triton.language as tl

from vaelor_moe_kernel import (
    ASIDE_IN_CAPTURE, CHECK_VARIABLE, D2_FROM, EXPERTS, GROUP, HIDDEN, INTERMEDIATE,
    PREFILL_TO, RAN_EAGER, RAN_IN_CAPTURE, TOP_K, uses_d2, uses_hip, uses_prefill,
)
from vaelor_moe_d2 import run_d2
import vaelor_moe_d2_hip
from vaelor_moe_prefill import run_prefill

#: This kernel's launch plans, for the counts below `D2_FROM` (d2 serves the
#: rest, `vaelor_moe_d2`), measured on gfx1151 (side containers, 2026-10-12).
#: Each row: the largest token count it serves; gate+up (pairs in expert
#: order, rows per program, packed bytes per step, warps); down (rows per
#: program, packed bytes per step, warps).
PLANS = (
    (1, (False, 4, 512, 8), (64, 128, 4)),
    (D2_FROM - 1, (False, 4, 512, 8), (32, 128, 2)),
)

#: A packed byte whose two nibbles both decode to zero (8 - 8), the value a
#: masked weight load reads. A constexpr: a jitted kernel reads no other global.
_ZERO_PAIR = tl.constexpr(0x88)


def plan_for(tokens):
    """The launch plan for ``tokens`` (1..D2_FROM - 1): the first row that covers it."""
    for largest, gate_up, down in PLANS:
        if tokens <= largest:
            return gate_up, down
    raise ValueError("no launch plan covers {} tokens".format(tokens))


@triton.jit
def _group_sums(w_base, s_base, rows, rows_ok, kb, x_low, x_high,
                KH: tl.constexpr, G: tl.constexpr, BN: tl.constexpr,
                BKH: tl.constexpr, HG: tl.constexpr, GPS: tl.constexpr):
    """Scaled 32-value group sums of BN weight rows against one activation slice.

    ``w_base``/``s_base`` point at one expert's packed rows and scales;
    ``kb`` is the first packed column of the slice. Masked rows read the zero
    pair and a zero scale, so they add nothing.
    """
    cols = kb + tl.arange(0, BKH)
    groups = kb // HG + tl.arange(0, GPS)
    packed = tl.load(
        w_base + rows[:, None] * KH + cols[None, :],
        mask=rows_ok[:, None] & (cols[None, :] < KH), other=_ZERO_PAIR,
    )
    low = (packed & 15).to(tl.float32) - 8.0
    high = (packed >> 4).to(tl.float32) - 8.0
    products = low * x_low[None, :] + high * x_high[None, :]
    sums = tl.sum(tl.reshape(products, (BN, GPS, HG)), axis=2)
    scales = tl.load(
        s_base + rows[:, None] * G + groups[None, :],
        mask=rows_ok[:, None] & (groups[None, :] < G), other=0.0,
    )
    return sums * scales.to(tl.float32)


@triton.jit
def _activation_halves(row_base, kb, live, LENGTH: tl.constexpr, BKH: tl.constexpr):
    """The even and odd activations of one packed slice, as fp32 (zero when masked)."""
    offsets = 2 * kb + tl.arange(0, 2 * BKH)
    values = tl.load(row_base + offsets, mask=live & (offsets < LENGTH), other=0.0)
    return tl.split(tl.reshape(values.to(tl.float32), (BKH, 2)))


@triton.jit
def vaelor_moe_order_pairs(ids_ptr, order_ptr, P, E: tl.constexpr, PB: tl.constexpr):
    """The route pairs in expert order, stable, as a permutation of ``[0, P)``.

    An expert id outside ``[0, E)`` sorts after every valid one, so the
    permutation never holds a value outside the pairs.
    """
    slots = tl.arange(0, PB)
    live = slots < P
    ids = tl.load(ids_ptr + slots, mask=live, other=E).to(tl.int32)
    ids = tl.where((ids >= 0) & (ids < E), ids, E)
    keys = tl.sort(ids * PB + slots)
    tl.store(order_ptr + slots, keys % PB, mask=live)


@triton.jit
def vaelor_moe_gate_up(x_ptr, ids_ptr, w_ptr, s_ptr, inter_ptr, order_ptr, P,
                       E: tl.constexpr, K: tl.constexpr, I: tl.constexpr,
                       TOPK: tl.constexpr, GS: tl.constexpr, BN: tl.constexpr,
                       BKH: tl.constexpr, ORDERED: tl.constexpr):
    """Gate and up for one route pair and BN intermediate columns, then SiLU(gate) * up."""
    if ORDERED:
        slot = tl.program_id(0)
        column_block = tl.program_id(1)
        pair = tl.load(order_ptr + slot, mask=slot < P, other=P)
    else:
        column_block = tl.program_id(0)
        pair = tl.program_id(1)
    pair_ok = (pair >= 0) & (pair < P)
    pair = tl.where(pair_ok, pair, 0)
    expert = tl.load(ids_ptr + pair, mask=pair_ok, other=-1).to(tl.int64)
    live = pair_ok & (expert >= 0) & (expert < E)
    expert = tl.where(live, expert, 0)
    KH: tl.constexpr = K // 2
    G: tl.constexpr = K // GS
    HG: tl.constexpr = GS // 2
    GPS: tl.constexpr = BKH // HG
    rows = column_block * BN + tl.arange(0, BN)
    rows_ok = live & (rows < I)
    w_expert = w_ptr + expert * (2 * I * KH)
    s_expert = s_ptr + expert * (2 * I * G)
    x_row = x_ptr + (pair // TOPK).to(tl.int64) * K
    gate = tl.zeros((BN, GPS), tl.float32)
    up = tl.zeros((BN, GPS), tl.float32)
    for kb in range(0, KH, BKH):
        x_low, x_high = _activation_halves(x_row, kb, live, K, BKH)
        gate += _group_sums(w_expert, s_expert, rows, rows_ok, kb, x_low, x_high,
                            KH, G, BN, BKH, HG, GPS)
        up += _group_sums(w_expert, s_expert, rows + I, rows_ok, kb, x_low, x_high,
                          KH, G, BN, BKH, HG, GPS)
    g = tl.sum(gate, axis=1)
    u = tl.sum(up, axis=1)
    hidden = g / (1.0 + tl.exp(-g)) * u
    tl.store(inter_ptr + pair.to(tl.int64) * I + rows, hidden.to(tl.bfloat16), mask=rows_ok)


@triton.jit
def vaelor_moe_down_token(inter_ptr, ids_ptr, weights_ptr, w_ptr, s_ptr, out_ptr, M,
                          E: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                          TOPK: tl.constexpr, GS: tl.constexpr, BN: tl.constexpr,
                          BKH: tl.constexpr):
    """Down for one token and BN output columns, its experts added with their routing weights."""
    column_block = tl.program_id(0)
    token = tl.program_id(1)
    token_ok = token < M
    KH: tl.constexpr = K // 2
    G: tl.constexpr = K // GS
    HG: tl.constexpr = GS // 2
    GPS: tl.constexpr = BKH // HG
    rows = column_block * BN + tl.arange(0, BN)
    total = tl.zeros((BN,), tl.float32)
    for j in range(TOPK):
        pair = token * TOPK + j
        expert = tl.load(ids_ptr + pair, mask=token_ok, other=-1).to(tl.int64)
        live = token_ok & (expert >= 0) & (expert < E)
        expert = tl.where(live, expert, 0)
        weight = tl.load(weights_ptr + pair, mask=live, other=0.0).to(tl.float32)
        rows_ok = live & (rows < N)
        w_expert = w_ptr + expert * (N * KH)
        s_expert = s_ptr + expert * (N * G)
        a_row = inter_ptr + pair.to(tl.int64) * K
        sums = tl.zeros((BN, GPS), tl.float32)
        for kb in range(0, KH, BKH):
            a_low, a_high = _activation_halves(a_row, kb, live, K, BKH)
            sums += _group_sums(w_expert, s_expert, rows, rows_ok, kb, a_low, a_high,
                                KH, G, BN, BKH, HG, GPS)
        total += weight * tl.sum(sums, axis=1)
    tl.store(out_ptr + token.to(tl.int64) * N + rows, total.to(tl.bfloat16),
             mask=token_ok & (rows < N))


def plan_workspace(workspace2, tokens):
    """``(intermediate, order)``: views of ``workspace2``, never an allocation.

    The intermediate rows (pairs x I, bf16) first, then the pair order (pairs
    int32) right after them. `vaelor_moe_kernel.qualifies` has already proved
    the workspace is large enough and does not overlap the output.
    """
    pairs = tokens * TOP_K
    flat = workspace2.view(-1)
    intermediate = flat[: pairs * INTERMEDIATE].view(pairs, INTERMEDIATE)
    order = flat[pairs * INTERMEDIATE: pairs * INTERMEDIATE + 2 * pairs].view(torch.int32)
    return intermediate, order


def check_call(hidden_states, topk_ids, topk_weights):
    """The host-side index proof: tests and ``VAELOR_MOE_KERNEL_CHECK=1`` only.

    It reads the ids back to the host, which synchronises, so it is never
    run while a GPU graph is being captured.
    """
    tokens = hidden_states.shape[0]
    ids = topk_ids.reshape(-1).to("cpu")
    if not 1 <= tokens <= PREFILL_TO or ids.numel() != tokens * TOP_K:
        raise ValueError("the expert call's token count is outside the kernel's range")
    if int(ids.min()) < 0 or int(ids.max()) >= EXPERTS:
        raise ValueError("an expert id is outside the layer's experts")
    if topk_weights.numel() != ids.numel():
        raise ValueError("the routing weights do not match the expert ids")


#: The token counts already launched outside a graph capture in this process,
#: so their kernels are compiled; a capture of any other count stands aside.
_WARM = set()


def run_experts(output, hidden_states, w1, w2, w1_scale, w2_scale,
                topk_weights, topk_ids, workspace13, workspace2):
    """Run one qualifying expert call into ``output``; say how it ran.

    :data:`RAN_EAGER` or :data:`RAN_IN_CAPTURE` when the kernels were
    launched (recorded into the graph being captured, in the second case),
    :data:`ASIDE_IN_CAPTURE` when it stood aside and launched nothing.

    The caller (`vaelor_moe_kernel`) has checked every shape, type and
    layout. The one thing decided here is the compile state: a graph capture
    of a token count this process has never run eagerly stands aside, so no
    kernel is ever compiled while a stream is capturing. This is the rare
    case, not the default: vLLM 0.27 runs every capture size eagerly first
    (``cudagraph_num_of_warmups`` 1, read in the pinned image, and the newer
    runner's ``capture`` warms each size before it records it), so on the
    served path every count of up to 40 tokens is warm when its graph is
    captured and the kernels are recorded into it. From `D2_FROM` tokens the
    call is d2's (`vaelor_moe_d2.run_d2`), on the same rule.
    """
    tokens = int(hidden_states.shape[0])
    capturing = hidden_states.is_cuda and torch.cuda.is_current_stream_capturing()
    if capturing and tokens not in _WARM:
        return ASIDE_IN_CAPTURE
    if not capturing and os.environ.get(CHECK_VARIABLE) == "1":
        check_call(hidden_states, topk_ids, topk_weights)
    if uses_hip(tokens) and vaelor_moe_d2_hip.run_hip(
            output, hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids,
            workspace13, capturing):
        return _ran(capturing, tokens)
    if uses_prefill(tokens):
        run_prefill(output, hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids,
                    workspace13, workspace2)
        return _ran(capturing, tokens)
    if uses_d2(tokens):
        run_d2(output, hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids,
               workspace13, workspace2, vaelor_moe_order_pairs)
        return _ran(capturing, tokens)
    pairs = tokens * TOP_K
    intermediate, order = plan_workspace(workspace2, tokens)
    ids = topk_ids.view(-1)
    weights = topk_weights.view(-1)
    (ordered, gate_rows, gate_bytes, gate_warps), (down_rows, down_bytes, down_warps) = plan_for(tokens)
    if ordered:
        vaelor_moe_order_pairs[(1,)](
            ids, order, pairs, E=EXPERTS, PB=triton.next_power_of_2(pairs), num_warps=4,
        )
        grid = (pairs, INTERMEDIATE // gate_rows)
    else:
        grid = (INTERMEDIATE // gate_rows, pairs)
    vaelor_moe_gate_up[grid](
        hidden_states, ids, w1, w1_scale, intermediate, order, pairs,
        E=EXPERTS, K=HIDDEN, I=INTERMEDIATE, TOPK=TOP_K, GS=GROUP,
        BN=gate_rows, BKH=gate_bytes, ORDERED=ordered, num_warps=gate_warps, num_stages=2,
    )
    vaelor_moe_down_token[(HIDDEN // down_rows, tokens)](
        intermediate, ids, weights, w2, w2_scale, output, tokens,
        E=EXPERTS, K=INTERMEDIATE, N=HIDDEN, TOPK=TOP_K, GS=GROUP,
        BN=down_rows, BKH=down_bytes, num_warps=down_warps, num_stages=2,
    )
    return _ran(capturing, tokens)


def _ran(capturing, tokens):
    """How a launched call ran; an eager one marks its count compiled."""
    if capturing:
        return RAN_IN_CAPTURE
    _WARM.add(tokens)
    return RAN_EAGER
