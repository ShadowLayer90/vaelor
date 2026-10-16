# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Vaelor contributors.
"""Vaelor's matrix-unit kernel for the 30B's dense 4-bit layers, 1-64 tokens and short prompts (VD-216).

Original Vaelor work, in Triton's Gluon dialect (explicit register layouts),
written from GEAK's measured findings on gfx1151 - not from its code (owner,
2026-10-14) - and from our own measurements on the Z2. No line of it comes
from vLLM, GEAK, Ornith/Ciru or any other project.

**What it does.** One program computes BN output rows of the layer for every
token of the call, so each weight is read from memory once. WK warps split
the K walk: at step ``s`` warp ``w`` takes 32-value group ``s * WK + w``, so a
program's warps read neighbouring 16-byte pieces of each row together. A
program starts its walk at a different step (``program * ROT``), so programs
running at once read different memory channels.

**Dequantisation in the matrix unit's own registers.** RDNA 3's 16x16x16 WMMA
takes each B column in one lane's registers. A lane loads its column's four
packed words of a group and turns each into four bf16 pairs with one shift and
one and-or: ``((word >> 4c) & 0x000F000F) | 0x43004300`` is the pair
``(128 + n, 128 + m)`` exactly (bf16 128.0 has a mantissa step of 1). The
weights' stored order puts value pairs (2j, 2j+1) in one 16-bit half each, so
the pairs come out in K order and are the WMMA operand as they stand - no
shared memory, no conversion. The +128 and the format's -8 come off once per
group: the accumulator starts at ``-136 x`` each row's activation sum (16
``v_dot2`` per lane, then spread to the accumulator's row order with DPP), and
the group's scale multiplies the fp32 group result once.

**Tiles.** Full 16-row tiles; for a remainder of up to 8 rows, a split tile:
RDNA 3 computes a WMMA's even output rows in the lower half-wave and its odd
rows in the upper, each half from its own copy of the operands, so the lower
half takes K 0-15 of the group and the upper K 16-31 of the same 8 rows, and
output rows 2j and 2j+1 add up to row j - 8 rows x 32 values in one WMMA, with
half the dequantisation per lane. The registers that carry this differ
between the half-waves, which the layouts here do not describe: every
operation on them is lane-local (no layout conversion moves them between
lanes), and the GPU tests compare every plan against an fp32 reference.

**Memory.** Every load is a buffer load whose offsets are in range by
construction - :func:`plan_refusal` states the conditions, and
`vaelor_dense_w4` refuses to load a plan table that breaks one: rows ``n < N``
because the grid is ``N / BN``; K steps modulo the step count; activation rows
clamped to ``M - 1`` (a padded row is computed and never stored). Stores are
masked. ``CAP`` int32 of shared memory held for the whole program caps how
many programs share a compute unit (fewer weight streams read memory better,
GEAK's finding, re-measured here); with
``SLDS`` the group scales live there, read per step from shared memory instead
of one 2-byte global load per lane.

**Prompt reading (``TB``, WO1).** With ``TB`` > 0 the grid has a second axis
of ``TB``-row token blocks: each program reads its block's activations and
writes its block's outputs from its own base, with ``M`` the rows left in the
block, so the clamps and masks above hold per block. One launch serves a whole
65-200-token call (`vaelor_dense_w4.PROMPT_PLANS`).
"""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout
from triton.experimental.gluon.language.amd.cdna3 import buffer_load, buffer_store
from triton.experimental.gluon.language.amd.rdna3 import wmma

#: The plan fields, in order: output rows per program, warps on K, the
#: program's start step multiplier, int32 of shared memory held (0 or a power
#: of two), scales staged in shared memory, activations loaded half per lane
#: and completed from the neighbouring lane.
PLAN_FIELDS = ("rows", "warps", "stagger", "held", "staged_scales", "paired")


def tiles_for(tokens):
    """``(full 16-row tiles, extra full tile, 8-row split tile)`` covering ``tokens`` rows.

    The full tiles are one tensor, so their count is a power of two; three is
    two plus the extra tile.
    """
    full, rest = divmod(int(tokens), 16)
    split = 1 if 0 < rest <= 8 else 0
    if rest > 8:
        full += 1
    if full in (0, 1, 2, 4):
        return full, 0, split
    if full == 3:
        return 2, 1, split
    raise ValueError("no tile set covers {} tokens".format(tokens))


def plan_refusal(n, k, tokens, plan):
    """Why ``plan`` cannot serve an ``(n, k)`` call of ``tokens``, or ``None``.

    These are the conditions the kernel's unmasked loads rely on.
    """
    rows, warps, _stagger, held, _staged, _paired = plan
    if n % rows or rows not in (16, 32, 64):
        return "rows per program must divide N"
    if warps not in (2, 4, 8) or k % (32 * warps * 2):
        return "each warp needs an even number of K steps"
    if held and held & (held - 1):
        return "held shared memory must be a power of two"
    if held and held < 32 * warps:
        return "held shared memory must cover one int32 per thread (32 x warps)"
    if not 1 <= int(tokens) <= 64:
        return "decode is 1-64 tokens"
    tiles_for(tokens)
    return None


def _bits(n):
    return [1 << i for i in range(int(n).bit_length() - 1)]


def _linear(regs, lanes, warps, shape):
    rank = len(shape)
    return gl.DistributedLinearLayout(
        reg_bases=regs, lane_bases=lanes,
        warp_bases=[[b] + [0] * (rank - 1) for b in _bits(warps)], block_bases=[], shape=shape)


def layouts(rows, warps, full):
    """The register layouts of one (rows, warps, full tiles) shape of the kernel.

    A lane holds column ``n % 16`` (both half-waves) of the B operand and row
    ``m % 16`` of the A operand; registers hold K, then 16-wide column or row
    tiles; warps split K.
    """
    tiles = max(full, 1)
    by_warp = [[b, 0, 0] for b in _bits(warps)]
    columns = [[0, 0, b] for b in _bits(rows // 16)]

    def accumulator(count):
        return AMDWMMALayout(version=1, transposed=False, warp_bases=by_warp,
                             reg_bases=columns + [[0, b, 0] for b in _bits(count)],
                             instr_shape=[16, 16, 16], rank=3)

    def one_column_tile(count):
        return AMDWMMALayout(version=1, transposed=False, warp_bases=by_warp,
                             reg_bases=[[0, b, 0] for b in _bits(count)],
                             instr_shape=[16, 16, 16], rank=3)

    lane3 = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 0]]
    lane4 = [[0, 1, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 0, 0, 0]]
    column_tiles = [[0, 16 * b, 0, 0] for b in _bits(rows // 16)]
    words = _linear([[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 0], [0, 0, 2, 0]] + column_tiles,
                    lane4, warps, [warps, rows, 4, 4])
    half_words = _linear([[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 0]] + column_tiles,
                         lane4, warps, [warps, rows, 2, 4])

    def activations(count, pairs):
        return _linear([[0, 0, b] for b in _bits(pairs)] + [[0, 16 * b, 0] for b in _bits(count)],
                       lane3, warps, [warps, 16 * count, pairs])

    def spread(count):
        return _linear([[0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0]]
                       + [[0, b, 0, 0, 0] for b in _bits(count)],
                       [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 0, 4], [0, 0, 0, 0, 8],
                        [0, 0, 0, 1, 0]],
                       warps, [warps, count, 8, 2, 16])

    return dict(ACC=accumulator(tiles), ONE=one_column_tile(tiles), ACC1=accumulator(1),
                ONE1=one_column_tile(1), WORDS=words, HWORDS=half_words,
                XP=activations(tiles, 16), XP1=activations(1, 16), XPH=activations(1, 8),
                XPF8=activations(tiles, 8), SPREAD=spread(tiles), SPREAD1=spread(1))


@gluon.jit
def _lane_id(like):
    """Each lane's index in its wave (0-31), shaped like ``like``."""
    return gl.inline_asm_elementwise("v_mbcnt_lo_u32_b32 $0, -1, 0", "=v,v", [like * 0],
                                     dtype=gl.int32, is_pure=True, pack=1)


@gluon.jit
def _as_bf16(pairs):
    """int32 registers holding bf16 pairs, read as the bf16 values (no instruction)."""
    return gl.inline_asm_elementwise("; bf16 pairs of $1 in $0", "=v,0,v", [gl.join(pairs, pairs)],
                                     dtype=gl.bfloat16, is_pure=True, pack=2)


def _row_bias_asm(pairs):
    """-136 x the sum of a lane's ``pairs`` bf16 pairs, then DPP moves putting the
    value of lane 2j (lower half-wave) / 2j+1 (upper) into register j."""
    return gl.constexpr(
        "".join("v_dot2_f32_bf16 $0, ${}, 0xC308C308, {}\n".format(8 + i, "0" if i == 0 else "$0")
                for i in range(pairs))
        + "s_nop 1\nv_mov_b32_dpp $0, $0 row_shl:1 row_mask:0x2 bank_mask:0xf\ns_nop 1\n"
        + "".join("v_mov_b32_dpp ${}, $0 row_share:{} row_mask:0xf bank_mask:0xf\n".format(j, 2 * j)
                  for j in range(1, 8))
        + "v_mov_b32_dpp $0, $0 row_share:0 row_mask:0xf bank_mask:0xf")


_BIAS16 = _row_bias_asm(16)
_BIAS8 = _row_bias_asm(8)
_BIAS16_CONSTRAINTS = gl.constexpr(",".join(["=&v"] * 8 + ["v"] * 16))
_BIAS8_CONSTRAINTS = gl.constexpr(",".join(["=&v"] * 8 + ["v"] * 8))
#: Each lane takes its neighbour's register (lanes 2j <-> 2j+1), four at a time.
_NEIGHBOUR4 = gl.constexpr("s_nop 1\n" + "".join(
    "v_mov_b32_dpp ${}, ${} quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n".format(i, 4 + i)
    for i in range(4)))


@gluon.jit
def _eight(x, D0: gl.constexpr, D1: gl.constexpr):
    """[D0, D1, 8] -> eight [D0, D1] tensors (one register each)."""
    t = gl.reshape(x, [D0, D1, 2, 2, 2])
    t0, t1 = gl.split(t)
    t00, t01 = gl.split(t0)
    t10, t11 = gl.split(t1)
    a, b = gl.split(t00)
    c, d = gl.split(t01)
    e, f = gl.split(t10)
    g, h = gl.split(t11)
    return a, b, c, d, e, f, g, h


@gluon.jit
def _row_bias(x, PAIRS: gl.constexpr, WK: gl.constexpr, T: gl.constexpr, BN: gl.constexpr,
              ONE: gl.constexpr, ACC: gl.constexpr, SPREAD: gl.constexpr):
    """The accumulator's start: -136 x the activation sum each output row reads."""
    RT: gl.constexpr = 16 * T
    if PAIRS == 16:
        lo, hi = gl.split(gl.reshape(x, [WK, RT, 8, 2]))
        a0, a1, a2, a3, a4, a5, a6, a7 = _eight(lo, WK, RT)
        b0, b1, b2, b3, b4, b5, b6, b7 = _eight(hi, WK, RT)
        by_row = gl.inline_asm_elementwise(
            _BIAS16, _BIAS16_CONSTRAINTS,
            [a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7],
            dtype=(gl.float32,) * 8, is_pure=False, pack=1)
    else:
        a0, a1, a2, a3, a4, a5, a6, a7 = _eight(x, WK, RT)
        by_row = gl.inline_asm_elementwise(_BIAS8, _BIAS8_CONSTRAINTS,
                                           [a0, a1, a2, a3, a4, a5, a6, a7],
                                           dtype=(gl.float32,) * 8, is_pure=False, pack=1)
    # Register j of the asm's output is the accumulator's register j: place it
    # there with compile-time selects (no data moves between lanes).
    COLUMN: gl.constexpr = gl.SliceLayout(2, gl.SliceLayout(3, SPREAD))
    jj = gl.arange(0, 8, layout=gl.SliceLayout(
        0, gl.SliceLayout(1, gl.SliceLayout(3, gl.SliceLayout(4, SPREAD)))))
    j5 = gl.expand_dims(gl.expand_dims(gl.expand_dims(gl.expand_dims(jj, 0), 1), 3), 4)
    placed = gl.zeros([WK, T, 8, 2, 16], gl.float32, layout=SPREAD)
    for j in gl.static_range(8):
        value = gl.convert_layout(gl.reshape(by_row[j], [WK, T, 16]), COLUMN)
        placed = gl.where(j5 == j, gl.expand_dims(gl.expand_dims(value, 2), 3), placed)
    bias = gl.convert_layout(gl.reshape(placed, [WK, RT, 16]), ONE)
    if BN >= 32:
        bias = gl.reshape(gl.permute(gl.join(bias, bias), [0, 1, 3, 2]), [WK, RT, 32])
    if BN >= 64:
        bias = gl.reshape(gl.permute(gl.join(bias, bias), [0, 1, 3, 2]), [WK, RT, 64])
    return gl.convert_layout(bias, ACC)


@gluon.jit
def _bf16_pairs(words, shift):
    """Packed words [.., words] -> int32 bf16 pairs (128 + n) [.., words, 4], K order."""
    return ((gl.expand_dims(words, 3) >> shift) & 0x000F000F) | 0x43004300


@gluon.jit
def _with_neighbour(x, LAYOUT: gl.constexpr):
    """[.., 8] own pairs -> [.., 16]: the lane's own, then its neighbour's."""
    other = gl.inline_asm_elementwise(_NEIGHBOUR4, "=&v,=&v,=&v,=&v,v,v,v,v", [x],
                                      dtype=gl.int32, is_pure=True, pack=4)
    D0: gl.constexpr = x.shape[0]
    D1: gl.constexpr = x.shape[1]
    return gl.convert_layout(gl.reshape(gl.permute(gl.join(x, other), [0, 1, 3, 2]),
                                        [D0, D1, 16]), LAYOUT)


@gluon.jit
def _load_step(x_ptr, w_ptr, s_ptr, w_off, s_off, xf_off, x1_off, xh_off, group,
               FT: gl.constexpr, X1: gl.constexpr, SP: gl.constexpr, SLDS: gl.constexpr):
    """One K step's weights, scales (unless staged) and activations, the scalar
    base advanced by ``group``."""
    w = buffer_load(w_ptr + group * 4, w_off)
    s = w
    if SLDS == 0:
        s = buffer_load(s_ptr + group, s_off)
    xf = w
    x1 = w
    xh = w
    if FT > 0:
        xf = buffer_load(x_ptr + group * 16, xf_off)
    if X1:
        x1 = buffer_load(x_ptr + group * 16, x1_off)
    if SP:
        xh = buffer_load(x_ptr + group * 16, xh_off)
    return w, s, xf, x1, xh


@gluon.jit
def _scales(s, staged, step, SLDS: gl.constexpr, SL: gl.constexpr):
    """The step's scales: as loaded, or from shared memory when staged."""
    if SLDS:
        s = staged.index(step).load(SL)
    return s


@gluon.jit
def _step(w, s, xf, x1, xh, af, a1, ah, shift, half_shift, WK: gl.constexpr,
          BN: gl.constexpr, FT: gl.constexpr, X1: gl.constexpr, SP: gl.constexpr,
          PAIRED: gl.constexpr, ACC: gl.constexpr, ONE: gl.constexpr, ACC1: gl.constexpr,
          ONE1: gl.constexpr, SPREAD: gl.constexpr, SPREAD1: gl.constexpr,
          HWORDS: gl.constexpr, XP: gl.constexpr, XP1: gl.constexpr):
    """One 32-value K group into the accumulators of every tile."""
    s1 = gl.expand_dims(s.to(gl.float32), 1)
    if PAIRED:
        if FT > 0:
            xf = _with_neighbour(xf, XP)
        if X1:
            x1 = _with_neighbour(x1, XP1)
    if FT + X1 > 0:
        p = _bf16_pairs(w, shift)
        DB1: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=ACC1, k_width=16)
        b = gl.convert_layout(gl.permute(gl.reshape(_as_bf16(p), [WK, BN, 32]), [0, 2, 1]), DB1)
        if FT > 0:
            DBF: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=ACC, k_width=16)
            DAF: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=ACC, k_width=16)
            bias = _row_bias(xf, 16, WK, FT, BN, ONE, ACC, SPREAD)
            a = gl.convert_layout(gl.reshape(_as_bf16(xf), [WK, 16 * FT, 32]), DAF)
            t = wmma(a, gl.convert_layout(b, DBF), bias)
            af = af + t * gl.convert_layout(s1, ACC)
        if X1:
            DA1: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=ACC1, k_width=16)
            bias = _row_bias(x1, 16, WK, 1, BN, ONE1, ACC1, SPREAD1)
            a = gl.convert_layout(gl.reshape(_as_bf16(x1), [WK, 16, 32]), DA1)
            t = wmma(a, b, bias)
            a1 = a1 + t * s1
        if SP:
            # The split tile's half of the group: words 0-1 below, 2-3 above.
            low, high = gl.split(gl.permute(gl.reshape(w, [WK, BN, 2, 2]), [0, 1, 3, 2]))
            upper = (_lane_id(low) >> 4) != 0
            own = gl.convert_layout(gl.where(upper, high, low), gl.SliceLayout(3, HWORDS))
            ph = gl.reshape(_bf16_pairs(own, half_shift), [WK, BN, 8])
    else:
        ph = gl.reshape(_bf16_pairs(w, shift), [WK, BN, 8])
    if SP:
        DBH: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=ACC1, k_width=16)
        DAH: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=ACC1, k_width=16)
        bh = gl.convert_layout(gl.permute(gl.reshape(_as_bf16(ph), [WK, BN, 16]), [0, 2, 1]), DBH)
        bias = _row_bias(xh, 8, WK, 1, BN, ONE1, ACC1, SPREAD1)
        a = gl.convert_layout(gl.reshape(_as_bf16(xh), [WK, 16, 16]), DAH)
        t = wmma(a, bh, bias)
        ah = ah + t * s1
    return af, a1, ah


@gluon.jit
def _activation_offsets(row0, M, NT: gl.constexpr, K2: gl.constexpr, WK: gl.constexpr,
                        R: gl.constexpr, P: gl.constexpr, MODE: gl.constexpr, XL: gl.constexpr):
    """int32-pair offsets of one tile set's activations; rows clamp to M - 1.

    MODE 0: lane row = tile row, all 16 pairs. MODE 1 (split tile): rows
    row0 + r // 2, the lower half-wave K 0-15 and the upper K 16-31. MODE 2
    (paired): lanes 2j and 2j+1 share a row (2j below, 2j+1 above) and load
    one K half each, the used lane its first.
    """
    ak = gl.arange(0, WK, layout=gl.SliceLayout(1, gl.SliceLayout(2, XL)))
    am = gl.arange(0, R, layout=gl.SliceLayout(0, gl.SliceLayout(2, XL)))
    ae = gl.arange(0, P, layout=gl.SliceLayout(0, gl.SliceLayout(1, XL)))
    am3 = gl.expand_dims(gl.expand_dims(am, 0), 2)
    base = gl.expand_dims(gl.expand_dims(ak * 16, 1), 2) + gl.expand_dims(gl.expand_dims(ae, 0), 1)
    last = M - 1    # the one bound every row form clamps to (a row past it is never stored)
    if MODE == 0:
        return base + gl.minimum(row0 + am3, last) * K2
    upper = _lane_id(am3) >> 4
    if MODE == 1:
        return base + gl.minimum(row0 + am3 // 2, last) * K2 + upper * 8
    rows = row0 + (am3 & -2) + upper
    return base + gl.minimum(rows, last) * K2 + ((am3 & 1) ^ upper) * 8


@gluon.jit
def vaelor_dense_mx(x_ptr, w_ptr, s_ptr, out_ptr, M, N: gl.constexpr, K: gl.constexpr,
                    BN: gl.constexpr, WK: gl.constexpr, FT: gl.constexpr, X1: gl.constexpr,
                    SP: gl.constexpr, ROT: gl.constexpr, CAP: gl.constexpr,
                    SLDS: gl.constexpr, PAIRED: gl.constexpr, TB: gl.constexpr,
                    ACC: gl.constexpr, ONE: gl.constexpr, ACC1: gl.constexpr, ONE1: gl.constexpr,
                    WORDS: gl.constexpr, HWORDS: gl.constexpr, XP: gl.constexpr,
                    XP1: gl.constexpr, XPH: gl.constexpr, XPF8: gl.constexpr,
                    SPREAD: gl.constexpr, SPREAD1: gl.constexpr):
    """BN output rows x M tokens (module docstring). x: int32 view of bf16 [M, K].

    TB > 0 (prompt reading): program (pid, t) takes rows t * TB .. t * TB + TB - 1
    of the call - its own base for x and out, and M becomes the rows left in its
    block, at most TB - so every clamp and store mask below holds per block."""
    G: gl.constexpr = K // 32
    KW: gl.constexpr = K // 8
    K2: gl.constexpr = K // 2
    if TB > 0:
        row_base = gl.program_id(1) * TB
        x_ptr = x_ptr + row_base * K2
        out_ptr = out_ptr + row_base * N
        M = gl.minimum(M - row_base, TB)
    NT: gl.constexpr = G // WK
    RT: gl.constexpr = 16 * (FT if FT > 0 else 1)
    R0: gl.constexpr = 16 * (FT + X1)
    pid = gl.program_id(0)
    n0 = pid * BN
    first = (pid * ROT) % NT

    if FT + X1 > 0:
        WL: gl.constexpr = gl.SliceLayout(3, WORDS)
        wj = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, WL)))
        sc = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, WORDS))))
    else:
        WL: gl.constexpr = gl.SliceLayout(3, HWORDS)
        wj = gl.arange(0, 2, layout=gl.SliceLayout(0, gl.SliceLayout(1, WL)))
        sc = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, HWORDS))))
    wk = gl.arange(0, WK, layout=gl.SliceLayout(1, gl.SliceLayout(2, WL)))
    wn = n0 + gl.arange(0, BN, layout=gl.SliceLayout(0, gl.SliceLayout(2, WL)))
    wn3 = gl.expand_dims(gl.expand_dims(wn, 0), 2)
    w_off = (gl.expand_dims(gl.expand_dims(wk * 4, 1), 2) + wn3 * KW
             + gl.expand_dims(gl.expand_dims(wj, 0), 1))
    if FT + X1 == 0:
        w_off = w_off + (_lane_id(wn3) >> 4) * 2         # the half-wave's own two words
    shift = gl.expand_dims(gl.expand_dims(gl.expand_dims(sc * 4, 0), 1), 2)
    hsc = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, HWORDS))))
    half_shift = gl.expand_dims(gl.expand_dims(gl.expand_dims(hsc * 4, 0), 1), 2)

    SL: gl.constexpr = gl.SliceLayout(1, ACC1)
    sk = gl.arange(0, WK, layout=gl.SliceLayout(1, SL))
    sn = n0 + gl.arange(0, BN, layout=gl.SliceLayout(0, SL))
    s_off = gl.expand_dims(sk, 1) + gl.expand_dims(sn * G, 0)
    staged = 0
    if SLDS:
        staged = gl.allocate_shared_memory(gl.bfloat16, [NT, WK, BN],
                                           gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        for t in gl.static_range(NT):
            staged.index(t).store(buffer_load(s_ptr + t * WK, s_off))
        gl.barrier()

    if PAIRED:
        xf_off = _activation_offsets(0, M, NT, K2, WK, RT, 8, 2, XPF8)
        x1_off = _activation_offsets(16 * FT, M, NT, K2, WK, 16, 8, 2, XPH)
    else:
        xf_off = _activation_offsets(0, M, NT, K2, WK, RT, 16, 0, XP)
        x1_off = _activation_offsets(16 * FT, M, NT, K2, WK, 16, 16, 0, XP1)
    xh_off = _activation_offsets(R0, M, NT, K2, WK, 16, 8, 1, XPH)

    af = gl.zeros([WK, RT, BN], gl.float32, layout=ACC)
    a1 = gl.zeros([WK, 16, BN], gl.float32, layout=ACC1)
    ah = gl.zeros([WK, 16, BN], gl.float32, layout=ACC1)

    # Two steps per turn, each step's loads issued one step ahead.
    step = first
    w0, s0, xf0, x10, xh0 = _load_step(x_ptr, w_ptr, s_ptr, w_off, s_off, xf_off, x1_off,
                                       xh_off, step * WK, FT, X1, SP, SLDS)
    for turn in range(NT // 2 - 1):
        nxt = (2 * turn + 1 + first) % NT
        w1, s1, xf1, x11, xh1 = _load_step(x_ptr, w_ptr, s_ptr, w_off, s_off, xf_off, x1_off,
                                           xh_off, nxt * WK, FT, X1, SP, SLDS)
        af, a1, ah = _step(w0, _scales(s0, staged, step, SLDS, SL), xf0, x10, xh0, af, a1, ah,
                           shift, half_shift, WK, BN, FT, X1, SP, PAIRED, ACC, ONE, ACC1, ONE1,
                           SPREAD, SPREAD1, HWORDS, XP, XP1)
        step = (2 * turn + 2 + first) % NT
        w0, s0, xf0, x10, xh0 = _load_step(x_ptr, w_ptr, s_ptr, w_off, s_off, xf_off, x1_off,
                                           xh_off, step * WK, FT, X1, SP, SLDS)
        af, a1, ah = _step(w1, _scales(s1, staged, nxt, SLDS, SL), xf1, x11, xh1, af, a1, ah,
                           shift, half_shift, WK, BN, FT, X1, SP, PAIRED, ACC, ONE, ACC1, ONE1,
                           SPREAD, SPREAD1, HWORDS, XP, XP1)
    last = (NT - 1 + first) % NT
    w1, s1, xf1, x11, xh1 = _load_step(x_ptr, w_ptr, s_ptr, w_off, s_off, xf_off, x1_off,
                                       xh_off, last * WK, FT, X1, SP, SLDS)
    af, a1, ah = _step(w0, _scales(s0, staged, step, SLDS, SL), xf0, x10, xh0, af, a1, ah,
                       shift, half_shift, WK, BN, FT, X1, SP, PAIRED, ACC, ONE, ACC1, ONE1,
                       SPREAD, SPREAD1, HWORDS, XP, XP1)
    af, a1, ah = _step(w1, _scales(s1, staged, last, SLDS, SL), xf1, x11, xh1, af, a1, ah,
                       shift, half_shift, WK, BN, FT, X1, SP, PAIRED, ACC, ONE, ACC1, ONE1,
                       SPREAD, SPREAD1, HWORDS, XP, XP1)

    if CAP > 0:
        # Held for the whole program, so fewer programs share a compute unit.
        held = gl.allocate_shared_memory(gl.int32, [CAP], gl.SwizzledSharedLayout(1, 1, 1, [0]))
        held.slice(0, 32 * WK).store(gl.zeros([32 * WK], gl.int32,
                                              layout=gl.BlockedLayout([1], [32], [WK], [0])))
    if FT > 0:
        out = gl.sum(af, axis=0)
        OL: gl.constexpr = gl.SliceLayout(0, ACC)
        om = gl.arange(0, RT, layout=gl.SliceLayout(1, OL))
        on = n0 + gl.arange(0, BN, layout=gl.SliceLayout(0, OL))
        buffer_store(out.to(gl.bfloat16), out_ptr, gl.expand_dims(om * N, 1) + gl.expand_dims(on, 0),
                     mask=gl.expand_dims(om < M, 1) & gl.expand_dims(on < N, 0))
    if X1:
        out = gl.sum(a1, axis=0)
        OL1: gl.constexpr = gl.SliceLayout(0, ACC1)
        om = 16 * FT + gl.arange(0, 16, layout=gl.SliceLayout(1, OL1))
        on = n0 + gl.arange(0, BN, layout=gl.SliceLayout(0, OL1))
        buffer_store(out.to(gl.bfloat16), out_ptr, gl.expand_dims(om * N, 1) + gl.expand_dims(on, 0),
                     mask=gl.expand_dims(om < M, 1) & gl.expand_dims(on < N, 0))
    if SP:
        # Output rows 2j and 2j+1 of the split tile are K's two halves of row j.
        out = gl.sum(gl.reshape(gl.sum(ah, axis=0), [8, 2, BN]), axis=1)
        OLH: gl.constexpr = out.type.layout
        om = R0 + gl.arange(0, 8, layout=gl.SliceLayout(1, OLH))
        on = n0 + gl.arange(0, BN, layout=gl.SliceLayout(0, OLH))
        buffer_store(out.to(gl.bfloat16), out_ptr, gl.expand_dims(om * N, 1) + gl.expand_dims(on, 0),
                     mask=gl.expand_dims(om < M, 1) & gl.expand_dims(on < N, 0))


#: Layouts by (rows, warps, full tiles); built once per process.
_LAYOUTS = {}


def _arguments(a, b_q, scales, out, plan, block=0):
    """``(grid, arguments, named)``; ``block`` > 0 splits the call's tokens into
    blocks of that many rows, one grid row each (prompt reading, one launch)."""
    tokens, k = (int(v) for v in a.shape)
    n = int(b_q.shape[0])
    rows, warps, stagger, held, staged, paired = plan
    full, extra, split = tiles_for(block or tokens)
    key = (rows, warps, full)
    if key not in _LAYOUTS:
        _LAYOUTS[key] = layouts(rows, warps, full)
    args = (a.view(torch.int32), b_q, scales, out, tokens)
    named = dict(N=n, K=k, BN=rows, WK=warps, FT=full, X1=extra, SP=split, ROT=stagger,
                 CAP=held, SLDS=int(staged), PAIRED=int(paired) if full + extra else 0,
                 TB=block, num_warps=warps, **_LAYOUTS[key])
    if block:
        return (n // rows, -(-tokens // block)), args, named
    return (n // rows,), args, named


def run_mx(a, b_q, scales, out, plan, block=0):
    """Launch :func:`vaelor_dense_mx` for one checked call into ``out``."""
    grid, args, named = _arguments(a, b_q, scales, out, plan, block)
    vaelor_dense_mx[grid](*args, **named)


def compile_mx(a, b_q, scales, out, plan, block=0):
    """Compile :func:`vaelor_dense_mx` for this call's shape and plan without running it."""
    grid, args, named = _arguments(a, b_q, scales, out, plan, block)
    vaelor_dense_mx.warmup(*args, grid=grid, **named)
