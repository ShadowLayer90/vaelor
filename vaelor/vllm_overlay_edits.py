"""Vaelor's own edits to vLLM's attention file, as text placed by line number and hash.

Split out of `vllm_overlay_build` (VD-216), which fetches the upstream diff,
applies its hunks and checks every hash; this module holds only what Vaelor
writes into the patched file and where each piece stands:

* VD-215 - the ``VLLM_TRITON_3D_MAX_Q`` environment read (an overlay replaces
  one file, so ``vllm/envs.py`` cannot register it) and the gfx1151
  prompt-reading launch shape (:data:`PROMPT_TUNING`);
* VD-216 - Vaelor's decode/verify kernel (the whole query group of a sequence
  packed into one program's rows so each KV tile is read once, the context
  split so the grid fills whole rounds of the GPU's WGPs, two tiles in flight),
  a prompt-reading kernel with the same packing, the cheaper bf16 rounding of
  the softmax weights in both, and the dispatch to them for plain causal
  attention only (:data:`VERIFY_TUNING`); -t3 sizes the decode/verify
  kernel's query slots per step and per launch mode, and launches the blocks
  a longer replay may need only into a captured graph.

The VD-216 mechanisms are findings verified on gfx1151 by an outside
optimisation run (owner: use its findings, not its code); the code is
Vaelor's, GPL-2.0-only, and no line of vLLM's is quoted here
(`tests/test_vllm_overlays.NoVllmCodeTests`): where a vLLM line is reused, it
is taken from the file being built, by position and hash. Every value here is
part of the built file, so a change to any of it is a new pinned
``built_sha256`` and a new overlay key (`vllm_overlays`).
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Sequence, Tuple

from .vllm_overlays import OverlayUnavailable

#: The prompt-reading launch shape on gfx1151, Vaelor's own tuning (VD-215).
#: Measured on the Z2 in a side test: a 3,240-token causal prompt at the 30B's
#: shape (32 query heads, 4 KV heads, head 128, block 16, bf16) took 2.91 ms
#: against 12.0 ms with vLLM's default shape, with the same largest error
#: against a dense reference, and confirmed in the live server (first word at
#: 3,260 / 6,440 tokens 1.6 / 3.8 s against 2.1 / 5.3 s). ONE block, so a later
#: measurement changes the values here and nowhere else - and changes the
#: pinned final sha256 (and so the overlay's key) with them.
PROMPT_TUNING = {
    "min_query_tokens": 64,
    "block_m": 128,
    "tile_size_prefill": 32,
    "num_warps": 8,
    "num_stages": 1,
}

#: The decode and verify launch shapes on gfx1151, Vaelor's own tuning (VD-216).
#: ``one_*`` is a single sequence whose context is split across workgroups
#: (one program per WGP: latency-bound); ``split_*`` two or more split
#: sequences (several programs per WGP: -t3 measured 16/4 at 3-17% under
#: 32/8 for 2-4 sequences, and 32/8 still best for one); ``whole_*`` one
#: program per (sequence, KV head) (enough of them to fill the GPU:
#: bandwidth-bound). ``graph_blocks_seqs``: the most sequences a captured
#: graph covers with idle blocks rather than the ceiling's slots (-t3, kb
#: R43). ONE block, like :data:`PROMPT_TUNING`: a change here is a
#: change of the pinned final sha256.
VERIFY_TUNING = {
    "one_tile": 32,
    "one_warps": 8,
    "split_tile": 16,
    "split_warps": 4,
    "whole_tile": 16,
    "whole_warps": 4,
    "max_rows": 128,
    "graph_blocks_seqs": 2,
}

#: The lines of the PATCHED file (pristine plus the PR's hunks, both pinned)
#: that Vaelor's edits stand on: role -> (1-based line number, sha256 of the
#: whole line). A file whose line there hashes differently is refused. The
#: text of these lines lives in vLLM, never here.
#:
#: * ``env`` - the PR's environment read, which Vaelor's replaces;
#: * ``kernels`` - the first line after the segment reducer; Vaelor's VD-216
#:   kernels and their launcher go before it;
#: * ``same_rows`` - vLLM's own bound on query heads per KV head, reused as-is;
#: * ``block_q`` - vLLM's own BLOCK_Q computation, reused as-is;
#: * ``launch`` - the large-head tuning's last line; Vaelor's shape goes after it;
#: * ``tile`` - the large-head tile line; Vaelor's tile goes after it;
#: * ``dispatch`` - the first line of the 2D/3D launch comment; Vaelor's
#:   decode, verify and prompt dispatch goes before it (VD-216).
ANCHORS: Dict[str, Tuple[int, str]] = {
    "env": (33, "9e2d073f01b4eec5ad46e634da810d92a71a7192bdb82c5be80291d6daeab3c6"),
    "kernels": (786, "a820ea3b154ee770eac65fd3ed5a76ec47b54f834d0b52ae72c02f77714fd080"),
    "same_rows": (960, "d825be5ef41cdd8367d5914e19b9b6efccd859eeea565652b35448d87e809381"),
    "block_q": (965, "70e2595894512f73f7cc24f41289641e30e7ef7855f8eb45048167a1b322785a"),
    "launch": (967, "62fbfda001aea2dc497f40cf1e5ea7bce937ca2d128ad044f4a72e8ae397cbc4"),
    "tile": (1000, "617a2ac315a455d4d6f3dd7288a51c9577b7c42e72db7d31f0a27e538a339607"),
    "dispatch": (1048, "b276648dece83d2352f4b221fe404c85095a347ec88df67cf0fcec5b59f933cc"),
}

#: Vaelor's own lines. ``{same_rows}`` and ``{block_q}`` stand for the vLLM
#: lines named above, taken from the file being built.
_ENV_LINES = (
    "# Vaelor (VD-215): read from the environment - an overlay replaces this one\n"
    "# file, so vllm/envs.py cannot register the variable.\n"
    "_TRITON_3D_MAX_Q = int(__import__(\"os\").environ.get(\"VLLM_TRITON_3D_MAX_Q\", \"1\"))\n"
    "\n"
    "\n"
    "def _vaelor_on_gfx1151() -> bool:\n"
    "    # Vaelor (VD-215): vLLM's own arch probe (vllm/platforms/rocm.py).\n"
    "    if not current_platform.is_rocm():\n"
    "        return False\n"
    "    from vllm.platforms.rocm import on_gfx1151\n"
    "\n"
    "    return on_gfx1151()\n"
    "\n"
    "\n"
    "_VAELOR_ON_GFX1151 = _vaelor_on_gfx1151()\n"
)
_LAUNCH_LINES = (
    "\n"
    "    # Vaelor (VD-215): the prompt-reading launch shape measured on gfx1151.\n"
    "    vaelor_prompt_tuned = (\n"
    "        _VAELOR_ON_GFX1151\n"
    "        and not tuned_large_head\n"
    "        and max_seqlen_q > {min_query_tokens}\n"
    "{same_rows}"
    "    )\n"
    "    if vaelor_prompt_tuned:\n"
    "        BLOCK_M = {block_m}\n"
    "{block_q}"
    "        launch_num_warps = {num_warps}\n"
    "        launch_num_stages = {num_stages}\n"
)
_TILE_LINES = (
    "    if vaelor_prompt_tuned:\n"
    "        TILE_SIZE_PREFILL = {tile_size_prefill}\n"
)

# --- VD-216: Vaelor's own decode/verify and prompt kernels -------------------
#
# Written by Vaelor from mechanisms GEAK measured on gfx1151 (owner, VD-216:
# use its findings, not its code). GPL-2.0-only, like the rest of Vaelor; it
# enters vLLM's file only on the machine that builds it.

_KERNEL_LINES = r'''# Vaelor (VD-216), GPL-2.0-only: decode/verify kernels for gfx1151.
#
# P (the softmax weights) is finite and in [0, 1], so rounding it to bf16 by
# adding half an ulp to its fp32 bits and keeping the top half differs from
# round-to-nearest-even only on exact ties, at a fraction of the work.
_VAELOR_CHEAP_P = tl.constexpr(_VAELOR_ON_GFX1151)


@triton.jit
def _vaelor_p_operand(p, v):
    if _VAELOR_CHEAP_P and v.dtype == tl.bfloat16:
        upper = (p.to(tl.uint32, bitcast=True) + 0x8000) >> 16
        return upper.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    else:
        return p.to(v.dtype)


@triton.jit
def _vaelor_pages(pages, tile, tile_to, n_keys, col, TILE: tl.constexpr, PAGE: tl.constexpr):
    # The physical block of each key of one tile; nothing is read past the range.
    key = tile * TILE + col
    wanted = (key < n_keys) & (tile < tile_to)
    return tl.load(pages + key // PAGE, mask=wanted, other=0).to(tl.int64)


@triton.jit
def _vaelor_tile(
    keys_at, values_at, page, tile, tile_to, n_keys, col, dim, in_dim,
    k_block, k_slot, v_block, v_slot, TILE: tl.constexpr, PAGE: tl.constexpr,
):
    # One tile's keys as (DIM, TILE) and values as (TILE, DIM); keys past the
    # sequence (or past this part) are never read and load as zero.
    key = tile * TILE + col
    present = (key < n_keys) & (tile < tile_to)
    within = key % PAGE
    keys = tl.load(
        keys_at + page[None, :] * k_block + within[None, :] * k_slot + dim[:, None],
        mask=in_dim[:, None] & present[None, :], other=0.0,
    )
    values = tl.load(
        values_at + page[:, None] * v_block + within[:, None] * v_slot + dim[None, :],
        mask=present[:, None] & in_dim[None, :], other=0.0,
    )
    return keys, values


@triton.jit
def _vaelor_absorb(
    query, keys, values, tile, newest, run_max, run_sum, acc, col, log2_scale,
    TILE: tl.constexpr,
):
    # One online-softmax step over a loaded tile, in the exp2 domain.
    key = tile * TILE + col
    score = tl.dot(query, keys) * log2_scale
    score = tl.where(key[None, :] <= newest[:, None], score, float("-inf"))
    new_max = tl.maximum(run_max, tl.max(score, axis=1))
    # A row with no visible key yet keeps a finite base, so no -inf - -inf.
    base = tl.where(new_max == float("-inf"), 0.0, new_max)
    weight = tl.exp2(score - base[:, None])
    rescale = tl.exp2(run_max - base)
    run_sum = run_sum * rescale + tl.sum(weight, axis=1)
    acc = acc * rescale[:, None] + tl.dot(_vaelor_p_operand(weight, values), values)
    return new_max, run_sum, acc


@triton.jit
def _vaelor_verify_part(
    q_ptr, k_ptr, v_ptr, out_ptr, part_acc_ptr, part_max_ptr, part_sum_ptr,
    tables_ptr, kv_lens_ptr, starts_ptr, log2_scale,
    table_row: tl.int64, q_tok: tl.int64, q_head: tl.int64,
    o_tok: tl.int64, o_head: tl.int64,
    k_block: tl.int64, k_slot: tl.int64, k_group: tl.int64,
    v_block: tl.int64, v_slot: tl.int64, v_group: tl.int64,
    HEADS: tl.constexpr, GROUP: tl.constexpr, PAGE: tl.constexpr,
    TILE: tl.constexpr, DIM: tl.constexpr, DIM_P2: tl.constexpr,
    ROWS: tl.constexpr, PARTS: tl.constexpr, BLOCKS: tl.constexpr,
):
    # One program per (sequence x block of query slots, KV head, part of the
    # context). Its ROWS pack a block of the sequence's query tokens times
    # every query head sharing this KV head, so each key/value tile is read
    # once for all of them; a block past the sequence's own query returns.
    seq = tl.program_id(0) // BLOCKS
    block = tl.program_id(0) % BLOCKS
    group = tl.program_id(1)
    part = tl.program_id(2)
    first_token = tl.load(starts_ptr + seq)
    n_query = tl.load(starts_ptr + seq + 1) - first_token
    if BLOCKS > 1:
        if block * (ROWS // GROUP) >= n_query:
            return
    n_keys = tl.load(kv_lens_ptr + seq)
    n_tiles = tl.cdiv(n_keys, TILE)
    tiles_each = tl.cdiv(n_tiles, PARTS)
    tile_from = part * tiles_each
    tile_to = tl.minimum(tile_from + tiles_each, n_tiles)

    row = tl.arange(0, ROWS)
    dim = tl.arange(0, DIM_P2)
    col = tl.arange(0, TILE)
    token = block * (ROWS // GROUP) + row // GROUP
    head = group * GROUP + row % GROUP
    live = token < n_query
    in_dim = dim < DIM
    token_at = (first_token + token).to(tl.int64)
    query = tl.load(
        q_ptr + token_at[:, None] * q_tok + head[:, None] * q_head + dim[None, :],
        mask=live[:, None] & in_dim[None, :], other=0.0,
    )
    # Causal: the row of query token t sees keys up to its own position.
    newest = n_keys - n_query + token

    run_max = tl.full([ROWS], float("-inf"), tl.float32)
    run_sum = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, DIM_P2], tl.float32)
    pages = tables_ptr + seq.to(tl.int64) * table_row
    keys_at = k_ptr + group * k_group
    values_at = v_ptr + group * v_group
    # Two tiles in flight, written out by hand (a loop-carried prefetch is
    # copied through the loop and waits early): tile A is used while tile B's
    # loads are outstanding, then the roles swap; block numbers run two
    # tiles further ahead still.
    page_a = _vaelor_pages(pages, tile_from, tile_to, n_keys, col, TILE, PAGE)
    keys_a, values_a = _vaelor_tile(
        keys_at, values_at, page_a, tile_from, tile_to, n_keys, col, dim, in_dim,
        k_block, k_slot, v_block, v_slot, TILE, PAGE)
    page_b = _vaelor_pages(pages, tile_from + 1, tile_to, n_keys, col, TILE, PAGE)
    page_a = _vaelor_pages(pages, tile_from + 2, tile_to, n_keys, col, TILE, PAGE)
    for tile in range(tile_from, tile_to, 2):
        keys_b, values_b = _vaelor_tile(
            keys_at, values_at, page_b, tile + 1, tile_to, n_keys, col, dim, in_dim,
            k_block, k_slot, v_block, v_slot, TILE, PAGE)
        page_b = _vaelor_pages(pages, tile + 3, tile_to, n_keys, col, TILE, PAGE)
        run_max, run_sum, acc = _vaelor_absorb(
            query, keys_a, values_a, tile, newest, run_max, run_sum, acc, col,
            log2_scale, TILE)
        keys_a, values_a = _vaelor_tile(
            keys_at, values_at, page_a, tile + 2, tile_to, n_keys, col, dim, in_dim,
            k_block, k_slot, v_block, v_slot, TILE, PAGE)
        page_a = _vaelor_pages(pages, tile + 4, tile_to, n_keys, col, TILE, PAGE)
        if tile + 1 < tile_to:
            run_max, run_sum, acc = _vaelor_absorb(
                query, keys_b, values_b, tile + 1, newest, run_max, run_sum, acc,
                col, log2_scale, TILE)

    keep = live[:, None] & in_dim[None, :]
    if PARTS == 1:
        whole = acc / tl.where(run_sum == 0.0, 1.0, run_sum)[:, None]
        tl.store(
            out_ptr + token_at[:, None] * o_tok + head[:, None] * o_head + dim[None, :],
            whole.to(out_ptr.dtype.element_ty), mask=keep,
        )
    else:
        # Partials as [token, head, part, DIM_P2]; an empty part stores -inf / 0.
        slot = (token_at * HEADS + head) * PARTS + part
        tl.store(part_acc_ptr + slot[:, None] * DIM_P2 + dim[None, :], acc, mask=keep)
        tl.store(part_max_ptr + slot, run_max, mask=live)
        tl.store(part_sum_ptr + slot, run_sum, mask=live)


@triton.jit
def _vaelor_verify_merge(
    out_ptr, part_acc_ptr, part_max_ptr, part_sum_ptr, starts_ptr,
    o_tok: tl.int64, o_head: tl.int64,
    HEADS: tl.constexpr, DIM: tl.constexpr, DIM_P2: tl.constexpr,
    PARTS: tl.constexpr, PARTS_P2: tl.constexpr, HEAD_BLOCK: tl.constexpr,
    TOKENS: tl.constexpr,
):
    # One program per (sequence x query slot, block of heads): weighs each
    # part by exp2(its max - the overall max) and divides once.
    seq = tl.program_id(0) // TOKENS
    token = tl.program_id(0) % TOKENS
    first_token = tl.load(starts_ptr + seq)
    if token >= tl.load(starts_ptr + seq + 1) - first_token:
        return
    token_at = (first_token + token).to(tl.int64)
    head = tl.program_id(1) * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    part = tl.arange(0, PARTS_P2)
    dim = tl.arange(0, DIM_P2)
    real = (head < HEADS)[:, None] & (part < PARTS)[None, :]
    slot = (token_at * HEADS + head[:, None]) * PARTS + part[None, :]
    maxima = tl.load(part_max_ptr + slot, mask=real, other=float("-inf"))
    top = tl.max(maxima, axis=1)
    top = tl.where(top == float("-inf"), 0.0, top)
    share = tl.exp2(maxima - top[:, None])
    total = tl.sum(tl.load(part_sum_ptr + slot, mask=real, other=0.0) * share, axis=1)
    sums = tl.load(
        part_acc_ptr + slot[:, :, None] * DIM_P2 + dim[None, None, :],
        mask=real[:, :, None] & (dim < DIM)[None, None, :], other=0.0,
    )
    merged = tl.sum(sums * share[:, :, None], axis=1)
    merged = merged / tl.where(total == 0.0, 1.0, total)[:, None]
    tl.store(
        out_ptr + token_at * o_tok + head[:, None] * o_head + dim[None, :],
        merged.to(out_ptr.dtype.element_ty),
        mask=(head < HEADS)[:, None] & (dim < DIM)[None, :],
    )


@triton.jit
def _vaelor_prompt_part(
    q_ptr, k_ptr, v_ptr, out_ptr, tables_ptr, kv_lens_ptr, starts_ptr, log2_scale,
    num_seqs: tl.int32, table_row: tl.int64, q_tok: tl.int64, q_head: tl.int64,
    o_tok: tl.int64, o_head: tl.int64,
    k_block: tl.int64, k_slot: tl.int64, k_group: tl.int64,
    v_block: tl.int64, v_slot: tl.int64, v_group: tl.int64,
    GROUP: tl.constexpr, PAGE: tl.constexpr, TILE: tl.constexpr,
    DIM: tl.constexpr, DIM_P2: tl.constexpr, TOKENS: tl.constexpr, ROWS: tl.constexpr,
):
    # Prompt reading: one program per (block of TOKENS query tokens, KV head),
    # with the same row packing and softmax step as above.
    seq, block, first_token, n_query, n_keys = resolve_seq_and_query_len(
        starts_ptr, kv_lens_ptr, tl.program_id(0), num_seqs, TOKENS)
    if block * TOKENS >= n_query:
        return
    group = tl.program_id(1)
    row = tl.arange(0, ROWS)
    dim = tl.arange(0, DIM_P2)
    col = tl.arange(0, TILE)
    token = block * TOKENS + row // GROUP
    head = group * GROUP + row % GROUP
    live = token < n_query
    in_dim = dim < DIM
    token_at = (first_token + token).to(tl.int64)
    query = tl.load(
        q_ptr + token_at[:, None] * q_tok + head[:, None] * q_head + dim[None, :],
        mask=live[:, None] & in_dim[None, :], other=0.0,
    )
    newest = n_keys - n_query + token
    # The block's last row sees the most keys; no tile past it is read.
    tile_to = tl.cdiv(tl.minimum(n_keys - n_query + block * TOKENS + TOKENS, n_keys), TILE)

    run_max = tl.full([ROWS], float("-inf"), tl.float32)
    run_sum = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, DIM_P2], tl.float32)
    pages = tables_ptr + seq.to(tl.int64) * table_row
    keys_at = k_ptr + group * k_group
    values_at = v_ptr + group * v_group
    # A plain tile loop: at 128 rows, holding a second tile in flight (either
    # way above) spills and measured 12-40% slower on gfx1151.
    for tile in range(0, tile_to):
        page = _vaelor_pages(pages, tile, tile_to, n_keys, col, TILE, PAGE)
        keys, values = _vaelor_tile(
            keys_at, values_at, page, tile, tile_to, n_keys, col, dim, in_dim,
            k_block, k_slot, v_block, v_slot, TILE, PAGE)
        run_max, run_sum, acc = _vaelor_absorb(
            query, keys, values, tile, newest, run_max, run_sum, acc, col,
            log2_scale, TILE)

    whole = acc / tl.where(run_sum == 0.0, 1.0, run_sum)[:, None]
    tl.store(
        out_ptr + token_at[:, None] * o_tok + head[:, None] * o_head + dim[None, :],
        whole.to(out_ptr.dtype.element_ty), mask=live[:, None] & in_dim[None, :],
    )


#: Once-per-process marker lines in the server's log (VD-216, LESSONS 1), the
#: way Vaelor's expert-kernel plugin writes its own: what served, and the first
#: time each reason sent a call back to vLLM's own paths.
_VAELOR_MARKER = "VAELOR_ATTENTION"
_VAELOR_SAID: set = set()


def _vaelor_say(event: str, detail: str) -> None:
    if (event, detail) in _VAELOR_SAID:
        return
    _VAELOR_SAID.add((event, detail))
    logger.info("%s %s: %s", _VAELOR_MARKER, event, detail)


def _vaelor_aside(
    q, k, v, out, *, use_causal, use_per_seq_causal, sliding_window_val, softcap,
    sinks, use_alibi_slopes, use_qq_bias, output_scale, use_mm_prefix, use_rswa,
    use_td, kv_quant_mode, num_queries_per_kv,
) -> str:
    """Why this call is not the plain causal attention Vaelor's kernels
    implement, or ``""`` when it is. The one gate for both kernels; anything
    named here stays on vLLM's own paths (the DFlash drafter's sliding-window
    layers among them)."""
    if not _VAELOR_ON_GFX1151:
        return "not gfx1151"
    if is_batch_invariant:
        return "batch-invariant mode"
    if not use_causal:
        return "non-causal attention"
    if use_per_seq_causal:
        return "per-sequence causal flags"
    if sliding_window_val != 0:
        return "a sliding window"
    if softcap > 0:
        return "a logit softcap"
    if sinks is not None:
        return "attention sinks"
    if use_alibi_slopes:
        return "ALiBi slopes"
    if use_qq_bias:
        return "a query-query bias"
    if output_scale is not None:
        return "an fp8 output scale"
    if use_mm_prefix:
        return "a multimodal prefix"
    if use_rswa:
        return "R-SWA"
    if use_td:
        return "tensor-descriptor loads"
    if kv_quant_mode != KVQuantMode.NONE:
        return "a quantised KV cache"
    if q.dtype not in (torch.bfloat16, torch.float16):
        return "a query that is not bf16 or fp16"
    if k.dtype != q.dtype:
        return "keys in another dtype"
    if v.dtype != q.dtype:
        return "values in another dtype"
    if out.dtype != q.dtype:
        return "an output in another dtype"
    if q.stride(2) != 1:
        return "a strided query head dimension"
    if out.stride(2) != 1:
        return "a strided output head dimension"
    if k.stride(3) != 1:
        return "a strided key head dimension"
    if v.stride(3) != 1:
        return "a strided value head dimension"
    if num_queries_per_kv & (num_queries_per_kv - 1):
        # The kernels pack the group into tl.arange rows: a power of two only.
        return "query heads per KV head not a power of two"
    return ""


_VAELOR_WGPS: list = []


def _vaelor_wgp_count() -> int:
    """The GPU's workgroup-processor count as torch reports it (20 on gfx1151)."""
    if not _VAELOR_WGPS:
        index = torch.cuda.current_device()
        _VAELOR_WGPS.append(int(torch.cuda.get_device_properties(index).multi_processor_count))
    return _VAELOR_WGPS[0]


def _vaelor_parts(pairs: int, room: int) -> int:
    """How many parts each sequence's context is split into.

    Enough programs to fill whole rounds of the GPU's WGPs - a part count
    that leaves a partial round runs that round as a tail - and none once the
    (sequence, KV head) pairs alone fill it (more streams lower the achieved
    bandwidth on this part). ``room`` is what the scratch holds.
    """
    from math import gcd

    wgps = _vaelor_wgp_count()
    if pairs >= wgps:
        return 1
    parts = wgps // gcd(pairs, wgps)
    if parts > room:
        parts = wgps // pairs
    return max(1, min(parts, room))


def _vaelor_slots(query: int, group: int) -> int:
    """Query slots per program for ``query`` tokens: a power of two, at
    least 16 rows (``tl.dot``'s smallest M), and never 32 rows (measured up
    to 3x slower than 16 or 64 on gfx1151)."""
    tokens = triton.next_power_of_2(max(query, 1))
    while tokens * group < 16 or 16 < tokens * group < 64:
        tokens *= 2
    return tokens


def _vaelor_verify(
    q, k, v, out, starts, kv_lens, tables, scale, max_query, num_seqs,
    num_kv_heads, part_acc, part_max, part_sum,
) -> bool:
    """Launch Vaelor's decode/verify kernel; ``False`` (nothing launched) when
    the configured ceiling's rows would not fit.

    Two sizes (VD-216 -t3). The rows each program packs follow THIS step
    (``max_query``), so a single-token step does not pay for the ceiling's
    slots. The programs launched cover the configured ceiling
    (``VLLM_TRITON_3D_MAX_Q``): a block of slots past a sequence's own query
    returns at once, so a FULL graph captured at a shorter step still writes
    every row a replay may carry (review R30). Every other choice is from
    tensor shapes."""
    heads, dim = q.shape[1], q.shape[2]
    group = heads // num_kv_heads
    ceiling = max(_TRITON_3D_MAX_Q, 1)
    if _vaelor_slots(ceiling, group) * group > {max_rows}:
        _vaelor_say("stood-aside", "verify rows over {max_rows}")
        return False
    dim_p2 = triton.next_power_of_2(dim)
    per_part = q.shape[0] * heads
    room = min(part_acc.numel() // max(1, per_part * dim_p2),
               part_max.numel() // max(1, per_part),
               part_sum.numel() // max(1, per_part))
    parts = _vaelor_parts(num_seqs * num_kv_heads, room)
    # Measured on gfx1151 (VD-216 -t3): a split step that fits the smallest
    # dot packs its own slots only; a longer one, and every whole-sequence
    # launch, packs the ceiling's (16 rows ran slower than 64 per whole
    # sequence). A step run now covers its own rows. Inside a graph, blocks
    # past the step cover the ceiling for up to {graph_blocks_seqs} sequences; more
    # sequences pack the ceiling's slots instead (idle blocks there measured
    # 4-11% slower than packing, kb R43).
    tokens = _vaelor_slots(min(max(max_query, 1), ceiling), group)
    if parts == 1 or tokens * group > 16:
        tokens = _vaelor_slots(ceiling, group)
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and num_seqs > {graph_blocks_seqs}:
        tokens = _vaelor_slots(ceiling, group)
    blocks = triton.cdiv(ceiling, tokens) if capturing else 1
    if parts == 1:
        tile, warps = {whole_tile}, {whole_warps}
    elif num_seqs == 1:
        tile, warps = {one_tile}, {one_warps}
    else:
        tile, warps = {split_tile}, {split_warps}
    _vaelor_verify_part[(num_seqs * blocks, num_kv_heads, parts)](
        q, k, v, out, part_acc, part_max, part_sum, tables, kv_lens, starts,
        scale * 1.4426950408889634,
        tables.stride(0), q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
        HEADS=heads, GROUP=group, PAGE=v.shape[1], TILE=tile, DIM=dim,
        DIM_P2=dim_p2, ROWS=tokens * group, PARTS=parts, BLOCKS=blocks,
        num_warps=warps, num_stages=1,
    )
    if parts > 1:
        head_block = max(1, min(8, 128 // triton.next_power_of_2(parts)))
        _vaelor_verify_merge[(num_seqs * blocks * tokens, triton.cdiv(heads, head_block))](
            out, part_acc, part_max, part_sum, starts, out.stride(0), out.stride(1),
            HEADS=heads, DIM=dim, DIM_P2=dim_p2, PARTS=parts,
            PARTS_P2=triton.next_power_of_2(parts), HEAD_BLOCK=head_block,
            TOKENS=blocks * tokens, num_warps=4,
        )
    return True


def _vaelor_prompt(
    q, k, v, out, starts, kv_lens, tables, scale, num_seqs, num_kv_heads,
    block_m, tile, warps, stages,
) -> None:
    """Launch Vaelor's prompt-reading kernel with VD-215's gfx1151 shape."""
    heads, dim = q.shape[1], q.shape[2]
    group = heads // num_kv_heads
    tokens = block_m // group
    _vaelor_prompt_part[(q.shape[0] // tokens + num_seqs, num_kv_heads)](
        q, k, v, out, tables, kv_lens, starts, scale * 1.4426950408889634, num_seqs,
        tables.stride(0), q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
        GROUP=group, PAGE=v.shape[1], TILE=tile, DIM=dim,
        DIM_P2=triton.next_power_of_2(dim), TOKENS=tokens, ROWS=tokens * group,
        num_warps=warps, num_stages=stages,
    )


'''

_DISPATCH_LINES = r'''    # Vaelor (VD-216): on gfx1151, plain causal attention over a bf16/fp16
    # paged cache takes Vaelor's kernels above: decode steps, short verify steps
    # the split-context one, prompts (VD-215's tuned shape) the prompt one.
    # Anything else - a sliding window (the DFlash drafter's layers),
    # non-causal, sinks, softcap, quantised KV - keeps the paths below.
    vaelor_aside = _vaelor_aside(
        q, k, v, out, use_causal=use_causal, use_per_seq_causal=use_per_seq_causal,
        sliding_window_val=sliding_window_val, softcap=softcap, sinks=sinks,
        use_alibi_slopes=use_alibi_slopes, use_qq_bias=use_qq_bias,
        output_scale=output_scale, use_mm_prefix=use_mm_prefix, use_rswa=use_rswa,
        use_td=use_td, kv_quant_mode=kv_quant_mode, num_queries_per_kv=num_queries_per_kv,
    )
    if vaelor_aside:
        _vaelor_say("stood-aside", vaelor_aside)
    elif (
        seq_threshold_3D is not None
        and softmax_segm_output is not None
        and softmax_segm_max is not None
        and softmax_segm_expsum is not None
        and 1 <= max_seqlen_q <= _TRITON_3D_MAX_Q
        and 1 <= num_seqs <= seq_threshold_3D
        and softmax_segm_output.dtype == torch.float32
        and softmax_segm_output.is_contiguous()
        and softmax_segm_max.is_contiguous()
        and softmax_segm_expsum.is_contiguous()
        and _vaelor_verify(
            q, k, v, out, cu_seqlens_q, seqused_k, block_table, softmax_scale,
            max_seqlen_q, num_seqs, num_kv_heads, softmax_segm_output,
            softmax_segm_max, softmax_segm_expsum,
        )
    ):
        _vaelor_say("served", "decode and verify steps")
        return
    elif vaelor_prompt_tuned:
        _vaelor_prompt(
            q, k, v, out, cu_seqlens_q, seqused_k, block_table, softmax_scale,
            num_seqs, num_kv_heads, BLOCK_M, TILE_SIZE_PREFILL, launch_num_warps,
            launch_num_stages,
        )
        _vaelor_say("served", "prompt reading")
        return

'''


NOT_PLACED = (
    "Vaelor's edit to the patched attention file did not find its place in it, "
    "so the file was not built."
)


def _line(lines: Sequence[str], anchor: Tuple[int, str]) -> str:
    """The whole line an anchor names, refused unless its sha256 is the pinned one."""
    number, digest = anchor
    if not 1 <= number <= len(lines):
        raise OverlayUnavailable(NOT_PLACED)
    line = lines[number - 1]
    if hashlib.sha256(line.encode("utf-8")).hexdigest() != digest:
        raise OverlayUnavailable(NOT_PLACED)
    return line


#: How a block stands on its anchor line: in its place, after it, or before it.
_REPLACE, _AFTER, _BEFORE = "replace", "after", "before"


def vaelor_edits(
    text: str, tuning: Optional[dict] = None, verify: Optional[dict] = None,
) -> str:
    """Vaelor's own changes to the patched file.

    VD-215: the environment read and the prompt-reading launch tuning. VD-216:
    Vaelor's decode/verify and prompt kernels with their launchers, and the
    dispatch to them. Placed by :data:`ANCHORS` -
    whole lines, by number and hash, never by searching for text - from the
    bottom of the file up, so each number still names the line it was checked at.
    """
    values = dict(PROMPT_TUNING if tuning is None else tuning)
    shapes = dict(VERIFY_TUNING if verify is None else verify)
    lines = text.splitlines(keepends=True)
    found = {role: _line(lines, anchor) for role, anchor in ANCHORS.items()}
    launch = _LAUNCH_LINES.format(
        same_rows=found["same_rows"], block_q=found["block_q"], **values)
    blocks = {
        "env": (_REPLACE, _ENV_LINES),
        "kernels": (_BEFORE, _KERNEL_LINES.format(**shapes)),
        "launch": (_AFTER, launch),
        "tile": (_AFTER, _TILE_LINES.format(**values)),
        "dispatch": (_BEFORE, _DISPATCH_LINES),
    }
    for role in sorted(blocks, key=lambda name: ANCHORS[name][0], reverse=True):
        number = ANCHORS[role][0]
        how, block = blocks[role]
        start = {_REPLACE: number - 1, _AFTER: number, _BEFORE: number - 1}[how]
        end = number if how == _REPLACE else start
        lines[start:end] = [block]
    return "".join(lines)
