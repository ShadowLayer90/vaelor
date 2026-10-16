// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Vaelor contributors.
//
// Vaelor's "d2" expert kernels in HIP, for the cluster 30B's expert layer at
// decode and verify sizes on gfx1151 (DECISIONS.md VD-216).
//
// Original Vaelor work. It does the same arithmetic as the Triton d2 kernels
// in vaelor_moe_d2.py - int4 turned into bf16 pairs by a magic number and
// multiplied with v_dot2_f32_bf16, the -136 x group-sum correction, one scale
// per 32-value group, fp32 accumulation - in two launches instead of up to
// five: no permuted copy of the input, no pair-ordering pass, and the down
// projection summed per token in fp32 rather than per pair in bf16.
//
// The layer, as vLLM 0.27 hands it over (vaelor_moe_kernel.refusal checks
// every shape, type and buffer before this is reached):
//   x        [tokens, 2048] bf16          w13 [128, 1536, 1024] uint8 (rows N-first,
//   ids      [tokens, 8]   int32/int64          two nibbles per byte, low first)
//   weights  [tokens, 8]   fp32           s13 [128, 1536, 64] bf16 (gate rows, then up)
//   out      [tokens, 2048] bf16          w2  [128, 2048, 384] uint8, s2 [128, 2048, 24] bf16
// value = nibble - 8 (symmetric, no zero points). An expert id outside
// [0, 128) contributes nothing and reads nothing.
//
// Built once per machine, as root, by Vaelor's installer from this exact
// file (its sha256 is pinned in vaelor_moe_d2_hip.py) into a code object the
// plugin loads with hipModuleLoadData after checking its sha256 against the
// build's record. The launch shapes are fixed here and mirrored by the
// plugin's constants (a test reads them out of this file).

#include <hip/hip_runtime.h>
#include <stdint.h>

#define V_EXPERTS 128
#define V_HIDDEN 2048
#define V_INTER 768
#define V_TOPK 8

// gate+up: VG_WAVES waves per block, each owning VG_ROWS intermediate rows
// (their gate row and their up row).
#define VG_WAVES 16
#define VG_ROWS 2
#define VG_BLOCK_ROWS (VG_WAVES * VG_ROWS)
// down: VD_WAVES waves per block; a template parameter says how many lanes
// share one output row (2 or 4: the plugin picks by token count).
#define VD_WAVES 8
#define VD_GROUPS (V_INTER / 32)  // 24 groups of 32 values in a down row

static_assert(V_INTER % VG_BLOCK_ROWS == 0, "gate+up rows must tile the intermediate size");

typedef __bf16 v_bf16x2 __attribute__((ext_vector_type(2)));

// Two bf16 x two bf16, added to an fp32 accumulator (v_dot2_f32_bf16).
__device__ __forceinline__ float v_dot2(uint32_t a, uint32_t b, float acc) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(v_bf16x2, a),
                                         __builtin_bit_cast(v_bf16x2, b), acc, false);
}

// bf16 (1, 1): dotted with a pair, the pair's sum in fp32.
#define V_BF16_ONES 0x3F803F80u

// fp32 -> bf16, round to nearest even, as Triton's .to(tl.bfloat16) does.
__device__ __forceinline__ uint16_t v_to_bf16(float value) {
  uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((bits >> 16) | 0x0040u);
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return (uint16_t)(bits >> 16);
}

__device__ __forceinline__ float v_from_bf16(uint16_t half) {
  return __uint_as_float(((uint32_t)half) << 16);
}

// Eight bf16 values (one uint4, natural order x0..x7) regrouped into the
// pairs the nibble unpack yields: (x0,x4), (x1,x5), (x2,x6), (x3,x7).
__device__ __forceinline__ uint4 v_pair_up(uint4 v) {
  uint4 p;
  p.x = (v.x & 0xFFFFu) | (v.z << 16);
  p.y = (v.x >> 16) | (v.z & 0xFFFF0000u);
  p.z = (v.y & 0xFFFFu) | (v.w << 16);
  p.w = (v.y >> 16) | (v.w & 0xFFFF0000u);
  return p;
}

// The sum of a uint4's eight bf16 values, in fp32.
__device__ __forceinline__ float v_sum8(uint4 v, float acc) {
  acc = v_dot2(v.x, V_BF16_ONES, acc);
  acc = v_dot2(v.y, V_BF16_ONES, acc);
  acc = v_dot2(v.z, V_BF16_ONES, acc);
  return v_dot2(v.w, V_BF16_ONES, acc);
}

// One packed word (eight nibbles) against its eight paired activations: the
// sum of (128 + nibble) * x. 0x43004300 is bf16 (128, 128), whose last
// mantissa bit is worth 1, so OR-ing a nibble in gives 128 + n exactly.
__device__ __forceinline__ float v_word(uint32_t word, uint4 pairs, float acc) {
  acc = v_dot2((word & 0x000F000Fu) | 0x43004300u, pairs.x, acc);
  acc = v_dot2(((word >> 4) & 0x000F000Fu) | 0x43004300u, pairs.y, acc);
  acc = v_dot2(((word >> 8) & 0x000F000Fu) | 0x43004300u, pairs.z, acc);
  return v_dot2(((word >> 12) & 0x000F000Fu) | 0x43004300u, pairs.w, acc);
}

// One 32-value group: four packed words against four uint4 of paired
// activations, starting from ``acc`` (the group's -136 x sum, so the result
// is the sum of (nibble - 8) * x).
__device__ __forceinline__ float v_group(uint4 words, const uint4* pairs, float acc) {
  acc = v_word(words.x, pairs[0], acc);
  acc = v_word(words.y, pairs[1], acc);
  acc = v_word(words.z, pairs[2], acc);
  return v_word(words.w, pairs[3], acc);
}

// Sum over the 32 lanes of a wave, every lane receiving the total: within a
// row of 16 lanes by DPP (swap neighbours, swap pairs, rotate by 4, rotate by
// 8), then across the two rows by permlanex16.
__device__ __forceinline__ float v_wave_total(float value) {
  value += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(value), 0xB1, 0xF, 0xF, false));
  value += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(value), 0x4E, 0xF, 0xF, false));
  value += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(value), 0x124, 0xF, 0xF, false));
  value += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(value), 0x128, 0xF, 0xF, false));
  value += __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(value), 0x76543210u,
                                                       0xFEDCBA98u, false, false));
  return value;
}

// ---- gate + up -------------------------------------------------------------
// Grid (pairs, 768 / VG_BLOCK_ROWS). Pairs vary fastest, so the blocks in
// flight together read the rows of the experts they share from L2.
// A 2,048-value row is 1,024 bytes = 64 groups of 16 bytes; lane l takes the
// adjacent groups 2l and 2l + 1 - one 32-byte read of weights and one 4-byte
// read of scales per row - so a row is one contiguous 1,024-byte read across
// the wave, and the lane's 64 activations stay in registers for every row the
// wave owns. One lane per row writes SiLU(gate) * up, rounded to bf16, at
// [pair, row] of ``inter``.
template <typename Id>
__device__ __forceinline__ void v_gate_up(const uint16_t* __restrict__ x, const Id* __restrict__ ids,
                                          const uint8_t* __restrict__ w13,
                                          const uint16_t* __restrict__ s13,
                                          uint16_t* __restrict__ inter) {
  const int pair = blockIdx.x;
  const int64_t expert = (int64_t)ids[pair];
  if (expert < 0 || expert >= V_EXPERTS) return;  // the same for the whole block
  const int lane = threadIdx.x & 31;
  const int first = blockIdx.y * VG_BLOCK_ROWS + (threadIdx.x >> 5) * VG_ROWS;
  constexpr int GROUPS = V_HIDDEN / 32;             // 16-byte groups per row
  const uint4* gate_rows = reinterpret_cast<const uint4*>(w13) + (expert * 2 * V_INTER + first) * GROUPS;
  const uint4* up_rows = gate_rows + (int64_t)V_INTER * GROUPS;
  const uint32_t* gate_scales =
      reinterpret_cast<const uint32_t*>(s13 + (expert * 2 * V_INTER + first) * GROUPS);
  const uint32_t* up_scales = gate_scales + (int64_t)V_INTER * GROUPS / 2;

  // Every weight and scale this lane needs, issued before the activations.
  uint4 gate_w[VG_ROWS][2], up_w[VG_ROWS][2];
  uint32_t gate_s[VG_ROWS], up_s[VG_ROWS];
#pragma unroll
  for (int r = 0; r < VG_ROWS; ++r) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      gate_w[r][h] = gate_rows[r * GROUPS + 2 * lane + h];
      up_w[r][h] = up_rows[r * GROUPS + 2 * lane + h];
    }
    gate_s[r] = gate_scales[r * (GROUPS / 2) + lane];
    up_s[r] = up_scales[r * (GROUPS / 2) + lane];
  }

  // The token's activations for groups 2l and 2l + 1, paired, and each
  // group's -136 x sum: (128 + n) x summed, less 136 x, is (n - 8) x.
  const uint4* row = reinterpret_cast<const uint4*>(x + (int64_t)(pair / V_TOPK) * V_HIDDEN);
  uint4 pairs[2][4];
  float start[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 v = row[(2 * lane + h) * 4 + i];
      pairs[h][i] = v_pair_up(v);
      sum = v_sum8(v, sum);
    }
    start[h] = -136.f * sum;
  }

  float gate_mine = 0.f, up_mine = 0.f;
#pragma unroll
  for (int r = 0; r < VG_ROWS; ++r) {
    float gate = v_group(gate_w[r][0], pairs[0], start[0]) * v_from_bf16((uint16_t)gate_s[r]) +
                 v_group(gate_w[r][1], pairs[1], start[1]) * v_from_bf16((uint16_t)(gate_s[r] >> 16));
    float up = v_group(up_w[r][0], pairs[0], start[0]) * v_from_bf16((uint16_t)up_s[r]) +
               v_group(up_w[r][1], pairs[1], start[1]) * v_from_bf16((uint16_t)(up_s[r] >> 16));
    gate = v_wave_total(gate);
    up = v_wave_total(up);
    if (lane == r) { gate_mine = gate; up_mine = up; }
  }
  if (lane < VG_ROWS) {
    const float hidden = gate_mine / (1.f + __expf(-gate_mine)) * up_mine;
    inter[(int64_t)pair * V_INTER + first + lane] = v_to_bf16(hidden);
  }
}

// ---- down ------------------------------------------------------------------
// Grid (tokens, 2048 / (VD_WAVES * 32 / LANES)). The block first copies the
// token's eight intermediate rows into LDS, paired, with each (row, group)'s
// -136 x sum; then each wave walks the eight experts for its output rows -
// LANES lanes per row, lane q reading 16-byte groups q, q + LANES, ... of the
// 384-byte row - adds each expert's row times its routing weight in fp32, and
// one lane per row writes the bf16 result. No per-pair rows, no second pass.
template <typename Id, int LANES>
__device__ __forceinline__ void v_down(const uint16_t* __restrict__ inter, const Id* __restrict__ ids,
                                       const float* __restrict__ weights,
                                       const uint8_t* __restrict__ w2,
                                       const uint16_t* __restrict__ s2,
                                       uint16_t* __restrict__ out) {
  constexpr int ROWS_PER_WAVE = 32 / LANES;
  constexpr int BLOCK_ROWS = VD_WAVES * ROWS_PER_WAVE;
  constexpr int STEPS = VD_GROUPS / LANES;
  static_assert(V_HIDDEN % BLOCK_ROWS == 0, "down rows must tile the hidden size");
  static_assert(VD_GROUPS % LANES == 0, "a down row's groups must split evenly over its lanes");
  __shared__ uint4 staged[V_TOPK][V_INTER / 8];
  __shared__ float starts[V_TOPK][VD_GROUPS];
  const int token = blockIdx.x;
  const int64_t first_pair = (int64_t)token * V_TOPK;
  constexpr int THREADS = VD_WAVES * 32;
  constexpr int UNITS = V_TOPK * (V_INTER / 8);    // uint4 of the eight rows
  // Every thread runs every pass, so the shuffles below never meet an idle lane.
  static_assert(UNITS % THREADS == 0, "the staging passes must cover the block exactly");
  for (int unit = threadIdx.x; unit < UNITS; unit += THREADS) {
    const int j = unit / (V_INTER / 8), at = unit % (V_INTER / 8);
    const int64_t expert = (int64_t)ids[first_pair + j];
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (expert >= 0 && expert < V_EXPERTS)
      v = reinterpret_cast<const uint4*>(inter + (first_pair + j) * V_INTER)[at];
    staged[j][at] = v_pair_up(v);
    // A group is four consecutive uint4, held by four adjacent lanes.
    float sum = v_sum8(v, 0.f);
    sum += __shfl_xor(sum, 1, 32);
    sum += __shfl_xor(sum, 2, 32);
    if ((at & 3) == 0) starts[j][at >> 2] = -136.f * sum;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int q = lane % LANES;
  const int out_row = blockIdx.y * BLOCK_ROWS + (threadIdx.x >> 5) * ROWS_PER_WAVE + lane / LANES;
  float total = 0.f;
  for (int j = 0; j < V_TOPK; ++j) {
    const int64_t expert = (int64_t)ids[first_pair + j];  // the same for the whole block
    if (expert < 0 || expert >= V_EXPERTS) continue;
    const uint4* groups = reinterpret_cast<const uint4*>(w2) + (expert * V_HIDDEN + out_row) * VD_GROUPS;
    const uint16_t* scales = s2 + (expert * V_HIDDEN + out_row) * VD_GROUPS;
    uint4 words[STEPS];
    uint16_t scale[STEPS];
#pragma unroll
    for (int step = 0; step < STEPS; ++step) {
      words[step] = groups[q + step * LANES];
      scale[step] = scales[q + step * LANES];
    }
    float part = 0.f;
#pragma unroll
    for (int step = 0; step < STEPS; ++step) {
      const int group = q + step * LANES;
      part += v_group(words[step], &staged[j][group * 4], starts[j][group]) * v_from_bf16(scale[step]);
    }
    total += weights[first_pair + j] * part;
  }
#pragma unroll
  for (int apart = 1; apart < LANES; apart <<= 1) total += __shfl_xor(total, apart, 32);
  if (q == 0) out[(int64_t)token * V_HIDDEN + out_row] = v_to_bf16(total);
}

// The entry points the plugin looks up by name: per expert-id type, and for
// the down projection per lanes-per-row.
#define V_GATE_UP(NAME, ID)                                                              \
  extern "C" __global__ __launch_bounds__(VG_WAVES * 32) void NAME(                      \
      const uint16_t* x, const ID* ids, const uint8_t* w13, const uint16_t* s13,          \
      uint16_t* inter) {                                                                  \
    v_gate_up<ID>(x, ids, w13, s13, inter);                                               \
  }
#define V_DOWN(NAME, ID, LANES)                                                          \
  extern "C" __global__ __launch_bounds__(VD_WAVES * 32) void NAME(                      \
      const uint16_t* inter, const ID* ids, const float* weights, const uint8_t* w2,      \
      const uint16_t* s2, uint16_t* out) {                                                \
    v_down<ID, LANES>(inter, ids, weights, w2, s2, out);                                  \
  }

V_GATE_UP(vaelor_d2_gate_up_i32, int32_t)
V_GATE_UP(vaelor_d2_gate_up_i64, int64_t)
V_DOWN(vaelor_d2_down2_i32, int32_t, 2)
V_DOWN(vaelor_d2_down2_i64, int64_t, 2)
V_DOWN(vaelor_d2_down4_i32, int32_t, 4)
V_DOWN(vaelor_d2_down4_i64, int64_t, 4)
