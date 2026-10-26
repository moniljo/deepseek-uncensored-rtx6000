// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT).
// Derived ordered-key and gather logic from Strata's ds4_kernels.cu; its
// third_party/project-maya/NOTICE.md and LICENSE remain applicable.
// Experimental exact multi-CTA selection for Strata's indexer.
// Production staging only: persistent POD scratch is supplied by the caller.
// The threshold and output contract follow Strata's original topk_kernel:
// raw ordered float bits; above-cutoff positions ascending, then earliest
// cutoff ties ascending. Candidate expansion is mapped only at output.
#pragma once

#include "ds4_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::ds4::topk_multi {

constexpr int Threads = 256;
constexpr int MaxParts = 32;
using State = TopkState;
using Count = TopkCount;
static_assert(sizeof(State) == 12 && sizeof(Count) == 8, "TopK POD layout");
constexpr int MinValues = 131072; // measured large-array entry point; small arrays retain the original CTA

__host__ __device__ inline void sizes(const Rows& r, int t, int& n, int& k) {
    if (r.gen == 0) { n = r.n[t]; k = r.k[t]; return; }
    const int nn = int((r.p0 + t + 1) / r.R);
    const int raw_cb = (nn + r.blk - 1) / r.blk;
    const int cb = raw_cb < r.ccap ? raw_cb : r.ccap;
    switch (r.gen) {
    case 1: n = nn; k = r.kcap < nn ? r.kcap : nn; break;
    case 2: n = nn; k = cb; break;
    case 3: n = cb * r.blk; k = r.kcap < nn ? r.kcap : nn; break;
    default: n = (nn + r.blk - 1) / r.blk; k = r.kcap; break;
    }
}

__device__ __forceinline__ uint32_t key(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ int first(int n, int part, int parts) {
    return int((static_cast<long long>(n) * part) / parts);
}
__device__ __forceinline__ int map_position(int i, const int* cd, int blk) {
    return cd ? cd[i / blk] * blk + i % blk : i;
}

static __global__ void init(State* states, int T, Rows r) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < T) {
        int n, k; sizes(r, row, n, k);
        states[row] = {0u, 0u, k};
    }
}

static __global__ void histogram(const float* __restrict__ vals, long long stride, Rows r,
                                const State* __restrict__ states, unsigned* __restrict__ hist,
                                int parts, int shift) {
    const int row = blockIdx.y, part = blockIdx.x, tid = threadIdx.x;
    int n, k; sizes(r, row, n, k);
    if (n <= 0 || k <= 0 || k >= n) return;
    __shared__ unsigned bins[256];
    bins[tid] = 0;
    __syncthreads();
    const State state = states[row];
    const float* v = vals + static_cast<long long>(row) * stride;
    const int lo = first(n, part, parts), hi = first(n, part + 1, parts);
    for (int i = lo + tid; i < hi; i += Threads) {
        const uint32_t u = key(v[i]);
        if ((u & state.mask) == state.prefix) {
            const unsigned bin = (u >> shift) & 255u;
#if __CUDA_ARCH__ >= 700
            // Exact integer multiplicity; this changes no float comparisons.
            const unsigned active = __activemask();
            const unsigned peers = __match_any_sync(active, bin);
            if ((tid & 31) == __ffs(peers) - 1) atomicAdd(&bins[bin], unsigned(__popc(peers)));
#else
            atomicAdd(&bins[bin], 1u);
#endif
        }
    }
    __syncthreads();
    hist[(size_t(row) * parts + part) * 256 + tid] = bins[tid];
}

static __global__ void select(const unsigned* __restrict__ hist, State* __restrict__ states,
                             Rows r, int parts, int shift) {
    const int row = blockIdx.x, tid = threadIdx.x;
    int n, k; sizes(r, row, n, k);
    if (n <= 0 || k <= 0 || k >= n) return;
    __shared__ unsigned bins[256];
    unsigned total = 0;
    for (int p = 0; p < parts; ++p) total += hist[(size_t(row) * parts + p) * 256 + tid];
    bins[tid] = total;
    __syncthreads();
    if (tid == 0) {
        State state = states[row];
        int cum = 0;
        for (int d = 255; d >= 0; --d) {
            const int h = int(bins[d]);
            if (cum + h >= state.remaining) {
                state.prefix |= uint32_t(d) << shift;
                state.mask |= 0xffu << shift;
                state.remaining -= cum;
                break;
            }
            cum += h;
        }
        states[row] = state;
    }
}

static __global__ void count(const float* __restrict__ vals, long long stride, Rows r,
                             const State* __restrict__ states, Count* __restrict__ counts, int parts) {
    const int row = blockIdx.y, part = blockIdx.x, tid = threadIdx.x;
    int n, k; sizes(r, row, n, k);
    if (n <= 0 || k <= 0 || k >= n) return;
    const float* v = vals + static_cast<long long>(row) * stride;
    const uint32_t threshold = states[row].prefix;
    const int lo = first(n, part, parts), hi = first(n, part + 1, parts);
    int ng = 0, ne = 0;
    for (int i = lo + tid; i < hi; i += Threads) {
        const uint32_t u = key(v[i]);
        ng += u > threshold; ne += u == threshold;
    }
    for (int delta = 16; delta > 0; delta >>= 1) {
        ng += __shfl_down_sync(0xffffffffu, ng, delta);
        ne += __shfl_down_sync(0xffffffffu, ne, delta);
    }
    __shared__ Count warp[Threads / 32];
    if ((tid & 31) == 0) warp[tid / 32] = {ng, ne};
    __syncthreads();
    if (tid == 0) {
        Count total{0, 0};
        for (int w = 0; w < Threads / 32; ++w) { total.gt += warp[w].gt; total.eq += warp[w].eq; }
        counts[size_t(row) * parts + part] = total;
    }
}

static __global__ void prefix(const Count* __restrict__ counts, Count* __restrict__ offsets,
                              Rows r, int parts) {
    const int row = blockIdx.x, lane = threadIdx.x;
    int n, k; sizes(r, row, n, k);
    if (n <= 0 || k <= 0 || k >= n) return;
    const Count own = lane < parts ? counts[size_t(row) * parts + lane] : Count{0, 0};
    Count accum = own;
    for (int delta = 1; delta < 32; delta <<= 1) {
        const int g = __shfl_up_sync(0xffffffffu, accum.gt, delta);
        const int e = __shfl_up_sync(0xffffffffu, accum.eq, delta);
        if (lane >= delta) { accum.gt += g; accum.eq += e; }
    }
    if (lane < parts) offsets[size_t(row) * parts + lane] = {accum.gt - own.gt, accum.eq - own.eq};
}

static __global__ void gather(const float* __restrict__ vals, long long stride, Rows r,
                              const int* __restrict__ cand, int cand_stride, int blk,
                              const State* __restrict__ states, const Count* __restrict__ offsets,
                              int parts, int* __restrict__ out, int out_stride) {
    const int row = blockIdx.y, part = blockIdx.x, tid = threadIdx.x;
    int n, k; sizes(r, row, n, k);
    if (n <= 0 || k <= 0) return;
    const int lo = first(n, part, parts), hi = first(n, part + 1, parts);
    const int* cd = cand ? cand + size_t(row) * cand_stride : nullptr;
    int* dest = out + size_t(row) * out_stride;
    if (k >= n) {
        for (int i = lo + tid; i < hi; i += Threads) dest[i] = map_position(i, cd, blk);
        return;
    }
    const State state = states[row];
    const int n_gt = k - state.remaining;
    const float* v = vals + static_cast<long long>(row) * stride;
    __shared__ int wgt[Threads / 32], weq[Threads / 32], cnt_gt, cnt_eq;
    if (tid == 0) {
        const Count offset = offsets[size_t(row) * parts + part];
        cnt_gt = offset.gt; cnt_eq = offset.eq;
    }
    __syncthreads();
    const int lane = tid & 31, wid = tid >> 5;
    const unsigned below = (1u << lane) - 1u;
    // Each partition is a contiguous input interval. Tile/warp/lane scans
    // preserve ascending positions even across partitions and arbitrary tails.
    for (int i0 = lo; i0 < hi; i0 += Threads) {
        const int i = i0 + tid;
        const uint32_t u = i < hi ? key(v[i]) : 0u;
        const bool gt = i < hi && u > state.prefix, eq = i < hi && u == state.prefix;
        const unsigned bg = __ballot_sync(0xffffffffu, gt), be = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) { wgt[wid] = __popc(bg); weq[wid] = __popc(be); }
        __syncthreads();
        if (wid == 0) {
            const int g0 = lane < Threads / 32 ? wgt[lane] : 0;
            const int e0 = lane < Threads / 32 ? weq[lane] : 0;
            int gs = g0, es = e0;
            for (int delta = 1; delta < 32; delta <<= 1) {
                const int g = __shfl_up_sync(0xffffffffu, gs, delta);
                const int e = __shfl_up_sync(0xffffffffu, es, delta);
                if (lane >= delta) { gs += g; es += e; }
            }
            if (lane < Threads / 32) { wgt[lane] = cnt_gt + gs - g0; weq[lane] = cnt_eq + es - e0; }
            __syncwarp();
            if (lane == 31) { cnt_gt += gs; cnt_eq += es; }
        }
        __syncthreads();
        int slot = -1;
        if (gt) slot = wgt[wid] + __popc(bg & below);
        else if (eq) {
            const int e = weq[wid] + __popc(be & below);
            if (e < state.remaining) slot = n_gt + e;
        }
        if (slot >= 0) dest[slot] = map_position(i, cd, blk);
        __syncthreads();
    }
}

// parts is fixed per call but partitions each query's actual visible n.
// Using <=32 allows prefix() to use a full warp without inter-block atomics.
inline cudaError_t launch(const float* vals, long long stride, int T, const Rows& r,
                          const int* cand, int cand_stride, int blk, int* out, int out_stride,
                          const TopkScratch& ws, int parts, cudaStream_t stream) {
    if (T == 0) return cudaSuccess;
    if (T < 0 || T > 65535 || parts < 1 || parts > MaxParts || T > ws.rows || parts > ws.parts ||
        stride < 0 || out_stride < 0 || blk < 1 || (r.gen == 0 && T > kWin) ||
        (r.gen != 0 && (r.R < 1 || r.blk < 1))) return cudaErrorInvalidValue;
    init<<<(T + Threads - 1) / Threads, Threads, 0, stream>>>(ws.states, T, r);
    const dim3 grid(parts, T);
    for (int shift = 24; shift >= 0; shift -= 8) {
        histogram<<<grid, Threads, 0, stream>>>(vals, stride, r, ws.states, ws.hist, parts, shift);
        select<<<T, Threads, 0, stream>>>(ws.hist, ws.states, r, parts, shift);
    }
    count<<<grid, Threads, 0, stream>>>(vals, stride, r, ws.states, ws.counts, parts);
    prefix<<<T, 32, 0, stream>>>(ws.counts, ws.offsets, r, parts);
    gather<<<grid, Threads, 0, stream>>>(vals, stride, r, cand, cand_stride, blk,
                                      ws.states, ws.offsets, parts, out, out_stride);
    return cudaPeekAtLastError();
}

// Admission is entirely host-side and precedes every launch. If it is false,
// the caller queues only the untouched original CTA. A launched CUDA failure
// remains visible to the model's normal error checks; it is not hidden by a
// second fallback launch after partially executing the new pipeline.
inline bool eligible(long long stride, int T, const Rows& r, const int* cand,
                     int cand_stride, int blk, int out_stride, const TopkScratch* ws) {
    if (!ws || !ws->states || !ws->hist || !ws->counts || !ws->offsets ||
        T < 1 || T > 65535 || T > ws->rows || ws->parts < 1 || ws->parts > MaxParts ||
        blk < 1 || stride < 0 || out_stride < 0 || (r.gen == 0 && T > kWin) ||
        (r.gen != 0 && (r.R < 1 || r.blk < 1))) return false;
    int max_n = 0;
    for (int t = 0; t < T; ++t) {
        int n, k; sizes(r, t, n, k);
        if (n < 0 || k < 0 || static_cast<long long>(n) > stride ||
            (k < n ? k : n) > out_stride ||
            (cand && cand_stride < (n + blk - 1) / blk)) return false;
        max_n = n > max_n ? n : max_n;
    }
    return max_n >= MinValues;
}

} // namespace strata::ds4::topk_multi
