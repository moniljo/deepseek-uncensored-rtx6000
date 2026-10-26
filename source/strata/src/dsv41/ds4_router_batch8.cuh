#pragma once
// SOURCE-ONLY ISOLATED STUDY. Include after the frozen actual ds4_kernels.cu in
// a fixture TU so the exact original warp_sum and tail kernel are visible.
// No native public wrapper, environment parser or model file is modified.
#include <cstdint>

namespace router_f32_batch_study {
constexpr int Tokens = 8, Threads = 256, Experts = 384, Dim = 5120, OutputStride = 1024;

// The valid full groups contain eight complete tokens. No tail guard, shared X,
// barriers, reduced-precision operation or different summation tree is added.
__global__ void __launch_bounds__(Threads) router8_kernel(
        const float* __restrict__ W, const float* __restrict__ x,
        float* __restrict__ y) {
    const int row = blockIdx.x * (Threads / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, t0 = blockIdx.y * Tokens;
    if (row >= Experts) return; // uniform within a warp; there are no CTA barriers
    const float4* w = (const float4*)(W + (size_t)row * Dim);
    float acc[Tokens];
#pragma unroll
    for (int t = 0; t < Tokens; ++t) acc[t] = 0.0f;
    for (int c = lane; c < Dim / 4; c += 32) {
        const float4 a = w[c]; // one load reused for eight independent original chains
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            const float4* xv = (const float4*)(x + (size_t)(t0 + t) * Dim);
            const float4 b = xv[c];
            acc[t] = fmaf(a.x, b.x, acc[t]);
            acc[t] = fmaf(a.y, b.y, acc[t]);
            acc[t] = fmaf(a.z, b.z, acc[t]);
            acc[t] = fmaf(a.w, b.w, acc[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < Tokens; ++t) {
        // Exact existing XOR16/8/4/2/1 tree, including its FP32 add semantics.
        const float value = strata::ds4::warp_sum(acc[t]);
        if (lane == 0) y[(size_t)(t0 + t) * OutputStride + row] = value;
    }
}

inline bool span(const void* pointer, size_t bytes, uintptr_t& begin, uintptr_t& end) {
    begin = reinterpret_cast<uintptr_t>(pointer);
    if (!pointer || bytes > ~uintptr_t(0) - begin) return false;
    end = begin + bytes;
    return true;
}
inline bool overlaps(uintptr_t a, uintptr_t b, uintptr_t c, uintptr_t d) { return a < d && c < b; }

// A false result promises no CUDA calls. Enabled is supplied by a future
// default-off dispatcher/fixture; there is no new getenv/static state here.
// Finite host fixture families and actual finite F32 router weights must be
// audited separately. This host guard cannot inspect device activation values.
inline bool supported(bool enabled, const float* W, const float* x,
                      int n, int k, int T, float* y, int ldy) {
    if (!enabled || n != Experts || k != Dim || ldy != OutputStride || T < 32 || T > 65280)
        return false; // includes every existing <=16-token decode/lookahead call
    if ((reinterpret_cast<uintptr_t>(W) & 15) || (reinterpret_cast<uintptr_t>(x) & 15) ||
        (reinterpret_cast<uintptr_t>(y) & 3)) return false;
    uintptr_t wb, we, xb, xe, yb, ye;
    if (!span(W, (size_t)Experts * Dim * sizeof(float), wb, we) ||
        !span(x, (size_t)T * Dim * sizeof(float), xb, xe) ||
        !span(y, (size_t)T * OutputStride * sizeof(float), yb, ye)) return false;
    // Preserve the old restrict contract. Read-only W/x overlap is harmless,
    // but the output must not overlap either input region.
    return !overlaps(wb, we, yb, ye) && !overlaps(xb, xe, yb, ye);
}

inline bool try_launch_8(bool enabled, const float* W, const float* x,
                         int n, int k, int T, float* y, int ldy, cudaStream_t stream) {
    if (!supported(enabled, W, x, n, k, T, y, ldy)) return false;
    const int full = T / Tokens * Tokens, tail = T - full;
    router8_kernel<<<dim3((Experts + Threads / 32 - 1) / (Threads / 32), full / Tokens), Threads, 0, stream>>>(W, x, y);
    if (tail) {
        // Reuse the actual ORIGINAL kernel for the final <=7 rows. Its t starts
        // at zero while the two base pointers select the original full offset.
        strata::ds4::gemv_f32_kernel<<<dim3((Experts + Threads / 32 - 1) / (Threads / 32), tail), Threads, 0, stream>>>(
            W, x + (size_t)full * Dim, Experts, Dim, y + (size_t)full * OutputStride, OutputStride);
    }
    return true; // normal launch-error checking belongs to the owning fixture/caller
}
} // namespace router_f32_batch_study
