// Isolated Strata decoder store-width experiment. No native integration.
// The surrounding decoder is Copyright (c) 2026 Peasant Smith and the Project Maya
// contributors, MIT; its notice and ggml attribution remain in ds4_iq.cu.
// Included inside iq's private namespace, after the original dq_rows_kernel.
#if !defined(STRATA_USE_HIP)

inline bool dq_vector_store_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("DS4_DQ_VECTOR_STORES");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

// Bound this study to the actual model's four routed IQ/K types and both expert
// roles, either one expert or eight contiguous experts. Other calls retain the
// unchanged original implementation. MXFP4 uses a different decoder altogether.
inline bool dq_vector_store_eligible(int type, const void* raw, size_t ld, int k, int n,
                                     const uint16_t* dst) {
    if (type != kQ6_K && type != kIQ3_XXS && type != kIQ3_S && type != kIQ2_S) return false;
    if (raw == nullptr || dst == nullptr || (reinterpret_cast<uintptr_t>(dst) & 15u) != 0) return false;
    if (!((k == 5120 && (n == 2304 || n == 8 * 2304)) ||
          (k == 2304 && (n == 5120 || n == 8 * 5120)))) return false;
    const size_t row = row_bytes(type, k);
    if (ld < row) return false;
    const size_t last_row = static_cast<size_t>(n - 1);
    if (last_row != 0 && ld > (SIZE_MAX - row) / last_row) return false;
    const size_t read_span = last_row * ld + row;
    const size_t write_span = static_cast<size_t>(n) * static_cast<size_t>(k) * sizeof(uint16_t);
    return reinterpret_cast<uintptr_t>(raw) <= UINTPTR_MAX - read_span &&
           reinterpret_cast<uintptr_t>(dst) <= UINTPTR_MAX - write_span;
}

__device__ __forceinline__ void dq_store_128(uint16_t* destination, uint32_t a, uint32_t b,
                                            uint32_t c, uint32_t d) {
    // Explicit vector PTX establishes the experiment's requested store contract.
    // Actual emitted PTX/SASS width and register/spill cost still require a build.
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
                 : : "l"(destination), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

template <int T>
__global__ void __launch_bounds__(kThreads) dq_rows_vector128_kernel(
        const uint8_t* __restrict__ raw, size_t ld, int k, int n,
        uint16_t* __restrict__ dst, bool bf16) {
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    load_tab<T>(tab);
    __syncthreads();
    const int nsb = k / 32;
    const size_t total = (size_t) n * (size_t) nsb;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t) gridDim.x * blockDim.x) {
        const size_t r = i / (size_t) nsb;
        const int kb = (int) (i - r * (size_t) nsb);
        Sub sb;
        sb.m0 = sb.m1 = 0.0f;
        decode<T>(raw + r * ld, kb, tab, sb);
        const float d0 = __fmul_rn(sb.d, (float) sb.s0), d1 = __fmul_rn(sb.d, (float) sb.s1);
        uint16_t* o = dst + r * (size_t) k + (size_t) kb * 32;
        uint32_t packed[16];
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int q = (int) (int8_t) ((uint32_t) sb.q[j >> 2] >> (8 * (j & 3)));
            const float v = j < 16 ? __fsub_rn(__fmul_rn(d0, (float) q), sb.m0) : __fsub_rn(__fmul_rn(d1, (float) q), sb.m1);
            const uint16_t bits = bf16 ? strata::ds4::bf16_rn(v) : __half_as_ushort(__float2half_rn(v));
            if ((j & 1) == 0) packed[j >> 1] = static_cast<uint32_t>(bits);
            else packed[j >> 1] |= static_cast<uint32_t>(bits) << 16;
        }
#pragma unroll
        for (int v = 0; v < 4; ++v)
            dq_store_128(o + v * 8, packed[v * 4], packed[v * 4 + 1], packed[v * 4 + 2], packed[v * 4 + 3]);
    }
}

#endif  // !STRATA_USE_HIP: leave the complete original HIP decoder unchanged.
