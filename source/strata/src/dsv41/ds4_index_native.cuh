// Experimental exact Blackwell FP4 indexer. See ds4_index_native_impl.cuh and
// third_party/xolotl-indexer/NOTICE.md for provenance and the numerical guard.
#pragma once
#include "ds4_kernels.cuh"

namespace strata::ds4 {
// Diagnostic entry point for the isolated numeric fixture. The ordinary indexer
// uses this implementation only with DS4_INDEX_NATIVE=1. Return false means the
// caller must use index_score; true queues either native or guarded scalar math.
// Optional device_state receives 18 integers after the score kernels: [0] native
// admission, [1] reason bits, then exponent/weight bounds; [16]/[17] count actual
// native/scalar CTAs, to distinguish admission from a dispatched no-op branch.
bool index_score_native_diagnostic(const float* q, const float* w, const void* k,
    int heads, int hd, int T, int n, const int* cand, int cand_stride, int blk,
    const Rows& lim, float* out, long long out_stride, cudaStream_t s,
    int* device_state = nullptr);

// DS4_INDEX_STATS=1 enables per-stream device counters for actual executed
// admission decisions. This explicit hook drains the supplied stream, prints
// one summary to stderr, and optionally queues a reset in that stream. Call at
// a completed prompt/decode boundary, never for each index_score invocation.
// With stats absent, this hook does nothing and queues no device operations.
void index_native_profile_dump(cudaStream_t stream, bool reset = false);
}
