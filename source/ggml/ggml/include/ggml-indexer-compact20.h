#pragma once

// Fork-local explicit CUDA extension. No ggml.h enum or ABI change. Normal
// constructors and decode nodes never carry these operation tags.
#include "ggml.h"
#include <stdint.h>
#include <stddef.h>
#include <limits.h>

#define GGML_COMPACT20_LI_TAG 3
#define GGML_COMPACT20_TOPK_TAG 0x4332304b
#define GGML_COMPACT20_MAX_SLOTS 16384
#define GGML_COMPACT20_MAX_K 512

#if defined(__CUDACC__)
#define GGML_COMPACT20_HD __host__ __device__
#else
#define GGML_COMPACT20_HD
#endif

// Exact ordinal of id in the existing dense radix gather: blocks, then
// threads, then each thread's strided columns. Works for ragged capacities.
GGML_COMPACT20_HD static inline uint32_t ggml_compact20_rank(uint32_t id, uint32_t cap) {
    const uint32_t B = ((cap + 1023u) / 1024u < 32u) ? (cap + 1023u) / 1024u : 32u;
    const uint32_t step = B * 256u, m = cap / step, tail = cap % step;
    const uint32_t b = (id / 256u) % B, t = id % 256u, s = id / step;
    const uint32_t before_b = b * 256u * m + (tail < b * 256u ? tail : b * 256u);
    const uint32_t extra = tail > b * 256u ? tail - b * 256u : 0u;
    return before_b + t * m + (extra < t ? extra : t) + s;
}

GGML_COMPACT20_HD static inline uint32_t ggml_compact20_id(uint32_t rank, uint32_t cap) {
    const uint32_t B = ((cap + 1023u) / 1024u < 32u) ? (cap + 1023u) / 1024u : 32u;
    const uint32_t step = B * 256u, m = cap / step, tail = cap % step;
    uint32_t b = 0;
    for (; b + 1u < B; ++b) {
        const uint32_t end = (b + 1u) * 256u * m + (tail < (b + 1u) * 256u ? tail : (b + 1u) * 256u);
        if (rank < end) break;
    }
    const uint32_t begin = b * 256u * m + (tail < b * 256u ? tail : b * 256u);
    const uint32_t extra0 = tail > b * 256u ? tail - b * 256u : 0u;
    const uint32_t extra = extra0 < 256u ? extra0 : 256u;
    const uint32_t r = rank - begin, first = extra * (m + 1u);
    const uint32_t t = r < first ? r / (m + 1u) : extra + (r - first) / m;
    const uint32_t s = r < first ? r % (m + 1u) : (r - first) % m;
    return b * 256u + t + s * step;
}
#undef GGML_COMPACT20_HD

#ifdef __cplusplus
#include <cstdlib>
#include <cstring>
static inline bool ggml_compact20_enabled() {
    static const bool enabled = [] {
        const char * e = std::getenv("STRATA_LI_COMPACT_PREFILL");
        const char * r = std::getenv("STRATA_TOPK_RADIX_PREFILL");
        return e && std::strcmp(e,"1")==0 && r && std::strcmp(r,"1")==0;
    }();
    return enabled;
}

static inline bool ggml_compact20_shape_eligible(int64_t cap, int64_t blocks, int64_t queries, int64_t topk) {
    return queries >= 64 && queries <= 2048 && blocks >= 1 && blocks <= 2048 &&
           topk >= 1 && topk <= GGML_COMPACT20_MAX_K &&
           blocks * 8 >= topk &&
           cap > 1024 && cap >= blocks * 8 + topk && cap <= 1048576;
}

// Mask is analytic canonical STEP(-id + visible), hence no cap*queries mask
// allocation. positive_zero MUST contain runtime positive zero, exactly as the
// old scalar ADD. Duplicated valid block IDs have identical per-key arithmetic;
// invalid IDs are ignored just as in the existing bounds-checked candidate LI.
static inline ggml_tensor * ggml_lightning_indexer_compact20(
        ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * weights,
        ggml_tensor * visible, ggml_tensor * candidates, ggml_tensor * positive_zero) {
    const int64_t slots = candidates->ne[0] * 8;
    if (!ggml_compact20_enabled() || !ggml_compact20_shape_eligible(k->ne[2], candidates->ne[0], q->ne[2], 512)) return nullptr;
    ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, slots, 1, q->ne[2], 1);
    result->op = GGML_OP_LIGHTNING_INDEXER;
    result->src[0] = q; result->src[1] = k; result->src[2] = weights;
    result->src[4] = visible; result->src[5] = candidates; result->src[6] = positive_zero;
    result->op_params[1] = GGML_COMPACT20_LI_TAG;
    result->op_params[2] = 8;
    return result;
}

// Produces GLOBAL raw IDs with exactly the original dense greater/equal
// traversal order. Selected causal masks are an auxiliary output; a view with
// an explicit TopK dependency makes its write-before-read order unambiguous.
static inline ggml_tensor * ggml_top_k_compact20(
        ggml_context * ctx, ggml_tensor * compact, int topk, ggml_tensor ** selected_mask) {
    GGML_ASSERT(compact->op == GGML_OP_LIGHTNING_INDEXER && compact->op_params[1] == GGML_COMPACT20_LI_TAG);
    GGML_ASSERT(ggml_compact20_shape_eligible(compact->src[1]->ne[2], compact->src[5]->ne[0], compact->ne[2], topk));
    ggml_tensor * scores = ggml_reshape_2d(ctx, compact, compact->ne[0], compact->ne[2]);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, topk, compact->ne[2]);
    ggml_set_output(mask);
    ggml_tensor * result = ggml_top_k(ctx, scores, topk);
    result->op_params[1] = GGML_COMPACT20_TOPK_TAG;
    result->op_params[2] = (int32_t)compact->src[1]->ne[2];
    result->src[1] = compact->src[5]; result->src[2] = compact->src[4]; result->src[3] = mask;
    ggml_tensor * dependent_mask = ggml_view_tensor(ctx, mask);
    dependent_mask->src[1] = result;
    *selected_mask = dependent_mask;
    return result;
}

#if defined(STRATA_COMPACT20_CAPTURE_OBSERVATION)
// Isolated consumer-only tests inject arbitrary score bit patterns, including
// nonfinite values outside the finite LI producer contract. This bypasses only
// the producer provenance check; it uses the identical actual CUDA consumer.
// The marker and its recognizer do not exist in production compilation.
static inline ggml_tensor * ggml_top_k_compact20_scores_for_test(
        ggml_context * ctx, ggml_tensor * scores, ggml_tensor * candidates,
        ggml_tensor * visible, int cap, int topk, ggml_tensor ** selected_mask) {
    GGML_ASSERT(ggml_compact20_shape_eligible(cap,candidates->ne[0],scores->ne[1],topk));
    GGML_ASSERT(scores->type==GGML_TYPE_F32 && scores->ne[0]==candidates->ne[0]*8);
    ggml_tensor * mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,topk,scores->ne[1]);
    ggml_set_output(mask);
    ggml_tensor * result=ggml_top_k(ctx,scores,topk);
    result->op_params[1]=GGML_COMPACT20_TOPK_TAG; result->op_params[2]=cap;
    result->op_params[3]=0x54455354;
    result->src[1]=candidates; result->src[2]=visible; result->src[3]=mask;
    ggml_tensor * view=ggml_view_tensor(ctx,mask); view->src[1]=result;
    *selected_mask=view; return result;
}
#endif

// Standalone compile-time-only observation. Production builds omit the macro,
// including every atomic and cudaStreamIsCapturing query. Both old/reference
// and compact dispatches are counted, so same-exec alone is never replay proof.
extern "C" void compact20_li_host_counts(uint64_t out[2]);
extern "C" void compact20_topk_host_counts(uint64_t out[2]);
#endif
