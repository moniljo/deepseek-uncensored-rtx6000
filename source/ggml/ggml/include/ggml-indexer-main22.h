#pragma once

#include "ggml.h"
#include <cstdlib>
#include <cstring>

#define GGML_MAIN22_LI_TAG 4

extern "C" void main22_host_counts(uint64_t out[2]);

static inline bool ggml_main22_enabled() {
    const char * v = std::getenv("STRATA_LI_MAIN_COMPACT_PREFILL");
    return v && std::strcmp(v, "1") == 0;
}

static inline int ggml_main22_query_tile() {
    static const int tile=[] {
        const char * v=std::getenv("STRATA_LI_MAIN_QUERY_TILE");
        if (v && std::strcmp(v,"32")==0) return 32;
        if (v && std::strcmp(v,"128")==0) return 128;
        if (v && std::strcmp(v,"256")==0) return 256;
        return 64;
    }();
    return tile;
}

// Native key-operand reuse is independent of score-scratch tiling. Read the
// setting only while constructing the node, then freeze it in op_params[4].
// Unknown values preserve the source22 QT16 path; decode/source21/compact20
// never consult this setting.
static inline int ggml_main22_native_query_tile_setting(const char * value) {
    if (value && std::strcmp(value,"32")==0) return 32;
    if (value && std::strcmp(value,"64")==0) return 64;
    return 16;
}

static inline int ggml_main22_native_query_tile() {
    return ggml_main22_native_query_tile_setting(std::getenv("STRATA_LI_MAIN_NATIVE_QUERY_TILE"));
}

static inline bool ggml_main22_shape_eligible(int64_t cap, int64_t queries, int64_t topk, int64_t blocks) {
    return cap > 1024 && cap <= 1048576 && cap % 8 == 0 && queries >= 64 && queries <= 2048 &&
        topk >= 1 && topk <= 512 && topk <= cap && blocks >= 0 && blocks <= 2048 && blocks <= cap / 8 &&
        (blocks == 0 || cap / 8 > 1024);
}

// All visible keys are scored. Auxiliary writes have explicit producer dependencies.
static inline ggml_tensor * ggml_lightning_indexer_main22(
        ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * weights,
        ggml_tensor * visible, ggml_tensor * last_block, int topk, int blocks,
        ggml_tensor ** selected_mask, ggml_tensor ** candidate_ids) {
    if (!ggml_main22_enabled() || !ggml_main22_shape_eligible(k->ne[2], q->ne[2], topk, blocks)) return nullptr;
    GGML_ASSERT(selected_mask && candidate_ids && ((blocks > 0) == (last_block != nullptr)));
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, topk, q->ne[2]);
    ggml_set_output(mask);
    ggml_tensor * candidates = blocks ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, blocks, q->ne[2]) : nullptr;
    if (candidates) ggml_set_output(candidates);
    ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, topk, q->ne[2]);
    result->op = GGML_OP_LIGHTNING_INDEXER;
    result->src[0] = q; result->src[1] = k; result->src[2] = weights;
    result->src[3] = mask; result->src[4] = visible; result->src[5] = last_block; result->src[6] = candidates;
    result->op_params[1] = GGML_MAIN22_LI_TAG;
    result->op_params[2] = blocks;
    result->op_params[3] = ggml_main22_query_tile();
    result->op_params[4] = ggml_main22_native_query_tile();
    ggml_tensor * mask_view = ggml_view_tensor(ctx, mask);
    mask_view->src[1] = result;
    *selected_mask = mask_view;
    *candidate_ids = nullptr;
    if (candidates) {
        ggml_tensor * candidate_view = ggml_view_tensor(ctx, candidates);
        candidate_view->src[1] = result;
        *candidate_ids = candidate_view;
    }
    return result;
}
