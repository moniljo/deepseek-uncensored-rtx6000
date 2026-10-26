#pragma once
#include "ggml.h"
#include <string.h>
#include <stdlib.h>

#define GGML_INDEXED_ATTN22_TAG 0x49413232
#define GGML_INDEXED_ATTN22_SLOT 15
#define GGML_INDEXED_ATTN22_REUSE_SLOT 13
#define GGML_INDEXED_ATTN24_PADDING_SLOT 12
#define GGML_INDEXED_ATTN24_DIAGNOSTIC_SLOT 11
#define GGML_INDEXED_ATTN22_MODE_SLOT 14

#ifdef __cplusplus
extern "C" void indexed_attn22_host_counts(uint64_t out[2]);
extern "C" int indexed_attn22_resources(int out[14]);
extern "C" int indexed_attn22_reuse_resources(int out[14]);
extern "C" int indexed_attn24_padding_reset(void);
extern "C" int indexed_attn24_padding_counts(uint64_t out[2]);
extern "C" int indexed_attn24_window_tile_probe(const float * window, int64_t rows,
                                               const int32_t * indices128, uint16_t * output128x512);
#endif

static inline bool ggml_is_indexed_attn22(const struct ggml_tensor * t) {
    return t && t->op == GGML_OP_FLASH_ATTN_EXT && t->op_params[GGML_INDEXED_ATTN22_SLOT] == GGML_INDEXED_ATTN22_TAG;
}

static inline bool ggml_split_attn22_reuse_setting(const char * value) {
    return value && strcmp(value,"1")==0;
}

static inline struct ggml_tensor * ggml_indexed_attn22(
        struct ggml_context * ctx, struct ggml_tensor * q, struct ggml_tensor * window,
        struct ggml_tensor * window_indices, struct ggml_tensor * compressed,
        struct ggml_tensor * selected, struct ggml_tensor * mask, float scale) {
    const int64_t width = 128 + selected->ne[0];
    const int64_t padded = (width + 255) / 256 * 256;
    struct ggml_tensor * virtual_k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 512, padded, 1, q->ne[3]);
    struct ggml_tensor * result = ggml_flash_attn_ext(ctx, q, virtual_k, virtual_k, mask, scale, 0.0f, 0.0f);
    result->src[1] = window;
    result->src[2] = compressed;
    result->src[5] = window_indices;
    result->src[6] = selected;
    // Packed mode keeps its original implementation regardless of split flags.
    result->op_params[GGML_INDEXED_ATTN24_DIAGNOSTIC_SLOT] = 0;
    result->op_params[GGML_INDEXED_ATTN24_PADDING_SLOT] = 0;
    result->op_params[GGML_INDEXED_ATTN22_REUSE_SLOT] = 0;
    result->op_params[GGML_INDEXED_ATTN22_SLOT] = GGML_INDEXED_ATTN22_TAG;
    return result;
}

static inline struct ggml_tensor * ggml_split_attn22(
        struct ggml_context * ctx, struct ggml_tensor * q, struct ggml_tensor * window,
        struct ggml_tensor * compressed, struct ggml_tensor * mask, float scale) {
    const int64_t padded=(128+compressed->ne[1]+255)/256*256;
    struct ggml_tensor * virtual_k=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,512,padded,1,q->ne[3]);
    struct ggml_tensor * result=ggml_flash_attn_ext(ctx,q,virtual_k,virtual_k,mask,scale,0.0f,0.0f);
    result->src[1]=window; result->src[2]=compressed;
    result->op_params[14]=2;
    // Both attention inputs are the very same virtual K tensor above: this
    // constructor proves K==V, while backend admission proves the supported
    // contiguous F16 geometry. Freeze the opt-in before graph capture/replay.
    result->op_params[GGML_INDEXED_ATTN22_REUSE_SLOT]=
        ggml_split_attn22_reuse_setting(getenv("STRATA_ATTN_REUSE_KV_PREFILL")) &&
        q->ne[0]==512 && window->ne[0]==512 && compressed->ne[0]==512 &&
        window->type==GGML_TYPE_F16 && compressed->type==GGML_TYPE_F16 ? 1 : 0;
    // Metadata, not ambient environment, controls each captured execution.
    result->op_params[GGML_INDEXED_ATTN24_PADDING_SLOT]=
        ggml_split_attn22_reuse_setting(getenv("STRATA_ATTN_SKIP_PADDING_PREFILL")) ? 1 : 0;
    result->op_params[GGML_INDEXED_ATTN24_DIAGNOSTIC_SLOT]=
        ggml_split_attn22_reuse_setting(getenv("STRATA_ATTN_PADDING_DIAGNOSTICS24")) ? 1 : 0;
    result->op_params[GGML_INDEXED_ATTN22_SLOT]=GGML_INDEXED_ATTN22_TAG;
    return result;
}

// Hybrid24 retains the materialized compressed F16 rows from split22. Only
// window rows are loaded from their original F32 source and rounded to F16
// inside the attention tile loader, using the same per-element RN conversion.
// The explicit constructor freezes mode3; the caller chooses it while building
// the graph. Ordinary/split graphs never consult the window flag at execution.
// Native NVIDIA CUDA-only contract: callers must establish that backend before
// construction. Generic FA cannot interpret these replaced K/V source tensors.
static inline struct ggml_tensor * ggml_hybrid_attn24(
        struct ggml_context * ctx, struct ggml_tensor * q, struct ggml_tensor * window,
        struct ggml_tensor * window_indices, struct ggml_tensor * compressed,
        struct ggml_tensor * mask, float scale) {
    const int64_t padded=(128+compressed->ne[1]+255)/256*256;
    struct ggml_tensor * virtual_k=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,512,padded,1,q->ne[3]);
    struct ggml_tensor * result=ggml_flash_attn_ext(ctx,q,virtual_k,virtual_k,mask,scale,0.0f,0.0f);
    result->src[1]=window; result->src[2]=compressed; result->src[5]=window_indices;
    result->op_params[GGML_INDEXED_ATTN22_MODE_SLOT]=3;
    result->op_params[GGML_INDEXED_ATTN22_REUSE_SLOT]=
        ggml_split_attn22_reuse_setting(getenv("STRATA_ATTN_REUSE_KV_PREFILL")) &&
        q->ne[0]==512 && window->ne[0]==512 && compressed->ne[0]==512 &&
        window->type==GGML_TYPE_F32 && compressed->type==GGML_TYPE_F16 ? 1 : 0;
    result->op_params[GGML_INDEXED_ATTN24_PADDING_SLOT]=
        ggml_split_attn22_reuse_setting(getenv("STRATA_ATTN_SKIP_PADDING_PREFILL")) ? 1 : 0;
    result->op_params[GGML_INDEXED_ATTN24_DIAGNOSTIC_SLOT]=
        ggml_split_attn22_reuse_setting(getenv("STRATA_ATTN_PADDING_DIAGNOSTICS24")) ? 1 : 0;
    result->op_params[GGML_INDEXED_ATTN22_SLOT]=GGML_INDEXED_ATTN22_TAG;
    return result;
}
