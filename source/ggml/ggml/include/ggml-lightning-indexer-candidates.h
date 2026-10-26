#pragma once

// Fork-local additive API. Including this header does not change ggml.h or
// force unrelated GGML/CUDA translation units to rebuild.
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

    // CUDA-only exact FP32 vector scoring inside model-supplied candidate blocks.
    // Same causal visible/mask contract as the causal API. All unmasked score
    // arithmetic (operands, dots, weighted terms/accumulations) must stay finite.
    // candidates: I32 [n_blocks,n_batch,1,1], nb[0]==4, nb[1]>=4*n_blocks;
    // columns may be strided. Each column MUST contain unique in-range block IDs.
    // block_size is currently8; heads32/64, dim128, one stream; K F32/BF16/MXFP4.
    // Result is -inf outside those blocks and original score+mask inside them.
    // No sentinel IDs, reordering of per-key arithmetic, or BF16 MMA is used.
    // This does not change ordinary lightning_indexer or its decode dispatch.
    GGML_API struct ggml_tensor * ggml_lightning_indexer_candidates(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * weights,
        struct ggml_tensor  * mask,
        struct ggml_tensor  * visible,
        struct ggml_tensor  * candidates,
        int32_t               block_size);

#ifdef __cplusplus
}
#endif
