#pragma once
// Explicit native-CUDA decode extension. Normal constructors remain untagged.
#include "ggml.h"
#include <stdint.h>
#ifdef __cplusplus
#include <cstdlib>
#include <cstring>
#define GGML_DECODE26_LI_TAG 5
#define GGML_DECODE26_TOPK_TAG 0x4432364b
static inline bool ggml_decode26_enabled() {
    static const bool enabled=[] {
        const char * e=std::getenv("STRATA_DECODE_DSA_COMPACT");
        return e && std::strcmp(e,"1")==0;
    }();
    return enabled;
}
// Diagnostic only. Production observes tagged dispatch/capture entries, never
// normal nodes or GPU replay invocation counts. Timed clean runs leave it OFF.
static inline bool ggml_decode26_observe_enabled() {
    static const bool enabled=[] {const char * e=std::getenv("STRATA_DECODE_DSA_OBSERVE26");return e&&std::strcmp(e,"1")==0;}();
    return enabled;
}
static inline bool ggml_decode26_shape_eligible(int64_t cap,int64_t blocks,int64_t queries,int64_t topk) {
    return queries==1 && cap>1024 && cap<=1048576 && blocks>=1 && blocks<=2048 &&
        blocks<=(cap+7)/8 && topk>=1 && topk<=512 && topk<=blocks*8 && cap>=blocks*8+topk;
}
// Candidates and visibility are direct runtime dependencies. The producer keeps
// the existing candidate vector arithmetic, analytic causal mask and runtime+0.
static inline ggml_tensor * ggml_lightning_indexer_decode26(
        ggml_context * ctx,ggml_tensor * q,ggml_tensor * k,ggml_tensor * weights,
        ggml_tensor * visible,ggml_tensor * candidates,ggml_tensor * positive_zero) {
    if (!ggml_decode26_enabled() ||
        !ggml_decode26_shape_eligible(k->ne[2],candidates->ne[0],q->ne[2],512)) return nullptr;
    ggml_tensor * result=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,candidates->ne[0]*8,1,1,1);
    result->op=GGML_OP_LIGHTNING_INDEXER;
    result->src[0]=q;result->src[1]=k;result->src[2]=weights;
    result->src[4]=visible;result->src[5]=candidates;result->src[6]=positive_zero;
    result->op_params[1]=GGML_DECODE26_LI_TAG;result->op_params[2]=8;
    return result;
}
static inline ggml_tensor * ggml_top_k_decode26(
        ggml_context * ctx,ggml_tensor * compact,int topk,ggml_tensor ** selected_mask) {
    GGML_ASSERT(compact->op==GGML_OP_LIGHTNING_INDEXER && compact->op_params[1]==GGML_DECODE26_LI_TAG);
    GGML_ASSERT(ggml_decode26_shape_eligible(compact->src[1]->ne[2],compact->src[5]->ne[0],1,topk));
    ggml_tensor * scores=ggml_reshape_2d(ctx,compact,compact->ne[0],1);
    ggml_tensor * mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,topk,1);ggml_set_output(mask);
    ggml_tensor * result=ggml_top_k(ctx,scores,topk);
    result->op_params[1]=GGML_DECODE26_TOPK_TAG;result->op_params[2]=(int32_t)compact->src[1]->ne[2];
    result->src[1]=compact->src[5];result->src[2]=compact->src[4];result->src[3]=mask;
    ggml_tensor * dependent=ggml_view_tensor(ctx,mask);dependent->src[1]=result;
    *selected_mask=dependent;return result;
}
#if defined(STRATA_DECODE26_CAPTURE_OBSERVATION)
// Fixture-only arbitrary-score injection. Not a production provenance bypass.
static inline ggml_tensor * ggml_top_k_decode26_scores_for_test(
        ggml_context * ctx,ggml_tensor * scores,ggml_tensor * candidates,
        ggml_tensor * visible,int cap,int topk,ggml_tensor ** selected_mask) {
    GGML_ASSERT(ggml_decode26_shape_eligible(cap,candidates->ne[0],ggml_nrows(scores),topk));
    GGML_ASSERT(scores->type==GGML_TYPE_F32 && scores->ne[0]==candidates->ne[0]*8);
    ggml_tensor * mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,topk,1);ggml_set_output(mask);
    ggml_tensor * result=ggml_top_k(ctx,scores,topk);
    result->op_params[1]=GGML_DECODE26_TOPK_TAG;result->op_params[2]=cap;result->op_params[3]=0x54455354;
    result->src[1]=candidates;result->src[2]=visible;result->src[3]=mask;
    ggml_tensor * dependent=ggml_view_tensor(ctx,mask);dependent->src[1]=result;
    *selected_mask=dependent;return result;
}
#endif
// Standalone fixture observes all nodes; production optional diagnostic observes
// tagged nodes only. Counts are host dispatch/capture entries plus the
// tagged subset. Replay proofs require frozen counts and actual graph identity.
extern "C" void decode26_li_host_counts(uint64_t out[4]);
extern "C" void decode26_topk_host_counts(uint64_t out[4]);
#endif
