#pragma once
// Native-Windows CUDA decode-only mask elimination. Scores remain full-capacity
// float arrays and selection remains the original float* CUB MaxPairs call.
#include "ggml.h"
#include <stdint.h>
#ifdef __cplusplus
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#define GGML_DECODE27_LI_TAG 6
#define GGML_DECODE27_TOPK_TAG 0x4432374b
#define GGML_DECODE27_MASK_TAG 0x4432374d
static inline bool ggml_decode27_enabled() {
    static const bool enabled=[] { const char * e=std::getenv("STRATA_DECODE_DSA_MASK27"); return e&&std::strcmp(e,"1")==0; }();
    return enabled;
}
static inline bool ggml_decode27_observe_enabled() {
    static const bool enabled=[] { const char * e=std::getenv("STRATA_DECODE_DSA_OBSERVE27"); return e&&std::strcmp(e,"1")==0; }();
    return enabled;
}
static inline bool ggml_decode27_shape_eligible(int64_t cap,int64_t blocks,int64_t queries,int64_t topk) {
    return queries==1&&cap>1024&&cap<=1048576&&blocks>=1&&blocks<=2048&&
        blocks<=(cap+7)/8&&topk>=1&&topk<=512&&topk<=blocks*8&&cap>=blocks*8+topk;
}
static inline bool ggml_decode27_inputs_eligible(const ggml_tensor * q,const ggml_tensor * k,
        const ggml_tensor * w,const ggml_tensor * v,const ggml_tensor * c,const ggml_tensor * z) {
    if (!q||!k||!w||!v||!c||!z||q->type!=GGML_TYPE_F32||w->type!=GGML_TYPE_F32||
        v->type!=GGML_TYPE_F32||c->type!=GGML_TYPE_I32||z->type!=GGML_TYPE_F32||
        ggml_nelements(z)!=1||!ggml_is_contiguous(z)||!ggml_is_contiguous(v)||
        q->ne[0]!=128||(q->ne[1]!=32&&q->ne[1]!=64)||q->ne[2]!=1||q->ne[3]!=1||
        k->ne[0]!=128||k->ne[1]!=1||k->ne[3]!=1||
        (k->type!=GGML_TYPE_F32&&k->type!=GGML_TYPE_BF16&&k->type!=GGML_TYPE_MXFP4)||
        w->ne[0]!=q->ne[1]||w->ne[1]!=1||w->ne[2]!=1||w->ne[3]!=1||
        v->ne[0]!=1||v->ne[1]!=1||v->ne[2]!=1||v->ne[3]!=1||
        c->ne[1]!=1||c->ne[2]!=1||c->ne[3]!=1||c->nb[0]!=sizeof(int32_t)||
        c->nb[1]<(size_t)c->ne[0]*sizeof(int32_t)||c->nb[1]%sizeof(int32_t)||
        !ggml_decode27_shape_eligible(k->ne[2],c->ne[0],1,512)) return false;
    for (const ggml_tensor * t : {q,k}) {
        if (ggml_is_quantized(t->type)) continue;
        if (t->nb[0]!=ggml_type_size(t->type)) return false;
        for (int i=1;i<GGML_MAX_DIMS;++i) if (t->nb[i]%16) return false;
    }
    return w->nb[0]==sizeof(float)&&w->nb[1]>=(size_t)w->ne[0]*sizeof(float);
}
static inline bool ggml_decode27_producer_eligible(const ggml_tensor * d) {
    return d&&d->op==GGML_OP_LIGHTNING_INDEXER&&d->op_params[1]==GGML_DECODE27_LI_TAG&&
        d->op_params[2]==8&&!d->src[3]&&
        ggml_decode27_inputs_eligible(d->src[0],d->src[1],d->src[2],d->src[4],d->src[5],d->src[6])&&
        d->type==GGML_TYPE_F32&&ggml_is_contiguous(d)&&d->ne[0]==d->src[1]->ne[2]&&
        d->ne[1]==1&&d->ne[2]==1&&d->ne[3]==1;
}
static inline ggml_tensor * ggml_lightning_indexer_decode27(ggml_context * ctx,ggml_tensor * q,
        ggml_tensor * k,ggml_tensor * w,ggml_tensor * v,ggml_tensor * c,ggml_tensor * z) {
    if (!ggml_decode27_enabled()||!ggml_decode27_inputs_eligible(q,k,w,v,c,z)) return nullptr;
    ggml_tensor * d=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,k->ne[2],1,1,1);
    d->op=GGML_OP_LIGHTNING_INDEXER;d->op_params[1]=GGML_DECODE27_LI_TAG;d->op_params[2]=8;
    d->src[0]=q;d->src[1]=k;d->src[2]=w;d->src[4]=v;d->src[5]=c;d->src[6]=z;
    return d;
}
static inline bool ggml_decode27_topk_eligible(const ggml_tensor * d) {
    if (!d||d->op!=GGML_OP_TOP_K||d->op_params[1]!=GGML_DECODE27_TOPK_TAG||
        !d->src[0]||!d->src[1]||!d->src[2]||!d->src[3]||d->src[4]||
        d->type!=GGML_TYPE_I32||!ggml_is_contiguous(d)||ggml_nrows(d)!=1) return false;
    const ggml_tensor * score=d->src[0],*producer=score;
    if (producer->op==GGML_OP_RESHAPE) producer=producer->src[0];
    const ggml_tensor * mask=d->src[3];
    return ggml_decode27_producer_eligible(producer)&&score->type==GGML_TYPE_F32&&
        ggml_is_contiguous(score)&&score->ne[0]==d->op_params[2]&&ggml_nrows(score)==1&&
        d->op_params[2]==producer->src[1]->ne[2]&&d->src[1]==producer->src[5]&&
        d->src[2]==producer->src[4]&&ggml_decode27_shape_eligible(score->ne[0],d->src[1]->ne[0],1,d->ne[0])&&
        mask->type==GGML_TYPE_F32&&ggml_is_contiguous(mask)&&mask->ne[0]==d->ne[0]&&ggml_nrows(mask)==1;
}
static inline ggml_tensor * ggml_top_k_decode27(ggml_context * ctx,ggml_tensor * full,int topk,
        ggml_tensor ** selected_mask) {
    if (!selected_mask||!ggml_decode27_enabled()||!ggml_decode27_producer_eligible(full)||
        !ggml_decode27_shape_eligible(full->ne[0],full->src[5]->ne[0],1,topk)) return nullptr;
    ggml_tensor * score=ggml_reshape_2d(ctx,full,full->ne[0],1);
    ggml_tensor * mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,topk,1);ggml_set_output(mask);
    ggml_tensor * ids=ggml_top_k(ctx,score,topk);
    ids->op_params[1]=GGML_DECODE27_TOPK_TAG;ids->op_params[2]=(int32_t)full->ne[0];
    ids->src[1]=full->src[5];ids->src[2]=full->src[4];ids->src[3]=mask;
    ggml_tensor * dependent=ggml_view_tensor(ctx,mask);dependent->src[1]=ids;
    *selected_mask=dependent;return ids;
}
static inline bool ggml_decode27_test_mask_eligible(const ggml_tensor * d) {
#if defined(STRATA_DECODE27_CAPTURE_OBSERVATION)
    if (!d||d->op!=GGML_OP_TOP_K||d->op_params[1]!=GGML_DECODE27_MASK_TAG||
        !d->src[0]||!d->src[1]||d->src[2]||d->type!=GGML_TYPE_F32||
        !ggml_is_contiguous(d)||ggml_nrows(d)!=1||d->ne[0]<1||d->ne[0]>512||
        d->op_params[2]<=1024||d->op_params[2]>1048576) return false;
    const ggml_tensor * ids=d->src[0],*v=d->src[1];
    return ids->type==GGML_TYPE_I32&&ggml_is_contiguous(ids)&&ids->ne[0]==d->ne[0]&&
        ggml_nrows(ids)==1&&v->type==GGML_TYPE_F32&&ggml_is_contiguous(v)&&ggml_nelements(v)==1;
#else
    (void)d;return false;
#endif
}
#if defined(STRATA_DECODE27_CAPTURE_OBSERVATION)
// Test-only consumer: the SINGLE ordinary selector's IDs feed both independent
// mask/gather arms. It does not create or replace a selector.
static inline ggml_tensor * ggml_decode27_selected_mask_for_test(ggml_context * ctx,
        ggml_tensor * ids,ggml_tensor * visible,int cap) {
    if (!ggml_decode27_enabled()||!ids||!visible||ids->type!=GGML_TYPE_I32||
        !ggml_is_contiguous(ids)||ggml_nrows(ids)!=1||ids->ne[0]<1||ids->ne[0]>512||
        visible->type!=GGML_TYPE_F32||!ggml_is_contiguous(visible)||ggml_nelements(visible)!=1||
        cap<=1024||cap>1048576) return nullptr;
    ggml_tensor * mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ids->ne[0],1);
    mask->op=GGML_OP_TOP_K;mask->op_params[1]=GGML_DECODE27_MASK_TAG;mask->op_params[2]=cap;
    mask->src[0]=ids;mask->src[1]=visible;return mask;
}
#endif
// Host construction/capture entries, never GPU replay invocation counts.
extern "C" void decode27_li_host_counts(uint64_t out[4]);
extern "C" void decode27_topk_host_counts(uint64_t out[4]);
extern "C" void decode27_mask_host_counts(uint64_t out[2]);
#endif
