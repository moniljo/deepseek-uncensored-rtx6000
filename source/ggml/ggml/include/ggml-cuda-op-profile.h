#pragma once
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif
// Host-thread-local diagnostic only. Must be called on the same owning thread
// that submits this backend's graphs, idle between requests. Environment-based
// profiling must be off. begin:0 success,1 invalid backend,2 env profiler on,
// 3 another next-request session active. No CUDA operation in begin.
GGML_BACKEND_API int ggml_backend_cuda_op_profile_next_begin(ggml_backend_t backend);
// Optional bounded metadata note, no CUDA operation. Inactive session returns0
// without even examining backend. Current active owning thread/device only.
// entry_kind is exactly attn_router_span / attn_router_span_kv / attn_router.
// Labels identify the entry point, not proof of producer-vs-YOCO-replay role;
// actual absolute position is logged to permit explicit later classification.
// Invalid metadata returns5 and clears note-known; no stale label is reused.
GGML_BACKEND_API int ggml_backend_cuda_op_profile_next_span(ggml_backend_t backend,int il,int64_t col0,
                                                          int64_t n,int64_t first_position,const char * entry_kind);
// Final cumulative report and event destruction after sampled graph streams
// have completed. end:0 success,1 invalid backend,4 absent/wrong-device session.
GGML_BACKEND_API int ggml_backend_cuda_op_profile_next_end(ggml_backend_t backend);
// Separate default-off whole-prefill diagnostics. Set
// GGML_CUDA_OP_PROFILE_PREFILL=1 before process start. Notes are consumed at
// every graph begin, including rejected graphs. This mode is independent of
// the legacy ordinal sampler and mutually exclusive with PROFILE_NEXT.
// entry_kind: attn_router_span / attn_router_span_kv / finish_span. The existing
// attn_router note is also accepted and consumed, but not sampled by this mode.
// Inactive calls return0 without examining the backend or doing CUDA work.
GGML_BACKEND_API int ggml_backend_cuda_op_profile_prefill_span(ggml_backend_t backend,int il,int64_t col0,
                                                             int64_t n,int64_t first_position,const char * entry_kind);
// Flush complete selected graphs, print every coverage stratum and free the
// event pool before decode. The process-local session cannot sample another
// request; use a fresh process for a fresh diagnostic observation.
GGML_BACKEND_API int ggml_backend_cuda_op_profile_prefill_end(ggml_backend_t backend);
// Host-thread-local nested stage diagnostics for a currently sampled graph.
// No CUDA operation when inactive. stream is an existing cudaStream_t cast to
// void*. Labels must contain only letters, digits, '.', '_' or '-', max63bytes.
// Tokens remain valid until graph completion; end must be called once before
// that completion. Nested timings overlap their enclosing op and are excluded
// from the enclosing graph's nonnested stream-time sum.
GGML_BACKEND_API int ggml_backend_cuda_op_profile_sample_active(void);
GGML_BACKEND_API void * ggml_backend_cuda_op_profile_nested_begin(const char * label,void * stream);
GGML_BACKEND_API void ggml_backend_cuda_op_profile_nested_end(void * token);
#ifdef __cplusplus
}
#endif
