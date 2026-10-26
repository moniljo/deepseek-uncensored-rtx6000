#pragma once
// Isolated MIT-derived G32 paired-PV experiment; no production dispatch changes.
namespace strata::ds4 { namespace { namespace attn_g32_pairs_study {
__global__ void __launch_bounds__(1024, 1) sparse_attn_g32_vec_kernel(const float* __restrict__ q,
                                                                      const uint16_t* __restrict__ win,
                                                                      const uint8_t* __restrict__ ckv,
                                                                      const int* __restrict__ sel, int sel_stride,
                                                                      AttnRows ar, const float* __restrict__ sink,
                                                                      int heads, float scale, float* __restrict__ out) {
    constexpr int d = 512, G = 32, W = 1, Stage = 32;   // 1024-thread CTA:32warps,one/head
    const int hb = blockIdx.x * G, t = blockIdx.y;
    int n_win, n_sel, w0;
    if (ar.gen) {
        const long long ta = ar.t0 + t, pa = ar.p0 + ta;
        const long long first = max((long long) ar.win - ar.p0, ta + 1);
        w0 = (int) first;
        n_win = (int) (ar.win + ta - first + 1);
        n_sel = ar.R > 0 ? (int) min((long long) ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_win = ar.nwin[t], n_sel = ar.nsel[t], w0 = ar.win0[t];
    }
    const uint16_t* wbase = win + (size_t) w0 * d;
    const int* sl = sel + (size_t) t * sel_stride;
    const size_t crow = kv_row_bytes(kKvFp4, d);
    __shared__ __align__(16) uint16_t tile[Stage * d];
    __shared__ float sc[G][kAttTile];
    __shared__ const uint8_t* rows[kAttTile];
    __shared__ float m_run_s[G], tile_scale_s[G], den_s[G];
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, gw = wp / W, wr = wp % W;
    float qr[16];   // this lane's 16 values of its warp's head
#pragma unroll
    for (int x = 0; x < 16; ++x) qr[x] = q[((size_t) t * heads + hb + gw) * d + lane * 16 + x];
    if (threadIdx.x < G) {
        m_run_s[threadIdx.x] = -1e30f;
        den_s[threadIdx.x] = 0.0f;
    }
    float o0[8], o1[8];
#pragma unroll
    for (int g = 0; g < 8; ++g) o0[g] = o1[g] = 0.0f;
    const int c = 2 * (threadIdx.x & 255);
    const int headbase = (threadIdx.x >> 8) * 8;
    const int sr = threadIdx.x >> 4, part = threadIdx.x & 15;   // the staging: a thread's row and 32 values
    // rows [s0, s0 + n) of the tile at t0 into tile[] as BF16
    auto stage = [&](int t0, int s0, int n) {
        if (sr >= n) return;
        const int j = s0 + sr;
        uint4* dst = (uint4*) (tile + sr * d + part * 32);
        if (t0 + j < n_win) {   // a window row: BF16 already
            const uint4* src = (const uint4*) (rows[j] + part * 64);
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = src[3];
            return;
        }
        unpack32(rows[j], part, dst);
    };
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += kAttTile) {
        const int nt = min(kAttTile, total - t0);
        if (threadIdx.x < nt) {
            const int j = t0 + threadIdx.x;
            rows[threadIdx.x] = j < n_win ? (const uint8_t*) (wbase + (size_t) j * d) : ckv + (size_t) sl[j - n_win] * crow;
        }
        __syncthreads();
        for (int s0 = 0; s0 < nt; s0 += Stage) {   // the dots
            const int n = min(Stage, nt - s0);
            stage(t0, s0, n);
            __syncthreads();
#if !defined(STRATA_USE_HIP)   // (HIP leaves a loop with shuffles and a remainder rolled anyway)
#pragma unroll 2
#endif
            for (int r = wr; r < n; r += W) {   // (two rows' chains at once)
                const uint16_t* vr = tile + r * d + lane * 16;
                const uint4 a = *(const uint4*) vr;
                const uint4 b = *(const uint4*) (vr + 8);
                const uint16_t* pa = (const uint16_t*) &a;
                const uint16_t* pb = (const uint16_t*) &b;
                float s = 0.0f;
#pragma unroll
                for (int x = 0; x < 8; ++x) s += qr[x] * bf16f(pa[x]);
#pragma unroll
                for (int x = 0; x < 8; ++x) s += qr[8 + x] * bf16f(pb[x]);
                s = warp_sum(s);
                if (lane == 0) sc[gw][s0 + r] = s * scale;
            }
            __syncthreads();
        }
        if (wp < G) {   // head wp's softmax over the tile
            float* scg = sc[wp];
            float m = -INFINITY;
            for (int j = lane; j < nt; j += 32) m = fmaxf(m, scg[j]);
            m = warp_max(m);
            const float m_old = m_run_s[wp];
            const float m_new = fmaxf(m_old, m);
            float s = 0.0f;
            for (int j = lane; j < nt; j += 32) {
                const float p = expf(scg[j] - m_new);
                scg[j] = p;
                s += p;
            }
            s = warp_sum(s);
            __syncwarp();
            if (lane == 0) {
                tile_scale_s[wp] = expf(m_old - m_new);
                den_s[wp] = den_s[wp] * tile_scale_s[wp] + s;
                m_run_s[wp] = m_new;
            }
        }
        __syncthreads();
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const float cs = tile_scale_s[headbase + g];
            o0[g] *= cs;
            o1[g] *= cs;
        }
        for (int s0 = 0; s0 < nt; s0 += Stage) {   // the weighted sums
            const int n = min(Stage, nt - s0);
            stage(t0, s0, n);
            __syncthreads();
            for (int r = 0; r < n; ++r) {
                const uint32_t pr = *(const uint32_t*) (tile + r * d + c);
                const float v0 = bf16f((uint16_t) (pr & 0xffffu)), v1 = bf16f((uint16_t) (pr >> 16));
#pragma unroll
                for (int g = 0; g < 8; ++g) {
                    o0[g] += sc[headbase + g][s0 + r] * v0;
                    o1[g] += sc[headbase + g][s0 + r] * v1;
                }
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        const int head = headbase + g;
        const float den = den_s[head] + expf(sink[hb + head] - m_run_s[head]);
        float* ov = out + ((size_t) t * heads + hb + head) * d;
        ov[c] = o0[g] / den;
        ov[c + 1] = o1[g] / den;
    }
}
inline bool supported(const float* q,const uint16_t* win,const void* ckv,const int* sel,
                      int sel_stride,const AttnRows& a,const float* sink,int heads,int d,int T,
                      float* out,int ckv_fmt,int heads_per_block) {
    return heads_per_block==0 && a.gen==1 && a.R>0 && a.kcap==512 &&
        ckv_fmt==kKvFp4 && d==512 && heads==64 && T>=1024 && T<=65535 &&
        sel_stride>=512 && q!=nullptr && win!=nullptr && ckv!=nullptr &&
        sel!=nullptr && sink!=nullptr && out!=nullptr;
}
inline bool try_launch(bool enabled,const float* q,const uint16_t* win,const void* ckv,
                       const int* sel,int sel_stride,const AttnRows& a,const float* sink,
                       int heads,int d,int T,float scale,float* out,cudaStream_t stream,
                       int ckv_fmt,int heads_per_block) {
    if(!enabled || !supported(q,win,ckv,sel,sel_stride,a,sink,heads,d,T,out,ckv_fmt,heads_per_block))return false;
    sparse_attn_g32_vec_kernel<<<dim3(heads/32,T),1024,0,stream>>>(
        q,win,(const uint8_t*)ckv,sel,sel_stride,a,sink,heads,scale,out);
    return true;
}
}}} // strata::ds4 / private / study
