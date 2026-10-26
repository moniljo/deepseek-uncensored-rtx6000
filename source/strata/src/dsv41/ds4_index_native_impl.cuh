// Experimental exact Blackwell index scores, adapted from the local Xolotl
// source25 lightning-indexer.cu guarded native21 path and ggml mma.cuh (MIT).
// Copyright (c) 2023-2026 The ggml authors. Full license and source hashes:
// third_party/xolotl-indexer/NOTICE.md. Included inside ds4_kernels.cu's private
// namespace, after rows_at and the unchanged scalar index_score_kernel.
//
// No activation is requantized. Pack Q only when its F32 bits are exactly on a
// per-32-value E2M1/power-of-two grid. Guard exponents so every subset of each
// 128-term dot is exactly representable in F32. Native MMA then preserves the
// original sequential dot, followed by the same ordered per-head ReLU/FMA.
// Off-grid/nonfinite/unsafe inputs run a copy of the original scalar kernel.
#pragma once

namespace index_native {
enum Reason : int { OffGrid=1, NonfiniteQ=2, BadKScale=4, OperandRange=8,
    ProductRange=16, DotOverflow=32, Spread=64, NonfiniteW=128,
    WeightOverflow=256, Unsupported=512 };

__device__ __forceinline__ uint32_t decoded_bits(int code, int E) {
    const uint32_t twice[8]={0,1,2,3,4,6,8,12};
    const uint32_t m=twice[code&7], sign=(code&8)?0x80000000u:0;
    if (!m) return sign;
    const int l=31-__clz(m), e=E-1+l;
    if (e>127) return sign|0x7f800000u;
    if (e>=-126) return sign|(uint32_t(e+127)<<23)|((m<<(23-l))&0x7fffffu);
    return sign|(m<<(E+148));
}
__device__ __forceinline__ int exact_code(uint32_t u,int E) {
    #pragma unroll
    for(int c=0;c<8;++c)
        if(decoded_bits(c,E)==(u&0x7fffffffu)) return c|int((u>>28)&8);
    return -1;
}
__global__ void init(int* st,int nq,int nk) {
    const int i=threadIdx.x;
    if(i<18) st[i]=(i==2 || i==6)?1000:(i==3 || i==7)?-1000:0;
    if(i==14) st[i]=nq;
    if(i==15) st[i]=nk;
}
__global__ void pack_q(const float* q,int* words,uint32_t* scales,int* st,int nq) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=nq*32) return;
    const float* src=q+(size_t)row*128;
    int lo=1000,hi=-1000,count=0;
    #pragma unroll
    for(int b=0;b<4;++b) {
        uint32_t mx=0; bool finite=true;
        #pragma unroll
        for(int j=0;j<32;++j) {
            const uint32_t u=__float_as_uint(src[b*32+j])&0x7fffffffu;
            mx=max(mx,u); finite=finite && u<0x7f800000u;
        }
        int E=-126; bool found=mx==0;
        if(!finite) atomicOr(st+1,NonfiniteQ);
        if(finite && mx) {
            const int field=int(mx>>23);
            const int floor=field?field-127:31-__clz(mx)-149;
            #pragma unroll
            for(int c=0;c<4;++c) {
                const int candidate=floor-2+c;
                if(found || candidate< -126 || candidate>127) continue;
                bool all=true;
                #pragma unroll
                for(int j=0;j<32;++j)
                    all=all && exact_code(__float_as_uint(src[b*32+j]),candidate)>=0;
                if(all) { E=candidate; found=true; }
            }
            if(!found) atomicOr(st+1,OffGrid);
        }
        #pragma unroll
        for(int word=0;word<4;++word) {
            uint32_t packed=0;
            #pragma unroll
            for(int j=0;j<4;++j) {
                const int at=word*4+j;
                const int a=exact_code(__float_as_uint(src[b*32+at]),E);
                const int z=exact_code(__float_as_uint(src[b*32+at+16]),E);
                packed |= uint32_t((a>=0?a:0)|((z>=0?z:0)<<4))<<(8*j);
            }
            words[(size_t)row*16+b*4+word]=int(packed);
        }
        // This thread alone owns both scale words; no read/modify/write race.
        reinterpret_cast<uint8_t*>(scales+(size_t)row*2)[(b/2)*4+b%2]=uint8_t(E+127);
        if(mx) { lo=min(lo,E); hi=max(hi,E); ++count; }
    }
    if(count) {
        atomicMin(st+2,lo); atomicMax(st+3,hi);
        atomicMax(st+4,hi-lo); atomicAdd(st+5,count);
    }
}

// Strata's row is 64 adjacent-pair code bytes then four UE8M0 scales,
// padded to 80 bytes. A zero scale means an unwritten zero row in Strata;
// explicitly zero those packed operands instead of interpreting UE8M0 zero
// as the hardware's smallest nonzero scale.
__device__ __forceinline__ int code_at(const uint8_t* row,int i) {
    return (row[i/2]>>(4*(i&1)))&15;
}
__device__ __forceinline__ int key_word(const uint8_t* row,int b,int word) {
    if(row[64+b]==0) return 0;
    uint32_t v=0;
    #pragma unroll
    for(int j=0;j<4;++j) {
        const int at=b*32+word*4+j;
        v |= uint32_t(code_at(row,at)|(code_at(row,at+16)<<4))<<(j*8);
    }
    return int(v);
}
__global__ void k_stats(const uint8_t* k,int* st,int nk) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    int lo=1000,hi=-1000,count=0,reasons=0;
    if(row<nk) {
        const uint8_t* src=k+(size_t)row*kv_row_bytes(kKvFp4p,128);
        #pragma unroll
        for(int b=0;b<4;++b) {
            const int byte=src[64+b], E=byte-127;
            if(byte==255) reasons|=BadKScale;
            bool nz=false;
            #pragma unroll
            for(int j=0;j<16;++j) nz=nz || (src[b*16+j]&0x77)!=0;
            if(byte!=0 && nz) { lo=min(lo,E); hi=max(hi,E); ++count; }
        }
    }
    int spread=count?hi-lo:0;
    #pragma unroll
    for(int o=16;o;o>>=1) {
        lo=min(lo,__shfl_down_sync(0xffffffffu,lo,o));
        hi=max(hi,__shfl_down_sync(0xffffffffu,hi,o));
        spread=max(spread,__shfl_down_sync(0xffffffffu,spread,o));
        count+=__shfl_down_sync(0xffffffffu,count,o);
        reasons|=__shfl_down_sync(0xffffffffu,reasons,o);
    }
    __shared__ int partial[5][4];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    if(lane==0) {
        partial[0][warp]=lo; partial[1][warp]=hi; partial[2][warp]=spread;
        partial[3][warp]=count; partial[4][warp]=reasons;
    }
    __syncthreads();
    if(warp==0) {
        lo=lane<4?partial[0][lane]:1000; hi=lane<4?partial[1][lane]:-1000;
        spread=lane<4?partial[2][lane]:0; count=lane<4?partial[3][lane]:0;
        reasons=lane<4?partial[4][lane]:0;
        #pragma unroll
        for(int o=16;o;o>>=1) {
            lo=min(lo,__shfl_down_sync(0xffffffffu,lo,o));
            hi=max(hi,__shfl_down_sync(0xffffffffu,hi,o));
            spread=max(spread,__shfl_down_sync(0xffffffffu,spread,o));
            count+=__shfl_down_sync(0xffffffffu,count,o);
            reasons|=__shfl_down_sync(0xffffffffu,reasons,o);
        }
        if(lane==0) {
            if(reasons) atomicOr(st+1,reasons);
            if(count) {
                atomicMin(st+6,lo); atomicMax(st+7,hi);
                atomicMax(st+8,spread); atomicAdd(st+9,count);
            }
        }
    }
}
__global__ void w_stats(const float* w,int* st,int nq) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nq*32) return;
    const uint32_t u=__float_as_uint(w[(size_t)(i/32)*64+i%32])&0x7fffffffu;
    if(u>=0x7f800000u) atomicOr(st+1,NonfiniteW);
    else atomicMax(st+10,int(u));
}
__global__ void finalize(int* st) {
    int reasons=st[1],wc=0; const uint32_t wm=uint32_t(st[10]);
    if(wm) {
        const int field=int(wm>>23);
        const int floor=field?field-127:31-__clz(wm)-149;
        const bool power=field?(wm&0x7fffffu)==0:(wm&(wm-1))==0;
        wc=floor+int(!power);
    }
    st[13]=wc;
    if((st[5] && (st[2]<-125 || st[3]>125)) ||
       (st[9] && (st[6]<-125 || st[7]>125))) reasons|=OperandRange;
    if(st[5] && st[9]) {
        const int pmin=st[2]+st[6],pmax=st[3]+st[7]; st[11]=pmin; st[12]=pmax;
        if(pmin<-124) reasons|=ProductRange;
        if(pmax>115) reasons|=DotOverflow;
        // Four 32-product blocks; every subset is below 2^24 integer units:
        // 32*144*(1+3*2^10)=14,160,384. Therefore MMA's partial order is exact.
        if(st[4]+st[8]>10) reasons|=Spread;
        if(wm && pmax+wc>108) reasons|=WeightOverflow;
    }
#if !defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__HIPCC__)
    reasons|=Unsupported;
#endif
    st[1]=reasons; st[0]=reasons==0;
}

__device__ __forceinline__ void mma(float (&d)[4],const int (&a)[4],
                                    const int (&b)[2],uint32_t as,uint32_t bs) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) && !defined(__HIPCC__)
    asm volatile(
        "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(as), "r"(bs));
#endif
}
template<int QT,bool CAND>
__global__ void native_scores(int* st,const int* words,const uint32_t* scales,
    const uint8_t* k,const float* w,int nq,int n,int nk,const int* cand,int cand_stride,
    int blk,Rows lim,float* out,long long stride,bool observe) {
    if(!st[0]) return;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) && !defined(__HIPCC__)
    if(observe && threadIdx.x==0 && threadIdx.y==0) atomicAdd(st+16,1);
    constexpr int KT=128,GROUPS=4;
    const int lane=threadIdx.x,warp=threadIdx.y,tid=warp*32+lane;
    const int qfirst=blockIdx.y*QT,kfirst=blockIdx.x*KT;
    int B[GROUPS][2][2]; uint32_t sb[GROUPS][2];
    int ln=0,lk=0;
    if(CAND) rows_at(lim,qfirst,ln,lk);
    #pragma unroll
    for(int group=0;group<GROUPS;++group) {
        const int i=kfirst+group*32+warp*8+lane/4;
        int pos=i;
        bool valid=i<n && i<nk;
        if(CAND) {
            valid=i<n && i<lk*blk;
            if(valid) { pos=cand[(size_t)qfirst*cand_stride+i/blk]*blk+i%blk; valid=pos>=0 && pos<ln; }
        }
        const uint8_t* row=valid?k+(size_t)pos*kv_row_bytes(kKvFp4p,128):nullptr;
        #pragma unroll
        for(int frag=0;frag<2;++frag) {
            #pragma unroll
            for(int l=0;l<2;++l) {
                const int word=l*4+lane%4,b32=frag*2+word/4;
                B[group][frag][l]=valid?key_word(row,b32,word%4):0;
            }
            sb[group][frag]=valid?uint32_t(row[64+frag*2])|(uint32_t(row[65+frag*2])<<8):0x7f7f;
        }
    }
    __shared__ float hd[16][KT];
    __shared__ __align__(32) int qw[16][16];
    #pragma unroll 1
    for(int qi=0;qi<QT;++qi) {
        const int query=qfirst+qi;
        if(query>=nq) break;
        float score[GROUPS]={};
        #pragma unroll
        for(int h0=0;h0<32;h0+=16) {
            #pragma unroll
            for(int at=tid;at<16*16;at+=128)
                qw[at/16][at%16]=words[((size_t)query*32+h0+at/16)*16+at%16];
            __syncthreads();
            float d[GROUPS][4]={};
            #pragma unroll
            for(int frag=0;frag<2;++frag) {
                int a[4];
                const int* xs=&qw[lane%16][frag*8+(lane/16)*4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "l"(xs));
                const int sh=h0+lane/4+(lane%2)*8;
                const uint32_t sa=scales[((size_t)query*32+sh)*2+frag];
                #pragma unroll
                for(int group=0;group<GROUPS;++group) mma(d[group],a,B[group][frag],sa,sb[group][frag]);
            }
            #pragma unroll
            for(int group=0;group<GROUPS;++group) {
                #pragma unroll
                for(int l=0;l<4;++l)
                    hd[(l/2)*8+lane/4][group*32+warp*8+(lane%4)*2+l%2]=d[group][l];
            }
            __syncthreads();
            if(warp==0) {
                #pragma unroll
                for(int h=0;h<16;++h) {
                    #pragma unroll
                    for(int group=0;group<GROUPS;++group)
                        score[group]+=fmaxf(hd[h][group*32+lane],0.0f)*w[(size_t)query*64+h0+h];
                }
            }
            __syncthreads();
        }
        if(warp==0) {
            int qn,qk; rows_at(lim,query,qn,qk);
            #pragma unroll
            for(int group=0;group<GROUPS;++group) {
                const int i=kfirst+group*32+lane;
                if(i>=n) continue;
                int pos=i; bool valid=i<qn;
                if(CAND) {
                    valid=i<qk*blk;
                    if(valid) { pos=cand[(size_t)query*cand_stride+i/blk]*blk+i%blk; valid=pos>=0 && pos<qn; }
                }
                out[(long long)query*stride+i]=valid?score[group]:-INFINITY;
            }
        }
    }
#endif
}

// Intentionally preserve the original scalar FP4 operation order and layout.
// This copy is separate so DS4_INDEX_NATIVE absent uses the unchanged kernel.
__global__ void __launch_bounds__(kThreads) scalar_fallback(int* st,const float* q,const float* w,
    const uint8_t* kc,int n,const int* cand,int cand_stride,int blk,Rows lim,float* out,long long stride,bool observe) {
    if(st[0]) return;
    if(observe && threadIdx.x==0) atomicAdd(st+17,1);
    extern __shared__ float qs[];
    const int t=blockIdx.y;
    for(int i=threadIdx.x;i<32*128;i+=blockDim.x) qs[i]=q[(size_t)t*32*128+i];
    for(int i=threadIdx.x;i<32;i+=blockDim.x) qs[32*128+i]=w[(size_t)t*64+i];
    __syncthreads();
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float* o=out+(long long)t*stride;
    int ln,lk; rows_at(lim,t,ln,lk);
    int p=i;
    if(cand!=nullptr) {
        if(i>=lk*blk) { o[i]=-INFINITY; return; }
        p=cand[(size_t)t*cand_stride+i/blk]*blk+i%blk;
    }
    if(p>=ln) { o[i]=-INFINITY; return; }
    const uint8_t* kr=kc+(size_t)p*kv_row_bytes(kKvFp4p,128);
    float acc[32];
    #pragma unroll
    for(int h=0;h<32;++h) acc[h]=0.0f;
    #pragma unroll 2
    for(int c=0;c<128;c+=8) {
        float kv[8];
        const uint32_t cw=*(const uint32_t*)(kr+c/2);
        const float ks=ue8m0_val(kr[64+c/32]);
        #pragma unroll
        for(int x=0;x<8;++x) kv[x]=e2m1_val14(cw>>(4*x))*16384.0f*ks;
        #pragma unroll
        for(int h=0;h<32;++h) {
            const float* qh=qs+h*128+c;
            #pragma unroll
            for(int x=0;x<8;++x) acc[h]+=qh[x]*kv[x];
        }
    }
    float sum=0.0f;
    #pragma unroll
    for(int h=0;h<32;++h) sum+=fmaxf(acc[h],0.0f)*qs[32*128+h];
    o[i]=sum;
}

struct Profile {
    unsigned long long calls=0, queries=0, native_calls=0, fallback_calls=0;
    unsigned long long native_queries=0, fallback_queries=0, reason_or=0;
};
// One thread updates its stream's counters in stream order. Captured graph
// replay counts device executions, rather than host dispatch or graph capture.
__global__ void profile_record(const int* state, Profile* p, int queries) {
    ++p->calls;
    p->queries += (unsigned long long) queries;
    if(state[0]) { ++p->native_calls; p->native_queries += (unsigned long long) queries; }
    else { ++p->fallback_calls; p->fallback_queries += (unsigned long long) queries; }
    p->reason_or |= (unsigned int) state[1];
}
static bool profile_enabled() {
    static const bool on=[] { const char* v=std::getenv("DS4_INDEX_STATS"); return v && std::strcmp(v,"1")==0; }();
    return on;
}
struct Workspace {
    int* state=nullptr; int* words=nullptr; uint32_t* scales=nullptr; size_t rows=0;
    Profile* profile=nullptr;
};
// Bounded lifetime: one workspace per (device, stream), reused in stream order.
// Allocation growth is exceptional; cudaFree drains any use of the old allocation.
static std::mutex ws_mutex;
static std::map<std::pair<int,cudaStream_t>,Workspace> workspaces;
static int ceil_div(int a,int b) { return (a+b-1)/b; }
static bool launch(const float* q,const float* w,const void* k,int heads,int hd,int T,int n,
    const int* cand,int cand_stride,int blk,const Rows& lim,float* out,long long stride,cudaStream_t s,int* diag) {
#if defined(__HIPCC__)
    return false;
#else
    if(heads!=32 || hd!=128 || T<1 || n<1 || blk<1 || (lim.gen==0 && T>kWin)) return false;
    int device=0; if(cudaGetDevice(&device)!=cudaSuccess) return false;
    cudaDeviceProp prop{};
    if(cudaGetDeviceProperties(&prop,device)!=cudaSuccess || prop.major!=12 || prop.minor!=0) return false;
    int nk=0;
    for(int t=0;t<T;++t) {
        int ln=0;
        if(lim.gen==0) ln=lim.n[t];
        else {
            if(lim.R<1) return false;
            const int nn=int((lim.p0+t+1)/lim.R);
            const int cb=std::min(lim.ccap,(nn+lim.blk-1)/lim.blk);
            ln=lim.gen==3?cb*lim.blk:lim.gen==4?(nn+lim.blk-1)/lim.blk:nn;
        }
        nk=std::max(nk,ln);
    }
    if(nk<1) return false;
    std::lock_guard<std::mutex> lock(ws_mutex);
    Workspace& ws=workspaces[{device,s}];
    const size_t rows=(size_t)T*32;
    if(ws.state==nullptr && cudaMalloc((void**)&ws.state,18*sizeof(int))!=cudaSuccess) return false;
    if(rows>ws.rows) {
        int* nw=nullptr; uint32_t* ns=nullptr;
        if(cudaMalloc((void**)&nw,rows*16*sizeof(int))!=cudaSuccess) return false;
        if(cudaMalloc((void**)&ns,rows*2*sizeof(uint32_t))!=cudaSuccess) { cudaFree(nw); return false; }
        if(ws.words) cudaFree(ws.words);
        if(ws.scales) cudaFree(ws.scales);
        ws.words=nw; ws.scales=ns; ws.rows=rows;
    }
    // Stats are a separate opt-in. Lazy allocation/initialization occurs once
    // per stream; there is no D2H copy or stream synchronization in score calls.
    const bool profiling=profile_enabled();
    if(profiling && ws.profile==nullptr) {
        if(cudaMalloc((void**)&ws.profile,sizeof(Profile))!=cudaSuccess) return false;
        cudaMemsetAsync(ws.profile,0,sizeof(Profile),s);
    }
    init<<<1,32,0,s>>>(ws.state,T,nk);
    cudaMemsetAsync(ws.scales,0,rows*2*sizeof(uint32_t),s);
    pack_q<<<ceil_div((int)rows,128),128,0,s>>>(q,ws.words,ws.scales,ws.state,T);
    k_stats<<<ceil_div(nk,128),128,0,s>>>((const uint8_t*)k,ws.state,nk);
    w_stats<<<ceil_div((int)rows,128),128,0,s>>>(w,ws.state,T);
    finalize<<<1,1,0,s>>>(ws.state);
    if(profiling) profile_record<<<1,1,0,s>>>(ws.state,ws.profile,T);
    if(cand) native_scores<1,true><<<dim3(ceil_div(n,128),T),dim3(32,4),0,s>>>(ws.state,ws.words,ws.scales,
        (const uint8_t*)k,w,T,n,nk,cand,cand_stride,blk,lim,out,stride,diag!=nullptr);
    else native_scores<16,false><<<dim3(ceil_div(n,128),ceil_div(T,16)),dim3(32,4),0,s>>>(ws.state,ws.words,ws.scales,
        (const uint8_t*)k,w,T,n,nk,nullptr,0,blk,lim,out,stride,diag!=nullptr);
    scalar_fallback<<<dim3(ceil_div(n,kThreads),T),kThreads,(32*128+32)*sizeof(float),s>>>(ws.state,q,w,
        (const uint8_t*)k,n,cand,cand_stride,blk,lim,out,stride,diag!=nullptr);
    if(diag) cudaMemcpyAsync(diag,ws.state,18*sizeof(int),cudaMemcpyDeviceToDevice,s);
    return true;
#endif
}
static void profile_dump(cudaStream_t s,bool reset) {
    if(!profile_enabled()) return;
#if !defined(__HIPCC__)
    int device=0;
    const cudaError_t dev_status=cudaGetDevice(&device);
    if(dev_status!=cudaSuccess) {
        std::fprintf(stderr,"DS4_INDEX_STATS error=%d stage=device\n",(int)dev_status);
        return;
    }
    std::lock_guard<std::mutex> lock(ws_mutex);
    const auto found=workspaces.find({device,s});
    Profile* ptr=found==workspaces.end()?nullptr:found->second.profile;
    Profile totals{};
    if(ptr) {
        cudaError_t status=cudaMemcpyAsync(&totals,ptr,sizeof(totals),cudaMemcpyDeviceToHost,s);
        if(status==cudaSuccess) status=cudaStreamSynchronize(s);
        if(status!=cudaSuccess) {
            std::fprintf(stderr,"DS4_INDEX_STATS device=%d error=%d stage=read\n",device,(int)status);
            return;
        }
    }
    std::fprintf(stderr,
        "DS4_INDEX_STATS device=%d allocated=%d calls=%llu queries=%llu native_calls=%llu fallback_calls=%llu "
        "native_queries=%llu fallback_queries=%llu reason_or=0x%llx reset=%d\n",
        device,ptr!=nullptr,totals.calls,totals.queries,totals.native_calls,totals.fallback_calls,
        totals.native_queries,totals.fallback_queries,totals.reason_or,reset);
    std::fflush(stderr);
    if(reset && ptr) {
        const cudaError_t status=cudaMemsetAsync(ptr,0,sizeof(Profile),s);
        if(status!=cudaSuccess)
            std::fprintf(stderr,"DS4_INDEX_STATS device=%d error=%d stage=reset\n",device,(int)status);
    }
#endif
}
static bool enabled() {
    static const bool on=[] { const char* v=std::getenv("DS4_INDEX_NATIVE"); return v && std::strcmp(v,"1")==0; }();
    return on;
}
} // namespace index_native
