// Synthetic gates for the experimental exact Blackwell FP4 indexer.
// No model, tokenizer, server, credentials, forced tokens or prompt shortcuts.
#include "ds4_index_native.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::ds4;
static void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static uint32_t bits(float x) { uint32_t u; std::memcpy(&u,&x,4); return u; }
static float frombits(uint32_t u) { float x; std::memcpy(&x,&u,4); return x; }
static constexpr uint32_t Canary=0x4bbad00d;
template<class T> struct Device {
    T* p=nullptr; size_t n;
    explicit Device(size_t count):n(count) { ck(cudaMalloc((void**)&p,n*sizeof(T))); }
    ~Device() { if(p) cudaFree(p); }
    Device(const Device&)=delete;
    void put(const std::vector<T>& x,cudaStream_t s) { if(x.size()!=n) throw std::runtime_error("input size"); ck(cudaMemcpyAsync(p,x.data(),n*sizeof(T),cudaMemcpyHostToDevice,s)); }
    std::vector<T> get(cudaStream_t s) { std::vector<T> x(n); ck(cudaMemcpyAsync(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost,s)); ck(cudaStreamSynchronize(s)); return x; }
};
static uint32_t rnd(uint32_t& s) { s^=s<<13; s^=s>>17; s^=s<<5; return s; }
static float value(int code,int exponent) {
    static const float v[8]={0,.5f,1,1.5f,2,3,4,6};
    return std::copysign(std::ldexp(v[code&7],exponent),(code&8)?-1.0f:1.0f);
}
static std::pair<int,int> sizes(const Rows& r,int t) {
    if(!r.gen) return {r.n[t],r.k[t]};
    int nn=int((r.p0+t+1)/r.R),cb=std::min(r.ccap,(nn+r.blk-1)/r.blk);
    switch(r.gen) {
    case 1:return {nn,std::min(r.kcap,nn)};
    case 2:return {nn,cb};
    case 3:return {cb*r.blk,std::min(r.kcap,nn)};
    default:return {(nn+r.blk-1)/r.blk,r.kcap};
    }
}
struct Case {
    std::string name; int T=1,n=129,keys=129,blk=8,mode=0;
    bool candidates=false,expect_native=true,cpu=true,capture=false;
    Rows lim{};
};
static void compare_float_bits(const std::vector<float>& a,const std::vector<float>& b,const char* label) {
    if(a.size()!=b.size()) throw std::runtime_error("comparison size");
    for(size_t i=0;i<a.size();++i) if(bits(a[i])!=bits(b[i])) {
        std::fprintf(stderr,"%s at %zu: %08x != %08x\n",label,i,bits(a[i]),bits(b[i]));
        throw std::runtime_error(label);
    }
}
static void compare_ints(const std::vector<int>& a,const std::vector<int>& b,const char* label) {
    if(a!=b) throw std::runtime_error(label);
}
static void run(Case c,cudaStream_t stream,bool bench=false) {
    constexpr int H=32,D=128,CS=64,Guard=32;
    const int stride=c.n+11,ts=37,cs=17;
    std::vector<float> q((size_t)c.T*H*D),w((size_t)c.T*64,frombits(Canary));
    std::vector<uint8_t> key((size_t)c.keys*kv_row_bytes(kKvFp4p,D),0);
    std::vector<int> cand((size_t)c.T*CS,0);
    uint32_t seed=0x7147abcdu;
    for(int t=0;t<c.T;++t) {
        for(int h=0;h<H;++h) {
            w[(size_t)t*64+h]=float(int(rnd(seed)%25)-12)*.125f;
            for(int b=0;b<4;++b) {
                const int E=(b+h+t)%4-3;
                for(int j=0;j<32;++j) q[((size_t)t*H+h)*D+b*32+j]=value(rnd(seed)%16,E);
            }
        }
        const auto sz=sizes(c.lim,t);
        for(int b=0;b<CS;++b)
            cand[(size_t)t*CS+b]=((b*7+t*3)%std::max(1,(sz.first+c.blk-1)/c.blk));
    }
    for(int p=0;p<c.keys;++p) {
        uint8_t* row=key.data()+(size_t)p*kv_row_bytes(kKvFp4p,D);
        for(int b=0;b<4;++b) {
            row[64+b]=uint8_t(127+(b+p)%3-2);
            for(int j=0;j<16;++j) row[b*16+j]=uint8_t(rnd(seed));
        }
    }
    // Modes exercise exact ties, zero scales, guarded transitions and unsafe inputs.
    if(c.mode==1) std::fill(q.begin(),q.end(),0.0f);
    if(c.mode==2) for(int p=0;p<c.keys;++p) std::copy(key.begin(),key.begin()+80,key.begin()+(size_t)p*80);
    if(c.mode==3) q[0]=.1f;
    if(c.mode==4) q[5]=std::numeric_limits<float>::infinity();
    if(c.mode==5) w[3]=std::numeric_limits<float>::quiet_NaN();
    if(c.mode==6) key[64]=255;
    if(c.mode==7) for(int t=0;t<c.T;++t) for(int h=0;h<H;++h)
        for(int b=0;b<4;++b) for(int j=0;j<32;++j)
            q[((size_t)t*H+h)*D+b*32+j]=value((j+3)%16,b==0?-16:0);
    if(c.mode==8) for(int p=0;p<c.keys;++p) for(int b=0;b<4;++b) key[(size_t)p*80+64+b]=0;
    if(c.mode==9) {
        for(auto& x:q) x=value(7,-100);
        for(int p=0;p<c.keys;++p) for(int b=0;b<4;++b) key[(size_t)p*80+64+b]=27;
    }
    const auto q0=q,w0=w; const auto k0=key; const auto ca0=cand;
    const size_t os=(size_t)c.T*stride+2*Guard;
    std::vector<float> sentinel(os,frombits(Canary));
    Device<float> dq(q.size()),dw(w.size()),ref(os),got(os);
    Device<uint8_t> dk(key.size()); Device<int> dc(cand.size()),stats(18);
    dq.put(q,stream); dw.put(w,stream); dk.put(key,stream); dc.put(cand,stream);
    ref.put(sentinel,stream); got.put(sentinel,stream);
    const int* cp=c.candidates?dc.p:nullptr;
    index_score(dq.p,dw.p,dk.p,H,D,c.T,c.n,cp,CS,c.blk,c.lim,ref.p+Guard,stride,stream);
    if(!index_score_native_diagnostic(dq.p,dw.p,dk.p,H,D,c.T,c.n,cp,CS,c.blk,c.lim,
                                     got.p+Guard,stride,stream,stats.p)) throw std::runtime_error("native shape declined");
    ck(cudaGetLastError());
    auto a=ref.get(stream),b=got.get(stream); auto st=stats.get(stream);
    compare_float_bits(a,b,"GPU score bits or guards");
    if(bool(st[0])!=c.expect_native || (st[0] && (st[16]<=0 || st[17]!=0)) ||
       (!st[0] && (st[16]!=0 || st[17]<=0))) throw std::runtime_error("guard/dispatch admission");
    if(c.cpu) {
        auto expected=sentinel;
        // Independent scalar F32/FMA oracle, directly decoding Strata cache bytes.
        for(int t=0;t<c.T;++t) {
            auto sz=sizes(c.lim,t);
            for(int i=0;i<c.n;++i) {
                int p=i; bool valid=true;
                if(c.candidates) { valid=i<sz.second*c.blk; if(valid) p=cand[(size_t)t*CS+i/c.blk]*c.blk+i%c.blk; }
                if(!valid || p>=sz.first) { expected[Guard+(size_t)t*stride+i]=-INFINITY; continue; }
                const uint8_t* kr=key.data()+(size_t)p*80;
                float sum=0;
                for(int h=0;h<H;++h) {
                    float dot=0;
                    for(int d=0;d<D;++d) {
                        const int code=(kr[d/2]>>(4*(d&1)))&15;
                        const float scale=frombits(uint32_t(kr[64+d/32])<<23);
                        const float kv=value(code,0)*scale;
                        dot=std::fma(q[((size_t)t*H+h)*D+d],kv,dot);
                    }
                    sum=std::fma(std::fmax(dot,0.0f),w[(size_t)t*64+h],sum);
                }
                expected[Guard+(size_t)t*stride+i]=sum;
            }
        }
        compare_float_bits(a,expected,"independent CPU F32 oracle");
    }
    // Verify original TopK raw order/ties and candidate-block producer byte for byte.
    Rows tr=c.lim;
    if(c.candidates) {
        if(tr.gen) tr.gen=3;
        else for(int t=0;t<c.T;++t) { tr.n[t]=std::min(c.n,tr.k[t]*c.blk); tr.k[t]=std::min(23,tr.n[t]); }
    } else {
        if(tr.gen) tr.kcap=23;
        else for(int t=0;t<c.T;++t) tr.k[t]=std::min(23,tr.n[t]);
    }
    std::vector<int> ti((size_t)c.T*ts+2*Guard,int(Canary));
    Device<int> ta(ti.size()),tb(ti.size()); ta.put(ti,stream); tb.put(ti,stream);
    topk(ref.p+Guard,stride,c.T,tr,cp,CS,c.blk,ta.p+Guard,ts,stream);
    topk(got.p+Guard,stride,c.T,tr,cp,CS,c.blk,tb.p+Guard,ts,stream);
    compare_ints(ta.get(stream),tb.get(stream),"raw TopK IDs/ties/guards");
    if(!c.candidates) {
        Rows cr=c.lim;
        if(cr.gen) cr.kcap=std::min(7,(c.n+c.blk-1)/c.blk);
        else for(int t=0;t<c.T;++t) cr.k[t]=std::min(7,(cr.n[t]+c.blk-1)/c.blk);
        const int bs=(c.n+c.blk-1)/c.blk+7;
        Device<float> ma((size_t)c.T*bs),mb((size_t)c.T*bs);
        std::vector<float> ms((size_t)c.T*bs,frombits(Canary)); ma.put(ms,stream); mb.put(ms,stream);
        std::vector<int> ci((size_t)c.T*cs+2*Guard,int(Canary));
        Device<int> ca(ci.size()),cb(ci.size()); ca.put(ci,stream); cb.put(ci,stream);
        cand_blocks(ref.p+Guard,stride,c.T,cr,c.blk,ma.p,bs,ca.p+Guard,cs,stream);
        cand_blocks(got.p+Guard,stride,c.T,cr,c.blk,mb.p,bs,cb.p+Guard,cs,stream);
        compare_ints(ca.get(stream),cb.get(stream),"candidate producer IDs/guards");
        compare_float_bits(ma.get(stream),mb.get(stream),"candidate producer maxima");
    }
    // Repeat native->fallback->native cases in the same stream/workspace. A graph
    // replay below changes input bytes in place without renewed host dispatch.
    if(c.capture) {
        cudaGraph_t graph=nullptr; cudaGraphExec_t exec=nullptr;
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        if(!index_score_native_diagnostic(dq.p,dw.p,dk.p,H,D,c.T,c.n,cp,CS,c.blk,c.lim,
                                         got.p+Guard,stride,stream,stats.p)) throw std::runtime_error("capture declined");
        ck(cudaStreamEndCapture(stream,&graph)); ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        ck(cudaGraphLaunch(exec,stream)); compare_float_bits(a,got.get(stream),"captured replay");
        q[0]=.1f; dq.put(q,stream); ref.put(sentinel,stream);
        index_score(dq.p,dw.p,dk.p,H,D,c.T,c.n,cp,CS,c.blk,c.lim,ref.p+Guard,stride,stream);
        ck(cudaGraphLaunch(exec,stream)); compare_float_bits(ref.get(stream),got.get(stream),"captured fallback transition");
        if(stats.get(stream)[0]!=0) throw std::runtime_error("capture must fallback");
        dq.put(q0,stream); ck(cudaGraphLaunch(exec,stream)); compare_float_bits(a,got.get(stream),"captured native restoration");
        if(stats.get(stream)[0]!=1) throw std::runtime_error("capture must restore native");
        ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph));
    }
    float scalar_ms=0,native_ms=0;
    if(bench) {
        cudaEvent_t start,stop; ck(cudaEventCreate(&start)); ck(cudaEventCreate(&stop));
        constexpr int rounds=5;
        for(int arm=0;arm<2;++arm) {
            ck(cudaEventRecord(start,stream));
            for(int rep=0;rep<rounds;++rep) {
                if(!arm) index_score(dq.p,dw.p,dk.p,H,D,c.T,c.n,cp,CS,c.blk,c.lim,ref.p+Guard,stride,stream);
                else index_score_native_diagnostic(dq.p,dw.p,dk.p,H,D,c.T,c.n,cp,CS,c.blk,c.lim,got.p+Guard,stride,stream,nullptr);
            }
            ck(cudaEventRecord(stop,stream)); ck(cudaEventSynchronize(stop));
            float ms=0; ck(cudaEventElapsedTime(&ms,start,stop));
            (arm?native_ms:scalar_ms)=ms/rounds;
        }
        ck(cudaEventDestroy(start)); ck(cudaEventDestroy(stop));
        compare_float_bits(ref.get(stream),got.get(stream),"post-timing exact scores");
    }
    compare_float_bits(dq.get(stream),q0,"Q input preserved");
    compare_float_bits(dw.get(stream),w0,"weights preserved");
    if(dk.get(stream)!=k0 || dc.get(stream)!=ca0) throw std::runtime_error("cache/candidate input changed");
    std::printf("{\"case\":\"%s\",\"passed\":true,\"queries\":%d,\"positions\":%d,\"native\":%d,\"reasons\":%d,\"native_ctas\":%d,\"scalar_ctas\":%d,\"cpu_oracle\":%s,\"scalar_ms\":%.6f,\"native_ms\":%.6f}\n",
        c.name.c_str(),c.T,c.n,st[0],st[1],st[16],st[17],c.cpu?"true":"false",scalar_ms,native_ms);
}
int main(int argc,char** argv) {
    try {
        if(std::getenv("DS4_INDEX_NATIVE")) throw std::runtime_error("unset DS4_INDEX_NATIVE: reference must use ordinary path");
        cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        bool bench=argc==2 && std::strcmp(argv[1],"--bench")==0;
        if(bench) {
            for(int n:{262144,524288,1048576}) {
                Case c; c.name="timing-"+std::to_string(n); c.T=32; c.n=c.keys=n; c.cpu=false;
                c.lim.gen=1; c.lim.R=1; c.lim.p0=n-32; c.lim.kcap=23;
                run(c,stream,true);
            }
        } else {
            { Case c; c.name="one-key"; c.n=c.keys=1; c.lim.n[0]=1; c.lim.k[0]=1; run(c,stream); }
            for(int T:{1,4,16,17,33}) for(int n:{7,129,513}) {
                if(T>2*n) continue;
                Case c; c.name="causal-"+std::to_string(T)+"-"+std::to_string(n); c.T=T; c.n=c.keys=n;
                c.lim.gen=1; c.lim.R=2; c.lim.p0=std::max(0,n*2-T); c.lim.kcap=23;
                c.capture=T==17 && n==129; run(c,stream);
            }
            for(int mode=1;mode<=9;++mode) {
                Case c; c.name="guard-mode-"+std::to_string(mode); c.T=4; c.mode=mode;
                c.expect_native=mode==1 || mode==2 || mode==8; c.cpu=mode!=4 && mode!=5 && mode!=6 && mode!=9;
                for(int t=0;t<c.T;++t) { c.lim.n[t]=c.n-t*5; c.lim.k[t]=23; }
                run(c,stream);
            }
            for(int gen:{0,2}) for(int T:{1,5,16}) {
                Case c; c.name="candidates-"+std::to_string(gen)+"-"+std::to_string(T);
                c.T=T; c.n=64; c.keys=std::max(513,499+T); c.candidates=true;
                c.lim.gen=gen; c.lim.R=1; c.lim.p0=499; c.lim.ccap=8; c.lim.blk=8; c.lim.kcap=23;
                for(int t=0;t<std::min(T,kWin);++t) { c.lim.n[t]=513-t*13; c.lim.k[t]=8-t%4; }
                run(c,stream);
            }
        }
        // With DS4_INDEX_STATS=1 these explicit summaries also exercise reset.
        // Default fixture: 31 calls/279 queries, 24 native/7 fallback calls,
        // 238 native/41 fallback queries (including three captured executions).
        // --bench: 18 calls/576 queries, all native. The second dump is all zero.
        // With stats unset these calls queue no work and print nothing.
        index_native_profile_dump(stream,true);
        index_native_profile_dump(stream,false);
        ck(cudaStreamDestroy(stream));
        std::puts("{\"passed\":true,\"scope\":\"synthetic exact index scores, raw topk, candidate producer, guards and capture\"}");
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"index native test failed: %s\n",e.what()); return 1;
    }
}
