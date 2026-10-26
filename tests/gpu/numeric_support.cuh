// Fixture support only. The production staging method is extracted separately,
// byte for byte, after its source freeze. No model weights or services are read.
#pragma once
#include "ds4_gguf.cuh"
#include "ds4_iq.cuh"
#include "strata/dsv41/moe_win.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gg = strata::ds4::gguf;
namespace iq = strata::ds4::gguf::iq;
namespace x3 = strata::dsv41k::exl3;
namespace gf = strata::dsv41k::glmf;
inline void need(bool ok,const std::string& what){if(!ok)throw std::runtime_error(what);}
inline void cu(cudaError_t e,const char* at){if(e!=cudaSuccess)throw std::runtime_error(std::string(at)+": "+cudaGetErrorString(e));}
#define CU(x) cu((x),#x)
constexpr int D=5120,F=2304,K=6,PICKS=8,MAXT=4;
constexpr size_t GUARD=256;
inline size_t align256(size_t n){return (n+255)&~size_t(255);}
inline uint64_t hash64(const void* p,size_t n){auto b=static_cast<const uint8_t*>(p);uint64_t h=14695981039346656037ull;for(size_t i=0;i<n;++i){h^=b[i];h*=1099511628211ull;}return h;}
template<class T> struct Device {
    uint8_t* owner=nullptr;T* p=nullptr;size_t count=0;
    explicit Device(size_t n):count(n){CU(cudaMalloc(&owner,GUARD+n*sizeof(T)+GUARD));p=reinterpret_cast<T*>(owner+GUARD);CU(cudaMemset(owner,0xa5,GUARD+n*sizeof(T)+GUARD));CU(cudaDeviceSynchronize());} // join legacy initialization before nonblocking uploads
    Device(const Device&)=delete;Device&operator=(const Device&)=delete;
    ~Device(){if(owner)cudaFree(owner);}
    void set(const std::vector<T>& v,cudaStream_t s){need(v.size()==count,"upload extent");CU(cudaMemcpyAsync(p,v.data(),count*sizeof(T),cudaMemcpyHostToDevice,s));}
    std::vector<T> get(cudaStream_t s)const{std::vector<T>v(count);CU(cudaMemcpyAsync(v.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost,s));CU(cudaStreamSynchronize(s));return v;}
    void audit(cudaStream_t s)const{std::array<uint8_t,GUARD>a{},b{};CU(cudaMemcpyAsync(a.data(),owner,GUARD,cudaMemcpyDeviceToHost,s));CU(cudaMemcpyAsync(b.data(),owner+GUARD+count*sizeof(T),GUARD,cudaMemcpyDeviceToHost,s));CU(cudaStreamSynchronize(s));for(size_t i=0;i<GUARD;++i)need(a[i]==0xa5&&b[i]==0xa5,"device canary");}
};
struct Family {int gate,down;const char* name;};
inline const std::array<Family,8> families{{{22,18,"IQ2_S-IQ3_XXS"},{18,22,"IQ3_XXS-IQ2_S"},{18,18,"IQ3_XXS-IQ3_XXS"},{18,21,"IQ3_XXS-IQ3_S"},{18,14,"IQ3_XXS-Q6_K"},{21,14,"IQ3_S-Q6_K"},{14,21,"Q6_K-IQ3_S"},{39,39,"MXFP4-MXFP4"}}};

struct Experts {
    int ne; x3::ExpertLayout lay{};size_t blob=0;
    std::array<size_t,3> bytes{};
    std::vector<uint8_t> host;
    std::unique_ptr<Device<uint8_t>> resident;
    Experts(Family family,int count,cudaStream_t s):ne(count){
        lay.fmt=gg::kFmtMXFP4;
        for(int r=0;r<3;++r){const int type=r<2?family.gate:family.down;lay.K[r]=type;lay.trellis[r]=blob;lay.suh[r]=lay.svh[r]=blob;bytes[r]=gg::type_row_bytes(type,r<2?D:F)*(r<2?F:D);need(bytes[r]>0,"supported matrix type");blob+=align256(bytes[r]);}
        host.assign(blob*(size_t)ne,0x6d);
        for(int e=0;e<ne;++e)for(int r=0;r<3;++r){
            const int type=r<2?family.gate:family.down;
            const int bv=type==39?32:iq::block_values(type),bb=type==39?17:iq::block_bytes(type);
            need(bv>0&&bb>0&&bytes[r]%(size_t)bb==0,"whole legal blocks");
            auto p=host.data()+(size_t)e*blob+lay.trellis[r];
            for(size_t j=0;j<bytes[r];++j)p[j]=static_cast<uint8_t>(j*73u+(size_t)e*29u+(size_t)r*11u+(size_t)type);
            for(size_t j=0;j<bytes[r];j+=(size_t)bb){
                if(type==39)p[j]=static_cast<uint8_t>(111+(e+r)%4); // finite E8M0, all nibble codes retained
                else {const uint16_t h=static_cast<uint16_t>(0x1400+((e+r)&3)*0x40);std::memcpy(p+j+(type==14?208:0),&h,2);} // Q6 d is at208, not0
            }
        }
        resident=std::make_unique<Device<uint8_t>>(host.size());resident->set(host,s);CU(cudaStreamSynchronize(s));
    }
    unsigned long long ptr(int e)const{return reinterpret_cast<unsigned long long>(resident->p+(size_t)e*blob);}
    const uint8_t* role(int e,int r)const{return host.data()+(size_t)e*blob+lay.trellis[r];}
};

struct MathBuffers {
    int T;Device<float>x,weights,shared,h,part,out;Device<unsigned long long>plan,group;Device<int>starts,entries,ng;
    std::vector<float>input,weight,share;
    explicit MathBuffers(int t,cudaStream_t s):T(t),x((size_t)t*D),weights((size_t)t*8),shared((size_t)t*D),h((size_t)t*8*F),part((size_t)t*8*D),out((size_t)t*D),plan((size_t)t*8),group((size_t)t*8),starts((size_t)t*8+1),entries((size_t)t*8),ng(1){
        input.resize((size_t)t*D);share.resize(input.size());weight.assign((size_t)t*8,0);
        for(size_t i=0;i<input.size();++i){input[i]=static_cast<float>(static_cast<int>((i*17+11)%101)-50)*0.002f;share[i]=static_cast<float>(static_cast<int>(i%13)-6)*0.001f;}
        for(int q=0;q<t;++q)for(int k=0;k<K;++k)weight[(size_t)q*8+k]=0.07f+0.013f*k;
        x.set(input,s);weights.set(weight,s);shared.set(share,s);CU(cudaStreamSynchronize(s));
    }
    void launch(const Experts& E,const std::vector<unsigned long long>& pointers,cudaStream_t s,const unsigned long long* published=nullptr,bool synchronize=true){
        if(!published)plan.set(pointers,s);const auto live_plan=published?published:plan.p;CU(cudaMemsetAsync(h.p,0,h.count*4,s));CU(cudaMemsetAsync(part.p,0,part.count*4,s));CU(cudaMemsetAsync(out.p,0xff,out.count*4,s));
        if(T==1){x3::MoeArgs a{};a.plan_ptr=live_plan;a.plan_w=weights.p;a.lay=E.lay;a.k=K;a.n_embd=D;a.n_ff=F;a.limit=10.0f;gg::moe_gate_up(a,x.p,h.p,s);gg::moe_down(a,h.p,shared.p,out.p,s);}
        else {gf::MoeWin g{};g.plan_ptr=live_plan;g.T=T;g.k=K;g.grp_ptr=group.p;g.grp_start=starts.p;g.grp_ent=entries.p;g.n_groups=ng.p;gf::moe_win_group(g,s);
            x3::WinArgs a{};a.grp_ptr=group.p;a.grp_start=starts.p;a.grp_ent=entries.p;a.n_groups=ng.p;a.plan_ptr=live_plan;a.plan_w=weights.p;a.x=x.p;a.x_stride=D;a.h=h.p;a.part=part.p;a.sh_out=shared.p;a.out=out.p;a.out_stride=D;a.lay=E.lay;a.T=T;a.k=K;a.n_embd=D;a.n_ff=F;a.limit=10.0f;need(gg::win_experts(a,s),"actual public window experts refused");}
        CU(cudaGetLastError());if(synchronize)CU(cudaStreamSynchronize(s));
    }
    void audit(cudaStream_t s){x.audit(s);weights.audit(s);shared.audit(s);h.audit(s);part.audit(s);out.audit(s);plan.audit(s);group.audit(s);starts.audit(s);entries.audit(s);ng.audit(s);need(x.get(s)==input&&weights.get(s)==weight&&shared.get(s)==share,"math inputs changed");}
};
inline void exact_finite(const std::vector<float>& a,const std::vector<float>& b,const char* stage){need(a.size()==b.size(),"output extent");for(size_t i=0;i<a.size();++i){need(std::isfinite(a[i])&&std::isfinite(b[i]),std::string(stage)+" nonfinite");if(std::memcmp(&a[i],&b[i],4)!=0){uint32_t x,y;std::memcpy(&x,&a[i],4);std::memcpy(&y,&b[i],4);char q[180];std::snprintf(q,sizeof(q),"%s first mismatch i=%zu original=%08x staged=%08x",stage,i,x,y);throw std::runtime_error(q);}}}
