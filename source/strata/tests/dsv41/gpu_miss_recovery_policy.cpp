// Pure actual-header policy/registration ownership tests. Root compiles/runs.
#include "gpu-miss-support/recovery_policy.hpp"
#include "phase-support/registered_copy_transaction.hpp"
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>
namespace gm=strata::ds4::gpumiss;
namespace rc=strata::ds4::regcopy;
int main() {
    int checks=0;
    auto check=[&](bool b){assert(b);++checks;};
    for(int bits=0;bits<16;++bits)for(int n:{-1,0,1,168,32767,32768,999764}) {
        const bool direct=bits&1,needed=bits&2,live=bits&4,long_feed=bits&8;
        check(gm::recovery_request(direct,needed,live,n,long_feed)==
              (direct&&needed&&!live&&n>0&&!long_feed));
    }
    check(gm::recovery_geometry(1024,1024,128,1024,65280,4096,128));
    check(gm::recovery_geometry(65280,4096,128,65280,65280,4096,128));
    for(int i=0;i<7;++i) {
        int v[7]={1024,1024,128,1024,65280,4096,128};++v[i];
        check(!gm::recovery_geometry(v[0],v[1],v[2],v[3],v[4],v[5],v[6]));
    }
    check(gm::recovery_lease(true,true,true,2,2,true,false,false,true));
    for(int i=0;i<7;++i) {
        bool v[7]={true,true,true,true,false,false,true};v[i]=!v[i];
        check(!gm::recovery_lease(v[0],v[1],v[2],2,2,v[3],v[4],v[5],v[6]));
    }
    check(!gm::recovery_lease(true,true,true,0,0,true,false,false,true));
    check(!gm::recovery_lease(true,true,true,1,2,true,false,false,true));
    const size_t MiB=size_t(1024)*1024,GiB=size_t(1024)*MiB;
    check(gm::recovery_registration_reserve==512*MiB);
    check(gm::recovery_pool_cap(80*GiB)==80*GiB-512*MiB);
    check(gm::recovery_pool_budget(100.0*GiB,60*GiB,80*GiB)==80*GiB-512*MiB);
    check(gm::recovery_pool_budget(70.0*GiB,60*GiB,80*GiB)==70*GiB);
    check(gm::recovery_pool_budget(100.0*GiB,90*GiB,80*GiB)==90*GiB);
    check(gm::recovery_pool_budget(100.0*GiB,40*GiB,0)==100*GiB-512*MiB);
    check(gm::recovery_pool_budget(512*MiB,0,0)==0);
    check(gm::recovery_pool_budget(512*MiB-1,0,0)==0);
    check(gm::recovery_pool_budget(512*MiB+1,0,0)==1);
    for(size_t healthy:{size_t(0),512*MiB-1,512*MiB,512*MiB+1,80*GiB}) {
        for(size_t current:{size_t(0),size_t(1),60*GiB,90*GiB}) {
            const size_t expected=healthy>512*MiB?healthy-512*MiB:0;
            check(gm::recovery_pool_cap(healthy)==expected);
            check(gm::recovery_pool_needs_shrink(current,healthy)==(healthy==0||current>expected));
        }
    }
    // Actual failed-run values are scalar evidence, not an allocation claim.
    const size_t healthy=83167395840ULL,initial=63929364480ULL;
    check(gm::recovery_pool_cap(healthy)==82630524928ULL);
    check(healthy-gm::recovery_pool_cap(healthy)==512*MiB);
    check(!gm::recovery_pool_needs_shrink(initial,healthy));
    // Already-large, partial-above-cap and post-shrink states use actual bytes.
    check(gm::recovery_pool_needs_shrink(83155322880ULL,healthy));
    check(gm::recovery_pool_needs_shrink(gm::recovery_pool_cap(healthy)+1,healthy));
    check(!gm::recovery_pool_needs_shrink(gm::recovery_pool_cap(healthy),healthy));
    check(!gm::recovery_pool_needs_shrink(initial,healthy));
    // Repeated cancelled first phases must keep taking the checked shrink
    // path until a NORMAL healthy target is captured; no synthetic capture.
    for(int cancel=0;cancel<4;++cancel) {
        check(gm::recovery_pool_needs_shrink(83155322880ULL,0));
        check(gm::recovery_pool_needs_shrink(initial,0));
        check(gm::recovery_pool_budget(100.0*GiB,initial,0)==100*GiB-512*MiB);
        check(gm::recovery_pool_cap(healthy)==82630524928ULL);
    }
    const std::vector<rc::Range> ranges={{0x100000,4096},{0x200000,8192},{0x300000,4096}};
    // Actual Ledger/acquire/retire; integer addresses are never dereferenced.
    for(int stop_at=0;stop_at<=6;++stop_at) {
        rc::Ledger ledger;int visits=0,registered=0,retired=0;std::string error;
        rc::Callbacks c;
        c.headroom=[&]{return visits++==stop_at?rc::Step::unavailable:rc::Step::ready;};
        c.register_one=[&](const rc::Range&){++registered;return rc::Step::ready;};
        c.unregister_one=[&](const rc::Range&){++retired;return true;};
        const auto result=rc::acquire(ledger,ranges,c,error);
        if(stop_at<6) {check(result==rc::Result::fallback);check(ledger.held.empty()&&!ledger.published);check(registered==retired);}
        else {check(result==rc::Result::enabled);check(ledger.published&&ledger.held.size()==3);check(ledger.covers(0x200000,8192));check(rc::retire(ledger,c.unregister_one));check(ledger.held.empty()&&!ledger.published);}
    }
    {rc::Ledger ledger;std::string error;int visits=0;rc::Callbacks c;
     c.headroom=[&]{return visits++==3?rc::Step::unavailable:rc::Step::ready;};
     c.register_one=[](const rc::Range&){return rc::Step::ready;};
     c.unregister_one=[](const rc::Range&){return false;};
     check(rc::acquire(ledger,ranges,c,error)==rc::Result::fatal);check(!ledger.published&&!ledger.held.empty());
     check(rc::retire(ledger,[](const rc::Range&){return true;}));check(ledger.held.empty());}
    std::printf("recovery actual-header policy/ledger checks: %d PASS; no CUDA/Model/physical allocation\n",checks);
    return 0;
}
