#pragma once
// Pure policy for a future held-epoch maintenance transaction. No CUDA, source
// reads, free-hole byte assumptions, or persistent ownership ledger.
#include "../ram-domain-support/copied_ram_domains.hpp"
#include "../phase-support/expert_grow_transaction.hpp"
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace strata::ds4::copiedshrink {
struct Plan {
    int layer=-1, old_slots=0, old_spare=-1, next_slots=0, next_spare=-1, resident=0;
    size_t old_bytes=0, next_bytes=0;
    std::vector<int> slot, kept, demoted, destinations;
};

inline bool plan(const ramdomain::Domain& domain,const std::vector<int>& old,
                 int old_slots,int old_spare,const std::vector<int>& ram,int ram_spare,
                 const std::vector<int>& free_holes,size_t unit,
                 Plan& output,std::string& err) {
    output={};
    auto fail=[&](const char* why){err=std::string("copied-pool shrink: ")+why;return false;};
    const size_t maxs=std::numeric_limits<size_t>::max();
    if(old.size()>size_t(INT_MAX)||old.empty()||ram.size()!=old.size()||!unit||
       domain.expert_count!=(int)old.size()||domain.layer<0||!domain.has_spare||
       !domain.slot_count||domain.slot_count>old.size()+1||domain.part_stride!=unit||
       domain.first_slot>size_t(INT_MAX)||domain.slot_count>size_t(INT_MAX)-domain.first_slot||
       old_slots<2||old_slots>(int)old.size()||old_spare<0||old_spare>=old_slots||
       size_t(old_slots)>maxs/unit)
        return fail("unsupported extent/spare/stride");
    const size_t end=domain.first_slot+domain.slot_count;
    auto belongs=[&](int s){return s>=0&&size_t(s)>=domain.first_slot&&size_t(s)<end;};
    if(!belongs(ram_spare))return fail("CPU spare is outside the immutable domain");
    std::vector<int> gpu_claims(size_t(old_slots),-1),cpu_claims(domain.slot_count,-1),resident;
    cpu_claims[size_t(ram_spare)-domain.first_slot]=-2;
    for(size_t e=0;e<old.size();++e) {
        const int v=old[e],s=ram[e];
        if(v < -1||s < -1)return fail("invalid absent map value");
        if(v>=0) {
            if(v>=old_slots||v==old_spare||gpu_claims[size_t(v)]!=-1||s!=-1)
                return fail("GPU duplicate/spare/CPU conflict");
            gpu_claims[size_t(v)]=(int)e;resident.push_back((int)e);
        } else {
            if(!belongs(s)||cpu_claims[size_t(s)-domain.first_slot]!=-1)
                return fail("missing/cross-domain/duplicate CPU copy");
            cpu_claims[size_t(s)-domain.first_slot]=(int)e;
        }
    }
    if(resident.size()!=size_t(old_slots-1))return fail("GPU capacity contains an unexplained hole");
    std::vector<int> actual_free;
    for(size_t s=0;s<cpu_claims.size();++s)if(cpu_claims[s]==-1)actual_free.push_back((int)(domain.first_slot+s));
    auto supplied=free_holes;std::sort(supplied.begin(),supplied.end());
    if(supplied!=actual_free)return fail("supplied free holes differ from CURRENT ownership");
    const int minimum=(int)old.size()-(int)(domain.slot_count-1);
    // The first prototype deliberately keeps both spares and at least one
    // resident. Empty pools/all-CPU startup require a separate policy.
    if(minimum<1||minimum>(int)resident.size())return fail("minimum resident capacity is unsupported");
    Plan next;
    next.layer=domain.layer;next.old_slots=old_slots;next.old_spare=old_spare;next.resident=minimum;
    next.next_slots=minimum+1;
    // Retain current physical prefix and spare whenever possible. If the old
    // spare falls outside, its replacement is the last prefix slot; the current
    // expert in that slot is demoted. Never consult stale initial identities.
    next.next_spare=old_spare<next.next_slots?old_spare:next.next_slots-1;
    next.old_bytes=size_t(old_slots)*unit;next.next_bytes=size_t(next.next_slots)*unit;
    next.slot.assign(old.size(),-1);
    for(int e:resident) {
        const int v=old[size_t(e)];
        if(v<next.next_slots&&v!=next.next_spare)next.kept.push_back(e);
        else next.demoted.push_back(e);
    }
    if(next.kept.size()!=size_t(minimum))return fail("physical prefix retained count mismatch");
    if(next.demoted.size()!=actual_free.size())return fail("demotions do not exactly fit free capacity");
    // Prefix retention permits one contiguous D2D, with no expert permutation.
    auto by_old_slot=[&](int a,int b){return old[size_t(a)]<old[size_t(b)];};
    std::sort(next.kept.begin(),next.kept.end(),by_old_slot);
    std::sort(next.demoted.begin(),next.demoted.end(),by_old_slot);
    for(int e:next.kept)next.slot[size_t(e)]=old[size_t(e)];
    next.destinations=std::move(actual_free);
    output=std::move(next);return true;
}
} // namespace strata::ds4::copiedshrink
