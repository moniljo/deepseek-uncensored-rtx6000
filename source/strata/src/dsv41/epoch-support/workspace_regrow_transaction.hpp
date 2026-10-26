#pragma once
// Pure policy only. CUDA/host allocation and owner bookkeeping are callbacks.
// Uses the same actual Buffer/Kind representation as the reviewed shrink47.
#include "../shrink-support/chunk_shrink_transaction.hpp"
#include <limits>
#include <cmath>
#include <cstdint>

namespace strata::ds4::chunkregrow {
using Buffer=shrink::Buffer;
using Kind=shrink::Kind;
using Callbacks=shrink::Callbacks;
enum class State { kept_original, committed, fatal };
struct Result {
    State state=State::kept_original;
    bool publication_started=false,published=false,stopped=false;
    size_t device_old=0,device_new=0,host_old=0,host_new=0;
    std::string reason;
};
inline bool add_size(size_t& sum,size_t n) {
    if(sum>std::numeric_limits<size_t>::max()-n)return false;
    sum+=n;return true;
}
inline bool budget_allows(size_t free_device,uint64_t free_ram,double commit_room,
                          size_t new_device,size_t new_host) {
    constexpr size_t floor=(size_t)3072<<20;
    constexpr uint64_t margin=(uint64_t)8<<30;
    const size_t max=std::numeric_limits<size_t>::max();
    if(new_device>max-floor||free_device<new_device+floor||
       free_ram==~uint64_t{0}||new_host>std::numeric_limits<uint64_t>::max()-margin||free_ram<new_host+margin||
       new_device>max-new_host||!std::isfinite(commit_room)||commit_room<0||
       commit_room<(double)(new_device+new_host))return false;
    return true;
}
inline Result replace(std::vector<Buffer>& buffers,const Callbacks& ops) {
    Result result;
    auto rollback=[&] {
        bool ok=true;
        for(auto i=buffers.rbegin();i!=buffers.rend();++i)if(i->next) {
            bool alias=false;
            for(const auto& other:buffers)alias=alias||i->next==other.old_pointer||
                (&other!=&*i&&other.next==i->next);
            if(alias){ok=false;continue;} // never free an old/ambiguous allocation
            // A failed discard retains its owner entry; do not claim it freed.
            bool freed=false;try{freed=ops.discard_new(*i);}catch(...){freed=false;}
            ok=freed&&ok;if(freed)i->next=nullptr;
        }
        return ok;
    };
    auto keep=[&](const char* why) {
        result.reason=why;
        if(!rollback()){result.state=State::fatal;result.reason+="; rollback failed";}
        return result;
    };
    try {
        if(buffers.size()!=47){result.state=State::fatal;result.reason="expected exactly47 buffers";return result;}
        size_t devices=0,hosts=0;
        for(size_t i=0;i<buffers.size();++i) {
            const Buffer& b=buffers[i];void* current=nullptr;
            if(b.slot)std::memcpy(&current,b.slot,sizeof current);
            bool duplicate=false;
            for(size_t j=0;j<i;++j)duplicate=duplicate||buffers[j].slot==b.slot||buffers[j].old_pointer==b.old_pointer;
            if(!b.slot||!b.old_pointer||!b.old_bytes||b.new_bytes<=b.old_bytes||current!=b.old_pointer||
               duplicate||b.next||!ops.owns_old(b)) {
                result.state=State::fatal;result.reason="invalid/aliased/unowned old regrow buffer";return result;
            }
            if(b.kind==Kind::device) {
                ++devices;
                if(!add_size(result.device_old,b.old_bytes)||!add_size(result.device_new,b.new_bytes)) {
                    result.state=State::fatal;result.reason="device sum overflow";return result;
                }
            } else {
                ++hosts;
                if(!add_size(result.host_old,b.old_bytes)||!add_size(result.host_new,b.new_bytes)) {
                    result.state=State::fatal;result.reason="host sum overflow";return result;
                }
            }
        }
        if(devices!=41||hosts!=6){result.state=State::fatal;result.reason="expected41device/6host";return result;}
        if(ops.cancelled()){result.stopped=true;return keep("Stop before allocation");}
        for(Buffer& b:buffers) {
            if(ops.cancelled()){result.stopped=true;return keep("Stop during preparation");}
            const bool prepared=ops.allocate(b);
            if(b.next)for(const auto& other:buffers)if(b.next==other.old_pointer||(&other!=&b&&other.next==b.next)) {
                result.state=State::fatal;result.reason="replacement aliases tracked allocation";return result;
            }
            if(!prepared||!b.next)return keep("allocation/initialization unavailable");
        }
        if(!ops.reserve_publication(buffers))return keep("publication reservation unavailable");
        if(ops.cancelled()){result.stopped=true;return keep("Stop before commit");}
        // No observers exist under HELD epoch. All pointers and shapes publish
        // in one nonthrowing caller transaction; preserve BOTH ledgers if this
        // contract is violated instead of freeing possibly-published pointers.
        result.publication_started=true;
        ops.publish(buffers);
        result.published=true;result.state=State::committed;
        for(const Buffer& b:buffers)if(!ops.retire_old(b)) {
            result.state=State::fatal;result.reason="old buffer retirement failed";return result;
        }
        result.reason="committed";return result;
    } catch(...) {
        result.state=State::fatal;result.reason="regrow callback exception";
        if(!result.publication_started&&!rollback())result.reason+="; rollback failed";
        return result;
    }
}
} // namespace strata::ds4::chunkregrow
