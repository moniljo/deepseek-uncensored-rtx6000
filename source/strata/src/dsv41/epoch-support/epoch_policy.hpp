// Pure CPU protocol policy; no CUDA, threads, model data or sequence mutation.
#pragma once
#include <cstddef>
#include <limits>
namespace strata::ds4::epochpolicy {
enum class Mailbox { pending, complete, invalid };
inline Mailbox mailbox(unsigned issued,unsigned completed,const unsigned* req,const unsigned* resp,size_t n) {
    if(!req||!resp||!n||completed>issued)return Mailbox::invalid;
    unsigned published=0;bool equal=true;
    for(size_t i=0;i<n;++i) {
        if(req[i]>issued||resp[i]>req[i])return Mailbox::invalid; // includes unsupported sequence wrap
        if(req[i]>published)published=req[i];
        equal=equal&&req[i]==resp[i];
    }
    return published==issued&&completed==issued&&equal?Mailbox::complete:Mailbox::pending;
}
enum class Phase { idle, publishers, cache, acknowledging, held, failed };
struct Machine {
    Phase phase=Phase::idle;unsigned serial=0;double deadline=0;
    bool start(double now,int timeout_ms) {
        if(phase!=Phase::idle||timeout_ms<1||timeout_ms>60000||serial==std::numeric_limits<unsigned>::max())return false;
        ++serial;deadline=now+timeout_ms;phase=Phase::publishers;return true;
    }
    bool publishers_done(){if(phase!=Phase::publishers)return false;phase=Phase::cache;return true;}
    bool cache_done(){if(phase!=Phase::cache)return false;phase=Phase::acknowledging;return true;}
    bool acknowledged(){if(phase!=Phase::acknowledging)return false;phase=Phase::held;return true;}
    bool expired(double now)const{return now>=deadline;}
    void release(bool terminal=false){phase=terminal?Phase::failed:Phase::idle;}
};
} // namespace strata::ds4::epochpolicy
