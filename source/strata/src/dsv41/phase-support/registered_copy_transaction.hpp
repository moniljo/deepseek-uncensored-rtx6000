#pragma once
// Pure CPU range selection/ownership transaction. CUDA calls are injected by the model.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>
namespace strata::ds4::regcopy {
struct Range { uintptr_t base=0; size_t bytes=0; };
inline bool valid(const Range& range) {
    return range.base!=0 && range.bytes!=0 && range.bytes<=std::numeric_limits<uintptr_t>::max()-range.base;
}
inline bool contains(const Range& range,uintptr_t address,size_t bytes) {
    return valid(range)&&bytes!=0&&address>=range.base&&bytes<=range.bytes&&address-range.base<=range.bytes-bytes;
}
inline bool select(const std::vector<Range>& arenas,size_t cap,std::vector<Range>& result,std::string& error) {
    result.clear();
    std::vector<Range> sorted=arenas;
    std::sort(sorted.begin(),sorted.end(),[](const Range&a,const Range&b){return a.base<b.base;});
    for(size_t i=0;i<sorted.size();++i) {
        if(!valid(sorted[i])||(i&&sorted[i-1].bytes>sorted[i].base-sorted[i-1].base)) {
            error="invalid or overlapping copied-RAM arenas";return false;
        }
    }
    size_t used=0;
    // Deterministic whole-arena first fit. No adjacent-layer prefix or cross-allocation registration.
    for(const Range& range:arenas)if(range.bytes<=cap-used){result.push_back(range);used+=range.bytes;}
    return true;
}
struct Ledger {
    std::vector<Range> held;
    bool published=false;
    bool covers(uintptr_t address,size_t bytes)const {
        if(!published)return false;
        return std::any_of(held.begin(),held.end(),[&](const Range&r){return contains(r,address,bytes);});
    }
    size_t bytes()const {size_t total=0;for(const Range&r:held)total+=r.bytes;return total;}
};
enum class Step { ready, unavailable, fatal };
enum class Result { enabled, fallback, fatal };
struct Callbacks {
    std::function<Step()> headroom;
    std::function<Step(const Range&)> register_one;
    std::function<bool(const Range&)> unregister_one;
};
inline bool retire(Ledger& ledger,const std::function<bool(const Range&)>& unregister_one) {
    ledger.published=false; // no new direct source may be admitted after cleanup starts
    bool complete=true;
    for(size_t i=ledger.held.size();i>0;--i) {
        if(unregister_one(ledger.held[i-1]))ledger.held.erase(ledger.held.begin()+(ptrdiff_t)i-1);
        else complete=false; // keep every uncertain registration owned for terminal cleanup
    }
    return complete;
}
inline Result acquire(Ledger& ledger,const std::vector<Range>& selected,const Callbacks& calls,std::string& error) {
    if(ledger.published||!ledger.held.empty()){error="registration ledger was already used";return Result::fatal;}
    if(selected.empty())return Result::fallback;
    try{ledger.held.reserve(selected.size());}catch(...){error="registration owner capacity unavailable";return Result::fallback;}
    auto fail=[&](Step cause,const char* why) {
        error=why;
        if(!retire(ledger,calls.unregister_one)){error+="; registration rollback failed";return Result::fatal;}
        return cause==Step::fatal?Result::fatal:Result::fallback;
    };
    for(const Range& range:selected) {
        Step step=calls.headroom();
        if(step!=Step::ready)return fail(step,"headroom unavailable before registration");
        step=calls.register_one(range);
        if(step!=Step::ready)return fail(step,"host registration unavailable");
        ledger.held.push_back(range); // capacity reserved BEFORE any CUDA call; publication cannot allocate
        step=calls.headroom();
        if(step!=Step::ready)return fail(step,"headroom unavailable after registration");
    }
    ledger.published=true;
    return Result::enabled;
}
} // namespace strata::ds4::regcopy
