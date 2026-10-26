#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
namespace strata::ds4::gpumiss {
inline bool direct_cap(size_t mib,bool all){return !all&&(mib==98304||mib==196608);}
// Pure actual admission arithmetic: physical slots, never remembered expert IDs.
inline bool whole_slot(uintptr_t arena,size_t arena_bytes,size_t first,size_t count,size_t stride,
                       size_t slot,uintptr_t current,size_t blob,const size_t* cpu_offsets,
                       const size_t* gpu_offsets,const size_t* role_bytes) {
    if(!arena||!arena_bytes||!count||!stride||!blob||stride!=blob||!cpu_offsets||!gpu_offsets||!role_bytes||
       arena_bytes>std::numeric_limits<uintptr_t>::max()-arena||count>arena_bytes/stride||
       slot<first||slot-first>=count)return false;
    const size_t delta=(slot-first)*stride;
    if(blob>arena_bytes-delta||current!=arena+delta)return false;
    size_t end=0;
    for(int r=0;r<3;++r) {
        if(!role_bytes[r]||cpu_offsets[r]!=gpu_offsets[r]||cpu_offsets[r]<end||
           cpu_offsets[r]>blob||role_bytes[r]>blob-cpu_offsets[r])return false;
        end=cpu_offsets[r]+role_bytes[r];
    }
    return true;
}
} // namespace strata::ds4::gpumiss
