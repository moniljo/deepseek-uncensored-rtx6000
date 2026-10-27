// Optional027+V6b port: single ownership of registrations and mutable prompt scratch.
#pragma once
#include <cstddef>
#include <cstring>
namespace strata::ds4::upgrade027 {
inline bool custom_owner(bool phase,bool epoch,bool misses,bool all,std::size_t mib) {
    return phase||epoch||misses||all||mib>0;
}
inline bool full_pick_count(int active,int model) { return active==model; }
inline bool exclusive_options(bool custom,const char* copy_pin,const char* warm) {
    return !custom||((!copy_pin||std::strcmp(copy_pin,"0")==0)&&(!warm||std::strcmp(warm,"0")==0));
}
}
