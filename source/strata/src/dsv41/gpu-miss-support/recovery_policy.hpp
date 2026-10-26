#pragma once
#include <cstddef>
#include <cstdint>
namespace strata::ds4::gpumiss {
// Request-boundary eligibility only. An eligible long feed still uses the
// existing repeated-prefill transaction, never this decode-only maintenance.
inline bool recovery_request(bool direct,bool needed,bool live_phase,int remaining,
                             bool long_rearm) {
    return direct&&needed&&!live_phase&&remaining>0&&!long_rearm;
}
inline bool recovery_geometry(int C,int A,int AI,int chunk,int restore_C,
                              int restore_A,int restore_AI) {
    return AI==128&&restore_C==65280&&restore_A==4096&&restore_AI==128&&C==chunk&&
           ((C==1024&&A==1024)||(C==65280&&A==4096));
}
inline bool recovery_lease(bool direct,bool enabled,bool preparing,unsigned lease,
                           unsigned serial,bool held,bool live_phase,bool legacy_phase,
                           bool healthy) {
    return direct&&enabled&&preparing&&lease!=0&&lease==serial&&held&&
           !live_phase&&!legacy_phase&&healthy;
}
// Recovery registration consumes driver/device headroom too. Leave an
// additional 512 MiB below the last successful compact pool before growing.
// The recovery coordinator must shrink an already-larger pool before growth.
// Every physical registration still has its original live 3 GiB floor check.
inline constexpr size_t recovery_registration_reserve = size_t(512)*1024*1024;
inline size_t recovery_pool_cap(size_t healthy) {
    return healthy>recovery_registration_reserve?
        healthy-recovery_registration_reserve:0;
}
inline bool recovery_pool_needs_shrink(size_t current,size_t healthy) {
    return healthy==0||current>recovery_pool_cap(healthy);
}
inline double recovery_pool_budget(double available,size_t current,size_t healthy) {
    if(healthy==0)return available>(double)recovery_registration_reserve?
        available-(double)recovery_registration_reserve:0;
    const size_t target=recovery_pool_cap(healthy);
    const size_t limit=current>target?current:target;
    return available<(double)limit?available:(double)limit;
}
}
