#pragma once
// Explicit first-cold policies only. Pure arithmetic/parsing: no CUDA, allocation, or environment reads.
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::ds4::phasepolicy {
constexpr size_t MiB = size_t(1) << 20;
constexpr size_t relative_allowance = 512 * MiB;
constexpr size_t minimum_register_floor = 2048 * MiB;

// Zero means the original policy. Explicit strings are nonempty decimal digits, without signs/whitespace.
inline bool parse_mib(const char* text, size_t minimum, size_t maximum, size_t& out) {
    if(!text||!*text||maximum<minimum)return false;
    size_t value=0;
    for(const char* p=text;*p;++p) {
        if(*p<'0'||*p>'9')return false;
        const size_t digit=(size_t)(*p-'0');
        if(value>maximum/10||(value==maximum/10&&digit>maximum%10))return false;
        value=value*10+digit;
    }
    if(value!=0&&value<minimum)return false;
    out=value;return true;
}

inline size_t final_reserve_bytes(int64_t initial_reserve_mib,size_t explicit_final_mib) {
    // Original max(4096, --reserve-mib) is preserved exactly when the independent flag is zero.
    return explicit_final_mib ? explicit_final_mib*MiB : (size_t)std::max<int64_t>(4096,initial_reserve_mib)*MiB;
}

inline size_t registration_admission_bytes(size_t explicit_floor_mib) {
    return explicit_floor_mib ? explicit_floor_mib*MiB : minimum_register_floor+relative_allowance;
}

// Precondition: baseline >= registration_admission_bytes(). The original relative policy is unchanged at zero.
inline size_t registration_floor_bytes(size_t baseline_bytes,size_t configured_bytes,size_t explicit_floor_mib) {
    return explicit_floor_mib ? explicit_floor_mib*MiB
                             : std::max(minimum_register_floor,std::min(configured_bytes,baseline_bytes-relative_allowance));
}
} // namespace strata::ds4::phasepolicy
