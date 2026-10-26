// SPDX-License-Identifier: MIT
// Host-only admission policy. No allocation, GPU call, or model state mutation.
#pragma once

namespace strata::ds4::repeatpolicy {
inline constexpr int default_min_tokens = 130560;
inline constexpr int lowest_min_tokens = 8192;
inline constexpr int highest_min_tokens = 1048576;

// A missing setting preserves the old threshold even when rearm is disabled.
// A present setting is an explicit capability and requires repeat rearm.
// Failure leaves the caller's value unchanged; only ASCII decimal is accepted.
inline bool parse_min_tokens(const char* value, bool rearm_enabled, int& result) {
    if (value == nullptr) {
        result = default_min_tokens;
        return true;
    }
    if (!rearm_enabled || *value == '\0') return false;
    unsigned parsed = 0;
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(value); *p != 0; ++p) {
        if (*p < '0' || *p > '9') return false;
        const unsigned digit = *p - '0';
        if (parsed > (static_cast<unsigned>(highest_min_tokens) - digit) / 10u) return false;
        parsed = parsed * 10u + digit;
    }
    if (parsed < static_cast<unsigned>(lowest_min_tokens)) return false;
    result = static_cast<int>(parsed);
    return true;
}

inline bool eligible(bool rearm_enabled, bool phase_pending, int feed_tokens, int minimum_tokens) {
    return rearm_enabled && !phase_pending && feed_tokens >= minimum_tokens;
}
} // namespace strata::ds4::repeatpolicy
