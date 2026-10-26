#pragma once
// Pure host planning for the isolated bounded cold-prefill F16 adapter.
// No CUDA allocation, output publication, or model state is changed here.
#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
namespace strata::ds4::coldf16 {
enum class Rule { power2, ceil128, exact };
inline const char* name(Rule rule) {
    return rule == Rule::power2 ? "power2" : rule == Rule::ceil128 ? "ceil128" : "exact";
}
struct Row {
    int original = 0; // original bounds/blob member index, never a resident expert ID
    int row0 = 0, rows = 0, pad0 = 0, padded = 0;
};
struct Batch {
    std::vector<Row> rows;
    int padded = 0, valid = 0, max_pad = 0;
};
struct Plan {
    std::vector<Batch> batches;
    int total = 0, hot_rows = 0, hot_experts = 0, light_rows = 0, light_experts = 0, max_rows = 0;
};
inline bool make(const int* bounds, int i0, int i1, int a, int capacity,
                 Rule rule, Plan& result, std::string& err) {
    constexpr int slots = 8;
    result = Plan{};
    if (!bounds || i0 < 0 || i1 <= i0 || i1 >= 2048 || capacity <= 0 || a != bounds[i0]) {
        err = "invalid original cold expert range"; return false;
    }
    const int64_t total = (int64_t)bounds[i1] - a;
    if (total <= 0 || total > capacity) { err = "cold expert range exceeds compact capacity"; return false; }
    result.total = (int)total;
    Batch batch;
    auto finish = [&] {
        if (batch.rows.empty()) return;
        if (rule != Rule::exact)
            std::stable_sort(batch.rows.begin(),batch.rows.end(),[](const Row& x,const Row& y) { return x.padded < y.padded; });
        int offset = 0;
        for (Row& row : batch.rows) { row.pad0 = offset; offset += row.padded; }
        result.batches.push_back(std::move(batch));
        batch = Batch{};
    };
    for (int i = i0; i < i1; ++i) {
        const int64_t n64 = (int64_t)bounds[i+1] - bounds[i];
        if (n64 < 0 || n64 > capacity || (int64_t)bounds[i]-a < 0 || (int64_t)bounds[i+1]-a > total) {
            err = "nonmonotonic cold expert bounds"; return false;
        }
        const int n = (int)n64;
        result.max_rows = std::max(result.max_rows,n);
        if (!n) continue;
        if (n <= 16) { result.light_rows += n; ++result.light_experts; continue; }
        int padded = n;
        if (rule == Rule::ceil128) padded = (n+127)/128*128;
        else if (rule == Rule::power2) { padded = 32; while (padded < n && padded <= capacity/2) padded *= 2; }
        if (padded < n || padded > capacity) { err = "one cold expert padding exceeds capacity"; return false; }
        if ((int)batch.rows.size() == slots || padded > capacity-batch.padded) finish();
        batch.rows.push_back({i,bounds[i]-a,n,0,padded});
        batch.padded += padded; batch.valid += n; batch.max_pad = std::max(batch.max_pad,padded);
        result.hot_rows += n; ++result.hot_experts;
    }
    finish();
    if (result.hot_rows+result.light_rows != result.total) { err = "cold expert plan lost compact rows"; return false; }
    return true;
}
} // namespace strata::ds4::coldf16
