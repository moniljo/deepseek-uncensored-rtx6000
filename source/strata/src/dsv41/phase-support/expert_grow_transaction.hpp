#pragma once
// Experimental first-feed expert-pool growth. Pure CPU policy/transaction code;
// CUDA ownership and table copies are supplied by the model. No global state.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

namespace strata::ds4::grow {
struct Plan {
    int old_slots = 0, next_slots = 0, old_spare = -1, next_spare = -1;
    int resident = 0;
    std::vector<int> slot, promoted;
};
inline bool monotonic_plan(const std::vector<int>& old, int old_slots, int old_spare,
                           int next_slots, bool cache, const uint32_t* counts,
                           Plan& out, std::string& err) {
    const int ne = (int)old.size();
    if (old_slots < 0 || next_slots < old_slots || next_slots > ne ||
        old_spare < -1 || old_spare >= old_slots || !counts) {
        err = "invalid monotonic expert-pool extent"; return false;
    }
    std::vector<int> seen((size_t)old_slots, 0);
    int nr = 0;
    for (int s : old) {
        if (s < -1 || s >= old_slots || (s == old_spare && s >= 0) ||
            (s >= 0 && seen[(size_t)s]++)) {
            err = "invalid/aliased resident expert slots"; return false;
        }
        nr += s >= 0;
    }
    if (nr != old_slots - (old_spare >= 0 ? 1 : 0)) {
        err = "resident pool contains an unexplained hole"; return false;
    }
    out = Plan{};
    out.old_slots = old_slots; out.next_slots = next_slots;
    out.old_spare = old_spare;
    // Existing resident slots AND the old spare index never move. A formerly
    // empty pool may introduce its one spare at the end of its new allocation.
    out.next_spare = old_spare >= 0 ? old_spare :
        old_slots == 0 && cache && next_slots >= 2 && next_slots < ne ? next_slots - 1 : -1;
    out.slot = old;
    std::vector<int> order((size_t)ne);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return counts[a] > counts[b]; });
    int ns = old_slots;
    for (int e : order) {
        if (out.slot[(size_t)e] >= 0) continue;
        while (ns == out.next_spare) ++ns;
        if (ns >= next_slots) break;
        out.slot[(size_t)e] = ns++;
        out.promoted.push_back(e);
    }
    out.resident = nr + (int)out.promoted.size();
    if (out.resident != next_slots - (out.next_spare >= 0 ? 1 : 0)) {
        err = "monotonic expert plan did not cover its allocation"; return false;
    }
    return true;
}
enum class Prepare { ready, unavailable, fatal };
enum class State { kept, committed, fatal };
struct Result { State state = State::kept; bool stopped = false; std::string reason; };
struct Callbacks {
    std::function<bool()> cancelled;
    std::function<Prepare()> prepare; // owned new slab + all bytes/views ready; old state unchanged
    std::function<bool()> discard;  // safe before publication only
    std::function<bool()> publish_table; // failure is terminal: table may be partially changed
    std::function<void()> publish_host; // no allocation, cannot fail/throw
    std::function<bool()> retire_old;
};
inline Result replace(const Callbacks& o) {
    Result r;
    auto keep = [&](const char* why, bool stopped) {
        r.reason = why; r.stopped = stopped;
        if (!o.discard()) { r.state = State::fatal; r.reason += "; discard failed"; }
        return r;
    };
    if (o.cancelled()) return keep("Stop before layer preparation", true);
    const Prepare p = o.prepare();
    if (p == Prepare::fatal) {
        r.state = State::fatal; r.reason = "layer preparation device/ownership error";
        (void)o.discard(); return r;
    }
    if (p != Prepare::ready) {
        const bool stopped = o.cancelled();
        return keep(stopped ? "Stop during layer preparation" : "layer allocation/headroom unavailable", stopped);
    }
    if (o.cancelled()) return keep("Stop before layer commit", true);
    // No readers exist here. Device table first; the old slab remains owned
    // until both table and nonthrowing host publication have completed.
    if (!o.publish_table()) {
        r.state = State::fatal; r.reason = "device table publication failed";
        return r; // retain BOTH tracked slabs for terminal cleanup
    }
    o.publish_host();
    r.state = State::committed;
    if (!o.retire_old()) { r.state = State::fatal; r.reason = "old slab retirement failed"; }
    else r.reason = "committed";
    return r;
}
} // namespace strata::ds4::grow
