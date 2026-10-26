// Source-only rearm prerequisite. No CUDA, memory reads, allocation ownership
// changes, expert moves or synchronization. Free-slot contents are UNTRUSTED.
#pragma once
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace strata::ds4::ramdomain {
struct Arena { uintptr_t base = 0; size_t bytes = 0; };
struct Domain {
    int layer = -1, expert_count = 0;
    size_t first_slot = 0, slot_count = 0, part_stride = 0, arena_index = 0;
    uintptr_t base = 0;
    size_t arena_bytes = 0;
    std::array<size_t, 4> offsets{};
    std::array<size_t, 3> role_bytes{};
    bool has_spare = false;
};
struct LayerView {
    int layer = -1, expert_count = 0, gpu_capacity = 0, gpu_resident = 0;
    int gpu_spare = -1, ram_spare = -1;
    std::vector<int> gpu, ram;
    std::array<size_t, 4> offsets{};
    std::array<size_t, 3> role_bytes{};
};
struct Geometry {
    int nh = 0;
    bool copy_mode = false, has_store = false, ram_tier = false;
    size_t ram_held = 0;
};
struct LayerInventory {
    int layer = -1, spare = -1;
    size_t capacity = 0, occupied = 0;
    std::vector<int> free_slots;
};
struct Inventory {
    std::vector<LayerInventory> layers;
    size_t cpu_experts = 0, gpu_experts = 0, free_holes = 0, reserved_spares = 0;
    size_t free_capacity_bytes = 0;
    bool free_contents_trusted = false; // Always false. A demotion must overwrite first.
};

inline bool reconstruct(const Geometry& g, const std::vector<Domain>& domains,
                        const std::vector<Arena>& arenas,
                        const std::vector<std::array<uintptr_t, 2>>& slots,
                        const std::vector<LayerView>& live, Inventory& output, std::string& err) {
    output = {};
    auto fail = [&](const char* why) { err = std::string("copied-RAM domains: ") + why; return false; };
    if (g.nh != 1 || !g.copy_mode || g.has_store || g.ram_tier)
        return fail("requires NH1 copied RAM, no store and no RAM tier");
    if (live.empty() || slots.empty() || domains.empty() || domains.size() != arenas.size())
        return fail("missing published layer domains/arenas/slots");
    if (slots.size() > size_t(INT_MAX)) return fail("slot index does not fit int");
    const size_t maxs = std::numeric_limits<size_t>::max();
    const uintptr_t maxp = std::numeric_limits<uintptr_t>::max();
    std::vector<int> layer_index(live.size(), -1), domain_index(live.size(), -1);
    for (size_t i = 0; i < live.size(); ++i) {
        const auto& l = live[i];
        if (l.layer < 0 || size_t(l.layer) >= live.size() || layer_index[size_t(l.layer)] != -1)
            return fail("invalid or duplicate live layer");
        layer_index[size_t(l.layer)] = int(i);
        if (l.expert_count <= 0 || l.gpu.size() != size_t(l.expert_count) || l.ram.size() != size_t(l.expert_count) ||
            l.gpu_capacity < 0 || l.gpu_capacity > l.expert_count ||
            l.gpu_resident < 0 || l.gpu_resident > l.expert_count || l.gpu_spare < -1 ||
            (l.gpu_spare >= 0 && l.gpu_spare >= l.gpu_capacity) || l.ram_spare < -1)
            return fail("invalid live layer sizes/spares");
    }
    for (int i : layer_index) if (i < 0) return fail("live layer ids are not complete");

    // Ephemeral reconstruction only. No persistent per-expert owner ledger.
    std::vector<int> slot_domain(slots.size(), -1), arena_domain(arenas.size(), -1);
    for (size_t di = 0; di < domains.size(); ++di) {
        const auto& d = domains[di];
        if (d.layer < 0 || size_t(d.layer) >= live.size() || domain_index[size_t(d.layer)] != -1)
            return fail("invalid or duplicate domain layer");
        const auto& l = live[size_t(layer_index[size_t(d.layer)])];
        if (d.expert_count != l.expert_count || !d.slot_count || !d.part_stride ||
            d.first_slot > slots.size() || d.slot_count > slots.size() - d.first_slot ||
            d.slot_count > maxs / d.part_stride || d.arena_index >= arenas.size())
            return fail("invalid domain capacity/index");
        const size_t bytes = d.slot_count * d.part_stride;
        const auto& a = arenas[d.arena_index];
        if (!d.base || d.base != a.base || bytes != d.arena_bytes || bytes != a.bytes || bytes > maxp - d.base ||
            arena_domain[d.arena_index] != -1)
            return fail("domain arena identity/extent/ownership mismatch");
        if (d.offsets != l.offsets || d.role_bytes != l.role_bytes || d.offsets[0] != 0 || d.offsets[3] != d.part_stride)
            return fail("three-role layout changed");
        size_t expected = 0;
        for (size_t r = 0; r < 3; ++r) {
            if (!d.role_bytes[r] || d.offsets[r] != expected || d.role_bytes[r] > maxs - expected ||
                expected + d.role_bytes[r] > maxs - 255)
                return fail("three-role stride overflow/overlap");
            expected = (expected + d.role_bytes[r] + 255) & ~size_t(255);
        }
        if (expected != d.part_stride) return fail("three-role allocation stride mismatch");
        for (size_t other = 0; other < di; ++other) {
            const auto& b = domains[other];
            if (!(d.base + bytes <= b.base || b.base + b.arena_bytes <= d.base))
                return fail("owned RAM arena overlap");
        }
        for (size_t j = 0; j < d.slot_count; ++j) {
            const size_t s = d.first_slot + j;
            if (slot_domain[s] != -1 || slots[s][0] != d.base + j * d.part_stride || slots[s][1] != 0)
                return fail("slot domain overlap/NUMA pointer mismatch");
            slot_domain[s] = int(di);
        }
        arena_domain[d.arena_index] = int(di);
        domain_index[size_t(d.layer)] = int(di);
    }
    for (int i : arena_domain) if (i < 0) return fail("unrecorded owned RAM arena");
    for (int i : slot_domain) if (i < 0) return fail("unrecorded RAM slot");

    Inventory result;
    result.layers.resize(live.size());
    std::vector<int> claimed(slots.size(), -1); // -1 free, -2 reserved spare, otherwise expert id
    for (const auto& l : live) {
        auto& row = result.layers[size_t(l.layer)];
        row.layer = l.layer;
        const int di = domain_index[size_t(l.layer)];
        const Domain* d = di >= 0 ? &domains[size_t(di)] : nullptr;
        if (d) {
            row.capacity = d->slot_count;
            if (d->has_spare != (l.ram_spare >= 0)) return fail("missing or unexpected RAM spare");
        } else if (l.ram_spare != -1) return fail("RAM spare has no layer domain");
        std::vector<int> gpu_claims(size_t(l.gpu_capacity), -1);
        size_t resident = 0;
        for (int e = 0; e < l.expert_count; ++e) {
            const int v = l.gpu[size_t(e)], s = l.ram[size_t(e)];
            if (v < -1 || s < -1) return fail("negative map value other than absent");
            if (v >= 0) {
                if (v >= l.gpu_capacity || v == l.gpu_spare || gpu_claims[size_t(v)] != -1 || s != -1)
                    return fail("GPU slot duplicate/spare conflict or GPU-vs-CPU conflict");
                gpu_claims[size_t(v)] = e;
                ++resident; ++result.gpu_experts;
                continue;
            }
            if (s < 0) return fail("CPU expert missing RAM copy");
            if (!d || size_t(s) >= slots.size() || slot_domain[size_t(s)] != di)
                return fail("CPU slot outside its layer domain");
            if (claimed[size_t(s)] != -1) return fail("duplicate RAM ownership");
            claimed[size_t(s)] = e;
            ++row.occupied; ++result.cpu_experts;
        }
        if (resident != size_t(l.gpu_resident)) return fail("GPU resident count mismatch");
        for (int v = 0; v < l.gpu_capacity; ++v)
            if (v != l.gpu_spare && gpu_claims[size_t(v)] == -1) return fail("unaccounted GPU capacity");
        if (l.ram_spare >= 0) {
            const size_t s = size_t(l.ram_spare);
            if (!d || s >= slots.size() || slot_domain[s] != di || claimed[s] != -1)
                return fail("RAM spare outside domain or already occupied");
            claimed[s] = -2;
            row.spare = l.ram_spare;
            ++result.reserved_spares;
        }
    }
    if (result.cpu_experts != g.ram_held) return fail("RAM-held count mismatch");
    for (size_t s = 0; s < slots.size(); ++s) if (claimed[s] == -1) {
        const auto& d = domains[size_t(slot_domain[s])];
        result.layers[size_t(d.layer)].free_slots.push_back(int(s));
        if (result.free_capacity_bytes > maxs - d.part_stride) return fail("free-capacity overflow");
        result.free_capacity_bytes += d.part_stride;
        ++result.free_holes;
    }
    output = std::move(result);
    return true;
}
} // namespace strata::ds4::ramdomain
