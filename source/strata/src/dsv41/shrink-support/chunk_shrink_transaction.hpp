#pragma once
// Standalone CPU-testable transaction; no CUDA calls or process-global state.
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace strata::ds4::shrink {
enum class Kind { device, pinned, mapped };
struct Buffer {
    const char* name;
    void* slot;                  // address of the typed pointer; publish uses memcpy, never void** aliasing
    void* old_pointer;
    size_t old_bytes, new_bytes;
    Kind kind;
    void* next = nullptr;
};
struct Callbacks {
    std::function<bool(const Buffer&)> owns_old;
    std::function<bool(Buffer&)> allocate;
    std::function<bool(const Buffer&)> discard_new;
    std::function<bool(const std::vector<Buffer>&)> reserve_publication;
    std::function<void(const std::vector<Buffer>&)> publish; // nonthrowing after reserve; includes shape update
    std::function<bool(const Buffer&)> retire_old;
    std::function<bool()> cancelled;
};
enum class State { kept_original, committed, fatal };
struct Result {
    State state = State::kept_original;
    bool published = false;
    size_t device_old = 0, device_new = 0, host_old = 0, host_new = 0;
    std::string reason;
};
inline Result replace(std::vector<Buffer>& buffers, const Callbacks& ops) {
    Result result;
    auto rollback = [&] {
        bool ok = true;
        for (auto i = buffers.rbegin(); i != buffers.rend(); ++i)
            if (i->next != nullptr) { ok = ops.discard_new(*i) && ok; i->next = nullptr; }
        return ok;
    };
    auto keep = [&](const char* why) {
        result.reason = why;
        if (!rollback()) { result.state = State::fatal; result.reason += "; replacement rollback failed"; }
        return result;
    };
    // Validate ALL ownership before allocating. An alias/interior pointer may not
    // be freed as if it were the original tracked allocation.
    for (size_t i = 0; i < buffers.size(); ++i) {
        const Buffer& b = buffers[i]; void* current = nullptr;
        if (b.slot != nullptr) std::memcpy(&current, b.slot, sizeof(current));
        bool duplicate = false;
        for (size_t j = 0; j < i; ++j) duplicate = duplicate || buffers[j].old_pointer == b.old_pointer || buffers[j].slot == b.slot;
        if (!b.slot || !b.old_pointer || !b.new_bytes || b.new_bytes > b.old_bytes || current != b.old_pointer || duplicate || !ops.owns_old(b)) {
            result.state = State::fatal; result.reason = "invalid/aliased/untracked old buffer ledger"; return result;
        }
        if (b.kind == Kind::device) { result.device_old += b.old_bytes; result.device_new += b.new_bytes; }
        else { result.host_old += b.old_bytes; result.host_new += b.new_bytes; }
    }
    if (ops.cancelled()) return keep("Stop before replacement allocation");
    for (Buffer& b : buffers) {
        if (ops.cancelled()) return keep("Stop while preparing replacements");
        if (!ops.allocate(b) || b.next == nullptr) return keep("replacement allocation/initialization unavailable");
    }
    if (!ops.reserve_publication(buffers)) return keep("cannot reserve publication ledger");
    if (ops.cancelled()) return keep("Stop before commit");
    // Linearization point: all new buffers are ready, all old pointers are still
    // owned, streams/readers are quiescent. A later Stop retains the valid small
    // layout. Publication and shape update cannot fail after reserve_publication.
    ops.publish(buffers);
    result.published = true;
    result.state = State::committed;
    for (const Buffer& b : buffers) {
        if (!ops.retire_old(b)) {
            result.state = State::fatal; result.reason = "retiring old buffer after commit failed"; return result;
        }
    }
    result.reason = "committed";
    return result;
}
} // namespace strata::ds4::shrink
