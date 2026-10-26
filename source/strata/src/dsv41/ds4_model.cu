// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_model.cu - DeepSeek-V4.1-Flash on Maya (see ds4_model.hpp).
//
// The layer, for one token at position p (DeepSeek's inference/model.py, Block.forward with the lagged pre-mix):
//   [Engram, layers 1 and 14]  h += gate(h, key) * value from 24 n-gram rows of the FP8 tables
//   a_pre, a_post, a_comb = hc_mixes(h);  x = rmsnorm(sum_s pre[s] h[s])   (pre: the previous layer's f_pre)
//   attention(x): q from the low-rank wq_a / wq_b, one shared K = V row per token in a 128-slot window ring, plus
//                 (compress ratio > 0) up to 512 compressed rows the indexer picks; a learned sink per head; the
//                 output's rotary part rotated back, rounded to FP8, through the grouped wo_a and wo_b
//   h = post * attn + comb^T h;  f_* = hc_mixes(h);  x = rmsnorm(sum_s a_pre[s] h[s])
//   moe(x): 6 of 384 routed experts (sqrt-softplus scores, biased top-k) + the shared expert
//   h = post * moe + comb^T h;  pre = f_pre
// and after the last layer logits = head(rmsnorm(sum_s pre[s] h[s])).
#include "ds4_model.hpp"
#include "ds4_engram_io.hpp"

#include "ds4_gguf.cuh"
#include "ds4_iq.cuh"
#include "ds4_gguf_cpu.hpp"
#include "ds4_kernels.cuh"
#include "ds4_os.hpp"
#include "ds4_pool.hpp"
#include "ds4_store.hpp"
#include "ds4_vision.cuh"
#include "shrink-support/chunk_shrink_transaction.hpp"
#include "phase-support/expert_grow_transaction.hpp"
#include "phase-support/cold_f16_plan.hpp"
#include "phase-support/registered_copy_transaction.hpp"
#include "phase-support/prefill_reserve_policy.hpp"
#include "ram-domain-support/copied_ram_domains.hpp"
#include "epoch-support/epoch_policy.hpp"
#include "epoch-support/probe_protocol.hpp"
#include "epoch-support/pool_probe_protocol.hpp"
#include "epoch-support/workspace_probe_protocol.hpp"
#include "epoch-support/workspace_regrow_layout.hpp"
#include "epoch-support/expert_pool_shrink.hpp"
#include "gpu-miss-support/recovery_policy.hpp"
#include "gpu-miss-support/direct_policy.hpp"
#include "gpu-miss-support/policy.hpp"
#include "interactive-support/repeat_min_policy.hpp"
#include "strata/dsv41/exl3_cpu.hpp"
#include "strata/dsv41/exl3.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/dsv41/moe_win.hpp"
#include "strata/dsv41/sampler.hpp"

#include <cublas_v2.h>
#include "ds4_index_native.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_map>

#if defined(_MSC_VER)
#define strtok_r strtok_s
#endif

namespace strata::ds4 {
namespace x3 = strata::dsv41k::exl3;
namespace kc = strata::dsv41k::cpu;
namespace gf = strata::dsv41k::glmf;
namespace gg = strata::ds4::gguf;

#define CK(call)                                                                            \
    do {                                                                                    \
        cudaError_t e_ = (call);                                                            \
        if (e_ != cudaSuccess) {                                                            \
            err = std::string(#call) + ": " + cudaGetErrorString(e_);                       \
            return false;                                                                   \
        }                                                                                   \
    } while (0)

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

size_t align256(size_t n) { return (n + 255) & ~(size_t) 255; }

// part p of np of `whole` (a multiple of 128) in 128-blocks, as even as they go (whole / 128 % np parts a block more:
// those from part rot on, cyclically - so a group's layers, each its own rot, even out what each member holds):
// tensor parallel's columns of an FFN or of the head - an even split is what whole / np gave before
void split128(int whole, int np, int p, int& off, int& n, int rot = 0) {
    const int b = whole / 128, q = b / np, r = b % np;
    auto blocks = [&](int i) { return q + (((i - rot) % np + np) % np < r ? 1 : 0); };
    int at = 0;
    for (int i = 0; i < p; ++i) at += blocks(i);
    off = 128 * at;
    n = 128 * blocks(p);
}

std::string read_file(const std::string& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
std::vector<int> cpu_list(const std::string& s) {   // "0-3,8,10-11"
    std::vector<int> v;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty() || part == "\n") continue;
        const size_t dash = part.find('-');
        const int a = std::atoi(part.c_str());
        const int b = dash == std::string::npos ? a : std::atoi(part.c_str() + dash + 1);
        for (int c = a; c <= b; ++c) v.push_back(c);
    }
    return v;
}
// each NUMA node's physical cores (the first logical CPU of each core), in order
std::vector<std::vector<int>> node_cores() {
#if defined(_WIN32)
    static const bool smt = [] {   // DS4_CPU_SMT=1: the second hardware thread of every core as well (memory-bound rows gain from it)
        const char* v = std::getenv("DS4_CPU_SMT");
        return v != nullptr && std::atoi(v) != 0;
    }();
    std::vector<std::vector<int>> nodes = os::node_cores(smt);
#else
    std::vector<std::vector<int>> nodes;
    for (int n = 0; n < 64; ++n) {
        const std::string cl = read_file("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist");
        if (cl.empty()) break;
        std::vector<int> cores;
        for (int c : cpu_list(cl)) {
            const std::vector<int> sib = cpu_list(
                read_file("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"));
            if (sib.empty() || sib[0] == c) cores.push_back(c);
        }
        if (!cores.empty()) nodes.push_back(cores);
    }
#endif
    if (nodes.empty()) {
        std::vector<int> all;
        for (unsigned c = 0; c < std::max(1u, std::thread::hardware_concurrency() / 2); ++c) all.push_back((int) c);
        nodes.push_back(all);
    }
    return nodes;
}
// anonymous memory on one NUMA node (node < 0: wherever the first touch puts it), transparent huge pages
void* node_map(size_t bytes, int node, int n_nodes) { return os::anon_alloc(bytes, node, n_nodes); }
void pin_self(int cpu) { os::pin_self(cpu); }
// the RAM this process can still get: MemAvailable, and below it the tightest cgroup v2 limit on the way to the root
// (its limit less what is charged, the page cache counted as free: the kernel takes it back); ~0: not known
uint64_t ram_avail() {
#if defined(_WIN32)
    return os::ram_avail();
#endif
    uint64_t avail = ~uint64_t{0};
    {
        std::istringstream mi(read_file("/proc/meminfo"));
        std::string k;
        uint64_t kb = 0;
        while (mi >> k >> kb) {
            if (k == "MemAvailable:") avail = kb << 10;
            mi.ignore(64, '\n');
        }
    }
    std::string cg = read_file("/proc/self/cgroup");   // "0::/user.slice/..."
    const size_t at = cg.find("0::");
    if (at == std::string::npos) return avail;
    cg = cg.substr(at + 3);
    cg = cg.substr(0, cg.find('\n'));
    for (;;) {
        const std::string dir = "/sys/fs/cgroup" + (cg == "/" ? std::string() : cg);
        const std::string mx = read_file(dir + "/memory.max");
        if (!mx.empty() && mx.compare(0, 3, "max") != 0) {
            const uint64_t lim = std::strtoull(mx.c_str(), nullptr, 10);
            const uint64_t cur = std::strtoull(read_file(dir + "/memory.current").c_str(), nullptr, 10);
            uint64_t file = 0;
            std::istringstream st(read_file(dir + "/memory.stat"));
            std::string k;
            uint64_t v = 0;
            while (st >> k >> v)
                if (k == "file") file = v;
            const uint64_t used = cur > file ? cur - file : 0;
            avail = std::min(avail, lim > used ? lim - used : 0);
        }
        if (cg.empty() || cg == "/") break;
        const size_t sl = cg.rfind('/');
        cg = sl == 0 || sl == std::string::npos ? "/" : cg.substr(0, sl);
    }
    return avail;
}

float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float f16_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else {   // subnormal
            int ee = -1;
            uint32_t mm = m;
            do {
                ++ee;
                mm <<= 1;
            } while ((mm & 0x400) == 0);
            u = s | (uint32_t) (127 - 15 - ee) << 23 | (mm & 0x3ff) << 13;
        }
    } else if (e == 31) u = s | 0x7f800000u | m << 13;
    else u = s | (e - 15 + 127) << 23 | m << 13;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ------------------------------------------------------------------ the pack
struct Tensor {
    std::string dtype;
    int file = -1;
    uint64_t off = 0, bytes = 0;
    std::vector<int64_t> shape;
};

struct Pack {
    std::string dir, model_dir;
    std::unordered_map<std::string, std::string> kv;
    std::vector<std::string> files;
    std::vector<os::fd_t> fds;
    std::unordered_map<std::string, Tensor> t;
    int eg_vocab = 0, eg_layers = 0, eg_ngram = 0, eg_heads = 0, eg_hd = 0, eg_pad = 0;
    std::vector<int32_t> eg_map;
    std::vector<int64_t> eg_mult, eg_primes, eg_off;
    // format gguf (tools/ds4/pack_gguf.py): the files are a GGUF's shards, every one mapped at load (map()) - the
    // routed experts are always read in place (the OS page cache their RAM tier, the disk the rest)
    bool gguf = false;

    std::vector<std::pair<uint8_t*, size_t>> maps;   // file -> its read-only mapping (map(): the experts in place)
    std::vector<os::fd_t> dfds;   // (gguf) file -> unbuffered (os::open_direct; kBadFd: buffered reads only)
    std::vector<os::fd_t> pfds;   // (gguf) file -> for reads from many threads at once (os::open_par)

    ~Pack() {
        for (auto& m : maps) os::unmap(m.first, m.second);
        for (os::fd_t fd : fds) os::close_fd(fd);
        for (os::fd_t fd : dfds) os::close_fd(fd);
        for (os::fd_t fd : pfds) os::close_fd(fd);
    }
    // file i mapped whole, read-only (once; not thread-safe: called while loading)
    const uint8_t* map(int file) {
        if (maps.size() < fds.size()) maps.resize(fds.size(), {nullptr, 0});
        auto& m = maps[(size_t) file];
        if (m.first == nullptr) {
            const uint64_t size = os::file_size(fds[(size_t) file]);
            void* p = os::map_ro(fds[(size_t) file], size);
            if (p == nullptr) return nullptr;
            m = {(uint8_t*) p, (size_t) size};
        }
        return m.first;
    }

    bool load(const std::string& d, std::string& err) {
        dir = d;
        {
            std::ifstream f(d + "/ds4.txt");
            if (!f) {
                err = d + "/ds4.txt: not found (run tools/ds4/pack.py --model <model>)";
                return false;
            }
            std::string line;
            while (std::getline(f, line)) {
                const size_t sp = line.find(' ');
                if (sp == std::string::npos) continue;
                kv[line.substr(0, sp)] = line.substr(sp + 1);
            }
        }
        model_dir = str("model_dir");
        std::FILE* f = std::fopen((d + "/tensors.txt").c_str(), "r");
        if (f == nullptr) {
            err = d + "/tensors.txt: not found";
            return false;
        }
        std::vector<char> buf(1 << 16);
        while (std::fgets(buf.data(), (int) buf.size(), f) != nullptr) {
            char* s = buf.data();
            if (s[0] == 'f') {   // file <i> <name>
                char name[4096];
                int i = 0;
                if (std::sscanf(s, "file %d %4095s", &i, name) == 2) {
                    if ((int) files.size() <= i) files.resize((size_t) i + 1);
                    files[(size_t) i] = name;
                }
                continue;
            }
            if (s[0] != 't') continue;
            char* save = nullptr;
            ::strtok_r(s, " \n", &save);
            const char* name = ::strtok_r(nullptr, " \n", &save);
            const char* dt = ::strtok_r(nullptr, " \n", &save);
            const char* fi = ::strtok_r(nullptr, " \n", &save);
            const char* off = ::strtok_r(nullptr, " \n", &save);
            const char* nb = ::strtok_r(nullptr, " \n", &save);
            if (name == nullptr || nb == nullptr) continue;
            Tensor x;
            x.dtype = dt;
            x.file = std::atoi(fi);
            x.off = std::strtoull(off, nullptr, 10);
            x.bytes = std::strtoull(nb, nullptr, 10);
            while (const char* dim = ::strtok_r(nullptr, " \n", &save)) x.shape.push_back(std::atoll(dim));
            t.emplace(name, std::move(x));
        }
        std::fclose(f);
        for (const std::string& fn : files) {
            const std::string p = model_dir + "/" + fn;
            const os::fd_t fd = os::open_ro(p);
            if (fd == os::kBadFd) {
                err = p + ": cannot open";
                return false;
            }
            fds.push_back(fd);
        }
        gguf = str("format") == "gguf";
        if (gguf)
            for (size_t i = 0; i < fds.size(); ++i) {
                if (map((int) i) == nullptr) {
                    err = model_dir + "/" + files[i] + ": cannot map";
                    return false;
                }
                dfds.push_back(os::open_direct(model_dir + "/" + files[i]));
                pfds.push_back(os::open_par(model_dir + "/" + files[i]));
            }
        // the Engram tables
        std::FILE* e = std::fopen((d + "/engram.bin").c_str(), "rb");
        if (e == nullptr) {
            err = d + "/engram.bin: not found";
            return false;
        }
        char magic[4];
        int32_t hdr[6];
        bool ok = std::fread(magic, 1, 4, e) == 4 && std::memcmp(magic, "DSE1", 4) == 0 &&
                  std::fread(hdr, 4, 6, e) == 6;
        if (ok) {
            eg_vocab = hdr[0], eg_layers = hdr[1], eg_ngram = hdr[2], eg_heads = hdr[3], eg_hd = hdr[4], eg_pad = hdr[5];
            const int cols = (eg_ngram - 1) * eg_heads;
            eg_map.resize((size_t) eg_vocab);
            eg_mult.resize((size_t) eg_layers * eg_ngram);
            eg_primes.resize((size_t) eg_layers * cols);
            eg_off.resize((size_t) eg_layers * cols);
            ok = std::fread(eg_map.data(), 4, eg_map.size(), e) == eg_map.size() &&
                 std::fread(eg_mult.data(), 8, eg_mult.size(), e) == eg_mult.size() &&
                 std::fread(eg_primes.data(), 8, eg_primes.size(), e) == eg_primes.size() &&
                 std::fread(eg_off.data(), 8, eg_off.size(), e) == eg_off.size();
        }
        std::fclose(e);
        if (!ok) {
            err = d + "/engram.bin: unreadable";
            return false;
        }
        return true;
    }

    std::string str(const std::string& k, const std::string& def = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second;
    }
    double num(const std::string& k, double def) const {
        auto it = kv.find(k);
        return it == kv.end() ? def : std::atof(it->second.c_str());
    }
    std::vector<int> list(const std::string& k) const {
        std::vector<int> v;
        std::stringstream ss(str(k));
        std::string x;
        while (std::getline(ss, x, ','))
            if (!x.empty()) v.push_back(std::atoi(x.c_str()));
        return v;
    }
    const Tensor* get(const std::string& n) const {
        auto it = t.find(n);
        return it == t.end() ? nullptr : &it->second;
    }
    bool read_at(int file, uint64_t off, uint64_t n, void* dst) const {   // (any thread; side by side where pfds is)
        if ((size_t) file < pfds.size() && pfds[(size_t) file] != os::kBadFd)
            return os::pread_par(pfds[(size_t) file], dst, n, off);
        return os::pread_full(fds[(size_t) file], dst, n, off);
    }
    bool read(const Tensor& x, void* dst) const { return read_at(x.file, x.off, x.bytes, dst); }
    const uint8_t* at(int file, uint64_t off) const { return maps[(size_t) file].first + off; }   // (gguf: mapped)
};

// ------------------------------------------------------------------ geometry
struct Geo {
    int vocab = 0, dim = 0, nl = 0, heads = 0, hd = 0, rd = 0, q_lora = 0, o_lora = 0, o_groups = 0, n_exp = 0,
        k_exp = 0, ff = 0, win = 0, hc = 0, sinkhorn = 0, ix_heads = 0, ix_hd = 0, ix_topk = 0, cand_src = -1,
        cand_topk = 0, cand_blk = 0, orig_ctx = 0, eos = 1;
    float eps = 1e-20f, hc_eps = 1e-6f, route_scale = 1.0f, lim = 0.0f;
    double rope_theta = 1e4, comp_theta = 1e4, factor = 1.0, beta_fast = 32, beta_slow = 1;
    std::vector<int> ratios, kv_src, ix_src, eg_layers;
    // the DSpark drafter (the checkpoint's mtp.* blocks): a block of [seed, noise x (block - 1)] through n_mtp blocks
    int n_mtp = 0, d_block = 0, d_noise = 0, d_rank = 0, d_n_exp = 0, d_k = 0;
    std::vector<int> d_targets;

    void from(const Pack& p) {
        vocab = (int) p.num("vocab_size", 0), dim = (int) p.num("hidden_size", 0);
        nl = (int) p.num("num_hidden_layers", 0), heads = (int) p.num("num_attention_heads", 0);
        hd = (int) p.num("head_dim", 0), rd = (int) p.num("qk_rope_head_dim", 0);
        q_lora = (int) p.num("q_lora_rank", 0), o_lora = (int) p.num("o_lora_rank", 0);
        o_groups = (int) p.num("o_groups", 0), n_exp = (int) p.num("n_routed_experts", 0);
        k_exp = (int) p.num("num_experts_per_tok", 0), ff = (int) p.num("moe_intermediate_size", 0);
        win = (int) p.num("sliding_window", 0), hc = (int) p.num("hc_mult", 0);
        sinkhorn = (int) p.num("hc_sinkhorn_iters", 0), ix_heads = (int) p.num("index_n_heads", 0);
        ix_hd = (int) p.num("index_head_dim", 0), ix_topk = (int) p.num("index_topk", 0);
        cand_src = (int) p.num("candidate_source_layer_id", -1), cand_topk = (int) p.num("candidate_topk_blocks", 0);
        cand_blk = (int) p.num("candidate_block_size", 0);
        orig_ctx = (int) p.num("rope_scaling.original_max_position_embeddings", 0);
        eos = (int) p.num("eos_token_id", 1);
        eps = (float) p.num("rms_norm_eps", 1e-20), hc_eps = (float) p.num("hc_eps", 1e-6);
        route_scale = (float) p.num("routed_scaling_factor", 1.0), lim = (float) p.num("swiglu_limit", 0.0);
        rope_theta = p.num("rope_theta", 1e4), comp_theta = p.num("compress_rope_theta", 1e4);
        factor = p.num("rope_scaling.factor", 1.0), beta_fast = p.num("rope_scaling.beta_fast", 32);
        beta_slow = p.num("rope_scaling.beta_slow", 1);
        ratios = p.list("compress_ratios"), kv_src = p.list("kv_source_layer_ids");
        ix_src = p.list("index_source_layer_ids"), eg_layers = p.list("engram_layer_ids");
        d_block = (int) p.num("dspark_block_size", 0), d_noise = (int) p.num("dspark_noise_token_id", 0);
        d_rank = (int) p.num("dspark_markov_rank", 0), d_n_exp = (int) p.num("dspark_n_routed_experts", 0);
        d_k = (int) p.num("dspark_num_experts_per_tok", 0), d_targets = p.list("dspark_target_layer_ids");
        // the DSpark blocks: an EXL3 pack's, or a GGUF pack's from DeepSeek's checkpoint (tools/ds4/drafter_pack.py)
        n_mtp = d_block > 0 && (int) ratios.size() > nl &&
                        (p.get("mtp.0.attn.wq_a.trellis") != nullptr || p.get("mtp.0.attn.wq_a.weight") != nullptr)
                    ? (int) ratios.size() - nl : 0;
    }
};

// DeepSeek's precompute_freqs_cis frequencies (YaRN when original_seq_len > 0), in F32 as torch computes them
std::vector<float> rope_freqs(int dim, int original, double base, double factor, double beta_fast, double beta_slow) {
    std::vector<float> f((size_t) dim / 2);
    for (int i = 0; i < dim / 2; ++i) {
        const float e = (float) (2 * i) / (float) dim;
        f[(size_t) i] = 1.0f / std::pow((float) base, e);
    }
    if (original > 0) {
        auto corr = [&](double rot) {
            return dim * std::log(original / (rot * 2 * 3.14159265358979323846)) / (2 * std::log(base));   // (M_PI: not MSVC's)
        };
        const double low = std::max(std::floor(corr(beta_fast)), 0.0);
        const double high = std::min(std::ceil(corr(beta_slow)), (double) dim - 1);
        const float den = (float) std::max(high - low, 1e-3);
        for (int i = 0; i < dim / 2; ++i) {
            float ramp = ((float) i - (float) low) / den;
            ramp = std::min(1.0f, std::max(0.0f, ramp));
            const float smooth = 1.0f - ramp;
            f[(size_t) i] = f[(size_t) i] / (float) factor * (1.0f - smooth) + f[(size_t) i] * smooth;
        }
    }
    return f;
}

// ------------------------------------------------------------------ devices, layers
struct Dev {
    int id = 0;
    cudaStream_t st = nullptr;
    cudaEvent_t ev = nullptr;
    size_t free0 = 0, total = 0, weights = 0, experts = 0, caches = 0, scratch = 0;
    std::vector<void*> allocs;
    // a window's scratch, rows of up to kWin tokens (the routed experts' rows: kWin * kPicks)
    float *h = nullptr, *h2 = nullptr, *x = nullptr, *qr = nullptr, *q = nullptr, *kvr = nullptr, *ckr = nullptr,
          *cgr = nullptr, *latent = nullptr, *ikr = nullptr, *iq = nullptr, *iw = nullptr, *scores = nullptr,
          *bm = nullptr, *o = nullptr, *oa = nullptr, *ao = nullptr, *mix = nullptr, *a_pre = nullptr,
          *a_post = nullptr, *a_comb = nullptr, *f_pre = nullptr, *f_post = nullptr, *f_comb = nullptr, *pre = nullptr,
          *glog = nullptr, *rw = nullptr, *moe_out = nullptr, *sg = nullptr, *su = nullptr, *sh = nullptr,
          *sh_out = nullptr, *cpu_part = nullptr, *eg_x = nullptr, *eg_kv = nullptr, *logits = nullptr,
          *freqs_win = nullptr, *freqs_comp = nullptr, *xe = nullptr, *xg = nullptr, *xu = nullptr, *xhh = nullptr,
          *ed = nullptr;
    float* glog_a = nullptr;   // lookahead: the next two layers' gate logits on this layer's FFN input (rows 1024 apart)
    int *ridx = nullptr, *topk = nullptr, *cand = nullptr, *tok = nullptr;
    unsigned long long* plan_ptr = nullptr;           // the decode path's VRAM experts (0: on the CPU lane)
    uint8_t* eg_rows = nullptr;
    uint16_t *emb_rows = nullptr, *ext = nullptr;   // the window's embedding rows; the window attention's rows
    TopkScratch topk_scr{}; // optional load-owned multi-CTA selection scratch
    AttnScratch attn_scr;   // a few queries' selected compressed rows decoded (sparse_attn)
    // the hand-off to the next device through pinned host memory (these cards copy peer to peer only through the
    // driver's staging, ~1.7 ms a copy): h | pre | topk | cand
    uint8_t* bounce = nullptr;
    cudaEvent_t ev_out = nullptr;
    float* tap = nullptr;   // a window's stream means before a DSpark target layer
    // threads / tensor parallelism: its group and its index in it, the group's all-reduces it took part in, xfer_put's
    // block counter, the hand-off to the next group (host-mapped: h | pre | topk | cand, then its flag), the window
    // MoE's grouping scratch
    int grp = 0, mi = 0;
    unsigned ar_seq = 0;
    unsigned* xcnt = nullptr;
    unsigned* hccnt = nullptr; // decode fusion: kWin counters, load-owned and zeroed
    // the head group's members (the last layer's group): each its columns (split128 of V) of the head and the norms;
    // logits_part: a member's share of the logits (the lead's: where the others' land before they join its rows)
    x3::Mat head;
    float *norm_w = nullptr, *d_norm = nullptr, *logits_part = nullptr;
    unsigned head_seq = 0;
    // the drafter's markov bias over the head group (markov_share): this member's rows [mk_v0, mk_v0 + mk_vn) of the
    // markov head, the previous token's E row, the drafted tokens (the lead's: d_ints)
    uint16_t *mk_share = nullptr, *mk_erow = nullptr;
    int* mk_ints = nullptr;
    int mk_v0 = 0, mk_vn = 0;
    uint8_t* hand = nullptr;
    unsigned* hand_flag = nullptr;
    unsigned long long* grp_ptr = nullptr;
    int *grp_start = nullptr, *grp_ent = nullptr, *n_groups = nullptr;
};

// the chunked prompt path's buffers on a device: a chunk of C tokens through a layer at once (the attention in slabs of
// A queries, the indexer in slabs of AI), its GEMMs through FP16 reconstructions of the weights and cuBLAS
// DS4_PRED_DUMP: a fixed +-1 random projection of the router input (D -> 256), one block a token
__global__ void pred_proj_k(const float* x, int D, float* out) {
    const int t = blockIdx.x, j = threadIdx.x;   // 256 threads
    const float* xr = x + (size_t) t * D;
    float a = 0.0f;
    for (int d = 0; d < D; ++d) {
        unsigned h = (unsigned) d * 2654435761u ^ ((unsigned) j + 1u) * 40503u;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
        a += (h & 1u) ? xr[d] : -xr[d];
    }
    out[(size_t) t * 256 + j] = a * rsqrtf((float) D);
}

struct Chunk {
    int C = 0, A = 0, AI = 0;
    int restore_C = 0, restore_A = 0, restore_AI = 0; // captured once after successful diagnostic allocation
    float *h = nullptr, *x = nullptr, *acc = nullptr, *qr = nullptr, *kvr = nullptr, *ckr = nullptr, *cgr = nullptr,
          *latent = nullptr, *ikr = nullptr, *q = nullptr, *o = nullptr, *oa = nullptr, *iq = nullptr, *iw = nullptr,
          *scores = nullptr, *bm = nullptr, *mix = nullptr, *a_pre = nullptr, *a_post = nullptr, *a_comb = nullptr,
          *f_pre = nullptr, *f_post = nullptr, *f_comb = nullptr, *pre = nullptr, *glog = nullptr, *rw = nullptr,
          *sg = nullptr, *su = nullptr, *y = nullptr, *eg_x = nullptr, *eg_kv = nullptr;
    uint16_t *x16 = nullptr, *qr16 = nullptr, *o16 = nullptr, *oa16 = nullptr, *lat16 = nullptr, *sh16 = nullptr,
             *h16 = nullptr, *eg_x16 = nullptr, *ext = nullptr, *w16 = nullptr, *emb = nullptr;
    size_t w16_elems = 0;
    // Default-off, FIRST chunk-major prefill only. One immutable Engram matrix at a time;
    // original RN-F16 weights, original 1280-column GEMM geometry and existing c.blas.
    uint16_t* eg_weight_cache = nullptr;
    size_t eg_weight_cache_bytes = 0;
    x3::Mat eg_weight_tag{};
    int eg_weight_layer = -1;
    int64_t eg_weight_p0 = -1;
    bool eg_weight_valid = false; // accepted work is ordered on d.st, never a cross-stream promise
    uint64_t eg_weight_fills = 0, eg_weight_hits = 0, eg_weight_gemms = 0;
    // Default-off first chunk-major prefill only: WqB + eight WoA + WoB, 224 MiB.
    // Slots are disjoint from w16/expert/Engram scratch; each retains its original RN-F16 weights.
    uint16_t* dense_projection_cache = nullptr;
    size_t dense_projection_cache_bytes = 0;
    x3::Mat dense_projection_tags[10]{};
    int dense_projection_layers[10] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    int64_t dense_projection_p0[10] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    bool dense_projection_valid[10]{}; // publication and consumption both ordered on d.st
    uint64_t dense_projection_fills = 0, dense_projection_hits = 0, dense_projection_gemms = 0;
    int *ridx = nullptr, *topk = nullptr, *cand = nullptr, *bounds = nullptr, *row_tok = nullptr, *pos = nullptr;
    uint8_t* eg_rows = nullptr;
    TopkScratch topk_scr{}; // optional load-owned multi-CTA selection scratch
    int* bounds_pin = nullptr;   // pinned: the sort's bounds, read back
    // the CPU lane's experts streamed in: ring_n VRAM slots of ring_stride bytes in kParts parts, a batch a part,
    // kParts batches in flight on the copy stream (filled / freed: each part's last copy, its last use)
    static constexpr int kParts = 4;
    uint8_t* ring = nullptr;
    size_t ring_stride = 0;
    int ring_n = 0;
    cudaStream_t cs = nullptr;
    cudaEvent_t filled[kParts] = {}, freed[kParts] = {};
    // in place (file_experts): each streamed expert's file region read into pinned staging (ring_n slots of
    // stage_stride bytes, a part's slots free again once its `filled` copies are done) by the readers, in parallel,
    // and DMA'd from there - a copy from the files' pages goes through the driver's staging, one thread at ~1.3 GB/s
    uint8_t* stage = nullptr;
    size_t stage_stride = 0;
    std::unique_ptr<Pool> readers;
    cublasHandle_t blas = nullptr;
    // host-mapped: a chunk's hand-off to the next group (h | pre | topk | cand), two in flight (slot = chunk % 2), each
    // with its flag, and the next group's members' acknowledgements [slot][member] (32 apart)
    uint8_t* hand[2] = {nullptr, nullptr};
    unsigned* hand_flag[2] = {nullptr, nullptr};
    unsigned* hand_ack = nullptr;
    unsigned arb_seq = 0;        // the group's chunk all-reduces
    // pinned, two chunks in flight: the embedding rows (the first layer's device), the Engram rows (an Engram layer's)
    uint16_t* emb_host[2] = {nullptr, nullptr};
    uint8_t* eg_host[2] = {nullptr, nullptr};
    cudaEvent_t emb_ev[2] = {nullptr, nullptr}, eg_ev[2] = {nullptr, nullptr};
    double stream_bytes = 0;     // the experts streamed in, in all
    double registered_stream_begin = 0;
    uint64_t registered_h2d_payload_bytes = 0, registered_direct_bytes = 0, registered_bounce_bytes = 0;
    // the flow (a GGUF's in-place experts, every layer of the device on it alone; DS4_FLOW=0: the batches above): a
    // (chunk, layer)'s streamed experts are those it routes to - its likely ones (the profile's chance of a pick by the
    // chunk's tokens) queued once the layer before it routed, read while that one computes, the rest of its picks once
    // it routed itself; never an expert no token of the chunk picked unless it was a likely one.  Each item read by io
    // (unbuffered: the disk straight into its staging slot, every part of every item in flight at once - none waits
    // for a batch) or taken from its RAM-tier slot as it lies, then DMA'd into the ring in order, item i into slot
    // i % ring_n; the compute takes the items as they land, in runs of what has.
    struct Item {
        int il = 0, k = 0, e = 0, v = 0, stage = -1;   // layer, chunk, expert, virtual slot, staging slot (-1: RAM)
        int mi = 0;                                    // the device's member index in the layer's group
        const uint8_t* src[3] = {nullptr, nullptr, nullptr};   // where its gate / up / down (its parts) lie once read
        const uint8_t* ram = nullptr;                  // its RAM-tier blob, copied into staging (not pinned there)
        uint8_t* keep = nullptr;                       // (layer-major) its layer-cache slot: what is read kept there
        int64_t after = -1;                            // (layer-major) the items up to this one read first
        std::atomic<int> left{0};                      // parts still being read
        std::atomic<bool> bad{false};
        bool lm_gpu_read_skipped = false;             // the phase cache supplied this item; no host job owns its stage
        bool registered_copy_source = false;          // source roles are covered by our first-phase registration ledger
    };
    bool flow = false;
    int dev = -1;                            // the device (index into devs) this chunk state belongs to
    std::unique_ptr<IoQueue> io;
    std::deque<Item> items;                  // items[0] is item `base`; the indices below are absolute
    size_t base = 0, n_read = 0, n_dma = 0, n_comp = 0;
    uint64_t n_file = 0;                     // file items so far (staging slot = ordinal % S)
    int S = 0;                               // staging slots, stage_stride apart
    std::vector<int64_t> stage_item;         // staging slot -> the item last read into it (-1: none)
    std::vector<cudaEvent_t> stage_ev, ring_fill, ring_free;   // a staging slot's DMA out; a ring slot's DMA in, its use
    int spec_k = -1, spec_il = -1, n_spec = 0;   // the likely items queued last: (chunk, layer), how many, from where
    size_t spec_begin = 0;
    std::vector<int> fl_layers, fl_T;        // the device's trunk layers in order; the run's chunk sizes
    int fl_yoco = -1, fl_cut_k = 0, fl_cut_r = 0;   // Run::yoco / cut_k / cut_r: what a layer's chunk streams for
    int bc_n = 0, bc_w = 0;                  // a batch's compacted bounds: bc_n regions of bc_w, by batch count
    int *bc_pin = nullptr, *bc_dev = nullptr;
    uint64_t n_batch = 0;
    double wait_ms = 0;                      // the compute waiting for a read
    uint64_t it_file = 0, it_ram = 0, it_spec = 0, it_spec_unused = 0;
    uint64_t io_d0 = 0, io_b0 = 0;           // io's unbuffered / buffered bytes when the run started
    // layer-major (a run of several chunks, every layer on this device): the layers one after another, each through
    // all the chunks; a chunk's state between its layers (h | pre | topk | cand, chunk k at k chunk_hand_bytes()) in
    // host memory, and the layer cache - RAM slots where the experts a layer's chunk reads from the files are kept for
    // its later chunks, so each is read once a run
    bool lm = false, lm_pinned = false;
    uint8_t* lm_state = nullptr;
    size_t lm_state_bytes = 0;
    uint8_t* lc = nullptr;                   // lc_bytes of slots, a layer's experts' blobs side by side
    size_t lc_bytes = 0;
    int lc_il = -1, lc_used = 0, lc_n = 0;   // the layer the cache holds, its slots taken, its slots in all
    size_t lc_first = 0, lc_chk = 0;         // the layer's first item; the items before lc_chk all read
    std::vector<int> lc_of;                  // expert -> its slot (-1: none)
    std::vector<int64_t> lc_writer;          // slot -> the item reading its expert into it
    uint64_t it_lc = 0;                      // items taken from the layer cache
    // Opt-in compressed GPU layer cache. Allocated before the persistent expert placement, never aliases it.
    // The copy stream fills this arena and copies hits into the ORIGINAL ring slots. The original set batches,
    // kernels and pick-order reductions therefore stay exactly the same. No eviction within a layer.
    uint8_t* lm_gpu = nullptr;
    size_t lm_gpu_bytes = 0, lm_gpu_stride = 0, lm_gpu_used = 0, lm_gpu_peak_used = 0;
    int lm_gpu_il = -1;
    std::vector<int> lm_gpu_of;              // expert -> arena slot for lm_gpu_il, -1 before its first H2D
    uint64_t lm_gpu_hits = 0, lm_gpu_fills = 0, lm_gpu_read_skips = 0, lm_gpu_no_room = 0;
    uint64_t lm_gpu_h2d_bytes = 0, lm_gpu_d2d_bytes = 0, lm_gpu_saved_h2d_bytes = 0;
    // First-prefill-only F32 residual arena. Whole chunks alias their own slice; no arithmetic or extra copies.
    float* prefill_gpu_h = nullptr;
    size_t prefill_gpu_h_bytes = 0, prefill_gpu_h_rows = 0;
    uint64_t prefill_gpu_h_avoided = 0;
    // Opt-in first-prefill F16 cache. Only the active trunk layer lives here; no weights are requantized.
    uint16_t* lm_f16 = nullptr;
    size_t lm_f16_bytes = 0, lm_f16_unit = 0;
    int lm_f16_il = -1;
    std::vector<uint8_t> lm_f16_valid;
    float* lm_f16_y = nullptr;   // padded down outputs; compact c.y and original moe_combine are retained
    size_t lm_f16_y_bytes = 0;
    cublasHandle_t lm_f16_blas = nullptr;
    uint8_t *lm_f16_meta_host = nullptr, *lm_f16_meta_dev = nullptr;
    size_t lm_f16_meta_bytes = 0, lm_f16_meta_stride = 0;
    int lm_f16_meta_cap = 0;
    static constexpr int kF16MetaSlots = 16;
    cudaEvent_t lm_f16_meta_done[kF16MetaSlots] = {};
    bool lm_f16_meta_recorded[kF16MetaSlots] = {};
    uint64_t lm_f16_meta_count = 0, lm_f16_hits = 0, lm_f16_fills = 0, lm_f16_sets = 0, lm_f16_buckets = 0;
    uint64_t lm_f16_rows = 0, lm_f16_padded_rows = 0, lm_f16_fallback_light = 0, lm_f16_fallback_shape = 0,
             lm_f16_fallback_format = 0, lm_f16_fallback_blas = 0;
    uint64_t lm_f16_mixed_sets = 0, lm_f16_light_experts = 0, lm_f16_light_rows = 0;
    // Independent bounded cold-prefill arena. Never aliases resident experts or the old layer-major cache.
    static constexpr int kColdF16Experts = 8, kColdF16MetaSlots = 16;
    uint16_t* cold_f16 = nullptr;
    float* cold_f16_y = nullptr;
    uint8_t *cold_f16_meta_host = nullptr, *cold_f16_meta_dev = nullptr, *cold_f16_workspace = nullptr;
    size_t cold_f16_bytes = 0, cold_f16_unit = 0, cold_f16_y_bytes = 0,
           cold_f16_meta_bytes = 0, cold_f16_meta_stride = 0, cold_f16_workspace_bytes = 0;
    cublasHandle_t cold_f16_blas = nullptr;
    cudaEvent_t cold_f16_meta_done[kColdF16MetaSlots] = {};
    bool cold_f16_meta_recorded[kColdF16MetaSlots] = {};
    uint64_t cold_f16_meta_count = 0, cold_f16_sets = 0, cold_f16_batches = 0, cold_f16_buckets = 0,
             cold_f16_unpacks = 0, cold_f16_rows = 0, cold_f16_padded_rows = 0,
             cold_f16_light_experts = 0, cold_f16_light_rows = 0,
             cold_f16_fallback_shape = 0, cold_f16_fallback_format = 0,
             cold_f16_fallback_light = 0, cold_f16_fallback_blas = 0;
};

struct F16PhaseRow {
    int row0, rows, pad0, padded;   // row0 relative to the ORIGINAL run_set range; pad0 in reused chunk buffers
};
#if !defined(STRATA_USE_HIP)
__global__ static void f16_phase_gather(const F16PhaseRow* ds, int ne, int max_rows, const float* x, int D,
                                      const int* idx, uint16_t* xp) {
    const int e = (int) blockIdx.y;
    if (e >= ne) return;
    const F16PhaseRow r = ds[e];
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < (size_t) r.padded * D;
         i += (size_t) gridDim.x * blockDim.x) {
        const int row = (int) (i / D), col = (int) (i % D);
        const float v = row < r.rows ? x[(size_t) idx[r.row0 + row] * D + col] : 0.0f;
        xp[(size_t) r.pad0 * D + i] = __half_as_ushort(__float2half_rn(v));
    }
    (void) max_rows;
}
__global__ static void f16_phase_swiglu(const float* g, const float* u, size_t n, float limit, uint16_t* h) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        float gv = g[i], uv = u[i];
        if (limit > 0) { gv = fminf(gv, limit); uv = fminf(fmaxf(uv, -limit), limit); }
        h[i] = __half_as_ushort(__float2half_rn(__fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0f, expf(-gv))), uv)));
    }
}
__global__ static void f16_phase_scatter(const F16PhaseRow* ds, int ne, const uint16_t* hp, const float* yp,
                                       int D, int FF, uint16_t* h, float* y) {
    const int e = (int) blockIdx.y;
    if (e >= ne) return;
    const F16PhaseRow r = ds[e];
    const int width = D > FF ? D : FF;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < (size_t) r.rows * width;
         i += (size_t) gridDim.x * blockDim.x) {
        const int row = (int) (i / width), col = (int) (i % width);
        if (col < D) y[(size_t) (r.row0 + row) * D + col] = yp[(size_t) (r.pad0 + row) * D + col];
        if (col < FF) h[(size_t) (r.row0 + row) * FF + col] = hp[(size_t) (r.pad0 + row) * FF + col];
    }
}
#endif

struct Layer {
    int il = 0, dev = 0, ratio = 0;
    // its group (tensor parallel: nm members, each a copy of the device state in mates, member mi's share of the
    // heads, of wo_a's groups, of wo_b's rows and of the FFNs: gate / up columns, down rows; its part of an expert
    // laid out as hlay, hblob bytes - the host state, the CPU lane and the full blob layout stay with member 0)
    int grp = 0, nm = 1, mi = 0;
    std::vector<Layer> mates;
    x3::ExpertLayout hlay;
    size_t hblob = 0;
    int ff0 = 0, ffn = 0;     // this member's FFN columns [ff0, ff0 + ffn) (split128 of n_ff; the whole one alone)
    // this member's heads [ah0, ah0 + ahn) and wo_a groups [og0, og0 + ogn) - all of them alone; in a group of more
    // members than the heads split among evenly (head_members: 9, 7 or 6 over 64 heads), the first ones none: their
    // attention adds nothing to the sum
    int ah0 = 0, ahn = 0, og0 = 0, ogn = 0;
    // the chunked prompt path: each expert's virtual slot (its VRAM slot, or n_slots + its place among the CPU lane's
    // experts, which a chunk streams in that order) - pinned on member 0, a device copy on each member
    int* vs_pin = nullptr;
    int* vs_dev = nullptr;
    std::vector<int> host_list;
    std::string name;         // "layers.N." or, for the DSpark drafter's blocks, "mtp.N."
    int n_exp = 0, k_exp = 0; // routed experts, and how many a token takes (the drafter's: 128, 3)
    bool mtp = false;         // a DSpark block (index stage = il - n_layers)
    bool kv_src = false, ix_src = false, owns_k = false, cand_src = false, uses_cand = false;
    int src = -1;      // the kv source whose compressed KV this layer reads (index into Impl::srcs)
    int ik_src = -1;   // the kv source whose index keys its indexer reads
    int eg = -1;       // engram layer index, or -1
    x3::Mat wq_a, wq_b, wkv, wo_a[16], wo_b, s1, s2, s3, c_wkv, c_wgate, i_wq_b, i_wk, eg_wkv;
    float *q_norm = nullptr, *kv_norm = nullptr, *attn_norm = nullptr, *ffn_norm = nullptr, *c_norm = nullptr,
          *ik_norm = nullptr, *sink = nullptr, *hca_fn = nullptr, *hca_scale = nullptr, *hca_base = nullptr,
          *hcf_fn = nullptr, *hcf_scale = nullptr, *hcf_base = nullptr, *gate_bias = nullptr, *eg_w = nullptr,
          *gate_bias_vl = nullptr;   // (the image tokens' routing bias, with the vision tower)
    uint16_t* i_wproj = nullptr;
    float* gate_w = nullptr;   // the router's weights, F32 (DeepSeek's inference computes the gate in F32)
    uint8_t* win = nullptr;                     // [win] rows of kKvFp8 (kv_row_bytes)
    float *c_kv = nullptr, *c_score = nullptr;  // the compressor's partial group (ratio > 1)
    // a run of chunks' mid-run checkpoint (feed_mark): the ring and the partial group as they were before a row
    uint8_t* snap_win = nullptr;
    float *snap_kv = nullptr, *snap_sc = nullptr;
    // routed experts
    x3::ExpertLayout lay;
    size_t blob = 0;                            // bytes of one expert's blob
    int n_vram = 0;
    bool cpu_only = false;   // a role's type has no device kernel: every expert of this layer is the CPU lane's
    uint8_t* vram_base = nullptr;
    std::vector<int> slot;                      // expert -> VRAM slot or -1
    std::vector<int> ram;                       // expert -> RAM-tier slot or -1 (a GGUF's in-place experts; empty: none)
    std::vector<uint8_t*> host;                 // expert -> host (non-null: on the CPU lane; its halves in hh)
    // the CPU lane keeps an expert as NH column parts, part X on NUMA node X: gate / up columns [X n_ff / NH, ..),
    // down columns [X dim / NH, ..), each a standalone EXL3 matrix (its own trellis, the full suh, its part of svh)
    struct Half {
        size_t trellis[3], suh[3], svh[3], bytes;
        int k[3], n[3];
    } hl;
    std::vector<std::array<uint8_t*, 4>> hh;     // expert -> its parts on the nodes
    // the expert cache: one VRAM slot and one set of host parts kept free, so an expert can move in while the one it
    // replaces moves out; heat is how much each expert was used lately (the profile's share to start with)
    int n_slots = 0, spare_vram = -1;
    std::array<uint8_t*, 4> spare_host = {nullptr, nullptr, nullptr, nullptr};
    std::vector<float> heat;
    std::vector<int> cpu_hits;                  // CPU-lane experts used since the cache last looked
    bool swap_pending = false;
    unsigned long long* tab_pin = nullptr;      // pinned: the table entries a swap writes
    // a speculative window's backups (trunk layers): the ring rows it replaced, and a compressor's state before it and
    // the window's raw rows (kv, then score) to replay the accepted ones
    uint8_t* ring_bak = nullptr;
    float *c_snap = nullptr, *c_rows = nullptr;
    unsigned long long* vram_tab = nullptr;     // device [n_exp]
    std::vector<uint8_t> zc;                    // (the RAM store) expert -> 1: its store slot is GPU-visible, the device computes it over PCIe
    unsigned long long* zc_tab = nullptr;       // device [n_exp]: those slots' device addresses (0: none) - the decode token's zc_stage reads it
    std::vector<std::array<x3::Mat, 3>> vm;     // expert -> its gate, up, down as EXL3 views of its VRAM blob
    std::vector<uint64_t> region_off;           // expert -> file region (offset, bytes, file)
    std::vector<uint64_t> region_bytes;
    std::vector<int> region_file;
    size_t piece_src[3][3];                     // part (gate, up, down) x (trellis, suh, svh): offset in the region
    size_t piece_bytes[3][3];
    size_t dense_bytes = 0;
    // a GGUF's MXFP4 experts (lay.fmt == gg::kFmtMXFP4): matrix r (gate, up, down) of expert e lies at mx_off[r] +
    // e mx_stride[r] in file mx_file[r] - loaded from there, and read in place (the file's mapping) by the CPU lane
    int mx_file[3] = {-1, -1, -1};
    uint64_t mx_off[3] = {0, 0, 0}, mx_stride[3] = {0, 0, 0};
};

struct Src {
    int layer = 0, ratio = 1;
    int64_t cap = 0;
    // per device (nullptr where no layer reads it): rows of kKvFp4 (the compressed KV), kKvFp4p (the index keys)
    std::vector<uint8_t*> ckv, ik;
    uint8_t *h_ckv = nullptr, *h_ik = nullptr;   // pinned: a window's new rows, on their way to the other devices
    cudaEvent_t ev_ckv = nullptr, ev_ik = nullptr;
    // threads: whether devices outside the source's group read it, and the host-mapped hand-off of a window's new rows
    // (ckv rows, then ik rows at kWin * hd) and its flag
    bool remote = false;
    uint8_t* x_stage = nullptr;
    unsigned* x_flag = nullptr;
    // pipelined chunks: a second staging slot (slot 1; slot 0 is x_stage), its flag, and the readers' acknowledgements
    // [slot][device] (32 apart); readers: the devices outside the source's group that read it
    uint8_t* x_stage1 = nullptr;
    unsigned* x_flag1 = nullptr;
    unsigned* x_ack = nullptr;
    std::vector<int> readers;
    uint8_t* stage(int slot) const { return slot ? x_stage1 : x_stage; }
    unsigned* flag(int slot) const { return slot ? x_flag1 : x_flag; }
};

// a host thread that enqueues one device's work (threads mode)
struct Worker {
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    std::function<void()> job;
    bool busy = false, quit = false;
    void start() {
        th = std::thread([this] {
            for (;;) {
                std::function<void()> j;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return quit || job != nullptr; });
                    if (quit) return;
                    j = std::move(job);
                    job = nullptr;
                }
                j();
                {
                    std::lock_guard<std::mutex> lk(mu);
                    busy = false;
                }
                cv.notify_all();
            }
        });
    }
    void submit(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lk(mu);
            job = std::move(f);
            busy = true;
        }
        cv.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !busy; });
    }
    bool idle() {
        std::lock_guard<std::mutex> lk(mu);
        return !busy;
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        cv.notify_all();
        if (th.joinable()) th.join();
    }
};

struct Group {
    std::vector<int> mem;   // device indices, member order (member 0 the lead)
    ArGroup ar, arb;        // a window's all-reduces, a chunk's
    // the head group: the members' logits shares on their way to the lead, two slots (by seq), their flags and the
    // lead's acknowledgements (host-mapped)
    float* lg_stage = nullptr;
    unsigned *lg_flag = nullptr, *lg_ack = nullptr;
};

constexpr int kMaxL = 64;   // layers (the trunk's and the drafter's) a window plan covers

// what a window's layers need from the host, decided before they run (threads enqueue the layers concurrently): the
// mailbox seq of each layer (async: the device publishes its routing), the indexer's selection sizes and the candidate
// blocks as each layer sees them, the window's hand-off seq
struct WinPlan {
    bool par = false, async = false;
    unsigned win = 0;
    int k = 0;              // (a run of chunks) the chunk's index: its hand-off slot k % 2
    int snap_t = -1;        // (a run of chunks) the chunk's row whose state before it the layers keep (-1: none)
    int eg_slot = -1;       // (a run of chunks, layer-major) the Engram rows' slot (-1: k % 2)
    unsigned vl = 0;        // (a window) bit t: token t is inside a picture's span (routed with the image bias)
    unsigned seq[kMaxL] = {};
    int topk[kMaxL][kWin] = {}, cand[kMaxL][kWin] = {};
};

}  // namespace

// ------------------------------------------------------------------ the model
struct Model::Impl {
    Options o;
    Pack pk;
    Geo g;
    std::vector<Dev> devs;
    std::vector<Layer> L;
    std::vector<Src> srcs;
    std::vector<Group> groups;
    std::vector<std::unique_ptr<Worker>> workers;   // threads mode: one per device
    bool par = false;                 // threads mode (the windows: decode)
    bool prompt_par = false;          // the prompt path's chunks (a thread a GPU), as --tp / --threads asked
    unsigned win_seq = 0;             // the windows the threads ran (the hand-offs' seq)
    // the chunked prompt path (threads mode): chunk tokens a pass, runs shorter than chunk_min in windows
    int chunk = 0, chunk_min = 512;
    static constexpr bool kChunkedPromptDefault = true;   // the layer-major prompt is the default (gate: KL 0.1685 vs 0.1686, 8K prompt 6x, decode within 2% - ds4 notes-dsv41-r97)
    static constexpr int kFlowChunkMin = 2;   // chunk_min with every device's layers on the flow (see load)
    // --prefill-chunk at most (route() runs in kVlRows-row slabs).  The default stays kVlRows: a 16384-token chunk
    // reads a 16K prompt's experts once instead of twice (V100: 461.8 -> 321.3 s, 736 -> 506 GB read) but its buffers
    // take ~1.9 GB of VRAM experts (745 -> 644 on a 32 GB V100) from every decoded token
    static constexpr int kMaxChunk = 4 * kVlRows;
    std::vector<std::unique_ptr<Chunk>> chunks;   // per device
    size_t eg_cap = kWin;             // tokens h_eg holds
    int tap_cap = kWin;               // rows h_taps holds (a chunk seeds the drafter with its last 128)
    static constexpr int kYocoMin = 1024;   // a long prompt's skip (forward_chunks): the fewest rows worth skipping
    bool logits_all = false;          // the last feed was a chunk with logits for every row (chunk_logits)
    std::vector<float> chunk_logits;
    std::function<void(int64_t)> progress;   // a run of chunks: the tokens done so far (the main thread, while it waits)
    // a Stop while a run of chunks goes through the layers (Model::set_cancel, asked while forward_chunks waits): one
    // GPU stops after the layer (or the layer's chunk) it is on (halt); several stop at a chunk every one of them
    // reaches - none starts chunk halt_k or later, halt_k one past the latest any started - so no device waits on a
    // hand-off that never comes (Project Maya v1.1.20)
    std::function<bool()> cancel;
    std::atomic<bool> halt{false};
    std::mutex halt_mu;
    int halt_k = INT_MAX, halt_seen = -1;
    bool halted = false;                      // the last run ended early on a Stop
    static constexpr const char* kHalted = "stopped";
    bool chunk_go(int k) {   // chunk k may start on this device (several GPUs)
        std::lock_guard<std::mutex> lk(halt_mu);
        if (k >= halt_k) return false;
        halt_seen = std::max(halt_seen, k);
        return true;
    }
    bool halt_now() const { return devs.size() == 1 && halt.load(std::memory_order_acquire); }
    // a run's early end on device di: the expert reads still in flight land first (they count down items the next run
    // clears), then its streams are idle
    bool halt_end(int di, std::string& err) {
        Chunk& c = *chunks[(size_t) di];
        cudaSetDevice(devs[(size_t) di].id);
        if (c.flow)
            for (size_t i = c.base; i < c.n_read; ++i)
                while (fit(c, i).left.load(std::memory_order_acquire) != 0) c.io->wait_any(500);
        if (c.cs != nullptr) cudaStreamSynchronize(c.cs);
        cudaStreamSynchronize(devs[(size_t) di].st);
        err = kHalted;
        return false;
    }
    std::vector<cudaEvent_t> chunk_done;     // on the last layer's device: chunk k through every layer
    std::unique_ptr<std::atomic<int>[]> chunk_rec;   // chunk k's event recorded
    // a prompt the flow could take: the flow or windows of 16, whichever this PC runs it faster (pick_path) - this
    // PC's speeds measured at load (calibrate_paths), each path's predictions scaled by how long it really took
    struct PathModel {
        bool on = false;
        double dma = 0, copy = 0, disk = 0, cpu = 0, cpu_cold = 0;   // bytes/s (disk, cpu_cold 0: not measured)
        // the flow's taken / predicted so far [0] by the prompt's size class (2-31, 32-127, 128-511, 512 tokens and
        // more: a short one's fixed costs weigh more), and how many prompts set it ([1] unused)
        double k[2][4] = {{1, 1, 1, 1}, {1, 1, 1, 1}};
        int64_t kn[2][4] = {};
        double win_s = 0;                                              // windows: seconds a window took, so far
        int64_t win_n = 0;
        int last_cls = 0, last_nw = 0;                                 // the last pick's size class, its windows
        double m = 1;                                                  // the picks' spread, against the profile's
        int64_t n_m = 0;
        int last = -1;                                                 // the last pick: 1 the flow, 0 windows
        double last_pred = 0;                                          // its prediction (s, before its k)
    } pm;
    static constexpr int kPickMax = 2048;   // (from here on the flow: windows' work grows with every 16 tokens)
    static constexpr double kPickRes = 0.9;  // (this share of the CPU-lane experts in RAM: windows if predicted faster)
    std::atomic<int> lm_layers{0};           // (layer-major) the layers every chunk went through
    WinPlan wplan;
    int head_dev = 0;
    x3::Mat head;
    float* norm_w = nullptr;          // on the head device
    // the drafter: main_proj / main_norm (on its first block's device), its final norm, the markov head and the
    // confidence head (on the head device: the drafts' logits come from the trunk's head)
    x3::Mat d_main_proj;
    float *d_main_norm = nullptr, *d_norm = nullptr, *d_conf = nullptr;
    uint16_t *d_mk_embed = nullptr, *d_mk_head = nullptr;   // BF16 [vocab][rank], F16 [vocab][rank]
    // a head group of two or more: the markov steps over its members (each its rows of the bias and an argmax of its
    // share, the global one through host-mapped slots [step][member] {value, index, seq}) in place of the lead's whole
    // rows - E host-mapped (each member reads the picked token's row), a draft's seq in mk_seq
    bool mk_split = false;
    uint16_t* h_mk_embed = nullptr;
    unsigned* mk_slots = nullptr;
    unsigned mk_seq = 0;
    int d_dev = 0;                    // the drafter's first block's device
    bool spec_win = false;            // the window being run is a draft's verification (layer() keeps backups)
    float* h_taps = nullptr;          // pinned [kWin][targets][dim]: the last window's taps
    cudaEvent_t ev_seed = nullptr;    // the drafter's seeding read h_taps: the next window's taps wait for it
    float* d_confv = nullptr;         // on the head device
    int* d_ints = nullptr;            // on the head device: drafts in, picks out
    float conf_threshold = std::getenv("DS4_DSPARK_CONF") ? (float) std::atof(std::getenv("DS4_DSPARK_CONF")) : 0.5f;
    // a sampled round's drafts: the distribution q each one was drawn from (its kept list, on the head device) for
    // verify()'s rule - a point at the draft for the drafter's argmax; and the chain they were drawn with (verify()
    // checks it is its own)
    strata::dsv41k::SpecList* d_specq = nullptr;
    std::vector<strata::dsv41k::SpecList> h_specq = std::vector<strata::dsv41k::SpecList>(kWin);
    bool drafts_q = false;
    float dq_temp = 0.0f, dq_top_p = 1.0f, dq_min_p = 0.0f;
    int dq_top_k = 0;
    uint64_t dq_seed = 0, dq_counter = 0;
    // (tests) DS4_DSPARK_SAMPLE=1: a sampled request's drafts drawn from the drafter's distribution through the
    // request's chain, as DeepSeek's reference does (else its argmax: the same acceptance on Uranus, 31-32% of 5
    // drafts a round either way); DS4_DSPARK_QTEMP=f: those draws at f x the request's temperature (any q keeps the
    // rule exact); DS4_DSPARK_MAX=n: at most n drafts a round; DS4_DSPARK_DUMP=file: the drafter's inputs and outputs
    // (tools/ds4/dspark_ref.py checks them against DeepSeek's checkpoint)
    bool sample_drafts = std::getenv("DS4_DSPARK_SAMPLE") != nullptr && std::atoi(std::getenv("DS4_DSPARK_SAMPLE")) != 0;
    float q_temp_scale = std::getenv("DS4_DSPARK_QTEMP") ? (float) std::atof(std::getenv("DS4_DSPARK_QTEMP")) : 1.0f;
    int max_drafts = std::getenv("DS4_DSPARK_MAX") ? std::atoi(std::getenv("DS4_DSPARK_MAX")) : 0;
    std::FILE* dump_d = std::getenv("DS4_DSPARK_DUMP") ? std::fopen(std::getenv("DS4_DSPARK_DUMP"), "wb") : nullptr;
    float* h_dump = nullptr;          // (the dump's pinned copies: each block's stream, each block's ring)
    uint16_t* h_dump_ring = nullptr;

    uint16_t* embed_pinned = nullptr;
    std::vector<float> freqs_win, freqs_comp;
    std::vector<std::unique_ptr<Pool>> pools;   // one per NUMA node, its workers on that node's cores
    std::vector<std::vector<int>> cores;        // each node's physical cores
    int NH = 1;                                 // the CPU lane's parts of an expert (the NUMA nodes)
    // the CPU lane's experts read in place from the model's files (one part, the files' layout; the page cache keeps
    // what fits) instead of copied into RAM, and how much of them the load reads ahead (the most-routed first)
    bool file_experts = false;
    uint64_t warm_bytes = 0;
    static constexpr uint64_t kRamMargin = 8ull << 30;   // RAM left to the rest (the system, the page cache's churn)
    uint8_t* file_part(const Layer& l, int e) const {
        return pk.maps[(size_t) l.region_file[(size_t) e]].first + l.region_off[(size_t) e];
    }
    // host side of the CPU lane, a window's rows
    float *h_x = nullptr, *h_w = nullptr, *h_part = nullptr;
    int* h_idx = nullptr;
    uint8_t* h_eg = nullptr;          // the window's Engram rows: [engram layer][token] of [cols][hd] then the scales
    // the Engram rows' reader (its own threads and file handles), the rows already fetched, the window's batches (one
    // an Engram layer: asked for when the window's tokens are known, collected right before that layer)
    EgReader eg_io;
    EgRowCache eg_cache;
    std::mutex eg_cache_mu;
    std::vector<std::unique_ptr<EgBatch>> eg_win;
    // Engram layer li's embed rows, as the pack stores them: 0 MXFP8 ([scale, 32 codes] blocks, 33 bytes a 32), 1
    // ggml Q6_K (210 bytes a 256), 2 ggml IQ4_NL (18 bytes a 32: an fp16 d and 16 nibble bytes), 3 an old pack's FP8
    // codes with their E8M0 scales in a tensor of their own (see eg_issue)
    std::vector<uint8_t> eg_fmt;
    std::atomic<int64_t> eg_rows_n{0}, eg_hits_n{0}, eg_reads_n{0}, eg_wait_ns{0}, eg_io_ns{0};
    uint16_t* h_emb = nullptr;        // the window's embedding rows
    std::vector<float> c_xh, c_xsb, c_acc, c_xhd, c_xsd, c_accd, c_yd;
    // the state
    int64_t pos = 0;
    std::vector<int32_t> toks, ctoks;
    struct Check {                    // the state at a position: every window ring, the compressors' partial groups
        int64_t pos = -1;
        std::vector<std::vector<uint8_t>> win;
        std::vector<std::vector<float>> comp;
    };
    static constexpr int kChecks = 4;
    Check ck[kChecks];                // the end of the last prompt, the position before it, the anchor (the position
                                      // before the end of the first prompt of the run of prompts that each continued
                                      // the last in place: a chat template that drops earlier turns' reasoning at a new
                                      // user message re-renders the history from about there) and the prefix (where the
                                      // server said the prompt's shared start - system prompt and tools - ends: another
                                      // agent behind the same start goes back to it)
    bool slot_held = false;           // slot_save took this state and nothing has changed it since
    std::vector<uint32_t> prof;       // [nl][n_exp] routed-expert hits
    Stats st;
    // Experimental one-shot memory phase. The initial placement has ZERO resident slots; after the first feed
    // the phase arenas are retired and the ordinary placement/load runs before any decode can be submitted.
    size_t prefill_gpu_h_mb = 0;
    int prefill_decode_chunk = 0; // strict first-cold opt-in; default path unchanged
    size_t lm_f16_cache_mb = 0;
    bool lm_f16_mixed_light = false; // opt8: preserve whole-range NE16 light arithmetic beside hot BLAS experts
    bool prefill_phase_pending = false;
    bool prefill_copy_ram = false;   // explicit one-shot late NH1 copy; never runs again after the phase is consumed
    std::string prefill_phase_error;   // a failed transition is terminal: never decode through a partial pool
    // Distinct first-feed experiment: startup normal GPU pool and complete CPU
    // copies, chunk-major residuals on GPU, then monotonic pool growth.
    bool chunk_major_phase_enabled = false, chunk_major_phase_pending = false;
    bool decode_fusion = false; // separate default-off official decode-only fusion
    bool normalize_prefill_heat = false; // independent opt-in: same profile scale as fresh place_experts
    bool cold_f16_enabled = false;
    bool prefill_engram_cache = false; // DS4_PREFILL_ENGRAM_CACHE=1, no default-path allocation
    bool prefill_dense_projection_cache = false; // DS4_PREFILL_DENSE_PROJ_CACHE=1, separate experiment
    coldf16::Rule cold_f16_rule = coldf16::Rule::power2;
    static constexpr int chunk_major_decode_chunk = 1024;
    std::string chunk_major_phase_error;
    int cur_topk_n[kWin] = {}, cur_cand_n[kWin] = {};   // what the latest indexer / candidate source left, per query
                                                        // (plan_window's: the windows' plans are made in order)
    int n_logits = 0;                 // the last forward's logit rows
    // the decode path (windows of one): the device publishes each layer's routing into reqs, the service thread
    // answers in resps with the CPU lane's part, and the device waits for the answer only before the expert sum
    Req* reqs = nullptr;
    Resp* resps = nullptr;
    unsigned seq_issued = 0;
    std::atomic<int> layer_of[kRing];
    std::thread service;
    std::atomic<bool> quit{false};
    bool sync_decode = std::getenv("DS4_SYNC") != nullptr;
    // a window of one through moe_gate_up / moe_down (DS4_MOE_TOKEN=0: win_experts, whose down rows run
    // before the wait for the CPU lane - the same bits, no faster)
    bool moe_token_path = std::getenv("DS4_MOE_TOKEN") == nullptr || std::atoi(std::getenv("DS4_MOE_TOKEN")) != 0;
    int async_max_t = std::getenv("DS4_ASYNC_T") ? std::atoi(std::getenv("DS4_ASYNC_T")) : 8;   // windows up to this
    // lookahead: while layer L runs, the device guesses the next layer's picks (its gate on L's FFN input, the same
    // choice) and publishes them with L's routing; the CPU lane asks for the guessed experts it will read from the files
    // at once (a thread of its own, os::prefetch), so the SSD reads them while L computes instead of on the next
    // layer's critical path.  Only the page cache sees it - the outputs are the same.  DS4_AHEAD=0: off, 2: also the
    // layer after (from L's input too); DS4_AHEAD_EXTRA: guesses past the next layer's top k (< 0: fewer - the guess's
    // picks by rank on Sun: 97 91 81 70 57 41 %, past the top 6 29 20 %: reads wasted on an SSD-bound PC).
    int ahead_mode = std::getenv("DS4_AHEAD") ? std::atoi(std::getenv("DS4_AHEAD")) : 1;
    int ahead_extra = std::getenv("DS4_AHEAD_EXTRA") ? std::atoi(std::getenv("DS4_AHEAD_EXTRA")) : 0;
    // (DS4_AHEAD_ORDER=0: the CPU lane computes a layer's experts in expert order, not the ones asked for early first)
    bool ahead_order = std::getenv("DS4_AHEAD_ORDER") == nullptr || std::atoi(std::getenv("DS4_AHEAD_ORDER")) != 0;
    std::vector<uint8_t> ahead_fl;     // [layer][1024]: 1 the top-k guess, 2 any guess, a layer before; 4 two layers before;
                                       // 8 asked for (prefetch)
    std::vector<uint8_t> ahead_rank;   // [layer][1024]: 1 + the expert's rank in the guess a layer before (0: not guessed)
    std::vector<unsigned> ahead_for;   // [layer]: the request (seq) its flags are for
    std::thread pf_th;
    std::mutex pf_mu;
    std::condition_variable pf_cv;
    std::vector<std::pair<const void*, size_t>> pf_q;
    // the expert cache: swaps prepared on threads of their own (DS4_CACHE_THREADS, 1 by default - 4 measured no faster
    // on one V100 - more for a PC to try: each its staging buffers and
    // copy streams), switched between windows
    struct Swap {
        int layer = 0, in = -1, out = -1;
        int ram = -1;   // (copy_mode, `in` held in RAM) the spare slot `out` is copied into from VRAM
        bool epoch_prepared_ok = false; // opt-in success bookkeeping only, never assumed from an empty queue
    };
    static constexpr int kMaxSwaps = 8;
    // the swap rule: a CPU-lane expert in when its heat passes the coldest VRAM one's x margin + bonus; heat x decay
    // every 256 tokens; at most `inflight` swaps under way (DS4_CACHE_MARGIN / _BONUS / _DECAY / _INFLIGHT)
    static float env_f(const char* k, float d) { return std::getenv(k) ? (float) std::atof(std::getenv(k)) : d; }
    const float c_margin = env_f("DS4_CACHE_MARGIN", 1.25f), c_bonus = env_f("DS4_CACHE_BONUS", 1.0f),
                c_decay = env_f("DS4_CACHE_DECAY", 0.9f);
    const int c_inflight = std::max(1, (int) env_f("DS4_CACHE_INFLIGHT", (float) kMaxSwaps));
    // the RAM tier (a GGUF's in-place experts, as GLM's: the hottest in VRAM, the next hottest here, the coldest in the
    // files): RAM slots of one expert blob each, filled at load with the hottest experts VRAM does not hold, so the
    // CPU lane reads them from RAM; between windows (cache_tick) a hotter expert from the files takes the place of the
    // layer's coldest held one - read into a free slot by ram_th, switched while nothing reads them.  Opt-in
    // (STRATA_DS4_RAM_TIER=1): on a V100 with 64 GB of RAM it was slower than the page cache alone (1.4 against 2.2-2.4
    // tokens/s sampled) - the cache follows a conversation's experts at once, the tier's moves lag behind.
    struct RamSwap {
        int layer = 0, in = -1, out = -1, slot = -1;
    };
    std::vector<uint8_t*> ram_slots, ram_chunks;
    std::vector<size_t> ram_chunk_bytes;
    bool ram_pinned = false;   // the slots are pinned memory (the prompt path DMAs them from where they lie)
    size_t flow_part[3] = {0, 0, 0};   // the flow's staging room for each part: its largest over the trunk's layers
    std::vector<int> ram_free;
    std::deque<RamSwap> ram_queue;
    std::vector<RamSwap> ram_ready;
    std::condition_variable ram_cv;
    std::thread ram_th;
    int ram_inflight = 0, ram_held = 0;
    bool ram_tier = std::getenv("STRATA_DS4_RAM_TIER") != nullptr && std::atoi(std::getenv("STRATA_DS4_RAM_TIER")) != 0;
    // STRATA_DS4_DISK_SKIP_W (experimental opt-in, CHANGES THE ANSWERS; 0 = off): on the small windows (decode, a
    // draft's check - never the prompt path) the CPU lane leaves out a routed expert whose share of its token's route
    // weights is below this and whose pages are not in RAM now (the page cache or the RAM tier; os::in_ram - Linux every
    // page, Windows sampled pages of the working set), so a token waits on the SSD only for the experts that weigh; the
    // weight it carried is lost (no renormalisation).  On layers from STRATA_DS4_DISK_SKIP_FROM (0) on.  An expert left
    // out still counts as used (the expert cache, the profile).
    const float disk_skip_w = [] {
        const char* v = std::getenv("STRATA_DS4_DISK_SKIP_W");
        return v != nullptr ? std::max(0.0f, (float) std::atof(v)) : 0.0f;
    }();
    const int disk_skip_from = [] {
        const char* v = std::getenv("STRATA_DS4_DISK_SKIP_FROM");
        return v != nullptr ? std::max(0, std::atoi(v)) : 0;
    }();
    // the RAM store (--ssd-cache): the CPU lane's experts in slots a policy owns, read unbuffered (null: off)
    std::unique_ptr<ExpertStore> store;
    // Experimental copied-RAM GPU decode misses; all owners remain outside Chunk.
    bool gpu_misses_enabled=false;
    bool gpu_miss_direct_enabled=false,gpu_misses_decode_pins=false;
    bool gpu_misses_recovery_needed=false,gpu_misses_recovery_preparing=false;
    unsigned gpu_misses_recovery_lease=0;
    uint64_t gpu_misses_recovery_attempts=0;
    size_t gpu_misses_last_decode_pool=0;
    uint8_t* gpu_misses_stage=nullptr;
    size_t gpu_misses_stride=0,gpu_misses_bytes=0;
    uint8_t* gpu_misses_host[2]={nullptr,nullptr};
    cudaEvent_t gpu_misses_done[2]={nullptr,nullptr};
    bool gpu_misses_busy[2]={false,false};
    cudaStream_t gpu_misses_cs=nullptr;
    cudaEvent_t gpu_misses_ready=nullptr;
    unsigned long long* gpu_misses_plan_pin=nullptr;
    uint8_t* zc_stage_buf = nullptr;   // device: k staging slots for the store's experts of a decode token
    size_t zc_stride = 0;
    bool zc_copy = true;               // DS4_STORE_STAGE=0: the expert kernels read the arena over PCIe themselves
    bool in_ram(const Layer& l, int e) const {   // (DISK_SKIP_W) expert e's pages in RAM now
        if (!l.ram.empty() && l.ram[(size_t) e] >= 0) return true;
        if (l.lay.fmt == gg::kFmtMXFP4) {
            for (int r = 0; r < 3; ++r)
                if (!os::in_ram(mx_mat(l, e, r), mx_part_bytes(l, r))) return false;
            return true;
        }
        return !file_experts || os::in_ram(file_part(l, e), l.blob);
    }
    // a GGUF's CPU-lane experts copied into RAM (copy_experts, after VRAM took its share: --mmap-experts off - auto reads
    // them in place; DS4_COPY_GB=<n>: the hottest n GB, the rest in place - static, no moves between
    // them): RAM slots (l.ram, as the tier's) of NH parts, part X on NUMA node X with rows [X n / NH, (X + 1) n / NH) of
    // each matrix at cp_lay[layer][r] - whole rows, so the split never cuts a GGUF block (MXFP4's 32, the IQ / K types'
    // 256 values: blocks run along a row); one spare slot a layer for the cache's swaps (the expert VRAM gives back takes
    // the RAM place of the one it takes in)
    // Windows' commit on an integrated GPU: its allocations are charged to the commit (RAM + page file) like any private
    // memory - measured on a Radeon 8065S with 160 GiB of its own: the system's free commit fell by each one, whatever
    // its size (one above ~3 GiB also counts as the process's private bytes, a smaller one does not, the commit the
    // same), so a page file at least the GPU's share is what lets it hold experts at all.  The experts it holds
    // (place_experts) and the copies stay within what Windows can still commit, less max(8 GB, 10% of the RAM) for the
    // other programs - commit_kept: the experts that kept off the GPU, pagefile_hint: the page file (GB) that would hold
    // them (the log's WARNING, a NOTE and INFO for the server)
    double commit_kept = 0, pagefile_hint = 0;
    double commit_margin() const {
        const os::Commit c = os::commit();
        return std::max(8.0 * (double) (1ull << 30), c.ram == ~uint64_t{0} ? 0.0 : 0.1 * (double) c.ram);
    }
    // what may still be committed beside the margin (~inf: nothing limits it; DS4_COMMIT_CAP_GB: at most that, tests)
    double commit_room() const {
        const os::Commit c = os::commit();
        double avail = c.avail == ~uint64_t{0} ? 1e300 : (double) c.avail;
        if (const char* v = std::getenv("DS4_COMMIT_CAP_GB")) avail = std::min(avail, std::atof(v) * 1e9);
        return avail >= 1e300 ? 1e300 : std::max(0.0, avail - commit_margin());
    }
    // a device whose allocations Windows charges to the commit: an integrated GPU (DS4_COMMIT_CAP_GB: any, tests)
    bool commit_charged(const Dev& d) const {
        if (std::getenv("DS4_COMMIT_CAP_GB") != nullptr) return true;
#if defined(_WIN32)
        int v = 0;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrIntegrated, d.id) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        return v != 0;
#else
        (void) d;
        return false;
#endif
    }
    bool copy_mode = false;
    size_t register_ram_mb = 0; bool register_ram_all = false;
    size_t prefill_final_reserve_mb = 0, register_gpu_floor_mb = 0; // independent default-off first-cold policies
    regcopy::Ledger registered_copy;
    double registered_copy_setup_ms = 0, registered_copy_driver_ms = 0, registered_copy_unregister_ms = 0;
    std::vector<std::array<uint8_t*, 2>> ram_part;   // slot -> its parts
    std::vector<std::array<size_t, 4>> cp_lay;       // layer -> part offsets of gate, up, down rows; a part's bytes
    std::vector<int> copy_spare;                      // layer -> its spare slot (-1: none)
    // Experimental prerequisite only: no rearm, moves, resize or changed cache policy.
    bool record_copied_ram_domains = false, copied_ram_domains_published = false;
    std::vector<ramdomain::Domain> copied_ram_domains; // immutable after successful initial-copy publication
    bool cache_epoch_enabled=false;
    bool cache_epoch_probe_enabled=false; // explicit native EPOCH command only; no automatic request overhead
    bool cache_pool_shrink_probe_enabled=false; // separate explicit diagnostic only
    bool cache_workspace_regrow_probe_enabled=false; // separate native diagnostic only
    bool prefill_repeat_rearm=false, prefill_repeat_preparing=false; // independent explicit capability
    int prefill_repeat_min_tokens=repeatpolicy::default_min_tokens; // optional interactive threshold; old default retained
    uint64_t prefill_repeat_attempts=0,prefill_repeat_generation=0;
    uint64_t prefill_repeat_cold_sets=0,prefill_repeat_cold_unpacks=0;
    uint64_t prefill_repeat_eg_fills=0,prefill_repeat_eg_hits=0,prefill_repeat_dense_fills=0,prefill_repeat_dense_hits=0;
    epochpolicy::Machine cache_epoch_machine; // caller-owned, serialized Model API boundary only
    std::atomic<bool> cache_epoch_block_enqueues{false};
    std::atomic_flag cache_epoch_owner=ATOMIC_FLAG_INIT;
    std::atomic<unsigned> cache_epoch_request{0},cache_epoch_service_target{0},cache_epoch_service_ack{0};
    std::atomic<unsigned> cache_epoch_service_completed{0};
    size_t cache_epoch_worker_acks=0; // protected by sw_mu, one acknowledgment per worker/epoch
    std::mutex sw_mu;
    std::condition_variable sw_cv;
    std::deque<Swap> sw_queue;
    std::vector<Swap> sw_ready;
    struct SwapBufs {
        uint8_t *a = nullptr, *b = nullptr, *h = nullptr;
        std::vector<cudaStream_t> st;   // a copy stream a device
    };
    std::vector<SwapBufs> sw_bufs;
    std::vector<std::thread> cache_ths;
    int sw_inflight = 0;
    int64_t decay_tokens = 0;
    bool check_swaps = std::getenv("DS4_CHECK_SWAPS") != nullptr;
    std::vector<float> host_logits;
    std::vector<void*> host_allocs;
    std::FILE* dump_h = std::getenv("DS4_DUMP_H") ? std::fopen(std::getenv("DS4_DUMP_H"), "wb") : nullptr;
    // DS4_DUMP_L: (threads) every layer's residual streams as records [layer, p0, T, T x hc x dim floats], window or
    // chunk path (tests: where the two part)
    std::FILE* dump_l = std::getenv("DS4_DUMP_L") ? std::fopen(std::getenv("DS4_DUMP_L"), "wb") : nullptr;
    // DS4_DUMP_R: the chunk path's routing, a record a layer [layer, p0, T, k, n_exp, n_vs, n_slots, kPicks] (int64),
    // then the picks' experts [T kPicks] (the first k of each token's), each expert's virtual slot [n_exp], the
    // slots' row bounds [n_vs + 1], the sorted rows' tokens [T k] (int32), the gate's logits [T n_exp] and its bias
    // [n_exp] (F32)
    std::FILE* dump_r = std::getenv("DS4_DUMP_R") ? std::fopen(std::getenv("DS4_DUMP_R"), "wb") : nullptr;
    // DS4_DUMP_Y=<layer>:<file> (layer -1: every layer): its expert sets on the chunk path, a record a piece [layer,
    // row0, rows, n_exp, wbase kind (0 VRAM, 1 ring), max_rows] (int64), its bounds [n_exp + 1], its h rows [rows ff]
    // (F16) and y rows [rows dim] (F32)
    int dump_y_layer = std::getenv("DS4_DUMP_Y") ? std::atoi(std::getenv("DS4_DUMP_Y")) : -1;
    std::FILE* dump_y = [] {
        const char* v = std::getenv("DS4_DUMP_Y");
        const char* f = v != nullptr ? std::strchr(v, ':') : nullptr;
        return f != nullptr ? std::fopen(f + 1, "wb") : nullptr;
    }();
    std::mutex dump_mu;
    // DS4_PROF: the GPU's time a decoded token by stage - events in stream order at a layer's start, after its
    // attention, after the routing and the shared expert, after the experts' gate / up, after the wait for the CPU
    // lane, after the down pass and the post - read after the token's sync, each the time since the mark before it
    bool prof_on = std::getenv("DS4_PROF") != nullptr;
    // DS4_PRED_DUMP=file (predictor data, the chunked prompt path): per layer and chunk u8 layer, u16 T, T x 384 fp16 router
    // logits, T x 256 fp16 projection of the router input.  DS4_XIN_DUMP=file (expert similarity inputs): u8 layer, u16 T,
    // T x D fp32 router inputs, the first DS4_XIN_N (1024) tokens of each layer.
    std::FILE* pred_f = std::getenv("DS4_PRED_DUMP") != nullptr ? std::fopen(std::getenv("DS4_PRED_DUMP"), "wb") : nullptr;
    std::FILE* xin_f = std::getenv("DS4_XIN_DUMP") != nullptr ? std::fopen(std::getenv("DS4_XIN_DUMP"), "wb") : nullptr;
    int xin_n = std::getenv("DS4_XIN_N") ? std::atoi(std::getenv("DS4_XIN_N")) : 1024;
    std::array<int, 64> xin_got{};
    std::FILE* trace_f = std::getenv("DS4_TRACE") != nullptr ? std::fopen(std::getenv("DS4_TRACE"), "wb") : nullptr;
    struct ProfMark {
        const Dev* d;
        cudaEvent_t ev;
        int stage;
    };
    std::mutex prof_mu;
    std::vector<ProfMark> prof_marks;
    std::map<int, std::vector<cudaEvent_t>> prof_pool;   // by device: an event belongs to the device it was made on
    bool prof_window = false;   // (decode windows only: T == 1)
    // ... and a run of chunks' (prof_chunk): a layer's start, after its attention, after the routing and the shared
    // expert, after its experts (the streamed ones' waits for their reads in it); printed after the run
    bool prof_chunk = false;
    // DS4_PROF stages of the chunked prompt (finer ones than the decode's 0..5): 0 between layers, 1 attention rest (wq_b, wo,
    // hc_post), 2 router-side tail (shared expert), 3 experts (+ waits), 4 Engram, 5 attn mHC (mix, split, norm), 6 attention
    // projections (wq_a, wkv, compressor), 7 compressor + indexer, 8 q_b + rope, 9 sparse attention, 10 ffn mHC + router + sort
    double cprof_ms[16] = {};
    // DS4_PROF: the experts' kernels alone, by type (events around the gate/up and down calls of run_set)
    struct EkMark {
        cudaEvent_t e0, e1, e2;
        int tg, td, nrows, nexp, maxrows, dev, ff;
    };
    std::vector<EkMark> ek_marks;
    std::map<int, std::vector<cudaEvent_t>> ek_pool;   // by device
    cudaEvent_t ek_ev() {
        int cur = 0;
        cudaGetDevice(&cur);
        std::vector<cudaEvent_t>& pl = ek_pool[cur];
        if (!pl.empty()) {
            cudaEvent_t e = pl.back();
            pl.pop_back();
            return e;
        }
        cudaEvent_t e = nullptr;
        cudaEventCreate(&e);
        return e;
    }
    void prof_mark(const Dev& d, int stage) {
        if (!prof_window && !prof_chunk) return;
        std::lock_guard<std::mutex> lk(prof_mu);
        int cur = 0;
        cudaGetDevice(&cur);
        cudaSetDevice(d.id);   // (events of another device are an "invalid resource handle")
        cudaEvent_t e = nullptr;
        std::vector<cudaEvent_t>& pool = prof_pool[d.id];
        if (pool.empty()) cudaEventCreate(&e);
        else {
            e = pool.back();
            pool.pop_back();
        }
        cudaEventRecord(e, d.st);
        cudaSetDevice(cur);
        prof_marks.push_back({&d, e, stage});
    }
    void prof_collect() {
        if (!prof_window && !prof_chunk) return;
        std::lock_guard<std::mutex> lk(prof_mu);
        std::vector<std::pair<const Dev*, cudaEvent_t>> last;
        for (const ProfMark& m : prof_marks) {
            auto it = std::find_if(last.begin(), last.end(), [&](const auto& x) { return x.first == m.d; });
            if (it != last.end()) {
                float ms = 0.0f;
                cudaSetDevice(m.d->id);
                if (cudaEventElapsedTime(&ms, it->second, m.ev) == cudaSuccess)
                    (prof_chunk ? cprof_ms : st.prof_ms)[m.stage] += ms;
                it->second = m.ev;
            } else {
                last.push_back({m.d, m.ev});
            }
        }
        for (const ProfMark& m : prof_marks) prof_pool[m.d->id].push_back(m.ev);
        prof_marks.clear();
        if (!prof_chunk) ++st.prof_tokens;
    }
    void dump_layer(Dev& d, const float* h, int il, int64_t p0, int T) {
        std::vector<float> v((size_t) T * g.hc * g.dim);
        cudaSetDevice(d.id);
        cudaMemcpyAsync(v.data(), h, v.size() * 4, cudaMemcpyDeviceToHost, d.st);
        cudaStreamSynchronize(d.st);
        std::lock_guard<std::mutex> lk(dump_mu);
        const int64_t hdr[3] = {il, p0, T};
        std::fwrite(hdr, 8, 3, dump_l);
        std::fwrite(v.data(), 4, v.size(), dump_l);
        std::fflush(dump_l);
    }

    ~Impl() {
        eg_io.stop();   // (its threads write into h_eg)
        for (auto& w : workers) w->stop();
        for (size_t i = 0; i < chunks.size(); ++i)
            if (chunks[i]) {
                cudaSetDevice(devs[i].id);
                if (chunks[i]->blas) cublasDestroy(chunks[i]->blas);
                if (chunks[i]->lm_f16_blas) cublasDestroy(chunks[i]->lm_f16_blas);
                for (cudaEvent_t e : chunks[i]->lm_f16_meta_done) if (e) cudaEventDestroy(e);
                if (chunks[i]->cold_f16_blas) cublasDestroy(chunks[i]->cold_f16_blas);
                for (cudaEvent_t e : chunks[i]->cold_f16_meta_done) if (e) cudaEventDestroy(e);
            }
        quit.store(true);
        if (service.joinable()) service.join();
        {
            std::lock_guard<std::mutex> lk(pf_mu);
        }
        pf_cv.notify_all();
        if (pf_th.joinable()) pf_th.join();
        {
            std::lock_guard<std::mutex> lk(sw_mu);
        }
        sw_cv.notify_all();
        ram_cv.notify_all();
        for (std::thread& t : cache_ths)
            if (t.joinable()) t.join();
        if (ram_th.joinable()) ram_th.join();
        bool release_gpu_misses=true;
        if(gpu_misses_enabled) {
            std::string why;
            try { release_gpu_misses=drain_gpu_misses(why); }
            catch(...) { release_gpu_misses=false; }
            if(!release_gpu_misses)std::fprintf(stderr,"ds4: ERROR: GPU miss exit drain unproven; all device/host/source owners retained for process teardown\n");
            std::fprintf(stderr,"ds4: GPU_MISS exit: layer_windows=%lld kernel_windows=%lld staged_entries=%lld unique_blobs=%lld h2d_bytes=%lld fallbacks=%lld drained=%d\n",
                (long long)st.gpu_miss_windows,(long long)st.gpu_miss_kernel_windows,(long long)st.gpu_miss_entries,
                (long long)st.gpu_miss_unique,(long long)st.gpu_miss_bytes,(long long)st.gpu_miss_fallbacks,
                release_gpu_misses?1:0);
        }
        std::string why_for_gpu_miss_exit;
        bool release_cpu_arenas=release_gpu_misses;
        if(release_cpu_arenas&&!registered_copy.held.empty()) {
            try { release_cpu_arenas=gpu_misses_decode_pins?gpu_misses_retire_pins("exit",why_for_gpu_miss_exit):drain_and_unregister_copied_ram_at_exit(); }
            catch(...) { release_cpu_arenas=false; }
            if(!release_cpu_arenas)
                std::fprintf(stderr,"ds4: ERROR: registered-copy exit drain/unregister unproven; retaining CPU source arenas for process teardown\n");
        }
        if(release_cpu_arenas)
            for (size_t i = 0; i < ram_chunks.size(); ++i) os::anon_free(ram_chunks[i], ram_chunk_bytes[i]);
        for (Dev& d : devs) {
            if(!release_gpu_misses)continue;
            cudaSetDevice(d.id);
            if(prefill_dense_projection_cache) for(size_t di=0;di<chunks.size();++di) if(chunks[di] && &devs[di]==&d && chunks[di]->dense_projection_cache) {
                std::string why;
                if(!retire_prefill_dense_projection_cache(d,*chunks[di],why))
                    std::fprintf(stderr,"ds4: ERROR: %s; remaining dense projection cache stays in device owner ledger until teardown\n",why.c_str());
            }
            if(prefill_engram_cache) for(size_t di=0;di<chunks.size();++di) if(chunks[di] && &devs[di]==&d && chunks[di]->eg_weight_cache) {
                std::string why;
                if(!retire_prefill_engram_cache(d,*chunks[di],why))
                    std::fprintf(stderr,"ds4: ERROR: %s; remaining Engram cache stays in device owner ledger until teardown\n",why.c_str());
            }
            for (void* p : d.allocs) cudaFree(p);
            if (d.st) cudaStreamDestroy(d.st);
            if (d.ev) cudaEventDestroy(d.ev);
        }
        if(release_gpu_misses)for (void* p : host_allocs) cudaFreeHost(p);
        for (Layer& l : L)
            for (uint8_t* b : l.host)
                if (b != nullptr) {
                    (void) b;   // host blobs live in one mapping per layer, unmapped below
                }
        if(release_gpu_misses)for (auto& m : host_maps) {
            if (chunk > 0) cudaHostUnregister(m.first);
            os::anon_free(m.first, m.second);
        }
    }
    std::vector<std::pair<void*, size_t>> host_maps;

    template <class T>
    bool dalloc(Dev& d, T*& p, size_t n, std::string& err, size_t* acct = nullptr) {
        void* v = nullptr;
        cudaSetDevice(d.id);
        const size_t b = align256(n * sizeof(T));
        if (cudaMalloc(&v, b) != cudaSuccess) {
            err = "cudaMalloc " + std::to_string(b >> 20) + " MiB on CUDA" + std::to_string(d.id) + " failed";
            return false;
        }
        // zeroed before anything lands in it: the loads copy in on the default stream, which the non-blocking d.st
        // does not order against - a zeroing still queued there could wipe the weights just copied
        if (cudaMemsetAsync(v, 0, b, d.st) != cudaSuccess || cudaStreamSynchronize(d.st) != cudaSuccess) {
            cudaFree(v);
            err = "zeroing " + std::to_string(b >> 20) + " MiB on CUDA" + std::to_string(d.id) + " failed";
            return false;
        }
        d.allocs.push_back(v);
        if (acct) *acct += b;
        else d.scratch += b;
        p = (T*) v;
        return true;
    }
    static Layer& member(Layer& l, int m) { return m == 0 ? l : l.mates[(size_t) m - 1]; }
    static const Layer& member(const Layer& l, int m) { return m == 0 ? l : l.mates[(size_t) m - 1]; }
    // bytes of a VRAM slot of l's experts (a member's part of each, in a group)
    static size_t unit(const Layer& l) { return l.nm > 1 ? l.hblob : l.blob; }
    // the members of a group of nm that take heads: the most the heads and the wo_a groups split among evenly (64
    // heads and 8 groups: 9 -> 8, 7 or 6 -> 4, 5 -> 4, 3 -> 2); the first members the rest, only their FFN share
    int head_members(int nm) const {
        int na = nm;
        while (na > 1 && (g.heads % na != 0 || g.o_groups % na != 0)) --na;
        return na;
    }
    // host-mapped memory (the same address on the host and every device), zeroed
    template <class T>
    bool hmap(T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        if (cudaHostAlloc(&v, n * sizeof(T) + 64, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            err = "host-mapped memory (" + std::to_string((n * sizeof(T)) >> 20) + " MiB)";
            return false;
        }
        std::memset(v, 0, n * sizeof(T) + 64);
        host_allocs.push_back(v);
        p = (T*) v;
        return true;
    }
    template <class T>
    bool halloc(T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        if (cudaHostAlloc(&v, n * sizeof(T) + 64, cudaHostAllocPortable) != cudaSuccess) {
            err = "cudaHostAlloc " + std::to_string((n * sizeof(T)) >> 20) + " MiB failed";
            return false;
        }
        host_allocs.push_back(v);
        p = (T*) v;
        return true;
    }

    // Permanent decode scratch, allocated before expert placement. No CPU arena is
    // borrowed by asynchronous DMA: only these two owned pinned bounces are sources.
    bool allocate_gpu_misses(std::string& err) {
        if(!gpu_misses_enabled)return true;
#if defined(STRATA_USE_HIP)
        err="DS4_GPU_MISSES requires CUDA";return false;
#else
        bool supported=pk.gguf&&devs.size()==1&&NH==1&&o.mmap_experts==0&&o.ssd_cache_gb==0&&
            !ram_tier&&g.dim==5120&&g.ff==2304&&g.n_exp==384&&g.k_exp==6&&
            !gpu_misses_stage&&!gpu_misses_host[0]&&!gpu_misses_host[1]&&!gpu_misses_plan_pin;
        size_t largest=0;
        for(const Layer& l:L) {
            supported=supported&&l.nm==1&&l.dev==0&&!l.cpu_only&&l.lay.fmt==gg::kFmtMXFP4&&
                l.k_exp>=1&&l.k_exp<=gpumiss::max_picks;
            for(int r=0;r<3;++r)supported=supported&&gg::type_on_device(gg::expert_type(l.lay,r));
            largest=std::max(largest,l.blob);
        }
        if(!supported||!gpumiss::sizes(largest,gpu_misses_stride,gpu_misses_bytes)) {
            err="DS4_GPU_MISSES requires copied-RAM/store0 single-GPU/NH1 GGUF [5120,2304], six picks and device-supported expert types";
            return false;
        }
        if(gpu_miss_direct_enabled&&(!cache_epoch_enabled||!record_copied_ram_domains||!prefill_repeat_rearm||
            !chunk_major_phase_enabled||!gpumiss::direct_cap(register_ram_mb,register_ram_all))) {
            err="DS4_GPU_MISS_DIRECT requires copied-RAM epoch/domains/repeat phase and explicit98304 or196608MiB registration cap (separate contrasts)";return false;
        }
        Dev& d=devs[0];
        if(cudaSetDevice(d.id)!=cudaSuccess){err="GPU miss scratch device selection";return false;}
        const size_t host_need=gpu_misses_stride*2+gpumiss::physical_stride*gpumiss::max_tokens*8+3*64;
        const uint64_t avail=ram_avail();const os::Commit cm=os::commit();
        if((avail!=~uint64_t{0}&&(avail<(8ull<<30)||host_need>avail-(8ull<<30)))||
           (cm.avail!=~uint64_t{0}&&(cm.avail<(512ull<<20)||host_need>cm.avail-(512ull<<20)))) {
            err="GPU miss pinned scratch exceeds RAM/commit margins";return false;
        }
        // Reserve owner-ledger capacity before the first CUDA allocation.
        d.allocs.reserve(d.allocs.size()+1);host_allocs.reserve(host_allocs.size()+3);
        if(!dalloc(d,gpu_misses_stage,gpu_misses_bytes,err)||
           !halloc(gpu_misses_host[0],gpu_misses_stride,err)||
           !halloc(gpu_misses_host[1],gpu_misses_stride,err)||
           !halloc(gpu_misses_plan_pin,(size_t)gpumiss::physical_stride*gpumiss::max_tokens,err))return false;
        if(cudaStreamCreateWithFlags(&gpu_misses_cs,cudaStreamNonBlocking)!=cudaSuccess||
           cudaEventCreateWithFlags(&gpu_misses_ready,cudaEventDisableTiming)!=cudaSuccess) {
            err="GPU miss private copy stream/ready event allocation";return false;
        }
        for(int i=0;i<2;++i) {
            std::memset(gpu_misses_host[i],0,gpu_misses_stride);
            if(cudaEventCreateWithFlags(&gpu_misses_done[i],cudaEventDisableTiming)!=cudaSuccess) {
                err="GPU miss scratch event allocation";return false;
            }
        }
        std::memset(gpu_misses_plan_pin,0,gpumiss::physical_stride*gpumiss::max_tokens*8);
        std::fprintf(stderr,"ds4: GPU_MISS setup: capacity24 maxT4 maxK6 stride%zu device_bytes%zu host_bounce_bytes%zu\n",
                     gpu_misses_stride,gpu_misses_bytes,gpu_misses_stride*2);
        return true;
#endif
    }

    // Also covers H2D success followed by EventRecord failure, before busy=true.
    // A false result forbids freeing any source/device owner; attempt BOTH drains.
    bool drain_gpu_misses(std::string& err) {
        if(!gpu_misses_stage&&!gpu_misses_host[0]&&!gpu_misses_host[1]&&!gpu_misses_plan_pin&&
           !gpu_misses_cs&&!gpu_misses_ready)return true;
        if(devs.size()!=1||cudaSetDevice(devs[0].id)!=cudaSuccess) {
            err="GPU miss exit drain device selection";return false;
        }
        const cudaError_t copy_status=gpu_misses_cs?cudaStreamSynchronize(gpu_misses_cs):cudaSuccess;
        const cudaError_t compute_status=devs[0].st?cudaStreamSynchronize(devs[0].st):cudaErrorInvalidValue;
        if(copy_status!=cudaSuccess||compute_status!=cudaSuccess) {
            err="GPU miss copy/compute exit drain unproven";return false;
        }
        for(int i=0;i<2;++i) {
            gpu_misses_busy[i]=false;
            if(gpu_misses_done[i]) {
                if(cudaEventDestroy(gpu_misses_done[i])!=cudaSuccess) {
                    err="GPU miss event retirement failed";return false;
                }
                gpu_misses_done[i]=nullptr;
            }
        }
        if(gpu_misses_ready) {
            if(cudaEventDestroy(gpu_misses_ready)!=cudaSuccess){err="GPU miss ready event retirement failed";return false;}
            gpu_misses_ready=nullptr;
        }
        if(gpu_misses_cs) {
            if(cudaStreamDestroy(gpu_misses_cs)!=cudaSuccess){err="GPU miss copy stream retirement failed";return false;}
            gpu_misses_cs=nullptr;
        }
        return true;
    }

    // ---- weights
    bool upload(Dev& d, const Tensor& t, void* dst, std::string& err) {
        std::vector<uint8_t> tmp(t.bytes);
        if (!pk.read(t, tmp.data())) {
            err = "read failed";
            return false;
        }
        cudaSetDevice(d.id);
        CK(cudaMemcpy(dst, tmp.data(), t.bytes, cudaMemcpyHostToDevice));
        return true;
    }
    // a dense matrix in a ggml block type (x3::Mat fmt kFmtGG, row = k / 256 blocks): whole, or part of its rows / inputs
    bool mat_gg(Dev& d, const std::string& pre, const Tensor& t, int ggt, x3::Mat& m, std::string& err, size_t* acct, int part,
                int np, bool cols) {
        if (t.shape.size() != 2 || t.shape[1] % 256 != 0) {
            err = pre + ": a " + t.dtype + " tensor that is not a matrix of whole blocks";
            return false;
        }
        int n = (int) t.shape[0], k = (int) t.shape[1];
        const size_t rb = gg::type_row_bytes(ggt, k), bb = rb / ((size_t) k / 256);
        std::vector<uint8_t> w((size_t) n * rb);
        if (t.bytes != w.size() || !pk.read(t, w.data())) {
            err = pre + ": read failed";
            return false;
        }
        if (np > 1) {
            const int whole = cols ? n : k, wd = whole / np;
            if (whole % np != 0 || wd % (cols ? 1 : 256) != 0) {
                err = pre + ": " + std::to_string(whole) + (cols ? " outputs" : " inputs") + " do not split in " + std::to_string(np) + " parts";
                return false;
            }
            std::vector<uint8_t> p;
            if (cols) {
                p.assign(w.begin() + (long) ((size_t) part * wd * rb), w.begin() + (long) ((size_t) (part + 1) * wd * rb));
                n = wd;
            } else {
                const size_t pb = (size_t) (wd / 256) * bb, off = (size_t) (part * wd / 256) * bb;
                p.resize((size_t) n * pb);
                for (size_t r = 0; r < (size_t) n; ++r) std::memcpy(p.data() + r * pb, w.data() + r * rb + off, pb);
                k = wd;
            }
            w.swap(p);
        }
        uint8_t* dp = nullptr;
        if (!dalloc(d, dp, w.size(), err, acct)) return false;
        cudaSetDevice(d.id);
        CK(cudaMemcpy(dp, w.data(), w.size(), cudaMemcpyHostToDevice));
        m = x3::Mat{};
        m.fmt = gg::kFmtGG;
        m.K = ggt;
        m.raw = dp;
        m.k = k;
        m.n = n;
        m.ld = (int) gg::type_row_bytes(ggt, k);
        return true;
    }
    // a GGUF's BF16 or Q8_0 matrix ([out][in]) onto the device, as mat() takes an EXL3 one: whole, or part `part` of its
    // columns (outputs: a range of its rows) or of its rows (inputs: a range of each row, whole Q8_0 blocks) - a
    // standalone [n][k] matrix
    bool mat_bf16(Dev& d, const std::string& pre, const Tensor& t, x3::Mat& m, std::string& err, size_t* acct, int part,
                  int np, bool cols) {
        const bool q8 = t.dtype == "Q8_0";
        // a ggml K-quant dense matrix (Q4_K / Q5_K / Q6_K: 256 values a block; the GEMV path of ds4_iq)
        const int ggt = (t.dtype == "Q4_K" || t.dtype == "Q5_K" || t.dtype == "Q6_K") ? gg_type_of(t.dtype) : -1;
        if (ggt > 0) return mat_gg(d, pre, t, ggt, m, err, acct, part, np, cols);
        if (t.shape.size() != 2 || (q8 && t.shape[1] % 32 != 0)) {
            err = pre + ": a " + t.dtype + " tensor that is not a matrix";
            return false;
        }
        int n = (int) t.shape[0], k = (int) t.shape[1];
        // a row's bytes, and the bytes of v of its values (BF16: 2 each; Q8_0: 34 every 32)
        auto bytes_of = [&](size_t v) { return q8 ? v / 32 * gg::kQ8Bytes : v * 2; };
        std::vector<uint8_t> w((size_t) n * bytes_of((size_t) k));
        if (t.bytes != w.size() || !pk.read(t, w.data())) {
            err = pre + ": read failed";
            return false;
        }
        if (np > 1) {
            const int whole = cols ? n : k, wd = whole / np;
            if (whole % np != 0 || wd % (q8 ? 32 : 8) != 0) {
                err = pre + ": " + std::to_string(whole) + (cols ? " outputs" : " inputs") + " do not split in " +
                      std::to_string(np) + " parts";
                return false;
            }
            const size_t rb = bytes_of((size_t) k);
            std::vector<uint8_t> p;
            if (cols) {
                p.assign(w.begin() + (long) ((size_t) part * wd * rb), w.begin() + (long) ((size_t) (part + 1) * wd * rb));
                n = wd;
            } else {
                const size_t pb = bytes_of((size_t) wd);
                p.resize((size_t) n * pb);
                for (size_t r = 0; r < (size_t) n; ++r)
                    std::memcpy(p.data() + r * pb, w.data() + r * rb + bytes_of((size_t) part * wd), pb);
                k = wd;
            }
            w.swap(p);
        }
        uint8_t* dp = nullptr;
        if (!dalloc(d, dp, w.size(), err, acct)) return false;
        cudaSetDevice(d.id);
        CK(cudaMemcpy(dp, w.data(), w.size(), cudaMemcpyHostToDevice));
        m = x3::Mat{};
        m.fmt = q8 ? gg::kFmtQ8_0 : gg::kFmtBF16;
        m.raw = dp;
        m.k = k;
        m.n = n;
        m.ld = q8 ? (int) bytes_of((size_t) k) : k;
        return true;
    }
    // a Q8_0 tensor's values as F32 (host)
    static std::vector<float> q8_values(const uint8_t* q, size_t n) {
        std::vector<float> v(n);
        for (size_t b = 0; b < n / 32; ++b) {
            const uint8_t* p = q + b * gg::kQ8Bytes;
            const float d = f16_to_f32((uint16_t) (p[0] | p[1] << 8));
            for (int j = 0; j < 32; ++j) v[b * 32 + j] = d * (float) (int8_t) p[2 + j];
        }
        return v;
    }
    // an EXL3 matrix onto the device: whole, or (np > 1) part `part` of its columns (cols) or of its rows - a standalone
    // matrix of its own (the rotations' 128-blocks stay whole)
    // (rot: split128's - an EXL3 matrix's uneven 128-block shares going round the members; a GGUF's BF16 / Q8_0 matrix
    // splits evenly, its groups only as wide as that goes)
    bool mat(Dev& d, const std::string& pre, x3::Mat& m, std::string& err, size_t* acct, int part = 0, int np = 1,
             bool cols = true, int rot = 0) {
        if (const Tensor* tw = pk.get(pre + ".weight"); tw != nullptr && (tw->dtype == "BF16" || tw->dtype == "Q8_0" || tw->dtype == "Q4_K" || tw->dtype == "Q5_K" || tw->dtype == "Q6_K"))
            return mat_bf16(d, pre, *tw, m, err, acct, part, np, cols);
        const Tensor* tr = pk.get(pre + ".trellis");
        const Tensor* su = pk.get(pre + ".suh");
        const Tensor* sv = pk.get(pre + ".svh");
        if (tr == nullptr || su == nullptr || sv == nullptr || tr->shape.size() != 3) {
            err = pre + ": not an EXL3 matrix in the pack";
            return false;
        }
        if (pk.get(pre + ".mul1") == nullptr) {
            err = pre + ": only the mul1 codebook is covered";
            return false;
        }
        m.k = (int) tr->shape[0] * 16;
        m.n = (int) tr->shape[1] * 16;
        m.K = (int) tr->shape[2] / 16;
        m.cb = 2;
        m.ld = 0;
        if (np > 1) {
            const int whole = cols ? m.n : m.k;
            int at = 0, w = 0;
            if (whole % 128 != 0 || whole / 128 < np) {
                err = pre + ": " + std::to_string(whole) + (cols ? " columns" : " rows") + " do not split in " +
                      std::to_string(np) + " parts of 128-blocks";
                return false;
            }
            split128(whole, np, part, at, w, rot);
            std::vector<uint8_t> t0(tr->bytes), u0(su->bytes), v0(sv->bytes);
            if (!pk.read(*tr, t0.data()) || !pk.read(*su, u0.data()) || !pk.read(*sv, v0.data())) {
                err = pre + ": read failed";
                return false;
            }
            const size_t tile = tr->bytes / ((size_t) (m.k / 16) * (m.n / 16));
            std::vector<uint8_t> t1, u1, v1;
            if (cols) {   // the tiles [c0 / 16, (c0 + w) / 16) of each 16-row slice, svh [c0, c0 + w)
                const size_t c0 = (size_t) at, nt = (size_t) m.n / 16, nth = (size_t) w / 16;
                t1.resize((size_t) (m.k / 16) * nth * tile);
                for (size_t sl = 0; sl < (size_t) m.k / 16; ++sl)
                    std::memcpy(t1.data() + sl * nth * tile, t0.data() + (sl * nt + c0 / 16) * tile, nth * tile);
                u1 = u0;
                v1.assign(v0.begin() + (long) (c0 * 2), v0.begin() + (long) ((c0 + w) * 2));
                m.n = w;
            } else {      // the 16-row slices [r0 / 16, (r0 + w) / 16), suh [r0, r0 + w)
                const size_t r0 = (size_t) at, row = (size_t) (m.n / 16) * tile;
                t1.assign(t0.begin() + (long) (r0 / 16 * row), t0.begin() + (long) ((r0 + w) / 16 * row));
                u1.assign(u0.begin() + (long) (r0 * 2), u0.begin() + (long) ((r0 + w) * 2));
                v1 = v0;
                m.k = w;
            }
            uint32_t* dt = nullptr;
            uint16_t *ds = nullptr, *dv = nullptr;
            if (!dalloc(d, dt, t1.size() / 4, err, acct) || !dalloc(d, ds, u1.size() / 2, err, acct) ||
                !dalloc(d, dv, v1.size() / 2, err, acct))
                return false;
            cudaSetDevice(d.id);
            CK(cudaMemcpy(dt, t1.data(), t1.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(ds, u1.data(), u1.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dv, v1.data(), v1.size(), cudaMemcpyHostToDevice));
            m.trellis = dt;
            m.suh = ds;
            m.svh = dv;
            return x3::supported(m) || (err = pre + ": a part not covered by the EXL3 kernels", false);
        }
        uint32_t* dt = nullptr;
        uint16_t *ds = nullptr, *dv = nullptr;
        if (!dalloc(d, dt, tr->bytes / 4, err, acct) || !dalloc(d, ds, su->bytes / 2, err, acct) ||
            !dalloc(d, dv, sv->bytes / 2, err, acct))
            return false;
        if (!upload(d, *tr, dt, err) || !upload(d, *su, ds, err) || !upload(d, *sv, dv, err)) return false;
        m.trellis = dt;
        m.suh = ds;
        m.svh = dv;
        if (!x3::supported(m)) {
            err = pre + ": k " + std::to_string(m.k) + " n " + std::to_string(m.n) + " K " + std::to_string(m.K) +
                  " not covered by the EXL3 kernels";
            return false;
        }
        return true;
    }
    // a float tensor (F32 / F16 / BF16) as F32 on the device
    bool f32(Dev& d, const std::string& name, float*& p, std::string& err, size_t* acct, size_t want = 0) {
        const Tensor* t = pk.get(name);
        if (t == nullptr) {
            err = name + ": not in the pack";
            return false;
        }
        std::vector<uint8_t> raw(t->bytes);
        if (!pk.read(*t, raw.data())) {
            err = name + ": read failed";
            return false;
        }
        size_t n = 0;
        std::vector<float> v;
        if (t->dtype == "F32") {
            n = t->bytes / 4;
            v.resize(n);
            std::memcpy(v.data(), raw.data(), t->bytes);
        } else if (t->dtype == "F16" || t->dtype == "BF16") {
            n = t->bytes / 2;
            v.resize(n);
            const uint16_t* s = (const uint16_t*) raw.data();
            for (size_t i = 0; i < n; ++i) v[i] = t->dtype == "F16" ? f16_to_f32(s[i]) : bf16_to_f32(s[i]);
        } else if (t->dtype == "Q8_0") {   // (a GGUF's router may come so)
            n = t->bytes / gg::kQ8Bytes * 32;
            v = q8_values(raw.data(), n);
        } else {
            err = name + ": dtype " + t->dtype + " is not a float";
            return false;
        }
        if (want != 0 && n != want) {
            err = name + ": " + std::to_string(n) + " values, expected " + std::to_string(want);
            return false;
        }
        if (!dalloc(d, p, n, err, acct)) return false;
        cudaSetDevice(d.id);
        CK(cudaMemcpy(p, v.data(), n * 4, cudaMemcpyHostToDevice));
        return true;
    }
    bool f16raw(Dev& d, const std::string& name, uint16_t*& p, std::string& err, size_t* acct) {
        const Tensor* t = pk.get(name);
        if (t != nullptr && (t->dtype == "F32" || t->dtype == "BF16" || t->dtype == "Q8_0")) {   // (a GGUF's)
            const size_t n = t->dtype == "Q8_0" ? t->bytes / gg::kQ8Bytes * 32 : t->bytes / (t->dtype == "F32" ? 4 : 2);
            std::vector<uint8_t> raw(t->bytes);
            if (!pk.read(*t, raw.data())) {
                err = name + ": read failed";
                return false;
            }
            const std::vector<float> q = t->dtype == "Q8_0" ? q8_values(raw.data(), n) : std::vector<float>();
            std::vector<uint16_t> h(n);
            for (size_t i = 0; i < n; ++i) {
                float v;
                if (t->dtype == "F32") std::memcpy(&v, raw.data() + i * 4, 4);
                else if (t->dtype == "Q8_0") v = q[i];
                else v = bf16_to_f32(((const uint16_t*) raw.data())[i]);
                h[i] = strata::kernels::f16_from_f32(v);
            }
            if (!dalloc(d, p, n, err, acct)) return false;
            cudaSetDevice(d.id);
            CK(cudaMemcpy(p, h.data(), n * 2, cudaMemcpyHostToDevice));
            return true;
        }
        if (t == nullptr || t->dtype != "F16") {
            err = name + ": not an F16 tensor in the pack";
            return false;
        }
        if (!dalloc(d, p, t->bytes / 2, err, acct)) return false;
        return upload(d, *t, p, err);
    }

    size_t mat_bytes(const std::string& pre) const {
        if (const Tensor* t = pk.get(pre + ".weight")) return align256(t->bytes);
        size_t b = 0;
        for (const char* s : {".trellis", ".suh", ".svh"})
            if (const Tensor* t = pk.get(pre + s)) b += align256(t->bytes);
        return b;
    }
    size_t tensor_bytes(const std::string& name) const {
        const Tensor* t = pk.get(name);
        return t ? align256(t->bytes * (t->dtype == "F32" ? 1 : 2)) : 0;
    }

    // ---- expert regions and blob layout
    // a GGUF's MXFP4 experts: the blob is gate | up | down ([out][in / 32] blocks each, as the file holds them), every
    // expert's matrix r at the same stride from expert 0's in one file (the GGUF's [n_exp][out][in] tensor)
    // (the other block types: lay.fmt stays kFmtMXFP4 - read in place as these - and lay.K[r] is role r's ggml type)
    static int gg_type_of(const std::string& dtype) {
        for (int t : {gg::kTypeMXFP4, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 145, 146, 224, 225, 226, 227})
            if (dtype == gg::type_name(t)) return t;
        return -1;
    }
    bool expert_layout_mx(Layer& l, std::string& err) {
        const std::string P = l.name + "ffn.experts.";
        const char* parts[3] = {"w1", "w3", "w2"};   // gate, up, down
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            const int n = r < 2 ? g.ff : g.dim, k = r < 2 ? g.dim : g.ff;
            const Tensor* t0 = pk.get(P + "0." + parts[r] + ".weight");
            const Tensor* t1 = l.n_exp > 1 ? pk.get(P + "1." + parts[r] + ".weight") : t0;
            const int ty = t0 != nullptr ? gg_type_of(t0->dtype) : -1;
            const size_t bytes = ty > 0 ? (size_t) n * gg::type_row_bytes(ty, k) : 0;
            if (t0 == nullptr || t1 == nullptr || bytes == 0 || t0->bytes != bytes) {
                err = P + "0." + parts[r] + ": not an MXFP4 or low-bit [" + std::to_string(n) + "][" + std::to_string(k) +
                      "] matrix the engine reads (" + (t0 != nullptr ? t0->dtype : std::string("missing")) + ")";
                return false;
            }
            l.mx_file[r] = t0->file;
            l.mx_off[r] = t0->off;
            l.mx_stride[r] = l.n_exp > 1 ? t1->off - t0->off : bytes;
            for (int e = 0; e < l.n_exp; ++e) {
                const Tensor* t = pk.get(P + std::to_string(e) + "." + parts[r] + ".weight");
                if (t == nullptr || t->dtype != t0->dtype || t->bytes != bytes || t->file != l.mx_file[r] ||
                    t->off != l.mx_off[r] + (uint64_t) e * l.mx_stride[r]) {
                    err = P + std::to_string(e) + "." + parts[r] + ": missing, or not where the layer's others are";
                    return false;
                }
            }
            l.lay.trellis[r] = off;
            l.lay.suh[r] = l.lay.svh[r] = off;
            l.lay.K[r] = ty;
            off += align256(bytes) + (store_gb() > 0 ? ExpertStore::kSlack : 0);   // (the store's aligned reads spill into the gaps)
        }
        l.lay.fmt = gg::kFmtMXFP4;
        if (!gg::mx_experts(l.lay) && (gg::expert_type(l.lay, 0) != gg::expert_type(l.lay, 1) ||
                                       gg::expert_type(l.lay, 0) == gg::kTypeMXFP4 ||
                                       gg::expert_type(l.lay, 2) == gg::kTypeMXFP4)) {
            err = P + ": gate / up / down of types " + gg::type_name(l.lay.K[0]) + " / " + gg::type_name(l.lay.K[1]) +
                  " / " + gg::type_name(l.lay.K[2]) + " (gate and up must match; MXFP4 does not mix with the others)";
            return false;
        }
        l.blob = off;
        // a type the device kernels do not take (none of the i-quants now: gg::type_on_device lists them): the layer stays on the CPU lane, where
        // ggml-cpu's row dots read it - one warning, then the placement code keeps it out of VRAM
        for (int r = 0; r < 3; ++r)
            if (!gg::type_on_device(l.lay.K[r])) {
                l.cpu_only = true;
                std::fprintf(stderr,
                             "ds4: layer %d %s is %s: no GPU kernel for it, its experts stay on the CPU lane "
                             "(ggml-cpu's rows; VRAM tier: none)\n", l.il, r == 2 ? "down" : "gate/up",
                             gg::type_name(l.lay.K[r]));
                break;
            }
        return true;
    }
    const uint8_t* mx_mat(const Layer& l, int e, int r) const {   // expert e's matrix r: its RAM-tier slot, else the file's
        if (store) {
            const int s = store->find(l.il, e);
            if (s >= 0) return store->mat(s, r);
        }
        if (!l.ram.empty() && l.ram[(size_t) e] >= 0)   // (a copy with NH > 1: part 0's rows only - whole: read_mx)
            return copy_mode ? ram_part[(size_t) l.ram[(size_t) e]][0] + cp_lay[(size_t) l.il][(size_t) r]
                             : ram_slots[(size_t) l.ram[(size_t) e]] + l.lay.trellis[r];
        return pk.at(l.mx_file[r], l.mx_off[r] + (uint64_t) e * l.mx_stride[r]);
    }
    size_t mx_part_bytes(const Layer& l, int r) const {   // (its type's)
        return r < 2 ? (size_t) g.ff * gg::type_row_bytes(gg::expert_type(l.lay, r), g.dim)
                     : (size_t) g.dim * gg::type_row_bytes(gg::expert_type(l.lay, 2), g.ff);
    }
    // (copy_mode) matrix r's rows, and the rows of one NUMA part of it
    int mx_n(int r) const { return r < 2 ? g.ff : g.dim; }
    size_t mx_row_b(const Layer& l, int r) const { return mx_part_bytes(l, r) / (size_t) mx_n(r); }
    // part X's rows [X n / NH, ..) of expert e's matrix r: its copy's part on node X, else where the file has them
    const uint8_t* mx_part_ptr(const Layer& l, int e, int r, int X) const {
        if (copy_mode && l.ram[(size_t) e] >= 0)
            return ram_part[(size_t) l.ram[(size_t) e]][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r];
        return mx_mat(l, e, r) + (size_t) X * (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
    }
    // a copy's NH parts -> the blob, and back (each matrix's rows of part X at X n / NH rows into the blob's matrix)
    void ram_to_blob(const Layer& l, int s, uint8_t* blob) const {
        for (int r = 0; r < 3; ++r) {
            const size_t nb = (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
            for (int X = 0; X < NH; ++X)
                std::memcpy(blob + l.lay.trellis[r] + (size_t) X * nb, ram_part[(size_t) s][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r], nb);
        }
    }
    void blob_to_ram(const Layer& l, const uint8_t* blob, int s) const {
        for (int r = 0; r < 3; ++r) {
            const size_t nb = (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
            for (int X = 0; X < NH; ++X)
                std::memcpy(ram_part[(size_t) s][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r], blob + l.lay.trellis[r] + (size_t) X * nb, nb);
        }
    }
    // expert e (from the file) -> its blob; and member m of nm's part of a blob (gate / up rows [m n_ff / nm, ..), down
    // inputs of the same range: in each row the blocks [m n_ff / nm / 32, ..)) and back
    bool read_mx(const Layer& l, int e, uint8_t* blob) const {   // (from its RAM-tier slot or copy when it has one)
        if (!l.ram.empty() && l.ram[(size_t) e] >= 0) {
            if (copy_mode) ram_to_blob(l, l.ram[(size_t) e], blob);
            else std::memcpy(blob, ram_slots[(size_t) l.ram[(size_t) e]], l.blob);
            return true;
        }
        return read_mx_file(l, e, blob);
    }
    bool read_mx_file(const Layer& l, int e, uint8_t* blob) const {   // (the movers: never a slot that may change)
        for (int r = 0; r < 3; ++r)
            if (!pk.read_at(l.mx_file[r], l.mx_off[r] + (uint64_t) e * l.mx_stride[r], mx_part_bytes(l, r),
                            blob + l.lay.trellis[r]))
                return false;
        return true;
    }
    void mx_blob_to_half(const Layer& l, const uint8_t* blob, int m, uint8_t* dst) const {
        const int ffh = g.ff / l.nm;
        const size_t gr = gg::type_row_bytes(gg::expert_type(l.lay, 0), g.dim),
                     dr = gg::type_row_bytes(gg::expert_type(l.lay, 2), g.ff),
                     dh = gg::type_row_bytes(gg::expert_type(l.lay, 2), ffh);
        for (int r = 0; r < 2; ++r)
            std::memcpy(dst + l.hlay.trellis[r], blob + l.lay.trellis[r] + (size_t) m * ffh * gr, (size_t) ffh * gr);
        for (size_t row = 0; row < (size_t) g.dim; ++row)
            std::memcpy(dst + l.hlay.trellis[2] + row * dh, blob + l.lay.trellis[2] + row * dr + (size_t) m * dh, dh);
    }
    void mx_half_to_blob(const Layer& l, const uint8_t* src, int m, uint8_t* blob) const {
        const int ffh = g.ff / l.nm;
        const size_t gr = gg::type_row_bytes(gg::expert_type(l.lay, 0), g.dim),
                     dr = gg::type_row_bytes(gg::expert_type(l.lay, 2), g.ff),
                     dh = gg::type_row_bytes(gg::expert_type(l.lay, 2), ffh);
        for (int r = 0; r < 2; ++r)
            std::memcpy(blob + l.lay.trellis[r] + (size_t) m * ffh * gr, src + l.hlay.trellis[r], (size_t) ffh * gr);
        for (size_t row = 0; row < (size_t) g.dim; ++row)
            std::memcpy(blob + l.lay.trellis[2] + row * dr + (size_t) m * dh, src + l.hlay.trellis[2] + row * dh, dh);
    }

    bool expert_layout(Layer& l, std::string& err) {
        if (const Tensor* t = pk.get(l.name + "ffn.experts.0.w1.weight"); t != nullptr && gg_type_of(t->dtype) > 0)
            return expert_layout_mx(l, err);
        const std::string P = l.name + "ffn.experts.";
        const char* parts[3] = {"w1", "w3", "w2"};   // gate, up, down
        const char* pieces[3] = {".trellis", ".suh", ".svh"};
        l.region_off.resize((size_t) l.n_exp);
        l.region_bytes.resize((size_t) l.n_exp);
        l.region_file.resize((size_t) l.n_exp);
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            const Tensor* tr = pk.get(P + "0." + parts[r] + ".trellis");
            if (tr == nullptr || pk.get(P + "0." + parts[r] + ".mul1") == nullptr) {
                err = P + "0." + parts[r] + ": not an EXL3 (mul1) matrix";
                return false;
            }
            l.lay.K[r] = (int) tr->shape[2] / 16;
            l.lay.trellis[r] = off;
            off += align256(tr->bytes);
            l.lay.suh[r] = off;
            off += align256(pk.get(P + "0." + parts[r] + ".suh")->bytes);
            l.lay.svh[r] = off;
            off += align256(pk.get(P + "0." + parts[r] + ".svh")->bytes);
        }
        l.lay.cb = 2;
        l.blob = off;
        for (int e = 0; e < l.n_exp; ++e) {
            const std::string E = P + std::to_string(e) + ".";
            uint64_t lo = UINT64_MAX, hi = 0;
            int file = -1;
            for (int r = 0; r < 3; ++r)
                for (int q = 0; q < 3; ++q) {
                    const Tensor* t = pk.get(E + parts[r] + pieces[q]);
                    if (t == nullptr) {
                        err = E + parts[r] + pieces[q] + ": missing";
                        return false;
                    }
                    if (file < 0) file = t->file;
                    if (t->file != file) {
                        err = E + ": split across files";
                        return false;
                    }
                    lo = std::min(lo, t->off);
                    hi = std::max(hi, t->off + t->bytes);
                    if (e == 0) l.piece_bytes[r][q] = t->bytes;
                    else if (t->bytes != l.piece_bytes[r][q]) {
                        err = E + parts[r] + ": not the shape of expert 0's";
                        return false;
                    }
                }
            l.region_off[(size_t) e] = lo;
            l.region_bytes[(size_t) e] = hi - lo;
            l.region_file[(size_t) e] = file;
            for (int r = 0; r < 3; ++r)
                for (int q = 0; q < 3; ++q) {
                    const size_t src = pk.get(E + parts[r] + pieces[q])->off - lo;
                    if (e == 0) l.piece_src[r][q] = src;
                    else if (src != l.piece_src[r][q]) {
                        err = E + ": the pieces lie differently from expert 0's";
                        return false;
                    }
                }
        }
        return true;
    }
    // the region of expert e (read from the file) -> its blob
    void repack(const Layer& l, const uint8_t* region, uint8_t* blob) const {
        for (int r = 0; r < 3; ++r) {
            const size_t dst[3] = {l.lay.trellis[r], l.lay.suh[r], l.lay.svh[r]};
            for (int q = 0; q < 3; ++q) std::memcpy(blob + dst[q], region + l.piece_src[r][q], l.piece_bytes[r][q]);
        }
    }

    // the region of expert e -> its CPU-lane part X (columns [X n / NH, (X + 1) n / NH) of each matrix)
    void repack_part(const Layer& l, const uint8_t* region, int X, uint8_t* dst) const {
        for (int r = 0; r < 3; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r];
            const int k = l.hl.k[r], nt = l.hl.n[r] * NH / 16, nth = l.hl.n[r] / 16;
            const uint8_t* tr = region + l.piece_src[r][0];
            for (int sl = 0; sl < k / 16; ++sl)
                std::memcpy(dst + l.hl.trellis[r] + (size_t) sl * nth * tile, tr + ((size_t) sl * nt + (size_t) X * nth) * tile,
                            (size_t) nth * tile);
            std::memcpy(dst + l.hl.suh[r], region + l.piece_src[r][1], (size_t) k * 2);
            std::memcpy(dst + l.hl.svh[r], region + l.piece_src[r][2] + (size_t) X * l.hl.n[r] * 2, (size_t) l.hl.n[r] * 2);
        }
    }

    // Explicit HELD maintenance admission. This never impersonates a live
    // prefill phase: phase flags, generation and work-delta counters stay put.
    bool gpu_misses_recovery_context() const {
        return gpumiss::recovery_lease(gpu_miss_direct_enabled,gpu_misses_enabled,
            gpu_misses_recovery_preparing,gpu_misses_recovery_lease,cache_epoch_machine.serial,
            cache_epoch_machine.phase==epochpolicy::Phase::held,chunk_major_phase_pending,
            prefill_phase_pending,chunk_major_phase_error.empty());
    }

    // The native GEN path always feeds at least its final input token. Called
    // only at that serialized request boundary, before token/checkpoint work.
    bool gpu_misses_recover_decode(int remaining,std::string& err) {
        const bool long_rearm=repeatpolicy::eligible(prefill_repeat_rearm,chunk_major_phase_pending,
                                                   remaining,prefill_repeat_min_tokens);
        if(!gpumiss::recovery_request(gpu_miss_direct_enabled,gpu_misses_recovery_needed,
                                     chunk_major_phase_pending,remaining,long_rearm))return true;
#if defined(STRATA_USE_HIP)
        err="direct decode recovery requires CUDA";return false;
#else
        auto hard=[&](const std::string& why){
            gpu_misses_decode_pins=false;registered_copy.published=false;
            st.gpu_miss_retained_ranges=st.gpu_miss_retained_bytes=0;
            const std::string first=chunk_major_phase_error.empty()?"GPU miss decode recovery: "+why:chunk_major_phase_error;
            if(cache_epoch_machine.phase==epochpolicy::Phase::failed){err=first;return false;}
            return cache_epoch_terminal(first,err);
        };
        unsigned lease=0;bool released=false;
        try {
            if(gpu_misses_retained_ready()){gpu_misses_recovery_needed=false;return true;}
            if(gpu_misses_recovery_preparing||prefill_repeat_preparing||!chunk_major_phase_error.empty()||
               !cache_epoch_enabled||!copied_ram_domains_published||!record_copied_ram_domains||
               devs.size()!=1||chunks.size()!=1||!chunks[0]||NH!=1||!copy_mode||store||ram_tier||
               prefill_phase_pending||registered_copy.published||!registered_copy.held.empty()||
               !pk.gguf||!file_experts||!o.expert_cache||o.max_context!=1048576)
                return hard("unsupported consumed/unregistered state");
            Dev& d=devs[0];Chunk& c=*chunks[0];
            if(!gpumiss::recovery_geometry(c.C,c.A,c.AI,chunk,c.restore_C,c.restore_A,c.restore_AI)||
               c.cold_f16||c.cold_f16_y||c.cold_f16_meta_host||c.cold_f16_meta_dev||c.cold_f16_workspace||
               c.cold_f16_blas||c.eg_weight_cache||c.dense_projection_cache)
                return hard("unrecorded geometry or inactive phase assets remain");
            if(cancel&&cancel()){halted=true;err="stopped before GPU miss decode recovery";return false;}
            struct Preparing {Impl* p;Preparing(Impl* v):p(v){p->gpu_misses_recovery_preparing=true;}
                ~Preparing(){p->gpu_misses_recovery_preparing=false;p->gpu_misses_recovery_lease=0;}
            } preparing{this};
            const double started=now_ms();const CacheProbeState before=cache_probe_state();
            const uint64_t attempt=++gpu_misses_recovery_attempts;
            auto same=[](const CacheProbeState& a,const CacheProbeState& b){return a.position==b.position&&
                a.tokens==b.tokens&&a.ctokens==b.ctokens&&a.issued==b.issued&&a.token_hash==b.token_hash&&
                a.checkpoint_hash==b.checkpoint_hash;};
            std::string why;
            if(!begin_cache_worker_epoch(lease,60000,why)) {
                if(cache_epoch_machine.phase==epochpolicy::Phase::failed||!chunk_major_phase_error.empty())return hard(why);
                const bool stopped=cancel&&cancel();
                std::fprintf(stderr,"ds4: GPU_MISS_RECOVERY declined: attempt=%llu epoch=0 reason=acquire stopped=%d registration_ranges=%zu registration_bytes=%zu; current layout retained\n",
                    (unsigned long long)attempt,stopped?1:0,registered_copy.held.size(),registered_copy.bytes());
                if(stopped){halted=true;err="stopped while acquiring GPU miss decode recovery";return false;}
                return true;
            }
            gpu_misses_recovery_lease=lease;
            struct LeaseGuard {Impl* p;unsigned lease;bool& released;
                ~LeaseGuard(){if(!released&&p->cache_epoch_machine.serial==lease&&
                    p->cache_epoch_machine.phase==epochpolicy::Phase::held)try{
                    std::string why;p->cache_epoch_terminal("abandoned GPU miss decode recovery",why);
                }catch(...){}}
            } guard{this,lease,released};
            if(!gpu_misses_recovery_context()||gpu_misses_decode_pins||registered_copy.published||!registered_copy.held.empty())
                return hard("epoch did not prove exact unregistered ownership");
            const CacheProbeState held=cache_probe_state();
            if(!same(before,held))return hard("acquire changed tokens/checkpoints/issued");
            std::fprintf(stderr,"ds4: GPU_MISS_RECOVERY begin: attempt=%llu epoch=%u issued=%u C=%d A=%d AI=%d pool_bytes=%zu registration_ranges=0 registration_bytes=0\n",
                (unsigned long long)attempt,lease,seq_issued,c.C,c.A,c.AI,d.experts);
            auto abort=[&](bool stopped,const char* reason)->bool {
                // No new GPU/CPU producer is allowed before the request feed.
                // The actual retirement helper drains CS, d.st and c.cs even
                // after a failed enqueue/event operation, and owns failures.
                if(!gpu_misses_retire_pins(stopped?"recovery-stop":"recovery-abort",why))return hard(why);
                if(registered_copy.published||!registered_copy.held.empty()||gpu_misses_decode_pins)
                    return hard("abort registration did not physically reach zero");
                const CacheProbeState post=cache_probe_state();
                if(!same(held,post))return hard("abort changed tokens/checkpoints/issued");
                if(!released){if(!end_cache_worker_epoch(lease,why))return hard(why);released=true;}
                const CacheProbeState after=cache_probe_state();
                if(!same(post,after)||post.map_hash!=after.map_hash)return hard("abort release changed completed maps");
                std::fprintf(stderr,"ds4: GPU_MISS_RECOVERY declined: attempt=%llu epoch=%u reason=%s stopped=%d C=%d A=%d AI=%d pool_bytes=%zu registration_ranges=0 registration_bytes=0; valid current layout retained\n",
                    (unsigned long long)attempt,lease,reason,stopped?1:0,c.C,c.A,c.AI,d.experts);
                if(stopped){halted=true;err="stopped during GPU miss decode recovery";return false;}
                return true;
            };
            if(cancel&&cancel())return abort(true,"before-compact");
            const int oldC=c.C,oldA=c.A;const size_t old_pool=d.experts;
            const double compact_started=now_ms();
            // An early rearm Stop may leave the small workspace with the old
            // large decode pool. Reuse the checked current-map shrink before
            // regrowing with registration headroom; never infer initial IDs.
            if(gpumiss::recovery_pool_needs_shrink(d.experts,gpu_misses_last_decode_pool)) {
                const size_t pool_before=d.experts;
                bool stopped=false;
                const bool shrunk=shrink_copied_expert_pool(lease,stopped,why);
                if(cache_epoch_machine.phase==epochpolicy::Phase::failed||!chunk_major_phase_error.empty())return hard(why);
                std::fprintf(stderr,"ds4: GPU_MISS_RECOVERY_RESERVE attempt=%llu epoch=%u pool_before_bytes=%zu pool_after_bytes=%zu target_cap_bytes=%zu reserve_bytes=%zu shrink_completed=%d stopped=%d\n",
                    (unsigned long long)attempt,lease,pool_before,d.experts,
                    gpumiss::recovery_pool_cap(gpu_misses_last_decode_pool),
                    gpumiss::recovery_registration_reserve,shrunk?1:0,stopped?1:0);
                if(stopped||(cancel&&cancel()))return abort(true,"headroom-shrink");
                if(!shrunk)return abort(false,"headroom-shrink-unavailable");
                if(gpu_misses_last_decode_pool) {
                    if(gpumiss::recovery_pool_needs_shrink(d.experts,gpu_misses_last_decode_pool))
                        return abort(false,"headroom-shrink-unavailable");
                } else {
                    // First-ever phase cancellation has no normal compact
                    // capture. Require the actual immutable-domain targets,
                    // exactly as the existing long-rearm coordinator does.
                    ramdomain::Inventory inventory;
                    if(!inspect_copied_ram_inventory(inventory,why))return hard(why);
                    bool attained=inventory.free_holes==0&&!copied_ram_domains.empty();
                    for(const auto& domain:copied_ram_domains) {
                        const Layer& l=L[(size_t)domain.layer];copiedshrink::Plan target;
                        if(!copiedshrink::plan(domain,l.slot,l.n_slots,l.spare_vram,l.ram,copy_spare[(size_t)l.il],
                            inventory.layers[(size_t)l.il].free_slots,unit(l),target,why))return hard(why);
                        attained=attained&&l.n_slots==target.next_slots&&inventory.layers[(size_t)l.il].free_slots.empty();
                    }
                    if(!attained)return abort(false,"initial-target-shrink-unavailable");
                }
            }
            if(!finish_chunk_major_phase(why,true,true)) {
                if(!chunk_major_phase_error.empty()||cache_epoch_machine.phase==epochpolicy::Phase::failed)return hard(why);
                if(halted||(cancel&&cancel()))return abort(true,"compact");
                return hard(why);
            }
            const double compact_ms=now_ms()-compact_started;
            if(cancel&&cancel())return abort(true,"after-compact");
            if(c.C!=1024||c.A!=1024||chunk!=1024)return abort(false,"compact-unavailable");
            const CacheProbeState compact=cache_probe_state();
            if(!same(held,compact))return hard("compact changed tokens/checkpoints/issued");
            std::fprintf(stderr,"ds4: GPU_MISS_RECOVERY compact: attempt=%llu epoch=%u C_before=%d A_before=%d C_after=%d A_after=%d AI=%d pool_before_bytes=%zu pool_after_bytes=%zu last_healthy_pool_bytes=%zu compact_ms=%.3f\n",
                (unsigned long long)attempt,lease,oldC,oldA,c.C,c.A,c.AI,old_pool,d.experts,gpu_misses_last_decode_pool,compact_ms);
            const double registration_started=now_ms();
            if(!register_copied_ram_for_prefill(why))return hard(why);
            const double registration_ms=now_ms()-registration_started;
            if(cancel&&cancel())return abort(true,"registration");
            if(!registered_copy.published||registered_copy.held.empty())return abort(false,"registration-unavailable");
            const CacheProbeState registered=cache_probe_state();
            if(!same(compact,registered)||compact.map_hash!=registered.map_hash)return hard("registration changed completed state");
            if(!end_cache_worker_epoch(lease,why))return hard(why);
            released=true;
            const CacheProbeState after=cache_probe_state();
            if(!same(registered,after)||registered.map_hash!=after.map_hash)return hard("release changed completed state");
            if(cancel&&cancel())return abort(true,"before-publication");
            // Commit only after checked registration, exact lease release and
            // late Stop. A later soft Stop may keep this healthy DECODE owner.
            gpu_misses_decode_pins=true;
            if(!gpu_misses_retained_ready())return hard("DECODE publication refused");
            st.gpu_miss_retained_ranges=(int64_t)registered_copy.held.size();
            st.gpu_miss_retained_bytes=(int64_t)registered_copy.bytes();
            // Preserve the last NORMAL compact target. Repeated recovery
            // must not subtract another reserve from a prior recovery pool.
            gpu_misses_recovery_needed=false;
            const auto fingerprint=gpu_misses_pin_fingerprint();
            std::fprintf(stderr,"ds4: GPU_MISS_RECOVERY ready: attempt=%llu epoch=%u issued=%u C=%d A=%d AI=%d pool_bytes=%zu ranges=%zu bytes=%zu fingerprint=%016llx tokens_unchanged=1 checkpoints_unchanged=1 sequence_unchanged=1 streams_drained=1 compact_ms=%.3f registration_ms=%.3f elapsed_ms=%.3f\n",
                (unsigned long long)attempt,lease,seq_issued,c.C,c.A,c.AI,d.experts,registered_copy.held.size(),
                registered_copy.bytes(),(unsigned long long)fingerprint,compact_ms,registration_ms,now_ms()-started);
            std::fprintf(stderr,"ds4: GPU_MISS_PIN retained: context=DECODE ranges=%zu bytes=%zu fingerprint=%016llx epoch=%u issued=%u streams_drained=1\n",
                registered_copy.held.size(),registered_copy.bytes(),(unsigned long long)fingerprint,lease,seq_issued);
            return true;
        }catch(...){return hard("exception during tracked request-boundary recovery");}
#endif
    }

    // SAME physical registration ledger, explicit consumed-phase decode lifetime.
    // The owner ranges do not move; the CURRENT expert -> RAM slot does.
    bool gpu_misses_retained_ready() const {
        return gpu_miss_direct_enabled&&gpu_misses_enabled&&gpu_misses_decode_pins&&
            !chunk_major_phase_pending&&!prefill_phase_pending&&chunk_major_phase_error.empty()&&
            registered_copy.published&&!registered_copy.held.empty()&&copy_mode&&NH==1&&!store&&
            devs.size()==1&&copied_ram_domains_published;
    }
    uint64_t gpu_misses_pin_fingerprint() const {
        uint64_t h=14695981039346656037ull;
        for(const auto& range:registered_copy.held)for(uint64_t value:{(uint64_t)range.base,(uint64_t)range.bytes})
            for(int shift=0;shift<64;shift+=8){h^=(value>>shift)&255u;h*=1099511628211ull;}
        return h;
    }
    const uint8_t* gpu_misses_direct_source(const Layer& l,int e) const {
        if(!gpu_misses_retained_ready()||e<0||e>=l.n_exp||l.nm!=1||l.dev!=0||
           (size_t)l.il>=cp_lay.size()||(size_t)e>=l.ram.size())return nullptr;
        const int slot=l.ram[(size_t)e];
        if(slot<0||(size_t)slot>=ram_part.size()||!ram_part[(size_t)slot][0]||ram_part[(size_t)slot][1])return nullptr;
        const uint8_t* source=ram_part[(size_t)slot][0];
        const auto found=std::find_if(copied_ram_domains.begin(),copied_ram_domains.end(),
                                    [&](const ramdomain::Domain& x){return x.layer==l.il;});
        if(found==copied_ram_domains.end())return nullptr;
        const auto& domain=*found;size_t gpu_offsets[3],role_bytes[3];
        if(domain.arena_index>=ram_chunks.size()||domain.arena_index>=ram_chunk_bytes.size()||
           domain.base!=(uintptr_t)ram_chunks[domain.arena_index]||domain.arena_bytes!=ram_chunk_bytes[domain.arena_index]||
           domain.part_stride!=cp_lay[(size_t)l.il][3]||domain.offsets!=cp_lay[(size_t)l.il])return nullptr;
        for(int r=0;r<3;++r){gpu_offsets[r]=l.lay.trellis[r];role_bytes[r]=mx_part_bytes(l,r);
            if(domain.role_bytes[(size_t)r]!=role_bytes[r])return nullptr;}
        if(!gpumiss::whole_slot(domain.base,domain.arena_bytes,domain.first_slot,domain.slot_count,
            domain.part_stride,(size_t)slot,(uintptr_t)source,l.blob,cp_lay[(size_t)l.il].data(),gpu_offsets,role_bytes)||
           !registered_copy.covers((uintptr_t)source,l.blob))return nullptr;
        return source;
    }
    // Normal finish/epoch callers already prove serialized idle workers. Errors
    // may retain uncertain owners, but never unregister while a DMA is unproven.
    bool gpu_misses_retire_pins(const char* reason,std::string& err) {
        if(registered_copy.held.empty()) {
            gpu_misses_decode_pins=false;st.gpu_miss_retained_ranges=st.gpu_miss_retained_bytes=0;return true;
        }
        const size_t before_ranges=registered_copy.held.size(),before_bytes=registered_copy.bytes();
        const uint64_t fingerprint=gpu_misses_pin_fingerprint();
        gpu_misses_decode_pins=false;st.gpu_miss_retained_ranges=st.gpu_miss_retained_bytes=0;
        for(const auto& worker:workers)if(!worker->idle()){err="retained-pin retirement requires idle prompt workers";return false;}
        if(devs.size()!=1||chunks.size()!=1||!chunks[0]||cudaSetDevice(devs[0].id)!=cudaSuccess){
            err="retained-pin retirement geometry/device";return false;}
        const cudaError_t copy=gpu_misses_cs?cudaStreamSynchronize(gpu_misses_cs):cudaSuccess;
        const cudaError_t compute=cudaStreamSynchronize(devs[0].st);
        const cudaError_t flow=cudaStreamSynchronize(chunks[0]->cs);
        if(copy!=cudaSuccess||compute!=cudaSuccess||flow!=cudaSuccess){err="retained-pin retirement drain unproven";return false;}
        if(!retire_registered_copy_ram(err)) {
            std::fprintf(stderr,"ds4: GPU_MISS_PIN retirement_failed: reason=%s context=TERMINAL ranges_before=%zu bytes_before=%zu fingerprint=%016llx ranges_after=%zu bytes_after=%zu epoch=%u issued=%u streams_drained=1\n",
                reason,before_ranges,before_bytes,(unsigned long long)fingerprint,registered_copy.held.size(),registered_copy.bytes(),
                cache_epoch_machine.serial,seq_issued);return false;
        }
        if(registered_copy.published||!registered_copy.held.empty()){err="retained-pin retirement did not reach zero";return false;}
        std::fprintf(stderr,"ds4: GPU_MISS_PIN retired: reason=%s context=NONE ranges_before=%zu bytes_before=%zu fingerprint=%016llx ranges_after=0 bytes_after=0 epoch=%u issued=%u streams_drained=1\n",
            reason,before_ranges,before_bytes,(unsigned long long)fingerprint,cache_epoch_machine.serial,seq_issued);
        return true;
    }

    // Actual shared staging adapter; fixtures extract this body verbatim.
    // Caller has synchronized the current route event on d.st. All earlier GPU
    // consumers are finished. Copies use private cs; d.st waits ready before expert use.
    bool gpu_misses_prepare(Layer& l,Dev& d,int T,const int* ids,bool& used,std::string& err) {
        used=false;
        if(!gpumiss::window(gpu_misses_enabled,T))return true;
        if(gpu_misses_decode_pins&&!gpu_misses_retained_ready())
            return cache_epoch_terminal("stale decode pin context before staging",err);
        if(!copy_mode||NH!=1||devs.size()!=1||l.nm!=1||l.dev!=0||d.id!=devs[0].id||l.cpu_only||
           l.lay.fmt!=gg::kFmtMXFP4||l.k_exp<1||l.k_exp>gpumiss::max_picks||
           !gpu_misses_cs||!gpu_misses_ready||!gpu_misses_stage||!gpu_misses_host[0]||!gpu_misses_host[1]||!gpu_misses_plan_pin||
           l.blob>gpu_misses_stride||l.slot.size()!=(size_t)l.n_exp||l.ram.size()!=(size_t)l.n_exp) {
            ++st.gpu_miss_fallbacks;return true;
        }
        gpumiss::Plan plan;
        const auto admission=gpumiss::plan(T,l.k_exp,l.n_exp,ids,l.slot.data(),l.ram.data(),plan);
        if(admission==gpumiss::Admission::unavailable){++st.gpu_miss_fallbacks;return true;}
        auto hard=[&](const std::string& why){return cache_epoch_terminal("GPU miss staging: "+why,err);};
        if(admission==gpumiss::Admission::invalid)return hard("invalid routed expert");
        size_t role_end=0;
        for(int r=0;r<3;++r) {
            const size_t n=mx_part_bytes(l,r),off=l.lay.trellis[r];
            if(!gg::type_on_device(gg::expert_type(l.lay,r))||off<role_end||off>l.blob||n>l.blob-off)
                return hard("unsupported or out-of-bounds compressed role");
            role_end=off+n;
        }
        for(int t=0;t<T;++t)for(int i=0;i<l.k_exp;++i) {
            const int e=ids[t*gpumiss::physical_stride+i],v=l.slot[(size_t)e];
            if(v>=0&&(!l.vram_base||v>=l.n_slots||!unit(l)))return hard("invalid resident slot");
        }
        // Validate the complete borrowed-source plan before queuing any DMA.
        for(int j=0;j<plan.count;++j)for(int r=0;r<3;++r)
            if(!mx_mat(l,plan.unique[j],r))return hard("missing copied-RAM role");
        for(int j=0;j<plan.count;++j) {
            const int slot=j&1,e=plan.unique[j];
            if(const uint8_t* source=gpu_misses_direct_source(l,e)) {
                const double copy_started=now_ms();
                const cudaError_t status=cudaMemcpyAsync(gpu_misses_stage+(size_t)j*gpu_misses_stride,
                    source,l.blob,cudaMemcpyHostToDevice,gpu_misses_cs);
                st.gpu_miss_h2d_call_ms+=now_ms()-copy_started;
                if(status!=cudaSuccess)return hard("registered whole-slot H2D");
                ++st.gpu_miss_direct_unique;st.gpu_miss_direct_bytes+=(int64_t)l.blob;
                st.gpu_miss_bytes+=(int64_t)l.blob;continue;
            }
            if(gpu_misses_busy[slot]) {
                const double started=now_ms();
                const cudaError_t status=cudaEventSynchronize(gpu_misses_done[slot]);
                st.gpu_miss_bounce_wait_ms+=now_ms()-started;
                if(status!=cudaSuccess)return hard("pinned bounce reuse completion");
            }
            const double pack_started=now_ms();
            // Keep padding deterministic without another full expert-sized RAM write.
            size_t end=0;
            for(int r=0;r<3;++r) {
                const size_t off=l.lay.trellis[r],n=mx_part_bytes(l,r);
                std::memset(gpu_misses_host[slot]+end,0,off-end);
                std::memcpy(gpu_misses_host[slot]+off,mx_mat(l,e,r),n);end=off+n;
            }
            std::memset(gpu_misses_host[slot]+end,0,l.blob-end);
            st.gpu_miss_pack_ms+=now_ms()-pack_started;
            const double copy_started=now_ms();
            const cudaError_t copy_status=cudaMemcpyAsync(gpu_misses_stage+(size_t)j*gpu_misses_stride,
                gpu_misses_host[slot],l.blob,cudaMemcpyHostToDevice,gpu_misses_cs);
            st.gpu_miss_h2d_call_ms+=now_ms()-copy_started;
            if(copy_status!=cudaSuccess)return hard("compressed H2D");
            if(cudaEventRecord(gpu_misses_done[slot],gpu_misses_cs)!=cudaSuccess)return hard("bounce completion event");
            gpu_misses_busy[slot]=true;
            st.gpu_miss_bytes+=(int64_t)l.blob;
            if(gpu_miss_direct_enabled){++st.gpu_miss_bounce_unique;st.gpu_miss_bounce_bytes+=(int64_t)l.blob;}
        }
        // The copy stream never waits on current d.st work: the caller already
        // synchronized d.ev. This ready event creates only a compute-side wait.
        if(cudaEventRecord(gpu_misses_ready,gpu_misses_cs)!=cudaSuccess)return hard("stage ready event");
        for(int i=0;i<T*gpumiss::physical_stride;++i)gpu_misses_plan_pin[i]=0;
        for(int t=0;t<T;++t)for(int i=0;i<l.k_exp;++i) {
            const int at=t*gpumiss::physical_stride+i,e=ids[at],staged=plan.pick[at];
            const uint8_t* raw=staged>=0?gpu_misses_stage+(size_t)staged*gpu_misses_stride:
                l.vram_base+(size_t)l.slot[(size_t)e]*unit(l);
            gpu_misses_plan_pin[at]=(unsigned long long)raw;
        }
        if(cudaMemcpyAsync(d.plan_ptr,gpu_misses_plan_pin,(size_t)T*gpumiss::physical_stride*8,
                           cudaMemcpyHostToDevice,d.st)!=cudaSuccess)return hard("window plan H2D");
        if(cudaStreamWaitEvent(d.st,gpu_misses_ready,0)!=cudaSuccess)return hard("compute wait for staged blobs");
        ++st.gpu_miss_windows;st.gpu_miss_unique+=plan.count;st.gpu_miss_entries+=plan.entries;
        used=true;return true;
    }

    // ---- loading
    bool load(const Options& opt, std::string& err) {
        if (const char* v = std::getenv("DS4_DECODE_FUSION")) {
            if (std::strcmp(v, "0") != 0 && std::strcmp(v, "1") != 0) {
                err = "DS4_DECODE_FUSION must be 0 or 1"; return false;
            }
            decode_fusion = v[0] == '1';
        }
        if(!gpumiss::parse(std::getenv("DS4_GPU_MISSES"),gpu_misses_enabled)) {
            err="DS4_GPU_MISSES must be exactly 0 or 1";return false;
        }
#if defined(STRATA_USE_HIP)
        if(gpu_misses_enabled){err="DS4_GPU_MISSES requires CUDA";return false;}
#endif
        if(!gpumiss::parse(std::getenv("DS4_GPU_MISS_DIRECT"),gpu_miss_direct_enabled)||
           (gpu_miss_direct_enabled&&!gpu_misses_enabled)) {
            err="DS4_GPU_MISS_DIRECT must be exactly0 or1 and requires DS4_GPU_MISSES=1";return false;
        }
        st.gpu_miss_direct_mode=gpu_miss_direct_enabled?1:0;
        st.gpu_miss_enabled=gpu_misses_enabled?1:0;
        o = opt;
        if (const char* v = std::getenv("DS4_PREFILL_DENSE_PROJ_CACHE")) {
            if (std::strcmp(v,"0") != 0 && std::strcmp(v,"1") != 0) {
                err="DS4_PREFILL_DENSE_PROJ_CACHE must be0 or1"; return false;
            }
            prefill_dense_projection_cache = v[0]=='1';
        }
#if defined(STRATA_USE_HIP)
        if(prefill_dense_projection_cache) { err="DS4_PREFILL_DENSE_PROJ_CACHE requires CUDA";return false; }
#endif
        if (const char* v = std::getenv("DS4_PREFILL_ENGRAM_CACHE")) {
            if (std::strcmp(v,"0") != 0 && std::strcmp(v,"1") != 0) {
                err = "DS4_PREFILL_ENGRAM_CACHE must be0 or1"; return false;
            }
            prefill_engram_cache = v[0] == '1';
        }
#if defined(STRATA_USE_HIP)
        if (prefill_engram_cache) { err="DS4_PREFILL_ENGRAM_CACHE requires CUDA"; return false; }
#endif
        if (const char* v = std::getenv("DS4_PREFILL_CHUNK_MAJOR_PHASE")) {
            if (std::strcmp(v,"0") != 0 && std::strcmp(v,"1") != 0) {
                err = "DS4_PREFILL_CHUNK_MAJOR_PHASE must be0 or1"; return false;
            }
            chunk_major_phase_enabled = v[0] == '1';
        }
        if(prefill_dense_projection_cache && !chunk_major_phase_enabled) {
            err="DS4_PREFILL_DENSE_PROJ_CACHE requires DS4_PREFILL_CHUNK_MAJOR_PHASE=1";return false;
        }
        if (const char* value=std::getenv("DS4_RECORD_RAM_DOMAINS")) {
            if(std::strcmp(value,"0")!=0 && std::strcmp(value,"1")!=0) {
                err="DS4_RECORD_RAM_DOMAINS must be0 or1";return false;
            }
            record_copied_ram_domains=value[0]=='1';
        }
        if(record_copied_ram_domains && !chunk_major_phase_enabled) {
            err="DS4_RECORD_RAM_DOMAINS requires DS4_PREFILL_CHUNK_MAJOR_PHASE=1";return false;
        }
#if defined(STRATA_USE_HIP)
        if(record_copied_ram_domains) {err="DS4_RECORD_RAM_DOMAINS is a CUDA-only experiment";return false;}
#endif
        if(const char* value=std::getenv("DS4_CACHE_WORKER_EPOCH")) {
            if(std::strcmp(value,"0")!=0&&std::strcmp(value,"1")!=0) {
                err="DS4_CACHE_WORKER_EPOCH must be0 or1";return false;
            }
            cache_epoch_enabled=value[0]=='1';
        }
        if(cache_epoch_enabled&&!record_copied_ram_domains) {
            err="DS4_CACHE_WORKER_EPOCH requires DS4_RECORD_RAM_DOMAINS=1";return false;
        }
        if(!probeproto::flag(std::getenv("DS4_CACHE_EPOCH_PROBE"),cache_epoch_probe_enabled)) {
            err="DS4_CACHE_EPOCH_PROBE must be0 or1";return false;
        }
        if(cache_epoch_probe_enabled&&!cache_epoch_enabled) {
            err="DS4_CACHE_EPOCH_PROBE requires DS4_CACHE_WORKER_EPOCH=1";return false;
        }
        if(!probeproto::flag(std::getenv("DS4_CACHE_POOL_SHRINK_PROBE"),cache_pool_shrink_probe_enabled)) {
            err="DS4_CACHE_POOL_SHRINK_PROBE must be0 or1";return false;
        }
        if(cache_pool_shrink_probe_enabled&&!cache_epoch_probe_enabled) {
            err="DS4_CACHE_POOL_SHRINK_PROBE requires DS4_CACHE_EPOCH_PROBE=1";return false;
        }
        if(!probeproto::flag(std::getenv("DS4_CACHE_WORKSPACE_REGROW_PROBE"),cache_workspace_regrow_probe_enabled)) {
            err="DS4_CACHE_WORKSPACE_REGROW_PROBE must be0 or1";return false;
        }
        if(cache_workspace_regrow_probe_enabled&&!cache_pool_shrink_probe_enabled) {
            err="DS4_CACHE_WORKSPACE_REGROW_PROBE requires DS4_CACHE_POOL_SHRINK_PROBE=1";return false;
        }
        if(!probeproto::flag(std::getenv("DS4_PREFILL_REPEAT_REARM"),prefill_repeat_rearm)) {
            err="DS4_PREFILL_REPEAT_REARM must be0 or1";return false;
        }
        if(!repeatpolicy::parse_min_tokens(std::getenv("DS4_PREFILL_REPEAT_MIN_TOKENS"),
            prefill_repeat_rearm,prefill_repeat_min_tokens)) {
            err="DS4_PREFILL_REPEAT_MIN_TOKENS requires DS4_PREFILL_REPEAT_REARM=1 and a decimal integer8192..1048576";return false;
        }
#if defined(STRATA_USE_HIP)
        if(prefill_repeat_rearm){err="DS4_PREFILL_REPEAT_REARM requires CUDA";return false;}
#endif
        if (prefill_engram_cache && !chunk_major_phase_enabled) {
            err="DS4_PREFILL_ENGRAM_CACHE requires DS4_PREFILL_CHUNK_MAJOR_PHASE=1"; return false;
        }
        if(const char* value=std::getenv("DS4_PREFILL_NORMALIZE_HEAT")) {
            if(std::strcmp(value,"0")!=0&&std::strcmp(value,"1")!=0) {
                err="DS4_PREFILL_NORMALIZE_HEAT must be0 or1";return false;
            }
            normalize_prefill_heat=value[0]=='1';
        }
        if(normalize_prefill_heat&&!chunk_major_phase_enabled) {
            err="DS4_PREFILL_NORMALIZE_HEAT requires DS4_PREFILL_CHUNK_MAJOR_PHASE=1";return false;
        }
        if(const char* value=std::getenv("DS4_PREFILL_COLD_F16")) {
            if(std::strcmp(value,"0")!=0&&std::strcmp(value,"1")!=0) {
                err="DS4_PREFILL_COLD_F16 must be0 or1";return false;
            }
            cold_f16_enabled=value[0]=='1';
        }
        if(cold_f16_enabled&&!chunk_major_phase_enabled) {
            err="DS4_PREFILL_COLD_F16 requires DS4_PREFILL_CHUNK_MAJOR_PHASE=1";return false;
        }
        if(const char* value=std::getenv("DS4_PREFILL_COLD_F16_BUCKET")) {
            if(!cold_f16_enabled) {err="cold F16 bucket rule requires DS4_PREFILL_COLD_F16=1";return false;}
            if(std::strcmp(value,"power2")==0)cold_f16_rule=coldf16::Rule::power2;
            else if(std::strcmp(value,"128")==0)cold_f16_rule=coldf16::Rule::ceil128;
            else if(std::strcmp(value,"exact")==0)cold_f16_rule=coldf16::Rule::exact;
            else {err="DS4_PREFILL_COLD_F16_BUCKET must be power2,128 or exact";return false;}
        }
        if(const char* value=std::getenv("DS4_PREFILL_REGISTER_RAM_MB")) {
            if(std::strcmp(value,"all")==0)register_ram_all=true;
            else {
                char* tail=nullptr;const unsigned long long mb=std::strtoull(value,&tail,10);
                if(value[0]=='-'||tail==value||*tail!='\0'||mb>(~size_t(0)>>20)) {
                    err="DS4_PREFILL_REGISTER_RAM_MB must be a nonnegative MiB integer or all";return false;
                }
                register_ram_mb=(size_t)mb;
            }
        }
        if((register_ram_all||register_ram_mb>0)&&!chunk_major_phase_enabled) {
            err="registered-copy DMA requires DS4_PREFILL_CHUNK_MAJOR_PHASE=1";return false;
        }
        if(const char* value=std::getenv("DS4_PREFILL_FINAL_RESERVE_MB")) {
            if(!strata::ds4::phasepolicy::parse_mib(value,3072,8192,prefill_final_reserve_mb)) {
                err="DS4_PREFILL_FINAL_RESERVE_MB must be0 or a decimal integer3072..8192";return false;
            }
        }
        if(const char* value=std::getenv("DS4_PREFILL_REGISTER_GPU_FLOOR_MB")) {
            if(!strata::ds4::phasepolicy::parse_mib(value,2048,~size_t(0)>>20,register_gpu_floor_mb)) {
                err="DS4_PREFILL_REGISTER_GPU_FLOOR_MB must be0 or a decimal MiB integer>=2048 that fits size_t bytes";return false;
            }
        }
        if((prefill_final_reserve_mb>0||register_gpu_floor_mb>0)&&!chunk_major_phase_enabled) {
            err="first-prefill reserve policies require DS4_PREFILL_CHUNK_MAJOR_PHASE=1";return false;
        }
        if(register_gpu_floor_mb>0&&!register_ram_all&&register_ram_mb==0) {
            err="explicit registered GPU floor requires DS4_PREFILL_REGISTER_RAM_MB>0 or all";return false;
        }
        if (const char* value = std::getenv("DS4_PREFILL_GPU_H_MB")) {
            char* tail = nullptr;
            const unsigned long long mb = std::strtoull(value, &tail, 10);
            if (value[0] == '-' || tail == value || *tail != '\0' || mb > (~(size_t) 0 >> 20)) {
                err = "DS4_PREFILL_GPU_H_MB must be a nonnegative integer MiB budget";
                return false;
            }
            prefill_gpu_h_mb = (size_t) mb;
        }
        if (const char* value = std::getenv("DS4_PREFILL_COPY_RAM")) {
            if (std::strcmp(value, "0") != 0 && std::strcmp(value, "1") != 0) {
                err = "DS4_PREFILL_COPY_RAM must be 0 or 1";
                return false;
            }
            prefill_copy_ram = value[0] == '1';
        }
        if (prefill_copy_ram && prefill_gpu_h_mb == 0) {
            err = "DS4_PREFILL_COPY_RAM requires DS4_PREFILL_GPU_H_MB > 0";
            return false;
        }
        if (const char* value = std::getenv("DS4_LM_F16_CACHE_MB")) {
            char* tail = nullptr;
            const unsigned long long mb = std::strtoull(value, &tail, 10);
            if (value[0] == '-' || tail == value || *tail != '\0' || mb > (~(size_t) 0 >> 20)) {
                err = "DS4_LM_F16_CACHE_MB must be a nonnegative integer MiB budget";
                return false;
            }
            lm_f16_cache_mb = (size_t) mb;
            if (lm_f16_cache_mb > 0 && prefill_gpu_h_mb == 0) {
                err = "DS4_LM_F16_CACHE_MB requires the existing first-prefill DS4_PREFILL_GPU_H_MB phase";
                return false;
            }
        }
        if (const char* value = std::getenv("DS4_PREFILL_DECODE_CHUNK")) {
            if (std::strcmp(value, "1024") != 0 || prefill_gpu_h_mb == 0) {
                err = "DS4_PREFILL_DECODE_CHUNK must be exactly 1024 and requires DS4_PREFILL_GPU_H_MB > 0";
                return false;
            }
            prefill_decode_chunk = 1024;
        }
        if (const char* value = std::getenv("DS4_LM_F16_MIXED_LIGHT")) {
            if (std::strcmp(value, "0") != 0 && std::strcmp(value, "1") != 0) {
                err = "DS4_LM_F16_MIXED_LIGHT must be 0 or 1";
                return false;
            }
            lm_f16_mixed_light = value[0] == '1';
            if (lm_f16_mixed_light && lm_f16_cache_mb == 0) {
                err = "DS4_LM_F16_MIXED_LIGHT requires the first-phase DS4_LM_F16_CACHE_MB cache";
                return false;
            }
        }
        const double t0 = now_ms();
        if (!pk.load(o.pack, err)) return false;
        g.from(pk);
        if (g.dim == 0 || g.nl == 0 || (int) g.ratios.size() < g.nl) {
            err = o.pack + "/ds4.txt: incomplete geometry";
            return false;
        }
        if (g.hd != 512 || g.ix_hd != 128 || g.k_exp > 8 || g.hc > 8) {
            err = "geometry not covered (head_dim " + std::to_string(g.hd) + ", index_head_dim " +
                  std::to_string(g.ix_hd) + ", experts per token " + std::to_string(g.k_exp) + ")";
            return false;
        }
        freqs_win = rope_freqs(g.rd, 0, g.rope_theta, g.factor, g.beta_fast, g.beta_slow);
        freqs_comp = rope_freqs(g.rd, g.orig_ctx, g.comp_theta, g.factor, g.beta_fast, g.beta_slow);

        // devices
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
            err = "no CUDA device";
            return false;
        }
        std::vector<int> ids = o.gpus;
        if (ids.empty())
            for (int i = 0; i < ndev; ++i) ids.push_back(i);
        devs.resize(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            Dev& d = devs[i];
            d.id = ids[i];
            CK(cudaSetDevice(d.id));
            CK(cudaStreamCreateWithFlags(&d.st, cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&d.ev, cudaEventDisableTiming));
            if (!x3::prepare(d.st) || (pk.gguf && !gg::prepare(d.st))) {
                err = "exl3 scratch on CUDA" + std::to_string(d.id);
                return false;
            }
            CK(cudaMemGetInfo(&d.free0, &d.total));
        }
        // the groups' width: o.tp, less while the experts' blocks do not split in it (a member takes n_ff / tp of each
        // expert: the down's inputs in whole blocks - a GGUF's 256-value types split n_ff 2304 in 1, 3 or 9 parts)
        int tp = o.tp;
        if (pk.gguf && tp > 1) {
            int blk = 32;
            for (const auto& kv : pk.t) {
                const std::string& n = kv.first;
                if (n.find(".ffn.experts.") == std::string::npos || n.size() < 9 ||
                    n.compare(n.size() - 9, 9, "w2.weight") != 0)
                    continue;
                const int ty = gg_type_of(kv.second.dtype);
                if (ty >= 0) blk = std::max(blk, gg::type_block(ty));
            }
            while (tp > 1 && (g.ff % tp != 0 || (g.ff / tp) % blk != 0 || (g.ff / tp) % (std::getenv("DS4_TP_ALIGN") ? std::atoi(std::getenv("DS4_TP_ALIGN")) : 128) != 0 || g.heads % tp != 0 ||
                              g.o_groups % tp != 0))
                --tp;
            if (tp != o.tp)
                std::fprintf(stderr, "ds4: tensor parallel %d -> %d: the experts' %d-value blocks split n_ff %d no other way "
                                     "(each GPU its own layers)\n", o.tp, tp, blk, g.ff);
        }
        // the groups: runs of tp devices of one kind (their memory), a device unlike the next alone
        // - or, tp the number of GPUs, one group of all of them whatever their kind (its tightest member sets the
        // experts' room); threads mode for the windows (decode) where a group is 2 wide or more or --threads on, the
        // prompt path (its own thread a GPU) as asked for - also where the experts' blocks took the groups' width down
        // to 1 (a 256-value type's: Maya's quants), which left every prompt in windows of 16
        par = o.threads || tp > 1;
        // the chunked (layer-major) prompt path: with --tp / --threads as before; --prefill-chunk N (N > 0) or
        // kChunkedPromptDefault bring it to a single card's default run too (a host thread a GPU for the prompt only;
        // decode stays as asked).  --prefill-chunk 0 keeps the windows of 16 (DS4_WINDOWS=1 the same).  Off by default
        // until a prompt costs the decode nothing: its buffers hold VRAM the windowed path gives to experts
        prompt_par = par || o.tp > 1 ||
                     ((o.prefill_chunk > 0 || (o.prefill_chunk < 0 && kChunkedPromptDefault)) &&
                      std::getenv("DS4_WINDOWS") == nullptr);
        const bool tp_all = tp > 1 && tp == (int) devs.size();
        for (size_t i = 0; i < devs.size();) {
            Group G;
            G.mem.push_back((int) i);
            size_t j = i + 1;
            while ((int) G.mem.size() < tp && j < devs.size() && (tp_all || devs[j].total == devs[i].total))
                G.mem.push_back((int) j++);
            if ((int) G.mem.size() < tp && G.mem.size() > 1) {   // a short run at the end: devices of their own
                for (int m : G.mem) {
                    Group one;
                    one.mem.push_back(m);
                    groups.push_back(one);
                }
            } else groups.push_back(G);
            i = j;
        }
        for (size_t gi = 0; gi < groups.size(); ++gi)
            for (size_t m = 0; m < groups[gi].mem.size(); ++m) {
                devs[(size_t) groups[gi].mem[m]].grp = (int) gi;
                devs[(size_t) groups[gi].mem[m]].mi = (int) m;
            }
        // whether the CPU lane's experts are read in place (--mmap-experts; auto: the pack's expert bytes beyond what the
        // GPUs can take - their free memory less the other weights and the prompt path's buffers - do not fit the RAM
        // this process can get beside kRamMargin), decided here because it sets the prompt chunk's default: in place
        // every chunk streams every expert VRAM does not hold from the files, so the larger the chunk the fewer times
        if (pk.gguf) file_experts = true;   // (a GGUF's: the files stay their source - copy_experts copies them into RAM
                                            // after VRAM's are in, by --mmap-experts)
        else if (o.mmap_experts >= 0) file_experts = o.mmap_experts > 0;
        else {
            double exp_b = 0, other_b = 0, room = 0;
            for (const auto& kt : pk.t) {
                const std::string& n = kt.first;
                if (n.find(".engram.embed.") != std::string::npos || n.rfind("vision.", 0) == 0 || n == "embed.weight" ||
                    (!o.dspark && n.rfind("mtp.", 0) == 0))
                    continue;
                (n.find(".ffn.experts.") != std::string::npos ? exp_b : other_b) += (double) kt.second.bytes;
            }
            for (const Dev& d : devs)
                room += std::max(0.0, (double) d.free0 - (double) ((size_t) o.reserve_mib << 20) -
                                          (prompt_par ? 1.5 : 0.4) * (1 << 30)) *
                        o.vram_experts;
            file_experts = exp_b - std::max(0.0, room - other_b) + (double) kRamMargin > (double) ram_avail();
        }
        const bool own_default = !(par || o.tp > 1);   // (the single-card default run the chunked path newly takes: chunks of 1024 - 8K prompt 370 tok/s on 4 x R9700, 214 at 8192; the threads / tp runs keep theirs)
        chunk = prompt_par ? (o.prefill_chunk >= 0 ? o.prefill_chunk : own_default ? 1024 : file_experts ? kVlRows : 2048) : 0;
        const char* dc = std::getenv("DS4_CHUNK");   // (tests: down to 16, a short prompt in several chunks)
        if (dc != nullptr) chunk = prompt_par ? std::atoi(dc) : 0;
        if (const char* v = std::getenv("DS4_CHUNK_MIN")) chunk_min = std::max(1, std::atoi(v));
        if (chunk > 0) {
            chunk = std::min(chunk_major_phase_enabled ? 65280 : kMaxChunk, std::max(dc != nullptr ? 16 : 256, chunk / 16 * 16));
            eg_cap = (size_t) chunk;
            tap_cap = std::max(kWin, 128);
        } else   // (only as asked for: --prefill-chunk 0, or --tp 1 with --threads off)
            std::fprintf(stderr,
                         "ds4: prompts run in windows of %d tokens (%s): long prompts prefill far slower than in\n"
                         "     chunks (a 7-GPU RTX 3090 rig, Maya-S read in place: 8K / 32K tokens 115 / 99 tokens/s,\n"
                         "     461 / 1010 in chunks of %d) - %s\n",
                         kWin, o.prefill_chunk == 0 ? "--prefill-chunk 0" : "--tp 1 with --threads off", kVlRows,
                         o.prefill_chunk == 0 ? "leave --prefill-chunk out for chunks"
                                              : "leave --tp 1 out, or pass --threads on, for chunks");
        // the vision tower on the first GPU, before the split: the layers get what it leaves
        if (o.vision && !load_vision(err)) {   // (a tower that cannot load leaves the model reading text: said loudly,
            if (o.vision_only) return false;   // and the server is told pictures are off - info_lines)
            std::fprintf(stderr, "ds4: WARNING: the vision tower did not load (%s): pictures are OFF, the model reads "
                                 "text only\n", err.c_str());
            vision_err = err.empty() ? "unknown reason" : err;
            err.clear();
            vt.release();
            img_tok = -1;
        }
        if (o.vision_only) return true;

        // layers: roles and the sources they read
        if (o.dspark_auto && o.dspark && g.n_mtp > 0) {   // (see Options::dspark_auto)
            double exp_b = 0, room_b = 0;
            for (const auto& kt : pk.t)
                if (kt.first.rfind("layers.", 0) == 0 && kt.first.find(".ffn.experts.") != std::string::npos) exp_b += (double) kt.second.bytes;
            for (const Dev& d : devs) room_b += std::max(0.0, (double) d.free0 - (double) ((size_t) o.reserve_mib << 20));
            if (exp_b > 1.5 * room_b) {
                o.dspark = false;
                std::fprintf(stderr, "ds4: DSpark not loaded: the routed experts (%.0f GB) are far more than the GPU holds (%.0f GB), where its verify "
                                     "rounds read 3.3x a token's bytes and gain nothing (--dspark on loads it anyway)\n", exp_b / 1e9, room_b / 1e9);
            }
        }
        if (!o.dspark) g.n_mtp = 0;
        if (o.dspark_block > 0 && std::getenv("DS4_DSPARK_MAX") == nullptr) max_drafts = o.dspark_block;   // (--dspark-block N: at most N drafts a round)
        L.resize((size_t) (g.nl + g.n_mtp));
        int last_src = -1;
        for (int il = 0; il < g.nl + g.n_mtp; ++il) {
            Layer& l = L[(size_t) il];
            l.il = il;
            l.mtp = il >= g.nl;
            l.name = l.mtp ? "mtp." + std::to_string(il - g.nl) + "." : "layers." + std::to_string(il) + ".";
            l.n_exp = l.mtp ? g.d_n_exp : g.n_exp;
            l.k_exp = l.mtp ? g.d_k : g.k_exp;
            l.ratio = g.ratios[(size_t) il];
            if (l.mtp) {   // window attention only: no compressor, indexer or Engram
                if (l.ratio != 0 || l.n_exp <= 0 || l.k_exp <= 0 || l.k_exp > kPicks) {
                    err = l.name + ": a DSpark block of compress ratio " + std::to_string(l.ratio) + ", " +
                          std::to_string(l.n_exp) + " experts / " + std::to_string(l.k_exp) + " is not covered";
                    return false;
                }
                if (!expert_layout(l, err)) return false;
                continue;
            }
            l.kv_src = std::find(g.kv_src.begin(), g.kv_src.end(), il) != g.kv_src.end();
            l.ix_src = std::find(g.ix_src.begin(), g.ix_src.end(), il) != g.ix_src.end();
            l.owns_k = l.kv_src;
            l.cand_src = il == g.cand_src;
            l.uses_cand = g.cand_src >= 0 && g.cand_src < il && l.ix_src;
            if (l.kv_src) {
                Src s;
                s.layer = il;
                s.ratio = l.ratio;
                s.cap = (o.max_context + l.ratio - 1) / l.ratio + 1;
                srcs.push_back(s);
                last_src = (int) srcs.size() - 1;
            }
            if (l.ratio > 0) {
                if (last_src < 0 || srcs[(size_t) last_src].ratio != l.ratio) {
                    err = "layer " + std::to_string(il) + ": compress ratio " + std::to_string(l.ratio) +
                          " without a KV source of that ratio before it";
                    return false;
                }
                l.src = last_src;
                l.ik_src = last_src;
            }
            auto eg = std::find(g.eg_layers.begin(), g.eg_layers.end(), il);
            if (eg != g.eg_layers.end()) l.eg = (int) (eg - g.eg_layers.begin());
            if (!expert_layout(l, err)) return false;
        }

        // the split: each layer's dense weights + its experts, the head on the last GPU
        for (Layer& l : L) {
            const std::string& P = l.name;
            size_t b = 0;
            if (l.mtp && l.il == g.nl) b += mat_bytes("mtp.0.main_proj");
            for (const char* m : {"attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b", "ffn.shared_experts.w1",
                                  "ffn.shared_experts.w2", "ffn.shared_experts.w3", "attn.compressor.wkv",
                                  "attn.compressor.wgate", "attn.indexer.wq_b", "attn.indexer.wk", "engram.wkv"})
                b += mat_bytes(P + m);
            for (int gi = 0; gi < g.o_groups; ++gi) b += mat_bytes(P + "attn.wo_a.slice." + std::to_string(gi));
            b += tensor_bytes(P + "hc_attn_fn") + tensor_bytes(P + "hc_ffn_fn") + tensor_bytes(P + "ffn.gate.weight");
            b += (size_t) g.win * win_rb() + (2u << 20);
            l.dense_bytes = b;
        }
        const size_t head_b = mat_bytes("head") + (size_t) g.vocab * 4 + (16u << 20);
        const size_t per_dev_fixed = (size_t) o.reserve_mib << 20;
        const size_t kv_per_src_dev = 0;   // counted after the split
        (void) kv_per_src_dev;
        std::vector<double> avail(groups.size(), 0.0);
        double avail_sum = 0;
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            for (int m : groups[gi].mem)
                avail[gi] += std::max(0.0, (double) devs[(size_t) m].free0 - (double) per_dev_fixed - 400.0 * (1 << 20));
            avail_sum += avail[gi];
        }
        double need_sum = (double) head_b;
        for (const Layer& l : L) need_sum += (double) l.dense_bytes + (double) l.blob * l.n_exp;
        {
            // contiguous layer ranges, each group's share of the total by its free memory
            size_t d = 0;
            double cum = 0, target = avail[0] / avail_sum * need_sum;
            std::vector<Layer*> order;   // the drafter's blocks first (the first GPU), then the trunk's layers
            for (Layer& l : L)
                if (l.mtp) order.push_back(&l);
            for (Layer& l : L)
                if (!l.mtp) order.push_back(&l);
            for (Layer* lp : order) {
                Layer& l = *lp;
                const double lb = (double) l.dense_bytes + (double) l.blob * l.n_exp;
                if (cum + lb / 2 > target && d + 1 < groups.size()) {
                    ++d;
                    target += avail[d] / avail_sum * need_sum;
                }
                l.grp = (int) d;
                cum += lb;
            }
            // with the chunked prompt path: the contiguous split that streams the least into its busiest member (its
            // VRAM short of its layers' experts, per member) - a chunk's groups run as a pipeline, its slowest stage
            // sets the pace; a token's decode passes every layer either way, the experts in VRAM the same in all
            if (chunk > 0 && groups.size() > 1 && groups.size() <= 6 && std::getenv("DS4_SPLIT") == nullptr) {
                const int G = (int) groups.size(), NL = g.nl;
                std::vector<double> cap(groups.size());
                for (int gi = 0; gi < G; ++gi) {
                    const int nm = (int) groups[(size_t) gi].mem.size();
                    double mn = 1e300;
                    for (int m : groups[(size_t) gi].mem)
                        mn = std::min(mn, std::max(0.0, (double) devs[(size_t) m].free0 - (double) per_dev_fixed -
                                                            400.0 * (1 << 20) - 1.1 * (1 << 30)));   // (the chunk's buffers)
                    cap[(size_t) gi] = mn * nm;
                }
                for (const Layer& l : L)
                    if (l.mtp) cap[0] -= (double) l.dense_bytes + (double) l.blob * l.n_exp;
                cap[(size_t) G - 1] -= (double) head_b;
                std::vector<double> pe(NL + 1, 0.0), pd(NL + 1, 0.0);   // prefix sums: experts, dense
                for (int il = 0; il < NL; ++il) {
                    pe[(size_t) il + 1] = pe[(size_t) il] + (double) L[(size_t) il].blob * L[(size_t) il].n_exp;
                    pd[(size_t) il + 1] = pd[(size_t) il] + (double) L[(size_t) il].dense_bytes;
                }
                auto cost = [&](int gi, int a, int b) {   // per-member streaming of layers [a, b) on group gi
                    const int nm = (int) groups[(size_t) gi].mem.size();
                    const double dense = (pd[(size_t) b] - pd[(size_t) a]) * (nm > 1 ? 1.3 : 1.0);
                    return std::max(0.0, pe[(size_t) b] - pe[(size_t) a] - (cap[(size_t) gi] - dense)) / nm;
                };
                // best[gi][b]: the least max cost of layers [0, b) over groups 0..gi (each at least one layer)
                std::vector<std::vector<double>> best((size_t) G, std::vector<double>((size_t) NL + 1, 1e300));
                std::vector<std::vector<int>> cut((size_t) G, std::vector<int>((size_t) NL + 1, 0));
                for (int b = 1; b <= NL; ++b) best[0][(size_t) b] = cost(0, 0, b);
                for (int gi = 1; gi < G; ++gi)
                    for (int b = gi + 1; b <= NL; ++b)
                        for (int a = gi; a < b; ++a) {
                            const double v = std::max(best[(size_t) gi - 1][(size_t) a], cost(gi, a, b));
                            if (v < best[(size_t) gi][(size_t) b]) {
                                best[(size_t) gi][(size_t) b] = v;
                                cut[(size_t) gi][(size_t) b] = a;
                            }
                        }
                int b = NL;
                for (int gi = G - 1; gi >= 0; --gi) {
                    const int a = gi > 0 ? cut[(size_t) gi][(size_t) b] : 0;
                    for (int il = a; il < b; ++il) L[(size_t) il].grp = gi;
                    b = a;
                }
            }
            // DS4_SPLIT=a,b,..: the trunk layers each group takes (the drafter stays with the first)
            if (const char* sp = std::getenv("DS4_SPLIT")) {
                std::vector<int> n;
                for (const char* q = sp; *q;) {
                    n.push_back(std::atoi(q));
                    while (*q && *q != ',') ++q;
                    if (*q == ',') ++q;
                }
                int il = 0;
                for (size_t gi = 0; gi < n.size() && gi < groups.size(); ++gi)
                    for (int j = 0; j < n[gi] && il < g.nl; ++j) L[(size_t) il++].grp = (int) gi;
                for (; il < g.nl; ++il) L[(size_t) il].grp = (int) groups.size() - 1;
            }
            for (Layer& l : L) {
                l.dev = groups[(size_t) l.grp].mem[0];
                l.nm = l.mtp ? 1 : (int) groups[(size_t) l.grp].mem.size();   // (the drafter's blocks: on the lead alone)
            }
            head_dev = L[(size_t) g.nl - 1].dev;
            d_dev = g.n_mtp > 0 ? L[(size_t) g.nl].dev : 0;
        }
        // the groups' members: copies of each layer's device state, a member's part of an expert
        for (Layer& l : L) {
            l.ff0 = 0;
            l.ffn = g.ff;
            l.ah0 = l.og0 = 0;
            l.ahn = g.heads;
            l.ogn = g.o_groups;
            if (l.nm == 1) continue;
            const int na = head_members(l.nm);
            if (g.heads % na != 0 || g.o_groups % na != 0 || g.ff % 128 != 0 || g.ff / 128 < l.nm) {
                err = "tensor parallel " + std::to_string(l.nm) + ": the heads, output groups or FFN do not split";
                return false;
            }
            for (int m = 1; m < l.nm; ++m) {
                Layer c = l;
                c.mates.clear();
                c.mi = m;
                c.dev = groups[(size_t) l.grp].mem[(size_t) m];
                l.mates.push_back(std::move(c));
            }
            // each member's FFN columns and its part of an expert's layout: an EXL3 expert's in whole 128-blocks, as even
            // as they go (split128, the odd blocks going round the members layer by layer); a GGUF's in even shares (its
            // groups are only as wide as its experts' blocks split evenly: mx_blob_to_half's n_ff / nm)
            const bool gguf_x = l.lay.fmt == gg::kFmtMXFP4;
            for (int m = 0; m < l.nm; ++m) {
                Layer& x = member(l, m);
                const int ha = m - (l.nm - na);   // (its place among the members with heads)
                x.ahn = ha < 0 ? 0 : g.heads / na;
                x.ah0 = ha < 0 ? 0 : ha * x.ahn;
                x.ogn = ha < 0 ? 0 : g.o_groups / na;
                x.og0 = ha < 0 ? 0 : ha * x.ogn;
                if (gguf_x) {
                    x.ffn = g.ff / l.nm;
                    x.ff0 = m * x.ffn;
                } else
                    split128(g.ff, l.nm, m, x.ff0, x.ffn, l.il % l.nm);   // (rot: the odd blocks go round the members)
                size_t off = 0;
                for (int r = 0; r < 3; ++r) {
                    const int k = r < 2 ? g.dim : x.ffn, n = r < 2 ? x.ffn : g.dim;
                    x.hlay.K[r] = l.lay.K[r];
                    x.hlay.trellis[r] = off;
                    if (gguf_x) {   // gate / up rows, down inputs: [n][k / block] blocks
                        const int ty = gg::expert_type(l.lay, r);
                        if (k % gg::type_block(ty) != 0) {
                            err = "tensor parallel " + std::to_string(l.nm) + ": the experts' " + gg::type_name(ty) +
                                  " blocks do not split";
                            return false;
                        }
                        x.hlay.suh[r] = x.hlay.svh[r] = off;
                        off += align256((size_t) n * gg::type_row_bytes(ty, k));
                        continue;
                    }
                    off += align256(x3::trellis_bytes(k, n, l.lay.K[r]));
                    x.hlay.suh[r] = off;
                    off += align256((size_t) k * 2);
                    x.hlay.svh[r] = off;
                    off += align256((size_t) n * 2);
                }
                x.hlay.cb = l.lay.cb;
                x.hlay.fmt = l.lay.fmt;
                x.hblob = off;
            }
        }
        // which devices hold each source's compressed KV / index keys
        for (Src& s : srcs) {
            s.ckv.assign(devs.size(), nullptr);
            s.ik.assign(devs.size(), nullptr);
        }
        for (const Layer& l0 : L)
            for (int m = 0; m < l0.nm; ++m) {
                const Layer& l = member(l0, m);
                if (l.src >= 0) {
                    srcs[(size_t) l.src].ckv[(size_t) l.dev] = (uint8_t*) 1;
                    if (l0.grp != L[(size_t) srcs[(size_t) l.src].layer].grp) srcs[(size_t) l.src].remote = true;
                }
                if (l.ix_src && l.ik_src >= 0) {
                    srcs[(size_t) l.ik_src].ik[(size_t) l.dev] = (uint8_t*) 1;
                    if (l0.grp != L[(size_t) srcs[(size_t) l.ik_src].layer].grp) srcs[(size_t) l.ik_src].remote = true;
                }
            }

        // per device: the scratch, the caches, the dense weights
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[di];
            const int D = g.dim, H = g.hc;
            const size_t W = kWin, P = (size_t) kWin * kPicks, cap = (size_t) o.max_context + 64;
            bool ok = dalloc(d, d.h, W * H * D, err) && dalloc(d, d.h2, W * H * D, err) && dalloc(d, d.x, W * D, err) &&
                      dalloc(d, d.qr, W * g.q_lora, err) && dalloc(d, d.q, W * g.heads * g.hd, err) &&
                      dalloc(d, d.kvr, W * g.hd, err) && dalloc(d, d.ckr, W * g.hd, err) &&
                      dalloc(d, d.cgr, W * g.hd, err) && dalloc(d, d.latent, W * g.hd, err) &&
                      dalloc(d, d.ikr, W * g.ix_hd, err) && dalloc(d, d.iq, W * g.ix_heads * g.ix_hd, err) &&
                      dalloc(d, d.iw, W * 64, err) && dalloc(d, d.scores, W * cap, err) &&
                      dalloc(d, d.bm, W * bm_stride(), err) && dalloc(d, d.o, W * g.heads * g.hd, err) &&
                      dalloc(d, d.oa, W * g.o_groups * g.o_lora, err) && dalloc(d, d.ao, W * D, err) &&
                      dalloc(d, d.mix, W * 32, err) && dalloc(d, d.a_pre, W * 8, err) && dalloc(d, d.a_post, W * 8, err) &&
                      dalloc(d, d.a_comb, W * 64, err) && dalloc(d, d.f_pre, W * 8, err) &&
                      dalloc(d, d.f_post, W * 8, err) && dalloc(d, d.f_comb, W * 64, err) && dalloc(d, d.pre, W * 8, err) &&
                      dalloc(d, d.glog, W * 1024, err) && dalloc(d, d.glog_a, 2 * W * 1024, err) &&
                      dalloc(d, d.rw, W * kPicks, err) &&
                      dalloc(d, d.ridx, W * kPicks, err) && dalloc(d, d.moe_out, W * D, err) &&
                      dalloc(d, d.sg, W * g.ff, err) && dalloc(d, d.su, W * g.ff, err) && dalloc(d, d.sh, W * g.ff, err) &&
                      dalloc(d, d.sh_out, W * D, err) && dalloc(d, d.cpu_part, W * D, err) &&
                      dalloc(d, d.freqs_win, 64, err) && dalloc(d, d.freqs_comp, 64, err) &&
                      dalloc(d, d.topk, W * topk_stride(), err) && dalloc(d, d.cand, W * cand_stride(), err) &&
                      dalloc(d, d.tok, 16, err) && dalloc(d, d.emb_rows, W * D, err) &&
                      dalloc(d, d.ext, ((size_t) g.win + W) * g.hd, err) && dalloc(d, d.xe, P * D, err) &&
                      dalloc(d, d.xg, P * g.ff, err) && dalloc(d, d.xu, P * g.ff, err) && dalloc(d, d.xhh, P * g.ff, err) &&
                      dalloc(d, d.ed, P * D, err) && dalloc(d, d.plan_ptr, P, err);
            if (!ok || !alloc_topk_scratch(d, d.topk_scr, W, err)) return false;
            d.attn_scr.max_t = kAttnScratchT;
            d.attn_scr.cap = (int) topk_stride();
            if (!dalloc(d, d.attn_scr.rows, (size_t) kAttnScratchT * topk_stride() * g.hd, err)) return false;
            if (!g.eg_layers.empty()) {
                const int cols = (pk.eg_ngram - 1) * pk.eg_heads;
                ok = dalloc(d, d.eg_rows, W * cols * (pk.eg_hd + pk.eg_hd / 32), err) &&
                     dalloc(d, d.eg_x, W * cols * pk.eg_hd, err) && dalloc(d, d.eg_kv, W * (g.hc + 1) * D, err);
                if (!ok) return false;
            }
            if ((int) di == head_dev && !dalloc(d, d.logits, (size_t) kWin * g.vocab, err)) return false;
            if (!halloc(d.bounce, bounce_bytes(), err)) return false;
            if (decode_fusion && (2 + H) * H <= 32 && !dalloc(d, d.hccnt, kWin, err)) return false;
            if (!dalloc(d, d.xcnt, 4, err) || !hmap(d.hand, bounce_bytes(), err) || !hmap(d.hand_flag, 32, err) ||
                !dalloc(d, d.grp_ptr, P, err) || !dalloc(d, d.grp_start, P + 1, err) || !dalloc(d, d.grp_ent, P, err) ||
                !dalloc(d, d.n_groups, 4, err))
                return false;
            if (g.n_mtp > 0 && !dalloc(d, d.tap, (size_t) tap_cap * D, err)) return false;
            if (g.n_mtp > 0 && (int) di == head_dev &&
                (!dalloc(d, d_confv, kWin, err) || !dalloc(d, d_ints, 64, err) || !dalloc(d, d_specq, kWin, err)))
                return false;
            CK(cudaEventCreateWithFlags(&d.ev_out, cudaEventDisableTiming));
            cudaSetDevice(d.id);
            CK(cudaMemcpy(d.freqs_win, freqs_win.data(), freqs_win.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d.freqs_comp, freqs_comp.data(), freqs_comp.size() * 4, cudaMemcpyHostToDevice));
            for (Src& s : srcs) {
                if (s.h_ckv == nullptr) {   // (its events belong to the source layer's device: it records them)
                    if (!halloc(s.h_ckv, (size_t) kWin * ckv_rb(), err) || !halloc(s.h_ik, (size_t) kWin * ik_rb(), err))
                        return false;
                    CK(cudaSetDevice(devs[(size_t) L[(size_t) s.layer].dev].id));
                    CK(cudaEventCreateWithFlags(&s.ev_ckv, cudaEventDisableTiming));
                    CK(cudaEventCreateWithFlags(&s.ev_ik, cudaEventDisableTiming));
                    CK(cudaSetDevice(d.id));
                }
                if (s.ckv[di] != nullptr && !dalloc(d, s.ckv[di], (size_t) s.cap * ckv_rb(), err, &d.caches))
                    return false;
                if (s.ik[di] != nullptr && !dalloc(d, s.ik[di], (size_t) s.cap * ik_rb(), err, &d.caches))
                    return false;
            }
        }
        for (Src& sr : srcs) {
            if (sr.remote && (!hmap(sr.x_stage, x_cap() * (ckv_rb() + ik_rb()), err) || !hmap(sr.x_flag, 32, err)))
                return false;
            if (sr.remote && chunk > 0 &&
                (!hmap(sr.x_stage1, x_cap() * (ckv_rb() + ik_rb()), err) || !hmap(sr.x_flag1, 32, err) ||
                 !hmap(sr.x_ack, (size_t) 2 * devs.size() * 32, err)))
                return false;
            const int sg = L[(size_t) sr.layer].grp;
            for (size_t di = 0; di < devs.size(); ++di)
                if ((sr.ckv[di] != nullptr || sr.ik[di] != nullptr) && devs[di].grp != sg) sr.readers.push_back((int) di);
        }
        for (Group& G : groups) {
            G.ar.nm = (int) G.mem.size();
            G.ar.slots = 4;
            G.ar.stride = (size_t) kWin * g.dim;
            if (G.ar.nm > 1 && (!hmap(G.ar.data, (size_t) G.ar.nm * G.ar.slots * G.ar.stride, err) ||
                                !hmap(G.ar.flags, (size_t) G.ar.nm * G.ar.slots * 32, err)))
                return false;
            // three members or more: the reduce-scatter's sums and its per-block flags (allreduce)
            if (G.ar.nm > 2 && (!hmap(G.ar.sums, (size_t) G.ar.slots * G.ar.stride, err) ||
                                !hmap(G.ar.bflags, (size_t) 2 * G.ar.nm * G.ar.slots * kArBlocks * 8, err)))
                return false;
            if (G.ar.nm > 1 && &G == &groups[(size_t) devs[(size_t) head_dev].grp] &&
                (!hmap(G.lg_stage, (size_t) 2 * (G.ar.nm - 1) * kWin * ((g.vocab / 128 + G.ar.nm - 1) / G.ar.nm * 128), err) ||
                 !hmap(G.lg_flag, (size_t) 2 * (G.ar.nm - 1) * 32, err) || !hmap(G.lg_ack, (size_t) 2 * 32, err)))
                return false;
            G.arb = G.ar;   // (a chunk's: two slots of chunk rows)
            G.arb.slots = 2;
            G.arb.stride = (size_t) chunk * g.dim;
            if (chunk > 0 && G.arb.nm > 1 && (!hmap(G.arb.data, (size_t) G.arb.nm * G.arb.slots * G.arb.stride, err) ||
                                              !hmap(G.arb.flags, (size_t) G.arb.nm * G.arb.slots * 32, err)))
                return false;
            if (chunk > 0 && G.arb.nm > 2 &&
                (!hmap(G.arb.sums, (size_t) G.arb.slots * G.arb.stride, err) ||
                 !hmap(G.arb.bflags, (size_t) 2 * G.arb.nm * G.arb.slots * kArBlocks * 8, err)))
                return false;
        }
        if (g.n_mtp > 0) {
            if (!halloc(h_taps, (size_t) tap_cap * g.d_targets.size() * g.dim, err)) return false;
            CK(cudaSetDevice(devs[(size_t) d_dev].id));
            CK(cudaEventCreateWithFlags(&ev_seed, cudaEventDisableTiming));
            CK(cudaEventRecord(ev_seed, devs[(size_t) d_dev].st));
        }
        // the dense weights, each device's layers by a thread of its own
        {
            std::vector<std::string> errs(devs.size());
            std::vector<std::thread> th;
            for (size_t di = 0; di < devs.size(); ++di)
                th.emplace_back([this, di, &errs] {
                    std::string e;
                    if (!load_dense((int) di, e)) errs[di] = e;
                });
            for (auto& t : th) t.join();
            for (auto& e : errs)
                if (!e.empty()) {
                    err = e;
                    return false;
                }
        }
        if (o.verbose)
            std::fprintf(stderr, "ds4: dense weights loaded in %.1f s\n", (now_ms() - t0) / 1000.0);
        // the markov steps over a head group of two or more (DS4_MK_SPLIT=0: the lead's whole rows, as before): E
        // host-mapped, the steps' slots
        // (only where every member holds its rows of the markov head: a type load_dense does not take keeps the lead's)
        bool mk_rows = g.n_mtp > 0 && head_nm() > 1;
        for (int m : groups[(size_t) devs[(size_t) head_dev].grp].mem) mk_rows = mk_rows && devs[(size_t) m].mk_share != nullptr;
        if (mk_rows && !(std::getenv("DS4_MK_SPLIT") != nullptr && std::atoi(std::getenv("DS4_MK_SPLIT")) == 0)) {
            const Tensor* te = pk.get("mtp." + std::to_string(g.n_mtp - 1) + ".markov_head.embed.weight");
            if (te == nullptr || te->dtype != "BF16" || te->bytes != (uint64_t) g.vocab * g.d_rank * 2) {
                err = "the markov embed: not a BF16 [vocab][rank] table";
                return false;
            }
            if (!hmap(h_mk_embed, (size_t) g.vocab * g.d_rank, err) || !pk.read(*te, h_mk_embed) ||
                !hmap(mk_slots, (size_t) kWin * head_nm() * 4, err)) {
                if (err.empty()) err = "the markov embed: read failed";
                return false;
            }
            devs[(size_t) head_dev].mk_ints = d_ints;   // (the lead's drafted tokens: where the confidence head reads them)
            mk_split = true;
        }

        // the embedding table, pinned on the host
        {
            const Tensor* t = pk.get("embed.weight");
            const size_t nv = (size_t) g.vocab * g.dim;
            const bool q8 = t != nullptr && t->dtype == "Q8_0" && t->bytes == nv / 32 * gg::kQ8Bytes;
            // other block types (a GGUF whose dense tensors are Q4_K / Q5_K / Q6_K: the IST-DAS sets' Q5_K embed.weight):
            // each row through ggml's to_float, to the same BF16 the host table holds
            const int ety = t != nullptr ? gg_type_of(t->dtype) : -1;
            const size_t erb = ety > 0 ? gg::type_row_bytes(ety, g.dim) : 0;
            const bool qrow = t != nullptr && !(t->dtype == "BF16") && erb > 0 && t->bytes == (size_t) g.vocab * erb;
            if (t == nullptr || (!q8 && !qrow && (t->dtype != "BF16" || t->bytes != nv * 2))) {
                err = "embed.weight: not a BF16 (or Q8_0 / a ggml block type) [vocab][dim] table";
                return false;
            }
            if (!halloc(embed_pinned, nv, err)) return false;
            if (qrow) {   // (ggml's decode of each row, rounded to BF16 as the Q8_0 path rounds)
                std::vector<uint8_t> row(erb);
                for (size_t r = 0; r < (size_t) g.vocab; ++r) {
                    if (!pk.read_at(t->file, t->off + r * erb, erb, row.data())) {
                        err = "embed.weight: read failed";
                        return false;
                    }
                    if (!gg::gg_row_to_bf16(ety, row.data(), g.dim, embed_pinned + r * g.dim)) {
                        err = "embed.weight: ggml has no decode for " + t->dtype;
                        return false;
                    }
                }
            } else if (q8) {   // (a GGUF's: each value rounded to BF16, a row at a time)
                std::vector<uint8_t> row((size_t) g.dim / 32 * gg::kQ8Bytes);
                for (size_t r = 0; r < (size_t) g.vocab; ++r) {
                    if (!pk.read_at(t->file, t->off + r * row.size(), row.size(), row.data())) {
                        err = "embed.weight: read failed";
                        return false;
                    }
                    const std::vector<float> v = q8_values(row.data(), (size_t) g.dim);
                    for (size_t c = 0; c < (size_t) g.dim; ++c) {
                        uint32_t u;
                        std::memcpy(&u, &v[c], 4);
                        u += 0x7fffu + ((u >> 16) & 1u);   // round to nearest even
                        embed_pinned[r * g.dim + c] = (uint16_t) (u >> 16);
                    }
                }
            } else if (!pk.read(*t, embed_pinned)) {
                err = "embed.weight: read failed";
                return false;
            }
        }
        // the host side of the CPU lane and the Engram rows
        const int cols = (pk.eg_ngram - 1) * pk.eg_heads;
        if (!halloc(h_x, (size_t) kWin * g.dim, err) || !halloc(h_w, (size_t) kWin * kPicks, err) ||
            !halloc(h_part, (size_t) kWin * g.dim, err) || !halloc(h_idx, (size_t) kWin * kPicks, err) ||
            !halloc(h_emb, (size_t) std::max(kWin, chunk) * g.dim, err) ||
            !halloc(h_eg, (size_t) std::max(1, pk.eg_layers) * eg_cap * cols * (pk.eg_hd + pk.eg_hd / 32) + 64, err))
            return false;
        if (pk.eg_layers > 0) {
            // the Engram rows' reader: threads enough to keep an SSD's queue busy (reads from the page cache need few);
            // the row cache a small share of the RAM free (DS4_ENGRAM_CACHE_MB: its size, 0 off; DS4_ENGRAM_THREADS)
            const size_t row = (size_t) pk.eg_hd + pk.eg_hd / 32;
#if defined(_WIN32)
            // (Windows: each thread's ReadFile is synchronous, so the threads are the queue depth - Project Maya: a 28K-token
            //  prompt in one chunk, 10 threads 17.1 s, 64 13.4 s on a 20-core PC.  Here (6 cores, shipped profile) 64 readers
            //  gave +4.5% on a 4K prompt but no decode gain in every pair: DS4_ENGRAM_THREADS=64 turns them on)
#endif
            const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
            int nth = (int) std::min(16u, std::max(4u, hw / 2));
            if (const char* e = std::getenv("DS4_ENGRAM_THREADS")) nth = std::max(1, std::atoi(e));
            const uint64_t avail = ram_avail();
            uint64_t mb = avail == ~uint64_t{0} ? 64 : std::min<uint64_t>(256, std::max<uint64_t>(16, (avail >> 20) / 64));
            if (const char* e = std::getenv("DS4_ENGRAM_CACHE_MB")) mb = std::strtoull(e, nullptr, 10);
            eg_cache.init((size_t) ((mb << 20) / row), row);
            std::vector<std::string> paths;
            for (const std::string& fn : pk.files) paths.push_back(pk.model_dir + "/" + fn);
            // Windows: the Engram tables' files that hold nothing else are unmapped and read unbuffered - an SSD's
            // queue kept by the reader threads instead of the page cache's (whose reads of one file from many threads
            // went ~255K rows a second here); unbuffered reads of a mapped file take a slow path, so only unmapped
            // ones (off by default: -8% on a 4K prompt here; DS4_EG_DIRECT=1 turns it on)
            std::vector<char> eg_direct(pk.files.size(), 0);
#if defined(_WIN32)
            if (pk.gguf && std::getenv("DS4_EG_DIRECT") != nullptr && std::atoi(std::getenv("DS4_EG_DIRECT")) != 0) {
                std::vector<char> eg_file(pk.files.size(), 0), other(pk.files.size(), 0);
                for (const auto& kt : pk.t)
                    if (kt.second.file >= 0 && (size_t) kt.second.file < pk.files.size())
                        (kt.first.find(".engram.embed.") != std::string::npos ? eg_file : other)[(size_t) kt.second.file] = 1;
                for (size_t i = 0; i < pk.files.size(); ++i)
                    if (eg_file[i] && !other[i]) {
                        if (i < pk.maps.size() && pk.maps[i].first != nullptr) {
                            os::unmap(pk.maps[i].first, pk.maps[i].second);
                            pk.maps[i] = {nullptr, 0};
                        }
                        eg_direct[i] = 1;
                        if (o.verbose) std::fprintf(stderr, "ds4: %s unmapped, its Engram rows read unbuffered\n",
                                                    pk.files[i].c_str());
                    }
            }
#endif
            eg_io.start(paths, nth, eg_direct);
            for (int li = 0; li < pk.eg_layers; ++li) {
                eg_win.emplace_back(new EgBatch());
                const std::string id = std::to_string(g.eg_layers[(size_t) li]);
                const Tensor* tw = pk.get("layers." + id + ".engram.embed.weight");
                // The row layout the reader and the dequant take.  A type not here (or no tensor at all) would be read
                // as some other format's bytes - the old code then read the scale tensor of a pack that has none and
                // crashed in eg_issue - so refuse the pack once, with the type's name, before any token is read.
                int fmt;
                if (tw == nullptr) {
                    std::fprintf(stderr, "ds4: Engram layer %s: engram.embed.weight is missing from the pack\n", id.c_str());
                    std::exit(2);
                } else if (tw->dtype == "MXFP8") {
                    fmt = 0;
                } else if (tw->dtype == "Q6_K") {
                    fmt = 1;
                } else if (tw->dtype == "IQ4_NL") {
                    fmt = 2;
                } else if (pk.get("layers." + id + ".engram.embed.scale") != nullptr) {
                    fmt = 3;   // (an old pack: FP8 codes in one tensor, their E8M0 scales in another)
                } else {
                    std::fprintf(stderr,
                                 "ds4: Engram layer %s: engram.embed.weight type %s is not one the Engram reader handles "
                                 "(MXFP8, Q6_K, IQ4_NL)\n",
                                 id.c_str(), tw->dtype.c_str());
                    std::exit(2);
                }
                if (fmt == 1 && pk.eg_hd % 256 != 0) {
                    err = "Engram layer " + id + ": Q6_K rows of " + std::to_string(pk.eg_hd) +
                          " values (not whole superblocks of 256)";
                    return false;
                }
                if (fmt == 2 && pk.eg_hd % 32 != 0) {
                    err = "Engram layer " + id + ": IQ4_NL rows of " + std::to_string(pk.eg_hd) +
                          " values (not whole blocks of 32)";
                    return false;
                }
                eg_fmt.push_back((uint8_t) fmt);
            }
            if (o.verbose)
                std::fprintf(stderr, "ds4: Engram rows read on %d threads, issued with the tokens, collected before "
                                     "their layer; row cache %llu MB (%zu rows)\n",
                             nth, (unsigned long long) mb, eg_cache.capacity());
        }
        prof.assign((size_t) (g.nl + g.n_mtp) * g.n_exp, 0u);
        if (!o.profile.empty()) {
            std::FILE* f = std::fopen(o.profile.c_str(), "rb");
            if (f != nullptr) {
                std::vector<uint32_t> p(prof.size());
                const size_t n = std::fread(p.data(), 4, p.size(), f);
                if (n == p.size() || n == (size_t) g.nl * g.n_exp) std::copy(p.begin(), p.begin() + (long) n, prof.begin());
                std::fclose(f);
            }
        }

        if (chunk_major_phase_enabled) {
            const char* lm = std::getenv("DS4_LAYER_MAJOR");
            const char* flow = std::getenv("DS4_FLOW");
            const char* gc = std::getenv("DS4_LM_GPU_CACHE_MB");
            const char* cn = std::getenv("DS4_COPY_NUMA");
            bool supported = pk.gguf && file_experts && devs.size()==1 && chunk>1024 && chunk<=65280 &&
                o.mmap_experts==0 && store_gb()==0 && !ram_tier && o.vram_experts==1.0 && o.reserve_mib>=4096 &&
                lm && std::strcmp(lm,"0")==0 && (!flow || std::strcmp(flow,"0")!=0) &&
                (!gc || std::strcmp(gc,"0")==0) && (!cn || std::atoi(cn)<2) &&
                prefill_gpu_h_mb==0 && !prefill_copy_ram && prefill_decode_chunk==0 && lm_f16_cache_mb==0;
            for (const Layer& l : L) supported = supported && l.nm==1;
            if (!supported) {
                err = "DS4_PREFILL_CHUNK_MAJOR_PHASE requires one GPU, GGUF, --mmap-experts off, --ssd-cache0, "
                      "--vram-experts1, reserve>=4096MiB, C1025..65280, DS4_LAYER_MAJOR=0, flow on, NH1, "
                      "and no GPUH/LM-cache phase";
                return false;
            }
            chunk_major_phase_pending = true;
        }
        if(prefill_dense_projection_cache) {
            bool supported=chunk_major_phase_pending && devs.size()==1 && NH==1 && pk.gguf &&
                           g.nl==40 && g.dim==5120 && g.q_lora==1280 && g.hd==512 &&
                           g.heads==64 && g.o_groups==8 && g.o_lora==1024;
            int layers=0;
            for(const Layer& l:L) if(!l.mtp) {
                ++layers;
                supported=supported && l.nm==1 && l.dev==0 && l.ahn==64 && l.ogn==8;
                auto role_ok=[](const x3::Mat& m,int n,int k,int type) {
                    // mat_bf16 loads Q8_0 as fmtQ8_0; Mat.K is unused/default0, not GG's type8.
                    const bool q8=type==8;
                    const size_t rb=q8?gg::q8_row_bytes(k):gg::type_row_bytes(type,k);
                    return m.fmt==(q8?gg::kFmtQ8_0:gg::kFmtGG) && (q8 || m.K==type) &&
                           m.raw!=nullptr && m.n==n && m.k==k && (m.ld==0 || m.ld==(int)rb);
                };
                supported=supported && role_ok(l.wq_b,32768,1280,14) && role_ok(l.wo_b,5120,8192,14);
                for(int gi=0;gi<8;++gi) supported=supported && role_ok(l.wo_a[gi],1024,4096,8);
            }
            if(!supported || layers!=40) {
                err="DS4_PREFILL_DENSE_PROJ_CACHE requires single-GPU/NH1 chunk-major GGUF,40 validated "
                    "Q6_K WqB[32768,1280]/WoB[5120,8192] and8 Q8_0 WoA[1024,4096] per trunk layer";
                return false;
            }
        }
        if (prefill_engram_cache) {
            bool supported = chunk_major_phase_pending && devs.size()==1 && NH==1 && pk.gguf;
            int matrices = 0;
            for (const Layer& l : L) if (!l.mtp && l.eg>=0) {
                const x3::Mat& m = l.eg_wkv;
                ++matrices;
                // These are the actual two matrices covered by the full-byte/bit fixture.
                supported = supported && l.nm==1 && l.dev==0 && m.fmt==gg::kFmtGG && m.K==14 &&
                            m.raw!=nullptr && m.n==25600 && m.k==6144 &&
                            (m.ld==0 || m.ld==(int)gg::type_row_bytes(14,m.k));
            }
            if (!supported || matrices!=2) {
                err="DS4_PREFILL_ENGRAM_CACHE requires single-GPU/NH1 chunk-major GGUF and exactly two validated "
                    "Q6_K Engram matrices [25600,6144]"; return false;
            }
        }
        if(prefill_repeat_rearm) {
            bool supported=chunk_major_phase_pending&&cache_epoch_enabled&&record_copied_ram_domains&&
                cold_f16_enabled&&prefill_engram_cache&&prefill_dense_projection_cache&&normalize_prefill_heat&&
                chunk==65280&&devs.size()==1&&NH==1&&o.max_context==1048576&&g.dim==5120&&g.hc==4&&
                g.hd==512&&g.ff==2304&&g.ix_heads==32&&g.ix_hd==128&&
                (register_ram_all||register_ram_mb>0);
            for(const Layer& l:L)supported=supported&&l.nm==1&&l.dev==0&&(l.mtp||l.k_exp==6);
            if(!supported){err="DS4_PREFILL_REPEAT_REARM requires the exact copied-RAM single-GPU Opt17 geometry, C65280, six experts, all three caches, heat normalization, worker epoch/domains and registered-copy policy";return false;}
        }
        if (prefill_gpu_h_mb > 0) {
            bool supported = pk.gguf && file_experts && devs.size() == 1 && chunk > 0 &&
                             o.mmap_experts != 0 && !ram_tier;
            for (const Layer& l : L) supported = supported && l.nm == 1;
            if (const char* v = std::getenv("DS4_FLOW")) supported = supported && std::atoi(v) != 0;
            if (const char* v = std::getenv("DS4_COPY_GB")) supported = supported && std::atof(v) <= 0.0;
            if (const char* v = std::getenv("DS4_STORE_ZC")) supported = supported && std::atoi(v) == 0;
            if (const char* v = std::getenv("DS4_LAYER_MAJOR")) supported = supported && std::atoi(v) != 0;
            if (!supported) {
                err = "DS4_PREFILL_GPU_H_MB requires one GPU, in-place GGUF experts, chunked flow/layer-major, "
                      "no copied RAM tier and no store zero-copy";
                return false;
            }
            if (prefill_copy_ram && store_gb() != 0.0) {
                err = "DS4_PREFILL_COPY_RAM requires --ssd-cache 0 (or DS4_STORE_GB=0); the late copies replace the RAM store";
                return false;
            }
            if (prefill_copy_ram && std::getenv("DS4_COPY_NUMA") && std::atoi(std::getenv("DS4_COPY_NUMA")) >= 2) {
                err = "DS4_PREFILL_COPY_RAM requires NH1; DS4_COPY_NUMA must be absent, 0 or 1";
                return false;
            }
            prefill_phase_pending = true;
        }
        // the chunked prompt path's buffers (before the experts: they take what is left)
        if (chunk > 0) {
            // recommend, never force: a chunk the cards cannot hold in buffers is halved (down to 256 rows) and then the
            // prompt runs in windows of 16, said in one line - the load only fails for a reason that is not this
            std::vector<size_t> na0, sc0;
            for (Dev& d : devs) {
                na0.push_back(d.allocs.size());
                sc0.push_back(d.scratch);
            }
            const size_t nh = host_allocs.size();
            std::string aerr;
            while (chunk > 0 && !alloc_chunks(aerr)) {
                for (size_t di = 0; di < devs.size(); ++di) {
                    Dev& d = devs[di];
                    cudaSetDevice(d.id);
                    cudaDeviceSynchronize();
                    if (di < chunks.size() && chunks[di]) {
                        if (chunks[di]->blas) cublasDestroy(chunks[di]->blas);
                        if (chunks[di]->cs) cudaStreamDestroy(chunks[di]->cs);
                        if (chunks[di]->lm_f16_blas) cublasDestroy(chunks[di]->lm_f16_blas);
                        for (cudaEvent_t e : chunks[di]->lm_f16_meta_done) if (e) cudaEventDestroy(e);
                        if (chunks[di]->cold_f16_blas) cublasDestroy(chunks[di]->cold_f16_blas);
                        for (cudaEvent_t e : chunks[di]->cold_f16_meta_done) if (e) cudaEventDestroy(e);
                    }
                    while (d.allocs.size() > na0[di]) {
                        cudaFree(d.allocs.back());
                        d.allocs.pop_back();
                    }
                    d.scratch = sc0[di];
                }
                while (host_allocs.size() > nh) {
                    cudaFreeHost(host_allocs.back());
                    host_allocs.pop_back();
                }
                chunks.clear();
                const int was = chunk;
                chunk = chunk > 512 ? chunk / 2 / 16 * 16 : 0;
                if (chunk > 0) {
                    eg_cap = (size_t) chunk;
                    std::fprintf(stderr, "ds4: the prompt path's buffers for chunks of %d rows did not fit (%s): trying %d\n",
                                 was, aerr.c_str(), chunk);
                } else {
                    prompt_par = false;
                    eg_cap = (size_t) kWin;
                    tap_cap = kWin;
                    std::fprintf(stderr, "ds4: the prompt path's buffers did not fit (%s): prompts run in windows of %d "
                                         "tokens (slower on long prompts; more free GPU memory or a lower --reserve-mib "
                                         "brings the chunks back)\n", aerr.c_str(), kWin);
                }
                aerr.clear();
            }
            if (chunk > 0 && aerr.empty()) err.clear();
        }
        if(prefill_repeat_rearm) {
            if(chunks.size()!=1||!chunks[0]||chunks[0]->restore_C!=65280||chunks[0]->restore_A!=4096||
               chunks[0]->restore_AI!=128||chunks[0]->C!=65280||chunks[0]->A!=4096||chunks[0]->AI!=128||
               !chunks[0]->cold_f16||!chunks[0]->cold_f16_blas||
               !chunks[0]->eg_weight_cache||!chunks[0]->dense_projection_cache) {
                err="repeat rearm requires successful original65280/4096/128 and all phase arenas";return false;
            }
        }
        if (chunk_major_phase_pending && chunk<=1024) {
            err = "chunk-major first-feed buffers did not fit above1024 rows"; return false;
        }
        if (prefill_decode_chunk > 0 && chunk <= prefill_decode_chunk) {
            err = "DS4_PREFILL_DECODE_CHUNK requires an initial chunk larger than 1024";
            return false;
        }
        if (prefill_phase_pending && (chunk <= 0 || chunks.size() != 1 || !chunks[0]->prefill_gpu_h)) {
            err = "DS4_PREFILL_GPU_H_MB could not reserve a usable chunked residual arena";
            return false;
        }
        // the CPU lane's experts read in place (decided above): how much of them the load reads ahead, from what VRAM
        // now leaves over
        if (file_experts) {
            double room = 0, all = 0;   // the VRAM place_experts will find for experts; every expert's bytes
            for (Dev& d : devs) {
                cudaSetDevice(d.id);
                size_t fr = 0, tot = 0;
                CK(cudaMemGetInfo(&fr, &tot));
                room += std::max(0.0, (double) fr - (double) ((size_t) o.reserve_mib << 20)) * o.vram_experts;
            }
            for (const Layer& l : L) all += (double) l.blob * l.n_exp;
            const double need = std::max(0.0, all - room);
            const uint64_t avail = ram_avail();
            warm_bytes = avail > kRamMargin ? std::min<uint64_t>(avail - kRamMargin, (uint64_t) need) : 0;
            if (avail == ~uint64_t{0}) warm_bytes = (uint64_t) need;
            if (store_gb() > 0) warm_bytes = 0;   // (the store is the RAM tier: nothing to warm in the OS cache)
            if (prefill_phase_pending && prefill_copy_ram) warm_bytes = 0;   // copy only the final CPU remainder AFTER prefill
            std::fprintf(stderr,
                         "ds4: the experts VRAM leaves over (~%.1f GB) are read in place from the model's files%s "
                         "(%.1f GB of RAM free to keep them: the most-routed %.1f GB read ahead; prompts in chunks of %d)\n",
                         need / 1e9, o.mmap_experts > 0 ? " (--mmap-experts on)"
                                     : pk.gguf ? " (--mmap-experts off copies them into RAM)" : ", not copied into RAM",
                         avail == ~uint64_t{0} ? 0.0 : avail / 1e9, warm_bytes / 1e9, chunk);
        }
        // the CPU lane: one pool of workers per NUMA node, each expert split in NH column parts (part X on node X); in
        // place, one part (the files' layout) and one pool over every node's cores (the pages lie where the cache put them)
        cores = node_cores();
        NH = file_experts ? 1 : (int) std::min<size_t>(2, cores.size());
        if (NH > 1 && ((g.ff / NH) % 128 != 0 || (g.dim / NH) % 128 != 0)) NH = 1;
        if (!pk.gguf) make_pools();   // (a GGUF's after its experts: copy_experts decides NH)
        {
            const size_t E = (size_t) g.dim, FF = (size_t) g.ff, GQ = E / 256, DQ = FF / 128, S = kSlots;
            c_xh.resize((size_t) NH * S * 2 * E);
            c_xsb.resize((size_t) NH * S * 2 * (E / 128));
            c_acc.resize((size_t) NH * S * 2 * GQ * (FF / NH));
            c_xhd.resize(S * FF);
            c_xsd.resize(S * (FF / 128));
            c_accd.resize((size_t) NH * S * DQ * (E / NH));
            c_yd.resize(S * E);
        }
        if(!allocate_gpu_misses(err))return false;
        // the experts: VRAM slots in what each device has left, the rest on the host
        if (!place_experts(err, prefill_phase_pending)) return false;
        if (!load_experts(err)) return false;
        if(gpu_misses_enabled) {
            bool valid=copy_mode&&NH==1&&!store&&devs.size()==1;
            for(const Layer& l:L)for(int e=0;e<l.n_exp;++e)
                if(l.slot[(size_t)e]<0)valid=valid&&!l.ram.empty()&&l.ram[(size_t)e]>=0;
            if(!valid){err="DS4_GPU_MISSES requires every CPU expert in the owned copied-RAM domain";return false;}
        }
        if (chunk_major_phase_pending) {
            size_t copied = 0, cpu = 0;
            for (const Layer& l : L) for (int e=0;e<l.n_exp;++e) if(l.slot[(size_t)e]<0) {
                ++cpu; copied += !l.ram.empty() && l.ram[(size_t)e]>=0;
            }
            if (NH!=1 || !copy_mode || copied!=cpu || (size_t)ram_held!=cpu) {
                err = "chunk-major phase requires ALL CPU experts copied before any feed; partial RAM copying refused";
                return false;
            }
            std::fprintf(stderr,"ds4: experimental chunk-major phase: C=%d, %zu/%zu CPU experts copied at startup; "
                                "normal GPU pool retained; post-feed shrink1024 and monotonic growth timed before first token\n",
                         chunk,copied,cpu);
        }
        if (prefill_phase_pending)
            std::fprintf(stderr, "ds4: experimental prefill phase: zero resident experts until the FIRST feed ends; "
                                 "the normal decode pool refill is included before its first token\n");
        if (prefill_phase_pending && prefill_copy_ram)
            std::fprintf(stderr, "ds4: experimental late RAM copy: mapped experts during prefill, then only the final "
                                 "CPU experts copied in NH1 before first token; a failed/Stopped phase consumes the "
                                 "experiment and retains mapping for this process\n");
        if (pk.gguf) make_pools();
        if (chunk > 0) {
            // the CPU lane's parts pinned (the chunks stream them to the GPUs by DMA), each layer's virtual slots
            const double tr = now_ms();
            for (auto& m : host_maps)
                if (cudaHostRegister(m.first, m.second, cudaHostRegisterPortable) != cudaSuccess) {
                    err = "pinning the CPU lane's experts (" + std::to_string(m.second >> 20) + " MiB) failed";
                    return false;
                }
            if (o.verbose) std::fprintf(stderr, "ds4: CPU lane's experts pinned in %.1f s\n", (now_ms() - tr) / 1000.0);
            if (file_experts) {   // in place: each device's staging for its ring, and its readers (DS4_READERS, 8)
                size_t mr = 0;
                for (const Layer& l : L)
                    if (!l.mtp) {
                        for (uint64_t r : l.region_bytes) mr = std::max<size_t>(mr, r);
                        if (l.lay.fmt == gg::kFmtMXFP4) {   // (a GGUF's: read as its blob)
                            mr = std::max(mr, l.blob);
                            for (int r = 0; r < 3; ++r) flow_part[r] = std::max(flow_part[r], mx_part_bytes(l, r));
                        }
                    }
                const char* nr = std::getenv("DS4_READERS");
                const int n_readers = std::max(1, nr != nullptr ? std::atoi(nr) : 8);
                const bool flow_on = std::getenv("DS4_FLOW") == nullptr || std::atoi(std::getenv("DS4_FLOW")) != 0;
                // (a tensor-parallel group's members on the flow only with DS4_FLOW_TP=1 for now: each reads the
                // whole down of every expert it streams, 25 MB an expert for a pair, 18.8 alone - 3x V100, the
                // 17-token prompt from the page cache: 2.5-2.7 s on the flow, 1.3-1.6 s in the group's windows)
                const bool flow_tp = std::getenv("DS4_FLOW_TP") != nullptr && std::atoi(std::getenv("DS4_FLOW_TP")) != 0;
                for (size_t di = 0; di < chunks.size(); ++di) {
                    auto& cp = chunks[di];
                    if (cp == nullptr || cp->ring == nullptr) continue;
                    cp->stage_stride = (mr + 4095) & ~(size_t) 4095;
                    // the flow: every trunk layer of the device a GGUF's (alone or a tensor-parallel group's member)
                    bool fl = flow_on;
                    int n_exp_max = 0;
                    for (const Layer& l : L)
                        if (!l.mtp)
                            for (int m = 0; m < l.nm; ++m)
                                if (member(l, m).dev == (int) di) {
                                    // (a group's members: each its share of the down's blocks in every row)
                                    fl = fl && l.lay.fmt == gg::kFmtMXFP4 &&
                                         (l.nm == 1 || (flow_tp && (g.ff / l.nm) %
                                                        std::max(1, gg::type_block(gg::expert_type(l.lay, 2))) == 0));
                                    n_exp_max = std::max(n_exp_max, l.n_exp);
                                }
                    if (fl)   // (each part read unbuffered from its offset rounded down: a page more each)
                        cp->stage_stride = std::max(cp->stage_stride, flow_stride());
                    cp->S = cp->ring_n;
                    if (!halloc(cp->stage, (size_t) cp->S * cp->stage_stride, err)) return false;
                    cp->readers = std::make_unique<Pool>(n_readers - 1, std::vector<int>{}, 50);
                    if (!fl) continue;
                    cudaSetDevice(devs[di].id);
                    cp->flow = true;
                    cp->dev = (int) di;
                    cp->io = std::make_unique<IoQueue>(n_readers);
                    cp->stage_item.assign((size_t) cp->S, -1);
                    cp->stage_ev.resize((size_t) cp->S);
                    cp->ring_fill.resize((size_t) cp->ring_n);
                    cp->ring_free.resize((size_t) cp->ring_n);
                    for (auto* v : {&cp->stage_ev, &cp->ring_fill, &cp->ring_free})
                        for (cudaEvent_t& e : *v) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
                    // (a region is written again only bc_n batches later - more than a layer's items, so after that
                    // layer's bounds wait, which every copy from it precedes)
                    cp->bc_n = n_exp_max + 2;
                    cp->bc_w = cp->ring_n + 1;
                    if (!halloc(cp->bc_pin, (size_t) cp->bc_n * cp->bc_w, err) ||
                        !dalloc(devs[di], cp->bc_dev, (size_t) cp->bc_n * cp->bc_w, err))
                        return false;
                }
                // every device with layers on the flow: a prompt reads only the experts it picks, once, so a short one
                // takes the chunked path too (one pass, where windows of 16 read the CPU lane's picks again for each
                // window, a page fault at a time) - DS4_CHUNK_MIN still overrides
                bool all_flow = true;
                for (const auto& cp : chunks) all_flow = all_flow && (cp == nullptr || cp->ring == nullptr || cp->flow);
                if (all_flow && std::getenv("DS4_CHUNK_MIN") == nullptr) chunk_min = kFlowChunkMin;
            }
            for (Layer& l : L) {
                if (l.mtp) continue;
                for (int m = 0; m < l.nm; ++m) {   // (each member its own: the flow's members decide them each)
                    Layer& x = member(l, m);
                    if (!halloc(x.vs_pin, (size_t) l.n_exp, err)) return false;
                    if (!dalloc(devs[(size_t) x.dev], x.vs_dev, (size_t) l.n_exp, err)) return false;
                }
            }
        }
        {
            void* m = nullptr;
            if (cudaHostAlloc(&m, sizeof(Req) * kRing, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                err = "host-mapped memory for the decode path's requests";
                return false;
            }
            host_allocs.push_back(m);
            reqs = (Req*) m;
            if (cudaHostAlloc(&m, sizeof(Resp) * kRing, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                err = "host-mapped memory for the decode path's answers";
                return false;
            }
            host_allocs.push_back(m);
            resps = (Resp*) m;
            std::memset((void*) reqs, 0, sizeof(Req) * kRing);
            std::memset((void*) resps, 0, sizeof(Resp) * kRing);
            for (auto& x : layer_of) x.store(-1);
            ahead_fl.assign(L.size() * 1024, 0);
            ahead_rank.assign(L.size() * 1024, 0);
            ahead_for.assign(L.size(), 0);
            const int core = !cores.empty() && !cores[0].empty() ? cores[0][0] : -1;
            service = std::thread([this, core] {
                if (core >= 0) pin_self(core);
                serve_lane();
            });
            if (ahead_mode > 0) pf_th = std::thread([this] { prefetch_loop(); });
        }
        if (o.expert_cache) {
            size_t maxb = 0;
            for (const Layer& l : L) maxb = std::max(maxb, l.blob);
            const int nt = std::max(1, std::min(c_inflight, std::getenv("DS4_CACHE_THREADS") ? std::atoi(std::getenv("DS4_CACHE_THREADS")) : 1));
            sw_bufs.resize((size_t) nt);
            for (SwapBufs& B : sw_bufs) {
                if (!halloc(B.a, maxb, err) || !halloc(B.b, maxb, err) || !halloc(B.h, maxb, err)) return false;
                for (Dev& d : devs) {
                    cudaSetDevice(d.id);
                    cudaStream_t cs = nullptr;
                    CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
                    B.st.push_back(cs);
                }
            }
            for (SwapBufs& B : sw_bufs) cache_ths.emplace_back([this, &B] { cache_loop(B); });
        }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            CK(cudaStreamSynchronize(d.st));
        }
        if (par || chunk > 0)   // (the windows' threads; the prompt path's, whatever mode the windows take)
            for (size_t di = 0; di < devs.size(); ++di) {
                workers.push_back(std::make_unique<Worker>());
                workers.back()->start();
            }
        calibrate_paths();
        init_store();      // (last: the store's arena is registered with the driver - page-locked memory the other pinned allocations must come before)
        preload_store();
        if (o.verbose) std::fprintf(stderr, "ds4: loaded in %.1f s\n%s", (now_ms() - t0) / 1000.0, describe().c_str());
        reset();
        if(prefill_repeat_rearm&&!chunks[0]->flow) {
            err="repeat rearm requires the initialized single-GPU streaming flow";return false;
        }
        if(!register_copied_ram_for_prefill(err))return false;
        return true;
    }


    // the caches' row bytes, packed as DeepSeek keeps them (ds4_kernels.cuh): the window ring's (FP8), the compressed
    // KV's (FP4 + E4M3 scales), the index keys' (FP4 + power-of-two scales)
    size_t win_rb() const { return kv_row_bytes(kKvFp8, g.hd); }
    size_t ckv_rb() const { return kv_row_bytes(kKvFp4, g.hd); }
    size_t ik_rb() const { return kv_row_bytes(kKvFp4p, g.ix_hd); }

    // ---- the chunked prompt path
    size_t x_cap() const { return (size_t) std::max<int>(kWin, chunk + 2); }   // a window's / chunk's new source rows
    size_t chunk_hand_bytes() const {
        return (size_t) chunk * ((size_t) g.hc * g.dim * 4 + 8 * 4 + topk_stride() * 4 + cand_stride() * 4);
    }

    // the staging ring's size (MB) for the experts a chunk streams from the CPU lane: 400 where a good part of them are
    // streamed; 16 where the cards hold most of the experts (they are within 30% of fitting) (a 4-card box) - the ring's VRAM is then worth more
    // as expert slots, and the decode after a prompt keeps the windowed path's speed (R9700 x4: 29.6 -> 31.7 tok/s)
    size_t ring_mb_auto() const {
        double exp_b = 0, other_b = 0, room = 0;
        for (const auto& kt : pk.t) {
            const std::string& n = kt.first;
            if (n.find(".engram.embed.") != std::string::npos || n.rfind("vision.", 0) == 0 || n == "embed.weight" ||
                (!o.dspark && n.rfind("mtp.", 0) == 0))
                continue;
            (n.find(".ffn.experts.") != std::string::npos ? exp_b : other_b) += (double) kt.second.bytes;
        }
        for (const Dev& d : devs)
            room += std::max(0.0, (double) d.free0 - (double) ((size_t) o.reserve_mib << 20) - 1.5 * (1 << 30)) * o.vram_experts;
        const bool fit = exp_b > 0 && exp_b <= 1.3 * std::max(1.0, room - other_b);   // (R9700 x4: 1.16 -> ring 16: decode -1.8% after a prompt, 64: -4.4%, 400: -4.6%)
        if (o.verbose)
            std::fprintf(stderr, "ds4: staging ring: experts %.1f GB, VRAM room %.1f GB, dense %.1f GB -> %d MB\n", exp_b / 1e9,
                         room / 1e9, other_b / 1e9, fit ? 16 : 400);
        return fit ? 16 : 400;
    }
    bool alloc_topk_scratch(Dev& d, TopkScratch& scratch, size_t rows, std::string& err) {
        const char* flag = std::getenv("DS4_TOPK_MULTI");
        if (flag == nullptr || std::strcmp(flag, "1") != 0) return true;
        constexpr size_t parts = 32;
        // Every pointer joins d.allocs through dalloc, including partial failure.
        // Rows/parts are published only when the complete POD is usable. These
        // allocations remain alive through decode, capture and phase promotion.
        if (!dalloc(d, scratch.states, rows, err) ||
            !dalloc(d, scratch.hist, rows * parts * 256, err) ||
            !dalloc(d, scratch.counts, rows * parts, err) ||
            !dalloc(d, scratch.offsets, rows * parts, err)) return false;
        scratch.rows = (int) rows;
        scratch.parts = (int) parts;
        if (o.verbose) std::fprintf(stderr,
            "ds4: multi-CTA TopK CUDA%d: %d rows,32 partitions,large arrays>=131072 only\n", d.id, scratch.rows);
        return true;
    }
    bool alloc_chunks(std::string& err) {
        chunks.resize(devs.size());
        const int D = g.dim, H = g.hc, HD = g.hd;
        const int cols = (pk.eg_ngram - 1) * pk.eg_heads, ehd = pk.eg_hd;
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[di];
            cudaSetDevice(d.id);
            chunks[di] = std::make_unique<Chunk>();
            Chunk& c = *chunks[di];
            const int nm = (int) groups[(size_t) d.grp].mem.size();
            const int NHL = g.heads / head_members(nm), HQ = NHL * HD, OG = g.o_groups / head_members(nm);
            int FF = 0, ff0_ = 0;
            split128(g.ff, nm, 0, ff0_, FF);   // (the widest member's share: member 0 takes the odd blocks)
            const size_t C = (size_t) chunk;
            c.C = chunk;
            c.A = std::max(64, (int) (((size_t) 8 << 20) / (size_t) HQ) / 64 * 64);
            // Opt-in larger dense-attention GEMM slabs. All queries, keys and
            // causal rows stay unchanged; buffers are allocated before experts.
            if (const char* v = std::getenv("DS4_ATTN_ROWS")) {
                const int requested = std::atoi(v);
                if (requested < 64 || requested > 8192 || requested % 64 != 0) {
                    err = "DS4_ATTN_ROWS must be a multiple of 64 between 64 and 8192";
                    return false;
                }
                c.A = std::min(chunk, requested);
                std::fprintf(stderr, "ds4: dense attention slab CUDA%d: %d rows (DS4_ATTN_ROWS)\n", d.id, c.A);
            }
            // (the indexer's slab: its scores rows are the context wide - about 64 MB of them)
            c.AI = std::max(32, std::min(128, (int) (((size_t) 64 << 20) / (((size_t) o.max_context + 64) * 4)) / 32 * 32));
            // Opt-in indexer query slabs. Generated Rows, native score tiles,
            // the scalar fallback and TopK all keep their per-query indexing.
            // This also enlarges the persistent score/Q/block-max buffers;
            // the index Q GEMM shape and call-wide admission guard can change.
            if (const char* v = std::getenv("DS4_INDEX_ROWS")) {
                char* end = nullptr;
                const long requested = std::strtol(v, &end, 10);
                if (end == v || *end != '\0' || requested < 32 || requested > 256 || requested % 32 != 0) {
                    err = "DS4_INDEX_ROWS must be a multiple of 32 between 32 and 256";
                    return false;
                }
                c.AI = std::min(chunk, (int) requested);
                std::fprintf(stderr, "ds4: index score slab CUDA%d: %d rows (DS4_INDEX_ROWS)\n", d.id, c.AI);
            }
            c.w16_elems = (size_t) 8 << 20;
            bool has_eg = false, has_first = false;
            size_t stride = 0;
            for (const Layer& l : L)
                for (int m = 0; m < l.nm; ++m)
                    if (member(l, m).dev == (int) di && !l.mtp) {
                        has_eg = has_eg || l.eg >= 0;
                        has_first = has_first || l.il == 0;
                        stride = std::max(stride, align256(unit(member(l, m))));
                    }
            const size_t A = (size_t) c.A, AI = (size_t) c.AI, ss = (size_t) o.max_context + 64;
            if(prefill_dense_projection_cache && chunk_major_phase_pending) {
                constexpr size_t values=(size_t)32768*1280+(size_t)8*1024*4096+(size_t)5120*8192;
                constexpr size_t bytes=values*sizeof(uint16_t); // exactly224MiB, bounded by fixed role admission
                size_t free_b=0,total_b=0;
                CK(cudaMemGetInfo(&free_b,&total_b));
                const size_t reserve=(size_t)std::max<int64_t>(0,o.reserve_mib)<<20;
                if(free_b>reserve && free_b-reserve>=bytes) {
                    if(!dalloc(d,c.dense_projection_cache,values,err))return false;
                    c.dense_projection_cache_bytes=align256(bytes);
                    std::fprintf(stderr,"ds4: experimental first-prefill dense projection cache: %.1f MiB CUDA%d, "
                                        "10 matrices/19 original tiled GEMMs per slab; retired before decode pool growth\n",
                                 c.dense_projection_cache_bytes/1048576.0,d.id);
                } else {
                    std::fprintf(stderr,"ds4: experimental first-prefill dense projection cache: no224MiB room "
                                        "beside the initial reserve; original GEMM path retained\n");
                }
            }
            if (prefill_engram_cache && chunk_major_phase_pending && has_eg) {
                const size_t values = (size_t)25600 * 6144; // checked fixed admission above; 300 MiB maximum
                if (!dalloc(d,c.eg_weight_cache,values,err)) return false;
                c.eg_weight_cache_bytes=align256(values*sizeof(uint16_t));
                std::fprintf(stderr,"ds4: experimental first-prefill Engram weight cache: %.1f MiB CUDA%d, "
                                    "one matrix, original20x1280 GEMMs; retired before first decode pool growth\n",
                             c.eg_weight_cache_bytes/1048576.0,d.id);
            }
            bool ok = dalloc(d, c.h, C * H * D, err) && dalloc(d, c.x, C * D, err) && dalloc(d, c.acc, C * D, err) &&
                      dalloc(d, c.x16, C * D, err) && dalloc(d, c.qr, C * g.q_lora, err) &&
                      dalloc(d, c.qr16, C * g.q_lora, err) && dalloc(d, c.kvr, C * HD, err) &&
                      dalloc(d, c.ckr, C * HD, err) && dalloc(d, c.cgr, C * HD, err) &&
                      dalloc(d, c.latent, (C + 2) * HD, err) && dalloc(d, c.lat16, (C + 2) * HD, err) &&
                      dalloc(d, c.ikr, (C + 2) * g.ix_hd, err) && dalloc(d, c.ext, ((size_t) g.win + C) * HD, err) &&
                      dalloc(d, c.q, A * HQ, err) && dalloc(d, c.o, A * HQ, err) && dalloc(d, c.o16, A * HQ, err) &&
                      dalloc(d, c.oa, A * OG * g.o_lora, err) && dalloc(d, c.oa16, A * OG * g.o_lora, err) &&
                      dalloc(d, c.iq, AI * g.ix_heads * g.ix_hd, err) && dalloc(d, c.iw, C * 64, err) &&
                      dalloc(d, c.scores, AI * ss, err) && dalloc(d, c.bm, AI * bm_stride(), err) &&
                      dalloc(d, c.topk, C * topk_stride(), err) && dalloc(d, c.cand, C * cand_stride(), err) &&
                      dalloc(d, c.mix, C * 32, err) && dalloc(d, c.a_pre, C * 8, err) && dalloc(d, c.a_post, C * 8, err) &&
                      dalloc(d, c.a_comb, C * 64, err) && dalloc(d, c.f_pre, C * 8, err) &&
                      dalloc(d, c.f_post, C * 8, err) && dalloc(d, c.f_comb, C * 64, err) && dalloc(d, c.pre, C * 8, err) &&
                      dalloc(d, c.glog, C * 1024, err) && dalloc(d, c.ridx, C * kPicks, err) &&
                      dalloc(d, c.rw, C * kPicks, err) && dalloc(d, c.bounds, 2048, err) &&
                      dalloc(d, c.row_tok, C * kPicks, err) && dalloc(d, c.pos, C * kPicks, err) &&
                      dalloc(d, c.sg, C * FF, err) && dalloc(d, c.su, C * FF, err) && dalloc(d, c.sh16, C * FF, err) &&
                      dalloc(d, c.h16, C * FF, err) && dalloc(d, c.y, C * D, err) && dalloc(d, c.w16, c.w16_elems, err) &&
                      halloc(c.bounds_pin, 2048, err);
            if (!ok || !alloc_topk_scratch(d, c.topk_scr, AI, err)) return false;
            if (has_first && !dalloc(d, c.emb, C * D, err)) return false;
            if (has_eg) {
                const size_t ES = 256, per = (size_t) cols * (ehd + ehd / 32);
                if (!dalloc(d, c.eg_x, ES * cols * ehd, err) || !dalloc(d, c.eg_x16, ES * cols * ehd, err) ||
                    !dalloc(d, c.eg_kv, ES * (H + 1) * D, err) || !dalloc(d, c.eg_rows, C * per, err))
                    return false;
            }
            if (stride > 0) {   // DS4_RING_MB (400) of slots in kParts batches
                const size_t mb = std::getenv("DS4_RING_MB") ? (size_t) std::atoi(std::getenv("DS4_RING_MB")) : ring_mb_auto();
                c.ring_stride = stride;
                c.ring_n = std::max(Chunk::kParts, (int) ((mb << 20) / stride) / Chunk::kParts * Chunk::kParts);
                if (!dalloc(d, c.ring, (size_t) c.ring_n * stride, err)) return false;
            }
            CK(cudaStreamCreateWithFlags(&c.cs, cudaStreamNonBlocking));
            for (int i = 0; i < Chunk::kParts; ++i) {
                CK(cudaEventCreateWithFlags(&c.filled[i], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&c.freed[i], cudaEventDisableTiming));
            }
            if (cublasCreate(&c.blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(c.blas, d.st) != CUBLAS_STATUS_SUCCESS) {
                err = "cuBLAS on CUDA" + std::to_string(d.id);
                return false;
            }
            for (int sl = 0; sl < 2; ++sl) {
                if (d.mi == 0 && (!hmap(c.hand[sl], chunk_hand_bytes(), err) || !hmap(c.hand_flag[sl], 32, err))) return false;
                if (has_first && !halloc(c.emb_host[sl], C * D, err)) return false;
                if (has_eg && !halloc(c.eg_host[sl], C * pk.eg_layers * (size_t) cols * (ehd + ehd / 32), err)) return false;
                CK(cudaEventCreateWithFlags(&c.emb_ev[sl], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&c.eg_ev[sl], cudaEventDisableTiming));
            }
            if (d.mi == 0 && !hmap(c.hand_ack, (size_t) 2 * 8 * 32, err)) return false;
            // An explicit experimental budget, in MiB. It is reserved HERE, before persistent expert placement;
            // their existing cudaMemGetInfo budget consequently leaves the arena and --reserve-mib untouched.
            // Only a single-card GGUF flow can use it. No allocation or traversal change when the flag is absent.
            const char* lm_gpu_value = std::getenv("DS4_LM_GPU_CACHE_MB");
            if (lm_gpu_value == nullptr && prefill_phase_pending) lm_gpu_value = "8192";
            if (const char* value = lm_gpu_value) {
                char* tail = nullptr;
                const unsigned long long asked_mb = std::strtoull(value, &tail, 10);
                if (value[0] != '-' && tail != value && *tail == '\0' && asked_mb > 0) {
                    if (pk.gguf && devs.size() == 1 && nm == 1) {
                        size_t free_b = 0, total_b = 0;
                        CK(cudaMemGetInfo(&free_b, &total_b));
                        const size_t reserve = (size_t) std::max<int64_t>(0, o.reserve_mib) << 20;
                        const size_t asked = (size_t) std::min<unsigned long long>(asked_mb, total_b >> 20) << 20;
                        const size_t room = free_b > reserve ? free_b - reserve : 0;
                        const size_t bytes = std::min(asked, room) / 256 * 256;
                        if (bytes > 0) {
                            if (!dalloc(d, c.lm_gpu, bytes, err)) return false;
                            c.lm_gpu_bytes = bytes;
                            std::fprintf(stderr, "ds4: experimental layer-major GPU expert cache: %.1f MiB reserved "
                                                 "before persistent expert placement; original ring/reduction order retained\n",
                                         (double) bytes / 1048576.0);
                        } else
                            std::fprintf(stderr, "ds4: experimental layer-major GPU expert cache: no room beside the reserve; off\n");
                    } else
                        std::fprintf(stderr, "ds4: experimental layer-major GPU expert cache: only single-card GGUF flows; off\n");
                }
            }
            if (chunk_major_phase_pending && cold_f16_enabled) {
#if defined(STRATA_USE_HIP)
                err = "bounded cold F16 prefill requires CUDA TC/F16 operands and F32 outputs";
                return false;
#else
                int major = 0;
                CK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, d.id));
                const char* no_tc = std::getenv("STRATA_DS4_NO_TC");
                if (major < 7 || (no_tc && no_tc[0]=='1')) {
                    err = "bounded cold F16 prefill requires the original CUDA TC/F16 contract";
                    return false;
                }
                const size_t unit_values = (size_t)3 * D * FF;
                const size_t cache_b = align256((size_t)Chunk::kColdF16Experts * unit_values * sizeof(uint16_t));
                const size_t y_b = align256(C * D * sizeof(float));
                const size_t meta_stride = align256((size_t)Chunk::kColdF16Experts * sizeof(F16PhaseRow)) +
                                           align256((size_t)8 * Chunk::kColdF16Experts * sizeof(unsigned long long));
                const size_t meta_b = meta_stride * Chunk::kColdF16MetaSlots;
                const size_t workspace_b = (size_t)32 << 20;
                size_t free_b = 0, total_b = 0;
                CK(cudaMemGetInfo(&free_b,&total_b));
                const size_t reserve = (size_t)std::max<int64_t>(4096,o.reserve_mib) << 20;
                const size_t need = cache_b + y_b + meta_b + workspace_b;
                if (free_b <= reserve || need > free_b-reserve) {
                    err = "bounded cold F16 cache/workspace does not fit beside configured reserve";
                    return false; // ordinary allocation retry may choose a smaller chunk, never an untracked arena
                }
                if (!dalloc(d,c.cold_f16,(size_t)Chunk::kColdF16Experts * unit_values,err) ||
                    !dalloc(d,c.cold_f16_y,C * D,err) || !dalloc(d,c.cold_f16_meta_dev,meta_b,err) ||
                    !dalloc(d,c.cold_f16_workspace,workspace_b,err) || !halloc(c.cold_f16_meta_host,meta_b,err))
                    return false;
                c.cold_f16_bytes=cache_b; c.cold_f16_unit=unit_values; c.cold_f16_y_bytes=y_b;
                c.cold_f16_meta_bytes=meta_b; c.cold_f16_meta_stride=meta_stride;
                c.cold_f16_workspace_bytes=workspace_b;
                if (cublasCreate(&c.cold_f16_blas)!=CUBLAS_STATUS_SUCCESS ||
                    cublasSetStream(c.cold_f16_blas,d.st)!=CUBLAS_STATUS_SUCCESS ||
                    cublasSetPointerMode(c.cold_f16_blas,CUBLAS_POINTER_MODE_HOST)!=CUBLAS_STATUS_SUCCESS ||
                    cublasSetMathMode(c.cold_f16_blas,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)!=CUBLAS_STATUS_SUCCESS ||
                    cublasSetAtomicsMode(c.cold_f16_blas,CUBLAS_ATOMICS_NOT_ALLOWED)!=CUBLAS_STATUS_SUCCESS ||
                    cublasSetWorkspace(c.cold_f16_blas,c.cold_f16_workspace,workspace_b)!=CUBLAS_STATUS_SUCCESS) {
                    err="bounded cold F16 cuBLAS handle/workspace"; return false;
                }
                for(cudaEvent_t& e:c.cold_f16_meta_done) CK(cudaEventCreateWithFlags(&e,cudaEventDisableTiming));
                std::fprintf(stderr,"ds4: experimental bounded cold F16:8 experts %.1f MiB, paddedY %.1f MiB, "
                                    "fixed cuBLAS workspace32 MiB, rule=%s; allocated before resident placement, "
                                    "same F16 operands/F32 GU+down, original light/one-combine; speed unproven\n",
                             cache_b/1048576.0,y_b/1048576.0,coldf16::name(cold_f16_rule));
#endif
            }
            if (prefill_phase_pending && lm_f16_cache_mb > 0) {
#if defined(STRATA_USE_HIP)
                std::fprintf(stderr, "ds4: experimental F16 expert cache: CUDA TC contract only; original path retained\n");
#else
                int major = 0;
                CK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, d.id));
                const char* no_tc = std::getenv("STRATA_DS4_NO_TC");
                const bool tc_contract = major >= 7 && !(no_tc != nullptr && no_tc[0] == '1');
                int max_exp = 0;
                for (const Layer& l : L) if (!l.mtp) max_exp = std::max(max_exp, l.n_exp);
                const size_t unit_values = (size_t) 3 * D * FF;
                const size_t cache_b = align256((size_t) max_exp * unit_values * sizeof(uint16_t));
                const size_t y_b = align256(C * D * sizeof(float));
                const size_t meta_stride = align256((size_t) max_exp * sizeof(F16PhaseRow)) +
                                           align256((size_t) 8 * max_exp * sizeof(unsigned long long));
                const size_t meta_b = meta_stride * Chunk::kF16MetaSlots;
                size_t free_b = 0, total_b = 0;
                CK(cudaMemGetInfo(&free_b, &total_b));
                const size_t reserve = (size_t) std::max<int64_t>(0, o.reserve_mib) << 20;
                const size_t one_h = align256(C * (size_t) H * D * sizeof(float));
                const size_t room = free_b > reserve + one_h ? free_b - reserve - one_h : 0;
                if (!tc_contract || (lm_f16_cache_mb << 20) < cache_b || cache_b + y_b + meta_b > room) {
                    std::fprintf(stderr, "ds4: experimental F16 expert cache: full-layer/cache/workspace/TC "
                                         "requirements unavailable; original TC retained (need %.1f MiB cache + %.1f MiB "
                                         "workspace, room %.1f MiB beside reserve and one H chunk)\n",
                                 cache_b / 1048576.0, (y_b + meta_b) / 1048576.0, room / 1048576.0);
                } else {
                    if (!dalloc(d, c.lm_f16, (size_t) max_exp * unit_values, err) ||
                        !dalloc(d, c.lm_f16_y, C * D, err) || !dalloc(d, c.lm_f16_meta_dev, meta_b, err) ||
                        !halloc(c.lm_f16_meta_host, meta_b, err)) return false;
                    c.lm_f16_bytes = cache_b;
                    c.lm_f16_unit = unit_values;
                    c.lm_f16_y_bytes = y_b;
                    c.lm_f16_meta_bytes = meta_b;
                    c.lm_f16_meta_stride = meta_stride;
                    c.lm_f16_meta_cap = max_exp;
                    if (cublasCreate(&c.lm_f16_blas) != CUBLAS_STATUS_SUCCESS ||
                        cublasSetStream(c.lm_f16_blas, d.st) != CUBLAS_STATUS_SUCCESS ||
                        cublasSetMathMode(c.lm_f16_blas, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION) != CUBLAS_STATUS_SUCCESS ||
                        cublasSetAtomicsMode(c.lm_f16_blas, CUBLAS_ATOMICS_NOT_ALLOWED) != CUBLAS_STATUS_SUCCESS) {
                        err = "experimental F16 phase cuBLAS handle";
                        return false;
                    }
                    for (cudaEvent_t& e : c.lm_f16_meta_done) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
                    std::fprintf(stderr, "ds4: experimental F16 expert cache: %.1f MiB active layer, %.1f MiB extra "
                                         "workspace; original set ranges/combiner, padded rows <= %d, CUDA F16/F32 only\n",
                                 cache_b / 1048576.0, (y_b + meta_b) / 1048576.0, c.C);
                }
#endif
            }
            if (prefill_phase_pending) {
                size_t free_b = 0, total_b = 0;
                CK(cudaMemGetInfo(&free_b, &total_b));
                const size_t reserve = (size_t) std::max<int64_t>(0, o.reserve_mib) << 20;
                const size_t row_b = (size_t) g.hc * g.dim * sizeof(float);
                const size_t asked = std::min(prefill_gpu_h_mb << 20, (size_t) o.max_context * row_b);
                const size_t room = (free_b > reserve ? free_b - reserve : 0) / 256 * 256;
                const size_t rows = std::min(asked, room) / row_b;
                if (c.lm_gpu == nullptr || c.lm_gpu_bytes < stride || rows < C) {
                    err = "DS4_PREFILL_GPU_H_MB needs a compressed layer cache and at least one full F32 H chunk "
                          "beside --reserve-mib";
                    return false;
                }
                if (!dalloc(d, c.prefill_gpu_h, rows * (size_t) g.hc * g.dim, err)) return false;
                c.prefill_gpu_h_bytes = align256(rows * row_b);
                c.prefill_gpu_h_rows = rows;
                std::fprintf(stderr, "ds4: experimental prefill F32 H: %.1f MiB (%zu rows) reserved, requested %zu MiB; "
                                     "whole cached chunks alias GPU slices, remaining fields retain host handoffs\n",
                             (double) c.prefill_gpu_h_bytes / 1048576.0, rows, prefill_gpu_h_mb);
            }
        }
        if(cache_pool_shrink_probe_enabled||prefill_repeat_rearm) {
            for(auto& owned:chunks)if(owned) {
                if(owned->restore_C||owned->restore_A||owned->restore_AI) {
                    err="original workspace geometry already published";return false;
                }
                owned->restore_C=owned->C;owned->restore_A=owned->A;owned->restore_AI=owned->AI;
            }
        }
        return true;
    }

    // y[r][c] = x16[r] . W[:, c] (+ beta y) for r < rows: W reconstructed to FP16 in whole 128-column blocks, cuBLAS
    bool gemm(Dev& d, Chunk& c, const x3::Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta,
              std::string& err) {
        if (rows <= 0) return true;
        const int cols = std::max(128, (int) std::min<size_t>((size_t) m.n, c.w16_elems / (size_t) m.k) / 128 * 128);
        for (int c0 = 0; c0 < m.n; c0 += cols) {
            const int nc = std::min(cols, m.n - c0);
            if (m.fmt != 0) gg::rows(nc == m.n ? m : gg::view(m, c0, nc), c.w16, false, d.st);   // (a GGUF's)
            else x3::reconstruct_rows(nc == m.n ? m : x3::view(m, c0, nc), c.w16, false, d.st);
            const float one = 1.0f;
            const cublasStatus_t st = cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, nc, rows, m.k, &one, c.w16, CUDA_R_16F,
                                                   m.k, x16, CUDA_R_16F, ldx, &beta, y + c0, CUDA_R_32F, ldy,
                                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (st != CUBLAS_STATUS_SUCCESS) {
                err = "cublasGemmEx: status " + std::to_string((int) st);
                return false;
            }
        }
        return true;
    }
    // The SAME tile sizes, alpha/beta, ldx/ldy, c.blas and compute mode as gemm(). Only weight lifetime changes.
    bool gemm_cached_dense_projection(Dev& d,Chunk& c,const Layer& l,int role,int64_t p0,int total_rows,
                                      const x3::Mat& m,const uint16_t* x16,int ldx,int rows,float* y,int ldy,
                                      float beta,std::string& err) {
        if(!prefill_dense_projection_cache || !chunk_major_phase_pending || !c.dense_projection_cache ||
           total_rows<=c.A || rows<=0)
            return gemm(d,c,m,x16,ldx,rows,y,ldy,beta,err);
        const bool is_q=role==0,is_a=role>=1&&role<=8,is_b=role==9;
        const int n=is_q?32768:is_a?1024:5120,k=is_q?1280:is_a?4096:8192,type=is_a?8:14;
        const size_t offset=is_q?0:is_a?(size_t)32768*1280+(size_t)(role-1)*1024*4096:
                                       (size_t)32768*1280+(size_t)8*1024*4096;
        constexpr size_t capacity=(size_t)224<<20;
        const int expected_ldx=is_q?1280:is_a?32768:8192,expected_ldy=is_q?32768:is_a?8192:5120;
        if((!is_q&&!is_a&&!is_b) || NH!=1 || devs.size()!=1 || l.mtp || l.nm!=1 || l.dev!=c.dev ||
           m.fmt!=(is_a?gg::kFmtQ8_0:gg::kFmtGG) || (!is_a&&m.K!=type) || !m.raw || m.n!=n || m.k!=k ||
           (m.ld!=0&&m.ld!=(int)(is_a?gg::q8_row_bytes(k):gg::type_row_bytes(type,k))) ||
           c.dense_projection_cache_bytes!=capacity || offset+(size_t)n*k>capacity/sizeof(uint16_t) ||
           ldx!=expected_ldx || ldy!=expected_ldy || rows>c.A || total_rows>c.C || beta!=0.0f) {
            // A benign admission refusal touches no outputs; the complete original call remains available.
            if(role>=0&&role<10)c.dense_projection_valid[role]=false;
            return gemm(d,c,m,x16,ldx,rows,y,ldy,beta,err);
        }
        const x3::Mat& tag=c.dense_projection_tags[role];
        const bool hit=c.dense_projection_valid[role] && c.dense_projection_layers[role]==l.il &&
            c.dense_projection_p0[role]==p0 && tag.raw==m.raw && tag.fmt==m.fmt && tag.K==m.K &&
            tag.n==m.n && tag.k==m.k && tag.ld==m.ld && tag.cb==m.cb && tag.trellis==m.trellis &&
            tag.suh==m.suh && tag.svh==m.svh;
        uint16_t* const cached=c.dense_projection_cache+offset;
        if(!hit) {
            c.dense_projection_valid[role]=false;
            const int before=gg::launch_errors();
            gg::rows(m,cached,false,d.st);
            const cudaError_t status=cudaPeekAtLastError();
            if(status!=cudaSuccess || gg::launch_errors()!=before) {
                err=std::string("dense projection cache decode launch: ")+cudaGetErrorString(status);return false;
            }
            c.dense_projection_tags[role]=m;c.dense_projection_layers[role]=l.il;
            c.dense_projection_p0[role]=p0;c.dense_projection_valid[role]=true;
            ++c.dense_projection_fills;
        } else ++c.dense_projection_hits;
        const int cols=std::max(128,(int)std::min<size_t>((size_t)m.n,c.w16_elems/(size_t)m.k)/128*128);
        const float one=1.0f;
        for(int c0=0;c0<m.n;c0+=cols) {
            const int nc=std::min(cols,m.n-c0);
            const cublasStatus_t status=cublasGemmEx(c.blas,CUBLAS_OP_T,CUBLAS_OP_N,nc,rows,m.k,&one,
                cached+(size_t)c0*m.k,CUDA_R_16F,m.k,x16,CUDA_R_16F,ldx,&beta,y+c0,CUDA_R_32F,ldy,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if(status!=CUBLAS_STATUS_SUCCESS) {
                c.dense_projection_valid[role]=false;
                err="cublasGemmEx dense projection cached weights: status "+std::to_string((int)status);return false;
            }
            ++c.dense_projection_gemms;
        }
        return true;
    }
    bool retire_prefill_dense_projection_cache(Dev& d,Chunk& c,std::string& err) {
        for(int r=0;r<10;++r) {
            c.dense_projection_valid[r]=false;c.dense_projection_tags[r]={};
            c.dense_projection_layers[r]=-1;c.dense_projection_p0[r]=-1;
        }
        if(!c.dense_projection_cache) {
            if(c.dense_projection_cache_bytes!=0){err="dense projection cache missing owned pointer";return false;}
            return true;
        }
        cudaSetDevice(d.id);
        const cudaError_t drained=cudaStreamSynchronize(d.st);
        if(drained!=cudaSuccess){err=std::string("dense projection cache retire compute drain: ")+cudaGetErrorString(drained);return false;}
        auto it=std::find(d.allocs.begin(),d.allocs.end(),(void*)c.dense_projection_cache);
        const size_t bytes=align256(c.dense_projection_cache_bytes);
        if(it==d.allocs.end() || std::count(d.allocs.begin(),d.allocs.end(),(void*)c.dense_projection_cache)!=1 || d.scratch<bytes) {
            err="dense projection cache retirement ownership/accounting";return false;
        }
        const cudaError_t status=cudaFree(c.dense_projection_cache);
        if(status!=cudaSuccess){err=std::string("dense projection cache retire: ")+cudaGetErrorString(status);return false;}
        d.allocs.erase(it);d.scratch-=bytes;c.dense_projection_cache=nullptr;c.dense_projection_cache_bytes=0;
        std::fprintf(stderr,"ds4: experimental first-prefill dense projection cache retired %.1f MiB: "
                            "%llu matrix fills/%llu slab hits/%llu original-shape GEMMs; all10 valid tags cleared\n",
                     bytes/1048576.0,(unsigned long long)c.dense_projection_fills,
                     (unsigned long long)c.dense_projection_hits,(unsigned long long)c.dense_projection_gemms);
        return true;
    }
    // Called only for the admitted Engram role, after all inputs land on the SAME d.st.
    // Tagging accepted asynchronous work is safe because every later read/overwrite uses that stream.
    bool gemm_cached_engram(Dev& d, Chunk& c, const Layer& l, int64_t p0,
                            const uint16_t* x16, int ldx, int rows, float* y, int ldy, std::string& err) {
        const x3::Mat& m=l.eg_wkv;
        if (!prefill_engram_cache || !chunk_major_phase_pending || !c.eg_weight_cache)
            return gemm(d,c,m,x16,ldx,rows,y,ldy,0.0f,err);
        if (rows<=0) return true;
        const int cols=std::max(128,(int)std::min<size_t>((size_t)m.n,c.w16_elems/(size_t)m.k)/128*128);
        const size_t bytes=(size_t)m.n*(size_t)m.k*sizeof(uint16_t);
        if (NH!=1 || devs.size()!=1 || l.nm!=1 || l.dev!=c.dev || m.fmt!=gg::kFmtGG || m.K!=14 ||
            !m.raw || m.n!=25600 || m.k!=6144 || (m.ld!=0 && m.ld!=(int)gg::type_row_bytes(14,m.k)) || cols!=1280 ||
            bytes!=c.eg_weight_cache_bytes || ldx!=m.k || ldy!=m.n || rows>256) {
            err="Engram weight cache shape/role/lifetime changed after strict admission"; return false;
        }
        const x3::Mat& tag=c.eg_weight_tag;
        const bool hit=c.eg_weight_valid && c.eg_weight_layer==l.il && c.eg_weight_p0==p0 &&
            tag.raw==m.raw && tag.fmt==m.fmt && tag.K==m.K && tag.n==m.n && tag.k==m.k && tag.ld==m.ld;
        if (!hit) {
            c.eg_weight_valid=false; c.eg_weight_layer=-1; c.eg_weight_p0=-1; c.eg_weight_tag={};
            const int before=gg::launch_errors();
            gg::rows(m,c.eg_weight_cache,false,d.st);
            const cudaError_t status=cudaPeekAtLastError();
            if (status!=cudaSuccess || gg::launch_errors()!=before) {
                err=std::string("Engram weight-cache decode launch: ")+cudaGetErrorString(status); return false;
            }
            // Publish only after successful launch. Completion is guaranteed before each GEMM by d.st ordering.
            c.eg_weight_tag=m; c.eg_weight_layer=l.il; c.eg_weight_p0=p0; c.eg_weight_valid=true;
            ++c.eg_weight_fills;
        } else ++c.eg_weight_hits;
        const float one=1.0f,zero=0.0f;
        for(int c0=0;c0<m.n;c0+=cols) {
            const int nc=std::min(cols,m.n-c0);
            const cublasStatus_t status=cublasGemmEx(c.blas,CUBLAS_OP_T,CUBLAS_OP_N,nc,rows,m.k,
                &one,c.eg_weight_cache+(size_t)c0*m.k,CUDA_R_16F,m.k,x16,CUDA_R_16F,ldx,
                &zero,y+c0,CUDA_R_32F,ldy,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if(status!=CUBLAS_STATUS_SUCCESS) {
                c.eg_weight_valid=false;
                err="cublasGemmEx Engram cached weights: status "+std::to_string((int)status);return false;
            }
            ++c.eg_weight_gemms;
        }
        return true;
    }
    bool retire_prefill_engram_cache(Dev& d, Chunk& c, std::string& err) {
        // Idempotent, including short first feed, Stop, error, load failure and destructor.
        c.eg_weight_valid=false;c.eg_weight_tag={};c.eg_weight_layer=-1;c.eg_weight_p0=-1;
        if(!c.eg_weight_cache) {
            if(c.eg_weight_cache_bytes!=0){err="Engram cache missing owned pointer";return false;}
            return true;
        }
        cudaSetDevice(d.id);
        const cudaError_t drained=cudaStreamSynchronize(d.st);
        if(drained!=cudaSuccess){err=std::string("Engram cache retire compute drain: ")+cudaGetErrorString(drained);return false;}
        auto it=std::find(d.allocs.begin(),d.allocs.end(),(void*)c.eg_weight_cache);
        const size_t bytes=align256(c.eg_weight_cache_bytes);
        if(it==d.allocs.end() || std::count(d.allocs.begin(),d.allocs.end(),(void*)c.eg_weight_cache)!=1 || d.scratch<bytes) {
            err="Engram cache retirement ownership/accounting";return false;
        }
        const cudaError_t status=cudaFree(c.eg_weight_cache);
        if(status!=cudaSuccess){err=std::string("Engram cache retire: ")+cudaGetErrorString(status);return false;}
        d.allocs.erase(it);d.scratch-=bytes;c.eg_weight_cache=nullptr;c.eg_weight_cache_bytes=0;
        std::fprintf(stderr,"ds4: experimental first-prefill Engram weight cache retired %.1f MiB: "
                            "%llu fills/%llu slab hits/%llu original-shape GEMMs; valid tag cleared\n",
                     bytes/1048576.0,(unsigned long long)c.eg_weight_fills,(unsigned long long)c.eg_weight_hits,
                     (unsigned long long)c.eg_weight_gemms);
        return true;
    }
    // y[r][c] = alpha x16[r] . W[c] for F16 weights W [n][k] (the indexer's weights_proj)
    bool gemm_f16w(Chunk& c, const uint16_t* W, int n, int k, const uint16_t* x16, int ldx, int rows, float alpha,
                   float* y, int ldy, std::string& err) {
        const float zero = 0.0f;
        const cublasStatus_t st = cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &alpha, W, CUDA_R_16F, k,
                                               x16, CUDA_R_16F, ldx, &zero, y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) {
            err = "cublasGemmEx (F16 weights): status " + std::to_string((int) st);
            return false;
        }
        return true;
    }
    // hc_mix over a chunk: the dots by an F32 GEMM, then the rms scale
    bool hc_mix_chunk(Dev& d, Chunk& c, const float* h, const float* fn, int T, float* mixes, std::string& err) {
        const int hcd = g.hc * g.dim, n_mix = (2 + g.hc) * g.hc;
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, n_mix, T, hcd, &one, fn, hcd, h, hcd, &zero, mixes, 32) !=
            CUBLAS_STATUS_SUCCESS) {
            err = "cublasSgemm (mHC mixes)";
            return false;
        }
        hc_mix_scale(h, hcd, n_mix, T, g.eps, mixes, d.st);
        return true;
    }

    // expert e of l (on the CPU lane: its NH column parts) -> member m's VRAM copy at dst, in the GPU layout (the full
    // blob, or a group member's part), by DMA from the pinned parts on the copy stream (in place: from `src`, its file
    // region read into pinned staging)
    bool stream_expert(const Layer& l, int m, int e, uint8_t* dst, cudaStream_t cs, std::string& err,
                       const uint8_t* src = nullptr) {
        if (l.lay.fmt == gg::kFmtMXFP4) {   // the blob, or member m's rows / blocks: from `src` (the expert read into
            // pinned staging as its blob) or from the file's mapping
            auto mat_r = [&](int r) { return src != nullptr ? src + l.lay.trellis[r] : mx_mat(l, e, r); };
            bool ok = true;
            if (l.nm == 1) {
                if (src != nullptr)
                    ok = cudaMemcpyAsync(dst, src, l.blob, cudaMemcpyHostToDevice, cs) == cudaSuccess;
                else
                    for (int r = 0; r < 3; ++r)
                        ok = ok && cudaMemcpyAsync(dst + l.lay.trellis[r], mat_r(r), mx_part_bytes(l, r),
                                                   cudaMemcpyHostToDevice, cs) == cudaSuccess;
            } else {
                const int ffh = g.ff / l.nm;
                const size_t gr = gg::type_row_bytes(gg::expert_type(l.lay, 0), g.dim),
                             dr = gg::type_row_bytes(gg::expert_type(l.lay, 2), g.ff),
                             dh = gg::type_row_bytes(gg::expert_type(l.lay, 2), ffh);
                for (int r = 0; r < 2; ++r)
                    ok = ok && cudaMemcpyAsync(dst + l.hlay.trellis[r], mat_r(r) + (size_t) m * ffh * gr,
                                               (size_t) ffh * gr, cudaMemcpyHostToDevice, cs) == cudaSuccess;
                ok = ok && cudaMemcpy2DAsync(dst + l.hlay.trellis[2], dh, mat_r(2) + (size_t) m * dh, dr, dh,
                                             (size_t) g.dim, cudaMemcpyHostToDevice, cs) == cudaSuccess;
            }
            if (!ok) err = "streaming layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
            return ok;
        }
        const auto& parts = l.hh[(size_t) e];
        const int D = g.dim, FF = g.ff;
        auto cp2 = [&](uint8_t* d0, size_t dp, const uint8_t* s0, size_t sp, size_t w, size_t h) {
            return cudaMemcpy2DAsync(d0, dp, s0, sp, w, h, cudaMemcpyHostToDevice, cs) == cudaSuccess;
        };
        auto cp = [&](uint8_t* d0, const uint8_t* s0, size_t n) {
            return cudaMemcpyAsync(d0, s0, n, cudaMemcpyHostToDevice, cs) == cudaSuccess;
        };
        bool ok = true;
        if (NH == 1) {   // one part: the whole expert in exllamav3's layout, its pieces at hl's offsets
            const uint8_t* p = src != nullptr ? src : parts[0];
            if (l.nm == 1) {
                for (int r = 0; r < 3; ++r)
                    ok = ok && cp(dst + l.lay.trellis[r], p + l.hl.trellis[r], l.piece_bytes[r][0]) &&
                         cp(dst + l.lay.suh[r], p + l.hl.suh[r], (size_t) l.hl.k[r] * 2) &&
                         cp(dst + l.lay.svh[r], p + l.hl.svh[r], (size_t) l.hl.n[r] * 2);
            } else {   // member m's part (its FFN columns [f0, f0 + fn)), cut as blob_to_half cuts a blob
                const Layer& x = member(l, m);
                const int f0 = x.ff0, fn = x.ffn;
                for (int r = 0; r < 2; ++r) {
                    const size_t tile = (size_t) 32 * l.lay.K[r], nt = (size_t) FF / 16, nth = (size_t) fn / 16;
                    ok = ok && cp2(dst + x.hlay.trellis[r], nth * tile, p + l.hl.trellis[r] + (size_t) (f0 / 16) * tile,
                                   nt * tile, nth * tile, (size_t) D / 16) &&
                         cp(dst + x.hlay.suh[r], p + l.hl.suh[r], (size_t) D * 2) &&
                         cp(dst + x.hlay.svh[r], p + l.hl.svh[r] + (size_t) f0 * 2, (size_t) fn * 2);
                }
                const size_t row = (size_t) (D / 16) * 32 * l.lay.K[2];
                ok = ok && cp(dst + x.hlay.trellis[2], p + l.hl.trellis[2] + (size_t) (f0 / 16) * row,
                              (size_t) (fn / 16) * row) &&
                     cp(dst + x.hlay.suh[2], p + l.hl.suh[2] + (size_t) f0 * 2, (size_t) fn * 2) &&
                     cp(dst + x.hlay.svh[2], p + l.hl.svh[2], (size_t) D * 2);
            }
        } else if (l.nm == 1) {   // the full blob: each matrix's columns from the two parts side by side
            for (int r = 0; r < 3; ++r) {
                const size_t tile = (size_t) 32 * l.lay.K[r];
                const int k = l.hl.k[r], nh = l.hl.n[r];   // part: k x nh
                for (int X = 0; X < 2; ++X) {
                    ok = ok && cp2(dst + l.lay.trellis[r] + (size_t) X * (nh / 16) * tile, (size_t) (2 * nh / 16) * tile,
                                   parts[(size_t) X] + l.hl.trellis[r], (size_t) (nh / 16) * tile, (size_t) (nh / 16) * tile,
                                   (size_t) (k / 16));
                    ok = ok && cp(dst + l.lay.svh[r] + (size_t) X * nh * 2, parts[(size_t) X] + l.hl.svh[r], (size_t) nh * 2);
                }
                ok = ok && cp(dst + l.lay.suh[r], parts[0] + l.hl.suh[r], (size_t) k * 2);
            }
        } else {   // member m (FFN columns [f0, f0 + fn)): gate / up from the parts holding them; down rows [f0, ..) of
                   // both parts' columns
            const Layer& x = member(l, m);
            const int f0 = x.ff0, fn = x.ffn;
            for (int r = 0; r < 2; ++r) {
                const size_t tile = (size_t) 32 * l.lay.K[r];
                const int hn = l.hl.n[r];   // part X: columns [X hn, (X + 1) hn)
                for (int X = 0; X < 2; ++X) {
                    const int a = std::max(f0, X * hn), b = std::min(f0 + fn, (X + 1) * hn);
                    if (a >= b) continue;
                    ok = ok && cp2(dst + x.hlay.trellis[r] + (size_t) ((a - f0) / 16) * tile, (size_t) (fn / 16) * tile,
                                   parts[(size_t) X] + l.hl.trellis[r] + (size_t) ((a - X * hn) / 16) * tile,
                                   (size_t) (hn / 16) * tile, (size_t) ((b - a) / 16) * tile, (size_t) (D / 16)) &&
                         cp(dst + x.hlay.svh[r] + (size_t) (a - f0) * 2, parts[(size_t) X] + l.hl.svh[r] + (size_t) (a - X * hn) * 2,
                            (size_t) (b - a) * 2);
                }
                ok = ok && cp(dst + x.hlay.suh[r], parts[0] + l.hl.suh[r], (size_t) D * 2);
            }
            const size_t tile = (size_t) 32 * l.lay.K[2];
            const int nh = D / 2;
            for (int X = 0; X < 2; ++X) {
                ok = ok && cp2(dst + x.hlay.trellis[2] + (size_t) X * (nh / 16) * tile, (size_t) (D / 16) * tile,
                               parts[(size_t) X] + l.hl.trellis[2] + (size_t) (f0 / 16) * (nh / 16) * tile,
                               (size_t) (nh / 16) * tile, (size_t) (nh / 16) * tile, (size_t) (fn / 16));
                ok = ok && cp(dst + x.hlay.svh[2] + (size_t) X * nh * 2, parts[(size_t) X] + l.hl.svh[2], (size_t) nh * 2);
            }
            ok = ok && cp(dst + x.hlay.suh[2], parts[0] + l.hl.suh[2] + (size_t) f0 * 2, (size_t) fn * 2);
        }
        if (!ok) err = "streaming layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
        return ok;
    }

    // a device's streamed experts: batches of ring_n / kParts of each of its layers' CPU-lane experts, chunk by chunk
    // in layer order; the copies run at most kParts batches ahead of the compute
    struct Batch {
        int k = 0, il = 0, j0 = 0, n = 0;   // chunk, layer, its host_list range
    };
    struct Streamer {
        std::vector<Batch> b;
        size_t cp = 0, kp = 0;        // the next batch to copy, to compute
        std::atomic<size_t> done{0};  // cp, for read_ahead's thread
    };
    // in place (file_experts): the batches' file regions read ahead of their copies - POSIX_FADV_WILLNEED, the kernel
    // reading them in the background in large requests, where the copies alone fault the pages in a few at a time -
    // at most DS4_READAHEAD_MB (2048) beyond the batch being copied; each region by its layer's first member alone
    void read_ahead(const Streamer& S, int di, const std::atomic<bool>& stop) const {
        static const double cap = (std::getenv("DS4_READAHEAD_MB") ? std::atof(std::getenv("DS4_READAHEAD_MB")) : 2048.0) *
                                  1048576.0;
        std::deque<std::pair<size_t, double>> issued;   // (batch, bytes) not yet copied
        double ahead = 0;
        for (size_t q = 0; q < S.b.size() && !stop.load(std::memory_order_relaxed);) {
            const size_t done = S.done.load(std::memory_order_relaxed);
            while (!issued.empty() && issued.front().first < done) {
                ahead -= issued.front().second;
                issued.pop_front();
            }
            if (ahead > cap) {
                std::this_thread::sleep_for(std::chrono::microseconds(500));
                continue;
            }
            const Batch& bt = S.b[q];
            const Layer& l0 = L[(size_t) bt.il];
            double nb = 0;
            if (l0.dev == di)
                for (int i = 0; i < bt.n; ++i) {
                    const int e = l0.host_list[(size_t) (bt.j0 + i)];
                    if (l0.lay.fmt == gg::kFmtMXFP4) {
                        if (copy_mode && l0.ram[(size_t) e] >= 0) continue;   // (in RAM: nothing to read)
                        for (int r = 0; r < 3; ++r)
                            os::advise(pk.fds[(size_t) l0.mx_file[r]], (l0.mx_off[r] + (uint64_t) e * l0.mx_stride[r]), mx_part_bytes(l0, r), true);
                        nb += (double) l0.blob;
                        continue;
                    }
                    os::advise(pk.fds[(size_t) l0.region_file[(size_t) e]], l0.region_off[(size_t) e], l0.region_bytes[(size_t) e], true);
                    nb += (double) l0.region_bytes[(size_t) e];
                }
            issued.push_back({q++, nb});
            ahead += nb;
        }
    }
    bool copy_ahead(Dev& d, Chunk& c, Streamer& S, int di, std::string& err) {
        const int P = Chunk::kParts;
        while (S.cp < S.b.size() && S.cp < S.kp + P) {
            const Batch& bt = S.b[S.cp];
            const int half = (int) (S.cp % P);
            const Layer& l0 = L[(size_t) bt.il];
            int mi = 0;
            for (int m = 0; m < l0.nm; ++m)
                if (member(l0, m).dev == di) mi = m;
            if (S.cp >= (size_t) P) CK(cudaStreamWaitEvent(c.cs, c.freed[half], 0));
            uint8_t* sb = nullptr;   // in place: the batch's regions in the part's staging slots
            if (c.stage != nullptr) {
                if (S.cp >= (size_t) P) CK(cudaEventSynchronize(c.filled[half]));   // (its last batch's copies out)
                sb = c.stage + (size_t) half * (c.ring_n / P) * c.stage_stride;
                std::atomic<bool> bad{false};
                c.readers->run(bt.n, [&](int i) {
                    const int e = l0.host_list[(size_t) (bt.j0 + i)];
                    if (l0.lay.fmt == gg::kFmtMXFP4 ? !read_mx(l0, e, sb + (size_t) i * c.stage_stride)
                                                    : !pk.read_at(l0.region_file[(size_t) e], l0.region_off[(size_t) e],
                                                                  l0.region_bytes[(size_t) e],
                                                                  sb + (size_t) i * c.stage_stride))
                        bad.store(true);
                });
                if (bad.load()) {
                    err = "reading layer " + std::to_string(l0.il) + "'s experts from the model's files failed";
                    return false;
                }
            }
            for (int i = 0; i < bt.n; ++i) {
                uint8_t* dst = c.ring + ((size_t) half * (c.ring_n / P) + i) * c.ring_stride;
                if (!stream_expert(l0, mi, l0.host_list[(size_t) (bt.j0 + i)], dst, c.cs, err,
                                   sb != nullptr ? sb + (size_t) i * c.stage_stride : nullptr))
                    return false;
                c.stream_bytes += (double) unit(member(l0, mi));
            }
            CK(cudaEventRecord(c.filled[half], c.cs));
            ++S.cp;
            S.done.store(S.cp, std::memory_order_relaxed);
        }
        return true;
    }

    bool registered_ram_role(const Layer& l,int expert,int role,const uint8_t*& source,size_t& bytes) const {
        source=nullptr;bytes=0;
        if(!chunk_major_phase_pending||!registered_copy.published||!copy_mode||NH!=1||store||l.nm!=1||
           expert<0||expert>=l.n_exp||role<0||role>=3||l.il<0||(size_t)l.il>=cp_lay.size()||
           (size_t)expert>=l.ram.size())return false;
        const int slot=l.ram[(size_t)expert];
        if(slot<0||(size_t)slot>=ram_part.size()||!ram_part[(size_t)slot][0])return false;
        source=ram_part[(size_t)slot][0]+cp_lay[(size_t)l.il][(size_t)role];
        bytes=mx_part_bytes(l,role);
        return registered_copy.covers((uintptr_t)source,bytes);
    }

    // Production's same three H2Ds, factored only for registered NH1 sources so the real path is fixture-testable.
    bool dma_registered_copy_roles(Chunk& c,const Layer& l,const Chunk::Item& item,uint8_t* destination,std::string& err) {
        if(!item.registered_copy_source||!chunk_major_phase_pending||!registered_copy.published||
           !copy_mode||NH!=1||store||l.nm!=1){err="registered source outlived its first-prefill geometry";return false;}
        for(int role=0;role<3;++role)
            if(!registered_copy.covers((uintptr_t)item.src[role],mx_read_len(l,role))){
                err="direct H2D role lost registered-range coverage";return false;
            }
        const x3::ExpertLayout& layout=l.lay;
        for(int role=0;role<2;++role)
            CK(cudaMemcpyAsync(destination+layout.trellis[role],item.src[role],mx_read_len(l,role),cudaMemcpyHostToDevice,c.cs));
        CK(cudaMemcpyAsync(destination+layout.trellis[2],item.src[2],mx_part_bytes(l,2),cudaMemcpyHostToDevice,c.cs));
        return true;
    }

    // Exact registration ledger, independent from host_allocs/host_maps and ram_pinned/ram_slots.
    // Normal callers have already drained I/O and both streams before entering this function.
    bool retire_registered_copy_ram(std::string& err) {
        if(gpu_miss_direct_enabled){gpu_misses_decode_pins=false;st.gpu_miss_retained_ranges=st.gpu_miss_retained_bytes=0;}
        if(registered_copy.held.empty())return true;
        const double started=now_ms();
        cudaError_t last=cudaSuccess;
        const bool complete=regcopy::retire(registered_copy,[&](const regcopy::Range& range){
            const cudaError_t status=cudaHostUnregister((void*)range.base);
            if(status!=cudaSuccess)last=status;
            return status==cudaSuccess;
        });
        registered_copy_unregister_ms+=now_ms()-started;
        std::fprintf(stderr,"ds4: experimental registered-copy unregister: %.3f s this call, %.3f s total; "
                            "%zu ranges/%.1f MiB still owned; original pageable CPU copies retained\n",
                     (now_ms()-started)/1000.0,registered_copy_unregister_ms/1000.0,
                     registered_copy.held.size(),registered_copy.bytes()/1048576.0);
        if(!complete){err=std::string("unregistering copied-RAM source: ")+cudaGetErrorString(last);return false;}
        return true;
    }

    // Setup is transactional: until every selected arena is registered and budget-checked, direct admission is off.
    // Recoverable setup failures reverse-unregister all successes, then use the original bounce path for this process.
    bool register_copied_ram_for_prefill(std::string& err) {
        const bool decode_recovery=gpu_misses_recovery_context();
        if(!register_ram_all&&register_ram_mb==0)return true;
#if defined(STRATA_USE_HIP)
        std::fprintf(stderr,"ds4: experimental registered-copy setup: CUDA host registration unavailable; original staging retained\n");
        return true;
#else
        if((!chunk_major_phase_pending&&!decode_recovery)||devs.size()!=1||chunks.size()!=1||!chunks[0]||!chunks[0]->flow||
           !copy_mode||NH!=1||store||ram_chunks.size()!=ram_chunk_bytes.size()||ram_chunks.empty()||
           !registered_copy.held.empty()||registered_copy.published){err="registered-copy phase geometry/ownership";return false;}
        for(const Layer& l:L)for(int e=0;e<l.n_exp;++e)
            if(l.slot[(size_t)e]<0&&(l.ram.empty()||l.ram[(size_t)e]<0)){
                err="registered-copy setup requires every CPU expert copied";return false;
            }
        for(const auto& w:workers)if(!w->idle()){err="registered-copy setup before prompt-worker quiescence";return false;}
        {
            std::lock_guard<std::mutex> lock(sw_mu);
            if(sw_inflight||ram_inflight||!sw_queue.empty()||!sw_ready.empty()||!ram_queue.empty()||!ram_ready.empty()){
                err="registered-copy setup found cache work";return false;
            }
        }
        Dev& d=devs[0];Chunk& c=*chunks[0];CK(cudaSetDevice(d.id));
        if(gpu_misses_cs)CK(cudaStreamSynchronize(gpu_misses_cs));
        CK(cudaStreamSynchronize(c.cs));CK(cudaStreamSynchronize(d.st));
        int supported=0;CK(cudaDeviceGetAttribute(&supported,cudaDevAttrHostRegisterSupported,d.id));
        if(!supported){std::fprintf(stderr,"ds4: experimental registered-copy setup: device has no host-register support; original staging retained\n");return true;}
        const double started=now_ms();
        std::vector<regcopy::Range> arenas,selected;
        arenas.reserve(ram_chunks.size());
        for(size_t i=0;i<ram_chunks.size();++i)arenas.push_back({(uintptr_t)ram_chunks[i],ram_chunk_bytes[i]});
        const size_t cap=register_ram_all?~size_t(0):register_ram_mb<<20;
        std::string detail;
        if(!regcopy::select(arenas,cap,selected,detail)){err=detail;return false;}
        size_t baseline_free=0,total=0;CK(cudaMemGetInfo(&baseline_free,&total));
        const size_t MiB=(size_t)1<<20;
        const size_t admission=strata::ds4::phasepolicy::registration_admission_bytes(register_gpu_floor_mb);
        if(baseline_free<admission){
            if(register_gpu_floor_mb==0)
                std::fprintf(stderr,"ds4: experimental registered-copy setup: %.1f MiB CUDA free <2560 MiB admission; original staging retained\n",baseline_free/1048576.0);
            else
                std::fprintf(stderr,"ds4: experimental registered-copy setup: reason=explicit-baseline-GPU-floor; "
                                    "%.1f MiB CUDA free <%.1f MiB chosen floor; original staging retained\n",
                             baseline_free/1048576.0,admission/1048576.0);
            return true;
        }
        // Zero retains the original baseline-minus512MiB policy. An explicit absolute floor is an unmeasured opt-in;
        // all original per-arena RAM/commit/CUDA checks and reverse rollback below are unchanged.
        const size_t configured=(size_t)std::max<int64_t>(0,o.reserve_mib)<<20;
        const size_t gpu_floor=strata::ds4::phasepolicy::registration_floor_bytes(baseline_free,configured,register_gpu_floor_mb);
        if(register_gpu_floor_mb>0)
            std::fprintf(stderr,"ds4: experimental registered-copy policy: explicit absolute CUDA floor%zu MiB; "
                                "baseline%.1f MiB, same per-arena RAM/commit checks and rollback\n",
                         register_gpu_floor_mb,baseline_free/1048576.0);
        cudaError_t api_status=cudaSuccess;
        // Diagnostic observations only: registration admission and rollback policy stay unchanged.
        cudaError_t registration_status=cudaSuccess,meminfo_status=cudaSuccess;
        size_t attempts=0,successes=0,last_free=baseline_free,min_free=baseline_free;
        uint64_t last_ram=~uint64_t{0},min_ram=~uint64_t{0};
        double last_commit=-1,min_commit=1e300;
        const char* rejected_by="none";
        regcopy::Callbacks calls;
        calls.headroom=[&]{
            if(decode_recovery&&cancel&&cancel()){rejected_by="recovery-stop";return regcopy::Step::unavailable;}
            size_t free_b=0,total_b=0;
            api_status=cudaMemGetInfo(&free_b,&total_b);
            meminfo_status=api_status;
            if(api_status!=cudaSuccess){rejected_by="cudaMemGetInfo";return regcopy::Step::fatal;}
            last_free=free_b;min_free=std::min(min_free,free_b);
            const uint64_t available=ram_avail();
            last_ram=available;
            if(available!=~uint64_t{0})min_ram=std::min(min_ram,available);
            if(available==~uint64_t{0}){rejected_by="RAM-query-unavailable";return regcopy::Step::unavailable;}
            if(available<kRamMargin){rejected_by="RAM-margin";return regcopy::Step::unavailable;}
            last_commit=commit_room();min_commit=std::min(min_commit,last_commit);
            if(last_commit<512.0*MiB){rejected_by="commit-margin";return regcopy::Step::unavailable;}
            if(free_b<gpu_floor){rejected_by="CUDA-free-floor";return regcopy::Step::unavailable;}
            return regcopy::Step::ready;
        };
        calls.register_one=[&](const regcopy::Range& range){
            ++attempts;
            const double t=now_ms();
            api_status=cudaHostRegister((void*)range.base,range.bytes,cudaHostRegisterPortable);
            registration_status=api_status;
            registered_copy_driver_ms+=now_ms()-t;
            if(api_status==cudaSuccess){++successes;return regcopy::Step::ready;}
            rejected_by="cudaHostRegister";
            if(api_status==cudaErrorMemoryAllocation||api_status==cudaErrorInvalidValue||
               api_status==cudaErrorNotSupported||api_status==cudaErrorHostMemoryAlreadyRegistered){
                (void)cudaGetLastError();return regcopy::Step::unavailable;
            }
            return regcopy::Step::fatal;
        };
        calls.unregister_one=[&](const regcopy::Range& range){
            const double t=now_ms();
            const cudaError_t status=cudaHostUnregister((void*)range.base);
            registered_copy_unregister_ms+=now_ms()-t;
            return status==cudaSuccess;
        };
        const regcopy::Result result=regcopy::acquire(registered_copy,selected,calls,detail);
        registered_copy_setup_ms+=now_ms()-started;
        std::fprintf(stderr,"ds4: experimental registered-copy setup: selected %zu/%zu whole CPU arenas, "
                            "published %zu ranges/%.1f MiB, CUDA floor %.1f MiB; %.3f s setup/%.3f s register calls; %s\n",
                     selected.size(),arenas.size(),registered_copy.held.size(),registered_copy.bytes()/1048576.0,
                     gpu_floor/1048576.0,registered_copy_setup_ms/1000.0,registered_copy_driver_ms/1000.0,
                     result==regcopy::Result::enabled?"direct H2D first-prefill only":
                     result==regcopy::Result::fallback?"rolled back; original staging retained":"terminal setup/rollback failure");
        std::fprintf(stderr,"ds4: registered-copy admission diagnostic: reason=%s detail=%s; attempts=%zu successes=%zu; "
                            "register_status=%d(%s) meminfo_status=%d(%s); CUDA baseline/min/last/floor "
                            "%.1f/%.1f/%.1f/%.1f MiB; RAM min/last %.1f/%.1f MiB; commit min/last %.1f/%.1f MiB\n",
                     rejected_by,detail.empty()?"none":detail.c_str(),attempts,successes,
                     (int)registration_status,cudaGetErrorString(registration_status),(int)meminfo_status,cudaGetErrorString(meminfo_status),
                     baseline_free/1048576.0,min_free/1048576.0,last_free/1048576.0,gpu_floor/1048576.0,
                     min_ram==~uint64_t{0}?-1.0:min_ram/1048576.0,last_ram==~uint64_t{0}?-1.0:last_ram/1048576.0,
                     min_commit>1e290?-1.0:min_commit/1048576.0,last_commit/1048576.0);
        if(result==regcopy::Result::fatal){err="registered-copy setup failed: "+detail;return false;}
        return true;
#endif
    }

    // Exit/load-error cleanup only. Never release an anonymous CPU source while a direct DMA or uncertain
    // registration may still own it. A terminal failure retains those arenas for process/driver teardown.
    bool drain_and_unregister_copied_ram_at_exit() {
        if(registered_copy.held.empty())return true;
        registered_copy.published=false;
        bool drained=true;
        for(size_t di=0;di<chunks.size();++di)if(chunks[di]){
            Chunk& c=*chunks[di];
            for(size_t i=c.base;i<c.n_read;++i)
                while(fit(c,i).left.load(std::memory_order_acquire)!=0){if(!c.io)return false;c.io->wait_any(500);}
            if(di<devs.size()&&cudaSetDevice(devs[di].id)!=cudaSuccess)drained=false;
            if(c.cs&&cudaStreamSynchronize(c.cs)!=cudaSuccess)drained=false;
            if(di<devs.size()&&devs[di].st&&cudaStreamSynchronize(devs[di].st)!=cudaSuccess)drained=false;
            c.io.reset(); // join any old queued CPU jobs before the source arenas could be released
        }
        if(!drained)return false;
        std::string why;
        return retire_registered_copy_ram(why);
    }

    // ---- the flow (see Chunk)
    // Windows (WDDM: HIP's runtime and CUDA's driver alike) submits queued work only when the host waits on the GPU:
    // the flow's copies and the compute it queued between its waits are submitted at once (win_submit); elsewhere
    // nothing.  (CUDA too since #303: two RTX 3080s hung before the first layer, one GPU spinning on the other's flow)
    static void kick(cudaStream_t s) { win_submit(s); }
    size_t flow_region(int r) const {   // a part's room in a staging slot (read from its offset rounded down)
        return ((flow_part[r] + os::kDirectAlign - 1) & ~(size_t) (os::kDirectAlign - 1)) + os::kDirectAlign;
    }
    size_t flow_stride() const { return flow_region(0) + flow_region(1) + flow_region(2); }
    static Chunk::Item& fit(Chunk& c, size_t i) { return c.items[i - c.base]; }
    // what member m of l's group reads of matrix r of an expert: gate / up its rows [m n_ff / nm, ..) - one run, from
    // this offset in the matrix, this long; the down the whole matrix (its part, the blocks [m n_ff / nm / 32, ..) of
    // every row, is DMA'd out of it row by row); alone (nm 1) every matrix whole
    size_t mx_read_off(const Layer& l, int m, int r) const {
        return r < 2 && l.nm > 1 ? (size_t) m * (size_t) (g.ff / l.nm) * gg::type_row_bytes(gg::expert_type(l.lay, r), g.dim)
                                 : 0;
    }
    size_t mx_read_len(const Layer& l, int r) const {
        return r < 2 && l.nm > 1 ? (size_t) (g.ff / l.nm) * gg::type_row_bytes(gg::expert_type(l.lay, r), g.dim)
                                 : mx_part_bytes(l, r);
    }
    // the device's member index in L0's group (0 alone)
    int flow_mi(const Chunk& c, const Layer& L0) const {
        for (int m = 0; m < L0.nm; ++m)
            if (member(L0, m).dev == c.dev) return m;
        return 0;
    }
    void flow_push(Chunk& c, const Layer& l, int mi, int k, int e, int v) {
        c.items.emplace_back();
        Chunk::Item& it = c.items.back();
        it.il = l.il;
        it.mi = mi;
        it.k = k;
        it.e = e;
        it.v = v;
        bool in_ram = !l.ram.empty() && l.ram[(size_t) e] >= 0;
        const uint8_t* store_blob = nullptr;   // (the RAM store holds it: copied into staging like a RAM-tier slot, no read)
        if (!in_ram && store && l.nm == 1) {
            const int ss = store->find(l.il, e);
            if (ss >= 0) {
                store_blob = store->blob(ss);
                in_ram = true;
            }
        }
        if(in_ram&&store_blob==nullptr&&registered_copy.published&&chunk_major_phase_pending) {
            const uint8_t* source[3]={nullptr,nullptr,nullptr};size_t bytes[3]={};bool covered=true;
            for(int r=0;r<3;++r)covered=registered_ram_role(l,e,r,source[r],bytes[r])&&covered;
            if(covered) {
                // Private copied-RAM slots, NOT ram_slots/ram_pinned. All three roles are covered by owned
                // successful registrations; no CPU bounce job owns a staging slot for this item.
                for(int r=0;r<3;++r)it.src[r]=source[r];
                it.ram=ram_part[(size_t)l.ram[(size_t)e]][0];
                it.registered_copy_source=true;it.stage=-1;it.left.store(0,std::memory_order_release);
                ++c.it_ram;return;
            }
        }
        if (in_ram && ram_pinned && store_blob == nullptr) {   // (DMA'd from where it lies)
            const uint8_t* p = ram_slots[(size_t) l.ram[(size_t) e]];
            for (int r = 0; r < 3; ++r) it.src[r] = p + l.lay.trellis[r] + mx_read_off(l, mi, r);
            ++c.it_ram;
            return;
        }
        if (in_ram)   // (copied into staging by io, then DMA'd; copy_mode: from its NH parts, flow_pump)
            it.ram = store_blob != nullptr ? store_blob
                     : copy_mode ? ram_part[(size_t) l.ram[(size_t) e]][0] : ram_slots[(size_t) l.ram[(size_t) e]];
        bool hit = false;
        if (!in_ram && c.lc != nullptr && l.nm == 1) {   // layer-major: from the layer cache, or kept for the chunks after
            const size_t i = c.base + c.items.size() - 1, stride = (l.blob + 63) & ~(size_t) 63;
            if (c.lc_il != l.il) {   // (a layer's first item: the slots the layer before used are its own again
                c.lc_il = l.il;      // once that layer's items are all read - `after` below)
                c.lc_used = 0;
                c.lc_n = (int) std::min<size_t>(c.lc_of.size(), c.lc_bytes / stride);
                c.lc_first = i;
                std::fill(c.lc_of.begin(), c.lc_of.end(), -1);
            }
            const int j = c.lc_of[(size_t) e];
            if (j >= 0) {   // (copied into staging by io once its reader is done, then DMA'd)
                it.ram = c.lc + (size_t) j * stride;
                it.after = c.lc_writer[(size_t) j];
                hit = true;
            } else if (k + 1 < (int) c.fl_T.size() && c.lc_used < c.lc_n) {   // (a later chunk may pick it)
                const int jn = c.lc_used++;
                c.lc_of[(size_t) e] = jn;
                c.lc_writer[(size_t) jn] = (int64_t) i;
                it.keep = c.lc + (size_t) jn * stride;
                it.after = (int64_t) c.lc_first - 1;
            }
        }
        it.stage = (int) (c.n_file++ % (uint64_t) c.S);
        uint8_t* sb = c.stage + (size_t) it.stage * c.stage_stride;
        for (int r = 0; r < 3; ++r) {
            const uint64_t off = l.mx_off[r] + (uint64_t) e * l.mx_stride[r] + mx_read_off(l, mi, r);
            it.src[r] = sb + (in_ram || hit ? 0 : off % os::kDirectAlign);
            sb += flow_region(r);
        }
        it.left.store(in_ram && copy_mode && store_blob == nullptr ? 3 * NH : 3, std::memory_order_relaxed);
        ++(in_ram ? c.it_ram : hit ? c.it_lc : c.it_file);
    }
    // the likely experts of (chunk k, layer il) for T tokens, first in its stream: the CPU lane's experts whose chance
    // to be picked at least once, 1 - (1 - share)^(picks), is at least DS4_SPEC_Q (0.9: the reads are what takes the
    // time, an unpicked one costs a whole read, and the routing is known a few ms after the layer before is done -
    // early reads pay where a chunk's long attention keeps the disk otherwise idle, and there nearly every expert is
    // that likely), likeliest first, at most what the staging and the ring hold (no more can be read before the layer
    // routes); the layer's virtual slots put them first, then its other CPU-lane experts in index order (the order the
    // rest of its picks are streamed in)
    // (a tensor-parallel group's members each decide their own stream and virtual slots - the same, from the same
    // profile and the same routing - and each reads only its part of every expert)
    void flow_spec(Chunk& c, int k, int il, int T) {
        static const double spec_q = std::getenv("DS4_SPEC_Q") ? std::atof(std::getenv("DS4_SPEC_Q")) : 0.9;
        Layer& l = L[(size_t) il];
        const int mi = flow_mi(c, l);
        int* vs = member(l, mi).vs_pin;
        c.spec_k = k;
        c.spec_il = il;
        c.spec_begin = c.base + c.items.size();
        const uint32_t* pr = prof.data() + (size_t) il * g.n_exp;
        double sum = 0;
        for (int e = 0; e < l.n_exp; ++e) sum += pr[e];
        std::vector<std::pair<double, int>> cand;
        const double picks = (double) l.k_exp * T;
        // (early reads only while they hit: once a run's likely ones went unpicked more often than DS4_SPEC_WASTE
        // (0.2) of the time, the rest of the run reads only what the routing picked - an unpicked one costs a whole
        // read, and how well the profile guesses depends on the PC's history and the prompt: V100, 64 tokens - 376
        // of 921 likely ones unpicked, 7 GB of 84 read for nothing; Radeon 8065S, 128 and 512 tokens - 44 of 945 and
        // 21 of 1568, the early reads 9-12% faster there)
        static const double waste = std::getenv("DS4_SPEC_WASTE") ? std::atof(std::getenv("DS4_SPEC_WASTE")) : 0.2;
        const bool early = c.it_spec < 2u * (uint64_t) c.ring_n || (double) c.it_spec_unused <= waste * (double) c.it_spec;
        for (int e : l.host_list) {
            if (!early) break;
            const double p = sum > 0 ? pr[e] / sum : 0.0;
            const double q = p >= 1.0 ? 1.0 : 1.0 - std::exp(picks * std::log1p(-p));
            if (q >= spec_q && spec_q <= 1.0) cand.push_back({q, e});
        }
        std::stable_sort(cand.begin(), cand.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first > b.first;
        });
        if (cand.size() > (size_t) (c.S + c.ring_n)) cand.resize((size_t) (c.S + c.ring_n));
        std::vector<char> is((size_t) l.n_exp, 0);
        int j = 0;
        for (const auto& x : cand) {
            vs[x.second] = l.n_slots + j++;
            is[(size_t) x.second] = 1;
        }
        for (int e : l.host_list)
            if (!is[(size_t) e]) vs[e] = l.n_slots + j++;
        for (const auto& x : cand) flow_push(c, l, mi, k, x.second, vs[x.second]);
        c.n_spec = (int) cand.size();
        c.it_spec += cand.size();
    }
    // (chunk k, layer il) routed (bounds b by virtual slot): the rest of its picks after its likely ones
    void flow_rest(Chunk& c, int k, int il, const std::vector<int>& b) {
        const Layer& l = L[(size_t) il];
        const int mi = flow_mi(c, l);
        const int* vs = member(l, mi).vs_pin;
        for (size_t i = c.spec_begin; i < c.spec_begin + (size_t) c.n_spec; ++i) {
            const int v = fit(c, i).v;
            if (b[(size_t) v + 1] == b[(size_t) v]) ++c.it_spec_unused;
        }
        for (int e : l.host_list) {   // (index order: their virtual slots ascending)
            const int v = vs[e];
            if (v >= l.n_slots + c.n_spec && b[(size_t) v + 1] > b[(size_t) v]) flow_push(c, l, mi, k, e, v);
        }
    }
    // the next (chunk, layer) of the device after (k, il) in the run's order that streams experts (false: none) - a
    // long prompt's skip leaves out its early chunks' layers from Run::yoco on (kv_only there: no experts)
    bool flow_next(const Chunk& c, int k, int il, int& k2, int& il2) const {
        while (flow_next1(c, k, il, k2, il2)) {
            if (c.fl_yoco < 0 || k2 >= c.fl_cut_k || il2 < c.fl_yoco) return true;
            k = k2;
            il = il2;
        }
        return false;
    }
    // the rows chunk k runs layer il on (a long prompt's skip: the cut chunk's from cut_r on, from the yoco layer on)
    static int fl_rows(const Chunk& c, int k, int il) {
        const int T = c.fl_T[(size_t) k];
        return c.fl_yoco >= 0 && k == c.fl_cut_k && il >= c.fl_yoco ? T - c.fl_cut_r : T;
    }
    bool flow_next1(const Chunk& c, int k, int il, int& k2, int& il2) const {
        const auto it = std::find(c.fl_layers.begin(), c.fl_layers.end(), il);
        const size_t i = (size_t) (it - c.fl_layers.begin());
        if (c.lm) {   // (layer-major: the layer's next chunk, then the next layer's first)
            if (k + 1 < (int) c.fl_T.size()) {
                k2 = k + 1;
                il2 = il;
                return true;
            }
            if (i + 1 >= c.fl_layers.size()) return false;
            k2 = 0;
            il2 = c.fl_layers[i + 1];
            return true;
        }
        if (i + 1 < c.fl_layers.size()) {
            k2 = k;
            il2 = c.fl_layers[i + 1];
            return true;
        }
        if (k + 1 < (int) c.fl_T.size() && !c.fl_layers.empty()) {
            k2 = k + 1;
            il2 = c.fl_layers[0];
            return true;
        }
        return false;
    }
    // the reads (in order, each into its staging slot once that slot's last DMA is done) and the DMAs (in order, each
    // into its ring slot once that slot's last use is queued: the copy stream waits for it) as far as they can go now
    bool flow_pump(Chunk& c, std::string& err) {
        const size_t end = c.base + c.items.size();
        IoQueue::Job jobs[6];
        while (c.n_read < end) {
            Chunk::Item& it = fit(c, c.n_read);
            if (it.after >= 0) {   // layer-major: a slot's expert copied out of it once its reader is done; a slot
                // written again once the layer before's items are all read (none still copies out of it)
                bool wait = false;
                if (it.keep == nullptr)
                    wait = (size_t) it.after >= c.base && fit(c, (size_t) it.after).left.load(std::memory_order_acquire) != 0;
                else {
                    while (c.lc_chk < c.n_read &&
                           (c.lc_chk < c.base || fit(c, c.lc_chk).left.load(std::memory_order_acquire) == 0))
                        ++c.lc_chk;
                    wait = c.lc_chk <= (size_t) it.after;
                }
                if (wait) break;
            }
            // A preceding chunk already queued this expert into the phase cache on c.cs. Skip its host read.
            // Metadata is reset only as the DMA cursor reaches a new layer, after every prior layer's item was
            // queued. The old layer's D2D reads precede every overwrite on that same copy stream.
            if (c.lm && c.lm_gpu != nullptr && c.lm_gpu_il == it.il && it.e >= 0 &&
                (size_t) it.e < c.lm_gpu_of.size() && c.lm_gpu_of[(size_t) it.e] >= 0) {
                it.lm_gpu_read_skipped = true;
                it.stage = -1;   // no outstanding host read uses the originally proposed staging slot
                it.left.store(0, std::memory_order_release);
                ++c.lm_gpu_read_skips;
                ++c.n_read;
                continue;
            }
            if (it.stage >= 0) {
                const int64_t prev = c.stage_item[(size_t) it.stage];
                if (prev >= 0) {
                    if ((size_t) prev >= c.n_dma) break;
                    const cudaError_t q = cudaEventQuery(c.stage_ev[(size_t) it.stage]);
                    if (q == cudaErrorNotReady) break;
                    CK(q);
                }
                c.stage_item[(size_t) it.stage] = (int64_t) c.n_read;
                const Layer& l = L[(size_t) it.il];
                uint8_t* sb = c.stage + (size_t) it.stage * c.stage_stride;
                const int cs = copy_mode && it.ram != nullptr ? l.ram[(size_t) it.e] : -1;   // (a copy's slot)
                int nj = 0;
                for (int r = 0; r < 3; ++r) {
                    if (cs >= 0) {   // a copy: each matrix's NH parts of rows, side by side into staging
                        const size_t nb = (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
                        for (int X = 0; X < NH; ++X) {
                            IoQueue::Job& j = jobs[nj++];
                            j = IoQueue::Job{};
                            j.len = nb;
                            j.dst = sb + (size_t) X * nb;
                            j.left = &it.left;
                            j.bad = &it.bad;
                            j.mem = ram_part[(size_t) cs][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r];
                        }
                        sb += flow_region(r);
                        continue;
                    }
                    IoQueue::Job& j = jobs[nj++];
                    const os::fd_t pf = pk.pfds.empty() ? os::kBadFd : pk.pfds[(size_t) l.mx_file[r]];
                    j.fd = pf != os::kBadFd ? pf : pk.fds[(size_t) l.mx_file[r]];
                    j.dfd = pk.dfds.empty() ? os::kBadFd : pk.dfds[(size_t) l.mx_file[r]];
                    j.off = l.mx_off[r] + (uint64_t) it.e * l.mx_stride[r] + mx_read_off(l, it.mi, r);
                    j.len = mx_read_len(l, r);
                    j.dst = sb;
                    j.left = &it.left;
                    j.bad = &it.bad;
                    j.mem = it.ram != nullptr ? it.ram + l.lay.trellis[r] + mx_read_off(l, it.mi, r) : nullptr;
                    j.map = pk.at(l.mx_file[r], j.off);
                    j.keep = it.keep != nullptr ? it.keep + l.lay.trellis[r] : nullptr;
                    sb += flow_region(r);
                }
                c.io->push(jobs, nj);
            }
            ++c.n_read;
        }
        const size_t R = (size_t) c.ring_n, dma0 = c.n_dma;
        struct Kick {   // (every way out: what was queued is submitted)
            cudaStream_t s;
            const size_t &a, b;
            ~Kick() {
                if (a != b) kick(s);
            }
        } kq{c.cs, c.n_dma, dma0};
        while (c.n_dma < c.n_read && c.n_dma < c.n_comp + R) {
            Chunk::Item& it = fit(c, c.n_dma);
            if (it.left.load(std::memory_order_acquire) != 0) break;
            if (it.bad.load(std::memory_order_acquire)) {
                err = "reading layer " + std::to_string(it.il) + " expert " + std::to_string(it.e) +
                      " from the model's files failed";
                return false;
            }
            const size_t r = c.n_dma % R;
            if (c.n_dma >= R) CK(cudaStreamWaitEvent(c.cs, c.ring_free[r], 0));
            const Layer& l = L[(size_t) it.il];
            uint8_t* dst = c.ring + r * c.ring_stride;
            const x3::ExpertLayout& lay = l.nm > 1 ? l.hlay : l.lay;
            const bool cache_on = c.lm && c.lm_gpu != nullptr && l.nm == 1;
            if (cache_on && (it.e < 0 || it.e >= l.n_exp || l.blob > c.ring_stride)) {
                err = "layer-major GPU expert cache item is outside its expert/ring bounds";
                return false;
            }
            if (cache_on && c.lm_gpu_il != it.il) {
                // GPU operations from the previous layer have already been queued in order on c.cs. Reusing
                // its storage for this layer cannot overtake a previous cache-to-ring copy.
                c.lm_gpu_il = it.il;
                c.lm_gpu_stride = align256(l.blob);
                c.lm_gpu_used = 0;
                c.lm_gpu_of.assign((size_t) l.n_exp, -1);
            }
            const int cache_slot = cache_on && it.e >= 0 && (size_t) it.e < c.lm_gpu_of.size()
                                       ? c.lm_gpu_of[(size_t) it.e] : -1;
            if (it.lm_gpu_read_skipped && cache_slot < 0) {
                err = "layer-major GPU expert cache lost the source of a skipped host read";
                return false;
            }
            const size_t payload = mx_read_len(l, 0) + mx_read_len(l, 1) + mx_read_len(l, 2);
            if (cache_slot >= 0) {
                CK(cudaMemcpyAsync(dst, c.lm_gpu + (size_t) cache_slot * c.lm_gpu_stride,
                                   l.blob, cudaMemcpyDeviceToDevice, c.cs));
                ++c.lm_gpu_hits;
                c.lm_gpu_saved_h2d_bytes += payload;
                c.lm_gpu_d2d_bytes += l.blob;
            } else {
                if(it.registered_copy_source) {
                    if(!dma_registered_copy_roles(c,l,it,dst,err))return false;
                } else {
                    // The unchanged host-to-ring path, including tensor-parallel members when the cache is off.
                    for (int p = 0; p < 2; ++p)
                        CK(cudaMemcpyAsync(dst + lay.trellis[p], it.src[p], mx_read_len(l, p), cudaMemcpyHostToDevice, c.cs));
                    if (l.nm > 1) {
                        const int t2 = gg::expert_type(l.lay, 2);
                        const size_t dr = gg::type_row_bytes(t2, g.ff), dh = gg::type_row_bytes(t2, g.ff / l.nm);
                        CK(cudaMemcpy2DAsync(dst + lay.trellis[2], dh, it.src[2] + (size_t) it.mi * dh, dr, dh, (size_t) g.dim,
                                             cudaMemcpyHostToDevice, c.cs));
                    } else
                        CK(cudaMemcpyAsync(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2), cudaMemcpyHostToDevice, c.cs));
                }
                c.stream_bytes += (double) unit(l);
                if(register_ram_all||register_ram_mb>0) {
                    c.registered_h2d_payload_bytes+=payload;
                    if(it.registered_copy_source)c.registered_direct_bytes+=payload;
                    else if(it.ram)c.registered_bounce_bytes+=payload;
                }
                if (cache_on) {
                    c.lm_gpu_h2d_bytes += payload;
                    // Keep a bounded prefix when a budget cannot hold this layer. Entries never move or evict
                    // until the DMA cursor enters the next layer. A later chunk may use a filled slot immediately:
                    // both its fill and its D2D read are ordered on c.cs before the ring_fill dependency.
                    if (c.lm_gpu_stride > 0 && c.lm_gpu_used < c.lm_gpu_bytes / c.lm_gpu_stride) {
                        const size_t slot = c.lm_gpu_used++;
                        CK(cudaMemcpyAsync(c.lm_gpu + slot * c.lm_gpu_stride, dst, l.blob,
                                           cudaMemcpyDeviceToDevice, c.cs));
                        c.lm_gpu_of[(size_t) it.e] = (int) slot;
                        ++c.lm_gpu_fills;
                        c.lm_gpu_peak_used = std::max(c.lm_gpu_peak_used, c.lm_gpu_used);
                        c.lm_gpu_d2d_bytes += l.blob;
                    } else
                        ++c.lm_gpu_no_room;
                }
            }
            CK(cudaEventRecord(c.ring_fill[r], c.cs));
            if (it.stage >= 0) CK(cudaEventRecord(c.stage_ev[(size_t) it.stage], c.cs));
            ++c.n_dma;
        }
        return true;
    }
    // layer L0 of chunk k routed (bounds b, the device's copy d_bounds): its streamed experts as they land, the VRAM
    // ones in shares between them (the device works while the reads run)
    bool flow_layer(Dev& d, Chunk& c, Layer& L0, int mi, int k, int T, const std::vector<int>& b, std::string& err) {
        cudaStream_t s = d.st;
        const uint8_t* vbase = member(L0, mi).vram_base;   // (a member's: its part of each VRAM expert, L0's slots)
        if (c.spec_k != k || c.spec_il != L0.il) {
            err = "the prompt path's expert stream is out of step (layer " + std::to_string(L0.il) + ")";
            return false;
        }
        flow_rest(c, k, L0.il, b);
        const size_t lb = c.spec_begin, le = c.base + c.items.size(), n_lay = le - lb;
        int k2 = 0, il2 = 0;
        if (flow_next(c, k, L0.il, k2, il2)) flow_spec(c, k2, il2, fl_rows(c, k2, il2));
        const size_t R = (size_t) c.ring_n, bmax = std::max<size_t>(1, R / 2);
        int v_done = 0;
        while (c.n_comp < le) {
            // a batch's bounds are fixed by the stream alone (bmax items, never across the ring's end), never by what
            // has landed: the experts' outputs are summed batch after batch (the VRAM ones' shares between them), so
            // the sums come out the same on every run whatever the reads' timing
            const size_t i0 = c.n_comp, i1 = std::min({le, i0 + bmax, (i0 / R + 1) * R});
            if (!flow_pump(c, err)) return false;
            if (c.n_dma < i1) {   // not all of the batch landed: wait for a read, or for a staging slot's copy
                const double tw = now_ms();
                kick(s);
                if (c.n_read > c.n_dma) c.io->wait_any(500);
                else {
                    const Chunk::Item& it = fit(c, c.n_read);
                    if (it.stage >= 0) CK(cudaEventSynchronize(c.stage_ev[(size_t) it.stage]));
                }
                c.wait_ms += now_ms() - tw;
                continue;
            }
            const int n = (int) (i1 - i0);
            int* bc = c.bc_pin + (size_t) (c.n_batch % (uint64_t) c.bc_n) * c.bc_w;
            int* bcd = c.bc_dev + (size_t) (c.n_batch % (uint64_t) c.bc_n) * c.bc_w;
            ++c.n_batch;
            // (the items' virtual slots ascend and the slots between them have no rows: their bounds side by side)
            for (int m = 0; m < n; ++m) bc[m] = b[(size_t) fit(c, i0 + (size_t) m).v];
            bc[n] = b[(size_t) fit(c, i1 - 1).v + 1];
            if (bc[n] > bc[0]) {
                CK(cudaStreamWaitEvent(s, c.ring_fill[(i1 - 1) % R], 0));
                CK(cudaMemcpyAsync(bcd, bc, (size_t) (n + 1) * 4, cudaMemcpyHostToDevice, s));
                std::vector<int> ids;
                if (c.lm_f16 != nullptr) {
                    ids.reserve((size_t) n);
                    for (int m = 0; m < n; ++m) ids.push_back(fit(c, i0 + (size_t) m).e);
                }
                if (!run_set(d, c, L0, c.ring + (i0 % R) * c.ring_stride, 0, c.ring_stride, bcd, bc, 0, n, T, err,
                             ids.empty() ? nullptr : ids.data(), (int) ids.size()))
                    return false;
            }
            for (size_t i = i0; i < i1; ++i) CK(cudaEventRecord(c.ring_free[i % R], s));
            kick(s);
            c.n_comp = i1;
            if (!flow_pump(c, err)) return false;
            const int v_to = (int) ((int64_t) L0.n_slots * (int64_t) (c.n_comp - lb) / (int64_t) std::max<size_t>(1, n_lay));
            if (!run_set(d, c, L0, vbase, 0, unit(L0), c.bounds, b.data(), v_done, v_to, T, err)) return false;
            v_done = v_to;
        }
        if (!run_set(d, c, L0, vbase, 0, unit(L0), c.bounds, b.data(), v_done, L0.n_slots, T, err)) return false;
        while (c.base < c.n_comp && c.items.size() > 4096) {   // (done with: read, copied and used)
            c.items.pop_front();
            ++c.base;
        }
        return true;
    }

    // Try one ORIGINAL run_set range. All host checks happen before changing work buffers. A supported range may
    // compute its experts in padded row buckets, but scatters back to the same compact rows before the UNCHANGED
    // moe_combine call. It never merges/splits accumulator ranges or routes int8-light experts through F16.
    bool try_f16_phase_set(Dev& d, Chunk& c, const Layer& l, const uint8_t* wbase, int wb0, size_t stride,
                           const int* d_bounds, const int* b, int i0, int i1, const int* expert_ids, int n_ids, int a,
                           cudaEvent_t gu_event, bool& used, std::string& err) {
        used = false;
#if defined(STRATA_USE_HIP)
        (void) d; (void) c; (void) l; (void) wbase; (void) wb0; (void) stride; (void) d_bounds; (void) b; (void) i0; (void) i1;
        (void) expert_ids; (void) n_ids; (void) a; (void) gu_event; (void) err;
        return true;
#else
        if (c.lm_f16 == nullptr || !prefill_phase_pending || !c.lm || !c.flow) return true;
        auto supported = [](int t) { return t == 14 || t == 18 || t == 21 || t == 22; };
        const int tg = gg::expert_type(l.lay, 0), tu = gg::expert_type(l.lay, 1), td = gg::expert_type(l.lay, 2);
        if (l.mtp || l.nm != 1 || l.lay.fmt != gg::kFmtMXFP4 || tg != tu || !supported(tg) || !supported(td)) {
            ++c.lm_f16_fallback_format;
            return true;
        }
        if (l.n_slots != 0 || !expert_ids || wb0 < 0 || i0 < wb0 || i1 > n_ids || i1 - i0 > c.lm_f16_meta_cap ||
            c.lm_f16_blas == nullptr || !c.lm_f16_meta_host || !c.lm_f16_meta_dev || !c.lm_f16_y ||
            l.n_exp > c.lm_f16_meta_cap || l.ffn != g.ff) {
            ++c.lm_f16_fallback_shape;
            return true;
        }
        struct Planned { F16PhaseRow row; int original, expert; };
        std::vector<Planned> plan;
        int padded_rows = 0, valid_rows = 0, max_pad = 0, light_rows = 0, light_experts = 0, original_max_rows = 0;
        for (int i = i0; i < i1; ++i) {
            const int rows = b[i + 1] - b[i], e = expert_ids[i];
            if (rows == 0) continue;
            if (rows < 0 || rows > c.C || e < 0 || e >= l.n_exp) { ++c.lm_f16_fallback_shape; return true; }
            original_max_rows = std::max(original_max_rows, rows);
            if (rows <= 16) {
                const char* disabled = std::getenv("STRATA_DS4_NO_LIGHT");
                if (!lm_f16_mixed_light || (disabled != nullptr && disabled[0] == '1') || d_bounds == nullptr) {
                    ++c.lm_f16_fallback_light;
                    return true;
                }
                light_rows += rows;
                ++light_experts;
                continue;   // keep these ORIGINAL compact positions for the exact full-range light APIs
            }
            int pad = 32;
            while (pad < rows && pad <= c.C / 2) pad *= 2;
            if (pad < rows || pad > c.C - padded_rows) { ++c.lm_f16_fallback_shape; return true; }
            padded_rows += pad;
            valid_rows += rows;
            max_pad = std::max(max_pad, pad);
            plan.push_back({{b[i] - a, rows, 0, pad}, i, e});
        }
        if (plan.empty()) { // all-light ranges keep the original dispatch and its max_rows-dependent kernel choice
            if (light_experts > 0) ++c.lm_f16_fallback_light;
            else ++c.lm_f16_fallback_shape;
            return true;
        }
        if (valid_rows + light_rows != b[i1] - a || padded_rows > c.C) {
            ++c.lm_f16_fallback_shape;
            return true;
        }
        std::stable_sort(plan.begin(), plan.end(), [](const Planned& x, const Planned& y) { return x.row.padded < y.row.padded; });
        int p0 = 0;
        for (Planned& p : plan) { p.row.pad0 = p0; p0 += p.row.padded; }
        if (c.lm_f16_il != l.il) {
            // Only d.st reads/writes F16 weights. New-layer decodes follow every old-layer use on that stream.
            c.lm_f16_il = l.il;
            c.lm_f16_valid.assign((size_t) l.n_exp, 0);
        }
        const int D = g.dim, FF = g.ff, ne = (int) plan.size(), cap = c.lm_f16_meta_cap;
        const size_t role_values = (size_t) D * FF;
        for (const Planned& p : plan) {
            if (c.lm_f16_valid[(size_t) p.expert]) { ++c.lm_f16_hits; continue; }
            for (int r = 0; r < 3; ++r) {
                const int t = gg::expert_type(l.lay, r), k = r < 2 ? D : FF, n = r < 2 ? FF : D;
                const uint8_t* raw = wbase + (size_t) (p.original - wb0) * stride + l.lay.trellis[r];
                if (!gg::iq::dequant_rows(t, raw, gg::iq::row_bytes(t, k), k, n,
                                         c.lm_f16 + (size_t) p.expert * c.lm_f16_unit + (size_t) r * role_values,
                                         false, d.st)) {
                    err = "experimental F16 expert unpack failed";
                    return false;
                }
            }
            c.lm_f16_valid[(size_t) p.expert] = 1;
            ++c.lm_f16_fills;
        }
        const int slot = (int) (c.lm_f16_meta_count++ % Chunk::kF16MetaSlots);
        if (c.lm_f16_meta_recorded[slot]) CK(cudaEventSynchronize(c.lm_f16_meta_done[slot]));
        uint8_t* host = c.lm_f16_meta_host + (size_t) slot * c.lm_f16_meta_stride;
        uint8_t* device = c.lm_f16_meta_dev + (size_t) slot * c.lm_f16_meta_stride;
        const size_t desc_bytes = align256((size_t) cap * sizeof(F16PhaseRow));
        F16PhaseRow* hd = (F16PhaseRow*) host;
        auto* hp = (unsigned long long*) (host + desc_bytes);
        auto* dp = (unsigned long long*) (device + desc_bytes);
        for (int q = 0; q < ne; ++q) {
            const Planned& p = plan[(size_t) q];
            hd[q] = p.row;
            const size_t xoff = (size_t) p.row.pad0 * D, hoff = (size_t) p.row.pad0 * FF;
            const uint16_t* w = c.lm_f16 + (size_t) p.expert * c.lm_f16_unit;
            hp[q] = (unsigned long long) w;
            hp[cap + q] = (unsigned long long) (w + role_values);
            hp[2 * cap + q] = (unsigned long long) (w + 2 * role_values);
            hp[3 * cap + q] = (unsigned long long) (c.x16 + xoff);
            hp[4 * cap + q] = (unsigned long long) (c.sh16 + hoff);
            hp[5 * cap + q] = (unsigned long long) (c.sg + hoff);
            hp[6 * cap + q] = (unsigned long long) (c.su + hoff);
            hp[7 * cap + q] = (unsigned long long) (c.lm_f16_y + xoff);
        }
        CK(cudaMemcpyAsync(device, host, c.lm_f16_meta_stride, cudaMemcpyHostToDevice, d.st));
        auto record_meta_done = [&]() {
            const cudaError_t status = cudaEventRecord(c.lm_f16_meta_done[slot], d.st);
            if (status != cudaSuccess) { err = std::string("F16 metadata event: ") + cudaGetErrorString(status); return false; }
            c.lm_f16_meta_recorded[slot] = true;
            return true;
        };
        auto blas_failed = [&](cublasStatus_t status) {
            if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH ||
                status == CUBLAS_STATUS_ALLOC_FAILED) {
                if (++c.lm_f16_fallback_blas == 1)
                    std::fprintf(stderr, "ds4: experimental F16 cuBLAS range unsupported (status %d); original TC retained\n", (int) status);
                if (status == CUBLAS_STATUS_ALLOC_FAILED) (void) cudaGetLastError();
                return record_meta_done();   // no compact c.h16/c.y data was published; original TC safely overwrites
            }
            err = "experimental F16 cuBLAS failed, status " + std::to_string((int) status);
            return false;
        };
        const unsigned gx = (unsigned) std::min<size_t>(((size_t) max_pad * D + 255) / 256, 65535);
        f16_phase_gather<<<dim3(gx, (unsigned) ne), 256, 0, d.st>>>((const F16PhaseRow*) device, ne, max_pad,
                                                                                c.x, D, c.row_tok + a, c.x16);
        CK(cudaGetLastError());
        struct Bucket { int first, n, rows; };
        std::vector<Bucket> buckets;
        for (int i = 0; i < ne;) {
            int j = i + 1;
            while (j < ne && plan[(size_t) j].row.padded == plan[(size_t) i].row.padded) ++j;
            buckets.push_back({i, j - i, plan[(size_t) i].row.padded});
            i = j;
        }
        const float one = 1.0f, zero = 0.0f;
        auto mm = [&](const Bucket& bkt, int wi, int ai, int yi, int n, int k) {
            return cublasGemmBatchedEx(c.lm_f16_blas, CUBLAS_OP_T, CUBLAS_OP_N, n, bkt.rows, k, &one,
                                      (const void* const*) (dp + (size_t) wi * cap + bkt.first), CUDA_R_16F, k,
                                      (const void* const*) (dp + (size_t) ai * cap + bkt.first), CUDA_R_16F, k, &zero,
                                      (void* const*) (dp + (size_t) yi * cap + bkt.first), CUDA_R_32F, n,
                                      bkt.n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        };
        for (const Bucket& bkt : buckets) {
            cublasStatus_t status = mm(bkt, 0, 3, 5, FF, D);
            if (status != CUBLAS_STATUS_SUCCESS) return blas_failed(status);
            status = mm(bkt, 1, 3, 6, FF, D);
            if (status != CUBLAS_STATUS_SUCCESS) return blas_failed(status);
        }
        const size_t hn = (size_t) padded_rows * FF;
        f16_phase_swiglu<<<(unsigned) std::min<size_t>((hn + 255) / 256, 65535), 256, 0, d.st>>>(c.sg, c.su, hn, g.lim, c.sh16);
        CK(cudaGetLastError());
        if (gu_event != nullptr) CK(cudaEventRecord(gu_event, d.st));
        for (const Bucket& bkt : buckets) {
            const cublasStatus_t status = mm(bkt, 2, 4, 7, D, FF);
            if (status != CUBLAS_STATUS_SUCCESS) return blas_failed(status);
        }
        if (light_experts > 0) {
            // All benign BLAS failures are behind us; no compact H/Y was published before here. Preserve the
            // ORIGINAL full range/max_rows, hence the exact NE16 kernel, q8 scratch indexing and FMA order.
            gg::iq::SetArgs q{};
            q.wbase = wbase + (size_t) (i0 - wb0) * stride;
            q.stride = stride;
            for (int r = 0; r < 3; ++r) q.tr[r] = l.lay.trellis[r];
            q.bounds = d_bounds + i0;
            q.row0 = a;
            q.n_exp = i1 - i0;
            q.max_rows = original_max_rows;
            q.n_embd = D;
            q.n_ff = FF;
            q.limit = g.lim;
            q.tg = tg;
            q.td = td;
            q.xf = c.x;
            q.idx = c.row_tok + a;
            q.ldx = D;
            q.h = c.h16;
            q.ldo = FF;
            if (!gg::iq::light_gate_up(q, d.st)) {
                ++c.lm_f16_fallback_light;
                return record_meta_done();   // light unavailable: original full-range dispatch recomputes everything
            }
            CK(cudaGetLastError());
            q.xf = nullptr;
            q.idx = nullptr;
            q.xh = c.h16;
            q.ldx = FF;
            q.y = c.y;
            q.ldo = D;
            if (!gg::iq::light_down(q, d.st)) {
                ++c.lm_f16_fallback_light;
                return record_meta_done();   // no accumulation occurred; original dispatch overwrites partial H/Y
            }
            CK(cudaGetLastError());
            ++c.lm_f16_mixed_sets;
            c.lm_f16_light_experts += (uint64_t) light_experts;
            c.lm_f16_light_rows += (uint64_t) light_rows;
        }
        const unsigned sx = (unsigned) std::min<size_t>(((size_t) max_pad * std::max(D, FF) + 255) / 256, 65535);
        f16_phase_scatter<<<dim3(sx, (unsigned) ne), 256, 0, d.st>>>((const F16PhaseRow*) device, ne, c.sh16,
                                                                 c.lm_f16_y, D, FF, c.h16, c.y);
        CK(cudaGetLastError());
        if (!record_meta_done()) return false;
        ++c.lm_f16_sets;
        c.lm_f16_buckets += buckets.size();
        c.lm_f16_rows += (uint64_t) valid_rows;
        c.lm_f16_padded_rows += (uint64_t) padded_rows;
        used = true;
        return true;
#endif
    }

    // Bounded cold strategy: at most8 experts unpacked at any instant, reused by their GU/down GEMMs only.
    // Both resident and streamed blob pointers are accepted. No resident table, router, or combine is changed.
    // Completed batches may publish compact hot H/Y. On ANY benign failure the caller reruns the ENTIRE original
    // TC range before the single combine, overwriting every partial compact output in stream order.
    bool try_cold_f16_set(Dev& d, Chunk& c, const Layer& l, const uint8_t* wbase, int wb0, size_t stride,
                          const int* d_bounds, const int* b, int i0, int i1, int a,
                          cudaEvent_t gu_event, bool& used, std::string& err) {
        used=false;
#if defined(STRATA_USE_HIP)
        (void)d;(void)c;(void)l;(void)wbase;(void)wb0;(void)stride;(void)d_bounds;(void)b;
        (void)i0;(void)i1;(void)a;(void)gu_event;(void)err;return true;
#else
        if(!cold_f16_enabled || !chunk_major_phase_pending || !c.cold_f16 || c.lm || !c.flow) return true;
        auto supported=[](int t){return t==14||t==18||t==21||t==22;};
        const int tg=gg::expert_type(l.lay,0),tu=gg::expert_type(l.lay,1),td=gg::expert_type(l.lay,2);
        if(l.mtp||l.nm!=1||l.lay.fmt!=gg::kFmtMXFP4||tg!=tu||!supported(tg)||!supported(td)) {
            ++c.cold_f16_fallback_format; return true;
        }
        if(!wbase||!stride||wb0<0||i0<wb0||i1<=i0||i1>=2048||!d_bounds||!b||l.ffn!=g.ff||
           !c.cold_f16_blas||!c.cold_f16_y||!c.cold_f16_meta_host||!c.cold_f16_meta_dev||!c.cold_f16_workspace) {
            ++c.cold_f16_fallback_shape;return true;
        }
        coldf16::Plan plan;std::string why;
        if(!coldf16::make(b,i0,i1,a,c.C,cold_f16_rule,plan,why)||plan.batches.empty()) {
            ++c.cold_f16_fallback_shape;return true;
        }
        if(plan.light_experts) {
            const char* disabled=std::getenv("STRATA_DS4_NO_LIGHT");
            if(disabled&&disabled[0]=='1'){++c.cold_f16_fallback_light;return true;}
        }
        const int D=g.dim,FF=g.ff,cap=Chunk::kColdF16Experts;
        const size_t role_values=(size_t)D*FF;
        uint64_t padded_total=0,bucket_total=0;
        for(const coldf16::Batch& batch:plan.batches) {
            if(halt.load(std::memory_order_acquire)){err=kHalted;return false;}
            const int ne=(int)batch.rows.size();
            // Weight slots0..ne-1 may be recycled only on d.st, after every previous batch's last GEMM/scatter.
            for(int q=0;q<ne;++q) {
                const auto& row=batch.rows[(size_t)q];
                for(int r=0;r<3;++r) {
                    const int t=gg::expert_type(l.lay,r),k=r<2?D:FF,n=r<2?FF:D;
                    const uint8_t* raw=wbase+(size_t)(row.original-wb0)*stride+l.lay.trellis[r];
                    if(!gg::iq::dequant_rows(t,raw,gg::iq::row_bytes(t,k),k,n,
                        c.cold_f16+(size_t)q*c.cold_f16_unit+(size_t)r*role_values,false,d.st)) {
                        err="bounded cold F16 unpack failed";return false;
                    }
                }
                ++c.cold_f16_unpacks;
            }
            const int slot=(int)(c.cold_f16_meta_count++%Chunk::kColdF16MetaSlots);
            if(c.cold_f16_meta_recorded[slot]) CK(cudaEventSynchronize(c.cold_f16_meta_done[slot]));
            uint8_t* host=c.cold_f16_meta_host+(size_t)slot*c.cold_f16_meta_stride;
            uint8_t* device=c.cold_f16_meta_dev+(size_t)slot*c.cold_f16_meta_stride;
            const size_t desc_bytes=align256((size_t)cap*sizeof(F16PhaseRow));
            auto* hd=(F16PhaseRow*)host;
            auto* hp=(unsigned long long*)(host+desc_bytes);
            auto* dp=(unsigned long long*)(device+desc_bytes);
            for(int q=0;q<ne;++q) {
                const auto& row=batch.rows[(size_t)q];
                hd[q]={row.row0,row.rows,row.pad0,row.padded};
                const uint16_t* weight=c.cold_f16+(size_t)q*c.cold_f16_unit;
                const size_t xoff=(size_t)row.pad0*D,hoff=(size_t)row.pad0*FF;
                hp[q]=(unsigned long long)weight;
                hp[cap+q]=(unsigned long long)(weight+role_values);
                hp[2*cap+q]=(unsigned long long)(weight+2*role_values);
                hp[3*cap+q]=(unsigned long long)(c.x16+xoff);
                hp[4*cap+q]=(unsigned long long)(c.sh16+hoff);
                hp[5*cap+q]=(unsigned long long)(c.sg+hoff);
                hp[6*cap+q]=(unsigned long long)(c.su+hoff);
                hp[7*cap+q]=(unsigned long long)(c.cold_f16_y+xoff);
            }
            CK(cudaMemcpyAsync(device,host,c.cold_f16_meta_stride,cudaMemcpyHostToDevice,d.st));
            auto record_done=[&] {
                const cudaError_t status=cudaEventRecord(c.cold_f16_meta_done[slot],d.st);
                if(status!=cudaSuccess){err=std::string("cold F16 metadata event: ")+cudaGetErrorString(status);return false;}
                c.cold_f16_meta_recorded[slot]=true;return true;
            };
            auto blas_failed=[&](cublasStatus_t status) {
                if(status==CUBLAS_STATUS_NOT_SUPPORTED||status==CUBLAS_STATUS_ARCH_MISMATCH||status==CUBLAS_STATUS_ALLOC_FAILED) {
                    ++c.cold_f16_fallback_blas;
                    if(status==CUBLAS_STATUS_ALLOC_FAILED)(void)cudaGetLastError();
                    return record_done(); // used remains false: original full-range TC overwrites previous batches
                }
                err="bounded cold F16 cuBLAS failed, status "+std::to_string((int)status);return false;
            };
            const unsigned gx=(unsigned)std::min<size_t>(((size_t)batch.max_pad*D+255)/256,65535);
            f16_phase_gather<<<dim3(gx,(unsigned)ne),256,0,d.st>>>((const F16PhaseRow*)device,ne,batch.max_pad,
                                                                               c.x,D,c.row_tok+a,c.x16);
            CK(cudaGetLastError());
            struct Bucket{int first,n,rows;};std::vector<Bucket>buckets;
            for(int i=0;i<ne;) {
                int j=i+1;
                if(cold_f16_rule!=coldf16::Rule::exact)
                    while(j<ne&&batch.rows[(size_t)j].padded==batch.rows[(size_t)i].padded)++j;
                buckets.push_back({i,j-i,batch.rows[(size_t)i].padded});i=j;
            }
            const float one=1.f,zero=0.f;
            auto mm=[&](const Bucket& bucket,int wi,int ai,int yi,int n,int k) {
                if(cold_f16_rule==coldf16::Rule::exact) {
                    return cublasGemmEx(c.cold_f16_blas,CUBLAS_OP_T,CUBLAS_OP_N,n,bucket.rows,k,&one,
                        (const void*)hp[wi*cap+bucket.first],CUDA_R_16F,k,
                        (const void*)hp[ai*cap+bucket.first],CUDA_R_16F,k,&zero,
                        (void*)hp[yi*cap+bucket.first],CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                }
                return cublasGemmBatchedEx(c.cold_f16_blas,CUBLAS_OP_T,CUBLAS_OP_N,n,bucket.rows,k,&one,
                    (const void*const*)(dp+(size_t)wi*cap+bucket.first),CUDA_R_16F,k,
                    (const void*const*)(dp+(size_t)ai*cap+bucket.first),CUDA_R_16F,k,&zero,
                    (void*const*)(dp+(size_t)yi*cap+bucket.first),CUDA_R_32F,n,bucket.n,
                    CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            };
            for(const Bucket& bucket:buckets) {
                cublasStatus_t status=mm(bucket,0,3,5,FF,D);if(status!=CUBLAS_STATUS_SUCCESS)return blas_failed(status);
                status=mm(bucket,1,3,6,FF,D);if(status!=CUBLAS_STATUS_SUCCESS)return blas_failed(status);
            }
            const size_t hn=(size_t)batch.padded*FF;
            f16_phase_swiglu<<<(unsigned)std::min<size_t>((hn+255)/256,65535),256,0,d.st>>>(c.sg,c.su,hn,g.lim,c.sh16);
            CK(cudaGetLastError());
            for(const Bucket& bucket:buckets) {
                const cublasStatus_t status=mm(bucket,2,4,7,D,FF);if(status!=CUBLAS_STATUS_SUCCESS)return blas_failed(status);
            }
            const unsigned sx=(unsigned)std::min<size_t>(((size_t)batch.max_pad*std::max(D,FF)+255)/256,65535);
            f16_phase_scatter<<<dim3(sx,(unsigned)ne),256,0,d.st>>>((const F16PhaseRow*)device,ne,c.sh16,
                                                                 c.cold_f16_y,D,FF,c.h16,c.y);
            CK(cudaGetLastError());
            if(!record_done())return false;
            ++c.cold_f16_batches;padded_total+=(uint64_t)batch.padded;bucket_total+=buckets.size();
        }
        if(plan.light_experts) {
            // Preserve the ORIGINAL full range and maximum: mixed hot ranges select the exact NE16 light math.
            gg::iq::SetArgs q{};q.wbase=wbase+(size_t)(i0-wb0)*stride;q.stride=stride;
            for(int r=0;r<3;++r)q.tr[r]=l.lay.trellis[r];
            q.bounds=d_bounds+i0;q.row0=a;q.n_exp=i1-i0;q.max_rows=plan.max_rows;
            q.n_embd=D;q.n_ff=FF;q.limit=g.lim;q.tg=tg;q.td=td;
            q.xf=c.x;q.idx=c.row_tok+a;q.ldx=D;q.h=c.h16;q.ldo=FF;
            if(!gg::iq::light_gate_up(q,d.st)){++c.cold_f16_fallback_light;return true;}
            CK(cudaGetLastError());
            if(gu_event)CK(cudaEventRecord(gu_event,d.st));
            q.xf=nullptr;q.idx=nullptr;q.xh=c.h16;q.ldx=FF;q.y=c.y;q.ldo=D;
            if(!gg::iq::light_down(q,d.st)){++c.cold_f16_fallback_light;return true;}
            CK(cudaGetLastError());
        } else if(gu_event)CK(cudaEventRecord(gu_event,d.st));
        ++c.cold_f16_sets;c.cold_f16_rows+=(uint64_t)plan.hot_rows;c.cold_f16_padded_rows+=padded_total;
        c.cold_f16_buckets+=bucket_total;c.cold_f16_light_rows+=(uint64_t)plan.light_rows;
        c.cold_f16_light_experts+=(uint64_t)plan.light_experts;
        used=true;return true; // caller retains exactly ONE original moe_combine for the whole original range
#endif
    }

    // the experts [i0, i1) of a sorted set (blobs at wbase + i * stride; bounds b on the host, d_bounds on the device)
    // over the chunk's rows, in pieces of at most C rows, each added into acc in pick order
    bool run_set(Dev& d, Chunk& c, const Layer& l, const uint8_t* wbase, int wb0, size_t stride, const int* d_bounds,
                 const int* b, int i0, int i1, int T, std::string& err, const int* expert_ids = nullptr, int n_ids = 0) {
        const int D = g.dim, FF = l.ffn;
        while (i0 < i1) {
            int j = i0 + 1;
            while (j < i1 && b[j + 1] - b[i0] <= c.C) ++j;
            const int a = b[i0], nrows = b[j] - a;
            int max_rows = 0;
            for (int i = i0; i < j; ++i) max_rows = std::max(max_rows, b[i + 1] - b[i]);
            if (nrows > 0) {
                x3::SetArgs sa;
                sa.wbase = wbase + (size_t) (i0 - wb0) * stride;   // (wbase: set member wb0's blob)
                sa.stride = stride;
                sa.lay = l.nm > 1 ? l.hlay : l.lay;
                sa.n_exp = j - i0;
                sa.bounds = d_bounds + i0;
                sa.row0 = a;
                sa.max_rows = max_rows;
                sa.n_embd = D;
                sa.n_ff = FF;
                sa.limit = g.lim;
                EkMark ek{};
                const bool ekp = prof_chunk;
                if (ekp) {
                    std::lock_guard<std::mutex> lk(prof_mu);
                    ek.e0 = ek_ev();
                    ek.e1 = ek_ev();
                    ek.e2 = ek_ev();
                    ek.tg = gg::expert_type(sa.lay, 0);
                    ek.td = gg::expert_type(sa.lay, 2);
                    ek.nrows = nrows;
                    ek.nexp = j - i0;
                    ek.maxrows = max_rows;
                    ek.ff = FF;
                    cudaGetDevice(&ek.dev);
                    cudaEventRecord(ek.e0, d.st);
                }
                bool f16_used = false;
                if (!try_cold_f16_set(d,c,l,wbase,wb0,stride,d_bounds,b,i0,j,a,
                                      ekp ? ek.e1 : nullptr,f16_used,err)) return false;
                if (!f16_used && !try_f16_phase_set(d, c, l, wbase, wb0, stride, d_bounds, b, i0, j, expert_ids, n_ids, a,
                                      ekp ? ek.e1 : nullptr, f16_used, err)) return false;
                if (f16_used) {
                    // Both original compact outputs are ready; retain the original event/combiner/range below.
                } else if (sa.lay.fmt == gg::kFmtMXFP4) {
                    gg::set_gate_up(sa, c.x, D, c.row_tok + a, c.h16, FF, d.st);
                    if (ekp) cudaEventRecord(ek.e1, d.st);
                    gg::set_down(sa, c.h16, FF, c.y, D, d.st);
                } else {
                    x3::set_gate_up(sa, c.x, D, c.row_tok + a, c.h16, FF, d.st);
                    if (ekp) cudaEventRecord(ek.e1, d.st);
                    x3::set_down(sa, c.h16, FF, c.y, D, d.st);
                }
                if (ekp) {
                    cudaEventRecord(ek.e2, d.st);
                    std::lock_guard<std::mutex> lk(prof_mu);
                    ek_marks.push_back(ek);
                }
                if (dump_y != nullptr && (dump_y_layer < 0 || l.il == dump_y_layer)) {
                    std::vector<int> bb(b + i0, b + j + 1);
                    std::vector<uint16_t> hv((size_t) nrows * FF);
                    std::vector<float> yv((size_t) nrows * D);
                    CK(cudaStreamSynchronize(d.st));
                    CK(cudaMemcpy(hv.data(), c.h16, hv.size() * 2, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(yv.data(), c.y, yv.size() * 4, cudaMemcpyDeviceToHost));
                    std::lock_guard<std::mutex> lk(dump_mu);
                    const int64_t hdr[6] = {l.il, a, nrows, j - i0, wbase == c.ring || (wbase > c.ring &&
                                            wbase < c.ring + (size_t) c.ring_n * c.ring_stride) ? 1 : 0, max_rows};
                    std::fwrite(hdr, 8, 6, dump_y);
                    std::fwrite(bb.data(), 4, bb.size(), dump_y);
                    std::fwrite(hv.data(), 2, hv.size(), dump_y);
                    std::fwrite(yv.data(), 4, yv.size(), dump_y);
                    std::fflush(dump_y);
                }
#if !defined(STRATA_USE_HIP)
                // Same-stream matching moe_sort outputs. False means no launch; preserve the complete original call.
                if (!(devs.size()==1 && NH==1 && l.nm==1 &&
                      try_moe_combine_range_rows(c.acc,c.y,c.pos,c.rw,c.row_tok,T*l.k_exp,
                                                 T,l.k_exp,a,a+nrows,D,d.st)))
#endif
                moe_combine(c.acc, c.y, c.pos, c.rw, T, l.k_exp, a, a + nrows, D, d.st);
            }
            i0 = j;
        }
        (void) err;
        return true;
    }

    bool load_dense(int di, std::string& err) {
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        size_t* acct = &d.weights;
        for (Layer& l0 : L)
          for (int mi = 0; mi < l0.nm; ++mi) {
            Layer& l = member(l0, mi);
            if (l.dev != di) continue;
            // (a group's member: its share of the heads (wq_b columns, the sinks), of wo_a's groups and wo_b's rows,
            // of the shared expert's FFN; the rest whole)
            const int nm = l0.nm, na = g.heads / std::max(1, l.ahn), ha = l.ah0 / std::max(1, l.ahn);
            const std::string& P = l.name;
            const std::string A = P + "attn.";
            bool ok = mat(d, A + "wq_a", l.wq_a, err, acct) &&
                      (l.ahn == 0 || mat(d, A + "wq_b", l.wq_b, err, acct, ha, na, true)) &&
                      mat(d, A + "wkv", l.wkv, err, acct) &&
                      (l.ahn == 0 || mat(d, A + "wo_b", l.wo_b, err, acct, ha, na, false)) &&
                      f32(d, A + "q_norm.weight", l.q_norm, err, acct, (size_t) g.q_lora) &&
                      f32(d, A + "kv_norm.weight", l.kv_norm, err, acct, (size_t) g.hd) &&
                      f32(d, A + "attn_sink", l.sink, err, acct, (size_t) g.heads) &&
                      f32(d, P + "attn_norm.weight", l.attn_norm, err, acct, (size_t) g.dim) &&
                      f32(d, P + "ffn_norm.weight", l.ffn_norm, err, acct, (size_t) g.dim) &&
                      f32(d, P + "hc_attn_fn", l.hca_fn, err, acct) && f32(d, P + "hc_attn_scale", l.hca_scale, err, acct) &&
                      f32(d, P + "hc_attn_base", l.hca_base, err, acct) && f32(d, P + "hc_ffn_fn", l.hcf_fn, err, acct) &&
                      f32(d, P + "hc_ffn_scale", l.hcf_scale, err, acct) &&
                      f32(d, P + "hc_ffn_base", l.hcf_base, err, acct) &&
                      f32(d, P + "ffn.gate.weight", l.gate_w, err, acct, (size_t) l.n_exp * g.dim) &&
                      f32(d, P + "ffn.gate.bias", l.gate_bias, err, acct, (size_t) l.n_exp) &&
                      (l.mtp || img_tok < 0 || pk.get(P + "ffn.gate.bias_vl") == nullptr ||
                       f32(d, P + "ffn.gate.bias_vl", l.gate_bias_vl, err, acct, (size_t) l.n_exp)) &&
                      mat(d, P + "ffn.shared_experts.w1", l.s1, err, acct, mi, nm, true, l.il % nm) &&
                      mat(d, P + "ffn.shared_experts.w2", l.s2, err, acct, mi, nm, false, l.il % nm) &&
                      mat(d, P + "ffn.shared_experts.w3", l.s3, err, acct, mi, nm, true, l.il % nm);
            if (!ok) return false;
            l.sink += (size_t) l.ah0;
            for (int gi = 0; gi < l.ogn; ++gi)
                if (!mat(d, A + "wo_a.slice." + std::to_string(l.og0 + gi), l.wo_a[gi], err, acct)) return false;
            if (l.kv_src) {
                const std::string C = A + "compressor.";
                if (!mat(d, C + "wkv", l.c_wkv, err, acct) || !f32(d, C + "norm.weight", l.c_norm, err, acct)) return false;
                if (l.ratio > 1 && !mat(d, C + "wgate", l.c_wgate, err, acct)) return false;
                if (l.ratio > 1 && (!dalloc(d, l.c_kv, (size_t) l.ratio * g.hd, err) ||
                                    !dalloc(d, l.c_score, (size_t) l.ratio * g.hd, err)))
                    return false;
            }
            if (l.ix_src) {
                const std::string I = A + "indexer.";
                if (!mat(d, I + "wq_b", l.i_wq_b, err, acct) || !f16raw(d, I + "weights_proj.weight", l.i_wproj, err, acct))
                    return false;
                if (l.owns_k && (!mat(d, I + "wk", l.i_wk, err, acct) || !f32(d, I + "k_norm.weight", l.ik_norm, err, acct)))
                    return false;
            }
            if (l.eg >= 0) {
                if (!mat(d, P + "engram.wkv", l.eg_wkv, err, acct)) return false;
                if (pk.get(P + "engram.gate.weight") != nullptr) {   // a GGUF: q_weight * k_weight, folded already
                    if (!f32(d, P + "engram.gate.weight", l.eg_w, err, acct, (size_t) g.hc * g.dim)) return false;
                } else {
                    float *qw = nullptr, *kw = nullptr;
                    if (!f32(d, P + "engram.q_weight", qw, err, acct) || !f32(d, P + "engram.k_weight", kw, err, acct))
                        return false;
                    std::vector<float> a((size_t) g.hc * g.dim), b(a.size());
                    CK(cudaMemcpy(a.data(), qw, a.size() * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(b.data(), kw, b.size() * 4, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < a.size(); ++i) a[i] *= b[i];   // only ever used as a product
                    CK(cudaMemcpy(qw, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
                    l.eg_w = qw;
                }
            }
            if (!dalloc(d, l.win, (size_t) g.win * win_rb(), err, &d.caches)) return false;
            if (g.n_mtp > 0 && !l.mtp) {
                if (!dalloc(d, l.ring_bak, (size_t) kWin * win_rb(), err)) return false;
                if (l.kv_src && l.ratio > 1 && (!dalloc(d, l.c_snap, (size_t) 2 * l.ratio * g.hd, err) ||
                                                !dalloc(d, l.c_rows, (size_t) 2 * kWin * g.hd, err)))
                    return false;
            }
            if (!dalloc(d, l.vram_tab, (size_t) l.n_exp, err, acct)) return false;
          }
        if (d.grp == devs[(size_t) head_dev].grp) {   // the head: each member of its group a share of the columns
            const int hn = head_nm();
            if (!mat(d, "head", d.head, err, acct, d.mi, hn, true) || !f32(d, "norm.weight", d.norm_w, err, acct, (size_t) g.dim))
                return false;
            if (g.n_mtp > 0 && !f32(d, "mtp." + std::to_string(g.n_mtp - 1) + ".norm.weight", d.d_norm, err, acct, (size_t) g.dim))
                return false;
            if (!dalloc(d, d.logits_part, (size_t) kWin * ((g.vocab / 128 + hn - 1) / hn * 128), err)) return false;
            if (g.n_mtp > 0 && hn > 1) {   // the markov steps over the group (mk_split): this member's rows of the head
                const Tensor* th = pk.get("mtp." + std::to_string(g.n_mtp - 1) + ".markov_head.head.weight");
                const int R = g.d_rank;
                head_cols(hn, d.mi, d.mk_v0, d.mk_vn);   // (its share of the head's columns: the same rows)
                // F16 as the pack holds it, or a GGUF drafter's BF16 rounded to F16 as f16raw rounds d_mk_head (the same
                // values as the lead's whole rows); another type: no mk_split (the lead's whole rows)
                if (th != nullptr && (th->dtype == "F16" || th->dtype == "BF16") && th->bytes == (uint64_t) g.vocab * R * 2) {
                    std::vector<uint16_t> rows((size_t) d.mk_vn * R);
                    if (!pk.read_at(th->file, th->off + (uint64_t) d.mk_v0 * R * 2, rows.size() * 2, rows.data())) {
                        err = "the markov head: read failed";
                        return false;
                    }
                    if (th->dtype == "BF16")
                        for (uint16_t& h : rows) h = strata::kernels::f16_from_f32(bf16_to_f32(h));
                    if (!dalloc(d, d.mk_share, rows.size(), err, acct) || !dalloc(d, d.mk_erow, (size_t) R, err) ||
                        !dalloc(d, d.mk_ints, (size_t) kWin + 2, err))
                        return false;
                    CK(cudaMemcpy(d.mk_share, rows.data(), rows.size() * 2, cudaMemcpyHostToDevice));
                }
            }
        }
        if (di == head_dev) {
            head = d.head;
            norm_w = d.norm_w;
            d_norm = d.d_norm;
            if (g.n_mtp > 0) {
                const std::string Z = "mtp." + std::to_string(g.n_mtp - 1) + ".";
                if (!f32(d, Z + "confidence_head.proj.weight", d_conf, err, acct, (size_t) (g.dim + g.d_rank)))
                    return false;
                const Tensor* te = pk.get(Z + "markov_head.embed.weight");
                const Tensor* th = pk.get(Z + "markov_head.head.weight");
                if (te == nullptr || th == nullptr || te->dtype != "BF16" || (th->dtype != "F16" && th->dtype != "BF16")) {
                    err = Z + "markov_head: not the BF16 embed and F16 / BF16 head expected";
                    return false;
                }
                if (!dalloc(d, d_mk_embed, te->bytes / 2, err, acct) || !upload(d, *te, d_mk_embed, err) ||
                    !f16raw(d, Z + "markov_head.head.weight", d_mk_head, err, acct))
                    return false;
            }
        }
        if (g.n_mtp > 0 && di == d_dev) {
            if (!mat(d, "mtp.0.main_proj", d_main_proj, err, acct) ||
                !f32(d, "mtp.0.main_norm.weight", d_main_norm, err, acct, (size_t) g.dim))
                return false;
        }
        return true;
    }

    void set_vm(Layer& l, int e, int slot) {
        uint8_t* b = l.vram_base + (size_t) slot * l.blob;
        if (l.lay.fmt == gg::kFmtMXFP4) {
            for (int r = 0; r < 3; ++r) {
                x3::Mat& m = l.vm[(size_t) e][(size_t) r];
                m = x3::Mat{};
                m.K = gg::expert_type(l.lay, r);   // (a low-bit type: its GEMV, ds4_iq's)
                m.fmt = m.K == gg::kTypeMXFP4 ? gg::kFmtMXFP4 : gg::kFmtGG;
                m.raw = b + l.lay.trellis[r];
                m.k = r < 2 ? g.dim : g.ff;
                m.n = r < 2 ? g.ff : g.dim;
            }
            return;
        }
        for (int r = 0; r < 3; ++r) {
            x3::Mat& m = l.vm[(size_t) e][(size_t) r];
            m.trellis = (const uint32_t*) (b + l.lay.trellis[r]);
            m.suh = (const uint16_t*) (b + l.lay.suh[r]);
            m.svh = (const uint16_t*) (b + l.lay.svh[r]);
            m.k = r < 2 ? g.dim : g.ff;
            m.n = r < 2 ? g.ff : g.dim;
            m.K = l.lay.K[r];
            m.cb = 2;
            m.ld = 0;
        }
    }
    // an expert's VRAM blob -> its CPU-lane part X, and its NH parts -> the blob (the cache's swaps)
    void blob_to_part(const Layer& l, const uint8_t* blob, int X, uint8_t* dst) const {
        for (int r = 0; r < 3; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r];
            const int k = l.hl.k[r], nt = l.hl.n[r] * NH / 16, nth = l.hl.n[r] / 16;
            const uint8_t* tr = blob + l.lay.trellis[r];
            for (int sl = 0; sl < k / 16; ++sl)
                std::memcpy(dst + l.hl.trellis[r] + (size_t) sl * nth * tile, tr + ((size_t) sl * nt + (size_t) X * nth) * tile,
                            (size_t) nth * tile);
            std::memcpy(dst + l.hl.suh[r], blob + l.lay.suh[r], (size_t) k * 2);
            std::memcpy(dst + l.hl.svh[r], blob + l.lay.svh[r] + (size_t) X * l.hl.n[r] * 2, (size_t) l.hl.n[r] * 2);
        }
    }
    void parts_to_blob(const Layer& l, const std::array<uint8_t*, 4>& parts, uint8_t* blob) const {
        for (int r = 0; r < 3; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r];
            const int k = l.hl.k[r], nt = l.hl.n[r] * NH / 16, nth = l.hl.n[r] / 16;
            uint8_t* tr = blob + l.lay.trellis[r];
            for (int X = 0; X < NH; ++X)
                for (int sl = 0; sl < k / 16; ++sl)
                    std::memcpy(tr + ((size_t) sl * nt + (size_t) X * nth) * tile,
                                parts[(size_t) X] + l.hl.trellis[r] + (size_t) sl * nth * tile, (size_t) nth * tile);
            std::memcpy(blob + l.lay.suh[r], parts[0] + l.hl.suh[r], (size_t) k * 2);
            for (int X = 0; X < NH; ++X)
                std::memcpy(blob + l.lay.svh[r] + (size_t) X * l.hl.n[r] * 2, parts[(size_t) X] + l.hl.svh[r],
                            (size_t) l.hl.n[r] * 2);
        }
    }

    // an expert's blob -> member m's part (gate / up columns [ff0, ff0 + ffn), down rows of the same), and back
    void blob_to_half(const Layer& l, const uint8_t* blob, int m, uint8_t* dst) const {
        if (l.lay.fmt == gg::kFmtMXFP4) return mx_blob_to_half(l, blob, m, dst);
        const Layer& x = member(l, m);
        const int f0 = x.ff0, fn = x.ffn;
        for (int r = 0; r < 2; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r], nt = (size_t) g.ff / 16, nth = (size_t) fn / 16;
            for (size_t sl = 0; sl < (size_t) g.dim / 16; ++sl)
                std::memcpy(dst + x.hlay.trellis[r] + sl * nth * tile,
                            blob + l.lay.trellis[r] + (sl * nt + (size_t) f0 / 16) * tile, nth * tile);
            std::memcpy(dst + x.hlay.suh[r], blob + l.lay.suh[r], (size_t) g.dim * 2);
            std::memcpy(dst + x.hlay.svh[r], blob + l.lay.svh[r] + (size_t) f0 * 2, (size_t) fn * 2);
        }
        const size_t row = (size_t) (g.dim / 16) * 32 * l.lay.K[2];
        std::memcpy(dst + x.hlay.trellis[2], blob + l.lay.trellis[2] + (size_t) (f0 / 16) * row, (size_t) (fn / 16) * row);
        std::memcpy(dst + x.hlay.suh[2], blob + l.lay.suh[2] + (size_t) f0 * 2, (size_t) fn * 2);
        std::memcpy(dst + x.hlay.svh[2], blob + l.lay.svh[2], (size_t) g.dim * 2);
    }
    void half_to_blob(const Layer& l, const uint8_t* src, int m, uint8_t* blob) const {
        if (l.lay.fmt == gg::kFmtMXFP4) return mx_half_to_blob(l, src, m, blob);
        const Layer& x = member(l, m);
        const int f0 = x.ff0, fn = x.ffn;
        for (int r = 0; r < 2; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r], nt = (size_t) g.ff / 16, nth = (size_t) fn / 16;
            for (size_t sl = 0; sl < (size_t) g.dim / 16; ++sl)
                std::memcpy(blob + l.lay.trellis[r] + (sl * nt + (size_t) f0 / 16) * tile,
                            src + x.hlay.trellis[r] + sl * nth * tile, nth * tile);
            if (m == 0) std::memcpy(blob + l.lay.suh[r], src + x.hlay.suh[r], (size_t) g.dim * 2);
            std::memcpy(blob + l.lay.svh[r] + (size_t) f0 * 2, src + x.hlay.svh[r], (size_t) fn * 2);
        }
        const size_t row = (size_t) (g.dim / 16) * 32 * l.lay.K[2];
        std::memcpy(blob + l.lay.trellis[2] + (size_t) (f0 / 16) * row, src + x.hlay.trellis[2], (size_t) (fn / 16) * row);
        std::memcpy(blob + l.lay.suh[2] + (size_t) f0 * 2, src + x.hlay.suh[2], (size_t) fn * 2);
        if (m == 0) std::memcpy(blob + l.lay.svh[2], src + x.hlay.svh[2], (size_t) g.dim * 2);
    }

    bool place_experts(std::string& err, bool empty = false) {
        // (Windows, an integrated GPU: what its experts may commit - commit_room - and what that kept off the GPU)
        double room = 1e300, kept = 0;
        for (const Dev& d : devs)
            if (commit_charged(d)) room = commit_room();
        const double room0 = room;
        // per group: what its tightest member has left (each member holds its part of every VRAM expert of the
        // group's layers, its unit bytes a slot - counted at the members' mean: an uneven split's odd blocks go round
        // the members layer by layer, so each member's total comes to about the mean's)
        auto ucount = [](const Layer& l) {
            double s = 0;
            for (int m = 0; m < l.nm; ++m) s += (double) unit(member(l, m));
            return s / l.nm;
        };
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            double budget = 1e300;
            bool charged = false;
            for (int m : groups[gi].mem) {
                cudaSetDevice(devs[(size_t) m].id);
                size_t fr = 0, tot = 0;
                CK(cudaMemGetInfo(&fr, &tot));
                budget = std::min(budget, std::max(0.0, ((double) fr - (double) ((size_t) o.reserve_mib << 20))) *
                                              (empty ? 0.0 : o.vram_experts));
                charged = charged || commit_charged(devs[(size_t) m]);
            }
            if (charged && room < 1e300) {   // (each member's slots charged: the group's share of the room)
                double all = 0;
                for (const Layer& l : L)
                    if (l.grp == (int) gi) all += (double) unit(l) * l.n_exp;
                const double nm = (double) groups[gi].mem.size(), capped = std::min(budget, room / nm);
                kept += std::max(0.0, std::min(budget, all) - std::min(capped, all)) * nm;
                budget = capped;
                room = std::max(0.0, room - std::min(budget, all) * nm);
            }
            double budget_t = budget;   // the drafter's experts first, all of them (every draft uses its blocks)
            for (Layer& l : L)
                if (l.grp == (int) gi && l.mtp) {
                    l.n_vram = std::min(l.n_exp, (int) std::floor(budget_t / (double) ucount(l)));
                    budget_t -= (double) l.n_vram * ucount(l);
                }
            double want = 0;
            for (const Layer& l : L)
                if (l.grp == (int) gi && !l.mtp && !l.cpu_only) want += (double) ucount(l) * l.n_exp;
            const double frac = want > 0 ? std::min(1.0, std::max(0.0, budget_t) / want) : 0.0;
            for (Layer& l : L) {
                if (l.grp != (int) gi || l.mtp) continue;
                if (l.cpu_only) {   // (its type has no device kernel: the CPU lane reads every expert of it)
                    l.n_vram = 0;
                    continue;
                }
                l.n_vram = std::min(l.n_exp, (int) std::floor(frac * l.n_exp));
                // never more than fits: the floor of each layer's share can still round up the total
                while (l.n_vram > 0 && (double) ucount(l) * l.n_vram > budget) --l.n_vram;
            }
            double used = 0;
            for (Layer& l : L)
                if (l.grp == (int) gi) used += (double) ucount(l) * l.n_vram;
            // what the floors left: whole experts, layer by layer
            for (Layer& l : L) {
                if (l.grp != (int) gi || l.cpu_only) continue;
                while (l.n_vram < l.n_exp && used + (double) ucount(l) <= budget) {
                    ++l.n_vram;
                    used += (double) ucount(l);
                }
            }
        }
        if (kept > 0) {   // the commit kept experts off the GPU: say so, and the page file that keeps them on it
            const os::Commit c = os::commit();
            const double gib = (double) (1ull << 30);
            const double pf = c.limit != ~uint64_t{0} && c.ram != ~uint64_t{0} && c.limit > c.ram ? (double) (c.limit - c.ram) : 0.0;
            commit_kept = kept;
            pagefile_hint = std::ceil((pf + kept) / gib);
            std::fprintf(stderr,
                         "ds4: WARNING: %.1f GB of experts stay off the GPU - Windows charges the GPU's memory to its "
                         "commit (RAM + page file) and can commit only %.1f GB more for them (%.0f GB kept for the "
                         "other programs). A page file of %.0f GB (%.0f MB; now %.0f GB) keeps them on the GPU: System "
                         "> About > Advanced system settings > Performance > Advanced > Virtual memory, then restart\n",
                         kept / 1e9, room0 / 1e9, commit_margin() / 1e9, pagefile_hint, pagefile_hint * 1024.0, pf / gib);
        } else if (room0 < 1e300 && o.verbose) {
            std::fprintf(stderr, "ds4: Windows' commit has room for the GPU's experts (%.1f GB beside %.0f GB for the other "
                                 "programs)\n", room0 / 1e9, commit_margin() / 1e9);
        }
        for (Layer& l : L) {
            // the hottest experts of the profile first, then by index
            std::vector<int> order((size_t) l.n_exp);
            std::iota(order.begin(), order.end(), 0);
            const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b]; });
            l.slot.assign((size_t) l.n_exp, -1);
            l.host.assign((size_t) l.n_exp, nullptr);
            // a layer with experts on both sides keeps one VRAM slot free for the cache's swaps
            l.n_slots = std::min(l.n_vram, l.n_exp);
            if (o.expert_cache && l.n_slots >= 2 && l.n_slots < l.n_exp) {   // (every expert fits: nothing to swap)
                l.n_vram = l.n_slots - 1;
                l.spare_vram = l.n_slots - 1;
            } else {
                l.n_vram = l.n_slots;
                l.spare_vram = -1;
            }
            for (int i = 0; i < l.n_vram; ++i) l.slot[(size_t) order[(size_t) i]] = i;
            for (int m = 0; m < l.nm && l.n_slots > 0; ++m) {
                Layer& x = member(l, m);
                Dev& d = devs[(size_t) x.dev];
                cudaSetDevice(d.id);
                void* v = nullptr;
                if (cudaMalloc(&v, (size_t) l.n_slots * unit(x)) != cudaSuccess) {
                    err = "cudaMalloc of layer " + std::to_string(l.il) + "'s " + std::to_string(l.n_slots) +
                          " experts on CUDA" + std::to_string(d.id) + " failed";
                    return false;
                }
                d.allocs.push_back(v);
                d.experts += (size_t) l.n_slots * unit(x);
                x.vram_base = (uint8_t*) v;
            }
            l.heat.assign((size_t) l.n_exp, 0.0f);
            {   // the profile as the starting heat: its counts scaled to ~512 hits a layer
                double sum = 0;
                for (int e = 0; e < l.n_exp; ++e) sum += p[e];
                const double sc = sum > 0 ? 512.0 / sum : 0.0;
                for (int e = 0; e < l.n_exp; ++e) l.heat[(size_t) e] = (float) (p[e] * sc);
            }
            const int n_host = l.n_exp - l.n_vram;
            {   // the CPU lane's parts of an expert (in place: the whole expert, its pieces where the file has them)
                size_t off = 0;
                for (int r = 0; r < 3; ++r) {
                    const int k = r < 2 ? g.dim : g.ff, nf = r < 2 ? g.ff : g.dim, n = nf / NH;
                    l.hl.k[r] = k;
                    l.hl.n[r] = n;
                    l.hl.trellis[r] = file_experts ? l.piece_src[r][0] : off;
                    off += align256((size_t) (k / 16) * (n / 16) * 32 * l.lay.K[r]);
                    l.hl.suh[r] = file_experts ? l.piece_src[r][1] : off;
                    off += align256((size_t) k * 2);
                    l.hl.svh[r] = file_experts ? l.piece_src[r][2] : off;
                    off += align256((size_t) n * 2);
                }
                l.hl.bytes = off;
            }
            l.hh.assign((size_t) l.n_exp, {nullptr, nullptr, nullptr, nullptr});
            if (l.lay.fmt == gg::kFmtMXFP4) {   // a GGUF's: read in place from the file's mapping
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) l.host[(size_t) e] = (uint8_t*) mx_mat(l, e, 0);
            } else if (file_experts) {   // every expert's file mapped (the cache's swaps hand VRAM experts back to the files)
                for (int e = 0; e < l.n_exp; ++e)
                    if (pk.map(l.region_file[(size_t) e]) == nullptr) {
                        err = "mapping " + pk.files[(size_t) l.region_file[(size_t) e]] + " failed";
                        return false;
                    }
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) {
                        l.hh[(size_t) e][0] = file_part(l, e);
                        l.host[(size_t) e] = l.hh[(size_t) e][0];
                    }
            } else if (n_host > 0) {
                const int n_parts = n_host + (l.spare_vram >= 0 ? 1 : 0);   // (+ the cache's spare)
                const size_t bytes = (size_t) n_parts * l.hl.bytes;
                for (int X = 0; X < NH; ++X) {
                    void* m = node_map(bytes, X, NH);
                    if (m == nullptr) {
                        err = "host memory for layer " + std::to_string(l.il) + "'s " + std::to_string(n_host) +
                              " CPU experts (" + std::to_string(bytes >> 20) + " MiB on node " + std::to_string(X) + ")";
                        return false;
                    }
                    host_maps.emplace_back(m, bytes);
                    int hi = 0;
                    for (int e = 0; e < l.n_exp; ++e)
                        if (l.slot[(size_t) e] < 0) l.hh[(size_t) e][(size_t) X] = (uint8_t*) m + (size_t) (hi++) * l.hl.bytes;
                    if (l.spare_vram >= 0) l.spare_host[(size_t) X] = (uint8_t*) m + (size_t) hi * l.hl.bytes;
                }
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) l.host[(size_t) e] = l.hh[(size_t) e][0];
            }
            l.vm.assign((size_t) l.n_exp, {});
            if (l.nm == 1)   // (the host-routed window path's views: a group's members take the device-routed path)
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] >= 0) set_vm(l, e, l.slot[(size_t) e]);
            // The one-shot prefill phase reuses the original ZERO-slot plan, never a populated pool.
            // Its pinned publication scratch is already allocated; do not leak a second copy on promotion.
            if (l.tab_pin == nullptr && !halloc(l.tab_pin, (size_t) 2 * l.nm, err)) return false;
            for (int m = 0; m < l.nm; ++m) {
                Layer& x = member(l, m);
                std::vector<unsigned long long> tab((size_t) l.n_exp, 0ull);
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] >= 0)
                        tab[(size_t) e] = (unsigned long long) (x.vram_base + (size_t) l.slot[(size_t) e] * unit(x));
                cudaSetDevice(devs[(size_t) x.dev].id);
                CK(cudaMemcpy(x.vram_tab, tab.data(), tab.size() * 8, cudaMemcpyHostToDevice));
            }
        }
        return true;
    }

    // a GGUF's VRAM experts, in file order, by a few threads through pinned staging blobs (the CPU lane's stay in the
    // file: the mapping reads them on first use)
    bool load_experts_mx(std::string& err, bool resident_only = false) {
        struct Job {
            int layer, e;
        };
        std::vector<Job> jobs;
        size_t maxb = 0;
        for (const Layer& l : L) {
            maxb = std::max(maxb, l.blob);
            for (int e = 0; e < l.n_exp; ++e)
                if (l.slot[(size_t) e] >= 0) jobs.push_back({l.il, e});
        }
        std::sort(jobs.begin(), jobs.end(), [&](const Job& a, const Job& b) {
            const Layer &la = L[(size_t) a.layer], &lb = L[(size_t) b.layer];
            if (la.mx_file[0] != lb.mx_file[0]) return la.mx_file[0] < lb.mx_file[0];
            return la.mx_off[0] + (uint64_t) a.e * la.mx_stride[0] < lb.mx_off[0] + (uint64_t) b.e * lb.mx_stride[0];
        });
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::mutex emu;
        std::string ferr;
        std::atomic<int64_t> done_bytes{0};
        const double t0 = now_ms();
        std::vector<std::thread> th;
        for (int t = 0; t < 12; ++t)
            th.emplace_back([&] {
                uint8_t *stage = nullptr, *stage2 = nullptr;
                cudaHostAlloc((void**) &stage, maxb, cudaHostAllocPortable);
                cudaHostAlloc((void**) &stage2, maxb, cudaHostAllocPortable);
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= jobs.size() || failed.load()) break;
                    const Layer& l = L[(size_t) jobs[i].layer];
                    const int e = jobs[i].e;
                    bool ok = stage != nullptr && stage2 != nullptr && read_mx(l, e, stage);
                    for (int m = 0; m < l.nm && ok; ++m) {   // (a group: each member its part)
                        const Layer& x = member(l, m);
                        cudaSetDevice(devs[(size_t) x.dev].id);
                        if (l.nm > 1) blob_to_half(l, stage, m, stage2);
                        ok = cudaMemcpy(x.vram_base + (size_t) l.slot[(size_t) e] * unit(l), l.nm > 1 ? stage2 : stage,
                                        unit(l), cudaMemcpyHostToDevice) == cudaSuccess;
                    }
                    if (!ok) {
                        std::lock_guard<std::mutex> lk(emu);
                        ferr = "loading layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
                        failed.store(true);
                        break;
                    }
                    const int64_t db = done_bytes.fetch_add((int64_t) l.blob) + (int64_t) l.blob;
                    if (o.verbose && (i % 1024) == 0)
                        std::fprintf(stderr, "ds4: experts %zu / %zu (%.1f GB, %.2f GB/s)\n", i, jobs.size(), db / 1e9,
                                     db / 1e6 / std::max(1.0, now_ms() - t0));
                }
                if (stage) cudaFreeHost(stage);
                if (stage2) cudaFreeHost(stage2);
            });
        for (auto& t : th) t.join();
        if (failed.load()) {
            err = ferr;
            return false;
        }
        // Promotion after the one-shot phase must not recreate/clear an existing RAM store or copied tier.
        if (resident_only) return true;
        // as load_experts in place: the VRAM experts' pages leave the cache, the most-routed of the others (the
        // profile's share of their layer, then by index) are read once, up to warm_bytes, so the cache starts with them
        struct Warm {
            int layer, e;
            double share;
        };
        std::vector<Warm> warm;
        for (const Layer& l : L) {
            const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
            double sum = 0;
            for (int e = 0; e < l.n_exp; ++e) sum += p[e];
            for (int e = 0; e < l.n_exp; ++e) {
                if (l.slot[(size_t) e] >= 0) {
                    for (int r = 0; r < 3; ++r)
                        os::advise(pk.fds[(size_t) l.mx_file[r]], (l.mx_off[r] + (uint64_t) e * l.mx_stride[r]), mx_part_bytes(l, r), false);
                } else {
                    warm.push_back({l.il, e, sum > 0 ? p[e] / sum : 0.0});
                }
            }
        }
        std::stable_sort(warm.begin(), warm.end(), [](const Warm& a, const Warm& b) { return a.share > b.share; });
        if (!ram_tier && !warm.empty()) {   // copied into RAM (copy_experts decides), the rest read ahead into what is left
            std::vector<std::pair<int, int>> hot;
            for (const Warm& w : warm) hot.push_back({w.layer, w.e});
            if (!copy_experts(hot, err)) return false;
            if (copy_mode) {
                warm.erase(std::remove_if(warm.begin(), warm.end(),
                                          [&](const Warm& w) { return L[(size_t) w.layer].ram[(size_t) w.e] >= 0; }),
                           warm.end());
                const uint64_t av = ram_avail();
                if (av != ~uint64_t{0}) warm_bytes = av > kRamMargin ? std::min<uint64_t>(warm_bytes, av - kRamMargin) : 0;
            }
        }
        if (ram_tier && warm_bytes > 0 && !warm.empty()) {
            // chunks of 32 slots up to warm_bytes (all the RAM the rest leaves, kRamMargin kept), one slot kept free for
            // a move in; the hottest experts read in, their pages then dropped from the page cache.  Plain memory, not
            // pinned: only the CPU reads it (what goes to a GPU passes the pinned staging), and pinned memory is capped on
            // Windows - the engine's own pinned buffers come after this
            const size_t want = std::min(warm.size(), (size_t) (warm_bytes / maxb));
            while (ram_slots.size() < want + 1) {
                const size_t n = std::min<size_t>(32, want + 1 - ram_slots.size());
                uint8_t* c = (uint8_t*) os::anon_alloc(n * maxb, -1, 1);
                if (c == nullptr) break;
                ram_chunks.push_back(c);
                ram_chunk_bytes.push_back(n * maxb);
                for (size_t i = 0; i < n; ++i) ram_slots.push_back(c + i * maxb);
            }
            ram_held = ram_slots.empty() ? 0 : (int) ram_slots.size() - 1;
            for (Layer& l : L)
                if (l.lay.fmt == gg::kFmtMXFP4) l.ram.assign((size_t) l.n_exp, -1);
            std::atomic<size_t> ri{0};
            std::atomic<bool> rfail{false};
            std::vector<std::thread> rt;
            for (int t = 0; t < 12 && ram_held > 0; ++t)
                rt.emplace_back([&] {
                    for (size_t i; (i = ri.fetch_add(1)) < (size_t) ram_held;) {
                        const Layer& l = L[(size_t) warm[i].layer];
                        if (!read_mx_file(l, warm[i].e, ram_slots[i])) rfail.store(true);
                        for (int r = 0; r < 3; ++r)
                            os::advise(pk.fds[(size_t) l.mx_file[r]], l.mx_off[r] + (uint64_t) warm[i].e * l.mx_stride[r],
                                       mx_part_bytes(l, r), false);
                    }
                });
            for (auto& t : rt) t.join();
            if (rfail.load()) {
                err = "reading the RAM tier's experts from the model's files failed";
                return false;
            }
            for (int i = 0; i < ram_held; ++i) L[(size_t) warm[(size_t) i].layer].ram[(size_t) warm[(size_t) i].e] = i;
            if (ram_held > 0) ram_free.push_back(ram_held);
            if (o.expert_cache && ram_held > 0) ram_th = std::thread([this] { ram_loop(); });
            if (o.verbose)
                std::fprintf(stderr, "ds4: RAM tier: %d experts (%.1f GB of RAM), the hottest after VRAM's; the other %zu "
                                     "read from the model's files (%.1f s)\n",
                             ram_held, ram_slots.size() * (double) maxb / 1e9, warm.size() - (size_t) ram_held,
                             (now_ms() - t0) / 1000.0);
            return true;
        }
        size_t nw = 0;
        for (double b = 0; nw < warm.size() && b + (double) L[(size_t) warm[nw].layer].blob <= (double) warm_bytes; ++nw)
            b += (double) L[(size_t) warm[nw].layer].blob;
        std::atomic<size_t> wi{0};
        std::vector<std::thread> wt;
        for (int t = 0; t < 12 && nw > 0; ++t)
            wt.emplace_back([&] {
                std::vector<uint8_t> buf(maxb);
                for (size_t i; (i = wi.fetch_add(1)) < nw;) read_mx(L[(size_t) warm[i].layer], warm[i].e, buf.data());
            });
        for (auto& t : wt) t.join();
        if (o.verbose && nw > 0)
            std::fprintf(stderr, "ds4: %zu CPU-lane experts read ahead (%.1f GB) in %.1f s\n", nw,
                         nw * (double) L[0].blob / 1e9, (now_ms() - t0) / 1000.0);
        return true;
    }

    // PURE metadata snapshot/validation. It does NOT synchronize or establish quiescence.
    // The initial-copy caller has no live readers. A future rearm caller must first
    // prove worker/cache/CPU-lane/DMA quiescence before calling this unchanged routine.
    bool reconstruct_copied_ram_inventory(const std::vector<ramdomain::Domain>& domains,
                                         ramdomain::Inventory& inventory,std::string& err) const {
        inventory={};
        if(!record_copied_ram_domains || !copy_mode || NH!=1 || store || ram_tier || devs.size()!=1 ||
           ram_held<0 || ram_chunks.size()!=ram_chunk_bytes.size() ||
           cp_lay.size()!=L.size() || copy_spare.size()!=L.size()) {
            err="copied-RAM domains: actual model geometry/layout is unavailable";return false;
        }
        std::vector<ramdomain::Arena> arenas;
        std::vector<std::array<uintptr_t,2>> slots;
        std::vector<ramdomain::LayerView> live;
        try {
            arenas.reserve(ram_chunks.size());slots.reserve(ram_part.size());live.reserve(L.size());
            for(size_t i=0;i<ram_chunks.size();++i)
                arenas.push_back({(uintptr_t)ram_chunks[i],ram_chunk_bytes[i]});
            for(const auto& p:ram_part)slots.push_back({(uintptr_t)p[0],(uintptr_t)p[1]});
            for(const Layer& l:L) {
                if(l.il<0 || (size_t)l.il>=L.size() || l.nm!=1 || l.dev!=0 || l.lay.fmt!=gg::kFmtMXFP4) {
                    err="copied-RAM domains: actual single-card GGUF layer binding changed";return false;
                }
                ramdomain::LayerView v;
                v.layer=l.il;v.expert_count=l.n_exp;v.gpu_capacity=l.n_slots;v.gpu_resident=l.n_vram;
                v.gpu_spare=l.spare_vram;v.ram_spare=copy_spare[(size_t)l.il];
                v.gpu=l.slot;v.ram=l.ram;v.offsets=cp_lay[(size_t)l.il];
                for(int r=0;r<3;++r)v.role_bytes[(size_t)r]=mx_part_bytes(l,r);
                live.push_back(std::move(v));
            }
            const ramdomain::Geometry geometry{NH,copy_mode,store!=nullptr,ram_tier,(size_t)ram_held};
            return ramdomain::reconstruct(geometry,domains,arenas,slots,live,inventory,err);
        } catch(...) {
            inventory={};err="copied-RAM domains: temporary inventory memory unavailable";return false;
        }
    }
    bool inspect_copied_ram_inventory(ramdomain::Inventory& inventory,std::string& err) const {
        if(!copied_ram_domains_published) {
            inventory={};err="copied-RAM domains have not been successfully published";return false;
        }
        return reconstruct_copied_ram_inventory(copied_ram_domains,inventory,err);
    }

    // a GGUF's CPU-lane experts copied into RAM (copy_mode), decided once VRAM's are in - on a unified-memory APU the GPU
    // has taken its share of the RAM by then, so the copies only get what it leaves.  `hot`: the CPU lane's experts, the
    // most-routed first.  --mmap-experts off: all that fit beside kRamMargin (a warning if not all do); on and auto:
    // none - read in place: the page cache follows a conversation's experts where a static set (the profile's
    // hottest) does not (the RAM tier's lesson), and copies that all fit were no faster; DS4_COPY_GB=<n>: the
    // hottest n GB (the static hybrid, for tests).  Each copy in NH parts on the NUMA nodes (DS4_COPY_NUMA=1: one), read from the files in file
    // order by a few threads, anonymous memory on huge pages (Windows: plain), not pinned (only the CPU reads it; what
    // goes to a GPU passes the pinned staging) - a layer that cannot get its memory keeps its experts in place
    bool copy_experts(const std::vector<std::pair<int, int>>& hot, std::string& err, bool late = false,
                      bool* interrupted_out = nullptr) {
        if (interrupted_out) *interrupted_out = false;
        std::vector<ramdomain::Domain> pending_ram_domains;
        if(record_copied_ram_domains) {
            if(late || !pk.gguf || devs.size()!=1 || o.mmap_experts!=0 || copy_mode || ram_tier || store ||
               !ram_chunks.empty() || !ram_part.empty() || copied_ram_domains_published || !copied_ram_domains.empty()) {
                err="RAM-domain recording requires a fresh initial single-GPU copied-RAM load";return false;
            }
            try {pending_ram_domains.reserve(L.size());}
            catch(...) {err="RAM-domain metadata capacity unavailable";return false;}
        }
        if ((!late && o.mmap_experts > 0) || hot.empty()) return true;
        if (late && (NH != 1 || copy_mode || ram_held != 0 || !ram_chunks.empty() || !ram_part.empty() || store)) {
            err = "late RAM copy requires an unused NH1 mapped CPU lane and no RAM store";
            return false;
        }
        if (late && cancel && cancel()) {
            if (interrupted_out) *interrupted_out = true;
            return true;
        }
        const double t0 = now_ms();
        auto spare_of = [&](const Layer& l) { return o.expert_cache && l.spare_vram >= 0 ? 1 : 0; };
        double all = 0;
        {
            std::vector<char> seen(L.size(), 0);
            for (const auto& h : hot) {
                const Layer& l = L[(size_t) h.first];
                all += (double) l.blob * (1 + (seen[(size_t) l.il]++ == 0 ? spare_of(l) : 0));
            }
        }
        const uint64_t avail = ram_avail();
        // (Windows: the copies are committed memory, within what its commit still has beside the other programs' margin)
        const double can = std::min(avail == ~uint64_t{0} ? 1e300 : std::max(0.0, (double) avail - (double) kRamMargin),
                                    commit_room());
        // (auto: none - copies that all fit were measured no faster than the page cache: the CPU lane's rows are
        // compute-bound, ~46 GB/s from either on 2 Xeon sockets, NUMA parts included; a part copied cost decode speed
        // on 64 GB and on 32 GB of RAM)
        double budget = late || o.mmap_experts == 0 ? std::min(all, can) : 0.0;
        if (!late)
            if (const char* v = std::getenv("DS4_COPY_GB")) budget = std::atof(v) * 1e9;
        if ((late || o.mmap_experts == 0) && all > can)
            std::fprintf(stderr, "ds4: --mmap-experts off: the CPU lane's experts (%.1f GB) do not all fit the RAM this "
                                 "process can get (%.1f GB beside %.0f GB for the rest) - the most-routed copied, the "
                                 "others read in place\n", all / 1e9, can / 1e9, kRamMargin / 1e9);
        if (budget <= 0) {
            if (o.verbose)
                std::fprintf(stderr, "ds4: the CPU lane's experts (%.1f GB) stay in place: the RAM this process can get "
                                     "(%.1f GB beside %.0f GB for the rest) does not hold them all\n",
                             all / 1e9, can / 1e9, kRamMargin / 1e9);
            return true;
        }
        // NUMA parts: whole rows of each matrix, so any row count the nodes divide
        int nh = late ? 1 : (int) std::min<size_t>(2, std::max<size_t>(1, cores.size()));
        if (!late)
            if (const char* v = std::getenv("DS4_COPY_NUMA"))   // (0 / 1: one part; 2: two, even on one node - a test)
                nh = std::atoi(v) >= 2 ? 2 : 1;
        if (g.ff % nh != 0 || g.dim % nh != 0) nh = 1;
        NH = nh;
        if(record_copied_ram_domains && NH!=1) {err="RAM-domain recording requires NH1";return false;}
        cp_lay.assign(L.size(), {0, 0, 0, 0});
        for (const Layer& l : L) {
            size_t off = 0;
            for (int r = 0; r < 3; ++r) {
                cp_lay[(size_t) l.il][(size_t) r] = off;
                off += align256((size_t) (mx_n(r) / NH) * mx_row_b(l, r));
            }
            cp_lay[(size_t) l.il][3] = off;
        }
        // the hottest that fit the budget (a layer's first one brings its spare)
        std::vector<int> n_copy(L.size(), 0);
        double used = 0;
        size_t take = 0;
        for (; take < hot.size(); ++take) {
            const Layer& l = L[(size_t) hot[take].first];
            const double add = (double) l.blob * (1 + (n_copy[(size_t) l.il] == 0 ? spare_of(l) : 0));
            if (used + add > budget) break;
            used += add;
            ++n_copy[(size_t) l.il];
        }
        // each layer's slots: one block per node of (copies + spare) parts
        std::vector<int> first(L.size(), -1);
        copy_spare.assign(L.size(), -1);
        for (Layer& l : L) {
            l.ram.assign((size_t) l.n_exp, -1);
            const int ns = n_copy[(size_t) l.il] + (n_copy[(size_t) l.il] > 0 ? spare_of(l) : 0);
            if (ns == 0) continue;
            const size_t part = cp_lay[(size_t) l.il][3], bytes = (size_t) ns * part;
            std::array<uint8_t*, 2> base = {nullptr, nullptr};
            bool ok = true;
            for (int X = 0; X < NH && ok; ++X) {
                base[(size_t) X] = (uint8_t*) node_map(bytes, X, NH);
                ok = base[(size_t) X] != nullptr;
                if (ok) {
                    ram_chunks.push_back(base[(size_t) X]);
                    ram_chunk_bytes.push_back(bytes);
                }
            }
            if (!ok) {
                std::fprintf(stderr, "ds4: no RAM for layer %d's %d copied experts (%.1f GB): they stay in place\n", l.il,
                             ns, (double) bytes * NH / 1e9);
                n_copy[(size_t) l.il] = 0;
                continue;
            }
            first[(size_t) l.il] = (int) ram_part.size();
            for (int i = 0; i < ns; ++i)
                ram_part.push_back({base[0] + (size_t) i * part, NH > 1 ? base[1] + (size_t) i * part : nullptr});
            if (spare_of(l)) copy_spare[(size_t) l.il] = first[(size_t) l.il] + ns - 1;
            if(record_copied_ram_domains) {
                ramdomain::Domain domain;
                domain.layer=l.il;domain.expert_count=l.n_exp;
                domain.first_slot=(size_t)first[(size_t)l.il];domain.slot_count=(size_t)ns;
                domain.part_stride=part;domain.arena_index=ram_chunks.size()-1;
                domain.base=(uintptr_t)base[0];domain.arena_bytes=bytes;domain.offsets=cp_lay[(size_t)l.il];
                domain.has_spare=spare_of(l)!=0;
                for(int r=0;r<3;++r)domain.role_bytes[(size_t)r]=mx_part_bytes(l,r);
                pending_ram_domains.push_back(domain); // capacity reserved before any new RAM allocation
            }
        }
        struct Job {
            int layer, e, slot;
        };
        std::vector<Job> jobs;
        {
            std::vector<int> next = first;
            for (size_t i = 0; i < take; ++i) {
                const int il = hot[i].first;
                if (n_copy[(size_t) il] == 0) continue;
                jobs.push_back({il, hot[i].second, next[(size_t) il]++});
            }
        }
        std::sort(jobs.begin(), jobs.end(), [&](const Job& a, const Job& b) {   // (file order: the disk reads ahead)
            const Layer &la = L[(size_t) a.layer], &lb = L[(size_t) b.layer];
            if (la.mx_file[0] != lb.mx_file[0]) return la.mx_file[0] < lb.mx_file[0];
            return la.mx_off[0] + (uint64_t) a.e * la.mx_stride[0] < lb.mx_off[0] + (uint64_t) b.e * lb.mx_stride[0];
        });
        std::atomic<size_t> ji{0};
        std::atomic<bool> bad{false};
        std::atomic<bool> interrupted{false};
        std::atomic<int> active{0};
        std::atomic<int64_t> done_b{0};
        std::vector<std::thread> th;
        for (int t = 0; t < 12 && !jobs.empty(); ++t) {
            if (late) active.fetch_add(1, std::memory_order_relaxed);
            th.emplace_back([&] {
                for (size_t i; (i = ji.fetch_add(1)) < jobs.size() && !bad.load();) {
                    if (late && interrupted.load(std::memory_order_acquire)) break;
                    const Job& j = jobs[i];
                    const Layer& l = L[(size_t) j.layer];
                    for (int r = 0; r < 3; ++r) {
                        const size_t nb = (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
                        for (int X = 0; X < NH; ++X)
                            if (!pk.read_at(l.mx_file[r], l.mx_off[r] + (uint64_t) j.e * l.mx_stride[r] + (uint64_t) X * nb, nb,
                                            ram_part[(size_t) j.slot][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r]))
                                bad.store(true);
                    }
                    const int64_t db = done_b.fetch_add((int64_t) l.blob) + (int64_t) l.blob;
                    if (o.verbose && i % 2048 == 0)
                        std::fprintf(stderr, "ds4: copying experts %zu / %zu (%.1f GB, %.2f GB/s)\n", i, jobs.size(), db / 1e9,
                                     db / 1e6 / std::max(1.0, now_ms() - t0));
                }
                if (late) active.fetch_sub(1, std::memory_order_release);
            });
        }
        if (late) {   // poll the callback ONLY on the model's caller thread, never concurrently on readers
            while (active.load(std::memory_order_acquire) > 0) {
                if (cancel && cancel()) interrupted.store(true, std::memory_order_release);
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            if (cancel && cancel()) interrupted.store(true, std::memory_order_release);
        }
        for (auto& t : th) t.join();
        if (bad.load()) {
            err = "copying the CPU lane's experts from the model's files failed";
            return false;
        }
        if (late && interrupted.load(std::memory_order_acquire)) {
            // No l.ram entry or copy_mode was published. Every read joined; the optional fresh arenas can go away
            // without touching file maps, the normal GPU pool, or the already-created NH1 worker pools.
            for (size_t i = 0; i < ram_chunks.size(); ++i) os::anon_free(ram_chunks[i], ram_chunk_bytes[i]);
            ram_chunks.clear();
            ram_chunk_bytes.clear();
            ram_part.clear();
            cp_lay.clear();
            copy_spare.clear();
            for (Layer& l : L) l.ram.assign((size_t) l.n_exp, -1);
            if (interrupted_out) *interrupted_out = true;
            return true;
        }
        for (const Job& j : jobs) L[(size_t) j.layer].ram[(size_t) j.e] = j.slot;
        copy_mode = true;
        ram_held = (int) jobs.size();
        if(record_copied_ram_domains) {
            ramdomain::Inventory initial;
            if(!reconstruct_copied_ram_inventory(pending_ram_domains,initial,err))return false;
            copied_ram_domains.swap(pending_ram_domains);
            copied_ram_domains_published=true;
            std::fprintf(stderr,"ds4: experimental copied-RAM domains recorded: %zu layers/%zu slots, "
                                "%zu occupied/%zu spares/%zu free holes; free bytes untrusted, "
                                "no rearm or expert move performed\n",
                         copied_ram_domains.size(),ram_part.size(),initial.cpu_experts,
                         initial.reserved_spares,initial.free_holes);
        }
        double cb = 0;
        for (const Job& j : jobs) cb += (double) L[(size_t) j.layer].blob;
        std::fprintf(stderr, "ds4: %zu of the CPU lane's %zu experts copied into RAM (%.1f GB, %d NUMA part%s) in %.1f s%s\n",
                     jobs.size(), hot.size(), cb / 1e9, NH, NH > 1 ? "s" : "", (now_ms() - t0) / 1000.0,
                     jobs.size() < hot.size() ? "; the others read in place" : "");
        return true;
    }

    bool load_experts(std::string& err) {
        if (!L.empty() && L[0].lay.fmt == gg::kFmtMXFP4) return load_experts_mx(err);
        // every expert's region, in file order, by a few threads; VRAM ones through a pinned staging blob.  In place
        // (file_experts) the CPU lane's experts are not copied: the most-routed of them (the profile's share of their
        // layer) are read once, up to warm_bytes, so the page cache starts with them, and the VRAM ones' pages are
        // dropped from it again
        struct Job {
            int layer, e;
        };
        std::vector<Job> jobs;
        for (const Layer& l : L)
            for (int e = 0; e < l.n_exp; ++e)
                if (!file_experts || l.slot[(size_t) e] >= 0) jobs.push_back({l.il, e});
        uint64_t warm = 0;
        if (file_experts) {
            std::vector<std::pair<double, Job>> cand;
            for (const Layer& l : L) {
                const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
                double sum = 0;
                for (int e = 0; e < l.n_exp; ++e) sum += p[e];
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) cand.push_back({sum > 0 ? p[e] / sum : 0.0, Job{l.il, e}});
            }
            std::stable_sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            for (const auto& c : cand) {
                const uint64_t b = L[(size_t) c.second.layer].region_bytes[(size_t) c.second.e];
                if (warm + b > warm_bytes) break;
                warm += b;
                jobs.push_back(c.second);
            }
        }
        std::sort(jobs.begin(), jobs.end(), [&](const Job& a, const Job& b) {
            const Layer &la = L[(size_t) a.layer], &lb = L[(size_t) b.layer];
            if (la.region_file[(size_t) a.e] != lb.region_file[(size_t) b.e])
                return la.region_file[(size_t) a.e] < lb.region_file[(size_t) b.e];
            return la.region_off[(size_t) a.e] < lb.region_off[(size_t) b.e];
        });
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::mutex emu;
        std::string ferr;
        const int nth = 12;
        size_t maxr = 0, maxb = 0;
        for (const Layer& l : L) {
            maxb = std::max(maxb, l.blob);
            for (uint64_t r : l.region_bytes) maxr = std::max<size_t>(maxr, r);
        }
        std::atomic<int64_t> done_bytes{0};
        const double t0 = now_ms();
        std::vector<std::thread> th;
        for (int t = 0; t < nth; ++t)
            th.emplace_back([&] {
                std::vector<uint8_t> region(maxr);
                uint8_t *stage = nullptr, *stage2 = nullptr;
                cudaHostAlloc((void**) &stage, maxb, cudaHostAllocPortable);
                cudaHostAlloc((void**) &stage2, maxb, cudaHostAllocPortable);
                int cur_dev = -1;
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= jobs.size() || failed.load()) break;
                    const Layer& l = L[(size_t) jobs[i].layer];
                    const int e = jobs[i].e;
                    if (!pk.read_at(l.region_file[(size_t) e], l.region_off[(size_t) e], l.region_bytes[(size_t) e],
                                    region.data())) {
                        std::lock_guard<std::mutex> lk(emu);
                        ferr = "reading layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
                        failed.store(true);
                        break;
                    }
                    if (l.slot[(size_t) e] >= 0) {
                        repack(l, region.data(), stage);
                        bool up = true;
                        for (int m = 0; m < l.nm && up; ++m) {   // (a group: each member its part)
                            const Layer& x = member(l, m);
                            const Dev& d = devs[(size_t) x.dev];
                            if (cur_dev != d.id) {
                                cudaSetDevice(d.id);
                                cur_dev = d.id;
                            }
                            if (l.nm > 1) blob_to_half(l, stage, m, stage2);
                            up = cudaMemcpy(x.vram_base + (size_t) l.slot[(size_t) e] * unit(x), l.nm > 1 ? stage2 : stage,
                                            unit(x), cudaMemcpyHostToDevice) == cudaSuccess;
                        }
                        if (!up) {
                            std::lock_guard<std::mutex> lk(emu);
                            ferr = "uploading layer " + std::to_string(l.il) + " expert " + std::to_string(e);
                            failed.store(true);
                            break;
                        }
                        if (file_experts)
                            os::advise(pk.fds[(size_t) l.region_file[(size_t) e]], l.region_off[(size_t) e], l.region_bytes[(size_t) e], false);
                    } else if (!file_experts) {
                        for (int X = 0; X < NH; ++X) repack_part(l, region.data(), X, l.hh[(size_t) e][(size_t) X]);
                    }
                    const int64_t db = done_bytes.fetch_add((int64_t) l.region_bytes[(size_t) e]) +
                                       (int64_t) l.region_bytes[(size_t) e];
                    if (o.verbose && (i % 1024) == 0)
                        std::fprintf(stderr, "ds4: experts %zu / %zu (%.1f GB, %.2f GB/s)\n", i, jobs.size(),
                                     db / 1e9, db / 1e6 / std::max(1.0, now_ms() - t0));
                }
                if (stage) cudaFreeHost(stage);
                if (stage2) cudaFreeHost(stage2);
            });
        for (auto& t : th) t.join();
        if (failed.load()) {
            err = ferr;
            return false;
        }
        if (file_experts)
            std::fprintf(stderr, "ds4: experts loaded in %.1f s (%.1f GB of the CPU lane's read ahead)\n",
                         (now_ms() - t0) / 1000.0, warm / 1e9);
        return true;
    }

    std::string describe() const {
        std::ostringstream s;
        for (size_t di = 0; di < devs.size(); ++di) {
            const Dev& d = devs[di];
            int a = -1, b = -1, ev = 0, et = 0;
            for (const Layer& l : L)
                for (int m = 0; m < l.nm; ++m)
                    if (member(l, m).dev == (int) di) {
                        if (a < 0) a = l.il;
                        b = l.il;
                        ev += l.n_vram;
                        et += l.n_exp;
                    }
            size_t fr = 0, tot = 0;
            cudaSetDevice(d.id);
            cudaMemGetInfo(&fr, &tot);
            const size_t gsz = groups.empty() ? 1 : groups[(size_t) d.grp].mem.size();
            char tp[64] = "";
            if (gsz > 1) std::snprintf(tp, sizeof tp, " (tensor parallel %zu, part %d)", gsz, d.mi);
            char line[384];
            std::snprintf(line, sizeof line,
                          "ds4: CUDA%d layers [%d, %d]%s%s: weights %.2f GB, experts %.2f GB (%d of %d in VRAM), caches "
                          "%.2f GB, %.0f MiB free\n",
                          d.id, a, b, tp, (int) di == head_dev ? " + head" : "", d.weights / 1e9, d.experts / 1e9, ev, et,
                          d.caches / 1e9, fr / 1048576.0);
            s << line;
        }
        int nh = 0;
        size_t hb_ = 0;
        for (const Layer& l : L) {
            nh += l.n_exp - l.n_vram;
            hb_ += (size_t) (l.n_exp - l.n_vram) * l.blob;
        }
        char line[200];
        if (copy_mode)
            std::snprintf(line, sizeof line, "ds4: %d routed experts on the CPU lane (%.1f GB): %d copied into RAM (%d NUMA "
                                             "parts), the rest in place in the model's files; %d threads\n",
                          nh, hb_ / 1e9, ram_held, NH, cpu_threads());
        else if (file_experts)
            std::snprintf(line, sizeof line,
                          "ds4: %d routed experts on the CPU lane (%.1f GB, in place in the model's files), %d threads\n",
                          nh, hb_ / 1e9, cpu_threads());
        else
            std::snprintf(line, sizeof line, "ds4: %d routed experts on the CPU lane (%.1f GB of RAM, %d NUMA parts), %d threads\n", nh,
                          hb_ / 1e9, NH, cpu_threads());
        s << line;
        return s.str();
    }

    double store_gb() const {
        static double cached = -2.0;   // (decided once: the layout of the blobs depends on it)
        if (cached > -1.5) return cached;
        double gb = o.ssd_cache_gb;
        if (const char* v = std::getenv("DS4_STORE_GB")) gb = std::atof(v);
        if (!pk.gguf) gb = 0.0;
        else if (gb < 0.0) {   // auto: the RAM free now less 12 GB for the system and the owner's programs, 48 GB at most
            const uint64_t avail = ram_avail();
            const double room = avail == ~uint64_t{0} ? 0.0 : (double) avail / 1e9 - 12.0;
            gb = room >= 4.0 ? std::min(48.0, room) : 0.0;
        }
        return cached = gb;
    }
    void init_store() {
        const double gb = store_gb();
        if (gb <= 0.0 || L.empty() || L[0].lay.fmt != gg::kFmtMXFP4 || !file_experts) return;
        std::vector<ExpertStore::LayerGeo> geo((size_t) g.nl);   // (the trunk's layers: the drafter's bigger experts stay in place)
        for (const Layer& l : L) {
            if (l.il >= g.nl) continue;
            ExpertStore::LayerGeo& gl = geo[(size_t) l.il];
            gl.n_exp = l.n_exp;
            for (int r = 0; r < 3; ++r) {
                gl.file[r] = l.mx_file[r];
                gl.off[r] = l.mx_off[r];
                gl.stride[r] = l.mx_stride[r];
                gl.bytes[r] = mx_part_bytes(l, r);
                gl.tr[r] = l.lay.trellis[r];
            }
            gl.blob = l.blob;
        }
        ExpertStore::Cfg cfg;
        cfg.budget_bytes = (size_t) (gb * 1e9);
        cfg.threads = std::getenv("DS4_STORE_THREADS") ? std::atoi(std::getenv("DS4_STORE_THREADS")) : 8;
        cfg.floor_bytes = (size_t) (std::getenv("DS4_STORE_FLOOR_GB") ? std::atof(std::getenv("DS4_STORE_FLOOR_GB")) : 8.0) << 30;
        cfg.policy = (ExpertStore::Policy) (std::getenv("DS4_STORE_POLICY") ? std::atoi(std::getenv("DS4_STORE_POLICY")) : o.ssd_policy);
        cfg.ram_avail = [] { return ram_avail(); };
        cfg.reserve = (size_t) (std::getenv("DS4_STORE_RESERVE") ? std::atoi(std::getenv("DS4_STORE_RESERVE")) : 192);
        cfg.vis_percent = std::getenv("DS4_STORE_ZC_FRAC") ? std::atoi(std::getenv("DS4_STORE_ZC_FRAC")) : 50;
        // (the GPU computing the store's experts over PCIe is opt-in, DS4_STORE_ZC=1: measured slower than the CPU lane alone on a
        // 5070 / PCIe 5.0 - the staging copy sits in the layer's critical path - and its answers depend on the timing of which lane
        // took which expert, the CPU lane's are the mmap path's byte for byte)
        const bool want_zc = std::getenv("DS4_STORE_ZC") && std::atoi(std::getenv("DS4_STORE_ZC")) != 0 && devs.size() == 1 &&
                             L[0].nm == 1;
        if (want_zc) {   // the arena's chunks registered for the GPU to read in place (zero-copy over PCIe)
            const int dev0 = devs[0].id;
            cfg.map_chunk = [dev0](void* h, size_t n) -> void* {
                cudaSetDevice(dev0);
                if (cudaHostRegister(h, n, cudaHostRegisterMapped | cudaHostRegisterPortable) != cudaSuccess) {
                    (void) cudaGetLastError();
                    return nullptr;
                }
                void* d = nullptr;
                if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) {
                    (void) cudaGetLastError();
                    cudaHostUnregister(h);
                    return nullptr;
                }
                if (d != h) std::fprintf(stderr, "ds4: RAM store: chunk host %p -> device %p (not the identity)\n", h, d);
                return d;
            };
            cfg.unmap_chunk = [dev0](void* h) {
                cudaSetDevice(dev0);
                cudaHostUnregister(h);
            };
        }
        bool direct = true;
        for (os::fd_t f : pk.dfds) direct = direct && f != os::kBadFd;
        auto st = std::make_unique<ExpertStore>();
        if (!st->init(cfg, geo, pk.dfds, pk.pfds)) {
            std::fprintf(stderr, "ds4: --ssd-cache: the store could not start\n");
            return;
        }
        store = std::move(st);
        // (the router lookahead's guesses are not read ahead with the store: measured 4.5 against 4.7 tok/s, 62% of the guesses are right and
        // the SSD is the narrow part; DS4_AHEAD=1 turns them on)
        if (std::getenv("DS4_AHEAD") == nullptr) ahead_mode = 0;
        for (Layer& l : L)
            if (l.il < g.nl && want_zc) l.zc.assign((size_t) l.n_exp, 0);
        std::fprintf(stderr, "ds4: RAM store: up to %zu slots of %.1f MB (%.1f GB), policy %d, %d reader threads, %s reads\n",
                     store->slots_max(), store->slot_bytes() / 1e6, store->slots_max() * (double) store->slot_bytes() / 1e9,
                     (int) cfg.policy, cfg.threads, direct ? "unbuffered" : "BUFFERED (no direct handle)");
        if (want_zc) {
            if (!store->gpu_mapped()) {
                std::fprintf(stderr, "ds4: RAM store: the arena could not be mapped for the GPU: zero-copy off\n");
                for (Layer& l : L) l.zc.clear();
            } else {
                zc_stride = (store->max_blob() + 4095) & ~(size_t) 4095;
                cudaSetDevice(devs[0].id);
                if (cudaMalloc((void**) &zc_stage_buf, zc_stride * kPicks) != cudaSuccess) {
                    (void) cudaGetLastError();
                    zc_stage_buf = nullptr;
                    for (Layer& l : L) l.zc.clear();
                } else {
                    for (Layer& l : L)
                        if (!l.zc.empty() && cudaMalloc((void**) &l.zc_tab, (size_t) l.n_exp * 8) == cudaSuccess) cudaMemset(l.zc_tab, 0, (size_t) l.n_exp * 8);
                        else if (!l.zc.empty()) {
                            (void) cudaGetLastError();
                            l.zc.clear();
                        }
                }
            }
            std::fprintf(stderr, "ds4: RAM store: the GPU takes its slots over PCIe (%s), %zu slots kept evictable\n",
                         zc_stage_buf != nullptr ? "copied into a staging area at the PCIe rate first" : "off", cfg.reserve);
        }
    }
    // the profile's hottest experts (VRAM's left out) read into the store before the first token (DS4_STORE_PRELOAD=0: not)
    void preload_store() {
        if (!store) return;
        if (std::getenv("DS4_STORE_PRELOAD") && std::atoi(std::getenv("DS4_STORE_PRELOAD")) == 0) return;
        std::vector<std::pair<double, std::pair<int, int>>> cand;
        double tot = 0;
        for (const Layer& l : L) {
            if (l.il >= g.nl) continue;
            const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
            for (int e = 0; e < l.n_exp; ++e) tot += p[e];
        }
        if (tot <= 0) return;   // (no profile yet: the first tokens fill it)
        for (const Layer& l : L) {
            if (l.il >= g.nl) continue;
            const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
            for (int e = 0; e < l.n_exp; ++e)
                if (l.slot[(size_t) e] < 0 && p[e] > 0) cand.push_back({(double) p[e] / (double) l.blob, {l.il, e}});
        }
        std::stable_sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<std::pair<int, int>> list;
        list.reserve(cand.size());
        for (const auto& c : cand) list.push_back(c.second);
        const double t0 = now_ms();
        const size_t leave = (size_t) (std::getenv("DS4_STORE_RESERVE") ? std::atoi(std::getenv("DS4_STORE_RESERVE")) : 192) + 64;
        const size_t n = store->preload(list, leave);
        std::fprintf(stderr, "ds4: RAM store: %zu of the profile's hottest experts read in (%.1f GB) in %.1f s\n", n,
                     (double) n * (double) L[0].blob / 1e9, (now_ms() - t0) / 1000.0);
    }
    // at a window boundary: the store's ready slots become the device's (zero-copy) experts, taken back when it needs the room;
    // each touched layer's whole table is written again (VRAM slots, the store's, 0), in stream order before the next window
    void store_sync() {
        if (!store || !store->gpu_mapped()) return;
        std::vector<ExpertStore::Vis> ev;
        store->sync(ev, [&](int layer, int e) { return L[(size_t) layer].slot[(size_t) e] < 0; });
        if (ev.empty()) return;
        std::vector<char> dirty(L.size(), 0);
        for (const auto& v : ev) {
            Layer& l = L[(size_t) v.layer];
            if (l.zc.empty()) continue;
            l.zc[(size_t) v.e] = v.dev != 0 ? 1 : 0;
            dirty[(size_t) v.layer] = 1;
        }
        for (Layer& l : L) {
            if (!dirty[(size_t) l.il]) continue;
            if (l.zc_tab == nullptr) continue;
            std::vector<unsigned long long> tab((size_t) l.n_exp, 0ull);
            for (int e = 0; e < l.n_exp; ++e)
                if (l.zc[(size_t) e]) {
                    const int s = l.slot[(size_t) e] < 0 ? store->find(l.il, e) : -1;
                    tab[(size_t) e] = s >= 0 ? store->dev_blob(s) : 0ull;
                    if (s < 0) l.zc[(size_t) e] = 0;
                }
            cudaSetDevice(devs[0].id);
            cudaMemcpyAsync(l.zc_tab, tab.data(), tab.size() * 8, cudaMemcpyHostToDevice, devs[0].st);
        }
    }

    int cpu_threads() const {
        int n = 0;
        for (const auto& pl : pools) n += pl->size() - 1;
        return n + 1;
    }
    // the CPU lane's workers: a pool per NUMA node (NH of them) on its cores; in place with one part, one pool over
    // every node's cores (the pages lie where the cache put them)
    void make_pools() {
        std::vector<std::vector<int>> cs = cores;
        if ((int) cs.size() < NH && !cs.empty()) {   // (DS4_COPY_NUMA=2 on one node, a test: its cores split in NH)
            std::vector<int> all = cs[0];
            cs.assign((size_t) NH, {});
            for (size_t i = 0; i < all.size(); ++i) cs[i * (size_t) NH / all.size()].push_back(all[i]);
        }
        static const int lane_free = [] {
            const char* v = std::getenv("STRATA_DS4_LANE_FREE");
            return v != nullptr ? std::max(0, std::atoi(v)) : 2;
        }();
        size_t lane_cpus = 0;
        for (const auto& v : cs) lane_cpus += v.size();
        const bool lane_takes_all = lane_cpus >= (size_t) std::max(1u, std::thread::hardware_concurrency());
        for (int X = 0; X < (int) cs.size() && X < std::max(NH, 1); ++X) {
            std::vector<int> c = cs[(size_t) X];
            if (file_experts && NH == 1)
                for (size_t Y = 1; Y < cs.size(); ++Y) c.insert(c.end(), cs[Y].begin(), cs[Y].end());
            if (X == 0 && !c.empty()) c.erase(c.begin());   // the host thread's core (it helps node 0's pool)
            // cores of the first pool left free beside the host thread's, for the thread that launches the decode's kernels
            // (and waits on the device) and whatever else the PC runs: with a worker spinning on every other logical CPU
            // that thread waits for a core (Project Maya #359, a 20-core Windows 11 PC: decode 28.3 -> 55.5 tok/s).  Only
            // when the lane would take every logical CPU and --cpu-threads is not given.
            // STRATA_DS4_LANE_FREE: the cores left free (2 by default, 0: as before)
            if (X == 0 && o.cpu_threads <= 0 && lane_free > 0 && (int) c.size() > lane_free + 1 && lane_takes_all)
                c.resize(c.size() - (size_t) lane_free);
            int nw = (int) c.size();
            if (o.cpu_threads > 0) nw = std::max(X == 0 ? 0 : 1, std::min(nw, o.cpu_threads / NH - (X == 0 ? 1 : 0)));
            pools.push_back(std::make_unique<Pool>(nw, c, 20000));   // spin through a token's layers (20 ms)
        }
        (void) gg::mx_tier();   // (measured here, not on the first token)
        if (o.verbose) {
            int nthr = 0;
            for (auto& p : pools) nthr += p->size();
            std::fprintf(stderr, "ds4: CPU lane: %d threads, MXFP4 rows %s (%s)\n", nthr, gg::mx_tier_name(gg::mx_tier()),
                         gg::mx_tier_report());
        }
    }

    void reset() {
        pos = 0;
        toks.clear();
        ctoks.clear();
        slot_held = false;
        for (Check& c0 : ck) c0.pos = -1;
        for (Layer& l0 : L)
            for (int m = 0; m < l0.nm; ++m) {
                Layer& l = member(l0, m);
                Dev& d = devs[(size_t) l.dev];
                cudaSetDevice(d.id);
                cudaMemsetAsync(l.win, 0, (size_t) g.win * win_rb(), d.st);
                if (l.c_kv) cudaMemsetAsync(l.c_kv, 0, (size_t) l.ratio * g.hd * 4, d.st);
                if (l.c_score) cudaMemsetAsync(l.c_score, 0, (size_t) l.ratio * g.hd * 4, d.st);
            }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
    }

    // the state at pos into checkpoint `slot` (the window rings and the compressors' partial groups: the rest of the
    // state is indexed by position, and what lies past a restored position is written again)
    void checkpoint(int slot) {
        Check& c = ck[slot & (kChecks - 1)];
        c.pos = pos;
        c.win.resize(L.size());
        c.comp.resize(L.size() * 2);
        for (size_t i = 0; i < L.size(); ++i) {
            Layer& l = L[i];
            Dev& d = devs[(size_t) l.dev];
            cudaSetDevice(d.id);
            c.win[i].resize((size_t) g.win * win_rb());
            cudaMemcpyAsync(c.win[i].data(), l.win, c.win[i].size(), cudaMemcpyDeviceToHost, d.st);
            if (l.c_kv) {
                c.comp[2 * i].resize((size_t) l.ratio * g.hd);
                c.comp[2 * i + 1].resize((size_t) l.ratio * g.hd);
                cudaMemcpyAsync(c.comp[2 * i].data(), l.c_kv, c.comp[2 * i].size() * 4, cudaMemcpyDeviceToHost, d.st);
                cudaMemcpyAsync(c.comp[2 * i + 1].data(), l.c_score, c.comp[2 * i + 1].size() * 4,
                                cudaMemcpyDeviceToHost, d.st);
            }
        }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
    }

    // the checkpoint at or before position limit (the state's tokens agree with a new prompt up to there), or -1
    int best_check(int64_t limit) const {
        int b = -1;
        for (int i = 0; i < kChecks; ++i)
            if (ck[i].pos >= 0 && ck[i].pos <= limit && ck[i].pos <= pos && (b < 0 || ck[i].pos > ck[b].pos)) b = i;
        return b;
    }
    int64_t restorable(int64_t limit) const {
        const int b = best_check(limit);
        return b < 0 ? -1 : ck[b].pos;
    }

    // checkpoint 1 (the position before the last prompt's end) becomes the anchor too
    void pin() {
        if (ck[1].pos > 0) ck[2] = ck[1];
    }
    // the checkpoints the state can go back to (what a slot of it keeps), latest first
    std::vector<int64_t> check_positions() const {
        std::vector<int64_t> v;
        for (const Check& c : ck)
            if (c.pos > 0 && c.pos <= pos && c.win.size() == L.size() &&
                std::find(v.begin(), v.end(), c.pos) == v.end())
                v.push_back(c.pos);
        std::sort(v.rbegin(), v.rend());
        return v;
    }
    // the checkpoints a slot keeps (each with its role: 0, 1 or the anchor)
    std::vector<int> slot_checks() const {
        std::vector<int> v;
        for (int i = 0; i < kChecks; ++i)
            if (ck[i].pos > 0 && ck[i].pos <= pos && ck[i].win.size() == L.size()) v.push_back(i);
        return v;
    }
    size_t check_bytes() const {
        size_t n = 16;
        for (const Layer& l : L) n += (size_t) g.win * win_rb() + (l.c_kv ? 2 * (size_t) l.ratio * g.hd * 4 : 0);
        return n;
    }

    bool restore(int64_t limit) {
        const int b = best_check(limit);
        if (b < 0) return false;
        const Check& c = ck[b];
        for (size_t i = 0; i < L.size(); ++i)
            for (int m = 0; m < L[i].nm; ++m) {   // (a group's members hold the same state: the lead's copy)
                Layer& l = member(L[i], m);
                Dev& d = devs[(size_t) l.dev];
                cudaSetDevice(d.id);
                cudaMemcpyAsync(l.win, c.win[i].data(), c.win[i].size(), cudaMemcpyHostToDevice, d.st);
                if (l.c_kv) {
                    cudaMemcpyAsync(l.c_kv, c.comp[2 * i].data(), c.comp[2 * i].size() * 4, cudaMemcpyHostToDevice, d.st);
                    cudaMemcpyAsync(l.c_score, c.comp[2 * i + 1].data(), c.comp[2 * i + 1].size() * 4,
                                    cudaMemcpyHostToDevice, d.st);
                }
            }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
        pos = c.pos;
        toks.resize((size_t) pos);
        ctoks.resize((size_t) pos);
        slot_held = false;
        for (Check& o2 : ck)   // a later checkpoint saw tokens the state no longer has
            if (o2.pos > pos) o2.pos = -1;
        return true;
    }

    // ---- the kept conversations (a slot): the state out of the model and back, through a file or a host buffer.
    // A prompt that continues the state is reused in place (checkpoint / restore, the compressed KV left where it is);
    // a conversation that leaves the model for another one is a slot - the whole state: the position, its tokens,
    // every window ring (the DSpark blocks' included), the compressors' partial groups and each KV source's compressed
    // KV and index keys up to it (rows 0 .. pos / ratio, the groups its position completed; one the prompt has not
    // completed yet lives in the compressor's partial group).  A tensor parallel group's members hold the same state,
    // so one member's ring is saved and written back into all of them.  Text only: a conversation that holds pictures
    // is not a slot (the Engram history's rows are the tokens' again, no -1 among them).
    static constexpr uint64_t kSlotMagic = 0x31544f4c53345344ull;   // "DS4SLOT1" (little endian)

    int64_t src_rows(const Src& s) const { return pos / std::max(1, s.ratio); }

    // the slot's bytes: the header, the tokens, every layer's ring and partial group, every source's rows, the end
    size_t slot_bytes() const {
        size_t n = 8 + 8 + slot_identity().size() + 8 + (size_t) pos * 4;
        for (const Layer& l : L) n += 8 + (size_t) g.win * win_rb() + 8 + (l.c_kv ? 2 * (size_t) l.ratio * g.hd * 4 : 0);
        for (const Src& s : srcs) n += 8 + (size_t) src_rows(s) * (ckv_rb() + ik_rb());
        n += 8 + slot_checks().size() * check_bytes();   // (the checkpoints: what the slot can be taken back at)
        return n + 8;
    }

    uint64_t slot_save(std::vector<uint8_t>& out, std::string& err) {
        if (slot_held) {
            err = "it is already set aside";
            return 0;
        }
        if (pos <= 0) {
            err = "there is no conversation";
            return 0;
        }
        if ((int64_t) toks.size() != pos) {
            err = "the state's tokens do not match its position";
            return 0;
        }
        if (!imgs.empty()) {
            err = "it holds pictures";
            return 0;
        }
        const std::string id = slot_identity();
        out.assign(slot_bytes(), 0);
        uint8_t* p = out.data();
        size_t off = 0;
        const auto put = [&](const void* s, size_t b) {
            std::memcpy(p + off, s, b);
            off += b;
        };
        const uint64_t il = id.size(), n = (uint64_t) pos;
        put(&kSlotMagic, 8);
        put(&il, 8);
        put(id.data(), (size_t) il);
        put(&n, 8);
        put(toks.data(), (size_t) n * 4);
        for (Dev& d : devs) {   // (the devices' work ends before their state is read)
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
        for (size_t i = 0; i < L.size(); ++i) {
            const Layer& l = L[i];
            const Dev& d = devs[(size_t) l.dev];
            cudaSetDevice(d.id);
            const uint64_t wb = (uint64_t) g.win * win_rb();
            put(&wb, 8);
            if (cudaMemcpy(p + off, l.win, (size_t) wb, cudaMemcpyDeviceToHost) != cudaSuccess) {
                err = l.name + "window ring: " + cudaGetErrorString(cudaGetLastError());
                return 0;
            }
            off += (size_t) wb;
            const uint64_t cn = l.c_kv ? (uint64_t) l.ratio * (uint64_t) g.hd : 0;
            put(&cn, 8);
            if (cn > 0) {
                const size_t cb = (size_t) cn * 4;
                if (cudaMemcpy(p + off, l.c_kv, cb, cudaMemcpyDeviceToHost) != cudaSuccess ||
                    cudaMemcpy(p + off + cb, l.c_score, cb, cudaMemcpyDeviceToHost) != cudaSuccess) {
                    err = l.name + "compressor's group: " + cudaGetErrorString(cudaGetLastError());
                    return 0;
                }
                off += 2 * cb;
            }
        }
        for (const Src& s : srcs) {
            const uint64_t rows = (uint64_t) src_rows(s);
            put(&rows, 8);
            if (rows == 0) continue;
            const int di = L[(size_t) s.layer].dev;
            const uint8_t* ckv = s.ckv[(size_t) di];
            const uint8_t* ik = s.ik[(size_t) di];   // (a pack whose sources are not index sources: none, zeros)
            if (ckv == nullptr) {
                err = "source layer " + std::to_string(s.layer) + "'s rows are not on its device";
                return 0;
            }
            const size_t cb = (size_t) rows * ckv_rb(), ib = (size_t) rows * ik_rb();
            cudaSetDevice(devs[(size_t) di].id);
            if (cudaMemcpy(p + off, ckv, cb, cudaMemcpyDeviceToHost) != cudaSuccess ||
                (ik != nullptr && cudaMemcpy(p + off + cb, ik, ib, cudaMemcpyDeviceToHost) != cudaSuccess)) {
                err = "source layer " + std::to_string(s.layer) + "'s compressed KV: " + cudaGetErrorString(cudaGetLastError());
                return 0;
            }
            off += cb + ib;
        }
        {
            const std::vector<int> cs = slot_checks();
            const uint64_t nc = cs.size();
            put(&nc, 8);
            for (int k : cs) {
                const Check& c = ck[k];
                const uint64_t role = (uint64_t) k, cp = (uint64_t) c.pos;
                put(&role, 8);
                put(&cp, 8);
                for (size_t i = 0; i < L.size(); ++i) {
                    put(c.win[i].data(), (size_t) g.win * win_rb());
                    if (L[i].c_kv) {
                        put(c.comp[2 * i].data(), (size_t) L[i].ratio * g.hd * 4);
                        put(c.comp[2 * i + 1].data(), (size_t) L[i].ratio * g.hd * 4);
                    }
                }
            }
        }
        if (off + 8 != out.size()) {
            err = "the state's size does not add up";
            return 0;
        }
        put(&kSlotMagic, 8);
        slot_held = true;
        return (uint64_t) off;
    }

    bool slot_load(const uint8_t* data, size_t bytes, int64_t n_pos, std::string& err) {
        size_t off = 0;
        bool wrote = false, ok = false;
        // a load that stops after some of another conversation's state went into the layers leaves a state no
        // checkpoint can repair: it is reset (the prompt is read from token 0), never served from the mix
        struct FailClosed {
            Impl& p;
            const bool& wrote;
            const bool& ok;
            ~FailClosed() {
                if (!ok && wrote) p.reset();
            }
        } guard{*this, wrote, ok};
        const auto room = [&](size_t b) { return b <= bytes && off <= bytes - b; };
        uint64_t magic = 0, il = 0, n = 0;
        if (!room(16)) {
            err = "the slot is cut short";
            return false;
        }
        std::memcpy(&magic, data + off, 8);
        std::memcpy(&il, data + off + 8, 8);
        off += 16;
        if (magic != kSlotMagic || il == 0 || il > (1u << 20)) {
            err = "not a saved conversation";
            return false;
        }
        if (!room((size_t) il + 8)) {
            err = "the slot is cut short";
            return false;
        }
        {
            std::string id((size_t) il, '\0');
            std::memcpy(id.data(), data + off, (size_t) il);
            off += (size_t) il;
            std::memcpy(&n, data + off, 8);
            off += 8;
            if (id != slot_identity()) {
                err = "another model's, or another engine version's, conversation";
                return false;
            }
        }
        if (n != (uint64_t) n_pos || n_pos <= 0 || n_pos > o.max_context || !room((size_t) n * 4)) {
            err = "the conversation's position is not this prompt's";
            return false;
        }
        std::vector<int32_t> toks_in((size_t) n);
        std::memcpy(toks_in.data(), data + off, (size_t) n * 4);
        off += (size_t) n * 4;
        for (Dev& d : devs) {   // (their work ends before the state is written over)
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
        for (size_t i = 0; i < L.size(); ++i) {
            Layer& l = L[i];
            const Dev& d = devs[(size_t) l.dev];
            cudaSetDevice(d.id);
            uint64_t wb = 0, cn = 0;
            if (!room(8)) {
                err = "the slot is cut short";
                return false;
            }
            std::memcpy(&wb, data + off, 8);
            off += 8;
            if (wb != (uint64_t) g.win * win_rb() || !room((size_t) wb + 8)) {
                err = l.name + "window ring of another size";
                return false;
            }
            wrote = true;
            for (int m = 0; m < l.nm; ++m) {
                Layer& lm = member(l, m);
                if (cudaMemcpy(lm.win, data + off, (size_t) wb, cudaMemcpyHostToDevice) != cudaSuccess) {
                    err = l.name + "window ring: " + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
            }
            off += (size_t) wb;
            std::memcpy(&cn, data + off, 8);
            off += 8;
            if (cn != (l.c_kv ? (uint64_t) l.ratio * (uint64_t) g.hd : 0)) {
                err = l.name + "compressor's group of another size";
                return false;
            }
            if (cn > 0) {
                const size_t cb = (size_t) cn * 4;
                if (!room(2 * cb)) {
                    err = "the slot is cut short";
                    return false;
                }
                for (int m = 0; m < l.nm; ++m) {
                    Layer& lm = member(l, m);
                    if (cudaMemcpy(lm.c_kv, data + off, cb, cudaMemcpyHostToDevice) != cudaSuccess ||
                        cudaMemcpy(lm.c_score, data + off + cb, cb, cudaMemcpyHostToDevice) != cudaSuccess) {
                        err = l.name + "compressor's group: " + cudaGetErrorString(cudaGetLastError());
                        return false;
                    }
                }
                off += 2 * cb;
            }
        }
        for (const Src& s : srcs) {
            uint64_t rows = 0;
            if (!room(8)) {
                err = "the slot is cut short";
                return false;
            }
            std::memcpy(&rows, data + off, 8);
            off += 8;
            if (rows != (uint64_t) (n_pos / std::max(1, s.ratio))) {
                err = "source layer " + std::to_string(s.layer) + "'s rows are not this conversation's";
                return false;
            }
            if (rows == 0) continue;
            const size_t cb = (size_t) rows * ckv_rb(), ib = (size_t) rows * ik_rb();
            if (!room(cb + ib)) {
                err = "the slot is cut short";
                return false;
            }
            for (size_t di = 0; di < devs.size(); ++di) {
                if (s.ckv[di] == nullptr && s.ik[di] == nullptr) continue;
                cudaSetDevice(devs[di].id);
                if (s.ckv[di] != nullptr && cudaMemcpy(s.ckv[di], data + off, cb, cudaMemcpyHostToDevice) != cudaSuccess) {
                    err = "source layer " + std::to_string(s.layer) + "'s compressed KV: " + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
                if (s.ik[di] != nullptr &&
                    cudaMemcpy(s.ik[di], data + off + cb, ib, cudaMemcpyHostToDevice) != cudaSuccess) {
                    err = "source layer " + std::to_string(s.layer) + "'s index keys: " + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
            }
            off += cb + ib;
        }
        Check got[kChecks];   // the slot's checkpoints: the positions it can be taken back at
        {
            uint64_t nc = 0;
            if (!room(8)) {
                err = "the slot is cut short";
                return false;
            }
            std::memcpy(&nc, data + off, 8);
            off += 8;
            if (nc > (uint64_t) kChecks) {
                err = "the slot's checkpoints are damaged";
                return false;
            }
            for (uint64_t k = 0; k < nc; ++k) {
                uint64_t role = 0, cp = 0;
                if (!room(16)) {
                    err = "the slot is cut short";
                    return false;
                }
                std::memcpy(&role, data + off, 8);
                std::memcpy(&cp, data + off + 8, 8);
                off += 16;
                if (role >= (uint64_t) kChecks || got[role].pos >= 0 || cp == 0 || cp > (uint64_t) n_pos) {
                    err = "the slot's checkpoints are damaged";
                    return false;
                }
                Check& c = got[role];
                c.pos = (int64_t) cp;
                c.win.resize(L.size());
                c.comp.resize(L.size() * 2);
                for (size_t i = 0; i < L.size(); ++i) {
                    const size_t wb = (size_t) g.win * win_rb(), cb = L[i].c_kv ? (size_t) L[i].ratio * g.hd * 4 : 0;
                    if (!room(wb + 2 * cb)) {
                        err = "the slot is cut short";
                        return false;
                    }
                    c.win[i].assign(data + off, data + off + wb);
                    off += wb;
                    if (cb > 0) {
                        c.comp[2 * i].resize(cb / 4);
                        c.comp[2 * i + 1].resize(cb / 4);
                        std::memcpy(c.comp[2 * i].data(), data + off, cb);
                        std::memcpy(c.comp[2 * i + 1].data(), data + off + cb, cb);
                        off += 2 * cb;
                    }
                }
            }
        }
        uint64_t end = 0;
        if (!room(8)) {
            err = "the slot is cut short";
            return false;
        }
        std::memcpy(&end, data + off, 8);
        off += 8;
        if (end != kSlotMagic) {
            err = "the slot does not end where it should";
            return false;
        }
        pos = n_pos;
        toks.swap(toks_in);
        ctoks.resize((size_t) pos);
        for (size_t i = 0; i < (size_t) pos; ++i)
            ctoks[i] = toks[i] < pk.eg_vocab ? pk.eg_map[(size_t) toks[i]] : 0;
        for (int k = 0; k < kChecks; ++k) ck[k] = std::move(got[k]);   // (the slot's own; the old ones held the state
                                                               // this one replaced)
        slot_held = false;
        ok = true;
        return true;
    }

    std::string slot_identity() const {
        const auto size_of = [](const std::string& p) {
            std::error_code ec;
            const auto s = std::filesystem::file_size(p, ec);
            return ec ? 0ull : (unsigned long long) s;
        };
        std::ostringstream id;
        id << "strata-ds4-session-3 engine=" << STRATA_VERSION << " mtp=" << g.n_mtp << " dim=" << g.dim << " hd=" << g.hd
           << " win=" << g.win << " ratios=";
        for (int r : g.ratios) id << r << ",";
        id << " kv=";
        for (int v : g.kv_src) id << v << ",";
        id << " ix=";
        for (int v : g.ix_src) id << v << ",";
        id << " ds4.txt:" << size_of(pk.dir + "/ds4.txt") << " engram.bin:" << size_of(pk.dir + "/engram.bin")
           << " tensors.txt:" << size_of(pk.dir + "/tensors.txt");
        for (size_t i = 0; i < pk.files.size(); ++i)
            id << " " << pk.files[i] << ":" << os::file_size(pk.fds[i]);
        return id.str();
    }

    size_t bounce_bytes() const {
        return (size_t) kWin * ((size_t) g.hc * g.dim * 4 + 8 * 4 + topk_stride() * 4 + cand_stride() * 4);
    }
    size_t topk_stride() const { return (size_t) std::max(g.ix_topk, 32); }
    size_t cand_stride() const { return (size_t) std::max(g.cand_topk, 32); }
    size_t bm_stride() const { return (size_t) (o.max_context + 64) / std::max(1, g.cand_blk) + 64; }
    static constexpr int kSlots = kWin * kPicks;   // the CPU lane's (token, expert) rows a window can need

    // ---- the vision tower (the checkpoint's vision.* / aligner.*) on the first GPU, its weights streamed from the
    // host; a prompt's pictures as their spans' rows (set_images)
    VisionTower vt;
    VisionGeo vg;
    int img_tok = -1;
    std::string vision_err;              // why the tower did not load (pictures off), or empty
    std::vector<uint16_t> img_mark[3];   // image_start, image_newline, image_end: BF16 [dim]
    std::vector<Image> imgs;             // the pictures of the prompt being fed, by start
    // a BF16 (or, rounded to nearest even, F32) tensor of n values into dst
    bool read_bf16(const Tensor& t, size_t n, void* dst) {
        if (t.dtype == "BF16") return t.bytes == n * 2 && pk.read(t, dst);
        if (t.dtype != "F32" || t.bytes != n * 4) return false;
        std::vector<float> f(n);
        if (!pk.read(t, f.data())) return false;
        uint16_t* o = (uint16_t*) dst;
        for (size_t i = 0; i < n; ++i) {
            uint32_t u;
            std::memcpy(&u, &f[i], 4);
            o[i] = (uint16_t) ((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
        }
        return true;
    }
    bool load_vision(std::string& err) {
        vg.layers = (int) pk.num("vision.num_hidden_layers", 0);
        const int tok = (int) pk.num("image_token_id", -1);
        if (vg.layers <= 0 || tok < 0 || pk.get("vision.patch_embed.proj.weight") == nullptr) return true;   // none
        vg.dim = (int) pk.num("vision.hidden_size", 1024), vg.heads = (int) pk.num("vision.num_attention_heads", 16);
        vg.inter = (int) pk.num("vision.intermediate_size", 2816), vg.patch = (int) pk.num("vision.patch_size", 14);
        vg.merge = (int) pk.num("vision.downsample_ratio", 3), vg.theta = (float) pk.num("vision.rope_theta", 1e4);
        vg.max_tokens = (int) pk.num("vision.max_image_tokens", 1024), vg.out = g.dim;
        const char* mark[3] = {"image_start", "image_newline", "image_end"};
        for (int i = 0; i < 3; ++i) {
            const Tensor* t = pk.get(mark[i]);
            img_mark[i].resize((size_t) g.dim);
            if (t == nullptr || !read_bf16(*t, (size_t) g.dim, img_mark[i].data())) {
                err = std::string(mark[i]) + ": not a BF16 (or F32) row of the model's width";
                return false;
            }
        }
        const double t0 = now_ms();
        Dev& d = devs[0];
        // the tower's tensors as BF16: the checkpoint's own, or a GGUF's mmproj (F32 norms / biases / marker rows, which
        // came from BF16, converted back exactly; mlp.w1 as the two files' gate and up, the halves of its rows)
        auto read = [this](const std::string& n, std::vector<int64_t>& shape, void* dst) {
            const Tensor* t = pk.get(n);
            if (t == nullptr && n.size() > 13 && n.compare(n.size() - 13, 13, "mlp.w1.weight") == 0) {
                const std::string pre = n.substr(0, n.size() - 13);
                const Tensor* tg = pk.get(pre + "mlp.gate.weight");
                const Tensor* tu = pk.get(pre + "mlp.up.weight");
                if (tg == nullptr || tu == nullptr || tg->shape.size() != 2 || tg->shape != tu->shape) return false;
                shape = {tg->shape[0] * 2, tg->shape[1]};
                const size_t half = (size_t) tg->shape[0] * (size_t) tg->shape[1];
                return dst == nullptr || (read_bf16(*tg, half, dst) && read_bf16(*tu, half, (uint16_t*) dst + half));
            }
            if (t == nullptr) return false;
            shape = t->shape;
            size_t n_el = 1;
            for (int64_t x : t->shape) n_el *= (size_t) x;
            return dst == nullptr || read_bf16(*t, n_el, dst);
        };
        if (!vt.load(vg, d.id, read, err)) return false;
        img_tok = tok;
        CK(cudaSetDevice(d.id));
        CK(cudaMemGetInfo(&d.free0, &d.total));   // (the split sees what the tower left)
        if (o.verbose)
            std::fprintf(stderr, "ds4: vision tower (%d blocks) on CUDA%d: %.0f MiB, up to %d patches, %.1f s\n",
                         vg.layers, d.id, (double) vt.device_bytes() / (1 << 20), vt.max_patches(),
                         (now_ms() - t0) / 1000.0);
        return true;
    }
    bool encode_image(const uint8_t* rgb, int w, int h, std::vector<uint16_t>& span, std::string& err) {
        if (img_tok < 0) {
            err = "this model has no vision tower (or it is off)";
            return false;
        }
        std::vector<uint16_t> rows;
        int nh = 0, nw = 0;
        if (!vt.encode(rgb, w, h, rows, nh, nw, err)) return false;
        const size_t D = (size_t) g.dim;
        span.resize(((size_t) nh * (nw + 1) + 2) * D);
        uint16_t* o = span.data();
        auto put = [&](const uint16_t* r) {
            std::memcpy(o, r, D * 2);
            o += D;
        };
        put(img_mark[0].data());
        for (int i = 0; i < nh; ++i) {
            for (int j = 0; j < nw; ++j) put(rows.data() + ((size_t) i * nw + j) * D);
            put(img_mark[1].data());
        }
        put(img_mark[2].data());
        return true;
    }
    // the row of a picture's span at position p (null: p is in none)
    const uint16_t* image_row(int64_t p) const {
        if (imgs.empty()) return nullptr;
        auto it = std::upper_bound(imgs.begin(), imgs.end(), p, [](int64_t v, const Image& im) { return v < im.start; });
        if (it == imgs.begin()) return nullptr;
        const Image& im = *(it - 1);
        const int64_t r = p - im.start;
        if (!im.rows || r >= (int64_t) (im.rows->size() / (size_t) g.dim)) return nullptr;
        return im.rows->data() + (size_t) r * g.dim;
    }
    // token tk's embedding row at position p: inside a picture's span, the picture's
    const uint16_t* emb_row(int64_t p, int tk) const {
        const uint16_t* r = tk == img_tok ? image_row(p) : nullptr;
        return r != nullptr ? r : embed_pinned + (size_t) tk * g.dim;
    }
    bool in_image(int64_t p) const { return img_tok >= 0 && toks[(size_t) p] == img_tok && image_row(p) != nullptr; }
    // the Engram's compressed id of token tk at p (-1: a picture's, which no n-gram spans)
    int32_t ctok(int64_t p, int tk) const {
        if (tk == img_tok && image_row(p) != nullptr) return -1;
        return tk < pk.eg_vocab ? pk.eg_map[(size_t) tk] : 0;
    }
    // a chunk's image rows (route()'s bits); false: none
    bool vl_bits(int64_t p0, int T, VlBits& b) const {
        if (imgs.empty()) return false;
        std::memset(&b, 0, sizeof b);
        bool any = false;
        for (int t = 0; t < T && t < kVlRows; ++t)
            if (in_image(p0 + t)) {
                b.w[t >> 5] |= 1u << (t & 31);
                any = true;
            }
        return any;
    }

    // ---- Engram: a window's rows of one table (Engram layer li), tokens p0 .. p0 + T - 1 into dst ([token] of [cols]
    // [hd] codes then the scales; nullptr: h_eg).  eg_issue serves the rows already fetched from the row cache (and a
    // row the window needs twice from one read) and hands the rest to the reader's threads; eg_finish waits for them,
    // copies the repeats and keeps the new rows in the cache.  The rows are the file's bytes either way.
    void eg_issue(int64_t p0, int T, int li, uint8_t* dst, EgBatch& b) {
        const int cols = (pk.eg_ngram - 1) * pk.eg_heads, hd = pk.eg_hd;
        const size_t per = (size_t) cols * (hd + hd / 32);
        b.clear();
        b.reads.reserve((size_t) T * cols);
        const std::string P = "layers." + std::to_string(g.eg_layers[(size_t) li]) + ".engram.embed.";
        const Tensor* tw = pk.get(P + "weight");
        const Tensor* ts = pk.get(P + "scale");
        const int fmt = eg_fmt[(size_t) li];
        const bool mx8 = fmt == 0;   // (a GGUF: the scales between the codes, one per 32)
        const bool q6k = fmt == 1;   // (a GGUF: one Q6_K superblock a 256 values, its scales inside)
        const bool iq4 = fmt == 2;   // (a GGUF: 18-byte blocks a 32 values, d and 16 nibble bytes, no scales)
        std::unordered_map<uint64_t, int> first;   // a row this batch reads already -> its read
        int64_t hits = 0;
        {
            std::lock_guard<std::mutex> lk(eg_cache_mu);
            for (int t = 0; t < T; ++t) {
                const int64_t p = p0 + t;
                uint8_t* rows = dst != nullptr ? dst + (size_t) t * per : h_eg + ((size_t) li * eg_cap + t) * per;
                uint8_t* scales = rows + (size_t) cols * hd;
                if (ctoks[(size_t) p] < 0) {   // a picture's token: no rows (zeros: its gate adds nothing)
                    std::memset(rows, 0, per);
                    continue;
                }
                int64_t prod[8];
                bool blocked = false;
                for (int s = 0; s < pk.eg_ngram; ++s) {   // (the look-back stops at the start and at a picture)
                    blocked = blocked || p < s || ctoks[(size_t) (p - s)] < 0;
                    const int64_t tk = blocked ? pk.eg_pad : ctoks[(size_t) (p - s)];
                    prod[s] = tk * pk.eg_mult[(size_t) li * pk.eg_ngram + s];
                }
                int64_t rolling = prod[0];
                for (int n = 1; n < pk.eg_ngram; ++n) {
                    rolling ^= prod[n];
                    for (int h = 0; h < pk.eg_heads; ++h) {
                        const int c = (n - 1) * pk.eg_heads + h;
                        const int64_t id = rolling % pk.eg_primes[(size_t) li * cols + c] + pk.eg_off[(size_t) li * cols + c];
                        uint8_t* rd = rows + (size_t) c * hd;
                        uint8_t* sd = scales + (size_t) c * (hd / 32);
                        const uint64_t key = (uint64_t) id * 8 + (uint64_t) li;
                        if (const uint8_t* hit = eg_cache.find(key)) {
                            std::memcpy(rd, hit, (size_t) hd);
                            std::memcpy(sd, hit + hd, (size_t) hd / 32);
                            ++hits;
                            continue;
                        }
                        const auto fi = first.find(key);
                        if (fi != first.end()) {
                            b.dups.push_back({fi->second, {rd, sd}});
                            continue;
                        }
                        first.emplace(key, (int) b.reads.size());
                        EgRead r;
                        r.file = tw->file;
                        r.dst = rd;
                        r.scales = sd;
                        r.key = key;
                        if (mx8) {
                            r.mx8 = true;
                            r.n = (uint32_t) (hd / 32 * 33);
                            r.off = tw->off + (uint64_t) id * r.n;
                        } else if (q6k || iq4) {   // (into the row's code bytes; its scale bytes unused)
                            r.n = (uint32_t) (q6k ? hd / 256 * 210 : hd / 32 * 18);
                            r.off = tw->off + (uint64_t) id * r.n;
                        } else {
                            r.n = (uint32_t) hd;
                            r.off = tw->off + (uint64_t) id * hd;
                            r.file2 = ts->file;
                            r.n2 = (uint32_t) (hd / 32);
                            r.off2 = ts->off + (uint64_t) id * (hd / 32);
                        }
                        b.reads.push_back(r);
                    }
                }
            }
        }
        eg_rows_n += hits + (int64_t) b.reads.size() + (int64_t) b.dups.size();
        eg_hits_n += hits + (int64_t) b.dups.size();
        eg_reads_n += (int64_t) b.reads.size();
        eg_io.submit(b);
    }
    // b's rows all in place (any thread; a group's members may ask at once: one finishes, the others wait for it)
    void eg_finish(EgBatch& b) {
        if (b.post.load(std::memory_order_acquire) == 2) return;
        const int64_t t0 = now_ns();
        eg_io.wait(b);
        int want = 0;
        if (b.post.compare_exchange_strong(want, 1, std::memory_order_acq_rel)) {
            const int hd = pk.eg_hd;
            for (const auto& d : b.dups) {
                const EgRead& r = b.reads[(size_t) d.first];
                std::memcpy(d.second.first, r.dst, (size_t) hd);
                std::memcpy(d.second.second, r.scales, (size_t) hd / 32);
            }
            {
                std::lock_guard<std::mutex> lk(eg_cache_mu);
                for (size_t i = 0; i < b.reads.size(); ++i) {
                    const EgRead& r = b.reads[i];
                    if (!b.ok[i]) {   // (a failed read: zeros, its gate adds nothing; not kept)
                        std::memset(r.dst, 0, (size_t) hd);
                        std::memset(r.scales, 0, (size_t) hd / 32);
                        continue;
                    }
                    if (uint8_t* c = eg_cache.put(r.key)) {
                        std::memcpy(c, r.dst, (size_t) hd);
                        std::memcpy(c + hd, r.scales, (size_t) hd / 32);
                    }
                }
            }
            eg_io_ns += b.t1.load() - b.t0;
            b.live = false;
            b.post.store(2, std::memory_order_release);
        } else {
            while (b.post.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        }
        eg_wait_ns += now_ns() - t0;
    }
    void eg_stats() {
        st.eg_rows = eg_rows_n.load();
        st.eg_hits = eg_hits_n.load();
        st.eg_reads = eg_reads_n.load();
        st.eg_io_ms = (double) eg_io_ns.load() / 1e6;
        std::lock_guard<std::mutex> lk(eg_cache_mu);
        st.eg_cached = (int64_t) eg_cache.size();
        st.eg_cache_cap = (int64_t) eg_cache.capacity();
    }
    // the chunked prompt path: rows of Engram layer `only` into dst, now (a helper thread, a chunk ahead)
    void engram_rows(int64_t p0, int T, int only, uint8_t* dst) {
        EgBatch b;
        eg_issue(p0, T, only, dst, b);
        eg_finish(b);
    }

    // ---- the CPU lane: a window's routed experts that live on the host, as entries (an expert and up to 8 of its
    // tokens, in rows slot0 ..) computed together: each node its column part of every matrix, from memory on that node,
    // in four rounds (the gate / up split-K pieces, each preparing its two input blocks of every token; each band's
    // sums / codebook / Hadamard / swiglu and that block of the down input, which both nodes then read whole; the down
    // pieces; the down bands).  yd[slot] gets each row's expert output.
    static constexpr int kEntMax = kWin > kc::kExl3CpuMaxN ? kWin : kc::kExl3CpuMaxN;
    struct Ent {
        int e = 0, n = 0, slot0 = 0;
        int tok[kEntMax] = {};
    };
    // tokens an entry holds: the i-quant GGUF experts take a whole window's tokens of an expert in one call (the CPU
    // lane's exact multi-token rows, 7+ tokens an expert: decoded once for all), the others (MXFP4, EXL3) up to 8
    static int ent_cap(const Layer& l) {
        return l.lay.fmt == gg::kFmtMXFP4 && !gg::mx_experts(l.lay) ? kEntMax : kc::kExl3CpuMaxN;
    }
    void run_nodes(int n, const std::function<void(int)>* f) {
        for (int X = 0; X < NH; ++X) pools[(size_t) X]->start(n, f[X]);
        pools[0]->help();
        for (int X = 0; X < NH; ++X) pools[(size_t) X]->wait();
    }

    // a GGUF's MXFP4 experts on the CPU lane, read from the file's mapping or their copy in RAM (mx_part_ptr): each
    // node its rows of every matrix - the gate / up rows [X n_ff / NH, ..) of each entry (the swiglu into its h row),
    // then (h whole) the down rows [X dim / NH, ..) into yd.  A row's sum is the same whatever the split: the same bits
    // in a copy as in place.  The device's arithmetic (gg::mx_rows: int8 activations rounded as the device rounds
    // them, exact integer sums a block; every CPU tier the same bits): each token's input quantized once, whatever
    // number of entries take it, and each job its band of h straight into the down's int8 input.
    // (ord: the order the entries are computed in - their slots, so the sums, stay the same; null: as they are)
    void cpu_experts_mx(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx, const int* ord) {
        if (!gg::mx_experts(l.lay)) return cpu_experts_iq(l, x, ents, ldx, ord);
        constexpr int RB = 64;   // rows a job (whole blocks of h: RB % 32 == 0)
        const int E = g.dim, FF = g.ff, FFh = FF / NH, Eh = E / NH;
        const int GB = (FFh + RB - 1) / RB, DB = (Eh + RB - 1) / RB, ne = (int) ents.size();
        const size_t gr = gg::mx_row_bytes(E), dr = gg::mx_row_bytes(FF);
        const size_t xa = gg::mx_act_bytes(E), ha = gg::mx_act_bytes(FF);
        const bool band_q = FFh % 32 == 0;   // (each job's rows whole blocks of h)
        const float lim = g.lim;
        static_assert(kc::kExl3CpuMaxN <= gg::kMxCpuMaxN, "a CPU-lane entry's tokens in one call");
        {   // the experts read from the files (not in the RAM tier) that are not resident: all their pages asked for at
            // once, so the SSD reads them with large requests side by side while the first entries compute
            // (os::prefetch_cold: the resident ones are left alone - the hint walks every page of a range, which cost
            // more than computing a warm expert, 0.34 ms a 6.3 MB matrix on Windows)
            std::vector<std::pair<const void*, size_t>> want;
            for (const Ent& en : ents)
                if (!store && (l.ram.empty() || l.ram[(size_t) en.e] < 0))
                    for (int r = 0; r < 3; ++r) want.push_back({mx_mat(l, en.e, r), mx_part_bytes(l, r)});
            os::prefetch_cold(want);
        }
        // the int8 inputs in c_xh's bytes: token t's (t < kWin) at t xa, then the slots' h at (kWin + slot) ha; a token
        // index past the window (none on the decode or chunk paths) gets a row of its own after them
        uint8_t* xq = (uint8_t*) c_xh.data();
        uint8_t* hq = xq + (size_t) kWin * xa;
        uint8_t* xq_extra = hq + (size_t) kSlots * ha;
        auto x_act = [&](const Ent& en, int i) -> const void* {
            const int t = en.tok[i];
            return t < kWin ? xq + (size_t) t * xa : xq_extra + (size_t) (en.slot0 + i) * xa;
        };
        {
            bool done[kWin] = {};
            for (const Ent& en : ents)
                for (int i = 0; i < en.n; ++i) {
                    const int t = en.tok[i];
                    if (t < kWin && done[t]) continue;
                    if (t < kWin) done[t] = true;
                    gg::mx_quant(x + (size_t) t * ldx, E, (void*) x_act(en, i));
                }
        }
        std::function<void(int)> f[4];
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (ord != nullptr ? ord[j / GB] : j / GB)];
                const int r0 = X * FFh + (j % GB) * RB, r1 = std::min(X * FFh + FFh, r0 + RB);
                const void* ap[gg::kMxCpuMaxN];
                float yg[gg::kMxCpuMaxN][RB], yu[gg::kMxCpuMaxN][RB];
                float *gp[gg::kMxCpuMaxN], *up[gg::kMxCpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    ap[i] = x_act(en, i);
                    gp[i] = yg[i];
                    up[i] = yu[i];
                }
                // (part X's rows, from where they are: a copy's part on node X, or the file's)
                gg::mx_rows(mx_part_ptr(l, en.e, 0, X), gr, E, r0 - X * FFh, r1 - X * FFh, ap, en.n, gp);
                gg::mx_rows(mx_part_ptr(l, en.e, 1, X), gr, E, r0 - X * FFh, r1 - X * FFh, ap, en.n, up);
                for (int i = 0; i < en.n; ++i) {
                    float* h = band_q ? yg[i] : c_xhd.data() + (size_t) (en.slot0 + i) * FF + r0;
                    for (int r = r0; r < r1; ++r) {
                        float gv = yg[i][r - r0], uv = yu[i][r - r0];
                        if (lim > 0.0f) {
                            gv = std::min(gv, lim);
                            uv = std::min(std::max(uv, -lim), lim);
                        }
                        h[r - r0] = gv / (1.0f + std::exp(-gv)) * uv;
                    }
                    if (band_q) gg::mx_quant_blocks(h, r0 / 32, r1 / 32, hq + (size_t) (en.slot0 + i) * ha, FF);
                }
            };
        run_nodes(ne * GB, f);
        if (!band_q)
            for (const Ent& en : ents)
                for (int i = 0; i < en.n; ++i)
                    gg::mx_quant(c_xhd.data() + (size_t) (en.slot0 + i) * FF, FF, hq + (size_t) (en.slot0 + i) * ha);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (ord != nullptr ? ord[j / DB] : j / DB)];
                const int r0 = X * Eh + (j % DB) * RB, r1 = std::min(X * Eh + Eh, r0 + RB);
                const void* ap[gg::kMxCpuMaxN];
                float* yp[gg::kMxCpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    ap[i] = hq + (size_t) (en.slot0 + i) * ha;
                    yp[i] = c_yd.data() + (size_t) (en.slot0 + i) * E + r0;
                }
                gg::mx_rows(mx_part_ptr(l, en.e, 2, X), dr, FF, r0 - X * Eh, r1 - X * Eh, ap, en.n, yp);
            };
        run_nodes(ne * DB, f);
    }
    // ... the low-bit ones (ds4_iq's types): as above, through ggml-cpu's dots (gg::iq_cpu_*) - each token's input
    // quantized once to the weight type's activation (Q8_K) in its c_xh row's bytes, then (h whole) its h the same
    void cpu_experts_iq(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx, const int* ord) {
        constexpr int RB = 64;   // rows a job
        const int E = g.dim, FF = g.ff, FFh = FF / NH, Eh = E / NH;
        const int GB = (FFh + RB - 1) / RB, DB = (Eh + RB - 1) / RB, ne = (int) ents.size();
        gg::IqCpu fm;
        std::string ferr;
        if (!gg::iq_cpu_fmt(gg::expert_type(l.lay, 0), gg::expert_type(l.lay, 2), E, FF, fm, ferr) ||
            fm.act_bytes > (size_t) E * 4 || fm.h_bytes > (size_t) E * 4) {
            static std::atomic<bool> said{false};
            if (!said.exchange(true)) std::fprintf(stderr, "ds4: layer %d's CPU-lane experts: %s\n", l.il, ferr.c_str());
            for (const Ent& en : ents) std::memset(c_yd.data() + (size_t) en.slot0 * E, 0, (size_t) en.n * E * 4);
            return;
        }
        const float lim = g.lim;
        {   // (os::prefetch_cold, as above: the resident ones are left alone - on Windows the hint walks every page of a
            // range, ~0.13 ms an IQ2_XXS expert, on the lane's critical path before the first entry computes.
            // off by default (the mmap path, owner PC: not faster in every pair); DS4_IQ_PF_COLD=1 turns it on)
            std::vector<std::pair<const void*, size_t>> want;
            for (const Ent& en : ents)
                if (!store && (l.ram.empty() || l.ram[(size_t) en.e] < 0))
                    for (int r = 0; r < 3; ++r) want.push_back({mx_mat(l, en.e, r), mx_part_bytes(l, r)});
            static const bool pf_all = std::getenv("DS4_IQ_PF_COLD") == nullptr || std::atoi(std::getenv("DS4_IQ_PF_COLD")) == 0;
            if (pf_all) os::prefetch(want);
            else os::prefetch_cold(want);
        }
        // the activations in c_xh's bytes, E 4 apart: token t's (t < kWin) at row t - quantized once, whatever number of
        // entries take it - then a row a slot (kWin + slot): a token past the window's input (none on the decode or
        // chunk paths), then (x no longer needed) the slot's h
        uint8_t* aq = (uint8_t*) c_xh.data();
        auto x_act = [&](const Ent& en, int i) -> uint8_t* {
            const int t = en.tok[i];
            return aq + (size_t) (t < kWin ? t : kWin + en.slot0 + i) * E * 4;
        };
        {
            bool done[kWin] = {};
            for (const Ent& en : ents)
                for (int i = 0; i < en.n; ++i) {
                    const int t = en.tok[i];
                    if (t < kWin && done[t]) continue;
                    if (t < kWin) done[t] = true;
                    gg::iq_cpu_quant(fm.gu_act, x + (size_t) t * ldx, E, x_act(en, i));
                }
        }
        std::function<void(int)> f[4];
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (ord != nullptr ? ord[j / GB] : j / GB)];
                const int r0 = X * FFh + (j % GB) * RB, r1 = std::min(X * FFh + FFh, r0 + RB);
                const void* ap[kEntMax];
                float* hp[kEntMax];
                for (int i = 0; i < en.n; ++i) {
                    ap[i] = x_act(en, i);
                    hp[i] = c_xhd.data() + (size_t) (en.slot0 + i) * FF + (size_t) X * FFh;   // (part X's rows of h)
                }
                // (part X's rows, from where they are: a copy's part on node X, or the file's - whole rows, so any
                // block size of the type)
                gg::iq_cpu_gate_up(fm, mx_part_ptr(l, en.e, 0, X), mx_part_ptr(l, en.e, 1, X), ap, en.n, hp,
                                   r0 - X * FFh, r1 - X * FFh, lim);
            };
        run_nodes(ne * GB, f);
        for (const Ent& en : ents)
            for (int i = 0; i < en.n; ++i)
                gg::iq_cpu_quant(fm.d_act, c_xhd.data() + (size_t) (en.slot0 + i) * FF, FF,
                                 aq + (size_t) (kWin + en.slot0 + i) * E * 4);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (ord != nullptr ? ord[j / DB] : j / DB)];
                const int r0 = X * Eh + (j % DB) * RB, r1 = std::min(X * Eh + Eh, r0 + RB);
                const void* hp[kEntMax];
                float* yp[kEntMax];
                for (int i = 0; i < en.n; ++i) {
                    hp[i] = aq + (size_t) (kWin + en.slot0 + i) * E * 4;
                    yp[i] = c_yd.data() + (size_t) (en.slot0 + i) * E + (size_t) X * Eh;   // (part X's rows of it)
                }
                gg::iq_cpu_down(fm, mx_part_ptr(l, en.e, 2, X), hp, en.n, yp, r0 - X * Eh, r1 - X * Eh);
            };
        run_nodes(ne * DB, f);
    }

    // the window's CPU-lane experts made ready (and pinned) in the store for the length of the call
    struct StoreHold {
        Impl* m = nullptr;
        int layer = 0;
        std::vector<int> es;
        std::vector<char> ok;
        StoreHold(Impl* m_, const Layer& l, const std::vector<Ent>& ents) {
            if (!m_->store) return;
            m = m_;
            layer = l.il;
            for (const Ent& en : ents)
                if ((l.ram.empty() || l.ram[(size_t) en.e] < 0) && std::find(es.begin(), es.end(), en.e) == es.end()) es.push_back(en.e);
            m->store->ensure(layer, es.data(), (int) es.size(), ok);
        }
        ~StoreHold() {
            if (m) m->store->release(layer, es.data(), (int) es.size(), ok);
        }
    };
    void cpu_experts(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx = 0,
                     const int* ord = nullptr) {
        if (ldx == 0) ldx = (size_t) g.dim;
        if (l.lay.fmt == gg::kFmtMXFP4) {
            StoreHold hold(this, l, ents);
            return cpu_experts_mx(l, x, ents, ldx, ord);
        }
        const int E = g.dim, FF = g.ff, FFh = FF / NH, Eh = E / NH, GQ = E / 256, DQ = FF / 128;
        const int GB = FFh / 128, DB = Eh / 128, EB = E / 128, FB = FF / 128;
        const int ne = (int) ents.size();
        const float lim = g.lim;
        auto part = [&](int e, int X, int r) {
            const uint8_t* b = l.hh[(size_t) e][(size_t) X];
            kc::Exl3Part p;
            p.trellis = (const uint32_t*) (b + l.hl.trellis[r]);
            p.suh = (const uint16_t*) (b + l.hl.suh[r]);
            p.svh = (const uint16_t*) (b + l.hl.svh[r]);
            p.k = l.hl.k[r];
            p.n = l.hl.n[r];
            p.K = l.lay.K[r];
            return p;
        };
        auto xh = [&](int X, int sl, int r) { return c_xh.data() + ((size_t) (X * kSlots + sl) * 2 + r) * E; };
        auto xsb = [&](int X, int sl, int r) { return c_xsb.data() + ((size_t) (X * kSlots + sl) * 2 + r) * EB; };
        auto acc = [&](int X, int sl, int r, int q) {
            return c_acc.data() + (((size_t) (X * kSlots + sl) * 2 + r) * GQ + q) * FFh;
        };
        auto xhd = [&](int sl) { return c_xhd.data() + (size_t) sl * FF; };
        auto xsd = [&](int sl) { return c_xsd.data() + (size_t) sl * FB; };
        auto accd = [&](int X, int sl, int q) { return c_accd.data() + ((size_t) (X * kSlots + sl) * DQ + q) * Eh; };
        std::function<void(int)> f[4];
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / (2 * GQ))];
                const int r = (j / GQ) % 2, q = j % GQ;
                const kc::Exl3Part p = part(en.e, X, r);
                const float* xp[kc::kExl3CpuMaxN];
                float* ap[kc::kExl3CpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    const int sl = en.slot0 + i;
                    const float* xt = x + (size_t) en.tok[i] * ldx;
                    for (int b = 2 * q; b < 2 * q + 2; ++b)
                        xsb(X, sl, r)[b] = kc::exl3_cpu_prepare_block(p.suh + b * 128, xt + b * 128, xh(X, sl, r) + b * 128);
                    xp[i] = xh(X, sl, r);
                    ap[i] = acc(X, sl, r, q);
                }
                if (en.n == 1) kc::exl3_cpu_rows_acc(p, xp[0], q * 16, (q + 1) * 16, ap[0]);
                else kc::exl3_cpu_rows_acc_n(p, xp, en.n, q * 16, (q + 1) * 16, ap);
            };
        run_nodes(ne * 2 * GQ, f);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / GB)];
                const int b = j % GB;
                const kc::Exl3Part pg = part(en.e, X, 0), pu = part(en.e, X, 1), pd = part(en.e, X, 2);
                for (int i = 0; i < en.n; ++i) {
                    const int sl = en.slot0 + i;
                    float y[2][128], a[128], hb[128];
                    for (int r = 0; r < 2; ++r) {
                        for (int c = 0; c < 128; ++c) a[c] = 0.0f;
                        for (int q = 0; q < GQ; ++q) {
                            const float* aq = acc(X, sl, r, q) + b * 128;
                            for (int c = 0; c < 128; ++c) a[c] += aq[c];
                        }
                        double xs = 0.0;
                        for (int k = 0; k < EB; ++k) xs += xsb(X, sl, r)[k];
                        kc::exl3_cpu_band_out(r == 0 ? pg : pu, a, (float) xs, b * 128, y[r]);
                    }
                    for (int c = 0; c < 128; ++c) {
                        float gv = y[0][c], uv = y[1][c];
                        if (lim > 0.0f) {
                            gv = std::min(gv, lim);
                            uv = std::min(std::max(uv, -lim), lim);
                        }
                        hb[c] = gv / (1.0f + std::exp(-gv)) * uv;
                    }
                    const int gb = X * GB + b;
                    xsd(sl)[gb] = kc::exl3_cpu_prepare_block(pd.suh + gb * 128, hb, xhd(sl) + gb * 128);
                }
            };
        run_nodes(ne * GB, f);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / DQ)];
                const int q = j % DQ;
                const kc::Exl3Part p = part(en.e, X, 2);
                const float* xp[kc::kExl3CpuMaxN];
                float* ap[kc::kExl3CpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    xp[i] = xhd(en.slot0 + i);
                    ap[i] = accd(X, en.slot0 + i, q);
                }
                if (en.n == 1) kc::exl3_cpu_rows_acc(p, xp[0], q * 8, (q + 1) * 8, ap[0]);
                else kc::exl3_cpu_rows_acc_n(p, xp, en.n, q * 8, (q + 1) * 8, ap);
            };
        run_nodes(ne * DQ, f);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / DB)];
                const int b = j % DB;
                const kc::Exl3Part pd = part(en.e, X, 2);
                for (int i = 0; i < en.n; ++i) {
                    const int sl = en.slot0 + i;
                    float a[128];
                    for (int c = 0; c < 128; ++c) a[c] = 0.0f;
                    for (int q = 0; q < DQ; ++q) {
                        const float* aq = accd(X, sl, q) + b * 128;
                        for (int c = 0; c < 128; ++c) a[c] += aq[c];
                    }
                    double xs = 0.0;
                    for (int k = 0; k < FB; ++k) xs += xsd(sl)[k];
                    kc::exl3_cpu_band_out(pd, a, (float) xs, b * 128, c_yd.data() + (size_t) sl * E + (size_t) X * Eh + b * 128);
                }
            };
        run_nodes(ne * DB, f);
    }

    // ---- the expert cache.  A swap moves a CPU-lane expert (`in`) into the layer's spare VRAM slot and the VRAM expert it
    // replaces (`out`) into the spare host parts - both copies of weights that do not change, made while the model runs
    // (the copy stream, the host's repacking) - and cache_tick switches the tables between windows, so the device's
    // table and the host's view of a layer always agree within a window.  In place (file_experts) `out` needs no copy:
    // the CPU lane reads it from the file again.
    void ram_loop() {   // the RAM tier's moves in: an expert read from the file into a free slot
        for (;;) {
            RamSwap rs;
            {
                std::unique_lock<std::mutex> lk(sw_mu);
                ram_cv.wait(lk, [&] { return quit.load() || !ram_queue.empty(); });
                if (quit.load()) return;
                rs = ram_queue.front();
                ram_queue.pop_front();
            }
            const Layer& l = L[(size_t) rs.layer];
            if (read_mx_file(l, rs.in, ram_slots[(size_t) rs.slot])) {
                for (int r = 0; r < 3; ++r)   // (one copy of it in RAM)
                    os::advise(pk.fds[(size_t) l.mx_file[r]], l.mx_off[r] + (uint64_t) rs.in * l.mx_stride[r],
                               mx_part_bytes(l, r), false);
            } else {
                rs.in = -1;   // (failed: the slot goes back)
            }
            std::lock_guard<std::mutex> lk(sw_mu);
            ram_ready.push_back(rs);
        }
    }

    bool cache_epoch_commit_ready(std::string& err) {
#define EPOCH_CK(expr) do {const cudaError_t e=(expr);if(e!=cudaSuccess){err=std::string("cache epoch commit: ")+cudaGetErrorString(e);return false;}} while(0)
        if (!o.expert_cache || cache_ths.empty()) return true;
        std::vector<Swap> ready;
        std::vector<RamSwap> rready;
        {
            std::lock_guard<std::mutex> lk(sw_mu);
            ready.swap(sw_ready);
            rready.swap(ram_ready);
        }
        for (const RamSwap& rs : rready) {   // (nothing reads the tier between windows)
            --ram_inflight;
            Layer& l = L[(size_t) rs.layer];
            if (rs.in < 0 || l.ram[(size_t) rs.in] >= 0 || l.slot[(size_t) rs.in] >= 0) {
                ram_free.push_back(rs.slot);
                continue;
            }
            l.ram[(size_t) rs.in] = rs.slot;
            if (rs.out >= 0 && l.ram[(size_t) rs.out] >= 0) {
                ram_free.push_back(l.ram[(size_t) rs.out]);
                l.ram[(size_t) rs.out] = -1;
            }
            ++st.ram_moves;
        }
        for (const Swap& sw : ready) {
            if(!sw.epoch_prepared_ok) {err="cache epoch: pending swap preparation failed";return false;}
            Layer& l = L[(size_t) sw.layer];
            const int old_slot = l.slot[(size_t) sw.out];
            const std::array<uint8_t*, 4> old_parts = l.hh[(size_t) sw.in];
            if (copy_mode) {   // (`out` takes the RAM place `in` leaves: through the spare, which `in`'s slot becomes)
                const int s_in = l.ram[(size_t) sw.in];
                l.ram[(size_t) sw.in] = -1;
                if (sw.ram >= 0) {
                    l.ram[(size_t) sw.out] = sw.ram;
                    copy_spare[(size_t) l.il] = s_in;
                } else if (s_in >= 0) {   // (the move into RAM failed: `out` back to the file, one copy fewer)
                    --ram_held;
                    if (copy_spare[(size_t) l.il] < 0) copy_spare[(size_t) l.il] = s_in;
                }
            } else if (!l.ram.empty() && l.ram[(size_t) sw.in] >= 0) {   // (in VRAM now: its RAM slot for the next hottest)
                ram_free.push_back(l.ram[(size_t) sw.in]);
                l.ram[(size_t) sw.in] = -1;
            }
            l.slot[(size_t) sw.in] = l.spare_vram;
            if (!l.zc.empty() && l.zc[(size_t) sw.in]) {   // (VRAM's now: not the store's to compute any more)
                l.zc[(size_t) sw.in] = 0;
                if (l.zc_tab != nullptr) {   // (the device's table too: its entry pointed into the slot the store now reuses - when `in` left VRAM later,
                                             // zc_stage copied whatever lay there, another expert's blob, into the plan)
                    EPOCH_CK(cudaSetDevice(devs[0].id));
                    EPOCH_CK(cudaMemsetAsync(l.zc_tab + sw.in, 0, 8, devs[0].st));
                }
                store->unmark(l.il, sw.in);
            }
            if (l.nm == 1) set_vm(l, sw.in, l.spare_vram);
            l.hh[(size_t) sw.in] = {nullptr, nullptr, nullptr, nullptr};
            l.host[(size_t) sw.in] = nullptr;
            l.slot[(size_t) sw.out] = -1;
            if (l.lay.fmt == gg::kFmtMXFP4) {   // (the file has it)
                l.host[(size_t) sw.out] = (uint8_t*) mx_mat(l, sw.out, 0);
            } else if (file_experts) {   // (back to its pages in the file)
                l.hh[(size_t) sw.out] = {file_part(l, sw.out), nullptr, nullptr, nullptr};
                l.host[(size_t) sw.out] = l.hh[(size_t) sw.out][0];
            } else {
                l.hh[(size_t) sw.out] = l.spare_host;
                l.host[(size_t) sw.out] = l.spare_host[0];
                l.spare_host = old_parts;
            }
            l.spare_vram = old_slot;
            for (int m = 0; m < l.nm; ++m) {
                Layer& x = member(l, m);
                const Dev& dm = devs[(size_t) x.dev];
                l.tab_pin[2 * m] = (unsigned long long) (x.vram_base + (size_t) l.slot[(size_t) sw.in] * unit(x));
                l.tab_pin[2 * m + 1] = 0ull;
                EPOCH_CK(cudaSetDevice(dm.id));   // (stream order: before the next window)
                EPOCH_CK(cudaMemcpyAsync(x.vram_tab + sw.in, &l.tab_pin[2 * m], 8, cudaMemcpyHostToDevice, dm.st));
                EPOCH_CK(cudaMemcpyAsync(x.vram_tab + sw.out, &l.tab_pin[2 * m + 1], 8, cudaMemcpyHostToDevice, dm.st));
            }
            l.swap_pending = false;
            --sw_inflight;
            ++st.swaps;
        }
        return true;
#undef EPOCH_CK
    }

    // No pool/workspace changes. A future caller must own the serialized Model API
    // boundary and call end_cache_worker_epoch before issuing any new GPU work.
    void cache_epoch_release(bool terminal=false) {
        cache_epoch_request.store(0,std::memory_order_release);
        cache_epoch_block_enqueues.store(terminal,std::memory_order_release);
        cache_epoch_machine.release(terminal);
        cache_epoch_owner.clear(std::memory_order_release);
        sw_cv.notify_all();
    }
    bool cache_epoch_terminal(const std::string& why,std::string& err) {
        err=why;
        chunk_major_phase_error="cache-worker epoch terminal error: "+why;
        cache_epoch_release(true); // workers may exit; no future Model API may compute through failed tables
        std::fprintf(stderr,"ds4: ERROR: %s\n",chunk_major_phase_error.c_str());
        return false;
    }
    bool cache_epoch_wait_worker(std::unique_lock<std::mutex>& lock,unsigned& last_ack) {
        for(;;) {
            if(quit.load())return false;
            if(!sw_queue.empty())return true; // all old work completes before acknowledging the pause
            const unsigned e=cache_epoch_request.load(std::memory_order_acquire);
            if(e) {
                if(last_ack!=e){last_ack=e;++cache_epoch_worker_acks;sw_cv.notify_all();}
                sw_cv.wait(lock,[&]{return quit.load()||cache_epoch_request.load(std::memory_order_acquire)!=e;});
            } else sw_cv.wait(lock,[&]{return quit.load()||!sw_queue.empty()||cache_epoch_request.load(std::memory_order_acquire)!=0;});
        }
    }
    bool begin_cache_worker_epoch(unsigned& lease,int timeout_ms,std::string& err) {
        lease=0;
        if(gpu_misses_decode_pins&&!gpu_misses_retained_ready())return cache_epoch_terminal("stale retained decode pin context",err);
        if(!cache_epoch_enabled||!copied_ram_domains_published||NH!=1||!copy_mode||store||ram_tier||
           devs.size()!=1||chunks.size()!=1||!chunks[0]||!chunks[0]->flow||!service.joinable()||!reqs||!resps||
           !chunk_major_phase_error.empty()||((registered_copy.published||!registered_copy.held.empty())&&!gpu_misses_retained_ready())) {
            err="cache epoch requires a healthy, consumed, unregistered NH1 copied-RAM model";return false;
        }
        if(chunk_major_phase_pending||prefill_phase_pending) {err="cache epoch cannot overlap a live prefill phase";return false;}
        for(const auto& w:workers)if(!w->idle()){err="cache epoch requires serialized idle prompt workers";return false;}
        if(cache_epoch_owner.test_and_set(std::memory_order_acquire)){err="cache epoch already owned";return false;}
        if(!cache_epoch_machine.start(now_ms(),timeout_ms)) {
            cache_epoch_owner.clear(std::memory_order_release);err="cache epoch policy/state/timeout refused";return false;
        }
        struct EpochRollback {
            Impl* p;unsigned epoch;
            ~EpochRollback(){const auto phase=p->cache_epoch_machine.phase;
                if(p->cache_epoch_machine.serial==epoch&&phase!=epochpolicy::Phase::idle&&
                   phase!=epochpolicy::Phase::held&&phase!=epochpolicy::Phase::failed)p->cache_epoch_release(false);}
        } rollback{this,cache_epoch_machine.serial}; // callback/metadata exceptions cannot strand an acknowledged worker
        const unsigned target=seq_issued;
        cache_epoch_block_enqueues.store(true,std::memory_order_release);
        auto soft=[&](const char* why){err=why;cache_epoch_release(false);return false;};
        auto wait_ok=[&] {
            if(quit.load()){err="cache epoch teardown requested";return false;}
            if(cancel&&cancel()){err="cache epoch canceled";return false;}
            if(cache_epoch_machine.expired(now_ms())){err="cache epoch timed out; sequences/maps retained";return false;}
            return true;
        };
        auto streams=[&](bool& idle)->bool {
            idle=true;
            auto query=[&](cudaStream_t s){if(!s)return true;const cudaError_t e=cudaStreamQuery(s);
                if(e==cudaErrorNotReady){idle=false;return true;}if(e!=cudaSuccess){err=std::string("cache epoch stream: ")+cudaGetErrorString(e);return false;}return true;};
            if(cudaSetDevice(devs[0].id)!=cudaSuccess){err="cache epoch selecting CUDA device";return false;}
            if(!query(devs[0].st)||!query(chunks[0]->cs)||!query(gpu_misses_cs))return false;
            for(const auto& B:sw_bufs)for(cudaStream_t s:B.st)if(!query(s))return false;
            Chunk& c=*chunks[0];
            for(size_t i=c.base;i<c.n_read;++i)if(fit(c,i).left.load(std::memory_order_acquire)!=0)idle=false;
            return true;
        };
        auto mailbox=[&] {
            const unsigned completed=cache_epoch_service_completed.load(std::memory_order_acquire);
            if(completed>target)return epochpolicy::Mailbox::invalid;
            if(completed<target)return epochpolicy::Mailbox::pending;
            unsigned q[kRing],r[kRing];
            for(int i=0;i<kRing;++i){q[i]=*(volatile unsigned*)&reqs[i].seq;r[i]=*(volatile unsigned*)&resps[i].seq;}
            return epochpolicy::mailbox(target,completed,q,r,kRing);
        };
        // Keep the service live until every already-issued GPU publication and response finishes.
        for(;;) {
            if(!wait_ok()){cache_epoch_release(false);return false;}
            if(seq_issued!=target)return cache_epoch_terminal("Model API serialization changed during epoch",err);
            bool idle=false;if(!streams(idle))return cache_epoch_terminal(err,err);
            if(idle) {
                const auto m=mailbox();
                if(m==epochpolicy::Mailbox::invalid)return soft("cache epoch mailbox/sequence wrap refused");
                if(m==epochpolicy::Mailbox::complete)break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        cache_epoch_machine.publishers_done();
        for(;;) {
            if(!wait_ok()){cache_epoch_release(false);return false;}
            if(!cache_epoch_commit_ready(err))return cache_epoch_terminal(err,err);
            bool idle=false;if(!streams(idle))return cache_epoch_terminal(err,err);
            bool done=false;
            {std::lock_guard<std::mutex> lock(sw_mu);
             done=sw_inflight==0&&ram_inflight==0&&sw_queue.empty()&&sw_ready.empty()&&ram_queue.empty()&&ram_ready.empty();}
            if(done&&idle)break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        for(const Layer& l:L)if(l.swap_pending)return cache_epoch_terminal("cache epoch pending-layer count mismatch",err);
        cache_epoch_machine.cache_done();
        {std::lock_guard<std::mutex> lock(sw_mu);cache_epoch_worker_acks=0;
         cache_epoch_service_target.store(target,std::memory_order_release);
         cache_epoch_request.store(cache_epoch_machine.serial,std::memory_order_release);}
        sw_cv.notify_all();
        for(;;) {
            if(!wait_ok()){cache_epoch_release(false);return false;}
            size_t ack=0;{std::lock_guard<std::mutex> lock(sw_mu);ack=cache_epoch_worker_acks;}
            if(ack==cache_ths.size()&&cache_epoch_service_ack.load(std::memory_order_acquire)==cache_epoch_machine.serial)break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if(mailbox()!=epochpolicy::Mailbox::complete)return cache_epoch_terminal("cache epoch mailbox changed after pause",err);
        if(gpu_misses_decode_pins&&!gpu_misses_retire_pins("epoch",err))return cache_epoch_terminal(err,err);
        if(gpu_miss_direct_enabled&&(gpu_misses_decode_pins||registered_copy.published||!registered_copy.held.empty()))
            return cache_epoch_terminal("retained pins survived epoch drain",err);
        ramdomain::Inventory inventory;
        if(!inspect_copied_ram_inventory(inventory,err)){cache_epoch_release(false);return false;}
        cache_epoch_machine.acknowledged();lease=cache_epoch_machine.serial;
        return true;
    }
    bool end_cache_worker_epoch(unsigned lease,std::string& err) {
        if(!lease||cache_epoch_machine.phase!=epochpolicy::Phase::held||lease!=cache_epoch_machine.serial) {
            err="cache epoch release lease mismatch";return false;
        }
        cache_epoch_release(false);return true;
    }

    // UNUSED prerequisite: maintenance DMA is allowed ONLY while the actual
    // epoch lease is HELD. Ordinary Model/prompt/decode work remains forbidden.
    // No caller, new flag, workspace change, registration or first-phase reset.
    bool shrink_copied_expert_pool(unsigned lease,bool& stopped,std::string& err) {
        stopped=false;
#if defined(STRATA_USE_HIP)
        (void)lease;err="copied-pool shrink is a CUDA-only unused prerequisite";return false;
#else
        if(!cache_epoch_enabled||!record_copied_ram_domains||!copied_ram_domains_published||
           !lease||cache_epoch_machine.phase!=epochpolicy::Phase::held||lease!=cache_epoch_machine.serial||
           cache_epoch_request.load(std::memory_order_acquire)!=lease||
           !cache_epoch_block_enqueues.load(std::memory_order_acquire)||
           cache_epoch_service_ack.load(std::memory_order_acquire)!=lease||
           cache_epoch_service_completed.load(std::memory_order_acquire)!=seq_issued||
           !chunk_major_phase_error.empty()||chunk_major_phase_pending||prefill_phase_pending||
           devs.size()!=1||chunks.size()!=1||!chunks[0]||NH!=1||!copy_mode||store||ram_tier||
           !pk.gguf||!file_experts||!o.expert_cache||registered_copy.published||!registered_copy.held.empty()) {
            err="copied-pool shrink requires the healthy unregistered HELD single-GPU copied-RAM epoch";return false;
        }
        for(const auto& w:workers)if(!w->idle()){err="copied-pool shrink requires idle prompt workers";return false;}
        {
            std::lock_guard<std::mutex> lock(sw_mu);
            if(cache_epoch_worker_acks!=cache_ths.size()||sw_inflight||ram_inflight||
               !sw_queue.empty()||!sw_ready.empty()||!ram_queue.empty()||!ram_ready.empty()) {
                err="copied-pool shrink requires acknowledged empty cache queues";return false;
            }
        }
        Dev& d=devs[0];Chunk& c=*chunks[0];
        if(!c.flow||c.C!=chunk||c.lm||c.lm_state||c.lc||c.prefill_gpu_h||c.lm_gpu||c.lm_f16||
           c.cold_f16||c.eg_weight_cache||c.dense_projection_cache||sw_bufs.empty()||!sw_bufs[0].b||
           std::count(host_allocs.begin(),host_allocs.end(),(void*)sw_bufs[0].b)!=1) {
            err="copied-pool shrink requires consumed phase and owned paused swap staging";return false;
        }
        auto hard=[&](const std::string& why){return cache_epoch_terminal("copied-pool shrink: "+why,err);};
        cudaError_t ce=cudaSetDevice(d.id);
        if(ce!=cudaSuccess)return hard(std::string("device selection: ")+cudaGetErrorString(ce));
        for(size_t i=c.base;i<c.n_read;++i)if(fit(c,i).left.load(std::memory_order_acquire)!=0)
            return hard("a flow read outlived the held epoch");
        ce=cudaStreamQuery(c.cs);if(ce!=cudaSuccess)return hard("copy stream not idle at held boundary");
        ce=cudaStreamQuery(d.st);if(ce!=cudaSuccess)return hard("compute stream not idle at held boundary");
        ramdomain::Inventory inventory;
        if(!inspect_copied_ram_inventory(inventory,err))return hard(err);
        std::vector<copiedshrink::Plan> plans;size_t planned_demotions=0;
        try {
            plans.reserve(copied_ram_domains.size());
            for(const auto& domain:copied_ram_domains) {
                if(domain.layer<0||(size_t)domain.layer>=L.size())return hard("domain layer index changed");
                Layer& l=L[(size_t)domain.layer];
                if(l.mtp||l.cpu_only||l.nm!=1||l.dev!=0||l.lay.fmt!=gg::kFmtMXFP4||
                   l.vm.size()!=(size_t)l.n_exp||l.host.size()!=(size_t)l.n_exp||l.hh.size()!=(size_t)l.n_exp||
                   !l.vram_tab||!l.vs_pin||l.swap_pending||cp_lay[(size_t)l.il][3]!=unit(l))
                    return hard("unsupported current layer/view binding");
                for(int r=0;r<3;++r)if(l.lay.trellis[r]!=cp_lay[(size_t)l.il][(size_t)r])
                    return hard("GPU blob/CPU role offsets differ");
                copiedshrink::Plan plan;
                if(!copiedshrink::plan(domain,l.slot,l.n_slots,l.spare_vram,l.ram,copy_spare[(size_t)l.il],
                       inventory.layers[(size_t)l.il].free_slots,unit(l),plan,err))
                    return hard(err);
                if(plan.next_slots<plan.old_slots) {
                    if(plan.demoted.size()>(size_t)std::numeric_limits<int>::max()-(size_t)ram_held-planned_demotions)
                        return hard("RAM-held counter overflow");
                    planned_demotions+=plan.demoted.size();plans.push_back(std::move(plan));
                }
            }
            // No allocation is permitted in the host publication or final
            // plan_streams rebuild after any layer has committed.
            for(Layer& l:L)if(!l.mtp)l.host_list.reserve((size_t)l.n_exp);
            std::stable_sort(plans.begin(),plans.end(),[](const auto& a,const auto& b){return a.next_bytes<b.next_bytes;});
        } catch(...) {err="copied-pool shrink host planning memory unavailable; pool unchanged";return false;}
        const double started=now_ms();const size_t before=d.experts;
        size_t committed=0,skipped=0,demoted_count=0,demoted_bytes=0,prefix_bytes=0,completed_d2h=0,completed_d2d=0;
        uint8_t* const bounce=sw_bufs[0].b;
        // The whole loop is guarded: callbacks and std::function setup can
        // throw even BETWEEN layer transactions after an earlier commit.
        try {
        for(const copiedshrink::Plan& plan:plans) {
            if(cancel&&cancel()){stopped=true;break;}
            Layer& l=L[(size_t)plan.layer];
            uint8_t* const old=l.vram_base;uint8_t* next=nullptr;
            decltype(l.vm) next_vm;decltype(l.host) next_host;decltype(l.hh) next_hh;decltype(l.ram) next_ram;
            std::vector<int> next_slot;std::vector<unsigned long long> table;
            grow::Callbacks ops;
            bool observed_stop=false;
            ops.cancelled=[&]{if(cancel&&cancel())observed_stop=true;return observed_stop;};
            ops.discard=[&] {
                if(!next)return true;
                if(cudaStreamSynchronize(d.st)!=cudaSuccess)return false;
                auto it=std::find(d.allocs.begin(),d.allocs.end(),(void*)next);
                if(it==d.allocs.end()||cudaFree(next)!=cudaSuccess)return false;
                d.allocs.erase(it);next=nullptr;return true;
            };
            ops.prepare=[&] {
                if(!old||std::count(d.allocs.begin(),d.allocs.end(),(void*)old)!=1||
                   d.experts<plan.old_bytes||l.n_slots!=plan.old_slots||l.spare_vram!=plan.old_spare)
                    return grow::Prepare::fatal;
                try {
                    d.allocs.reserve(d.allocs.size()+1);
                    next_vm.resize((size_t)l.n_exp);next_host=l.host;next_hh=l.hh;next_ram=l.ram;
                    next_slot=plan.slot;table.assign((size_t)l.n_exp,0ull);
                } catch(...) {return grow::Prepare::unavailable;}
                size_t free=0,total=0;ce=cudaMemGetInfo(&free,&total);
                if(ce!=cudaSuccess)return grow::Prepare::fatal;
                const size_t floor=(size_t)1024<<20;
                if(free<plan.next_bytes||free-plan.next_bytes<floor||
                   (commit_charged(d)&&commit_room()<(double)plan.next_bytes))return grow::Prepare::unavailable;
                ce=cudaMalloc((void**)&next,plan.next_bytes);
                if(ce==cudaErrorMemoryAllocation){(void)cudaGetLastError();next=nullptr;return grow::Prepare::unavailable;}
                if(ce!=cudaSuccess)return grow::Prepare::fatal;
                d.allocs.push_back(next);
                for(size_t i=0;i<plan.demoted.size();++i) {
                    if(ops.cancelled())return grow::Prepare::unavailable;
                    const int e=plan.demoted[i],slot=plan.destinations[i];
                    ce=cudaMemcpyAsync(bounce,old+(size_t)l.slot[(size_t)e]*unit(l),unit(l),cudaMemcpyDeviceToHost,d.st);
                    if(ce!=cudaSuccess)return grow::Prepare::fatal;
                    ce=cudaStreamSynchronize(d.st);if(ce!=cudaSuccess)return grow::Prepare::fatal;
                    completed_d2h+=unit(l);
                    blob_to_ram(l,bounce,slot); // CURRENT free hole, not its stale prior contents
                    for(int r=0;r<3;++r)if(std::memcmp(bounce+l.lay.trellis[r],
                         ram_part[(size_t)slot][0]+cp_lay[(size_t)l.il][(size_t)r],mx_part_bytes(l,r))!=0)
                        return grow::Prepare::fatal;
                    next_ram[(size_t)e]=slot;next_host[(size_t)e]=ram_part[(size_t)slot][0]+cp_lay[(size_t)l.il][0];
                    next_hh[(size_t)e]={nullptr,nullptr,nullptr,nullptr};
                }
                if(ops.cancelled())return grow::Prepare::unavailable;
                // All retained physical slot numbers are unchanged. Copy the
                // complete target prefix once, including inert spare bytes.
                ce=cudaMemcpyAsync(next,old,plan.next_bytes,cudaMemcpyDeviceToDevice,d.st);
                if(ce!=cudaSuccess)return grow::Prepare::fatal;
                for(int e:plan.kept) {
                    uint8_t* raw=next+(size_t)plan.slot[(size_t)e]*unit(l);
                    table[(size_t)e]=(unsigned long long)raw;
                    for(int r=0;r<3;++r) {
                        x3::Mat& m=next_vm[(size_t)e][(size_t)r];m=x3::Mat{};
                        m.K=gg::expert_type(l.lay,r);m.fmt=m.K==gg::kTypeMXFP4?gg::kFmtMXFP4:gg::kFmtGG;
                        m.raw=raw+l.lay.trellis[r];m.k=r<2?g.dim:g.ff;m.n=r<2?g.ff:g.dim;
                    }
                    next_host[(size_t)e]=nullptr;next_hh[(size_t)e]={nullptr,nullptr,nullptr,nullptr};
                }
                ce=cudaStreamSynchronize(d.st);
                if(ce==cudaSuccess)completed_d2d+=plan.next_bytes;
                return ce==cudaSuccess?grow::Prepare::ready:grow::Prepare::fatal;
            };
            ops.publish_table=[&] {
                ce=cudaMemcpyAsync(l.vram_tab,table.data(),table.size()*sizeof(table[0]),cudaMemcpyHostToDevice,d.st);
                if(ce!=cudaSuccess)return false;
                ce=cudaStreamSynchronize(d.st);return ce==cudaSuccess;
            };
            ops.publish_host=[&] {
                l.vram_base=next;l.slot.swap(next_slot);l.vm.swap(next_vm);l.host.swap(next_host);l.hh.swap(next_hh);l.ram.swap(next_ram);
                l.n_slots=plan.next_slots;l.n_vram=plan.resident;l.spare_vram=plan.next_spare;
                ram_held+=(int)plan.demoted.size();d.experts-=plan.old_bytes-plan.next_bytes;
            };
            ops.retire_old=[&] {
                auto it=std::find(d.allocs.begin(),d.allocs.end(),(void*)old);
                if(it==d.allocs.end()||cudaFree(old)!=cudaSuccess)return false;
                d.allocs.erase(it);return true;
            };
            grow::Result result;
            try {result=grow::replace(ops);}
            catch(...) {
                // A callback exception cannot justify releasing either slab:
                // table publication may already have begun. Both remain in
                // the existing owner ledger for terminal teardown.
                return hard("maintenance callback exception; tracked slabs retained");
            }
            if(result.state==grow::State::fatal)return hard("layer "+std::to_string(l.il)+": "+result.reason);
            if(result.state==grow::State::committed) {
                ++committed;demoted_count+=plan.demoted.size();demoted_bytes+=plan.old_bytes-plan.next_bytes;
                prefix_bytes+=plan.next_bytes;
                if(!inspect_copied_ram_inventory(inventory,err))return hard(err);
            } else ++skipped;
            if(result.stopped){stopped=true;break;}
        }
        // A partial smaller pool remains valid with the unchanged small Chunk.
        // No request IDs are cleared and no workspace or phase flag is changed.
        plan_streams();
        if(!inspect_copied_ram_inventory(inventory,err))return hard(err);
        } catch(...) {
            return hard("maintenance-loop exception; tracked ownership retained, ordinary work forbidden");
        }
        std::fprintf(stderr,"ds4: experimental HELD copied-pool shrink: epoch%u, layers%zu/%zu skipped%zu; "
                            "experts %.3f -> %.3f GB, committed demoted%zu/%zu bytes, committed prefix D2D%zu bytes; "
                            "completed D2H%zu D2D%zu bytes; "
                            "floor1024 MiB, free holes%zu; workspace/sequence/phase unchanged, %.3f s%s\n",
                     lease,committed,plans.size(),skipped,before/1e9,d.experts/1e9,demoted_count,demoted_bytes,
                     prefix_bytes,completed_d2h,completed_d2d,inventory.free_holes,(now_ms()-started)/1000.0,
                     stopped?"; stopped, valid partial pool retained":"");
        return true; // lease remains HELD; caller must end it before ordinary Model APIs
#endif
    }

    // UNUSED next prerequisite. Only the proven HELD epoch permits these
    // maintenance allocations; no caller, cache arena or phase activation.
    bool regrow_copied_chunk_buffers(unsigned lease,bool& stopped,bool& published,std::string& err) {
        stopped=false;published=false;
#if defined(STRATA_USE_HIP)
        (void)lease;err="chunk workspace regrow is CUDA-only";return false;
#else
        if((!cache_pool_shrink_probe_enabled&&!prefill_repeat_rearm)||!cache_epoch_enabled||!copied_ram_domains_published||
           !lease||cache_epoch_machine.phase!=epochpolicy::Phase::held||lease!=cache_epoch_machine.serial||
           cache_epoch_request.load(std::memory_order_acquire)!=lease||
           !cache_epoch_block_enqueues.load(std::memory_order_acquire)||
           cache_epoch_service_ack.load(std::memory_order_acquire)!=lease||
           cache_epoch_service_completed.load(std::memory_order_acquire)!=seq_issued||
           !chunk_major_phase_error.empty()||chunk_major_phase_pending||prefill_phase_pending||
           devs.size()!=1||chunks.size()!=1||!chunks[0]||NH!=1||!copy_mode||store||ram_tier||
           registered_copy.published||!registered_copy.held.empty()) {
            err="workspace regrow requires healthy unregistered HELD single-GPU copied-RAM epoch";return false;
        }
        auto hard=[&](const std::string& why){return cache_epoch_terminal("workspace regrow: "+why,err);};
        try {
        Dev& d=devs[0];Chunk& c=*chunks[0];
        if(c.C!=1024||chunk!=1024||c.A!=1024||c.AI!=128||c.restore_C!=65280||c.restore_A!=4096||c.restore_AI!=128||
           !c.flow||c.lm||c.lm_state||c.lc||c.prefill_gpu_h||c.lm_gpu||c.lm_f16||c.lm_f16_y||c.lm_f16_meta_dev||
           c.lm_f16_meta_host||c.lm_f16_blas||c.cold_f16||c.cold_f16_y||c.cold_f16_meta_host||c.cold_f16_meta_dev||
           c.cold_f16_workspace||c.cold_f16_blas||c.eg_weight_cache||c.dense_projection_cache||
           o.max_context!=1048576||g.ix_heads!=32||g.ix_hd!=128||g.cand_blk!=8||
           topk_stride()!=512||cand_stride()!=2048||!h_emb||!h_eg||eg_cap<(size_t)c.restore_C||
           (g.n_mtp>0&&(!h_taps||tap_cap<128))) {
            err="workspace regrow requires recorded65280/4096/128 and current1024/1024/128, retired phase arenas";return false;
        }
        for(const auto& w:workers)if(!w->idle()){err="workspace regrow requires idle prompt workers";return false;}
        {
            std::lock_guard<std::mutex> lock(sw_mu);
            if(cache_epoch_worker_acks!=cache_ths.size()||sw_inflight||ram_inflight||
               !sw_queue.empty()||!sw_ready.empty()||!ram_queue.empty()||!ram_ready.empty()) {
                err="workspace regrow requires acknowledged empty cache queues";return false;
            }
        }
        if(cudaSetDevice(d.id)!=cudaSuccess)return hard("device selection");
        if(cudaStreamQuery(d.st)!=cudaSuccess||cudaStreamQuery(c.cs)!=cudaSuccess)return hard("held streams not idle");
        for(size_t i=c.base;i<c.n_read;++i)if(fit(c,i).left.load(std::memory_order_acquire)!=0)return hard("flow read still owns payload");
        ramdomain::Inventory inventory;if(!inspect_copied_ram_inventory(inventory,err))return hard(err);
        const auto owns_device=[&](const void* p){return p&&std::count(d.allocs.begin(),d.allocs.end(),p)==1;};
        const auto owns_host=[&](const void* p){return p&&std::count(host_allocs.begin(),host_allocs.end(),p)==1;};
        // AI=128 was never shrunk. Its immutable recorded rows + original
        // one-time allocation formulas establish capacities; no Driver size or
        // GPU contents readback is claimed. Retain these exact owner pointers.
        if(!owns_device(c.iq)||!owns_device(c.scores)||!owns_device(c.bm)||!owns_device(c.w16)||
           c.w16_elems!=((size_t)8<<20)||!owns_device(c.bounds)||!owns_host(c.bounds_pin)||
           !owns_host(h_emb)||!owns_host(h_eg)||(g.n_mtp>0&&!owns_host(h_taps)))return hard("retained AI/global owners");
        if(c.topk_scr.states) {
            if(c.topk_scr.rows!=128||c.topk_scr.parts!=32||!owns_device(c.topk_scr.states)||
               !owns_device(c.topk_scr.hist)||!owns_device(c.topk_scr.counts)||!owns_device(c.topk_scr.offsets))
                return hard("retained AI TopK capacities/owners");
        } else if(c.topk_scr.hist||c.topk_scr.counts||c.topk_scr.offsets||c.topk_scr.rows||c.topk_scr.parts)
            return hard("partial retained AI TopK owner");
        const size_t retained_AI=(size_t)128*g.ix_heads*g.ix_hd*sizeof(float)+
            (size_t)128*((size_t)o.max_context+64)*sizeof(float)+(size_t)128*bm_stride()*sizeof(float);
        if(d.scratch<retained_AI)return hard("retained AI accounting");
        std::vector<shrink::Buffer> buffers;
        bool layout_ok=true;
        auto add=[&](const char* name,auto*& pointer,size_t before,size_t after,shrink::Kind kind=shrink::Kind::device) {
            const size_t item=sizeof(*pointer),max=std::numeric_limits<size_t>::max();
            if(!before||after<=before||before>max/item||after>max/item){layout_ok=false;return;}
            size_t old_bytes=before*item,new_bytes=after*item;
            const size_t suffix=kind==shrink::Kind::device?255:64;
            if(old_bytes>max-suffix||new_bytes>max-suffix){layout_ok=false;return;}
            if(kind==shrink::Kind::device){old_bytes=align256(old_bytes);new_bytes=align256(new_bytes);}
            else {old_bytes+=64;new_bytes+=64;}
            buffers.push_back({name,(void*)&pointer,(void*)pointer,old_bytes,new_bytes,kind,nullptr});
        };
        try {
            buffers.reserve(47);
            if(!chunkregrow::append_layout(c,g,pk,(size_t)c.C,(size_t)c.restore_C,c.A,c.restore_A,c.AI,c.restore_AI,
                                          topk_stride(),cand_stride(),add)||!layout_ok||buffers.size()!=47)
                return hard("unsupported original/current layout");
            d.allocs.reserve(d.allocs.size()+41);host_allocs.reserve(host_allocs.size()+6);
        } catch(...) {err="workspace regrow planning memory unavailable; small layout unchanged";return false;}
        size_t old_device=0,new_device=0,new_host=0;
        for(const auto& b:buffers) {
            if(!chunkregrow::add_size(b.kind==shrink::Kind::device?new_device:new_host,b.new_bytes)||
               (b.kind==shrink::Kind::device&&!chunkregrow::add_size(old_device,b.old_bytes)))return hard("allocation sum overflow");
        }
        if(d.scratch<old_device||d.scratch>std::numeric_limits<size_t>::max()-new_device)return hard("scratch accounting overflow/underflow");
        size_t remaining_device=new_device,remaining_host=new_host;
        bool hard_error=false,observed_stop=false;
        const auto budget=[&] {
            size_t free=0,total=0;if(cudaMemGetInfo(&free,&total)!=cudaSuccess||!total||free>total){hard_error=true;return false;}
            const uint64_t ram=ram_avail();const os::Commit cm=os::commit();
            return cm.avail!=~uint64_t{0}&&(double)cm.avail>=commit_margin()&&
                chunkregrow::budget_allows(free,ram,commit_room(),remaining_device,remaining_host);
        };
        if(!budget()) {
            if(hard_error)return hard("initial live budget query");
            err="workspace regrow needs all new device/host buffers beside live3GiB CUDA and RAM/commit margins";return false;
        }
        const double started=now_ms();
        shrink::Callbacks ops;
        ops.owns_old=[&](const shrink::Buffer& b){return b.kind==shrink::Kind::device?owns_device(b.old_pointer):owns_host(b.old_pointer);};
        ops.cancelled=[&]{if(cancel&&cancel())observed_stop=true;return observed_stop;};
        ops.allocate=[&](shrink::Buffer& b) {
            if(!budget())return false;
            cudaError_t status;
            if(b.kind==shrink::Kind::device) {
                status=cudaMalloc(&b.next,b.new_bytes);
                if(status==cudaSuccess){d.allocs.push_back(b.next);d.scratch+=b.new_bytes;remaining_device-=b.new_bytes;}
                if(status==cudaSuccess)status=cudaMemsetAsync(b.next,0,b.new_bytes,d.st);
                if(status==cudaSuccess)status=cudaStreamSynchronize(d.st);
            } else {
                const unsigned flags=cudaHostAllocPortable|(b.kind==shrink::Kind::mapped?cudaHostAllocMapped:0);
                status=cudaHostAlloc(&b.next,b.new_bytes,flags);
                if(status==cudaSuccess){host_allocs.push_back(b.next);remaining_host-=b.new_bytes;std::memset(b.next,0,b.new_bytes);}
            }
            if(status==cudaErrorMemoryAllocation)(void)cudaGetLastError();
            else if(status!=cudaSuccess)hard_error=true;
            return status==cudaSuccess&&budget();
        };
        ops.discard_new=[&](const shrink::Buffer& b) {
            if(cudaStreamSynchronize(d.st)!=cudaSuccess)return false;
            auto& owner=b.kind==shrink::Kind::device?d.allocs:host_allocs;
            auto it=std::find(owner.begin(),owner.end(),b.next);
            if(it==owner.end())return false;
            const cudaError_t status=b.kind==shrink::Kind::device?cudaFree(b.next):cudaFreeHost(b.next);
            if(status!=cudaSuccess)return false;
            owner.erase(it);if(b.kind==shrink::Kind::device)d.scratch-=b.new_bytes;return true;
        };
        ops.reserve_publication=[&](const std::vector<shrink::Buffer>& bs) {
            if(remaining_device||remaining_host||!budget())return false;
            for(const auto& b:bs) {
                const auto& owner=b.kind==shrink::Kind::device?d.allocs:host_allocs;
                if(std::count(owner.begin(),owner.end(),b.old_pointer)!=1||std::count(owner.begin(),owner.end(),b.next)!=1)return false;
            }
            return c.C==1024&&c.A==1024&&c.AI==128&&chunk==1024;
        };
        ops.publish=[&](const std::vector<shrink::Buffer>& bs) {
            for(const auto& b:bs)std::memcpy(b.slot,&b.next,sizeof b.next);
            chunk=c.C=c.restore_C;c.A=c.restore_A; // AI/index buffers remain128 and unmoved
        };
        ops.retire_old=[&](const shrink::Buffer& b) {
            auto& owner=b.kind==shrink::Kind::device?d.allocs:host_allocs;
            auto it=std::find(owner.begin(),owner.end(),b.old_pointer);
            if(it==owner.end())return false;
            const cudaError_t status=b.kind==shrink::Kind::device?cudaFree(b.old_pointer):cudaFreeHost(b.old_pointer);
            if(status!=cudaSuccess)return false;
            owner.erase(it);if(b.kind==shrink::Kind::device)d.scratch-=b.old_bytes;return true;
        };
        const chunkregrow::Result result=chunkregrow::replace(buffers,ops);
        stopped=result.stopped||observed_stop;published=result.published;
        if(hard_error||result.state==chunkregrow::State::fatal)return hard("allocation/initialization/ownership transaction: "+result.reason);
        if(!result.published) {
            std::fprintf(stderr,"ds4: experimental HELD workspace regrow kept small C=1024/A=1024/AI=128: %s%s\n",
                         result.reason.c_str(),stopped?"; stopped":"");return true;
        }
        std::fprintf(stderr,"ds4: experimental HELD workspace regrow committed47 buffers: C1024->%d A1024->%d AI128 retained; "
                            "GPU %.3f->%.3f GiB, host %.3f->%.3f GiB, floor3072MiB; no phase/cache/pin/sequence change, %.3f s\n",
                     c.C,c.A,result.device_old/1073741824.0,result.device_new/1073741824.0,
                     result.host_old/1073741824.0,result.host_new/1073741824.0,(now_ms()-started)/1000.0);
        std::fprintf(stderr,"ds4: workspace regrow exact committed totals: device_old=%zu device_new=%zu host_old=%zu host_new=%zu; "
                            "buffers=47 device=41 host=6 published=47 retired=47\n",
                     result.device_old,result.device_new,result.host_old,result.host_new);
        return true; // coherent HELD lease; caller must release before ordinary work
        } catch(...) {
            return hard("maintenance exception; tracked owners retained, ordinary work forbidden");
        }
#endif
    }

    struct CacheProbeState {
        int64_t position=0;size_t tokens=0,ctokens=0;uint64_t token_hash=0,checkpoint_hash=0,map_hash=0;
        unsigned issued=0;int64_t swaps=0;
    };
    CacheProbeState cache_probe_state() const {
        // Caller-owned token/checkpoint and authoritative map state ONLY. No
        // heat/prof/cpu_hits/service-written stats or device memory is read.
        auto word=[](uint64_t& h,uint64_t v){for(int i=0;i<8;++i){h^=(v>>(i*8))&255u;h*=1099511628211ull;}};
        auto bytes=[](uint64_t& h,const void* p,size_t n){const auto* b=(const uint8_t*)p;
            for(size_t i=0;i<n;++i){h^=b[i];h*=1099511628211ull;}};
        CacheProbeState s;s.position=pos;s.tokens=toks.size();s.ctokens=ctoks.size();s.issued=seq_issued;s.swaps=st.swaps;
        s.token_hash=s.checkpoint_hash=s.map_hash=1469598103934665603ull;
        word(s.token_hash,toks.size());for(int32_t v:toks)word(s.token_hash,(uint32_t)v);
        word(s.token_hash,ctoks.size());for(int32_t v:ctoks)word(s.token_hash,(uint32_t)v);
        word(s.checkpoint_hash,slot_held);
        for(const Check& c:ck) {
            word(s.checkpoint_hash,(uint64_t)c.pos);word(s.checkpoint_hash,c.win.size());word(s.checkpoint_hash,c.comp.size());
            for(const auto& v:c.win){word(s.checkpoint_hash,v.size());bytes(s.checkpoint_hash,v.data(),v.size());}
            for(const auto& v:c.comp){word(s.checkpoint_hash,v.size());bytes(s.checkpoint_hash,v.data(),v.size()*sizeof(float));}
        }
        for(const Layer& l:L) {
            word(s.map_hash,(uint64_t)l.il);word(s.map_hash,(uint64_t)l.n_slots);word(s.map_hash,(uint64_t)l.n_vram);
            word(s.map_hash,(uint64_t)(int64_t)l.spare_vram);
            word(s.map_hash,(uint64_t)(int64_t)copy_spare[(size_t)l.il]);
            for(int v:l.slot)word(s.map_hash,(uint64_t)(int64_t)v);
            for(int v:l.ram)word(s.map_hash,(uint64_t)(int64_t)v);
        }
        return s;
    }
    bool probe_cache_worker_epoch(int timeout_ms,std::string& diagnostic,unsigned& epoch,bool& terminal,std::string& err) {
        diagnostic.clear();epoch=0;terminal=false;
        if(!cache_epoch_probe_enabled) {err="explicit cache epoch probe is disabled";return false;}
        if(!copied_ram_domains_published||copy_spare.size()!=L.size()) {err="probe requires published RAM domains";return false;}
        const CacheProbeState before=cache_probe_state();
        int pending=0;size_t queued=0,ready=0;
        {std::lock_guard<std::mutex> lock(sw_mu);pending=sw_inflight;queued=sw_queue.size();ready=sw_ready.size();}
        const unsigned previous=cache_epoch_machine.serial;
        unsigned lease=0;
        const bool began=begin_cache_worker_epoch(lease,timeout_ms,err);
        epoch=cache_epoch_machine.serial>previous?cache_epoch_machine.serial:0;
        if(!began) {
            terminal=cache_epoch_machine.phase==epochpolicy::Phase::failed||!chunk_major_phase_error.empty();
            std::ostringstream o;o<<"{\"acquired\":false,\"terminal\":"<<(terminal?"true":"false")
                <<",\"epoch\":"<<epoch<<",\"issued_before\":"<<before.issued<<",\"issued_after\":"<<seq_issued
                <<",\"pending_before\":"<<pending<<",\"queued_before\":"<<queued<<",\"ready_before\":"<<ready
                <<",\"sequence_reset\":false,\"next_GEN_resume_proven\":false}";
            diagnostic=o.str();return false;
        }
        struct ReleaseLease {
            Impl* p;unsigned lease;bool& terminal;bool released=false;
            ~ReleaseLease(){if(released||p->cache_epoch_machine.phase!=epochpolicy::Phase::held)return;
                try{std::string why;if(!p->end_cache_worker_epoch(lease,why)){terminal=true;p->cache_epoch_terminal("probe lease cleanup failed",why);}}
                catch(...){terminal=true;p->chunk_major_phase_error="probe lease cleanup threw";p->cache_epoch_release(true);}}
        } release{this,lease,terminal};
        const CacheProbeState held=cache_probe_state();
        ramdomain::Inventory inventory;
        if(!inspect_copied_ram_inventory(inventory,err))return false; // lease RAII always resumes this soft refusal
        size_t acks=0;{std::lock_guard<std::mutex> lock(sw_mu);acks=cache_epoch_worker_acks;}
        const unsigned service_ack=cache_epoch_service_ack.load(std::memory_order_acquire);
        const unsigned completed=cache_epoch_service_completed.load(std::memory_order_acquire);
        size_t published_slots=0,response_slots=0;
        for(int i=0;i<kRing;++i){published_slots+=reqs[i].seq!=0;response_slots+=resps[i].seq!=0;}
        if(!end_cache_worker_epoch(lease,err)){terminal=true;cache_epoch_terminal("probe exact lease release failed",err);return false;}
        release.released=true;
        const CacheProbeState after=cache_probe_state();
        const bool no_reset=before.position==held.position&&held.position==after.position&&
            before.tokens==held.tokens&&held.tokens==after.tokens&&before.ctokens==held.ctokens&&held.ctokens==after.ctokens&&
            before.token_hash==held.token_hash&&held.token_hash==after.token_hash&&
            before.checkpoint_hash==held.checkpoint_hash&&held.checkpoint_hash==after.checkpoint_hash&&
            before.issued==held.issued&&held.issued==after.issued;
        const bool maps_stable=held.map_hash==after.map_hash;
        if(!no_reset||!maps_stable){terminal=true;cache_epoch_terminal("probe changed token/checkpoint/issued or held-to-released maps",err);return false;}
        std::ostringstream o;o<<"{\"acquired\":true,\"released\":true,\"terminal\":false,\"epoch\":"<<epoch
            <<",\"issued_before\":"<<before.issued<<",\"issued_held\":"<<held.issued<<",\"issued_after\":"<<after.issued
            <<",\"service_completed\":"<<completed<<",\"service_ack\":"<<service_ack
            <<",\"published_request_slots\":"<<published_slots<<",\"response_slots\":"<<response_slots
            <<",\"cache_worker_acks\":"<<acks<<",\"cache_workers\":"<<cache_ths.size()
            <<",\"pending_before\":"<<pending<<",\"queued_before\":"<<queued<<",\"ready_before\":"<<ready
            <<",\"committed_swaps\":"<<(after.swaps-before.swaps)
            <<",\"position\":"<<after.position<<",\"token_count\":"<<after.tokens<<",\"ctoken_count\":"<<after.ctokens
            <<",\"token_fnv64\":\""<<std::hex<<after.token_hash<<"\",\"checkpoint_fnv64\":\""<<after.checkpoint_hash
            <<"\",\"host_maps_before_fnv64\":\""<<before.map_hash<<"\",\"host_maps_held_fnv64\":\""<<held.map_hash
            <<"\",\"host_maps_released_fnv64\":\""<<after.map_hash<<std::dec<<"\""
            <<",\"CPU_copies\":"<<inventory.cpu_experts<<",\"GPU_experts\":"<<inventory.gpu_experts
            <<",\"free_holes\":"<<inventory.free_holes<<",\"spares\":"<<inventory.reserved_spares
            <<",\"maps_held_release_equal\":true,\"token_checkpoint_issued_unchanged\":true,\"sequence_reset\":false"
            <<",\"rearm_or_pool_workspace_changes\":false,\"next_GEN_resume_proven\":false}";
        diagnostic=o.str();return true;
    }

    // Explicit native diagnostic only. f071 shrink helper is byte-unchanged.
    bool probe_copied_pool_shrink(int timeout_ms,std::string& diagnostic,unsigned& epoch,
                                 bool& terminal,bool& full,std::string& err) {
        diagnostic.clear();epoch=0;terminal=false;full=false;
#if defined(STRATA_USE_HIP)
        (void)timeout_ms;err="pool-shrink probe is CUDA-only";return false;
#else
        if(!cache_pool_shrink_probe_enabled||!cache_epoch_probe_enabled||timeout_ms<1||timeout_ms>60000) {
            err="pool-shrink probe disabled or timeout outside1..60000";return false;
        }
        if(!copied_ram_domains_published||copy_spare.size()!=L.size()) {
            err="pool-shrink probe requires published RAM domains";return false;
        }
        const double started=now_ms(),deadline=started+timeout_ms;
        bool timed_out=false,client_stop=false;
        struct RestoreCancel {Impl* p;std::function<bool()> saved;~RestoreCancel(){p->cancel.swap(saved);}}
            restore_cancel{this,std::move(cancel)};
        cancel=[&] {
            if(now_ms()>=deadline){timed_out=true;return true;}
            if(restore_cancel.saved&&restore_cancel.saved()){client_stop=true;return true;}
            return false;
        };
        const CacheProbeState before=cache_probe_state();
        const unsigned previous=cache_epoch_machine.serial;unsigned lease=0;
        const bool began=begin_cache_worker_epoch(lease,timeout_ms,err);
        epoch=cache_epoch_machine.serial>previous?cache_epoch_machine.serial:0;
        if(!began) {
            terminal=cache_epoch_machine.phase==epochpolicy::Phase::failed||!chunk_major_phase_error.empty();
            std::ostringstream o;o<<"{\"acquired\":false,\"terminal\":"<<(terminal?"true":"false")
                <<",\"epoch\":"<<epoch<<",\"issued_before\":"<<before.issued<<",\"issued_after\":"<<seq_issued
                <<",\"target_attained\":false,\"full_success\":false,\"timed_out\":"<<(timed_out?"true":"false")
                <<",\"client_stop\":"<<(client_stop?"true":"false")<<",\"next_GEN_resume_proven\":false}";
            diagnostic=o.str();return false;
        }
        // Unexpected acquired-path exit is terminal; never release changed
        // ownership as healthy without the complete verification below.
        struct ReleaseLease {
            Impl* p;unsigned lease;bool& terminal;bool released=false;
            ~ReleaseLease(){if(released||p->cache_epoch_machine.phase!=epochpolicy::Phase::held)return;
                terminal=true;
                try{std::string why;p->cache_epoch_terminal("pool probe aborted while lease held",why);}
                catch(...){p->chunk_major_phase_error="pool probe cleanup threw";p->cache_epoch_release(true);}}
        } release{this,lease,terminal};
        auto hard=[&](const std::string& why){terminal=true;full=false;return cache_epoch_terminal("pool probe: "+why,err);};
        const CacheProbeState held=cache_probe_state();
        ramdomain::Inventory pre;if(!inspect_copied_ram_inventory(pre,err))return hard(err);
        const size_t pool_before=devs[0].experts,scratch_before=devs[0].scratch;
        const int C_before=chunks[0]->C,A_before=chunks[0]->A,chunk_before=chunk;
        size_t free_before=0,total=0;
        if(cudaMemGetInfo(&free_before,&total)!=cudaSuccess)return hard("initial headroom query");
        size_t acks=0;{std::lock_guard<std::mutex> lock(sw_mu);acks=cache_epoch_worker_acks;}
        const unsigned service_ack=cache_epoch_service_ack.load(std::memory_order_acquire);
        const unsigned completed=cache_epoch_service_completed.load(std::memory_order_acquire);
        bool stopped=false;const bool operation_ok=shrink_copied_expert_pool(lease,stopped,err);
        if(cache_epoch_machine.phase==epochpolicy::Phase::failed||!chunk_major_phase_error.empty()) {
            terminal=true;
            diagnostic="{\"acquired\":true,\"released\":false,\"terminal\":true,\"operation_ok\":false,\"full_success\":false}";
            return false;
        }
        stopped=stopped||cancel();
        ramdomain::Inventory post_inventory;
        if(!inspect_copied_ram_inventory(post_inventory,err))return hard(err);
        struct Target {int layer,before,after,target,experts,capacity;size_t unit_bytes,free_before,free_after;};
        std::vector<Target> targets;targets.reserve(copied_ram_domains.size());bool attained=true;
        for(const auto& domain:copied_ram_domains) {
            const Layer& l=L[(size_t)domain.layer];copiedshrink::Plan p;
            if(!copiedshrink::plan(domain,l.slot,l.n_slots,l.spare_vram,l.ram,copy_spare[(size_t)l.il],
                post_inventory.layers[(size_t)l.il].free_slots,unit(l),p,err))return hard(err);
            const int prior_slots=l.n_exp-(int)pre.layers[(size_t)l.il].occupied+1;
            targets.push_back({l.il,prior_slots,l.n_slots,p.next_slots,l.n_exp,(int)domain.slot_count,unit(l),
                pre.layers[(size_t)l.il].free_slots.size(),post_inventory.layers[(size_t)l.il].free_slots.size()});
            attained=attained&&l.n_slots==p.next_slots&&post_inventory.layers[(size_t)l.il].free_slots.empty();
        }
        attained=attained&&post_inventory.free_holes==0;
        // Independent table-address/Mat metadata checks. Dormant CPU Mat entries
        // can be stale in the original cache path, so check ONLY GPU residents.
        size_t accounted=0,table_entries=0;
        std::vector<std::pair<uintptr_t,size_t>> ranges;ranges.reserve(L.size());
        for(const Layer& l:L) {
            if(l.nm!=1||l.dev!=0||l.n_slots<0||l.n_slots>l.n_exp||!unit(l)||
               (size_t)l.n_slots>std::numeric_limits<size_t>::max()/unit(l)||
               l.slot.size()!=(size_t)l.n_exp||l.vm.size()!=(size_t)l.n_exp||!l.vram_tab)
                return hard("GPU pool geometry");
            const size_t bytes=(size_t)l.n_slots*unit(l);
            if(bytes) {
                if(!l.vram_base||std::count(devs[0].allocs.begin(),devs[0].allocs.end(),(void*)l.vram_base)!=1||
                   (uintptr_t)l.vram_base>std::numeric_limits<uintptr_t>::max()-bytes)return hard("GPU owner/span");
                ranges.emplace_back((uintptr_t)l.vram_base,bytes);
            }
            if(accounted>std::numeric_limits<size_t>::max()-bytes)return hard("pool accounting overflow");
            accounted+=bytes;std::vector<unsigned long long> table((size_t)l.n_exp);
            if(cudaMemcpy(table.data(),l.vram_tab,table.size()*sizeof(table[0]),cudaMemcpyDeviceToHost)!=cudaSuccess)
                return hard("GPU table readback");
            for(int e=0;e<l.n_exp;++e) {
                const int slot=l.slot[(size_t)e];const uintptr_t raw=slot>=0?(uintptr_t)(l.vram_base+(size_t)slot*unit(l)):0;
                if(table[(size_t)e]!=(unsigned long long)raw)return hard("GPU table/current slots mismatch");
                if(slot>=0)for(int r=0;r<3;++r) {
                    const x3::Mat& m=l.vm[(size_t)e][(size_t)r];const int type=gg::expert_type(l.lay,r);
                    if((uintptr_t)m.raw!=raw+l.lay.trellis[r]||m.K!=type||
                       m.fmt!=(type==gg::kTypeMXFP4?gg::kFmtMXFP4:gg::kFmtGG)||
                       m.k!=(r<2?g.dim:g.ff)||m.n!=(r<2?g.ff:g.dim))return hard("GPU Mat metadata mismatch");
                }
            }
            table_entries+=table.size();
        }
        std::sort(ranges.begin(),ranges.end());
        for(size_t i=1;i<ranges.size();++i)if(ranges[i-1].first+ranges[i-1].second>ranges[i].first)return hard("GPU slabs overlap");
        if(accounted!=devs[0].experts)return hard("GPU byte accounting mismatch");
        const CacheProbeState post=cache_probe_state();
        const auto same=[](const CacheProbeState& a,const CacheProbeState& b) {
            return a.position==b.position&&a.tokens==b.tokens&&a.ctokens==b.ctokens&&a.token_hash==b.token_hash&&
                a.checkpoint_hash==b.checkpoint_hash&&a.issued==b.issued;
        };
        const bool workspace_same=chunks[0]->C==C_before&&chunks[0]->A==A_before&&chunk==chunk_before&&
            devs[0].scratch==scratch_before&&!registered_copy.published&&registered_copy.held.empty();
        if(!same(before,held)||!same(held,post)||!workspace_same)return hard("token/checkpoint/issued/workspace changed");
        size_t free_after=0,total_after=0;
        if(cudaMemGetInfo(&free_after,&total_after)!=cudaSuccess||!total_after||free_after>total_after)return hard("final headroom query");
        const size_t floor=(size_t)1024<<20;const bool headroom_ok=free_after>=floor;
        full=operation_ok&&attained&&headroom_ok&&!stopped;
        if(!end_cache_worker_epoch(lease,err))return hard("exact lease release failed");
        release.released=true;const CacheProbeState after=cache_probe_state();
        if(!same(post,after)||post.map_hash!=after.map_hash)return hard("post-shrink state/maps changed on release");
        std::ostringstream o;o<<"{\"acquired\":true,\"released\":true,\"terminal\":false,\"epoch\":"<<epoch
            <<",\"operation_ok\":"<<(operation_ok?"true":"false")<<",\"target_attained\":"<<(attained?"true":"false")
            <<",\"headroom_ok\":"<<(headroom_ok?"true":"false")<<",\"full_success\":"<<(full?"true":"false")
            <<",\"coherent_partial\":"<<(operation_ok&&!full?"true":"false")<<",\"stopped\":"<<(stopped?"true":"false")
            <<",\"timed_out\":"<<(timed_out?"true":"false")<<",\"client_stop\":"<<(client_stop?"true":"false")
            <<",\"issued_before\":"<<before.issued<<",\"issued_held\":"<<held.issued<<",\"issued_post\":"<<post.issued
            <<",\"issued_after\":"<<after.issued<<",\"service_completed\":"<<completed<<",\"service_ack\":"<<service_ack
            <<",\"cache_worker_acks\":"<<acks<<",\"cache_workers\":"<<cache_ths.size()<<",\"committed_pending_swaps\":"<<(held.swaps-before.swaps)
            <<",\"pool_bytes_before\":"<<pool_before<<",\"pool_bytes_after\":"<<devs[0].experts
            <<",\"GPU_experts_before\":"<<pre.gpu_experts<<",\"GPU_experts_after\":"<<post_inventory.gpu_experts
            <<",\"CPU_copies_before\":"<<pre.cpu_experts<<",\"CPU_copies_after\":"<<post_inventory.cpu_experts
            <<",\"free_holes_before\":"<<pre.free_holes<<",\"free_holes_after\":"<<post_inventory.free_holes
            <<",\"CUDA_free_before\":"<<free_before<<",\"CUDA_free_after\":"<<free_after<<",\"CUDA_total\":"<<total_after
            <<",\"temporary_floor_bytes\":"<<floor<<",\"GPU_table_entries_checked\":"<<table_entries
            <<",\"position\":"<<after.position<<",\"token_count\":"<<after.tokens<<",\"ctoken_count\":"<<after.ctokens
            <<",\"token_fnv64\":\""<<std::hex<<after.token_hash<<"\",\"checkpoint_fnv64\":\""<<after.checkpoint_hash
            <<"\",\"host_maps_held_before_fnv64\":\""<<held.map_hash<<"\",\"host_maps_post_shrink_fnv64\":\""<<post.map_hash
            <<"\",\"host_maps_released_fnv64\":\""<<after.map_hash<<std::dec<<"\""
            <<",\"maps_post_release_equal\":true,\"token_checkpoint_issued_unchanged\":true,\"sequence_reset\":false"
            <<",\"workspace_or_registration_changes\":false,\"next_GEN_resume_proven\":false,\"targets\":[";
        for(size_t i=0;i<targets.size();++i){const auto& t=targets[i];if(i)o<<',';
            o<<"{\"layer\":"<<t.layer<<",\"before\":"<<t.before<<",\"after\":"<<t.after<<",\"target\":"<<t.target
             <<",\"expert_count\":"<<t.experts<<",\"domain_capacity\":"<<t.capacity<<",\"unit_bytes\":"<<t.unit_bytes
             <<",\"reserved_CPU_spare\":1,\"reserved_GPU_spare\":1,\"free_holes_before\":"<<t.free_before
             <<",\"free_holes_after\":"<<t.free_after<<'}';}
        o<<"],\"C_before\":"<<C_before<<",\"C_after\":"<<chunks[0]->C<<",\"A_before\":"<<A_before
         <<",\"A_after\":"<<chunks[0]->A<<",\"chunk_before\":"<<chunk_before<<",\"chunk_after\":"<<chunk
         <<",\"scratch_before\":"<<scratch_before<<",\"scratch_after\":"<<devs[0].scratch
         <<",\"registration_ranges_after\":"<<registered_copy.held.size();
        const auto state_json=[](std::ostream& out,const CacheProbeState& s) {
            out<<"{\"position\":"<<s.position<<",\"tokens\":"<<s.tokens<<",\"ctokens\":"<<s.ctokens<<",\"issued\":"<<s.issued
               <<",\"token_fnv64\":\""<<std::hex<<s.token_hash<<"\",\"checkpoint_fnv64\":\""<<s.checkpoint_hash<<std::dec<<"\"}";
        };
        o<<",\"state_before\":";state_json(o,before);o<<",\"state_held\":";state_json(o,held);
        o<<",\"state_post\":";state_json(o,post);o<<",\"state_after\":";state_json(o,after);
        o<<",\"elapsed_ms\":"<<(now_ms()-started)<<",\"timeout_scope\":\"cooperative epoch and maintenance; CUDA calls drain\"}";
        diagnostic=o.str();return operation_ok;
#endif
    }

    bool probe_copied_workspace_regrow(int timeout_ms,std::string& diagnostic,unsigned& epoch,bool& terminal,
                                      bool& operation_ok,bool& published,std::string& err) {
        diagnostic.clear();epoch=0;terminal=false;operation_ok=false;published=false;
#if defined(STRATA_USE_HIP)
        (void)timeout_ms;err="workspace-regrow probe is CUDA-only";return false;
#else
        if(!cache_workspace_regrow_probe_enabled||!cache_pool_shrink_probe_enabled||timeout_ms<1||timeout_ms>60000||
           !copied_ram_domains_published||copy_spare.size()!=L.size()||chunks.size()!=1||!chunks[0]||
           chunks[0]->C!=1024||chunks[0]->A!=1024||chunks[0]->AI!=128||
           chunks[0]->restore_C!=65280||chunks[0]->restore_A!=4096||chunks[0]->restore_AI!=128) {
            err="workspace probe disabled, unsupported recorded/current geometry, or timeout outside1..60000";return false;
        }
        const double started=now_ms(),deadline=started+timeout_ms;bool timed_out=false,client_stop=false;
        struct RestoreCancel {Impl* p;std::function<bool()> saved;~RestoreCancel(){p->cancel.swap(saved);}}
            restore_cancel{this,std::move(cancel)};
        cancel=[&]{if(now_ms()>=deadline){timed_out=true;return true;}
            if(restore_cancel.saved&&restore_cancel.saved()){client_stop=true;return true;}return false;};
        const CacheProbeState before=cache_probe_state();unsigned lease=0;
        const unsigned previous=cache_epoch_machine.serial;
        const bool began=begin_cache_worker_epoch(lease,timeout_ms,err);
        epoch=cache_epoch_machine.serial>previous?cache_epoch_machine.serial:0;
        if(!began) {
            terminal=cache_epoch_machine.phase==epochpolicy::Phase::failed||!chunk_major_phase_error.empty();
            std::ostringstream o;o<<"{\"acquired\":false,\"terminal\":"<<(terminal?"true":"false")<<",\"epoch\":"<<epoch
                <<",\"issued_before\":"<<before.issued<<",\"issued_after\":"<<seq_issued
                <<",\"timed_out\":"<<(timed_out?"true":"false")<<",\"client_stop\":"<<(client_stop?"true":"false")
                <<",\"published\":false,\"next_GEN_resume_proven\":false}";
            diagnostic=o.str();return false;
        }
        struct ReleaseLease {
            Impl* p;bool& terminal;bool released=false;
            ~ReleaseLease(){if(released||p->cache_epoch_machine.phase!=epochpolicy::Phase::held)return;
                terminal=true;try{std::string why;p->cache_epoch_terminal("workspace probe aborted while HELD",why);}
                catch(...){p->chunk_major_phase_error="workspace probe cleanup threw";p->cache_epoch_release(true);}}
        } release{this,terminal};
        auto hard=[&](const std::string& why){terminal=true;return cache_epoch_terminal("workspace probe: "+why,err);};
        try {
        Dev& d=devs[0];Chunk& c=*chunks[0];const CacheProbeState held=cache_probe_state();
        auto spec=[&] {
            std::vector<shrink::Buffer> out;out.reserve(47);bool good=true;
            auto add=[&](const char* name,auto*& pointer,size_t small,size_t large,shrink::Kind kind=shrink::Kind::device) {
                const size_t width=sizeof(*pointer),max=std::numeric_limits<size_t>::max();
                if(!small||small>max/width||large>max/width){good=false;return;}
                size_t a=small*width,b=large*width,suffix=kind==shrink::Kind::device?255:64;
                if(a>max-suffix||b>max-suffix){good=false;return;}
                if(kind==shrink::Kind::device){a=align256(a);b=align256(b);}else {a+=64;b+=64;}
                out.push_back({name,(void*)&pointer,(void*)pointer,a,b,kind,nullptr});
            };
            if(!chunkregrow::append_layout(c,g,pk,1024,65280,1024,4096,128,128,topk_stride(),cand_stride(),add)||!good||out.size()!=47)
                throw std::runtime_error("workspace diagnostic layout unavailable");
            return out;
        };
        auto own=[&](const shrink::Buffer& b,const void* ptr) {
            const auto& owner=b.kind==shrink::Kind::device?d.allocs:host_allocs;
            return ptr&&std::count(owner.begin(),owner.end(),ptr)==1;
        };
        auto validate_spec=[&](const std::vector<shrink::Buffer>& bs) {
            for(size_t i=0;i<bs.size();++i){if(!own(bs[i],bs[i].old_pointer))return false;
                for(size_t j=0;j<i;++j)if(bs[i].slot==bs[j].slot||bs[i].old_pointer==bs[j].old_pointer)return false;}return true;
        };
        // Hash tracked protected owner multisets and explicit retained bindings.
        // This attests addresses/metadata, NOT all GPU KV/weight contents.
        auto bindings=[&](const std::vector<shrink::Buffer>& payload) {
            std::array<uint64_t,6> out;out.fill(1469598103934665603ull);
            auto word=[](uint64_t& h,uint64_t x){for(int i=0;i<8;++i){h^=x&255u;h*=1099511628211ull;x>>=8;}};
            auto ptr=[&](uint64_t& h,const void* p){word(h,(uintptr_t)p);};
            auto owners=[&](const std::vector<void*>& ledger,shrink::Kind category,uint64_t& hash) {
                std::vector<uintptr_t> kept;
                for(void* p:ledger){bool excluded=false;for(const auto& b:payload)
                    excluded=excluded||p==b.old_pointer&&((category==shrink::Kind::device)==(b.kind==shrink::Kind::device));
                    if(!excluded)kept.push_back((uintptr_t)p);}
                std::sort(kept.begin(),kept.end());word(hash,kept.size());for(auto p:kept)word(hash,p);
            };
            owners(d.allocs,shrink::Kind::device,out[0]);owners(host_allocs,shrink::Kind::pinned,out[1]);
            word(out[2],c.AI);ptr(out[2],c.iq);ptr(out[2],c.scores);ptr(out[2],c.bm);ptr(out[2],c.w16);word(out[2],c.w16_elems);
            ptr(out[2],c.topk_scr.states);ptr(out[2],c.topk_scr.hist);ptr(out[2],c.topk_scr.counts);ptr(out[2],c.topk_scr.offsets);
            word(out[2],c.topk_scr.rows);word(out[2],c.topk_scr.parts);ptr(out[2],c.bounds);ptr(out[2],c.bounds_pin);
            ptr(out[3],h_emb);ptr(out[3],h_eg);ptr(out[3],h_taps);ptr(out[3],embed_pinned);ptr(out[3],h_x);ptr(out[3],h_w);ptr(out[3],h_part);
            word(out[3],eg_cap);word(out[3],tap_cap);word(out[3],c.restore_C);word(out[3],c.restore_A);word(out[3],c.restore_AI);
            ptr(out[3],c.ring);word(out[3],c.ring_stride);word(out[3],c.ring_n);ptr(out[3],c.stage);word(out[3],c.stage_stride);
            ptr(out[3],c.readers.get());ptr(out[3],c.io.get());ptr(out[3],c.cs);ptr(out[3],c.blas);ptr(out[3],c.hand_ack);word(out[3],c.arb_seq);
            for(int i=0;i<2;++i){ptr(out[3],c.hand_flag[i]);ptr(out[3],c.emb_ev[i]);ptr(out[3],c.eg_ev[i]);}
            word(out[3],workers.size());for(const auto& w:workers)ptr(out[3],w.get());
            word(out[3],o.max_context);word(out[3],chunk_major_phase_pending);word(out[3],prefill_phase_pending);
            word(out[3],registered_copy.published);word(out[3],registered_copy.held.size());
            ptr(out[3],c.cold_f16);ptr(out[3],c.eg_weight_cache);ptr(out[3],c.dense_projection_cache);
            word(out[4],d.experts);for(const Layer& l:L){word(out[4],l.il);ptr(out[4],l.vram_base);ptr(out[4],l.vram_tab);
                word(out[4],l.n_slots);word(out[4],l.n_vram);word(out[4],(uint64_t)(int64_t)l.spare_vram);
                for(int e=0;e<l.n_exp;++e)if(l.slot[(size_t)e]>=0)for(int r=0;r<3;++r)ptr(out[4],l.vm[(size_t)e][(size_t)r].raw);}
            word(out[5],ram_held);word(out[5],ram_chunks.size());
            for(size_t i=0;i<ram_chunks.size();++i){ptr(out[5],ram_chunks[i]);word(out[5],ram_chunk_bytes[i]);}
            for(const auto& p:ram_part){ptr(out[5],p[0]);ptr(out[5],p[1]);}
            for(const auto& row:cp_lay)for(size_t x:row)word(out[5],x);
            return out;
        };
        const auto old=spec();if(!validate_spec(old))return hard("old47 ownership/alias");
        const auto protected_before=bindings(old);const size_t scratch_before=d.scratch,pool_before=d.experts;
        const size_t device_owners_before=d.allocs.size(),host_owners_before=host_allocs.size();
        const int C_before=c.C,A_before=c.A,AI_before=c.AI,chunk_before=chunk;
        ramdomain::Inventory inv_before;if(!inspect_copied_ram_inventory(inv_before,err))return hard(err);
        size_t free_before=0,total=0;if(cudaMemGetInfo(&free_before,&total)!=cudaSuccess)return hard("initial headroom");
        size_t acks=0;{std::lock_guard<std::mutex> lock(sw_mu);acks=cache_epoch_worker_acks;}
        const unsigned service_ack=cache_epoch_service_ack.load(std::memory_order_acquire),completed=cache_epoch_service_completed.load(std::memory_order_acquire);
        bool stopped=false;operation_ok=regrow_copied_chunk_buffers(lease,stopped,published,err);
        if(cache_epoch_machine.phase==epochpolicy::Phase::failed||!chunk_major_phase_error.empty()) {
            terminal=true;diagnostic="{\"acquired\":true,\"released\":false,\"terminal\":true,\"verified\":false}";return false;
        }
        stopped=stopped||cancel();const auto current=spec();if(!validate_spec(current))return hard("current47 ownership/alias");
        size_t device_old=0,device_new=0,host_old=0,host_new=0,changed=0,retired=0,zero_bytes=0;
        auto sample_zero=[&](const shrink::Buffer& b) {
            std::array<uint8_t,64> bytes{};const size_t n=std::min<size_t>(bytes.size(),b.new_bytes);
            for(size_t off:{size_t(0),b.new_bytes-n}) {
                if(b.kind==shrink::Kind::device) {
                    if(cudaMemcpy(bytes.data(),(uint8_t*)b.old_pointer+off,n,cudaMemcpyDeviceToHost)!=cudaSuccess)return false;
                } else std::memcpy(bytes.data(),(uint8_t*)b.old_pointer+off,n);
                for(size_t i=0;i<n;++i)if(bytes[i])return false;zero_bytes+=n;
            }return true;
        };
        for(size_t i=0;i<old.size();++i) {
            const auto& a=old[i];const auto& b=current[i];
            if(a.slot!=b.slot||a.kind!=b.kind||a.old_bytes!=b.old_bytes||a.new_bytes!=b.new_bytes)return hard("layout binding changed");
            if(a.kind==shrink::Kind::device){device_old+=a.old_bytes;device_new+=a.new_bytes;}else {host_old+=a.old_bytes;host_new+=a.new_bytes;}
            const bool different=a.old_pointer!=b.old_pointer;changed+=different;
            const auto& owner=a.kind==shrink::Kind::device?d.allocs:host_allocs;
            const bool old_absent=std::count(owner.begin(),owner.end(),a.old_pointer)==0;retired+=old_absent;
            if(published?(!different||!old_absent):different)return hard("publication/old retirement mismatch");
            if(published&&!sample_zero(b))return hard("new47 zero edge sample failed");
        }
        const auto protected_after=bindings(current);ramdomain::Inventory inv_after;
        if(!inspect_copied_ram_inventory(inv_after,err))return hard(err);
        const CacheProbeState post=cache_probe_state();
        auto same=[](const CacheProbeState& a,const CacheProbeState& b){return a.position==b.position&&a.tokens==b.tokens&&a.ctokens==b.ctokens&&
            a.issued==b.issued&&a.token_hash==b.token_hash&&a.checkpoint_hash==b.checkpoint_hash;};
        const bool shape=published?c.C==65280&&c.A==4096&&c.AI==128&&chunk==65280:
                                   c.C==1024&&c.A==1024&&c.AI==128&&chunk==1024;
        const size_t expected_scratch=published?scratch_before+device_new-device_old:scratch_before;
        if(!shape||d.scratch!=expected_scratch||d.experts!=pool_before||device_owners_before!=d.allocs.size()||host_owners_before!=host_allocs.size()||
           protected_before!=protected_after||held.map_hash!=post.map_hash||!same(before,held)||!same(held,post)||
           inv_before.cpu_experts!=inv_after.cpu_experts||inv_before.gpu_experts!=inv_after.gpu_experts||inv_before.free_holes!=inv_after.free_holes)
            return hard("protected state/shape/accounting changed");
        size_t free_after=0,total_after=0;
        if(cudaMemGetInfo(&free_after,&total_after)!=cudaSuccess||!total_after||free_after>total_after)return hard("final headroom");
        const bool headroom_ok=free_after>=((size_t)3072<<20);
        if(published&&!headroom_ok)return hard("published workspace below3GiB live floor");
        if(!end_cache_worker_epoch(lease,err))return hard("exact lease release");
        release.released=true;const CacheProbeState after=cache_probe_state();const auto protected_released=bindings(current);
        if(!same(post,after)||post.map_hash!=after.map_hash||protected_after!=protected_released)return hard("release changed protected state");
        std::ostringstream o;o<<"{\"acquired\":true,\"released\":true,\"terminal\":false,\"verified\":true,\"epoch\":"<<epoch
            <<",\"operation_ok\":"<<(operation_ok?"true":"false")<<",\"published\":"<<(published?"true":"false")
            <<",\"kept_small\":"<<(!published?"true":"false")<<",\"stopped\":"<<(stopped?"true":"false")
            <<",\"timed_out\":"<<(timed_out?"true":"false")<<",\"client_stop\":"<<(client_stop?"true":"false")
            <<",\"buffer_count\":47,\"device_buffers\":41,\"host_buffers\":6,\"changed_buffers\":"<<changed<<",\"retired_old_buffers\":"<<retired
            <<",\"published_initialized_buffers\":"<<(published?47:0)<<",\"zero_edge_sample_bytes\":"<<zero_bytes
            <<",\"zero_edge_samples_checked\":"<<(published?"true":"false")
            <<",\"device_old\":"<<device_old<<",\"device_new\":"<<device_new<<",\"host_old\":"<<host_old<<",\"host_new\":"<<host_new
            <<",\"C_before\":"<<C_before<<",\"C_after\":"<<c.C<<",\"A_before\":"<<A_before<<",\"A_after\":"<<c.A
            <<",\"AI_before\":"<<AI_before<<",\"AI_after\":"<<c.AI<<",\"chunk_before\":"<<chunk_before<<",\"chunk_after\":"<<chunk
            <<",\"scratch_before\":"<<scratch_before<<",\"scratch_after\":"<<d.scratch<<",\"pool_before\":"<<pool_before<<",\"pool_after\":"<<d.experts
            <<",\"device_owners_before\":"<<device_owners_before<<",\"device_owners_after\":"<<d.allocs.size()
            <<",\"host_owners_before\":"<<host_owners_before<<",\"host_owners_after\":"<<host_allocs.size()
            <<",\"CUDA_free_before\":"<<free_before<<",\"CUDA_free_after\":"<<free_after<<",\"CUDA_total\":"<<total_after
            <<",\"temporary_floor_bytes\":3221225472,\"headroom_ok\":"<<(headroom_ok?"true":"false")
            <<",\"service_completed\":"<<completed<<",\"service_ack\":"<<service_ack<<",\"cache_worker_acks\":"<<acks<<",\"cache_workers\":"<<cache_ths.size()
            <<",\"CPU_copies_before\":"<<inv_before.cpu_experts<<",\"CPU_copies_after\":"<<inv_after.cpu_experts
            <<",\"GPU_experts_before\":"<<inv_before.gpu_experts<<",\"GPU_experts_after\":"<<inv_after.gpu_experts
            <<",\"free_holes_before\":"<<inv_before.free_holes<<",\"free_holes_after\":"<<inv_after.free_holes
            <<",\"host_maps_held\":\""<<std::hex<<held.map_hash<<"\",\"host_maps_post\":\""<<post.map_hash<<"\",\"host_maps_released\":\""<<after.map_hash<<std::dec<<"\""
            <<",\"protected_bindings\":[";
        for(size_t i=0;i<protected_before.size();++i){if(i)o<<',';o<<"{\"before\":\""<<std::hex<<protected_before[i]<<"\",\"post\":\""<<protected_after[i]
            <<"\",\"released\":\""<<protected_released[i]<<std::dec<<"\"}";}
        auto state_json=[](std::ostream& out,const CacheProbeState& s){out<<"{\"position\":"<<s.position<<",\"tokens\":"<<s.tokens<<",\"ctokens\":"<<s.ctokens
            <<",\"issued\":"<<s.issued<<",\"token_fnv64\":\""<<std::hex<<s.token_hash<<"\",\"checkpoint_fnv64\":\""<<s.checkpoint_hash<<std::dec<<"\"}";};
        o<<"],\"state_before\":";state_json(o,before);o<<",\"state_held\":";state_json(o,held);o<<",\"state_post\":";state_json(o,post);o<<",\"state_after\":";state_json(o,after);
        o<<",\"buffers\":[";for(size_t i=0;i<old.size();++i){if(i)o<<',';const auto& b=old[i];
            o<<"{\"ordinal\":"<<i<<",\"name\":"<<probeproto::quote(b.name)<<",\"kind\":"<<(int)b.kind<<",\"old_bytes\":"<<b.old_bytes
             <<",\"new_bytes\":"<<b.new_bytes<<",\"changed\":"<<(b.old_pointer!=current[i].old_pointer?"true":"false")<<",\"owned_before\":true,\"owned_after\":true}";}
        o<<"],\"sequence_reset\":false,\"pool_AI_globals_unchanged\":true,\"registration_ranges_after\":"<<registered_copy.held.size()
         <<",\"phase_or_cache_rearm\":false,\"next_GEN_resume_proven\":false,\"elapsed_ms\":"<<(now_ms()-started)
         <<",\"init_scope\":\"primitive checked full initialization plus bounded first/last64-byte samples; not full-content attestation\"}";
        diagnostic=o.str();return true;
        } catch(...) {
            return hard("acquired-path exception, including post-release reporting; ordinary work forbidden");
        }
#endif
    }

    void cache_loop(SwapBufs& B) {
        unsigned last_epoch_ack=0;
        for (;;) {
            Swap sw;
            {
                std::unique_lock<std::mutex> lk(sw_mu);
                if(cache_epoch_enabled) {if(!cache_epoch_wait_worker(lk,last_epoch_ack))return;}
                else {
                sw_cv.wait(lk, [&] { return quit.load() || !sw_queue.empty(); });
                }
                if (quit.load()) return;
                sw = sw_queue.front();
                sw_queue.pop_front();
            }
            Layer& l = L[(size_t) sw.layer];
            if (l.lay.fmt == gg::kFmtMXFP4) {   // in: from the file; out: nothing to keep (the file has it)
                swap_in_mx(l, sw, B);
                std::lock_guard<std::mutex> lk(sw_mu);
                sw_ready.push_back(sw);
                continue;
            }
            parts_to_blob(l, l.hh[(size_t) sw.in], B.a);
            if (l.nm == 1) {
                const Dev& d = devs[(size_t) l.dev];
                cudaSetDevice(d.id);
                cudaStream_t cs = B.st[(size_t) l.dev];
                cudaMemcpyAsync(l.vram_base + (size_t) l.spare_vram * l.blob, B.a, l.blob, cudaMemcpyHostToDevice, cs);
                if (!file_experts)
                    cudaMemcpyAsync(B.b, l.vram_base + (size_t) l.slot[(size_t) sw.out] * l.blob, l.blob,
                                    cudaMemcpyDeviceToHost, cs);
                cudaStreamSynchronize(cs);
            } else {   // a group: each member's part in, each member's part out
                for (int m = 0; m < l.nm; ++m) {
                    const Layer& x = member(l, m);
                    cudaSetDevice(devs[(size_t) x.dev].id);
                    cudaStream_t cs = B.st[(size_t) x.dev];
                    blob_to_half(l, B.a, m, B.h);
                    cudaMemcpyAsync(x.vram_base + (size_t) l.spare_vram * x.hblob, B.h, x.hblob, cudaMemcpyHostToDevice, cs);
                    cudaStreamSynchronize(cs);
                    if (file_experts) continue;
                    cudaMemcpyAsync(B.h, x.vram_base + (size_t) l.slot[(size_t) sw.out] * x.hblob, x.hblob,
                                    cudaMemcpyDeviceToHost, cs);
                    cudaStreamSynchronize(cs);
                    half_to_blob(l, B.h, m, B.b);
                }
            }
            if (!file_experts)
                for (int X = 0; X < NH; ++X) blob_to_part(l, B.b, X, l.spare_host[(size_t) X]);
            if (check_swaps) {   // DS4_CHECK_SWAPS: both moves against the model file, byte for byte
                std::vector<uint8_t> reg, blob(l.blob), part(l.hl.bytes);
                bool ok = true;
                for (int pass = 0; pass < (file_experts ? 1 : 2); ++pass) {   // (in place: no host copy of `out`)
                    const int e = pass == 0 ? sw.in : sw.out;
                    reg.resize(l.region_bytes[(size_t) e]);
                    pk.read_at(l.region_file[(size_t) e], l.region_off[(size_t) e], reg.size(), reg.data());
                    if (pass == 0) {
                        repack(l, reg.data(), blob.data());
                        std::vector<uint8_t> dev(l.blob);
                        if (l.nm == 1)
                            cudaMemcpy(dev.data(), l.vram_base + (size_t) l.spare_vram * l.blob, l.blob, cudaMemcpyDeviceToHost);
                        else
                            for (int m = 0; m < l.nm; ++m) {   // (the members' parts, put together again)
                                std::vector<uint8_t> h(member(l, m).hblob);
                                cudaSetDevice(devs[(size_t) member(l, m).dev].id);
                                cudaMemcpy(h.data(), member(l, m).vram_base + (size_t) l.spare_vram * member(l, m).hblob, member(l, m).hblob,
                                           cudaMemcpyDeviceToHost);
                                half_to_blob(l, h.data(), m, dev.data());
                            }
                        for (int r = 0; r < 3; ++r)
                            ok = ok && std::memcmp(dev.data() + l.lay.trellis[r], blob.data() + l.lay.trellis[r], l.piece_bytes[r][0]) == 0 &&
                                 std::memcmp(dev.data() + l.lay.suh[r], blob.data() + l.lay.suh[r], l.piece_bytes[r][1]) == 0 &&
                                 std::memcmp(dev.data() + l.lay.svh[r], blob.data() + l.lay.svh[r], l.piece_bytes[r][2]) == 0;
                    } else {
                        for (int X = 0; X < NH; ++X) {
                            repack_part(l, reg.data(), X, part.data());
                            for (int r = 0; r < 3; ++r) {
                                const size_t tb = (size_t) (l.hl.k[r] / 16) * (l.hl.n[r] / 16) * 32 * l.lay.K[r];
                                ok = ok && std::memcmp(part.data() + l.hl.trellis[r], l.spare_host[(size_t) X] + l.hl.trellis[r], tb) == 0 &&
                                     std::memcmp(part.data() + l.hl.suh[r], l.spare_host[(size_t) X] + l.hl.suh[r], (size_t) l.hl.k[r] * 2) == 0 &&
                                     std::memcmp(part.data() + l.hl.svh[r], l.spare_host[(size_t) X] + l.hl.svh[r], (size_t) l.hl.n[r] * 2) == 0;
                            }
                        }
                    }
                }
                std::fprintf(stderr, "ds4 cache: layer %d expert %d in, %d out: %s\n", l.il, sw.in, sw.out,
                             ok ? "both match the model file" : "MISMATCH");
            }
            std::lock_guard<std::mutex> lk(sw_mu);
            sw_ready.push_back(sw);
        }
    }

    // a GGUF's swap: expert sw.in from the file (copy_mode: its copy) into the spare VRAM slot (each member its part);
    // the expert it replaces goes back to the CPU lane, which reads it from the file's mapping - or (copy_mode, sw.ram)
    // from the layer's spare RAM slot, which it is copied into from VRAM here
    void swap_in_mx(Layer& l, Swap& sw, SwapBufs& B) {
        bool epoch_ok=!copy_mode||sw.ram>=0;
        uint8_t *sw_a = B.a, *sw_b = B.b, *sw_h = B.h;   // (this thread's)
        const std::vector<cudaStream_t>& copy_st = B.st;
        bool ok = copy_mode ? read_mx(l, sw.in, sw_a) : read_mx_file(l, sw.in, sw_a);   // (a copy holds its slot till
                                                                                          // this swap's switch)
        for (int m = 0; m < l.nm && ok; ++m) {
            const Layer& x = member(l, m);
            if(cache_epoch_enabled) {if(cudaSetDevice(devs[(size_t) x.dev].id)!=cudaSuccess)epoch_ok=false;}
            else cudaSetDevice(devs[(size_t) x.dev].id);
            cudaStream_t cs = copy_st[(size_t) x.dev];
            if (l.nm > 1) blob_to_half(l, sw_a, m, sw_h);
            ok = cudaMemcpyAsync(x.vram_base + (size_t) l.spare_vram * unit(l), l.nm > 1 ? sw_h : sw_a, unit(l),
                                 cudaMemcpyHostToDevice, cs) == cudaSuccess &&
                 cudaStreamSynchronize(cs) == cudaSuccess;
        }
        if (sw.ram >= 0) {   // `out` from its VRAM slot (each member its part) into the spare RAM slot
            bool got = ok;
            for (int m = 0; m < l.nm && got; ++m) {
                const Layer& x = member(l, m);
                if(cache_epoch_enabled) {if(cudaSetDevice(devs[(size_t) x.dev].id)!=cudaSuccess)epoch_ok=false;}
            else cudaSetDevice(devs[(size_t) x.dev].id);
                cudaStream_t cs = copy_st[(size_t) x.dev];
                got = cudaMemcpyAsync(l.nm > 1 ? sw_h : sw_b, x.vram_base + (size_t) l.slot[(size_t) sw.out] * unit(l),
                                      unit(l), cudaMemcpyDeviceToHost, cs) == cudaSuccess &&
                      cudaStreamSynchronize(cs) == cudaSuccess;
                if (got && l.nm > 1) half_to_blob(l, sw_h, m, sw_b);
            }
            if(cache_epoch_enabled)epoch_ok=epoch_ok&&got;
            if (got) blob_to_ram(l, sw_b, sw.ram);
            else sw.ram = -1;   // (back to the file's pages then)
        }
        if (check_swaps) {   // DS4_CHECK_SWAPS: the slot (and `out`'s RAM copy) against the model file, byte for byte
            std::vector<uint8_t> dev(l.blob), h(unit(l)), file(l.blob);
            for (int m = 0; m < l.nm && ok; ++m) {
                const Layer& x = member(l, m);
                if(cache_epoch_enabled) {if(cudaSetDevice(devs[(size_t) x.dev].id)!=cudaSuccess)epoch_ok=false;}
            else cudaSetDevice(devs[(size_t) x.dev].id);
                if(cache_epoch_enabled) {
                const cudaError_t epoch_copy=cudaMemcpy(l.nm > 1 ? h.data() : dev.data(), x.vram_base + (size_t) l.spare_vram * unit(l), unit(l),
                           cudaMemcpyDeviceToHost);
                if(epoch_copy!=cudaSuccess)epoch_ok=false;
            } else {
            cudaMemcpy(l.nm > 1 ? h.data() : dev.data(), x.vram_base + (size_t) l.spare_vram * unit(l), unit(l),
                           cudaMemcpyDeviceToHost);
            }
                if (l.nm > 1) half_to_blob(l, h.data(), m, dev.data());
            }
            ok = ok && read_mx_file(l, sw.in, file.data());
            for (int r = 0; r < 3 && ok; ++r)
                ok = std::memcmp(dev.data() + l.lay.trellis[r], file.data() + l.lay.trellis[r], mx_part_bytes(l, r)) == 0;
            bool ok_out = true;
            if (sw.ram >= 0) {
                ok_out = read_mx_file(l, sw.out, file.data());
                ram_to_blob(l, sw.ram, dev.data());
                for (int r = 0; r < 3 && ok_out; ++r)
                    ok_out = std::memcmp(dev.data() + l.lay.trellis[r], file.data() + l.lay.trellis[r], mx_part_bytes(l, r)) == 0;
            }
            if(cache_epoch_enabled)epoch_ok=epoch_ok&&ok_out;
            std::fprintf(stderr, "ds4 cache: layer %d expert %d in, %d out%s: %s\n", l.il, sw.in, sw.out,
                         sw.ram >= 0 ? " (into RAM)" : "", ok && ok_out ? "matches the model file" : "MISMATCH");
        } else if (!ok) {
            std::fprintf(stderr, "ds4 cache: layer %d expert %d: the move into VRAM failed\n", l.il, sw.in);
        }
        if(cache_epoch_enabled)sw.epoch_prepared_ok=ok&&epoch_ok;
    }

    // after a chunked prompt: the expert cache catches up with what the prompt picked at once - a window prompt gets
    // its swaps one tick at a time over its hundreds of windows, a chunked one has a single tick - so the decode that
    // follows finds the hot experts in VRAM.  Rounds of the ordinary swap rule (the hottest CPU-lane pick of each layer
    // in, its coldest VRAM expert out) until nothing is worth moving or DS4_SETTLE_MS (6000) is spent
    void cache_settle() {
        if (!o.expert_cache || cache_ths.empty()) return;
        static const double budget = env_f("DS4_SETTLE_MS", 6000.0f);
        if (budget <= 0.0) return;
        const double t0 = now_ms();
        const int64_t s0 = st.swaps;
        while (now_ms() - t0 < budget) {
            for (Layer& l : L) {
                if (l.mtp || l.spare_vram < 0 || l.swap_pending) continue;
                l.cpu_hits.clear();
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0 && l.heat[(size_t) e] >= 2.0f) l.cpu_hits.push_back(e);
            }
            cache_tick_swaps(0);
            if (sw_inflight == 0) break;   // (nothing queued: every layer's best CPU-lane expert is not hotter than its coldest VRAM one)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (store && env_f("DS4_SETTLE_STORE", 1.0f) != 0.0f) {   // the RAM store too: a window prompt fills it through the CPU
            // lane's picks, a chunked one streams past it - the hottest CPU-lane experts of the prompt into its free slots
            std::vector<std::pair<float, std::pair<int, int>>> hot;
            for (const Layer& l : L) {
                if (l.mtp) continue;
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0 && l.heat[(size_t) e] >= 4.0f) hot.push_back({l.heat[(size_t) e], {l.il, e}});
            }
            std::sort(hot.begin(), hot.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            std::vector<std::pair<int, int>> list;
            for (const auto& h : hot) {
                if (list.size() >= store->slots_max()) break;
                list.push_back(h.second);
            }
            const double ts = now_ms();
            const size_t n = store->preload(list, 0);
            if (o.verbose) std::fprintf(stderr, "ds4: RAM store: %zu hot experts read after the prompt (%.0f ms)\n", n, now_ms() - ts);
        }
        if (o.verbose && st.swaps > s0)
            std::fprintf(stderr, "ds4: expert cache: %lld swaps after the prompt (%.0f ms)\n", (long long) (st.swaps - s0),
                         now_ms() - t0);
    }
    void cache_tick(int T) {
        cache_tick_swaps(T);
        store_sync();
        store_watch();
    }
    // the store gives RAM back when the PC runs short (the owner opened something): below 4 GB free, a 0.7 GB chunk at a time
    // until 5 GB are free again (checked every 16 windows); it grows again by itself once 8 GB beside its chunk are free
    void store_watch() {
        if (!store) return;
        static int tick = 0;
        if (++tick % 16 != 0) return;
        const uint64_t avail = ram_avail();
        if (avail == ~uint64_t{0} || avail >= (4ull << 30)) return;
        const size_t chunks = 1 + (size_t) (((5ull << 30) - avail) / (700ull << 20));
        const size_t n = store->shrink_chunks(chunks);
        if (n > 0) std::fprintf(stderr, "ds4: RAM store: %.1f GB free: gave back %zu chunks (%zu slots left)\n", avail / 1e9, n, store->slots_allocated());
    }
    void cache_tick_swaps(int T) {
        if (!o.expert_cache || cache_ths.empty()) return;
        std::vector<Swap> ready;
        std::vector<RamSwap> rready;
        {
            std::lock_guard<std::mutex> lk(sw_mu);
            ready.swap(sw_ready);
            rready.swap(ram_ready);
        }
        for (const RamSwap& rs : rready) {   // (nothing reads the tier between windows)
            --ram_inflight;
            Layer& l = L[(size_t) rs.layer];
            if (rs.in < 0 || l.ram[(size_t) rs.in] >= 0 || l.slot[(size_t) rs.in] >= 0) {
                ram_free.push_back(rs.slot);
                continue;
            }
            l.ram[(size_t) rs.in] = rs.slot;
            if (rs.out >= 0 && l.ram[(size_t) rs.out] >= 0) {
                ram_free.push_back(l.ram[(size_t) rs.out]);
                l.ram[(size_t) rs.out] = -1;
            }
            ++st.ram_moves;
        }
        for (const Swap& sw : ready) {
            if(cache_epoch_enabled&&!sw.epoch_prepared_ok) {
                std::string why;cache_epoch_terminal("cache swap preparation failed before ordinary commit",why);return;
            }
            Layer& l = L[(size_t) sw.layer];
            const int old_slot = l.slot[(size_t) sw.out];
            const std::array<uint8_t*, 4> old_parts = l.hh[(size_t) sw.in];
            if (copy_mode) {   // (`out` takes the RAM place `in` leaves: through the spare, which `in`'s slot becomes)
                const int s_in = l.ram[(size_t) sw.in];
                l.ram[(size_t) sw.in] = -1;
                if (sw.ram >= 0) {
                    l.ram[(size_t) sw.out] = sw.ram;
                    copy_spare[(size_t) l.il] = s_in;
                } else if (s_in >= 0) {   // (the move into RAM failed: `out` back to the file, one copy fewer)
                    --ram_held;
                    if (copy_spare[(size_t) l.il] < 0) copy_spare[(size_t) l.il] = s_in;
                }
            } else if (!l.ram.empty() && l.ram[(size_t) sw.in] >= 0) {   // (in VRAM now: its RAM slot for the next hottest)
                ram_free.push_back(l.ram[(size_t) sw.in]);
                l.ram[(size_t) sw.in] = -1;
            }
            l.slot[(size_t) sw.in] = l.spare_vram;
            if (!l.zc.empty() && l.zc[(size_t) sw.in]) {   // (VRAM's now: not the store's to compute any more)
                l.zc[(size_t) sw.in] = 0;
                if (l.zc_tab != nullptr) {   // (the device's table too: its entry pointed into the slot the store now reuses - when `in` left VRAM later,
                                             // zc_stage copied whatever lay there, another expert's blob, into the plan)
                    cudaSetDevice(devs[0].id);
                    cudaMemsetAsync(l.zc_tab + sw.in, 0, 8, devs[0].st);
                }
                store->unmark(l.il, sw.in);
            }
            if (l.nm == 1) set_vm(l, sw.in, l.spare_vram);
            l.hh[(size_t) sw.in] = {nullptr, nullptr, nullptr, nullptr};
            l.host[(size_t) sw.in] = nullptr;
            l.slot[(size_t) sw.out] = -1;
            if (l.lay.fmt == gg::kFmtMXFP4) {   // (the file has it)
                l.host[(size_t) sw.out] = (uint8_t*) mx_mat(l, sw.out, 0);
            } else if (file_experts) {   // (back to its pages in the file)
                l.hh[(size_t) sw.out] = {file_part(l, sw.out), nullptr, nullptr, nullptr};
                l.host[(size_t) sw.out] = l.hh[(size_t) sw.out][0];
            } else {
                l.hh[(size_t) sw.out] = l.spare_host;
                l.host[(size_t) sw.out] = l.spare_host[0];
                l.spare_host = old_parts;
            }
            l.spare_vram = old_slot;
            for (int m = 0; m < l.nm; ++m) {
                Layer& x = member(l, m);
                const Dev& dm = devs[(size_t) x.dev];
                l.tab_pin[2 * m] = (unsigned long long) (x.vram_base + (size_t) l.slot[(size_t) sw.in] * unit(x));
                l.tab_pin[2 * m + 1] = 0ull;
                cudaSetDevice(dm.id);   // (stream order: before the next window)
                cudaMemcpyAsync(x.vram_tab + sw.in, &l.tab_pin[2 * m], 8, cudaMemcpyHostToDevice, dm.st);
                cudaMemcpyAsync(x.vram_tab + sw.out, &l.tab_pin[2 * m + 1], 8, cudaMemcpyHostToDevice, dm.st);
            }
            l.swap_pending = false;
            --sw_inflight;
            ++st.swaps;
        }
        decay_tokens += T;
        if (decay_tokens >= 256) {   // older use counts less: x0.9 every 256 tokens
            decay_tokens = 0;
            for (Layer& l : L)
                for (float& h : l.heat) h *= c_decay;
        }
        if(cache_epoch_enabled&&cache_epoch_block_enqueues.load(std::memory_order_acquire))return;
        bool queued = false, rqueued = false;
        for (Layer& l : L) {
            if (l.cpu_hits.empty()) continue;
            std::vector<int> hits;
            hits.swap(l.cpu_hits);
            int best = -1, vin = -1;
            for (int e : hits)
                if (l.slot[(size_t) e] < 0 && (best < 0 || l.heat[(size_t) e] > l.heat[(size_t) best])) best = e;
            if (best >= 0 && l.spare_vram >= 0 && !l.swap_pending && sw_inflight < c_inflight) {   // VRAM: the hottest in
                int cold = -1;
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] >= 0 && (cold < 0 || l.heat[(size_t) e] < l.heat[(size_t) cold])) cold = e;
                if (cold >= 0 && l.heat[(size_t) best] > l.heat[(size_t) cold] * c_margin + c_bonus) {
                    l.swap_pending = true;
                    ++sw_inflight;
                    vin = best;
                    Swap sw{l.il, best, cold};
                    if (copy_mode && l.ram[(size_t) best] >= 0) sw.ram = copy_spare[(size_t) l.il];   // (-1: none)
                    std::lock_guard<std::mutex> lk(sw_mu);
                    sw_queue.push_back(sw);
                    queued = true;
                }
            }
            if (l.ram.empty() || ram_free.empty() || ram_inflight >= kMaxSwaps) continue;
            int rb = -1;   // RAM: the hottest read from the file, in place of the layer's coldest held (or a free slot)
            for (int e : hits)
                if (e != vin && l.slot[(size_t) e] < 0 && l.ram[(size_t) e] < 0 &&
                    (rb < 0 || l.heat[(size_t) e] > l.heat[(size_t) rb]))
                    rb = e;
            if (rb < 0) continue;
            int rc = -1;
            for (int e = 0; e < l.n_exp; ++e)
                if (l.ram[(size_t) e] >= 0 && (rc < 0 || l.heat[(size_t) e] < l.heat[(size_t) rc])) rc = e;
            const bool room = ram_free.size() > 1;   // (slots VRAM freed: filled at once)
            if (!room && !(rc >= 0 && l.heat[(size_t) rb] > l.heat[(size_t) rc] * 1.25f + 1.0f)) continue;
            RamSwap rs;
            rs.layer = l.il;
            rs.in = rb;
            rs.out = room ? -1 : rc;
            rs.slot = ram_free.back();
            ram_free.pop_back();
            ++ram_inflight;
            std::lock_guard<std::mutex> lk(sw_mu);
            ram_queue.push_back(rs);
            rqueued = true;
        }
        if (queued) sw_cv.notify_all();
        if (rqueued) ram_cv.notify_one();
    }

    // lookahead's thread: the ranges the CPU lane asked for, read in (os::prefetch) off the lane, which computes
    void prefetch_loop() {
        std::vector<std::pair<const void*, size_t>> b;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(pf_mu);
                pf_cv.wait(lk, [&] { return quit.load(std::memory_order_relaxed) || !pf_q.empty(); });
                if (quit.load(std::memory_order_relaxed)) return;
                b.swap(pf_q);
            }
            os::prefetch_cold(b);   // (the resident ones would only cost the page walk)
            b.clear();
        }
    }
    // an expert the CPU lane reads from the files (neither in VRAM nor in the RAM tier)
    static bool file_read(const Layer& l, int e) {
        return l.slot[(size_t) e] < 0 && (l.ram.empty() || l.ram[(size_t) e] < 0);
    }
    // the layer whose picks layer l0 guesses a layers ahead (1, 2): the next ones of its stack on its device, MXFP4
    // experts read in place, some of them from the files now (none: nothing to read early - null)
    const Layer* ahead_layer(const Layer& l0, int a) const {
        const int j = l0.il + a, end = l0.mtp ? g.nl + g.n_mtp : g.nl;
        if (j >= end || j >= (int) L.size()) return nullptr;
        const Layer& n = L[(size_t) j];
        if (n.dev != l0.dev || n.mtp != l0.mtp || n.lay.fmt != gg::kFmtMXFP4 || n.gate_w == nullptr || n.n_exp > 1024 ||
            (int) n.slot.size() < n.n_exp)
            return nullptr;
        for (int e = 0; e < n.n_exp; ++e)
            if (file_read(n, e)) return &n;
        return nullptr;
    }
    // lookahead on the CPU lane: request r (layer li, seq) carries the device's guesses for the next layers - the
    // guessed experts read from the files asked for now (each once a window), the guesses kept for the hit counts
    std::vector<std::pair<const void*, size_t>> pf_want;
    void ahead_ask(int li, const Req& r, unsigned seq, int T) {
        pf_want.clear();
        for (int a = 1; a <= 2; ++a) {
            const int m = std::min(r.na[a - 1], a == 1 ? kAhead1 : kPicks), j = li + a;
            if (m <= 0 || j >= (int) L.size()) continue;
            const Layer& n = L[(size_t) j];
            uint8_t* fl = ahead_fl.data() + (size_t) j * 1024;
            uint8_t* rk = ahead_rank.data() + (size_t) j * 1024;
            if (ahead_for[(size_t) j] != seq + (unsigned) a) {
                std::memset(fl, 0, 1024);
                std::memset(rk, 0, 1024);
                ahead_for[(size_t) j] = seq + (unsigned) a;
            }
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < m; ++i) {
                    const int e = r.ahead[t][(a - 1) * kAhead1 + i];
                    if (e < 0 || e >= n.n_exp || e >= (int) n.slot.size()) continue;
                    fl[e] |= a == 2 ? 4 : (i < n.k_exp ? 3 : 2);
                    if (a == 1 && (rk[e] == 0 || rk[e] > i + 1)) rk[e] = (uint8_t) (i + 1);
                    if ((fl[e] & 8) || !file_read(n, e)) continue;
                    fl[e] |= 8;
                    if (store) store->prefetch(j, e);
                    else
                        for (int q = 0; q < 3; ++q) pf_want.push_back({mx_mat(n, e, q), mx_part_bytes(n, q)});
                    ++st.ahead_pf;
                }
        }
        if (pf_want.empty()) return;
        {
            std::lock_guard<std::mutex> lk(pf_mu);
            pf_q.insert(pf_q.end(), pf_want.begin(), pf_want.end());
        }
        pf_cv.notify_one();
    }

    // the decode path's host side: each published routing in turn - its CPU lane experts (if any) into the answer's
    // part, then the answer's seq.  Spins while requests come (a token's 40 layers), naps after 2 ms without one.
    void serve_lane() {
        unsigned next = 1;
        auto last = std::chrono::steady_clock::now();
        std::vector<Ent> ents;
        while (!quit.load(std::memory_order_relaxed)) {
            Req& r = reqs[next % kRing];
            if (*(volatile unsigned*) &r.seq != next) {
                if(cache_epoch_enabled) {
                    const unsigned e=cache_epoch_request.load(std::memory_order_acquire);
                    if(e&&next-1==cache_epoch_service_target.load(std::memory_order_acquire)) {
                        cache_epoch_service_ack.store(e,std::memory_order_release);
                        while(!quit.load()&&cache_epoch_request.load(std::memory_order_acquire)==e)
                            std::this_thread::sleep_for(std::chrono::microseconds(20));
                        continue;
                    }
                }
                cpu_relax();
                if (std::chrono::steady_clock::now() - last > std::chrono::milliseconds(2))
                    std::this_thread::sleep_for(std::chrono::microseconds(20));
                continue;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            const int li = layer_of[next % kRing].load(std::memory_order_acquire);
            Resp& rp = resps[next % kRing];
            const int T = std::max(1, std::min(kWin, r.T));
            for (int t = 0; t < T; ++t) rp.cpu[t] = 0;
            if (li >= 0 && li < (int) L.size()) {
                Layer& l = L[(size_t) li];
                uint32_t* pr = prof.data() + (size_t) li * g.n_exp;
                ents.clear();
                if (store && li > 0 && li < g.nl) store->cancel_stale(li - 1);   // (guesses for layers the token has passed: dropped unread)
                // lookahead: the guesses made for this layer (when they are for this request), counted on decode
                const uint8_t* fl = T == 1 && !ahead_for.empty() && ahead_for[(size_t) li] == next
                                        ? ahead_fl.data() + (size_t) li * 1024 : nullptr;
                // the window's CPU-lane picks by expert (up to 8 tokens an entry), their weights by entry row
                float wv[kWin * kPicks];
                int tok_of[kWin * kPicks];
                int nslots = 0;
                std::vector<std::pair<int, int>> picks;   // (expert, t * kPicks + i), CPU-lane only
                if (trace_f != nullptr && !l.mtp) {   // DS4_TRACE=file: one record a route request: u8 layer, u8 T, T x k int16 expert ids
                    // DS4_TRACE_AHEAD=1: T has bit 7 set and the guesses follow: u8 na1, u8 na2, T x (na1 + na2) int16 (the
                    // next layer's likely picks by rank, then the one after's), as the lookahead made them for THIS layer's
                    // predecessors (the request carries the guesses for the layers after it)
                    static const bool with_ahead = std::getenv("DS4_TRACE_AHEAD") != nullptr;
                    uint8_t hd[2] = {(uint8_t) li, (uint8_t) (T | (with_ahead ? 0x80 : 0))};
                    std::fwrite(hd, 1, 2, trace_f);
                    for (int t = 0; t < T; ++t)
                        for (int i = 0; i < l.k_exp; ++i) {
                            const int16_t v = (int16_t) r.ids[t][i];
                            std::fwrite(&v, 2, 1, trace_f);
                        }
                    if (with_ahead) {
                        const uint8_t na[2] = {(uint8_t) std::max(0, std::min(r.na[0], kAhead1)), (uint8_t) std::max(0, std::min(r.na[1], kPicks))};
                        std::fwrite(na, 1, 2, trace_f);
                        for (int t = 0; t < T; ++t) {
                            for (int i = 0; i < na[0]; ++i) {
                                const int16_t v = (int16_t) r.ahead[t][i];
                                std::fwrite(&v, 2, 1, trace_f);
                            }
                            for (int i = 0; i < na[1]; ++i) {
                                const int16_t v = (int16_t) r.ahead[t][kAhead1 + i];
                                std::fwrite(&v, 2, 1, trace_f);
                            }
                        }
                    }
                }
                const bool dsk = disk_skip_w > 0.0f && !l.mtp && li >= disk_skip_from && li < g.nl;
                for (int t = 0; t < T; ++t) {
                    float wsum = 0.0f;   // (DISK_SKIP_W: the token's route weights)
                    if (dsk)
                        for (int i = 0; i < l.k_exp; ++i)
                            if (r.ids[t][i] >= 0 && r.ids[t][i] < l.n_exp) wsum += r.w[t][i];
                    for (int i = 0; i < l.k_exp; ++i) {
                        const int e = r.ids[t][i];
                        if (e < 0 || e >= l.n_exp) continue;
                        ++pr[e];
                        l.heat[(size_t) e] += 1.0f;
                        if (fl != nullptr) {
                            ++st.ahead_picks;
                            st.ahead_hit_k += fl[e] & 1;
                            st.ahead_hit += (fl[e] & 6) != 0;
                            st.ahead_hit2 += (fl[e] >> 2) & 1;
                        }
                        if (l.slot[(size_t) e] < 0) {
                            l.cpu_hits.push_back(e);
                            if (T == 1 && !l.zc.empty() && l.zc[(size_t) e]) {   // the device computed it from the store's slot
                                store->touch(li, e);
                                ++st.store_zc;
                                continue;
                            }
                            if (dsk && r.w[t][i] < disk_skip_w * wsum && !in_ram(l, e)) {
                                ++st.skipped;
                                continue;
                            }
                            if (!l.ram.empty() && l.ram[(size_t) e] >= 0) ++st.ram_experts;
                            else if (fl != nullptr) {
                                ++st.ahead_cold;
                                st.ahead_cold_hit += (fl[e] >> 3) & 1;
                            }
                            picks.push_back({e, t * kPicks + i});
                        }
                    }
                }
                if (fl != nullptr) {   // the guesses by rank: how many were picked
                    const uint8_t* rk = ahead_rank.data() + (size_t) li * 1024;
                    for (int e = 0; e < l.n_exp && e < 1024; ++e) {
                        if (rk[e] == 0 || rk[e] > 16) continue;
                        bool hit = false;
                        for (int i = 0; i < l.k_exp; ++i) hit = hit || r.ids[0][i] == e;
                        ++st.ahead_rank_n[rk[e] - 1];
                        st.ahead_rank_hit[rk[e] - 1] += hit;
                    }
                }
                if (ahead_mode > 0 && !ahead_for.empty()) ahead_ask(li, r, next, T);
                std::sort(picks.begin(), picks.end());
                for (size_t a = 0; a < picks.size();) {
                    Ent en;
                    en.e = picks[a].first;
                    en.slot0 = nslots;
                    en.n = 0;
                    while (a < picks.size() && picks[a].first == en.e && en.n < ent_cap(l)) {
                        const int t = picks[a].second / kPicks, i = picks[a].second % kPicks;
                        en.tok[en.n++] = t;
                        tok_of[nslots] = t;
                        wv[nslots] = r.w[t][i];
                        ++nslots;
                        ++a;
                    }
                    ents.push_back(en);
                }
                // lookahead: the entries asked for early (or in the RAM tier) computed first, so the ones asked for
                // only now get the others' time to come in from the SSD (their slots, so the sums, stay the same)
                int ord[kWin * kPicks];
                const int* po = nullptr;
                if (ahead_order && !ahead_for.empty() && ahead_for[(size_t) li] == next && ents.size() > 1 &&
                    ents.size() <= (size_t) (kWin * kPicks)) {
                    const uint8_t* fa = ahead_fl.data() + (size_t) li * 1024;
                    int n = 0;
                    for (int pass = 0; pass < 2; ++pass)
                        for (size_t q = 0; q < ents.size(); ++q) {
                            const int e = ents[q].e;
                            const bool early = !file_read(l, e) || (fa[e] & 8) != 0;
                            if (early == (pass == 0)) ord[n++] = (int) q;
                        }
                    po = ord;
                }
                if (!ents.empty()) {
                    const double tc = now_ms();
                    cpu_experts(l, &r.x[0][0], ents, (size_t) kMaxDim, po);
                    const int D = g.dim;
                    for (int t = 0; t < T; ++t) {
                        bool any = false;
                        for (int sl = 0; sl < nslots; ++sl) any = any || tok_of[sl] == t;
                        if (!any) continue;
                        float* o = rp.part[t];
                        for (int e = 0; e < D; ++e) o[e] = 0.0f;
                        for (int sl = 0; sl < nslots; ++sl) {
                            if (tok_of[sl] != t) continue;
                            const float* y = c_yd.data() + (size_t) sl * D;
                            for (int e = 0; e < D; ++e) o[e] += wv[sl] * y[e];
                        }
                        rp.cpu[t] = 1;
                    }
                    st.cpu_ms += now_ms() - tc;
                    st.cpu_experts += nslots;
                }
            }
            std::atomic_thread_fence(std::memory_order_release);
            *(volatile unsigned*) &rp.seq = next;
            if(cache_epoch_enabled)cache_epoch_service_completed.store(next,std::memory_order_release);
            ++next;
            last = std::chrono::steady_clock::now();
        }
    }

    // a GGUF's matrices (BF16 / MXFP4) to its own kernels, EXL3 ones to x3's
    static bool mv_any(const x3::MvJob* j, int n, cudaStream_t s) {
        bool g = false;
        for (int i = 0; i < n; ++i) g = g || j[i].m->fmt != 0;
        return g ? gg::mv(j, n, s) : x3::mv(j, n, s);
    }
    bool mv(Dev& d, std::initializer_list<x3::MvJob> jobs, std::string& err) {
        x3::MvJob j[x3::kMaxMvJobs];
        int n = 0;
        for (const x3::MvJob& x : jobs) j[n++] = x;
        if (!mv_any(j, n, d.st)) {
            err = "exl3 mv failed";
            return false;
        }
        return true;
    }
    bool mv_n(Dev& d, x3::MvJob* j, int n, std::string& err) {
        for (int a = 0; a < n; a += x3::kMaxMvJobs)
            if (!mv_any(j + a, std::min(x3::kMaxMvJobs, n - a), d.st)) {
                err = "exl3 mv failed";
                return false;
            }
        return true;
    }
    // y[c] = x[c] W for the T activations x + c * x_cs (floats), into y + c * y_cs
    static x3::MvJob job(const x3::Mat& m, const float* x, float* y, int T = 1, size_t x_cs = 0, size_t y_cs = 0,
                         float alpha = 1.0f) {
        x3::MvJob j;
        j.m = &m;
        j.xf = x;
        j.y = y;
        j.alpha = alpha;
        j.ncol = T;
        j.xf_cs = x_cs;
        j.y_cs = y_cs;
        return j;
    }

    // ---- the DSpark drafter (DeepSeek's DSparkBlock / forward_spec).  Its blocks' rings hold the trunk's main KV:
    // seed() turns a window's taps (the streams' mean entering layers 37..39) into it, through main_proj / main_norm
    // and each block's wkv / kv_norm / rope / FP8, for the positions the trunk kept.  draft() runs the block [y, noise x
    // (block - 1)] at positions pos .. through the blocks (each query attending to the ring and to the whole block),
    // the trunk's head, the markov bias token by token (greedy) and the confidence head.
    bool seed_drafter(int64_t p0, int n, int row0, std::string& err) {
        if (g.n_mtp == 0 || n <= 0) return true;
        if (n > kWin * kPicks) {
            err = "seeding the drafter with " + std::to_string(n) + " rows";
            return false;
        }
        Dev& d = devs[(size_t) d_dev];
        cudaSetDevice(d.id);
        const int nt = (int) g.d_targets.size(), D = g.dim, HD = g.hd;
        if (dump_d != nullptr) {   // DS4_DSPARK_DUMP: 'S', the first position, the rows, then the taps
            for (Dev& e : devs) {
                cudaSetDevice(e.id);
                CK(cudaStreamSynchronize(e.st));
            }
            cudaSetDevice(d.id);
            const int64_t hdr[4] = {'S', p0, n, (int64_t) nt * D};
            std::fwrite(hdr, 8, 4, dump_d);
            std::fwrite(h_taps + (size_t) row0 * nt * D, 4, (size_t) n * nt * D, dump_d);
            std::fflush(dump_d);
        }
        for (int a = 0; a < n; a += kWin) {   // (a chunk's last 128 rows: 16 a GEMV)
            const int na = std::min(kWin, n - a);
            CK(cudaMemcpyAsync(d.xe, h_taps + (size_t) (row0 + a) * nt * D, (size_t) na * nt * D * 4,
                               cudaMemcpyHostToDevice, d.st));
            if (!mv(d, {job(d_main_proj, d.xe, d.ed + (size_t) a * D, na, (size_t) nt * D, D)}, err)) return false;
        }
        CK(cudaEventRecord(ev_seed, d.st));
        Finish fm;
        fm.w = d_main_norm;
        fm.eps = g.eps;
        finish(d.ed, n, D, D, fm, d.st);
        for (int i = 0; i < g.n_mtp; ++i) {
            Layer& l = L[(size_t) (g.nl + i)];
            if (l.dev != d_dev) {
                err = "the DSpark blocks are split across devices";
                return false;
            }
            for (int a = 0; a < n; a += kWin) {
                const int na = std::min(kWin, n - a);
                if (!mv(d, {job(l.wkv, d.ed + (size_t) a * D, d.kvr, na, D, HD)}, err)) return false;
                Finish fk;
                fk.w = l.kv_norm;
                fk.eps = g.eps;
                fk.rd = g.rd;
                fk.pos = (int) (p0 + a);
                fk.freqs = d.freqs_win;
                fk.quant = 1;
                fk.dst = d.ext + (size_t) g.win * HD;
                fk.dst_stride = HD;
                finish(d.kvr, na, HD, HD, fk, d.st);
                ext_to_ring(d.ext, l.win, p0 + a, g.win, na, HD, d.st);
            }
        }
        return true;
    }

    bool draft(int y, std::vector<int>& out, std::vector<float>& conf, std::string& err, float temperature, float top_p,
               int top_k, float min_p, uint64_t seed, uint64_t counter) {
        const double t0 = now_ms();
        drafts_q = temperature > 0.0f && sample_drafts;
        dq_temp = temperature;
        dq_top_p = top_p;
        dq_top_k = top_k;
        dq_min_p = min_p;
        dq_seed = seed;
        dq_counter = counter;
        const bool ok = draft_(y, out, conf, err);
        if (!ok) drafts_q = false;
        if (max_drafts > 0 && (int) out.size() > max_drafts) {
            out.resize((size_t) max_drafts);
            conf.resize((size_t) max_drafts);
        }
        st.draft_ms += now_ms() - t0;
        ++st.drafts_rounds;
        return ok;
    }
    bool draft_(int y, std::vector<int>& out, std::vector<float>& conf, std::string& err) {
        out.clear();
        conf.clear();
        const int T = g.d_block, D = g.dim, H = g.hc, V = g.vocab, R = g.d_rank;
        const int64_t p0 = pos;
        if (g.n_mtp == 0 || T < 1 || T > kWin || p0 + T > o.max_context) return true;
        int tk[kWin];
        tk[0] = y;
        for (int t = 1; t < T; ++t) tk[t] = g.d_noise;
        Dev& d = devs[(size_t) d_dev];
        cudaSetDevice(d.id);
        for (int t = 0; t < T; ++t) std::memcpy(h_emb + (size_t) t * D, embed_pinned + (size_t) tk[t] * D, (size_t) D * 2);
        CK(cudaMemcpyAsync(d.emb_rows, h_emb, (size_t) T * D * 2, cudaMemcpyHostToDevice, d.st));
        embed(d.emb_rows, T, d.h, D, H, d.st);
        init_pre(d.pre, T, d.st);
        WinPlan dp;   // (window attention alone: the mailbox seqs)
        dp.async = !sync_decode && T <= async_max_t && !gpumiss::window(gpu_misses_enabled,T);
        for (int i = 0; i < g.n_mtp && dp.async; ++i) {
            const int il = g.nl + i;
            dp.seq[il] = ++seq_issued;
            layer_of[dp.seq[il] % kRing].store(il, std::memory_order_release);
        }
        const size_t hs = (size_t) T * H * D, rs = (size_t) g.win * g.hd;
        if (dump_d != nullptr && h_dump == nullptr &&
            (!halloc(h_dump, (size_t) g.n_mtp * kWin * H * D, err) || !halloc(h_dump_ring, (size_t) g.n_mtp * rs, err)))
            return false;
        for (int i = 0; i < g.n_mtp; ++i) {
            Layer& l = L[(size_t) (g.nl + i)];
            if (dump_d != nullptr) {   // (stream order: the ring the block reads - as BF16 rows - the stream it leaves)
                ring_to_ext(l.win, d.ext, g.win, g.win, g.hd, d.st);
                CK(cudaMemcpyAsync(h_dump_ring + (size_t) i * rs, d.ext, rs * 2, cudaMemcpyDeviceToHost, d.st));
            }
            if (!layer(l, 0, p0, T, dp, err)) return false;
            if (dump_d != nullptr) {
                cudaSetDevice(d.id);
                CK(cudaMemcpyAsync(h_dump + (size_t) i * hs, d.h, hs * 4, cudaMemcpyDeviceToHost, d.st));
            }
        }
        cudaSetDevice(d.id);
        hc_pre_norm(d.h, d.pre, nullptr, D, H, T, g.eps, d.x, d.st);   // (the confidence head reads it before the norm)
        Dev& hd = devs[(size_t) head_dev];
        // (through the host to the head's device, and to the head group's other members - the drafter's device can be
        // the head group's lead itself, a group of every GPU)
        if (d_dev != head_dev || groups[(size_t) hd.grp].mem.size() > 1) {
            CK(cudaMemcpyAsync(d.bounce, d.x, (size_t) T * D * 4, cudaMemcpyDeviceToHost, d.st));
            CK(cudaEventRecord(d.ev_out, d.st));
        }
        if (d_dev != head_dev) {
            cudaSetDevice(hd.id);
            CK(cudaStreamWaitEvent(hd.st, d.ev_out, 0));
            CK(cudaMemcpyAsync(hd.x, d.bounce, (size_t) T * D * 4, cudaMemcpyHostToDevice, hd.st));
        }
        // the drafter's final norm and the trunk's head, on every member of the head group (the others' shares first)
        const std::vector<int>& hg = groups[(size_t) hd.grp].mem;
        const bool mk = mk_split && !drafts_q;   // (sampled drafts: the lead's whole rows, gathered)
        for (int m = (int) hg.size() - 1; m >= 0; --m) {
            Dev& e = devs[(size_t) hg[(size_t) m]];
            cudaSetDevice(e.id);
            if (m > 0) {
                CK(cudaStreamWaitEvent(e.st, d.ev_out, 0));
                CK(cudaMemcpyAsync(e.ao, d.bounce, (size_t) T * D * 4, cudaMemcpyHostToDevice, e.st));
            } else
                CK(cudaMemcpyAsync(e.ao, e.x, (size_t) T * D * 4, cudaMemcpyDeviceToDevice, e.st));
            Finish fn;
            fn.w = e.d_norm;
            fn.eps = g.eps;
            finish(e.ao, T, D, D, fn, e.st);
            if (!head_share(e, m, e.ao, T, err, !mk)) return false;
        }
        h_idx[0] = y;
        if (mk) {
            // the markov steps over the group: each member its rows of the bias and its share's argmax, every member
            // then the global one (the sampler's tie rule) and its E row - the same drafts as the lead's whole rows,
            // each step ~1/nm of their 66 MB and no gather of the shares
            const int nm = (int) hg.size();
            const unsigned seq = ++mk_seq;
            for (int m = 0; m < nm; ++m) {
                Dev& e = devs[(size_t) hg[(size_t) m]];
                cudaSetDevice(e.id);
                CK(cudaMemcpyAsync(e.mk_ints, h_idx, 4, cudaMemcpyHostToDevice, e.st));
                CK(cudaMemcpyAsync(e.mk_erow, h_mk_embed + (size_t) y * R, (size_t) R * 2, cudaMemcpyHostToDevice, e.st));
            }
            for (int i = 0; i < T; ++i)
                for (int m = 0; m < nm; ++m) {
                    Dev& e = devs[(size_t) hg[(size_t) m]];
                    cudaSetDevice(e.id);
                    float* row = m == 0 ? hd.logits + (size_t) i * V : e.logits_part + (size_t) i * e.mk_vn;
                    markov_share(row, e.mk_share, e.mk_erow, e.mk_vn, R, e.st);
                    argmax_share(row, e.mk_vn, e.mk_v0, mk_slots + ((size_t) i * nm + m) * 4, seq, e.st);
                    markov_pick(mk_slots + (size_t) i * nm * 4, nm, seq, e.mk_ints + i + 1, h_mk_embed, e.mk_erow, R,
                                e.st);
                }
            cudaSetDevice(hd.id);
        } else {
            CK(cudaMemcpyAsync(d_ints, h_idx, 4, cudaMemcpyHostToDevice, hd.st));
            strata::dsv41k::SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            if (drafts_q) {   // the request's own chain: draft i drawn as the model's token i would be (verify's counter + i)
                sp.greedy = false;
                sp.temperature = dq_temp * (q_temp_scale > 0.0f ? q_temp_scale : 1.0f);
                sp.top_p = dq_top_p;
                sp.top_k = dq_top_k;
                sp.min_p = dq_min_p;
                sp.seed = dq_seed;
            }
            for (int i = 0; i < T; ++i) {   // the markov bias of the token before, then that position's draft
                markov_add(hd.logits + (size_t) i * V, d_mk_embed, d_mk_head, d_ints + i, V, R, hd.st);
                if (drafts_q) {
                    sp.counter = dq_counter + (uint64_t) i;
                    strata::dsv41k::sample_draft(hd.logits + (size_t) i * V, 1, V, sp, d_specq + i, d_ints + i + 1, hd.st);
                } else
                    strata::dsv41k::sample_tokens(hd.logits + (size_t) i * V, 1, V, nullptr, 0, sp, d_ints + i + 1, hd.st);
            }
        }
        confidence(hd.x, d_mk_embed, d_ints, d_conf, T, D, R, d_confv, hd.st);
        int tok[kWin + 1];
        float cf[kWin];
        CK(cudaMemcpyAsync(h_idx, d_ints, (size_t) (T + 1) * 4, cudaMemcpyDeviceToHost, hd.st));
        CK(cudaMemcpyAsync(h_w, d_confv, (size_t) T * 4, cudaMemcpyDeviceToHost, hd.st));
        CK(cudaStreamSynchronize(hd.st));
        for (int i = 0; i <= T; ++i) tok[i] = h_idx[i];
        for (int i = 0; i < T; ++i) cf[i] = h_w[i];
        if (dump_d != nullptr) {   // 'D', the block's position, y, T, then the drafts, their confidences, the hidden
            std::vector<float> xs((size_t) T * D);   // rows before the final norm; then each block's ring and stream
            CK(cudaMemcpy(xs.data(), hd.x, xs.size() * 4, cudaMemcpyDeviceToHost));
            cudaSetDevice(d.id);
            CK(cudaStreamSynchronize(d.st));
            const int64_t hdr[4] = {'D', p0, y, T};
            std::fwrite(hdr, 8, 4, dump_d);
            std::fwrite(tok + 1, 4, (size_t) T, dump_d);
            std::fwrite(cf, 4, (size_t) T, dump_d);
            std::fwrite(xs.data(), 4, xs.size(), dump_d);
            for (int i = 0; i < g.n_mtp; ++i) {
                std::fwrite(h_dump_ring + (size_t) i * rs, 2, rs, dump_d);
                std::fwrite(h_dump + (size_t) i * hs, 4, hs, dump_d);
            }
            std::fflush(dump_d);
            cudaSetDevice(hd.id);
        }
        for (int i = 0; i < T; ++i) {
            out.push_back(tok[i + 1]);
            conf.push_back(cf[i]);
        }
        return true;
    }

    int sample_row(int row, float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter,
                   const float* bias) {
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        strata::dsv41k::SamplerParams sp;
        sp.temperature = temperature;
        sp.greedy = temperature <= 0.0f;
        sp.top_p = top_p;
        sp.top_k = top_k;
        sp.min_p = min_p;
        sp.seed = seed;
        sp.counter = counter;
        sp.logit_bias = bias;
        strata::dsv41k::sample_tokens(d.logits + (size_t) row * g.vocab, 1, g.vocab, nullptr, 0, sp, d.tok, d.st);
        int t = -1;
        cudaMemcpyAsync(&t, d.tok, 4, cudaMemcpyDeviceToHost, d.st);
        cudaStreamSynchronize(d.st);
        return t;
    }

    // [y, drafts] at positions pos ..: the trunk's verdict, its next token, and the state back to what it kept.
    // Greedy: the drafts the trunk's argmax agrees with.  Sampled: the speculative-sampling rule over the request's own
    // chain (top_k, top_p, min_p, temperature): draft d stands with probability min(1, p(d) / q(d)), q the distribution
    // it was drawn from (the drafter's argmax: a point at d, so p(d)), a rejected one is replaced by a draw from the
    // residual max(0, p - q) - so each kept token is distributed as the trunk's own draw.
    bool verify(int y, const std::vector<int>& drafts, float temperature, float top_p, int top_k, float min_p,
                uint64_t seed, uint64_t counter, int& accepted, int& next, std::string& err) {
        const int n = (int) std::min<size_t>(drafts.size(), (size_t) kWin - 1);
        const int T = n + 1, V = g.vocab;
        const int64_t p0 = pos;
        int tk[kWin];
        tk[0] = y;
        for (int i = 0; i < n; ++i) tk[i + 1] = drafts[(size_t) i];
        spec_win = true;
        const double tv = now_ms();
        const bool ok = forward(tk, T, T, err, 0);
        st.verify_ms += now_ms() - tv;
        spec_win = false;
        if (!ok) return false;
        Dev& hd = devs[(size_t) head_dev];
        cudaSetDevice(hd.id);
        int m = 0;
        if (temperature <= 0.0f) {
            strata::dsv41k::SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            strata::dsv41k::sample_tokens(hd.logits, T, V, nullptr, 0, sp, d_ints + 16, hd.st);   // (the T rows at once)
            CK(cudaMemcpyAsync(h_idx, d_ints + 16, (size_t) T * 4, cudaMemcpyDeviceToHost, hd.st));
            CK(cudaStreamSynchronize(hd.st));
            while (m < n && h_idx[m] == drafts[(size_t) m]) ++m;
            next = h_idx[m];
        } else {
            // row i keeps draft i with probability min(1, p(d) / q(d)), else draws from the residual max(0, p - q)
            // (never d itself): each row a token distributed as the model's own draw at that position, p through the
            // request's chain - the top_k / top_p / min_p cut included (the drafter's argmax can lie outside it: p(d) 0,
            // never kept); past the first rejection the rows do not count
            const bool own = drafts_q && dq_temp == temperature && dq_top_p == top_p && dq_top_k == top_k &&
                             dq_min_p == min_p && dq_seed == seed && dq_counter == counter;
            if (!own) {   // the drafter's argmax: q a point at each draft
                for (int i = 0; i < n; ++i) {
                    h_specq[(size_t) i].n = 1;
                    h_specq[(size_t) i].ids[0] = drafts[(size_t) i];
                    h_specq[(size_t) i].p[0] = 1.0;
                }
                CK(cudaMemcpyAsync(d_specq, h_specq.data(), (size_t) n * sizeof(strata::dsv41k::SpecList),
                                   cudaMemcpyHostToDevice, hd.st));
            }
            for (int i = 0; i < n; ++i) h_idx[i] = drafts[(size_t) i];
            CK(cudaMemcpyAsync(d_ints + 32, h_idx, (size_t) n * 4, cudaMemcpyHostToDevice, hd.st));
            strata::dsv41k::SamplerParams sp;
            sp.greedy = false;
            sp.temperature = temperature;
            sp.top_p = top_p;
            sp.top_k = top_k;
            sp.min_p = min_p;
            sp.seed = seed;
            sp.counter = counter;
            strata::dsv41k::sample_verify(hd.logits, n, V, sp, d_specq, d_ints + 32, d_ints + 16, hd.st);
            CK(cudaMemcpyAsync(h_idx, d_ints + 16, (size_t) n * 4, cudaMemcpyDeviceToHost, hd.st));
            CK(cudaStreamSynchronize(hd.st));
            while (m < n && h_idx[m] == drafts[(size_t) m]) ++m;
            next = m < n ? h_idx[m] : sample_row(n, temperature, top_p, top_k, min_p, seed, counter + (uint64_t) n, nullptr);
        }
        drafts_q = false;
        rollback(p0, T, m + 1);
        if (!seed_drafter(p0, m + 1, 0, err)) return false;
        accepted = m;
        st.drafts_offered += n;
        st.drafts_accepted += m;
        for (int i = 0; i < n && i <= m && i < 8; ++i) {
            ++st.draft_reached[i];
            if (i < m) ++st.draft_kept[i];
        }
        return true;
    }

    // a verified window keeps its first `keep` positions: the ring rows and the compressors' state of the rest undone
    // (the compressed KV and the index keys past them are written again when those positions come)
    void rollback(int64_t p0, int T, int keep) {
        if (keep >= T) return;
        // the ring rows: one launch a device for all its layers (a launch a layer and member was ~300 on 8-way tensor
        // parallel, a millisecond of the host's before the next draft)
        std::vector<RingBatch> rb(devs.size());
        for (int il = 0; il < g.nl; ++il)
            for (int m = 0; m < L[(size_t) il].nm; ++m) {
                Layer& l = member(L[(size_t) il], m);
                RingBatch& b = rb[(size_t) l.dev];
                if (b.n == kMaxRingBatch) {   // (more layers on a device than a batch takes: this one alone)
                    cudaSetDevice(devs[(size_t) l.dev].id);
                    ring_restore(l.win, l.ring_bak, p0, g.win, keep, T, g.hd, devs[(size_t) l.dev].st);
                    continue;
                }
                b.ring[b.n] = l.win;
                b.bak[b.n++] = l.ring_bak;
            }
        for (size_t di = 0; di < devs.size(); ++di)
            if (rb[di].n > 0) {
                cudaSetDevice(devs[di].id);
                ring_restore_batch(rb[di], p0, g.win, keep, T, g.hd, devs[di].st);
            }
        for (int il = 0; il < g.nl; ++il)
            for (int m = 0; m < L[(size_t) il].nm; ++m) {
                Layer& l = member(L[(size_t) il], m);
                Dev& d = devs[(size_t) l.dev];
                if (l.c_snap != nullptr) {
                    cudaSetDevice(d.id);
                    const size_t ns = (size_t) l.ratio * g.hd * 4;
                    cudaMemcpyAsync(l.c_kv, l.c_snap, ns, cudaMemcpyDeviceToDevice, d.st);
                    cudaMemcpyAsync(l.c_score, l.c_snap + (size_t) l.ratio * g.hd, ns, cudaMemcpyDeviceToDevice, d.st);
                    compress_win(l.c_rows, l.c_rows + (size_t) kWin * g.hd, l.c_kv, l.c_score, l.ratio, (int) p0, keep,
                                 g.hd, l.c_norm, g.eps, d.latent, d.st);
                }
            }
        pos = p0 + keep;
        toks.resize((size_t) pos);
        ctoks.resize((size_t) pos);
        slot_held = false;
    }

    // ---- a window: T tokens at positions pos .. pos + T - 1 through every layer; the logits of its last `logits` rows
    bool step(int token, bool want_logits, std::string& err) { return forward(&token, 1, want_logits ? 1 : 0, err); }


    // layer L0 for a chunk of T tokens at p0 as group member mi: layer()'s arithmetic over the chunk's rows (the
    // projections as GEMMs, the attention and the indexer in slabs, the routed experts as sorted rows: the VRAM ones in
    // place, the CPU lane's streamed in batches).  kv_only (a long prompt's first pass, Run::stop): only what the layers
    // after it read of these rows - its compressed KV and index keys - and nothing of its own attention or experts
    bool layer_chunk(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, Streamer& S, std::string& err,
                     bool kv_only = false) {
        Layer& l = member(L0, mi);
        Dev& d = devs[(size_t) l.dev];
        Chunk& c = *chunks[(size_t) l.dev];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        prof_mark(d, 0);
        const int nm = L0.nm;
        const int D = g.dim, H = g.hc, HD = g.hd, NHL = l.ahn, HQ = NHL * HD, OG = l.ogn, FF = l.ffn;
        const ArGroup& ab = groups[(size_t) L0.grp].arb;
        const float* freqs = l.ratio ? d.freqs_comp : d.freqs_win;
        // Engram, in slabs of 256 tokens
        if (l.eg >= 0) {
            const int cols = (pk.eg_ngram - 1) * pk.eg_heads, ehd = pk.eg_hd;
            const size_t per = (size_t) cols * (ehd + ehd / 32);
            const int slot = wp.eg_slot >= 0 ? wp.eg_slot : wp.k % 2;   // (read ahead by the run: two in flight)
            uint8_t* host = c.eg_host[slot] + (size_t) l.eg * c.C * per;
            CK(cudaMemcpyAsync(c.eg_rows, host, (size_t) T * per, cudaMemcpyHostToDevice, s));
            CK(cudaEventRecord(c.eg_ev[slot], s));
            for (int s0 = 0; s0 < T; s0 += 256) {
                const int ns = std::min(256, T - s0);
                engram_dequant_tok(c.eg_rows + (size_t) s0 * per, per, cols, ns, ehd, c.eg_x, s, (int) eg_fmt[(size_t) l.eg]);
                x3::rows_f16(c.eg_x, cols * ehd, nullptr, ns, cols * ehd, c.eg_x16, s);
                if (prefill_engram_cache && chunk_major_phase_pending && c.eg_weight_cache) {
                    if (!gemm_cached_engram(d,c,l,p0,c.eg_x16,cols*ehd,ns,c.eg_kv,(H+1)*D,err)) return false;
                } else {
                if (!gemm(d, c, l.eg_wkv, c.eg_x16, cols * ehd, ns, c.eg_kv, (H + 1) * D, 0.0f, err)) return false;
                }
                engram_gate(c.h + (size_t) s0 * H * D, c.eg_kv, l.eg_w, H, D, ns, g.eps, s);
            }
        }
        prof_mark(d, 4);
        // attention
        if (!hc_mix_chunk(d, c, c.h, l.hca_fn, T, c.mix, err)) return false;
        hc_split(c.mix, l.hca_scale, l.hca_base, H, g.sinkhorn, g.hc_eps, T, c.a_pre, c.a_post, c.a_comb, s);
        hc_pre_norm(c.h, c.pre, l.attn_norm, D, H, T, g.eps, c.x, s);
        x3::rows_f16(c.x, D, nullptr, T, D, c.x16, s);
        prof_mark(d, 5);
        if (!kv_only && (!gemm(d, c, l.wq_a, c.x16, D, T, c.qr, g.q_lora, 0.0f, err) ||
                         !gemm(d, c, l.wkv, c.x16, D, T, c.kvr, HD, 0.0f, err)))
            return false;
        if (l.kv_src && !gemm(d, c, l.c_wkv, c.x16, D, T, c.ckr, HD, 0.0f, err)) return false;
        if (l.kv_src && l.ratio > 1 && !gemm(d, c, l.c_wgate, c.x16, D, T, c.cgr, HD, 0.0f, err)) return false;
        prof_mark(d, 6);
        const bool snap = !kv_only && wp.snap_t >= 0 && mi == 0 && L0.snap_win != nullptr;
        if (!kv_only) {
            Finish fq;
            fq.w = l.q_norm;
            fq.eps = g.eps;
            finish(c.qr, T, g.q_lora, g.q_lora, fq, s);
            x3::rows_f16(c.qr, g.q_lora, nullptr, T, g.q_lora, c.qr16, s);
            // the window rows: the ring's (positions p0 - win ..) then the chunk's
            ring_to_ext(l.win, c.ext, p0, g.win, HD, s);
            Finish fk;
            fk.w = l.kv_norm;
            fk.eps = g.eps;
            fk.rd = g.rd;
            fk.pos = (int) p0;
            fk.freqs = freqs;
            fk.quant = 1;
            fk.dst = c.ext + (size_t) g.win * HD;
            fk.dst_stride = HD;
            finish(c.kvr, T, HD, HD, fk, s);
            // feed_mark's checkpoint: the ring as it was before row snap_t - the old ring, then the chunk's rows before it
            if (snap) {
                CK(cudaMemcpyAsync(L0.snap_win, l.win, (size_t) g.win * win_rb(), cudaMemcpyDeviceToDevice, s));
                if (wp.snap_t > 0) ext_to_ring(c.ext, L0.snap_win, p0, g.win, wp.snap_t, HD, s, nullptr);
            }
            ext_to_ring(c.ext, l.win, p0, g.win, T, HD, s, nullptr);
        }
        const uint8_t* ckv = nullptr;
        const long long ss = (long long) o.max_context + 64;
        if (l.ratio > 0) {
            Src& sr = srcs[(size_t) l.src];
            const int R = l.ratio;
            const int64_t g_first = p0 / R, g_end = (p0 + T) / R;
            const int ng = (int) std::max<int64_t>(0, g_end - g_first);
            const float* latent = c.latent;
            if (l.kv_src) {
                if (R == 1) {
                    Finish fc;
                    fc.w = l.c_norm;
                    fc.eps = g.eps;
                    finish(c.ckr, T, HD, HD, fc, s);
                    latent = c.ckr;
                } else {
                    if (snap && L0.snap_kv != nullptr) {   // (each slot: the latest row before snap_t that holds it)
                        const int64_t ps = p0 + wp.snap_t;
                        for (int sl = 0; sl < R; ++sl) {
                            const int64_t q = ps - 1 - (((ps - 1 - sl) % R) + R) % R;
                            const float* kv = q >= p0 ? c.ckr + (size_t) (q - p0) * HD : l.c_kv + (size_t) sl * HD;
                            const float* sc = q >= p0 ? c.cgr + (size_t) (q - p0) * HD : l.c_score + (size_t) sl * HD;
                            CK(cudaMemcpyAsync(L0.snap_kv + (size_t) sl * HD, kv, (size_t) HD * 4, cudaMemcpyDeviceToDevice, s));
                            CK(cudaMemcpyAsync(L0.snap_sc + (size_t) sl * HD, sc, (size_t) HD * 4, cudaMemcpyDeviceToDevice, s));
                        }
                    }
                    compress_win(c.ckr, c.cgr, l.c_kv, l.c_score, R, (int) p0, T, HD, l.c_norm, g.eps, c.latent, s);
                }
            }
            const int clen_max = (int) ((p0 + T) / R);
            if (l.ix_src) {
                Src& ks = srcs[(size_t) l.ik_src];
                if (l.owns_k && ng > 0) {
                    x3::rows_f16(latent, HD, nullptr, ng, HD, c.lat16, s);
                    if (!gemm(d, c, l.i_wk, c.lat16, HD, ng, c.ikr, g.ix_hd, 0.0f, err)) return false;
                    Finish fi;
                    fi.w = l.ik_norm;
                    fi.eps = g.eps;
                    fi.rd = g.rd;
                    fi.pos = (int) (g_first * R);
                    fi.pos_step = R;
                    fi.freqs = freqs;
                    fi.quant = 3;
                    fi.pdst = ks.ik[(size_t) l.dev] + (size_t) g_first * ik_rb();
                    fi.pdst_stride = (long long) ik_rb();
                    finish(c.ikr, ng, g.ix_hd, g.ix_hd, fi, s);
                }
                if (clen_max > 0 && !kv_only) {
                    if (!gemm_f16w(c, l.i_wproj, g.ix_heads, D, c.x16, D, T,
                                   1.0f / std::sqrt((float) g.ix_hd) / std::sqrt((float) g.ix_heads), c.iw, 64, err))
                        return false;
                    for (int a0 = 0; a0 < T; a0 += c.AI) {
                        const int na = std::min(c.AI, T - a0);
                        const int64_t pa = p0 + a0;
                        if (!gemm(d, c, l.i_wq_b, c.qr16 + (size_t) a0 * g.q_lora, g.q_lora, na, c.iq,
                                  g.ix_heads * g.ix_hd, 0.0f, err))
                            return false;
                        Finish fi;
                        fi.rd = g.rd;
                        fi.pos = (int) pa;
                        fi.vpp = g.ix_heads;
                        fi.freqs = freqs;
                        fi.quant = 3;
                        finish(c.iq, na * g.ix_heads, g.ix_hd, g.ix_hd, fi, s);
                        Rows lim;
                        lim.gen = 1;
                        lim.p0 = pa;
                        lim.R = R;
                        lim.kcap = g.ix_topk;
                        lim.ccap = g.cand_topk;
                        lim.blk = std::max(1, g.cand_blk);
                        const int nlast = (int) ((pa + na) / R);   // the slab's widest query
                        int* tk = c.topk + (size_t) a0 * topk_stride();
                        int* cd = c.cand + (size_t) a0 * cand_stride();
                        const float* iw = c.iw + (size_t) a0 * 64;
                        if (nlast <= 0) continue;
                        if (l.uses_cand && (nlast + g.cand_blk - 1) / g.cand_blk > g.cand_topk) {
                            Rows cl = lim;
                            cl.gen = 2;
                            index_score(c.iq, iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, na, g.cand_topk * g.cand_blk,
                                        cd, (int) cand_stride(), g.cand_blk, cl, c.scores, ss, s);
                            Rows tr = lim;
                            tr.gen = 3;
                            topk(c.scores, ss, na, tr, cd, (int) cand_stride(), g.cand_blk, tk, (int) topk_stride(), s, c.topk_scr.states ? &c.topk_scr : nullptr);
                        } else {
                            index_score(c.iq, iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, na, nlast, nullptr, 0, 1,
                                        lim, c.scores, ss, s);
                            if (l.cand_src) {
                                Rows cr = lim;
                                cr.kcap = g.cand_topk;
                                cand_blocks(c.scores, ss, na, cr, g.cand_blk, c.bm, (long long) bm_stride(), cd,
                                            (int) cand_stride(), s, c.topk_scr.states ? &c.topk_scr : nullptr);
                            }
                            topk(c.scores, ss, na, lim, nullptr, 0, 1, tk, (int) topk_stride(), s, c.topk_scr.states ? &c.topk_scr : nullptr);
                        }
                    }
                }
            }
            if (l.kv_src && ng > 0) {
                Finish fc;
                fc.rd = g.rd;
                fc.pos = (int) (g_first * R);
                fc.pos_step = R;
                fc.freqs = freqs;
                fc.quant = 2;
                fc.pdst = sr.ckv[(size_t) l.dev] + (size_t) g_first * ckv_rb();
                fc.pdst_stride = (long long) ckv_rb();
                finish(const_cast<float*>(latent), ng, HD, HD, fc, s);
                if (mi == 0 && sr.remote) {   // the new rows for the other groups' readers (slot k % 2, once
                    const int slot = wp.k % 2;  // they took the chunk before last's)
                    if (wp.k >= 2)
                        for (int r : sr.readers) wait_flags(sr.x_ack + ((size_t) slot * devs.size() + r) * 32, 1, 32, wp.win - 2, s);
                    Xfer x;
                    x.src[0] = fc.pdst;
                    x.dst[0] = sr.stage(slot);
                    x.bytes[0] = (size_t) ng * ckv_rb();
                    x.n = 1;
                    if (l.owns_k && srcs[(size_t) l.ik_src].ik[(size_t) l.dev] != nullptr) {
                        x.src[1] = srcs[(size_t) l.ik_src].ik[(size_t) l.dev] + (size_t) g_first * ik_rb();
                        x.dst[1] = sr.stage(slot) + x_cap() * ckv_rb();
                        x.bytes[1] = (size_t) ng * ik_rb();
                        x.n = 2;
                    }
                    xfer_put(x, sr.flag(slot), wp.win, d.xcnt, s);
                }
            }
            ckv = sr.ckv[(size_t) l.dev];
        }
        if (kv_only) return true;
        prof_mark(d, 7);
        // the attention in slabs: q, the sparse attention, wo_a, wo_b (this member's share) into acc
        if (NHL == 0) CK(cudaMemsetAsync(c.acc, 0, (size_t) T * D * sizeof(*c.acc), s));   // (no heads)
        for (int a0 = 0; a0 < T && NHL > 0; a0 += c.A) {
            const int na = std::min(c.A, T - a0);
            if(prefill_dense_projection_cache && chunk_major_phase_pending && c.dense_projection_cache) {
                if(!gemm_cached_dense_projection(d,c,l,0,p0,T,l.wq_b,c.qr16+(size_t)a0*g.q_lora,
                                                 g.q_lora,na,c.q,HQ,0.0f,err))return false;
            } else {
                if (!gemm(d, c, l.wq_b, c.qr16 + (size_t) a0 * g.q_lora, g.q_lora, na, c.q, HQ, 0.0f, err)) return false;
            }
            Finish fr;
            fr.rd = g.rd;
            fr.pos = (int) (p0 + a0);
            fr.vpp = NHL;
            fr.freqs = freqs;
            finish(c.q, na * NHL, HD, HD, fr, s);
            prof_mark(d, 8);
            AttnRows ar;
            ar.gen = 1;
            ar.p0 = p0;
            ar.t0 = a0;
            ar.win = g.win;
            ar.R = l.ratio;
            ar.kcap = g.ix_topk;
            sparse_attn(c.q, c.ext, ckv, c.topk + (size_t) a0 * topk_stride(), (int) topk_stride(), ar, l.sink, NHL, HD,
                        na, 1.0f / std::sqrt((float) HD), c.o, s, kKvFp4, 0, &d.attn_scr);   // (a short last slab)
            prof_mark(d, 9);
            Finish fo;
            fo.rd = g.rd;
            fo.pos = (int) (p0 + a0);
            fo.vpp = NHL;
            fo.freqs = freqs;
            fo.inverse = 1;
            fo.quant = 1;
            finish(c.o, na * NHL, HD, HD, fo, s);
            x3::rows_f16(c.o, HQ, nullptr, na, HQ, c.o16, s);
            const int per = HQ / OG;
            for (int gi = 0; gi < OG; ++gi) {
                if(prefill_dense_projection_cache && chunk_major_phase_pending && c.dense_projection_cache) {
                    if(!gemm_cached_dense_projection(d,c,l,1+gi,p0,T,l.wo_a[gi],c.o16+(size_t)gi*per,
                                                     HQ,na,c.oa+(size_t)gi*g.o_lora,OG*g.o_lora,0.0f,err))return false;
                } else {
                    if (!gemm(d, c, l.wo_a[gi], c.o16 + (size_t) gi * per, HQ, na, c.oa + (size_t) gi * g.o_lora,
                              OG * g.o_lora, 0.0f, err))
                        return false;
                }
            }
            x3::rows_f16(c.oa, OG * g.o_lora, nullptr, na, OG * g.o_lora, c.oa16, s);
            if(prefill_dense_projection_cache && chunk_major_phase_pending && c.dense_projection_cache) {
                if(!gemm_cached_dense_projection(d,c,l,9,p0,T,l.wo_b,c.oa16,OG*g.o_lora,na,
                                                 c.acc+(size_t)a0*D,D,0.0f,err))return false;
            } else {
                if (!gemm(d, c, l.wo_b, c.oa16, OG * g.o_lora, na, c.acc + (size_t) a0 * D, D, 0.0f, err)) return false;
            }
        }
        if (nm > 1) allreduce(c.acc, T * D, ab, mi, ++c.arb_seq, d.xcnt, s);
        hc_post_inplace(c.acc, c.h, c.a_post, c.a_comb, D, H, T, s);
        prof_mark(d, 1);
        // the MoE
        if (!hc_mix_chunk(d, c, c.h, l.hcf_fn, T, c.mix, err)) return false;
        hc_split(c.mix, l.hcf_scale, l.hcf_base, H, g.sinkhorn, g.hc_eps, T, c.f_pre, c.f_post, c.f_comb, s);
        hc_pre_norm(c.h, c.a_pre, l.ffn_norm, D, H, T, g.eps, c.x, s);
        x3::rows_f16(c.x, D, nullptr, T, D, c.x16, s);
        gemv_f32(l.gate_w, c.x, l.n_exp, D, T, c.glog, 1024, s);   // (F32 x: the token path's bits)
        for (int r0 = 0; r0 < T; r0 += kVlRows) {   // (route()'s image bits cover kVlRows rows a call)
            const int nr = std::min(kVlRows, T - r0);
            VlBits vb;
            const bool vl = l.gate_bias_vl != nullptr && vl_bits(p0 + r0, nr, vb);
            route(c.glog + (size_t) r0 * 1024, l.gate_bias, l.n_exp, l.k_exp, nr, g.route_scale,
                  c.ridx + (size_t) r0 * kPicks, c.rw + (size_t) r0 * kPicks, s, l.gate_bias_vl, vl ? &vb : nullptr);
        }
        const int n_vs = L0.n_slots + (int) L0.host_list.size();
        CK(cudaMemcpyAsync(l.vs_dev, l.vs_pin, (size_t) l.n_exp * 4, cudaMemcpyHostToDevice, s));   // (this member's)
        moe_sort(c.ridx, T, l.k_exp, l.vs_dev, n_vs, c.bounds, c.row_tok, c.pos, s);
        CK(cudaMemcpyAsync(c.bounds_pin, c.bounds, (size_t) (n_vs + 1) * 4, cudaMemcpyDeviceToHost, s));
        prof_mark(d, 10);
        // the shared expert into acc while the host waits for the bounds
        if (!gemm(d, c, l.s1, c.x16, D, T, c.sg, FF, 0.0f, err) || !gemm(d, c, l.s3, c.x16, D, T, c.su, FF, 0.0f, err))
            return false;
        swiglu(c.sg, c.su, T * FF, g.lim, c.sg, s);
        x3::rows_f16(c.sg, FF, nullptr, T, FF, c.sh16, s);
        if (!gemm(d, c, l.s2, c.sh16, FF, T, c.acc, D, 0.0f, err)) return false;
        prof_mark(d, 2);
        // the expert stream's copies keep going while the device runs this layer's attention, routing and shared expert:
        // the next items (this layer's likely experts, queued when the layer before routed) go out as they are ready and
        // ring slots free, not only once the bounds are back (Project Maya #360; DS4_FLOW_PUMP_WAIT=0: wait as before)
        static const bool pump_wait = std::getenv("DS4_FLOW_PUMP_WAIT") == nullptr ||
                                      std::atoi(std::getenv("DS4_FLOW_PUMP_WAIT")) != 0;
        if (c.flow && pump_wait)
            while (cudaStreamQuery(s) == cudaErrorNotReady) {
                if (!flow_pump(c, err)) return false;
                std::this_thread::yield();
            }
        CK(cudaStreamSynchronize(s));
        std::vector<int> b(c.bounds_pin, c.bounds_pin + n_vs + 1);
        if (dump_r != nullptr && mi == 0) {
            const size_t nk = (size_t) T * l.k_exp, np = (size_t) T * kPicks;
            std::vector<int> ri(np), rt(nk);
            std::vector<float> gl((size_t) T * l.n_exp), gb((size_t) l.n_exp);
            CK(cudaMemcpy(ri.data(), c.ridx, np * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(rt.data(), c.row_tok, nk * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy2D(gl.data(), (size_t) l.n_exp * 4, c.glog, 1024 * 4, (size_t) l.n_exp * 4, (size_t) T,
                            cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(gb.data(), l.gate_bias, gb.size() * 4, cudaMemcpyDeviceToHost));
            std::lock_guard<std::mutex> lk(dump_mu);
            const int64_t hdr[8] = {l.il, p0, T, l.k_exp, l.n_exp, n_vs, L0.n_slots, kPicks};
            std::fwrite(hdr, 8, 8, dump_r);
            std::fwrite(ri.data(), 4, np, dump_r);
            std::fwrite(L0.vs_pin, 4, (size_t) l.n_exp, dump_r);
            std::fwrite(b.data(), 4, b.size(), dump_r);
            std::fwrite(rt.data(), 4, nk, dump_r);
            std::fwrite(gl.data(), 4, gl.size(), dump_r);
            std::fwrite(gb.data(), 4, gb.size(), dump_r);
            std::fflush(dump_r);
        }
        if (mi == 0 && !l.mtp && (pred_f != nullptr || xin_f != nullptr)) {
            std::lock_guard<std::mutex> lk(dump_mu);
            if (pred_f != nullptr) {
                float* dp = nullptr;
                CK(cudaMalloc(&dp, (size_t) T * 256 * 4));
                pred_proj_k<<<T, 256, 0, s>>>(c.x, (int) D, dp);
                std::vector<float> hp((size_t) T * 256), hl((size_t) T * l.n_exp);
                CK(cudaMemcpyAsync(hp.data(), dp, hp.size() * 4, cudaMemcpyDeviceToHost, s));
                CK(cudaMemcpy2DAsync(hl.data(), (size_t) l.n_exp * 4, c.glog, 1024 * 4, (size_t) l.n_exp * 4, (size_t) T,
                                     cudaMemcpyDeviceToHost, s));
                CK(cudaStreamSynchronize(s));
                CK(cudaFree(dp));
                std::vector<uint16_t> o(hp.size() + hl.size());
                for (size_t i = 0; i < hl.size(); ++i) o[i] = strata::kernels::f16_from_f32(hl[i]);
                for (size_t i = 0; i < hp.size(); ++i) o[hl.size() + i] = strata::kernels::f16_from_f32(hp[i]);
                const uint8_t hd[3] = {(uint8_t) l.il, (uint8_t) (T & 255), (uint8_t) (T >> 8)};
                std::fwrite(hd, 1, 3, pred_f);
                std::fwrite(o.data(), 2, o.size(), pred_f);
            }
            if (xin_f != nullptr && xin_got[(size_t) l.il] < xin_n) {   // up to 32 evenly spaced tokens of the chunk
                const int tk = std::min(std::min(T, 32), xin_n - xin_got[(size_t) l.il]);
                const int stride = std::max(1, T / tk);
                std::vector<float> hx((size_t) T * D);
                CK(cudaMemcpyAsync(hx.data(), c.x, hx.size() * 4, cudaMemcpyDeviceToHost, s));
                CK(cudaStreamSynchronize(s));
                const uint8_t hd[3] = {(uint8_t) l.il, (uint8_t) (tk & 255), (uint8_t) (tk >> 8)};
                std::fwrite(hd, 1, 3, xin_f);
                for (int i = 0; i < tk; ++i) std::fwrite(hx.data() + (size_t) std::min(T - 1, i * stride) * D, 4, (size_t) D, xin_f);
                xin_got[(size_t) l.il] += tk;
            }
        }
        if (mi == 0 && trace_f != nullptr && !l.mtp) {   // DS4_TRACE: a chunk's picks, u8 (layer | 0x80), u16 T, T x k int16 ids
            std::vector<int> ri((size_t) T * kPicks);
            CK(cudaMemcpy(ri.data(), c.ridx, ri.size() * 4, cudaMemcpyDeviceToHost));
            std::lock_guard<std::mutex> lk(dump_mu);
            const uint8_t hd[3] = {(uint8_t) (l.il | 0x80), (uint8_t) (T & 255), (uint8_t) (T >> 8)};
            std::fwrite(hd, 1, 3, trace_f);
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < l.k_exp; ++i) {
                    const int16_t v = (int16_t) ri[(size_t) t * kPicks + i];
                    std::fwrite(&v, 2, 1, trace_f);
                }
        }
        if (mi == 0) {   // the profile (the CPU lane's experts: by the plan's order)
            uint32_t* pr = prof.data() + (size_t) l.il * g.n_exp;
            for (int e = 0; e < l.n_exp; ++e) {
                const int v = L0.vs_pin[e];
                const int cnt = b[(size_t) v + 1] - b[(size_t) v];
                pr[e] += (uint32_t) cnt;
                if (cnt > 0) {   // (the expert cache learns from a chunked prompt as from windows: heat, and the CPU lane's hits)
                    l.heat[(size_t) e] += (float) cnt;
                    if (l.slot[(size_t) e] < 0) l.cpu_hits.push_back(e);
                }
            }
            st.experts += (int64_t) T * l.k_exp;
        }
        if (c.flow) {
            if (!flow_layer(d, c, L0, mi, wp.k, T, b, err)) return false;
            if (nm > 1) allreduce(c.acc, T * D, ab, mi, ++c.arb_seq, d.xcnt, s);
            hc_post_inplace(c.acc, c.h, c.f_post, c.f_comb, D, H, T, s);
            CK(cudaMemcpyAsync(c.pre, c.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
            prof_mark(d, 3);
            return true;
        }
        // the streamed experts as their batches arrive (each batch's slots freed for the copies at once), the VRAM ones
        // in shares between them (the device works while the copies run)
        const int di = l.dev, P = Chunk::kParts;
        int nb = 0;
        for (size_t q = S.kp; q < S.b.size() && S.b[q].il == L0.il && S.b[q].k == wp.k; ++q) ++nb;
        int v_done = 0;
        for (int bi = 0; bi < nb; ++bi) {
            const Batch& bt = S.b[S.kp];
            const int half = (int) (S.kp % P);
            CK(cudaStreamWaitEvent(s, c.filled[half], 0));
            const int v0 = L0.n_slots + bt.j0;
            if (!run_set(d, c, l, c.ring + (size_t) half * (c.ring_n / P) * c.ring_stride, v0, c.ring_stride, c.bounds,
                         b.data(), v0, v0 + bt.n, T, err))
                return false;
            CK(cudaEventRecord(c.freed[half], s));
            ++S.kp;
            if (!copy_ahead(d, c, S, di, err)) return false;
            const int v_to = (int) ((int64_t) L0.n_slots * (bi + 1) / nb);
            if (!run_set(d, c, l, l.vram_base, 0, unit(l), c.bounds, b.data(), v_done, v_to, T, err)) return false;
            v_done = v_to;
        }
        if (!run_set(d, c, l, l.vram_base, 0, unit(l), c.bounds, b.data(), v_done, L0.n_slots, T, err)) return false;
        if (nm > 1) allreduce(c.acc, T * D, ab, mi, ++c.arb_seq, d.xcnt, s);
        hc_post_inplace(c.acc, c.h, c.f_post, c.f_comb, D, H, T, s);
        CK(cudaMemcpyAsync(c.pre, c.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
        prof_mark(d, 3);
        return true;
    }

    // the CPU lane's experts of every layer, in index order, and each expert's virtual slot (before a chunk)
    void plan_streams() {
        for (Layer& l : L) {
            if (l.mtp || l.vs_pin == nullptr) continue;
            l.host_list.clear();
            for (int e = 0; e < l.n_exp; ++e)
                if (l.slot[(size_t) e] < 0) l.host_list.push_back(e);
            for (int m = 0; m < l.nm; ++m) {   // (a group's members: each its own copy)
                Layer& x = member(l, m);
                int j = 0;
                for (int e = 0; e < l.n_exp; ++e) x.vs_pin[e] = l.slot[(size_t) e] >= 0 ? l.slot[(size_t) e] : l.n_slots + j++;
            }
        }
    }

    // a chunk's hand-off between groups: h | pre | topk | cand of a chunk of T rows
    Xfer chunk_xfer(uint8_t* host, Chunk& c, int T, bool put) const {
        const size_t nh = (size_t) T * g.hc * g.dim * 4, np = (size_t) T * 8 * 4, nk = (size_t) T * topk_stride() * 4,
                     nc = (size_t) T * cand_stride() * 4;
        void* dv[4] = {c.h, c.pre, c.topk, c.cand};
        uint8_t* hv[4] = {host, host + nh, host + nh + np, host + nh + np + nk};
        const size_t nb[4] = {nh, np, nk, nc};
        Xfer x;
        x.n = 4;
        for (int i = 0; i < 4; ++i) {
            x.src[i] = put ? (const void*) dv[i] : (const void*) hv[i];
            x.dst[i] = put ? (void*) hv[i] : dv[i];
            x.bytes[i] = nb[i];
        }
        return x;
    }

    // a run of chunks (start positions p0[k], sizes T[k], seqs seq[k]) on device di: its layers chunk after chunk, the
    // groups pipelined (a group takes chunk k + 1 while the next takes chunk k: the hand-offs and the sources' rows
    // two deep, each slot reused once its readers acknowledged it), the streamed experts' copies running ahead across
    // layers and chunks
    struct Run {
        std::vector<int64_t> p0;
        std::vector<int> T;
        std::vector<unsigned> seq;
        int64_t snap_pos = -1;   // feed_mark: the position whose state before it the layers keep (-1: none)
        // a long prompt's skip (forward_chunks): the cut at row cut_r of chunk cut_k - the chunks before it go through
        // the layers before `yoco`, that layer as layer_chunk's kv_only and none after it; chunk cut_k's rows before
        // cut_r the same, its rows from cut_r on and the chunks after it through every layer (yoco -1: no skip)
        int yoco = -1, cut_k = 0, cut_r = 0;
        bool runs(int k, int il) const { return yoco < 0 || k >= cut_k || il <= yoco; }
        bool kv_only(int k, int il) const { return yoco >= 0 && k < cut_k && il == yoco; }
        int r0(int k, int il) const { return yoco >= 0 && k == cut_k && il >= yoco ? cut_r : 0; }   // the first row
    };
    // layer L0 over chunk k per a long prompt's skip (Run): its kv_only, or its rows from Run::r0 on - the chunk's
    // buffers seen from that row (the yoco layer: its kv_only over the rows before it first)
    bool run_layer(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, Streamer& S, std::string& err,
                   const Run& run, int k, int il) {
        if (run.kv_only(k, il)) return layer_chunk(L0, mi, p0, T, wp, S, err, true);
        const int r0 = run.r0(k, il);
        if (r0 <= 0) return layer_chunk(L0, mi, p0, T, wp, S, err);
        if (il == run.yoco) {
            WinPlan w0 = wp;
            w0.snap_t = -1;
            if (!layer_chunk(L0, mi, p0, r0, w0, S, err, true)) return false;
        }
        Chunk& c = *chunks[(size_t) member(L0, mi).dev];
        float* const h0 = c.h;
        float* const pre0 = c.pre;
        c.h += (size_t) r0 * g.hc * g.dim;
        c.pre += (size_t) r0 * 8;
        WinPlan w1 = wp;
        w1.snap_t = wp.snap_t >= r0 ? wp.snap_t - r0 : -1;
        const bool ok = layer_chunk(L0, mi, p0 + r0, T - r0, w1, S, err);
        c.h = h0;
        c.pre = pre0;
        return ok;
    }

    // layer-major (c.lm: a run of several chunks, device di every trunk layer, the flow set up by run_dev_chunks):
    // layer after layer, each through all the chunks in order - a chunk's state to host memory after each of its
    // layers and back before the next - so a layer's experts are read from the files once for the whole run (the
    // layer cache keeps what a chunk read for the chunks after it), where chunk after chunk reads them once a chunk
    bool run_dev_layers(int di, const Run& run, std::string& err) {
        Dev& d = devs[(size_t) di];
        Chunk& c = *chunks[(size_t) di];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int D = g.dim, H = g.hc, nk = (int) run.T.size();
        float* const work_h = c.h;
        struct RestoreH {
            Chunk& c;
            float* work;
            ~RestoreH() { c.h = work; }
        } restore_h{c, work_h};   // success, Stop and any failed kernel/copy all restore the tracked work pointer
        Streamer S;   // (none: the flow streams)
        const size_t hb = chunk_hand_bytes();
        // the Engram rows: (layer, chunk) after (layer, chunk) in the run's order, each read on a helper thread while
        // the one before runs (slot u % 2, once unit u - 2's copy of it is done)
        std::vector<std::pair<int, int>> egu;   // (Engram layer, chunk)
        for (int il : c.fl_layers)
            if (L[(size_t) il].eg >= 0)
                for (int k = 0; k < nk; ++k)
                    if (run.runs(k, il)) egu.push_back({L[(size_t) il].eg, k});
        const size_t eg_per = (size_t) (pk.eg_ngram - 1) * pk.eg_heads * (pk.eg_hd + pk.eg_hd / 32);
        auto read_eg = [this, &c, &run, &egu, eg_per](size_t u) {
            if (u >= 2) cudaEventSynchronize(c.eg_ev[u % 2]);
            const int li = egu[u].first, k = egu[u].second;
            engram_rows(run.p0[(size_t) k], run.T[(size_t) k], li, c.eg_host[u % 2] + (size_t) li * c.C * eg_per);
        };
        size_t eg_u = 0;
        std::future<void> eg_next;
        if (!egu.empty()) eg_next = std::async(std::launch::async, read_eg, (size_t) 0);
        for (size_t li = 0; li < c.fl_layers.size(); ++li) {
            const int il = c.fl_layers[li];
            Layer& L0 = L[(size_t) il];
            const bool last_l = li + 1 == c.fl_layers.size();
            for (int k = 0; k < nk; ++k) {
                if (!run.runs(k, il)) continue;   // (a skipped early chunk: nothing after the yoco layer)
                if (halt_now()) return halt_end(di, err);   // (a Stop: one GPU, this one)
                const int64_t p0 = run.p0[(size_t) k];
                const int T = run.T[(size_t) k], slot = k % 2;
                const size_t h_row = (size_t) (p0 - run.p0[0]);
                const bool gpu_h = c.prefill_gpu_h != nullptr && h_row <= c.prefill_gpu_h_rows &&
                                   (size_t) T <= c.prefill_gpu_h_rows - h_row;
                c.h = gpu_h ? c.prefill_gpu_h + h_row * (size_t) H * D : work_h;
                const bool last_k = k + 1 == nk;
                WinPlan wp;
                wp.par = true;
                wp.async = true;
                wp.win = run.seq[(size_t) k];
                wp.k = k;
                wp.snap_t = run.snap_pos >= p0 && run.snap_pos < p0 + T ? (int) (run.snap_pos - p0) : -1;
                if (L0.eg >= 0) {   // this unit's rows ready, the next one's on their way
                    const double te = now_ms();
                    eg_next.get();
                    st.engram_ms += now_ms() - te;
                    wp.eg_slot = (int) (eg_u % 2);
                    if (++eg_u < egu.size()) eg_next = std::async(std::launch::async, read_eg, eg_u);
                }
                if (li == 0) {   // the chunk's input: its embedding rows
                    if (k >= 2) CK(cudaEventSynchronize(c.emb_ev[slot]));
                    for (int t = 0; t < T; ++t)
                        std::memcpy(c.emb_host[slot] + (size_t) t * D, emb_row(p0 + t, toks[(size_t) (p0 + t)]),
                                    (size_t) D * 2);
                    CK(cudaMemcpyAsync(c.emb, c.emb_host[slot], (size_t) T * D * 2, cudaMemcpyHostToDevice, s));
                    CK(cudaEventRecord(c.emb_ev[slot], s));
                    embed(c.emb, T, c.h, D, H, s);
                    init_pre(c.pre, T, s);
                } else {   // or its state after the layer before
                    const Xfer x = chunk_xfer(c.lm_state + (size_t) k * hb, c, T, false);
                    for (int i = gpu_h ? 1 : 0; i < x.n; ++i)
                        CK(cudaMemcpyAsync(x.dst[i], x.src[i], x.bytes[i], cudaMemcpyHostToDevice, s));
                    if (gpu_h) c.prefill_gpu_h_avoided += x.bytes[0];
                }
                if (last_k && g.n_mtp > 0) {   // the drafter's taps: the last chunk's last rows
                    const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
                    if (it != g.d_targets.end()) {
                        const size_t j = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
                        const int nr = std::min(T, tap_cap);
                        hc_mean(c.h + (size_t) (T - nr) * H * D, nr, H, D, d.tap, D, s);
                        CK(cudaStreamWaitEvent(s, ev_seed, 0));
                        CK(cudaMemcpy2DAsync(h_taps + j * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4,
                                             (size_t) nr, cudaMemcpyDeviceToHost, s));
                    }
                }
                const bool kvo = run.kv_only(k, il);
                if (!run_layer(L0, 0, p0, T, wp, S, err, run, k, il)) return false;
                if (dump_l != nullptr) dump_layer(d, c.h, il, p0, T);
                if (kvo) continue;   // (no layer after it reads this chunk's state)
                if (!last_l) {   // the chunk's state kept for the next layer
                    const Xfer x = chunk_xfer(c.lm_state + (size_t) k * hb, c, T, true);
                    for (int i = gpu_h ? 1 : 0; i < x.n; ++i)
                        CK(cudaMemcpyAsync(x.dst[i], x.src[i], x.bytes[i], cudaMemcpyDeviceToHost, s));
                    if (gpu_h) c.prefill_gpu_h_avoided += x.bytes[0];
                    continue;
                }
                // through every layer: the chunk's end, and the logits
                if ((size_t) k < chunk_done.size()) {
                    CK(cudaEventRecord(chunk_done[(size_t) k], s));
                    chunk_rec[(size_t) k].store(1, std::memory_order_release);
                }
                if (logits_all) {   // every row's logits (tests): the head over the chunk's rows, 16 at a time
                    const size_t base = (size_t) (p0 - run.p0[0]) * g.vocab;
                    for (int r0 = 0; r0 < T; r0 += kWin) {
                        const int nr = std::min(kWin, T - r0);
                        hc_pre_norm(c.h + (size_t) r0 * H * D, c.pre + (size_t) r0 * 8, d.norm_w, D, H, nr, g.eps, d.x, s);
                        if (!head_share(d, d.mi, d.x, nr, err)) return false;
                        CK(cudaMemcpyAsync(chunk_logits.data() + base + (size_t) r0 * g.vocab, d.logits,
                                           (size_t) nr * g.vocab * 4, cudaMemcpyDeviceToHost, s));
                        CK(cudaStreamSynchronize(s));
                    }
                } else if (last_k && n_logits > 0) {   // (the last rows' logits, into the window path's buffers)
                    const int t0 = T - n_logits;
                    hc_pre_norm(c.h + (size_t) t0 * H * D, c.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, s);
                    if (!head_share(d, d.mi, d.x, n_logits, err)) return false;
                }
            }
            lm_layers.store((int) li + 1, std::memory_order_release);
        }
        return true;
    }
    bool run_dev_chunks(int di, const Run& run, std::string& err) {
        Dev& d = devs[(size_t) di];
        Chunk& c = *chunks[(size_t) di];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int D = g.dim, H = g.hc, nk = (int) run.T.size();
        Streamer S;
        const int B = std::max(1, c.ring_n / Chunk::kParts);
        if (c.flow) {   // (the flow decides its streams as the layers route; nothing planned here)
            CK(cudaStreamSynchronize(s));
            CK(cudaStreamSynchronize(c.cs));
            c.items.clear();
            c.base = c.n_read = c.n_dma = c.n_comp = 0;
            c.n_file = 0;
            std::fill(c.stage_item.begin(), c.stage_item.end(), -1);
            c.fl_layers.clear();
            for (int il = 0; il < g.nl; ++il)   // (a group's layers: on each member)
                for (int m = 0; m < L[(size_t) il].nm; ++m)
                    if (member(L[(size_t) il], m).dev == di) c.fl_layers.push_back(il);
            c.fl_T = run.T;
            c.fl_yoco = run.yoco;
            c.fl_cut_k = run.cut_k;
            c.fl_cut_r = run.cut_r;
            c.spec_k = c.spec_il = -1;
            c.wait_ms = 0;
            c.it_file = c.it_ram = c.it_spec = c.it_spec_unused = c.it_lc = 0;
            if(register_ram_all||register_ram_mb>0) {
                c.registered_stream_begin=c.stream_bytes;
                c.registered_h2d_payload_bytes=c.registered_direct_bytes=c.registered_bounce_bytes=0;
            }
            if (c.lm_gpu != nullptr) {
                // A fresh forward run must never reuse an expert under a previous layer's blob layout. Both
                // streams were synchronized above. The arena itself remains reserved for later forward runs.
                c.lm_gpu_il = -1;
                c.lm_gpu_stride = c.lm_gpu_used = c.lm_gpu_peak_used = 0;
                c.lm_gpu_of.clear();
                c.lm_gpu_hits = c.lm_gpu_fills = c.lm_gpu_read_skips = c.lm_gpu_no_room = 0;
                c.lm_gpu_h2d_bytes = c.lm_gpu_d2d_bytes = c.lm_gpu_saved_h2d_bytes = 0;
            }
            c.lc_il = -1;
            c.lc_used = 0;
            c.lc_first = c.lc_chk = 0;
            c.io_d0 = c.io->direct_bytes();
            c.io_b0 = c.io->buffered_bytes();
            if (!c.fl_layers.empty()) {
                flow_spec(c, 0, c.fl_layers[0], run.T[0]);
                if (!flow_pump(c, err)) return false;
            }
            if (c.lm) return run_dev_layers(di, run, err);
        }
        for (int k = 0; k < nk && !c.flow; ++k)
            for (int il = 0; il < g.nl; ++il) {
                if (!run.runs(k, il) || run.kv_only(k, il)) continue;   // (a long prompt's skip: no experts there)
                const Layer& L0 = L[(size_t) il];
                bool here = false;
                for (int m = 0; m < L0.nm; ++m) here = here || member(L0, m).dev == di;
                if (!here) continue;
                for (int j0 = 0; j0 < (int) L0.host_list.size(); j0 += B)
                    S.b.push_back({k, il, j0, std::min(B, (int) L0.host_list.size() - j0)});
            }
        if (!S.b.empty() && c.ring == nullptr) {
            err = "no slots to stream experts into on CUDA" + std::to_string(d.id);
            return false;
        }
        std::atomic<bool> ra_stop{false};
        std::thread ra;
        if (file_experts && !S.b.empty()) ra = std::thread([this, &S, di, &ra_stop] { read_ahead(S, di, ra_stop); });
        struct Join {   // (every way out of the run: the read-ahead ends before S does)
            std::atomic<bool>& stop;
            std::thread& th;
            ~Join() {
                stop.store(true);
                if (th.joinable()) th.join();
            }
        } ra_join{ra_stop, ra};
        if (!copy_ahead(d, c, S, di, err)) return false;
        // this device's Engram layers: chunk k's rows read on a helper thread while chunk k - 1 runs (slot k % 2, once
        // chunk k - 2's copy of it is done)
        std::vector<int> egs;
        for (const Layer& l0 : L)
            for (int m = 0; m < l0.nm; ++m)
                if (member(l0, m).dev == di && !l0.mtp && l0.eg >= 0) egs.push_back(l0.eg);
        const size_t eg_per = (size_t) (pk.eg_ngram - 1) * pk.eg_heads * (pk.eg_hd + pk.eg_hd / 32);
        auto read_eg = [this, &c, &run, &egs, eg_per](int k) {
            if (k >= 2) cudaEventSynchronize(c.eg_ev[k % 2]);
            for (int li : egs)
                engram_rows(run.p0[(size_t) k], run.T[(size_t) k], li, c.eg_host[k % 2] + (size_t) li * c.C * eg_per);
        };
        std::future<void> eg_next;
        if (!egs.empty()) eg_next = std::async(std::launch::async, read_eg, 0);
        for (int k = 0; k < nk; ++k) {
            if (!chunk_go(k)) return halt_end(di, err);   // (a Stop: the chunks every device reaches are done)
            const int64_t p0 = run.p0[(size_t) k];
            const int T = run.T[(size_t) k], slot = k % 2;
            const bool last_k = k + 1 == nk;
            if (!egs.empty()) {   // chunk k's rows ready, chunk k + 1's on their way
                const double te = now_ms();
                eg_next.get();
                if (d.mi == 0) st.engram_ms += now_ms() - te;
                if (k + 1 < nk) eg_next = std::async(std::launch::async, read_eg, k + 1);
            }
            WinPlan wp;
            wp.par = true;
            wp.async = true;
            wp.win = run.seq[(size_t) k];
            wp.k = k;
            wp.snap_t = run.snap_pos >= p0 && run.snap_pos < p0 + T ? (int) (run.snap_pos - p0) : -1;
            bool started = false;
            std::vector<char> got(srcs.size(), 0);
            for (int il = 0; il < g.nl && run.runs(k, il); ++il) {
                if (halt_now()) return halt_end(di, err);   // (one GPU: after the layer it is on)
                Layer& L0 = L[(size_t) il];
                int mi = -1;
                for (int m = 0; m < L0.nm; ++m)
                    if (member(L0, m).dev == di) mi = m;
                if (mi < 0) continue;
                if (!started) {   // the chunk's input: its embedding rows, or the previous group's hand-off
                    started = true;
                    if (il == 0) {
                        if (k >= 2) CK(cudaEventSynchronize(c.emb_ev[slot]));
                        for (int t = 0; t < T; ++t)
                            std::memcpy(c.emb_host[slot] + (size_t) t * D, emb_row(p0 + t, toks[(size_t) (p0 + t)]),
                                        (size_t) D * 2);
                        CK(cudaMemcpyAsync(c.emb, c.emb_host[slot], (size_t) T * D * 2, cudaMemcpyHostToDevice, s));
                        CK(cudaEventRecord(c.emb_ev[slot], s));
                        embed(c.emb, T, c.h, D, H, s);
                        init_pre(c.pre, T, s);
                    } else {
                        Chunk& a = *chunks[(size_t) L[(size_t) il - 1].dev];
                        xfer_get(chunk_xfer(a.hand[slot], c, T, false), a.hand_flag[slot], wp.win, s);
                        set_flag(a.hand_ack + ((size_t) slot * 8 + mi) * 32, wp.win, s);
                    }
                }
                for (int sidx : {L0.src, L0.ix_src ? L0.ik_src : -1}) {
                    if (sidx < 0 || got[(size_t) sidx]) continue;
                    got[(size_t) sidx] = 1;
                    Src& sr = srcs[(size_t) sidx];
                    if (L[(size_t) sr.layer].grp == L0.grp) continue;
                    const int64_t g_first = p0 / sr.ratio;
                    const int ng = (int) std::max<int64_t>(0, (p0 + T) / sr.ratio - g_first);
                    Xfer x;
                    if (ng > 0 && sr.ckv[(size_t) di] != nullptr) {
                        x.src[x.n] = sr.stage(slot);
                        x.dst[x.n] = sr.ckv[(size_t) di] + (size_t) g_first * ckv_rb();
                        x.bytes[x.n++] = (size_t) ng * ckv_rb();
                    }
                    if (ng > 0 && sr.ik[(size_t) di] != nullptr) {
                        x.src[x.n] = sr.stage(slot) + x_cap() * ckv_rb();
                        x.dst[x.n] = sr.ik[(size_t) di] + (size_t) g_first * ik_rb();
                        x.bytes[x.n++] = (size_t) ng * ik_rb();
                    }
                    if (x.n > 0) xfer_get(x, sr.flag(slot), wp.win, s);
                    set_flag(sr.x_ack + ((size_t) slot * devs.size() + di) * 32, wp.win, s);   // (taken, or none)
                }
                if (last_k && mi == 0 && g.n_mtp > 0) {   // the drafter's taps: the last chunk's last rows
                    const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
                    if (it != g.d_targets.end()) {
                        const size_t j = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
                        const int nr = std::min(T, tap_cap);
                        hc_mean(c.h + (size_t) (T - nr) * H * D, nr, H, D, d.tap, D, s);
                        CK(cudaStreamWaitEvent(s, ev_seed, 0));
                        CK(cudaMemcpy2DAsync(h_taps + j * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4,
                                             (size_t) nr, cudaMemcpyDeviceToHost, s));
                    }
                }
                if (!run_layer(L0, mi, p0, T, wp, S, err, run, k, il)) return false;
                if (dump_l != nullptr && mi == 0) dump_layer(d, c.h, il, p0, T);
                const bool last = il + 1 == g.nl || L[(size_t) il + 1].grp != L0.grp;
                if (last && mi == 0 && il + 1 < g.nl) {   // the hand-off, once the next group took the chunk before last
                    const int nnext = L[(size_t) il + 1].nm;
                    if (k >= 2) wait_flags(c.hand_ack + (size_t) slot * 8 * 32, nnext, 32, wp.win - 2, s);
                    xfer_put(chunk_xfer(c.hand[slot], c, T, true), c.hand_flag[slot], wp.win, d.xcnt, s);
                }
            }
            if (di == head_dev && (size_t) k < chunk_done.size()) {
                CK(cudaEventRecord(chunk_done[(size_t) k], s));
                chunk_rec[(size_t) k].store(1, std::memory_order_release);
            }
            if (d.grp == devs[(size_t) head_dev].grp) {   // the head group: each member its share of the logits
                if (logits_all) {   // every row's logits (tests): the head over the chunk's rows, 16 at a time
                    const size_t base = (size_t) (p0 - run.p0[0]) * g.vocab;
                    for (int r0 = 0; r0 < T; r0 += kWin) {
                        const int nr = std::min(kWin, T - r0);
                        hc_pre_norm(c.h + (size_t) r0 * H * D, c.pre + (size_t) r0 * 8, d.norm_w, D, H, nr, g.eps, d.x, s);
                        if (!head_share(d, d.mi, d.x, nr, err)) return false;
                        if (di != head_dev) continue;
                        CK(cudaMemcpyAsync(chunk_logits.data() + base + (size_t) r0 * g.vocab, d.logits,
                                           (size_t) nr * g.vocab * 4, cudaMemcpyDeviceToHost, s));
                        CK(cudaStreamSynchronize(s));
                    }
                } else if (last_k && n_logits > 0) {   // (the last rows' logits, into the window path's buffers)
                    const int t0 = T - n_logits;
                    hc_pre_norm(c.h + (size_t) t0 * H * D, c.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, s);
                    if (!head_share(d, d.mi, d.x, n_logits, err)) return false;
                }
            }
        }
        return true;
    }

    // ---- a prompt's path: the flow or windows of 16
    // this PC's speeds for pick_path (every device on the flow, the experts in place): the DMA from pinned staging to a
    // GPU; the readers copying RAM into staging; the disk read by them (experts not in RAM); the CPU lane on experts in
    // RAM and on ones it reads from the disk first - each timed once, a few experts' worth (DS4_PATH_PICK=0: always the
    // flow, as before)
    void calibrate_paths() {
        if (chunk <= 0 || chunk_min != kFlowChunkMin || std::getenv("DS4_CHUNK_MIN") != nullptr) return;
        if (const char* v = std::getenv("DS4_PATH_PICK"))
            if (std::atoi(v) == 0) return;
        int di = -1;
        for (size_t i = 0; i < chunks.size() && di < 0; ++i)
            if (chunks[i] != nullptr && chunks[i]->flow && chunks[i]->S >= 2) di = (int) i;
        if (di < 0) return;
        Chunk& c = *chunks[(size_t) di];
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        const double t0 = now_ms();
        // the DMA: staging -> the ring (once to warm up, once timed)
        const size_t nb = std::min({(size_t) 256 << 20, (size_t) c.S * c.stage_stride, (size_t) c.ring_n * c.ring_stride});
        double ms = 0;
        for (int rep = 0; rep < 2; ++rep) {
            const double a = now_ms();
            if (cudaMemcpyAsync(c.ring, c.stage, nb, cudaMemcpyHostToDevice, c.cs) != cudaSuccess ||
                cudaStreamSynchronize(c.cs) != cudaSuccess) {
                cudaGetLastError();
                return;
            }
            ms = now_ms() - a;
        }
        pm.dma = (double) nb / std::max(1e-3, ms / 1000.0);
        // the readers: RAM -> staging (a buffer of its own, touched first), in as many jobs as they are
        auto run_jobs = [&](std::vector<IoQueue::Job>& jobs, std::atomic<int>& left) {
            const double a = now_ms();
            left.store((int) jobs.size());
            c.io->push(jobs.data(), (int) jobs.size());
            while (left.load(std::memory_order_acquire) != 0) c.io->wait_any(500);
            return now_ms() - a;
        };
        {
            const size_t part = std::min(nb, (size_t) 64 << 20) / 8 / 4096 * 4096;
            std::vector<uint8_t> src(part * 8, 1);
            std::vector<IoQueue::Job> jobs(8);
            std::atomic<int> left{0};
            std::atomic<bool> bad{false};
            for (int i = 0; i < 8; ++i) {
                jobs[(size_t) i].mem = src.data() + (size_t) i * part;
                jobs[(size_t) i].dst = c.stage + (size_t) i * part;
                jobs[(size_t) i].len = part;
                jobs[(size_t) i].left = &left;
                jobs[(size_t) i].bad = &bad;
            }
            run_jobs(jobs, left);
            const double t1 = run_jobs(jobs, left), t2 = run_jobs(jobs, left);
            pm.copy = (double) (part * 8) / std::max(1e-3, std::min(t1, t2) / 1000.0);
        }
        // experts not in RAM (none of their sampled pages resident), of the trunk layer with the most CPU-lane ones:
        // up to min(8, staging slots) read by the readers as the flow reads them, then 8 others through the CPU lane
        // three times (from the disk, then in RAM: the faster of two)
        int best = -1, most = 0;
        for (int il = 0; il < g.nl; ++il) {
            int n = 0;
            for (int e = 0; e < L[(size_t) il].n_exp; ++e) n += L[(size_t) il].slot[(size_t) e] < 0;
            if (n > most) most = n, best = il;
        }
        if (best < 0) return;
        const Layer& l = L[(size_t) best];
        std::vector<int> cold;
        const int n_read = std::min(8, c.S), n_cpu = 8;
        for (int e = 0; e < l.n_exp && (int) cold.size() < n_read + n_cpu; ++e) {
            if (l.slot[(size_t) e] >= 0 || (!l.ram.empty() && l.ram[(size_t) e] >= 0)) continue;
            bool res = false;
            for (int r = 0; r < 3; ++r) res = res || os::resident_sampled(mx_mat(l, e, r), mx_part_bytes(l, r), 4);
            if (!res) cold.push_back(e);
        }
        if ((int) cold.size() >= n_read + n_cpu) {
            std::vector<IoQueue::Job> jobs;
            std::atomic<int> left{0};
            std::atomic<bool> bad{false};
            for (int i = 0; i < n_read; ++i) {
                uint8_t* sb = c.stage + (size_t) i * c.stage_stride;
                for (int r = 0; r < 3; ++r) {
                    IoQueue::Job j;
                    const os::fd_t pf = pk.pfds.empty() ? os::kBadFd : pk.pfds[(size_t) l.mx_file[r]];
                    j.fd = pf != os::kBadFd ? pf : pk.fds[(size_t) l.mx_file[r]];
                    j.dfd = pk.dfds.empty() ? os::kBadFd : pk.dfds[(size_t) l.mx_file[r]];
                    j.off = l.mx_off[r] + (uint64_t) cold[(size_t) i] * l.mx_stride[r];
                    j.len = mx_part_bytes(l, r);
                    j.dst = sb;
                    j.left = &left;
                    j.bad = &bad;
                    j.map = pk.at(l.mx_file[r], j.off);
                    jobs.push_back(j);
                    sb += flow_region(r);
                }
            }
            const double rd = run_jobs(jobs, left);
            if (!bad.load()) pm.disk = (double) n_read * (double) l.blob / std::max(1e-3, rd / 1000.0);
        }
        {   // the CPU lane: entries of one token (as most of a window's are), from the disk (if any were cold), then RAM
            const size_t first = (int) cold.size() >= n_read + n_cpu ? (size_t) n_read : 0;
            std::vector<Ent> ents;
            for (int i = 0; i < n_cpu; ++i) {
                Ent en;
                en.e = first + (size_t) i < cold.size() ? cold[first + (size_t) i] : -1;
                if (en.e < 0)   // (no cold ones: the first CPU-lane experts, wherever they are)
                    for (int e = 0, k = 0; e < l.n_exp && en.e < 0; ++e)
                        if (l.slot[(size_t) e] < 0 && k++ == i) en.e = e;
                if (en.e < 0) break;
                en.n = 1;
                en.slot0 = i;
                en.tok[0] = 0;
                ents.push_back(en);
            }
            if (ents.empty()) return;
            std::vector<float> x((size_t) g.dim);
            for (size_t i = 0; i < x.size(); ++i) x[i] = 0.02f * std::sin(0.37f * (float) i);
            const double bytes = (double) ents.size() * (double) l.blob;
            double a = now_ms();
            cpu_experts(l, x.data(), ents, (size_t) g.dim);
            const double t_first = now_ms() - a;
            double t_warm = 1e30;
            for (int rep = 0; rep < 2; ++rep) {
                a = now_ms();
                cpu_experts(l, x.data(), ents, (size_t) g.dim);
                t_warm = std::min(t_warm, now_ms() - a);
            }
            pm.cpu = bytes / std::max(1e-3, t_warm / 1000.0);
            if (first > 0 || (int) cold.size() >= n_cpu) pm.cpu_cold = bytes / std::max(1e-3, t_first / 1000.0);
        }
        pm.on = pm.dma > 0 && pm.copy > 0 && pm.cpu > 0;
        if (o.verbose) {
            char dk[32] = "not measured", cc[32] = "not measured";
            if (pm.disk > 0) std::snprintf(dk, sizeof dk, "%.2f GB/s", pm.disk / 1e9);
            if (pm.cpu_cold > 0) std::snprintf(cc, sizeof cc, "%.2f GB/s", pm.cpu_cold / 1e9);
            std::fprintf(stderr,
                         "ds4: prompts through the flow or windows by this PC's speeds: DMA %.1f GB/s, RAM to staging "
                         "%.1f GB/s, the disk %s; the CPU lane %.1f GB/s in RAM, %s from the disk (%.0f ms)\n",
                         pm.dma / 1e9, pm.copy / 1e9, dk, pm.cpu / 1e9, cc, now_ms() - t0);
        }
    }
    // layer l's CPU-lane experts (VRAM holds the rest) and the chance a token picks a given one of them: the profile's
    // share of the layer's picks they took, spread evenly over them, times pm.m (what the flow's runs found)
    double pm_rate(const Layer& l, int& n_cold) const {
        const uint32_t* pr = prof.data() + (size_t) l.il * g.n_exp;
        double sum = 0, cold = 0;
        n_cold = 0;
        for (int e = 0; e < l.n_exp; ++e) {
            sum += pr[e];
            if (l.slot[(size_t) e] < 0) cold += pr[e], ++n_cold;
        }
        if (n_cold == 0) return 0.0;
        const double share = sum > 0 && cold > 0 ? cold / sum : (double) n_cold / l.n_exp;
        return pm.m * (double) l.k_exp * share / n_cold;
    }
    // the CPU-lane experts n tokens pick in the trunk's layers, expected (each layer's n_cold (1 - e^(-rate n)))
    double pm_picks(int n) const {
        double d = 0;
        for (int il = 0; il < g.nl; ++il) {
            int nc = 0;
            const double a = pm_rate(L[(size_t) il], nc);
            d += nc * (1.0 - std::exp(-a * n));
        }
        return d;
    }
    // a run of one chunk of n tokens on the flow picked `picked` CPU-lane experts: pm.m to what makes the expectation
    // that (bisection over its log; a running mean of the log)
    void pm_learn(int n, double picked) {
        if (!pm.on || n < 2 || picked < 1) return;
        const double m0 = pm.m;
        double lo = std::log(1e-3), hi = std::log(1e3);
        for (int it = 0; it < 40; ++it) {
            pm.m = std::exp(0.5 * (lo + hi));
            (pm_picks(n) < picked ? lo : hi) = 0.5 * (lo + hi);
        }
        const double obs = 0.5 * (lo + hi);
        pm.m = std::exp(pm.n_m == 0 ? obs : 0.7 * std::log(m0) + 0.3 * obs);
        ++pm.n_m;
    }
    // a prompt of n tokens that the flow could take (n >= chunk_min): true the flow, false windows of 16 - each path's
    // time predicted from the bytes of CPU-lane experts it would move (the flow: each picked one once, n_cold (1 -
    // e^(-rate n)) a layer, pm_rate) or compute (windows: each window's own picks), those in RAM (each layer's share:
    // sampled pages of up to 32 of its CPU-lane experts, mincore / QueryWorkingSetEx; the RAM tier's) and those not at
    // this PC's speeds - the flow's times what it really took against its predictions so far for prompts of that size
    // (pm_k), windows' once some ran what a window took times the windows; the flow unless windows are predicted
    // faster (by half, with less than kPickRes in RAM); said with --verbose
    bool pick_path(int n) {
        pm.last = -1;
        if (!pm.on || n >= kPickMax) return true;
        double f_res = 0, f_cold = 0, w = 0;
        const int nw = n / kWin, rem = n % kWin;
        for (int il = 0; il < g.nl; ++il) {
            const Layer& l = L[(size_t) il];
            int nc = 0;
            const double a = pm_rate(l, nc);
            if (nc == 0) continue;
            int sampled = 0, res = 0, seen = 0;
            const int step = std::max(1, nc / 32);
            for (int e = 0; e < l.n_exp; ++e) {
                if (l.slot[(size_t) e] >= 0 || seen++ % step != 0) continue;
                bool r = !l.ram.empty() && l.ram[(size_t) e] >= 0;
                if (!r) {
                    r = true;
                    for (int p = 0; p < 3 && r; ++p) r = os::resident_sampled(mx_mat(l, e, p), mx_part_bytes(l, p), 2);
                }
                ++sampled;
                res += r;
            }
            const double share = sampled > 0 ? (double) res / sampled : 0.0, b = (double) l.blob;
            auto u = [&](int t) { return nc * (1.0 - std::exp(-a * t)); };
            f_res += share * u(n) * b;
            f_cold += (1.0 - share) * u(n) * b;
            w += ((double) nw * u(kWin) + (rem > 0 ? u(rem) : 0.0)) * b;
        }
        const double r_fr = std::min(pm.copy, pm.dma), r_fc = pm.disk > 0 ? std::min(pm.disk, pm.dma) : r_fr;
        const double r_wc = pm.cpu_cold > 0 ? pm.cpu_cold : 1.0 / (1.0 / r_fc + 1.0 / pm.cpu);
        const double t_flow = f_res / r_fr + f_cold / r_fc;
        const double t_win = std::max(0.0, w - f_cold) / pm.cpu + f_cold / r_wc;
        const int cls = pm_cls(n), n_win = (n + kWin - 1) / kWin;
        // (windows: once some ran, what a window took on average times the windows the prompt needs - each pays the
        // fixed costs of every layer on every GPU again, which a pass of the flow pays once)
        const double pf = pm_k(cls) * t_flow, pw = pm.win_n > 0 ? pm.win_s * n_win : t_win;
        // (the flow unless windows are predicted faster - in half its time at least, unless kPickRes of the CPU-lane
        // experts' bytes are in RAM: the flow wins on every PC whose experts come from an SSD, where a wrong pick would
        // cost the most; windows win where they are all in RAM)
        const double res_share = f_res / std::max(1.0, f_res + f_cold);
        const bool flow = !(pw < pf && (res_share >= kPickRes || 2.0 * pw <= pf));
        pm.last = flow ? 1 : 0;
        pm.last_cls = cls;
        pm.last_nw = n_win;
        pm.last_pred = flow ? t_flow : t_win;
        if (o.verbose)
            std::fprintf(stderr,
                         "ds4: a prompt of %d tokens through %s: predicted %.2f s, %s %.2f s (%.0f%% of its CPU-lane "
                         "experts' %.1f GB in RAM)\n",
                         n, flow ? "the flow" : "windows", flow ? pf : pw, flow ? "windows" : "the flow", flow ? pw : pf,
                         100.0 * res_share, (f_res + f_cold) / 1e9);
        return flow;
    }
    // a prompt's size class for pm.k: 2-31, 32-127, 128-511, 512 tokens and more
    static int pm_cls(int n) { return n < 32 ? 0 : n < 128 ? 1 : n < 512 ? 2 : 3; }
    // the flow's time factor for size class c: its own, else its nearest class's, else 1 (never the windows': theirs
    // is mostly the fixed costs a window pays again and again)
    double pm_k(int c) const {
        for (int dc = 0; dc < 4; ++dc)
            for (int cc : {c - dc, c + dc})
                if (cc >= 0 && cc < 4 && pm.kn[0][cc] > 0) return pm.k[0][cc];
        return 1.0;
    }
    // n tokens through windows of 16, the last one's logits_rows rows of logits
    bool feed_windows(const int* tokens, int n, int logits_rows, std::string& err) {
        logits_all = false;
        for (int a = 0; a < n; a += kWin) {
            const int T = std::min(kWin, n - a);
            if (!forward(tokens + a, T, a + T == n ? logits_rows : 0, err)) return false;
        }
        return true;
    }
    // the picked path took ms: the flow's predictions for the prompt's size class scaled by what it really takes (taken
    // / predicted), windows' time a window - running means
    void path_done(double ms) {
        if (pm.last < 0 || pm.last_pred <= 0) return;
        if (pm.last == 0) {
            const double w = ms / 1000.0 / std::max(1, pm.last_nw);
            pm.win_s = pm.win_n == 0 ? w : 0.7 * pm.win_s + 0.3 * w;
            ++pm.win_n;
        } else {
            const double ratio = std::min(8.0, std::max(0.25, ms / 1000.0 / pm.last_pred));
            double& k = pm.k[0][pm.last_cls];
            int64_t& n = pm.kn[0][pm.last_cls];
            k = n == 0 ? ratio : 0.7 * k + 0.3 * ratio;
            ++n;
        }
        pm.last = -1;
    }
    // a run layer-major? - one device holding every layer on the flow, several chunks, and what the chunks after the
    // first would read from the files again more than the chunks' states moved to host memory and back at every layer
    // (DS4_LAYER_MAJOR=0: never); then its host memory: the chunks' states (pinned, else pageable) and the layer cache,
    // the largest layer's experts read from the files if RAM allows (ram_avail() less kRamMargin; DS4_LAYER_CACHE_MB
    // sets it) - at least one expert, or not layer-major
    bool lm_begin(const Run& run) {
        static const char* env = std::getenv("DS4_LAYER_MAJOR");
        if (devs.size() != 1 || run.T.size() < 2 || (env != nullptr && std::atoi(env) == 0)) return false;
        Chunk& c = *chunks[0];
        if (!c.flow) return false;
        size_t need = 0, n_exp = 0, blob_max = 0;
        double file_b = 0;
        for (int il = 0; il < g.nl; ++il) {
            const Layer& l = L[(size_t) il];
            size_t ne = 0;
            for (int e : l.host_list)
                if (l.ram.empty() || l.ram[(size_t) e] < 0) ++ne;
            const size_t stride = (l.blob + 63) & ~(size_t) 63;
            need = std::max(need, ne * stride);
            n_exp = std::max(n_exp, (size_t) l.n_exp);
            blob_max = std::max(blob_max, stride);
            file_b += (double) ne * (double) l.blob;
        }
        int64_t rows = 0;
        for (int T : run.T) rows += T;
        const size_t hb = chunk_hand_bytes(), state_b = run.T.size() * hb;
        const double moved = 2.0 * (double) rows * ((double) hb / (double) chunk) * (double) g.nl;
        if (need == 0) return false;
        // The explicit GPU-cache experiment prefers layer-major even where the original host-I/O-only
        // heuristic selected chunk-major: repeated expert H2D is now avoided as well. Keep every RAM check
        // below, and keep the original decision byte for byte when no usable GPU cache was requested.
        const bool gpu_phase = c.lm_gpu != nullptr && c.lm_gpu_bytes >= blob_max;
        if (!gpu_phase && (double) (run.T.size() - 1) * file_b <= moved) return false;
        size_t lc_bytes = need;
        if (const char* mb = std::getenv("DS4_LAYER_CACHE_MB")) lc_bytes = std::min(need, (size_t) std::atoll(mb) << 20);
        else {
            const uint64_t avail = ram_avail();
            if (avail != ~uint64_t{0})
                lc_bytes = std::min<uint64_t>(need, avail > kRamMargin + state_b ? avail - kRamMargin - state_b : 0);
        }
        if (lc_bytes < blob_max) return false;
        void* st_p = nullptr;
        c.lm_pinned = cudaHostAlloc(&st_p, state_b, cudaHostAllocPortable) == cudaSuccess;
        if (!c.lm_pinned) {
            cudaGetLastError();
            st_p = os::anon_alloc(state_b, -1, 1);
            if (st_p == nullptr) return false;
        }
        c.lc = (uint8_t*) os::anon_alloc(lc_bytes, -1, 1);
        if (c.lc == nullptr) {
            if (c.lm_pinned) cudaFreeHost(st_p);
            else os::anon_free(st_p, state_b);
            return false;
        }
        c.lm_state = (uint8_t*) st_p;
        c.lm_state_bytes = state_b;
        c.lc_bytes = lc_bytes;
        c.lc_of.assign(n_exp, -1);
        c.lc_writer.assign(n_exp, -1);
        c.lm = true;
        lm_layers.store(0);
        return true;
    }
    void lm_end() {
        for (size_t di = 0; di < chunks.size(); ++di) {
            Chunk& c = *chunks[di];
            if (!c.lm) continue;
            cudaSetDevice(devs[di].id);
            cudaStreamSynchronize(devs[di].st);   // (nothing of the run's still reads or writes them: the device's
            cudaStreamSynchronize(c.cs);          // copies done, every read the run issued done - a failed run's too)
            for (size_t i = c.base; i < c.n_read; ++i)
                while (fit(c, i).left.load(std::memory_order_acquire) != 0) c.io->wait_any(500);
            if (c.lm_pinned) cudaFreeHost(c.lm_state);
            else os::anon_free(c.lm_state, c.lm_state_bytes);
            os::anon_free(c.lc, c.lc_bytes);
            c.lm_state = c.lc = nullptr;
            c.lm_state_bytes = c.lc_bytes = 0;
            c.lm = c.lm_pinned = false;
        }
    }

    // First-cold experiment ONLY. All run workers have returned, lm_end has
    // joined readers, streams have synchronized, and the one-shot H/cache arenas
    // have retired. Retain the Chunk object, I/O, events, BLAS, Dev/Layer/Src KV,
    // DSpark taps/rings and checkpoints. Future prompts use the smaller chunk.
    bool shrink_prefill_chunk(Dev& d, Chunk& c, std::string& err) {
        if (prefill_decode_chunk == 0 || c.C <= prefill_decode_chunk) return true;
        const size_t oldC = (size_t)c.C, newC = (size_t)prefill_decode_chunk;
        const int oldA = c.A, newA = std::min(c.A, prefill_decode_chunk);
        const size_t D = (size_t)g.dim, H = (size_t)g.hc, HD = (size_t)g.hd, FF = (size_t)g.ff;
        if (devs.size() != 1 || c.C != chunk || c.lm || c.lm_state || c.lc || c.prefill_gpu_h || c.lm_gpu || c.lm_f16 || c.lm_f16_y || c.lm_f16_meta_dev ||
            c.lm_f16_meta_host || c.lm_f16_blas || d.experts != 0) {
            err = "decode chunk shrink requires a retired, quiescent first-prefill phase";
            return false;
        }
        std::vector<shrink::Buffer> buffers;
        auto add = [&](const char* name, auto*& pointer, size_t before, size_t after, shrink::Kind kind = shrink::Kind::device) {
            if (pointer == nullptr || before == after) return;
            const size_t item = sizeof(*pointer);
            size_t ob = before * item, nb = after * item;
            if (kind == shrink::Kind::device) ob = align256(ob), nb = align256(nb);
            else ob += 64, nb += 64; // halloc/hmap's exact64-byte suffix
            buffers.push_back({name, (void*)&pointer, (void*)pointer, ob, nb, kind, nullptr});
        };
        // Proven C-scaled allocations, matching alloc_chunks's element counts.
        add("h", c.h, oldC*H*D, newC*H*D); add("x", c.x, oldC*D, newC*D);
        add("acc", c.acc, oldC*D, newC*D); add("x16", c.x16, oldC*D, newC*D);
        add("qr", c.qr, oldC*g.q_lora, newC*g.q_lora); add("qr16", c.qr16, oldC*g.q_lora, newC*g.q_lora);
        add("kvr", c.kvr, oldC*HD, newC*HD); add("ckr", c.ckr, oldC*HD, newC*HD); add("cgr", c.cgr, oldC*HD, newC*HD);
        add("latent", c.latent, (oldC+2)*HD, (newC+2)*HD); add("lat16", c.lat16, (oldC+2)*HD, (newC+2)*HD);
        add("ikr", c.ikr, (oldC+2)*g.ix_hd, (newC+2)*g.ix_hd);
        add("ext", c.ext, (oldC+g.win)*HD, (newC+g.win)*HD);
        add("iw", c.iw, oldC*64, newC*64); add("topk", c.topk, oldC*topk_stride(), newC*topk_stride());
        add("cand", c.cand, oldC*cand_stride(), newC*cand_stride()); add("mix", c.mix, oldC*32, newC*32);
        add("a_pre", c.a_pre, oldC*8, newC*8); add("a_post", c.a_post, oldC*8, newC*8);
        add("a_comb", c.a_comb, oldC*64, newC*64); add("f_pre", c.f_pre, oldC*8, newC*8);
        add("f_post", c.f_post, oldC*8, newC*8); add("f_comb", c.f_comb, oldC*64, newC*64);
        add("pre", c.pre, oldC*8, newC*8); add("glog", c.glog, oldC*1024, newC*1024);
        add("ridx", c.ridx, oldC*kPicks, newC*kPicks); add("rw", c.rw, oldC*kPicks, newC*kPicks);
        add("row_tok", c.row_tok, oldC*kPicks, newC*kPicks); add("pos", c.pos, oldC*kPicks, newC*kPicks);
        add("sg", c.sg, oldC*FF, newC*FF); add("su", c.su, oldC*FF, newC*FF);
        add("sh16", c.sh16, oldC*FF, newC*FF); add("h16", c.h16, oldC*FF, newC*FF);
        add("y", c.y, oldC*D, newC*D); add("emb", c.emb, oldC*D, newC*D);
        const size_t cols = (size_t)(pk.eg_ngram-1)*pk.eg_heads, per = cols*(pk.eg_hd+pk.eg_hd/32);
        add("eg_rows", c.eg_rows, oldC*per, newC*per);
        const size_t NHL = (size_t)g.heads / head_members(1), HQ = NHL*HD, OG = (size_t)g.o_groups / head_members(1);
        add("q", c.q, (size_t)oldA*HQ, (size_t)newA*HQ); add("o", c.o, (size_t)oldA*HQ, (size_t)newA*HQ);
        add("o16", c.o16, (size_t)oldA*HQ, (size_t)newA*HQ);
        add("oa", c.oa, (size_t)oldA*OG*g.o_lora, (size_t)newA*OG*g.o_lora);
        add("oa16", c.oa16, (size_t)oldA*OG*g.o_lora, (size_t)newA*OG*g.o_lora);
        auto hand_bytes = [&](size_t n) { return n*(H*D*4 + 8*4 + topk_stride()*4 + cand_stride()*4); };
        for (int slot = 0; slot < 2; ++slot) {
            add("emb_host", c.emb_host[slot], oldC*D, newC*D, shrink::Kind::pinned);
            add("eg_host", c.eg_host[slot], oldC*pk.eg_layers*per, newC*pk.eg_layers*per, shrink::Kind::pinned);
            add("hand", c.hand[slot], hand_bytes(oldC), hand_bytes(newC), shrink::Kind::mapped);
        }
        // Reserve owner capacity BEFORE the first allocation. Every new pointer
        // is tracked immediately, including failed zeroing/rollback cases.
        size_t device_count = 0, host_count = 0;
        for (const auto& b : buffers) if (b.kind == shrink::Kind::device) ++device_count; else ++host_count;
        try { d.allocs.reserve(d.allocs.size()+device_count); host_allocs.reserve(host_allocs.size()+host_count); }
        catch (...) {
            std::fprintf(stderr, "ds4: experimental decode chunk shrink skipped: owner ledger capacity unavailable; original C=%zu retained\n", oldC);
            return true;
        }
        bool allocation_hard_error = false;
        shrink::Callbacks ops;
        ops.owns_old = [&](const shrink::Buffer& b) {
            const auto& owner = b.kind == shrink::Kind::device ? d.allocs : host_allocs;
            return std::count(owner.begin(), owner.end(), b.old_pointer) == 1;
        };
        ops.cancelled = [&] { return cancel && cancel(); }; // caller thread only
        ops.allocate = [&](shrink::Buffer& b) {
            cudaError_t status;
            if (b.kind == shrink::Kind::device) {
                status = cudaMalloc(&b.next, b.new_bytes);
                if (status == cudaSuccess) { d.allocs.push_back(b.next); d.scratch += b.new_bytes; }
                if (status == cudaSuccess) status = cudaMemsetAsync(b.next, 0, b.new_bytes, d.st);
                if (status == cudaSuccess) status = cudaStreamSynchronize(d.st);
            } else {
                const unsigned flags = cudaHostAllocPortable | (b.kind == shrink::Kind::mapped ? cudaHostAllocMapped : 0);
                status = cudaHostAlloc(&b.next, b.new_bytes, flags);
                if (status == cudaSuccess) host_allocs.push_back(b.next);
                if (status == cudaSuccess) std::memset(b.next, 0, b.new_bytes);
            }
            if (status == cudaErrorMemoryAllocation) (void)cudaGetLastError(); // recoverable optional allocation
            else if (status != cudaSuccess) allocation_hard_error = true;
            return status == cudaSuccess;
        };
        ops.discard_new = [&](const shrink::Buffer& b) {
            auto& owner = b.kind == shrink::Kind::device ? d.allocs : host_allocs;
            auto it = std::find(owner.begin(), owner.end(), b.next);
            if (it == owner.end()) return false;
            const cudaError_t status = b.kind == shrink::Kind::device ? cudaFree(b.next) : cudaFreeHost(b.next);
            if (status != cudaSuccess) return false; // retain owner entry for terminal cleanup
            owner.erase(it);
            if (b.kind == shrink::Kind::device) d.scratch -= b.new_bytes;
            return true;
        };
        ops.reserve_publication = [&](const std::vector<shrink::Buffer>& bs) {
            size_t nd = 0, nh = 0, old_b = 0, new_b = 0;
            for (const auto& b : bs) if (b.kind == shrink::Kind::device) ++nd, old_b += b.old_bytes, new_b += b.new_bytes; else ++nh;
            if (old_b > ~(size_t)0-new_b || d.scratch < old_b+new_b) return false;
            for (const auto& b : bs) {
                const auto& owner = b.kind == shrink::Kind::device ? d.allocs : host_allocs;
                if (std::count(owner.begin(), owner.end(), b.old_pointer) != 1 || std::count(owner.begin(), owner.end(), b.next) != 1) return false;
            }
            return true; // no allocation remains in publication
        };
        ops.publish = [&](const std::vector<shrink::Buffer>& bs) {
            // COMMIT linearization: allocations initialized, ledgers reserved;
            // pointer+shape publication cannot fail. Stop AFTER here keeps this
            // valid small layout; no asynchronous writer remains on old buffers.
            for (const auto& b : bs) std::memcpy(b.slot, &b.next, sizeof(b.next));
            chunk = c.C = prefill_decode_chunk;
            c.A = newA;
        };
        ops.retire_old = [&](const shrink::Buffer& b) {
            auto& owner = b.kind == shrink::Kind::device ? d.allocs : host_allocs;
            auto it = std::find(owner.begin(), owner.end(), b.old_pointer);
            if (it == owner.end()) return false;
            const cudaError_t status = b.kind == shrink::Kind::device ? cudaFree(b.old_pointer) : cudaFreeHost(b.old_pointer);
            if (status != cudaSuccess) return false;
            owner.erase(it);
            if (b.kind == shrink::Kind::device) d.scratch -= b.old_bytes;
            return true;
        };
        const shrink::Result result = shrink::replace(buffers, ops);
        if (allocation_hard_error) { err = "decode chunk shrink replacement initialization failed with a CUDA execution error"; return false; }
        if (result.state == shrink::State::fatal) { err = "decode chunk shrink: " + result.reason; return false; }
        if (!result.published) {
            std::fprintf(stderr, "ds4: experimental decode chunk shrink skipped (%s); original C=%zu retained\n", result.reason.c_str(), oldC);
            return true;
        }
        std::fprintf(stderr, "ds4: experimental decode chunk shrink committed C=%zu -> %zu, A=%d -> %d; %zu buffers, "
                             "GPU %.3f -> %.3f GiB (%.3f GiB reclaimed), host %.3f -> %.3f GiB; future cold prompts "
                             "use smaller chunks and may be slower; KV/taps/checkpoints retained\n",
                     oldC, newC, oldA, newA, buffers.size(), result.device_old/1073741824.0, result.device_new/1073741824.0,
                     (result.device_old-result.device_new)/1073741824.0, result.host_old/1073741824.0, result.host_new/1073741824.0);
        return true;
    }

    bool shrink_chunk_major_buffers(Dev& d, Chunk& c, std::string& err) {
        if (chunk_major_decode_chunk == 0 || c.C <= chunk_major_decode_chunk) return true;
        const size_t oldC = (size_t)c.C, newC = (size_t)chunk_major_decode_chunk;
        const int oldA = c.A, newA = std::min(c.A, chunk_major_decode_chunk);
        const size_t D = (size_t)g.dim, H = (size_t)g.hc, HD = (size_t)g.hd, FF = (size_t)g.ff;
        if (devs.size() != 1 || c.C != chunk || c.lm || c.lm_state || c.lc || c.prefill_gpu_h || c.lm_gpu || c.lm_f16 || c.lm_f16_y || c.lm_f16_meta_dev ||
            c.lm_f16_meta_host || c.lm_f16_blas || c.cold_f16 || c.cold_f16_y || c.cold_f16_meta_host ||
            c.cold_f16_meta_dev || c.cold_f16_workspace || c.cold_f16_blas || c.eg_weight_cache || c.dense_projection_cache || (!chunk_major_phase_pending&&!gpu_misses_recovery_context())) {
            err = "decode chunk shrink requires a retired, quiescent first-prefill phase";
            return false;
        }
        std::vector<shrink::Buffer> buffers;
        auto add = [&](const char* name, auto*& pointer, size_t before, size_t after, shrink::Kind kind = shrink::Kind::device) {
            if (pointer == nullptr || before == after) return;
            const size_t item = sizeof(*pointer);
            size_t ob = before * item, nb = after * item;
            if (kind == shrink::Kind::device) ob = align256(ob), nb = align256(nb);
            else ob += 64, nb += 64; // halloc/hmap's exact64-byte suffix
            buffers.push_back({name, (void*)&pointer, (void*)pointer, ob, nb, kind, nullptr});
        };
        // Proven C-scaled allocations, matching alloc_chunks's element counts.
        add("h", c.h, oldC*H*D, newC*H*D); add("x", c.x, oldC*D, newC*D);
        add("acc", c.acc, oldC*D, newC*D); add("x16", c.x16, oldC*D, newC*D);
        add("qr", c.qr, oldC*g.q_lora, newC*g.q_lora); add("qr16", c.qr16, oldC*g.q_lora, newC*g.q_lora);
        add("kvr", c.kvr, oldC*HD, newC*HD); add("ckr", c.ckr, oldC*HD, newC*HD); add("cgr", c.cgr, oldC*HD, newC*HD);
        add("latent", c.latent, (oldC+2)*HD, (newC+2)*HD); add("lat16", c.lat16, (oldC+2)*HD, (newC+2)*HD);
        add("ikr", c.ikr, (oldC+2)*g.ix_hd, (newC+2)*g.ix_hd);
        add("ext", c.ext, (oldC+g.win)*HD, (newC+g.win)*HD);
        add("iw", c.iw, oldC*64, newC*64); add("topk", c.topk, oldC*topk_stride(), newC*topk_stride());
        add("cand", c.cand, oldC*cand_stride(), newC*cand_stride()); add("mix", c.mix, oldC*32, newC*32);
        add("a_pre", c.a_pre, oldC*8, newC*8); add("a_post", c.a_post, oldC*8, newC*8);
        add("a_comb", c.a_comb, oldC*64, newC*64); add("f_pre", c.f_pre, oldC*8, newC*8);
        add("f_post", c.f_post, oldC*8, newC*8); add("f_comb", c.f_comb, oldC*64, newC*64);
        add("pre", c.pre, oldC*8, newC*8); add("glog", c.glog, oldC*1024, newC*1024);
        add("ridx", c.ridx, oldC*kPicks, newC*kPicks); add("rw", c.rw, oldC*kPicks, newC*kPicks);
        add("row_tok", c.row_tok, oldC*kPicks, newC*kPicks); add("pos", c.pos, oldC*kPicks, newC*kPicks);
        add("sg", c.sg, oldC*FF, newC*FF); add("su", c.su, oldC*FF, newC*FF);
        add("sh16", c.sh16, oldC*FF, newC*FF); add("h16", c.h16, oldC*FF, newC*FF);
        add("y", c.y, oldC*D, newC*D); add("emb", c.emb, oldC*D, newC*D);
        const size_t cols = (size_t)(pk.eg_ngram-1)*pk.eg_heads, per = cols*(pk.eg_hd+pk.eg_hd/32);
        add("eg_rows", c.eg_rows, oldC*per, newC*per);
        const size_t NHL = (size_t)g.heads / head_members(1), HQ = NHL*HD, OG = (size_t)g.o_groups / head_members(1);
        add("q", c.q, (size_t)oldA*HQ, (size_t)newA*HQ); add("o", c.o, (size_t)oldA*HQ, (size_t)newA*HQ);
        add("o16", c.o16, (size_t)oldA*HQ, (size_t)newA*HQ);
        add("oa", c.oa, (size_t)oldA*OG*g.o_lora, (size_t)newA*OG*g.o_lora);
        add("oa16", c.oa16, (size_t)oldA*OG*g.o_lora, (size_t)newA*OG*g.o_lora);
        auto hand_bytes = [&](size_t n) { return n*(H*D*4 + 8*4 + topk_stride()*4 + cand_stride()*4); };
        for (int slot = 0; slot < 2; ++slot) {
            add("emb_host", c.emb_host[slot], oldC*D, newC*D, shrink::Kind::pinned);
            add("eg_host", c.eg_host[slot], oldC*pk.eg_layers*per, newC*pk.eg_layers*per, shrink::Kind::pinned);
            add("hand", c.hand[slot], hand_bytes(oldC), hand_bytes(newC), shrink::Kind::mapped);
        }
        // Reserve owner capacity BEFORE the first allocation. Every new pointer
        // is tracked immediately, including failed zeroing/rollback cases.
        size_t device_count = 0, host_count = 0;
        for (const auto& b : buffers) if (b.kind == shrink::Kind::device) ++device_count; else ++host_count;
        try { d.allocs.reserve(d.allocs.size()+device_count); host_allocs.reserve(host_allocs.size()+host_count); }
        catch (...) {
            std::fprintf(stderr, "ds4: experimental decode chunk shrink skipped: owner ledger capacity unavailable; original C=%zu retained\n", oldC);
            return true;
        }
        bool allocation_hard_error = false;
        shrink::Callbacks ops;
        ops.owns_old = [&](const shrink::Buffer& b) {
            const auto& owner = b.kind == shrink::Kind::device ? d.allocs : host_allocs;
            return std::count(owner.begin(), owner.end(), b.old_pointer) == 1;
        };
        ops.cancelled = [&] { return cancel && cancel(); }; // caller thread only
        ops.allocate = [&](shrink::Buffer& b) {
            cudaError_t status;
            if (b.kind == shrink::Kind::device) {
                status = cudaMalloc(&b.next, b.new_bytes);
                if (status == cudaSuccess) { d.allocs.push_back(b.next); d.scratch += b.new_bytes; }
                if (status == cudaSuccess) status = cudaMemsetAsync(b.next, 0, b.new_bytes, d.st);
                if (status == cudaSuccess) status = cudaStreamSynchronize(d.st);
            } else {
                const unsigned flags = cudaHostAllocPortable | (b.kind == shrink::Kind::mapped ? cudaHostAllocMapped : 0);
                status = cudaHostAlloc(&b.next, b.new_bytes, flags);
                if (status == cudaSuccess) host_allocs.push_back(b.next);
                if (status == cudaSuccess) std::memset(b.next, 0, b.new_bytes);
            }
            if (status == cudaErrorMemoryAllocation) (void)cudaGetLastError(); // recoverable optional allocation
            else if (status != cudaSuccess) allocation_hard_error = true;
            return status == cudaSuccess;
        };
        ops.discard_new = [&](const shrink::Buffer& b) {
            auto& owner = b.kind == shrink::Kind::device ? d.allocs : host_allocs;
            auto it = std::find(owner.begin(), owner.end(), b.next);
            if (it == owner.end()) return false;
            const cudaError_t status = b.kind == shrink::Kind::device ? cudaFree(b.next) : cudaFreeHost(b.next);
            if (status != cudaSuccess) return false; // retain owner entry for terminal cleanup
            owner.erase(it);
            if (b.kind == shrink::Kind::device) d.scratch -= b.new_bytes;
            return true;
        };
        ops.reserve_publication = [&](const std::vector<shrink::Buffer>& bs) {
            size_t nd = 0, nh = 0, old_b = 0, new_b = 0;
            for (const auto& b : bs) if (b.kind == shrink::Kind::device) ++nd, old_b += b.old_bytes, new_b += b.new_bytes; else ++nh;
            if (old_b > ~(size_t)0-new_b || d.scratch < old_b+new_b) return false;
            for (const auto& b : bs) {
                const auto& owner = b.kind == shrink::Kind::device ? d.allocs : host_allocs;
                if (std::count(owner.begin(), owner.end(), b.old_pointer) != 1 || std::count(owner.begin(), owner.end(), b.next) != 1) return false;
            }
            return true; // no allocation remains in publication
        };
        ops.publish = [&](const std::vector<shrink::Buffer>& bs) {
            // COMMIT linearization: allocations initialized, ledgers reserved;
            // pointer+shape publication cannot fail. Stop AFTER here keeps this
            // valid small layout; no asynchronous writer remains on old buffers.
            for (const auto& b : bs) std::memcpy(b.slot, &b.next, sizeof(b.next));
            chunk = c.C = chunk_major_decode_chunk;
            c.A = newA;
        };
        ops.retire_old = [&](const shrink::Buffer& b) {
            auto& owner = b.kind == shrink::Kind::device ? d.allocs : host_allocs;
            auto it = std::find(owner.begin(), owner.end(), b.old_pointer);
            if (it == owner.end()) return false;
            const cudaError_t status = b.kind == shrink::Kind::device ? cudaFree(b.old_pointer) : cudaFreeHost(b.old_pointer);
            if (status != cudaSuccess) return false;
            owner.erase(it);
            if (b.kind == shrink::Kind::device) d.scratch -= b.old_bytes;
            return true;
        };
        const shrink::Result result = shrink::replace(buffers, ops);
        if (allocation_hard_error) { err = "decode chunk shrink replacement initialization failed with a CUDA execution error"; return false; }
        if (result.state == shrink::State::fatal) { err = "decode chunk shrink: " + result.reason; return false; }
        if (!result.published) {
            std::fprintf(stderr, "ds4: experimental decode chunk shrink skipped (%s); original C=%zu retained\n", result.reason.c_str(), oldC);
            return true;
        }
        std::fprintf(stderr, "ds4: experimental decode chunk shrink committed C=%zu -> %zu, A=%d -> %d; %zu buffers, "
                             "GPU %.3f -> %.3f GiB (%.3f GiB reclaimed), host %.3f -> %.3f GiB; future cold prompts "
                             "use smaller chunks and may be slower; KV/taps/checkpoints retained\n",
                     oldC, newC, oldA, newA, buffers.size(), result.device_old/1073741824.0, result.device_new/1073741824.0,
                     (result.device_old-result.device_new)/1073741824.0, result.host_old/1073741824.0, result.host_new/1073741824.0);
        return true;
    }


    // Cold-GEMM owner replaces this hook when its independent cache is merged.
    // Called only after worker/read/copy/compute drain, before shrink and monotonic expert-pool growth.
    bool retire_chunk_major_cold_cache(Dev& d, Chunk& c, std::string& err) {
        if(c.cold_f16_blas) {
            if(cublasDestroy(c.cold_f16_blas)!=CUBLAS_STATUS_SUCCESS){err="retiring cold F16 cuBLAS handle";return false;}
            c.cold_f16_blas=nullptr;
        }
        for(cudaEvent_t& event:c.cold_f16_meta_done)if(event) {
            const cudaError_t status=cudaEventDestroy(event);
            if(status!=cudaSuccess){err=std::string("retiring cold metadata event: ")+cudaGetErrorString(status);return false;}
            event=nullptr;
        }
        auto gpu=[&](auto*& pointer,size_t bytes) {
            if(!pointer)return true;
            const size_t accounted=align256(bytes);
            auto it=std::find(d.allocs.begin(),d.allocs.end(),(void*)pointer);
            if(it==d.allocs.end()||std::count(d.allocs.begin(),d.allocs.end(),(void*)pointer)!=1||d.scratch<accounted) {
                err="cold F16 retirement ownership/accounting";return false;
            }
            const cudaError_t status=cudaFree(pointer);
            if(status!=cudaSuccess){err=std::string("retiring cold F16 buffer: ")+cudaGetErrorString(status);return false;}
            d.allocs.erase(it);d.scratch-=accounted;pointer=nullptr;return true;
        };
        const size_t bytes=c.cold_f16_bytes+c.cold_f16_y_bytes+c.cold_f16_meta_bytes+c.cold_f16_workspace_bytes;
        if(!gpu(c.cold_f16,c.cold_f16_bytes)||!gpu(c.cold_f16_y,c.cold_f16_y_bytes)||
           !gpu(c.cold_f16_meta_dev,c.cold_f16_meta_bytes)||!gpu(c.cold_f16_workspace,c.cold_f16_workspace_bytes))return false;
        if(c.cold_f16_meta_host) {
            auto it=std::find(host_allocs.begin(),host_allocs.end(),(void*)c.cold_f16_meta_host);
            if(it==host_allocs.end()||std::count(host_allocs.begin(),host_allocs.end(),(void*)c.cold_f16_meta_host)!=1) {
                err="cold F16 pinned metadata retirement ownership";return false;
            }
            const cudaError_t status=cudaFreeHost(c.cold_f16_meta_host);
            if(status!=cudaSuccess){err=std::string("retiring cold pinned metadata: ")+cudaGetErrorString(status);return false;}
            host_allocs.erase(it);c.cold_f16_meta_host=nullptr;
        }
        if(bytes)std::fprintf(stderr,"ds4: experimental bounded cold F16 retired %.1f MiB: %llu sets/%llu batches/%llu buckets, "
            "%llu expert unpacks, %llu valid/%llu padded hot rows, %llu light experts/%llu rows; "
            "fallback shape=%llu format=%llu light=%llu blas=%llu. Total expert timing valid; GU/down split approximate\n",
            bytes/1048576.0,(unsigned long long)c.cold_f16_sets,(unsigned long long)c.cold_f16_batches,
            (unsigned long long)c.cold_f16_buckets,(unsigned long long)c.cold_f16_unpacks,
            (unsigned long long)c.cold_f16_rows,(unsigned long long)c.cold_f16_padded_rows,
            (unsigned long long)c.cold_f16_light_experts,(unsigned long long)c.cold_f16_light_rows,
            (unsigned long long)c.cold_f16_fallback_shape,(unsigned long long)c.cold_f16_fallback_format,
            (unsigned long long)c.cold_f16_fallback_light,(unsigned long long)c.cold_f16_fallback_blas);
        c.cold_f16_bytes=c.cold_f16_unit=c.cold_f16_y_bytes=c.cold_f16_meta_bytes=0;
        c.cold_f16_meta_stride=c.cold_f16_workspace_bytes=0;
        return true;
    }

    // Same all-layer profile-to-heat normalization as fresh place_experts, after the first feed's
    // worker/stream/cache drain. Chunk routing has already counted every pick into prof synchronously.
    // Do not replay cpu_hits or touch prof/decay_tokens: the following ordinary cache_tick owns those.
    bool reset_prefill_heat_from_profile(std::string& err) {
        if(!chunk_major_phase_pending){err="heat normalization outside first copied-RAM phase";return false;}
        for(const Layer& l:L) {
            if(l.il<0||g.n_exp<=0||l.n_exp<0||l.n_exp>g.n_exp||
               (size_t)l.il>=prof.size()/(size_t)g.n_exp||l.heat.size()!=(size_t)l.n_exp) {
                err="heat normalization profile/layer bounds";return false;
            }
        }
        double before_trunk=0,before_mtp=0,after_trunk=0,after_mtp=0;
        uint64_t profile_hits=0,profile_hash=14695981039346656037ull;
        size_t trunk_layers=0,mtp_layers=0,entries=0;
        for(Layer& l:L) {
            const uint32_t* p=prof.data()+(size_t)l.il*g.n_exp;
            double before=0;
            for(float h:l.heat)before+=h;
            // Literal Opt8 place_experts normalization; preserve its double sum/scale and final F32 rounding.
            l.heat.assign((size_t) l.n_exp, 0.0f);
            {   // the profile as the starting heat: its counts scaled to ~512 hits a layer
                double sum = 0;
                for (int e = 0; e < l.n_exp; ++e) sum += p[e];
                const double sc = sum > 0 ? 512.0 / sum : 0.0;
                for (int e = 0; e < l.n_exp; ++e) l.heat[(size_t) e] = (float) (p[e] * sc);
            }
            double after=0;
            for(int e=0;e<l.n_exp;++e) {
                after+=l.heat[(size_t)e];profile_hits+=p[e];++entries;
                // Explicit byte order makes this a reproducible diagnostic of counts, never prompt contents.
                for(int shift=0;shift<32;shift+=8){profile_hash^=(p[e]>>shift)&255u;profile_hash*=1099511628211ull;}
            }
            if(l.mtp){before_mtp+=before;after_mtp+=after;++mtp_layers;}
            else {before_trunk+=before;after_trunk+=after;++trunk_layers;}
        }
        std::fprintf(stderr,"ds4: experimental first-prefill heat normalized from profile: trunk%zu %.9g -> %.9g, "
                            "MTP%zu %.9g -> %.9g; %zu entries/%llu hits, counts FNV1a64=%016llx; "
                            "no placement or profile change\n",
                     trunk_layers,before_trunk,after_trunk,mtp_layers,before_mtp,after_mtp,
                     entries,(unsigned long long)profile_hits,(unsigned long long)profile_hash);
        return true;
    }

    // Unused draft: called only after a successful workspace restoration under
    // an acknowledged worker epoch. This does not activate a prefill phase.
    bool rebuild_copied_phase_assets(unsigned lease,bool& ready,bool& stopped,std::string& err) {
        ready=false;stopped=false;
#if defined(STRATA_USE_HIP)
        (void)lease;err="copied phase asset rebuild requires CUDA";return false;
#else
        if(!cache_epoch_enabled||!lease||cache_epoch_machine.phase!=epochpolicy::Phase::held||
           lease!=cache_epoch_machine.serial||cache_epoch_request.load(std::memory_order_acquire)!=lease||
           !cache_epoch_block_enqueues.load(std::memory_order_acquire)||
           cache_epoch_service_ack.load(std::memory_order_acquire)!=lease||
           cache_epoch_service_completed.load(std::memory_order_acquire)!=seq_issued||
           !chunk_major_phase_error.empty()||chunk_major_phase_pending||prefill_phase_pending||
           !chunk_major_phase_enabled||!cold_f16_enabled||!prefill_engram_cache||!prefill_dense_projection_cache||
           devs.size()!=1||chunks.size()!=1||!chunks[0]||NH!=1||!copy_mode||store||ram_tier||
           registered_copy.published||!registered_copy.held.empty()) {
            err="phase asset rebuild requires unregistered HELD copied-RAM Opt17 geometry";return false;
        }
        auto hard=[&](const std::string& why){return cache_epoch_terminal("phase asset rebuild: "+why,err);};
        try {
        Dev& d=devs[0];Chunk& c=*chunks[0];
        if(c.C!=65280||chunk!=65280||c.A!=4096||c.AI!=128||c.restore_C!=65280||
           c.restore_A!=4096||c.restore_AI!=128||!c.flow||c.lm||c.lm_state||c.lc||
           c.prefill_gpu_h||c.lm_gpu||c.lm_f16||c.lm_f16_y||c.lm_f16_meta_dev||c.lm_f16_meta_host||c.lm_f16_blas||
           c.cold_f16||c.cold_f16_y||c.cold_f16_meta_host||c.cold_f16_meta_dev||c.cold_f16_workspace||c.cold_f16_blas||
           c.eg_weight_cache||c.dense_projection_cache||c.cold_f16_bytes||c.cold_f16_y_bytes||
           c.cold_f16_meta_bytes||c.cold_f16_workspace_bytes||c.eg_weight_cache_bytes||c.dense_projection_cache_bytes||
           g.dim!=5120||g.hc!=4||g.hd!=512||g.ff!=2304||o.max_context!=1048576||!pk.gguf)
            {err="phase asset rebuild requires restored workspace and completely retired arenas";return false;}
        for(cudaEvent_t event:c.cold_f16_meta_done)if(event)return hard("retired cold metadata event remains owned");
        for(const auto& worker:workers)if(!worker->idle()){err="phase asset rebuild needs idle prompt workers";return false;}
        {
            std::lock_guard<std::mutex> lock(sw_mu);
            if(cache_epoch_worker_acks!=cache_ths.size()||sw_inflight||ram_inflight||!sw_queue.empty()||
               !sw_ready.empty()||!ram_queue.empty()||!ram_ready.empty())
                {err="phase asset rebuild needs acknowledged empty cache queues";return false;}
        }
        if(cudaSetDevice(d.id)!=cudaSuccess||cudaStreamQuery(d.st)!=cudaSuccess||cudaStreamQuery(c.cs)!=cudaSuccess)
            return hard("held device streams not idle");
        for(size_t i=c.base;i<c.n_read;++i)if(fit(c,i).left.load(std::memory_order_acquire)!=0)
            return hard("a flow reader still owns a payload");
        ramdomain::Inventory inventory;if(!inspect_copied_ram_inventory(inventory,err))return hard(err);
        const size_t D=(size_t)g.dim,FF=(size_t)g.ff,C=(size_t)c.C;
        const size_t dense_values=(size_t)32768*1280+(size_t)8*1024*4096+(size_t)5120*8192;
        const size_t eg_values=(size_t)25600*6144,unit_values=(size_t)3*D*FF;
        const size_t dense_b=align256(dense_values*sizeof(uint16_t)),eg_b=align256(eg_values*sizeof(uint16_t));
        const size_t cold_b=align256((size_t)Chunk::kColdF16Experts*unit_values*sizeof(uint16_t));
        const size_t y_b=align256(C*D*sizeof(float));
        const size_t meta_stride=align256((size_t)Chunk::kColdF16Experts*sizeof(F16PhaseRow))+
                                 align256((size_t)8*Chunk::kColdF16Experts*sizeof(unsigned long long));
        const size_t meta_b=meta_stride*Chunk::kColdF16MetaSlots,workspace_b=(size_t)32<<20;
        const size_t gpu_need=dense_b+eg_b+cold_b+y_b+meta_b+workspace_b,host_need=meta_b+64;
        size_t remaining_host=host_need;
        // Startup's6144MiB placement reserve is not the repeat-phase reserve.
        // Require the explicit final reserve and a minimum3GiB of live VRAM.
        const size_t floor=strata::ds4::phasepolicy::final_reserve_bytes(3072,prefill_final_reserve_mb);
        bool budget_hard_error=false;
        auto budget=[&](size_t remaining)->bool {
            size_t free_b=0,total_b=0;
            if(cudaMemGetInfo(&free_b,&total_b)!=cudaSuccess||!total_b||free_b>total_b)
                {budget_hard_error=true;return false;}
            const os::Commit cm=os::commit();
            return free_b>=floor&&remaining<=free_b-floor&&cm.avail!=~uint64_t{0}&&
                (double)cm.avail>=commit_margin()&&
                chunkregrow::budget_allows(free_b,ram_avail(),commit_room(),remaining,remaining_host);
        };
        if(cancel&&cancel()){stopped=true;return true;}
        if(!budget(gpu_need)) {
            if(budget_hard_error)return hard("initial live CUDA budget query");
            err="phase asset rebuild does not fit beside repeat-phase live reserve";return false;
        }
        try{d.allocs.reserve(d.allocs.size()+6);host_allocs.reserve(host_allocs.size()+1);}
        catch(...){err="phase asset rebuild owner planning unavailable";return false;}
        const size_t scratch_before=d.scratch;const double started=now_ms();size_t remaining=gpu_need;
        auto rollback=[&](const std::string& reason)->bool {
            std::string why;
            if(!retire_prefill_dense_projection_cache(d,c,why)||!retire_prefill_engram_cache(d,c,why)||
               !retire_chunk_major_cold_cache(d,c,why))return hard("rollback: "+why);
            if(d.scratch!=scratch_before)return hard("rollback scratch accounting");
            err=reason;return true;
        };
        auto interrupted=[&]()->bool{stopped=cancel&&cancel();return stopped;};
        auto allocation=[&](auto*& ptr,size_t elements,size_t bytes,size_t& accounted)->bool {
            if(!budget(remaining)){err="phase asset rebuild live budget changed";return false;}
            void* next=nullptr;
            if(align256(elements*sizeof(*ptr))!=bytes||cudaMalloc(&next,bytes)!=cudaSuccess)
                {err="phase asset allocation failed";return false;}
            d.allocs.push_back(next);ptr=static_cast<decltype(ptr+0)>(next);
            d.scratch+=bytes;accounted=bytes;remaining-=bytes;
            // Ownership is visible even if initialization fails or throws.
            if(cudaMemsetAsync(next,0,bytes,d.st)!=cudaSuccess||cudaStreamSynchronize(d.st)!=cudaSuccess)
                {err="phase asset initialization failed";return false;}
            return true;
        };
        // Every pointer is added to the existing teardown ledger immediately;
        // set each byte field only after that allocation actually succeeds.
        if(!allocation(c.dense_projection_cache,dense_values,dense_b,c.dense_projection_cache_bytes))return hard(err);
        if(interrupted())return rollback("stopped during phase asset rebuild");
        if(!allocation(c.eg_weight_cache,eg_values,eg_b,c.eg_weight_cache_bytes))return hard(err);
        if(interrupted())return rollback("stopped during phase asset rebuild");
        if(!allocation(c.cold_f16,(size_t)Chunk::kColdF16Experts*unit_values,cold_b,c.cold_f16_bytes))return hard(err);
        c.cold_f16_unit=unit_values;
        if(interrupted())return rollback("stopped during phase asset rebuild");
        if(!allocation(c.cold_f16_y,C*D,y_b,c.cold_f16_y_bytes))return hard(err);
        if(interrupted())return rollback("stopped during phase asset rebuild");
        if(!allocation(c.cold_f16_meta_dev,meta_b,meta_b,c.cold_f16_meta_bytes))return hard(err);
        c.cold_f16_meta_stride=meta_stride;
        if(interrupted())return rollback("stopped during phase asset rebuild");
        if(!allocation(c.cold_f16_workspace,workspace_b,workspace_b,c.cold_f16_workspace_bytes))return hard(err);
        void* next_host=nullptr;
        if(cudaHostAlloc(&next_host,host_need,cudaHostAllocPortable)!=cudaSuccess)return hard("cold metadata pinned allocation");
        host_allocs.push_back(next_host);c.cold_f16_meta_host=(uint8_t*)next_host;remaining_host=0;
        std::memset(c.cold_f16_meta_host,0,host_need);
        if(interrupted())return rollback("stopped during phase asset rebuild");
        if(cublasCreate(&c.cold_f16_blas)!=CUBLAS_STATUS_SUCCESS||
           cublasSetStream(c.cold_f16_blas,d.st)!=CUBLAS_STATUS_SUCCESS||
           cublasSetPointerMode(c.cold_f16_blas,CUBLAS_POINTER_MODE_HOST)!=CUBLAS_STATUS_SUCCESS||
           cublasSetMathMode(c.cold_f16_blas,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)!=CUBLAS_STATUS_SUCCESS||
           cublasSetAtomicsMode(c.cold_f16_blas,CUBLAS_ATOMICS_NOT_ALLOWED)!=CUBLAS_STATUS_SUCCESS||
           cublasSetWorkspace(c.cold_f16_blas,c.cold_f16_workspace,workspace_b)!=CUBLAS_STATUS_SUCCESS)
            return hard("cold F16 cuBLAS handle/workspace");
        for(cudaEvent_t& event:c.cold_f16_meta_done)
            if(cudaEventCreateWithFlags(&event,cudaEventDisableTiming)!=cudaSuccess)return hard("cold F16 metadata event");
        if(interrupted())return rollback("stopped during phase asset rebuild");
        if(!budget(0)||remaining||d.scratch!=scratch_before+gpu_need)return hard("final live budget/accounting");
        for(int r=0;r<10;++r){c.dense_projection_valid[r]=false;c.dense_projection_tags[r]={};
            c.dense_projection_layers[r]=-1;c.dense_projection_p0[r]=-1;}
        c.eg_weight_valid=false;c.eg_weight_tag={};c.eg_weight_layer=-1;c.eg_weight_p0=-1;
        for(bool& recorded:c.cold_f16_meta_recorded)recorded=false;
        c.cold_f16_meta_count=0;
        ready=true;
        std::fprintf(stderr,"ds4: experimental copied phase assets rebuilt: GPU %zu bytes, host %zu bytes, "
                            "16 new metadata events; tags invalidated, sequence untouched, %.3f s; phase NOT activated\n",
                     gpu_need,host_need,(now_ms()-started)/1000.0);
        return true;
        }catch(...){return hard("exception during tracked asset preparation");}
#endif
    }

    // Explicit capability only. Called at the serialized feed/feed_mark boundary,
    // before token/KV mutation, so the caller's GEN timing includes maintenance.
    bool maybe_rearm_copied_prefill(int n,std::string& err) {
        if(!repeatpolicy::eligible(prefill_repeat_rearm,chunk_major_phase_pending,n,prefill_repeat_min_tokens))return true;
#if defined(STRATA_USE_HIP)
        err="repeated copied prefill requires CUDA";return false;
#else
        if(!chunk_major_phase_error.empty()){err=chunk_major_phase_error;return false;}
        auto hard=[&](const std::string& why){return cache_epoch_terminal("repeat prefill rearm: "+why,err);};
        unsigned lease=0;bool released=false;
        try {
            if(prefill_repeat_preparing)return hard("nested serialized feed");
            struct Preparing {bool& value;Preparing(bool& v):value(v){value=true;}~Preparing(){value=false;}} preparing(prefill_repeat_preparing);
            if(devs.size()!=1||chunks.size()!=1||!chunks[0]||NH!=1||!copy_mode||store||ram_tier||
               !cache_epoch_enabled||!copied_ram_domains_published||!chunk_major_phase_enabled||
               !cold_f16_enabled||!prefill_engram_cache||!prefill_dense_projection_cache||
               ((registered_copy.published||!registered_copy.held.empty())&&!gpu_misses_retained_ready())||prefill_phase_pending||
               !pk.gguf||!file_experts||!o.expert_cache||o.max_context!=1048576||
               pos<0||n<0||pos>o.max_context-n)return hard("unsupported current capability/context");
            Dev& d=devs[0];Chunk& c=*chunks[0];
            const bool small=c.C==1024&&chunk==1024&&c.A==1024;
            const bool large=c.C==65280&&chunk==65280&&c.A==4096;
            if(c.restore_C!=65280||c.restore_A!=4096||c.restore_AI!=128||c.AI!=128||(!small&&!large))
                return hard("unrecorded small/restored geometry");
            // A late Stop may leave the restored workspace, but never unowned
            // caches or an unreachable active phase. This state is retryable.
            if(c.cold_f16||c.cold_f16_y||c.cold_f16_meta_host||c.cold_f16_meta_dev||c.cold_f16_workspace||
               c.cold_f16_blas||c.eg_weight_cache||c.dense_projection_cache)
                return hard("inactive phase still owns arenas");
            const double started=now_ms();const CacheProbeState before=cache_probe_state();
            ++prefill_repeat_attempts;std::string why;
            if(!begin_cache_worker_epoch(lease,60000,why)) {
                if(cache_epoch_machine.phase==epochpolicy::Phase::failed||!chunk_major_phase_error.empty())return hard(why);
                if(cancel&&cancel()){halted=true;err="stopped while acquiring repeat prefill epoch";return false;}
                std::fprintf(stderr,"ds4: experimental repeat prefill rearm declined before lease: %s; existing layout retained\n",why.c_str());
                return true;
            }
            if(gpu_miss_direct_enabled&&(gpu_misses_decode_pins||registered_copy.published||!registered_copy.held.empty()))
                return hard("rearm epoch did not physically retire decode pins");
            struct LeaseGuard {Impl* p;unsigned lease;bool& released;
                ~LeaseGuard(){if(!released&&p->cache_epoch_machine.serial==lease&&
                    p->cache_epoch_machine.phase==epochpolicy::Phase::held)try{std::string why;p->cache_epoch_terminal("repeat rearm abandoned held epoch",why);}catch(...){}}
            } guard{this,lease,released};
            const CacheProbeState held=cache_probe_state();
            auto same_state=[](const CacheProbeState& a,const CacheProbeState& b){return a.position==b.position&&a.tokens==b.tokens&&
                a.ctokens==b.ctokens&&a.issued==b.issued&&a.token_hash==b.token_hash&&a.checkpoint_hash==b.checkpoint_hash;};
            if(!same_state(before,held))return hard("acquisition changed token/checkpoint/issued state");
            auto release_abort=[&](bool stopped,const std::string& reason)->bool {
                // No prompt worker has started. Retire every tracked setup asset
                // before unblocking cache/service workers, including registration.
                if(cudaSetDevice(d.id)!=cudaSuccess||cudaStreamSynchronize(c.cs)!=cudaSuccess||
                   cudaStreamSynchronize(d.st)!=cudaSuccess)return hard("abort stream drain");
                if(!retire_registered_copy_ram(why)||!retire_prefill_dense_projection_cache(d,c,why)||
                   !retire_prefill_engram_cache(d,c,why)||!retire_chunk_major_cold_cache(d,c,why))return hard(why);
                chunk_major_phase_pending=false;
                if(gpu_miss_direct_enabled&&stopped)gpu_misses_recovery_needed=true;
                const CacheProbeState post=cache_probe_state();
                if(!same_state(held,post))return hard("abort changed token/checkpoint/issued state");
                if(!end_cache_worker_epoch(lease,why))return hard(why);
                released=true;
                std::fprintf(stderr,"ds4: experimental repeat prefill rearm not activated: attempt%llu C%d, %s%s; valid current pool/layout retained\n",
                    (unsigned long long)prefill_repeat_attempts,c.C,reason.c_str(),stopped?"; stopped":"");
                if(stopped){halted=true;err="stopped during repeat prefill rearm";return false;}
                return true;
            };
            bool stopped=false;
            const bool shrunk=shrink_copied_expert_pool(lease,stopped,why);
            if(cache_epoch_machine.phase==epochpolicy::Phase::failed||!chunk_major_phase_error.empty())return hard(why);
            if(stopped||(cancel&&cancel()))return release_abort(true,why);
            if(!shrunk)return release_abort(false,why);
            ramdomain::Inventory inventory;
            if(!inspect_copied_ram_inventory(inventory,why))return hard(why);
            bool attained=inventory.free_holes==0&&!copied_ram_domains.empty();
            for(const auto& domain:copied_ram_domains) {
                const Layer& l=L[(size_t)domain.layer];copiedshrink::Plan target;
                if(!copiedshrink::plan(domain,l.slot,l.n_slots,l.spare_vram,l.ram,copy_spare[(size_t)l.il],
                    inventory.layers[(size_t)l.il].free_slots,unit(l),target,why))return hard(why);
                attained=attained&&l.n_slots==target.next_slots&&inventory.layers[(size_t)l.il].free_slots.empty();
            }
            if(!attained)return release_abort(false,"valid partial pool shrink did not reach every immutable-domain target");
            bool published=false;
            if(small) {
                const bool restored=regrow_copied_chunk_buffers(lease,stopped,published,why);
                if(cache_epoch_machine.phase==epochpolicy::Phase::failed||!chunk_major_phase_error.empty())return hard(why);
                if(stopped||(cancel&&cancel()))return release_abort(true,why);
                if(!restored||!published)return release_abort(false,why.empty()?"workspace kept small":why);
            }
            if(c.C!=65280||chunk!=65280||c.A!=4096||c.AI!=128)return hard("workspace target not attained");
            bool ready=false;
            const bool rebuilt=rebuild_copied_phase_assets(lease,ready,stopped,why);
            if(cache_epoch_machine.phase==epochpolicy::Phase::failed||!chunk_major_phase_error.empty())return hard(why);
            if(stopped||(cancel&&cancel()))return release_abort(true,why);
            if(!rebuilt||!ready)return release_abort(false,why.empty()?"phase assets not ready":why);
            // All cache arenas are ready before pending becomes visible. The
            // epoch is still HELD throughout registration and its rollback.
            chunk_major_phase_pending=true;
            if(!register_copied_ram_for_prefill(why))return hard(why);
            if(cancel&&cancel())return release_abort(true,"stopped after registration");
            const CacheProbeState post=cache_probe_state();
            if(!same_state(held,post))return hard("maintenance changed token/checkpoint/issued state");
            prefill_repeat_cold_sets=c.cold_f16_sets;prefill_repeat_cold_unpacks=c.cold_f16_unpacks;
            prefill_repeat_eg_fills=c.eg_weight_fills;prefill_repeat_eg_hits=c.eg_weight_hits;
            prefill_repeat_dense_fills=c.dense_projection_fills;prefill_repeat_dense_hits=c.dense_projection_hits;
            if(!end_cache_worker_epoch(lease,why))return hard(why);
            released=true;
            const CacheProbeState after=cache_probe_state();
            if(!same_state(post,after)||post.map_hash!=after.map_hash)return hard("release changed completed maintenance state");
            ++prefill_repeat_generation;
            std::fprintf(stderr,"ds4: experimental repeat prefill rearm ready: generation%llu attempt%llu, remaining%d C%d A%d AI%d, "
                "epoch%u issued%u unchanged, pool%zu bytes, registration%zu bytes; %.3f s charged inside feed; next compute is existing chunk-major path\n",
                (unsigned long long)prefill_repeat_generation,(unsigned long long)prefill_repeat_attempts,n,c.C,c.A,c.AI,
                lease,seq_issued,d.experts,registered_copy.bytes(),(now_ms()-started)/1000.0);
            return true;
        } catch(...) {
            return hard("exception during serialized repeat preparation; owned allocations retained");
        }
#endif
    }

    bool finish_chunk_major_phase(std::string& err, bool grow_pool = true, bool shrink_buffers = true) {
        if (!chunk_major_phase_error.empty()) { err = chunk_major_phase_error; return false; }
        const bool decode_recovery=gpu_misses_recovery_context();
        if (!chunk_major_phase_pending&&!decode_recovery) return true;
        const double started = now_ms();
        auto poison = [&](const std::string& why) {
            chunk_major_phase_pending = false;
            chunk_major_phase_error = "experimental chunk-major phase: " + why;
            err = chunk_major_phase_error;
            std::fprintf(stderr, "ds4: ERROR: %s\n", err.c_str());
            return false;
        };
        if (devs.size() != 1 || chunks.size() != 1 || !chunks[0] || NH != 1 || !copy_mode || store)
            return poison("invalid single-card copied-RAM lifecycle");
        Dev& d = devs[0]; Chunk& c = *chunks[0];
        if(decode_recovery) {
            if(cudaSetDevice(d.id)!=cudaSuccess)return poison("recovery device selection");
        } else cudaSetDevice(d.id);
        for (const auto& w : workers) if (!w->idle()) return poison("a prompt worker still owns the chunk");
        // lm_end intentionally does nothing for !lm. Drain the actual flow here,
        // including speculative reads left behind by an interrupted run.
        if (c.lm || c.lm_state || c.lc || c.prefill_gpu_h || c.lm_gpu || prefill_phase_pending)
            return poison("chunk-major and GPUH/layer-major lifetimes overlapped");
        for (size_t i = c.base; i < c.n_read; ++i)
            while (fit(c, i).left.load(std::memory_order_acquire) != 0) c.io->wait_any(500);
        cudaError_t ce=cudaSuccess;
        if(gpu_miss_direct_enabled&&gpu_misses_cs) {ce=cudaStreamSynchronize(gpu_misses_cs);if(ce!=cudaSuccess)return poison("GPU miss copy drain");}
        ce = cudaStreamSynchronize(c.cs);
        if (ce != cudaSuccess) return poison(std::string("copy drain: ") + cudaGetErrorString(ce));
        ce = cudaStreamSynchronize(d.st);
        if (ce != cudaSuccess) return poison(std::string("compute drain: ") + cudaGetErrorString(ce));
        {
            std::lock_guard<std::mutex> lk(sw_mu);
            if (sw_inflight || ram_inflight || !sw_queue.empty() || !sw_ready.empty() || !ram_queue.empty() || !ram_ready.empty())
                return poison("cache work preceded first-feed publication");
        }
        for (const Layer& l : L) if (l.swap_pending) return poison("a layer has a pending cache swap");
        const bool retain_decode_pins=gpu_miss_direct_enabled&&grow_pool&&shrink_buffers&&!halted&&
            !(cancel&&cancel())&&registered_copy.published&&!registered_copy.held.empty();
        bool pins_committed=false;
        struct PhasePinRollback {Impl* p;bool active;bool& committed;
            ~PhasePinRollback(){if(active&&!committed&&!p->registered_copy.held.empty())try{
                std::string why;if(!p->gpu_misses_retire_pins("phase-abort",why))p->cache_epoch_terminal(why,why);
            }catch(...){p->gpu_misses_decode_pins=false;try{std::string why;p->cache_epoch_terminal("phase-pin rollback exception",why);}catch(...){}}}
        } phase_pin_rollback{this,retain_decode_pins,pins_committed};
        if(!retain_decode_pins&&!retire_registered_copy_ram(err))return poison(err);
        if(prefill_dense_projection_cache && !retire_prefill_dense_projection_cache(d,c,err))return poison(err);
        if(prefill_engram_cache && !retire_prefill_engram_cache(d,c,err))return poison(err);
        if (!retire_chunk_major_cold_cache(d, c, err)) return poison(err);
        if (shrink_buffers && !shrink_chunk_major_buffers(d, c, err)) return poison(err);
        bool stopped = cancel && cancel();
        size_t grown = 0, unavailable = 0, before = d.experts;
        const size_t final_reserve=strata::ds4::phasepolicy::final_reserve_bytes(o.reserve_mib,prefill_final_reserve_mb);
        if (grow_pool && !stopped) {
            size_t free_b = 0, total_b = 0;
            ce = cudaMemGetInfo(&free_b, &total_b);
            if (ce != cudaSuccess) return poison("reading the post-shrink GPU budget");
            const size_t reserve = final_reserve;
            const size_t hard_free = (size_t)1024 << 20;
            double budget = free_b + (double)d.experts > reserve ? free_b + (double)d.experts - reserve : 0;
            if(decode_recovery)budget=gpumiss::recovery_pool_budget(budget,d.experts,gpu_misses_last_decode_pool);
            std::vector<int> target(L.size());
            double mtp_bytes = 0, want = 0, used = 0;
            for (const Layer& l : L) {
                target[(size_t)l.il] = l.n_slots;
                used += (double)l.n_slots * unit(l);
                if (l.mtp) mtp_bytes += (double)l.n_slots * unit(l);
                else if (!l.cpu_only) want += (double)l.n_exp * unit(l);
            }
            const double fraction = want > 0 ? std::min(1.0, std::max(0.0, budget-mtp_bytes)/want) : 0;
            // Plan only additions. Loaded resident identities/spare never move;
            // even a profile rank change cannot evict an uncopied resident.
            for (const Layer& l : L) {
                if (l.mtp || l.cpu_only) continue;
                const int ideal = std::min(l.n_exp, (int)std::floor(fraction*l.n_exp));
                while (target[(size_t)l.il] < ideal && used + unit(l) <= budget) {
                    ++target[(size_t)l.il]; used += unit(l);
                }
            }
            for (const Layer& l : L) {
                if (l.mtp || l.cpu_only) continue;
                while (target[(size_t)l.il] < l.n_exp && used + unit(l) <= budget) {
                    ++target[(size_t)l.il]; used += unit(l);
                }
            }
            std::vector<int> order;
            for (const Layer& l : L) if (target[(size_t)l.il] > l.n_slots) order.push_back(l.il);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                return (size_t)L[(size_t)a].n_slots*unit(L[(size_t)a]) > (size_t)L[(size_t)b].n_slots*unit(L[(size_t)b]);
            });
            for (int il : order) {
                if (cancel && cancel()) { stopped = true; break; }
                Layer& l = L[(size_t)il];
                grow::Plan plan;
                std::string why;
                if (!grow::monotonic_plan(l.slot, l.n_slots, l.spare_vram, target[(size_t)il], o.expert_cache,
                                          prof.data()+(size_t)il*g.n_exp, plan, why)) return poison(why);
                const size_t old_bytes = (size_t)l.n_slots*unit(l), next_bytes = (size_t)plan.next_slots*unit(l);
                uint8_t* old = l.vram_base; uint8_t* next = nullptr;
                decltype(l.vm) next_vm; decltype(l.host) next_host; decltype(l.hh) next_hh;
                std::vector<unsigned long long> tab;
                grow::Callbacks ops;
                ops.cancelled = [&] { return cancel && cancel(); };
                ops.discard = [&] {
                    if (!next) return true;
                    auto it = std::find(d.allocs.begin(), d.allocs.end(), (void*)next);
                    if (it == d.allocs.end() || cudaFree(next) != cudaSuccess) return false;
                    d.allocs.erase(it); next = nullptr; return true;
                };
                ops.prepare = [&] {
                    if ((old_bytes && (!old || std::count(d.allocs.begin(), d.allocs.end(), (void*)old) != 1)) ||
                        d.experts < old_bytes || l.nm != 1 || l.lay.fmt != gg::kFmtMXFP4 ||
                        !c.stage || c.stage_stride < l.blob) return grow::Prepare::fatal;
                    for (int e : plan.promoted) if (l.ram[(size_t)e] < 0) return grow::Prepare::fatal;
                    try {
                        d.allocs.reserve(d.allocs.size()+1);
                        next_vm = l.vm; next_host = l.host; next_hh = l.hh;
                        tab.assign((size_t)l.n_exp, 0ull);
                    } catch (...) { return grow::Prepare::unavailable; }
                    size_t fr = 0, total = 0;
                    ce = cudaMemGetInfo(&fr, &total);
                    if (ce != cudaSuccess) return grow::Prepare::fatal;
                    // Final configured reserve is preserved; only the old/new
                    // slab overlap may temporarily use it, never below1GiB.
                    if (fr < next_bytes + hard_free || fr < next_bytes-old_bytes + reserve)
                        return grow::Prepare::unavailable;
                    ce = cudaMalloc((void**)&next, next_bytes);
                    if (ce == cudaErrorMemoryAllocation) { (void)cudaGetLastError(); next = nullptr; return grow::Prepare::unavailable; }
                    if (ce != cudaSuccess) return grow::Prepare::fatal;
                    d.allocs.push_back(next);
                    if (old_bytes) {
                        ce = cudaMemcpyAsync(next, old, old_bytes, cudaMemcpyDeviceToDevice, d.st);
                        if (ce != cudaSuccess) return grow::Prepare::fatal;
                    }
                    for (int e : plan.promoted) {
                        if (cancel && cancel()) {
                            ce = cudaStreamSynchronize(d.st);
                            return ce == cudaSuccess ? grow::Prepare::unavailable : grow::Prepare::fatal;
                        }
                        if (!read_mx(l, e, c.stage)) return grow::Prepare::fatal;
                        ce = cudaMemcpyAsync(next+(size_t)plan.slot[(size_t)e]*unit(l), c.stage, unit(l), cudaMemcpyHostToDevice, d.st);
                        if (ce != cudaSuccess) return grow::Prepare::fatal;
                        ce = cudaStreamSynchronize(d.st); // stage is reused only after its DMA completes
                        if (ce != cudaSuccess) return grow::Prepare::fatal;
                        next_host[(size_t)e] = nullptr;
                        next_hh[(size_t)e] = {nullptr,nullptr,nullptr,nullptr};
                    }
                    ce = cudaStreamSynchronize(d.st);
                    if (ce != cudaSuccess) return grow::Prepare::fatal;
                    for (int e = 0; e < l.n_exp; ++e) if (plan.slot[(size_t)e] >= 0) {
                        uint8_t* raw = next+(size_t)plan.slot[(size_t)e]*unit(l);
                        tab[(size_t)e] = (unsigned long long)raw;
                        for (int r = 0; r < 3; ++r) {
                            x3::Mat& m = next_vm[(size_t)e][(size_t)r]; m = x3::Mat{};
                            m.K = gg::expert_type(l.lay,r);
                            m.fmt = m.K == gg::kTypeMXFP4 ? gg::kFmtMXFP4 : gg::kFmtGG;
                            m.raw = raw+l.lay.trellis[r]; m.k = r < 2 ? g.dim : g.ff; m.n = r < 2 ? g.ff : g.dim;
                        }
                    }
                    return grow::Prepare::ready;
                };
                ops.publish_table = [&] {
                    ce = cudaMemcpy(l.vram_tab, tab.data(), tab.size()*8, cudaMemcpyHostToDevice);
                    return ce == cudaSuccess;
                };
                ops.publish_host = [&] {
                    l.vram_base = next;
                    l.slot.swap(plan.slot); l.vm.swap(next_vm); l.host.swap(next_host); l.hh.swap(next_hh);
                    l.n_slots = plan.next_slots; l.n_vram = plan.resident; l.spare_vram = plan.next_spare;
                    for (int e : plan.promoted) { l.ram[(size_t)e] = -1; --ram_held; }
                    d.experts += next_bytes-old_bytes;
                };
                ops.retire_old = [&] {
                    if (!old_bytes) return true;
                    auto it = std::find(d.allocs.begin(), d.allocs.end(), (void*)old);
                    if (it == d.allocs.end() || cudaFree(old) != cudaSuccess) return false;
                    d.allocs.erase(it); return true;
                };
                const grow::Result r = grow::replace(ops);
                if (r.state == grow::State::fatal) return poison("layer "+std::to_string(il)+": "+r.reason);
                if (r.state == grow::State::committed) ++grown; else ++unavailable;
                if (r.stopped) { stopped = true; break; }
            }
        }
        if(!gpu_miss_direct_enabled&&prefill_repeat_rearm&&prefill_repeat_generation) {
            std::fprintf(stderr,"ds4: experimental repeat prefill generation%llu work delta: cold %llu sets/%llu unpacks; Engram %llu fills/%llu hits; dense %llu fills/%llu hits; registration ranges%zu after retirement; sequence%u retained\n",
                (unsigned long long)prefill_repeat_generation,(unsigned long long)(c.cold_f16_sets-prefill_repeat_cold_sets),
                (unsigned long long)(c.cold_f16_unpacks-prefill_repeat_cold_unpacks),
                (unsigned long long)(c.eg_weight_fills-prefill_repeat_eg_fills),(unsigned long long)(c.eg_weight_hits-prefill_repeat_eg_hits),
                (unsigned long long)(c.dense_projection_fills-prefill_repeat_dense_fills),
                (unsigned long long)(c.dense_projection_hits-prefill_repeat_dense_hits),registered_copy.held.size(),seq_issued);
        }
        // Partial growth is valid. RAM arenas/part offsets/spare mappings were
        // never rebuilt; promoted arenas' unused holes remain owned until exit.
        if(!decode_recovery&&normalize_prefill_heat&&!reset_prefill_heat_from_profile(err))return poison(err);
        plan_streams();
        chunk_major_phase_pending = false;
        size_t free_b = 0, total_b = 0;
        ce = cudaMemGetInfo(&free_b, &total_b);
        if (ce != cudaSuccess) return poison("reading final GPU headroom");
        if(gpu_miss_direct_enabled) {
            // This is the transition's commit point. A Stop observed before it
            // cannot create a new DECODE context; later soft Stops may retain a
            // healthy existing context until real maintenance or teardown.
            stopped=stopped||halted||(cancel&&cancel());
            if(retain_decode_pins&&stopped) {
                if(!gpu_misses_retire_pins("phase-stop",err))return poison(err);
                ce=cudaMemGetInfo(&free_b,&total_b);if(ce!=cudaSuccess)return poison("headroom after phase-stop pin retirement");
            } else if(retain_decode_pins) {
                gpu_misses_decode_pins=true;
                if(!gpu_misses_retained_ready())return poison("retained decode pin publication refused");
                st.gpu_miss_retained_ranges=(int64_t)registered_copy.held.size();
                st.gpu_miss_retained_bytes=(int64_t)registered_copy.bytes();
                std::fprintf(stderr,"ds4: GPU_MISS_PIN retained: context=DECODE ranges=%zu bytes=%zu fingerprint=%016llx epoch=%u issued=%u streams_drained=1\n",
                    registered_copy.held.size(),registered_copy.bytes(),(unsigned long long)gpu_misses_pin_fingerprint(),cache_epoch_machine.serial,seq_issued);
            }
            if(!decode_recovery&&prefill_repeat_rearm&&prefill_repeat_generation) {
                std::fprintf(stderr,"ds4: experimental repeat prefill generation%llu work delta: cold %llu sets/%llu unpacks; Engram %llu fills/%llu hits; dense %llu fills/%llu hits; registration ranges%zu after %s; sequence%u retained\n",
                    (unsigned long long)prefill_repeat_generation,(unsigned long long)(c.cold_f16_sets-prefill_repeat_cold_sets),
                    (unsigned long long)(c.cold_f16_unpacks-prefill_repeat_cold_unpacks),
                    (unsigned long long)(c.eg_weight_fills-prefill_repeat_eg_fills),(unsigned long long)(c.eg_weight_hits-prefill_repeat_eg_hits),
                    (unsigned long long)(c.dense_projection_fills-prefill_repeat_dense_fills),
                    (unsigned long long)(c.dense_projection_hits-prefill_repeat_dense_hits),registered_copy.held.size(),
                    gpu_misses_decode_pins?"retention":"retirement",seq_issued);
            }
            pins_committed=true;
            if(!decode_recovery) {
                if(stopped&&!gpu_misses_retained_ready())gpu_misses_recovery_needed=true;
                if(!stopped&&c.C==1024&&c.A==1024)gpu_misses_last_decode_pool=d.experts;
                if(gpu_misses_retained_ready())gpu_misses_recovery_needed=false;
            }
        }
        if(decode_recovery)std::fprintf(stderr,"ds4: GPU_MISS_RECOVERY layout: C=%d A=%d AI=%d layers_grown=%zu skipped=%zu pool_before_bytes=%zu pool_after_bytes=%zu free_bytes=%zu stopped=%d\n",
            c.C,c.A,c.AI,grown,unavailable,before,d.experts,free_b,stopped?1:0);
        else std::fprintf(stderr, "ds4: experimental chunk-major phase ended: C=%d, layers grown=%zu skipped=%zu, "
                             "expert pool %.3f -> %.3f GB, free %.1f MiB, target final reserve%zu MiB (initial%lld MiB), "
                             "temporary hard floor1024 MiB, RAM maps retained, %.3f s%s\n",
                     c.C, grown, unavailable, before/1e9, d.experts/1e9, free_b/1048576.0,
                     final_reserve>>20,(long long)o.reserve_mib,
                     (now_ms()-started)/1000.0, stopped ? "; stopped, valid partial pool retained" : "");
        if (stopped && grow_pool) { halted = true; err = "stopped while preparing the chunk-major decode pool"; return false; }
        return true;
    }


    // End the FIRST-feed memory phase. There were no global expert allocations to retire: only the two tracked
    // scratch arenas are freed, after all prompt work/readers are quiescent. A failed promotion is terminal and
    // reported on every later feed/step, rather than allowing decode through a zero/partially-filled table.
    bool finish_prefill_phase(std::string& err, bool allow_ram_copy = true, bool allow_chunk_shrink = false) {
        if (!prefill_phase_error.empty()) {
            err = prefill_phase_error;
            return false;
        }
        if (!prefill_phase_pending) return true;
        const double t0 = now_ms();
        auto fail = [&](const std::string& why) {
            prefill_phase_pending = false;
            prefill_phase_error = "experimental prefill phase could not restore the decode pool: " + why;
            err = prefill_phase_error;
            std::fprintf(stderr, "ds4: ERROR: %s\n", err.c_str());
            return false;
        };
        if (devs.size() != 1 || chunks.size() != 1 || !chunks[0]) return fail("invalid device/chunk state");
        Dev& d = devs[0];
        Chunk& c = *chunks[0];
        cudaSetDevice(d.id);
        lm_end();   // waits for d.st, c.cs, and every issued host read before dropping host state
        cudaError_t ce = cudaStreamSynchronize(d.st);
        if (ce != cudaSuccess) return fail(std::string("prompt stream: ") + cudaGetErrorString(ce));
        ce = cudaStreamSynchronize(c.cs);
        if (ce != cudaSuccess) return fail(std::string("copy stream: ") + cudaGetErrorString(ce));
        {
            std::lock_guard<std::mutex> lk(sw_mu);
            if (sw_inflight != 0 || ram_inflight != 0 || !sw_queue.empty() || !sw_ready.empty() ||
                !ram_queue.empty() || !ram_ready.empty())
                return fail("a cache move was submitted before promotion");
        }
        if (d.experts != 0) return fail("initial resident expert accounting was not zero");
        for (const Layer& l : L)
            if (l.n_slots != 0 || l.n_vram != 0 || l.vram_base != nullptr || l.swap_pending)
                return fail("initial resident expert plan was not empty");
        auto retire = [&](void* p, size_t bytes) {
            if (p == nullptr) return true;
            const auto it = std::find(d.allocs.begin(), d.allocs.end(), p);
            if (it == d.allocs.end() || d.scratch < bytes) return false;
            if (cudaFree(p) != cudaSuccess) return false;
            d.allocs.erase(it);
            d.scratch -= bytes;
            return true;
        };
        const size_t h_bytes = c.prefill_gpu_h_bytes, exp_bytes = c.lm_gpu_bytes;
        const uint64_t avoided = c.prefill_gpu_h_avoided;
        if (c.lm_f16 != nullptr) {
            std::fprintf(stderr, "ds4: experimental F16 phase: %llu sets in %llu row buckets, %llu valid / %llu padded "
                                 "rows; %llu expert unpacks, %llu cache hits; TC fallbacks light=%llu shape=%llu "
                                 "format=%llu blas=%llu\n",
                         (unsigned long long) c.lm_f16_sets, (unsigned long long) c.lm_f16_buckets,
                         (unsigned long long) c.lm_f16_rows, (unsigned long long) c.lm_f16_padded_rows,
                         (unsigned long long) c.lm_f16_fills, (unsigned long long) c.lm_f16_hits,
                         (unsigned long long) c.lm_f16_fallback_light, (unsigned long long) c.lm_f16_fallback_shape,
                         (unsigned long long) c.lm_f16_fallback_format, (unsigned long long) c.lm_f16_fallback_blas);
            if (c.lm_f16_mixed_sets > 0)
                std::fprintf(stderr, "ds4: experimental F16 mixed-light: %llu sets, %llu exact-light experts, %llu "
                                     "exact-light rows; original full-range NE16/q8/FMA order retained; per-role GPU "
                                     "profile down segment includes delayed light GU (total expert time unchanged)\n",
                             (unsigned long long) c.lm_f16_mixed_sets, (unsigned long long) c.lm_f16_light_experts,
                             (unsigned long long) c.lm_f16_light_rows);
        }
        if (c.lm_f16_blas != nullptr) {
            if (cublasDestroy(c.lm_f16_blas) != CUBLAS_STATUS_SUCCESS) return fail("retiring the phase cuBLAS handle");
            c.lm_f16_blas = nullptr;   // its lazy workspaces are gone BEFORE the normal GPU placement budget is measured
        }
        for (cudaEvent_t& e : c.lm_f16_meta_done) {
            if (e != nullptr && cudaEventDestroy(e) != cudaSuccess) return fail("retiring F16 metadata events");
            e = nullptr;
        }
        if (!retire(c.lm_f16, c.lm_f16_bytes)) return fail("retiring the F16 active-layer arena");
        c.lm_f16 = nullptr;
        c.lm_f16_bytes = c.lm_f16_unit = 0;
        c.lm_f16_il = -1;
        c.lm_f16_valid.clear();
        if (!retire(c.lm_f16_y, c.lm_f16_y_bytes)) return fail("retiring F16 padded down output");
        c.lm_f16_y = nullptr;
        c.lm_f16_y_bytes = 0;
        if (!retire(c.lm_f16_meta_dev, c.lm_f16_meta_bytes)) return fail("retiring F16 device metadata");
        c.lm_f16_meta_dev = nullptr;
        if (c.lm_f16_meta_host != nullptr) {
            const auto it = std::find(host_allocs.begin(), host_allocs.end(), (void*) c.lm_f16_meta_host);
            if (it == host_allocs.end() || cudaFreeHost(c.lm_f16_meta_host) != cudaSuccess)
                return fail("retiring F16 pinned metadata");
            host_allocs.erase(it);
            c.lm_f16_meta_host = nullptr;
        }
        c.lm_f16_meta_bytes = c.lm_f16_meta_stride = 0;
        if (!retire(c.prefill_gpu_h, h_bytes)) return fail("retiring the F32 H arena");
        c.prefill_gpu_h = nullptr;
        c.prefill_gpu_h_bytes = c.prefill_gpu_h_rows = 0;
        if (!retire(c.lm_gpu, exp_bytes)) return fail("retiring the compressed expert arena");
        c.lm_gpu = nullptr;
        c.lm_gpu_bytes = c.lm_gpu_stride = c.lm_gpu_used = 0;
        c.lm_gpu_il = -1;
        c.lm_gpu_of.clear();
        if (prefill_decode_chunk > 0) {
            if (allow_chunk_shrink) {
                std::string shrink_error;
                if (!shrink_prefill_chunk(d, c, shrink_error)) return fail(shrink_error);
            } else {
                std::fprintf(stderr, "ds4: experimental decode chunk shrink skipped: first feed failed/Stopped, was short, "
                                     "or did not complete the multi-chunk phase; original C=%d retained\n", c.C);
            }
        }
        // Stop using the phase before placement: any error after here is fail-closed, never a second placement
        // over a partly populated pool. Its original pinned tab_pin and fixed vram_tab allocations are reused.
        prefill_phase_pending = false;
        std::string load_err;
        if (!place_experts(load_err) || !load_experts_mx(load_err, true)) return fail(load_err);
        bool copy_interrupted = false;
        if (prefill_copy_ram) {
            if (!allow_ram_copy) {
                std::fprintf(stderr, "ds4: experimental late RAM copy skipped after failed/Stopped first feed; "
                                     "phase consumed, mapped CPU experts retained for this process\n");
            } else {
                // The normal GPU pool (including all fitting DSpark experts) now exists. Copy ONLY what it left on
                // the CPU lane, after the ~1M-token host state and both GPU phase arenas have been retired.
                struct Hot { int il, e; double share; };
                std::vector<Hot> ranked;
                for (const Layer& l : L) {
                    const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
                    double sum = 0;
                    for (int e = 0; e < l.n_exp; ++e) sum += p[e];
                    for (int e = 0; e < l.n_exp; ++e)
                        if (l.slot[(size_t) e] < 0) ranked.push_back({l.il, e, sum > 0 ? p[e] / sum : 0.0});
                }
                std::stable_sort(ranked.begin(), ranked.end(), [](const Hot& a, const Hot& b) { return a.share > b.share; });
                std::vector<std::pair<int, int>> hot;
                hot.reserve(ranked.size());
                for (const Hot& h : ranked) hot.push_back({h.il, h.e});
                if (!copy_experts(hot, load_err, true, &copy_interrupted)) return fail(load_err);
                if (copy_interrupted)
                    std::fprintf(stderr, "ds4: experimental late RAM copy cancelled before publication; optional "
                                         "copy buffers discarded, phase consumed, mapped CPU experts retained for this process\n");
                else
                    std::fprintf(stderr, "ds4: experimental late RAM copy complete: %d / %zu final CPU experts copied, "
                                         "NH=%d, before first token\n", ram_held, hot.size(), NH);
            }
        }
        plan_streams();
        size_t slots = 0;
        for (const Layer& l : L) slots += (size_t) l.n_vram;
        std::fprintf(stderr, "ds4: experimental prefill phase ended: freed %.1f MiB F32 H + %.1f MiB expert cache; "
                             "avoided %.6f GB H PCIe transfers; normal decode pool %zu experts, %.3f GB loaded in %.3f s "
                             "before first token\n",
                     (double) h_bytes / 1048576.0, (double) exp_bytes / 1048576.0, (double) avoided / 1e9,
                     slots, (double) d.experts / 1e9, (now_ms() - t0) / 1000.0);
        if (copy_interrupted) {
            halted = true;
            err = "stopped while preparing the CPU experts; normal GPU pool restored, mapped CPU experts retained";
            return false;   // recoverable Stop, not prefill_phase_error: the next request remains usable
        }
        return true;
    }

    // the first layer of a long prompt's second half (forward_chunks' two passes), or -1: the first with compress ratio 1
    // (V4.1: 20), every one after it ratio 1 too and owning no compressed KV or index keys of its own (they read its),
    // every trunk layer on one GPU.  The settings' MAYA_DS4_PROMPT_SKIP (STRATA_DS4_PROMPT_SKIP; DS4_YOCO for tests):
    // 1 (the default) the skip, 2 DeepSeek's own serving's (the last 128 rows only: faster, the answers shift a
    // little), 0 off
    static int yoco_mode() {
        static const int m = [] {
            const char* v = std::getenv("DS4_YOCO");
            if (v == nullptr) v = std::getenv("STRATA_DS4_PROMPT_SKIP");
            return v != nullptr ? std::atoi(v) : 1;
        }();
        return m;
    }
    int yoco_layer() const {
        if (yoco_mode() == 0 || chunk <= 0 || g.nl < 2 || (int) g.ratios.size() < g.nl) return -1;
        int yl = -1;
        for (int il = 0; il < g.nl && yl < 0; ++il)
            if (g.ratios[(size_t) il] == 1) yl = il;
        if (yl <= 0 || !L[(size_t) yl].kv_src) return -1;
        for (int il = 0; il < g.nl; ++il) {
            const Layer& l = L[(size_t) il];
            if (l.nm != 1 || l.dev != L[0].dev) return -1;
            if (il > yl && (g.ratios[(size_t) il] != 1 || l.kv_src)) return -1;
        }
        return yl;
    }

    // n tokens at positions pos .. in chunks of equal size (at most `chunk`), pipelined over the groups; the logits of
    // the last min(logits_rows, 16) rows (logits_rows > 16: a run of one chunk, every row's into chunk_logits)
    bool forward_chunks(const int* tk, int n, int logits_rows, std::string& err, int snap_slot = -1) {
        if (n < 1) return true;
        if (!prefill_phase_error.empty()) { err = prefill_phase_error; return false; }
        if (!chunk_major_phase_error.empty()) { err=chunk_major_phase_error; return false; }
        struct RestoreChunkMajor {
            Impl* p; std::string& err; bool was_phase; bool succeeded=false;
            ~RestoreChunkMajor() {
                if(!was_phase||succeeded)return;
                // halted is the caller's recognized Stop result. The raw halt signal alone must not hide a
                // different device error (the worker-error loop explicitly gives real errors priority over Stop).
                const bool stopped=p->halted;
                if(p->chunk_major_phase_pending) {
                    std::string why;
                    if(!p->finish_chunk_major_phase(why,false,false)) {
                        if(!err.empty())err+="; ";err+=why;
                    }
                }
                if(!stopped) {
                    // Includes cache/seed/checkpoint failures AFTER successful phase publication. Do not use
                    // stale caller-owned err as a success test; the explicit succeeded latch alone admits reuse.
                    if(err.empty())err="chunk-major first feed failed without an error detail";
                    p->chunk_major_phase_pending=false;p->chunk_major_phase_error=err;
                    std::fprintf(stderr,"ds4: ERROR: chunk-major first feed failed; process poisoned: %s\n",err.c_str());
                }
            }
        } restore_chunk_major{this,err,chunk_major_phase_pending};
        struct RestorePhase {   // includes invalid input, Stop, and any prompt error; preserve the original error
            Impl* p;
            std::string& err;
            ~RestorePhase() {
                if (!p->prefill_phase_pending) return;
                std::string why;
                if (!p->finish_prefill_phase(why, false)) {
                    if (!err.empty()) err += "; ";
                    err += why;
                }
            }
        } restore_phase{this, err};
        if (pos + n > o.max_context) {
            err = "the context is full";
            return false;
        }
        for (int t = 0; t < n; ++t)
            if (tk[t] < 0 || tk[t] >= g.vocab) {
                err = "token " + std::to_string(tk[t]) + " out of range";
                return false;
            }
        // a long prompt's skip: the layers from yoco_layer() on (V4.1: 20-39) read the earlier positions only through
        // that layer's compressed KV and index keys, and each looks back one window (win - 1 positions) more than the
        // layer before - so they need their own work only on the last keep = (nl - yoco) (win - 1) + win rows for the
        // run's last rows and the rings the decode reads to be what the full work gives (up to the experts' summing
        // order).  The chunks stay as they are: the ones before the cut go through the layers before yoco and its
        // kv_only, the cut's chunk through those over all its rows and through the rest from the cut on (Run::r0, a
        // view of its buffers), the chunks after it through every layer - each layer's experts read as without the
        // skip.  One GPU (DS4_YOCO=1; 0: off); not with every row's logits (tests)
        const int yl = logits_rows > kWin ? -1 : yoco_layer();
        // (DS4_YOCO_KEEP=<rows>: fewer - DeepSeek's own serving replays the last 128 tokens through those layers
        //  (DeepInfer's "SWA bounded replay"), which shifts the answers a little; at least one window)
        static const int keep_env = std::getenv("DS4_YOCO_KEEP") ? std::atoi(std::getenv("DS4_YOCO_KEEP")) : 0;
        const int keep = yl <= 0 ? 0
                         : keep_env > 0 ? std::max(keep_env, g.win)
                         : yoco_mode() == 2 ? g.win : (g.nl - yl) * (g.win - 1) + g.win;
        const int a = yl > 0 && n - keep >= kYocoMin ? n - keep : 0;
        const int nc = (n + chunk - 1) / chunk, per = (n + nc - 1) / nc;
        Run run;
        for (int x = 0; x < n; x += per) {
            run.p0.push_back(pos + x);
            run.T.push_back(std::min(per, n - x));
            run.seq.push_back(++win_seq);
        }
        if (a > 0) {
            run.yoco = yl;
            run.cut_k = a / per;
            run.cut_r = a - run.cut_k * per;
        }
        logits_all = logits_rows > kWin;
        if (logits_all) chunk_logits.resize((size_t) n * g.vocab);
        for (int t = 0; t < n; ++t) {
            toks.push_back(tk[t]);
            ctoks.push_back(ctok(pos + t, tk[t]));
        }
        plan_streams();
        if (snap_slot >= 0) {   // feed_mark: the state before the run's last row, kept by its layers on the way
            run.snap_pos = pos + n - 1;
            for (int il = 0; il < g.nl; ++il) {
                Layer& l = L[(size_t) il];
                Dev& d = devs[(size_t) l.dev];
                if (l.snap_win == nullptr && !dalloc(d, l.snap_win, (size_t) g.win * win_rb(), err)) return false;
                if (l.c_kv != nullptr && l.snap_kv == nullptr &&
                    (!dalloc(d, l.snap_kv, (size_t) l.ratio * g.hd, err) || !dalloc(d, l.snap_sc, (size_t) l.ratio * g.hd, err)))
                    return false;
            }
        }
        const int T_last = run.T.back();
        n_logits = logits_all ? 0 : std::min(logits_rows, T_last);
        const double tl = now_ms();
        {   // each chunk's end on the head device, for the progress reports
            cudaSetDevice(devs[(size_t) head_dev].id);
            while (chunk_done.size() < run.T.size()) {
                cudaEvent_t e = nullptr;
                CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
                chunk_done.push_back(e);
            }
            chunk_rec.reset(new std::atomic<int>[chunk_done.size()]);
            for (size_t i = 0; i < chunk_done.size(); ++i) chunk_rec[i].store(0);
        }
        prof_chunk = prof_on;
        const bool lm = lm_begin(run);
        if (chunk_major_phase_pending && lm) { err="chunk-major phase unexpectedly entered layer-major"; return false; }
        if (prefill_phase_pending && !lm) {
            std::fprintf(stderr, "ds4: experimental prefill phase: layer-major prerequisites unavailable; "
                                 "restoring the normal pool before the ordinary prompt path\n");
            if (!finish_prefill_phase(err)) return false;
        }
        if (chunks.size() == 1 && chunks[0]->lm_gpu != nullptr)
            std::fprintf(stderr, "ds4: experimental GPU expert cache: %s for %zu chunks; %.1f MiB arena, "
                                 "%.1f GB host-state capacity\n",
                         lm ? "layer-major active" : "inactive (layer-major prerequisites did not fit)",
                         run.T.size(), (double) chunks[0]->lm_gpu_bytes / 1048576.0,
                         (double) (run.T.size() * chunk_hand_bytes()) / 1e9);
        struct LmEnd {   // (every way out: its host memory freed once the device is done with it)
            Impl* p;
            bool on;
            ~LmEnd() {
                if (on) p->lm_end();
            }
        } lm_end_{this, lm};
        std::vector<std::string> errs(devs.size());
        halted = false;
        halt.store(false);
        {
            std::lock_guard<std::mutex> lk(halt_mu);
            halt_k = INT_MAX;
            halt_seen = -1;
        }
        bool halting = false;
        for (size_t di = 0; di < devs.size(); ++di)
            workers[di]->submit([this, di, &run, &errs] {
                std::string e;
                if (!run_dev_chunks((int) di, run, e)) errs[di] = e.empty() ? "a device's thread failed" : e;
            });
        size_t reported = 0;
        for (;;) {
            bool all = true;
            for (auto& w : workers) all = all && w->idle();
            if (!all && !halting && cancel && cancel()) {   // a Stop: the devices end the run early
                halting = true;
                std::lock_guard<std::mutex> lk(halt_mu);
                halt_k = halt_seen + 1;
                halt.store(true, std::memory_order_release);
            }
            if (progress && lm) {   // (layer-major: the share of the layers every chunk went through)
                const int nd = lm_layers.load(std::memory_order_acquire);
                if (nd > 0 && nd < g.nl) progress((int64_t) n * nd / g.nl);
            } else if (progress) {   // chunks through every layer (the last layer's device past their event)
                while (reported < run.T.size() && chunk_rec[reported].load(std::memory_order_acquire) &&
                       cudaEventQuery(chunk_done[reported]) == cudaSuccess)
                    ++reported;
                if (reported > 0 && reported < run.T.size())
                    progress(run.p0[reported - 1] + run.T[reported - 1] - run.p0[0]);
            }
            if (all) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        for (const std::string& e : errs)   // (a device's failure first: a Stop's early end is not one)
            if (!e.empty() && e != kHalted) {
                err = e;
                return false;
            }
        for (const std::string& e : errs)
            if (e == kHalted) {
                halted = true;
                err = "stopped while reading the prompt";
                return false;
            }
        st.layer_ms += now_ms() - tl;
        eg_stats();
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        const double th = now_ms();
        CK(cudaStreamSynchronize(d.st));
        st.head_ms += now_ms() - th;
        if (prof_chunk) {   // DS4_PROF: the run's GPU time by stage, every device's summed
            for (Dev& dv : devs) {
                cudaSetDevice(dv.id);
                CK(cudaStreamSynchronize(dv.st));
            }
            cudaSetDevice(d.id);
            prof_collect();
            std::fprintf(stderr,
                         "ds4 prompt GPU time, %d tokens: attention %.0f ms, routing + shared expert %.0f ms, experts %.0f ms "
                         "(with the waits for streamed ones), between layers %.0f ms\n",
                         n, cprof_ms[1], cprof_ms[2], cprof_ms[3], cprof_ms[0]);
            {   // the finer table (ms, % of the sum)
                static const char* nm_[16] = {"between layers", "attn rest (q_b, wo_a/b, hc_post)", "shared expert", "experts (incl waits)",
                                              "engram", "attn mHC (mix/split/norm)", "attn projections", "compressor + indexer",
                                              "q_b + rope", "sparse attention", "ffn mHC + router + sort", "", "", "", "", ""};
                double sum = 0;
                for (double v : cprof_ms) sum += v;
                for (int i = 0; i < 11; ++i)
                    std::fprintf(stderr, "ds4 prof  %-34s %9.1f ms %5.1f%%\n", nm_[i], cprof_ms[i], sum > 0 ? 100.0 * cprof_ms[i] / sum : 0.0);
                struct Agg {
                    double ms = 0, flops = 0;
                    long calls = 0, rows = 0, experts = 0;
                };
                std::map<std::pair<int, int>, Agg> agg;   // (type, role 0 gate+up, 1 down)
                {
                    std::lock_guard<std::mutex> lk(prof_mu);
                    for (const EkMark& m : ek_marks) {
                        float a = 0, b = 0;
                        cudaSetDevice(m.dev);
                        cudaEventElapsedTime(&a, m.e0, m.e1);
                        cudaEventElapsedTime(&b, m.e1, m.e2);
                        Agg& x = agg[{m.tg, 0}];
                        x.ms += a;
                        x.flops += 2.0 * 2.0 * m.nrows * (double) g.dim * (double) m.ff;
                        x.calls++, x.rows += m.nrows, x.experts += m.nexp;
                        Agg& y = agg[{m.td, 1}];
                        y.ms += b;
                        y.flops += 2.0 * m.nrows * (double) g.dim * (double) m.ff;
                        y.calls++, y.rows += m.nrows, y.experts += m.nexp;
                        ek_pool[m.dev].push_back(m.e0);
                        ek_pool[m.dev].push_back(m.e1);
                        ek_pool[m.dev].push_back(m.e2);
                    }
                    ek_marks.clear();
                }
                double tot = 0;
                for (auto& kv : agg) tot += kv.second.ms;
                for (auto& kv : agg)
                    std::fprintf(stderr,
                                 "ds4 prof  expert kernels type %-8s %-7s %9.1f ms  calls %5ld  experts %6ld (%.1f us each, %.1f a call)  rows/expert %6.1f  ~%5.1f TFLOPS\n",
                                 gg::iq::name(kv.first.first), kv.first.second ? "down" : "gate+up", kv.second.ms, kv.second.calls,
                                 kv.second.experts, kv.second.experts > 0 ? 1e3 * kv.second.ms / (double) kv.second.experts : 0.0,
                                 kv.second.calls > 0 ? (double) kv.second.experts / (double) kv.second.calls : 0.0,
                                 kv.second.experts > 0 ? (double) kv.second.rows / (double) kv.second.experts : 0.0,
                                 kv.second.ms > 0 ? kv.second.flops / kv.second.ms / 1e9 : 0.0);
                std::fprintf(stderr, "ds4 prof  expert kernels total %.1f ms\n", tot);
            }
            for (double& v : cprof_ms) v = 0;
            prof_chunk = false;
        }
        if (std::getenv("DS4_INDEX_STATS") != nullptr) {
            for (Dev& dv : devs) {
                cudaSetDevice(dv.id);
                CK(cudaStreamSynchronize(dv.st));
                index_native_profile_dump(dv.st, true);
            }
            cudaSetDevice(d.id);
        }
        if (o.verbose || std::getenv("DS4_FLOW_STATS") != nullptr) {
            const double ms = now_ms() - tl;
            for (size_t di = 0; di < chunks.size(); ++di) {
                const Chunk& c = *chunks[di];
                if (!c.flow) continue;
                const double db = (double) (c.io->direct_bytes() - c.io_d0) / 1e9;
                const double bb = (double) (c.io->buffered_bytes() - c.io_b0) / 1e9;
                const double fb = db + bb, rb = (double) c.it_ram * (double) L[0].blob / 1e9;   // (a member: its parts)
                char lcs[160] = "";
                if (c.lm)
                    std::snprintf(lcs, sizeof lcs, ", %llu again from the layer cache (layer-major, %d chunks, %.1f GB "
                                  "of RAM)", (unsigned long long) c.it_lc, (int) run.T.size(), (double) c.lc_bytes / 1e9);
                std::fprintf(stderr,
                             "ds4 prompt stream, CUDA%d: %d tokens in %.0f ms; %llu experts streamed: %llu from the files "
                             "(%.1f GB, %.2f GB/s: %.1f GB from the disk unbuffered, %.1f GB through the page cache), %llu "
                             "from RAM (%.1f GB)%s; %llu likely ones queued early, %llu of them unpicked; the compute waited "
                             "%.0f ms for reads\n",
                             devs[di].id, n, ms, (unsigned long long) (c.it_file + c.it_ram + c.it_lc),
                             (unsigned long long) c.it_file, fb, fb / std::max(1e-3, ms / 1000.0), db, bb,
                             (unsigned long long) c.it_ram, rb, lcs, (unsigned long long) c.it_spec,
                             (unsigned long long) c.it_spec_unused, c.wait_ms);
                if(register_ram_all||register_ram_mb>0)
                    std::fprintf(stderr,"ds4 registered-copy flow, CUDA%d: engine-accounted H2D blob bytes %.0f, "
                        "actual H2D role bytes %llu, direct registered CPU bytes %llu, bounced CPU bytes %llu; "
                        "wait %.3f ms includes I/O and staging-event/ring backpressure; not a RAM bandwidth measurement\n",
                        devs[di].id,c.stream_bytes-c.registered_stream_begin,
                        (unsigned long long)c.registered_h2d_payload_bytes,
                        (unsigned long long)c.registered_direct_bytes,(unsigned long long)c.registered_bounce_bytes,c.wait_ms);
            }
        }
        for (size_t di = 0; di < chunks.size(); ++di) {
            const Chunk& c = *chunks[di];
            if (c.lm_gpu == nullptr) continue;
            std::fprintf(stderr, "ds4 experimental GPU expert cache, CUDA%d: hits %llu, fills %llu, "
                                 "host reads skipped %llu, no-room H2D %llu; expert H2D %.6f GB, "
                                 "H2D avoided %.6f GB, cache/ring D2D %.6f GB, peak layer slots %zu\n",
                         devs[di].id, (unsigned long long) c.lm_gpu_hits, (unsigned long long) c.lm_gpu_fills,
                         (unsigned long long) c.lm_gpu_read_skips, (unsigned long long) c.lm_gpu_no_room,
                         (double) c.lm_gpu_h2d_bytes / 1e9, (double) c.lm_gpu_saved_h2d_bytes / 1e9,
                         (double) c.lm_gpu_d2d_bytes / 1e9, c.lm_gpu_peak_used);
        }
        if (pm.on && run.T.size() == 1) {   // (the picks' spread for pick_path: the CPU-lane experts this run picked)
            bool tp = false;
            for (const Layer& l : L) tp = tp || l.nm > 1;
            double picked = 0;
            for (const auto& cp : chunks)
                if (cp != nullptr && cp->flow) picked += (double) (cp->it_file + cp->it_ram) - (double) cp->it_spec_unused;
            if (!tp) pm_learn(n, picked);
        }
        if (logits_all) n_logits = (T_last - 1) % kWin + 1;   // (sample() takes the last row of d.logits)
        pos += n;
        st.tokens += n;
        // No swap may be queued against the initial empty plan. Retire phase buffers and refill the NORMAL pool
        // while this feed is still timed, before its first cache tick, settle, sample or decode.
        if (prefill_phase_pending) {
            if (!finish_prefill_phase(err, true, true)) {
                lm_end_.on = false;
                // A Stop during the optional late RAM copy happened AFTER the complete trunk prefill. Keep its
                // DSpark rings coherent if the next request reuses that completed prefix without replaying it.
                if (halted && prefill_phase_error.empty()) {
                    std::string seed_err;
                    const int nr = std::min(T_last, tap_cap);
                    if (!seed_drafter(pos - nr, nr, 0, seed_err)) {
                        err += "; restoring the drafter failed: " + seed_err;
                        prefill_phase_error = err;   // a partially seeded drafter must never be reused
                    }
                }
                return false;
            }
            lm_end_.on = false;
        }
        if (chunk_major_phase_pending && !finish_chunk_major_phase(err)) {
            if (halted && chunk_major_phase_error.empty()) {
                std::string seed_err; const int nr=std::min(T_last,tap_cap);
                if (!seed_drafter(pos-nr,nr,0,seed_err)) {
                    err += "; restoring the drafter failed: "+seed_err;
                    chunk_major_phase_error=err;
                }
            }
            return false;
        }
        cache_tick(n);
        if (lm_end_.on) {   // (the layer cache's RAM back before the store and the swaps want it)
            lm_end();
            lm_end_.on = false;
        }
        cache_settle();
        const int nr = std::min(T_last, tap_cap);
        if (!seed_drafter(pos - nr, nr, 0, err)) return false;
        if (snap_slot >= 0) {   // checkpoint `snap_slot` from what the layers kept (the drafter's blocks: as they are -
            // the next feed seeds them again from its own rows, past this position)
            Check& ckp = ck[snap_slot & 1];
            ckp.pos = run.snap_pos;
            ckp.win.resize(L.size());
            ckp.comp.resize(L.size() * 2);
            for (size_t i = 0; i < L.size(); ++i) {
                Layer& l = L[i];
                Dev& dv = devs[(size_t) l.dev];
                cudaSetDevice(dv.id);
                const bool kept = !l.mtp && l.snap_win != nullptr;
                ckp.win[i].resize((size_t) g.win * win_rb());
                CK(cudaMemcpyAsync(ckp.win[i].data(), kept ? l.snap_win : l.win, ckp.win[i].size(),
                                   cudaMemcpyDeviceToHost, dv.st));
                if (l.c_kv) {
                    ckp.comp[2 * i].resize((size_t) l.ratio * g.hd);
                    ckp.comp[2 * i + 1].resize((size_t) l.ratio * g.hd);
                    CK(cudaMemcpyAsync(ckp.comp[2 * i].data(), kept && l.snap_kv ? l.snap_kv : l.c_kv,
                                       ckp.comp[2 * i].size() * 4, cudaMemcpyDeviceToHost, dv.st));
                    CK(cudaMemcpyAsync(ckp.comp[2 * i + 1].data(), kept && l.snap_sc ? l.snap_sc : l.c_score,
                                       ckp.comp[2 * i + 1].size() * 4, cudaMemcpyDeviceToHost, dv.st));
                }
            }
            for (Dev& dv : devs) {
                cudaSetDevice(dv.id);
                CK(cudaStreamSynchronize(dv.st));
            }
        }
        restore_chunk_major.succeeded=true;
        return true;
    }

    // the window's plan, in window order (the indexer's and the candidate source's sizes carry from window to window)
    void plan_window(int64_t p0, int T, WinPlan& wp) {
        wp.par = par;
        wp.async = (par || (!sync_decode && T <= async_max_t)) && !gpumiss::window(gpu_misses_enabled,T);
        wp.vl = 0;
        for (int t = 0; t < T && !imgs.empty(); ++t)
            if (in_image(p0 + t)) wp.vl |= 1u << t;
        for (int il = 0; il < g.nl; ++il) {
            const Layer& l = L[(size_t) il];
            if (l.ratio > 0) {
                const int R = l.ratio;
                int n[kWin], k[kWin], clen_max = 0;
                for (int t = 0; t < T; ++t) {
                    n[t] = (int) ((p0 + t + 1) / R);
                    k[t] = std::min(g.ix_topk, n[t]);
                    clen_max = std::max(clen_max, n[t]);
                }
                if (l.ix_src) {
                    for (int t = 0; t < T; ++t) wp.cand[il][t] = cur_cand_n[t];
                    if (clen_max > 0 && l.cand_src)
                        for (int t = 0; t < T; ++t) cur_cand_n[t] = std::min(g.cand_topk, (n[t] + g.cand_blk - 1) / g.cand_blk);
                    for (int t = 0; t < T; ++t) cur_topk_n[t] = k[t];
                }
                for (int t = 0; t < T; ++t) wp.topk[il][t] = cur_topk_n[t];
            }
            if (wp.async) {
                wp.seq[il] = ++seq_issued;
                layer_of[wp.seq[il] % kRing].store(il, std::memory_order_release);
            }
        }
    }

    int head_nm() const { return (int) groups[(size_t) devs[(size_t) head_dev].grp].mem.size(); }
    // the head's columns on member mi of its group of nm: an EXL3 head's split128 of the vocabulary (mat's), a GGUF's
    // BF16 / Q8_0 head's even shares (mat_bf16's) - head_share's, its gather's and the markov steps' rows
    void head_cols(int nm, int mi, int& v0, int& vn) const {
        const Tensor* tw = pk.get("head.weight");
        if (tw != nullptr && (tw->dtype == "BF16" || tw->dtype == "Q8_0")) {
            vn = g.vocab / nm;
            v0 = mi * vn;
        } else
            split128(g.vocab, nm, mi, v0, vn);
    }
    // the head over `rows` (<= 16) normed rows xn on member mi of the head group (device d): its share of each row's
    // logits (columns [v0, v0 + vn), head_cols); the lead gathers the others' into d.logits (rows V apart) - two
    // slots by seq (Vmax apart, the widest share's), a member reusing one once the lead acknowledged it.  gather
    // false: each member keeps its share (the drafter's markov steps over the group, mk_split)
    bool head_share(Dev& d, int mi, const float* xn, int rows, std::string& err, bool gather = true) {
        const int V = g.vocab, D = g.dim, nm = head_nm();
        int v0 = 0, vn = 0, Vmax = 0, z = 0;
        head_cols(nm, mi, v0, vn);
        head_cols(nm, 0, z, Vmax);
        Group& G = groups[(size_t) d.grp];
        float* out = mi == 0 ? d.logits : d.logits_part;
        const size_t ld = mi == 0 ? (size_t) V : (size_t) vn;
        cudaSetDevice(d.id);
        // (an EXL3 head: as many activations at a time as its split-K partials fit the scratch - 8 for a shared head, 4
        // for a whole one, each pass reading the whole share; a GGUF's: 4 at a time, as before)
        const int per = d.head.fmt != 0 ? 4 : std::max(1, x3::mv_cols(d.head));
        for (int c = 0; c < rows; c += per)
            if (!mv(d, {job(d.head, xn + (size_t) c * D, out + (size_t) c * ld, std::min(per, rows - c), D, ld)}, err))
                return false;
        if (nm == 1 || !gather) return true;
        const unsigned seq = ++d.head_seq;
        const int slot = (int) (seq % 2);
        auto stage = [&](int m) { return G.lg_stage + ((size_t) slot * (nm - 1) + (m - 1)) * kWin * Vmax; };
        auto flag = [&](int m) { return G.lg_flag + ((size_t) slot * (nm - 1) + (m - 1)) * 32; };
        if (mi > 0) {
            if (seq > 2) wait_flags(G.lg_ack + (size_t) slot * 32, 1, 32, seq - 2, d.st);
            Xfer x;
            x.n = 1;
            x.src[0] = d.logits_part;
            x.dst[0] = stage(mi);
            x.bytes[0] = (size_t) rows * vn * 4;
            xfer_put(x, flag(mi), seq, d.xcnt, d.st);
            return true;
        }
        for (int m = 1; m < nm; ++m) {
            int m0 = 0, mn = 0;
            head_cols(nm, m, m0, mn);
            Xfer x;
            x.n = 1;
            x.src[0] = stage(m);
            x.dst[0] = d.logits_part;
            x.bytes[0] = (size_t) rows * mn * 4;
            xfer_get(x, flag(m), seq, d.st);
            CK(cudaMemcpy2DAsync(d.logits + (size_t) m0, (size_t) V * 4, d.logits_part, (size_t) mn * 4,
                                 (size_t) mn * 4, (size_t) rows, cudaMemcpyDeviceToDevice, d.st));
        }
        set_flag(G.lg_ack + (size_t) slot * 32, seq, d.st);
        return true;
    }

    // the head's rows (the last n_logits of the window) on head group member di (its h / pre: the group's)
    bool head_rows_dev(int di, int T, std::string& err) {
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        const int D = g.dim, H = g.hc;
        if (n_logits <= 0) return true;
        const int t0 = T - n_logits;
        hc_pre_norm(d.h + (size_t) t0 * H * D, d.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, d.st);
        return head_share(d, d.mi, d.x, n_logits, err);
    }

    // the DSpark drafter's taps: the streams' mean entering target layer il, on device d (after the last seeding read
    // h_taps)
    bool tap(Dev& d, int il, int T, std::string& err) {
        const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
        if (g.n_mtp == 0 || it == g.d_targets.end()) return true;
        const size_t j = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
        const int D = g.dim;
        cudaSetDevice(d.id);
        hc_mean(d.h, T, g.hc, D, d.tap, D, d.st);
        CK(cudaStreamWaitEvent(d.st, ev_seed, 0));
        CK(cudaMemcpy2DAsync(h_taps + j * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4, (size_t) T,
                             cudaMemcpyDeviceToHost, d.st));
        return true;
    }

    bool forward(const int* tk, int T, int logits_rows, std::string& err, int seed_rows = -1) {
        if (T < 1 || T > kWin) {
            err = "a window of " + std::to_string(T) + " tokens";
            return false;
        }
        if (pos + T > o.max_context) {
            err = "the context is full";
            return false;
        }
        for (int t = 0; t < T; ++t)
            if (tk[t] < 0 || tk[t] >= g.vocab) {
                err = "token " + std::to_string(tk[t]) + " out of range";
                return false;
            }
        const int64_t p0 = pos;
        slot_held = false;   // (the state grows now: what slot_save took is no longer the conversation)
        for (int t = 0; t < T; ++t) {
            toks.push_back(tk[t]);
            ctoks.push_back(ctok(pos + t, tk[t]));
        }
        // the Engram rows asked for now (the first Engram layer's first), collected right before their layer: what is
        // left of their reads by then is the window's Engram time
        const double te = now_ms();
        const int64_t eg_w0 = eg_wait_ns.load();
        for (int li = 0; li < pk.eg_layers; ++li) {
            EgBatch& b = *eg_win[(size_t) li];
            if (b.live) eg_finish(b);   // (a window that failed midway: its reads end before h_eg is reused)
            eg_issue(p0, T, li, nullptr, b);
        }
        st.engram_ms += now_ms() - te;
        const int D = g.dim;
        for (int t = 0; t < T; ++t)   // (h_emb is written again by the next window, after this one's last sync)
            std::memcpy(h_emb + (size_t) t * D, emb_row(p0 + t, tk[t]), (size_t) D * 2);
        plan_window(p0, T, wplan);
        prof_window = prof_on && T == 1;
        n_logits = std::min(logits_rows, T);
        const double tl = now_ms();
        if (par) {
            if (!forward_par(p0, T, err)) return false;
        } else if (!forward_serial(p0, T, err)) return false;
        st.layer_ms += now_ms() - tl;
        for (auto& b : eg_win)
            if (b->live) eg_finish(*b);
        st.engram_ms += (double) (eg_wait_ns.load() - eg_w0) / 1e6;
        eg_stats();
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        const double th = now_ms();
        CK(cudaStreamSynchronize(d.st));
        st.head_ms += now_ms() - th;
        prof_collect();
        pos += T;
        st.tokens += T;
        cache_tick(T);
        if (seed_rows != 0 && !seed_drafter(p0, seed_rows < 0 ? T : seed_rows, 0, err)) return false;
        return true;
    }

    // one thread: the layers device after device, the hand-offs through pinned memory and events
    bool forward_serial(int64_t p0, int T, std::string& err) {
        const int D = g.dim, H = g.hc;
        int cur = L[0].dev;
        {
            Dev& d = devs[(size_t) cur];
            cudaSetDevice(d.id);
            CK(cudaMemcpyAsync(d.emb_rows, h_emb, (size_t) T * D * 2, cudaMemcpyHostToDevice, d.st));
            embed(d.emb_rows, T, d.h, D, H, d.st);
            init_pre(d.pre, T, d.st);
        }
        std::vector<float> dump;
        if (dump_h != nullptr) dump.resize((size_t) T * g.nl * H * D);
        for (int il = 0; il < g.nl; ++il) {
            Layer& l = L[(size_t) il];
            if (l.dev != cur) {
                Dev& a = devs[(size_t) cur];
                Dev& b = devs[(size_t) l.dev];
                cudaSetDevice(a.id);
                const size_t nh = (size_t) T * H * D * 4, np = (size_t) T * 8 * 4, nk = (size_t) T * topk_stride() * 4,
                             nc = (size_t) T * cand_stride() * 4;
                uint8_t* bb = a.bounce;
                CK(cudaMemcpyAsync(bb, a.h, nh, cudaMemcpyDeviceToHost, a.st));
                CK(cudaMemcpyAsync(bb + nh, a.pre, np, cudaMemcpyDeviceToHost, a.st));
                CK(cudaMemcpyAsync(bb + nh + np, a.topk, nk, cudaMemcpyDeviceToHost, a.st));
                CK(cudaMemcpyAsync(bb + nh + np + nk, a.cand, nc, cudaMemcpyDeviceToHost, a.st));
                CK(cudaEventRecord(a.ev_out, a.st));
                cudaSetDevice(b.id);
                CK(cudaStreamWaitEvent(b.st, a.ev_out, 0));
                CK(cudaMemcpyAsync(b.h, bb, nh, cudaMemcpyHostToDevice, b.st));
                CK(cudaMemcpyAsync(b.pre, bb + nh, np, cudaMemcpyHostToDevice, b.st));
                CK(cudaMemcpyAsync(b.topk, bb + nh + np, nk, cudaMemcpyHostToDevice, b.st));
                CK(cudaMemcpyAsync(b.cand, bb + nh + np + nk, nc, cudaMemcpyHostToDevice, b.st));
                cur = l.dev;
            }
            if (!tap(devs[(size_t) l.dev], il, T, err)) return false;
            if (!layer(l, 0, p0, T, wplan, err)) return false;
            if (dump_h != nullptr) {   // DS4_DUMP_H: every layer's residual stream, for tools/ds4/compare.py --layers
                Dev& dd = devs[(size_t) l.dev];
                std::vector<float> hv((size_t) T * H * D);
                cudaSetDevice(dd.id);
                CK(cudaMemcpyAsync(hv.data(), dd.h, hv.size() * 4, cudaMemcpyDeviceToHost, dd.st));
                CK(cudaStreamSynchronize(dd.st));
                for (int t = 0; t < T; ++t)
                    std::memcpy(dump.data() + ((size_t) t * g.nl + l.il) * H * D, hv.data() + (size_t) t * H * D,
                                (size_t) H * D * 4);
            }
        }
        if (dump_h != nullptr) std::fwrite(dump.data(), 4, dump.size(), dump_h);
        if (cur != head_dev) {
            err = "the head is not on the last layer's GPU";
            return false;
        }
        return head_rows_dev(head_dev, T, err);
    }

    // threads: each device's thread enqueues its layers (a group's members each their share) at once, the
    // devices waiting on each other through host-mapped flags (the window's seq)
    bool forward_par(int64_t p0, int T, std::string& err) {
        wplan.win = ++win_seq;
        std::vector<std::string> errs(devs.size());
        for (size_t di = 0; di < devs.size(); ++di)
            workers[di]->submit([this, di, p0, T, &errs] {
                std::string e;
                if (!run_dev((int) di, p0, T, wplan, e)) errs[di] = e.empty() ? "a device's thread failed" : e;
            });
        for (auto& w : workers) w->wait();
        for (const std::string& e : errs)
            if (!e.empty()) {
                err = e;
                return false;
            }
        return true;
    }

    // the hand-off between groups: h | pre | topk | cand of device d <-> the host-mapped buffer
    Xfer hand_xfer(uint8_t* host, Dev& d, int T, bool put) const {
        const size_t nh = (size_t) T * g.hc * g.dim * 4, np = (size_t) T * 8 * 4, nk = (size_t) T * topk_stride() * 4,
                     nc = (size_t) T * cand_stride() * 4;
        void* dv[4] = {d.h, d.pre, d.topk, d.cand};
        uint8_t* hv[4] = {host, host + nh, host + nh + np, host + nh + np + nk};
        const size_t nb[4] = {nh, np, nk, nc};
        Xfer x;
        x.n = 4;
        for (int i = 0; i < 4; ++i) {
            x.src[i] = put ? (const void*) dv[i] : (const void*) hv[i];
            x.dst[i] = put ? (void*) hv[i] : dv[i];
            x.bytes[i] = nb[i];
        }
        return x;
    }

    bool run_dev(int di, int64_t p0, int T, const WinPlan& wp, std::string& err) {
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int D = g.dim, H = g.hc;
        bool started = false;
        std::vector<char> got(srcs.size(), 0);
        for (int il = 0; il < g.nl; ++il) {
            Layer& L0 = L[(size_t) il];
            int mi = -1;
            for (int m = 0; m < L0.nm; ++m)
                if (member(L0, m).dev == di) mi = m;
            if (mi < 0) continue;
            if (!started) {   // the window's input: the embedding rows, or the previous group's hand-off
                started = true;
                if (il == 0) {
                    CK(cudaMemcpyAsync(d.emb_rows, h_emb, (size_t) T * D * 2, cudaMemcpyHostToDevice, s));
                    embed(d.emb_rows, T, d.h, D, H, s);
                    init_pre(d.pre, T, s);
                } else {
                    const Dev& a = devs[(size_t) L[(size_t) il - 1].dev];
                    xfer_get(hand_xfer(a.hand, d, T, false), a.hand_flag, wp.win, s);
                }
            }
            // the compressed KV / index keys another group's source layer wrote this window
            for (int sidx : {L0.src, L0.ix_src ? L0.ik_src : -1}) {
                if (sidx < 0 || got[(size_t) sidx]) continue;
                got[(size_t) sidx] = 1;
                Src& sr = srcs[(size_t) sidx];
                if (L[(size_t) sr.layer].grp == L0.grp) continue;
                const int64_t g_first = p0 / sr.ratio;
                const int ng = (int) std::max<int64_t>(0, (p0 + T) / sr.ratio - g_first);
                if (ng == 0) continue;
                Xfer x;
                if (sr.ckv[(size_t) di] != nullptr) {
                    x.src[x.n] = sr.x_stage;
                    x.dst[x.n] = sr.ckv[(size_t) di] + (size_t) g_first * ckv_rb();
                    x.bytes[x.n++] = (size_t) ng * ckv_rb();
                }
                if (sr.ik[(size_t) di] != nullptr) {
                    x.src[x.n] = sr.x_stage + x_cap() * ckv_rb();
                    x.dst[x.n] = sr.ik[(size_t) di] + (size_t) g_first * ik_rb();
                    x.bytes[x.n++] = (size_t) ng * ik_rb();
                }
                if (x.n > 0) xfer_get(x, sr.x_flag, wp.win, s);
            }
            if (mi == 0 && !tap(d, il, T, err)) return false;
            if (!layer(L0, mi, p0, T, wp, err)) return false;
            if (dump_l != nullptr && mi == 0) dump_layer(d, d.h, il, p0, T);
            const bool last = il + 1 == g.nl || L[(size_t) il + 1].grp != L0.grp;
            if (last && mi == 0 && il + 1 < g.nl) xfer_put(hand_xfer(d.hand, d, T, true), d.hand_flag, wp.win, d.xcnt, s);
        }
        if (d.grp == devs[(size_t) head_dev].grp) return head_rows_dev(di, T, err);
        return true;
    }

    // a source's new cache rows (src on device `from`, bytes of them) to the same place of its other devices' copies,
    // through pinned memory: the copy out on the source's stream, each copy in on its device's stream after it
    bool replicate(std::vector<uint8_t*>& copies, uint8_t* stage, cudaEvent_t ev, const uint8_t* src, size_t at,
                   size_t bytes, int from, std::string& err) {
        bool any = false;
        for (size_t od = 0; od < devs.size(); ++od) any = any || (copies[od] != nullptr && (int) od != from);
        if (!any) return true;
        Dev& d = devs[(size_t) from];
        CK(cudaMemcpyAsync(stage, src, bytes, cudaMemcpyDeviceToHost, d.st));
        CK(cudaEventRecord(ev, d.st));
        for (size_t od = 0; od < devs.size(); ++od) {
            if (copies[od] == nullptr || (int) od == from) continue;
            Dev& o2 = devs[od];
            cudaSetDevice(o2.id);
            CK(cudaStreamWaitEvent(o2.st, ev, 0));
            CK(cudaMemcpyAsync(copies[od] + at, stage, bytes, cudaMemcpyHostToDevice, o2.st));
        }
        cudaSetDevice(d.id);
        return true;
    }

    // layer L0 for the window, as group member mi (its share of the heads, of the output projection and of the FFNs;
    // the partial sums all-reduced across the group)
    bool layer(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, std::string& err) {
        Layer& l = member(L0, mi);
        Dev& d = devs[(size_t) l.dev];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int nm = L0.nm;
        const int D = g.dim, H = g.hc, HD = g.hd, NHL = l.ahn, HQ = NHL * HD, OG = l.ogn, FF = l.ffn;
        const ArGroup& ag = groups[(size_t) L0.grp].ar;
        const float* freqs = l.ratio ? d.freqs_comp : d.freqs_win;
        prof_mark(d, 0);
        // Engram
        if (l.eg >= 0) {
            const int cols = (pk.eg_ngram - 1) * pk.eg_heads, ehd = pk.eg_hd;
            const size_t per = (size_t) cols * (ehd + ehd / 32);
            eg_finish(*eg_win[(size_t) l.eg]);   // (asked for when the window began)
            uint8_t* src = h_eg + (size_t) l.eg * eg_cap * per;
            for (int t = 0; t < T; ++t)   // [token][rows | scales] -> [all rows][all scales]
                CK(cudaMemcpyAsync(d.eg_rows + (size_t) t * cols * ehd, src + (size_t) t * per, (size_t) cols * ehd,
                                   cudaMemcpyHostToDevice, s));
            uint8_t* sc = d.eg_rows + (size_t) T * cols * ehd;
            for (int t = 0; t < T; ++t)
                CK(cudaMemcpyAsync(sc + (size_t) t * cols * (ehd / 32), src + (size_t) t * per + (size_t) cols * ehd,
                                   (size_t) cols * (ehd / 32), cudaMemcpyHostToDevice, s));
            engram_dequant(d.eg_rows, sc, T * cols, ehd, d.eg_x, s, (int) eg_fmt[(size_t) l.eg]);
            if (!mv(d, {job(l.eg_wkv, d.eg_x, d.eg_kv, T, (size_t) cols * ehd, (size_t) (H + 1) * D)}, err)) return false;
            engram_gate(d.h, d.eg_kv, l.eg_w, H, D, T, g.eps, s);
        }
        // attention
        if (decode_fusion && d.hccnt != nullptr && H > 0 && (2 + H) * H <= 32 &&
            T > 0 && T <= kWin && (H * D) % 4 == 0) {
            hc_mix_split(d.h, l.hca_fn, H * D, (2 + H) * H, T, g.eps, d.mix, d.hccnt,
                         l.hca_scale, l.hca_base, H, g.sinkhorn, g.hc_eps, d.a_pre, d.a_post, d.a_comb, s);
        } else {
            hc_mix(d.h, l.hca_fn, H * D, (2 + H) * H, T, g.eps, d.mix, s);
            hc_split(d.mix, l.hca_scale, l.hca_base, H, g.sinkhorn, g.hc_eps, T, d.a_pre, d.a_post, d.a_comb, s);
        }
        hc_pre_norm(d.h, d.pre, l.attn_norm, D, H, T, g.eps, d.x, s);
        {
            x3::MvJob j[4];
            int nj = 0;
            j[nj++] = job(l.wq_a, d.x, d.qr, T, D, g.q_lora);
            j[nj++] = job(l.wkv, d.x, d.kvr, T, D, HD);
            if (l.kv_src) j[nj++] = job(l.c_wkv, d.x, d.ckr, T, D, HD);
            if (l.kv_src && l.ratio > 1) j[nj++] = job(l.c_wgate, d.x, d.cgr, T, D, HD);
            if (!mv_n(d, j, nj, err)) return false;
        }
        Finish fq;
        fq.w = l.q_norm;
        fq.eps = g.eps;
        finish(d.qr, T, g.q_lora, g.q_lora, fq, s);
        if (NHL > 0) {
            if (!mv(d, {job(l.wq_b, d.qr, d.q, T, g.q_lora, HQ)}, err)) return false;
            Finish fr;
            fr.rd = g.rd;
            fr.pos = (int) p0;
            fr.vpp = NHL;
            fr.freqs = freqs;
            finish(d.q, T * NHL, HD, HD, fr, s);
        }
        // the window rows: the ring's (positions p0 - win ..) then this window's, each query its own 128.  A DSpark
        // block's ring holds the trunk's main KV (positions <= p0 - 1, seeded); its block of drafts attends to all of
        // that ring and to the whole block, and leaves the ring as it was
        ring_to_ext(l.win, d.ext, l.mtp ? (int64_t) g.win : p0, g.win, HD, s);   // (a block's ring: its rows as they lie)
        Finish fk;
        fk.w = l.kv_norm;
        fk.eps = g.eps;
        fk.rd = g.rd;
        fk.pos = (int) p0;
        fk.freqs = freqs;
        fk.quant = 1;
        fk.dst = d.ext + (size_t) g.win * HD;
        fk.dst_stride = HD;
        finish(d.kvr, T, HD, HD, fk, s);
        if (!l.mtp) ext_to_ring(d.ext, l.win, p0, g.win, T, HD, s, spec_win ? l.ring_bak : nullptr);
        AttnRows ar;
        for (int t = 0; t < T; ++t) {
            if (l.mtp) {
                ar.win0[t] = 0;
                ar.nwin[t] = (int) std::min<int64_t>(g.win, p0);
                ar.nsel[t] = T;
                continue;
            }
            const int first = (int) std::max<int64_t>(g.win - p0, t + 1);   // ext row of position max(0, p - win + 1)
            ar.win0[t] = first;
            ar.nwin[t] = g.win + t - first + 1;
            ar.nsel[t] = 0;
        }
        const void* ckv = nullptr;
        int ckv_fmt = kKvFp4;
        if (l.ratio > 0) {
            Src& sr = srcs[(size_t) l.src];
            const int R = l.ratio;
            // the groups this window completes: g_first .. g_first + ng - 1
            const int64_t g_first = p0 / R;              // the first group whose last position is >= p0
            const int64_t g_end = (p0 + T) / R;          // groups completed by p0 + T - 1
            const int ng = (int) std::max<int64_t>(0, g_end - g_first);
            if (l.kv_src) {
                if (R == 1) {
                    Finish fc;
                    fc.w = l.c_norm;
                    fc.eps = g.eps;
                    finish(d.ckr, T, HD, HD, fc, s);
                    CK(cudaMemcpyAsync(d.latent, d.ckr, (size_t) T * HD * 4, cudaMemcpyDeviceToDevice, s));
                } else {
                    if (spec_win && l.c_snap != nullptr) {   // to replay the accepted rows if drafts are rejected
                        const size_t ns = (size_t) R * HD * 4, nr = (size_t) T * HD * 4;
                        CK(cudaMemcpyAsync(l.c_snap, l.c_kv, ns, cudaMemcpyDeviceToDevice, s));
                        CK(cudaMemcpyAsync(l.c_snap + (size_t) R * HD, l.c_score, ns, cudaMemcpyDeviceToDevice, s));
                        CK(cudaMemcpyAsync(l.c_rows, d.ckr, nr, cudaMemcpyDeviceToDevice, s));
                        CK(cudaMemcpyAsync(l.c_rows + (size_t) kWin * HD, d.cgr, nr, cudaMemcpyDeviceToDevice, s));
                    }
                    compress_win(d.ckr, d.cgr, l.c_kv, l.c_score, R, (int) p0, T, HD, l.c_norm, g.eps, d.latent, s);
                }
            }
            Rows lim;
            int clen_max = 0;
            for (int t = 0; t < T; ++t) {
                lim.n[t] = (int) ((p0 + t + 1) / R);
                lim.k[t] = std::min(g.ix_topk, lim.n[t]);
                clen_max = std::max(clen_max, lim.n[t]);
            }
            if (l.ix_src) {
                Src& ks = srcs[(size_t) l.ik_src];
                if (l.owns_k && ng > 0) {
                    if (!mv(d, {job(l.i_wk, d.latent, d.ikr, ng, HD, g.ix_hd)}, err)) return false;
                    Finish fi;
                    fi.w = l.ik_norm;
                    fi.eps = g.eps;
                    fi.rd = g.rd;
                    fi.pos = (int) (g_first * R);
                    fi.pos_step = R;
                    fi.freqs = freqs;
                    fi.quant = 3;
                    fi.pdst = ks.ik[(size_t) l.dev] + (size_t) g_first * ik_rb();
                    fi.pdst_stride = (long long) ik_rb();
                    finish(d.ikr, ng, g.ix_hd, g.ix_hd, fi, s);
                    if (!wp.par && !replicate(ks.ik, ks.h_ik, ks.ev_ik, fi.pdst, (size_t) g_first * ik_rb(),
                                              (size_t) ng * ik_rb(), l.dev, err))
                        return false;
                }
                if (clen_max > 0) {
                    if (!mv(d, {job(l.i_wq_b, d.qr, d.iq, T, g.q_lora, (size_t) g.ix_heads * g.ix_hd)}, err)) return false;
                    Finish fi;
                    fi.rd = g.rd;
                    fi.pos = (int) p0;
                    fi.vpp = g.ix_heads;
                    fi.freqs = freqs;
                    fi.quant = 3;
                    finish(d.iq, T * g.ix_heads, g.ix_hd, g.ix_hd, fi, s);
                    // (iw rows are 64 apart)
                    gemv_f16_rows(l.i_wproj, d.x, (size_t) D, g.ix_heads, D, T,
                                  1.0f / std::sqrt((float) g.ix_hd) / std::sqrt((float) g.ix_heads), d.iw, 64, s);
                    const long long ss = (long long) o.max_context + 64;
                    bool restrict_ = false;
                    for (int t = 0; t < T; ++t)
                        restrict_ = restrict_ || (lim.n[t] + g.cand_blk - 1) / g.cand_blk > g.cand_topk;
                    if (l.uses_cand && restrict_) {
                        Rows cl;   // the candidate positions of each query: its blocks x blk
                        int nmax = 0;
                        for (int t = 0; t < T; ++t) {
                            cl.n[t] = lim.n[t];
                            cl.k[t] = wp.cand[l.il][t];
                            nmax = std::max(nmax, wp.cand[l.il][t] * g.cand_blk);
                        }
                        index_score(d.iq, d.iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, T, nmax, d.cand,
                                    (int) cand_stride(), g.cand_blk, cl, d.scores, ss, s);
                        Rows tr;
                        for (int t = 0; t < T; ++t) {
                            tr.n[t] = wp.cand[l.il][t] * g.cand_blk;
                            tr.k[t] = lim.k[t];
                        }
                        topk(d.scores, ss, T, tr, d.cand, (int) cand_stride(), g.cand_blk, d.topk, (int) topk_stride(), s, d.topk_scr.states ? &d.topk_scr : nullptr);
                    } else {
                        index_score(d.iq, d.iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, T, clen_max, nullptr, 0, 1,
                                    lim, d.scores, ss, s);
                        if (l.cand_src) {
                            Rows cr;
                            for (int t = 0; t < T; ++t) {
                                cr.n[t] = lim.n[t];
                                cr.k[t] = g.cand_topk;
                            }
                            cand_blocks(d.scores, ss, T, cr, g.cand_blk, d.bm, (long long) bm_stride(), d.cand,
                                        (int) cand_stride(), s, d.topk_scr.states ? &d.topk_scr : nullptr);
                        }
                        topk(d.scores, ss, T, lim, nullptr, 0, 1, d.topk, (int) topk_stride(), s, d.topk_scr.states ? &d.topk_scr : nullptr);
                    }
                }
            }
            if (l.kv_src && ng > 0) {
                Finish fc;
                fc.rd = g.rd;
                fc.pos = (int) (g_first * R);
                fc.pos_step = R;
                fc.freqs = freqs;
                fc.quant = 2;
                fc.pdst = sr.ckv[(size_t) l.dev] + (size_t) g_first * ckv_rb();
                fc.pdst_stride = (long long) ckv_rb();
                finish(d.latent, ng, HD, HD, fc, s);
                if (!wp.par && !replicate(sr.ckv, sr.h_ckv, sr.ev_ckv, fc.pdst, (size_t) g_first * ckv_rb(),
                                          (size_t) ng * ckv_rb(), l.dev, err))
                    return false;
                if (wp.par && mi == 0 && sr.remote) {   // the new rows for the other groups' readers
                    Xfer x;
                    x.src[0] = fc.pdst;
                    x.dst[0] = sr.x_stage;
                    x.bytes[0] = (size_t) ng * ckv_rb();
                    x.n = 1;
                    if (l.owns_k && srcs[(size_t) l.ik_src].ik[(size_t) l.dev] != nullptr) {
                        x.src[1] = srcs[(size_t) l.ik_src].ik[(size_t) l.dev] + (size_t) g_first * ik_rb();
                        x.dst[1] = sr.x_stage + x_cap() * ckv_rb();
                        x.bytes[1] = (size_t) ng * ik_rb();
                        x.n = 2;
                    }
                    xfer_put(x, sr.x_flag, wp.win, d.xcnt, s);
                }
            }
            for (int t = 0; t < T; ++t) ar.nsel[t] = lim.n[t] > 0 ? wp.topk[l.il][t] : 0;
            ckv = sr.ckv[(size_t) l.dev];
        }
        if (l.mtp) {   // the block's rows (BF16), after the ring's in ext, as every query's selection
            ckv = d.ext + (size_t) g.win * HD;
            ckv_fmt = kKvBf16;
            iota_rows(d.topk, T, T, (int) topk_stride(), s);
        }
        if (NHL > 0) {
            sparse_attn(d.q, d.ext, ckv, d.topk, (int) topk_stride(), ar, l.sink, NHL, HD, T,
                        1.0f / std::sqrt((float) HD), d.o, s, ckv_fmt, 0, &d.attn_scr);
            Finish fo;
            fo.rd = g.rd;
            fo.pos = (int) p0;
            fo.vpp = NHL;
            fo.freqs = freqs;
            fo.inverse = 1;
            fo.quant = 1;
            finish(d.o, T * NHL, HD, HD, fo, s);
            x3::MvJob j[16];
            const int per = HQ / OG;
            for (int gi = 0; gi < OG; ++gi)
                j[gi] = job(l.wo_a[gi], d.o + (size_t) gi * per, d.oa + (size_t) gi * g.o_lora, T, HQ, (size_t) OG * g.o_lora);
            if (!mv_n(d, j, OG, err)) return false;
            if (!mv(d, {job(l.wo_b, d.oa, d.ao, T, (size_t) OG * g.o_lora, D)}, err)) return false;
        } else CK(cudaMemsetAsync(d.ao, 0, (size_t) T * D * sizeof(*d.ao), s));   // (no heads: nothing to add)
        if (nm > 1) allreduce(d.ao, T * D, ag, mi, ++d.ar_seq, d.xcnt, s);
        hc_post(d.ao, d.h, d.a_post, d.a_comb, D, H, T, d.h2, s);
        std::swap(d.h, d.h2);
        if (dump_l != nullptr && mi == 0) dump_layer(d, d.h, 1000 + l.il, p0, T);   // (DS4_DUMP_L: after the attention)
        prof_mark(d, 1);
        // the MoE
        if (decode_fusion && d.hccnt != nullptr && H > 0 && (2 + H) * H <= 32 &&
            T > 0 && T <= kWin && (H * D) % 4 == 0) {
            hc_mix_split(d.h, l.hcf_fn, H * D, (2 + H) * H, T, g.eps, d.mix, d.hccnt,
                         l.hcf_scale, l.hcf_base, H, g.sinkhorn, g.hc_eps, d.f_pre, d.f_post, d.f_comb, s);
        } else {
            hc_mix(d.h, l.hcf_fn, H * D, (2 + H) * H, T, g.eps, d.mix, s);
            hc_split(d.mix, l.hcf_scale, l.hcf_base, H, g.sinkhorn, g.hc_eps, T, d.f_pre, d.f_post, d.f_comb, s);
        }
        hc_pre_norm(d.h, d.a_pre, l.ffn_norm, D, H, T, g.eps, d.x, s);
        if (wp.async) {
            // the routing goes to the service thread through the mailbox (a group's lead publishes it, the members
            // make the same plan) while the device runs the shared expert and the VRAM experts (from the plan the
            // router made), and waits for the answer only before the experts' sum
            gemv_f32(l.gate_w, d.x, l.n_exp, D, T, d.glog, 1024, s);
            const unsigned sq = wp.seq[l.il];
            const int slot = (int) (sq % kRing);
            const bool lead = mi == 0;
            int na[2] = {0, 0};   // lookahead: the next layers' guesses (their gate on this FFN input), published below
            for (int a = 1; lead && a <= std::min(2, ahead_mode); ++a) {
                const Layer* nx = ahead_layer(L0, a);
                if (nx == nullptr) break;
                const int m = a == 1 ? std::max(1, std::min(kAhead1, nx->k_exp + ahead_extra)) : std::min(kPicks, nx->k_exp);
                float* gl = d.glog_a + (size_t) (a - 1) * kWin * 1024;
                gemv_f32(nx->gate_w, d.x, nx->n_exp, D, T, gl, 1024, s);   // (normed with the next layer's FFN
                                                                         // weights instead: the same hit rate)
                route_ahead(gl, nx->gate_bias, nx->n_exp, m, T, reqs + slot, (a - 1) * kAhead1, s);
                na[a - 1] = m;
            }
            route_publish(d.glog, l.gate_bias, l.n_exp, l.k_exp, T, g.route_scale, l.vram_tab, d.x, D, d.ridx, d.rw,
                          d.plan_ptr, lead ? reqs + slot : nullptr, sq, s, l.gate_bias_vl, l.mtp ? 0u : wp.vl, na[0],
                          na[1]);
            if (!mv(d, {job(l.s1, d.x, d.sg, T, D, FF), job(l.s3, d.x, d.su, T, D, FF)}, err)) return false;
            swiglu(d.sg, d.su, T * FF, g.lim, d.sh, s);
            if (!mv(d, {job(l.s2, d.sh, d.sh_out, T, FF, D)}, err)) return false;
            prof_mark(d, 2);
            const x3::ExpertLayout& lay = nm > 1 ? l.hlay : l.lay;
            if (T == 1 && moe_token_path) {
                x3::MoeArgs ma;
                ma.plan_ptr = d.plan_ptr;
                ma.plan_w = d.rw;
                ma.cpu_part = lead ? resps[slot].part[0] : nullptr;
                ma.cpu_flag = lead ? &resps[slot].cpu[0] : nullptr;
                ma.lay = lay;
                ma.k = l.k_exp;
                ma.n_embd = D;
                ma.n_ff = FF;
                ma.limit = g.lim;
                const bool mx = lay.fmt == gg::kFmtMXFP4;
                if (zc_stage_buf != nullptr && l.zc_tab != nullptr)   // (the store's experts copied to VRAM at the PCIe rate first)
                    zc_stage(d.plan_ptr, d.ridx, l.zc_tab, l.k_exp, zc_stage_buf, zc_stride, l.blob, s);
                if (mx) gg::moe_gate_up(ma, d.x, d.xg, s);
                else x3::moe_gate_up(ma, d.x, d.xg, s);
                prof_mark(d, 3);
                if (lead) wait_seq(&resps[slot].seq, sq, s);
                prof_mark(d, 4);
                if (mx) gg::moe_down(ma, d.xg, d.sh_out, d.moe_out, s);
                else x3::moe_down(ma, d.xg, d.sh_out, d.moe_out, s);
            } else {   // each distinct expert read once for the window's tokens
                gf::MoeWin mw;
                mw.plan_ptr = d.plan_ptr;
                mw.T = T;
                mw.k = l.k_exp;
                mw.grp_ptr = d.grp_ptr;
                mw.grp_start = d.grp_start;
                mw.grp_ent = d.grp_ent;
                mw.n_groups = d.n_groups;
                gf::moe_win_group(mw, s);
                x3::WinArgs wa;
                wa.grp_ptr = d.grp_ptr;
                wa.grp_start = d.grp_start;
                wa.grp_ent = d.grp_ent;
                wa.n_groups = d.n_groups;
                wa.plan_ptr = d.plan_ptr;
                wa.plan_w = d.rw;
                wa.cpu_part = lead ? &resps[slot].part[0][0] : nullptr;
                wa.cpu_stride = kMaxDim;
                wa.cpu_flag = lead ? resps[slot].cpu : nullptr;
                wa.wait_ptr = lead ? &resps[slot].seq : nullptr;
                wa.wait_val = sq;
                wa.x = d.x;
                wa.x_stride = (size_t) D;
                wa.h = d.xg;
                wa.part = d.ed;
                wa.sh_out = d.sh_out;
                wa.out = d.moe_out;
                wa.out_stride = (size_t) D;
                wa.lay = lay;
                wa.T = T;
                wa.k = l.k_exp;
                wa.n_embd = D;
                wa.n_ff = FF;
                wa.limit = g.lim;
                if (!(lay.fmt == gg::kFmtMXFP4 ? gg::win_experts(wa, s) : x3::win_experts(wa, s))) {
                    err = "exl3 win_experts failed";
                    return false;
                }
            }
            if (lead) st.experts += (int64_t) T * l.k_exp;
            if (nm > 1) allreduce(d.moe_out, T * D, ag, mi, ++d.ar_seq, d.xcnt, s);
            hc_post(d.moe_out, d.h, d.f_post, d.f_comb, D, H, T, d.h2, s);
            std::swap(d.h, d.h2);
            CK(cudaMemcpyAsync(d.pre, d.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
            prof_mark(d, 5);
            return true;
        }
        if (nm > 1) {
            err = "a tensor-parallel layer on the host-routed path";
            return false;
        }
        // (glog rows are 1024 apart: route reads them so)
        gemv_f32(l.gate_w, d.x, l.n_exp, D, T, d.glog, 1024, s);
        VlBits vb = {};
        vb.w[0] = l.mtp ? 0u : wp.vl;
        route(d.glog, l.gate_bias, l.n_exp, l.k_exp, T, g.route_scale, d.ridx, d.rw, s, l.gate_bias_vl,
              vb.w[0] != 0 ? &vb : nullptr);
        CK(cudaMemcpyAsync(h_idx, d.ridx, (size_t) T * kPicks * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(h_w, d.rw, (size_t) T * kPicks * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(h_x, d.x, (size_t) T * D * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaEventRecord(d.ev, s));
        // the shared expert while the host looks at the picks
        if (!mv(d, {job(l.s1, d.x, d.sg, T, D, g.ff), job(l.s3, d.x, d.su, T, D, g.ff)}, err)) return false;
        swiglu(d.sg, d.su, T * g.ff, g.lim, d.sh, s);
        if (!mv(d, {job(l.s2, d.sh, d.sh_out, T, g.ff, D)}, err)) return false;
        const double tw = now_ms();
        CK(cudaEventSynchronize(d.ev));
        st.gpu_wait_ms += now_ms() - tw;
        // No mailbox was reserved for this window. The completed route event
        // precedes plan reuse; prepare queued d.st's stage-ready wait before use.
        bool gpu_misses_used=false;
        if(!gpu_misses_prepare(l,d,T,h_idx,gpu_misses_used,err))return false;
        if(gpu_misses_used) {
            uint32_t* p=prof.data()+(size_t)l.il*g.n_exp;
            for(int t=0;t<T;++t)for(int i=0;i<l.k_exp;++i) {
                const int e=h_idx[t*kPicks+i];++p[e];l.heat[(size_t)e]+=1.0f;
                if(l.slot[(size_t)e]<0)l.cpu_hits.push_back(e);
            }
            const int before_errors=gg::launch_errors();
            if(T==1) {
                x3::MoeArgs ma;
                ma.plan_ptr=d.plan_ptr;ma.plan_w=d.rw;ma.cpu_part=nullptr;ma.cpu_flag=nullptr;
                ma.lay=l.lay;ma.k=l.k_exp;ma.n_embd=D;ma.n_ff=FF;ma.limit=g.lim;
                gg::moe_gate_up(ma,d.x,d.xg,s);
                gg::moe_down(ma,d.xg,d.sh_out,d.moe_out,s);
            } else {
                gf::MoeWin mw;
                mw.plan_ptr=d.plan_ptr;mw.T=T;mw.k=l.k_exp;mw.grp_ptr=d.grp_ptr;
                mw.grp_start=d.grp_start;mw.grp_ent=d.grp_ent;mw.n_groups=d.n_groups;
                gf::moe_win_group(mw,s);
                x3::WinArgs wa;
                wa.grp_ptr=d.grp_ptr;wa.grp_start=d.grp_start;wa.grp_ent=d.grp_ent;wa.n_groups=d.n_groups;
                wa.plan_ptr=d.plan_ptr;wa.plan_w=d.rw;wa.cpu_part=nullptr;wa.cpu_flag=nullptr;wa.wait_ptr=nullptr;
                wa.x=d.x;wa.x_stride=(size_t)D;wa.h=d.xg;wa.part=d.ed;wa.sh_out=d.sh_out;
                wa.out=d.moe_out;wa.out_stride=(size_t)D;wa.lay=l.lay;wa.T=T;wa.k=l.k_exp;
                wa.n_embd=D;wa.n_ff=FF;wa.limit=g.lim;
                if(!gg::win_experts(wa,s))return cache_epoch_terminal("GPU miss window kernels refused",err);
            }
            if(gg::launch_errors()!=before_errors||cudaPeekAtLastError()!=cudaSuccess)
                return cache_epoch_terminal("GPU miss expert kernel launch failed",err);
            ++st.gpu_miss_kernel_windows;st.experts+=(int64_t)T*l.k_exp;
            hc_post(d.moe_out,d.h,d.f_post,d.f_comb,D,H,T,d.h2,s);
            std::swap(d.h,d.h2);
            if(cudaMemcpyAsync(d.pre,d.f_pre,(size_t)T*8*4,cudaMemcpyDeviceToDevice,s)!=cudaSuccess)
                return cache_epoch_terminal("GPU miss post-mix state copy failed",err);
            return true;
        }

        // the picks by expert: the VRAM ones' rows (gathered inputs, one GEMV per matrix for all their tokens), the
        // CPU lane's entries
        Combine cm;
        Gather ga;
        uint32_t* pr = prof.data() + (size_t) l.il * g.n_exp;
        int vex[kWin * kPicks], vrow0[kWin * kPicks], vn[kWin * kPicks];
        int nvex = 0, nrows = 0;
        std::vector<Ent> ents;
        std::vector<std::pair<int, float>> cpu_rows;   // slot -> (token, weight)
        std::vector<std::vector<std::pair<int, int>>> per((size_t) l.n_exp);   // expert -> (token, pick)
        for (int t = 0; t < T; ++t) {
            cm.n[t] = 0;
            cm.cpu[t] = 0;
            for (int i = 0; i < l.k_exp; ++i) {
                const int e = h_idx[t * kPicks + i];
                if (e < 0 || e >= l.n_exp) {
                    err = "router picked expert " + std::to_string(e);
                    return false;
                }
                ++pr[e];
                l.heat[(size_t) e] += 1.0f;
                if (l.slot[(size_t) e] < 0) l.cpu_hits.push_back(e);
                per[(size_t) e].push_back({t, i});
            }
        }
        int nslots = 0;
        for (int e = 0; e < l.n_exp; ++e) {
            const auto& pe = per[(size_t) e];
            if (pe.empty()) continue;
            if (l.slot[(size_t) e] >= 0) {
                vex[nvex] = e;
                vrow0[nvex] = nrows;
                vn[nvex] = (int) pe.size();
                for (const auto& tp : pe) {
                    ga.src[nrows] = tp.first;
                    const int t = tp.first;
                    cm.row[t][cm.n[t]] = nrows;
                    cm.w[t][cm.n[t]] = h_w[t * kPicks + tp.second];
                    ++cm.n[t];
                    ++nrows;
                }
                ++nvex;
            } else {
                for (size_t a = 0; a < pe.size(); a += (size_t) ent_cap(l)) {
                    Ent en;
                    en.e = e;
                    en.slot0 = nslots;
                    en.n = (int) std::min<size_t>((size_t) ent_cap(l), pe.size() - a);
                    for (int i = 0; i < en.n; ++i) {
                        en.tok[i] = pe[a + (size_t) i].first;
                        cpu_rows.push_back({pe[a + (size_t) i].first, h_w[pe[a + (size_t) i].first * kPicks + pe[a + (size_t) i].second]});
                        cm.cpu[en.tok[i]] = 1;
                    }
                    nslots += en.n;
                    ents.push_back(en);
                }
            }
        }
        st.experts += (int64_t) T * l.k_exp;
        st.cpu_experts += nslots;
        if (nrows > 0) {
            gather_rows(d.x, ga, nrows, D, d.xe, s);
            x3::MvJob j[2 * kWin * kPicks];
            int nj = 0;
            for (int i = 0; i < nvex; ++i) {
                const auto& vm = l.vm[(size_t) vex[i]];
                j[nj++] = job(vm[0], d.xe + (size_t) vrow0[i] * D, d.xg + (size_t) vrow0[i] * g.ff, vn[i], D, g.ff);
                j[nj++] = job(vm[1], d.xe + (size_t) vrow0[i] * D, d.xu + (size_t) vrow0[i] * g.ff, vn[i], D, g.ff);
            }
            if (!mv_n(d, j, nj, err)) return false;
            swiglu(d.xg, d.xu, nrows * g.ff, g.lim, d.xhh, s);
            nj = 0;
            for (int i = 0; i < nvex; ++i)
                j[nj++] = job(l.vm[(size_t) vex[i]][2], d.xhh + (size_t) vrow0[i] * g.ff, d.ed + (size_t) vrow0[i] * D,
                              vn[i], g.ff, D);
            if (!mv_n(d, j, nj, err)) return false;
        }
        if (!ents.empty()) {
            const double tc = now_ms();
            cpu_experts(l, h_x, ents);
            std::fill(h_part, h_part + (size_t) T * D, 0.0f);
            for (int sl = 0; sl < nslots; ++sl) {
                float* o = h_part + (size_t) cpu_rows[(size_t) sl].first * D;
                const float wv = cpu_rows[(size_t) sl].second;
                const float* y = c_yd.data() + (size_t) sl * D;
                for (int e = 0; e < D; ++e) o[e] += wv * y[e];
            }
            st.cpu_ms += now_ms() - tc;
            CK(cudaMemcpyAsync(d.cpu_part, h_part, (size_t) T * D * 4, cudaMemcpyHostToDevice, s));
        }
        combine(d.ed, ents.empty() ? nullptr : d.cpu_part, d.sh_out, cm, T, D, d.moe_out, s);
        hc_post(d.moe_out, d.h, d.f_post, d.f_comb, D, H, T, d.h2, s);
        std::swap(d.h, d.h2);
        CK(cudaMemcpyAsync(d.pre, d.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
        // (h_part / h_x are written again only after the next layer's route has synchronized, which follows these)
        return true;
    }

    int sample(float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter) {
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        strata::dsv41k::SamplerParams sp;
        sp.temperature = temperature;
        sp.greedy = temperature <= 0.0f;
        sp.top_p = top_p;
        sp.top_k = top_k;
        sp.min_p = min_p;
        sp.seed = seed;
        sp.counter = counter;
        strata::dsv41k::sample_tokens(d.logits + (size_t) std::max(0, n_logits - 1) * g.vocab, 1, g.vocab, nullptr, 0,
                                       sp, d.tok, d.st);
        int t = -1;
        cudaMemcpyAsync(&t, d.tok, 4, cudaMemcpyDeviceToHost, d.st);
        cudaStreamSynchronize(d.st);
        return t;
    }

    bool logits(std::vector<float>& out) {
        if (logits_all) {
            out = chunk_logits;
            return true;
        }
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        out.resize((size_t) std::max(1, n_logits) * g.vocab);
        return cudaMemcpy(out.data(), d.logits, out.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    void save_profile() const {
        if (o.profile.empty()) return;
        const std::string tmp = o.profile + ".tmp";
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        if (f == nullptr) return;
        const bool ok = std::fwrite(prof.data(), 4, prof.size(), f) == prof.size();
        std::fclose(f);
        // (the old profile replaced: std::rename does not replace an existing file on Windows, so there the profile
        // never moved on and only .tmp files were left)
        if (!ok || !os::replace_file(tmp, o.profile)) std::remove(tmp.c_str());
    }

    std::vector<std::string> info_lines() const {
        std::vector<std::string> v;
        int nh = 0;
        for (const Layer& l : L) nh += l.n_exp - l.n_vram;
        char b[512];
        // kv=fp4: DeepSeek's own cache layout (the compressed KV and the index keys FP4, the window FP8)
        std::snprintf(b, sizeof b, "INFO context=%lld kv=fp4 kv_resident=%lld expert_slots=%d engine_kind=ds4 "
                                   "cpu_experts=%d ram_experts=%d engine=ds4-0.1",
                      (long long) o.max_context, (long long) o.max_context,
                      prefill_phase_pending ? 0 : g.nl * g.n_exp - nh, nh, ram_held);
        v.push_back(b);
        if (commit_kept > 0) {   // (Windows, an integrated GPU: the commit kept experts off the GPU - the server says so)
            std::snprintf(b, sizeof b,
                          "NOTE experts: %.0f GB stay off the GPU - Windows' page file is too small to hold them there; "
                          "a page file of %.0f GB keeps them on the GPU (System > About > Advanced system settings > "
                          "Performance > Advanced > Virtual memory), then restart Maya", commit_kept / 1e9, pagefile_hint);
            v.push_back(b);
            std::snprintf(b, sizeof b, "INFO gpu_commit_capped_gb=%.0f pagefile_hint_gb=%.0f", std::ceil(commit_kept / 1e9),
                          pagefile_hint);
            v.push_back(b);
        }
        if (pk.eg_layers > 0) {   // the n-gram memory's (Engram's) table rows a token reads, for the Monitor
            std::snprintf(b, sizeof b, "INFO engram_layers=%d engram_lookups=%d", pk.eg_layers,
                          pk.eg_layers * (pk.eg_ngram - 1) * pk.eg_heads);
            v.push_back(b);
        }
        if (img_tok >= 0) {   // the server's image resize (DeepSeek's image_processor.py)
            std::snprintf(b, sizeof b,
                          "INFO vision=ds4 vision_patch=%d vision_merge=%d vision_min_pixels=%lld vision_max_tokens=%d "
                          "vision_max_wh_ratio=%lld image_token=%d vision_gpu=%d vision_mib=%lld",
                          vg.patch, vg.merge, (long long) pk.num("vision.min_pixels", 0), vg.max_tokens,
                          (long long) pk.num("vision.max_wh_ratio", 0), img_tok, vt.device(),
                          (long long) (vt.device_bytes() >> 20));
            v.push_back(b);
        } else if (!vision_err.empty()) {   // (the server offers no pictures and says why)
            std::string why = vision_err;
            for (char& ch : why)
                if (ch == '\n' || ch == '\r') ch = ' ';
            v.push_back("NOTE images: OFF - the model's vision tower did not load (" + why + ")");
            v.push_back("INFO vision=off");
        }
        return v;
    }
};

Model::Model() : p_(new Impl) {}
Model::~Model() = default;
bool Model::load(const Options& o, std::string& err) { return p_->load(o, err); }
bool Model::step(int token, bool logits, std::string& err) {
    if (!p_->finish_chunk_major_phase(err)) return false;
    if (!p_->finish_prefill_phase(err)) return false;
    return p_->step(token, logits, err);
}
bool Model::feed(const int* tokens, int n, int logits_rows, std::string& err) {
    Impl& p = *p_;
    p.pm.last = -1;
    p.halted = false;
    if (!p.prefill_phase_error.empty()) { err = p.prefill_phase_error; return false; }
    if (!p.chunk_major_phase_error.empty()) { err=p.chunk_major_phase_error; return false; }
    if(p.gpu_miss_direct_enabled&&!p.gpu_misses_recover_decode(n,err))return false;
    if(p.prefill_repeat_rearm&&!p.maybe_rearm_copied_prefill(n,err))return false;
    if (p.chunk_major_phase_pending && n>0) {
        if (p.chunk>0 && n>=p.chunk_min) return p.forward_chunks(tokens,n,logits_rows,err);
        if (!p.finish_chunk_major_phase(err)) return false;
    }
    if (p.prefill_phase_pending && n > 0) {
        // The explicit one-shot experiment requires the layer-major path for a multi-chunk first feed. A short
        // first feed consumes the experiment by restoring the ordinary pool BEFORE any ordinary decode/windows.
        if (p.chunk > 0 && n > p.chunk && n >= p.chunk_min) return p.forward_chunks(tokens, n, logits_rows, err);
        std::fprintf(stderr, "ds4: experimental prefill phase: first feed is not multi-chunk; restoring the normal pool\n");
        if (!p.finish_prefill_phase(err)) return false;
    }
    if (p.chunk > 0 && n >= p.chunk_min) {   // a long run - or one the flow could take, if it is the faster path here
        const double t0 = now_ms();   // (every row's logits (tests): the chunked path, which keeps them all)
        const bool ok = logits_rows > kWin || p.pick_path(n) ? p.forward_chunks(tokens, n, logits_rows, err)
                                                             : p.feed_windows(tokens, n, logits_rows, err);
        if (ok) p.path_done(now_ms() - t0);
        return ok;
    }
    return p.feed_windows(tokens, n, logits_rows, err);
}
bool Model::feed_mark(const int* tokens, int n, int logits_rows, int slot, std::string& err) {
    Impl& p = *p_;
    p.pm.last = -1;
    p.halted = false;
    const double t0 = now_ms();
    if (!p.prefill_phase_error.empty()) { err = p.prefill_phase_error; return false; }
    if (!p.chunk_major_phase_error.empty()) { err=p.chunk_major_phase_error; return false; }
    if(p.gpu_miss_direct_enabled&&!p.gpu_misses_recover_decode(n,err))return false;
    if(p.prefill_repeat_rearm&&!p.maybe_rearm_copied_prefill(n,err))return false;
    if (p.chunk_major_phase_pending && n>0) {
        if (p.chunk>0 && n>=p.chunk_min) return p.forward_chunks(tokens,n,logits_rows,err, slot);
        if (!p.finish_chunk_major_phase(err)) return false;
    }
    if (p.prefill_phase_pending && n > 0) {
        if (p.chunk > 0 && n > p.chunk && n >= p.chunk_min)
            return p.forward_chunks(tokens, n, logits_rows, err, slot);
        std::fprintf(stderr, "ds4: experimental prefill phase: first feed is not multi-chunk; restoring the normal pool\n");
        if (!p.finish_prefill_phase(err)) return false;
    }
    if (n >= 2 && p.chunk > 0 && n >= p.chunk_min && p.pick_path(n)) {   // (the last token rides along; its layers
        const bool ok = p.forward_chunks(tokens, n, logits_rows, err, slot);   // keep the state before it)
        if (ok) p.path_done(now_ms() - t0);
        return ok;
    }
    if (n > 1 && !p.feed_windows(tokens, n - 1, 0, err)) return false;
    p.checkpoint(slot);
    const bool ok = n < 1 || p.feed_windows(tokens + n - 1, 1, logits_rows, err);
    if (ok) p.path_done(now_ms() - t0);   // (windows, when pick_path chose them)
    return ok;
}
int Model::feed_unit() const { return p_->chunk > 0 ? std::max(p_->chunk, (int) p_->o.max_context) : kWin; }
void Model::set_progress(std::function<void(int64_t)> f) { p_->progress = std::move(f); }
void Model::set_cancel(std::function<bool()> f) { p_->cancel = std::move(f); }
bool Model::cancelled() const { return p_->halted; }
int Model::sample(float t, float tp, int tk, float mp, uint64_t seed, uint64_t counter) {
    return p_->sample(t, tp, tk, mp, seed, counter);
}
bool Model::logits(std::vector<float>& out) { return p_->logits(out); }
int Model::image_token() const { return p_->img_tok; }
int Model::dim() const { return p_->g.dim; }
bool Model::encode_image(const uint8_t* rgb, int width, int height, std::vector<uint16_t>& span, std::string& err) {
    return p_->encode_image(rgb, width, height, span, err);
}
void Model::set_images(std::vector<Image> images) {
    std::sort(images.begin(), images.end(), [](const Image& a, const Image& b) { return a.start < b.start; });
    p_->imgs = std::move(images);
}
int Model::draft_block() const { return p_->g.n_mtp > 0 ? p_->g.d_block : 0; }
bool Model::draft(int y, std::vector<int>& d, std::vector<float>& c, std::string& err, float t, float tp, int tk,
                  float mp, uint64_t seed, uint64_t counter) {
    if (!p_->finish_chunk_major_phase(err)) return false;
    if (!p_->finish_prefill_phase(err)) return false;
    return p_->draft(y, d, c, err, t, tp, tk, mp, seed, counter);
}
bool Model::verify(int y, const std::vector<int>& d, float t, float tp, int tk, float mp, uint64_t seed, uint64_t counter,
                   int& accepted, int& next, std::string& err) {
    if (!p_->finish_chunk_major_phase(err)) return false;
    if (!p_->finish_prefill_phase(err)) return false;
    return p_->verify(y, d, t, tp, tk, mp, seed, counter, accepted, next, err);
}
int64_t Model::pos() const { return p_->pos; }
int64_t Model::max_context() const { return p_->o.max_context; }
int Model::vocab() const { return p_->g.vocab; }
const std::vector<int32_t>& Model::tokens() const { return p_->toks; }
void Model::reset() { p_->reset(); }
void Model::checkpoint(int slot) { p_->checkpoint(slot); }
bool Model::restore(int64_t limit) { return p_->restore(limit); }
void Model::pin() { p_->pin(); }
std::vector<int64_t> Model::check_positions() const { return p_->check_positions(); }
int64_t Model::restorable(int64_t limit) const { return p_->restorable(limit); }
uint64_t Model::slot_save(std::vector<uint8_t>& out, std::string& err) { return p_->slot_save(out, err); }
bool Model::slot_load(const uint8_t* data, size_t bytes, int64_t n_pos, std::string& err) {
    return p_->slot_load(data, bytes, n_pos, err);
}
std::string Model::slot_identity() const { return p_->slot_identity(); }
void Model::save_profile() const { p_->save_profile(); }
const Stats& Model::stats() const {
    Stats& st = p_->st;
    if (p_->store) {   // (the store keeps its own counters: copied in when asked)
        const ExpertStore& x = *p_->store;
        st.store_hits = x.stats().hits.load();
        st.store_misses = x.stats().misses.load();
        st.store_prefetched = x.stats().prefetched.load();
        st.store_pf_used = x.stats().prefetch_used.load();
        st.store_pf_wasted = x.stats().prefetch_wasted.load();
        st.store_evictions = x.stats().evictions.load();
        st.store_bytes = x.stats().bytes_read.load();
        st.store_fails = x.stats().fails.load();
        st.store_wait_ms = x.stats().wait_ns.load() / 1e6;
        st.store_busy_ms = x.busy_ms();
        st.store_zc_visible = (int64_t) x.visible_count();
        st.store_slots = (int64_t) x.slots_allocated();
        st.store_slots_max = (int64_t) x.slots_max();
    }
    return st;
}
std::string Model::describe() const { return p_->describe(); }
bool Model::probe_cache_worker_epoch(int timeout_ms,std::string& diagnostic,unsigned& epoch,bool& terminal,std::string& err) {
    return p_->probe_cache_worker_epoch(timeout_ms,diagnostic,epoch,terminal,err);
}
bool Model::probe_copied_pool_shrink(int timeout_ms,std::string& diagnostic,unsigned& epoch,bool& terminal,bool& full,std::string& err) {
    return p_->probe_copied_pool_shrink(timeout_ms,diagnostic,epoch,terminal,full,err);
}
bool Model::probe_copied_workspace_regrow(int timeout_ms,std::string& diagnostic,unsigned& epoch,bool& terminal,bool& operation_ok,bool& published,std::string& err) {
    return p_->probe_copied_workspace_regrow(timeout_ms,diagnostic,epoch,terminal,operation_ok,published,err);
}
std::vector<std::string> Model::info_lines() const { return p_->info_lines(); }

}  // namespace strata::ds4
