// Actual candidate method below is copied verbatim; fixture support surrounds it.
#include <mutex>
#include "recovery_policy.generated.hpp"
#include "epoch_policy.generated.hpp"
#include "repeat_policy.generated.hpp"
#include "reserve_policy.generated.hpp"
#include "policy.generated.hpp"
#include "direct_policy.generated.hpp"
#include "registered_copy.generated.hpp"
#include "frozen-support/epoch-support/expert_pool_shrink.hpp"
namespace gpumiss=strata::ds4::gpumiss;
namespace regcopy=strata::ds4::regcopy;
namespace ramdomain=strata::ds4::ramdomain;
namespace epochpolicy=strata::ds4::epochpolicy;
namespace repeatpolicy=strata::ds4::repeatpolicy;
namespace copiedshrink=strata::ds4::copiedshrink;
inline double now_ms(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct Layer {
    int il=0,nm=1,dev=0,n_exp=0,k_exp=6,n_slots=0,spare_vram=0;bool cpu_only=false,mtp=false;
    size_t blob=0;x3::ExpertLayout lay{};uint8_t* vram_base=nullptr;
    std::vector<int>slot,ram;std::vector<std::array<const uint8_t*,3>>fixture_roles;
};
struct Dev {int id=0;cudaStream_t st=nullptr;unsigned long long*plan_ptr=nullptr;size_t experts=1ull<<30;};
struct Actual {
    bool gpu_misses_enabled=true,copy_mode=true;int NH=1;bool terminal=false;
    bool gpu_miss_direct_enabled=false,gpu_misses_decode_pins=false,fixture_physical_sources=false;
    bool chunk_major_phase_pending=false,prefill_phase_pending=false,copied_ram_domains_published=false;
    std::string chunk_major_phase_error;void*store=nullptr;
    regcopy::Ledger registered_copy;std::vector<ramdomain::Domain>copied_ram_domains;
    std::vector<std::array<size_t,4>>cp_lay;std::vector<std::array<uint8_t*,2>>ram_part;
    std::vector<uint8_t*>ram_chunks;std::vector<size_t>ram_chunk_bytes;
    uint8_t*gpu_misses_stage=nullptr;size_t gpu_misses_stride=0,gpu_misses_bytes=0;
    uint8_t*gpu_misses_host[2]={};cudaEvent_t gpu_misses_done[2]={};bool gpu_misses_busy[2]={};cudaStream_t gpu_misses_cs=nullptr;cudaEvent_t gpu_misses_ready=nullptr;
    unsigned long long*gpu_misses_plan_pin=nullptr;
    struct Stats {int64_t gpu_miss_retained_ranges=0,gpu_miss_retained_bytes=0;int64_t gpu_miss_windows=0,gpu_miss_unique=0,gpu_miss_entries=0,gpu_miss_bytes=0,gpu_miss_fallbacks=0;
    int64_t gpu_miss_direct_unique=0,gpu_miss_direct_bytes=0,gpu_miss_bounce_unique=0,gpu_miss_bounce_bytes=0;
    double gpu_miss_pack_ms=0,gpu_miss_bounce_wait_ms=0,gpu_miss_h2d_call_ms=0;}st;
    std::vector<Dev>devs=std::vector<Dev>(1);
    // Fixture adapters ONLY: no claim of real Model worker/47-buffer execution.
    using Impl=Actual;
    struct CacheProbeState {int64_t position=31;size_t tokens=31,ctokens=3;unsigned issued=43;
        uint64_t token_hash=0x12345,checkpoint_hash=0x67890,map_hash=0xabcd;};
    struct FixtureWorker {bool blocked=false;bool idle()const{return !blocked;}};
    struct FixtureChunk {
        int C=65280,A=4096,AI=128,restore_C=65280,restore_A=4096,restore_AI=128;
        bool flow=true;cudaStream_t cs=nullptr;
        void*cold_f16=nullptr,*cold_f16_y=nullptr,*cold_f16_meta_host=nullptr,*cold_f16_meta_dev=nullptr,*cold_f16_workspace=nullptr,*cold_f16_blas=nullptr;
        void*eg_weight_cache=nullptr,*dense_projection_cache=nullptr;
    };
    using Chunk=FixtureChunk;
    struct FixtureOptions {bool expert_cache=true;int64_t max_context=1048576,reserve_mib=6144;}o;
    struct FixturePack {bool gguf=true;}pk;
    bool file_experts=true,ram_tier=false,cache_epoch_enabled=true,record_copied_ram_domains=true;
    bool prefill_repeat_rearm=true,prefill_repeat_preparing=false;
    int prefill_repeat_min_tokens=32768,chunk=65280;
    bool gpu_misses_recovery_needed=false,gpu_misses_recovery_preparing=false;
    unsigned gpu_misses_recovery_lease=0,seq_issued=43;
    uint64_t gpu_misses_recovery_attempts=0;
    size_t gpu_misses_last_decode_pool=2ull<<30;
    epochpolicy::Machine cache_epoch_machine;
    bool register_ram_all=false;size_t register_ram_mb=196608,register_gpu_floor_mb=3072;
    double registered_copy_driver_ms=0,registered_copy_setup_ms=0,registered_copy_unregister_ms=0;
    static constexpr uint64_t kRamMargin=8ull<<30;
    uint64_t fixture_ram_available=64ull<<30;double fixture_commit_available=64.0*(1ull<<30);
    bool halted=false;std::function<bool()>cancel;
    std::vector<Layer>L;
    std::vector<std::unique_ptr<FixtureWorker>>workers;
    std::vector<std::unique_ptr<FixtureChunk>>chunks;
    std::mutex sw_mu;int sw_inflight=0,ram_inflight=0;
    std::vector<int>sw_queue,sw_ready,ram_queue,ram_ready;
    bool fixture_recovery_adapters=false,fixture_stop=false,fixture_begin_refuse=false,fixture_end_fail=false;
    bool fixture_stop_after_compact=false,fixture_stop_after_release=false,fixture_compact_unavailable=false;
    int fixture_acquires=0,fixture_releases=0,fixture_compactions=0;
    CacheProbeState fixture_state;
    std::vector<int>copy_spare;
    int fixture_shrink_calls=0;
    size_t fixture_shrink_pool_after=1ull<<30;
    double fixture_available_pool_budget=double(3ull<<30);
    bool fixture_shrink_stop=false,fixture_shrink_hard=false,fixture_shrink_refuse=false;
    bool fixture_unknown_targets_attained=true,fixture_inventory_error=false;

    size_t unit(const Layer&l)const{return l.blob;}
    size_t mx_part_bytes(const Layer&l,int r)const{return gg::type_row_bytes(gg::expert_type(l.lay,r),r<2?D:F)*(r<2?F:D);}
    const uint8_t*mx_mat(const Layer&l,int e,int r)const{
        if(fixture_physical_sources){const int at=l.ram.at((size_t)e);if(at<0)return nullptr;
            return ram_part.at((size_t)at)[0]+cp_lay.at((size_t)l.il).at((size_t)r);}
        return l.fixture_roles.at((size_t)e)[(size_t)r];}
    bool cache_epoch_terminal(const std::string&why,std::string&err){terminal=true;err=why;if(fixture_recovery_adapters){chunk_major_phase_error=why;cache_epoch_machine.release(true);}return false;}
    uint64_t ram_avail()const{return fixture_ram_available;}
    double commit_room()const{return fixture_commit_available;}
    CacheProbeState cache_probe_state()const{auto v=fixture_state;v.issued=seq_issued;return v;}
    bool begin_cache_worker_epoch(unsigned&lease,int timeout,std::string&err){
        ++fixture_acquires;
        if(fixture_begin_refuse){err="fixture acquire refusal";return false;}
        if(!cache_epoch_machine.start(now_ms(),timeout)||!cache_epoch_machine.publishers_done()||
           !cache_epoch_machine.cache_done()||!cache_epoch_machine.acknowledged()){
            err="fixture epoch policy rejected";return false;
        }
        lease=cache_epoch_machine.serial;return true;
    }
    bool end_cache_worker_epoch(unsigned lease,std::string&err){
        if(fixture_end_fail){err="fixture exact release failure";return false;}
        if(!lease||cache_epoch_machine.serial!=lease||cache_epoch_machine.phase!=epochpolicy::Phase::held){
            err="fixture release lease mismatch";return false;
        }
        ++fixture_releases;cache_epoch_machine.release(false);
        if(fixture_stop_after_release)fixture_stop=true;
        return true;
    }
    bool finish_chunk_major_phase(std::string&err,bool=true,bool=true){
        ++fixture_compactions;
        if(fixture_compact_unavailable)return true;
        auto&c=*chunks.at(0);c.C=c.A=chunk=1024;
        devs.at(0).experts=(size_t)gpumiss::recovery_pool_budget(fixture_available_pool_budget,devs.at(0).experts,gpu_misses_last_decode_pool);
        ++fixture_state.map_hash; // expected pool publication may change maps
        if(fixture_stop_after_compact){fixture_stop=true;halted=true;err="fixture stopped during compaction";return false;}
        return true;
    }
    bool shrink_copied_expert_pool(unsigned lease,bool&stopped,std::string&err){
        ++fixture_shrink_calls;stopped=false;
        if(!lease||lease!=cache_epoch_machine.serial||cache_epoch_machine.phase!=epochpolicy::Phase::held){err="fixture shrink lease";return false;}
        if(fixture_shrink_hard)return cache_epoch_terminal("fixture terminal pool transaction",err);
        if(fixture_shrink_refuse){err="fixture coherent shrink refusal";return false;}
        devs.at(0).experts=fixture_shrink_pool_after;++fixture_state.map_hash;
        if(gpu_misses_last_decode_pool==0){
            // Valid synthetic minimum target for the ACTUAL copiedshrink::plan.
            // This is not a GPU allocation/owner transaction implementation.
            copy_spare.resize(L.size());
            for(auto&domain:copied_ram_domains){auto&l=L.at((size_t)domain.layer);
                domain.slot_count=(size_t)l.n_exp;l.n_slots=2;l.spare_vram=1;
                l.slot.assign((size_t)l.n_exp,-1);l.ram.resize((size_t)l.n_exp);
                for(int e=0;e<l.n_exp;++e)l.ram[(size_t)e]=(int)domain.first_slot+e;
                l.slot.back()=0;l.ram.back()=-1;
                copy_spare[(size_t)l.il]=(int)domain.first_slot+l.n_exp-1;
            }
        }
        if(fixture_shrink_stop){stopped=true;fixture_stop=true;}
        return true;
    }
    bool inspect_copied_ram_inventory(ramdomain::Inventory&out,std::string&err){
        if(fixture_inventory_error){err="fixture inventory refusal";return false;}
        out={};out.layers.resize(L.size());out.free_holes=fixture_unknown_targets_attained?0:1;
        for(size_t i=0;i<L.size();++i)out.layers[i].layer=(int)i;
        return true;
    }


#define CK(x) do {const auto fixture_status=(x);if(fixture_status!=cudaSuccess){err=std::string("fixture checked CUDA: ")+cudaGetErrorString(fixture_status);return false;}}while(0)
#define cudaHostRegister fixture_host_register
#define cudaHostUnregister fixture_host_unregister
#define cudaMemGetInfo fixture_registration_meminfo
#define cudaStreamSynchronize fixture_stream_sync
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

#define cudaMemcpyAsync fixture_memcpy
#define cudaEventRecord fixture_event_record
#define cudaEventSynchronize fixture_event_sync
#define cudaStreamWaitEvent fixture_stream_wait
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

#undef CK
#undef cudaHostRegister
#undef cudaHostUnregister
#undef cudaMemGetInfo
#undef cudaMemcpyAsync
#undef cudaEventRecord
#undef cudaEventSynchronize
#undef cudaStreamSynchronize
#undef cudaStreamWaitEvent
};
