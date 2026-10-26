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
