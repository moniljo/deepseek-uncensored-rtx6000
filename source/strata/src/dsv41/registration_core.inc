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

