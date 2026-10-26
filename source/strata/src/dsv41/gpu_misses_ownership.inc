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
