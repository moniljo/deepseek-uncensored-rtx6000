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
