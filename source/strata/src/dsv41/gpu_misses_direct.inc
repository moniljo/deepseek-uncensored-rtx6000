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
