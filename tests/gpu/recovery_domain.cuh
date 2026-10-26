// Synthetic ownership/layout only. Registration and recovery decisions come
// from the actual extracted native methods, not from this fixture allocator.
#pragma once

struct RecoveryDomains {
    Experts& experts;Stager& stage;
    static constexpr int count=2;
    std::array<uint8_t*,count> owners{},arenas{};
    size_t page=0,arena_bytes=0,rounded=0,owner_bytes=0;
    std::vector<int> current_slot;
    explicit RecoveryDomains(Experts&e,Stager&s):experts(e),stage(s),current_slot((size_t)e.ne){
        SYSTEM_INFO info{};GetSystemInfo(&info);page=info.dwPageSize;
        need(page&&!(page&(page-1)),"recovery page geometry");
        arena_bytes=(size_t)(e.ne+1)*e.blob;rounded=(arena_bytes+page-1)&~(page-1);
        owner_bytes=rounded+2*page;
        need(count*rounded<1024ull*1024*1024,"recovery physical registration bounded below1GiB");
        auto&A=s.actual;
        need(!A.registered_copy.published&&A.registered_copy.held.empty(),"new recovery fixture requires empty ledger");
        A.gpu_miss_direct_enabled=true;A.gpu_misses_decode_pins=false;
        A.fixture_physical_sources=true;A.copied_ram_domains_published=true;
        A.ram_chunks.clear();A.ram_chunk_bytes.clear();A.cp_lay.clear();A.ram_part.clear();A.copied_ram_domains.clear();
        const std::array<size_t,4>offsets{{e.lay.trellis[0],e.lay.trellis[1],e.lay.trellis[2],e.blob}};
        for(int n=0;n<count;++n){
            owners[(size_t)n]=static_cast<uint8_t*>(VirtualAlloc(nullptr,owner_bytes,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
            need(owners[(size_t)n]!=nullptr,"recovery VirtualAlloc owner");
            arenas[(size_t)n]=owners[(size_t)n]+page;std::memset(owners[(size_t)n],0xa5,owner_bytes);
            for(int x=0;x<e.ne;++x)std::memcpy(arenas[(size_t)n]+(size_t)x*e.blob,e.host.data()+(size_t)x*e.blob,e.blob);
            A.ram_chunks.push_back(arenas[(size_t)n]);A.ram_chunk_bytes.push_back(arena_bytes);A.cp_lay.push_back(offsets);
            for(int x=0;x<=e.ne;++x)A.ram_part.push_back({{arenas[(size_t)n]+(size_t)x*e.blob,nullptr}});
            ramdomain::Domain d{};d.layer=n;d.expert_count=e.ne;d.first_slot=(size_t)n*(e.ne+1);d.slot_count=(size_t)e.ne+1;
            d.part_stride=e.blob;d.arena_index=(size_t)n;d.base=(uintptr_t)arenas[(size_t)n];d.arena_bytes=arena_bytes;
            d.offsets=offsets;d.role_bytes=e.bytes;d.has_spare=true;A.copied_ram_domains.push_back(d);
        }
        for(int x=0;x<e.ne;++x)current_slot[(size_t)x]=x;
        bind(false);
    }
    RecoveryDomains(const RecoveryDomains&)=delete;
    void bind(bool mixed){
        for(int e=0;e<experts.ne;++e){
            const bool resident=mixed&&(e%3==1);stage.layer.slot[(size_t)e]=resident?e:-1;
            stage.layer.ram[(size_t)e]=resident?-1:current_slot[(size_t)e];
            for(int r=0;r<3;++r)stage.layer.fixture_roles[(size_t)e][(size_t)r]=
                arenas[0]+(size_t)current_slot[(size_t)e]*experts.blob+experts.lay.trellis[r];
        }
    }
    void rotate_first_into_spare(){
        stage.drain();const int old=current_slot[0],spare=experts.ne;
        std::memcpy(arenas[0]+(size_t)spare*experts.blob,experts.host.data(),experts.blob);
        std::memcpy(arenas[0]+(size_t)old*experts.blob,experts.host.data()+experts.blob,experts.blob);
        current_slot[0]=spare;bind(false);
    }
    void audit()const{
        for(int n=0;n<count;++n){
            for(size_t j=0;j<page;++j)need(owners[(size_t)n][j]==0xa5&&owners[(size_t)n][page+rounded+j]==0xa5,"recovery arena guard");
            for(size_t j=arena_bytes;j<rounded;++j)need(arenas[(size_t)n][j]==0xa5,"recovery page padding");
        }
    }
    bool physically_retire(){
        stage.drain();stage.actual.gpu_misses_decode_pins=false;
        return regcopy::retire(stage.actual.registered_copy,[](const regcopy::Range&r){return cudaHostUnregister((void*)r.base)==cudaSuccess;});
    }
    ~RecoveryDomains(){
        bool safe=true;
        try{safe=physically_retire();}catch(...){safe=false;}
        if(safe){for(auto*owner:owners)if(owner)VirtualFree(owner,0,MEM_RELEASE);}
        else std::fprintf(stderr,"recovery fixture retained uncertain source owner until process exit\n");
    }
};
