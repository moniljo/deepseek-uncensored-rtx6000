// New v3 fixture support only. Production direct/source/ledger methods are
// extracted verbatim separately. No real model payload or persistent pinning.
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

struct RegisteredDomain {
    Stager& stage;Experts& experts;
    uint8_t* owner=nullptr;uint8_t* arena=nullptr;
    size_t page=0,arena_bytes=0,rounded=0,owner_bytes=0,registered_bytes=0,cpu_stride=0;
    std::array<size_t,4>cpu_offsets{};
    int registers=0,unregisters=0;std::vector<int> physical;
    RegisteredDomain(Experts&E,Stager&S,size_t cpu_padding=0):stage(S),experts(E),physical((size_t)E.ne){
        SYSTEM_INFO info{};GetSystemInfo(&info);page=info.dwPageSize;
        need(page&&!(page&(page-1)),"Windows page geometry");
        cpu_stride=E.blob+cpu_padding;cpu_offsets={{E.lay.trellis[0],E.lay.trellis[1]+cpu_padding,E.lay.trellis[2]+cpu_padding,cpu_stride}};
        arena_bytes=(size_t)(E.ne+1)*cpu_stride;rounded=(arena_bytes+page-1)&~(page-1);
        need(rounded<1024ull*1024*1024,"fixture registration bounded below1GiB");
        owner_bytes=rounded+2*page;
        owner=static_cast<uint8_t*>(VirtualAlloc(nullptr,owner_bytes,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
        need(owner!=nullptr,"VirtualAlloc synthetic copied domain");arena=owner+page;
        std::memset(owner,0xa5,owner_bytes);
        for(int e=0;e<E.ne;++e){physical[(size_t)e]=e;copy_expert(e,e);}
        auto&A=stage.actual;A.gpu_miss_direct_enabled=true;A.gpu_misses_decode_pins=true;
        A.copied_ram_domains_published=true;A.fixture_physical_sources=true;A.ram_chunks={arena};A.ram_chunk_bytes={arena_bytes};
        A.cp_lay={cpu_offsets};
        A.ram_part.resize((size_t)E.ne+1);
        for(int s=0;s<=E.ne;++s)A.ram_part[(size_t)s]={{arena+(size_t)s*cpu_stride,nullptr}};
        ramdomain::Domain d{};d.layer=0;d.expert_count=E.ne;d.first_slot=0;d.slot_count=(size_t)E.ne+1;
        d.part_stride=cpu_stride;d.arena_index=0;d.base=(uintptr_t)arena;d.arena_bytes=arena_bytes;
        d.offsets=A.cp_lay[0];d.role_bytes=E.bytes;d.has_spare=true;A.copied_ram_domains={d};
        bind(false);
    }
    RegisteredDomain(const RegisteredDomain&)=delete;
    void copy_expert(int physical_slot,int expert){
        auto*dst=arena+(size_t)physical_slot*cpu_stride;
        if(cpu_stride==experts.blob)std::memcpy(dst,experts.host.data()+(size_t)expert*experts.blob,experts.blob);
        else for(int r=0;r<3;++r)std::memcpy(dst+cpu_offsets[(size_t)r],experts.role(expert,r),experts.bytes[(size_t)r]);
    }
    void bind(bool mixed){
        for(int e=0;e<experts.ne;++e){
            const bool resident=mixed&&e%3==1;stage.layer.slot[(size_t)e]=resident?e:-1;
            stage.layer.ram[(size_t)e]=resident?-1:physical[(size_t)e];
            for(int r=0;r<3;++r)stage.layer.fixture_roles[(size_t)e][(size_t)r]=
                arena+(size_t)physical[(size_t)e]*cpu_stride+cpu_offsets[(size_t)r];
        }
    }
    void register_prefix(size_t bytes=0){
        stage.drain();need(stage.actual.registered_copy.held.empty(),"registration ledger already owned");
        registered_bytes=bytes?bytes:rounded;
        need(registered_bytes<=rounded&&registered_bytes%page==0,"page-aligned bounded registration extent");
        regcopy::Callbacks calls;
        calls.headroom=[](){return regcopy::Step::ready;};
        calls.register_one=[&](const regcopy::Range&r){
            const auto rc=cudaHostRegister(reinterpret_cast<void*>(r.base),r.bytes,cudaHostRegisterPortable);
            if(rc!=cudaSuccess)return regcopy::Step::fatal;++registers;return regcopy::Step::ready;};
        calls.unregister_one=[&](const regcopy::Range&r){const auto rc=cudaHostUnregister(reinterpret_cast<void*>(r.base));if(rc==cudaSuccess)++unregisters;return rc==cudaSuccess;};
        std::string error;const auto result=regcopy::acquire(stage.actual.registered_copy,{{(uintptr_t)arena,registered_bytes}},calls,error);
        need(result==regcopy::Result::enabled,"REAL cudaHostRegister acquisition failed: "+error);
        unsigned flags=0;CU(cudaHostGetFlags(&flags,arena));
        need((flags&cudaHostRegisterPortable)!=0,"actual host registration not portable/owned");
        need(stage.actual.gpu_misses_retained_ready(),"actual retained-ready admission");
    }
    void retire(){
        stage.drain();
        need(regcopy::retire(stage.actual.registered_copy,[&](const regcopy::Range&r){
            const auto rc=cudaHostUnregister(reinterpret_cast<void*>(r.base));if(rc==cudaSuccess)++unregisters;return rc==cudaSuccess;}),"actual ledger unregister retained uncertain owner");
        need(!stage.actual.registered_copy.published&&stage.actual.registered_copy.held.empty(),"actual ledger retirement not zero");
    }
    void rotate_first_into_spare(){
        stage.drain();const auto fingerprint=stage.actual.gpu_misses_pin_fingerprint();
        const int old=physical[0],spare=experts.ne;
        copy_expert(spare,0);
        // Make stale remembered slot0 observably wrong without changing ownership.
        copy_expert(old,1);
        physical[0]=spare;bind(false);
        need(stage.actual.gpu_misses_pin_fingerprint()==fingerprint,"rotation changed physical registration ownership");
        need(stage.actual.gpu_misses_direct_source(stage.layer,0)==arena+(size_t)spare*cpu_stride,"actual direct source remembered old expert slot");
    }
    void audit()const{
        for(size_t i=0;i<page;++i)need(owner[i]==0xa5&&owner[page+rounded+i]==0xa5,"VirtualAlloc domain outer canary");
        for(size_t i=arena_bytes;i<rounded;++i)need(arena[i]==0xa5,"registration page padding changed");
    }
    ~RegisteredDomain(){
        if(!owner)return;
        // Never free a physical DMA source until BOTH streams are proven joined.
        const auto cs=stage.actual.gpu_misses_cs?cudaStreamSynchronize(stage.actual.gpu_misses_cs):cudaSuccess;
        const auto st=cudaStreamSynchronize(stage.device.st);
        bool released=cs==cudaSuccess&&st==cudaSuccess;
        if(released)released=regcopy::retire(stage.actual.registered_copy,[&](const regcopy::Range&r){const auto rc=cudaHostUnregister(reinterpret_cast<void*>(r.base));if(rc==cudaSuccess)++unregisters;return rc==cudaSuccess;});
        if(released)VirtualFree(owner,0,MEM_RELEASE);
        else std::fprintf(stderr,"fixture retained uncertain registered source owner until process exit\n");
    }
};

struct DirectResult {std::string family,kind;int T=0,direct=0,bounce=0,missing_entries=0;size_t registered_bytes=0;uint64_t output_hash=0;};
DirectResult direct_numeric(Experts&E,Stager&stage,RegisteredDomain&domain,int T,const char*kind,bool mixed=false,bool offset_mismatch=false,bool unique=false){
    domain.bind(mixed);const auto ids=picks(T,E.ne,unique);std::vector<unsigned long long>reference_plan((size_t)T*PICKS,0);
    for(int t=0;t<T;++t)for(int k=0;k<K;++k)reference_plan[t*PICKS+k]=E.ptr(ids[t*PICKS+k]);
    MathBuffers math(T,stage.device.st);math.launch(E,reference_plan,stage.device.st);
    const auto reference=math.out.get(stage.device.st),reference_h=math.h.get(stage.device.st),reference_part=math.part.get(stage.device.st);
    std::vector<int>missing;int entries=0;std::vector<bool>is_direct;
    for(int t=0;t<T;++t)for(int k=0;k<K;++k){const int e=ids[t*PICKS+k];if(stage.layer.slot[(size_t)e]>=0)continue;++entries;
        if(std::find(missing.begin(),missing.end(),e)==missing.end())missing.push_back(e);}
    int direct=0;
    for(int e:missing){const size_t offset=(size_t)domain.physical[(size_t)e]*domain.cpu_stride;
        const bool covered=!offset_mismatch&&offset<=domain.registered_bytes&&E.blob<=domain.registered_bytes-offset;
        is_direct.push_back(covered);direct+=covered;
        need(stage.actual.gpu_misses_direct_source(stage.layer,e)==(covered?domain.arena+offset:nullptr),"independent physical coverage expectation differs");}
    const auto source_hash=hash64(domain.arena,domain.arena_bytes),host_hash=hash64(E.host.data(),E.host.size());
    const auto before=stage.actual.st;api.reset();stage.trace();bool used=false;std::string error;
    need(stage.actual.gpu_misses_prepare(stage.layer,stage.device,T,ids.data(),used,error)&&used&&!stage.actual.terminal,"actual direct prepare: "+error);
    // NO copy drain, host readback or event sync between prepare and real GG use.
    math.launch(E,{},stage.device.st,stage.plan.p);
    exact_finite(reference,math.out.get(stage.device.st),"direct whole public expert output");
    exact_finite(reference_h,math.h.get(stage.device.st),"direct public gate/up");
    if(T>1)exact_finite(reference_part,math.part.get(stage.device.st),"direct public down partial");
    need(api.blob_copies==(int)missing.size()&&api.bytes==missing.size()*E.blob&&api.waits==1&&api.ready_records==1,"direct DMA count/dependency");
    need(stage.actual.st.gpu_miss_direct_unique-before.gpu_miss_direct_unique==direct&&
         stage.actual.st.gpu_miss_direct_bytes-before.gpu_miss_direct_bytes==(int64_t)(direct*E.blob)&&
         stage.actual.st.gpu_miss_bounce_unique-before.gpu_miss_bounce_unique==(int)missing.size()-direct&&
         stage.actual.st.gpu_miss_bounce_bytes-before.gpu_miss_bounce_bytes==(int64_t)((missing.size()-direct)*E.blob),"actual direct/bounce counters vs independent expectations");
    auto plan=stage.plan.get(stage.device.st);
    for(int t=0;t<T;++t)for(int k=0;k<PICKS;++k){const int at=t*PICKS+k;if(k>=K){need(plan[at]==0,"direct unused physical pick");continue;}
        const int e=ids[at];const auto p=std::find(missing.begin(),missing.end(),e);
        const auto want=p==missing.end()?E.ptr(e):reinterpret_cast<unsigned long long>(stage.gpu.p+(size_t)(p-missing.begin())*stage.stride);
        need(plan[at]==want,"direct original-pick order/dedup pointer");}
    std::vector<uint8_t>raw(E.blob),wanted(E.blob);
    for(size_t j=0;j<missing.size();++j){const int e=missing[j];
        if(is_direct[j])std::memcpy(wanted.data(),E.host.data()+(size_t)e*E.blob,E.blob);
        else {std::fill(wanted.begin(),wanted.end(),0);for(int r=0;r<3;++r)std::memcpy(wanted.data()+E.lay.trellis[r],E.role(e,r),E.bytes[r]);}
        CU(cudaMemcpyAsync(raw.data(),stage.gpu.p+j*stage.stride,E.blob,cudaMemcpyDeviceToHost,stage.device.st));stage.drain();need(raw==wanted,"actual direct/bounce full blob bytes");}
    need(source_hash==hash64(domain.arena,domain.arena_bytes)&&host_hash==hash64(E.host.data(),E.host.size()),"registered/source bytes changed by GPU helper");
    math.audit(stage.device.st);stage.audit();domain.audit();E.resident->audit(stage.device.st);
    return {"",kind,T,direct,(int)missing.size()-direct,entries,domain.registered_bytes,hash64(reference.data(),reference.size()*4)};
}

int direct_guards(Experts&E,Stager&stage,RegisteredDomain&domain){
    int checks=0;auto&A=stage.actual;auto&L=stage.layer;
    auto refused=[&](){api.reset();need(A.gpu_misses_direct_source(L,0)==nullptr&&api.copies==0&&api.event_records==0,"direct admission refusal attempted GPU operation");++checks;};
    auto bool_guard=[&](bool&value){const bool old=value;value=!old;refused();value=old;};
    bool_guard(A.gpu_miss_direct_enabled);bool_guard(A.gpu_misses_enabled);bool_guard(A.gpu_misses_decode_pins);
    bool_guard(A.chunk_major_phase_pending);bool_guard(A.prefill_phase_pending);bool_guard(A.copied_ram_domains_published);bool_guard(A.registered_copy.published);bool_guard(A.copy_mode);
    A.chunk_major_phase_error="fixture-terminal";refused();A.chunk_major_phase_error.clear();
    A.NH=2;refused();A.NH=1;A.store=&A;refused();A.store=nullptr;
    auto devs=A.devs;A.devs.clear();refused();A.devs=devs;
    auto held=A.registered_copy.held;A.registered_copy.held.clear();refused();A.registered_copy.held=held;
    L.nm=2;refused();L.nm=1;L.dev=1;refused();L.dev=0;
    const int oldram=L.ram[0];L.ram[0]=-1;refused();L.ram[0]=(int)A.ram_part.size();refused();L.ram[0]=oldram;
    auto oldpart=A.ram_part[(size_t)oldram];A.ram_part[(size_t)oldram][0]+=256;refused();A.ram_part[(size_t)oldram]=oldpart;
    A.ram_part[(size_t)oldram][1]=domain.arena;refused();A.ram_part[(size_t)oldram]=oldpart;
    auto domains=A.copied_ram_domains;A.copied_ram_domains.clear();refused();A.copied_ram_domains=domains;
    auto&d=A.copied_ram_domains[0];d.arena_index=1;refused();d=domains[0];d.base+=256;refused();d=domains[0];
    d.arena_bytes-=256;refused();d=domains[0];d.part_stride-=256;refused();d=domains[0];
    d.offsets[1]+=256;refused();d=domains[0];d.role_bytes[2]-=1;refused();d=domains[0];
    auto cp=A.cp_lay[0];A.cp_lay[0][1]+=256;refused();A.cp_lay[0]=cp;
    need(A.gpu_misses_direct_source(L,-1)==nullptr&&A.gpu_misses_direct_source(L,E.ne)==nullptr,"direct expert bounds");++checks;
    // The actual retained predicate does not inspect Stop flags or cached expert
    // IDs: published physical owners persist until real maintenance retirement.
    need(A.gpu_misses_retained_ready(),"valid retained state was not restored");++checks;
    return checks;
}

int direct_ready_lifetime(Experts&E,cudaStream_t stream){
    Stager stage(E,stream);RegisteredDomain domain(E,stage);domain.register_prefix();domain.bind(true);
    const auto ids=picks(4,E.ne,false);MathBuffers math(4,stream);std::vector<unsigned long long>ref(4*PICKS,0);
    for(int t=0;t<4;++t)for(int k=0;k<K;++k)ref[t*PICKS+k]=E.ptr(ids[t*PICKS+k]);
    math.launch(E,ref,stream);const auto reference=math.out.get(stream);
    Gate gate(stage.actual.gpu_misses_cs);CU(cudaLaunchHostFunc(stage.actual.gpu_misses_cs,wait_gate,&gate));
    api.reset();stage.trace();bool used=false;std::string error;
    need(stage.actual.gpu_misses_prepare(stage.layer,stage.device,4,ids.data(),used,error)&&used,"direct blocked-ready prepare");
    math.launch(E,{},stream,stage.plan.p,false);cudaEvent_t consumed=nullptr;CU(cudaEventCreateWithFlags(&consumed,cudaEventDisableTiming));CU(cudaEventRecord(consumed,stream));
    const bool pending=bounded_wait([&]{return gate.entered.load();})&&cudaEventQuery(consumed)==cudaErrorNotReady;
    // Retire the REAL ledger only after the same actual two-stream drain method
    // returns; keep the physical source owner and registration alive meanwhile.
    std::atomic<bool>drained{false};bool ok=false;std::exception_ptr failure;
    std::thread retire([&]{try{std::string err;ok=stage.actual.drain_gpu_misses(err);if(!ok)throw std::runtime_error(err);drained=true;}catch(...){failure=std::current_exception();drained=true;}});
    const bool entered=bounded_wait([&]{return api.entered_whole_stream_drain.load();});
    const bool still_owned=entered&&!drained.load()&&stage.actual.registered_copy.published&&stage.actual.registered_copy.held.size()==1&&domain.unregisters==0;
    gate.release=true;retire.join();gate.stream=nullptr;if(failure)std::rethrow_exception(failure);
    CU(cudaEventSynchronize(consumed));CU(cudaEventDestroy(consumed));
    need(pending&&still_owned&&ok&&api.copy_drains==1&&api.compute_drains==1,"direct DMA registration was not retained through both-stream drain");
    exact_finite(reference,math.out.get(stream),"direct blocked-ready whole public output");
    const auto fp=stage.actual.gpu_misses_pin_fingerprint();need(fp!=0,"actual ledger fingerprint");
    domain.retire();need(!stage.actual.gpu_misses_retained_ready()&&domain.unregisters==domain.registers,"actual physical registration retirement");
    math.audit(stream);stage.audit();domain.audit();return 2;
}
