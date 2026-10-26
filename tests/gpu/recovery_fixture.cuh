// Additive v5 fixture. REAL registration, retirement, streams and public GG
// math; actual recover/core methods. Epoch/47-buffer adapters are synthetic.
#pragma once
#include "recovery_domain.cuh"

void configure_recovery(Stager&s,RecoveryDomains&domains,bool compact=false){
    auto&A=s.actual;A.fixture_recovery_adapters=true;A.gpu_misses_recovery_needed=true;
    A.gpu_misses_decode_pins=false;A.chunk=compact?1024:65280;
    A.chunks.push_back(std::make_unique<Actual::FixtureChunk>());
    A.chunks[0]->C=A.chunk;A.chunks[0]->A=compact?1024:4096;A.chunks[0]->cs=s.device.st;
    A.L={s.layer,s.layer};A.L[1].il=1;
    for(int e=0;e<s.layer.n_exp;++e){A.L[1].slot[(size_t)e]=-1;A.L[1].ram[(size_t)e]=s.layer.n_exp+1+e;}
    A.cancel=[&A]{return A.fixture_stop||(registration_api.stop_after_registration&&registration_api.successes>0);};
    need(!A.gpu_misses_retained_ready()&&!A.registered_copy.published&&A.registered_copy.held.empty(),"recovery must start unregistered");
    domains.audit();
}

struct RecoveryResult {std::string family,kind;int T=0,direct=0;size_t registered_bytes=0;uint64_t output_hash=0;};
RecoveryResult recovery_numeric(Experts&E,Stager&s,RecoveryDomains&domains,int T,const char*kind,bool mixed=false){
    auto&A=s.actual;domains.bind(mixed);A.L[0]=s.layer;const auto ids=picks(T,E.ne,false);
    const int prior_registers=registration_api.registrations;const bool previously_healthy=A.gpu_misses_retained_ready();
    A.gpu_misses_recovery_needed=true; // also exercise idempotent already-healthy admission
    const auto before=A.cache_probe_state();const unsigned issued=A.seq_issued;
    std::string error;need(A.gpu_misses_recover_decode(17,error)&&!A.terminal,"actual recovery refused: "+error);
    need(A.gpu_misses_retained_ready()&&!A.gpu_misses_recovery_needed&&!A.gpu_misses_recovery_preparing&&
         A.gpu_misses_recovery_lease==0&&!A.chunk_major_phase_pending&&!A.prefill_phase_pending,"recovery publication/private context");
    const auto after=A.cache_probe_state();
    need(before.position==after.position&&before.tokens==after.tokens&&before.ctokens==after.ctokens&&
         before.token_hash==after.token_hash&&before.checkpoint_hash==after.checkpoint_hash&&issued==A.seq_issued,"recovery changed token/checkpoint/issued fixture state");
    need(A.registered_copy.held.size()==2&&A.registered_copy.bytes()==2*domains.arena_bytes,"actual recovery same-ledger coverage");
    need(registration_api.registrations-prior_registers==(previously_healthy?0:2),"recovery duplicate registration or missing acquisition");
    std::vector<unsigned long long>reference_plan((size_t)T*PICKS,0);std::vector<int>missing;
    for(int t=0;t<T;++t)for(int k=0;k<K;++k){const int e=ids[t*PICKS+k];reference_plan[t*PICKS+k]=E.ptr(e);
        if(s.layer.slot[(size_t)e]<0&&std::find(missing.begin(),missing.end(),e)==missing.end())missing.push_back(e);}
    for(int e:missing)need(A.gpu_misses_direct_source(s.layer,e)==domains.arenas[0]+(size_t)domains.current_slot[(size_t)e]*E.blob,"recovered direct source does not use current slot");
    MathBuffers math(T,s.device.st);math.launch(E,reference_plan,s.device.st);
    const auto reference=math.out.get(s.device.st),reference_h=math.h.get(s.device.st),reference_part=math.part.get(s.device.st);
    const auto stats=A.st;const auto source_hash=hash64(domains.arenas[0],domains.arena_bytes);
    api.reset();s.trace();bool used=false;need(A.gpu_misses_prepare(s.layer,s.device,T,ids.data(),used,error)&&used,"actual recovered prepare: "+error);
    // Consume the ACTUAL device plan before host readback/drain can hide a
    // missing ready dependency. Both weights and math are the production path.
    math.launch(E,{},s.device.st,s.plan.p);
    exact_finite(reference,math.out.get(s.device.st),"recovered whole output");
    exact_finite(reference_h,math.h.get(s.device.st),"recovered gate/up");
    if(T>1)exact_finite(reference_part,math.part.get(s.device.st),"recovered down partials");
    need(A.st.gpu_miss_direct_unique-stats.gpu_miss_direct_unique==(int64_t)missing.size()&&
         A.st.gpu_miss_bounce_unique==stats.gpu_miss_bounce_unique,"recovered covered slot fell back to bounce");
    need(api.blob_copies==(int)missing.size()&&api.bytes==missing.size()*E.blob&&api.waits==1&&api.ready_records==1,"recovered DMA bytes/ready ordering");
    auto plan=s.plan.get(s.device.st);
    for(int t=0;t<T;++t)for(int k=0;k<PICKS;++k){const size_t at=(size_t)t*PICKS+k;
        if(k>=K){need(plan[at]==0,"recovered physical pick tail");continue;}
        const int e=ids[at];auto it=std::find(missing.begin(),missing.end(),e);
        const auto want=it==missing.end()?E.ptr(e):reinterpret_cast<unsigned long long>(s.gpu.p+(size_t)(it-missing.begin())*s.stride);
        need(plan[at]==want,"recovered original pick/dedup plan");}
    std::vector<uint8_t>raw(E.blob);
    for(size_t j=0;j<missing.size();++j){CU(cudaMemcpyAsync(raw.data(),s.gpu.p+j*s.stride,E.blob,cudaMemcpyDeviceToHost,s.device.st));s.drain();
        need(std::memcmp(raw.data(),E.host.data()+(size_t)missing[j]*E.blob,E.blob)==0,"recovered full compressed blob differs");}
    need(source_hash==hash64(domains.arenas[0],domains.arena_bytes),"recovery direct DMA mutated RAM");
    math.audit(s.device.st);s.audit();domains.audit();E.resident->audit(s.device.st);
    return {"",kind,T,(int)missing.size(),A.registered_copy.bytes(),hash64(reference.data(),reference.size()*sizeof(float))};
}

int recovery_faults(Experts&E,cudaStream_t stream){
    int checks=0;
    // No new retained context on a pre-acquire Stop, compact Stop, partial
    // register Stop, or Stop after exact release but BEFORE publication.
    for(int mode=0;mode<4;++mode){
        Stager s(E,stream);RecoveryDomains d(E,s);configure_recovery(s,d);auto&A=s.actual;
        registration_api.reset();if(mode==0)A.fixture_stop=true;if(mode==1)A.fixture_stop_after_compact=true;
        if(mode==2)registration_api.stop_after_registration=true;if(mode==3)A.fixture_stop_after_release=true;
        std::string err;const bool ok=A.gpu_misses_recover_decode(17,err);
        need(!ok&&A.halted&&!A.terminal&&!A.gpu_misses_decode_pins&&!A.registered_copy.published&&A.registered_copy.held.empty(),"late Stop retained a failed recovery phase");
        need(registration_api.successes==registration_api.releases,"Stop did not reverse every real registration");
        need(!A.gpu_misses_recovery_preparing&&A.gpu_misses_recovery_lease==0&&A.seq_issued==43,"Stop leaked private lease/preparing state");
        need(A.cache_epoch_machine.phase==epochpolicy::Phase::idle,"soft Stop did not release exact fixture lease");
        d.audit();++checks;
    }
    // A recoverable second registration failure must retire the real first
    // registration and leave the SAME ledger usable for a subsequent request.
    {
        Stager s(E,stream);RecoveryDomains d(E,s);configure_recovery(s,d);auto&A=s.actual;
        registration_api.reset();registration_api.fail_registration=2;std::string err;
        need(A.gpu_misses_recover_decode(17,err)&&!A.terminal&&!A.gpu_misses_retained_ready()&&A.registered_copy.held.empty(),"partial registration refusal not coherent");
        need(registration_api.successes==1&&registration_api.releases==1&&A.fixture_releases==1,"partial acquisition did not roll back before release");
        registration_api.reset();auto c=recovery_numeric(E,s,d,4,"retry-same-ledger-after-registration-refusal",true);
        need(c.direct>0&&registration_api.successes==2&&A.gpu_misses_recovery_attempts==2,"same-ledger retry did not acquire coverage");
        need(A.gpu_misses_retire_pins("fixture-retry-retire",err)&&registration_api.releases==2,"recovered retry did not physically retire");checks+=2;
    }
    // Uncertain rollback must keep ownership and poison, never compute through
    // a partial ledger. A later fixture teardown clears the injected failure.
    {
        Stager s(E,stream);RecoveryDomains d(E,s);configure_recovery(s,d);auto&A=s.actual;
        registration_api.reset();registration_api.fail_registration=2;registration_api.fail_unregistration=1;std::string err;
        need(!A.gpu_misses_recover_decode(17,err)&&A.terminal&&!A.registered_copy.published&&A.registered_copy.held.size()==1,"uncertain unregister lost terminal ownership");
        need(!A.gpu_misses_decode_pins&&registration_api.successes==1&&registration_api.releases==0,"uncertain source was republished/released");
        unsigned flags=0;CU(cudaHostGetFlags(&flags,(void*)A.registered_copy.held[0].base));
        registration_api.fail_unregistration=0;need(A.gpu_misses_retire_pins("fixture-uncertain-cleanup",err)&&A.registered_copy.held.empty(),"fixture uncertain owner cleanup failed");++checks;
    }
    for(int mode=0;mode<2;++mode){
        Stager s(E,stream);RecoveryDomains d(E,s);configure_recovery(s,d);auto&A=s.actual;registration_api.reset();
        if(mode==0)registration_api.fail_meminfo=1;else A.fixture_end_fail=true;
        std::string err;need(!A.gpu_misses_recover_decode(17,err)&&A.terminal&&!A.gpu_misses_decode_pins&&!A.registered_copy.published,"CUDA/release error was not terminal");
        if(mode==1)need(A.registered_copy.held.size()==2&&registration_api.releases==0,"failed exact release discarded real registrations");
        A.fixture_end_fail=false;need(A.gpu_misses_retire_pins("fixture-terminal-cleanup",err),"terminal fixture registration cleanup");++checks;
    }
    // Long feeds and live phases retain their original paths. Refusal checks
    // do not register or launch anything through the recovery coordinator.
    for(int mode=0;mode<3;++mode){
        Stager s(E,stream);RecoveryDomains d(E,s);configure_recovery(s,d);auto&A=s.actual;registration_api.reset();
        int remaining=17;if(mode==0)remaining=32768;if(mode==1)A.chunk_major_phase_pending=true;if(mode==2)A.gpu_miss_direct_enabled=false;
        std::string err;need(A.gpu_misses_recover_decode(remaining,err)&&registration_api.registrations==0&&A.fixture_acquires==0&&!A.gpu_misses_decode_pins,"ineligible recovery performed maintenance");++checks;
    }
    {
        Stager s(E,stream);RecoveryDomains d(E,s);configure_recovery(s,d);auto&A=s.actual;registration_api.reset();
        A.gpu_misses_recovery_preparing=true;A.gpu_misses_recovery_lease=1;A.cache_epoch_machine.serial=2;A.cache_epoch_machine.phase=epochpolicy::Phase::held;
        std::string err;need(!A.gpu_misses_recovery_context()&&!A.register_copied_ram_for_prefill(err)&&registration_api.registrations==0,"stale recovery lease admitted actual core");
        A.gpu_misses_recovery_preparing=false;A.gpu_misses_recovery_lease=0;A.cache_epoch_machine.release(false);++checks;
    }
    return checks;
}

int recovery_retirement_lifetime(Experts&E,cudaStream_t stream){
    Stager s(E,stream);RecoveryDomains domain(E,s);configure_recovery(s,domain,true);auto&A=s.actual;
    registration_api.reset();std::string err;need(A.gpu_misses_recover_decode(17,err),"lifetime initial recovery");
    domain.bind(true);const auto ids=picks(4,E.ne,false);MathBuffers math(4,stream);std::vector<unsigned long long>ref(4*PICKS,0);
    for(int t=0;t<4;++t)for(int k=0;k<K;++k)ref[t*PICKS+k]=E.ptr(ids[t*PICKS+k]);
    math.launch(E,ref,stream);const auto reference=math.out.get(stream);
    Gate gate(A.gpu_misses_cs);CU(cudaLaunchHostFunc(A.gpu_misses_cs,wait_gate,&gate));api.reset();s.trace();bool used=false;
    need(A.gpu_misses_prepare(s.layer,s.device,4,ids.data(),used,err)&&used,"recovery gated direct prepare");
    math.launch(E,{},stream,s.plan.p,false);cudaEvent_t consumed=nullptr;CU(cudaEventCreateWithFlags(&consumed,cudaEventDisableTiming));CU(cudaEventRecord(consumed,stream));
    const bool pending=bounded_wait([&]{return gate.entered.load();})&&cudaEventQuery(consumed)==cudaErrorNotReady;
    bool ok=false;std::atomic<bool>done{false};std::exception_ptr failure;
    std::thread retire([&]{try{ok=A.gpu_misses_retire_pins("fixture-gated-retirement",err);}catch(...){failure=std::current_exception();}done=true;});
    const bool entered=bounded_wait([&]{return api.entered_whole_stream_drain.load();});
    // Observe atomic API counters while the retirement thread is blocked;
    // inspect the mutable ledger only after joining that thread.
    const bool kept=entered&&!done.load()&&registration_api.unregistrations==0&&registration_api.releases==0;
    gate.release=true;retire.join();gate.stream=nullptr;if(failure)std::rethrow_exception(failure);
    CU(cudaEventSynchronize(consumed));CU(cudaEventDestroy(consumed));
    need(pending&&kept&&ok&&registration_api.releases==2&&!A.registered_copy.published&&A.registered_copy.held.empty(),"actual retirement unpinned before both-stream completion");
    exact_finite(reference,math.out.get(stream),"recovery gated public output");
    // Actual physical zero, then actual coordinator re-acquires the SAME ledger.
    A.gpu_misses_recovery_needed=true;registration_api.reset();auto result=recovery_numeric(E,s,domain,4,"reregister-same-ledger-after-drained-retirement",true);
    need(result.direct>0&&A.cache_epoch_machine.serial==2,"same-ledger re-registration epoch");
    need(A.gpu_misses_retire_pins("fixture-final-retire",err)&&registration_api.releases==2,"final physical zero");
    domain.audit();math.audit(stream);s.audit();return 2;
}

struct RecoverySuite {std::vector<RecoveryResult>cases;int guards=0;};
RecoverySuite run_recovery_suite(cudaStream_t stream){
    RecoverySuite out;
    for(const auto&family:families){
        Experts E(family,6,stream);Stager s(E,stream);RecoveryDomains domain(E,s);configure_recovery(s,domain);registration_api.reset();
        for(int T=1;T<=4;++T){auto x=recovery_numeric(E,s,domain,T,T%2?"recovered-all-missing":"recovered-mixed",T%2==0);x.family=family.name;out.cases.push_back(x);}
        const auto fp=s.actual.gpu_misses_pin_fingerprint();domain.rotate_first_into_spare();
        auto x=recovery_numeric(E,s,domain,4,"recovered-current-slot-rotation");x.family=family.name;out.cases.push_back(x);
        need(s.actual.gpu_misses_pin_fingerprint()==fp,"slot rotation changed physical retained ownership");
        std::string err;need(s.actual.gpu_misses_retire_pins("fixture-family-retire",err),"family physical retirement");
        need(registration_api.successes==registration_api.releases,"family registration owner leak");domain.audit();
    }
    {Experts E(families[2],6,stream);out.guards+=recovery_faults(E,stream);out.guards+=recovery_retirement_lifetime(E,stream);}
    return out;
}
