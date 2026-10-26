// No CUDA call until explicit --run. No model, weight file, network or service.
#include "numeric_support.cuh"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>

struct ApiTrace {
    int copies=0,blob_copies=0,event_records=0,event_syncs=0;
    int fail_copy=0,fail_record=0;size_t bytes=0;
    int waits=0,fail_wait=0,ready_records=0,copy_drains=0,compute_drains=0;bool ready_recorded=false;cudaEvent_t ready=nullptr;cudaStream_t copy_stream=nullptr,compute_stream=nullptr;
    std::atomic<bool> saw_pending_bounce{false};
    std::atomic<bool> entered_whole_stream_drain{false};
    void reset(){copies=blob_copies=event_records=event_syncs=fail_copy=fail_record=0;waits=fail_wait=ready_records=copy_drains=compute_drains=0;ready_recorded=false;bytes=0;saw_pending_bounce=false;entered_whole_stream_drain=false;}
}api;
cudaError_t fixture_memcpy(void* dst,const void* src,size_t bytes,cudaMemcpyKind kind,cudaStream_t stream){
    ++api.copies;if(bytes>256){++api.blob_copies;api.bytes+=bytes;}
    if(api.fail_copy&&api.copies==api.fail_copy)return cudaErrorInvalidValue;
    return cudaMemcpyAsync(dst,src,bytes,kind,stream);
}
cudaError_t fixture_event_record(cudaEvent_t event,cudaStream_t stream){++api.event_records;if(event==api.ready)++api.ready_records;if(api.fail_record&&api.event_records==api.fail_record)return cudaErrorInvalidValue;auto rc=cudaEventRecord(event,stream);if(event==api.ready&&rc==cudaSuccess)api.ready_recorded=true;return rc;}
cudaError_t fixture_event_sync(cudaEvent_t event){++api.event_syncs;if(cudaEventQuery(event)==cudaErrorNotReady)api.saw_pending_bounce=true;return cudaEventSynchronize(event);}
cudaError_t fixture_stream_sync(cudaStream_t stream){if(stream==api.copy_stream)++api.copy_drains;if(stream==api.compute_stream)++api.compute_drains;api.entered_whole_stream_drain=true;return cudaStreamSynchronize(stream);}
cudaError_t fixture_stream_wait(cudaStream_t stream,cudaEvent_t event,unsigned flags){++api.waits;if(api.fail_wait&&api.waits==api.fail_wait)return cudaErrorInvalidValue;if(event!=api.ready||!api.ready_recorded)return cudaErrorInvalidResourceHandle;return cudaStreamWaitEvent(stream,event,flags);}
#include "registration_trace.cuh"
#include "actual_stager.generated.cuh"

struct Stager {
    Actual actual;Layer layer;Dev device;
    Device<uint8_t>gpu;Device<unsigned long long>plan;
    uint8_t*host_owner[2]={};uint8_t*plan_owner=nullptr;size_t stride=0;
    explicit Stager(Experts&E,cudaStream_t s):gpu(stage_bytes(E.blob)),plan(MAXT*PICKS){
        need(gpumiss::sizes(E.blob,stride,actual.gpu_misses_bytes),"actual capacity policy");
        actual.gpu_misses_stride=stride;actual.gpu_misses_stage=gpu.p;
        CU(cudaStreamCreateWithFlags(&actual.gpu_misses_cs,cudaStreamNonBlocking));CU(cudaEventCreateWithFlags(&actual.gpu_misses_ready,cudaEventDisableTiming));
        for(int i=0;i<2;++i){CU(cudaHostAlloc(&host_owner[i],stride+2*GUARD,cudaHostAllocPortable));std::memset(host_owner[i],0xa5,stride+2*GUARD);actual.gpu_misses_host[i]=host_owner[i]+GUARD;CU(cudaEventCreateWithFlags(&actual.gpu_misses_done[i],cudaEventDisableTiming));}
        CU(cudaHostAlloc(&plan_owner,MAXT*PICKS*8+2*GUARD,cudaHostAllocPortable));std::memset(plan_owner,0xa5,MAXT*PICKS*8+2*GUARD);actual.gpu_misses_plan_pin=reinterpret_cast<unsigned long long*>(plan_owner+GUARD);
        layer.n_exp=E.ne;layer.n_slots=E.ne;layer.blob=E.blob;layer.lay=E.lay;layer.vram_base=E.resident->p;
        layer.slot.assign(E.ne,-1);layer.ram.resize(E.ne);layer.fixture_roles.resize(E.ne);
        for(int e=0;e<E.ne;++e){layer.ram[e]=e;for(int r=0;r<3;++r)layer.fixture_roles[e][r]=E.role(e,r);}
        device.st=s;device.plan_ptr=plan.p;actual.devs[0]=device;
        CU(cudaDeviceSynchronize()); // all constructor/default-stream initialization precedes actual calls
    }
    static size_t stage_bytes(size_t blob){size_t stride=0,bytes=0;need(gpumiss::sizes(blob,stride,bytes),"stage capacity");return bytes;}
    void trace(){api.ready=actual.gpu_misses_ready;api.copy_stream=actual.gpu_misses_cs;api.compute_stream=device.st;}
    void drain(){if(actual.gpu_misses_cs)CU(cudaStreamSynchronize(actual.gpu_misses_cs));CU(cudaStreamSynchronize(device.st));}
    ~Stager(){if(actual.gpu_misses_cs)cudaStreamSynchronize(actual.gpu_misses_cs);cudaStreamSynchronize(device.st);for(int i=0;i<2;++i){if(actual.gpu_misses_done[i])cudaEventDestroy(actual.gpu_misses_done[i]);if(host_owner[i])cudaFreeHost(host_owner[i]);}if(actual.gpu_misses_ready)cudaEventDestroy(actual.gpu_misses_ready);if(actual.gpu_misses_cs)cudaStreamDestroy(actual.gpu_misses_cs);if(plan_owner)cudaFreeHost(plan_owner);}
    void audit(){gpu.audit(device.st);plan.audit(device.st);for(int i=0;i<2;++i)for(size_t j=0;j<GUARD;++j)need(host_owner[i][j]==0xa5&&host_owner[i][GUARD+stride+j]==0xa5,"pinned bounce canary");for(size_t j=0;j<GUARD;++j)need(plan_owner[j]==0xa5&&plan_owner[GUARD+MAXT*PICKS*8+j]==0xa5,"plan pin canary");}
};

struct Gate {std::atomic<bool>entered{false},release{false};cudaStream_t stream=nullptr;explicit Gate(cudaStream_t s):stream(s){}~Gate(){release=true;if(stream)cudaStreamSynchronize(stream);}};
void CUDART_CB wait_gate(void* p){auto*g=static_cast<Gate*>(p);g->entered=true;while(!g->release.load())std::this_thread::yield();}
bool bounded_wait(const std::function<bool()>&fn,int ms=4000){auto end=std::chrono::steady_clock::now()+std::chrono::milliseconds(ms);while(!fn()){if(std::chrono::steady_clock::now()>=end)return false;std::this_thread::sleep_for(std::chrono::milliseconds(1));}return true;}

struct CaseResult {std::string family,kind;int T=0,unique=0,missing_entries=0;size_t H2D_bytes=0;uint64_t output_hash=0;};
std::vector<int> picks(int T,int ne,bool unique){std::vector<int>ids((size_t)MAXT*PICKS,-991);for(int t=0;t<T;++t)for(int i=0;i<K;++i)ids[(size_t)t*8+i]=unique?t*K+i:(i+t)%ne;return ids;}
CaseResult numeric(Experts&E,Stager&stage,int T,const char*kind,bool unique=false){
    auto ids=picks(T,E.ne,unique);const bool all_resident=std::string(kind)=="all-resident",mixed=std::string(kind)=="mixed";
    for(int e=0;e<E.ne;++e){stage.layer.slot[e]=(all_resident||(mixed&&e%2==0))?e:-1;stage.layer.ram[e]=stage.layer.slot[e]>=0?-1:e;}
    std::vector<unsigned long long>ref((size_t)T*8,0);
    for(int t=0;t<T;++t)for(int i=0;i<K;++i)ref[(size_t)t*8+i]=E.ptr(ids[(size_t)t*8+i]);
    MathBuffers math(T,stage.device.st);math.launch(E,ref,stage.device.st);
    const auto reference=math.out.get(stage.device.st),reference_h=math.h.get(stage.device.st),reference_part=math.part.get(stage.device.st);
    const auto source_hash=hash64(E.host.data(),E.host.size());
    gpumiss::Plan expected;need(gpumiss::plan(T,K,E.ne,ids.data(),stage.layer.slot.data(),stage.layer.ram.data(),expected)==gpumiss::Admission::ready,"independent fixture case not admitted");
    // Compute expected counts independently, not from the production Plan.
    std::vector<int>missing;int entries=0;
    for(int t=0;t<T;++t)for(int i=0;i<K;++i){const int e=ids[t*8+i];if(stage.layer.slot[e]>=0)continue;++entries;if(std::find(missing.begin(),missing.end(),e)==missing.end())missing.push_back(e);}
    need((int)missing.size()==expected.count&&entries==expected.entries,"independent expected dispatch/count");
    const unsigned long long plan_tail=0x1122334455667788ull;
    stage.plan.set(std::vector<unsigned long long>(MAXT*PICKS,plan_tail),stage.device.st);stage.drain();
    api.reset();stage.trace();bool used=false;std::string err;
    need(stage.actual.gpu_misses_prepare(stage.layer,stage.device,T,ids.data(),used,err)&&used&&!stage.actual.terminal,"actual prepare refused: "+err);
    // Consume the ACTUAL published device plan immediately. No readback or copy
    // drain may mask a missing copy-ready dependency in the candidate helper.
    math.launch(E,{},stage.device.st,stage.plan.p);const auto candidate=math.out.get(stage.device.st);
    need(api.waits==1&&api.ready_records==1&&api.ready_recorded,"actual ready record/wait ordering not exercised");
    auto got_plan=stage.plan.get(stage.device.st);for(size_t i=(size_t)T*8;i<got_plan.size();++i)need(got_plan[i]==plan_tail,"unused whole-window plan tail overwritten");got_plan.resize((size_t)T*8);
    need(api.blob_copies==(int)missing.size()&&api.bytes==missing.size()*E.blob,"actual DMA count/bytes differs from independent expected");
    for(int t=0;t<T;++t)for(int i=0;i<8;++i){const int at=t*8+i;if(i>=K){need(got_plan[at]==0,"physical unused pick changed");continue;}const int e=ids[at];auto it=std::find(missing.begin(),missing.end(),e);const auto want=it==missing.end()?E.ptr(e):reinterpret_cast<unsigned long long>(stage.gpu.p+(size_t)(it-missing.begin())*stage.stride);need(got_plan[at]==want,"published original-pick plan pointer");}
    // Every used GPU blob byte, including inter-role zero padding, is checked.
    std::vector<uint8_t>readback(E.blob),want(E.blob);
    for(size_t j=0;j<missing.size();++j){std::fill(want.begin(),want.end(),0);for(int r=0;r<3;++r)std::memcpy(want.data()+E.lay.trellis[r],E.role(missing[j],r),E.bytes[r]);CU(cudaMemcpyAsync(readback.data(),stage.gpu.p+j*stage.stride,E.blob,cudaMemcpyDeviceToHost,stage.device.st));stage.drain();need(readback==want,"actual staged raw-role/gap bytes differ");}
    exact_finite(reference,candidate,"whole public expert output");exact_finite(reference_h,math.h.get(stage.device.st),"public gate/up output");if(T>1)exact_finite(reference_part,math.part.get(stage.device.st),"public down partial output");
    math.audit(stage.device.st);stage.audit();E.resident->audit(stage.device.st);need(source_hash==hash64(E.host.data(),E.host.size()),"synthetic source bytes changed");
    return {"",kind,T,(int)missing.size(),entries,api.bytes,hash64(candidate.data(),candidate.size()*4)};
}

int lifecycle(Experts&E,Stager&stage){
    int checks=0;auto ids=picks(4,E.ne,false);for(int e=0;e<E.ne;++e){stage.layer.slot[e]=-1;stage.layer.ram[e]=e;}
    auto refuse=[&](int T,bool hard){api.reset();stage.trace();stage.actual.terminal=false;bool used=true;std::string err;const bool ok=stage.actual.gpu_misses_prepare(stage.layer,stage.device,T,ids.data(),used,err);need(!used&&ok==!hard&&stage.actual.terminal==hard&&api.copies==0&&api.event_records==0,"guard attempted DMA or wrong terminal contract");++checks;};
    stage.actual.gpu_misses_enabled=false;refuse(4,false);stage.actual.gpu_misses_enabled=true;
    refuse(0,false);refuse(5,false);stage.actual.copy_mode=false;refuse(4,false);stage.actual.copy_mode=true;
    stage.layer.ram[0]=-1;refuse(4,false);stage.layer.ram[0]=0;
    ids[0]=E.ne;refuse(4,true);ids[0]=0;stage.actual.terminal=false;
    auto stride=stage.actual.gpu_misses_stride;stage.actual.gpu_misses_stride=E.blob-1;refuse(4,false);stage.actual.gpu_misses_stride=stride;
    const auto* saved=stage.layer.fixture_roles[0][0];stage.layer.fixture_roles[0][0]=nullptr;refuse(4,true);stage.layer.fixture_roles[0][0]=saved;stage.actual.terminal=false;
    stage.layer.k_exp=7;refuse(4,false);stage.layer.k_exp=K;
    stage.layer.slot[0]=0;const int slots=stage.layer.n_slots;stage.layer.n_slots=0;refuse(4,true);stage.layer.n_slots=slots;stage.actual.terminal=false;
    auto base=stage.layer.vram_base;stage.layer.vram_base=nullptr;refuse(4,true);stage.layer.vram_base=base;stage.layer.slot[0]=-1;stage.actual.terminal=false;
    const auto offset=stage.layer.lay.trellis[1];stage.layer.lay.trellis[1]=stage.layer.lay.trellis[0];refuse(4,true);stage.layer.lay.trellis[1]=offset;stage.actual.terminal=false;
    // One missing expert avoids bounce backpressure, so prepare can publish a
    // real pending ready-event dependency while copies are intentionally held.
    // Expert consumers use the actual device plan without any host-side drain.
    {
        Stager dep(E,stage.device.st);MathBuffers math(4,stage.device.st);std::vector<unsigned long long>ref(4*PICKS,0);
        for(int e=0;e<E.ne;++e){dep.layer.slot[e]=e;dep.layer.ram[e]=-1;}dep.layer.slot[0]=-1;dep.layer.ram[0]=0;
        for(int t=0;t<4;++t)for(int i=0;i<K;++i)ref[t*8+i]=E.ptr(ids[t*8+i]);
        math.launch(E,ref,stage.device.st);const auto reference=math.out.get(stage.device.st);
        CU(cudaMemsetAsync(dep.gpu.p,0xff,dep.gpu.count,dep.device.st));CU(cudaStreamSynchronize(dep.device.st));
        Gate held(dep.actual.gpu_misses_cs);CU(cudaLaunchHostFunc(dep.actual.gpu_misses_cs,wait_gate,&held));api.reset();dep.trace();
        bool u=false;std::string err;need(dep.actual.gpu_misses_prepare(dep.layer,dep.device,4,ids.data(),u,err)&&u,"pending-ready prepare failed");
        math.launch(E,{},dep.device.st,dep.plan.p,false);cudaEvent_t consumed=nullptr;CU(cudaEventCreateWithFlags(&consumed,cudaEventDisableTiming));CU(cudaEventRecord(consumed,dep.device.st));
        const bool pending=bounded_wait([&]{return held.entered.load();})&&cudaEventQuery(consumed)==cudaErrorNotReady&&cudaEventQuery(dep.actual.gpu_misses_ready)==cudaErrorNotReady;
        held.release=true;CU(cudaEventSynchronize(consumed));CU(cudaEventDestroy(consumed));
        need(pending&&api.blob_copies==1&&api.waits==1&&api.ready_recorded,"actual ready-event did not block the public expert consumers");
        exact_finite(reference,math.out.get(dep.device.st),"pending-ready public expert output");math.audit(dep.device.st);dep.audit();++checks;
    }
    // A real stream gate keeps the first H2D events incomplete. The actual third
    // bounce reuse must synchronize an event observed not-ready, not a fake flag.
    stage.drain();for(bool&b:stage.actual.gpu_misses_busy)b=false;Gate gate(stage.actual.gpu_misses_cs);CU(cudaLaunchHostFunc(stage.actual.gpu_misses_cs,wait_gate,&gate));
    bool result=false,used=false;std::string error;std::exception_ptr failure;api.reset();stage.trace();
    std::thread producer([&]{try{result=stage.actual.gpu_misses_prepare(stage.layer,stage.device,4,ids.data(),used,error);}catch(...){failure=std::current_exception();}});
    const bool entered=bounded_wait([&]{return gate.entered.load();});const bool pending=entered&&bounded_wait([&]{return api.saw_pending_bounce.load();});gate.release=true;producer.join();stage.drain();if(failure)std::rethrow_exception(failure);
    need(entered&&pending&&result&&used&&!stage.actual.terminal,"actual two-bounce completion backpressure not exercised");++checks;stage.audit();
    // Both failure modes happen after a real earlier transfer has been queued.
    // No fallback/compute is attempted; an explicit full-stream drain is required
    // even if EventRecord failed before its busy flag could be published.
    for(int mode=0;mode<2;++mode){
        Stager fail(E,stage.device.st);Gate blocked(fail.actual.gpu_misses_cs);cudaEvent_t complete=nullptr;CU(cudaEventCreateWithFlags(&complete,cudaEventDisableTiming));
        CU(cudaLaunchHostFunc(fail.actual.gpu_misses_cs,wait_gate,&blocked));api.reset();fail.trace();if(mode==0)api.fail_copy=2;else api.fail_record=1;
        bool u=true;std::string err;const bool ok=fail.actual.gpu_misses_prepare(fail.layer,fail.device,4,ids.data(),u,err);
        CU(cudaEventRecord(complete,fail.actual.gpu_misses_cs));
        const bool copy_pending=bounded_wait([&]{return blocked.entered.load();})&&cudaEventQuery(complete)==cudaErrorNotReady;
        bool retired=false;std::string why;std::atomic<bool>finished{false};std::exception_ptr drain_failure;
        std::thread retire([&]{try{retired=fail.actual.drain_gpu_misses(why);}catch(...){drain_failure=std::current_exception();}finished=true;});
        const bool drain_entered=bounded_wait([&]{return api.entered_whole_stream_drain.load();});const bool blocked_until_release=drain_entered&&!finished.load();
        blocked.release=true;retire.join();blocked.stream=nullptr;if(drain_failure)std::rethrow_exception(drain_failure);
        need(!ok&&!u&&fail.actual.terminal&&api.blob_copies>=1,"late injected failure did not terminate");
        need(copy_pending&&blocked_until_release&&retired&&api.copy_drains==1&&api.compute_drains==1&&cudaEventQuery(complete)==cudaSuccess,"actual retirement did not drain both pending-copy and compute streams");
        need(!fail.actual.gpu_misses_done[0]&&!fail.actual.gpu_misses_done[1]&&!fail.actual.gpu_misses_busy[0]&&!fail.actual.gpu_misses_busy[1],"actual drain did not retire events/clear busy");
        need(!fail.actual.gpu_misses_cs&&!fail.actual.gpu_misses_ready,"actual drain did not retire private stream/ready event");
        std::string again;need(fail.actual.drain_gpu_misses(again),"actual second retirement not idempotent");
        CU(cudaEventDestroy(complete));fail.audit();++checks;
    }
    // Ready publication and compute-wait failures occur after real blob H2Ds.
    // No original fallback or expert compute may be attempted after either.
    for(int mode=0;mode<2;++mode){Stager fail(E,stage.device.st);api.reset();fail.trace();if(mode==0)api.fail_record=7;else api.fail_wait=1;bool u=true;std::string err;const bool ok=fail.actual.gpu_misses_prepare(fail.layer,fail.device,4,ids.data(),u,err);need(!ok&&!u&&fail.actual.terminal&&api.blob_copies==6,"ready/wait failure did not terminate after real transfers");need(mode==0?api.waits==0:api.waits==1,"fallback or wait attempted after failed ready record");std::string why;need(fail.actual.drain_gpu_misses(why)&&api.copy_drains==1&&api.compute_drains==1,"ready/wait failure did not drain both streams");need(!fail.actual.gpu_misses_cs&&!fail.actual.gpu_misses_ready,"ready/wait failure owners not retired");fail.audit();++checks;}
    stage.actual.terminal=false;api.reset();return checks;
}

#include "direct_fixture.cuh"
#include "recovery_fixture.cuh"
#include "reserve_fixture.cuh"

int main(int argc,char**argv){
    bool run=false;std::string output;
    for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--run")run=true;else if(a=="--output"&&i+1<argc)output=argv[++i];else {std::fprintf(stderr,"unknown/incomplete argument\n");return 2;}}
    if(!run){std::puts("{\"prepared_only\":true,\"GPU_calls\":0,\"model_payloads_read\":0}");return 0;}
    if(output.empty()||std::filesystem::exists(output)){std::fprintf(stderr,"exclusive --output required\n");return 2;}
    std::ofstream f(output,std::ios::out);if(!f)return 2;std::vector<CaseResult>cases;std::vector<DirectResult>direct_cases;int life=0,direct_life=0;RecoverySuite recovery;ReserveSuite reserve;cudaStream_t stream=nullptr;
    try{
        CU(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        for(const auto&family:families){Experts E(family,6,stream);Stager stage(E,stream);for(int T=1;T<=4;++T){auto c=numeric(E,stage,T,T%2?"all-missing":"mixed");c.family=family.name;cases.push_back(c);}auto c=numeric(E,stage,4,"all-resident");c.family=family.name;cases.push_back(c);if(family.gate==18&&family.down==18)life+=lifecycle(E,stage);stage.drain();}
        {Experts E(families[6],24,stream);Stager stage(E,stream);auto c=numeric(E,stage,4,"all-24-unique",true);need(c.unique==24&&c.missing_entries==24,"maximum exact whole-window capacity not exercised");c.family=families[6].name;cases.push_back(c);stage.drain();}
        for(const auto&family:families){
            Experts E(family,6,stream);Stager stage(E,stream);RegisteredDomain domain(E,stage);domain.register_prefix();
            if(family.gate==18&&family.down==18)direct_life+=direct_guards(E,stage,domain);
            for(int T=1;T<=4;++T){auto c=direct_numeric(E,stage,domain,T,"registered-whole-slot",T%2==0);c.family=family.name;direct_cases.push_back(c);}
            domain.rotate_first_into_spare();auto rotated=direct_numeric(E,stage,domain,4,"current-slot-rotation");rotated.family=family.name;direct_cases.push_back(rotated);
            domain.retire();domain.register_prefix(((3*E.blob+domain.page-1)/domain.page)*domain.page);
            auto hybrid=direct_numeric(E,stage,domain,4,"covered-and-uncovered-mixed-resident",true);need(hybrid.direct>0&&hybrid.bounce>0,"real direct/bounce hybrid coverage not exercised");hybrid.family=family.name;direct_cases.push_back(hybrid);
            if(family.gate==18&&family.down==18){
                domain.retire();domain.register_prefix(((E.blob-1)/domain.page)*domain.page);
                auto partial=direct_numeric(E,stage,domain,4,"partial-slot-coverage-bounces");need(partial.direct==0&&partial.bounce>0,"partial span was admitted direct");partial.family=family.name;direct_cases.push_back(partial);
                domain.retire();
                {Stager alternate(E,stream);RegisteredDomain padded(E,alternate,256);padded.register_prefix();
                 auto mismatch=direct_numeric(E,alternate,padded,4,"cpu-gpu-offset-mismatch-bounces",false,true);need(mismatch.direct==0&&mismatch.bounce>0,"offset mismatch was admitted direct");mismatch.family=family.name;direct_cases.push_back(mismatch);padded.retire();}
            }
            domain.retire();need(domain.registers==domain.unregisters,"real owned registration not retired exactly once");domain.audit();
        }
        {Experts E(families[6],24,stream);Stager stage(E,stream);RegisteredDomain domain(E,stage);domain.register_prefix();auto c=direct_numeric(E,stage,domain,4,"registered-24-unique",false,false,true);need(c.direct==24&&c.bounce==0,"maximum24 direct slots not exercised");c.family=families[6].name;direct_cases.push_back(c);domain.retire();}
        {Experts E(families[2],6,stream);direct_life+=direct_ready_lifetime(E,stream);}
        recovery=run_recovery_suite(stream);
        reserve=run_reserve_suite(stream);
        CU(cudaStreamSynchronize(stream));CU(cudaStreamDestroy(stream));stream=nullptr;
        f<<"{\"passed\":true,\"finished\":true,\"actual_helper_extracted\":true,\"format_families\":8,\"case_count\":"<<cases.size()<<",\"lifecycle_guard_checks\":"<<life<<",\"precision_contract\":\"Full resident-GPU versus actual-staged-GPU output bits; not CPU-lane bit identity or independent new math proof\",\"model_or_API_calls\":0,\"performance_claim\":false,\"cases\":[";
        for(size_t i=0;i<cases.size();++i){auto&c=cases[i];if(i)f<<',';f<<"{\"family\":\""<<c.family<<"\",\"kind\":\""<<c.kind<<"\",\"T\":"<<c.T<<",\"unique_misses\":"<<c.unique<<",\"missing_entries\":"<<c.missing_entries<<",\"H2D_blob_bytes\":"<<c.H2D_bytes<<",\"output_FNV64\":"<<c.output_hash<<",\"all_bits_finite_guards_inputs\":true}";}f<<"],\"direct_case_count\":"<<direct_cases.size()<<",\"direct_guard_lifetime_checks\":"<<direct_life<<",\"registration\":\"real cudaHostRegister-owned page-aligned VirtualAlloc domains, less than1GiB simultaneously\",\"retention_scope\":\"actual retained-ready/current-source/ledger and two-stream drain; full Model epoch, phase retention, Stop and rearm remain separate loaded-model gates\",\"direct_cases\":[";
        for(size_t i=0;i<direct_cases.size();++i){const auto&c=direct_cases[i];if(i)f<<',';f<<"{\"family\":\""<<c.family<<"\",\"kind\":\""<<c.kind<<"\",\"T\":"<<c.T<<",\"direct_unique\":"<<c.direct<<",\"bounce_unique\":"<<c.bounce<<",\"missing_entries\":"<<c.missing_entries<<",\"registered_bytes\":"<<c.registered_bytes<<",\"output_FNV64\":"<<c.output_hash<<",\"all_bits_finite_guards_inputs\":true}";}
        f<<"],\"recovery_case_count\":"<<recovery.cases.size()<<",\"recovery_guard_lifetime_checks\":"<<recovery.guards<<",\"recovery_scope\":\"Actual coordinator/core/retire/policy with REAL CUDA registration, streams, same Ledger and public GG math; epoch and47-buffer compaction adapters are synthetic, full loaded Model gates remain separate\",\"recovery_cases\":[";
        for(size_t i=0;i<recovery.cases.size();++i){const auto&c=recovery.cases[i];if(i)f<<',';f<<"{\"family\":\""<<c.family<<"\",\"kind\":\""<<c.kind<<"\",\"T\":"<<c.T<<",\"direct_unique\":"<<c.direct<<",\"registered_bytes\":"<<c.registered_bytes<<",\"output_FNV64\":"<<c.output_hash<<",\"all_bits_finite_guards_inputs\":true}";}
        f<<"],\"reserve_case_count\":"<<reserve.cases.size()<<",\"reserve_guard_lifetime_checks\":"<<reserve.guards<<",\"reserve_scope\":\"Actual reserve/coordinator/registration and pure copiedshrink target plan; actual GPU pool transaction/inventory adapters remain synthetic, loaded Model gates required\",\"reserve_cases\":[";
        for(size_t i=0;i<reserve.cases.size();++i){const auto&c=reserve.cases[i];if(i)f<<',';f<<"{\"family\":\""<<c.family<<"\",\"kind\":\""<<c.kind<<"\",\"T\":"<<c.T<<",\"direct_unique\":"<<c.direct<<",\"registered_bytes\":"<<c.registered_bytes<<",\"output_FNV64\":"<<c.output_hash<<",\"all_bits_finite_guards_inputs\":true}";}
        f<<"]}\n";f.flush();std::printf("PASS %zu+%zu+%zu inherited and%zu reserve numeric cases; %d+%d+%d inherited and%d reserve guards\n",cases.size(),direct_cases.size(),recovery.cases.size(),reserve.cases.size(),life,direct_life,recovery.guards,reserve.guards);return 0;
    }catch(const std::exception&e){if(stream)cudaStreamSynchronize(stream);f<<"{\"passed\":false,\"finished\":true,\"completed_cases\":"<<cases.size()<<",\"error\":\"fixture failed; see stderr\"}\n";f.flush();std::fprintf(stderr,"FAIL: %s\n",e.what());if(stream)cudaStreamDestroy(stream);return 1;}
}
