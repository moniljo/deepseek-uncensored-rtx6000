// Actual v6b policy/coordinator/registration, synthetic GPU-pool transaction.
// No Model owner/worker proof is inferred from these adapters.
#pragma once
struct ReserveSuite {std::vector<RecoveryResult>cases;int guards=0;};
ReserveSuite run_reserve_suite(cudaStream_t stream){
    ReserveSuite result;constexpr size_t G=size_t(1)<<30,R=size_t(512)<<20;
    const bool policy[]={
        gpumiss::recovery_registration_reserve==R,
        gpumiss::recovery_pool_cap(0)==0,
        gpumiss::recovery_pool_cap(R-1)==0,
        gpumiss::recovery_pool_cap(R)==0,
        gpumiss::recovery_pool_cap(R+1)==1,
        gpumiss::recovery_pool_cap(2*G)==2*G-R,
        gpumiss::recovery_pool_cap(~size_t(0))==~size_t(0)-R,
        gpumiss::recovery_pool_needs_shrink(0,0),
        gpumiss::recovery_pool_needs_shrink(2*G,0),
        !gpumiss::recovery_pool_needs_shrink(G,2*G),
        !gpumiss::recovery_pool_needs_shrink(2*G-R,2*G),
        gpumiss::recovery_pool_needs_shrink(2*G-R+1,2*G),
        gpumiss::recovery_pool_budget(0,0,0)==0,
        gpumiss::recovery_pool_budget((double)(R-1),0,0)==0,
        gpumiss::recovery_pool_budget((double)R,0,0)==0,
        gpumiss::recovery_pool_budget((double)(R+1),0,0)==1,
        gpumiss::recovery_pool_budget((double)(3*G),G,2*G)==double(2*G-R),
        gpumiss::recovery_pool_budget((double)G,G,2*G)==double(G),
        gpumiss::recovery_pool_budget((double)(3*G),2*G,2*G)==double(2*G),
        gpumiss::recovery_pool_budget((double)(3*G),G,0)==double(3*G-R)
    };
    for(bool ok:policy){need(ok,"actual reserve policy boundary");++result.guards;}
    Experts E(families[2],6,stream);
    // Below cap, already-large known pool, and unknown first-phase cancellation.
    for(int mode=0;mode<3;++mode){
        Stager s(E,stream);RecoveryDomains d(E,s);configure_recovery(s,d,mode==1);auto&A=s.actual;
        A.devs[0].experts=mode==0?G:2*G;if(mode==2)A.gpu_misses_last_decode_pool=0;
        const size_t normal=A.gpu_misses_last_decode_pool;registration_api.reset();std::string err;
        need(A.gpu_misses_recover_decode(17,err)&&A.gpu_misses_retained_ready(),"actual reserved recovery: "+err);
        need(A.fixture_shrink_calls==(mode==0?0:1)&&A.gpu_misses_last_decode_pool==normal,"reserve shrink admission or normal target changed");
        const size_t expected=normal?normal-R:3*G-R;
        need(A.devs[0].experts==expected&&registration_api.successes==2,"reserve did not constrain subsequent grow/register");
        if(mode==1){auto c=recovery_numeric(E,s,d,4,"reserved-large-pool-direct-consumption",true);c.family=families[2].name;result.cases.push_back(c);}
        need(A.gpu_misses_retire_pins("fixture-reserve-retire",err)&&registration_api.releases==2,"reserve test physical retirement");++result.guards;
    }
    // Coherent partial-above-cap, unknown-target partial and both Stop origins.
    for(int mode=0;mode<4;++mode){
        Stager s(E,stream);RecoveryDomains d(E,s);configure_recovery(s,d,true);auto&A=s.actual;
        A.devs[0].experts=2*G;if(mode%2)A.gpu_misses_last_decode_pool=0;
        A.fixture_shrink_pool_after=2*G;
        if(mode==1)A.fixture_unknown_targets_attained=false;
        if(mode>=2)A.fixture_shrink_stop=true;
        registration_api.reset();std::string err;const bool ok=A.gpu_misses_recover_decode(17,err);
        need(ok==(mode<2)&&A.halted==(mode>=2)&&!A.terminal,"partial/Stop reserve status");
        need(A.fixture_shrink_calls==1&&A.fixture_compactions==0&&registration_api.registrations==0&&
             A.registered_copy.held.empty()&&!A.registered_copy.published&&!A.gpu_misses_decode_pins&&
             A.gpu_misses_recovery_needed&&A.cache_epoch_machine.phase==epochpolicy::Phase::idle,"partial/Stop reserve published or leaked lease");++result.guards;
    }
    // Hard shrink errors are terminal. A benign shrink refusal stays usable.
    for(int mode=0;mode<2;++mode){
        Stager s(E,stream);RecoveryDomains d(E,s);configure_recovery(s,d,true);auto&A=s.actual;A.devs[0].experts=2*G;
        if(mode==0)A.fixture_shrink_hard=true;else A.fixture_shrink_refuse=true;
        registration_api.reset();std::string err;const bool ok=A.gpu_misses_recover_decode(17,err);
        need(ok==(mode==1)&&A.terminal==(mode==0)&&registration_api.registrations==0&&A.registered_copy.held.empty(),"shrink refusal/error admission");++result.guards;
    }
    // Repeated early cancellations must not subtract another512MiB from a
    // prior recovery pool. Unknown remains unknown until normal phase success.
    for(int unknown=0;unknown<2;++unknown){
        Stager s(E,stream);RecoveryDomains d(E,s);configure_recovery(s,d,true);auto&A=s.actual;
        A.devs[0].experts=2*G;if(unknown)A.gpu_misses_last_decode_pool=0;
        const size_t normal=A.gpu_misses_last_decode_pool;registration_api.reset();std::string err;
        for(int cycle=0;cycle<3;++cycle){
            A.gpu_misses_recovery_needed=true;
            need(A.gpu_misses_recover_decode(17,err)&&A.gpu_misses_retained_ready(),"repeated reserved recovery");
            need(A.gpu_misses_last_decode_pool==normal&&A.devs[0].experts==(normal?normal-R:3*G-R),"repeated cancellation ratcheted pool");
            need(A.gpu_misses_retire_pins("fixture-repeat-reserve-retire",err),"repeated reserve physical retirement");
        }
        need(registration_api.successes==6&&registration_api.releases==6,"repeated same-ledger owner leak");++result.guards;
    }
    {
        Stager s(E,stream);RecoveryDomains d(E,s);configure_recovery(s,d);auto&A=s.actual;
        A.gpu_misses_last_decode_pool=0;A.fixture_inventory_error=true;registration_api.reset();std::string err;
        need(!A.gpu_misses_recover_decode(17,err)&&A.terminal&&registration_api.registrations==0&&A.registered_copy.held.empty(),"unknown inventory error not terminal");++result.guards;
    }
    return result;
}
