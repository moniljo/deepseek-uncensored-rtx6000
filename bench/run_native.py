"""Two fresh frozen1M waves using canonical timing/observation/graders.

Default dry. One owned process, all original13 scores retained. Corrected24
primary quality and original12 paired throughput are separate from rawstatus.
"""
from __future__ import annotations
import argparse
import copy
import json
import re
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from unittest import mock
import native_support as B
import prepare_inputs as I
import runtime as RT
import qualification_support as Q
import qualification_contract as C
import baseline_validation as BASE
fresh_seed,wave_metrics,grade_row,validate_cleanup=C.fresh_seed,C.wave_metrics,C.grade_row,C.validate_cleanup
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
need=B.need


def selected_generate(observer,*args):
    row=observer.generate(*args)
    g=row['native_GPU_miss_answer'];d=row['native_GPU_miss_direct_answer']
    need(d['retained_ranges']==40 and d['retained_bytes']==184926136320 and
         d['direct_unique']>0 and d['direct_bytes']>0 and d['bounce_unique']==d['bounce_bytes']==0 and
         g['fallbacks']==0,'Selected exactMix direct192 graded answer requires40ranges/directpositive/zeroBounce/zeroCPUfallback')
    return row


def run(a):
    need(sys.platform=='win32'and a.config and a.build_receipt,'Explicit native config/BUILD required')
    paired=bool(a.baseline_report is not None or a.expected_baseline_report_sha256 is not None)
    need(not a.require_paired or paired,'Explicit paired qualification requires the closed stock REPORT/SHA')
    stock=None;stock_metrics=None
    if paired:
        need(a.baseline_report is not None and re.fullmatch('[0-9a-f]{64}',a.expected_baseline_report_sha256 or'')and B.sha(a.baseline_report)==a.expected_baseline_report_sha256,'Explicit stock REPORT byte pin differs')
        stock=B.read(a.baseline_report);stock_metrics=BASE.report_check(stock,False)
    out=a.output.resolve();need(out.is_relative_to((ROOT/'.work').resolve())and not out.exists(),'Exclusive output under ignored .work required')
    import psutil
    busy=[p.pid for p in psutil.process_iter(['name'])if p.info['name']and (p.info['name'].lower().startswith(('strata','ds41_generate','ds4_generate'))or p.info['name'].lower()=='llama-server.exe')]
    need(not busy,'Stop competing model servers explicitly; this benchmark never stops them')
    prepared,prefix=RT.ready_inputs(a.inputs);cfg=B.read(a.config);RT.verify_prepared_tokenizer(cfg,prepared)
    protected=RT.validate_build(cfg,a.build_receipt)
    server,ST=RT.load_frontend(a);manifest=I.frozen_manifest()
    encode,decode,kind=I.reference_encoder(a,ST,manifest)
    sys.path.insert(0,str(HERE/'oracles'))
    original=B.load(HERE/'oracles/accuracy_cases.py','accuracy_cases')
    revised=B.load(HERE/'oracles/accuracy_cases_v2.py','portable_fullQ_v2')
    tests=[(entry['case'],prefix+row['suffix_ids'],False)for entry,row in zip(manifest['cases'],prepared['inputs'][:12])]
    v2=next(c for c in revised.CASES if c['id']=='code-account-balance');tests.append((v2,prefix+prepared['inputs'][12]['suffix_ids'],True))
    parity=[{'case':r['name'],'tokens':r['prompt_tokens'],'input_ids_sha256':r['input_ids_sha256'],'equal':True}for r in prepared['inputs']]
    baseline_rows={r['name']:r for r in prepared['inputs']}
    baseline={'ttft_seconds':342.46961880000026,'weighted_ID_intervals_s':32.60134856572709,
        'primary_REPORT_sha256':'e6ccf0e2dc4b09424c77f23adc92d08740734ed00be840abb2175f5b5d8a1d17',
        'primary_provenance':'Original retained primary report digest; questions/hashes packaged here, raw private report not redistributed',
        'quality_note':'Original incumbent11/12+v2; raworiginal13 stillgraded in bothwaves; predeclaredprimaryfirst11+existingv2 separatelyassessed'}
    interlude_ids=prepared['interlude_ids'];interlude_case=next(c for c in original.CASES if c['id']=='filtered-decimal-revenue')
    need(B.idsha(prefix)==manifest['prefix_sha256']and len(prefix)==999764,'Exact frozen seed differs')
    if a.initial_profile is None:a.initial_profile=a.inputs/'initial-routing-profile.bin'
    out.mkdir(parents=True,exist_ok=False);cfg,profile=RT.make_private_config(cfg,out,a.initial_profile)
    source_files=[HERE/name for name in ('run_native.py','runtime.py','prepare_inputs.py','native_support.py','job_support.py','identity_support.py','owned_windows.py',
        'gpu_miss_validation.py','direct_pin_validation.py','native_stat_tap.py','phase_support.py','initial_phase.py','qualification_support.py','qualification_contract.py',
        'shutdown_trace.py','grade_first_support.py','strict_json_support.py','baseline_validation.py','frozen-inputs.json','ORACLE-PROVENANCE.json','SOURCE-PROVENANCE.json')]
    source_files +=list((HERE/'oracles').glob('*.py'))+list((HERE/'vendor').glob('*'))
    source_files +=[a.config,a.initial_profile,a.inputs/'INPUTS.json',a.inputs/'prefix.u32le',ROOT/'pins/runtime-overlays.json']
    if paired:source_files.append(a.baseline_report)
    runtime_map=B.read(ROOT/'pins/runtime-overlays.json')['files']
    source_files +=[a.strata_source/name for name in runtime_map]+[ROOT/'source/runtime'/name for name in runtime_map]+[a.strata_source/'tools/strata_tokenizer.py']
    token_dir=Path(cfg['tokenizer']);token_dir=token_dir if token_dir.is_absolute()else Path(cfg.get('cwd',a.config.parent))/token_dir
    source_files +=[token_dir/name for name in ('vocab.json','merges.txt','token_type.json','tokenizer.json')]
    protected.update({str(p):B.sha(p)for p in source_files})
    report={'schema':'retained-direct-two-fresh-frozen-waves-v1','passed':False,'status':'preparing','waves':[],'interlude':None,
        'configured_context':1048576,'wave_count':2,'same_owned_process_required':True,'no_retries':True,'no_RESET_or_maintenance_wire':True,'frozen_grading_changed':False,
        'registered_ram_cap_MiB':196608,'overlap_mode':False,'expert_profile':profile,'baseline':baseline,'tokenizer_parity':parity,
        'reference_tokenizer':kind,'exact_seed_tokens':999764,'exact_seed_ids_sha256':B.idsha(prefix),'protected_files_before':protected,
        'interlude_input':{'case':'filtered-decimal-revenue','tokens':164,'input_ids_sha256':B.idsha(interlude_ids),'tokenizer_equal':True,'prefix_pin':0},
        'public_source_and_fresh_BUILD_provenance':{'BUILD_sha256':B.sha(a.build_receipt),'source_overlay_manifest_sha256':B.sha(ROOT/'pins/source-overlays.json'),
            'source_locks_sha256':B.sha(ROOT/'pins/source-locks.json'),'private_ancestor_packet_not_required_or_implied':True},
        'paired_stock_REPORT_sha256':a.expected_baseline_report_sha256,'paired_comparison_applicable':paired,
        'paired_comparison_scope':'Exact same-machine historical stock81 control, including unchanged model/native path namespace'if paired else'Unpaired standalone measurement; no >=5percent or speed-improvement claim',
        'primary_corrected24_policy_evaluated_separately':True,
        'raw_original_protocol_preserved':True,'GEN_timing':'Exact canonical pipe submission tofirstactualID; nativeautorearm insideGEN charged; loading/inputprep excluded',
        'rate':'Eachwave ORIGINAL12 sum(actualIDs-1)/sum(last-first actualIDinterval); v2excluded; reasoning/control/EOSincluded',
        'complete_inputs_prepared_once_before_model':True,'no_precision_or_CPU_GPU_bitidentity_claim':True}
    owner=None;observer=None;rc=1;path=out/'REPORT.json'
    def save():
        B.save(path,report)
        for w in report['waves']:B.save(out/('WAVE-'+str(w['index'])+'.json'),w)
    save()
    try:
        import job_support,identity_support
        owned_args=SimpleNamespace(**{**vars(a),'dspark':'on','prefill_chunk':'65280','seed_only':True,'probe_prefix_tokens':None,
            'seed_deadline':2*a.seed_deadline+26*a.case_deadline+a.interlude_deadline})
        with mock.patch.object(server,'StrataEngine',RT.Tap.capture_class(server.StrataEngine)):
            owner=B.OwnedNative(owned_args,cfg,out,server,job_support.OwnedJob,identity_support)
        observer=Q.Observer(owner,out,196608,None)
        need(len(owner.processes)==len(owner.identities)==1,'One fresh owned process required')
        proc,identity=owner.processes[0],owner.identities[0]
        def same():need(owner.engine.proc is proc and proc.pid==identity['pid']and proc.poll()is None,'Owned engine changed/exited; no retry')
        report.update(engine_info=dict(owner.engine.info),owned_identity=identity,effective_native_args=owner.effective_args,effective_native_env=owner.effective_env,
            resource_sampler_request_budget_seconds=owned_args.seed_deadline,status='running');save()
        if paired:BASE.contrast(stock,report)
        for index in (1,2):
            same();wave={"index":index,"rows":[],"accuracy":[],"status":"seed-pending","measurement_complete":False,
                         "same_owned_identity":identity,"input_parity":parity,"no_new_profile_or_engine":True}
            report["waves"].append(wave);report["current_stage"]="wave"+str(index)+"-fresh-prefix";owner.stage=report["current_stage"];save()
            # None here preserves a returned first DONE for audit before the
            # explicit zero-reuse gate; generation/sampling/timers are unchanged.
            engine_log=out/"ENGINE.log";log_start=engine_log.stat().st_size
            wave["seed_log_window_start_offset"]=log_start;save()
            seed=observer.generate(owner.measurement,report["current_stage"],prefix,1,a.seed_deadline,None)
            seed["measurement_stage"]=seed["name"];seed["name"]="fresh-prefix-initialization";wave["rows"].append(seed);save()
            log_end=engine_log.stat().st_size;wave["seed_log_window_end_offset"]=log_end
            need(log_start<=log_end and log_end-log_start<=64<<20,"Invalid/excessive exact seed log window")
            with engine_log.open("rb")as stream:stream.seek(log_start);window=stream.read(log_end-log_start)
            window_file=out/("SEED-"+str(index)+"-ENGINE-window.log");window_file.write_bytes(window)
            wave["seed_log_window_file"]=str(window_file);wave["seed_log_window_sha256"]=B.sha(window_file)
            previous=report["waves"][0].get("native_phase_evidence")if index==2 else None
            wave["native_phase_evidence"]=Q.seed_telemetry(window.decode("utf-8","replace"),index,previous);save()
            fresh_seed(seed,report["exact_seed_ids_sha256"])
            need(seed["prompt_read"]==B.PREFIX_N and seed["emitted_token_ids"]==seed["native_generated_tokens"]==1 and seed["requested_prefix_tokens"]==B.PREFIX_N,"Exactfreshseed/read/actualIDaccounting differs")
            wave["fresh_seed_zero_reuse_proven"]=True;wave["status"]="followups";save()
            for case,ids,is_v2 in tests:
                same();name=case["id"]+("-v2"if is_v2 else "")
                report["current_stage"]="wave"+str(index)+"-"+name;owner.stage=report["current_stage"];save()
                row=selected_generate(observer,owner.measurement,report["current_stage"],ids,B.MAXIMUM,a.case_deadline,None)
                row["measurement_stage"]=row["name"];row["name"]=name;wave["rows"].append(row);save()
                need(row["input_ids_sha256"]==baseline_rows[name]["input_ids_sha256"]and row["reused_tokens"]==B.PREFIX_N and row["natural_eos"] and row["emitted_token_ids"]==row["native_generated_tokens"],"Frozen followup input/full-prefix/EOS/actualID mismatch")
                worker=HERE/"oracles"/("code_worker_v2.py"if is_v2 else "code_worker.py")
                grade=grade_row(row,case,is_v2,original,decode,worker);wave["accuracy"].append(grade);save()
                print(json.dumps({"wave":index,"case":name,"joint_passed":grade["joint_passed"],"actualIDs":row["emitted_token_ids"],"reused":row["reused_tokens"]}),flush=True)
            wave.update(wave_metrics(wave,baseline));wave.update(status="measurement-complete",measurement_complete=True);save()
            if index==1:
                same();report["current_stage"]="unrelated-short-graded-interlude";owner.stage=report["current_stage"];save()
                short=B.Measurement(owner.engine,out,owner.kill_owned,server,prefix_tokens=0)
                inter=selected_generate(observer,short,report["current_stage"],interlude_ids,B.MAXIMUM,a.interlude_deadline,None)
                report["interlude"]=inter;save()
                grade=grade_row(inter,interlude_case,False,original,decode,HERE/"oracles/code_worker.py")
                report["interlude_accuracy"]=grade;save()
                need(grade["joint_passed"]and inter["natural_eos"],"FIRST unrelated interlude failed; no replacement retry or wave2")
                need(inter["prompt_read"]>0,"Short interlude performed no new prompt work")
                print(json.dumps({"stage":"short-interlude","quality_passed":True,"natural_EOS":True,"actual_reuse":inter["reused_tokens"]}),flush=True)
        report["both_waves_speed_targets_and_all13_quality_passed"]=all(w["both_speed_targets_and_all13_quality_passed"]for w in report["waves"])
        report["both_waves_native_phase_admission_passed"]=all(w["native_phase_evidence"]["admission_passed"]for w in report["waves"])
        report["passed"]=all(w["all13_quality_passed"]and w["fresh_seed_zero_reuse_proven"]for w in report["waves"])and report["interlude_accuracy"]["joint_passed"]and report["both_waves_native_phase_admission_passed"]
        report["both_waves_speed_quality_and_native_rearm_admission_passed"]=report["passed"]and report["both_waves_speed_targets_and_all13_quality_passed"]
        all_quality=all(w["all13_quality_passed"]for w in report["waves"])and report["interlude_accuracy"]["joint_passed"]
        report["status"]="measurement-complete"if report["passed"]else "measurement-complete-quality-failed"if not all_quality else "measurement-complete-rearm-admission-failed"
        rc=0 if report["passed"]else 1
    except BaseException as error:
        report.update(passed=False,status="interrupted"if isinstance(error,KeyboardInterrupt)else "failed",error_type=type(error).__name__,
                      error="Two-wave validation/deadline failed; first actual evidence retained, no retry")
        if owner is not None and owner.engine is not None:
            report["failure_last_native_DONE"]=getattr(owner.engine,"raw_done",None)
            report["failure_last_native_stats"]=dict(getattr(owner.engine,"last",{})or{})
        rc=130 if isinstance(error,KeyboardInterrupt)else 1
    finally:
        if owner is not None:
            try:
                owner.stage="closing-owned-EOF";report["shutdown_trace"]=RT.close_owned(owner,server,out/"SHUTDOWN.json");report["cleanup"]=owner.cleanup;validate_cleanup(owner.cleanup,owner.identities[0]);need(report["shutdown_trace"]["natural_normal0_without_escalation"],"Shutdown escalated or native did not exit0");report["cleanup_validated"]=True
            except BaseException as error:
                report.update(passed=False,status="cleanup-failed",cleanup_error_type=type(error).__name__);rc=1
        elif (out/"CLEANUP.json").is_file():report["constructor_cleanup"]=B.read(out/"CLEANUP.json")
        if observer is not None:
            try:
                report["native_retained_direct_qualification"]=observer.finish()
                evidence=report["native_retained_direct_qualification"];setup=evidence["initial_setup"]
                need(setup["device_bytes"]==24*setup["stride"]and setup["host_bounce_bytes"]==2*setup["stride"]and
                     evidence["exit"]["fallbacks"]==0 and all(e["ranges"]==40 and e["bytes"]==184926136320 for e in evidence["physical_pin_history"]),
                     "Selected staging allocation/40arena ownership/CPUfallback0 final evidence differs")
                report["selected_native_final_passed"]=True
            except BaseException as error:
                report.update(passed=False,native_evidence_error_type=type(error).__name__);rc=1
                if report["status"]=="measurement-complete":report["status"]="native-evidence-failed"
            if not report["passed"]:report["first_result_observer_rows"]=observer.rows
        report["protected_files_unchanged"]=bool(protected)and all(Path(f).is_file()and B.sha(f)==h for f,h in protected.items())
        if not report["protected_files_unchanged"]:report.update(passed=False,status="protected-pin-failed");rc=1
        if Path(profile["path"]).is_file():report["expert_profile"]["final_sha256"]=B.sha(profile["path"])
        report["finished"]=True
        report["both_waves_speed_quality_and_native_rearm_admission_passed"]=bool(report["passed"]and report.get("cleanup_validated")and
            report["protected_files_unchanged"]and report.get("both_waves_speed_targets_and_all13_quality_passed",False))
        for wave in report["waves"]:
            wave["whole_process_cleanup_passed"]=bool(report.get("cleanup_validated"))
            wave["whole_run_protected_files_unchanged"]=report["protected_files_unchanged"]
        report["primary_corrected24_quality_passed"]=bool(len(report["waves"])==2 and all(len(w["accuracy"])==13 and all(w["accuracy"][i]["joint_passed"] for i in (*range(11),12)) for w in report["waves"]))
        report["strict_original26_score"]=sum(sum(g["joint_passed"]for g in w["accuracy"])for w in report["waves"])
        if paired and len(report["waves"])==2 and all(w.get("measurement_complete")for w in report["waves"]):
            pairs=[]
            for stock,new in zip(stock_metrics,report["waves"]):
                rate=new["weighted_original12_observed_ID_intervals_s"]
                pairs.append({"wave":new["index"],"stock_original12_ID_intervals_s":stock["original12_ID_intervals_s"],
                    "candidate_original12_ID_intervals_s":rate,"candidate_over_stock":rate/stock["original12_ID_intervals_s"],
                    "at_least_five_percent":rate>=stock["original12_ID_intervals_s"]*1.05,
                    "stock_TTFT_seconds":stock["fresh_seed_TTFT_seconds"],"candidate_TTFT_seconds":new["fresh_seed_TTFT_seconds"]})
            report["paired_original12_waves"]=pairs
            report["paired_original12_rate_gate_passed"]=all(p["at_least_five_percent"]for p in pairs)
        report["predeclared_primary_Q_gate_passed"]=bool(report.get("primary_corrected24_quality_passed")and report.get("paired_original12_rate_gate_passed")and
            report.get("both_waves_native_phase_admission_passed")and report.get("native_retained_direct_qualification",{}).get("passed")and
            report.get("selected_native_final_passed")and report.get("cleanup_validated")and report.get("protected_files_unchanged")and report.get("shutdown_trace",{}).get("natural_normal0_without_escalation"))
        report["Stop5_and_production_API_admission_not_inferred"]=True
        report["standalone_final_quality_native_cleanup_passed"]=bool(report.get("primary_corrected24_quality_passed")and
            report.get("both_waves_native_phase_admission_passed")and report.get("native_retained_direct_qualification",{}).get("passed")and
            report.get("selected_native_final_passed")and report.get("cleanup_validated")and report.get("protected_files_unchanged")and report.get("shutdown_trace",{}).get("natural_normal0_without_escalation"))
        if not paired:
            report["paired_original12_rate_gate_passed"]=None
            report["speed_improvement_or_deployment_eligibility_claimed"]=False
        save();print(json.dumps({"receipt":str(path),"status":report["status"],"passed":report["passed"],
                                "both_waves_speed_and_quality":report.get("both_waves_speed_targets_and_all13_quality_passed",False)}),flush=True)
    return rc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--config',type=Path);p.add_argument('--build-receipt',type=Path)
    p.add_argument('--baseline-report',type=Path);p.add_argument('--expected-baseline-report-sha256')
    p.add_argument('--require-paired',action='store_true')
    p.add_argument('--strata-source',type=Path,default=ROOT/'.work/runtime')
    p.add_argument('--reference-gguf',type=Path);p.add_argument('--reference-tokenizer',type=Path)
    p.add_argument('--inputs',type=Path,default=ROOT/'.work/bench-inputs');p.add_argument('--initial-profile',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'.work/bench/qualification-01')
    p.add_argument('--startup-deadline',type=float,default=900);p.add_argument('--seed-deadline',type=float,default=680)
    p.add_argument('--case-deadline',type=float,default=1200);p.add_argument('--interlude-deadline',type=float,default=300)
    a=p.parse_args()
    need(all(0<value<=3600 for value in (a.startup_deadline,a.seed_deadline,a.case_deadline,a.interlude_deadline)),
         'All explicit deadlines must be finite and within 0..3600 seconds')
    if not a.run:
        print(json.dumps({'plan_only':True,'model_processes':0,'API_calls':0,'writes':0,'waves':2,'fresh_seed_tokens_each':999764,
            'raw_original13_grades_each':13,'primary_corrected_grades_total':24,'primary_rate':'ORIGINAL12 paired>=stock*1.05 EACHwave',
            'exact29GEN_native_observer_phase_handoff_exit0_V3_no_escalation_required':True,'Stop5_and_API_separate':True}))
        return 0
    need(a.reference_gguf is not None or a.reference_tokenizer is not None,'Use already-downloaded Mixshard1 as --reference-gguf')
    return run(a)


if __name__=='__main__':raise SystemExit(main())
