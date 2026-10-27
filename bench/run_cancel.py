"""Public owned cancellation/recovery campaign; default is dry.

Reuses the exact native token clock, single stdout consumer and reviewed V6b
cancellation/retention policies. No HTTP, RESET, deployment or server control.
"""
from __future__ import annotations
import argparse,copy,json,re,sys,time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import runtime as B
import runtime
import prepare_inputs as I
import cancel_support as C
import cancel_validation as G
from cancel_support import StopPinHistory,interrupt,read_window,phase_retry,validate_recovery_seed,validate_decode_cancel,validate_phase_cancel,validate_initial_cancel,MODES
P,V,Tap=runtime.P,runtime.V,runtime.Tap
F=runtime
W=SimpleNamespace(PROFILE_SHA=runtime.PROFILE_SHA,validate_cleanup=runtime.cleanup_ok)
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
need=B.need

def option(args,key):
    need(args.count(key)==1 and args.index(key)+1<len(args),'Missing/duplicate native option: '+key)
    return args[args.index(key)+1]

def reference_before(a,cfg,effective_env):
    need(a.reference_control_report and re.fullmatch('[0-9a-f]{64}',a.expected_reference_control_report_sha256 or''),
         'Initial-mode speed needs an explicit completed normal V6 setup-control REPORT/SHA')
    need(B.sha(a.reference_control_report)==a.expected_reference_control_report_sha256,'Reference control REPORT changed')
    ref=B.read(a.reference_control_report)
    need(ref.get('mode')=='setup-stop-twice'and ref.get('finished')is True and ref.get('passed')is True and
         ref.get('cleanup_validated')is True and ref.get('protected_files_unchanged')is True and ref.get('physical_exit_zero')is True,
         'Reference must be a passing normal setup recovery control')
    W.validate_cleanup(ref['cleanup'],ref['owned_identity'])
    need(ref['expert_profile']['initial_sha256']==W.PROFILE_SHA and
         all(x in ref['protected_files_before'].values()for x in (a.expected_exe_sha256,a.expected_source_manifest_sha256,B.sha(a.build_receipt))),
         'Cross-process reference engine/source/build/fresh66 differs')
    def args_without_profile(args):return B.replace_option(args,'--expert-profile','<private-profile>')
    def env_without_slot(env):return {k:v for k,v in env.items()if k!='STRATA_DS4_SLOT_DIR'}
    expected=B.replace_option(cfg['args'],'--dspark',a.dspark);expected=B.replace_option(expected,'--prefill-chunk',a.prefill_chunk)
    need(args_without_profile(ref['effective_native_args'])==args_without_profile(expected),
         'Cross-process reference native arguments differ')
    need(env_without_slot(ref['effective_native_env'])==env_without_slot(effective_env),
         'Cross-process reference effective native environment differs')
    before=ref['matching_BEFORE_short'];G.direct_no_bounce(before)
    need(before['input_ids_sha256']=='6b355c6f44d2cbace47309216051869e798220085d055af38b17ca5a8a820a90'and
         before['prompt_tokens']==168 and before['quality_passed']is True and before['natural_eos']is True,
         'Reference must preserve the matching FIRST corrected-v2 input and result')
    return before,ref


def run(a):
    need(sys.platform=='win32' and a.config and a.mode,'Explicit Windows mode/config required')
    need(a.mode in MODES and not a.overlap and a.registered_ram_mib==196608,'Exact direct192/no-overlap modes only')
    if a.mode=='initial-prefill-stop':
        need(a.reference_control_report is not None and
             re.fullmatch('[0-9a-f]{64}',a.expected_reference_control_report_sha256 or''),
             'Initial mode requires the completed control REPORT and its explicit SHA before model startup')
    out=a.output.resolve()
    need(out.is_relative_to((ROOT/'.work').resolve())and not out.exists(),'Exclusive output below ignored .work required')
    import psutil
    need(not [p.pid for p in psutil.process_iter(['name'])if p.info['name']and p.info['name'].lower()in
              ('strata-dsv41.exe','llama-server.exe','xolotl.exe')],'Stop other model servers explicitly; this script never stops them')
    cfg=copy.deepcopy(B.read(a.config));env=cfg.get('env',{})
    need(all(env.get(k)=='1'for k in ('DS4_GPU_MISSES','DS4_GPU_MISS_DIRECT','DS4_PREFILL_REPEAT_REARM')) and
         env.get('DS4_PREFILL_REGISTER_RAM_MB')=='196608'and env.get('DS4_PREFILL_REPEAT_MIN_TOKENS')=='32768'and
         env.get('DS4_GPU_MISS_OVERLAP','0')=='0'and env.get('DS4_CPU_SMT','0')=='0',
         'Explicit direct192/repeat32K/CPU16/no-overlap configuration required')
    for key,value in (('--max-context','1048576'),('--cpu-threads','16'),('--prefill-chunk','65280'),
                      ('--dspark','on'),('--dspark-block','3'),('--dspark-conf','0.65')):
        need(option(cfg['args'],key)==value,'Frozen geometry/DSpark differs: '+key)
    need('DS4_DSPARK_MAX'not in env and 'DS4_DSPARK_CONF'not in env,'Hidden DSpark override differs')
    protected=runtime.validate_build(cfg,a.build_receipt)
    prepared,prefix=runtime.ready_inputs(a.inputs)
    runtime.verify_prepared_tokenizer(cfg,prepared)
    server,ST=runtime.load_frontend(a)
    short,original,decode,parity=runtime.frozen_short_inputs(a,cfg)
    expected_short=[('filtered-decimal-revenue',164,'22464a7c6f125403f341c2f1905a5fe2dd332d1f6ad2ad074fcf1fc1a8e8b87a'),
                    ('code-account-balance',168,'6b355c6f44d2cbace47309216051869e798220085d055af38b17ca5a8a820a90'),
                    ('code-filtered-cents',126,'f51a813ee106aac5a13b2e69fb586b1f99593ce6255d829f9475c9383097b9e6')]
    need([(case['id'],len(ids),B.idsha(ids))for _,case,ids in short]==expected_short,'Frozen standalone input parity differs')
    frozen=I.frozen_manifest()
    full=[(next(c for c in original.CASES if c['id']==info['name']),prefix+info['suffix_ids'],False)
          for info in prepared['inputs'][:12]]
    probe,tests=C.probe_inputs(prefix,full,131072)
    profile_input=a.initial_profile or a.inputs/'initial-routing-profile.bin'
    need(B.sha(profile_input)==runtime.PROFILE_SHA,'Frozen initial66 profile required')
    files=[a.config,a.build_receipt,profile_input,a.inputs/'INPUTS.json',a.inputs/'prefix.u32le',
           *HERE.glob('*.py'),*HERE.glob('*PROVENANCE.json'),HERE/'frozen-inputs.json',
           *(HERE/'oracles').glob('*.py'),*(HERE/'vendor').glob('*'),
           a.strata_source/'serve/server.py',a.strata_source/'tools/strata_tokenizer.py']
    token=Path(cfg['tokenizer'])
    if not token.is_absolute():token=Path(cfg.get('cwd',a.config.parent))/token
    token_files=[token/name for name in ('vocab.json','merges.txt','token_type.json','tokenizer.json')]
    need(all(p.is_file()for p in token_files),'Complete packed tokenizer files required')
    files+=token_files
    if a.reference_gguf:files.append(a.reference_gguf) # metadata-only input; no full weight hash below
    if a.reference_tokenizer:files.append(a.reference_tokenizer)
    # The already-downloaded reference GGUF is inspected by the encoder only.
    # Do not hash or copy its multi-GiB payload in this small benchmark adapter.
    files=[p for p in files if p!=a.reference_gguf]
    protected.update({str(p):B.sha(p)for p in files if p.is_file()})
    if a.mode=='initial-prefill-stop':
        need(a.reference_control_report is not None,'Initial mode needs a completed matching V6 normal reference')
        protected[str(a.reference_control_report)]=B.sha(a.reference_control_report)
    out.mkdir(parents=True,exist_ok=False)
    cfg,profile=runtime.make_private_config(cfg,out,profile_input)
    build=B.read(a.build_receipt)
    a.expected_exe_sha256=build['exe_sha256']
    a.expected_source_manifest_sha256=build['source_overlay_manifest_sha256']
    # This exact source hash is already in validate_build's protected map.
    need(a.expected_source_manifest_sha256 in protected.values(),'Current public overlay identity must be protected')
    a.dspark='on';a.prefill_chunk='65280'
    path=out/'REPORT.json'
    report=dict(schema='portable-V6b-cancel-recovery-v1',mode=a.mode,passed=False,finished=False,rows=[],accuracy=[],
                expert_profile=profile,NON_TARGET_recovery=True,no_full13_claim=True,no_RESET_new_wire_or_retries=True,
                postcancel_matching_rate_minimum=0.95,declared_abort_gaps=[],protected_files_before=protected,
                tokenizer_parity=parity,public_adapter=True,
                raw_parent_stop_protocol_preserved_with_predeclared_BEFORE_and_eviction=a.mode!='initial-prefill-stop',
                short_inputs=[dict(case=case['id'],tokens=len(ids),ids_sha256=B.idsha(ids))for _,case,ids in short],
                source_execution_scope='This run covers only selected mode; private five-mode campaign is separate evidence')
    save=lambda:B.save(path,report);save();owner=None;rc=1;overlap=None
    try:
        import job_support,identity_support
        owned=SimpleNamespace(**vars(a));owned.seed_only=True;owned.probe_prefix_tokens=None
        owned.seed_deadline=a.stop_deadline+a.retry_deadline+9*a.case_deadline
        with mock.patch.object(server,'StrataEngine',Tap.capture_class(server.StrataEngine)):
            owner=B.OwnedNative(owned,cfg,out,server,job_support.OwnedJob,identity_support)
        need(len(owner.processes)==len(owner.identities)==1,'One owned child required')
        proc,identity=owner.processes[0],owner.identities[0]
        report.update(owned_identity=identity,engine_info=dict(owner.engine.info),
                      effective_native_args=owner.effective_args,effective_native_env=owner.effective_env);save()
        pins=StopPinHistory(a.registered_ram_mib);log=out/'ENGINE.log';cursor=log.stat().st_size
        def same():need(owner.engine.proc is proc and proc.pid==identity['pid'] and proc.poll()is None,'Owned engine changed/exited')
        def window_since(offset):
            until=time.monotonic()+1
            while True:
                text=read_window(log,offset)
                if (V.GPU_ANSWER.search(text) and P.ANSWER.search(text)) or time.monotonic()>=until:return text
                time.sleep(.005)
        def attach(row,text,work=True):
            pins.feed(text);actual_work=row.get('requested_output',B.MAXIMUM)!=1
            primary=V.gpu_answer(text,actual_work);row['native_GPU_miss_answer']=primary
            if pins.active:row['native_GPU_miss_direct_answer']=P.direct_answer(text,primary,pins.active,work)
            else:
                matches=P.ANSWER.findall(text);need(len(matches)==1 and matches[0][-2:]==('0','0'),'Inactive post-cancel ownership must be zero')
                need(int(matches[0][0])==int(matches[0][1])==0,'Unregistered recovery short request claimed direct DMA')
                need(int(matches[0][2])==primary['unique_blobs'] and int(matches[0][3])==primary['h2d_bytes'],'Recovery bounce partition differs')
            if overlap:row['native_GPU_miss_overlap_answer']=overlap.overlap_answer(text,primary,actual_work)
            if row.get('requested_output',B.MAXIMUM)!=1 and pins.active:G.direct_no_bounce(row)
            report['retained_pin_history']=list(pins.events)
        def grade(item,name,pin=0,reused=None,require_direct=True):
            nonlocal cursor
            same();module,case,ids=item;owner.stage=name;begin=log.stat().st_size
            row=F.grade_first(owner,B.Measurement(owner.engine,out,owner.kill_owned,server,prefix_tokens=pin),case,ids,decode,module,
                HERE/'oracles'/('code_worker_v2.py'if module is not original else'code_worker.py'),a.case_deadline,reused,report,save)
            need(row['quality_passed'] and row['natural_eos'] and row['prompt_read']>0,'FIRST frozen quality/EOS/newfeed failed')
            if case['kind']=='code':
                ex=row['grade']['execution'];need(ex.get('safe_policy_passed')is True and ex.get('tests_passed')==ex.get('tests_total')==11,'FIRST11-vector gate failed')
            text=window_since(begin);attach(row,text,require_direct);cursor=log.stat().st_size
            report['accuracy'].append(dict(case=name,joint_passed=True));save();return row,text
        if a.mode=='initial-prefill-stop':
            before,reference=reference_before(a,cfg,owner.effective_env);active=None
            report.update(matching_BEFORE_short=before,no_warmup_or_BEFORE_GEN_in_this_process=True,
                cross_process_speed_reference=dict(REPORT_sha256=a.expected_reference_control_report_sha256,
                    mode=reference['mode'],owned_identity=reference['owned_identity'],state_difference_disclosed=True))
            save()
        else:
            _,warmtext=grade(short[0],'initial-short-decimal',reused=0)
            report['initial_compact_retention']=P.compact_evidence(warmtext);active=copy.deepcopy(pins.active);save()
            matching=short[2] if a.mode=='decode-stop' else short[1]
            before,_=grade(matching,'BEFORE-matching-short-code',require_direct=True)
            report['matching_BEFORE_short']=before;save()
            grade(short[0],'before-cancel-unrelated-decimal',require_direct=True)
            need(pins.active==active,'BEFORE preparation changed healthy physical ownership')
        same();owner.stage='actual-'+a.mode
        interrupted,segment,offset=interrupt(owner,server,short[1][2]if a.mode=='decode-stop'else prefix,log,out,
            'setup-stop'if a.mode=='setup-stop-twice'else a.mode,a.stop_deadline)
        report['interrupted']=interrupted
        (out/'interrupted-ENGINE-window.log').write_text(segment,encoding='utf-8')
        if a.mode=='decode-stop':
            report['cancel_validation']=validate_decode_cancel(segment,interrupted,active);pins.feed(segment)
            if overlap:report['cancel_validation']['overlap']=overlap.overlap_answer(segment,report['cancel_validation']['actual_GPU_work'],True)
            need(pins.active==active,'Ordinary decode Stop lost healthy retained context')
            first,first_text=grade(short[2],'FIRST-after-decode-Stop-code-filtered',require_direct=True)
            need('GPU_MISS_RECOVERY ' not in first_text,'Ordinary decode Stop must keep healthy context without recovery')
            report['first_short_recovery']=dict(required=False,healthy_context_preserved=True)
            generation,attempt=1,1
        elif a.mode in ('setup-stop','active-prefill-stop','early-rearm-stop','setup-stop-twice'):
            report['cancel_validation']=validate_phase_cancel(segment,interrupted,active,
                'setup-stop'if a.mode=='setup-stop-twice'else a.mode);pins.feed(segment)
            need(pins.active is None,'Cancelled new phase left published DECODE context')
            gap=report['cancel_validation']['old_decode_owner_physically_retired']['epoch']+1
            pins.declare_gap(gap);report['declared_abort_gaps'].append(gap)
            report['unregistered_retry_epoch']=gap
            first,first_text=grade(short[1],'FIRST-after-prefill-Stop-corrected-v2',require_direct=True)
            report['first_short_recovery']=G.recovered(first_text,active,pins.active,2,
                before_geometry=(1024,1024)if a.mode=='early-rearm-stop'else(65280,4096))
            if a.mode=='early-rearm-stop':
                reserve=report['first_short_recovery']['actual_checked_reserve_shrink']
                report['above_cap_small_workspace_exercised']=bool(reserve and reserve['pool_before_bytes']>reserve['target_cap_bytes'])
                need(report['above_cap_small_workspace_exercised'],'Early Stop raced/missed required small-workspace above-cap state; no retry')
            need(report['first_short_recovery']['native_recovery_elapsed_ms'] <= first['ttft_seconds']*1000+10,'Recovery cost must remain inside actual FIRST GEN TTFT')
            generation,attempt=(1,2)if a.mode=='setup-stop'else(2,2)
            if a.mode in ('early-rearm-stop','setup-stop-twice'):generation,attempt=1,2
        elif a.mode=='initial-prefill-stop':
            report['cancel_validation']=validate_initial_cancel(segment,interrupted);pins.feed(segment)
            need(pins.active is None and not pins.events,'Initial cancellation must have no prior DECODE owner')
            first,first_text=grade(short[1],'FIRST-after-initial-prefill-Stop-corrected-v2',require_direct=True)
            report['first_short_recovery']=G.recovered(first_text,None,pins.active,1,allow_unknown_healthy=True)
            need(report['first_short_recovery']['saved_healthy_pool_bytes']==0 and
                 all(pins.active[k]==before['native_GPU_miss_direct_answer']['retained_'+k] for k in ('ranges','bytes')),
                 'Unknown anchor or full physical registration selection differs')
            need(report['first_short_recovery']['native_recovery_elapsed_ms']<=first['ttft_seconds']*1000+10,
                 'Initial recovery cost must remain inside FIRST GEN TTFT')
            generation,attempt=1,1
        report['FIRST_matching_short']=first
        report['postcancel_speed']=G.matching_speed(before,first);save()
        need(report['postcancel_speed']['passed'],'FIRST postcancel decode is below95percent of matching BEFORE')
        if a.mode=='setup-stop-twice':
            first1=first;recovery1=report['first_short_recovery'];speed1=report['postcancel_speed'];active2=copy.deepcopy(pins.active)
            report.update(FIRST_matching_shorts=[first1],first_short_recoveries=[recovery1],postcancel_speeds=[speed1],
                          interrupted_attempts=[interrupted],cancel_validations=[report['cancel_validation']]);save()
            grade(short[0],'between-two-setup-cancels-decimal',require_direct=True)
            need(pins.active==active2,'Short interlude changed physical context')
            second,segment2,_=interrupt(owner,server,prefix,log,out,'setup-stop',a.stop_deadline)
            report['interrupted_attempts'].append(second)
            (out/'interrupted-ENGINE-window-second.log').write_text(segment2,encoding='utf-8')
            validation2=validate_phase_cancel(segment2,second,active2,'setup-stop',2,3);report['cancel_validations'].append(validation2)
            pins.feed(segment2);need(pins.active is None,'Second cancelled setup retained a DECODE owner')
            pins.declare_gap(4);report['declared_abort_gaps'].append(4);report['unregistered_retry_epoch']=4
            first,first_text=grade(short[1],'FIRST-after-second-setup-Stop-corrected-v2',require_direct=True)
            recovery2=G.recovered(first_text,active2,pins.active,4,2)
            G.same_normal_anchor(recovery1,recovery2)
            need(recovery2['native_recovery_elapsed_ms']<=first['ttft_seconds']*1000+10,'Second recovery cost escaped FIRST TTFT')
            speed2=G.matching_speed(before,first)
            report['FIRST_matching_shorts'].append(first);report['first_short_recoveries'].append(recovery2);report['postcancel_speeds'].append(speed2)
            report['FIRST_matching_short']=first;report['first_short_recovery']=recovery2
            report['normal_anchor_did_not_ratchet']=True;report['postcancel_speed']=speed2;save()
            need(speed2['passed'],'Second FIRST postcancel decode is below95percent of matching BEFORE')
            generation,attempt=1,3
        save();same();owner.stage='NON_TARGET131072-recovery-seed';begin=log.stat().st_size
        seed=B.Measurement(owner.engine,out,owner.kill_owned,server,prefix_tokens=131072).generate(owner.stage,probe,1,a.retry_deadline,0)
        report['rows'].append(seed);validate_recovery_seed(seed)
        text=window_since(begin);report['retry_phase_evidence']=phase_retry(text,generation,attempt,131072);attach(seed,text,False);save()
        for case,ids,_ in tests:
            row,_=grade((original,case,ids),'NON_TARGET131072-'+case['id'],131072,None)
            need(row['reused_tokens']==131072,'Recovery suffix did not retain exact131072 prefix')
        report.update(passed=True,status='actual-Stop-FIRST-direct-recovery-speed-and-NON_TARGET-reuse-passed');rc=0
    except BaseException as error:
        report.update(passed=False,status='failed',error_type=type(error).__name__,error='FIRST Stop/recovery gate failed; evidence retained, no retry');rc=1
        if owner and owner.engine:report['failure_last_DONE']=getattr(owner.engine,'raw_done',None)
    finally:
        if owner:
            try:
                owner.stage='closing-owned-EOF';report['shutdown_trace']=runtime.close_owned(owner,server,out/'SHUTDOWN.json');report['cleanup']=owner.cleanup;W.validate_cleanup(owner.cleanup,owner.identities[0]);need(report['shutdown_trace']['natural_normal0_without_escalation'],'Shutdown escalated or native did not exit0');report['cleanup_validated']=True
                text=log.read_text(errors='replace');end=StopPinHistory(a.registered_ram_mib,report['declared_abort_gaps'])
                # Cancelled active/setup phases have no DECODE retained event;
                # normal recovery and exit events still use the strict history.
                end.feed(text);need(end.active is None and end.events and (not report['passed'] or end.events[-1]['reason']=='exit'),'Final retained owner not physically retired')
                epochs=([0,1,1,1]if a.mode=='decode-stop'else[1,2,2,2]if a.mode=='initial-prefill-stop'else
                        [0,1,2,3,4,5,5,5]if a.mode=='setup-stop-twice'else[0,1,2,3,3,3])
                expected_kinds=['retained','retired']*(len(epochs)//2)
                if report['passed']:
                    need([e['kind'] for e in end.events]==expected_kinds,'Recovery campaign physical transition count differs')
                    need([e['epoch'] for e in end.events]==epochs and not end.allowed_gaps,'Actual short-recovery and long-rearm lease accounting differs')
                    counts=(5,4)if a.mode=='initial-prefill-stop'else(10,9)if a.mode=='setup-stop-twice'else(8,7)
                    need((len(report['rows']),len(report['accuracy']))==counts,'Exact predeclared loaded rows/grades required')
                    if a.mode=='decode-stop':need('GPU_MISS_RECOVERY ' not in text,'Healthy decode Stop unexpectedly used recovery')
                report['retained_pin_history_complete']=end.events;report['physical_exit_zero']=True
                report['native_GPU_miss_setup']=V.gpu_setup(text)
                if report['rows'] and 'native_GPU_miss_answer'in report['rows'][-1]:
                    report['native_GPU_miss_exit']=V.gpu_exit(text,report['rows'][-1]['native_GPU_miss_answer'])
            except BaseException as error:report.update(passed=False,status='cleanup-failed',cleanup_error_type=type(error).__name__);rc=1
        elif (out/'CLEANUP.json').is_file():report['constructor_cleanup']=B.read(out/'CLEANUP.json')
        report['protected_files_unchanged']=bool(protected) and all(Path(p).is_file() and B.sha(p)==h for p,h in protected.items())
        if not report['protected_files_unchanged']:report.update(passed=False,status='protected-failed');rc=1
        if Path(profile['path']).is_file():report['expert_profile']['final_sha256']=B.sha(profile['path'])
        report['finished']=True;save();print(json.dumps(dict(receipt=str(path),passed=report['passed'],status=report['status'])),flush=True)
    return rc

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--mode',choices=MODES)
    p.add_argument('--config',type=Path);p.add_argument('--build-receipt',type=Path)
    p.add_argument('--strata-source',type=Path,default=ROOT/'.work/runtime-60')
    p.add_argument('--inputs',type=Path,default=ROOT/'.work/bench-inputs');p.add_argument('--initial-profile',type=Path)
    p.add_argument('--reference-gguf',type=Path);p.add_argument('--reference-tokenizer',type=Path)
    p.add_argument('--reference-control-report',type=Path);p.add_argument('--expected-reference-control-report-sha256')
    p.add_argument('--output',type=Path,default=ROOT/'.work/cancel/run-01')
    p.add_argument('--startup-deadline',type=float,default=900);p.add_argument('--stop-deadline',type=float,default=600)
    p.add_argument('--retry-deadline',type=float,default=900);p.add_argument('--case-deadline',type=float,default=600)
    p.add_argument('--registered-ram-mib',type=int,choices=(196608,),default=196608)
    a=p.parse_args();a.overlap=False
    need(all(0<getattr(a,k)<=3600 for k in ('startup_deadline','stop_deadline','retry_deadline','case_deadline')),
         'All deadlines must be0<seconds<=3600')
    if not a.run:
        print(json.dumps({'default_dry':True,'model_API_process_calls':0,'writes':0,'modes':MODES,
            'mandatory_campaign':['setup-stop-twice','decode-stop','active-prefill-stop','early-rearm-stop','initial-prefill-stop'],
            'primary_FIRST_rate_ratio':0.95,'context':1048576,'registered_ram_mib':196608}));return 0
    need(a.config and a.build_receipt and a.mode and (a.reference_gguf or a.reference_tokenizer),
         'Explicit config/freshBUILD/mode and downloaded Mix reference tokenizer/shard1 required')
    return run(a)

if __name__=='__main__':raise SystemExit(main())
