"""Compare two completed same-weight native reports. Default dry; never regrades.

This checks stored protocol/quality/accounting. Callers separately attest actual
report bytes, model files, source/build identity and the remaining Stop/API gates.
"""
from pathlib import Path
import argparse
import copy
import json
import re
import baseline_validation as B
import native_support as N

ROOT=Path(__file__).resolve().parents[1]
INCUMBENT='7bc19cda219953d887878ed408058eb1ad78f154c1200b675bc9d1ab89bd1545'
OLD_OVERLAY='9e949460c896c350945c381d161ad7adf7c0e44f1ad293db95da3df14ca028d2'
need=B.need

def validate(report,image_sha,allow_legacy=False):
    need(report.get('schema')=='retained-direct-two-fresh-frozen-waves-v1'and
         report.get('finished')is True and report.get('cleanup_validated')is True and
         report.get('protected_files_unchanged')is True and report.get('no_retries')is True and
         report.get('same_owned_process_required')is True and report.get('configured_context')==1048576,
         'Completed protected two-wave report required')
    pins=report['protected_files_before'];image=B.norm(report['owned_identity']['image'])
    need([h for p,h in pins.items()if B.norm(p)==image]==[image_sha],'Explicit actual image/report binding differs')
    B.cleanup(report)
    need(report['shutdown_trace']['natural_normal0_without_escalation']is True,'Natural native0 without escalation required')
    profile=report['expert_profile']
    need(profile['initial_sha256']==B.PROFILE and profile['per_run_copy_required']is True and
         profile['learning_may_update_only_this_copy']is True and
         B.norm(profile['path'])==B.norm(B.option(report['effective_native_args'],'--expert-profile')),
         'One private fresh66 profile required')
    need(len(report['waves'])==2,'Exactly two completed waves required')
    metrics=[];raw=0;primary=True
    parity=[{'case':n,'tokens':t,'input_ids_sha256':d,'equal':True}for n,t,d in B.FROZEN[1:]]
    need(report['tokenizer_parity']==parity,'Frozen tokenizer/input parity differs')
    for index,wave in enumerate(report['waves'],1):
        need(wave['index']==index and wave['measurement_complete']is True and
             wave['same_owned_identity']==report['owned_identity']and wave['no_new_profile_or_engine']is True and
             wave['fresh_seed_zero_reuse_proven']is True and wave['whole_process_cleanup_passed']is True and
             wave['whole_run_protected_files_unchanged']is True,'Wave owner/state/cleanup differs')
        rows=wave['rows'];grades=wave['accuracy']
        need(len(rows)==14 and len(grades)==13,'All original FIRST13 grades must remain present')
        for i,(row,frozen)in enumerate(zip(rows,B.FROZEN)):
            B.row_check(row,frozen,seed=i==0)
            if not i:continue
            grade=grades[i-1]
            need(grade['case']==row['name']and type(grade['joint_passed'])is bool and
                 grade['corrected_v2']is(i==13),'Raw grade identity differs')
            # The ambiguous legacy case is retained verbatim. It is excluded
            # only from the already-declared primary quality set, never timing.
            if grade['joint_passed']:B.grade(row,grade,code=i>=11,v2=i==13)
            if i!=12:primary=primary and grade['joint_passed']
            raw+=grade['joint_passed']
        count=sum(row['emitted_token_ids']-1 for row in rows[1:13])
        seconds=sum(B.number(row['generation_seconds'],True)for row in rows[1:13])
        rate=count/seconds;ttft=rows[0]['ttft_seconds'];raw_wave=sum(g['joint_passed']for g in grades)
        need(wave['original12_joint_score']==sum(g['joint_passed']for g in grades[:12])and
             wave['all13_quality_passed']is(raw_wave==13)and wave['corrected_v2_passed']is grades[12]['joint_passed'],
             'Stored raw quality aggregate differs')
        need(wave['original12_actual_ID_intervals']==count,'Stored interval count differs')
        B.close(wave['original12_observed_generation_seconds'],seconds)
        B.close(wave['weighted_original12_observed_ID_intervals_s'],rate);B.close(wave['fresh_seed_TTFT_seconds'],ttft)
        metrics.append({'wave':index,'original12_actual_ID_intervals':count,'original12_generation_seconds':seconds,
            'original12_ID_intervals_s':rate,'fresh_seed_TTFT_seconds':ttft,'raw13_passed':raw_wave})
    B.native_phase(report['waves'][0]['native_phase_evidence'],report['waves'][1]['native_phase_evidence'],True)
    B.direct_rows(report)
    anchor=report['native_retained_direct_qualification']['physical_pin_history'][0]
    need(anchor['ranges']==40 and 0<anchor['bytes']<=196608<<20,'Actual full40 bounded ledger required')
    legacy=allow_legacy and(image_sha==INCUMBENT or OLD_OVERLAY in report['protected_files_before'].values())and'startup_full_arena_inventory'not in report
    if not legacy:
        startup=report['startup_full_arena_inventory'];selected=report['normal_retained_ledger_anchor']
        need(startup['captured_before_first_GEN']is True and startup['layers']==startup['reserved_spares']==
             startup['selected_arenas']==startup['total_arenas']==startup['published_ranges']==40 and
             startup['free_holes']==0 and startup['occupied_CPU_experts']==startup['copied_CPU_experts']and
             startup['slots']==startup['occupied_CPU_experts']+startup['reserved_spares']and startup['NUMA_parts']==1 and
             all(selected[k]==anchor[k]for k in ('ranges','bytes','fingerprint')),
             'New candidate startup inventory/normal ledger binding differs')
        need(abs(anchor['bytes']/1048576-startup['published_MiB_rounded'])<=.050001,
             'Exact candidate normal ledger/rounded startup sum differs')
    for row in report['waves'][0]['rows']+[report['interlude']]+report['waves'][1]['rows']:
        d=row['native_GPU_miss_direct_answer']
        need(d['retained_ranges']==anchor['ranges']and d['retained_bytes']==anchor['bytes']and
             d['bounce_unique']==d['bounce_bytes']==0 and row['native_GPU_miss_answer']['fallbacks']==0,
             'Selected direct192 physical coverage/zero-bounce gate failed')
    B.row_check(report['interlude'],B.INTERLUDE,interlude=True)
    B.grade(report['interlude'],report['interlude_accuracy'])
    need(report.get('both_waves_native_phase_admission_passed')is True and
         report.get('standalone_final_quality_native_cleanup_passed')is primary and
         report.get('passed')is(raw==26)and
         report.get('status')==('measurement-complete'if raw==26 else'measurement-complete-quality-failed'),
         'Stored final raw/primary/native status differs')
    return metrics,raw,primary

def args_and_env(report,image_sha):
    args=copy.deepcopy(report['effective_native_args']);env=copy.deepcopy(report['effective_native_env'])
    args[args.index('--expert-profile')+1]='<private-profile>'
    if '--experts'in args:
        need(B.option(args,'--experts')=='6','All-six routing required')
        i=args.index('--experts');del args[i:i+2]
    else:
        need(image_sha==INCUMBENT or OLD_OVERLAY in report['protected_files_before'].values(),
             'Only the pinned pre-update six-default source/image may omit --experts6')
    need(bool(env.pop('STRATA_DS4_SLOT_DIR',None)),'A private disabled-slot directory is required')
    for key in ('DS4_WARM','DS4_COPY_PIN'):
        need(env.pop(key,'0')=='0','Update shared ownership requires WARM0/COPY_PIN0')
    return args,env

def compare(old,new,old_image,new_image):
    need(old_image!=new_image,'The declared update and control images must differ')
    a,ar,ap=validate(old,old_image,True);b,br,bp=validate(new,new_image)
    need(args_and_env(old,old_image)==args_and_env(new,new_image),'Effective settings differ beyond declared image/profile/WARM0/COPY_PIN0')
    versions=(old.get('controller_python_version'),new.get('controller_python_version'))
    if all(versions):need(versions[0]==versions[1],'Recorded controller Python versions differ')
    pairs=[]
    for x,y in zip(a,b):
        ratio=y['original12_ID_intervals_s']/x['original12_ID_intervals_s']
        ttft=y['fresh_seed_TTFT_seconds']/x['fresh_seed_TTFT_seconds']
        pairs.append({'wave':x['wave'],'baseline':x,'candidate':y,'decode_ratio':ratio,'TTFT_ratio':ttft,
            'update_nonregression_passed':ratio>=.95 and ttft<=1.05,'decode_improved':ratio>1})
    return {'schema':'v027-same-weight-native-update-assessment-v1','reports_validated':True,'waves':pairs,
        'baseline_raw26':ar,'candidate_raw26':br,'baseline_primary24_passed':ap,'candidate_primary24_passed':bp,
        'native_nonregression_and_primary_passed':ap and bp and all(p['update_nonregression_passed']for p in pairs),
        'controller_python_versions_recorded':list(versions),'unrecorded_historical_clock_version_requires_external_runplan_attestation':not all(versions),
        'historical_targets_and_raw_failures_not_rewritten':True,'Stop5_API_and_default_admission_not_inferred':True,
        'no_regrade_retry_or_automatic_promotion':True}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','candidate','output'):p.add_argument('--'+name,type=Path)
    for name in ('baseline','candidate','baseline-image','candidate-image'):p.add_argument('--expected-'+name+'-sha256')
    a=p.parse_args()
    if not a.baseline and not a.candidate:
        print(json.dumps({'default_dry':True,'writes':0,'model_API_actions':0,'decode_min_ratio':.95,'TTFT_max_ratio':1.05}));return 0
    for path,h in ((a.baseline,a.expected_baseline_sha256),(a.candidate,a.expected_candidate_sha256)):
        need(path and re.fullmatch('[0-9a-f]{64}',h or'')and N.sha(path)==h,'Explicit actual closed report SHA required')
    need(all(re.fullmatch('[0-9a-f]{64}',h or'')for h in(a.expected_baseline_image_sha256,a.expected_candidate_image_sha256)),
         'Both actual image pins required')
    need(a.output and a.output.resolve().is_relative_to((ROOT/'.work').resolve())and not a.output.exists(),'Exclusive ignored output required')
    result=compare(N.read(a.baseline),N.read(a.candidate),a.expected_baseline_image_sha256,a.expected_candidate_image_sha256)
    result.update(baseline_sha256=a.expected_baseline_sha256,candidate_sha256=a.expected_candidate_sha256)
    with a.output.open('x',encoding='utf-8')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result));return 0

if __name__=='__main__':raise SystemExit(main())
