"""Pure stored-JSON comparison; no model imports, file reads, or runtime actions.

The caller pins the actual report bytes, sources/build, and intended configs.
This module independently checks their stored protocol/accounting/quality and
recomputes paired original-12 rates. It cannot regrade generated code or attest
the model payload from a report. No threshold override is exposed.
"""
from __future__ import annotations
import copy
import hashlib
import math
import re
import struct

PREFIX = 999764
PROFILE = '66b2d59fa7fa02cb6ab641846640a8e8c5fbc50906b8a1892140787e8ccf3428'
PRIMARY = 'e6ccf0e2dc4b09424c77f23adc92d08740734ed00be840abb2175f5b5d8a1d17'
STOCK_EXE = '81a63276afde1aedeea1d943f6a1c971f73b7c47d2c5d574c69d07c9b05a0bad'
DIRECT_EXE = 'd23b3b20df1c8214e40b62c096fffb481ec127cbe02361b2221bfb22c3925d72'
STOCK_SOURCE = '80ddb59d2d589fed88715fe9419dd1df30d20ee5d866fc8692cc5b21e58afe8e'
DIRECT_SOURCE = 'f9da20278159fe77e4d8378f924708ca4b2ee383a6f30275a98aa3210d38055e'
REQUIRED_RATIO = 1.05
HISTORICAL_TTFT = 342.46961880000026
HISTORICAL_RATE = 32.60134856572709
FROZEN = (
    ('fresh-prefix-initialization',999764,'989dd0659bd086d9996cf370b4f656a727061dfd8289a26a462a71fc59aca54e'),
    ('audit-sum',999797,'7a70ba89c64c29248379220f3a85a79a6a6fe38dd89e7f8e30078600c677e482'),
    ('audit-range',999815,'81ee5f0d38f9ce2227044e71a8ed01d3267c7bd4ffaf0bf366cd34c4459dbd5f'),
    ('audit-median',999808,'6b27b31252ad09873ffce01427031a36391768ae5b14ee6b3268fb3435ab02f7'),
    ('audit-total',999810,'e7c42342470e9eeba341b54a80ae4636827e1258c7dd263f4529f241ca4a76ff'),
    ('filtered-decimal-revenue',999901,'593973ca1d0c0fe60cbf07604dd5931e9ceb7e99d0fe576a1ef6539c83b64650'),
    ('python-output',999827,'8ccadde85d303d30c07af73187c433b1dd227a397a5c26752bdd3fd6d11cdda0'),
    ('source-authority',999886,'1119099be8520a3fc7cedf4ae2f962b6b74644bb9979a13226428bcb195dfd8c'),
    ('archive-metadata-join',999937,'7babbc6d8bc332ba777df5c7e7e40d9c8116eef80aa6de647348b5dc44710f00'),
    ('missing-record',999814,'9cf037b2ef073f15b6ce44eec235664bd525533fb95cc527ba7cd1cfb1a18ff4'),
    ('evidence-limited-causality',999900,'b9a46238f603e8afd769dcbeee96de14530ffb5cd689bf4eff78cfab4dff58d4'),
    ('code-filtered-cents',999863,'ea30f58c19cbe6487ae0e0c95fd74e46f87870fdc55a0a3c2aab5a81fc6db55e'),
    ('code-account-balance',999868,'83e889f3c5c40d6b7abbd47cce151f9d69b7a0e27ccc6ec11856b709c56433b1'),
    ('code-account-balance-v2',999905,'0914279aec705a695e399c3768c409d2bfd3b2a3d73fe22230605ec9e954dc2c'),
)
INTERLUDE = ('unrelated-short-graded-interlude',164,'22464a7c6f125403f341c2f1905a5fe2dd332d1f6ad2ad074fcf1fc1a8e8b87a')

class Refused(ValueError):
    pass

def need(value, message):
    if not value: raise Refused(message)

def integer(value, minimum=0):
    need(type(value) is int and value >= minimum, 'Invalid integer accounting')
    return value

def number(value, positive=False):
    need(type(value) in (int,float) and math.isfinite(value) and (value>0 if positive else value>=0), 'Invalid finite timing')
    return value

def close(actual, expected):
    number(actual)
    need(math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12), 'Stored metric differs from raw recomputation')

def norm(path):
    need(isinstance(path,str) and bool(path), 'Missing stored path')
    return path.replace('\\','/').casefold()

def ids_sha(ids):
    need(isinstance(ids,list) and 0<len(ids)<=4096, 'Invalid actual ID sequence')
    need(all(type(x) is int and 0<=x<=0xffffffff for x in ids), 'Invalid actual token ID')
    return hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()

def option(args, name):
    need(isinstance(args,list) and all(isinstance(x,str) for x in args) and args.count(name)==1, 'Missing/duplicate effective option')
    i=args.index(name);need(i+1<len(args),'Missing option value')
    return args[i+1]

def cleanup(report):
    ident=report['owned_identity'];c=report['cleanup']
    integer(ident['pid'],1);integer(ident['created_filetime'],1);norm(ident['image'])
    need(c.get('cleanup_proof_version')==3 and c.get('passed') is True and c.get('owned_job_closed') is True and
         c.get('owned_job_empty_proven') is True and type(c.get('owned_job_active_processes_before_close')) is int and
         c['owned_job_active_processes_before_close']==0 and c.get('errors')==[], 'Exact V3 empty/closed Job proof failed')
    h=c['separate_process_handle_checks'];need(len(h)==1 and h[0].get('recorded_identity')==ident and
        h[0].get('separately_held_handle_signaled') is True and type(h[0].get('wait_result')) is int and h[0]['wait_result']==0 and
        h[0].get('pid_absence_claimed') is False and c.get('pid_absence_claimed') is False, 'Held identity/signal proof failed')
    need(c.get('recorded_processes')==[ident] and c.get('popen_metadata_not_exit_proof')==[{'pid':ident['pid'],'returncode':0}] and
         type(c['popen_metadata_not_exit_proof'][0]['returncode']) is int and c.get('graceful_engine_close_error_type') is None and
         c['resources'].get('sampler_stopped') is True and c['resources'].get('error_type') is None, 'Native normal0/graceful/sampler proof failed')

def grade(row, accuracy, code=False, v2=False):
    g=accuracy['grade']
    need(accuracy.get('case')==row['name'] and accuracy.get('joint_passed') is True and
         accuracy.get('corrected_v2') is v2 and g.get('semantic_correct') is True and g.get('format_compliant') is True and
         isinstance(row.get('final_text'),str) and bool(row['final_text'].strip()) and
         row.get('actual_thinking_close_observed') is True and 128822 in row['token_ids'], 'FIRST stored grade/reasoning-close evidence failed')
    if code:
        e=g['execution']
        need(g.get('all11_vectors_passed') is True and e.get('safe_policy_passed') is True and
             type(e.get('tests_passed')) is type(e.get('tests_total')) is int and e['tests_passed']==e['tests_total']==11,
             'Stored original code11 execution failed')
        if v2: need(type(e.get('tests_run')) is int and e['tests_run']==11 and e.get('status')=='executed', 'Stored corrected-v2 execution failed')

def row_check(row, frozen, seed=False, interlude=False):
    name,tokens,digest=frozen
    need((row['name'],row['prompt_tokens'],row['input_ids_sha256'])==(name,tokens,digest), 'Noncanonical frozen input identity')
    integer(row['prompt_tokens'],1);integer(row['requested_output'],1)
    ids=row['token_ids'];need(ids_sha(ids)==row['output_ids_sha256'], 'Actual output IDs/hash differ')
    n=integer(row['emitted_token_ids'],1)
    need(n==len(ids)==integer(row['native_generated_tokens'],1), 'Actual/native generated count differs')
    d=row['native_done'];need(integer(d['generated'])==n and integer(d['prompt_tokens'])==tokens and
        integer(d['reused'])==integer(row['reused_tokens']) and row.get('None_heartbeats_counted') is False, 'Native DONE accounting differs')
    read=integer(row['prompt_read']);need(0<=row['reused_tokens']<=tokens and read==tokens-row['reused_tokens'], 'Prompt read/reuse accounting differs')
    if 'prompt_read' in d:need(integer(d['prompt_read'])==read,'Native prompt-read field differs')
    need(row['requested_prefix_tokens']==(0 if interlude else PREFIX), 'Prefix pin differs')
    number(row['ttft_seconds'],True);number(row['native_prompt_ms']);number(row['native_decode_ms'])
    if seed:
        need(row['requested_output']==n==1 and row['reused_tokens']==0 and read==PREFIX and d['finish'] in ('stop','length'), 'Fresh one-ID seed incomplete/canceled/reused')
    else:
        need(row['requested_output']==4096 and row.get('natural_eos') is True and row.get('EOS_included_in_stream_IDs') is True and
             ids[-1]==1 and d['finish']=='stop' and read>0, 'Graded native natural EOS required')
        if not interlude:need(row['reused_tokens']==PREFIX,'Frozen followup did not reuse exact prefix')
        number(row['generation_seconds'],True)

def cumulative(phase):
    c=phase['cumulative_retirement_counters']
    for k in ('cold_sets','cold_unpacks','eg_fills','eg_hits','eg_gemms','dense_fills','dense_hits','dense_gemms'):integer(c[k],1)
    return c

def native_phase(first,second,direct):
    need(first.get('admission_passed') is True and second.get('admission_passed') is True,'Native phase admission failed')
    a,b=cumulative(first),cumulative(second);delta=second['actual_generation1_work_delta']
    for k in ('cold_sets','cold_unpacks','eg_fills','eg_hits','dense_fills','dense_hits'):
        need(integer(delta[k],1)==b[k]-a[k], 'Actual repeat delta differs from cumulative work')
    need(b['eg_gemms']>a['eg_gemms'] and b['dense_gemms']>a['dense_gemms'],'Repeated original-shape GEMMs missing')
    if direct:
        need(second.get('generation')==1 and second.get('remaining_feed_tokens',second.get('remaining'))==PREFIX,'Direct rearm generation/feed differs')
    else:
        q=second['native_rearm_ready']
        need(second.get('native_repeat_generation')==second.get('native_repeat_attempt')==1 and
             q['remaining']==PREFIX and (q['C'],q['A'],q['AI'])==(65280,4096,128) and
             integer(q['epoch'],1)>0 and integer(q['issued'],1)>0 and integer(q['pool_bytes'],1)>0 and
             second.get('registration_ranges_after_retirement')==0,'Stock actual rearm/retirement differs')

def direct_rows(report):
    final=report['native_retained_direct_qualification'];need(final.get('passed') is True and final.get('exact_completed_GENs')==29 and
        final.get('overlap_mode') is False and final.get('Stop_recovery_not_claimed') is True,'Final direct/29-GEN qualification missing')
    h=final['physical_pin_history'];need(len(h)==4 and [e['kind'] for e in h]==['retained','retired','retained','retired'] and
        h[1]['reason']=='epoch' and h[3]['reason']=='exit','Exact physical pin lifecycle missing')
    signature=tuple(h[0][k] for k in ('ranges','bytes','fingerprint'))
    need(integer(signature[0],1)>0 and 0<integer(signature[1],1)<=196608*1024**2 and re.fullmatch('[0-9a-f]{16}',signature[2]),'Invalid actual retained ownership')
    need(all(tuple(e[k]for k in ('ranges','bytes','fingerprint'))==signature for e in h) and h[0]['epoch']==0 and
        integer(h[1]['epoch'],1)==h[2]['epoch']<=h[3]['epoch'] and all(integer(a['issued'])<=integer(b['issued'])for a,b in zip(h,h[1:])),
        'Physical fingerprint/epoch/sequence changed')
    first=report['waves'][0]['native_phase_evidence']['initial_retained_decode_owner']
    second=report['waves'][1]['native_phase_evidence']['physical_registration_handoff']
    for event,owner in ((h[0],first),(h[1],second['retired_to_zero_before_HELD']),(h[2],second['new_retained_decode'])):
        need(all(event[k]==owner[k]for k in ('ranges','bytes','fingerprint','epoch','issued')),'Stored phase/physical handoff differs')
    rows=report['waves'][0]['rows']+[report['interlude']]+report['waves'][1]['rows'];last=(0,0)
    for index,r in enumerate(rows):
        g=r['native_GPU_miss_answer'];d=r['native_GPU_miss_direct_answer'];work=index not in (0,15)
        for k in ('layer_windows','kernel_windows','staged_entries','unique_blobs','h2d_bytes','fallbacks','cumulative_entries','cumulative_bytes'):integer(g[k])
        need(g['fallbacks']==0 and g['kernel_windows']==g['layer_windows'] and g['unique_blobs']<=g['staged_entries'] and
            g['cumulative_entries']>=max(g['staged_entries'],last[0]) and g['cumulative_bytes']>=max(g['h2d_bytes'],last[1]),'GPU-miss window/cumulative accounting failed')
        for k in ('direct_unique','direct_bytes','bounce_unique','bounce_bytes','retained_ranges','retained_bytes'):integer(d[k])
        need(d['direct_unique']+d['bounce_unique']==g['unique_blobs'] and d['direct_bytes']+d['bounce_bytes']==g['h2d_bytes'] and
             (d['retained_ranges'],d['retained_bytes'])==signature[:2],'Direct/bounce/physical partition differs')
        if work:need(all(g[k]>0 for k in ('layer_windows','kernel_windows','staged_entries','unique_blobs','h2d_bytes'))and d['direct_unique']>0 and d['direct_bytes']>0,'Positive real graded GPU/direct work missing')
        last=(g['cumulative_entries'],g['cumulative_bytes'])
    setup=final['initial_setup'];need((setup['capacity'],setup['maxT'],setup['maxK'])==(24,4,6) and
        integer(setup['stride'],1)>0 and setup['device_bytes']==24*setup['stride'] and setup['host_bounce_bytes']==2*setup['stride'],'Actual stage setup geometry differs')
    end=final['exit'];need(end['drained']==1 and end['fallbacks']==0 and end['kernel_windows']==end['layer_windows'] and
        0<end['unique_blobs']<=end['staged_entries']==last[0] and end['h2d_bytes']==last[1]>0,'Actual stage exit/drain differs')

def report_check(report, direct):
    need(report['schema']==('retained-direct-two-fresh-frozen-waves-v1'if direct else'strata-two-fresh-frozen-waves-v1') and
        report.get('finished') is True and report.get('passed') is True and report.get('status')=='measurement-complete' and
        report.get('cleanup_validated') is True and report.get('protected_files_unchanged') is True,'Completed passing report required')
    need(report.get('configured_context')==1048576 and report.get('wave_count')==2 and report.get('same_owned_process_required') is True and
        report.get('no_retries') is True and report.get('no_RESET_or_maintenance_wire') is True and report.get('frozen_grading_changed') is False,'Frozen two-wave protocol differs')
    need(report.get('exact_seed_tokens')==PREFIX and report.get('exact_seed_ids_sha256')==FROZEN[0][2] and
        report['baseline']['primary_REPORT_sha256']==PRIMARY,'Frozen seed/primary provenance differs')
    close(report['baseline']['ttft_seconds'],HISTORICAL_TTFT);close(report['baseline']['weighted_ID_intervals_s'],HISTORICAL_RATE)
    if direct:need(report.get('registered_ram_cap_MiB')==196608 and report.get('overlap_mode') is False,'Candidate selected variant differs')
    pins=report['protected_files_before'];need(isinstance(pins,dict)and bool(pins)and all(isinstance(k,str)and re.fullmatch('[0-9a-f]{64}',v)for k,v in pins.items()),'Invalid stored protected pin map')
    image=norm(report['owned_identity']['image']);digest=DIRECT_EXE if direct else STOCK_EXE
    need([v for k,v in pins.items()if norm(k)==image]==[digest] and PRIMARY in pins.values() and STOCK_SOURCE in pins.values() and
        (not direct or DIRECT_SOURCE in pins.values()),'Stored engine/source/primary pins differ')
    cleanup(report)
    profile=report['expert_profile'];need(profile.get('initial_sha256')==PROFILE and profile.get('per_run_copy_required') is True and
        profile.get('learning_may_update_only_this_copy') is True and norm(profile['path'])==norm(option(report['effective_native_args'],'--expert-profile')),
        'Private fresh66 profile evidence failed')
    parity=[{'case':n,'tokens':t,'input_ids_sha256':d,'equal':True}for n,t,d in FROZEN[1:]]
    need(report['tokenizer_parity']==parity,'Complete frozen tokenizer parity differs')
    waves=report['waves'];need(len(waves)==2,'Exactly two complete waves required');metrics=[]
    for index,w in enumerate(waves,1):
        need(w['index']==index and w.get('measurement_complete') is True and w.get('status')=='measurement-complete' and
            w.get('fresh_seed_zero_reuse_proven') is True and w.get('no_new_profile_or_engine') is True and
            w.get('same_owned_identity')==report['owned_identity'] and w.get('input_parity')==parity and
            w.get('whole_process_cleanup_passed') is True and w.get('whole_run_protected_files_unchanged') is True,'Wave identity/state/protection incomplete')
        rows=w['rows'];accuracy=w['accuracy'];need(len(rows)==14 and len(accuracy)==13,'Full14 inputs/13 grades required')
        for i,(r,f)in enumerate(zip(rows,FROZEN)):
            row_check(r,f,seed=i==0)
            if i:grade(r,accuracy[i-1],code=i>=11,v2=i==13)
        need(w.get('original12_joint_score')==12 and w.get('corrected_v2_passed') is True and w.get('all13_quality_passed') is True,'Full13 aggregate failed')
        n=sum(r['emitted_token_ids']-1 for r in rows[1:13]);seconds=sum(r['generation_seconds']for r in rows[1:13]);rate=n/seconds;ttft=rows[0]['ttft_seconds']
        need(w['original12_actual_ID_intervals']==n,'Stored actual interval count differs')
        close(w['original12_observed_generation_seconds'],seconds);close(w['weighted_original12_observed_ID_intervals_s'],rate);close(w['fresh_seed_TTFT_seconds'],ttft)
        need(w.get('v2_excluded_from_primary_rate') is True and w.get('EOS_and_reasoning_IDs_included') is True,'Primary rate scope differs')
        need(w.get('TTFT_target_passed') is (ttft<HISTORICAL_TTFT) and w.get('original12_decode_target_passed') is (rate>HISTORICAL_RATE) and
            w.get('both_speed_targets_and_all13_quality_passed') is (ttft<HISTORICAL_TTFT and rate>HISTORICAL_RATE),'Stored historical target flags differ')
        metrics.append({'wave':index,'original12_actual_ID_intervals':n,'original12_generation_seconds':seconds,'original12_ID_intervals_s':rate,'fresh_seed_TTFT_seconds':ttft})
    native_phase(waves[0]['native_phase_evidence'],waves[1]['native_phase_evidence'],direct)
    row_check(report['interlude'],INTERLUDE,interlude=True);grade(report['interlude'],report['interlude_accuracy'])
    ii=report['interlude_input'];need(ii=={'case':'filtered-decimal-revenue','tokens':164,'input_ids_sha256':INTERLUDE[2],'tokenizer_equal':True,'prefix_pin':0},'Unrelated frozen interlude identity differs')
    if direct:direct_rows(report)
    targets=all(w['both_speed_targets_and_all13_quality_passed']for w in waves)
    need(report.get('both_waves_native_phase_admission_passed') is True and report.get('both_waves_speed_targets_and_all13_quality_passed') is targets and
        report.get('both_waves_speed_quality_and_native_rearm_admission_passed') is targets,'Final combined historical/native flag is inconsistent')
    return metrics

def contrast(baseline,candidate):
    a,b=copy.deepcopy(baseline['effective_native_args']),copy.deepcopy(candidate['effective_native_args'])
    need(norm(option(a,'--expert-profile'))!=norm(option(b,'--expert-profile')),'Runs shared one learned profile')
    for args in (a,b):
        for name,value in (('--max-context','1048576'),('--cpu-threads','16'),('--prefill-chunk','65280'),('--dspark','on'),('--dspark-block','3'),('--dspark-conf','0.65')):
            need(option(args,name)==value,'Effective fixed capacity/CPU/DSpark geometry differs')
        args[args.index('--expert-profile')+1]='<private-profile>'
    need(a==b,'Effective model/native args differ beyond private profile')
    a,b=copy.deepcopy(baseline['effective_native_env']),copy.deepcopy(candidate['effective_native_env'])
    for env in (a,b):
        need(bool(env.pop('STRATA_DS4_SLOT_DIR',None)),'Unique disabled slot directory missing')
        for name in ('STRATA_DS4_SLOT_KEEP','STRATA_DS4_SLOT_CAP','STRATA_DS4_SLOT_GB','STRATA_DS4_AUTO_PIN'):
            need(env.get(name)=='0','Conversation disk/auto-pin settings differ')
        need(env.get('STRATA_DS4_REUSE')=='1' and env.get('DS4_PREFILL_REPEAT_REARM')=='1' and env.get('DS4_PREFILL_REPEAT_MIN_TOKENS')=='32768','Exact repeat/reuse settings differ')
        need(env.pop('DS4_GPU_MISS_OVERLAP','0')=='0','Overlap is outside selected V3 contrast')
    for flag in ('DS4_GPU_MISSES','DS4_GPU_MISS_DIRECT'):
        need(a.pop(flag,'0')=='0' and b.pop(flag,None)=='1','GPU-miss/direct contrast differs')
    need(a.get('DS4_PREFILL_REGISTER_RAM_MB')=='98304' and b.get('DS4_PREFILL_REGISTER_RAM_MB')=='196608','Declared96/192GiB contrast differs')
    b['DS4_PREFILL_REGISTER_RAM_MB']='98304';need(a==b,'Undeclared effective environment contrast')

def compare(baseline,candidate):
    """Validate stored proof; return raw paired metrics and a fixed >=5% gate.

    A valid but slower experiment returns paired_rate_gate_passed=False.
    Invalid/incomplete evidence raises ValueError; no replacement or regrade.
    """
    try:
        old=report_check(baseline,False);new=report_check(candidate,True);contrast(baseline,candidate)
        pairs=[]
        for a,b in zip(old,new):
            pairs.append({'wave':a['wave'],'baseline':a,'candidate':b,'candidate_over_baseline_rate':b['original12_ID_intervals_s']/a['original12_ID_intervals_s'],
                'candidate_over_baseline_TTFT':b['fresh_seed_TTFT_seconds']/a['fresh_seed_TTFT_seconds'],
                'at_least_five_percent_decode_gain':b['original12_ID_intervals_s']>=a['original12_ID_intervals_s']*REQUIRED_RATIO})
        paired=all(p['at_least_five_percent_decode_gain']for p in pairs)
        combined=candidate['both_waves_speed_quality_and_native_rearm_admission_passed']
        return {'reports_validated':True,'required_ratio':REQUIRED_RATIO,'paired_waves':pairs,
                'paired_rate_gate_passed':paired,'candidate_combined_target_flag_passed':combined,
                'promotion_comparison_admission_passed':paired and combined,
                'primary_rate':'Each wave separately: sum(original12 actual IDs-1)/sum(original12 last-first ID seconds); v2 excluded',
                'quality':'Both stored original12+v2 first grades, natural EOS, exact full reuse and fresh0 seeds passed',
                'candidate_historical_speed_quality_and_native_admission_required_for_promotion':True,
                'semantic_regrading_performed':False,'actual_file_payload_or_process_attestation_performed':False,
                'Stop3_and_production_publication_not_qualified_here':True}
    except Refused:
        raise
    except (KeyError,TypeError,IndexError,OverflowError,struct.error) as error:
        raise Refused('Malformed or incomplete stored qualification evidence') from error
