"""Unchanged V6b recovery observer bodies; public runtime supplies the same pin policy."""
import math
import re
from runtime import P

need=P.need
BEGIN=re.compile(r'^ds4: GPU_MISS_RECOVERY begin: attempt=(\d+) epoch=(\d+) issued=(\d+) C=(\d+) A=(\d+) AI=(\d+) pool_bytes=(\d+) registration_ranges=0 registration_bytes=0$',re.M)
LAYOUT=re.compile(r'^ds4: GPU_MISS_RECOVERY layout: C=(\d+) A=(\d+) AI=(\d+) layers_grown=(\d+) skipped=(\d+) pool_before_bytes=(\d+) pool_after_bytes=(\d+) free_bytes=(\d+) stopped=(0|1)$',re.M)
COMPACT=re.compile(r'^ds4: GPU_MISS_RECOVERY compact: attempt=(\d+) epoch=(\d+) C_before=(\d+) A_before=(\d+) C_after=(\d+) A_after=(\d+) AI=(\d+) pool_before_bytes=(\d+) pool_after_bytes=(\d+) last_healthy_pool_bytes=(\d+) compact_ms=([0-9.]+)$',re.M)
READY=re.compile(r'^ds4: GPU_MISS_RECOVERY ready: attempt=(\d+) epoch=(\d+) issued=(\d+) C=(\d+) A=(\d+) AI=(\d+) pool_bytes=(\d+) ranges=(\d+) bytes=(\d+) fingerprint=([0-9a-f]{16}) tokens_unchanged=1 checkpoints_unchanged=1 sequence_unchanged=1 streams_drained=1 compact_ms=([0-9.]+) registration_ms=([0-9.]+) elapsed_ms=([0-9.]+)$',re.M)
RESERVE=re.compile(r'^ds4: GPU_MISS_RECOVERY_RESERVE attempt=(\d+) epoch=(\d+) pool_before_bytes=(\d+) pool_after_bytes=(\d+) target_cap_bytes=(\d+) reserve_bytes=(\d+) shrink_completed=(0|1) stopped=(0|1)$',re.M)
RESERVE_BYTES=536870912

def direct_no_bounce(row):
    g=row['native_GPU_miss_answer'];d=row['native_GPU_miss_direct_answer']
    need(g['fallbacks']==0 and all(type(g[k])is int and g[k]>0 for k in
        ('layer_windows','kernel_windows','staged_entries','unique_blobs','h2d_bytes')),
        'Positive GPU miss work and zero CPU-expert fallback required')
    need(d['direct_unique']>0 and d['direct_bytes']>0 and d['bounce_unique']==d['bounce_bytes']==0 and
         d['direct_unique']==g['unique_blobs'] and d['direct_bytes']==g['h2d_bytes'],
         'Postcancel direct work must not use bounce copies')

def recovered(text,initial_owner,active_owner,expected_epoch=2,expected_attempt=1,before_geometry=(65280,4096),allow_unknown_healthy=False):
    text=text.replace('\r\n','\n')
    need('GPU_MISS_RECOVERY declined:'not in text and 'GPU_MISS_PIN retirement_failed:'not in text and
         'ds4: ERROR:'not in text and 'ERR 'not in text,'Declined/terminal recovery is not a successful repair')
    b,l,c,r=BEGIN.findall(text),LAYOUT.findall(text),COMPACT.findall(text),READY.findall(text)
    need(len(b)==len(l)==len(c)==len(r)==1,'One actual recovery begin/layout/compact/ready required')
    layout=list(map(int,l[0]))
    b=list(map(int,b[0]));compact_ms=float(c[0][10]);c=list(map(int,c[0][:10]));raw=r[0];r=list(map(int,raw[:9]))
    compact_ready_ms,registration_ms,elapsed=map(float,raw[10:13])
    need(all(math.isfinite(v)and v>=0 for v in(compact_ms,compact_ready_ms,registration_ms,elapsed))and
         compact_ms==compact_ready_ms and elapsed+0.01>=compact_ms+registration_ms,'Invalid native recovery CPU-wall durations')
    issued_ok=b[2]>0 or (b[2]==0 and allow_unknown_healthy and initial_owner is None and c[9]==0)
    need(b[0]==c[0]==r[0]==expected_attempt and b[1]==c[1]==r[1]==expected_epoch and b[2]==r[2] and issued_ok,
         'Recovery attempt/lease/issued state differs')
    need(tuple(before_geometry)in ((65280,4096),(1024,1024)) and b[3:6]==[*before_geometry,128] and c[2:4]==list(before_geometry) and c[4:7]==[1024,1024,128] and
         r[3:6]==[1024,1024,128],'Exact consumed-large to compact geometry required')
    anchor=c[9];cap=max(0,anchor-RESERVE_BYTES)
    need(b[6]==c[7]>0 and r[6]==c[8]>0,'Actual pool before/after fields disagree')
    if anchor:
        need(anchor>RESERVE_BYTES and 99*anchor<=100*c[8] and c[8]<=cap,
             'Known NORMAL anchor pool must stay99percent and leave the fixed512MiB reserve')
    else:
        need(allow_unknown_healthy and initial_owner is None,'Unknown initial anchor requires explicit initial-phase mode')
    reserve_rows=RESERVE.findall(text)
    need(len([line for line in text.splitlines()if 'GPU_MISS_RECOVERY_RESERVE 'in line])==len(reserve_rows),
         'Malformed reserve-shrink telemetry is not ignored')
    shrink_required=anchor==0 or c[7]>cap
    need(len(reserve_rows)==int(shrink_required),'Missing/unexpected checked reserve shrink record')
    reserve=None;layout_before=c[7]
    if shrink_required:
        q=list(map(int,reserve_rows[0]));reserve=dict(zip(('attempt','epoch','pool_before_bytes','pool_after_bytes','target_cap_bytes','reserve_bytes','shrink_completed','stopped'),q))
        need(q[:3]==[expected_attempt,expected_epoch,c[7]] and 0<q[3]<=q[2] and q[4:]==[cap,RESERVE_BYTES,1,0],
             'Checked reserve shrink did not complete with exact cap/owner state')
        if anchor:need(q[3]<=cap,'Reserve shrink remained above cap')
        need(text.index('GPU_MISS_RECOVERY begin:')<text.index('GPU_MISS_RECOVERY_RESERVE ')<text.index('GPU_MISS_RECOVERY layout:'),
             'Reserve shrink must precede normal compact publication')
        layout_before=q[3]
    need(layout[:3]==[1024,1024,128] and layout[5]==layout_before and layout[6]==c[8] and layout[8]==0 and c[8]>=layout_before,
         'Actual normal layout restoration disagrees with compact/pool or stopped')
    keep=P.retained(text)
    need(active_owner==keep and keep['epoch']==expected_epoch and keep['issued']==r[2] and
         (keep['ranges'],keep['bytes'],keep['fingerprint'])==(r[7],r[8],raw[9]) and
         (initial_owner is None or all(keep[k]==initial_owner[k]for k in('ranges','bytes','fingerprint'))),
         'Recovery publication changed physical selection or lease state')
    order=[text.index(s)for s in('GPU_MISS_RECOVERY begin:','GPU_MISS_RECOVERY layout:','GPU_MISS_RECOVERY compact:',
                               'GPU_MISS_RECOVERY ready:','GPU_MISS_PIN retained:')]
    need(order==sorted(order),'Recovery compact/ready/publication order differs')
    relevant=[line for line in text.splitlines()if 'GPU_MISS_RECOVERY 'in line or 'GPU_MISS_PIN 'in line]
    need(len(relevant)==5 and LAYOUT.fullmatch(relevant[1])and READY.fullmatch(relevant[3])and P.RETAIN.fullmatch(relevant[4]),
         'Ready must be followed by the matching actual DECODE publication')
    return dict(passed=True,attempt=r[0],epoch=r[1],issued_unchanged=r[2],
        geometry_before=dict(C=b[3],A=b[4],AI=b[5]),geometry_after=dict(C=r[3],A=r[4],AI=r[5]),
        pool_before_bytes=c[7],pool_after_bytes=c[8],saved_healthy_pool_bytes=anchor,pool_restore_ratio=c[8]/anchor if anchor else None,
        normal_anchor_known=bool(anchor),fixed_registration_reserve_bytes=RESERVE_BYTES,known_pool_cap_bytes=cap if anchor else None,
        actual_checked_reserve_shrink=reserve,unknown_initial_targets_source_checked_before_READY=anchor==0,
        normal_layout=dict(C=layout[0],A=layout[1],AI=layout[2],layers_grown=layout[3],skipped=layout[4],free_bytes=layout[7],stopped=False),
        actual_retained_owner=keep,native_recovery_elapsed_ms=elapsed,
        native_compact_CPU_wall_ms=compact_ms,native_registration_CPU_wall_ms=registration_ms,
        source_checked_tokens_checkpoints_sequence_unchanged=True,streams_drained_before_publication=True)


def same_normal_anchor(first,second):
    need(first['normal_anchor_known']is True and second['normal_anchor_known']is True and
         first['saved_healthy_pool_bytes']==second['saved_healthy_pool_bytes']>0 and
         first['known_pool_cap_bytes']==second['known_pool_cap_bytes'],
         'Repeated recovery ratcheted the saved NORMAL anchor')
    return True

def matching_speed(before,after):
    for key in ('prompt_tokens','input_ids_sha256','requested_output','requested_prefix_tokens'):
        need(before[key]==after[key],'BEFORE and FIRST-short fixed input/settings differ')
    need(before['requested_output']==4096 and before['requested_prefix_tokens']==0,'Matching short cap/pin differs')
    rates=[]
    for row in (before,after):
        need(row['quality_passed']is True and row['natural_eos']is True and row['token_ids'][-1]==1 and
             row['native_done']['finish']=='stop'and row['emitted_token_ids']==len(row['token_ids'])==row['native_generated_tokens']>1,
             'Matching FIRST result must finish naturally with actualID accounting')
        ex=row['grade']['execution'];need(ex['safe_policy_passed']is True and ex['tests_passed']==ex['tests_total']==11,
                                         'Matching code task must retain all11 unchanged safe vectors')
        if row.get('case')=='code-account-balance':
            need(ex.get('status')=='executed'and ex.get('tests_run')==11,'Matching corrected-v2 code must actually execute all11 vectors')
        seconds=row['generation_seconds'];need(type(seconds)in(int,float)and math.isfinite(seconds)and seconds>0,
                                               'ActualID interval duration invalid')
        rates.append((row['emitted_token_ids']-1)/seconds);direct_no_bounce(row)
    ratio=rates[1]/rates[0]
    return dict(passed=ratio>=0.95,minimum_ratio=0.95,actual_ratio=ratio,
        before_actual_ID_intervals_s=rates[0],FIRST_after_actual_ID_intervals_s=rates[1],
        before_IDs=before['emitted_token_ids'],FIRST_after_IDs=after['emitted_token_ids'],
        before_output_ids_sha256=before['output_ids_sha256'],FIRST_after_output_ids_sha256=after['output_ids_sha256'],
        output_IDs_identical=before['token_ids']==after['token_ids'],
        before_TTFT_seconds=before['ttft_seconds'],FIRST_after_TTFT_seconds=after['ttft_seconds'],
        recovery_cost_not_removed_from_TTFT=True,no_retries_or_output_length_selection=True)
