"""Exact reviewed cancellation evidence/one-consumer STOP blocks, with public paths.

Only the imported runtime bindings are portable; the copied function/class
bodies remain source-identical to the private V6b controller.
"""
import copy
import re
import threading
import time
from types import SimpleNamespace
import native_support as B
import runtime
P,V=runtime.P,runtime.V
need=B.need
MODES=('decode-stop','setup-stop','active-prefill-stop','early-rearm-stop','initial-prefill-stop','setup-stop-twice')
COMMIT=re.compile(r"^ds4: experimental HELD workspace regrow committed47 buffers: C1024->65280 A1024->4096 AI128 retained;.*$",re.M)
READY=re.compile(r"^ds4: experimental repeat prefill rearm ready: generation(\d+) attempt(\d+), remaining(\d+) C(\d+) A(\d+) AI(\d+), epoch(\d+) issued(\d+) unchanged, pool(\d+) bytes, registration(\d+) bytes; ([\d.]+) s charged inside feed; next compute is existing chunk-major path$",re.M)
END=re.compile(r"^ds4: experimental chunk-major phase ended: C=(\d+), layers grown=(\d+) skipped=(\d+), .*RAM maps retained, ([\d.]+) s.*$",re.M)
ABORT=re.compile(r'^ds4: experimental repeat prefill rearm not activated: attempt(\d+) C(1024|65280), .*; stopped; valid current pool/layout retained$',re.M)
DELTA=re.compile(r'^ds4: experimental repeat prefill generation(\d+) work delta: cold (\d+) sets/(\d+) unpacks; Engram (\d+) fills/(\d+) hits; dense (\d+) fills/(\d+) hits; registration ranges(\d+) after (retention|retirement); sequence(\d+) retained$',re.M)

def log_since(path,offset):
    with path.open('rb') as stream:stream.seek(offset);raw=stream.read(2*1024*1024+1)
    need(len(raw)<=2*1024*1024,'Bounded cancellation log observation exceeded2MiB')
    return raw.decode('utf-8',errors='replace').replace('\r\n','\n')

R=SimpleNamespace(COMMIT=COMMIT,READY=READY,END=END,log_since=log_since)
def read_window(log,offset):return R.log_since(log,offset)

class SendOnly:
    def __init__(self,engine,event):self.engine,self.event=engine,event
    def send(self,line):
        with self.engine.wlock:
            self.engine.proc.stdin.write(line+'\n')
            self.engine.proc.stdin.flush()
F=SimpleNamespace(FaultControl=SendOnly)

def validate_recovery_seed(seed):
    need(seed['emitted_token_ids']==1 and seed['prompt_read']==131072 and seed['reused_tokens']==0,
         'Fresh exact NON_TARGET recovery seed failed')
    need(seed['native_done']['finish'] in ('stop','length'),
         'Fresh recovery seed did not finish normally')


class StopPinHistory(P.PinHistory):
    """One predeclared unregistered retry after the independently proved abort."""
    def __init__(self,cap,unregistered_retry_epoch=None):
        super().__init__(cap)
        self.allowed_gaps=set(()if unregistered_retry_epoch is None else (unregistered_retry_epoch,)if type(unregistered_retry_epoch)is int else unregistered_retry_epoch)
    def declare_gap(self,epoch):
        need(type(epoch)is int and epoch>0 and epoch not in self.allowed_gaps,'Explicit unique abort gap required')
        self.allowed_gaps.add(epoch)
    def feed(self,text):
        for line in text.splitlines():
            if P.RETAIN.search(line) and self.previous_retired and self.active is None:
                event=P.retained(line);old=self.previous_retired
                if event['epoch']!=old['epoch']:
                    need(event['epoch'] in self.allowed_gaps and event['epoch']==old['epoch']+1 and
                         event['issued']>=old['issued'] and all(event[k]==old[k]for k in ('ranges','bytes','fingerprint')),
                         'Undeclared registration/epoch gap after abort')
                    self.active=event;self.events.append(dict(kind='retained',**event));self.allowed_gaps.remove(event['epoch']);continue
            super().feed(line)


def phase_retry(text,generation,attempt,remaining):
    """Actual attempt accounting includes the predeclared interrupted phase."""
    q=READY.findall(text);d=DELTA.findall(text)
    need(len(q)==len(d)==1,'One actual successful retry ready/workdelta required')
    values=list(map(int,q[0][:10]));work=d[0]
    old,new=P.retired(text,'epoch') if P.RETIRE.search(text) else None,P.retained(text)
    need(values[:6]==[generation,attempt,remaining,65280,4096,128] and values[6]>0 and values[8]>0 and values[9]>0,
         'Actual retry generation/attempt/geometry differs')
    need(int(work[0])==generation and all(int(v)>0 for v in work[1:7]) and int(work[7])==new['ranges'] and
         work[8]=='retention' and int(work[9])==new['issued']>=values[7] and new['epoch']==values[6] and new['bytes']==values[9],
         'Retry work/retained physical ownership differs')
    need(text.index('repeat prefill rearm ready:')<text.index('GPU_MISS_PIN retained:')<text.index(' work delta:')<text.index('chunk-major phase ended:'),
         'Retry ready/retention/work/end order differs')
    ends=R.END.findall(text);need(len(ends)==1 and int(ends[0][0])==1024 and 'stopped' not in text and 'ds4: ERROR:' not in text,
                                'Successful retry did not finish compact/healthy')
    if old:need(old['epoch']==values[6] and old['issued']==values[7] and all(old[k]==new[k] for k in ('ranges','bytes','fingerprint')),
                'Retired/new registration identity differs')
    return dict(generation=generation,attempt=attempt,remaining=remaining,retained=new,retired=old,
                source_bound_zero_before_HELD=True,no_speed_or_full1M_claim=True)


def validate_decode_cancel(text,row,active):
    done=row.get('native_done') or {}
    need(row.get('STOP_written') is True and row.get('observed_IDs_before_STOP',0)>=16 and
         row.get('consumer_alive_at_STOP') is True and row.get('DONE_drained') is True and
         row.get('iterator_error_type') is None,'Actual active-GEN STOP trigger/DONE missing')
    need(done.get('finish')=='cancel' and done.get('generated')==len(row['token_ids'])>=row['observed_IDs_before_STOP'] and
         done.get('prompt_tokens')==row['input_tokens'] and row['token_ids'][-1]!=1,'Actual partial cancel accounting differs')
    need(not P.RETAIN.search(text) and not P.RETIRE.search(text) and 'ds4: ERROR:' not in text and 'ERR ' not in text,
         'Healthy decode STOP unexpectedly changed registration lifetime or errored')
    primary=V.gpu_answer(text,True);direct=P.direct_answer(text,primary,active,True)
    return dict(healthy_retained_decode_context_preserved=copy.deepcopy(active),actual_GPU_work=primary,direct=direct,
                cancelled_actual_IDs=len(row['token_ids']),hardware_inflight_instant_not_claimed=True)


def validate_phase_cancel(text,row,active,mode,expected_attempt=1,expected_retire_epoch=1):
    done=row.get('native_done') or {};old=P.retired(text,'epoch')
    need(all(old[k]==active[k] for k in ('ranges','bytes','fingerprint')) and old['epoch']==expected_retire_epoch and row.get('STOP_written') is True and
         row.get('DONE_drained') is True and row.get('iterator_error_type') is None,'Interrupted phase ownership/STOP differs')
    need(done.get('finish')=='cancel' and done.get('generated')==0 and row['token_ids']==[] and
         done.get('prompt_tokens')==row['input_tokens']==B.PREFIX_N,'Actual prefill cancel0 required')
    need(not P.RETAIN.search(text) and 'ds4: ERROR:' not in text and 'ERR ' not in text,
         'Cancelled phase newly retained pins or errored')
    if mode in ('setup-stop','early-rearm-stop'):
        abort=ABORT.findall(text)
        commits=0 if mode=='early-rearm-stop'else 1;geometry='1024'if mode=='early-rearm-stop'else'65280'
        need(len(R.COMMIT.findall(text))==commits and abort==[(str(expected_attempt),geometry)] and not READY.search(text) and not DELTA.search(text) and not R.END.search(text),
             'Setup STOP missed committed47-beforeactivation window')
        acquired=list(re.finditer(r'registered-copy setup: selected [^\r\n]*published (\d+) ranges/',text))
        if any(int(m.group(1))>0 for m in acquired):
            unreg=list(re.finditer(r'registered-copy unregister: [^\r\n]*; 0 ranges/[0-9.]+ MiB still owned;',text))
            need(unreg and unreg[-1].start()>acquired[-1].start() and unreg[-1].start()<text.index('rearm not activated:'),
                 'Setup STOP did not retire newly acquired prefill registrations')
    else:
        ready=READY.findall(text);d=DELTA.findall(text);ends=R.END.findall(text)
        need(len(ready)==len(d)==len(ends)==1 and list(map(int,ready[0][:2]))==[1,1] and
             row.get('observed_prompt_progress',[0,0])[0]>0 and row['observed_prompt_progress'][1]==B.PREFIX_N and
             int(d[0][0])==1 and int(d[0][1])>0 and int(d[0][2])>0 and int(d[0][7])==0 and d[0][8]=='retirement' and
             int(ends[0][0])==65280 and 'stopped' in text,'Active prefill STOP/work/actual retirement/no-shrink proof missing')
        unreg=list(re.finditer(r'registered-copy unregister: [^\r\n]*; 0 ranges/[0-9.]+ MiB still owned;',text))
        need(unreg and unreg[-1].start()>text.index('repeat prefill rearm ready:') and
             unreg[-1].start()<text.index('chunk-major phase ended:'),'Active cancelled prefill physical unregister-tozero missing')
    return dict(cancelled_new_phase_did_not_retain=True,old_decode_owner_physically_retired=old,
                source_bound_no_unpublished_registration_after_abort=True,scope='NON_TARGET cancellation/recovery, no speed claim')


def validate_initial_cancel(text,row):
    done=row.get('native_done')or{}
    need(row.get('STOP_written')is True and row.get('consumer_alive_at_STOP')is True and row.get('DONE_drained')is True and
         row.get('iterator_error_type')is None and done.get('finish')=='cancel'and done.get('generated')==0 and
         row['token_ids']==[]and done.get('prompt_tokens')==row['input_tokens']==B.PREFIX_N,'Actual initial-prefill cancel0 required')
    pp=row.get('observed_prompt_progress');need(isinstance(pp,list)and len(pp)==2 and 0<pp[0]<pp[1]==B.PREFIX_N,'Actual initial prompt work required')
    cold=re.findall(r'bounded cold F16 retired [0-9.]+ MiB: (\d+) sets/\d+ batches/\d+ buckets, (\d+) expert unpacks',text)
    ends=R.END.findall(text)
    need(len(cold)==len(ends)==1 and all(int(x)>0 for x in cold[0])and int(ends[0][0])==65280 and 'stopped'in text and
         not P.RETAIN.search(text)and not P.RETIRE.search(text)and not READY.search(text)and not DELTA.search(text)and
         re.search(r'registered-copy unregister: [^\r\n]*; 0 ranges/0\.0 MiB still owned;',text)and
         'ds4: ERROR:'not in text and 'ERR 'not in text,'Initial phase work/physical0/no-retention state differs')
    return dict(initial_phase_actual_work=True,cancelled_new_phase_did_not_retain=True,physical_prefill_registration_zero=True,
                no_prior_NORMAL_or_DECODE_owner=True,no_normal_anchor_fabricated=True)


def interrupt(owner,server,ids,log,out,mode,deadline):
    """One iterator consumer. Observer never reads engine.lines/STDOUT."""
    engine=owner.engine;offset=log.stat().st_size;began=time.monotonic();limit=began+deadline
    result=dict(input_tokens=len(ids),input_ids_sha256=B.idsha(ids),token_ids=[],STOP_written=False,DONE_drained=False,
                iterator_error_type=None,actual_stdout_consumers=1,observer_queue_reads=0)
    lock=threading.Lock();cancel=threading.Event()
    sampling={'temperature':0,'seed':12345,'presence_penalty':0,'frequency_penalty':0,'repetition_penalty':1,
              'strata_checkpoint':True,'strata_prefix':{'tokens':0 if mode=='decode-stop' else B.PREFIX_N}}
    engine.last={};engine.raw_done=None
    def consume():
        try:
            for token in engine.generate(ids,4096 if mode=='decode-stop' else 1,sampling,cancel):
                if token is not None:
                    need(type(token)is int and token>=0,'Invalid native ID')
                    with lock:result['token_ids'].append(token)
            need(engine.raw_done is not None,'Interrupted GEN has no actual DONE')
            result['native_done']=B.parse_done(server,engine.raw_done);result['raw_DONE']=engine.raw_done;result['DONE_drained']=True
        except BaseException as error:result['iterator_error_type']=type(error).__name__
    thread=threading.Thread(target=consume,name='owned-direct-stop-stdout',daemon=True);thread.start()
    try:
        while thread.is_alive() and time.monotonic()<limit:
            text=read_window(log,offset)
            with lock:count=len(result['token_ids'])
            progress=engine.progress
            trigger=(mode=='decode-stop' and count>=16) or (mode=='setup-stop' and R.COMMIT.search(text) and not READY.search(text)) or (
                mode=='active-prefill-stop' and READY.search(text) and isinstance(progress,tuple) and len(progress)==2 and
                type(progress[0])is int and 0<progress[0]<progress[1]==len(ids))
            trigger=trigger or (mode=='early-rearm-stop'and P.RETIRE.search(text)) or (mode=='initial-prefill-stop'and
                isinstance(progress,tuple)and len(progress)==2 and type(progress[0])is int and 0<progress[0]<progress[1]==len(ids))
            if trigger:
                need(thread.is_alive(),'GEN ended before STOP write')
                result.update(observed_IDs_before_STOP=count,consumer_alive_at_STOP=True,
                              observed_prompt_progress=list(progress)if isinstance(progress,tuple)else None,
                              STOP_write_elapsed_seconds=time.monotonic()-began)
                F.FaultControl(engine,lambda _:None).send('STOP') # send-only; never acquires stdout queue
                result['STOP_written']=True;break
            time.sleep(.005)
        thread.join(max(0,limit-time.monotonic()))
        if thread.is_alive():
            result['deadline_expired']=True;B.save(out/'INCOMPLETE-interrupted-GEN.json',result)
            owner.kill_owned();thread.join(5);raise B.DeadlineExpired('Owned interrupted GEN did not drain')
        result.update(wall_seconds=time.monotonic()-began,output_ids_sha256=B.idsha(result['token_ids']))
        # stderr is independent of DONE delivery; no canonical timestamps change.
        until=time.monotonic()+1
        while time.monotonic()<until:
            text=read_window(log,offset)
            if mode!='decode-stop' or (V.GPU_ANSWER.search(text) and P.ANSWER.search(text)):break
            time.sleep(.005)
        return result,text,offset
    finally:
        if thread.is_alive():owner.kill_owned();thread.join(5)


PREFIX_N=B.PREFIX_N

def probe_inputs(prefix, tests, count):
    if count is None:
        return prefix, tests
    need(1024 <= count < PREFIX_N and len(prefix) == PREFIX_N, "Invalid NON-TARGET probe prefix")
    independent = {"filtered-decimal-revenue", "python-output", "code-filtered-cents"}
    selected = [(case, prefix[:count] + ids[PREFIX_N:], is_v2) for case, ids, is_v2 in tests if case["id"] in independent and not is_v2]
    need(len(selected) == 3 and all(ids[:count] == prefix[:count] for _, ids, _ in selected), "Probe independent-task selection differs")
    return prefix[:count], selected

