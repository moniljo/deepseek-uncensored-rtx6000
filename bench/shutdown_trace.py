"""Bounded owned shutdown observation; no native or inference changes."""
import json
import time
from pathlib import Path

GRACE_SECONDS=60


def close_owned(owner,server,path=None,clock=time.perf_counter):
    """Call the unchanged owner.close, observing the actual child's escalation.

    Only the imported frontend's quit-wait constant is temporarily changed.
    The original close still writes STOP/QUIT/flush/EOF and owns all waits.
    Neither return codes nor OwnedV3 evidence are rewritten.
    """
    need=lambda ok,msg:None if ok else (_ for _ in ()).throw(RuntimeError(msg))
    need(len(owner.processes)==len(owner.identities)==1,'One exact owned process required for shutdown trace')
    proc=owner.processes[0];identity=dict(owner.identities[0]);engine=owner.engine
    need(engine is not None and engine.proc is proc and proc.pid==identity['pid'],
         'Shutdown trace refuses a different process binding')
    old_grace=server.ENGINE_QUIT_WAIT_S
    need(type(old_grace)in (int,float)and old_grace in (20,60),'Expected frozen20/newpublic60 quit wait required')
    path=Path(path)if path is not None else Path(owner.output)/'SHUTDOWN.json'
    need(not path.exists(),'Shutdown evidence is exclusive')
    began=clock();calls=[];waits=[];in_engine_close=False
    result={'schema':'owned-graceful-shutdown-observation-v1','owned_identity':identity,
            'previous_quit_wait_seconds':old_grace,'applied_quit_wait_seconds':GRACE_SECONDS,
            'inference_or_generation_timing_changed':False,'native_returncodes_rewritten':False,
            'escalation_calls':calls,'actual_wait_calls':waits}
    saved={name:(name in proc.__dict__,getattr(proc,name))for name in ('terminate','kill','wait')}
    original_close=engine.close;had_close='close'in engine.__dict__
    def escalation(name):
        original=saved[name][1]
        def invoke(*args,**kwargs):
            calls.append({'method':name,'elapsed_seconds':clock()-began,'inside_actual_engine_close':in_engine_close})
            return original(*args,**kwargs)
        return invoke
    def wait(*args,**kwargs):
        started=clock();item={'timeout':kwargs.get('timeout',args[0]if args else None),
            'elapsed_before_call_seconds':started-began,'inside_actual_engine_close':in_engine_close}
        waits.append(item)
        try:
            value=saved['wait'][1](*args,**kwargs);item['returned_returncode']=value;return value
        except BaseException as error:
            item['error_type']=type(error).__name__;raise
        finally:item['wait_elapsed_seconds']=clock()-started
    def engine_close():
        nonlocal in_engine_close
        need(not result.get('actual_engine_close_called'),'Actual engine.close invoked more than once')
        result['actual_engine_close_called']=True;in_engine_close=True;started=clock()
        try:return original_close()
        except BaseException as error:
            result['engine_close_error_type']=type(error).__name__;raise
        finally:
            result['actual_engine_close_elapsed_seconds']=clock()-started
            result['actual_engine_close_returncode']=proc.poll();in_engine_close=False
    proc.terminate=escalation('terminate');proc.kill=escalation('kill');proc.wait=wait;engine.close=engine_close
    server.ENGINE_QUIT_WAIT_S=GRACE_SECONDS
    try:
        owner.close()
    except BaseException as error:
        result['owner_close_error_type']=type(error).__name__;raise
    finally:
        result['total_owned_close_elapsed_seconds']=clock()-began
        result['final_observed_returncode']=proc.poll()
        result['no_terminate_or_kill']=not calls
        result['natural_normal0_without_escalation']=bool(result.get('actual_engine_close_called')and
            result.get('actual_engine_close_returncode')==0 and result['final_observed_returncode']==0 and
            not calls and not result.get('engine_close_error_type')and not result.get('owner_close_error_type'))
        result['natural0_after_more_than_old20_seconds']=bool(result['natural_normal0_without_escalation']and
            result.get('actual_engine_close_elapsed_seconds',0)>20)
        result['natural0_quit_wait_exceeded_old20_seconds']=bool(result['natural_normal0_without_escalation']and
            any(w.get('inside_actual_engine_close')and w.get('timeout')==60 and w.get('wait_elapsed_seconds',0)>20
                for w in waits))
        for name,(had,original)in saved.items():
            if had:setattr(proc,name,original)
            else:delattr(proc,name)
        if had_close:engine.close=original_close
        else:delattr(engine,'close')
        server.ENGINE_QUIT_WAIT_S=old_grace
        result['method_bindings_and_module_constant_restored']=True
        with path.open('x',encoding='utf-8')as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return result
