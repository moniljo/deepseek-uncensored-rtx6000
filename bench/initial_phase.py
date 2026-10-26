"""Exact canonical initial-only cache retirement observer."""
import math
import re
import native_support as B
need=B.need

def seed_telemetry(text,index,previous=None):
    """Actual stderr evidence only; initial cumulative != repeat generation delta."""
    result={"admission_passed":False,"wave":index,"errors":[],"native_generation0_delta_expected":False}
    try:
        cold=re.findall(r"bounded cold F16 retired [0-9.]+ MiB: (\d+) sets/(\d+) batches/(\d+) buckets, (\d+) expert unpacks",text)
        eg=re.findall(r"first-prefill Engram weight cache retired [0-9.]+ MiB: (\d+) fills/(\d+) slab hits/(\d+) original-shape GEMMs",text)
        dense=re.findall(r"first-prefill dense projection cache retired [0-9.]+ MiB: (\d+) matrix fills/(\d+) slab hits/(\d+) original-shape GEMMs",text)
        phases=re.findall(r"chunk-major phase ended: C=(\d+), layers grown=(\d+) skipped=(\d+),[^\r\n]+",text)
        result["soft_declines"]=re.findall(r"[^\r\n]*repeat prefill rearm (?:declined before lease|not activated):[^\r\n]*",text)
        ready=re.findall(r"repeat prefill rearm ready: generation(\d+) attempt(\d+), remaining(\d+) C(\d+) A(\d+) AI(\d+), epoch(\d+) issued(\d+) unchanged, pool(\d+) bytes, registration(\d+) bytes; ([0-9.]+) s charged inside feed; next compute is existing chunk-major path",text)
        deltas=re.findall(r"repeat prefill generation(\d+) work delta: cold (\d+) sets/(\d+) unpacks; Engram (\d+) fills/(\d+) hits; dense (\d+) fills/(\d+) hits; registration ranges(\d+) after retirement; sequence(\d+) retained",text)
        need(len(cold)==len(eg)==len(dense)==len(phases)==1,"Exactly one phase retirement/cache-family record required in each seed GEN window")
        need(int(phases[0][0])==1024,"Seed phase did not end with existing C1024 path")
        phase_lines=re.findall(r"[^\r\n]*chunk-major phase ended:[^\r\n]*",text)
        need(len(phase_lines)==1 and "stopped"not in phase_lines[0],"Stopped/partial seed phase cannot establish full fresh-prefill admission")
        cumulative={"cold_sets":int(cold[0][0]),"cold_unpacks":int(cold[0][3]),"eg_fills":int(eg[0][0]),"eg_hits":int(eg[0][1]),"eg_gemms":int(eg[0][2]),
                    "dense_fills":int(dense[0][0]),"dense_hits":int(dense[0][1]),"dense_gemms":int(dense[0][2])}
        need(all(v>0 for v in cumulative.values()),"Real positive cache-family work required")
        result.update(cumulative_retirement_counters=cumulative,phase_end_C=1024,phase_end_layers_grown=int(phases[0][1]),phase_end_layers_skipped=int(phases[0][2]))
        need(not result["soft_declines"],"Actual soft rearm decline is failed admission even if fallback grades pass")
        if index==1:
            need(not ready and not deltas,"Initial phase must not be mislabeled a repeated generation")
            result["initial_only_work_from_cumulative_counters"]=True
        else:
            need(index==2 and previous and previous.get("admission_passed"),"Valid initial-only retirement evidence required")
            need(len(ready)==len(deltas)==1,"Actual generation1 ready AND generation1 delta required inside second seed GEN")
            q=ready[0];r=deltas[0]
            need(tuple(map(int,q[:2]))==(1,1) and int(q[2])>=130560 and int(q[2])<=B.PREFIX_N and
                 tuple(map(int,q[3:6]))==(65280,4096,128)and int(q[6])>0 and int(q[7])>0 and int(q[8])>0,"Actual repeat generation/attempt/remaining/shape invalid")
            need(math.isfinite(float(q[10]))and float(q[10])>=0,"Invalid native maintenance duration")
            need(int(r[0])==1 and int(r[7])==0 and int(r[8])>=int(q[7]),"Repeat delta generation/retirement/sequence invalid")
            delta=dict(zip(("cold_sets","cold_unpacks","eg_fills","eg_hits","dense_fills","dense_hits"),map(int,r[1:7])))
            need(all(v>0 for v in delta.values()),"Repeat cache work was not actually exercised")
            old=previous["cumulative_retirement_counters"]
            need(all(cumulative[k]-old[k]==v for k,v in delta.items()),"Native repeat deltas do not match independently differenced cumulative counters")
            need(cumulative["dense_gemms"]>old["dense_gemms"]and cumulative["eg_gemms"]>old["eg_gemms"],"Actual repeated original-shape GEMMs missing")
            need(text.index("repeat prefill rearm ready:")<text.index("repeat prefill generation1 work delta:")<text.index("chunk-major phase ended:"),"Repeat ready/work/end ordering invalid")
            result.update(native_repeat_generation=1,native_repeat_attempt=1,native_rearm_ready={"remaining":int(q[2]),"C":int(q[3]),"A":int(q[4]),"AI":int(q[5]),
                          "epoch":int(q[6]),"issued":int(q[7]),"pool_bytes":int(q[8]),"registered_bytes":int(q[9]),"maintenance_seconds_inside_feed":float(q[10])},
                          actual_generation1_work_delta=delta,registration_ranges_after_retirement=0,sequence_after_prefill=int(r[8]))
        result["admission_passed"]=True
    except (RuntimeError,ValueError)as error:
        result["errors"].append(str(error))
    return result
