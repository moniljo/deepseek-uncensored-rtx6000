"""Exact frozen first-result grading/metrics/cleanup functions."""
import json
import subprocess
import sys
import native_support as B
need=B.need

def fresh_seed(seed,expected_hash):
    need(seed["prompt_tokens"]==B.PREFIX_N and seed["input_ids_sha256"]==expected_hash and
         type(seed["reused_tokens"])is int and seed["reused_tokens"]==0 and
         seed["native_done"]["finish"]in ("stop","length"),
         "Fresh seed reused context, was cancelled, or input changed; no retry/fresh-wave claim")


def wave_metrics(wave,baseline):
    rows=wave["rows"]
    need(len(rows)==14 and len(wave["accuracy"])==13,"Complete frozen14inputs/13graded results required")
    originals=rows[1:13]
    need(all(r["generation_seconds"]and r["generation_seconds"]>0 for r in originals),"Every original12 needs positive actual ID intervals")
    n=sum(r["emitted_token_ids"]-1 for r in originals)
    seconds=sum(r["generation_seconds"]for r in originals)
    native_seconds=sum(r["native_decode_ms"]/1000 for r in originals)
    need(seconds>0 and native_seconds>0,"Invalid original12 timing denominator")
    original_pass=sum(a["joint_passed"]for a in wave["accuracy"][:12])
    v2=wave["accuracy"][12]["joint_passed"]
    ttft=rows[0]["ttft_seconds"]
    return {"original12_actual_ID_intervals":n,"original12_observed_generation_seconds":seconds,
            "weighted_original12_observed_ID_intervals_s":n/seconds,
            "weighted_original12_native_ID_intervals_s":n/native_seconds,
            "original12_joint_score":original_pass,"corrected_v2_passed":v2,
            "all13_quality_passed":original_pass==12 and v2,
            "fresh_seed_TTFT_seconds":ttft,"actual_seed_reused":rows[0]["reused_tokens"],
            "TTFT_target_passed":ttft<baseline["ttft_seconds"],
            "original12_decode_target_passed":n/seconds>baseline["weighted_ID_intervals_s"],
            "both_speed_targets_and_all13_quality_passed":ttft<baseline["ttft_seconds"]and n/seconds>baseline["weighted_ID_intervals_s"]and original_pass==12 and v2,
            "v2_excluded_from_primary_rate":True,"EOS_and_reasoning_IDs_included":True}


def grade_row(row,case,is_v2,original,decode,worker):
    need(128822 in row["token_ids"],"Actual reasoning close missing")
    final=decode(row["token_ids"][row["token_ids"].index(128822)+1:])
    row.update(final_text=final,actual_thinking_close_observed=True,corrected_v2=is_v2)
    grade=original.grade(case,final)
    if case["kind"]=="code":
        result=subprocess.run([sys.executable,"-I","-S","-B",str(worker)],input=json.dumps({"case":case["id"],"text":final}),
                              text=True,capture_output=True,timeout=5,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        need(result.returncode==0 and len(result.stdout)<16000,"Bounded frozen code worker failed")
        execution=json.loads(result.stdout)
        grade.update(semantic_correct=execution["semantic_correct"],format_compliant=execution.get("format_compliant",False),execution=execution)
        vectors=execution.get("safe_policy_passed")is True and execution.get("tests_passed")==execution.get("tests_total")==11
        if is_v2:vectors=vectors and execution.get("tests_run")==11 and execution.get("status")=="executed"
        grade["all11_vectors_passed"]=vectors
        if not vectors:grade["semantic_correct"]=False
    return {"case":row["name"],"corrected_v2":is_v2,"grade":grade,
            "joint_passed":bool(final.strip()and grade.get("semantic_correct")is True and grade.get("format_compliant")is True)}


def validate_cleanup(cleanup,identity):
    need(cleanup.get("cleanup_proof_version")==3 and cleanup.get("passed")is True and cleanup.get("owned_job_closed")is True and
         cleanup.get("owned_job_empty_proven")is True and cleanup.get("owned_job_active_processes_before_close")==0 and cleanup.get("errors")==[],"V3 exact Job cleanup unproven")
    held=cleanup.get("separate_process_handle_checks",[])
    need(len(held)==1 and held[0].get("recorded_identity")==identity and held[0].get("separately_held_handle_signaled")is True and
         held[0].get("wait_result")==0 and held[0].get("pid_absence_claimed")is False and cleanup.get("pid_absence_claimed")is False,"Exact independently held identity not signaled")
    need(cleanup.get("recorded_processes")==[identity] and cleanup.get("popen_metadata_not_exit_proof")==[{"pid":identity["pid"],"returncode":0}] and
         cleanup.get("graceful_engine_close_error_type")is None and (cleanup.get("resources")or{}).get("sampler_stopped")is True and
         (cleanup.get("resources")or{}).get("error_type")is None,"Normal native0/graceful/sampler cleanup required")
