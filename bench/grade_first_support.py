"""Exact canonical FIRST-grade/isolated worker invocation."""
import json
import subprocess
import sys
import native_support as B
from types import SimpleNamespace
from strict_json_support import strict_json
S=SimpleNamespace(strict_json=strict_json)
need=B.need

def grade_first(owner, measure, case, ids, decode, cases, worker, deadline, reused, report, save):
    row = measure.generate(owner.stage, ids, B.MAXIMUM, deadline, reused)
    report["rows"].append(row)
    need(128822 in row["token_ids"], "Actual reasoning close missing")
    final = decode(row["token_ids"][row["token_ids"].index(128822) + 1:])
    grade = cases.grade(case, final)
    if case["kind"] == "code":
        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(worker)],
                                input=json.dumps({"case": case["id"], "text": final}), text=True,
                                capture_output=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        need(result.returncode == 0 and len(result.stdout) < 16000, "Bounded original code worker failed")
        execution = S.strict_json(result.stdout)
        grade.update(execution=execution, semantic_correct=execution.get("semantic_correct"),
                     format_compliant=execution.get("format_compliant"), all11_vectors_passed=(
                         execution.get("safe_policy_passed") is True and
                         execution.get("tests_passed") == execution.get("tests_total") == 11))
    row.update(case=case["id"], final_text=final, grade=grade,
               quality_passed=bool(final.strip() and grade.get("semantic_correct") is True and
                                   grade.get("format_compliant") is True and
                                   (case["kind"] != "code" or grade.get("all11_vectors_passed") is True)))
    save()
    return row
