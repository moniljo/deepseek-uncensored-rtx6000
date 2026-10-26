"""Bounded CPU worker for the clarified v2 dictionary contract.

The original AST policy and test vectors are reused unchanged. Explicit run
counts distinguish rejected source from executed numeric tests.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import accuracy_cases_v2 as cases


def main():
    payload = sys.stdin.read(16000)
    if len(payload) >= 16000:
        raise ValueError('Worker input too large')
    request = json.loads(payload)
    case = next(c for c in cases.CASES
                if c['id'] == request['case'] and c['kind'] == 'code')
    tests = cases.code_tests(case)
    answer = dict(schema='bounded-code-v2-result-v1',
                  benchmark_version=cases.BENCHMARK_VERSION, case=case['id'],
                  semantic_correct=False, safe_policy_passed=False,
                  format_compliant=False, tests_passed=0, tests_run=0,
                  tests_total=len(tests), status='rejected_before_execution')
    try:
        tree, fenced = cases.extract_code(case, request['text'])
        answer.update(safe_policy_passed=True, format_compliant=not fenced,
                      status='executing')
        namespace = {'__builtins__': {}, 'sum': sum}
        exec(compile(tree, '<generated-function>', 'exec'), namespace)
        for test in tests:
            answer['tests_run'] += 1
            actual = namespace[case['function']](*test['arguments'])
            if type(actual) is int and actual == test['expected']:
                answer['tests_passed'] += 1
            elif 'first_counterexample' not in answer:
                answer['first_counterexample'] = {
                    **test, 'actual': actual, 'actual_type': type(actual).__name__}
        answer.update(semantic_correct=answer['tests_passed'] == len(tests),
                      status='executed')
    except Exception as exc:
        answer['failure_type'] = type(exc).__name__
        if answer['safe_policy_passed']:
            answer['status'] = 'execution_error'
    print(json.dumps(answer, sort_keys=True))


if __name__ == '__main__':
    main()
