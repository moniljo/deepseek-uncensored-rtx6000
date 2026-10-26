"""Bounded CPU-only worker for fixed one-return finite-input coding tasks."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import accuracy_cases as cases


def main():
    payload=sys.stdin.read(16000)
    if len(payload)>=16000:
        raise ValueError('Worker input too large')
    request=json.loads(payload)
    case=next(c for c in cases.CASES if c['id']==request['case'] and c['kind']=='code')
    answer={'case':case['id'],'semantic_correct':False,'safe_policy_passed':False,
            'tests_passed':0,'tests_total':len(cases.code_tests(case))}
    try:
        tree,fenced=cases.extract_code(case,request['text'])
        answer.update(safe_policy_passed=True,format_compliant=not fenced)
        namespace={'__builtins__':{},'sum':sum}
        exec(compile(tree,'<generated-function>','exec'),namespace)
        for test in cases.code_tests(case):
            actual=namespace[case['function']](*test['arguments'])
            if type(actual) is int and actual==test['expected']:
                answer['tests_passed']+=1
            elif 'first_counterexample' not in answer:
                answer['first_counterexample']={**test,'actual':actual,'actual_type':type(actual).__name__}
        answer['semantic_correct']=answer['tests_passed']==answer['tests_total']
    except Exception as exc:
        answer['failure_type']=type(exc).__name__
    print(json.dumps(answer,sort_keys=True))


if __name__=='__main__': main()
