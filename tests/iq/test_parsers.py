"""Pure expected-row and owned-cleanup parsers; no native process/API/GPU."""
from pathlib import Path
import argparse,copy,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import test_iq as Q

def run():
    expected=Q.read(ROOT/'tests/iq/EXPECTED-ROWS.json')['runs']['default-self']
    lines=expected['role_headers']+[f'{t} {n} worst 0.00e+00 (tol {tol:.0e}) ok'for t,n,tol in expected['rows']]+['ALL OK']
    valid='\n'.join(lines)
    assert Q.validate_numeric(valid,expected)['numeric_rows']==102
    count=1
    bad=[valid.replace('ALL OK','FAILED'),valid.replace('worst 0.00e+00','worst nan',1),
         valid.replace('worst 0.00e+00','worst 1.00e+20',1),'\n'.join(lines[:-2]+['ALL OK']),
         valid+'\nALL OK',valid.replace('---- role types','---- altered types',1),valid.replace(') ok',') FAIL',1)]
    for value in bad:
        try:Q.validate_numeric(value,expected)
        except RuntimeError:count+=1
        else:raise AssertionError('Invalid numeric output accepted')
    identity={'synthetic':True}
    good={'passed':True,'exit_code':0,'exe_sha256':'a'*64,'identity':identity,
        'cleanup':{'passed':True,'cleanup_proof_version':3,'owned_job_closed':True,'owned_job_active_processes_before_close':0,
            'owned_job_empty_proven':True,'errors':[],'separate_process_handle_checks':[{'role':'numeric-fixture',
                'separately_held_handle_signaled':True,'wait_result':0,'recorded_identity':identity}]}}
    assert Q.cleanup_ok(good,'a'*64);count+=1
    mutations=[lambda x:x.update(exit_code=1),lambda x:x.update(error_type='Synthetic'),
        lambda x:x['cleanup'].update(owned_job_active_processes_before_close=1),
        lambda x:x['cleanup'].update(separate_process_handle_checks=[]),
        lambda x:x['cleanup']['separate_process_handle_checks'][0].update(wait_result=258),
        lambda x:x.update(exe_sha256='b'*64)]
    for change in mutations:
        value=copy.deepcopy(good);change(value);assert not Q.cleanup_ok(value,'a'*64);count+=1
    return {'passed':True,'pure_checks':count,'synthetic_metadata_NOT_actual_ownership':True,
        'native_process_GPU_model_API_or_process_queries':0}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    print(json.dumps(run()if a.self_test else {'default_dry':True,'actions':0}))
