"""Build/run the bounded synthetic GPU fixture against a FRESH native build.

No model, server, HTTP, configuration, or tunnel is accessed. Default is dry.
The fixture tests actual staging/registration/GG consumers; Model workers,
real pool/47-buffer maintenance and cancellation are separate loaded tests.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'tests/gpu'
OBJECTS=[
    'CMakeFiles/strata-dsv41.dir/src/dsv41/'+name for name in (
        'ds4_kernels.cu.obj','ds4_vision.cu.obj','ds4_gguf.cu.obj',
        'ds4_gguf_cpu.cpp.obj','ds4_iq.cu.obj','ds4_iq_cpu.cpp.obj',
        'ds4_tq.cu.obj','ds4_tq_cpu.cpp.obj','ds4_gg_cpu.cpp.obj','ds4_packinfo.cpp.obj')]
LIBRARIES=['strata_dsv41_kernels.lib','strata_kernels_cpu.lib',
           'ggml/src/ggml-cpu.lib','ggml/src/ggml-base.lib']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def need(condition,message):
    if not condition:raise RuntimeError(message)
def dump(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2);f.write('\n')
def safe_arg(value):
    value=str(value)
    need(not any(c in value for c in '\r\n"%'), 'Unsupported batch-path character')
    return '"'+value+'"'
def option(args,key):
    values=[v.split('=',1)[1] for v in args if v.startswith(key+'=')]
    need(len(values)==1,'Missing/ambiguous native configure argument '+key)
    return values[0]

def verify_plan(plan):
    for group in ('fixture_source_pins','fresh_build_input_pins'):
        for name,digest in plan[group].items():
            need(sha(name)==digest,'Input changed: '+name)
    need(sha(plan['native_build_receipt'])==plan['native_build_receipt_sha256'],
         'Native build receipt changed')

def prepare(build_receipt,out,vcvars):
    manifest=read(FIXTURE/'SOURCE-MANIFEST.json')
    need(manifest['model_payloads_required'] is False,'Unexpected payload dependency')
    source_pins={}
    for name,digest in manifest['source_files_LF_sha256'].items():
        path=(FIXTURE/name).resolve()
        need(path.is_relative_to(FIXTURE.resolve()),'Fixture manifest traversal')
        need(sha(path)==digest,'Fixture source changed: '+name)
        source_pins[str(path)]=digest
    from gpu_fixture_admission import admit
    build,source,native,ggml,cuda=admit(build_receipt,manifest)
    for relative in ('scripts/test_gpu.py','scripts/gpu_fixture_admission.py','tests/gpu/SOURCE-MANIFEST.json'):
        source_pins[str(ROOT/relative)]=sha(ROOT/relative)
    if vcvars is None:
        # .../VC/Tools/MSVC/<version>/bin/Hostx64/x64/cl.exe -> VC
        compiler=Path(build['host_compiler']).resolve()
        vcvars=compiler.parents[6]/'Auxiliary/Build/vcvars64.bat'
    vcvars=Path(vcvars).resolve()
    need(vcvars.is_file(),'Pass --vcvars path to MSVC14.44 vcvars64.bat')
    inputs=[native/name for name in OBJECTS+LIBRARIES]
    inputs += [cuda/'lib/x64/cublas.lib',cuda/'lib/x64/cublasLt.lib']
    need(len(inputs)==16 and all(p.is_file() for p in inputs),'Fresh library/object missing')
    need(not out.exists(),'Use a new exclusive fixture output directory')
    out.mkdir(parents=True)
    exe=out/'gpu_misses_stage_test.exe'
    command=[str(cuda/'bin/nvcc.exe'),'-std=c++20','-O3','--ptxas-options=-v',
        '--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]',
        '--cudart','shared','-Xcompiler=/EHsc,/O2,/MT,/Z7',
        '-DSTRATA_DSV41_TQ_CPU=1','-DSTRATA_NATIVE_EXPERTS=1','-DSTRATA_Q6K_EXPERTS=1',
        '-I'+str(source/'src/dsv41'),'-I'+str(source/'src/dsv41/epoch-support'),
        '-I'+str(source/'include'),'-I'+str(ggml/'ggml/include'),
        str(FIXTURE/'stage_test.cu'),*[str(p) for p in inputs],'advapi32.lib','-o',str(exe)]
    plan={'schema':'fresh-portable027-synthetic-GPU-fixture-plan-v1',
        'native_build_receipt':str(build_receipt),'native_build_receipt_sha256':sha(build_receipt),
        'fixture_manifest_sha256':sha(FIXTURE/'SOURCE-MANIFEST.json'),
        'fixture_source_pins':source_pins,'fresh_build_input_pins':{str(p):sha(p) for p in inputs},
        'fourteen_custom_inputs_from_one_fresh_build':True,
        'old_native_or_GGML_objects_used':False,'main_is_freshly_compiled_not_reused':True,
        'command':command,'exe':str(exe),'expected_counts':manifest['expected_counts'],
        'CUDA_toolkit':str(cuda),'actual_Model_worker_or_47_transaction_proof':False,
        'model_or_API_calls':0}
    batch='\r\n'.join(['@echo off','setlocal','call '+safe_arg(vcvars)+' -vcvars_ver=14.44',
        'if errorlevel 1 exit /b %errorlevel%','set "_CL_=/Z7"',
        ' '.join(safe_arg(arg) for arg in command),'exit /b %errorlevel%',''])
    (out/'compile.bat').write_bytes(batch.encode('utf-8'))
    dump(out/'PLAN.json',plan)
    return plan

def validate_numeric(result,counts):
    need(result['passed'] is True and result['finished'] is True,'GPU fixture did not pass/finish')
    need(result['actual_helper_extracted'] is True and result['format_families']==8,
         'Actual helper/format evidence missing')
    for key,expected in counts.items():need(result[key]==expected,'Unexpected count: '+key)
    for name,count in (('cases','case_count'),('direct_cases','direct_case_count'),
                       ('recovery_cases','recovery_case_count')):
        need(len(result[name])==counts[count] and len({row['family'] for row in result[name]})==8,
             'Missing numeric rows/format family in '+name)
        need(all(row['all_bits_finite_guards_inputs'] is True for row in result[name]),
             'Numeric/input/guard parity failed in '+name)
        need(all(type(row['output_FNV64']) is int and 0<=row['output_FNV64']<2**64
                 for row in result[name]),'Missing numeric output hash in '+name)
    if 'reserve_case_count' in counts:
        need(len(result['reserve_cases'])==counts['reserve_case_count'], 'Missing reserve numeric rows')
        need(all(row['all_bits_finite_guards_inputs'] is True and
                 type(row['output_FNV64']) is int and 0<=row['output_FNV64']<2**64
                 for row in result['reserve_cases']), 'Reserve numeric/input/guard parity failed')
    need(result['model_or_API_calls']==0 and result['performance_claim'] is False,
         'Unexpected fixture scope')

def numeric_hashes(result):
    return {name:[{k:row[k] for k in ('family','kind','T','output_FNV64')}
                  for row in result[name]] for name in ('cases','direct_cases','recovery_cases')}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native-build',type=Path,default=ROOT/'.work/build-native/BUILD.json')
    ap.add_argument('--out',type=Path,default=ROOT/'.work/gpu-test-01')
    ap.add_argument('--vcvars',type=Path)
    ap.add_argument('--prepare',action='store_true',help='Pin fresh inputs and write a compile-only batch')
    ap.add_argument('--build',action='store_true',help='Prepare and compile/link the synthetic fixture')
    ap.add_argument('--run',action='store_true',help='Run only an already-built synthetic fixture')
    ap.add_argument('--compare-numeric',type=Path,help='Require all prior numeric FNV hashes identical')
    ap.add_argument('--timeout',type=int,default=300)
    a=ap.parse_args();need(1<=a.timeout<=3600,'Timeout must be1..3600 seconds')
    out=a.out.resolve()
    if not (a.prepare or a.build or a.run):
        print(json.dumps({'default_dry':True,'native_build':str(a.native_build),'output':str(out),
                          'fresh_custom_inputs':14,'models_or_API':False,'GPU_NOT_RUN':True}))
        return 0
    need(not(a.run and(a.prepare or a.build)),'Build and run are explicit separate invocations')
    if a.prepare or a.build:
        plan=prepare(a.native_build.resolve(),out,a.vcvars)
        if not a.build:
            print(json.dumps({'prepared':str(out/'PLAN.json'),'compiled':False,'GPU_NOT_RUN':True}))
            return 0
        verify_plan(plan)
        with (out/'compile.log').open('xb') as log:
            completed=subprocess.run(['cmd.exe','/d','/c',str(out/'compile.bat')],cwd=out,
                                     stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
        verify_plan(plan)
        receipt={'schema':'fresh-portable027-synthetic-GPU-fixture-build-v1',
            'compile_rc':completed.returncode,'plan_sha256':sha(out/'PLAN.json'),
            'source_and_inputs_unchanged':True,'fresh_main_and_fourteen_custom_inputs':True,
            'exe':plan['exe'],'exe_sha256':sha(plan['exe']) if completed.returncode==0 else None,
            'compile_log_sha256':sha(out/'compile.log'),'GPU_NOT_RUN':True,'models_or_API':False}
        dump(out/'BUILD.json',receipt)
        need(completed.returncode==0,'Fixture compilation failed; see compile.log')
        print(json.dumps(receipt));return 0
    plan=read(out/'PLAN.json');build=read(out/'BUILD.json');verify_plan(plan)
    need(build['compile_rc']==0 and build['plan_sha256']==sha(out/'PLAN.json') and
         sha(plan['exe'])==build['exe_sha256'],'Fixture binary/build receipt changed')
    output=out/'numeric.json';need(not output.exists(),'Numeric result already exists')
    env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','STRATA_','GGML_'))}
    cuda=Path(plan['CUDA_toolkit'])
    runtime=[cuda/'bin/x64',cuda/'bin']
    env['PATH']=os.pathsep.join(str(p) for p in runtime if p.is_dir())+os.pathsep+env.get('PATH','')
    with (out/'run.log').open('xb') as log:
        completed=subprocess.run([plan['exe'],'--run','--output',str(output)],cwd=out,
            env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
    verify_plan(plan)
    need(completed.returncode==0,'GPU fixture failed; inspect retained run.log')
    result=read(output);validate_numeric(result,plan['expected_counts'])
    same=None
    if a.compare_numeric:
        old=read(a.compare_numeric);same=numeric_hashes(old)==numeric_hashes(result)
        need(same,'Numeric hashes differ from the supplied previous fixture result')
    dump(out/'RUN.json',{'schema':'synthetic-GPU-fixture-run-v1','native_exit':completed.returncode,
        'passed':True,'numeric_sha256':sha(output),'run_log_sha256':sha(out/'run.log'),
        'previous_numeric_hashes_identical':same,'source_and_inputs_unchanged':True,
        'Model_worker_47_buffer_and_loaded_Stop_lifecycle_NOT_tested':True,
        'model_or_API_calls':0,'performance_claim':False})
    print(json.dumps({'passed':True,'numeric':str(output),'previous_hashes_identical':same}));return 0

if __name__=='__main__':sys.exit(main())
