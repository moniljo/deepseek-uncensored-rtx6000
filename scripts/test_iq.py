"""Default dry. Compile/run the bounded Q6/Mix numeric oracle; never a model.

Run --build, then a separate --run. Eight fresh owned processes cover four
kernel selectors and the exact self/mixed row matrices. No timing claim.
"""
from pathlib import Path
import argparse,collections,hashlib,json,math,os,re,subprocess,sys,time
from gpu_fixture_admission import admit
ROOT=Path(__file__).resolve().parents[1];FIXTURE=ROOT/'tests/iq'
ROW=re.compile(r'^(\S+)\s+(.+?)\s+worst\s+(\S+)\s+\(tol\s+(\S+)\)\s+(ok|FAIL)$')
ARMS={'default':{},'no-light':{'STRATA_DS4_NO_LIGHT':'1'},'no-tc':{'STRATA_DS4_NO_TC':'1'},
      'no-light-no-tc':{'STRATA_DS4_NO_LIGHT':'1','STRATA_DS4_NO_TC':'1'}}
def need(v,m):
    if not v:raise RuntimeError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    with Path(p).open('x',encoding='utf-8')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def validate_numeric(text,expected):
    need(len(text.encode())<=2<<20,'Bounded numeric output exceeded')
    lines=[x.strip()for x in text.splitlines()if x.strip()]
    need(lines and lines[-1]=='ALL OK'and lines.count('ALL OK')==1,'One final ALL OK required')
    need(not re.search(r'\b(?:FAIL|FAILED|nan|inf|infinity)\b',text,re.I),'Failure/nonfinite output')
    rows=[]
    for line in lines:
        if 'worst'not in line:continue
        m=ROW.fullmatch(line);need(m is not None,'Malformed numeric row')
        worst,tol=float(m[3]),float(m[4])
        need(math.isfinite(worst)and math.isfinite(tol)and 0<=worst<=tol and tol>=0 and m[5]=='ok','Numeric tolerance failed')
        rows.append((m[1],' '.join(m[2].split()),tol))
    need(collections.Counter(rows)==collections.Counter(tuple(x)for x in expected['rows']),'Missing/extra/changed expected numeric rows')
    need([x for x in lines if x.startswith('---- role types ')]==expected['role_headers'],'Role-pair coverage changed')
    return {'passed':True,'numeric_rows':len(rows),'all_expected_rows_and_tolerances':True,'all_final_OK':True}

def cleanup_ok(report,exe_sha):
    c=report.get('cleanup',{});checks=c.get('separate_process_handle_checks',[])
    return (report.get('passed')is True and report.get('exit_code')==0 and 'error_type'not in report and
        report.get('exe_sha256')==exe_sha and c.get('passed')is True and c.get('cleanup_proof_version')==3 and
        c.get('owned_job_closed')is True and c.get('owned_job_active_processes_before_close')==0 and
        c.get('owned_job_empty_proven')is True and c.get('errors')==[] and len(checks)==1 and
        checks[0].get('role')=='numeric-fixture'and checks[0].get('separately_held_handle_signaled')is True and
        checks[0].get('wait_result')==0 and checks[0].get('recorded_identity')==report.get('identity'))

def verify(pins):
    for p,d in pins.items():need(sha(p)==d,'Numeric source/input changed: '+str(p))

def prepare(receipt,out,vcvars):
    build,source,native,ggml,cuda=admit(receipt,read(ROOT/'tests/gpu/SOURCE-MANIFEST.json'))
    meta=read(FIXTURE/'SOURCE-MANIFEST.json')
    need(sha(FIXTURE/'iq_expert_test.cpp')==meta['source_sha256']and
         sha(FIXTURE/'EXPECTED-ROWS.json')==meta['expected_rows_sha256'],'Frozen IQ oracle changed')
    adapters=meta.get('adapter_source_files_sha256',{})
    need({'scripts/test_iq.py','scripts/run_numeric_owned.py','scripts/gpu_fixture_admission.py',
          'tests/numeric-owned-sources.json'}<=adapters.keys(),'Complete portable numeric adapter pins required')
    for relative,digest in adapters.items():
        path=(ROOT/relative).resolve()
        need(path.is_relative_to(ROOT)and sha(path)==digest,'Portable numeric adapter source changed')
    names=['ds4_iq.cu','ds4_iq_cpu.cpp','ds4_gg_cpu.cpp','ds4_tq.cu','ds4_tq_cpu.cpp']
    inputs=[native/'CMakeFiles/strata-dsv41.dir/src/dsv41'/(n+'.obj')for n in names]
    inputs +=[native/n for n in ('strata_kernels_cpu.lib','ggml/src/ggml-cpu.lib','ggml/src/ggml-base.lib')]
    inputs +=[cuda/'lib/x64/cudart.lib',cuda/'lib/x64/cudadevrt.lib']
    need(len(inputs)==10 and all(p.is_file()for p in inputs),'Current fresh IQ/GGML/CUDA inputs missing')
    if vcvars is None:vcvars=Path(build['host_compiler']).resolve().parents[6]/'Auxiliary/Build/vcvars64.bat'
    need(Path(vcvars).is_file(),'MSVC14.44 vcvars64.bat required')
    need(not out.exists(),'Use a new exclusive IQ output');out.mkdir(parents=True)
    exe=out/'ds4_iq_test.exe'
    includes=[source/'include',source/'src/dsv41',source/'third_party/ggml',ggml/'ggml/include',cuda/'include',cuda/'include/cccl']
    command=['cl.exe','/nologo','/std:c++20','/EHsc','/O2','/Ob2','/MT','/Z7','/DNDEBUG','/DSTRATA_NATIVE_EXPERTS=1',
        '/DSTRATA_DSV41_TQ_CPU=1','/DSTRATA_Q6K_EXPERTS=1',*['/I'+str(p)for p in includes],
        str(FIXTURE/'iq_expert_test.cpp'),'/Fo'+str(out/'iq_expert_test.cpp.obj'),'/Fe'+str(exe),*map(str,inputs),
        '/link','/INCREMENTAL:NO','/DEBUG','/PDB:'+str(out/'ds4_iq_test.pdb'),
        *['/DEFAULTLIB:'+n for n in ('kernel32.lib','user32.lib','gdi32.lib','winspool.lib','shell32.lib','ole32.lib',
            'oleaut32.lib','uuid.lib','comdlg32.lib','advapi32.lib')]]
    pins={str(p):sha(p)for p in inputs+[Path(receipt),FIXTURE/'iq_expert_test.cpp',FIXTURE/'EXPECTED-ROWS.json',FIXTURE/'SOURCE-MANIFEST.json',
        Path(__file__),ROOT/'scripts/gpu_fixture_admission.py',ROOT/'scripts/run_numeric_owned.py',ROOT/'tests/numeric-owned-sources.json']}
    pins.update({str((ROOT/n).resolve()):h for n,h in adapters.items()})
    plan={'schema':'portable027-IQ-numeric-compile-plan-v1','command':command,'pins':pins,'exe':str(exe),'CUDA_toolkit':str(cuda),
        'native_BUILD':str(receipt),'native_BUILD_sha256':sha(receipt),'old_objects_used':False,'new_fixture_main':True,'model_or_API_calls':0}
    # list2cmdline correctly quotes fixed local Windows path arguments; no shell substitutions.
    for value in [*command,str(vcvars)]:need(not any(c in value for c in '\r\n%"'),'Unsupported batch argument character')
    batch='\r\n'.join(['@echo off','setlocal EnableExtensions DisableDelayedExpansion',
        'call "'+str(vcvars)+'" -vcvars_ver=14.44','if errorlevel 1 exit /b %errorlevel%','set "_CL_=/Z7"',
        subprocess.list2cmdline(command),'exit /b %errorlevel%',''])
    (out/'compile.bat').write_bytes(batch.encode());save(out/'PLAN.json',plan);return plan

def run(out,timeout):
    plan,build=read(out/'PLAN.json'),read(out/'BUILD.json');verify(plan['pins'])
    need(build['compile_rc']==0 and build['plan_sha256']==sha(out/'PLAN.json')and sha(plan['exe'])==build['exe_sha256'],
         'Completed current IQ fixture image/BUILD required')
    root=out/'owned-matrix-01';need(not root.exists(),'Use a new matrix output');root.mkdir()
    expected=read(FIXTURE/'EXPECTED-ROWS.json')['runs'];rows=[]
    for arm,env in ARMS.items():
        for mode,values in [('self',['--type','14,18,21,22']),('mixed',['--mixed'])]:
            label=arm+'-'+mode;child=root/label
            command=[sys.executable,'-B',str(ROOT/'scripts/run_numeric_owned.py'),'--run','--exe',plan['exe'],
                '--expected-exe-sha256',build['exe_sha256'],'--output',str(child),'--cuda-root',plan['CUDA_toolkit'],
                '--timeout',str(timeout),'--env-json',json.dumps(env),'--',*values]
            with(root/(label+'-controller.log')).open('xb')as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout+180)
            report=read(child/'REPORT.json');need(completed.returncode==0 and cleanup_ok(report,build['exe_sha256']),'Strict native0/ownedV3 failed')
            need(report['arguments']==values and report['environment_overrides']==env,'Numeric matrix contrast differs')
            numeric=validate_numeric((child/'stdout.log').read_text(encoding='utf-8-sig'),expected[label])
            rows.append({'arm':label,'numeric':numeric,'native0_ownedV3':True,'REPORT_sha256':sha(child/'REPORT.json'),'stdout_sha256':sha(child/'stdout.log')})
            verify(plan['pins'])
    need(len(rows)==8 and sum(r['numeric']['numeric_rows']for r in rows)==1248,'Incomplete IQ matrix')
    save(root/'REPORT.json',{'schema':'portable027-IQ-eight-arm-numeric-result-v1','passed':True,'finished':True,'runs':rows,
        'numeric_rows':1248,'source_inputs_unchanged':True,'native0_ownedV3_all8':True,'model_or_API_calls':0,'performance_claim':False})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-build',type=Path,default=ROOT/'.work/build-native/BUILD.json')
    p.add_argument('--out',type=Path,default=ROOT/'.work/iq-test-01');p.add_argument('--vcvars',type=Path)
    p.add_argument('--build',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--timeout',type=float,default=300)
    a=p.parse_args();need(math.isfinite(a.timeout)and 0<a.timeout<=1800,'Finite timeout1..1800 required')
    out=a.out.resolve();need(out.is_relative_to(ROOT/'.work')and out!=ROOT/'.work','Exclusive private output under .work required')
    if not(a.build or a.run):print(json.dumps({'default_dry':True,'native_process_GPU_model_API_calls':0,'expected_numeric_rows':1248}));return 0
    need(not(a.build and a.run),'Build and run are separate explicit operations')
    if a.run:run(out,a.timeout);print(json.dumps({'passed':True,'numeric_rows':1248}));return 0
    plan=prepare(a.native_build.resolve(),out,a.vcvars);verify(plan['pins'])
    with(out/'build.log').open('xb')as log:completed=subprocess.run(['cmd.exe','/d','/c',str(out/'compile.bat')],cwd=out,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
    verify(plan['pins']);save(out/'BUILD.json',{'schema':'portable027-IQ-numeric-build-v1','compile_rc':completed.returncode,
        'plan_sha256':sha(out/'PLAN.json'),'exe':plan['exe'],'exe_sha256':sha(plan['exe'])if completed.returncode==0 else None,
        'source_inputs_unchanged':True,'GPU_NOT_RUN':True,'model_or_API_calls':0})
    need(completed.returncode==0,'Compilation failed; retained build.log')
    return 0
if __name__=='__main__':raise SystemExit(main())
