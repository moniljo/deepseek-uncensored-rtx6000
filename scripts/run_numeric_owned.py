"""Default dry; run only a pinned synthetic numeric executable in an owned Job.

No model, API, configuration or tunnel is accessed. No global process is killed.
"""
from pathlib import Path
import argparse,ctypes,hashlib,json,math,os,subprocess,sys,time
from ctypes import wintypes as W
import psutil

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--exe',type=Path)
    p.add_argument('--expected-exe-sha256');p.add_argument('--output',type=Path)
    p.add_argument('--cuda-root',type=Path);p.add_argument('--timeout',type=float,default=300)
    p.add_argument('--env-json',default='{}');p.add_argument('fixture_args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if not a.run:
        print(json.dumps({'default_dry':True,'GPU_model_process_or_API_actions':0}));return 0
    if sys.flags.optimize:
        raise RuntimeError('Run the numeric ownership checks with Python assertions enabled')
    assert sys.platform=='win32'and a.exe is not None and a.output is not None and a.cuda_root is not None
    exe,output,cuda=a.exe.resolve(),a.output.resolve(),a.cuda_root.resolve()
    assert exe.is_relative_to(ROOT/'.work')and exe.name in ('ds4_iq_test.exe','gpu_misses_stage_test.exe')
    assert len(a.expected_exe_sha256 or'')==64 and hashlib.sha256(exe.read_bytes()).hexdigest()==a.expected_exe_sha256
    assert output.is_relative_to(ROOT/'.work')and not output.exists()and math.isfinite(a.timeout)and 0<a.timeout<=1800
    manifest=json.loads((ROOT/'tests/numeric-owned-sources.json').read_bytes())
    for name,digest in manifest['files_sha256'].items():
        path=(ROOT/name).resolve()
        assert path.is_relative_to(ROOT)and hashlib.sha256(path.read_bytes()).hexdigest()==digest
    sys.path.insert(0,str(ROOT/'bench'))
    from owned_windows import Ownership
    import identity_support as native
    import job_support as common
    # Passive two-snapshot exclusion; never infer ownership or terminate by name.
    deadline,clear=time.monotonic()+30,0
    while clear<2:
        busy=[x.pid for x in psutil.process_iter(['name'])if x.info['name']and
              (x.info['name'].lower().startswith(('strata','ds41_generate','ds4_generate'))or x.info['name'].lower()=='llama-server.exe')]
        clear=0 if busy else clear+1
        assert time.monotonic()<deadline,'Competing inference process remains; no process was stopped'
        if clear<2:time.sleep(.5)
    output.mkdir(parents=True)
    kernel,job=native.windows(),common.OwnedJob()
    kernel.OpenThread.argtypes,kernel.OpenThread.restype=[W.DWORD,W.BOOL,W.DWORD],W.HANDLE
    kernel.ResumeThread.argtypes,kernel.ResumeThread.restype=[W.HANDLE],W.DWORD
    ownership=Ownership(kernel,job)
    env={k:v for k,v in os.environ.items()if not k.startswith(('DS4_','STRATA_','GGML_'))}
    sdk_dlls=cuda/'bin/x64';assert sdk_dlls.is_dir()
    env['CUDA_PATH']=str(cuda)
    env['PATH']=os.pathsep.join(map(str,(sdk_dlls,cuda/'bin',exe.parent)))+os.pathsep+env.get('PATH','')
    extra=json.loads(a.env_json)
    assert isinstance(extra,dict)and all(k in ('STRATA_DS4_NO_LIGHT','STRATA_DS4_NO_TC')and v=='1'for k,v in extra.items())
    env.update(extra)
    values=a.fixture_args[1:]if a.fixture_args[:1]==['--']else a.fixture_args
    report={'exe':str(exe),'exe_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),
        'arguments':values,'environment_overrides':extra,'GPU_fixture_only':True,'model_loaded':False,
        'runtime_DLL_directory':str(sdk_dlls)}
    began=time.perf_counter();process=None
    try:
        with(output/'stdout.log').open('x',encoding='utf-8')as log:
            process=subprocess.Popen([str(exe),*values],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,close_fds=True,creationflags=0x08000000|0x00000004)
            report['identity']=ownership.attach(process,exe,role='numeric-fixture')
            threads=psutil.Process(process.pid).threads();assert len(threads)==1
            handle=kernel.OpenThread(0x0002,False,threads[0].id);assert handle
            try:assert kernel.ResumeThread(handle)==1
            finally:kernel.CloseHandle(handle)
            process.wait(timeout=a.timeout);report['exit_code']=process.returncode
    except BaseException as error:report['error_type']=type(error).__name__
    finally:
        report['cleanup']=ownership.finish(30)
        if process is not None:process.wait(timeout=30)
        report['elapsed_seconds']=time.perf_counter()-began
        report['passed']=report.get('exit_code')==0 and report['cleanup']['passed']
        (output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(output),'passed':report['passed'],'exit_code':report.get('exit_code')}))
    return 0 if report['passed']else 1

if __name__=='__main__':raise SystemExit(main())
