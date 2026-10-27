"""Portable current-source admission; no model, process or network calls."""
from pathlib import Path
import hashlib
import json
import re

COMMIT='5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d'
PORT='799413c6b137a5457becf10c90ce471d57018cedbb467708066c041b7f7003d0'
PROMPT='6c4f575406cf9303355867354ced548d7c4789830ff6f817ae1f8de83fd17c2a'
ANSWER='93280b15904343560264c3cc4b564ccab82097833751482d454e11af71e16050'
PROFILE='66b2d59fa7fa02cb6ab641846640a8e8c5fbc50906b8a1892140787e8ccf3428'

def option(args,name,need):
    need(args.count(name)==1 and args.index(name)+1<len(args),'One option/value required: '+name)
    return args[args.index(name)+1]

def configure_option(args,name,need):
    values=[x.split('=',1)[1] for x in args if x.startswith(name+'=')]
    need(len(values)==1,'One configure value required: '+name)
    return values[0]

def build_contract(b,overlays,need):
    need(b.get('schema')=='fresh-portable-v027-V6b-native-build-v1' and
         b.get('compiled') is True and type(b.get('compile_and_link_rc')) is int and b['compile_and_link_rc']==0,
         'Actual successful new portable027 build required')
    for key in ('source_checkout_verified_before_build','source_checkout_verified_after_build',
                'source_overlay_verified_before_after','reference_raw_source_reconstructed',
                'all_native_GGML_objects_and_four_custom_static_libraries_rebuilt_from_source',
                'external_prebuilt_CUDA_toolkit_libraries','qualification_not_inferred_from_compilation'):
        need(b.get(key) is True,'Missing actual fresh build proof: '+key)
    need(b.get('old_objects_or_libraries_copied') is False and
         b.get('API_model_or_service_execution') is False and
         b.get('ported_source_manifest_sha256')==overlays.get('strata_reference_stage_manifest_sha256')==PORT and
         b.get('complete_raw_strata_source_files')==overlays.get('expected_complete_strata_files')==5185 and
         b.get('complete_raw_strata_source_tree_sha256')==overlays.get('expected_complete_strata_tree_sha256') and
         re.fullmatch('[0-9a-f]{64}',b.get('complete_raw_strata_source_tree_sha256')or''),
         'Raw source7994 closure/old-object scope differs')
    expected={Path(n).name:v['reference_raw_sha256'] for n,v in overlays['strata'].items()
              if Path(n).name in ('ds4_model.cu','ds4_main.cpp','ds4_model.hpp','ds4_iq.cu')}
    need(len(expected)==4 and b.get('source_pin_map')==b.get('compiled_source_pin_map_raw')==expected,
         'Four actual raw compiled source pins differ')
    need(b.get('CUDA_architecture')=='120a' and 'release 13.2' in b.get('CUDA_version_output','') and
         '19.44.' in b.get('host_compiler_version_output','') and
         b.get('build_id')=='5e8a4cca-strata027-custom-v6b-port2-sm120a',
         'Measured toolchain/build identity differs')

def prepare(a,here,root,need,digest,read,pack_metadata):
    pins={}
    def pin(path,expected):
        path=Path(path).resolve()
        need(re.fullmatch('[0-9a-f]{64}',expected or'') and path.is_file() and digest(path)==expected,
             'Explicit reviewed artifact byte pin differs')
        pins[str(path)]=expected;return path
    def child(base,rel):
        path=(base/rel).resolve()
        need(path.is_relative_to(base),'Source pin escapes its declared directory');return path
    manifest=pin(root/'pins/owned-api-adapter.json',a.expected_harness_manifest_sha256)
    mf=read(manifest)
    need(mf.get('schema')=='portable-owned-API-source-v1','Owned API source manifest differs')
    for name,h in mf['files_sha256'].items():pin(child(root,name),h)
    cfg_path=pin(a.config,a.expected_config_sha256)
    need(cfg_path.is_relative_to(root/'.work'),'Use a recipient-created private .work configuration')
    cfg=read(cfg_path);args=cfg['args'];env=cfg.get('env',{})
    need(cfg.get('host')=='127.0.0.1' and type(cfg.get('port')) is int and cfg['port']==18080 and
         cfg.get('arch')=='deepseek41' and isinstance(cfg.get('api_key'),str) and cfg['api_key'] and
         not any(x in cfg['api_key'] for x in '\r\n'),'Authenticated loopback18080 config required')
    for name,value in (('--max-context','1048576'),('--cpu-threads','16'),('--dspark','on'),
                       ('--dspark-block','3'),('--dspark-conf','0.65'),('--mmap-experts','off'),
                       ('--ssd-cache','0'),('--reserve-mib','6144'),('--experts','6'),('--prefill-chunk','65280')):
        need(option(args,name,need)==value,'Exact Mix/1M configuration differs: '+name)
    need(not cfg.get('vision') and not cfg.get('user_args') and
         env.get('DS4_GPU_MISSES')==env.get('DS4_GPU_MISS_DIRECT')=='1' and
         env.get('DS4_PREFILL_REGISTER_RAM_MB')=='196608' and
         env.get('DS4_WARM')==env.get('DS4_COPY_PIN')=='0','Reviewed full custom ownership profile required')
    need(not any(any(w in str(x).lower() for w in ('--api-key','--password','--authorization','--token='))
                 for x in args),'Credential-bearing native argument refused')
    bp=pin(a.build_receipt,a.expected_build_receipt_sha256);b=read(bp)
    sp=pin(a.source_manifest,a.expected_source_manifest_sha256);overlays=read(sp)
    need(sp==root/'pins/source-overlays.json' and b.get('source_manifest_sha256')==
         b.get('source_overlay_manifest_sha256')==a.expected_source_manifest_sha256 and
         Path(b['native_source_manifest']).resolve()==sp,'Build/source overlay identity differs')
    build_contract(b,overlays,need)
    engine=pin(cfg['exe'],a.expected_exe_sha256)
    need(Path(b['exe']).resolve()==engine and b['exe_sha256']==a.expected_exe_sha256,'Actual configured native image differs')
    locks=root/'pins/source-locks.json';pin(locks,b['source_locks_sha256'])
    need(read(locks)['strata']['commit']==COMMIT,'Wrong official base commit')
    for name,h in b['recipe_pins'].items():pin(child(root,name),h)
    source=Path(option(b['configure_arguments'],'-S',need)).resolve()
    ggml=Path(configure_option(b['configure_arguments'],'-DSTRATA_GGML_DIR',need)).resolve()
    prepared_path=pin(source.parent/'PREPARED-SOURCE.json',b['prepared_source_receipt_sha256'])
    prepared=read(prepared_path)
    need(prepared.get('schema')=='prepared-pinned-native-source-v1' and
         prepared.get('old_objects_libraries_copied') is False and
         Path(prepared['strata_source']).resolve()==source and Path(prepared['llama_source']).resolve()==ggml and
         prepared.get('complete_raw_strata_source_files')==b['complete_raw_strata_source_files'] and
         prepared.get('complete_raw_strata_source_tree_sha256')==b['complete_raw_strata_source_tree_sha256'],
         'Actual prepared source receipt differs from the completed build')
    for group,base in (('strata',source),('ggml',ggml)):
        for name,entry in overlays[group].items():pin(child(base,name),entry['reference_raw_sha256'])
    runtime=Path(cfg.get('server_runtime')or cfg['cwd']).resolve()
    need(runtime.is_relative_to(root/'.work'),'API runtime must stay in recipient .work')
    rp=pin(a.runtime_manifest,a.expected_runtime_manifest_sha256)
    need(rp==runtime/'RECIPE-RUNTIME.json','Actual prepared API runtime receipt required')
    rm=read(rp);overlay_path=root/'pins/api-runtime-overlays.json'
    pin(overlay_path,rm.get('runtime_overlay_manifest_sha256'));rmf=read(overlay_path)
    need(rm.get('schema')=='prepared-separate-recipe-runtime-v2' and rm.get('passed') is True and
         rm.get('runtime_kind')=='v027-retained-web-API' and rm.get('base_commit')==COMMIT and
         rm.get('complete_runtime_reconstructed') is True and
         rm.get('complete_runtime_files')==rmf.get('expected_complete_runtime_files')==5111 and
         rm.get('complete_runtime_tree_sha256')==rmf.get('expected_complete_runtime_tree_sha256') and
         re.fullmatch('[0-9a-f]{64}',rm.get('complete_runtime_tree_sha256')or'') and
         rmf.get('bounded_shutdown_grace')=={'quit_seconds':60,'terminate_seconds':5,'kill_seconds':20},
         'Complete current027 API runtime proof is missing or pending')
    if cfg.get('server_runtime') is not None:
        need(cfg.get('server_runtime_manifest_sha256')==a.expected_runtime_manifest_sha256,
             'Selected runtime route must bind the same prepared receipt')
    files={}
    for path in sorted(runtime.rglob('*')):
        if path.is_file() and path!=rp:
            actual=path.resolve();need(actual.is_relative_to(runtime),'Runtime source escapes its directory')
            files[path.relative_to(runtime).as_posix()]=digest(actual);pins[str(actual)]=files[path.relative_to(runtime).as_posix()]
    tree=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    need(len(files)==5111 and tree==rmf['expected_complete_runtime_tree_sha256'],'Actual complete API runtime bytes differ')
    for name,h in rmf['files'].items():need(files.get(name)==h,'API runtime overlay differs')
    for name,h in rmf['retained_web_files'].items():need(files.get(name)==h,'Retained web file differs')
    pack=Path(option(args,'--pack',need)).resolve()
    for name in ('ds4.txt','tensors.txt','engram.bin','dspark-merge-receipt.json'):
        path=pack/name;pins[str(path)]=digest(path)
    for path in sorted((pack/'tokenizer').rglob('*')):
        if path.is_file():pins[str(path)]=digest(path)
    payload=pack_metadata(pack,read(root/'pins/model-downloads.json'))
    profile=Path(option(args,'--expert-profile',need)).resolve()
    need(profile.parent==cfg_path.parent and profile.name=='expert-profile.bin' and
         profile.stat().st_size==61440 and digest(profile)==PROFILE,'Fresh private sibling66 required')
    cuda=Path(configure_option(b['configure_arguments'],'-DCUDAToolkit_ROOT',need)).resolve();dlls={}
    for name in ('cublas64_13.dll','cublasLt64_13.dll'):
        matches={(Path(folder)/name).resolve() for folder in (engine.parent,*map(Path,cfg['lib_dirs'])) if(Path(folder)/name).is_file()}
        need(len(matches)==1,'Exactly one selected toolkit DLL required');path=next(iter(matches))
        need(path in ((cuda/'bin/x64'/name).resolve(),(cuda/'bin'/name).resolve()),'DLL path differs from built toolkit')
        dlls[name.lower()]=path;pins[str(path)]=digest(path)
    pins[str(a.python.resolve())]=digest(a.python)
    need(a.long_prompt.is_file() and 0<a.long_prompt.stat().st_size<=12<<20 and digest(a.long_prompt)==PROMPT and
         hashlib.sha256(a.expected_long_answer.encode()).hexdigest()==ANSWER and
         a.long_thinking_preset=='high-budget256','Frozen FIRST body/exact answer/high256 required')
    pins[str(a.long_prompt.resolve())]=PROMPT
    return cfg,runtime,engine,pins,payload,dlls
