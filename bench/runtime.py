"""Shared portable bindings for the exact native Q/cancel protocol blocks."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys

from native_support import *
import native_support as B
import prepare_inputs as I
import gpu_miss_validation as V
import direct_pin_validation as P
import native_stat_tap as Tap
import phase_support as phase
from qualification_support import Observer,seed_telemetry
import qualification_contract as contract
from grade_first_support import grade_first
from shutdown_trace import close_owned

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
MAXIMUM=4096
PROFILE_SHA='66b2d59fa7fa02cb6ab641846640a8e8c5fbc50906b8a1892140787e8ccf3428'


def load_frontend(a):
    source=a.strata_source.resolve()
    overlays=read(ROOT/'pins/runtime-overlays.json')
    pins={}
    for name,digest in overlays['files'].items():
        p=ROOT/'source/runtime'/name
        need(sha(p)==digest,'Published runtime overlay changed: '+name)
        q=source/name
        need(sha(q)==digest,'Prepared runtime does not match the published overlay: '+name)
        pins[str(p)]=digest;pins[str(q)]=digest
    need(I.sha((source/'tools/strata_tokenizer.py').read_bytes().replace(b'\r\n',b'\n'))==I.TOKENIZER_LF_SHA,
         'Pinned Strata tokenizer source changed')
    sys.path.insert(0,str(source/'serve'));sys.path.insert(0,str(source/'tools'))
    server=load(source/'serve/server.py','portable_native_frontend')
    ST=load(source/'tools/strata_tokenizer.py','portable_native_tokenizer')
    need(server.ENGINE_QUIT_WAIT_S==60,'Use the published60-second graceful runtime')
    return server,ST


def validate_build(cfg,receipt):
    """Attest the public recipe and actual compiled checkout, not a private map."""
    receipt=Path(receipt);build=read(receipt)
    required=('source_checkout_verified_before_build','source_checkout_verified_after_build',
        'source_overlay_verified_before_after','all_native_GGML_objects_and_four_custom_static_libraries_rebuilt_from_source',
        'external_prebuilt_CUDA_toolkit_libraries','qualification_not_inferred_from_compilation')
    need(build.get('schema')=='fresh-portable-v027-V6b-native-build-v1'and build.get('compiled')is True and
         type(build.get('compile_and_link_rc'))is int and build['compile_and_link_rc']==0 and
         all(build.get(k)is True for k in required)and build.get('old_objects_or_libraries_copied')is False and
         build.get('API_model_or_service_execution')is False and 'all_reused_and_copied_inputs_unchanged'not in build,
         'Truthful completed fresh-source BUILD required')
    image=Path(cfg['exe']).resolve()
    need(image==Path(build['exe']).resolve()and sha(image)==build['exe_sha256'],'BUILD/config/executable differ')
    need(build.get('CUDA_architecture')=='120a','This recipe qualifies the Blackwell120a build')
    pins={str(receipt):sha(receipt),str(image):build['exe_sha256']}
    expected_recipes={'scripts/build_native.ps1','scripts/checkout_source.py','pins/source-locks.json','pins/source-overlays.json'}
    need(set(build['recipe_pins'])==expected_recipes,'Fresh build recipe closure differs')
    for name,digest in build['recipe_pins'].items():
        p=(ROOT/name).resolve();need(p.is_relative_to(ROOT)and sha(p)==digest,'Actual fresh-build recipe changed: '+name)
        pins[str(p)]=digest
    need(build['source_locks_sha256']==sha(ROOT/'pins/source-locks.json')and
         build['source_overlay_manifest_sha256']==sha(ROOT/'pins/source-overlays.json'),'BUILD source-lock/overlay provenance differs')
    overlay=read(ROOT/'pins/source-overlays.json')
    for project in ('strata','ggml'):
        for name,record in overlay[project].items():
            p=(ROOT/'source'/project/name).resolve()
            need(p.is_relative_to((ROOT/'source'/project).resolve())and sha(p)==record['sha256'],
                 'Published actual source changed: '+name)
            pins[str(p)]=record['sha256']
    expected={name:overlay['strata']['src/dsv41/'+name]['sha256']for name in ('ds4_model.cu','ds4_main.cpp','ds4_model.hpp','ds4_iq.cu')}
    need(build['source_pin_map']==build['compiled_source_pin_map_raw']==expected,'Actual compiled raw027 source map differs')
    need(build.get('reference_raw_source_reconstructed')is True and
         build.get('ported_source_manifest_sha256')==overlay['strata_reference_stage_manifest_sha256'] and
         build.get('complete_raw_strata_source_files')==overlay['expected_complete_strata_files'] and
         build.get('complete_raw_strata_source_tree_sha256')==overlay['expected_complete_strata_tree_sha256'],
         'Actual7994 complete raw source reconstruction required')
    args=build['configure_arguments'];need(args.count('-S')==1,'One actual configured source root required')
    source=Path(args[args.index('-S')+1]).resolve()
    prepared=source.parent/'PREPARED-SOURCE.json'
    need(source.is_relative_to((ROOT/'.work').resolve())and sha(prepared)==build['prepared_source_receipt_sha256'],
         'Fresh checkout receipt differs')
    pins[str(prepared)]=sha(prepared)
    # This explicit pre-start verifier checks the complete raw checkout, outside GEN timing.
    verify=ROOT/'scripts/checkout_source.py'
    import subprocess
    verified=subprocess.run([sys.executable,'-B',str(verify),'--verify','--work-dir',str(source.parent)],
                            capture_output=True,text=True,timeout=120)
    need(verified.returncode==0 and len(verified.stdout)<16384,'Complete raw source verification failed before native run')
    pins[str(verify)]=sha(verify)
    for name,digest in expected.items():
        p=source/'src/dsv41'/name;need(sha(p)==digest,'Actual compiled source changed: '+name);pins[str(p)]=digest
    need(cfg['args'].count('--pack')==1,'One exact model pack required')
    pack=Path(cfg['args'][cfg['args'].index('--pack')+1])
    for name in ('ds4.txt','tensors.txt'):
        p=pack/name;need(p.is_file()and p.stat().st_size<=16<<20,'Bounded pack metadata required')
        pins[str(p.resolve())]=sha(p)
    return pins


def profile_snapshot(cfg):
    args=cfg['args'];need(args.count('--expert-profile')==1,'One per-run profile required')
    p=Path(args[args.index('--expert-profile')+1]).resolve()
    need(p.is_relative_to((ROOT/'.work').resolve())and p.is_file()and p.stat().st_size==61440 and sha(p)==PROFILE_SHA,
         'Exact initial66 private profile below ignored .work required')
    need(p!=(ROOT/'.work/bench-inputs/initial-routing-profile.bin').resolve(),'Do not let native learning modify the shared initial input profile')
    return {'path':str(p),'initial_sha256':PROFILE_SHA,'per_run_copy_required':True,'learning_may_update_only_this_copy':True}


def make_private_config(cfg,out,initial_profile):
    cfg=copy.deepcopy(cfg);out=Path(out).resolve();initial_profile=Path(initial_profile)
    need(out.is_relative_to((ROOT/'.work').resolve())and out.is_dir(),'Exclusive local output required before profile copy')
    need(initial_profile.stat().st_size==61440 and sha(initial_profile)==PROFILE_SHA,'Frozen initial66 routing metadata required')
    # No API credential is needed for the native pipe. The source config is
    # never rewritten or copied to the output directory.
    forbidden=('api_key','authorization','password','secret','credential')
    for k in list(cfg):
        if any(x in k.lower()for x in forbidden):del cfg[k]
    cfg['env']={k:v for k,v in cfg.get('env',{}).items()if not any(x in k.lower()for x in forbidden)}
    need(not cfg.get('user_args')and not cfg.get('vision')and all(not any(word in str(arg).lower()for word in
         ('--api-key','--authorization','--password','--token='))for arg in cfg['args']),
         'Explicit credential-free text-only native arguments required')
    for key,value in (('--max-context','1048576'),('--cpu-threads','16'),('--dspark','on'),('--dspark-block','3'),
                      ('--dspark-conf','0.65'),('--prefill-chunk','65280')):
        need(cfg['args'].count(key)==1 and cfg['args'][cfg['args'].index(key)+1]==value,'Frozen benchmark geometry differs: '+key)
    need(cfg['args'].count('--experts')==1 and cfg['args'][cfg['args'].index('--experts')+1]=='6','Explicit all-six routing required')
    need(cfg['env'].get('DS4_WARM')==cfg['env'].get('DS4_COPY_PIN')=='0','027 shared ownership requires explicit WARM0/COPY_PIN0')
    env=cfg['env'];need(env.get('DS4_GPU_MISSES')==env.get('DS4_GPU_MISS_DIRECT')==env.get('DS4_PREFILL_REPEAT_REARM')=='1'and
        env.get('DS4_PREFILL_REGISTER_RAM_MB')=='196608'and env.get('DS4_PREFILL_REPEAT_MIN_TOKENS')=='32768'and
        env.get('DS4_GPU_MISS_OVERLAP','0')=='0'and env.get('DS4_CPU_SMT','0')=='0'and
        'DS4_DSPARK_MAX'not in env and 'DS4_DSPARK_CONF'not in env,'Exact selected direct192/on3/no-overlap profile required')
    p=out/'expert-profile.bin'
    need(cfg['args'].count('--expert-profile')==1 and cfg['args'].index('--expert-profile')+1<len(cfg['args']),
         'Exactly one existing expert-profile value required')
    with p.open('xb')as stream:stream.write(initial_profile.read_bytes())
    cfg['args'][cfg['args'].index('--expert-profile')+1]=str(p)
    return cfg,profile_snapshot(cfg)


def quality(row,case,is_v2,oracle,decode):
    worker=HERE/'oracles'/('code_worker_v2.py'if is_v2 else'code_worker.py')
    entry=contract.grade_row(row,case,is_v2,oracle,decode,worker)
    row.update(case=case['id'],grade=entry['grade'],quality_passed=entry['joint_passed'])
    return entry


def cleanup_ok(cleanup,identity):
    contract.validate_cleanup(cleanup,identity)
    return True


def frozen_short_inputs(a,cfg):
    server,ST=load_frontend(a);manifest=I.frozen_manifest()
    encode,decode,kind=I.reference_encoder(a,ST,manifest)
    directory=Path(cfg['tokenizer'])
    if not directory.is_absolute():directory=Path(cfg.get('cwd',a.config.parent))/directory
    token=I.packed_tokenizer(ST,directory)
    sys.path.insert(0,str(HERE/'oracles'))
    original=load(HERE/'oracles/accuracy_cases.py','accuracy_cases')
    revised=load(HERE/'oracles/accuracy_cases_v2.py','portable_short_v2')
    encoder=load(HERE/'vendor/dsv41_encoding.py','portable_short_encoder')
    specs=[(original,'filtered-decimal-revenue',164,'22464a7c6f125403f341c2f1905a5fe2dd332d1f6ad2ad074fcf1fc1a8e8b87a'),
           (revised,'code-account-balance',168,'6b355c6f44d2cbace47309216051869e798220085d055af38b17ca5a8a820a90'),
           (original,'code-filtered-cents',126,'f51a813ee106aac5a13b2e69fb586b1f99593ce6255d829f9475c9383097b9e6')]
    short=[];parity=[]
    for module,name,n,digest in specs:
        case=next(c for c in module.CASES if c['id']==name)
        text=encoder.encode_messages([{'role':'user','content':case['prompt']}],thinking_mode='thinking',reasoning_effort=75)
        ids=encode(text);need(token.encode(text,parse_special=True)==ids and len(ids)==n and idsha(ids)==digest,
                               'Frozen short input/reference tokenizer differs: '+name)
        short.append((module,case,ids));parity.append({'case':name,'tokens':n,'input_ids_sha256':digest,'equal':True})
    return short,original,decode,parity


def verify_prepared_tokenizer(cfg,prepared):
    directory=Path(cfg['tokenizer'])
    if not directory.is_absolute():directory=Path(cfg.get('cwd','.'))/directory
    expected=prepared['packed_tokenizer_files_sha256']
    need(set(expected)=={'vocab.json','merges.txt','token_type.json','tokenizer.json'}and
         prepared.get('strata_tokenizer_LF_sha256')==I.TOKENIZER_LF_SHA and
         all(sha(directory/name)==digest for name,digest in expected.items()),
         'Current model-pack tokenizer differs from the complete frozen input preparation')
    return True


def ready_inputs(directory):
    receipt = B.read(directory/'INPUTS.json')
    frozen = I.frozen_manifest()
    B.need(receipt.get('passed') is True and receipt.get('no_model_or_API_started') is True and
           receipt.get('input_manifest_sha256') == B.sha(HERE/'frozen-inputs.json') and
           receipt.get('oracle_provenance_sha256') == B.sha(HERE/'ORACLE-PROVENANCE.json'),
           'Prepare the exact frozen inputs first')
    for name,record in B.read(HERE/'ORACLE-PROVENANCE.json')['files'].items():
        B.need(B.sha(HERE/name)==record['portable_LF_sha256'],'Frozen oracle/vendor source changed: '+name)
    raw = (directory/'prefix.u32le').read_bytes()
    B.need(len(raw)==999764*4 and hashlib.sha256(raw).hexdigest()==frozen['prefix_sha256'],
           'Generated local prefix bytes differ')
    prefix = list(struct.unpack('<'+'I'*999764,raw))
    rows = receipt['inputs']
    B.need(len(receipt['interlude_ids'])==164 and B.idsha(receipt['interlude_ids'])==
           '22464a7c6f125403f341c2f1905a5fe2dd332d1f6ad2ad074fcf1fc1a8e8b87a',
           'Frozen short interlude differs')
    B.need(len(rows)==13 and len(frozen['cases'])==12,'Exactly original12 plus v2 required')
    expected = [(e['case']['id'],e)for e in frozen['cases']]+[('code-account-balance-v2',frozen['corrected_v2'])]
    for i,(row,(name,old)) in enumerate(zip(rows,expected)):
        ids = prefix+row['suffix_ids']
        B.need(row.get('name')==name and row.get('tokenizer_equal') is True and
               row.get('corrected_v2') is (i==12) and len(ids)==old['prompt_tokens'] and
               B.idsha(ids)==row['input_ids_sha256']==old['input_ids_sha256'],
               'Frozen prepared input differs: '+name)
    return receipt,prefix
