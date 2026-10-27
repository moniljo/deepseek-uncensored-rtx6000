"""Reconstruct the frozen public 1M inputs locally. Dry unless --prepare.

This script never starts an engine, opens an API connection, or executes the
concatenated corpus. Generated token IDs belong under the ignored .work tree.
"""
from __future__ import annotations
import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREFIX = 999764
CORPUS_SHA = '9fc1859dbb3fd21cd4c9b0a8498a0a3bf5e0429fd10502ac92bf79a2959c40ad'
GZIP_SHA = 'a36ec3a5729a168fd4d678a73041dbd3c9a35cfb956a74f5a341d801d3203c32'
ENCODER_SHA = '13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2'
TOKENIZER_LF_SHA = '6db1ea62d3edbaa64dce139088146ccf370933998d5f997cd5f41a39300075a6'
INPUT_MANIFEST_SHA = '31b51c0bf7e66e96f9aed6ea3a44abbd1e2643c31de1ea7d4f7d397714846f5e'
ORACLE_PROVENANCE_SHA = '1fe29bd113ffc6434d779508892797266244480b893c154b0a909fa6a7c1b793'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def idsha(ids):
    return sha(struct.pack('<'+'I'*len(ids), *ids))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def frozen_manifest():
    need(sha((HERE/'frozen-inputs.json').read_bytes())==INPUT_MANIFEST_SHA,
         'Frozen task/input/special-ID manifest changed')
    return read(HERE/'frozen-inputs.json')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def packed_tokenizer(ST, directory):
    vocab = read(directory/'vocab.json')
    tokens = [None]*len(vocab)
    for token, index in vocab.items():
        need(type(index) is int and 0 <= index < len(tokens) and tokens[index] is None,
             'Packed vocabulary indices must be unique and dense')
        tokens[index] = token
    need(all(isinstance(t,str) for t in tokens), 'Packed vocabulary has holes')
    cfg = read(directory/'tokenizer.json')
    return ST.Tokenizer(tokens, (directory/'merges.txt').read_text(encoding='utf-8').split('\n'),
                        read(directory/'token_type.json'), cfg['pre'], cfg['special_ids'])


def reference_encoder(a, ST, manifest):
    if a.reference_tokenizer is not None:
        need(sha(a.reference_tokenizer.read_bytes()) == manifest['reference_tokenizer_sha256'],
             'Pinned optional original reference tokenizer changed')
        try:
            from tokenizers import Tokenizer
            token = Tokenizer.from_file(str(a.reference_tokenizer))
            return (lambda text: token.encode(text, add_special_tokens=False).ids,
                    lambda ids: token.decode(ids, skip_special_tokens=True), 'pinned-HF-tokenizers')
        except Exception:
            pass
    need(a.reference_gguf is not None,
         'Supply already-downloaded OrcaRouter Mix shard1 as --reference-gguf; no second model is required')
    parser = load(a.strata_source/'tools/dsv41/pack_gguf.py', 'bench_reference_gguf')
    kv = parser.Gguf(a.reference_gguf).kv
    token = ST.Tokenizer(kv['tokenizer.ggml.tokens'], kv['tokenizer.ggml.merges'],
                         kv.get('tokenizer.ggml.token_type'), 'deepseek3',
                         {'bos':int(kv.get('tokenizer.ggml.bos_token_id',0)),
                          'eos':int(kv.get('tokenizer.ggml.eos_token_id',1))})
    # This small frozen ID list is from the pinned original tokenizer's special
    # entries. It avoids requiring that tokenizer JSON or a different model.
    special=set(manifest['reference_special_token_ids'])
    return (lambda text: token.encode(text,parse_special=True),
            lambda ids: token.decode([i for i in ids if i not in special]),
            'Mix-shard1-GGUF-BPE-full-frozen-hash-parity-required')


def archive_text():
    """Return checked public corpus text; never execute it or start a model."""
    compressed=(HERE/'data/retained-public-archive.txt.gz').read_bytes()
    need(sha(compressed)==GZIP_SHA,'Public corpus gzip changed')
    raw=gzip.decompress(compressed)
    need(len(raw)==3434689 and sha(raw)==CORPUS_SHA,'Public corpus content changed')
    return raw.decode('utf-8')


def long_prompt_text(case_id='audit-sum',corrected_v2=False):
    """Public user-message content for a normal API probe, not a request sender.

    The API renders its own chat wrapper. Native frozen IDs are validated by
    --prepare separately; this function performs no tokenization or API call.
    """
    manifest=frozen_manifest()
    if corrected_v2:
        need(case_id=='code-account-balance','Only pre-existing balance v2 exists')
        case=manifest['corrected_v2']['case_definition']
    else:
        case=next((e['case']for e in manifest['cases']if e['case']['id']==case_id),None)
        need(case is not None,'Unknown frozen task')
    return archive_text()+case['prompt']


def prepare(a):
    manifest = frozen_manifest()
    need(sha((HERE/'ORACLE-PROVENANCE.json').read_bytes())==ORACLE_PROVENANCE_SHA,
         'Frozen oracle provenance changed')
    provenance = read(HERE/'ORACLE-PROVENANCE.json')
    for name, record in provenance['files'].items():
        need(sha((HERE/name).read_bytes()) == record['portable_LF_sha256'], 'Frozen oracle/vendor source changed: '+name)
    need(sha((HERE/'vendor/dsv41_encoding.py').read_bytes()) == ENCODER_SHA, 'Frozen renderer changed')
    st_path = a.strata_source/'tools/strata_tokenizer.py'
    need(sha(st_path.read_bytes().replace(b'\r\n',b'\n')) == TOKENIZER_LF_SHA,
         'Use the pinned Strata tokenizer source obtained by the recipe checkout')
    output = a.output.resolve()
    need(output.is_relative_to((ROOT/'.work').resolve()) and not output.exists(),
         'Choose an unused output below the ignored repository .work directory')
    compressed = (HERE/manifest['corpus']).read_bytes()
    need(sha(compressed) == GZIP_SHA, 'Public corpus gzip changed')
    raw = gzip.decompress(compressed)
    need(len(raw) == 3434689 and sha(raw) == CORPUS_SHA, 'Public corpus content changed')
    archive = raw.decode('utf-8')
    # The corpus is text data: it is never imported or executed.
    sys.path.insert(0,str(HERE/'oracles'))
    original = load(HERE/'oracles/accuracy_cases.py','accuracy_cases')
    revised = load(HERE/'oracles/accuracy_cases_v2.py','portable_accuracy_cases_v2')
    encoder = load(HERE/'vendor/dsv41_encoding.py','portable_dsv41_encoding')
    ST = load(st_path,'portable_strata_tokenizer')
    candidate = packed_tokenizer(ST,a.tokenizer_dir)
    reference, decode, kind = reference_encoder(a,ST,manifest)
    cases = [(entry['case'],False,entry) for entry in manifest['cases']]
    need(len(cases) == 12 and [c['id']for c,_,_ in cases] == [c['id']for c in original.CASES[:12]],
         'Original twelve frozen task identities changed')
    fixed = next(c for c in revised.CASES if c['id']=='code-account-balance')
    cases.append((fixed,True,manifest['corrected_v2']))
    prefix = None
    rows = []
    for case,is_v2,expected in cases:
        name = case['id']+('-v2' if is_v2 else '')
        text = encoder.encode_messages([{'role':'user','content':archive+case['prompt']}],
                                       thinking_mode='thinking',reasoning_effort=75)
        ids = reference(text)
        need(candidate.encode(text,parse_special=True) == ids, 'Full reference/pack tokenizer mismatch: '+name)
        need(len(ids)==expected['prompt_tokens'] and idsha(ids)==expected['input_ids_sha256'],
             'Frozen full input count/hash differs: '+name)
        if prefix is None:
            prefix = ids[:PREFIX]
            need(idsha(prefix)==manifest['prefix_sha256'], 'Frozen 999764-token prefix differs')
        need(ids[:PREFIX] == prefix, 'Full input has a different retained prefix: '+name)
        if not is_v2:
            need(ids[PREFIX:] == expected['suffix_ids'], 'Frozen suffix IDs changed: '+name)
            need(sha(text.encode()) == expected['rendered_prompt_sha256'], 'Rendered task bytes changed: '+name)
        rows.append({'name':name,'prompt_tokens':len(ids),'input_ids_sha256':idsha(ids),
                     'tokenizer_equal':True,'suffix_ids':ids[PREFIX:],'corrected_v2':is_v2})
        print(json.dumps({'validated':name,'tokens':len(ids),'input_ids_sha256':idsha(ids)}),flush=True)
    interlude = next(c for c in original.CASES if c['id']=='filtered-decimal-revenue')
    text = encoder.encode_messages([{'role':'user','content':interlude['prompt']}],thinking_mode='thinking',reasoning_effort=75)
    short = reference(text)
    need(candidate.encode(text,parse_special=True)==short and len(short)==164 and
         idsha(short)=='22464a7c6f125403f341c2f1905a5fe2dd332d1f6ad2ad074fcf1fc1a8e8b87a',
         'Unrelated graded interlude changed')
    receipt = {'schema':'portable-frozen-input-preparation-v1','passed':True,'configured_context':1048576,
               'prefix_tokens':PREFIX,'prefix_sha256':idsha(prefix),'corpus_sha256':CORPUS_SHA,
               'reference_kind':kind,'inputs':rows,'interlude_ids':short,'input_manifest_sha256':sha((HERE/'frozen-inputs.json').read_bytes()),
               'packed_tokenizer_files_sha256':{name:sha((a.tokenizer_dir/name).read_bytes())for name in
                   ('vocab.json','merges.txt','token_type.json','tokenizer.json')},
               'strata_tokenizer_LF_sha256':TOKENIZER_LF_SHA,
               'oracle_provenance_sha256':sha((HERE/'ORACLE-PROVENANCE.json').read_bytes()),
               'no_model_or_API_started':True,'private_request_payloads_written':False,
               'generation_settings':{'temperature':0,'seed':12345,'maximum_new_tokens':4096,
                                      'repeat_penalty':1,'presence_penalty':0,'frequency_penalty':0}}
    profile_json=HERE/'data/initial-routing-profile.json'
    need(sha(profile_json.read_bytes())=='57ad1862b9d05fb816fb0645781cf0b9d4e8f9f910ffbcb6c1eb5ccbe71207cc',
         'Frozen initial routing metadata changed')
    profile_metadata=read(profile_json)
    profile=base64.b64decode(profile_metadata['data_base64'],validate=True)
    need(len(profile)==61440 and sha(profile)==profile_metadata['sha256']==
         '66b2d59fa7fa02cb6ab641846640a8e8c5fbc50906b8a1892140787e8ccf3428','Frozen initial66 profile differs')
    receipt['initial_routing_profile_sha256']=sha(profile)
    output.mkdir(parents=True,exist_ok=False)
    with (output/'prefix.u32le').open('xb') as stream:
        stream.write(struct.pack('<'+'I'*PREFIX,*prefix))
    with (output/'initial-routing-profile.bin').open('xb')as stream:stream.write(profile)
    with (output/'INPUTS.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps({'prepared':str(output),'prefix_tokens':PREFIX,'full_task_hashes_checked':13,
                      'native_processes':0,'upload_generated_ID_files':False}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--strata-source',type=Path,default=ROOT/'.work/runtime-60')
    p.add_argument('--tokenizer-dir',type=Path,default=ROOT/'.work/data/packs/strata-mix256g-1m/tokenizer')
    p.add_argument('--reference-tokenizer',type=Path)
    p.add_argument('--reference-gguf',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'.work/bench-inputs')
    a = p.parse_args()
    if not a.prepare:
        print(json.dumps({'plan_only':True,'native_or_API_calls':0,'writes':0,'full_tasks':13,
                          'retained_prefix_tokens':PREFIX,'generated_ID_files_private_and_ignored':True}))
        return 0
    need(a.reference_tokenizer is not None or a.reference_gguf is not None,
         'Pass the already-downloaded Mix shard1 as --reference-gguf')
    prepare(a)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
