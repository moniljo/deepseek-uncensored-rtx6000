"""Create this computer's private 1M configuration. Dry unless --write."""
from pathlib import Path
import argparse
import json
import os
import secrets
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=ROOT/'.work/runtime')
    ap.add_argument('--engine', type=Path, default=ROOT/'.work/build-native/strata-dsv41.exe')
    ap.add_argument('--pack', type=Path, default=ROOT/'.work/data/packs/strata-mix256g-1m')
    ap.add_argument('--cuda-root', type=Path, default=os.environ.get('CUDA_PATH'))
    ap.add_argument('--local-dir', type=Path, default=ROOT/'.work/local')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    source, engine, pack, local = [p.resolve() for p in (a.source,a.engine,a.pack,a.local_dir)]
    if not 1024 <= a.port <= 65535:
        raise RuntimeError('Choose a port from 1024 through 65535')
    if not a.write:
        print(json.dumps({'plan_only':True,'model':'deepseek-orcarouter-mix256g-uncensored-experimental',
                          'total_context':1048576,'source':str(source),'engine':str(engine),'pack':str(pack),
                          'private_config':str(local/'server.json'),'host':'127.0.0.1','port':a.port,
                          'credential':'reuse STRATA_API_KEY from the environment, or generate a new local key'},indent=2))
        return 0
    if not engine.is_file() or not (source/'serve/server.py').is_file():
        raise RuntimeError('Build the engine and prepare the API runtime first')
    for file in ('ds4.txt','tensors.txt','engram.bin','tokenizer/vocab.json','tokenizer/merges.txt'):
        if not (pack/file).is_file():
            raise RuntimeError('Incomplete model pack: '+file)
    if not a.cuda_root:
        raise RuntimeError('Pass --cuda-root for the CUDA13.2 toolkit')
    candidates = [a.cuda_root.resolve()/'bin/x64', a.cuda_root.resolve()/'bin']
    cuda_bin = next((p for p in candidates if all((p/n).is_file() for n in ('cublas64_13.dll','cublasLt64_13.dll'))),None)
    if cuda_bin is None:
        raise RuntimeError('CUDA13 cuBLAS DLLs are missing from the selected toolkit bin directory')
    if (local/'server.json').exists():
        raise RuntimeError('Private configuration already exists; preserve it or choose a different --local-dir')
    local.mkdir(parents=True,exist_ok=True)
    profile = local/'expert-profile.bin'
    if profile.exists():
        raise RuntimeError('An existing profile was found; use another --local-dir')
    shutil.copyfile(pack/'expert-profile.bin',profile)
    (local/'slots').mkdir(exist_ok=True)
    key = os.environ.get('STRATA_API_KEY','').strip() or secrets.token_urlsafe(32)
    if any(c in key for c in '\r\n'):
        raise RuntimeError('Invalid API key; its value was not printed')
    values = {'ENGINE':str(engine),'STRATA_SOURCE':str(source),'PACK':str(pack),'PROFILE':str(profile),
              'CUDA_BIN':str(cuda_bin),'API_KEY':key,'SLOT_DIR':str(local/'slots'),'LOG':str(local/'engine.log')}
    def expand(value):
        if isinstance(value,dict): return {k:expand(v) for k,v in value.items()}
        if isinstance(value,list): return [expand(v) for v in value]
        if not isinstance(value,str): return value
        for name,replacement in values.items():
            value = value.replace('{{'+name+'}}',replacement)
        if '{{' in value:
            raise RuntimeError('Unexpanded template value')
        return value
    config = expand(json.loads((ROOT/'config/server-1m.template.json').read_text()))
    config['port'] = a.port
    context_flag = config['args'].index('--max-context')
    if config['args'][context_flag+1] != '1048576':
        raise RuntimeError('The selected context changed')
    (local/'server.json').write_text(json.dumps(config,indent=2)+'\n')
    # This credential-bearing file is private, ignored, and never part of Git.
    connection = (f'OpenAI base: http://127.0.0.1:{a.port}/v1\n'
                  f'Anthropic base: http://127.0.0.1:{a.port}\n'
                  f'Model: {config["model_name"]}\nAPI key: {key}\nTotal context: 1048576\n')
    (local/'connection.txt').write_text(connection)
    print(json.dumps({'written':str(local/'server.json'),'private_connection_file':str(local/'connection.txt'),
                      'API_key_printed':False,'model':config['model_name'],'context':1048576}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
