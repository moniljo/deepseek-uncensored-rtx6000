"""Prepare the fixed public API input as binary UTF8; default dry, no tokenization."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
SHA256='6c4f575406cf9303355867354ced548d7c4789830ff6f817ae1f8de83fd17c2a'
BYTES=3434819

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'.work/api-first-long.txt')
    a=p.parse_args()
    if not a.prepare:
        print(json.dumps({'default_dry':True,'writes_process_API_model_or_tokenization_calls':0,
            'output':str(a.output),'expected_UTF8_bytes':BYTES,'expected_sha256':SHA256,
            'expected_exact_answer':'828639','long_thinking_preset':'high-budget256'}));return 0
    output=a.output.resolve()
    if not output.is_relative_to(ROOT/'.work') or output.exists():
        raise RuntimeError('Choose a new exclusive input file under .work')
    sys.path.insert(0,str(ROOT/'bench'))
    import prepare_inputs as I
    raw=I.long_prompt_text('audit-sum').encode('utf-8')
    if len(raw)!=BYTES or hashlib.sha256(raw).hexdigest()!=SHA256:
        raise RuntimeError('Frozen public API prompt bytes differ; no file written')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:
        stream.write(raw)  # Binary: Windows must not translate LF to CRLF.
    print(json.dumps({'prepared':True,'bytes':len(raw),'sha256':SHA256,
        'tokenization_model_process_or_API_calls':0,'plaintext_printed':False}));return 0

if __name__=='__main__':raise SystemExit(main())
