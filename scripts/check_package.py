"""Check the compact public recipe, or the exact Git staging area, before upload."""
from pathlib import Path
import argparse
import gzip
import hashlib
import io
import json
import os
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE = 2 << 20
MAX_TOTAL = 12 << 20
DENIED = {'.gguf','.safetensors','.pt','.pth','.bin','.dll','.exe','.obj','.lib','.pdb','.log','.part','.zip'}


def candidates(staged):
    if staged:
        raw = subprocess.check_output(['git','-C',str(ROOT),'ls-files','--stage','-z'])
        entries = []
        for item in raw.split(b'\0'):
            if not item:
                continue
            metadata, name = item.split(b'\t',1)
            mode, oid, stage = metadata.split()
            if mode not in {b'100644',b'100755'} or stage != b'0':
                raise RuntimeError('Only ordinary resolved files may be staged')
            relative = name.decode('utf-8')
            size = int(subprocess.check_output(['git','-C',str(ROOT),'cat-file','-s',oid.decode('ascii')]))
            if size > MAX_FILE:
                raise RuntimeError('Individual staged file exceeds 2 MiB: '+relative)
            data = subprocess.check_output(['git','-C',str(ROOT),'cat-file','blob',oid.decode('ascii')])
            entries.append((ROOT/relative,data))
        return entries
    paths = []
    for directory,folders,files in os.walk(ROOT):
        folders[:] = [n for n in folders if n not in {'.git','.work','.venv','__pycache__'}]
        paths.extend((Path(directory)/n,None) for n in files if not n.endswith(('.pyc','.pyo')))
    return paths


def inspect(path, forbidden, data=None):
    relative = path.relative_to(ROOT).as_posix()
    if relative.startswith(('.work/','.venv/','.git/')) or path.is_symlink():
        raise RuntimeError('Private/generated or symlink entry: '+relative)
    if (path.suffix.lower() in DENIED or path.name.lower().startswith('.env.')
            or path.name.lower() in {'.env','connection.txt','server.json','macbook-access.txt','internet-settings.json'}):
        raise RuntimeError('Payload, binary, log, or private configuration is forbidden: '+relative)
    if data is None:
        data = path.read_bytes()
    if len(data) > MAX_FILE:
        raise RuntimeError('Individual file exceeds 2 MiB: '+relative)
    text = data
    if path.suffix == '.gz':
        if not relative.startswith('bench/data/'):
            raise RuntimeError('Only the documented public benchmark text may be compressed')
        with gzip.GzipFile(fileobj=io.BytesIO(data),mode='rb') as stream:
            text = stream.read(5 << 20)
            if stream.read(1):
                raise RuntimeError('Benchmark snapshot exceeds its bounded text size')
    text.decode('utf-8')
    if b'\x00' in text:
        raise RuntimeError('Unexpected binary content: '+relative)
    patterns = [rb'[A-Za-z]:[\\/]Users[\\/](?!<)',
                rb'hf_[A-Za-z0-9]{20,}',rb'sk-(?:proj-)?[A-Za-z0-9_-]{24,}',
                rb'-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----']
    private_host = os.environ.get('RECIPE_FORBIDDEN_HOST','').encode()
    if private_host:
        patterns.append(re.escape(private_host))
    if any(re.search(p,text,re.I) for p in patterns) or (forbidden and forbidden in text):
        raise RuntimeError('Private path, service hostname or credential pattern found: '+relative)
    return relative, len(data), hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--staged',action='store_true')
    ap.add_argument('--seal',action='store_true',help='Write the public file manifest after validation')
    ap.add_argument('--verify-seal',action='store_true')
    a = ap.parse_args()
    if a.seal and a.verify_seal:
        raise RuntimeError('Choose one manifest action')
    forbidden = os.environ.get('RECIPE_FORBIDDEN_SECRET','').encode()
    entries = candidates(a.staged)
    checked = [inspect(p,forbidden,data) for p,data in entries]
    rows = [row for row in checked if row[0] != 'pins/package-files.json']
    total = sum(row[1] for row in checked)
    if total > MAX_TOTAL:
        raise RuntimeError('Public recipe exceeds the 12 MiB source/evidence budget')
    files = {name:{'bytes':size,'sha256':digest} for name,size,digest in sorted(rows)}
    result = {'schema':'compact-source-recipe-package-v1','passed':True,'file_count':len(files),
              'total_bytes':total,'max_file_bytes':MAX_FILE,'budget_bytes':MAX_TOTAL,
              'weights_binaries_credentials_and_private_state_included':False,'files':files}
    pin = ROOT/'pins/package-files.json'
    if a.seal:
        pin.write_text(json.dumps(result,indent=2)+'\n')
    if a.verify_seal:
        if a.staged:
            pin_data = next((data for p,data in entries if p == pin),None)
            if pin_data is None:
                raise RuntimeError('The sealed manifest must be staged too')
            expected = json.loads(pin_data)
        else:
            expected = json.loads(pin.read_text())
        if expected['files'] != files:
            raise RuntimeError('Public file set differs from its sealed manifest')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
