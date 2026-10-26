"""Plan, resume, and verify the exact model downloads. No weights enter Git."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            value.update(block)
    return value.hexdigest()


def verify(path, item):
    return path.is_file() and path.stat().st_size == item['bytes'] and digest(path) == item['sha256']


def download(item, target):
    import requests
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not verify(target, item):
            raise RuntimeError('Existing file checksum differs; preserve it and choose another data directory: ' + target.name)
        return 'already-verified'
    part = target.with_suffix(target.suffix + '.part')
    offset = part.stat().st_size if part.exists() else 0
    if offset > item['bytes']:
        raise RuntimeError('Partial file is larger than its pinned size: ' + part.name)
    if offset < item['bytes']:
        url = ('https://huggingface.co/' + item['repository'] + '/resolve/' + item['revision'] + '/' +
               quote(item['file'], safe='/') + '?download=true')
        headers = {'User-Agent': 'deepseek-rtx6000-recipe/1', 'Accept-Encoding': 'identity'}
        token = os.environ.get('HF_TOKEN', '').strip()
        if token and not re.fullmatch(r'hf_[A-Za-z0-9]{20,}', token):
            raise RuntimeError('HF_TOKEN format is invalid; its value was not printed')
        if token:
            headers['Authorization'] = 'Bearer ' + token
        if offset:
            headers['Range'] = f'bytes={offset}-'
        # Requests strips Authorization when Hugging Face redirects to its CDN.
        with requests.get(url, headers=headers, stream=True, timeout=(15, 90)) as response:
            if response.status_code in (401, 403):
                raise RuntimeError('Accept the repository terms and set HF_TOKEN locally for ' + item['repository'])
            expected_status = 206 if offset else 200
            if response.status_code != expected_status:
                raise RuntimeError(f'Unexpected download status {response.status_code}; partial file retained')
            if offset and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                raise RuntimeError('Server did not honor the resume offset')
            length = response.headers.get('Content-Length')
            if length is not None and int(length) != item['bytes'] - offset:
                raise RuntimeError('Download length differs from the pinned file size')
            last = time.monotonic()
            with part.open('ab' if offset else 'xb') as output:
                for block in response.iter_content(8 << 20):
                    output.write(block)
                    if output.tell() > item['bytes']:
                        raise RuntimeError('Download exceeds pinned size')
                    if time.monotonic() - last >= 10:
                        print(json.dumps({'file': item['file'], 'bytes': output.tell(), 'total_bytes': item['bytes']}), flush=True)
                        last = time.monotonic()
    if not verify(part, item):
        raise RuntimeError('Complete-file SHA-256 verification failed: ' + target.name)
    part.replace(target)
    return 'downloaded-and-verified'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=ROOT / '.work/data')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--download', action='store_true')
    mode.add_argument('--verify', action='store_true')
    a = ap.parse_args()
    manifest = json.loads((ROOT / 'pins/model-downloads.json').read_text())
    base = a.data_dir.resolve()
    files = manifest['files']
    if not a.download and not a.verify:
        print(json.dumps({'plan_only': True, 'files': len(files), 'download_bytes': manifest['download_bytes'],
                          'generated_sidecar_bytes': manifest['generated_lossless_compatibility_bytes'],
                          'data_directory': str(base), 'gated_repository': files[0]['repository'],
                          'credential': 'HF_TOKEN environment variable, never a command argument'}, indent=2))
        return 0
    missing = 0
    for item in files:
        target = base / item['destination']
        existing = target if target.is_file() else target.with_suffix(target.suffix + '.part')
        missing += max(0, item['bytes'] - (existing.stat().st_size if existing.is_file() else 0))
    if a.download:
        base.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(base).free < missing + (2 << 30):
            raise RuntimeError('Insufficient free space for the remaining downloads plus a 2 GiB margin')
    rows = []
    for item in files:
        target = (base / item['destination']).resolve()
        if not target.is_relative_to(base):
            raise RuntimeError('Invalid manifest destination')
        if a.verify:
            if not verify(target, item):
                raise RuntimeError('Missing or incorrect checksum: ' + item['file'])
            action = 'verified'
        else:
            action = download(item, target)
        rows.append({'file': item['file'], 'sha256': item['sha256'], 'bytes': item['bytes'], 'action': action})
        print(json.dumps(rows[-1]), flush=True)
    receipt = {'schema': 'model-download-verification-v1', 'passed': True, 'files': rows}
    if a.download:
        (base / 'download-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
