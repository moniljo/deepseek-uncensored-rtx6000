"""Build the lossless Q6 compatibility pack and add the pinned DSpark metadata."""
from pathlib import Path
import argparse
import base64
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=ROOT / '.work/data')
    ap.add_argument('--model-dir', type=Path)
    ap.add_argument('--draft', type=Path)
    ap.add_argument('--q6-pack', type=Path)
    ap.add_argument('--pack', type=Path)
    ap.add_argument('--packing-workdir', type=Path, default=ROOT/'.work/packing')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--codec-test', action='store_true')
    a = ap.parse_args()
    if a.run and a.codec_test:
        raise RuntimeError('Choose one action')
    if a.codec_test:
        q6 = load('recipe_q6', ROOT / 'tools/q6compat.py')
        print(json.dumps(q6.codec_tests(), indent=2))
        return 0
    data = a.data_dir.resolve()
    model = (a.model_dir or data/'models/Mix-256G').resolve()
    draft = (a.draft or data/'models/dspark/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf').resolve()
    q6_pack = (a.q6_pack or data/'packs/deepseek-orca-mix256g-q6compat').resolve()
    final = (a.pack or data/'packs/strata-mix256g-1m').resolve()
    if not a.run:
        print(json.dumps({'plan_only':True, 'model':str(model), 'drafter':str(draft),
                          'lossless_pack':str(q6_pack), 'final_pack':str(final),
                          'generated_sidecar_bytes':33443020800,
                          'original_GGUF_writes':False, 'original_model_requantization':False}, indent=2))
        return 0
    if final.exists():
        raise RuntimeError('Final pack already exists; preserve it and use another --pack destination')
    q6 = load('recipe_q6', ROOT / 'tools/q6compat.py')
    manifest = json.loads((ROOT/'pins/model-downloads.json').read_text())
    # Independently verify complete original files, including existing downloads.
    for row in manifest['files']:
        path = model/Path(row['file']).name if row['file'].startswith('Mix-256G/') else draft
        print(json.dumps({'verifying':row['file'], 'bytes':row['bytes']}), flush=True)
        if not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise RuntimeError('Pinned model checksum mismatch: '+row['file'])
    if not q6_pack.exists():
        q6.MODEL, q6.TARGET = model, q6_pack
        q6.build()
    audit = json.loads((q6_pack/'q6compat-manifest.json').read_text())
    if audit['different_f32_bits'] != 0 or audit['audited_values'] != manifest['compatibility_audited_values']:
        raise RuntimeError('Lossless audit differs from the selected model')
    expected = {r['converted_file']:r for r in manifest['compatibility_files']}
    for row in audit['tensors']:
        name = Path(row['converted_file']).name
        pin = expected.get(name)
        path = q6_pack/'q6compat'/name
        if not pin or path.stat().st_size != pin['converted_bytes'] or digest(path) != pin['converted_sha256']:
            raise RuntimeError('Generated compatibility checksum differs: '+name)
    if len(audit['tensors']) != len(expected):
        raise RuntimeError('Compatibility tensor count differs')
    geometry = dict(line.split(' ',1) for line in (q6_pack/'ds4.txt').read_text().splitlines() if ' ' in line)
    if Path(geometry['model_dir']).resolve() != model:
        raise RuntimeError('Existing compatibility pack points to another model directory')
    work = a.packing_workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    final.parent.mkdir(parents=True, exist_ok=True)
    drafter_pin = manifest['files'][-1]
    receipt = {'path':str(draft),'source_repository':drafter_pin['repository'],'revision':drafter_pin['revision'],
               'bytes':drafter_pin['bytes'],'SHA256':drafter_pin['sha256'],'verified':True}
    receipt_path = work/'dspark-download.json'
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise RuntimeError('Existing drafter receipt differs')
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    profile = work/'frozen-expert-profile.bin'
    profile_metadata = json.loads((ROOT/'bench/data/initial-routing-profile.json').read_text())
    initial_profile = base64.b64decode(profile_metadata['data_base64'],validate=True)
    if (len(initial_profile) != 61440 or hashlib.sha256(initial_profile).hexdigest() !=
            '66b2d59fa7fa02cb6ab641846640a8e8c5fbc50906b8a1892140787e8ccf3428'):
        raise RuntimeError('The declared initial routing metadata differs')
    if profile.exists():
        if profile.read_bytes() != initial_profile:
            raise RuntimeError('Existing frozen profile differs')
    else:
        profile.write_bytes(initial_profile)
    merge = load('recipe_dspark', ROOT/'tools/add_dspark.py')
    merge.OLD, merge.FINAL = q6_pack, final
    merge.HERE, merge.RAW = work, work/'raw-dspark-pack'
    saved = sys.argv
    try:
        sys.argv = [str(ROOT/'tools/add_dspark.py'),'--run']
        return merge.main()
    finally:
        sys.argv = saved


if __name__ == '__main__':
    raise SystemExit(main())
