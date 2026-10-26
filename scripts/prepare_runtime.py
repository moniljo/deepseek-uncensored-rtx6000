"""Prepare the pinned Python/API runtime separately from the native build tree."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native-source',type=Path,default=ROOT/'.work/source/strata')
    ap.add_argument('--output',type=Path,default=ROOT/'.work/runtime')
    ap.add_argument('--prepare',action='store_true')
    a = ap.parse_args()
    pin = json.loads((ROOT/'pins/runtime-overlays.json').read_text())
    source, output = a.native_source.resolve(), a.output.resolve()
    if not a.prepare:
        print(json.dumps({'plan_only':True,'source':str(source),'output':str(output),
                          'upstream_commit':pin['base_commit'],'runtime_overlay_files':len(pin['files'])}))
        return 0
    head = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if head != pin['base_commit']:
        raise RuntimeError('The source checkout is not the pinned upstream commit')
    if output.exists():
        raise RuntimeError('Use a new output directory; an existing runtime is never overwritten')
    for name,digest in pin['files'].items():
        path = (ROOT/'source/runtime'/name).resolve()
        if not path.is_relative_to((ROOT/'source/runtime').resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Runtime overlay differs: '+name)
    output.mkdir(parents=True)
    # Archive the pinned COMMIT, not a possibly modified working directory.
    with subprocess.Popen(['git','-C',str(source),'archive','--format=tar',head],stdout=subprocess.PIPE) as process:
        with tarfile.open(fileobj=process.stdout,mode='r|') as archive:
            archive.extractall(output,filter='data')
        if process.wait() != 0:
            raise RuntimeError('Pinned runtime extraction failed; partial output retained')
    for name,digest in pin['files'].items():
        target = (output/name).resolve()
        if not target.is_relative_to(output):
            raise RuntimeError('Invalid runtime overlay destination')
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/'source/runtime'/name,target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Runtime overlay copy differs')
    receipt = {'schema':'prepared-API-runtime-v1','passed':True,'base_commit':head,
               'runtime_overlay_sha256':hashlib.sha256((ROOT/'pins/runtime-overlays.json').read_bytes()).hexdigest(),
               'native_build_tree_modified':False,'model_or_GPU_started':False}
    (output/'RECIPE-RUNTIME.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'prepared':str(output),'overlay_files':len(pin['files']),'native_build_tree_modified':False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
