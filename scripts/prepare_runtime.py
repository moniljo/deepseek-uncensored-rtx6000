"""Prepare separate pinned API or native-benchmark runtimes. Default dry."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
URL='https://github.com/Niko1221/Strata.git'

def need(value,message):
    if not value:raise RuntimeError(message)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def safe(root,name):
    path=(root/name).resolve()
    need(path.is_relative_to(root.resolve()),'Runtime path escaped its declared root')
    return path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-source',type=Path,default=ROOT/'.work/source/strata')
    p.add_argument('--output',type=Path)
    p.add_argument('--asset-cache',type=Path,help='Optional existing exact retained assets; every file is still SHA-checked')
    p.add_argument('--benchmark',action='store_true',help='Keep the old reviewed frontend60 fixed for native comparison')
    p.add_argument('--prepare',action='store_true')
    a=p.parse_args()
    name='runtime-overlays.json'if a.benchmark else'api-runtime-overlays.json'
    pinpath=ROOT/'pins'/name;pin=json.loads(pinpath.read_bytes())
    overlays=ROOT/'source'/('runtime'if a.benchmark else'api-runtime')
    source=a.native_source.resolve();output=(a.output or ROOT/('.work/runtime-60'if a.benchmark else'.work/runtime')).resolve()
    if not a.prepare:
        print(json.dumps({'plan_only':True,'runtime_kind':'fixed-native-benchmark'if a.benchmark else'v027-retained-web-API',
            'source_commit':pin['base_commit'],'output':str(output),'native_build_tree_modified':False,
            'native_model_API_or_network_actions':0,'upstream_font_assets_not_bundled':len(pin.get('upstream_assets',{}))}))
        return 0
    need(not output.exists(),'Use an exclusive new runtime directory')
    if not a.benchmark:
        need(pin.get('expected_complete_runtime_files')==5111 and
             re.fullmatch('[0-9a-f]{64}',pin.get('expected_complete_runtime_tree_sha256')or''),
             'Candidate complete-runtime source proof is still pending')
    for name,digest in pin['files'].items():
        need(sha(safe(overlays,name))==digest,'Pinned runtime overlay differs: '+name)
    # The native checkout stays at the new tag. A separate pinned benchmark
    # commit may be fetched into its object database without changing HEAD/files.
    found=subprocess.run(['git','-C',str(source),'cat-file','-e',pin['base_commit']+'^{commit}'],capture_output=True)
    if found.returncode:
        subprocess.run(['git','-C',str(source),'fetch','--depth','1',URL,pin['base_commit']],check=True)
    need(subprocess.check_output(['git','-C',str(source),'rev-parse',pin['base_commit']+'^{commit}'],text=True).strip()==pin['base_commit'],
         'Runtime source commit differs')
    output.mkdir(parents=True,exist_ok=False)
    with subprocess.Popen(['git','-C',str(source),'archive','--format=tar',pin['base_commit']],stdout=subprocess.PIPE)as child:
        try:
            with tarfile.open(fileobj=child.stdout,mode='r|')as archive:
                for member in archive:
                    if not a.benchmark and(member.name=='serve/web'or member.name.startswith('serve/web/')):continue
                    safe(output,member.name)
                    archive.extract(member,output,filter='data')
            need(child.wait(timeout=30)==0,'Pinned runtime extraction failed; partial output retained')
        finally:
            if child.poll()is None:child.kill();child.wait(timeout=30)
    for name,digest in pin['files'].items():
        target=safe(output,name);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(safe(overlays,name),target)
        need(sha(target)==digest,'Copied runtime overlay differs')
    for name,entry in pin.get('upstream_assets',{}).items():
        need(entry['url'].startswith('https://raw.githubusercontent.com/Niko1221/Strata/dfa3f4025c9f1bc4af8cee201b20a2d223061813/'),
             'Unpinned retained asset URL')
        cached=safe(a.asset_cache.resolve(),name)if a.asset_cache else None
        if cached is not None and cached.is_file():
            need(cached.stat().st_size<=2<<20,'Retained asset exceeds the bounded font size')
            raw=cached.read_bytes()
        else:
            with urllib.request.urlopen(entry['url'],timeout=60)as response:raw=response.read((2<<20)+1)
        need(len(raw)<=2<<20 and hashlib.sha256(raw).hexdigest()==entry['sha256'],'Retained font asset differs')
        target=safe(output,name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    if not a.benchmark:
        web={p.relative_to(output).as_posix():sha(p)for p in(output/'serve/web').rglob('*')if p.is_file()}
        need(web==pin['retained_web_files']and len(web)==12,'Exact retained12 web closure differs')
        files={p.relative_to(output).as_posix():sha(p)for p in output.rglob('*')if p.is_file()}
        tree=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
        need(len(files)==pin['expected_complete_runtime_files']and tree==pin['expected_complete_runtime_tree_sha256'],
             'Complete027 retained-web runtime bytes differ from qualified composition')
    receipt={'schema':'prepared-separate-recipe-runtime-v2','passed':True,'base_commit':pin['base_commit'],
        'runtime_kind':'fixed-native-benchmark'if a.benchmark else'v027-retained-web-API',
        'runtime_overlay_manifest_sha256':sha(pinpath),'native_build_source_bytes_modified':False,
        'model_GPU_or_API_started':False,'loaded_API_qualification_inferred':False}
    if not a.benchmark:
        receipt.update(complete_runtime_files=len(files),complete_runtime_tree_sha256=tree,
                       complete_runtime_reconstructed=True)
    (output/'RECIPE-RUNTIME.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'prepared':str(output),'runtime_kind':receipt['runtime_kind'],'model_GPU_or_API_started':False}))
    return 0

if __name__=='__main__':raise SystemExit(main())
