"""Prepare pinned, overlaid source checkouts; dry unless --prepare/--verify."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
import shutil
import subprocess

REPO=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def need(value,message):
    if not value:raise ValueError(message)
def safe(root,relative):
    p=PurePosixPath(relative)
    need(not p.is_absolute() and p.parts and '..' not in p.parts and
         all(':' not in part and '\\' not in part for part in p.parts),'Unsafe source path')
    target=(root/Path(*p.parts)).resolve()
    need(target.is_relative_to(root.resolve()),'Source path escaped root')
    return target
def git(directory,*arguments):
    result=subprocess.run(['git','-C',str(directory),*arguments],check=True,
                          text=True,encoding='utf-8',stdout=subprocess.PIPE)
    return result.stdout.strip()
def metadata():
    locks=read(REPO/'pins/source-locks.json')
    manifest=safe(REPO/'pins',locks['overlay_manifest'])
    need(sha(manifest)==locks['overlay_manifest_sha256'],'Overlay manifest changed')
    overlays=read(manifest)
    for group in ('strata','ggml'):
        for name,item in overlays[group].items():
            need(sha(safe(REPO/'source'/group,name))==item['sha256'],'Overlay changed: '+name)
    for name,digest in overlays['licenses'].items():
        need(sha(safe(REPO/'source',name))==digest,'Source license changed')
    return locks,overlays
def verify(work,locks,overlays):
    receipt=read(work/'PREPARED-SOURCE.json')
    need(receipt['source_locks_sha256']==sha(REPO/'pins/source-locks.json') and
         receipt['overlay_manifest_sha256']==locks['overlay_manifest_sha256'],'Prepared source recipe changed')
    for key,group in (('strata','strata'),('llama_cpp','ggml')):
        spec=locks[key];target=safe(work,spec['destination'])
        need(git(target,'rev-parse','HEAD')==spec['commit'],'Upstream commit changed')
        for name,item in overlays[group].items():
            need(sha(safe(target,name))==item['sha256'],'Prepared overlay changed: '+name)
        # Everything outside the declared overlay must still be upstream-clean.
        changed=set(git(target,'diff','--name-only','HEAD').splitlines())
        extra=set(git(target,'ls-files','--others','--exclude-standard').splitlines())
        need((changed|extra)<=set(overlays[group]),'Undeclared checkout changes')
        need(not git(target,'diff','--name-only','--diff-filter=D','HEAD'),'A source file was removed')
    return receipt
def prepare(work,locks,overlays,caches):
    targets=[safe(work,locks[k]['destination'])for k in ('strata','llama_cpp')]
    need(not any(p.exists()for p in targets) and not(work/'PREPARED-SOURCE.json').exists(),
         'Use a new work directory; partial checkouts are preserved for inspection')
    work.mkdir(parents=True,exist_ok=True)
    for key,group in (('strata','strata'),('llama_cpp','ggml')):
        spec=locks[key];target=safe(work,spec['destination'])
        target.mkdir(exist_ok=False)
        git(target,'init','--quiet')
        git(target,'config','core.autocrlf','false')
        git(target,'config','core.longpaths','true') # repository-local; no OS/global policy change
        git(target,'remote','add','origin',spec['url'])
        fetch=str(caches[key].resolve())if caches[key]else'origin'
        git(target,'fetch','--quiet','--depth','1',fetch,spec['commit'])
        git(target,'checkout','--quiet','--detach',spec['commit'])
        need(git(target,'rev-parse','HEAD')==spec['commit'],'Fetched commit differs')
        for name,item in overlays[group].items():
            source=safe(REPO/'source'/group,name);destination=safe(target,name)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,destination)
    record={'schema':'prepared-pinned-native-source-v1','source_locks_sha256':sha(REPO/'pins/source-locks.json'),
            'overlay_manifest_sha256':locks['overlay_manifest_sha256'],
            'upstream_commits':{k:locks[k]['commit']for k in ('strata','llama_cpp')},
            'strata_source':str(targets[0]),'llama_source':str(targets[1]),
            'source_overlay_counts':{g:len(overlays[g])for g in ('strata','ggml')},
            'old_objects_libraries_copied':False,'model_or_GPU_calls':0}
    with(work/'PREPARED-SOURCE.json').open('x',encoding='utf-8')as stream:
        json.dump(record,stream,indent=2);stream.write('\n')
    return verify(work,locks,overlays)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--prepare',action='store_true');mode.add_argument('--verify',action='store_true')
    p.add_argument('--work-dir',type=Path,default=REPO/'.work/source')
    p.add_argument('--local-strata',type=Path,help='Optional existing Git object source for the SAME pinned public commit')
    p.add_argument('--local-llama',type=Path,help='Optional existing Git object source for the SAME pinned public commit')
    a=p.parse_args();locks,overlays=metadata();work=a.work_dir.resolve()
    for key in ('strata','llama_cpp'):
        need(re.fullmatch('[0-9a-f]{40}',locks[key]['commit']),'An exact full upstream commit is required')
    if not(a.prepare or a.verify):
        print(json.dumps({'dry':True,'work_dir':str(work),'upstream_commits':{k:locks[k]['commit']for k in ('strata','llama_cpp')},
                          'overlay_counts':{g:len(overlays[g])for g in ('strata','ggml')},'network_process_build_or_model_actions':0},indent=2));return 0
    result=verify(work,locks,overlays)if a.verify else prepare(work,locks,overlays,{'strata':a.local_strata,'llama_cpp':a.local_llama})
    print(json.dumps({'verified':True,'prepared':a.prepare,'source_receipt':str(work/'PREPARED-SOURCE.json'),
                      'upstream_commits':result['upstream_commits'],'model_or_GPU_calls':0},indent=2));return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except(ValueError,OSError,subprocess.CalledProcessError)as error:
        raise SystemExit('Source preparation refused: '+str(error))
