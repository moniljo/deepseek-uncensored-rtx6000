"""Narrow selected-profile guard over the preserved v1 source packet."""
from pathlib import Path
from types import SimpleNamespace
import re
import owned_api_admission as base

PARENT='8c59956ec50f97c7768493b9fc1f164d62bbb7ddcaa89d9f4b26e695ccabf112'
RECIPES={'scripts/build_native.ps1','scripts/checkout_source.py',
         'pins/source-locks.json','pins/source-overlays.json'}

def selected_profile(cfg,build,need):
    env=cfg.get('env',{})
    need(env.get('DS4_PREFILL_REPEAT_REARM')=='1' and
         env.get('DS4_PREFILL_REPEAT_MIN_TOKENS')=='32768' and
         env.get('DS4_SMT') in (None,'0') and
         env.get('DS4_CPU_SMT') in (None,'0') and
         env.get('DS4_GPU_MISS_OVERLAP') in (None,'0'),
         'Selected API arm requires repeat1/min32768/noSMT/noOverlap')
    need(isinstance(build.get('recipe_pins'),dict) and set(build['recipe_pins'])==RECIPES,
         'Exact four completed-build recipe pins required')

def prepare(a,here,root,need,digest,read,pack_metadata):
    manifest=root/'pins/owned-api-adapter-v2.json'
    need(re.fullmatch('[0-9a-f]{64}',a.expected_harness_manifest_sha256 or'') and
         digest(manifest)==a.expected_harness_manifest_sha256,
         'Explicit reviewed V2 owned API source manifest required')
    mf=read(manifest)
    need(mf.get('schema')=='portable-owned-API-source-v2' and
         mf.get('parent_manifest_sha256')==PARENT,'Owned API V2 source ancestry differs')
    extra={str(manifest.resolve()):a.expected_harness_manifest_sha256}
    for name,h in mf['files_sha256'].items():
        path=(root/name).resolve()
        need(path.is_relative_to(root) and digest(path)==h,'Owned API V2 source pin differs')
        extra[str(path)]=h
    # No mutation of the shared protocol/ownership modules or their globals.
    need(Path(a.config).resolve().is_relative_to(root/'.work') and Path(a.config).is_file() and
         Path(a.config).stat().st_size<=1<<20 and digest(a.config)==a.expected_config_sha256,
         'Bounded explicitly pinned private config required before parsing')
    need(Path(a.build_receipt).is_file() and Path(a.build_receipt).stat().st_size<=4<<20 and
         digest(a.build_receipt)==a.expected_build_receipt_sha256,
         'Bounded explicitly pinned completed build required before parsing')
    selected_profile(read(a.config),read(a.build_receipt),need)
    parent_args=SimpleNamespace(**vars(a));parent_args.expected_harness_manifest_sha256=PARENT
    cfg,runtime,engine,pins,payload,dlls=base.prepare(parent_args,here,root,need,digest,read,pack_metadata)
    pins.update(extra)
    return cfg,runtime,engine,pins,payload,dlls
