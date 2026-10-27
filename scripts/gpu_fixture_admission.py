"""Inert portable fixture admission; called only for an explicit fixture build.

No model, server, network or GPU is used. Full source verification is an explicit
CPU-only subprocess when admit() is called, never at import or default dry.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PORT = '799413c6b137a5457becf10c90ce471d57018cedbb467708066c041b7f7003d0'
STRATA = '5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d'
GGML = '3cf03257f219afbe7334045ff7c6a06ac68c627d'


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def option(args, name):
    if name in ('-S', '-B'):
        need(args.count(name) == 1 and args.index(name) + 1 < len(args), 'One configure source/build argument required')
        return args[args.index(name) + 1]
    values = [v.split('=', 1)[1] for v in args if v.startswith(name + '=')]
    need(len(values) == 1, 'One configure argument required: ' + name)
    return values[0]


def admit(build_receipt, fixture_manifest):
    """Return the actual fresh portable build and its source/build/toolkit roots.

    The synthetic fixture's numeric methods/oracles are independently pinned by
    its caller. This function requires the new raw027 source identity, rather
    than adapting an old private BUILD to a public schema.
    """
    receipt = Path(build_receipt).resolve()
    build = read(receipt)
    locks_path, overlays_path = ROOT/'pins/source-locks.json', ROOT/'pins/source-overlays.json'
    locks, overlays = read(locks_path), read(overlays_path)
    need(build.get('schema') == 'fresh-portable-v027-V6b-native-build-v1' and
         build.get('compiled') is True and type(build.get('compile_and_link_rc')) is int and
         build['compile_and_link_rc'] == 0, 'Successful fresh portable027 source build required')
    for key in ('source_checkout_verified_before_build', 'source_checkout_verified_after_build',
                'source_overlay_verified_before_after', 'reference_raw_source_reconstructed',
                'all_native_GGML_objects_and_four_custom_static_libraries_rebuilt_from_source'):
        need(build.get(key) is True, 'Missing actual fresh source evidence: ' + key)
    need(build.get('old_objects_or_libraries_copied') is False, 'Old compiled object reuse is forbidden')
    need(build.get('upstream_Strata_commit') == locks['strata']['commit'] == STRATA and
         build.get('upstream_llama_commit') == locks['llama_cpp']['commit'] == GGML and
         build.get('ported_source_manifest_sha256') == overlays['strata_reference_stage_manifest_sha256'] == PORT,
         'The new raw027 port or its upstream/GGML identity differs')
    need(build.get('source_manifest_sha256') == build.get('source_overlay_manifest_sha256') ==
         locks['overlay_manifest_sha256'] == sha(overlays_path) and
         build.get('source_locks_sha256') == sha(locks_path) and
         Path(build['native_source_manifest']).resolve() == overlays_path.resolve(),
         'Portable source locks/overlay manifest differ from this build')
    need(build.get('complete_raw_strata_source_files') == overlays['expected_complete_strata_files'] and
         build.get('complete_raw_strata_source_tree_sha256') == overlays['expected_complete_strata_tree_sha256'],
         'Complete raw source tree was not reconstructed from the tested source')
    for name, digest in build['recipe_pins'].items():
        path = (ROOT/name).resolve()
        need(not Path(name).is_absolute() and path.is_relative_to(ROOT) and sha(path) == digest,
             'Actual relative portable build recipe pin differs')
    for required in ('scripts/build_native.ps1', 'scripts/checkout_source.py',
                     'pins/source-locks.json', 'pins/source-overlays.json'):
        need(required in build['recipe_pins'], 'Required source build recipe is unpinned')
    need(build.get('CUDA_architecture') == '120a' and 'release 13.2' in build.get('CUDA_version_output', '') and
         '19.44.' in build.get('host_compiler_version_output', ''), 'CUDA13.2/SM120a/MSVC14.44 required')
    args = build['configure_arguments']
    source, native = Path(option(args, '-S')).resolve(), Path(option(args, '-B')).resolve()
    ggml, cuda = Path(option(args, '-DSTRATA_GGML_DIR')).resolve(), Path(option(args, '-DCUDAToolkit_ROOT')).resolve()
    need(native == receipt.parent and all(p.is_dir() for p in (source, native, ggml, cuda)),
         'Actual source/build/toolkit roots are missing or differ from the receipt')
    expected = {name: overlays['strata']['src/dsv41/'+name]['sha256']
                for name in ('ds4_model.cu', 'ds4_main.cpp', 'ds4_model.hpp', 'ds4_iq.cu')}
    need(build.get('source_pin_map') == build.get('compiled_source_pin_map_raw') == expected,
         'Actual compiled four-file raw source map differs')
    for name, digest in expected.items():
        need(sha(source/'src/dsv41'/name) == digest, 'Compiled source checkout changed: ' + name)
    need(expected['ds4_model.cu'] == fixture_manifest['bound_model_raw_stage_sha256'] and
         fixture_manifest['bound_native_stage_manifest_sha256'] == PORT and
         hashlib.sha256((source/'src/dsv41/ds4_model.cu').read_bytes().replace(b'\r\n', b'\n')).hexdigest() ==
         fixture_manifest['bound_model_LF_sha256'], 'Actual extracted helpers and new compiled model differ')
    need(source.name == locks['strata']['destination'] and
         ggml == (source.parent/locks['llama_cpp']['destination']).resolve(), 'Source checkout layout differs')
    prepared = source.parent/'PREPARED-SOURCE.json'
    need(sha(prepared) == build['prepared_source_receipt_sha256'], 'Prepared-source receipt differs')
    # The verifier is already recipe-pinned above; its full source check has no
    # model/GPU action. Root runs this after any timed qualification has closed.
    checked = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/checkout_source.py'),
        '--verify', '--work-dir', str(source.parent)], capture_output=True, text=True, timeout=180)
    need(checked.returncode == 0 and len(checked.stdout) < 16384,
         'Full reconstructed source verification failed before fixture compilation')
    verification = json.loads(checked.stdout)
    need(verification.get('verified') is True, 'Full source verifier did not confirm this checkout')
    return build, source, native, ggml, cuda
