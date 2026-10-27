"""Pure build-admission tests; no checkout, compiler, GPU, model or process call."""
from pathlib import Path
from unittest import mock
import argparse,copy,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import gpu_fixture_admission as A

def test_admission():
    model=b'synthetic model bytes\n';model_sha=hashlib.sha256(model).hexdigest()
    source=ROOT/'.work/synthetic/strata';ggml=source.parent/'llama.cpp';native=ROOT/'.work/synthetic-build';cuda=ROOT/'.work/synthetic-cuda'
    receipt=native/'BUILD.json'
    names=('ds4_model.cu','ds4_main.cpp','ds4_model.hpp','ds4_iq.cu')
    four={name:model_sha if name=='ds4_model.cu'else str(i)*64 for i,name in enumerate(names,1)}
    locks={'strata':{'commit':A.STRATA,'destination':'strata'},'llama_cpp':{'commit':A.GGML,'destination':'llama.cpp'},'overlay_manifest_sha256':'a'*64}
    overlays={'strata_reference_stage_manifest_sha256':A.PORT,'expected_complete_strata_files':5185,'expected_complete_strata_tree_sha256':'b'*64,
        'strata':{'src/dsv41/'+name:{'sha256':digest}for name,digest in four.items()}}
    fixture={'bound_model_raw_stage_sha256':model_sha,'bound_native_stage_manifest_sha256':A.PORT,'bound_model_LF_sha256':model_sha}
    recipes={n:'c'*64 for n in ('scripts/build_native.ps1','scripts/checkout_source.py','pins/source-locks.json','pins/source-overlays.json')}
    recipes['pins/source-overlays.json']='a'*64;recipes['pins/source-locks.json']='d'*64
    build={'schema':'fresh-portable-v027-V6b-native-build-v1','compiled':True,'compile_and_link_rc':0,
        **{k:True for k in ('source_checkout_verified_before_build','source_checkout_verified_after_build','source_overlay_verified_before_after',
            'reference_raw_source_reconstructed','all_native_GGML_objects_and_four_custom_static_libraries_rebuilt_from_source')},
        'old_objects_or_libraries_copied':False,'upstream_Strata_commit':A.STRATA,'upstream_llama_commit':A.GGML,'ported_source_manifest_sha256':A.PORT,
        'source_manifest_sha256':'a'*64,'source_overlay_manifest_sha256':'a'*64,'source_locks_sha256':'d'*64,
        'native_source_manifest':str(ROOT/'pins/source-overlays.json'),'complete_raw_strata_source_files':5185,
        'complete_raw_strata_source_tree_sha256':'b'*64,'recipe_pins':recipes,'CUDA_architecture':'120a','CUDA_version_output':'release 13.2',
        'host_compiler_version_output':'19.44.35217','configure_arguments':['-S',str(source),'-B',str(native),'-DSTRATA_GGML_DIR='+str(ggml),'-DCUDAToolkit_ROOT='+str(cuda)],
        'source_pin_map':four,'compiled_source_pin_map_raw':four,'prepared_source_receipt_sha256':'e'*64}
    digests={ROOT/n:d for n,d in recipes.items()};digests[source.parent/'PREPARED-SOURCE.json']='e'*64
    digests.update({source/'src/dsv41'/n:d for n,d in four.items()})
    def invoke(value,fixture_value=fixture):
        def fake_read(p):
            p=Path(p)
            if p==receipt:return value
            if p==ROOT/'pins/source-locks.json':return locks
            if p==ROOT/'pins/source-overlays.json':return overlays
            raise AssertionError('Unexpected read')
        with mock.patch.object(A,'read',side_effect=fake_read),mock.patch.object(A,'sha',side_effect=lambda p:digests[Path(p)]),\
             mock.patch.object(Path,'is_dir',return_value=True),mock.patch.object(Path,'read_bytes',return_value=model),\
             mock.patch.object(A.subprocess,'run',return_value=type('Completed',(),{'returncode':0,'stdout':'{"verified":true}'})()) as run:
            result=A.admit(receipt,fixture_value)
            assert result[1:]==(source,native,ggml,cuda)
            run.assert_called_once()
            assert '--verify'in run.call_args.args[0] and '--work-dir'in run.call_args.args[0]
    invoke(build);checks=1
    changes=[('schema','fresh-portable-native-build-v1'),('schema','fresh-custom-v027-V6b-port-native-build-v1'),
        ('compiled',False),('compile_and_link_rc',True),('compile_and_link_rc',1),('old_objects_or_libraries_copied',True),
        ('ported_source_manifest_sha256','0'*64),('source_manifest_sha256','0'*64),('source_locks_sha256','0'*64),
        ('source_pin_map',{}),('compiled_source_pin_map_raw',{}),('CUDA_architecture','120'),('CUDA_version_output','release 13.1'),
        ('complete_raw_strata_source_files',5184),('complete_raw_strata_source_tree_sha256','0'*64),('prepared_source_receipt_sha256','0'*64)]
    changes += [(k,False)for k in ('source_checkout_verified_before_build','source_checkout_verified_after_build','source_overlay_verified_before_after',
        'reference_raw_source_reconstructed','all_native_GGML_objects_and_four_custom_static_libraries_rebuilt_from_source')]
    for key,value in changes:
        bad=copy.deepcopy(build);bad[key]=value
        try:invoke(bad)
        except RuntimeError:checks+=1
        else:raise AssertionError('Invalid synthetic build accepted: '+key)
    return {'passed':True,'pure_checks':checks,'actual_compiler_process_checkout_GPU_model_or_API_calls':0,
        'synthetic_metadata_NOT_an_actual_build_receipt':True}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    print(json.dumps(test_admission()if a.self_test else {'default_dry':True,'actions':0}))
