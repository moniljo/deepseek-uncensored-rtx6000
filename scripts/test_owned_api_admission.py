"""Pure metadata checks; no model/process/API/filesystem mutation."""
import copy
import json
import owned_api_admission as A

def need(v,m):
    if not v:raise RuntimeError(m)

def run():
    names=('ds4_model.cu','ds4_main.cpp','ds4_model.hpp','ds4_iq.cu')
    pins={n:hex(i+1)[2:]*64 for i,n in enumerate(names)}
    overlays={'strata_reference_stage_manifest_sha256':A.PORT,'expected_complete_strata_files':5185,
              'expected_complete_strata_tree_sha256':'a'*64,
              'strata':{'src/dsv41/'+n:{'reference_raw_sha256':h}for n,h in pins.items()}}
    b={'schema':'fresh-portable-v027-V6b-native-build-v1','compiled':True,'compile_and_link_rc':0,
       'old_objects_or_libraries_copied':False,'API_model_or_service_execution':False,
       'ported_source_manifest_sha256':A.PORT,'complete_raw_strata_source_files':5185,
       'complete_raw_strata_source_tree_sha256':'a'*64,'source_pin_map':pins,'compiled_source_pin_map_raw':pins,
       'CUDA_architecture':'120a','CUDA_version_output':'release 13.2','host_compiler_version_output':'19.44.1',
       'build_id':'5e8a4cca-strata027-custom-v6b-port2-sm120a'}
    flags=('source_checkout_verified_before_build','source_checkout_verified_after_build',
           'source_overlay_verified_before_after','reference_raw_source_reconstructed',
           'all_native_GGML_objects_and_four_custom_static_libraries_rebuilt_from_source',
           'external_prebuilt_CUDA_toolkit_libraries','qualification_not_inferred_from_compilation')
    b.update({x:True for x in flags});A.build_contract(b,overlays,need);count=1
    for field,value in [*[(x,False)for x in flags],('schema','fresh-portable-native-build-v1'),
        ('compile_and_link_rc',True),('compile_and_link_rc',1),('old_objects_or_libraries_copied',True),
        ('API_model_or_service_execution',True),('source_pin_map',{}),('ported_source_manifest_sha256','b'*64),
        ('complete_raw_strata_source_files',5000),('complete_raw_strata_source_tree_sha256','b'*64),
        ('CUDA_architecture','120'),('CUDA_version_output','release 13.1'),('host_compiler_version_output','19.50')]:
        bad=copy.deepcopy(b);bad[field]=value
        try:A.build_contract(bad,overlays,need)
        except RuntimeError:count+=1
        else:raise AssertionError('Invalid actual build accepted: '+field)
    return {'passed':True,'checks':count,'process_model_API_GPU_calls':0,'files_written':0}

if __name__=='__main__':print(json.dumps(run()))
