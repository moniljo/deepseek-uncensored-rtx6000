"""Pure V2 guard cases. Run only after the current timing hold."""
import copy
import json
from owned_api_admission_v2 import RECIPES,selected_profile

def need(v,m):
    if not v:raise RuntimeError(m)

def run():
    cfg={'env':{'DS4_PREFILL_REPEAT_REARM':'1','DS4_PREFILL_REPEAT_MIN_TOKENS':'32768'}}
    build={'recipe_pins':{k:'a'*64 for k in RECIPES}}
    selected_profile(cfg,build,need);count=1
    for key,value in [('DS4_PREFILL_REPEAT_REARM','0'),('DS4_PREFILL_REPEAT_MIN_TOKENS','130560'),
                      ('DS4_SMT','1'),('DS4_CPU_SMT','1'),('DS4_GPU_MISS_OVERLAP','1')]:
        bad=copy.deepcopy(cfg);bad['env'][key]=value
        try:selected_profile(bad,build,need)
        except RuntimeError:count+=1
        else:raise AssertionError('Wrong selected profile accepted')
    for recipes in ({},{**build['recipe_pins'],'extra':'a'*64},list(RECIPES)):
        try:selected_profile(cfg,{'recipe_pins':recipes},need)
        except RuntimeError:count+=1
        else:raise AssertionError('Incomplete/extra recipe identity accepted')
    return {'passed':True,'checks':count,'process_model_API_GPU_calls':0,'files_written':0}

if __name__=='__main__':print(json.dumps(run()))
