"""Bounded pure tests; no frontend load, encoding, model, GPU or API calls."""
import copy
import unittest
import run_native as R
import qualification_contract as C


class Observer:
    def __init__(self,row):self.row=row
    def generate(self,*args):return self.row


def row():
    return {'native_GPU_miss_answer':{'fallbacks':0},'native_GPU_miss_direct_answer':
        {'retained_ranges':40,'retained_bytes':184926136320,'direct_unique':1,'direct_bytes':32,
         'bounce_unique':0,'bounce_bytes':0},'timestamp_sentinel':[.25,1.25],'token_ids':[7,1]}


class PublicQualification(unittest.TestCase):
    def test_post_timestamp_selected_gate_keeps_object_and_bytes(self):
        actual=row();old=copy.deepcopy(actual)
        self.assertIs(R.selected_generate(Observer(actual),'unchanged-call-args'),actual)
        self.assertEqual(actual,old)
        for key,value in [('retained_ranges',39),('retained_bytes',184926136319),('direct_unique',0),
                          ('direct_bytes',0),('bounce_unique',1),('bounce_bytes',1)]:
            changed=row();changed['native_GPU_miss_direct_answer'][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):R.selected_generate(Observer(changed))
        changed=row();changed['native_GPU_miss_answer']['fallbacks']=1
        with self.assertRaises(RuntimeError):R.selected_generate(Observer(changed))

    def test_exact_original12_formula_and_legacy_failure_stays_failed(self):
        rows=[{'emitted_token_ids':10+i,'generation_seconds':1+i,'native_decode_ms':1000*(1+i)}for i in range(14)]
        rows[0].update(ttft_seconds=2,reused_tokens=0)
        accuracy=[{'joint_passed':True}for _ in range(13)]
        wave={'rows':rows,'accuracy':accuracy};target={'ttft_seconds':3,'weighted_ID_intervals_s':.1}
        old=copy.deepcopy(wave);metrics=C.wave_metrics(wave,target)
        self.assertEqual(metrics['weighted_original12_observed_ID_intervals_s'],sum(r['emitted_token_ids']-1 for r in rows[1:13])/sum(r['generation_seconds']for r in rows[1:13]))
        self.assertEqual(wave,old)
        failed=copy.deepcopy(wave);failed['accuracy'][11]['joint_passed']=False
        current=C.wave_metrics(failed,target)
        self.assertFalse(current['all13_quality_passed']);self.assertEqual(current['original12_joint_score'],11)
        self.assertTrue(all(failed['accuracy'][i]['joint_passed']for i in (*range(11),12)))
        self.assertEqual(current['weighted_original12_observed_ID_intervals_s'],metrics['weighted_original12_observed_ID_intervals_s'])


if __name__=='__main__':unittest.main()
