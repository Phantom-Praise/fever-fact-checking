"""Full-dev controls: safe resumption, invalid handling, and correct evidence scoring."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from experiment_ollama_full_dev import analyze, make_prompt, request_one, validate_pairing, verify_prefix


def row(i=1):
    evidence=[dict(page_id='A',sentence_id=0,text='A fact.')]
    return dict(id=i,claim='A claim.',gold=dict(id=i,claim='A claim.',label='SUPPORTS',evidence=[[[1,1,'A',0]]]),
                evidence=dict(original=evidence,entity=[]),claim_probabilities=[.9,.05,.05],previous_llm_text=False,
                baseline_predictions=dict(evidence_512=0,claim_only_256=0))


class FullDevControls(unittest.TestCase):
    def test_full_population_rejects_a_subset(self):
        with self.assertRaises(ValueError):validate_pairing([],[],[])

    def test_resume_rejects_mismatched_id_configuration_or_prompt(self):
        r=row();good=dict(id=1,config='A_original',raw=dict(request=dict(messages=[dict(role='user',content=make_prompt(r,'A_original'))])))
        verify_prefix([r],[good],'A_original')
        for changed in [dict(good,id=2),dict(good,config='D_both'),dict(good,raw=dict(request=dict(messages=[])))]:
            with self.assertRaises(ValueError):verify_prefix([r],[changed],'A_original')

    def test_incomplete_response_stays_invalid_with_raw_response(self):
        raw=dict(request={},response=dict(done=True,done_reason='length',message=dict(content='{"label":"SUPPORTS","reason":"A fact."}')))
        with patch('experiment_ollama_full_dev.call',return_value=raw):result=request_one(None,'model',row(),'A_original')
        self.assertEqual(result['label'],'INVALID');self.assertEqual(result['raw'],raw)

    def test_hybrid_gain_cannot_claim_original_evidence_and_reanalysis_is_safe(self):
        r=row();result=dict(id=1,label='NOT ENOUGH INFO',wall_seconds=1.)
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);protocol=dict(model={'name':'test'},limitations=[])
            analyze(out,[r],[result],'D_both',protocol)
            analyze(out,[r],[result],'D_both',protocol)
            summary=json.loads((out/'D_both_summary.json').read_text())
            self.assertEqual(summary['baselines']['evidence_512']['fever_score'],1.)
            self.assertEqual(summary['baselines']['claim_only_256']['fever_score'],0.)
            self.assertEqual(summary['derived_hybrid']['accuracy'],1.)
            self.assertEqual(summary['derived_hybrid']['fever_score'],0.)
            self.assertEqual(summary['derived_hybrid']['overrides'],1)
            self.assertEqual(summary['condition']['accuracy'],0.)

if __name__=='__main__':unittest.main()
