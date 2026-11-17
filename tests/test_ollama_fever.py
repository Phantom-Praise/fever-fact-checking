"""Validate local-LLM evaluation controls without invoking a model."""
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from experiment_ollama_fever import prompt, validate_response, score


class OutputControls(unittest.TestCase):
    def test_generated_prompt_never_contains_gold_metadata(self):
        row={'claim':'Example claim','sentences':[{'text':'An evidence sentence.'}],
             'gold':{'label':'SECRET_GOLD','evidence':['SECRET_ANNOTATION']}}
        result=prompt(row)
        self.assertNotIn('SECRET_GOLD',result)
        self.assertNotIn('SECRET_ANNOTATION',result)
        self.assertIn('An evidence sentence.',result)

    def test_invalid_or_incomplete_responses_cannot_become_nei(self):
        for body in [dict(done=False,message={'content':'{"label":"SUPPORTS","reason":"x"}'}),
                     dict(done=True,done_reason='length',message={'content':'{"label":"SUPPORTS","reason":"x"}'}),
                     dict(done=True,message={'content':'{"label":"maybe","reason":"x"}'}),
                     dict(done=True,message={'content':'{"label":"NOT ENOUGH INFO"}'})]:
            with self.subTest(body=body):
                with self.assertRaises((ValueError,KeyError)):validate_response(body)
        self.assertEqual(validate_response(dict(done=True,done_reason='stop',message={'content':'{"label":"REFUTES","reason":"Contradiction."}'}))['label'],'REFUTES')

    def test_invalid_responses_remain_in_accuracy_denominator(self):
        rows=[dict(id=1,predicted_evidence=[['A',0]],gold=dict(label='SUPPORTS',evidence=[[[1,1,'A',0]]])),
              dict(id=2,predicted_evidence=[],gold=dict(label='NOT ENOUGH INFO',evidence=[[[None,None,None,None]]]))]
        value=score(rows,['SUPPORTS','INVALID'])
        self.assertEqual(value['accuracy'],.5)
        self.assertEqual(value['fever_score'],.5)
        self.assertEqual(value['invalid_responses'],1)
        self.assertEqual(value['confusion_matrix'][2][3],1)
        self.assertEqual(value['per_class']['NOT ENOUGH INFO']['FN'],1)

    def test_claim_only_strict_score_submits_no_evidence(self):
        rows=[dict(id=1,predicted_evidence=[['A',0]],gold=dict(label='SUPPORTS',evidence=[[[1,1,'A',0]]])),
              dict(id=2,predicted_evidence=[],gold=dict(label='NOT ENOUGH INFO',evidence=[[[None,None,None,None]]]))]
        labels=['SUPPORTS','NOT ENOUGH INFO']
        self.assertEqual(score(rows,labels)['fever_score'],1.)
        claim_only=score(rows,labels,submit_evidence=False)
        self.assertEqual(claim_only['accuracy'],1.)
        self.assertEqual(claim_only['fever_score'],.5)


RUN=ROOT/'reports/part2/ollama_comparison_20261028'
@unittest.skipUnless((RUN/'protocol.json').exists(),'Prepared local pilot required')
class PairedSample(unittest.TestCase):
    def test_balanced_frozen_sample_and_matched_full_visible_evidence(self):
        protocol=json.loads((RUN/'protocol.json').read_text())
        rows=[json.loads(line) for line in (RUN/'sample.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),300)
        self.assertEqual(len({r['id'] for r in rows}),300)
        self.assertNotIn(111897,{r['id'] for r in rows})
        for label in ['SUPPORTS','REFUTES','NOT ENOUGH INFO']:
            self.assertEqual(sum(r['gold']['label']==label for r in rows),100)
        self.assertEqual([r['id'] for r in rows],protocol['selected_ids'])
        self.assertEqual(protocol['roberta_512_truncated_sample'],0)
        for row in rows:
            self.assertLessEqual(row['roberta_full_input_tokens'],512)
            self.assertEqual([[s['page_id'],s['sentence_id']] for s in row['sentences']],row['predicted_evidence'])


if __name__=='__main__':unittest.main()
