"""Pilot sampling and scoring controls, with no model or network requests."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment_ephemeral_rag import LABELS,sample_rows,condition_metrics,old_prediction

class PilotControls(unittest.TestCase):
    def test_stratification_is_unique_reproducible_and_excludes_seen_texts(self):
        rows=[dict(id=i*10+j,claim=label+' claim '+str(j),label=label) for i,label in enumerate(LABELS) for j in range(10)]
        excluded={rows[0]['claim'].casefold()};a=sample_rows(rows,excluded);b=sample_rows(rows,excluded)
        self.assertEqual([r['id'] for r in a],[r['id'] for r in b]);self.assertEqual(len({r['claim'] for r in a}),10)
        self.assertEqual([sum(r['label']==l for r in a) for l in LABELS],[4,3,3]);self.assertNotIn(rows[0],a)

    def test_invalid_answers_stay_in_denominator(self):
        rows=[dict(gold='SUPPORTS',results={'c':dict(label='SUPPORTS',wall_seconds=1)}),dict(gold='REFUTES',results={'c':dict(label='INVALID',wall_seconds=1)})]
        m=condition_metrics(rows,'c');self.assertEqual(m['accuracy'],.5);self.assertEqual(m['invalid'],1)

    def test_original_prompt_never_receives_gold(self):
        def generate(prompt):
            self.assertNotIn('SECRET_GOLD',prompt)
            return {'response':{'done':True,'done_reason':'stop','message':{'content':'{"label":"SUPPORTS","reason":"Evidence supports it."}'}}}
        r=old_prediction(dict(claim='A claim',gold='SECRET_GOLD',original_evidence=[{'text':'An evidence sentence.'}]),generate)
        self.assertEqual(r['label'],'SUPPORTS')

if __name__=='__main__':unittest.main()
