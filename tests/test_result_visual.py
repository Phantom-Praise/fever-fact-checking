"""Check that chart drill-down populations match the measured cohorts and raw artifacts."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']

class SavedVisualData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT/'reports/result_analysis_20261028/visual/data.json').read_text())

    def test_every_chart_cell_matches_claim_catalog(self):
        d=self.data
        assessment=[dict(r,predictions=dict(A_original=r['original'],raw_component=r['raw_selected'],E_claim_fallback=r['selected'],neural_reference=r['neural_label'])) for r in d['waffle']]
        for models,rows in [(d['main'],d['catalog']),(d['assessment'],assessment),(d['heatmap_pairs']['pilot'],d['pilot_catalog'])]:
            for model in models:
                matrix=[[0]*len(model['matrix'][0]) for _ in LABELS]
                for r in rows:matrix[LABELS.index(r['gold'])][LABELS.index(r['predictions'][model['key']])]+=1
                self.assertEqual(matrix,model['matrix'],model['key'])

    def test_pilot_excludes_telemundo_and_assessment(self):
        d=self.data;pilot={r['id'] for r in d['pilot_catalog']};assessment={r['id'] for r in d['waffle']}
        self.assertEqual(len(pilot),300);self.assertEqual(len(assessment),120)
        self.assertNotIn(111897,pilot);self.assertFalse(pilot&assessment)

    def test_exact_requests_responses_and_fallback_are_preserved(self):
        raw={(r['config'],r['id']):r for r in map(json.loads,(ROOT/'reports/part2/llm_configurations_20261028/assessment_responses.jsonl').read_text().splitlines())}
        overrides=0
        for r in self.data['waffle']:
            for prefix,config in [('original','A_original'),('selected','D_both')]:
                self.assertEqual(r[prefix+'_request'],raw[(config,r['id'])]['raw']['request'])
                self.assertEqual(r[prefix+'_response'],raw[(config,r['id'])]['raw']['response'])
            if r['override']:
                overrides+=1;self.assertEqual(r['raw_selected'],'NOT ENOUGH INFO')
                self.assertNotEqual(r['raw_selected'],r['selected'])
        self.assertEqual(overrides,4)

    def test_neural_details_are_from_512_run(self):
        saved={r['id']:r for r in map(json.loads,(ROOT/'reports/part2/token_budget_20261028/outputs.jsonl').read_text().splitlines())}
        for row in self.data['waffle']+self.data['pilot_catalog']:
            self.assertEqual(row['neural'],saved[row['id']])
            self.assertIn('retained_tokens_512',row['neural'])

if __name__=='__main__':unittest.main()
