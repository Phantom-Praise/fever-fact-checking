"""Check paired error accounting and frozen-input experiment controls."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from experiment_token_budget import comparison, per_class


class PairedAccounting(unittest.TestCase):
    def test_fixes_and_regressions_account_for_accuracy_change(self):
        gold = np.array([0, 1, 2, 0, 1])
        old = [0, 0, 2, 2, 2]
        new = [1, 1, 2, 0, 0]
        result = comparison(gold, old, new)
        self.assertEqual(result['fixed_errors'], 2)
        self.assertEqual(result['newly_wrong'], 1)
        self.assertEqual(result['both_correct'], 1)
        self.assertEqual(result['both_wrong'], 1)
        self.assertAlmostEqual(result['accuracy_delta'], .2)
        self.assertEqual(result['n'], sum(result[k] for k in ('fixed_errors','newly_wrong','both_correct','both_wrong')))

    def test_unchanged_predictions_have_no_accuracy_effect(self):
        result = comparison(np.array([0,1,2]), [0,0,1], [0,0,1])
        self.assertEqual(result['paired_accuracy_ci95'], [0,0])
        self.assertEqual(result['mcnemar_exact_p_unadjusted'], 1.)

    def test_one_vs_rest_counts_sum_to_population(self):
        matrix = [[4,1,0],[2,3,1],[0,1,5]]
        values = per_class(matrix)
        self.assertEqual({k:values['SUPPORTS'][k] for k in ('TP','FP','FN','TN')},
                         dict(TP=4,FP=2,FN=1,TN=10))
        for label in values:
            self.assertEqual(sum(values[label][k] for k in ('TP','FP','FN','TN')), 17)
        self.assertEqual(sum(v['FP'] for v in values.values()),5)
        self.assertEqual(sum(v['FN'] for v in values.values()),5)


RUN = ROOT / 'reports/part2/token_budget_20261028_smoke_verified'
@unittest.skipUnless((RUN/'summary.json').exists(),'Local smoke artifacts required')
class FrozenInputControls(unittest.TestCase):
    def test_baseline_replay_and_exact_evidence_pairing(self):
        replay = json.loads((RUN/'baseline_replay.json').read_text())
        self.assertEqual(replay['label_mismatches'],0)
        self.assertLess(replay['max_probability_absolute_difference'],1e-5)
        saved = {r['id']:r for r in map(json.loads,(ROOT/'reports/part2/official_fever_dev_20261023/outputs.jsonl').read_text().splitlines())}
        rows = list(map(json.loads,(RUN/'outputs.jsonl').read_text().splitlines()))
        for r in rows:
            original = saved[r['id']]
            self.assertEqual(r['claim'],original['claim'])
            self.assertEqual(r['predicted_evidence'],original['predicted_evidence'])
            self.assertEqual(r['model_probabilities']['claim'],original['model_probabilities']['claim'])
            self.assertLessEqual(r['retained_tokens_512'],512)
            self.assertAlmostEqual(sum(r['model_probabilities']['evidence']),1.,places=5)
        telemundo = next(r for r in rows if r['id']==111897)
        self.assertEqual(telemundo['retained_tokens_512'],telemundo['full_input_tokens'])
        self.assertEqual(telemundo['full_input_tokens'],347)
        self.assertEqual(telemundo['predictions']['retrieved_roberta'],0)


if __name__=='__main__':unittest.main()
