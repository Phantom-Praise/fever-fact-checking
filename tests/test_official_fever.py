import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evaluate_official_fever import official_metrics
from vendor.fever_scorer import is_strictly_correct


class OfficialFever(unittest.TestCase):
    def test_label_correct_without_evidence_is_not_strict_correct(self):
        row={'label':'SUPPORTS','predicted_label':'SUPPORTS','evidence':[[[1,2,'A',0]]],'predicted_evidence':[]}
        self.assertFalse(is_strictly_correct(row,5))
        row['predicted_evidence']=[['A',0]]
        self.assertTrue(is_strictly_correct(row,5))

    def test_complete_group_and_top_five_are_required(self):
        row={'label':'REFUTES','predicted_label':'REFUTES','evidence':[[[1,2,'A',0],[1,3,'B',1]]],
             'predicted_evidence':[['A',0],['X',0],['X',1],['X',2],['X',3],['B',1]]}
        self.assertFalse(is_strictly_correct(row,5))
        self.assertTrue(is_strictly_correct(row,6))

    def test_nei_requires_correct_label_but_no_evidence(self):
        row={'label':'NOT ENOUGH INFO','predicted_label':'NOT ENOUGH INFO','evidence':[[[1,2,None,None]]],'predicted_evidence':[]}
        self.assertTrue(is_strictly_correct(row,5))

    def test_mismatched_sample_order_rejected(self):
        with self.assertRaises(ValueError):
            official_metrics([{'id':1}],[{'id':2}])

    def test_official_label_and_evidence_scores_are_distinct(self):
        gold=[{'id':1,'label':'SUPPORTS','evidence':[[[1,2,'A',0]]]},
              {'id':2,'label':'NOT ENOUGH INFO','evidence':[[[1,3,None,None]]]}]
        pred=[{'id':1,'predicted_label':'SUPPORTS','predicted_evidence':[]},
              {'id':2,'predicted_label':'NOT ENOUGH INFO','predicted_evidence':[]}]
        result=official_metrics(gold,pred)
        self.assertEqual(result['label_accuracy'],1)
        self.assertEqual(result['fever_score'],.5)
        self.assertNotIn('label',pred[0])


if __name__=='__main__': unittest.main()
