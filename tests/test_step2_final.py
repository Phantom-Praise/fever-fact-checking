import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from experiment_step2_final import run_pipeline, sample_records
from score_rq1_annotations import finalize_annotations
from experiment_rq1_automated import validate_annotation


class FinalStep2(unittest.TestCase):
    def test_machine_factual_label_requires_present_evidence(self):
        value = {'label': 'SUPPORTS', 'atomic': 'YES', 'claim_type': 'PERSON', 'error_type': 'NONE',
                 'evidence_ids': [], 'reason': 'Test fixture'}
        with self.assertRaises(ValueError):
            validate_annotation(value, 2)
        value['evidence_ids'] = [2]
        with self.assertRaises(ValueError):
            validate_annotation(value, 2)
        value['evidence_ids'] = [1]
        self.assertEqual(validate_annotation(value, 2)['label'], 'SUPPORTS')

    def test_adjudication_cannot_ignore_context_disagreement(self):
        key = {'r': {'source': 'generated', 'claim': 'Example'}}
        row = {'human_label': 'NOT ENOUGH INFO', 'atomic': 'YES', 'claim_type': 'PERSON',
               'error_type': 'UNSUPPORTED', 'temporal': 'NO', 'context_required': 'NO'}
        with self.assertRaises(ValueError):
            finalize_annotations({'r': row}, {'r': dict(row, context_required='YES')}, key, [])

    def test_resolved_claim_skips_actual_retrieval_and_model(self):
        classifier, retriever = Mock(), Mock()
        result = run_pipeline('Example', classifier, retriever, lambda _: (0, 'matched'))
        self.assertEqual(result['prediction'], 0)
        self.assertEqual(result['model_called'], 0)
        self.assertEqual(result['retrieval_called'], 0)
        classifier.assert_not_called()
        retriever.assert_not_called()

    def test_outage_is_explicit_operational_fallback(self):
        verifier = Mock(side_effect=RuntimeError('HTTP unavailable'))
        classifier = Mock(return_value=2)
        retriever = Mock(return_value=[{'text': 'Evidence'}])
        result = run_pipeline('Example', classifier, retriever, verifier)
        self.assertEqual(result['reason'], 'symbolic_service_failure_fallback')
        self.assertTrue(result['service_failure'])
        self.assertEqual(result['model_called'], 1)
        classifier.assert_called_once_with('Example', 'Evidence')

    def test_heldout_selection_does_not_depend_on_labels(self):
        records = [{'id': i, 'claim': str(i), 'label_id': i % 3} for i in range(20)]
        changed = [dict(r, label_id=2) for r in records]
        self.assertEqual([r['id'] for r in sample_records(records, 10, 874, [])],
                         [r['id'] for r in sample_records(changed, 10, 874, [])])

    def test_heldout_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            sample_records([{'id': 1, 'claim': '  REPEATED claim'}], 1, 874,
                           [{'id': 2, 'claim': 'repeated   claim'}])

    def test_no_automatic_adjudication_of_disputed_labels(self):
        key = {'r': {'source': 'generated', 'claim': 'Example'}}
        row = {'human_label': 'SUPPORTS', 'atomic': 'YES', 'claim_type': 'PERSON',
               'error_type': 'NONE', 'temporal': 'NO', 'context_required': 'NO'}
        first, second = {'r': row}, {'r': dict(row, human_label='REFUTES')}
        with self.assertRaises(ValueError):
            finalize_annotations(first, second, key, [])
        resolution = {'id': 'r', **{'adjudicated_' + field: value for field, value in row.items()},
                      'adjudication_source': 'test fixture source', 'adjudication_notes': 'test fixture decision'}
        final, summary = finalize_annotations(first, second, key, [resolution])
        self.assertEqual(final[0]['resolution'], 'adjudicated')
        self.assertEqual(summary['generated']['human_label'], {'SUPPORTS': 1})


if __name__ == '__main__':
    unittest.main()
