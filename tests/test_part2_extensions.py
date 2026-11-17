import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from build_controlled_corpora import nested_memberships
from experiment_evidence_topk import read_slice, retrieval_metrics
from local_llm_experiments import call, SCHEMA
from score_rq1_annotations import evaluate, load_annotations
import csv


class Part2Extensions(unittest.TestCase):
    def test_extension_excludes_repeated_text_under_different_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'validation.jsonl'
            rows = [{'id': 1, 'claim': 'Repeated'}, {'id': 2, 'claim': 'REPEATED'},
                    {'id': 3, 'claim': 'New claim'}, {'id': 4, 'claim': 'Another claim'}]
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            result = read_slice(path, 1, 2, excluded_ids=['1'], excluded_claims=['repeated'])
            self.assertEqual([r['id'] for r in result], [3, 4])

    def test_nested_corpora_keep_core_and_no_duplicate_pages(self):
        result = nested_memberships(['a', 'b'], ['b', 'c', 'd', 'e'], [2, 3, 5])
        self.assertEqual(result[2], ['a', 'b'])
        self.assertTrue(set(result[2]) < set(result[3]) < set(result[5]))
        self.assertEqual(len(set(result[5])), 5)
        with self.assertRaises(ValueError):
            nested_memberships(['a', 'b'], ['c'], [1])

    def test_retrieval_scoring_requires_gold(self):
        with self.assertRaises(ValueError):
            retrieval_metrics([{'label_id': 0, 'retrieved_evidence': []}], 5)

    def test_retrieval_complete_group_differs_from_any_hit(self):
        record = {'label_id': 0, 'evidence': [[[0, 0, 'Page', 1], [0, 0, 'Page', 2]]],
                  'retrieved_evidence': [{'page_id': 'Page', 'sentence_id': 1}, {'page_id': 'Page', 'sentence_id': 2}]}
        self.assertEqual(retrieval_metrics([record], 1)['any_recall'], 1)
        self.assertEqual(retrieval_metrics([record], 1)['complete_recall'], 0)
        self.assertEqual(retrieval_metrics([record], 5)['complete_recall'], 1)

    def test_local_judge_requests_have_fresh_context_and_fixed_options(self):
        session = Mock()
        session.post.return_value.json.return_value = {'message': {'content': '{"label":"SUPPORTS"}'}}
        result = call(session, 'test-model', 'Evidence task', 43, SCHEMA, .2, 32)
        payload = session.post.call_args.kwargs['json']
        self.assertEqual(payload['messages'], [{'role': 'user', 'content': 'Evidence task'}])
        self.assertEqual(payload['options']['seed'], 43)
        self.assertFalse(payload['stream'])
        self.assertEqual(result['response']['message']['content'], '{"label":"SUPPORTS"}')

    def test_human_disagreements_require_adjudication(self):
        key = {'review_000': {'claim': 'Example claim', 'source': 'generated'}}
        common = {'human_label': 'SUPPORTS', 'atomic': 'YES', 'claim_type': 'PERSON', 'error_type': 'NONE'}
        first = {'review_000': dict(common)}
        second = {'review_000': dict(common, human_label='REFUTES', error_type='ENTITY')}
        result = evaluate(first, second, key)
        self.assertEqual(result['label_agreement'], 0)
        self.assertEqual(len(result['requires_adjudication']), 1)
        self.assertEqual(result['requires_adjudication'][0]['adjudicated_label'], '')

    def test_blank_human_reference_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'review.csv'
            path.write_text('id,claim,human_label,atomic\nreview_000,Example claim,,\n')
            with self.assertRaises(ValueError):
                load_annotations(path, {'review_000': {'claim': 'Example claim'}})


if __name__ == '__main__':
    unittest.main()
