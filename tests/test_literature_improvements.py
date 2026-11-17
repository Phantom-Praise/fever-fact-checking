import sys
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evidence_pipeline_v2 import subject_prefix, evidence_text, fuse_probabilities, retrieve_bm25
from experiment_literature_improvements import split_validation
from prepare_retrieved_training import training_eligibility


class LiteratureImprovements(unittest.TestCase):
    def test_subject_names_keep_final_s(self):
        self.assertEqual(subject_prefix('Zeus is a Greek god.'), 'Zeus')
        self.assertEqual(subject_prefix('James was born in London.'), 'James')
        self.assertEqual(subject_prefix("James's work is well known."), "James's work")
        self.assertIsNone(subject_prefix('A sentence without a predicate.'))

    def test_optional_titles_keep_page_identity(self):
        rows = [{'page_id': 'Example_-LRB-film-RRB-', 'text': 'It was released.'}]
        self.assertEqual(evidence_text(rows), 'It was released.')
        self.assertEqual(evidence_text(rows, True), 'Example (film): It was released.')

    def test_subject_linking_adds_missing_page_and_handles_empty_tokens(self):
        connection = sqlite3.connect(':memory:')
        self.addCleanup(connection.close)
        connection.execute('CREATE TABLE wiki_titles(page_id TEXT)')
        connection.execute('CREATE TABLE page_sentences(page_id TEXT, sentence_id INTEGER, text TEXT)')
        connection.execute("INSERT INTO wiki_titles VALUES ('Zeus')")
        connection.execute("INSERT INTO page_sentences VALUES ('Zeus', 0, '!!!')")
        with patch('evidence_pipeline_v2.find_titles', return_value=[]):
            self.assertEqual(retrieve_bm25(connection, 'Zeus is a god.', link_subject=False), [])
            linked = retrieve_bm25(connection, 'Zeus is a god.', link_subject=True)
        self.assertEqual(linked[0]['page_id'], 'Zeus')
        self.assertTrue(linked[0]['subject_page_match'])
        self.assertTrue(np.isfinite(linked[0]['score']))

    def test_fusion_endpoints_and_nei_bias(self):
        claim = [[.7, .2, .1]]
        evidence = [[.1, .8, .1]]
        self.assertEqual(fuse_probabilities(claim, evidence, 1, 0).tolist(), [0])
        self.assertEqual(fuse_probabilities(claim, evidence, 0, 0).tolist(), [1])
        self.assertEqual(fuse_probabilities(claim, evidence, 1, 3).tolist(), [2])
        for bad in ([[np.nan, 0, 1]], [[-.1, .1, 1]], [[1, 1, 1]]):
            with self.assertRaises(ValueError):
                fuse_probabilities(bad, evidence, .5, 0)

    def test_assessment_excludes_entire_previous_tuning_population(self):
        rows = [{'id': i, 'claim': str(i)} for i in range(2000)]
        rows += [{'id': 2000, 'claim': '  1999  '}, {'id': 2001, 'claim': 'new'}]
        cal, assess = split_validation(rows, 2, 1)
        self.assertEqual([r['id'] for r in cal], [0, 1])
        self.assertEqual([r['id'] for r in assess], [2001])

    def test_training_never_relabels_missing_evidence(self):
        row = {'label_id': 0}
        retrieved = [{'page_id': 'A', 'sentence_id': 0, 'text': 'Example'}]
        with patch('prepare_retrieved_training.gold_evidence_groups', return_value=[{('B', 1)}]):
            self.assertEqual(training_eligibility(row, retrieved),
                             (False, 'verifiable_claim_missing_complete_group'))
        self.assertEqual(row['label_id'], 0)
        self.assertFalse(training_eligibility({'label_id': 2}, [])[0])
        self.assertTrue(training_eligibility({'label_id': 2}, retrieved)[0])


if __name__ == '__main__':
    unittest.main()
