import csv
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from experiment_symbolic_v2 import SnapshotVerifier, load_inputs, values
from analyze_controlled import paired
from controlled_judges import prepare, score


def statement(value, **extra):
    return {'mainsnak': {'datavalue': {'value': value}}, **extra}


class SymbolicRules(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.verifier = SnapshotVerifier(self.tmp.name, offline=True)

    def test_missing_snapshot_fails_instead_of_abstaining(self):
        with self.assertRaises(RuntimeError):
            self.verifier.query(action='missing')

    def test_exact_occupation_support_and_nonmatch_abstains(self):
        self.verifier.subject = Mock(return_value={'claims': {'P106': [statement({'id': 'Q1'})]}})
        self.verifier.named_match = Mock(side_effect=lambda qid, text: text == 'actor')
        self.assertEqual(self.verifier.verify('Example Person is an actor.')[0], 0)
        self.assertIsNone(self.verifier.verify('Example Person is a dancer.')[0])
        self.assertIsNone(self.verifier.verify('Example Person is an actor and dancer.')[0])
        self.assertIsNone(self.verifier.verify('Example Person is not an actor.')[0])

    def test_ambiguous_human_abstains(self):
        human = {'claims': {'P31': [statement({'id': 'Q5'})]}}
        self.verifier.query = Mock(return_value={'search': [{'id': 'Q1', 'label': 'Same Name'}, {'id': 'Q2', 'label': 'Same Name'}]})
        self.verifier.entity = Mock(return_value=human)
        self.assertIsNone(self.verifier.subject('Same Name', human=True))

    def test_human_type_constraint_and_truncation(self):
        self.verifier.query = Mock(return_value={'search': [{'id': 'Q1', 'label': 'Same Name'}, {'id': 'Q2', 'label': 'Same Name'}]})
        human = {'claims': {'P31': [statement({'id': 'Q5'})]}}
        self.verifier.entity = Mock(side_effect=lambda qid: human if qid == 'Q1' else {})
        self.assertEqual(self.verifier.subject('Same Name', human=True), human)
        self.verifier.query.return_value['search-continue'] = 50
        self.assertIsNone(self.verifier.subject('Same Name', human=True))

    def test_deprecated_and_qualified_values_excluded(self):
        entity = {'claims': {'P106': [statement(1, rank='deprecated'), statement(2, qualifiers={'P580': []}), statement(3)]}}
        self.assertEqual(values(entity, 'P106'), [3])

    def test_dates_precision_calendar_and_nonmatches(self):
        date = {'time': '+1996-02-21T00:00:00Z', 'precision': 11,
                'calendarmodel': 'http://www.wikidata.org/entity/Q1985727'}
        self.verifier.subject = Mock(return_value={'claims': {'P569': [statement(date)]}})
        self.assertEqual(self.verifier.verify('Example Person was born in the 1990s.')[0], 0)
        self.assertIsNone(self.verifier.verify('Example Person was born in 1995.')[0])
        date['precision'] = 8
        self.assertIsNone(self.verifier.verify('Example Person was born in 1996.')[0])
        date['calendarmodel'] = 'other'
        self.assertIsNone(self.verifier.verify('Example Person was born in the 1990s.')[0])

    def test_membership_missing_not_refuted(self):
        self.verifier.subject = Mock(return_value={'claims': {}})
        self.assertIsNone(self.verifier.verify('Example Country is a member of Example Organization.')[0])

    def test_input_duplicate_prediction_rejected(self):
        root = Path(self.tmp.name)
        (root / 'claims.jsonl').write_text(json.dumps({'id': 1, 'claim': 'A', 'label_id': 0}) + '\n')
        with (root / 'predictions.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['condition', 'id', 'gold_label_id', 'predicted_label_id'])
            writer.writeheader()
            row = dict(condition='retrieved_evidence', id=1, gold_label_id=0, predicted_label_id=0)
            writer.writerows([row, row])
        with self.assertRaises(ValueError):
            load_inputs(root / 'claims.jsonl', root / 'predictions.csv')

    def test_paired_no_changes_and_sign(self):
        result = paired([0, 1], [0, 0], [0, 1], samples=100)
        self.assertEqual(result['right_minus_left_accuracy'], .5)
        self.assertEqual(result['right_only_correct'], 1)
        same = paired([0, 1], [0, 0], [0, 0], samples=100)
        self.assertEqual(same['paired_bootstrap_accuracy_ci95'], [0, 0])
        self.assertEqual(same['mcnemar_exact_p_unadjusted'], 1)

    def test_judge_factorial_scoring_and_missing_response(self):
        root = Path(self.tmp.name)
        claims, predictions = root / 'claims.jsonl', root / 'predictions.csv'
        claims.write_text(json.dumps({'id': 1, 'claim': 'Example', 'label_id': 0, 'evidence_text': 'Evidence'}) + '\n')
        predictions.write_text('condition,id,gold_label_id,predicted_label_id\nretrieved_evidence,1,0,0\n')
        args = Namespace(retrieved=str(claims), predictions=str(predictions),
                         output=str(root / 'prepared'), repeats=3)
        prepare(args)
        tasks = root / 'prepared' / 'judge_tasks.jsonl'
        inputs = [json.loads(line) for line in tasks.read_text().splitlines()]
        self.assertEqual(len(inputs), 12)
        self.assertTrue(all('label_id' not in r for r in inputs))
        responses = root / 'responses.csv'
        with responses.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['task_id', 'model', 'predicted_label_id'])
            writer.writeheader()
            writer.writerows({'task_id': t['task_id'], 'model': 'test-fixture', 'predicted_label_id': 0} for t in inputs)
        args.tasks, args.responses, args.output = str(tasks), str(responses), str(root / 'score.json')
        score(args)
        report = json.loads(Path(args.output).read_text())
        condition = report['models']['test-fixture']['conditions']['retrieved:direct']
        self.assertEqual(condition['mean_accuracy'], 1)
        self.assertEqual(condition['all_repeats_agree_fraction'], 1)
        responses.write_text('\n'.join(responses.read_text().splitlines()[:-1]) + '\n')
        with self.assertRaises(ValueError):
            score(args)


if __name__ == '__main__':
    unittest.main()
