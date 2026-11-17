"""Offline correctness, isolation and resume tests for the new paired runner."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_parts_test as runner
from parts_rag_classifier import classify, check_budget, validate_parts_response
from test_ephemeral_rag import Embedder, Fetcher, selection


def response(prompt):
    payload = json.loads(prompt.split('\n\nINPUT JSON:\n')[1])
    ident = payload['passages'][0]['evidence'][0]['id']
    return dict(response=dict(done=True, done_reason='stop', prompt_eval_count=110, eval_count=35,
                message=dict(content=json.dumps(dict(parts=[dict(part=payload['claim'], verdict='SUPPORTS',
                    evidence_ids=[ident], justification='The excerpt states the birth date.')], label='NOT ENOUGH INFO')))))


class PartsAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = Embedder()

    def classify(self, temp, generate=response, count=lambda text: {'raw_prompt_tokens': 100}):
        return classify('Example Person was born.', self.embedder, generate, count, 10,
                        temp, fetcher_factory=Fetcher, select=selection)

    def test_uses_parts_prompt_and_code_label_without_losing_model_label(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = self.classify(temp)
            self.assertEqual(trace['label'], 'SUPPORTS')
            self.assertEqual(trace['model_label'], 'NOT ENOUGH INFO')
            self.assertFalse(trace['consistent'])
            self.assertIn('Identify the essential parts', trace['prompt'])
            self.assertTrue(trace['cleanup']['articles_and_database_deleted'])
            self.assertEqual(list(Path(temp).iterdir()), [])
            self.assertEqual(trace['cited_evidence'][0]['text'], 'Example Person was born in 1989.')

    def test_over_budget_input_is_not_silently_trimmed_or_sent(self):
        with tempfile.TemporaryDirectory() as temp:
            def forbidden(prompt):
                self.fail('Over-budget request was sent')
            trace = self.classify(temp, forbidden, lambda text: {'raw_prompt_tokens': 4000})
            self.assertEqual(trace['label'], 'INVALID')
            self.assertEqual(trace['stage'], 'token_preflight')
            self.assertTrue(trace['cleanup']['articles_and_database_deleted'])

    def test_token_count_mismatch_and_missing_counts_are_invalid(self):
        for count in [None, 109]:
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temp:
                def generate(prompt):
                    raw = response(prompt)
                    raw['response']['prompt_eval_count'] = count
                    return raw
                trace = self.classify(temp, generate)
                self.assertEqual(trace['label'], 'INVALID')
                self.assertIn('raw', trace)
                self.assertTrue(trace['cleanup']['articles_and_database_deleted'])

    def test_interrupt_still_cleans_workspace(self):
        with tempfile.TemporaryDirectory() as temp:
            def stop(prompt):
                raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                self.classify(temp, stop)
            self.assertEqual(list(Path(temp).iterdir()), [])


class RunnerControls(unittest.TestCase):
    def test_exclusive_atomic_save_preserves_existing_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'trace.json'
            runner.write_new(p, {'value': 1})
            with self.assertRaises(FileExistsError):
                runner.write_new(p, {'value': 2})
            self.assertEqual(runner.load(p), {'value': 1})
            self.assertEqual(list(Path(temp).iterdir()), [p])

    def test_sampling_is_seeded_balanced_and_preserves_no_labels(self):
        gold = {i: {'id': i, 'claim': 'Claim ' + str(i), 'label': runner.LABELS[i % 3]} for i in range(30)}
        eligible = [{'id': i, 'claim': r['claim']} for i, r in gold.items()]
        a = runner.select_sample(eligible, gold, 6)
        self.assertEqual(a, runner.select_sample(eligible, gold, 6))
        self.assertEqual(sorted([gold[r['id']]['label'] for r in a]), sorted(runner.LABELS * 2))
        self.assertTrue(all(set(r) == {'id', 'claim'} for r in a))

    def test_invalid_is_kept_in_denominator_and_false_negatives(self):
        m = runner.metrics(runner.LABELS, ['INVALID', 'REFUTES', 'NOT ENOUGH INFO'])
        self.assertEqual(m['accuracy'], 2/3)
        self.assertEqual(m['invalid'], 1)
        self.assertEqual(m['per_class']['SUPPORTS']['FN'], 1)
        self.assertEqual(m['confusion_matrix'][0], [0, 0, 0, 1])

    def test_resume_skips_completed_conditions_and_never_reads_gold(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            inputs = [{'id': i, 'claim': 'Claim ' + str(i)} for i in range(3)]
            (out / 'inputs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in inputs))
            (out / 'reference_labels.json').write_text('DO NOT READ DURING INFERENCE')
            (out / 'temporary').mkdir()
            baseline_calls, rag_calls = [], []
            class Baseline:
                def __call__(self, claim):
                    baseline_calls.append(claim)
                    return dict(label='SUPPORTS', wall_seconds=.1)
            class Backend:
                wrapper = 10
                calibration = []
                count = None
                generate = None
                def __init__(self, protocol):
                    pass
                def close(self):
                    pass
            interrupted = [False]
            def rag(claim, *args, **kwargs):
                if claim == 'Claim 1' and not interrupted[0]:
                    interrupted[0] = True
                    raise KeyboardInterrupt()
                rag_calls.append(claim)
                return dict(label='SUPPORTS', model_label='SUPPORTS', status='complete',
                            wall_seconds=.1, cleanup={'articles_and_database_deleted': True})
            def baseline_phase(folder, pending, all_inputs):
                baseline = Baseline()
                for row in pending:
                    runner.write_new(folder / 'traces/evidence_roberta' / (str(row['id'])+'.json'),
                                     dict(id=row['id'], **baseline(row['claim'])))
            with patch.object(runner, 'verify_frozen', return_value={'ids': [0, 1, 2]}), \
                 patch.object(runner, 'EvidenceBaseline', Baseline), \
                 patch.object(runner, 'baseline_phase', side_effect=baseline_phase), \
                 patch.object(runner, 'PartsBackend', Backend), \
                 patch('ephemeral_rag.MiniLMEmbedder', return_value=object()), \
                 patch('parts_rag_classifier.classify', side_effect=rag):
                with self.assertRaises(KeyboardInterrupt):
                    runner.execute(out)
                self.assertEqual(runner.load(out / 'progress.json')['status'], 'stopped')
                runner.execute(out)
            self.assertEqual(baseline_calls, ['Claim 0', 'Claim 1', 'Claim 2'])
            self.assertEqual(rag_calls, ['Claim 0', 'Claim 1', 'Claim 2'])
            self.assertEqual(runner.load(out / 'progress.json')['status'], 'complete_unscored')


if __name__ == '__main__':
    unittest.main()
