import json
from pathlib import Path
import sys
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evidence_explorer import Explorer, evidence_quality, make_handler, binary_outcome, confusion_summary

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/part2/official_fever_dev_20261023'


class CoverageGroups(unittest.TestCase):
    def test_one_vs_rest_includes_wrong_other_class_as_tn(self):
        cases = [('SUPPORTS', 'SUPPORTS', 'TP'), ('REFUTES', 'SUPPORTS', 'FP'),
                 ('SUPPORTS', 'NOT ENOUGH INFO', 'FN'), ('REFUTES', 'NOT ENOUGH INFO', 'TN')]
        records = []
        for actual, predicted, expected in cases:
            self.assertEqual(binary_outcome(actual, predicted, 'SUPPORTS'), expected)
            records.append({'label': actual, 'predictions': {'calibrated_ensemble': predicted}, 'quality': 'complete'})
        result = confusion_summary(records, 'calibrated_ensemble')
        self.assertEqual({k: result['per_class']['SUPPORTS'][k] for k in ('TP', 'FP', 'FN', 'TN')},
                         {'TP': 1, 'FP': 1, 'FN': 1, 'TN': 1})
        self.assertEqual(result['accuracy'], .25)
        self.assertEqual(result['per_class']['SUPPORTS']['f1'], .5)

    def test_complete_group_not_just_one_sentence(self):
        record = {'label': 'SUPPORTS', 'evidence': [[[1, 2, 'A', 0], [1, 3, 'B', 1]]]}
        self.assertEqual(evidence_quality(record, [['A', 0]]), 'partial')
        self.assertEqual(evidence_quality(record, [['A', 0], ['B', 1]]), 'complete')
        self.assertEqual(evidence_quality(record, [['C', 0]]), 'miss')

    def test_nei_is_not_a_retrieval_failure(self):
        self.assertEqual(evidence_quality({'label': 'NOT ENOUGH INFO'}, []), 'nei')


@unittest.skipUnless((RUN / 'outputs.jsonl').exists(), 'Local research data required')
class ExplorerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.viewer = Explorer(ROOT / 'data/official_fever/shared_task_dev.jsonl', RUN / 'outputs.jsonl',
                             ROOT / 'data/processed/wiki_fts.db', RUN / 'protocol.json')

    def test_summary_matches_frozen_official_results(self):
        meta = self.viewer.metadata()
        saved = json.loads((RUN / 'summary.json').read_text())
        self.assertEqual(meta['total'], 19998)
        self.assertEqual(meta['verifiable'], 13332)
        self.assertEqual(meta['complete_recall_at5'], saved['conditions']['retrieved_roberta']['evidence_macro_recall'])
        self.assertEqual(meta['accuracy'], saved['conditions']['calibrated_ensemble']['label_accuracy'])
        self.assertTrue(meta['retriever_source_matches_saved_run'])

    def test_all_models_confusion_and_strict_scores_match_saved_benchmark(self):
        saved = json.loads((RUN / 'summary.json').read_text())
        for model, expected in saved['conditions'].items():
            analysis = self.viewer.metadata(model)['analysis']
            self.assertEqual(analysis['matrix'], expected['confusion_matrix'])
            self.assertEqual(analysis['accuracy'], expected['label_accuracy'])
            self.assertEqual(analysis['strict_fever_score'], expected['fever_score'])
            self.assertAlmostEqual(analysis['macro_f1'], expected['classification_macro_f1'])
            self.assertEqual(sum(analysis['diagnostics'].values()), 19998)
            for counts in analysis['per_class'].values():
                self.assertEqual(sum(counts[k] for k in ('TP', 'FP', 'TN', 'FN')), 19998)

    def test_matrix_and_binary_filters_return_exact_cell_populations(self):
        analysis = self.viewer.metadata()['analysis']
        result = self.viewer.claims(label='REFUTES', prediction='SUPPORTS')
        self.assertEqual(result['total'], analysis['matrix'][1][0])
        self.assertTrue(all(r['label'] == 'REFUTES' and r['prediction'] == 'SUPPORTS' for r in result['items']))
        for outcome in ('TP', 'FP', 'TN', 'FN'):
            subset = self.viewer.claims(outcome=outcome, positive='NOT ENOUGH INFO')
            self.assertEqual(subset['total'], analysis['per_class']['NOT ENOUGH INFO'][outcome])
        subset = self.viewer.claims(diagnostic='wrong_complete', model='retrieved_roberta')
        self.assertEqual(subset['total'], self.viewer.metadata('retrieved_roberta')['analysis']['diagnostics']['wrong_complete'])
        self.assertTrue(all(not r['correct'] and r['quality'] == 'complete' for r in subset['items']))

    def test_tokenizer_trace_detects_selected_text_cut_off_before_model(self):
        selected = [{'page_id': 'A', 'sentence_id': 0, 'rank': 1, 'text': 'a long evidence sentence. ' * 400},
                    {'page_id': 'B', 'sentence_id': 0, 'rank': 2, 'text': 'The required fact is here.'}]
        trace = self.viewer.token_visibility('This is the claim.', selected)
        self.assertTrue(trace['available'])
        self.assertTrue(trace['truncated'])
        self.assertLessEqual(trace['tokens_after'], 256)
        self.assertEqual(trace['sentences'][0]['visibility'], 'partial')
        self.assertEqual(trace['sentences'][1]['visibility'], 'none')
        self.assertEqual(trace['sentences'][1]['tokens_kept'], 0)

    def test_trace_replays_scores_without_changing_saved_decisions(self):
        detail = self.viewer.detail('111897')
        self.assertTrue(detail['fusion_reproduced'])
        self.assertEqual(detail['current_quality'], 'complete')
        for label in ('SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'):
            self.assertEqual(detail['outcomes'][label], binary_outcome(detail['label'], detail['prediction'], label))
        self.assertEqual(self.viewer.detail('91198')['bottleneck'], 'nei')

    def test_reconstructed_rankings_match_independent_saved_predictions(self):
        ids = [str(r['id']) for r in self.viewer.listing[:8] + self.viewer.listing[::2500]]
        for claim_id in ids:
            with self.subTest(claim_id=claim_id):
                detail = self.viewer.detail(claim_id)
                self.assertTrue(detail['saved_top5_equal'])
                rows = detail['sentences']
                self.assertEqual(rows, sorted(rows, key=lambda r: (-r['score'], r['article_rank'], r['sentence_id'])))
                self.assertTrue(all(abs(row['score'] - row['overlap'] - row['title_bonus']) < 1e-12 for row in rows))
                self.assertEqual(sum(row['selected'] for row in rows), min(5, len(rows)))

    def test_missed_reference_is_available_for_inspection(self):
        detail = self.viewer.detail('137334')
        self.assertEqual(detail['quality'], 'miss')
        self.assertEqual(detail['gold_groups'][0][0]['page_id'], 'Soul_Food_-LRB-film-RRB-')
        self.assertFalse(detail['gold_groups'][0][0]['candidate'])
        article = self.viewer.article('Soul_Food_-LRB-film-RRB-')
        self.assertIn('Fox 2000 Pictures', article['sentences'][0]['text'])

    def test_search_filter_and_last_page_cover_real_claims(self):
        result = self.viewer.claims(query='TILDA SWINTON', quality='nei')
        self.assertTrue(result['items'])
        self.assertTrue(all('tilda swinton' in r['claim'].casefold() and r['quality'] == 'nei' for r in result['items']))
        end = self.viewer.claims(page=500)
        self.assertEqual(len(end['items']), 38)
        self.assertEqual(end['items'][-1]['id'], self.viewer.listing[-1]['id'])
        with self.assertRaises(ValueError): self.viewer.claims(quality='unknown')

    def test_http_read_only_routes_and_host_restriction(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.viewer, 0))
        server.RequestHandlerClass = make_handler(self.viewer, server.server_port)
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        base = 'http://127.0.0.1:' + str(server.server_port)
        try:
            with urlopen(base + '/api/claims?q=194462') as response:
                self.assertEqual(json.load(response)['items'][0]['id'], 194462)
            with urlopen(base + '/token-budget') as response:
                self.assertIn('Does a longer input help?', response.read().decode())
            with urlopen(base + '/llm-comparison') as response:
                self.assertIn('Local LLM versus RoBERTa', response.read().decode())
            with urlopen(base + '/api/llm-comparison') as response:
                local_llm = json.load(response)
                self.assertIn(local_llm['status'], ('not_started', 'prepared', 'inference_running', 'complete'))
                self.assertEqual(local_llm['progress']['total'], 300)
            with urlopen(base + '/ollama-traces') as response:
                self.assertIn('Inside the Ollama call', response.read().decode())
            with urlopen(base + '/api/ollama-traces?outcome=wrong') as response:
                self.assertEqual(json.load(response)['total'], 145)
            with urlopen(base + '/api/ollama-trace/111897') as response:
                trace = json.load(response)
                self.assertEqual(trace['prediction'], 'REFUTES')
                self.assertEqual(trace['response']['prompt_eval_count'], 525)
            with urlopen(base + '/live-llama') as response:
                self.assertIn('Live Llama Lab',response.read().decode())
            with urlopen(base + '/ephemeral-rag') as response:
                page=response.read().decode()
                self.assertIn('RAG: exact prompt',page)
                self.assertIn('Original: exact prompt and response',page)
            with urlopen(base + '/claim-parts') as response:
                page=response.read().decode()
                self.assertIn('Essential parts. Explicit verdicts.',page)
                self.assertIn('Updated: full exact prompt',page)
            with urlopen(base + '/exhaustive-reader') as response:
                self.assertIn('Every chunk, before the verdict.',response.read().decode())
            with urlopen(base + '/api/exhaustive-reader') as response:
                exhaustive=json.load(response)
                self.assertEqual(len(exhaustive['cases']),10)
                self.assertEqual(exhaustive['progress']['total'],10)
            if (ROOT/'reports/part2/exhaustive_token_verified_20261029/traces/130325/final_attempt_1.json').exists():
                with urlopen(base + '/api/exhaustive-verified/130325/final_attempt_1.json') as response:
                    final_trace=json.load(response)
                    self.assertEqual(final_trace['request']['model'],'llama3.1:8b')
                    self.assertIn('prompt_eval_count',final_trace['response'])
            with self.assertRaises(HTTPError) as forbidden_final:
                urlopen(base + '/api/exhaustive-verified/130325/protocol.json')
            self.assertEqual(forbidden_final.exception.code,404)
            with self.assertRaises(HTTPError) as invalid_trace:
                urlopen(base + '/api/exhaustive-reader/130325/protocol.json')
            self.assertEqual(invalid_trace.exception.code,404)
            with urlopen(base + '/api/ephemeral-rag') as response:
                pilot=json.load(response)
                self.assertEqual(pilot['progress']['total'],10)
                if 'summary' in pilot:self.assertTrue(pilot['summary']['all_workspaces_deleted'])
            with urlopen(base + '/api/live-llama/status') as response:
                live=json.load(response)
                self.assertEqual(live['conditions']['A_original']['total'],19998)
                self.assertGreater(live['conditions']['A_original']['completed'],0)
            with urlopen(base + '/api/live-llama/claims?outcome=invalid') as response:
                invalid=json.load(response)
                self.assertTrue(all(r['prediction']=='INVALID' for r in invalid['items']))
            with urlopen(base + '/api/live-llama/claim/91198') as response:
                live_trace=json.load(response)
                self.assertEqual(live_trace['result']['config'],'A_original')
                self.assertIn('request',live_trace['result']['raw'])
            with self.assertRaises(HTTPError) as bad:
                urlopen(base + '/api/live-llama/claims?config=unknown')
            self.assertEqual(bad.exception.code,400)
            with urlopen(base + '/api/llm-configurations') as response:
                configurations = json.load(response)
                self.assertIn(configurations['progress']['status'], ('prepared', 'running', 'calibration_complete', 'complete'))
                if 'assessment_summary' in configurations:
                    self.assertEqual(configurations['assessment_summary']['n'], 120)
            with urlopen(base + '/llm-configurations') as response:
                self.assertIn('FEVER', response.read().decode())
            with urlopen(base + '/result-analysis') as response:
                analysis = response.read().decode()
                self.assertIn('What helped.', analysis)
                self.assertIn('not preregistered', analysis)
                self.assertTrue(response.url.endswith('/result-analysis/'))
            with urlopen(base + '/result-analysis-details') as response:
                self.assertIn('At a glance: what the evidence supports', response.read().decode())
            with urlopen(base + '/result-analysis/data.json') as response:
                visual = json.load(response)
                self.assertEqual(len(visual['waffle']), 120)
                self.assertFalse(visual['provenance']['new_inference'])
            with urlopen(base + '/result-analysis.md') as response:
                self.assertIn('attachment', response.headers['Content-Disposition'])
                self.assertIn('Hypothesis verdicts with evidence', response.read().decode())
            with urlopen(base + '/result-analysis.zip') as response:
                self.assertEqual(response.read(2), b'PK')
            with urlopen(base + '/api/token-budget') as response:
                experiment = json.load(response)
                self.assertIn(experiment['status'], ('not_started', 'inference_running', 'complete'))
                if experiment['status'] == 'complete':
                    self.assertEqual(experiment['summary']['n'], 19998)
                else:
                    self.assertEqual(experiment['progress']['total'], 19998)
            with self.assertRaises(HTTPError) as denied:
                urlopen(Request(base + '/api/meta', headers={'Host': 'untrusted.example'}))
            self.assertEqual(denied.exception.code, 403)
            with self.assertRaises(HTTPError) as missing: urlopen(base + '/../../README.md')
            self.assertEqual(missing.exception.code, 404)
            with self.assertRaises(HTTPError) as invalid: urlopen(base + '/api/claims?page=0')
            self.assertEqual(invalid.exception.code, 400)
        finally:
            server.shutdown(); server.server_close(); worker.join()


if __name__ == '__main__': unittest.main()
