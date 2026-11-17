"""Verify saved request/response inspection without any model invocation."""
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ollama_trace_viewer import RUN, TraceStore


@unittest.skipUnless((RUN/'summary.json').exists(),'Completed local LLM artifacts required')
class SavedTraces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.store=TraceStore()

    def test_catalog_separates_diagnostic_and_sample(self):
        data=self.store.listing()
        self.assertEqual(data['traces_total'],301)
        self.assertEqual(data['sample_total'],300)
        self.assertEqual(data['diagnostic_total'],1)
        self.assertEqual(data['items'][0]['id'],111897)
        self.assertTrue(data['items'][0]['diagnostic'])
        self.assertEqual(self.store.listing(outcome='correct')['total'],156)
        self.assertEqual(self.store.listing(outcome='wrong')['total'],145)

    def test_all_exact_payloads_match_displayed_claim_and_evidence(self):
        for item in self.store.catalog:
            d=self.store.detail(item['id'])
            request=d['request']
            self.assertEqual(request['model'],'llama3.1:8b')
            self.assertEqual(len(request['messages']),1)
            inputs=json.loads(request['messages'][0]['content'].split('INPUT JSON:\n',1)[1])
            self.assertEqual(set(inputs),{'claim','evidence'})
            self.assertEqual(inputs['claim'],d['claim'])
            self.assertEqual(inputs['evidence'],[s['text'] for s in d['sentences']])
            self.assertEqual(json.loads(d['response']['message']['content']),d['parsed_output'])
            self.assertEqual(d['prediction'],d['parsed_output']['label'])

    def test_telemundo_tokens_and_actual_call_source(self):
        d=self.store.detail('111897')
        self.assertEqual(d['prediction'],'REFUTES')
        self.assertEqual(d['gold'],'REFUTES')
        self.assertEqual(len(d['sentences']),5)
        self.assertEqual(d['response']['prompt_eval_count'],525)
        self.assertEqual(d['response']['eval_count'],45)
        self.assertIn("session.post(URL + '/api/chat'",d['sources']['call']['code'])
        self.assertTrue(d['sources']['call']['matches_saved_run'])
        self.assertIn("json.loads(body['message']['content'])",d['sources']['parse']['code'])

    def test_search_pagination_validation_and_unknown_id(self):
        self.assertEqual(self.store.listing(query='111897')['items'][0]['id'],111897)
        pages=[self.store.listing(page=i)['items'] for i in range(1,17)]
        ids=[r['id'] for page in pages for r in page]
        self.assertEqual(len(ids),301)
        self.assertEqual(len(set(ids)),301)
        with self.assertRaises(ValueError):self.store.listing(page=0)
        with self.assertRaises(ValueError):self.store.listing(outcome='maybe')
        with self.assertRaises(KeyError):self.store.detail('../../README.md')


if __name__=='__main__':unittest.main()
