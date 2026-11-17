"""Live tail correctness: pending labels, partial writes, raw trace fidelity and filters."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from live_ollama_viewer import LiveStore

class LiveReadControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.folder=Path(self.temp.name)
        self.inputs=[]
        for i,label in [(1,'SUPPORTS'),(2,'REFUTES'),(3,'NOT ENOUGH INFO')]:
            self.inputs.append(dict(id=i,claim='Claim '+str(i),gold=dict(label=label,evidence=[[[1,1,'Page',0]]]),
                evidence=dict(original=[dict(page_id='Page',sentence_id=0,text='Exact saved evidence.')],entity=[]),
                claim_probabilities=[.9,.05,.05],baseline_predictions=dict(evidence_512=0)))
        (self.folder/'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.inputs))
        (self.folder/'protocol.json').write_text(json.dumps(dict(n=3,model={'name':'test'})))
        (self.folder/'progress.json').write_text('{}')
        (self.folder/'source_local_llm_experiments.py').write_text('def call():\n    return "saved source"\n')
        self.store=LiveStore(self.folder)
        self.response=dict(id=1,config='A_original',label='SUPPORTS',reason='A fact.',wall_seconds=1.,raw=dict(request={'messages':[{'role':'user','content':'EXACT PROMPT'}]},response={'message':{'content':'EXACT RAW RESPONSE'}}))

    def tearDown(self):self.temp.cleanup()

    def append(self,value,config='A_original'):
        with (self.folder/(config+'_responses.jsonl')).open('a') as stream:stream.write(json.dumps(value)+'\n')

    def test_pending_is_not_wrong_and_has_no_fabricated_output(self):
        self.assertEqual(self.store.listing(show='pending')['total'],3)
        self.assertEqual(self.store.listing(show='all',outcome='wrong')['total'],0)
        trace=self.store.detail(1);self.assertIsNone(trace['result']);self.assertEqual(trace['evidence'][0]['text'],'Exact saved evidence.')

    def test_partial_append_is_retried_once_complete(self):
        path=self.folder/'A_original_responses.jsonl';body=json.dumps(self.response)
        path.write_text(body[:20]);self.assertEqual(self.store.status()['conditions']['A_original']['completed'],0)
        with path.open('a') as stream:stream.write(body[20:]+'\n')
        self.assertEqual(self.store.status()['conditions']['A_original']['completed'],1)
        self.assertEqual(self.store.status()['conditions']['A_original']['completed'],1)

    def test_raw_trace_and_correctness_filters_match_saved_data(self):
        self.append(self.response)
        second=dict(self.response,id=2,label='INVALID',error='Token-limited');self.append(second)
        status=self.store.status()['conditions']['A_original']
        self.assertEqual((status['correct'],status['wrong'],status['invalid']),(1,1,1))
        self.assertEqual(status['accuracy'],.5);self.assertEqual(status['matrix'][1][3],1)
        self.assertEqual(self.store.listing(outcome='invalid')['items'][0]['id'],2)
        self.assertEqual(self.store.detail(1)['result'],self.response)
        self.assertEqual(self.store.listing(query='claim 1',gold='SUPPORTS')['total'],1)

    def test_hybrid_is_separate_from_raw_llm_and_does_not_fix_strict_evidence(self):
        r=dict(self.response,config='D_both',label='NOT ENOUGH INFO');self.append(r,'D_both')
        trace=self.store.detail(1,'D_both')
        self.assertEqual(trace['result']['label'],'NOT ENOUGH INFO')
        self.assertEqual(trace['hybrid']['final_label'],'SUPPORTS');self.assertTrue(trace['hybrid']['override'])
        self.assertFalse(trace['evidence_quality'])

    def test_invalid_filters_and_prefix_are_rejected(self):
        with self.assertRaises(ValueError):self.store.listing(config='../bad')
        with self.assertRaises(ValueError):self.store.listing(page=0)
        with self.assertRaises(KeyError):self.store.detail('unknown')
        self.append(dict(self.response,id=2))
        with self.assertRaises(ValueError):self.store.status()

if __name__=='__main__':unittest.main()
