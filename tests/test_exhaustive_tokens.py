import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from finalize_exhaustive_tokens import context_for,final_payload

class ExhaustiveTokenTests(unittest.TestCase):
    def test_reserve_and_real_context_limit(self):
        for count in [0,1000,8192,20000,29174]:
            context=context_for(count,10)
            self.assertGreaterEqual(context,count+10+3072+512)
            self.assertLessEqual(context,32768)
        with self.assertRaises(ValueError):context_for(29175,10)
    def test_all_selected_evidence_is_preserved(self):
        findings=[dict(evidence_id='C'+str(i),quote='Exact supplied text '+str(i),relevance='selected') for i in range(100)]
        trace=dict(claim='The full claim',findings=findings,cited_sources={f['evidence_id']:dict(title='Article',section_path=['Section']) for f in findings})
        payload=final_payload(trace)
        self.assertEqual([e['id'] for e in payload['evidence']],[f['evidence_id'] for f in findings])
        self.assertEqual([e['text'] for e in payload['evidence']],[f['quote'] for f in findings])
        self.assertEqual(set(payload),{'claim','evidence'})

if __name__=='__main__':unittest.main()
