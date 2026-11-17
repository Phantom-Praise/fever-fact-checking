import json
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment_exhaustive_reader import pack_batches, validate_map, validate_final, MAP_INSTRUCTION

class ExhaustiveReaderTests(unittest.TestCase):
    def test_every_chunk_exactly_once_and_bounded(self):
        chunks=[dict(id='C'+str(i),title='Example',section_path=[],text='evidence '*80) for i in range(30)]
        batches=pack_batches('A claim',chunks,budget=3000)
        self.assertGreater(len(batches),1)
        self.assertEqual([c['id'] for b in batches for c in b],[c['id'] for c in chunks])
        for b in batches:
            prompt=MAP_INSTRUCTION+'\n\nINPUT JSON:\n'+json.dumps(dict(claim='A claim',chunks=b),ensure_ascii=False)
            self.assertLessEqual(len(prompt.encode()),3000)
    def test_no_silent_oversize_drop(self):
        with self.assertRaises(ValueError):pack_batches('claim',[dict(id='C1',title='t',section_path=[],text='x'*3000)],budget=2000)
    def test_selects_known_ids_and_copies_original_text(self):
        chunks=[dict(id='C1',text='The supplied exact sentence.')]
        def body(ids,reason='stop'):return dict(done=True,done_reason=reason,message=dict(content=json.dumps(dict(evidence_ids=ids))))
        self.assertEqual(validate_map(body(['C1']),chunks,16384)[0]['quote'],chunks[0]['text'])
        for invalid in [['C9'],['C1','C1']]:
            with self.assertRaises(ValueError):validate_map(body(invalid),chunks,16384)
        with self.assertRaises(ValueError):validate_map(body(['C1'],'length'),chunks,16384)
    def test_model_and_derived_labels_separate(self):
        result=dict(parts=[dict(part='A is B',verdict='NOT ENOUGH INFO',evidence_ids=[],justification='Evidence missing')],label='SUPPORTS')
        body=dict(done=True,done_reason='stop',message=dict(content=json.dumps(result)))
        parsed=validate_final(body,set(),8192)
        self.assertEqual(parsed['model_label'],'SUPPORTS');self.assertEqual(parsed['derived_label'],'NOT ENOUGH INFO')

if __name__=='__main__':unittest.main()
