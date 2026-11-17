import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from refine_exhaustive_evidence import sentences,pack

class PreciseEvidenceTests(unittest.TestCase):
    def test_sentence_text_is_exact_source_span(self):
        chunk=dict(id='C4',title='t',section_path=[],text='First sentence. Second sentence!\nThird line without punctuation')
        spans=sentences(chunk)
        self.assertEqual(len(spans),3)
        for span in spans:
            self.assertEqual(chunk['text'][span['local_char_start']:span['local_char_end']],span['text'])
            self.assertEqual(span['parent_id'],'C4')
        self.assertEqual(len(set(s['id'] for s in spans)),3)
    def test_every_selected_chunk_survives_refinement_packing(self):
        chunks=[dict(id='C'+str(i),title='Topic',section_path=[],text='An exact sentence. '*20) for i in range(30)]
        batches=pack('claim',chunks,budget=4000)
        self.assertGreater(len(batches),1)
        self.assertEqual([c['chunk_id'] for b in batches for c in b],[c['id'] for c in chunks])

if __name__=='__main__':unittest.main()
