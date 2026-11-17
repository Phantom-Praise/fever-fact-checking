import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment_roberta_rag_replay import rag_excerpts,evidence_text


class SavedContextControls(unittest.TestCase):
    def setUp(self):
        self.trace=dict(claim='A claim.',label='INVALID',
                        input=dict(claim='A claim.',passages=[dict(evidence=[dict(id='E2',text='Second.'),dict(id='E1',text='First.')])]),
                        supplied_evidence={'E1':dict(text='First.',title='Title one'),
                                           'E2':dict(text='Second.',title='Title two')},
                        cited_evidence=[dict(id='E1')],parts=[dict(justification='Do not use this reason.')])

    def test_all_supplied_passages_keep_the_original_order_even_after_invalid_llama(self):
        selected=rag_excerpts(self.trace)
        self.assertEqual([r['id'] for r in selected],['E2','E1'])
        self.assertEqual(evidence_text(selected),'Second. First.')
        self.assertNotIn('Title',evidence_text(selected))
        self.assertNotIn('reason',evidence_text(selected))

    def test_text_mismatch_is_rejected(self):
        self.trace['supplied_evidence']['E2']['text']='Changed.'
        with self.assertRaises(ValueError):rag_excerpts(self.trace)

    def test_claim_mismatch_is_rejected(self):
        self.trace['input']['claim']='Other claim.'
        with self.assertRaises(ValueError):rag_excerpts(self.trace)


if __name__=='__main__':unittest.main()
