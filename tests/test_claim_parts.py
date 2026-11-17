import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from replay_claim_parts import derive_label,validate,replay_prompt,INSTRUCTION,MARKER


def part(verdict):return dict(part='An essential assertion.',verdict=verdict,evidence_ids=['X1'] if verdict!='NOT ENOUGH INFO' else [],justification='An evidence comparison.')
def raw(parts,label):return {'response':{'done':True,'done_reason':'stop','message':{'content':json.dumps({'parts':parts,'label':label})}}}

class PartsControls(unittest.TestCase):
    def test_general_aggregation(self):
        for verdicts,label in [(['SUPPORTS'],'SUPPORTS'),(['SUPPORTS','NOT ENOUGH INFO'],'NOT ENOUGH INFO'),(['REFUTES','NOT ENOUGH INFO'],'REFUTES')]:
            self.assertEqual(derive_label([part(v) for v in verdicts]),label)
        with self.assertRaises(ValueError):derive_label([])
    def test_model_disagreement_is_visible_not_overwritten(self):
        result=validate(raw([part('REFUTES')],'SUPPORTS'),{'X1':{}})
        self.assertEqual(result['model_label'],'SUPPORTS');self.assertEqual(result['derived_label'],'REFUTES');self.assertFalse(result['consistent'])
    def test_empty_parts_bad_citations_and_truncation_are_invalid(self):
        for r in [raw([],'SUPPORTS'),raw([part('SUPPORTS')],'SUPPORTS')]:
            with self.assertRaises(ValueError):validate(r,{})
        r=raw([part('SUPPORTS')],'SUPPORTS');r['response']['done_reason']='length'
        with self.assertRaises(ValueError):validate(r,{'X1':{}})
    def test_saved_context_suffix_is_exactly_preserved(self):
        suffix='{"claim": "A claim", "passages": [ ]}\n'
        result=replay_prompt('OLD INSTRUCTION'+MARKER+suffix)
        self.assertEqual(result,INSTRUCTION+MARKER+suffix)
        self.assertTrue(INSTRUCTION.endswith('Return an overall label.'))
        self.assertNotIn('Use no outside knowledge.',INSTRUCTION)

if __name__=='__main__':unittest.main()
