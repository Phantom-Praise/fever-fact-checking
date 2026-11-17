"""Check configuration controls and prefix retrieval without model inference."""
import json
from pathlib import Path
import sqlite3
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from experiment_llm_configurations import CONFIGS, decision, make_prompt, normalize, score
from retrieve_entity_anchored import exact_subject_page, retrieve_entity_anchored
from analyze_llm_configuration_results import paired_macro_f1_interval


class ConfigurationRules(unittest.TestCase):
    def test_paired_f1_bootstrap_preserves_pairing(self):
        self.assertEqual(paired_macro_f1_interval([0,1,2],[0,0,2],[0,0,2],samples=100),[0.,0.])
        lo,hi=paired_macro_f1_interval([0,1,2]*10,[1,2,0]*10,[0,1,2]*10,samples=100)
        self.assertAlmostEqual(lo,1.);self.assertAlmostEqual(hi,1.)
    def test_fallback_is_nei_only_and_fixed_threshold(self):
        row={'claim_probabilities':[.8,.1,.1]}
        self.assertEqual(decision(row,'NOT ENOUGH INFO','E_claim_fallback'),('SUPPORTS',True))
        self.assertEqual(decision(row,'REFUTES','E_claim_fallback'),('REFUTES',False))
        self.assertEqual(decision(row,'INVALID','E_claim_fallback'),('INVALID',False))
        self.assertEqual(decision(row,'NOT ENOUGH INFO','D_both'),('NOT ENOUGH INFO',False))
        self.assertEqual(decision({'claim_probabilities':[.79,.11,.1]},'NOT ENOUGH INFO','E_claim_fallback'),('NOT ENOUGH INFO',False))
        self.assertEqual(decision({'claim_probabilities':[.05,.05,.9]},'NOT ENOUGH INFO','E_claim_fallback'),('NOT ENOUGH INFO',False))

    def test_prompt_pair_changes_only_declared_factors_and_hides_gold(self):
        row={'claim':'Example claim','gold':{'label':'SECRET_GOLD'},'evidence':{'original':[{'text':'ORIGINAL_TEXT'}],'entity':[{'text':'ENTITY_TEXT'}]}}
        a=make_prompt(row,'A_original');b=make_prompt(row,'B_prompt');c=make_prompt(row,'C_retrieval');d=make_prompt(row,'D_both')
        self.assertEqual(a.split('INPUT JSON:',1)[0],c.split('INPUT JSON:',1)[0])
        self.assertEqual(b.split('INPUT JSON:',1)[0],d.split('INPUT JSON:',1)[0])
        self.assertEqual(a.split('INPUT JSON:',1)[1],b.split('INPUT JSON:',1)[1])
        self.assertEqual(c.split('INPUT JSON:',1)[1],d.split('INPUT JSON:',1)[1])
        self.assertEqual(d,make_prompt(row,'E_claim_fallback'))
        for text in [a,b,c,d]:self.assertNotIn('SECRET_GOLD',text)

    def test_invalid_response_stays_in_score_denominator(self):
        rows=[{'id':1,'gold':{'label':'SUPPORTS','evidence':[[[1,1,'A',0]]]},'evidence':{'original':[{'page_id':'A','sentence_id':0}]}},
              {'id':2,'gold':{'label':'NOT ENOUGH INFO','evidence':[[[None,None,None,None]]]},'evidence':{'original':[]}}]
        result=score(rows,['SUPPORTS','INVALID'],'A_original')
        self.assertEqual(result['accuracy'],.5);self.assertEqual(result['invalid'],1)
        self.assertEqual(result['confusion_matrix'][2][3],1)


class EntityPrefix(unittest.TestCase):
    def setUp(self):
        self.conn=sqlite3.connect(':memory:')
        self.conn.executescript('CREATE TABLE wiki_titles(page_id TEXT PRIMARY KEY,title TEXT); CREATE TABLE page_sentences(page_id TEXT,sentence_id INTEGER,text TEXT); CREATE VIRTUAL TABLE title_fts USING fts5(page_id UNINDEXED,title);')
        for page,title,text in [('Juventus_F.C.','Juventus F.C.','Juventus F.C. have worn black and white uniforms since 1903.'),('Black+White','Black White','Black White was a magazine.'),('The_New_York_Times','The New York Times','The New York Times is a newspaper.')]:
            self.conn.execute('INSERT INTO wiki_titles VALUES(?,?)',(page,title));self.conn.execute('INSERT INTO title_fts VALUES(?,?)',(page,title));self.conn.execute('INSERT INTO page_sentences VALUES(?,?,?)',(page,0,text))
    def tearDown(self):self.conn.close()
    def test_exact_leading_entity_is_kept_and_ranked(self):
        claim='Juventus F.C. rejected their traditional black-and-white uniform.'
        self.assertEqual(exact_subject_page(self.conn,claim),'Juventus_F.C.')
        rows=retrieve_entity_anchored(self.conn,claim)
        self.assertEqual(rows[0]['page_id'],'Juventus_F.C.')
        self.assertTrue(rows[0]['entity_anchor'])
        self.assertEqual(exact_subject_page(self.conn,'The New York Times publishes news.'),'The_New_York_Times')
    def test_no_matching_titles_returns_empty(self):
        self.assertIsNone(exact_subject_page(self.conn,'Qzxw asdfgh.'))
        self.assertEqual(retrieve_entity_anchored(self.conn,'Qzxw asdfgh.'),[])


RUN=ROOT/'reports/part2/llm_configurations_20261028'
@unittest.skipUnless((RUN/'protocol.json').exists(),'Prepared configuration experiment required')
class SampleControls(unittest.TestCase):
    def test_balanced_disjoint_splits_and_prior_pilot_exclusion(self):
        rows=[json.loads(line) for line in (RUN/'inputs.jsonl').read_text().splitlines()]
        cal=[r for r in rows if r['split']=='calibration'];assess=[r for r in rows if r['split']=='assessment']
        self.assertEqual(len(cal),60);self.assertEqual(len(assess),120)
        for label in ['SUPPORTS','REFUTES','NOT ENOUGH INFO']:
            self.assertEqual(sum(r['gold']['label']==label for r in cal),20)
            self.assertEqual(sum(r['gold']['label']==label for r in assess),40)
        self.assertFalse({r['id'] for r in cal}&{r['id'] for r in assess})
        self.assertFalse({normalize(r['claim']) for r in cal}&{normalize(r['claim']) for r in assess})
        old=[json.loads(line) for line in (ROOT/'reports/part2/ollama_comparison_20261028/sample.jsonl').read_text().splitlines()]
        self.assertFalse({normalize(r['claim']) for r in assess}&{normalize(r['claim']) for r in old})
        self.assertNotIn(111897,{r['id'] for r in assess})
        self.assertEqual(len({r['id'] for r in rows}),180)


if __name__=='__main__':unittest.main()
