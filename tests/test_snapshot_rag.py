"""Offline snapshot RAG checks: full text, provenance, cleanup and no HTTP."""
import json,re,sqlite3,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'colab'),str(ROOT/'src')]
import numpy as np
import fever_colab_runtime as core
from snapshot_rag import SnapshotCorpus

class Tokenizer:
 def encode(self,text,add_special_tokens=True):return list(range(len(text.split())+(2 if add_special_tokens else 0)))
 def __call__(self,text,**kwargs):return {'offset_mapping':[(m.start(),m.end()) for m in re.finditer(r'\S+',text)]}
class Encoder:
 tokenizer=Tokenizer()
 def encode(self,texts,progress=None):
  array=np.array([[len(s),len(s.split()),1,2] for s in texts],dtype='float32')
  return array/np.linalg.norm(array,axis=1,keepdims=True)

class SnapshotChecks(unittest.TestCase):
 def fixture(self,path,text=None,lines=''):
  project=path/'project';(project/'data').mkdir(parents=True);(project/'src').mkdir()
  for name in ['fever_colab_runtime.py','ephemeral_rag.py']:(project/'src'/name).write_text('frozen test source')
  row={'id':42,'claim':'The final fact is present.','ranked_articles':[{'rank':1,'page_id':'Test_page','url':'https://en.wikipedia.org/wiki/Test_page'}]}
  (project/'data/eval_inputs.jsonl').write_text(json.dumps(row)+'\n')
  folder=path/'snapshot';folder.mkdir();conn=sqlite3.connect(folder/'snapshot.sqlite')
  conn.execute('CREATE TABLE articles(page_id TEXT PRIMARY KEY,text TEXT,lines TEXT,source_file TEXT,source_line INTEGER,record_sha256 TEXT)')
  if text is None:text=('Earlier article information. '*250)+'\nThe final fact is present.'
  conn.execute('INSERT INTO articles VALUES(?,?,?,?,?,?)',('Test_page',text,lines,'wiki-001.jsonl',7,'raw-record-digest'));conn.commit();conn.close()
  manifest={'schema_version':1,'input_sha256':core.sha_file(project/'data/eval_inputs.jsonl'),
   'missing_pages':[],'packaged_pages':1,'selected_page_ids':1,'snapshot_sha256':core.sha_file(folder/'snapshot.sqlite')}
  (folder/'snapshot_manifest.json').write_text(json.dumps(manifest))
  return project,folder,row,text
 def test_full_text_is_chunked_without_http_and_citations_match_source(self):
  with tempfile.TemporaryDirectory() as d:
   root,folder,row,text=self.fixture(Path(d));corpus=SnapshotCorpus(folder,core.sha_file(root/'data/eval_inputs.jsonl'))
   try:
    with patch('requests.sessions.Session.request',side_effect=AssertionError('HTTP forbidden')):
     trace=corpus.context_for(row,root,Path(d)/'out',Encoder(),{'repo':'test'})
    self.assertEqual(trace['status'],'READY',trace.get('error'));self.assertEqual(trace['wikipedia_http_requests'],0)
    self.assertGreater(trace['chunks'],1);self.assertTrue(trace['articles_and_vectors_deleted'])
    self.assertEqual(trace['article_sources'][0]['body_characters'],len(text))
    self.assertTrue(any('final fact' in e['text'] for e in trace['supplied_evidence'].values()))
    for e in trace['supplied_evidence'].values():
     self.assertEqual(e['text'],text[e['article_char_start']:e['article_char_end']])
     self.assertIsNone(e['revision_id']);self.assertIsNone(e['revision_url'])
     self.assertEqual(e['record_sha256'],'raw-record-digest')
    with patch.object(corpus,'read_article',side_effect=AssertionError('Should reuse saved passages')):
     self.assertEqual(trace,corpus.context_for(row,root,Path(d)/'out',Encoder(),{'repo':'test'}))
   finally:corpus.close()
 def test_empty_aggregate_uses_all_sentence_text_and_not_link_fields(self):
  with tempfile.TemporaryDirectory() as d:
   root,folder,row,_=self.fixture(Path(d),text='',lines='0\tFirst fact.\tlink text\tTarget_page\n1\tSecond fact.\n2\tThird fact.')
   corpus=SnapshotCorpus(folder,core.sha_file(root/'data/eval_inputs.jsonl'))
   try:
    article=corpus.read_article('Test_page');body='\n'.join(b['text'] for b in article['blocks'])
    self.assertEqual(body,'First fact.\nSecond fact.\nThird fact.')
    self.assertEqual(article['text_source'],'all_sentence_text_from_lines')
   finally:corpus.close()
 def test_missing_page_has_no_network_or_gold_fallback(self):
  with tempfile.TemporaryDirectory() as d:
   root,folder,row,_=self.fixture(Path(d));corpus=SnapshotCorpus(folder,core.sha_file(root/'data/eval_inputs.jsonl'))
   row['ranked_articles'][0]['page_id']='Missing'
   try:
    with patch('requests.sessions.Session.request',side_effect=AssertionError('HTTP forbidden')):
     trace=corpus.context_for(row,root,Path(d)/'out',Encoder(),{'repo':'test'})
    self.assertEqual(trace['status'],'ERROR');self.assertIn('absent',trace['error']);self.assertTrue(trace['articles_and_vectors_deleted'])
   finally:corpus.close()
 def test_changed_database_or_ranking_file_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root,folder,_,_=self.fixture(Path(d))
   with self.assertRaisesRegex(ValueError,'input/ranking'):SnapshotCorpus(folder,'different')
   conn=sqlite3.connect(folder/'snapshot.sqlite');conn.execute("UPDATE articles SET text='tampered'");conn.commit();conn.close()
   with self.assertRaisesRegex(ValueError,'hash mismatch'):SnapshotCorpus(folder,core.sha_file(root/'data/eval_inputs.jsonl'))

if __name__=='__main__':unittest.main()
