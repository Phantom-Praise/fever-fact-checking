"""Per-claim isolation, source-citation fidelity and cleanup without real LLM calls."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ephemeral_rag import MODEL_DIR,ClaimVectorDatabase,article_chunks,classify_ephemeral,pack_context


def article():
    body='Example Person was born in 1989. The person is an actor.'
    return dict(title='Example Person',page_id=1,revision_id=10,revision_timestamp='2026-10-29T00:00:00Z',retrieved_utc='2026-10-29T01:00:00Z',
        url='https://en.wikipedia.org/wiki/Example_Person',revision_url='https://en.wikipedia.org/w/index.php?oldid=10',
        text=body,artifact_sha256={'article.txt':hashlib.sha256(body.encode()).hexdigest()},
        blocks=[dict(id='B1',kind='paragraph',text=body,section_path=['Early life'])])


class Embedder:
    def __init__(self):
        self.manifest={'repo':'offline-test','revision':'test'}
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True)
    def encode(self,texts,temp_folder=None):
        result=np.zeros((len(texts),384),dtype='float32');result[:,0]=1;return result


class Fetcher:
    def __init__(self,cache_dir):self.folder=Path(cache_dir);self.folder.mkdir()
    def fetch(self,url):
        (self.folder/'full_article.txt').write_text(article()['text']);return {'article':article()}
    def close(self):pass


def selection(**kwargs):return {'selected_articles':[{'rank':1,'url':'https://en.wikipedia.org/wiki/Example_Person'}]}

def answer(prompt,ids=None):
    payload=json.loads(prompt.split('\n\nINPUT JSON:\n')[1]);ident=payload['passages'][0]['evidence'][0]['id']
    parsed=dict(label='SUPPORTS',reason='A birth date is supplied.',evidence_ids=ids if ids is not None else [ident])
    return dict(request={'messages':[{'role':'user','content':prompt}]},response=dict(done=True,done_reason='stop',message={'content':json.dumps(parsed)},prompt_eval_count=300,eval_count=40))


class EphemeralControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.embedder=Embedder()

    def test_real_faiss_search_returns_corresponding_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            db=ClaimVectorDatabase(Path(temp)/'db',[{'text':'first'},{'text':'second'}],np.eye(2,dtype='float32'))
            self.assertEqual(db.search(np.array([[0,1]],dtype='float32'),1)[0]['text'],'second');db.close()

    def test_success_keeps_exact_citations_but_removes_articles_and_index(self):
        with tempfile.TemporaryDirectory() as temp:
            def generate(prompt):
                self.assertEqual(len(list(Path(temp).glob('claim-*/database/vectors.faiss'))),1)
                self.assertEqual(len(list(Path(temp).glob('claim-*/articles/full_article.txt'))),1)
                return answer(prompt)
            result=classify_ephemeral('Example Person was born.',self.embedder,generate,temp,Fetcher,selection)
            self.assertEqual(result['label'],'SUPPORTS');self.assertTrue(result['cleanup']['articles_and_database_deleted'])
            self.assertEqual(list(Path(temp).iterdir()),[])
            self.assertEqual(result['cited_evidence'][0]['text'],'Example Person was born in 1989.')
            self.assertNotIn('blocks',result['articles'][0]);self.assertNotIn('text',result['articles'][0])

    def test_generation_failure_also_deletes_workspace(self):
        with tempfile.TemporaryDirectory() as temp:
            def fail(prompt):raise TimeoutError('No Ollama answer')
            result=classify_ephemeral('A claim',self.embedder,fail,temp,Fetcher,selection)
            self.assertEqual(result['label'],'INVALID');self.assertIn('TimeoutError',result['error'])
            self.assertTrue(result['cleanup']['articles_and_database_deleted']);self.assertEqual(list(Path(temp).iterdir()),[])
            self.assertIn('prompt',result)

    def test_invented_citation_is_invalid_and_raw_output_is_kept(self):
        with tempfile.TemporaryDirectory() as temp:
            result=classify_ephemeral('A claim',self.embedder,lambda p:answer(p,['E999']),temp,Fetcher,selection)
            self.assertEqual(result['label'],'INVALID');self.assertIn('raw',result)
            self.assertTrue(result['cleanup']['articles_and_database_deleted'])

    def test_separate_claims_never_reuse_a_vector_workspace(self):
        with tempfile.TemporaryDirectory() as temp:
            first=classify_ephemeral('First claim',self.embedder,answer,temp,Fetcher,selection)
            second=classify_ephemeral('Second claim',self.embedder,answer,temp,Fetcher,selection)
            self.assertNotEqual(first['temporary_workspace'],second['temporary_workspace'])
            self.assertEqual(list(Path(temp).iterdir()),[])

    def test_long_article_chunks_are_bounded_and_source_offsets_are_exact(self):
        a=article();a['blocks'][0]['text']='An actor performed in several films. '*300
        chunks=article_chunks(a,self.embedder.tokenizer)
        self.assertGreater(len(chunks),1)
        for c in chunks:
            self.assertLessEqual(len(self.embedder.tokenizer.encode(c['embedding_text'])),256)
            self.assertEqual(a['blocks'][0]['text'][c['char_start']:c['char_end']],c['text'])
        payload,ids=pack_context('A claim',chunks)
        for evidence in ids.values():self.assertEqual(a['blocks'][0]['text'][evidence['char_start']:evidence['char_end']],evidence['text'])
        self.assertTrue(payload['passages'])

if __name__=='__main__':unittest.main()
