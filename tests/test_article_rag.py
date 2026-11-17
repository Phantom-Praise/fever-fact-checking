"""Automatic multiclass prototype controls without real Llama requests."""
import json
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from verify_article_rag import classify,make_prompt,pack,parse_response,retrieve_passages


def passage(page='Jack_Falahee',key='p1',sentence=0):
    return dict(key=key,page_id=page,sentences=[dict(sentence_id=sentence,text='Jack Falahee was born in 1989.')])


def response(label='SUPPORTS',citations=None):
    return dict(response=dict(done=True,done_reason='stop',message=dict(content=json.dumps(dict(label=label,reason='Evidence comparison.',evidence_ids=citations if citations is not None else ['E1'])))))


class ArticleRagControls(unittest.TestCase):
    def test_initial_search_reads_article_body_and_excludes_similarly_named_pages(self):
        conn=sqlite3.connect(':memory:');conn.execute('CREATE TABLE page_sentences(page_id TEXT,sentence_id INT,text TEXT)')
        conn.executemany('INSERT INTO page_sentences VALUES(?,?,?)',[
            ('Jack_Falahee',0,'An American actor.'),('Jack_Falahee',1,'A career detail.'),('Jack_Falahee',2,'Another detail.'),
            ('Jack_Falahee',3,'He was born on February 20, 1989.'),('Jack-Jack_Attack',0,'A Pixar short film.')])
        with patch('verify_article_rag.exact_subject_page',return_value='Jack_Falahee'),patch('verify_article_rag.find_titles',return_value=['Jack-Jack_Attack']):
            result=retrieve_passages(conn,'Jack Falahee was born.')
        conn.close();self.assertEqual(result['sentences_searched'],4)
        self.assertEqual(result['articles_searched'],['Jack_Falahee'])
        self.assertTrue(any('February' in s['text'] for p in result['passages'] for s in p['sentences']))

    def test_sentence_citations_map_back_to_original_article_ids(self):
        payload,ids=pack('A claim',[passage(),passage(key='overlap')])
        self.assertEqual(ids,{'E1':['Jack_Falahee',0]})
        self.assertNotIn('gold',make_prompt(payload))
        self.assertEqual(sum(len(p['sentences']) for p in payload['passages']),1)

    def test_invented_missing_or_truncated_citations_cannot_be_accepted(self):
        for value in [response(citations=['E99']),response(citations=[]),response(citations=['E1','E1'])]:
            with self.assertRaises(ValueError):parse_response(value['response'],{'E1':['Page',0]})
        value=response();value['response']['done_reason']='length'
        with self.assertRaises(ValueError):parse_response(value['response'],{'E1':['Page',0]})

    def test_nei_automatically_expands_once_and_stops_without_user_followup(self):
        initial=dict(passages=[passage()],anchor='Jack_Falahee')
        extra=dict(passages=[passage(page='Biography',key='p2')],anchor='Jack_Falahee')
        generate_calls=[]
        def generate(prompt):
            generate_calls.append(prompt)
            return response('NOT ENOUGH INFO',[]) if len(generate_calls)==1 else response('REFUTES',['E2'])
        with patch('verify_article_rag.retrieve_passages',side_effect=[initial,extra]) as retrieve:
            result=classify(None,'A claim',generate)
        self.assertEqual(len(generate_calls),2);self.assertTrue(retrieve.call_args.kwargs['expand'])
        self.assertEqual(result['label'],'REFUTES');self.assertEqual(result['predicted_evidence'],[['Biography',0]])
        self.assertEqual(result['rounds'][0]['parsed']['label'],'NOT ENOUGH INFO')

    def test_invalid_output_is_not_silently_treated_as_nei(self):
        with patch('verify_article_rag.retrieve_passages',return_value=dict(passages=[passage()])) as retrieve:
            result=classify(None,'A claim',lambda prompt:response(citations=['E99']))
        self.assertEqual(result['label'],'INVALID');self.assertEqual(retrieve.call_count,1)

    def test_failed_request_preserves_attempted_input(self):
        with patch('verify_article_rag.retrieve_passages',return_value=dict(passages=[passage()])):
            def failed(prompt):raise TimeoutError('No response')
            result=classify(None,'A claim',failed)
        self.assertEqual(result['label'],'INVALID')
        self.assertIn('INPUT JSON:',result['rounds'][0]['prompt'])
        self.assertIn('TimeoutError',result['error'])

if __name__=='__main__':unittest.main()
