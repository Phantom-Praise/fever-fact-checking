"""Full-content fetching, provenance, selection, immutable caching, and errors."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fetch_wikipedia_articles import API,ArticleFetchError,WikipediaArticleFetcher,article_target,extract_article,selected_urls,main,is_disambiguation

HTML='''<div class="mw-parser-output"><div class="toc">Navigation must disappear</div>
<table class="infobox"><tr><th>Born</th><td>February 20, 1989</td></tr><tr><th>Occupation</th><td>Actor</td></tr></table>
<p>Jack <b>Falahee</b> is an actor.<sup class="reference">[1]</sup></p>
<div class="mw-heading"><h2 id="Early_life">Early life<span class="mw-editsection">[edit]</span></h2></div>
<p>He grew up in Michigan.</p><h2 id="Career">Career</h2><p>Information from the article body, beyond the lead.</p>
<ul><li>First role<ul><li>A nested role</li></ul></li></ul>
<table class="wikitable"><caption>Filmography</caption><tr><th>Year</th><th>Title</th></tr><tr><td>2014</td><td>Example</td></tr></table>
<div class="reflist">Bibliography reference clutter</div><script>bad script()</script></div>'''


def parsed(revision=10):return dict(parse=dict(title='Jack Falahee',pageid=5,revid=revision,text=HTML,sections=[{'line':'Career'}],categories=[],properties=[],redirects=[]))
def metadata(revision=10):return dict(query=dict(pages=[dict(pageid=5,ns=0,title='Jack Falahee',fullurl='https://en.wikipedia.org/wiki/Jack_Falahee',revisions=[dict(revid=revision,timestamp='2026-10-28T12:00:00Z')])]))


class Response:
    def __init__(self,data,status=200):self.data=data;self.status_code=status;self.headers={}
    def json(self):return self.data
    def raise_for_status(self):
        if self.status_code>=400:raise requests.HTTPError(str(self.status_code))


class Session:
    def __init__(self,*responses):self.responses=list(responses);self.headers={};self.calls=[]
    def get(self,url,**kwargs):self.calls.append((url,kwargs));return self.responses.pop(0)


class FetchTests(unittest.TestCase):
    def test_article_body_sections_infobox_and_tables_survive_cleaning(self):
        result=extract_article(HTML)
        self.assertIn('beyond the lead',result['text']);self.assertIn('Born | February 20, 1989',result['text'])
        self.assertIn('2014 | Example',result['text']);self.assertIn('A nested role',result['text'])
        for unwanted in ['Navigation must disappear','[edit]','[1]','bad script','reference clutter']:self.assertNotIn(unwanted,result['text'])
        body=next(b for b in result['blocks'] if 'beyond the lead' in b['text'])
        self.assertEqual(body['section_path'],['Career'])
        self.assertEqual(len({b['id'] for b in result['blocks']}),len(result['blocks']))

    def test_url_normalization_and_revision_pinning(self):
        target=article_target('https://en.m.wikipedia.org/wiki/Jack_Falahee?oldid=42#Career')
        self.assertEqual(target['title'],'Jack Falahee');self.assertEqual(target['revision_id'],42)
        for url in ['https://example.com/wiki/Jack','https://en.wikipedia.org.evil.test/wiki/Jack','https://user:pass@en.wikipedia.org/wiki/Jack','https://en.wikipedia.org/wiki/Special:Random','file:///tmp/wiki','https://en.wikipedia.org/wiki/Jack?oldid=-1']:
            with self.subTest(url=url),self.assertRaises(ValueError):article_target(url)

    def test_full_fetch_keeps_raw_provenance_and_cached_read_has_no_network(self):
        session=Session(Response(parsed()),Response(metadata()))
        with tempfile.TemporaryDirectory() as cache:
            fetcher=WikipediaArticleFetcher(cache,session,min_interval=0)
            result=fetcher.fetch('https://en.wikipedia.org/wiki/Jack_Falahee')
            folder=Path(result['folder']);self.assertEqual((folder/'article.html').read_text(),HTML)
            self.assertIn('beyond the lead',(folder/'article.txt').read_text())
            self.assertEqual(result['article']['revision_id'],10);self.assertIn('oldid=10',result['article']['revision_url'])
            self.assertIn('metadata_response.json',result['article']['artifact_sha256'])
            again=fetcher.fetch('https://en.wikipedia.org/wiki/Jack_Falahee#Career')
            self.assertTrue(again['cache_hit']);self.assertEqual(fetcher.requests_made,2)
            for url,kwargs in session.calls:
                self.assertEqual(url,API);self.assertFalse(kwargs['allow_redirects'])
                self.assertNotIn('section',kwargs['params']);self.assertNotIn('exintro',kwargs['params'])
            (folder/'article.txt').write_text('CORRUPTED')
            with self.assertRaises(ArticleFetchError):fetcher.fetch('https://en.wikipedia.org/wiki/Jack_Falahee')

    def test_refresh_preserves_previous_revision(self):
        session=Session(Response(parsed(10)),Response(metadata(10)),Response(parsed(11)),Response(metadata(11)))
        with tempfile.TemporaryDirectory() as cache:
            fetcher=WikipediaArticleFetcher(cache,session,min_interval=0)
            a=fetcher.fetch('https://en.wikipedia.org/wiki/Jack_Falahee');b=fetcher.fetch('https://en.wikipedia.org/wiki/Jack_Falahee',refresh=True)
            self.assertNotEqual(a['folder'],b['folder']);self.assertTrue((Path(a['folder'])/'article.json').exists())
            self.assertEqual(fetcher.fetch('https://en.wikipedia.org/wiki/Jack_Falahee')['article']['revision_id'],11)

    def test_missing_page_and_wrong_revision_are_explicit_failures(self):
        with tempfile.TemporaryDirectory() as cache:
            fetcher=WikipediaArticleFetcher(cache,Session(Response({'error':{'code':'missingtitle','info':'Missing'}})),min_interval=0)
            with self.assertRaises(ArticleFetchError):fetcher.fetch('https://en.wikipedia.org/wiki/Nonexistent')
            self.assertFalse(list(Path(cache).glob('pages/*/article.json')))
            fetcher=WikipediaArticleFetcher(cache,Session(Response(parsed(10)),Response(metadata(11))),min_interval=0)
            with self.assertRaises(ArticleFetchError):fetcher.fetch('https://en.wikipedia.org/wiki/Jack_Falahee')

    def test_existing_ranking_is_preserved_as_ten_candidates_then_five_pages(self):
        with tempfile.TemporaryDirectory() as folder:
            db=Path(folder)/'test.db';sqlite3.connect(db).close()
            with patch('retrieve_title_aware.find_titles',return_value=['Page_'+str(i) for i in range(10)]) as find:
                result=selected_urls(claim='A claim',db=db)
            self.assertEqual(find.call_args.args[2],10)
            self.assertEqual(len(result['ranked_candidates']),10);self.assertEqual(len(result['selected_articles']),5)
            self.assertEqual([r['page_id'] for r in result['selected_articles']],['Page_'+str(i) for i in range(5)])
            self.assertEqual(selected_urls(urls=['https://en.wikipedia.org/wiki/One']*2)['selected_articles'][0]['rank'],1)

    def test_http_errors_are_bounded_and_do_not_become_cached_snippets(self):
        session=Session(Response({},status=503),Response(parsed()),Response(metadata()))
        with tempfile.TemporaryDirectory() as cache,patch('fetch_wikipedia_articles.time.sleep'):
            fetcher=WikipediaArticleFetcher(cache,session,min_interval=0,retries=1)
            fetcher.fetch('https://en.wikipedia.org/wiki/Jack_Falahee')
            self.assertEqual(fetcher.requests_made,3)
        self.assertTrue(is_disambiguation({'properties':{'disambiguation':''}}))

    def test_existing_run_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            sentinel=Path(folder)/'existing.txt';sentinel.write_text('keep this')
            argv=['fetch_wikipedia_articles','--url','https://en.wikipedia.org/wiki/Jack_Falahee','--output',folder]
            with patch('sys.argv',argv),self.assertRaises(FileExistsError):main()
            self.assertEqual(sentinel.read_text(),'keep this')

if __name__=='__main__':unittest.main()
