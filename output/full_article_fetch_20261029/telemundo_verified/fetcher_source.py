"""Fetch full parsed Wikipedia articles for selected URLs, separately from FEVER.

Uses existing requests + Python's HTML parser. No LLM, embeddings, retraining or
changes to the old corpus. Raw HTML/API responses and revision provenance stay
in an immutable cache; each invocation writes a separate selection manifest.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time
from urllib.parse import parse_qs, quote, unquote, urlsplit

import requests

ROOT=Path(__file__).resolve().parents[1]
API='https://en.wikipedia.org/w/api.php'
EXTRACTOR='article_blocks_v1'
USER_AGENT='COSC874FEVEREvidenceCollector/0.1 (academic full-article research prototype)'


def now():return datetime.now(timezone.utc).isoformat()
def sha(data):return hashlib.sha256(data).hexdigest()
def normalized(text):return re.sub(r'\s+',' ',unescape(text)).strip()


class ArticleFetchError(RuntimeError):pass


def article_target(url):
    parsed=urlsplit(url)
    if parsed.scheme not in ('https','http') or parsed.hostname not in ('en.wikipedia.org','en.m.wikipedia.org') or parsed.username or parsed.password or parsed.port not in (None,80,443):
        raise ValueError('Provide an English Wikipedia article URL without credentials')
    query=parse_qs(parsed.query)
    if parsed.path.startswith('/wiki/'):title=unquote(parsed.path[len('/wiki/'):])
    elif parsed.path=='/w/index.php':title=query.get('title',[''])[0]
    else:raise ValueError('Expected /wiki/Article or /w/index.php?title=Article')
    title=title.replace('_',' ').strip()
    if not title or len(title)>500 or any(ord(c)<32 for c in title):raise ValueError('Invalid article title')
    if title.split(':',1)[0].casefold() in {'special','file','image','category','template','talk','user','user talk','wikipedia','portal','help','module','draft','mediawiki'}:
        raise ValueError('Only article pages are supported')
    revision=None
    if 'oldid' in query:
        revision=int(query['oldid'][0])
        if revision<=0:raise ValueError('Invalid revision ID')
    canonical='https://en.wikipedia.org/wiki/'+quote(title.replace(' ','_'),safe='')
    if revision:canonical+='?oldid='+str(revision)
    return dict(title=title,revision_id=revision,normalized_url=canonical)


class Node:
    def __init__(self,tag='',attrs=(),parent=None):self.tag=tag;self.attrs=dict(attrs);self.children=[];self.parent=parent
    @property
    def classes(self):return set(self.attrs.get('class','').split())


class ArticleDOM(HTMLParser):
    VOID={'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}
    def __init__(self):super().__init__(convert_charrefs=True);self.root=Node('root');self.stack=[self.root]
    def handle_starttag(self,tag,attrs):
        n=Node(tag,attrs,self.stack[-1]);self.stack[-1].children.append(n)
        if tag not in self.VOID:self.stack.append(n)
    def handle_startendtag(self,tag,attrs):
        self.handle_starttag(tag,attrs)
        if tag not in self.VOID:self.handle_endtag(tag)
    def handle_endtag(self,tag):
        for i in range(len(self.stack)-1,0,-1):
            if self.stack[i].tag==tag:del self.stack[i:];break
    def handle_data(self,data):self.stack[-1].children.append(data)


SKIP_TAGS={'script','style','noscript'}
SKIP_CLASSES={'navbox','vertical-navbox','sidebar','toc','mw-editsection','reflist','references','reference','noprint','hatnote','ambox','authority-control','sistersitebox','catlinks','printfooter'}


def nodes(node):
    yield node
    for child in node.children:
        if isinstance(child,Node):yield from nodes(child)


def text_of(node,omit=()):
    if node.tag in SKIP_TAGS or node.tag in omit or node.classes&SKIP_CLASSES:return ''
    if node.tag in ('br','hr'):return ' '
    return ''.join(text_of(c,omit) if isinstance(c,Node) else c for c in node.children)


def extract_article(html):
    dom=ArticleDOM();dom.feed(html);dom.close()
    root=next((n for n in nodes(dom.root) if 'mw-parser-output' in n.classes),dom.root)
    blocks=[];headings=[]
    def add(kind,text,node):
        text='\n'.join(normalized(line) for line in text.splitlines() if normalized(line)) if kind in ('table','infobox') else normalized(text)
        if text:blocks.append(dict(id='B%05d'%(len(blocks)+1),kind=kind,text=text,section_path=[h[1] for h in headings],html_id=node.attrs.get('id')))
    def walk(n):
        if n.tag in SKIP_TAGS or n.classes&SKIP_CLASSES:return
        if re.fullmatch(r'h[1-6]',n.tag):
            level=int(n.tag[1]);title=normalized(text_of(n))
            if title:
                while headings and headings[-1][0]>=level:headings.pop()
                headings.append((level,title));add('heading',title,n)
            return
        if n.tag=='table':
            rows=[]
            for tr in nodes(n):
                if tr.tag!='tr':continue
                parent=tr.parent
                while parent and parent.tag!='table':parent=parent.parent
                if parent is not n:continue
                cells=[normalized(text_of(c)) for c in tr.children if isinstance(c,Node) and c.tag in ('th','td')]
                if cells:rows.append(' | '.join(cells))
            caption=next((normalized(text_of(c)) for c in n.children if isinstance(c,Node) and c.tag=='caption'),'')
            add('infobox' if 'infobox' in n.classes else 'table','\n'.join(([caption] if caption else [])+rows),n)
            return
        if n.tag in ('p','dd','dt'):
            add('paragraph' if n.tag=='p' else 'definition',text_of(n),n);return
        if n.tag=='li':
            add('list_item',text_of(n,omit={'ul','ol','table'}),n)
            for c in n.children:
                if isinstance(c,Node) and c.tag in ('ul','ol','table'):walk(c)
            return
        if n.classes&{'thumbcaption','mw-file-caption'}:add('caption',text_of(n),n);return
        for child in n.children:
            if isinstance(child,Node):walk(child)
    walk(root)
    text='\n\n'.join(b['text'] for b in blocks)
    if not text:raise ArticleFetchError('Article HTML contained no extractable content; no snippet fallback used')
    links=[];images=[]
    for n in nodes(root):
        if n.tag=='a' and n.attrs.get('href'):links.append(dict(href=n.attrs['href'],text=normalized(text_of(n))))
        if n.tag=='img':images.append({k:n.attrs[k] for k in ('src','alt','width','height') if k in n.attrs})
    return dict(text=text,blocks=blocks,links=links,images=images,
        extraction_note='Full parsed article requested, without a section/intro limit. Clean text keeps body headings, paragraphs, lists, tables and infoboxes; navigation, edit controls and citation markers/reference lists are excluded. The complete returned HTML/API response is retained, including content excluded from clean text.')


def atomic_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.write-',delete=False,encoding='utf-8') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2);stream.write('\n');temp=Path(stream.name)
    os.replace(temp,path)


class WikipediaArticleFetcher:
    def __init__(self,cache_dir=None,session=None,min_interval=.25,retries=2,user_agent=USER_AGENT):
        self.cache=Path(cache_dir or ROOT/'data/processed/wikipedia_full_articles').resolve()
        self.session=session or requests.Session();self.owns_session=session is None
        self.session.headers.update({'User-Agent':user_agent,'Accept':'application/json'})
        self.min_interval=min_interval;self.retries=retries;self.last_request=0.;self.requests_made=0

    def close(self):
        if self.owns_session:self.session.close()

    def _get(self,params):
        for attempt in range(self.retries+1):
            wait=self.min_interval-(time.monotonic()-self.last_request)
            if wait>0:time.sleep(wait)
            self.last_request=time.monotonic();self.requests_made+=1
            try:
                response=self.session.get(API,params={**params,'format':'json','formatversion':2,'maxlag':5},timeout=(10,40),allow_redirects=False)
                if response.status_code in (429,500,502,503,504):
                    if attempt<self.retries:
                        retry=response.headers.get('Retry-After','0')
                        time.sleep(min(30,max(2**attempt,float(retry) if retry.isdigit() else 0)));continue
                response.raise_for_status()
                if response.status_code!=200:raise ArticleFetchError('Unexpected HTTP status '+str(response.status_code))
                data=response.json()
                if 'error' in data:
                    err=data['error']
                    if err.get('code')=='maxlag' and attempt<self.retries:time.sleep(2**attempt);continue
                    raise ArticleFetchError('Wikipedia API '+str(err.get('code'))+': '+str(err.get('info')))
                return data
            except (requests.Timeout,requests.ConnectionError):
                if attempt==self.retries:raise
                time.sleep(2**attempt)
        raise ArticleFetchError('Wikipedia request failed')

    def _cached(self,target):
        pointer=self.cache/'urls'/(sha(target['normalized_url'].encode())+'.json')
        if not pointer.exists():return None
        record=read_json(pointer);folder=(self.cache/record['folder']).resolve();folder.relative_to(self.cache)
        article=read_json(folder/'article.json')
        if article['extractor_version']!=EXTRACTOR:return None
        for name,digest in article['artifact_sha256'].items():
            if sha((folder/name).read_bytes())!=digest:raise ArticleFetchError('Cached artifact failed hash check: '+name)
        if sha(article['text'].encode('utf-8'))!=article['artifact_sha256']['article.txt']:raise ArticleFetchError('Cached article text differs from saved text artifact')
        return dict(article=article,folder=str(folder),cache_hit=True)

    def fetch(self,url,refresh=False):
        target=article_target(url)
        if not refresh:
            cached=self._cached(target)
            if cached:return {**cached,'requested_url':url}
        params=dict(action='parse',prop='text|sections|revid|displaytitle|categories|properties',redirects=1)
        if target['revision_id']:params['oldid']=target['revision_id']
        else:params['page']=target['title']
        parsed_response=self._get(params);parsed=parsed_response.get('parse',{})
        if not isinstance(parsed.get('text'),str) or not parsed.get('revid'):raise ArticleFetchError('Missing full article HTML or revision metadata')
        revision=int(parsed['revid'])
        metadata_response=self._get(dict(action='query',prop='info|revisions',revids=revision,inprop='url',rvprop='ids|timestamp'))
        pages=metadata_response.get('query',{}).get('pages',[])
        if len(pages)!=1 or pages[0].get('missing') or pages[0].get('ns')!=0:raise ArticleFetchError('Resolved page is missing or outside the article namespace')
        page=pages[0];revisions=page.get('revisions',[])
        if not revisions or revisions[0]['revid']!=revision or page['pageid']!=parsed['pageid']:raise ArticleFetchError('Article/revision mismatch')
        if target['revision_id'] and target['revision_id']!=revision:raise ArticleFetchError('Requested revision was not returned')
        html=parsed['text'];extracted=extract_article(html)
        article=dict(corpus='live_wikipedia_full_articles',language='en',title=parsed['title'],page_id=parsed['pageid'],revision_id=revision,
            revision_timestamp=revisions[0]['timestamp'],retrieved_utc=now(),url=page['fullurl'],
            revision_url='https://en.wikipedia.org/w/index.php?title='+quote(parsed['title'].replace(' ','_'),safe='')+'&oldid='+str(revision),
            requested_url=url,redirects=parsed.get('redirects',[]),display_title=parsed.get('displaytitle',parsed['title']),
            api_sections=parsed.get('sections',[]),categories=parsed.get('categories',[]),properties=parsed.get('properties',[]),
            extractor_version=EXTRACTOR,**extracted)
        raw_files={'article.html':html.encode('utf-8'),'article.txt':extracted['text'].encode('utf-8'),
            'parse_response.json':json.dumps(parsed_response,ensure_ascii=False,indent=2).encode('utf-8'),
            'metadata_response.json':json.dumps(metadata_response,ensure_ascii=False,indent=2).encode('utf-8')}
        article['artifact_sha256']={name:sha(body) for name,body in raw_files.items()}
        # Preserve render changes (including templates) even at the same revision.
        fingerprint=sha(json.dumps(article['artifact_sha256'],sort_keys=True).encode())[:16]
        pages_dir=self.cache/'pages';pages_dir.mkdir(parents=True,exist_ok=True)
        folder=pages_dir/('enwiki_%s_%s_%s_%s'%(parsed['pageid'],revision,fingerprint,EXTRACTOR))
        if not folder.exists():
            scratch=Path(tempfile.mkdtemp(prefix='.pending-',dir=pages_dir))
            try:
                for name,body in raw_files.items():(scratch/name).write_bytes(body)
                (scratch/'article.json').write_text(json.dumps(article,ensure_ascii=False,indent=2)+'\n')
                try:scratch.rename(folder)
                except OSError:
                    if not folder.exists():raise
                    existing=read_json(folder/'article.json')
                    if existing['artifact_sha256']!=article['artifact_sha256']:raise ArticleFetchError('Concurrent cache write differs')
                    article=existing
            finally:
                if scratch.exists():shutil.rmtree(scratch)
        else:article=read_json(folder/'article.json')
        pointer=self.cache/'urls'/(sha(target['normalized_url'].encode())+'.json')
        atomic_json(pointer,dict(normalized_url=target['normalized_url'],folder=str(folder.relative_to(self.cache)),updated_utc=now()))
        return dict(article=article,folder=str(folder),cache_hit=False,requested_url=url)


def read_json(path):return json.loads(Path(path).read_text())


def is_disambiguation(article):
    properties=article.get('properties',{})
    return 'disambiguation' in properties if isinstance(properties,dict) else any(p.get('name')=='disambiguation' for p in properties)


def page_url(page):
    for token,value in [('-LRB-','('),('-RRB-',')'),('-LSB-','['),('-RSB-',']'),('-COLON-',':')]:page=page.replace(token,value)
    return 'https://en.wikipedia.org/wiki/'+quote(page,safe='')


def selected_urls(claim=None,urls=None,db=None,top_k=5):
    if not 1<=top_k<=5:raise ValueError('Select between one and five articles')
    if claim is not None:
        from retrieve_title_aware import find_titles
        db=Path(db or ROOT/'data/processed/wiki_fts.db').resolve()
        connection=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)
        try:pages=find_titles(connection,claim,10)
        finally:connection.close()
        ranked=[dict(rank=i,page_id=page,url=page_url(page)) for i,page in enumerate(pages,1)]
        method='Unchanged existing title FTS/BM25 ranking: retrieve 10 candidate articles, retain the first top_k. No new article reranker.'
    else:
        ranked=[dict(rank=i,url=u) for i,u in enumerate(urls or [],1)][:10]
        method='Caller-provided ranked Wikipedia URLs; retain the first top_k from up to 10 candidates.'
    seen=set();selected=[]
    for row in ranked:
        key=article_target(row['url'])['normalized_url']
        if key in seen:continue
        seen.add(key);selected.append(row)
        if len(selected)==top_k:break
    return dict(claim=claim,selection_method=method,ranked_candidates=ranked,selected_articles=selected)


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--claim');group.add_argument('--url',action='append');group.add_argument('--urls-file',help='JSON list of ranked URLs')
    parser.add_argument('--output',required=True,help='New run directory; existing directories are rejected')
    parser.add_argument('--cache');parser.add_argument('--db');parser.add_argument('--top-k',type=int,default=5)
    parser.add_argument('--refresh',action='store_true',help='Fetch current content again, preserving old cache artifacts')
    parser.add_argument('--user-agent',default=USER_AGENT);args=parser.parse_args()
    urls=read_json(args.urls_file) if args.urls_file else args.url
    if urls is not None and (not isinstance(urls,list) or any(not isinstance(u,str) for u in urls)):raise ValueError('Expected a JSON list of Wikipedia URL strings')
    selection=selected_urls(args.claim,urls,args.db,args.top_k)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'fetcher_source.py').write_bytes(Path(__file__).read_bytes())
    (out/'retriever_source.py').write_bytes((ROOT/'src/retrieve_title_aware.py').read_bytes())
    fetcher=WikipediaArticleFetcher(args.cache,user_agent=args.user_agent)
    manifest=dict(status='fetching',started_utc=now(),**selection,articles=[],errors=[],no_llm_calls=True,no_embedding_calls=True,
        source_sha256={str(Path(__file__).resolve()):sha(Path(__file__).read_bytes()),str(ROOT/'src/retrieve_title_aware.py'):sha((ROOT/'src/retrieve_title_aware.py').read_bytes())},
        corpus_note='Fetched Wikipedia revisions form a separate corpus. They are not the historical FEVER sentence snapshot or FEVER evidence IDs.')
    atomic_json(out/'manifest.json',manifest)
    try:
        for selected in selection['selected_articles']:
            try:
                result=fetcher.fetch(selected['url'],args.refresh);article=result['article']
                entry=dict(rank=selected['rank'],requested_url=selected['url'],title=article['title'],revision_id=article['revision_id'],
                    revision_timestamp=article['revision_timestamp'],retrieved_utc=article['retrieved_utc'],url=article['url'],revision_url=article['revision_url'],
                    folder=result['folder'],cache_hit=result['cache_hit'],characters=len(article['text']),words=len(article['text'].split()),
                    blocks=len(article['blocks']),body_headings=sum(b['kind']=='heading' for b in article['blocks']),
                    is_disambiguation=is_disambiguation(article))
                manifest['articles'].append(entry);print(json.dumps(entry),flush=True)
            except (ArticleFetchError,requests.RequestException,ValueError,KeyError,OSError) as exc:
                error=dict(rank=selected['rank'],url=selected['url'],error=type(exc).__name__+': '+str(exc));manifest['errors'].append(error);print(json.dumps(error),flush=True)
            atomic_json(out/'manifest.json',manifest)
        manifest['status']='complete' if not manifest['errors'] and manifest['articles'] else 'partial' if manifest['articles'] else 'failed'
        manifest['finished_utc']=now();manifest['http_requests']=fetcher.requests_made;atomic_json(out/'manifest.json',manifest)
    finally:fetcher.close()
    if manifest['errors'] or not manifest['articles']:raise SystemExit(1)


if __name__=='__main__':main()
