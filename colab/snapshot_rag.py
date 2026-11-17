"""Full FEVER Wikipedia snapshot -> per-claim temporary FAISS/SQLite RAG.

No HTTP calls and no gold-evidence lookup. The immutable source corpus is data;
only per-claim vectors/working copies are deleted. Saved traces retain supplied
passages and source provenance, not full working articles or vectors.
"""
import hashlib,json,re,sqlite3,tempfile,time
from pathlib import Path
from urllib.parse import quote
import fever_colab_runtime as core

def page_url(page):
 for token,value in [('-LRB-','('),('-RRB-',')'),('-LSB-','['),('-RSB-',']'),('-COLON-',':')]:page=page.replace(token,value)
 return 'https://en.wikipedia.org/wiki/'+quote(page,safe='')

class SnapshotCorpus:
 def __init__(self,folder,input_sha256):
  self.folder=Path(folder);self.manifest=json.loads((self.folder/'snapshot_manifest.json').read_text())
  if self.manifest['schema_version']!=1 or self.manifest['input_sha256']!=input_sha256:
   raise ValueError('Snapshot package does not match evaluation input/ranking file')
  if self.manifest['missing_pages'] or self.manifest['packaged_pages']!=self.manifest['selected_page_ids']:
   raise ValueError('Snapshot package has missing selected pages')
  database=self.folder/'snapshot.sqlite'
  if core.sha_file(database)!=self.manifest['snapshot_sha256']:raise ValueError('Snapshot database hash mismatch')
  self.connection=sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)
  if self.connection.execute('SELECT count(*) FROM articles').fetchone()[0]!=self.manifest['packaged_pages']:
   self.close();raise ValueError('Snapshot database page count mismatch')
 def close(self):self.connection.close()
 def protocol(self,root,embedding):
  return {'evidence':'Full article text supplied by the FEVER Wikipedia snapshot; first five existing ranked page IDs',
   'snapshot_sha256':self.manifest['snapshot_sha256'],'snapshot_schema':self.manifest['schema_version'],
   'input_sha256':core.sha_file(Path(root)/'data/eval_inputs.jsonl'),
   'embedding':embedding,'chunk_target_tokens':200,'vector_candidates':30,'context_character_budget':7600,
   'adapter_sha256':core.sha_file(__file__),
   'source_sha256':{n:core.sha_file(Path(root)/'src'/n) for n in ['fever_colab_runtime.py','ephemeral_rag.py']},
   'revision_note':'Historical snapshot record hashes and character spans; per-page revision IDs are unavailable',
   'retention':'Source corpus retained as input data; per-claim articles/chunks/vectors temporary; supplied passages retained',
   'wikipedia_http_requests':0}
 def read_article(self,page_id):
  row=self.connection.execute('SELECT page_id,text,lines,source_file,source_line,record_sha256 FROM articles WHERE page_id=?',(page_id,)).fetchone()
  if row is None:raise ValueError('Ranked page absent from packaged snapshot: '+page_id)
  ident,text,lines,filename,line_no,digest=row
  text_source='text'
  if not text.strip():
   # Some distributed records have sentence text but an empty aggregate field.
   text='\n'.join(parts[1] for line in lines.splitlines() for parts in [line.split('\t')] if len(parts)>1 and parts[0].isdigit() and parts[1].strip())
   text_source='all_sentence_text_from_lines'
  blocks=[]
  for match in re.finditer(r'[^\n]+',text):
   if not match.group().strip():continue
   blocks.append({'id':'B'+str(len(blocks)+1).zfill(5),'kind':'paragraph','text':match.group(),
                  'section_path':[],'source_char_start':match.start()})
  return {'page_id':ident,'title':ident.replace('_',' '),'url':page_url(ident),
   'revision_id':None,'revision_url':None,'blocks':blocks,
   'dataset':'FEVER Wikipedia snapshot','source_file':filename,'source_line':line_no,
   'record_sha256':digest,'text_source':text_source,'body_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),
   'body_characters':len(text)}
 def context_for(self,row,root,out,embedder,embedding,progress=None):
  from ephemeral_rag import article_chunks,ClaimVectorDatabase,pack_context
  root=Path(root);out=Path(out);protocol=self.protocol(root,embedding);core.freeze(out/'evidence',protocol)
  identity=core.signature({'id':row['id'],'claim':row['claim'],'ranked_articles':row['ranked_articles'],'protocol':core.signature(protocol)})
  path=out/'evidence'/('claim_'+str(row['id'])+'.json')
  if path.exists():
   saved=json.loads(path.read_text())
   if saved['input_identity']!=identity:raise ValueError('Cached snapshot evidence protocol differs')
   core.notify(progress,'evidence_cache_hit',evidence_status=saved['status'],passages=len(saved.get('supplied_evidence',{})))
   return saved
  started=time.perf_counter();sources=[];chunks=[];errors=[];workspace=None;database=None
  trace={'id':row['id'],'claim':row['claim'],'input_identity':identity,'status':'ERROR','wikipedia_http_requests':0,
         'source':'FEVER Wikipedia snapshot','snapshot_sha256':self.manifest['snapshot_sha256']}
  try:
   with tempfile.TemporaryDirectory(prefix='fever-snapshot-claim-') as temp:
    workspace=Path(temp)
    for selected in row['ranked_articles'][:5]:
     core.notify(progress,'reading_snapshot_article',page_id=selected['page_id'],rank=selected['rank'])
     article=self.read_article(selected['page_id'])
     sources.append({k:article[k] for k in ['page_id','title','url','revision_id','revision_url','dataset','source_file','source_line','record_sha256','text_source','body_sha256','body_characters']})
     if not article['blocks']:errors.append({'page_id':selected['page_id'],'error':'Empty snapshot article'})
     block_offsets={b['id']:b['source_char_start'] for b in article['blocks']}
     additions=article_chunks(article,embedder.tokenizer)
     for chunk in additions:
      chunk.update(dataset=article['dataset'],source_file=article['source_file'],source_line=article['source_line'],
       record_sha256=article['record_sha256'],body_sha256=article['body_sha256'],text_source=article['text_source'],
       snapshot_sha256=self.manifest['snapshot_sha256'],source_block_char_start=block_offsets[chunk['block_id']])
     chunks.extend(additions);del article
    trace['article_read_and_chunk_seconds']=time.perf_counter()-started
    if not chunks:raise ValueError('Selected snapshot pages contain no article text; no network or gold-evidence fallback')
    core.notify(progress,'embedding_started',chunks=len(chunks),snapshot_read_seconds=round(trace['article_read_and_chunk_seconds'],3))
    phase=time.perf_counter();vectors=embedder.encode([c['embedding_text'] for c in chunks]+[row['claim']],progress=progress)
    trace['embedding_seconds']=time.perf_counter()-phase
    database=ClaimVectorDatabase(workspace/'database',chunks,vectors[:-1])
    try:
     payload,citations=pack_context(row['claim'],database.search(vectors[-1:],30))
     if not citations:raise ValueError('No snapshot passages fit the fixed context budget')
     for citation in citations.values():
      citation['article_char_start']=citation['source_block_char_start']+citation['char_start']
      citation['article_char_end']=citation['source_block_char_start']+citation['char_end']
     trace.update(status='READY',payload=payload,supplied_evidence=citations,chunks=len(chunks))
    finally:database.close();database=None
  except Exception as exc:
   trace['error']=type(exc).__name__+': '+str(exc)[:1000]
   if database is not None:database.close()
  finally:
   trace.update(article_sources=sources,fetch_errors=errors,preparation_seconds=time.perf_counter()-started,
    articles_and_vectors_deleted=workspace is None or not workspace.exists(),created_utc=core.now())
  if not trace['articles_and_vectors_deleted']:raise RuntimeError('Temporary snapshot RAG database was not deleted')
  core.write_once(path,trace)
  core.notify(progress,'evidence_saved',evidence_status=trace['status'],supplied_passages=len(trace.get('supplied_evidence',{})),
              preparation_seconds=round(trace['preparation_seconds'],3),wikipedia_http_requests=0,exception=trace.get('error'))
  return trace
