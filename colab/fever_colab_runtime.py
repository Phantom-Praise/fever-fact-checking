"""Colab adapter for existing full-article RAG. No work or downloads on import.

One GPU model at a time; immutable per-claim checkpoints on Drive. This is a new
HF/4-bit experiment, not a replay of the earlier Ollama 8B measurement.
"""
from collections import Counter
from datetime import datetime,timezone
import gc,hashlib,importlib.metadata,json,math,os,random,re,tempfile,time
from pathlib import Path
LABELS=['SUPPORTS','REFUTES','NOT ENOUGH INFO']
MODELS={'qwen':'Qwen/Qwen2.5-3B-Instruct','llama':'meta-llama/Llama-3.2-3B-Instruct'}

def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def signature(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def read_rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def normalize(s):return ' '.join(s.casefold().split())
def now():return datetime.now(timezone.utc).isoformat()
def write_once(path,value):
 """Publish complete JSON only. Never replace a completed prediction/context."""
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  if json.loads(path.read_text())!=value:raise ValueError('Existing immutable artifact differs: '+str(path))
  return
 temp=path.with_name('.'+path.name+'.pending')
 with temp.open('w') as f:
  json.dump(value,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
 os.replace(temp,path)
def freeze(folder,protocol):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);path=folder/'protocol.json'
 if path.exists():
  if json.loads(path.read_text())!=protocol:raise ValueError('Protocol differs. Resume with the original settings, or choose a NEW run name.')
 else:write_once(path,protocol)
 return signature(protocol)
def verify_bundle(root):
 root=Path(root);m=json.loads((root/'manifest.json').read_text())
 for name,digest in m['files'].items():
  if sha_file(root/name)!=digest:raise ValueError('Bundle checksum failed: '+name)
 rows=read_rows(root/'data/eval_inputs.jsonl');refs=json.loads((root/'data/eval_labels.json').read_text())
 if len(rows)!=19998 or len({r['id'] for r in rows})!=19998 or {str(r['id']) for r in rows}!=set(refs):raise ValueError('Expected all 19,998 official development IDs')
 if Counter(refs.values())!=Counter({l:6666 for l in LABELS}):raise ValueError('Unexpected official label counts')
 if any('label' in r or 'gold' in r or 'evidence' in r for r in rows):raise ValueError('Evaluation features contain labels/annotations')
 eval_text={normalize(r['claim']) for r in rows}
 train=read_rows(root/'data/train_pilot.jsonl');val=read_rows(root/'data/validation_pilot.jsonl')
 sets=[{normalize(r['claim']) for r in x} for x in [train,val]]
 if sets[0]&sets[1] or any(s&eval_text for s in sets):raise ValueError('Normalized-claim split overlap')
 print('Verified bundle: 19,998 evaluation claims; labels separate; pilot training/validation disjoint.')
 return rows,refs

def load_hf_token():
 try:
  from google.colab import userdata
  return userdata.get('HF_TOKEN')
 except Exception:return os.environ.get('HF_TOKEN')
def remote_metadata(repo,requested_revision=None):
 from huggingface_hub import HfApi
 try:info=HfApi().model_info(repo,revision=requested_revision,files_metadata=True,token=load_hf_token())
 except Exception as e:raise RuntimeError('Cannot access '+repo+'. For Llama, obtain model access yourself and add HF_TOKEN in Colab Secrets. No automatic license acceptance. Error type: '+type(e).__name__) from None
 size=sum((getattr(s,'size',0) or 0) for s in (info.siblings or []) if s.rfilename.endswith('.safetensors'))
 print(repo,'revision',info.sha,'weight download approximately',round(size/2**30,2),'GiB before 4-bit runtime quantization')
 return {'repo':repo,'revision':info.sha,'safetensor_download_bytes':size}
def environment():
 import torch
 packages={p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate','bitsandbytes','numpy','scikit-learn','faiss-cpu']}
 return {'packages':packages,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'cuda':torch.version.cuda}
def check_cuda():
 import torch
 if not torch.cuda.is_available():raise RuntimeError('Select Runtime > Change runtime type > T4 GPU, then rerun setup. No full CPU run is started.')
 print('GPU:',torch.cuda.get_device_name(0),'VRAM GiB:',round(torch.cuda.get_device_properties(0).total_memory/2**30,1))
def release():
 import torch
 gc.collect()
 if torch.cuda.is_available():torch.cuda.empty_cache()

RUNTIME_REVISION = 'colab-debug-v5'

class RunMonitor:
 """Console progress plus durable operational logs; never modifies predictions.

The background thread prints stage elapsed time. It is not a Colab keep-alive.
"""
 def __init__(self,folder,condition,total,completed=0,invalid=0,heartbeat_seconds=20,show_progress=True,unit='claim'):
  import threading
  self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
  self.condition=condition;self.started=time.perf_counter();self.stage_started=self.started
  self.heartbeat_seconds=max(float(heartbeat_seconds),.05);self.show_progress=show_progress
  self.lock=threading.RLock();self.stop_event=threading.Event();self.durations=[];self.bar=None
  self.session_id=str(time.time_ns());self.log_path=self.folder/('session_'+self.session_id+'.jsonl')
  self.stream=self.log_path.open('x',encoding='utf-8')
  self.state={'session_id':self.session_id,'condition':condition,'total':total,'completed':completed,'invalid_total':invalid,
              'status':'running','stage':'starting','claim_id':None,'session_new_claims':0,'created_utc':now(),'unit':unit}
  if show_progress:
   try:
    from tqdm.auto import tqdm
    self.bar=tqdm(total=total,initial=completed,desc=condition,unit=unit,dynamic_ncols=True)
   except ImportError:pass
  self.thread=threading.Thread(target=self._heartbeat,daemon=True);self.thread.start()
  self.event('session_started',log_path=str(self.log_path))
 def _gpu(self):
  try:
   import torch
   if torch.cuda.is_available():return {'allocated_gib':round(torch.cuda.memory_allocated()/2**30,2),'reserved_gib':round(torch.cuda.memory_reserved()/2**30,2)}
  except Exception:pass
  return None
 def _save_status(self):
  status={**self.state,'updated_utc':now(),'session_elapsed_seconds':round(time.perf_counter()-self.started,2),
          'stage_elapsed_seconds':round(time.perf_counter()-self.stage_started,2),'gpu_memory':self._gpu()}
  temp=self.folder/('status_'+self.session_id+'.pending')
  temp.write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(temp,self.folder/'status.json')
 def _say(self,text):
  if not self.show_progress:return
  line='['+now()[11:19]+' UTC] '+self.condition+' | '+text
  if self.bar is not None:self.bar.write(line)
  else:print(line,flush=True)
 def event(self,stage,**fields):
  with self.lock:
   if self.stream.closed:return
   if stage!=self.state.get('stage'):
    self.stage_started=time.perf_counter()
   if stage=='claim_started':
    for key in ['prediction','input_tokens','output_tokens','raw_preview','exception']:self.state.pop(key,None)
   self.state['stage']=stage;self.state.update(fields)
   record={'utc':now(),'session_id':self.session_id,'condition':self.condition,'claim_id':self.state.get('claim_id'),'event':stage,**fields}
   self.stream.write(json.dumps(record,ensure_ascii=False)+'\n');self.stream.flush()
   self._save_status()
   show={k:v for k,v in fields.items() if k not in {'claim','raw_preview','exception'}}
   self._say(('claim #'+str(self.state['claim_id'])+' | ' if self.state.get('claim_id') is not None else '')+stage+' '+json.dumps(show,ensure_ascii=False))
   if fields.get('claim'):self._say('Claim: '+fields['claim'][:180])
   if fields.get('raw_preview'):self._say('Raw output preview: '+fields['raw_preview'])
   if fields.get('exception'):self._say('Error: '+fields['exception'])
 def saved(self,trace):
  seconds=trace['wall_seconds'];self.durations.append(seconds)
  self.state['completed']+=1;self.state['session_new_claims']+=1
  self.state['invalid_total']+=int(trace['label']=='INVALID')
  mean=sum(self.durations[-50:])/len(self.durations[-50:]);remaining=self.state['total']-self.state['completed']
  eta=remaining*mean if len(self.durations)>=3 else None
  if self.bar is not None:self.bar.update(1);self.bar.set_postfix(pred=trace['label'],invalid=self.state['invalid_total'],refresh=False)
  self.event('claim_saved',prediction=trace['label'],completed=self.state['completed'],total=self.state['total'],
   invalid_total=self.state['invalid_total'],claim_seconds=round(seconds,2),recent_mean_seconds=round(mean,2),
   rough_remaining_hours=round(eta/3600,2) if eta is not None else None,
   input_tokens=trace.get('input_tokens'),output_tokens=trace.get('output_tokens'),
   raw_preview=trace.get('response','')[:180],exception=trace.get('error'),
   prediction_path=str(trace.get('prediction_path','')))
 def _heartbeat(self):
  while not self.stop_event.wait(self.heartbeat_seconds):
   with self.lock:
    if self.stream.closed:return
    # Keep the active stage; a heartbeat is not progress or another model call.
    elapsed=round(time.perf_counter()-self.stage_started,1)
    self.stream.write(json.dumps({'utc':now(),'event':'heartbeat','session_id':self.session_id,'claim_id':self.state.get('claim_id'),
      'stage':self.state.get('stage'),'stage_elapsed_seconds':elapsed})+'\n');self.stream.flush();self._save_status()
    self._say('Still working | claim #'+str(self.state.get('claim_id'))+' | '+self.state.get('stage','')+' | '+str(elapsed)+' s in this stage')
 def close(self,status='paused',reason=None):
  self.stop_event.set();self.thread.join(timeout=2)
  with self.lock:
   if self.stream.closed:return
   self.state['status']=status;self.event('session_ended',status=status,reason=reason)
   self.stream.close()
   if self.bar is not None:self.bar.close()

def notify(progress,stage,**fields):
 if progress is not None:progress(stage,**fields)

def inspect_saved(out,condition,claim_id=None,show_reference=False,root=None,show_full=False):
 """Read saved results after inference; reference labels never enter model input."""
 folder=Path(out)/'predictions'/condition
 paths=[p for p in folder.glob('*.json') if p.name!='protocol.json']
 if claim_id is None:
  if not paths:print('No predictions saved yet. Check monitoring status below.');return None
  path=max(paths,key=lambda p:p.stat().st_mtime_ns)
 else:path=folder/(str(claim_id)+'.json')
 t=json.loads(path.read_text())
 summary={k:t.get(k) for k in ['id','claim','label','model_label','consistent','evidence_status','input_tokens','output_tokens','truncated',
                               'stopped_normally','wall_seconds','generation_seconds','error']}
 if show_reference:
  if root is None:raise ValueError('Provide the project root to inspect reference labels')
  summary['reference_label']=json.loads((Path(root)/'data/eval_labels.json').read_text())[str(t['id'])]
  summary['matches_reference']=summary['label']==summary['reference_label']
 print(json.dumps(summary,indent=2,ensure_ascii=False));print('Prediction file:',path)
 evidence_path=Path(out)/'evidence'/('claim_'+str(t['id'])+'.json')
 if show_full:
  print('\nEXACT PROMPT\n'+t.get('prompt','RoBERTa retained input: '+t.get('retained_input','Not saved')))
  print('\nFULL RAW RESPONSE\n'+t.get('response',json.dumps(t.get('probabilities'),indent=2)))
  if evidence_path.exists():
   evidence=json.loads(evidence_path.read_text());print('\nSUPPLIED PASSAGES\n'+json.dumps(evidence.get('supplied_evidence'),indent=2,ensure_ascii=False))
   if evidence.get('status')!='READY':print('\nEVIDENCE FAILURE DETAILS\n'+json.dumps(evidence,indent=2,ensure_ascii=False))
 return summary

def monitor_status(out,condition,last_events=8):
 folder=Path(out)/'monitor'/condition;path=folder/'status.json'
 if not path.exists():print('No monitoring session found for',condition);return None
 status=json.loads(path.read_text());print(json.dumps(status,indent=2,ensure_ascii=False))
 logs=sorted(folder.glob('session_*.jsonl'),key=lambda p:p.stat().st_mtime_ns)
 if logs:
  from collections import deque
  with logs[-1].open() as f:tail=list(deque(f,maxlen=last_events))
  print('\nRECENT EVENTS\n'+''.join(tail));print('Full log:',logs[-1])
 return status


class ResidentEmbedder:
 """Reuse MiniLM weights across claims; original chunking and vector search."""
 def __init__(self,metadata):
  import torch
  from transformers import AutoTokenizer,AutoModel
  self.torch=torch;self.tokenizer=AutoTokenizer.from_pretrained(metadata['repo'],revision=metadata['revision'],trust_remote_code=False)
  # Small encoder stays on CPU, keeping Colab VRAM free for the classifier.
  self.model=AutoModel.from_pretrained(metadata['repo'],revision=metadata['revision'],trust_remote_code=False).eval().to('cpu')
  torch.set_num_threads(2)
 def encode(self,texts,progress=None):
  import numpy as np
  result=[]
  for start in range(0,len(texts),32):
   batch=texts[start:start+32]
   if any(len(self.tokenizer.encode(x,add_special_tokens=True))>256 for x in batch):raise ValueError('Embedding text exceeds frozen 256-token limit')
   encoded=self.tokenizer(batch,padding=True,truncation=False,return_tensors='pt')
   with self.torch.inference_mode():
    hidden=self.model(**encoded).last_hidden_state;mask=encoded['attention_mask'].unsqueeze(-1)
    v=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
    result.append(self.torch.nn.functional.normalize(v,p=2,dim=1).numpy().astype('float32'))
   if start==0 or start+32>=len(texts) or (start//32+1)%10==0:notify(progress,'embedding_batch',embedded=min(start+32,len(texts)),total_texts=len(texts))
  return np.concatenate(result)

def context_protocol(root,embedding):
 root=Path(root)
 return {'evidence':'full live Wikipedia articles; first five of existing top-ten ranked URLs; all article/vector workspaces temporary',
 'embedding':embedding,'chunk_target_tokens':200,'vector_candidates':30,'context_character_budget':7600,
 'input_sha256':sha_file(root/'data/eval_inputs.jsonl'),'source_sha256':{n:sha_file(root/'src'/n) for n in ['fever_colab_runtime.py','fetch_wikipedia_articles.py','ephemeral_rag.py']}}
def context_for(row,root,out,embedder,embedding,progress=None):
 from fetch_wikipedia_articles import WikipediaArticleFetcher
 from ephemeral_rag import article_chunks,ClaimVectorDatabase,pack_context
 root=Path(root);out=Path(out);proto=context_protocol(root,embedding);freeze(out/'evidence',proto)
 input_identity=signature({'id':row['id'],'claim':row['claim'],'ranked_articles':row.get('ranked_articles',[]),'protocol':signature(proto)})
 path=out/'evidence'/('claim_'+str(row['id'])+'.json')
 if path.exists():
  saved=json.loads(path.read_text())
  if saved['input_identity']!=input_identity:raise ValueError('Cached evidence input/protocol mismatch')
  notify(progress,'evidence_cache_hit',passages=len(saved.get('supplied_evidence',{})),evidence_status=saved['status'])
  return saved
 started=time.perf_counter();chunks=[];sources=[];errors=[];db=None;workspace=None
 trace={'id':row['id'],'claim':row['claim'],'input_identity':input_identity,'status':'ERROR'}
 try:
  with tempfile.TemporaryDirectory(prefix='fever-claim-') as work:
   workspace=Path(work);fetcher=WikipediaArticleFetcher(cache_dir=workspace/'articles')
   try:
    for article_index,selected in enumerate(row.get('ranked_articles',[])[:5],1):
     notify(progress,'fetching_article',article=article_index,total_articles=min(5,len(row.get('ranked_articles',[]))),url=selected['url'])
     article_started=time.perf_counter()
     try:
      article=fetcher.fetch(selected['url'])['article']
      sources.append({k:article[k] for k in ['title','url','page_id','revision_id','revision_url','revision_timestamp','retrieved_utc','artifact_sha256']})
      chunks.extend(article_chunks(article,embedder.tokenizer))
      notify(progress,'article_fetched',title=article['title'],article_seconds=round(time.perf_counter()-article_started,2),chunks_so_far=len(chunks))
      del article
     except Exception as exc:
      errors.append({'url':selected['url'],'error':type(exc).__name__+': '+str(exc)[:500]})
      notify(progress,'article_fetch_failed',url=selected['url'],exception=errors[-1]['error'])
   finally:fetcher.close()
   trace['fetch_seconds']=time.perf_counter()-started
   if not chunks:raise ValueError('No full-article evidence. No title-only or gold-evidence fallback.')
   notify(progress,'embedding_started',chunks=len(chunks));phase=time.perf_counter()
   vectors=embedder.encode([x['embedding_text'] for x in chunks]+[row['claim']],progress=progress);trace['embedding_seconds']=time.perf_counter()-phase
   notify(progress,'vector_search_started',embedding_seconds=round(trace['embedding_seconds'],2))
   try:
    db=ClaimVectorDatabase(workspace/'database',chunks,vectors[:-1]);payload,citations=pack_context(row['claim'],db.search(vectors[-1:],30))
    if not citations:raise ValueError('No supplied passages fit the fixed context budget')
    trace.update(status='READY',payload=payload,supplied_evidence=citations,chunks=len(chunks),article_sources=sources,fetch_errors=errors)
   finally:
    if db is not None:db.close();db=None
 except Exception as exc:trace.update(error=type(exc).__name__+': '+str(exc)[:1000],fetch_errors=errors,article_sources=sources)
 finally:
  trace.update(preparation_seconds=time.perf_counter()-started,articles_and_vectors_deleted=workspace is None or not workspace.exists(),created_utc=now())
 if not trace['articles_and_vectors_deleted']:raise RuntimeError('Temporary article/vector cleanup failed')
 write_once(path,trace)
 notify(progress,'evidence_saved',evidence_status=trace['status'],supplied_passages=len(trace.get('supplied_evidence',{})),preparation_seconds=round(trace['preparation_seconds'],2),exception=trace.get('error'))
 return trace

def parse_parts(text,context,stopped_normally):
 if not stopped_normally:raise ValueError('Output reached its token limit before a normal stop')
 # Accept only an optional, complete Markdown wrapper, never extract JSON
 # from surrounding prose or repair incomplete/model-generated content.
 body=text.strip()
 fenced=re.fullmatch(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n```',body,re.DOTALL|re.IGNORECASE)
 if fenced:body=fenced.group(1)
 p=json.loads(body)
 if not isinstance(p,dict) or set(p) not in ({'parts'},{'parts','label'}) or ('label' in p and p['label'] not in LABELS) or not isinstance(p['parts'],list) or not p['parts']:raise ValueError('Invalid top-level parts schema')
 for item in p['parts']:
  if not isinstance(item,dict) or set(item)!={'part','verdict','evidence_ids','justification'}:raise ValueError('Invalid part schema')
  if any(not isinstance(item[k],str) or not item[k].strip() for k in ['part','justification']) or item['verdict'] not in LABELS:raise ValueError('Invalid part text or verdict')
  ids=item['evidence_ids']
  if not isinstance(ids,list) or any(not isinstance(i,str) or i not in context['supplied_evidence'] for i in ids) or len(set(ids))!=len(ids):raise ValueError('Invalid evidence IDs')
  if item['verdict']!='NOT ENOUGH INFO' and not ids:raise ValueError('Supported/refuted part needs a citation')
 verdicts=[i['verdict'] for i in p['parts']]
 derived='REFUTES' if 'REFUTES' in verdicts else 'SUPPORTS' if all(x=='SUPPORTS' for x in verdicts) else 'NOT ENOUGH INFO'
 return {'label':derived,'model_label':p.get('label'),'parts':p['parts'],
         'consistent':derived==p['label'] if 'label' in p else None,
         'label_source':'derived_from_validated_parts','model_label_missing':'label' not in p,
         'response_format':'markdown_json' if fenced else 'json'}

class CausalVerifier:
 def __init__(self,meta,root):
  import torch
  from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
  check_cuda();self.torch=torch;self.meta=meta;self.prompt=(Path(root)/'parts_prompt.txt').read_text()
  self.tokenizer=AutoTokenizer.from_pretrained(meta['repo'],revision=meta['revision'],token=load_hf_token(),trust_remote_code=False)
  self.tokenizer.pad_token=self.tokenizer.eos_token
  quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
  self.model=AutoModelForCausalLM.from_pretrained(meta['repo'],revision=meta['revision'],token=load_hf_token(),trust_remote_code=False,quantization_config=quant,device_map={'':0},torch_dtype=torch.float16).eval()
 def predict(self,row,context,max_new_tokens=512,max_context=4096,progress=None):
  if context['status']!='READY':raise ValueError('Evidence preparation failed; not a NEI prediction')
  prompt=self.prompt+'\n\nINPUT JSON:\n'+json.dumps(context['payload'],ensure_ascii=False)
  messages=[{'role':'user','content':prompt}]
  rendered=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  encoded=self.tokenizer(rendered,add_special_tokens=False,truncation=False,return_tensors='pt').to('cuda')
  count=int(encoded['input_ids'].shape[1])
  notify(progress,'prompt_ready',input_tokens=count,reserved_output_tokens=max_new_tokens,context_limit=max_context)
  if count+max_new_tokens>=max_context:
   error='Prompt exceeds frozen context budget; no silent truncation'
   notify(progress,'prompt_rejected',exception=error)
   return {'label':'INVALID','prompt':prompt,'rendered_prompt':rendered,'input_tokens':count,'output_tokens':0,'model_called':False,'error':error}
  notify(progress,'generating_response',input_tokens=count);generation_started=time.perf_counter()
  with self.torch.inference_mode():
   output=self.model.generate(**encoded,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=self.tokenizer.pad_token_id)
  generated=output[0,count:];raw=self.tokenizer.decode(generated,skip_special_tokens=True)
  eos=self.model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
  stop=bool(len(generated) and int(generated[-1]) in eos)
  notify(progress,'response_received',output_tokens=len(generated),stopped_normally=stop)
  result={'label':'INVALID','generation_seconds':time.perf_counter()-generation_started,'prompt':prompt,'rendered_prompt':rendered,'response':raw,'input_tokens':count,'output_tokens':len(generated),'stopped_normally':stop}
  try:result.update(parse_parts(raw,context,stop))
  except (ValueError,TypeError,KeyError) as exc:result['error']=str(exc)
  notify(progress,'response_validated' if result['label']!='INVALID' else 'response_invalid',prediction=result['label'],exception=result.get('error'))
  return result
 def close(self):self.model=None;self.tokenizer=None;release()

class RobertaVerifier:
 def __init__(self,path,max_length=512):
  import torch
  from transformers import AutoTokenizer,AutoModelForSequenceClassification
  check_cuda();self.torch=torch;self.max_length=max_length;self.path=Path(path)
  self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
  self.model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True,trust_remote_code=False).to('cuda').eval()
  if self.model.config.num_labels!=3:raise ValueError('Expected three-label RoBERTa checkpoint')
 def predict(self,row,context,progress=None,**unused):
  if context['status']!='READY':raise ValueError('Evidence preparation failed; not a NEI prediction')
  # Preserve original supplied order, not only evidence cited by a model.
  text=' '.join(e['text'] for p in context['payload']['passages'] for e in p['evidence'])
  uncut=self.tokenizer(row['claim'],text,truncation=False)['input_ids']
  encoded=self.tokenizer(row['claim'],text,truncation='only_second',max_length=self.max_length,return_tensors='pt').to('cuda')
  notify(progress,'roberta_inference',input_tokens=int(encoded['input_ids'].shape[1]),untruncated_tokens=len(uncut),truncated=len(uncut)>self.max_length)
  inference_started=time.perf_counter()
  with self.torch.inference_mode():prob=self.model(**encoded).logits.softmax(-1)[0].float().cpu().tolist()
  return {'label':LABELS[max(range(3),key=lambda i:prob[i])],'inference_seconds':time.perf_counter()-inference_started,'probabilities':prob,'input_tokens':encoded['input_ids'].shape[1],
    'untruncated_tokens':len(uncut),'truncated':len(uncut)>self.max_length,'retained_input':self.tokenizer.decode(encoded['input_ids'][0].cpu().tolist())}
 def close(self):self.model=None;self.tokenizer=None;release()

def checkpoint_identity(path):return {p.name:sha_file(p) for p in sorted(Path(path).iterdir()) if p.is_file() and p.suffix in {'.json','.txt','.safetensors','.bin'}}
def verify_diagnostic(root):
 root=Path(root);manifest=json.loads((root/'diagnostic.json').read_text())
 for name,digest in manifest['files'].items():
  if sha_file(root/'data'/name)!=digest:raise ValueError('Diagnostic checksum mismatch')
 rows=read_rows(root/'data/eval_inputs.jsonl');labels=json.loads((root/'data/eval_labels.json').read_text())
 if len(rows)!=90 or len({r['id'] for r in rows})!=90 or len({normalize(r['claim']) for r in rows})!=90:raise ValueError('Expected 90 distinct diagnostic claims')
 if set(labels)!={str(r['id']) for r in rows} or Counter(labels.values())!=Counter({l:30 for l in LABELS}):raise ValueError('Expected 30 diagnostic references per label')
 if any(set(r)!={'id','claim','ranked_articles'} for r in rows):raise ValueError('Diagnostic model inputs contain unexpected fields')
 return rows

def run_condition(rows,root,out,name,kind,metadata,embedding,minutes=45,max_new_tokens=512,max_length=512,
                  show_progress=True,heartbeat_seconds=20,max_new_claims=None,max_consecutive_invalid=10,scope='official_dev'):
 """Attempt every input in order, with stage logs and resumable session limits."""
 import torch
 if scope=='development90':
  if rows!=verify_diagnostic(root):raise ValueError('Diagnostic inputs differ from the fixed sample')
 elif scope!='official_dev' or len(rows)!=19998:raise ValueError('Full-run target must remain all 19,998 official claims; use a verified development diagnostic separately')
 if max_consecutive_invalid<1:raise ValueError('Invalid-output stop threshold must be positive')
 if minutes<=0:raise ValueError('Session minutes must be positive')
 if max_new_claims is not None and (not isinstance(max_new_claims,int) or max_new_claims<1):raise ValueError('Use a positive new-claim limit, or None')
 random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 root=Path(root);out=Path(out);folder=out/'predictions'/name
 identity=metadata if kind!='roberta' else checkpoint_identity(metadata['path'])
 protocol={'kind':kind,'identity':identity,'seed':42,'input_sha256':sha_file(root/'data/eval_inputs.jsonl'),
 'prompt_sha256':sha_file(root/'parts_prompt.txt'),'runtime_sha256':sha_file(root/'src/fever_colab_runtime.py'),'evidence_protocol':context_protocol(root,embedding),
 'max_new_tokens':max_new_tokens,'max_context':4096,'roberta_max_length':max_length,'decoding':'greedy; no silent retry or JSON repair',
 'environment':environment(),'max_consecutive_invalid':max_consecutive_invalid,
 'evaluation_scope':('90-claim original-validation diagnostic; not final evaluation' if scope=='development90' else 'official development, all 19998; not an untouched test or the old 60-claim experiment')}
 fingerprint=freeze(folder,protocol);pending=[];saved=[]
 print('Checking existing prediction files before resuming…',flush=True)
 for row in rows:
  path=folder/(str(row['id'])+'.json')
  if path.exists():
   t=json.loads(path.read_text())
   if t.get('id')!=row['id'] or t.get('claim')!=row['claim'] or t.get('protocol_identity')!=fingerprint:raise ValueError('Saved prediction does not match frozen protocol')
   saved.append(t)
  else:pending.append(row)
 if not pending:print(name,'already complete');return
 monitor=RunMonitor(out/'monitor'/name,name,len(rows),completed=len(saved),invalid=sum(t['label']=='INVALID' for t in saved),
                    heartbeat_seconds=heartbeat_seconds,show_progress=show_progress)
 embedder=verifier=None;started=time.perf_counter();new=0;consecutive_errors=0;invalid_streak=0
 final_status='paused';reason='session budget reached'
 try:
  monitor.event('loading_embedding_model',model=embedding['repo'])
  embedder=ResidentEmbedder(embedding)
  monitor.event('loading_classifier',model=metadata.get('repo',metadata.get('path')),remaining=len(pending))
  verifier=RobertaVerifier(metadata['path'],max_length) if kind=='roberta' else CausalVerifier(metadata,root)
  monitor.event('models_ready',setup_seconds=round(time.perf_counter()-started,2))
  for row in pending:
   if time.perf_counter()-started>=minutes*60:break
   if max_new_claims is not None and new>=max_new_claims:reason='requested per-session claim limit reached';break
   trace={'id':row['id'],'claim':row['claim'],'label':'INVALID','protocol_identity':fingerprint,'created_utc':now()};phase=time.perf_counter()
   monitor.event('claim_started',claim_id=row['id'],claim=row['claim'],position=len(saved)+new+1)
   try:
    context=context_for(row,root,out,embedder,embedding,progress=monitor.event)
    trace['evidence_identity']=signature(context);trace['evidence_status']=context['status']
    trace.update(verifier.predict(row,context,max_new_tokens=max_new_tokens,progress=monitor.event));consecutive_errors=0
   except torch.cuda.OutOfMemoryError:
    monitor.event('gpu_out_of_memory',exception='This claim remains pending; no prediction was fabricated.')
    raise RuntimeError('GPU out of memory. Free GPU memory; changing model/settings requires a new run.') from None
   except (ValueError,RuntimeError,KeyError,TypeError) as exc:
    trace['error']=type(exc).__name__+': '+str(exc)[:1000];consecutive_errors+=1
    monitor.event('claim_failed',failed_stage=monitor.state['stage'],exception=trace['error'])
   trace['wall_seconds']=time.perf_counter()-phase
   trace['timing_note']='Includes active monitoring overhead and preparation when newly built; cached-evidence timing is not end-to-end pipeline latency.'
   trace['prediction_path']=str(folder/(str(row['id'])+'.json'))
   write_once(folder/(str(row['id'])+'.json'),trace);new+=1;monitor.saved(trace)
   invalid_streak=invalid_streak+1 if trace['label']=='INVALID' else 0
   if consecutive_errors>=3:raise RuntimeError('Stopped after 3 consecutive infrastructure/preparation failures. Inspect the log; saved errors remain errors.')
   if invalid_streak>=max_consecutive_invalid:raise RuntimeError('Stopped after '+str(invalid_streak)+' consecutive INVALID outputs. Inspect prompts/responses before continuing; completed predictions are preserved.')
  if len(saved)+new==len(rows):final_status='complete';reason='all '+str(len(rows))+' predictions saved'
 except KeyboardInterrupt:
  final_status='interrupted';reason='user interrupted execution; current unsaved claim remains pending'
  monitor.event('interrupted',exception=reason);raise
 except Exception as exc:
  final_status='failed';reason=type(exc).__name__+': '+str(exc)
  monitor.event('run_failed',failed_stage=monitor.state['stage'],exception=reason);raise
 finally:
  try:
   monitor.event('releasing_models')
   if verifier is not None:verifier.close()
   embedder=None;release()
  finally:monitor.close(final_status,reason)
 print('Session saved',new,'new claims. Status:',final_status,'. Rerun unchanged settings to resume.',flush=True)

def score(root,out,names,require_full=True):
 import numpy as np
 from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
 root=Path(root);out=Path(out);inputs=read_rows(root/'data/eval_inputs.jsonl');gold=json.loads((root/'data/eval_labels.json').read_text());result={}
 if require_full and len(inputs)!=19998:raise ValueError('Final scoring requires the 19,998-claim official input file')
 for name in names:
  folder=out/'predictions'/name;saved={p.stem:json.loads(p.read_text()) for p in folder.glob('*.json') if p.name!='protocol.json'}
  expected={str(r['id']) for r in inputs}
  if set(saved)-expected:raise ValueError('Unexpected prediction IDs')
  if require_full and set(saved)!=expected:
   print(name,'INCOMPLETE:',len(saved),'/ 19998. No final accuracy produced.');continue
  order=[r for r in inputs if str(r['id']) in saved];y=[gold[str(r['id'])] for r in order];pred=[saved[str(r['id'])]['label'] for r in order]
  if not order:continue
  precision,recall,f1,_=precision_recall_fscore_support(y,pred,labels=LABELS,zero_division=0)
  matrix=confusion_matrix(y,pred,labels=LABELS+['INVALID'])[:3,:].tolist()
  result[name]={'status':'complete' if len(order)==19998 else 'partial_not_final','n':len(order),'accuracy':accuracy_score(y,pred),'macro_f1':float(np.mean(f1)),
   'invalid':pred.count('INVALID'),'prediction_columns':LABELS+['INVALID'],'confusion_matrix':matrix,
   'per_class':{label:{'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i])} for i,label in enumerate(LABELS)},
   'metric_note':'Label classification only. Full live-Wikipedia article citations do not yield an official strict FEVER score.'}
  print(name,json.dumps({k:result[name][k] for k in ['status','n','accuracy','macro_f1','invalid']}))
 if result:
  # Reports are derived files; uniquely named to preserve previous partial/complete reports.
  report=out/('metrics_'+str(time.time_ns())+'.json');write_once(report,result);print('Report:',report)
 return result

def diagnostic_report(root,out,names):
 """Score only saved predictions; error causes require trace review, not guessing."""
 import csv
 root=Path(root);out=Path(out);rows=verify_diagnostic(root)
 refs=json.loads((root/'data/references.json').read_text())
 results=score(root,out,names,require_full=False)
 review=[]
 for name,m in results.items():
  m['status']='development_diagnostic_complete' if m['n']==90 else 'development_diagnostic_incomplete'
  m['scope']='original validation diagnostic; never the final 19,998-claim result'
  for row in rows:
   path=out/'predictions'/name/(str(row['id'])+'.json')
   if not path.exists():continue
   trace=json.loads(path.read_text());context_path=out/'evidence'/('claim_'+str(row['id'])+'.json')
   context=json.loads(context_path.read_text()) if context_path.exists() else {}
   correct=trace['label']==refs[str(row['id'])]['label']
   category=('evidence_preparation_failure' if trace.get('evidence_status')!='READY' else
             'invalid_model_output' if trace['label']=='INVALID' else
             'correct_label_citations_need_review' if correct else 'evidence_or_reasoning_review_required')
   review.append({'condition':name,'id':row['id'],'claim':row['claim'],'reference':refs[str(row['id'])]['label'],
    'prediction':trace['label'],'correct':correct,'review_category':category,'error':trace.get('error',''),
    'annotated_reference_evidence':json.dumps(refs[str(row['id'])]['annotated_evidence']),
    'supplied_excerpt_count':len(context.get('supplied_evidence',{})),
    'prompt_response_file':str(path),'evidence_file':str(context_path),
    'reviewed_cause':'','review_notes':''})
 if review:
  path=out/('review_'+str(time.time_ns())+'.csv')
  with path.open('x',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(review[0]));writer.writeheader();writer.writerows(review)
  print('Review sheet:',path)
 if results:write_once(out/('diagnostic_metrics_'+str(time.time_ns())+'.json'),results)
 return results

def unfreeze_last_layers(model,n):
 if not 1<=n<=len(model.roberta.encoder.layer):raise ValueError('Invalid layer count')
 for p in model.parameters():p.requires_grad=False
 for p in model.classifier.parameters():p.requires_grad=True
 for block in model.roberta.encoder.layer[-n:]:
  for p in block.parameters():p.requires_grad=True
 return {'last_blocks':n,'classifier_head':True,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'total_parameters':sum(p.numel() for p in model.parameters())}

def finetune_roberta(root,out,last_layers=1,scope='pilot',minutes=30,checkpoint_steps=1000):
 """Continued fine-tuning from the identical existing checkpoint each time.

The included 96/96 data only verify training software. Full scope requires the
optional complete retrieved training export, never official-dev training.
"""
 import torch,numpy as np
 from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments,TrainerCallback,DataCollatorWithPadding,set_seed
 from transformers.trainer_utils import get_last_checkpoint
 from sklearn.metrics import f1_score
 check_cuda();set_seed(42);root=Path(root);out=Path(out)
 if not isinstance(checkpoint_steps,int) or checkpoint_steps<1:raise ValueError('Checkpoint interval must be a positive integer')
 if scope not in {'pilot','full'}:raise ValueError('Choose pilot or full')
 trainpath=root/('data/train_pilot.jsonl' if scope=='pilot' else 'data/train_full.jsonl')
 valpath=root/('data/validation_pilot.jsonl' if scope=='pilot' else 'data/validation_full.jsonl')
 if not trainpath.exists() or not valpath.exists():raise FileNotFoundError('Full training data are not in the starter bundle. See README and optional --include-training export. Do not substitute official dev.')
 train=read_rows(trainpath);val=read_rows(valpath);official=read_rows(root/'data/eval_inputs.jsonl');excluded={normalize(r['claim']) for r in official}
 # Small software pilots must also validate/save within their short run.
 checkpoint_steps=min(checkpoint_steps,max(1,math.ceil(len(train)/16)))
 trtext={normalize(r['claim']) for r in train};vtext={normalize(r['claim']) for r in val}
 if trtext&vtext or (trtext|vtext)&excluded:raise ValueError('Training/validation/evaluation text overlap')
 if scope=='full' and len(train)<=96:raise ValueError('Pilot data cannot be called full training')
 print(scope.upper(),'training:',len(train),'validation:',len(val),'Historical retrieved sentences, not new full-article RAG training contexts.')
 initial=root/'checkpoint/roberta_evidence';folder=out/'training'/('roberta_'+scope+'_last'+str(last_layers));folder.mkdir(parents=True,exist_ok=True)
 tokenizer=AutoTokenizer.from_pretrained(initial,local_files_only=True)
 model=AutoModelForSequenceClassification.from_pretrained(initial,local_files_only=True)
 trainable=unfreeze_last_layers(model,last_layers);print(trainable)
 protocol={'scope':scope,'train_n':len(train),'validation_n':len(val),'train_sha256':sha_file(trainpath),'validation_sha256':sha_file(valpath),
  'runtime_sha256':sha_file(root/'src/fever_colab_runtime.py'),'initial_checkpoint':checkpoint_identity(initial),'seed':42,'epochs':2,
  'learning_rate':2e-5,'max_length':512,'batch_size':4,'gradient_accumulation':4,'weight_decay':.01,
  'evaluation_and_checkpoint_strategy':'steps','checkpoint_steps':checkpoint_steps,'unfreezing':trainable,'environment':environment()}
 freeze(folder,protocol)
 if (folder/'complete.json').exists():print('Already trained:',folder/'final');return folder/'final'
 class ClaimDataset(torch.utils.data.Dataset):
  def __init__(self,rows):self.rows=rows
  def __len__(self):return len(self.rows)
  def __getitem__(self,i):
   r=self.rows[i];item=tokenizer(r['claim'],r['evidence_text'],truncation='only_second',max_length=512)
   item['labels']=LABELS.index(r['label']);return item
 monitor=RunMonitor(folder/'monitor','roberta_'+scope+'_last'+str(last_layers),total=0,show_progress=False,unit='optimizer step')
 monitor.show_progress=True  # Trainer supplies the training bar; monitor prints durable stage/loss events.
 class StopAtBudget(TrainerCallback):
  def __init__(self):self.started=time.perf_counter()
  def on_train_begin(self,args,state,control,**kwargs):
   monitor.event('training_started',completed=state.global_step,total=state.max_steps,train_rows=len(train),validation_rows=len(val),trainable_parameters=trainable['trainable_parameters'])
  def on_step_begin(self,args,state,control,**kwargs):
   if monitor.state['stage']!='training':monitor.event('training',completed=state.global_step,total=state.max_steps)
  def on_step_end(self,args,state,control,**kwargs):
   with monitor.lock:monitor.state.update(completed=state.global_step,total=state.max_steps,epoch=state.epoch)
   if state.global_step%10==0 or state.global_step==state.max_steps:monitor.event('training_step',completed=state.global_step,total=state.max_steps,epoch=state.epoch)
   if time.perf_counter()-self.started>minutes*60:
    monitor.event('training_budget_reached',completed=state.global_step)
    control.should_save=True;control.should_training_stop=True
   return control
  def on_log(self,args,state,control,logs=None,**kwargs):
   monitor.event('training_metrics',completed=state.global_step,metrics=logs or {})
  def on_prediction_step(self,args,state,control,**kwargs):
   if monitor.state['stage']!='validating':monitor.event('validating',completed=state.global_step)
  def on_evaluate(self,args,state,control,metrics=None,**kwargs):
   monitor.event('validation_complete',completed=state.global_step,metrics=metrics or {})
  def on_save(self,args,state,control,**kwargs):
   monitor.event('training_checkpoint_saved',completed=state.global_step,checkpoint=str(folder/'checkpoints'/('checkpoint-'+str(state.global_step))))
 args=TrainingArguments(output_dir=str(folder/'checkpoints'),num_train_epochs=2,learning_rate=2e-5,weight_decay=.01,
  per_device_train_batch_size=4,per_device_eval_batch_size=8,gradient_accumulation_steps=4,fp16=True,
  eval_strategy='steps',eval_steps=checkpoint_steps,save_strategy='steps',save_steps=checkpoint_steps,
  save_total_limit=2,logging_steps=10,seed=42,data_seed=42,report_to=[],
  load_best_model_at_end=True,metric_for_best_model='macro_f1',greater_is_better=True,
  gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},dataloader_num_workers=0)
 def compute(p):return {'macro_f1':f1_score(p.label_ids,p.predictions.argmax(-1),labels=[0,1,2],average='macro',zero_division=0)}
 trainer=Trainer(model=model,args=args,train_dataset=ClaimDataset(train),eval_dataset=ClaimDataset(val),processing_class=tokenizer,
  data_collator=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8),compute_metrics=compute,callbacks=[StopAtBudget()])
 previous=get_last_checkpoint(str(folder/'checkpoints')) if (folder/'checkpoints').exists() else None
 final=None;status='paused';reason='training session budget reached'
 try:
  trainer.train(resume_from_checkpoint=previous)
  if trainer.state.global_step>=trainer.state.max_steps:
   final=folder/'final';monitor.event('saving_final_model',path=str(final));trainer.save_model(final);tokenizer.save_pretrained(final)
   write_once(folder/'complete.json',{'scope':scope,'steps':trainer.state.global_step,'best_validation_checkpoint':trainer.state.best_model_checkpoint,'finished_utc':now()})
   status='complete';reason='all configured training steps completed';print('Training complete:',final,flush=True)
  else:print('Session budget reached; saved checkpoint. Rerun with unchanged settings to resume.',flush=True)
 except KeyboardInterrupt:
  status='interrupted';reason='user interruption; resume from latest completed training checkpoint';raise
 except Exception as exc:
  status='failed';reason=type(exc).__name__+': '+str(exc);monitor.event('training_failed',exception=reason);raise
 finally:
  monitor.close(status,reason);del trainer,model;release()
 return final
