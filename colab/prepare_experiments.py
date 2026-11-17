"""CPU-only, resumable preparation. Never trains or downloads models.

Keep the original training/validation partitions: the uploaded checkpoint has
already seen original training examples, so reshuffling them into validation
would make checkpoint selection misleading.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,random,sqlite3,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from retrieve_title_aware import retrieve,find_titles
from fetch_wikipedia_articles import page_url
LABELS=['SUPPORTS','REFUTES','NOT ENOUGH INFO']
def norm(s):return ' '.join(s.casefold().split())
def rows(path):return [json.loads(line) for line in path.open() if line.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def put(path,value):
 temp=path.with_suffix('.pending');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def partitions():
 official={norm(r['claim']) for r in rows(ROOT/'data/official_fever/shared_task_dev.jsonl')}
 train=[r for r in rows(ROOT/'data/processed/train.jsonl') if norm(r['claim']) not in official]
 traintext={norm(r['claim']) for r in train}
 val=[r for r in rows(ROOT/'data/processed/validation.jsonl') if norm(r['claim']) not in official|traintext]
 return train,val,official
def initialize():
 global DB
 DB=sqlite3.connect((ROOT/'data/processed/wiki_fts.db').as_uri()+'?mode=ro',uri=True)
def training_row(r):
 ev=retrieve(DB,r['claim'],5)
 return {'id':r['id'],'claim':r['claim'],'label':r['label'],
         'evidence_text':' '.join(x['text'] for x in ev),'evidence':ev}
def prepare(out,full=False,workers=2):
 out.mkdir(parents=True,exist_ok=True);data=out/'data';data.mkdir(exist_ok=True)
 train,val,official=partitions()
 proto={'purpose':'development diagnosis and original-split continued RoBERTa training',
        'seed':42,'train_n':len(train),'validation_n':len(val),'official_evaluation_n':19998,
        'sources':{n:sha(ROOT/n) for n in ['data/processed/train.jsonl','data/processed/validation.jsonl',
                    'data/official_fever/shared_task_dev.jsonl','src/retrieve_title_aware.py','colab/prepare_experiments.py']},
        'wiki_database':{'bytes':(ROOT/'data/processed/wiki_fts.db').stat().st_size,
                         'mtime_ns':(ROOT/'data/processed/wiki_fts.db').stat().st_mtime_ns},
        'training_evidence':'historical retrieved top-five sentences; not live full-article RAG; labels retained even when retrieval misses',
        'selection_note':'Official dev and training-text overlaps excluded from validation. Original internal test remains unused.',
        'full_training_requested':full}
 path=out/'preparation_protocol.json'
 if path.exists() and json.loads(path.read_text())!=proto:raise ValueError('Preparation inputs changed; use a new output folder')
 put(path,proto)
 # Diagnosis uses original validation only, excluding the included 96/96 pilots.
 excluded=set()
 for name in ['train_pilot','validation_pilot']:
  excluded|={norm(r['claim']) for r in rows(ROOT/('output/colab_20261104/data/'+name+'.jsonl'))}
 rng=random.Random(42);selected=[];used=set(excluded)
 for label in LABELS:
  pool=[r for r in val if r['label']==label];rng.shuffle(pool);chosen=[]
  for r in pool:
   if norm(r['claim']) in used:continue
   used.add(norm(r['claim']));chosen.append(r)
   if len(chosen)==30:break
  if len(chosen)!=30:raise ValueError('Insufficient distinct diagnostic examples')
  selected.extend(chosen)
 rng.shuffle(selected)
 initialize();features=[];references={}
 for r in selected:
  titles=find_titles(DB,r['claim'],10)
  features.append({'id':r['id'],'claim':r['claim'],'ranked_articles':[
      {'rank':i+1,'page_id':p,'url':page_url(p)} for i,p in enumerate(titles)]})
  references[str(r['id'])]={'label':r['label'],'annotated_evidence':r['evidence']}
 DB.close()
 files={'eval_inputs.jsonl':''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in features),
        'eval_labels.json':json.dumps({k:v['label'] for k,v in references.items()}),
        'references.json':json.dumps(references,ensure_ascii=False)}
 for name,content in files.items():
  p=data/name
  if p.exists() and p.read_text()!=content:raise ValueError('Existing diagnostic file differs')
  p.write_text(content)
 diagnostic={'scope':'90-claim development diagnostic, not final evaluation',
             'class_counts':dict(Counter(v['label'] for v in references.values())),
             'selection':'seed 42; original validation; excludes normalized official-dev/training/pilot overlaps',
             'files':{name:sha(data/name) for name in files}}
 put(out/'diagnostic.json',diagnostic)
 print('Diagnostic prepared:',len(features),'claims. Training:',len(train),'validation:',len(val),flush=True)
 if not full:return
 started=time.monotonic()
 with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
  for name,records in [('train_full',train),('validation_full',val)]:
   p=data/(name+'.jsonl');saved=rows(p) if p.exists() else []
   if [(r['id'],r['claim']) for r in saved]!=[(r['id'],r['claim']) for r in records[:len(saved)]]:raise ValueError('Resume prefix mismatch')
   counts=Counter();empty=0
   for r in saved:counts[r['label']]+=1;empty+=not bool(r['evidence'])
   with p.open('a') as stream:
    for index,r in enumerate(pool.map(training_row,records[len(saved):],chunksize=32),len(saved)+1):
     stream.write(json.dumps(r,ensure_ascii=False)+'\n');counts[r['label']]+=1;empty+=not bool(r['evidence'])
     if index%250==0 or index==len(records):
      stream.flush();elapsed=time.monotonic()-started
      status={'status':'preparing','split':name,'completed':index,'total':len(records),
              'class_counts':dict(counts),'empty_retrieval':empty,'elapsed_seconds':round(elapsed,1)}
      put(out/'preparation_status.json',status);print(json.dumps(status),flush=True)
 put(out/'training_ready.json',{'train_n':len(train),'validation_n':len(val),
     'files':{name+'.jsonl':sha(data/(name+'.jsonl')) for name in ['train_full','validation_full']},
     'scope':'Full original training/validation after overlap exclusion; not the 96/96 pilot'})
 archive=out/'FEVER_Training_Data.zip'
 if not archive.exists():
  with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
   for name in ['train_full','validation_full']:z.write(data/(name+'.jsonl'),'data/'+name+'.jsonl')
   z.write(out/'training_ready.json','training_ready.json');z.write(path,'preparation_protocol.json')
 put(out/'preparation_status.json',{'status':'complete','train_n':len(train),'validation_n':len(val),
     'archive':str(archive),'elapsed_seconds':round(time.monotonic()-started,1)})
 print('Training data ready:',archive,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--full',action='store_true');parser.add_argument('--workers',type=int,default=2)
 args=parser.parse_args();prepare(args.output,args.full,args.workers)
