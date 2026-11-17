"""Export Colab inputs without training, inference, downloads or old-result edits."""
import argparse
from collections import Counter
import hashlib,json,sqlite3,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from retrieve_title_aware import find_titles,retrieve
from fetch_wikipedia_articles import page_url
LABELS=['SUPPORTS','REFUTES','NOT ENOUGH INFO']
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def norm(s):return ' '.join(s.casefold().split())
def export(out,include_training=False):
 out.mkdir(parents=True,exist_ok=True);data=out/'data';data.mkdir(exist_ok=True)
 gold=load(ROOT/'data/official_fever/shared_task_dev.jsonl')
 assert len(gold)==19998 and Counter(r['label'] for r in gold)==Counter({l:6666 for l in LABELS})
 protocol={'official_dev_sha256':digest(ROOT/'data/official_fever/shared_task_dev.jsonl'),'retriever_sha256':digest(ROOT/'src/retrieve_title_aware.py'),'exporter_sha256':digest(__file__),'evaluation_n':19998}
 frozen=out/'export_protocol.json'
 if frozen.exists() and json.loads(frozen.read_text())!=protocol:raise ValueError('Export sources changed; choose a new output folder')
 frozen.write_text(json.dumps(protocol,indent=2))
 refs={str(r['id']):r['label'] for r in gold}
 (data/'eval_labels.json').write_text(json.dumps(refs))
 p=data/'eval_inputs.jsonl';saved=load(p) if p.exists() else []
 assert [r['id'] for r in saved]==[r['id'] for r in gold[:len(saved)]]
 c=sqlite3.connect((ROOT/'data/processed/wiki_fts.db').as_uri()+'?mode=ro',uri=True)
 started=time.monotonic()
 with p.open('a') as f:
  for i,r in enumerate(gold[len(saved):],len(saved)):
   pages=find_titles(c,r['claim'],10)
   row={'id':r['id'],'claim':r['claim'],'ranked_articles':[{'rank':j+1,'page_id':x,'url':page_url(x)} for j,x in enumerate(pages)]}
   f.write(json.dumps(row,ensure_ascii=False)+'\n')
   if (i+1)%500==0:
    f.flush();print('Ranked article inputs',i+1,'/ 19998; seconds',round(time.monotonic()-started),flush=True)
 # Engineering-only examples use the original training pool, never official dev.
 excluded={norm(r['claim']) for r in gold};used=set(excluded)
 for split,source in [('train_pilot','data/processed/train.jsonl'),('validation_pilot','data/processed/validation.jsonl')]:
  target=data/(split+'.jsonl')
  if target.exists():continue
  selected=[];counts=Counter()
  with (ROOT/source).open() as stream:
   for line in stream:
    r=json.loads(line);key=norm(r['claim']);label=LABELS.index(r['label'])
    if key in used or counts[label]>=32:continue
    ev=retrieve(c,r['claim'],5)
    if not ev:continue
    selected.append({'id':r['id'],'claim':r['claim'],'label':r['label'],'evidence_text':' '.join(x['text'] for x in ev),'evidence':ev,'purpose':'training_software_pilot_not_final_training'})
    used.add(key);counts[label]+=1
    if len(selected)==96:break
  assert len(selected)==96
  target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
  print('Prepared',split,len(selected),flush=True)
 if include_training:
  # Optional, CPU retrieval only. This can take hours locally; never runs by default.
  # Validation is a fixed normalized-claim group split of the official training pool.
  handles={k:(data/(k+'.jsonl')).open('x') for k in ['train_full','validation_full']}
  try:
   for i,r in enumerate(load(ROOT/'data/raw/train.jsonl')):
    key=norm(r['claim'])
    if key in excluded:continue
    split='validation_full' if int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%10==0 else 'train_full'
    ev=retrieve(c,r['claim'],5)
    handles[split].write(json.dumps({'id':r['id'],'claim':r['claim'],'label':r['label'],'evidence_text':' '.join(x['text'] for x in ev),'evidence':ev},ensure_ascii=False)+'\n')
    if (i+1)%1000==0:print('Full-training retrieval prepared',i+1,flush=True)
  finally:
   for f in handles.values():f.close()
 c.close()
 print('Input preparation complete.',flush=True)
def package(out):
 archive=out/'FEVER_Colab_Bundle.zip'
 if archive.exists():raise FileExistsError('Preserve existing bundle; choose a new output directory')
 files=[]
 for p in sorted((out/'data').glob('*')):files.append((p,'data/'+p.name))
 for p in [ROOT/'colab/fever_colab_runtime.py',ROOT/'src/fetch_wikipedia_articles.py',ROOT/'src/ephemeral_rag.py']:
  files.append((p,'src/'+p.name))
 # Exact selected essential-parts prompt, without importing model-dependent code.
 import ast
 tree=ast.parse((ROOT/'src/replay_claim_parts.py').read_text())
 prompt=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='INSTRUCTION' for t in n.targets))
 prompt_path=out/'parts_prompt.txt';prompt_path.write_text(prompt)
 files.append((prompt_path,'parts_prompt.txt'))
 checkpoint=ROOT/'checkpoints/roberta_evidence/epoch_1'
 for p in sorted(checkpoint.iterdir()):
  if p.is_file():files.append((p,'checkpoint/roberta_evidence/'+p.name))
 manifest={'evaluation':'All 19,998 official FEVER development claims, original order. Previously inspected benchmark, not a new untouched test.',
 'labels':LABELS,'retrieval':'Existing title FTS/BM25 selects top 10; first 5 used by full-article RAG. Labels never used for article selection.',
 'pilot_training':'96 training + 96 validation examples, historical retrieved five-sentence text for all three labels. These are software checks, not final fine-tuning.',
 'full_training_included':(out/'data/train_full.jsonl').exists(),'files':{dest:digest(src) for src,dest in files}}
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for src,dest in files:z.write(src,dest)
  z.writestr('manifest.json',json.dumps(manifest,indent=2))
 print('BUNDLE',archive,'MiB',round(archive.stat().st_size/2**20,1),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--include-training',action='store_true');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 export(a.output,a.include_training)
 if not a.prepare_only:package(a.output)
