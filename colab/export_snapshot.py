"""Package full FEVER snapshot records selected by existing claim-only retrieval.

No labels, gold evidence, network calls, inference or training are used here.
The package contains the union of the existing top-five page IDs for all 19,998
claims. Rankings and the claim order are not changed.
"""
import argparse,hashlib,json,sqlite3,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(2**20),b''):h.update(block)
 return h.hexdigest()
def export(inputs,wiki,out):
 out.mkdir(parents=True,exist_ok=False)
 rows=[json.loads(line) for line in inputs.open() if line.strip()]
 if len(rows)!=19998 or len({r['id'] for r in rows})!=19998:raise ValueError('Expected the full official input file')
 if any(set(r)!={'id','claim','ranked_articles'} for r in rows):raise ValueError('Unexpected fields in retrieval-only inputs')
 selected={p['page_id'] for r in rows for p in r['ranked_articles'][:5]}
 database=out/'snapshot.sqlite';conn=sqlite3.connect(database)
 conn.execute('CREATE TABLE articles(page_id TEXT PRIMARY KEY,text TEXT NOT NULL,lines TEXT NOT NULL,source_file TEXT NOT NULL,source_line INTEGER NOT NULL,record_sha256 TEXT NOT NULL)')
 found=set();sources=[];scanned=0;started=time.monotonic();files=sorted(wiki.glob('*.jsonl'))
 try:
  for index,path in enumerate(files,1):
   hasher=hashlib.sha256()
   with path.open('rb') as stream:
    for line_no,line in enumerate(stream,1):
     hasher.update(line);scanned+=1;record=json.loads(line)
     if record['id'] not in selected:continue
     if record['id'] in found:raise ValueError('Duplicate selected snapshot page: '+record['id'])
     if set(record)!={'id','text','lines'}:raise ValueError('Unexpected raw snapshot schema')
     conn.execute('INSERT INTO articles VALUES(?,?,?,?,?,?)',(record['id'],record['text'],record['lines'],path.name,line_no,hashlib.sha256(line).hexdigest()))
     found.add(record['id'])
   sources.append({'file':path.name,'sha256':hasher.hexdigest(),'bytes':path.stat().st_size});conn.commit()
   if index%10==0 or index==len(files):
    print('Snapshot files',index,'/',len(files),'; pages found',len(found),'/',len(selected),'; seconds',round(time.monotonic()-started,1),flush=True)
  missing=sorted(selected-found)
  if missing:raise ValueError('Selected pages missing from snapshot: '+repr(missing[:20]))
  nonempty=conn.execute("SELECT count(*) FROM articles WHERE length(trim(text))>0 OR length(trim(lines))>0").fetchone()[0]
  manifest={'source':'FEVER distributed Wikipedia snapshot; historical text, not current live Wikipedia',
   'schema_version':1,'input_sha256':sha_file(inputs),'claims':len(rows),'selected_page_ids':len(selected),
   'packaged_pages':len(found),'nonempty_pages':nonempty,'missing_pages':missing,
   'selection':'Union of the first five pages from existing top-ten claim-only rankings. No labels/gold evidence used.',
   'retention':'Static source dataset remains; per-claim vector databases and working article copies are temporary.',
   'revision_note':'FEVER records have no per-page revision IDs. Do not invent oldid URLs.',
   'sources':sources,'exporter_sha256':sha_file(__file__)}
 finally:conn.close()
 manifest['snapshot_sha256']=sha_file(database)
 (out/'snapshot_manifest.json').write_text(json.dumps(manifest,indent=2))
 archive=out/'FEVER_Wikipedia_Snapshot.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as zipped:
  zipped.write(database,'snapshot.sqlite');zipped.write(out/'snapshot_manifest.json','snapshot_manifest.json')
 print('READY:',archive,'MiB',round(archive.stat().st_size/2**20,2),'elapsed seconds',round(time.monotonic()-started,1),flush=True)
 return manifest
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--inputs',type=Path,default=ROOT/'output/colab_20261104/data/eval_inputs.jsonl')
 parser.add_argument('--wiki',type=Path,default=ROOT/'data/raw/wiki-pages')
 parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 export(args.inputs,args.wiki,args.output)
