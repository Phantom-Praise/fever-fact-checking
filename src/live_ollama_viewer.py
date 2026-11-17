"""Incremental read-only index of a running full-dev experiment. Never calls Ollama."""
from functools import lru_cache
import json
from pathlib import Path
import threading
from ollama_trace_viewer import source_function

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'reports/part2/ollama_full_dev_20261028'
LABELS=['SUPPORTS','REFUTES','NOT ENOUGH INFO']
CONDITIONS=('A_original','D_both')


def read_json(path):return json.loads(path.read_text())


def complete_evidence(gold,evidence):
    if gold['label']==LABELS[2]:return None
    selected={(e['page_id'],e['sentence_id']) for e in evidence}
    groups=[{(e[2],e[3]) for e in group if e[2] is not None and e[3] is not None} for group in gold['evidence']]
    return any(group and group<=selected for group in groups)


class LiveStore:
    def __init__(self,folder=RUN):
        self.folder=Path(folder);self.protocol=read_json(self.folder/'protocol.json')
        self.inputs={};self.order=[];self.lock=threading.RLock()
        with (self.folder/'inputs.jsonl').open('rb') as stream:
            while True:
                offset=stream.tell();line=stream.readline()
                if not line:break
                r=json.loads(line);ident=str(r['id'])
                if ident in self.inputs:raise ValueError('Duplicate input ID')
                self.order.append(ident)
                self.inputs[ident]=dict(id=r['id'],claim=r['claim'],gold=r['gold']['label'],offset=offset,
                    quality={k:complete_evidence(r['gold'],v) for k,v in r['evidence'].items()})
        if len(self.inputs)!=self.protocol['n']:raise ValueError('Incomplete frozen inputs')
        self.records={c:{} for c in CONDITIONS};self.offsets={c:0 for c in CONDITIONS}

    def sync(self,config):
        if config not in CONDITIONS:raise ValueError('Unknown condition')
        path=self.folder/(config+'_responses.jsonl')
        if not path.exists():return
        if path.stat().st_size<self.offsets[config]:raise ValueError('Response file shrank; restart viewer after inspection')
        with path.open('rb') as stream:
            stream.seek(self.offsets[config])
            while True:
                offset=stream.tell();line=stream.readline()
                if not line or not line.endswith(b'\n'):break # Retry an in-flight append next refresh.
                r=json.loads(line);ident=str(r['id']);index=len(self.records[config])
                if index>=len(self.order) or ident!=self.order[index] or r.get('config')!=config:raise ValueError('Response prefix/configuration mismatch')
                prediction=r['label']
                if prediction not in LABELS+['INVALID']:raise ValueError('Unknown saved label')
                self.records[config][ident]=dict(offset=offset,prediction=prediction,correct=prediction==self.inputs[ident]['gold'],
                    error=r.get('error'),wall_seconds=r['wall_seconds'])
                self.offsets[config]=stream.tell()

    def status(self):
        with self.lock:
            conditions={}
            for config in CONDITIONS:
                self.sync(config);records=self.records[config];matrix=[[0]*4 for _ in LABELS];strict=0;counts={label:0 for label in LABELS}
                retrieval='original' if config=='A_original' else 'entity'
                for ident,r in records.items():
                    source=self.inputs[ident];i=LABELS.index(source['gold']);j=LABELS.index(r['prediction']) if r['prediction'] in LABELS else 3
                    matrix[i][j]+=1;counts[source['gold']]+=1
                    strict+=bool(r['correct'] and (source['gold']==LABELS[2] or source['quality'][retrieval]))
                n=len(records);correct=sum(r['correct'] for r in records.values())
                conditions[config]=dict(completed=n,total=len(self.inputs),correct=correct,wrong=n-correct,
                    invalid=sum(r['prediction']=='INVALID' for r in records.values()),matrix=matrix,gold_counts=counts,
                    accuracy=correct/n if n else None,fever_score=strict/n if n else None,latest_id=int(next(reversed(records))) if records else None,
                    final=n==len(self.inputs) and (self.folder/(config+'_summary.json')).exists())
            return dict(conditions=conditions,progress=read_json(self.folder/'progress.json'),
                batch=read_json(self.folder/'batch_status.json') if (self.folder/'batch_status.json').exists() else {},
                model=self.protocol['model'],note='Updates after each completed saved request. Partial official-order metrics are not final benchmark results.')

    def listing(self,config='A_original',query='',outcome='all',gold='all',prediction='all',show='completed',order='latest',page=1):
        if config not in CONDITIONS or outcome not in ('all','correct','wrong','invalid') or gold not in ['all']+LABELS or prediction not in ['all']+LABELS+['INVALID'] or show not in ('completed','pending','all') or order not in ('latest','official') or not 1<=page<=10000 or len(query)>1000:
            raise ValueError('Invalid live trace filter')
        with self.lock:
            self.sync(config);records=self.records[config];q=query.casefold().strip();items=[]
            ids=list(reversed(records)) if order=='latest' and show=='completed' else (reversed(self.order) if order=='latest' else self.order)
            for ident in ids:
                source=self.inputs[ident];r=records.get(ident)
                if (show=='completed' and r is None) or (show=='pending' and r is not None):continue
                if q and q not in source['claim'].casefold() and q!=ident:continue
                if gold!='all' and source['gold']!=gold:continue
                if prediction!='all' and (r is None or r['prediction']!=prediction):continue
                if outcome!='all' and (r is None or (outcome=='invalid' and r['prediction']!='INVALID') or (outcome=='correct' and not r['correct']) or (outcome=='wrong' and r['correct'])):continue
                items.append(dict(id=source['id'],claim=source['claim'],gold=source['gold'],prediction=r['prediction'] if r else None,
                    correct=r['correct'] if r else None,error=r['error'] if r else None))
            return dict(total=len(items),page=page,page_size=20,items=items[(page-1)*20:page*20])

    def detail(self,ident,config='A_original'):
        if config not in CONDITIONS:raise ValueError('Unknown condition')
        ident=str(ident)
        with self.lock:
            self.sync(config);source=self.inputs[ident];saved=self.records[config].get(ident)
            with (self.folder/'inputs.jsonl').open('rb') as stream:stream.seek(source['offset']);row=json.loads(stream.readline())
            result=None
            if saved:
                with (self.folder/(config+'_responses.jsonl')).open('rb') as stream:stream.seek(saved['offset']);result=json.loads(stream.readline())
            retrieval='original' if config=='A_original' else 'entity'
            hybrid=None
            if config=='D_both' and result:
                p=row['claim_probabilities'];best=max(range(3),key=lambda i:p[i]);override=result['label']==LABELS[2] and best!=2 and p[best]>=.8
                hybrid=dict(raw_llm_label=result['label'],final_label=LABELS[best] if override else result['label'],override=override,
                    claim_probabilities=p,rule='Override raw LLM NEI only when claim-only SUPPORTS/REFUTES softmax ≥ 0.80. These scores do not measure truth confidence.')
            return dict(id=row['id'],claim=row['claim'],gold=row['gold']['label'],gold_annotations=row['gold']['evidence'],config=config,
                evidence=row['evidence'][retrieval],evidence_quality=source['quality'][retrieval],result=result,hybrid=hybrid,
                baselines={k:LABELS[v] for k,v in row['baseline_predictions'].items()},model=self.protocol['model'],
                provenance='Saved artifacts only; no new model request. Gold labels and annotations are displayed for evaluation and were not sent to Llama.',
                code=source_function(self.folder/'source_local_llm_experiments.py','call','source_local_llm_experiments.py (frozen snapshot)')['code'],
                source_path='source_local_llm_experiments.py (frozen snapshot)')


@lru_cache(maxsize=1)
def live_store():return LiveStore()
