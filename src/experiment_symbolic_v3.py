"""Freeze a fresh matched RQ3 comparison; no retraining or LLM calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import time

from experiment_parts_test import LABELS,metrics,write_new
from symbolic_kb_v3 import FactStore,verify,plan

ROOT=Path(__file__).resolve().parents[1]
BASE_CHECKPOINTS={'claim_only':'checkpoints/tuning_weighted/H2_seed42/best',
                  'retrieval_roberta':'checkpoints/roberta_evidence/epoch_1'}


def load(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key(text):return ' '.join(text.casefold().split())


def prepare(out,count=100):
    if out.exists():raise FileExistsError(out)
    eligible=ROOT/'output/plan_parts_test_20261030/eligible_claims_without_labels.jsonl'
    prior=[];excluded=set()
    for path in (ROOT/'reports').rglob('*.jsonl'):
        if path.name in {'inputs.jsonl','sample.jsonl','claims_without_labels.jsonl','results.jsonl','outputs.jsonl'} and 'source_snapshots' not in path.parts:
            prior.append(path)
            for r in rows(path):
                if isinstance(r.get('claim'),str):excluded.add(key(r['claim']))
    pool=[r for r in rows(eligible) if key(r['claim']) not in excluded]
    rng=random.Random(87420261030);chosen=rng.sample(pool,count)
    for r in chosen:
        r['condition_order']=['claim_only','retrieval_roberta','symbolic_first'];rng.shuffle(r['condition_order'])
    # Gold is used to create a separate scoring file, not to select claims.
    reference={r['id']:r['label'] for r in rows(ROOT/'data/processed/test.jsonl')}
    out.mkdir(parents=True);(out/'traces').mkdir()
    with (out/'inputs.jsonl').open('x') as stream:stream.writelines(json.dumps(r)+'\n' for r in chosen)
    write_new(out/'reference_labels.json',{str(r['id']):reference[r['id']] for r in chosen})
    files=[Path(__file__),ROOT/'src/symbolic_kb_v3.py',ROOT/'src/experiment_parts_test.py',ROOT/'src/retrieve_title_aware.py',eligible,
           out/'inputs.jsonl',out/'reference_labels.json',ROOT/'data/processed/test.jsonl',*prior]
    for checkpoint in BASE_CHECKPOINTS.values():files.extend(p for p in (ROOT/checkpoint).iterdir() if p.is_file())
    stat=(ROOT/'data/processed/wiki_fts.db').stat()
    protocol=dict(status='frozen_before_inference',n=count,seed=87420261030,
                  sampling='Uniform random unique claims from remaining eligible internal test population, excluding all located prior inputs/outputs. Labels not used for selection.',
                  ids=[r['id'] for r in chosen],parser_eligible_count=sum(plan(r['claim']) is not None for r in chosen),
                  eligible_pool=len(pool),conditions=['claim_only','retrieval_roberta','symbolic_first'],
                  max_length=256,batch_size=1,device='cpu',torch_threads=4,checkpoints=BASE_CHECKPOINTS,
                  primary='symbolic_first versus retrieval_roberta; same fallback checkpoint, retriever, token budget and claims.',
                  secondary='Compare both with claim-only H2; this uses a different trained checkpoint and is not an isolated symbolic effect.',
                  hypothesis='The repaired symbolic stage improves accuracy or macro-F1 over the identical retrieval-RoBERTa fallback.',
                  knowledge='Run-local SQLite fact table and immutable raw Wikidata responses, initially empty; facts/qualifiers/references/revisions retained.',
                  rules='Whole-claim birth/death dates (year, decade, month, month/year, full dates), exact birth/death locations, occupations and membership. Date contradictions only for adequately precise unique Gregorian facts; otherwise abstain. No arbitrary claim-to-triple LLM extraction.',
                  timing='Actual calls in all three conditions, shuffled condition order per claim. Model loading excluded. Live KB networking counted; one run with shared OS caches.',
                  sha256={str(p.resolve()):digest(p) for p in files},index=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns),
                  limitations=['Small single-run study; many FEVER claims lie outside the declared grammar.',
                               'Wikidata can contain errors, missing facts, ambiguity and temporal differences from FEVER.',
                               'Original fallback training mismatch remains. No graph model or broad semantic claim parser is reproduced.',
                               'Development repairs were checked on earlier inspected examples and synthetic controls; this new sample must not be used for retuning.',
                               'Live network latency and shared caches limit efficiency conclusions. API failures are separately recorded fallbacks.'])
    write_new(out/'protocol.json',protocol)
    print('Prepared',count,'fresh paired claims;',protocol['parser_eligible_count'],'fit the declared symbolic grammar. No inference yet.',flush=True)


def check(out):
    protocol=load(out/'protocol.json')
    for filename,expected in protocol['sha256'].items():
        if digest(filename)!=expected:raise ValueError('Frozen input changed: '+filename)
    stat=(ROOT/'data/processed/wiki_fts.db').stat()
    if (stat.st_size,stat.st_mtime_ns)!=(protocol['index']['bytes'],protocol['index']['mtime_ns']):raise ValueError('Retrieval index changed')
    return protocol


def run(out):
    protocol=check(out)
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from retrieve_title_aware import retrieve
    from fetch_wikipedia_articles import atomic_json
    torch.set_num_threads(4);torch.manual_seed(42)
    models={}
    for name,checkpoint in BASE_CHECKPOINTS.items():
        models[name]=(AutoTokenizer.from_pretrained(ROOT/checkpoint,local_files_only=True),
                      AutoModelForSequenceClassification.from_pretrained(ROOT/checkpoint,local_files_only=True).cpu().eval())
    store=FactStore(out/'knowledge_base');conn=sqlite3.connect((ROOT/'data/processed/wiki_fts.db').as_uri()+'?mode=ro',uri=True)
    def classify(claim,evidence,which):
        tokenizer,model=models[which]
        encoded=tokenizer(claim,text_pair=evidence,padding='max_length',truncation=True,max_length=256,return_tensors='pt')
        with torch.inference_mode():prob=torch.softmax(model(**encoded).logits[0],-1).tolist()
        return LABELS[max(range(3),key=prob.__getitem__)],prob
    try:
        data=rows(out/'inputs.jsonl')
        for i,row in enumerate(data,1):
            target=out/'traces'/(str(row['id'])+'.json')
            if target.exists():continue
            result=dict(id=row['id'],claim=row['claim'],conditions={})
            for condition in row['condition_order']:
                started=time.perf_counter();decision=None;symbolic=None;evidence=[];prob=None
                if condition=='symbolic_first':symbolic=verify(row['claim'],store);decision=symbolic['label']
                if condition=='claim_only':decision,prob=classify(row['claim'],None,'claim_only')
                elif decision is None:
                    evidence=retrieve(conn,row['claim'],5)
                    decision,prob=classify(row['claim'],' '.join(r['text'] for r in evidence),'retrieval_roberta')
                result['conditions'][condition]=dict(label=decision,wall_seconds=time.perf_counter()-started,
                    symbolic=symbolic,evidence=evidence,probabilities=prob,model_called=symbolic is None or symbolic['label'] is None,
                    retrieval_called=condition!='claim_only' and (symbolic is None or symbolic['label'] is None))
            write_new(target,result)
            atomic_json(out/'progress.json',dict(status='running',completed=i,total=len(data),latest_id=row['id']))
            print('SAVED',i,'/',len(data),row['id'],flush=True)
        check(out)
        atomic_json(out/'progress.json',dict(status='complete_unscored',completed=len(data),total=len(data)))
    finally:store.close();conn.close()


def score(out):
    protocol=check(out);inputs=rows(out/'inputs.jsonl');gold_map=load(out/'reference_labels.json')
    traces=[load(out/'traces'/(str(r['id'])+'.json')) for r in inputs]
    gold=[gold_map[str(r['id'])] for r in inputs];results={}
    for c in protocol['conditions']:
        results[c]=dict(**metrics(gold,[r['conditions'][c]['label'] for r in traces]),
                        mean_seconds=sum(r['conditions'][c]['wall_seconds'] for r in traces)/len(traces),
                        model_calls=sum(r['conditions'][c]['model_called'] for r in traces))
    symbolic=[r['conditions']['symbolic_first']['symbolic'] for r in traces]
    resolved=[(r,g) for r,g in zip(symbolic,gold) if r['label'] is not None]
    fixed=sum(r['conditions']['symbolic_first']['label']==g and r['conditions']['retrieval_roberta']['label']!=g for r,g in zip(traces,gold))
    introduced=sum(r['conditions']['symbolic_first']['label']!=g and r['conditions']['retrieval_roberta']['label']==g for r,g in zip(traces,gold))
    summary=dict(status='complete',n=len(traces),conditions=results,symbolic_resolved=len(resolved),
                 symbolic_correct=sum(r['label']==g for r,g in resolved),fixed=fixed,introduced=introduced,
                 reason_counts=dict(Counter(r['reason'] for r in symbolic)),
                 http_calls=sum(r['http_calls'] for r in symbolic),
                 service_failures=sum(r['reason']=='service_failure_fallback' for r in symbolic),
                 limitations=protocol['limitations'])
    write_new(out/'summary.json',summary)
    lines=['# Repaired dynamic symbolic stage: matched RQ3 study','','New run; no old predictions or labels overwritten. Same claims, fixed models, randomized condition order.','',
           '| Condition | Accuracy | Macro-F1 | Mean seconds | Model calls |','|---|---:|---:|---:|---:|']
    for c,m in results.items():lines.append('| %s | %.2f%% | %.4f | %.3f | %d |'%(c,m['accuracy']*100,m['macro_f1'],m['mean_seconds'],m['model_calls']))
    lines+=['',f"Symbolic decisions: {len(resolved)}/{len(traces)}; correct against FEVER: {summary['symbolic_correct']}. Baseline errors fixed: {fixed}; new errors: {introduced}.",
            '',f"Claims matching the declared grammar: {protocol['parser_eligible_count']}/{len(traces)}. Operational service failures: {summary['service_failures']}. These are not the same as absent knowledge.",'',
            'Wikidata facts and complete statements, including qualifiers and references, are stored in knowledge_base/knowledge.sqlite. Raw responses and fetch times are retained in knowledge_base/snapshots. Every trace records the parsed relationship, subject resolution, compared facts and fallback.','',
            'This is a limited-grammar symbolic verifier, not a general fact checker. Coverage and correctness must both be reported. No claim of general knowledge-graph superiority follows from this small experiment.','',
            *['- '+x for x in protocol['limitations']],'']
    (out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','score']);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=100)
    a=p.parse_args();{'prepare':prepare,'run':run,'score':score}[a.mode](a.output.resolve(),**({'count':a.count} if a.mode=='prepare' else {}))


if __name__=='__main__':main()
