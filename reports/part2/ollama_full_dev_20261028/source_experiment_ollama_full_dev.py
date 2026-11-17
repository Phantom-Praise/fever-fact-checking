"""Prepare/resume full official-dev Llama evaluation. No training or model pulls.

Preparation reads the local Wikipedia snapshot. Run makes sequential local-only
Ollama requests and preserves each raw response. Never overwrites earlier runs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import time

from experiment_llm_configurations import CONFIGS, make_prompt, decision, score as score_config
from experiment_ollama_fever import LABELS, SCHEMA, validate_response
from experiment_ollama_fever import score as score_original
from experiment_symbolic_v2 import digest
from experiment_token_budget import comparison, write_json
from local_llm_experiments import call, connect
from retrieve_entity_anchored import retrieve_entity_anchored

ROOT=Path(__file__).resolve().parents[1]
CONDITIONS=('A_original','D_both')


def now():return datetime.now(timezone.utc).isoformat()


def load(path):
    with Path(path).open() as stream:return [json.loads(line) for line in stream if line.strip()]


def validate_pairing(gold,saved,long):
    ids=[r['id'] for r in gold]
    if len(gold)!=19998 or len(set(ids))!=19998 or Counter(r['label'] for r in gold)!={label:6666 for label in LABELS}:
        raise ValueError('Expected all 19,998 balanced official development claims')
    if ids!=[r['id'] for r in saved] or ids!=[r['id'] for r in long]:raise ValueError('Baseline ID/order mismatch')
    for g,s,l in zip(gold,saved,long):
        if g['claim']!=s['claim'] or g['claim']!=l['claim'] or s['predicted_evidence']!=l['predicted_evidence']:
            raise ValueError('Claim or evidence pairing mismatch')


def prepare(args):
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError('Choose a new output folder')
    base=ROOT/'reports/part2/official_fever_dev_20261023';extended=ROOT/'reports/part2/token_budget_20261028'
    gold=load(ROOT/'data/official_fever/shared_task_dev.jsonl');saved=load(base/'outputs.jsonl');long=load(extended/'outputs.jsonl')
    validate_pairing(gold,saved,long)
    # Metadata requests only. connect explicitly refuses downloads.
    session,model,version=connect(args.model);session.close()
    previous=ROOT/'reports/part2/llm_configurations_20261028'
    old={r['id']:r for r in load(previous/'inputs.jsonl') if r['split']=='assessment'}
    prior_texts={' '.join(r['claim'].casefold().split()) for r in load(ROOT/'reports/part2/ollama_comparison_20261028/sample.jsonl')}
    prior_texts.update(' '.join(r['claim'].casefold().split()) for r in old.values())
    prior_texts.add(' '.join(json.loads((ROOT/'reports/part2/ollama_comparison_20261028/telemundo_input.json').read_text())['claim'].casefold().split()))
    out.mkdir(parents=True,exist_ok=False)
    connection=sqlite3.connect((ROOT/'data/processed/wiki_fts.db').resolve().as_uri()+'?mode=ro',uri=True)
    try:
        with (out/'inputs.jsonl').open('x') as stream:
            for index,(g,s,l) in enumerate(zip(gold,saved,long),1):
                original=[]
                for page,sid in s['predicted_evidence']:
                    found=connection.execute('SELECT text FROM page_sentences WHERE page_id=? AND sentence_id=?',(page,sid)).fetchone()
                    if found is None:raise ValueError('Saved evidence missing from local snapshot')
                    original.append(dict(page_id=page,sentence_id=sid,text=found[0]))
                entity=retrieve_entity_anchored(connection,g['claim'],5)
                if g['id'] in old:
                    for key,ev in [('original',original),('entity',entity)]:
                        if [(r['page_id'],r['sentence_id'],r['text']) for r in ev]!=[(r['page_id'],r['sentence_id'],r['text']) for r in old[g['id']]['evidence'][key]]:
                            raise ValueError('Retrieval differs from saved assessment: '+str(g['id']))
                row=dict(id=g['id'],claim=g['claim'],gold=g,evidence=dict(original=original,entity=entity),
                    claim_probabilities=s['model_probabilities']['claim'],
                    baseline_predictions=dict(evidence_256=s['predictions']['retrieved_roberta'],evidence_512=l['predictions']['retrieved_roberta'],
                        combined_256=s['predictions']['calibrated_ensemble'],combined_512=l['predictions']['calibrated_ensemble'],claim_only_256=s['predictions']['claim_only_H2']),
                    roberta_full_input_tokens=l['full_input_tokens'],previous_llm_text=' '.join(g['claim'].casefold().split()) in prior_texts)
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                if index%1000==0:
                    write_json(out/'progress.json',dict(status='preparing',completed=index,total=19998))
                    print('Prepared',index,'/ 19998',flush=True)
    finally:connection.close()
    sources=[Path(__file__).resolve(),ROOT/'src/experiment_llm_configurations.py',ROOT/'src/experiment_ollama_fever.py',
        ROOT/'src/local_llm_experiments.py',ROOT/'src/retrieve_entity_anchored.py',ROOT/'src/retrieve_title_aware.py',
        ROOT/'src/experiment_token_budget.py',ROOT/'src/experiment_symbolic_v2.py',ROOT/'src/evaluate_retrieval.py',ROOT/'src/vendor/fever_scorer.py',
        ROOT/'data/official_fever/shared_task_dev.jsonl',base/'outputs.jsonl',extended/'outputs.jsonl',out/'inputs.jsonl']
    timing=load(previous/'assessment_responses.jsonl')
    seconds=sum(r['raw']['wall_seconds'] for r in timing)/len(timing)
    rows=load(out/'inputs.jsonl')
    protocol=dict(created_utc=now(),status='prepared_not_started',n=19998,model=model,ollama=version,
        conditions={k:CONFIGS[k] for k in CONDITIONS},derived_hybrid='E_claim_fallback; derived from D_both, no additional LLM request',
        seed=42,temperature=0,num_predict=128,num_ctx=4096,schema=SCHEMA,
        primary_comparator='evidence_512',secondary_comparator='evidence_256',
        primary_control='A_original and evidence RoBERTa use exactly the same original five evidence sentences; different model sizes/tokenizers/context budgets remain.',
        system_comparison='D_both changes both prompting and retrieval. This does not isolate architecture or either intervention individually.',
        scope='All official development claims, including earlier inspected LLM samples and Telemundo. Expanded development evaluation, not an untouched test.',
        selection='Original A and previously tested D are frozen now. No configuration selection on these full-dev outcomes.',
        previous_llm_text_count=sum(r['previous_llm_text'] for r in rows),
        estimated_hours_per_condition=seconds*19998/3600,
        estimate_note='Extrapolation from prior 240 assessment request wall times; not a guaranteed completion time.',
        inputs_sha256={str(p):digest(p) for p in sources},
        limitations=['Full official development set has previously been inspected/evaluated. This is not a blind test.',
            'Llama pretraining exposure to FEVER/Wikipedia is unknown.',
            'No equal-resource or training-budget match; RoBERTa uses a 512-token input limit and Llama a 4096-token context.',
            'Keep any failed/invalid responses in the full denominator; do not silently convert to NEI.',
            'Reasons are generated explanations, not faithful hidden reasoning or human validation.',
            'Single local model, quantization and seed; no retraining or model-selection search.',
            'D and hybrid use different evidence from the original baseline; label gains do not prove better reasoning alone.'])
    write_json(out/'protocol.json',protocol)
    for path in sources[:10]:(out/('source_'+path.name)).write_bytes(path.read_bytes())
    write_json(out/'progress.json',dict(status='prepared_not_started',completed=0,total=19998,conditions=list(CONDITIONS),estimated_hours_per_condition=protocol['estimated_hours_per_condition']))
    (out/'PLAN.md').write_text('# Full FEVER development evaluation — prepared, not started\n\n'
        'Baseline: frozen evidence RoBERTa on all 19,998 claims.\n\n'
        'A_original: Llama with the identical original evidence (controlled system comparison).\n\n'
        'D_both: Llama with the previously fixed comparison prompt and entity-based evidence (pipeline comparison).\n\n'
        'E_claim_fallback: derive from D without another Llama request; label as hybrid.\n\n'
        'Approximately %.1f hours per Llama condition, %.1f hours for both, based on saved timings. No retraining, downloads or cloud calls.\n\n'
        'Every claim/request/response is saved. Resume uses an unchanged ordered prefix. Three consecutive invalid responses stop the run. Earlier experiments are untouched.\n\n'
        'All earlier inspected pilot/assessment texts remain included in the full benchmark; report their uninspected-text complement separately as a sensitivity analysis, not a blind test.\n'
        %(protocol['estimated_hours_per_condition'],2*protocol['estimated_hours_per_condition']))
    print('Ready; no inference performed:',out,flush=True)


def verify_prefix(rows,results,config):
    if len(results)>len(rows) or [r['id'] for r in results]!=[r['id'] for r in rows[:len(results)]]:raise ValueError('Response prefix mismatch')
    for row,result in zip(rows,results):
        if result['config']!=config:raise ValueError('Response configuration mismatch')
        request=result.get('raw',{}).get('request')
        if request and request['messages']!=[dict(role='user',content=make_prompt(row,config))]:raise ValueError('Saved prompt mismatch')


def request_one(session,model,row,config):
    started=time.perf_counter();result=dict(id=row['id'],config=config,label='INVALID',reason='')
    try:
        result['raw']=call(session,model,make_prompt(row,config),42,SCHEMA,0,128)
        body=result['raw']['response'];parsed=validate_response(body)
        if body.get('prompt_eval_count',0)+body.get('eval_count',0)>=4096:raise ValueError('Context budget may have been exhausted')
        result.update(parsed)
    except Exception as exc:result['error']=type(exc).__name__+': '+str(exc)
    result['wall_seconds']=time.perf_counter()-started
    return result


def analyze(out,rows,results,config,protocol):
    labels=[r['label'] for r in results]
    metrics=score_config(rows,labels,config)
    original=[dict(r,sentences=r['evidence']['original'],predicted_evidence=[[s['page_id'],s['sentence_id']] for s in r['evidence']['original']]) for r in rows]
    baselines={k:score_original(original,[LABELS[r['baseline_predictions'][k]] for r in rows],submit_evidence=k!='claim_only_256') for k in rows[0]['baseline_predictions']}
    gold=[LABELS.index(r['gold']['label']) for r in rows]
    pred=[LABELS.index(p) if p in LABELS else 3 for p in labels]
    summary=dict(status='complete',config=config,n=len(rows),condition=metrics,baselines=baselines,
        paired_vs_evidence_512=comparison(gold,[r['baseline_predictions']['evidence_512'] for r in rows],pred),
        measured_request_wall_seconds=sum(r['wall_seconds'] for r in results),model=protocol['model'],limitations=protocol['limitations'])
    fresh=[i for i,r in enumerate(rows) if not r['previous_llm_text']]
    summary['previously_uninspected_llm_text_subset']=dict(note='Sensitivity subset only; not a blind test',**score_config([rows[i] for i in fresh],[labels[i] for i in fresh],config))
    if config=='D_both':
        final=[decision(row,label,'E_claim_fallback') for row,label in zip(rows,labels)]
        summary['derived_hybrid']=dict(**score_config(rows,[v[0] for v in final],'E_claim_fallback'),overrides=sum(v[1] for v in final))
        derived=[dict(id=row['id'],raw_llm_label=label,final_label=value,claim_override=override) for row,label,(value,override) in zip(rows,labels,final)]
        path=out/'hybrid_decisions.jsonl'
        if path.exists():
            if load(path)!=derived:raise ValueError('Existing hybrid decisions differ; preserved for inspection')
        else:
            with path.open('x') as stream:
                for row in derived:stream.write(json.dumps(row)+'\n')
    write_json(out/(config+'_summary.json'),summary)
    report=['# Full official-development Llama evaluation: '+config,'',
        '| System | Claims | Accuracy | Macro-F1 | Strict FEVER |','|---|---:|---:|---:|---:|']
    for name,v in [('Evidence RoBERTa (512 tokens)',baselines['evidence_512']),('Llama '+config,metrics)]+([('Hybrid with claim-only fallback',summary['derived_hybrid'])] if 'derived_hybrid' in summary else []):
        report.append('| %s | %d | %.2f%% | %.4f | %.2f%% |'%(name,v['n'],v['accuracy']*100,v['macro_f1'],v['fever_score']*100))
    report+=['','Full development evaluation; not a blind test. A uses identical original evidence; D changes prompt and retrieval. Hybrid includes a claim-only override.','',*['- '+s for s in protocol['limitations']]]
    (out/(config+'_REPORT.md')).write_text('\n'.join(report)+'\n')


def run(args):
    out=Path(args.output).resolve();protocol=json.loads((out/'protocol.json').read_text());config=args.config
    with (out/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another full-dev run is active')
        for path,expected in protocol['inputs_sha256'].items():
            if digest(path)!=expected:raise ValueError('Frozen input/source changed: '+path)
        rows=load(out/'inputs.jsonl');path=out/(config+'_responses.jsonl')
        if path.exists() and not args.resume:raise FileExistsError('Use --resume; existing responses are preserved')
        results=load(path) if path.exists() else [];verify_prefix(rows,results,config)
        if (out/(config+'_summary.json')).exists():raise FileExistsError('This condition is already complete')
        session,model,version=connect(protocol['model']['name'])
        if model['digest']!=protocol['model']['digest'] or version!=protocol['ollama']:raise ValueError('Model/runtime mismatch')
        initial=len(results);started=time.perf_counter();consecutive=0;invalid=sum(r['label']=='INVALID' for r in results)
        try:
            with path.open('a' if path.exists() else 'x') as stream:
                for row in rows[initial:]:
                    result=request_one(session,model['name'],row,config)
                    stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno());results.append(result)
                    failed=result['label']=='INVALID';invalid+=failed;consecutive=consecutive+1 if failed else 0
                    elapsed=time.perf_counter()-started
                    progress=dict(status='running',config=config,completed=len(results),total=len(rows),invalid=invalid,updated_utc=now(),
                        session_seconds=elapsed,estimated_remaining_seconds=(len(rows)-len(results))*elapsed/(len(results)-initial))
                    write_json(out/'progress.json',progress)
                    if len(results)%25==0:print(json.dumps(progress),flush=True)
                    if consecutive>=3:raise RuntimeError('Three consecutive invalid responses; inspect saved artifacts before resuming')
            check,after,after_version=connect(model['name']);check.close()
            if after['digest']!=model['digest'] or after_version!=version:raise ValueError('Model/runtime changed during evaluation')
            analyze(out,rows,results,config,protocol)
            write_json(out/'progress.json',dict(status='condition_complete',config=config,completed=len(results),total=len(rows),updated_utc=now()))
        except BaseException as exc:
            write_json(out/'progress.json',dict(status='stopped',config=config,completed=len(results),total=len(rows),reason=type(exc).__name__+': '+str(exc),updated_utc=now()))
            raise
        finally:session.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','run'])
    parser.add_argument('--output',required=True);parser.add_argument('--model',default='llama3.1:8b')
    parser.add_argument('--config',choices=CONDITIONS,default='A_original');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    prepare(args) if args.mode=='prepare' else run(args)


if __name__=='__main__':main()
