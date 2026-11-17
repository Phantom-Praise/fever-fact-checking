"""Freeze and run a ten-claim, local, paired ephemeral-RAG pilot.

An exploratory system comparison: new article corpus, vector retrieval, context
and citation prompt change together. No broad improvement claim from ten cases.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import random
import signal
import sqlite3
import sys
import time

from ephemeral_rag import LABELS,MODEL_DIR,SCHEMA,INSTRUCTION,MiniLMEmbedder,classify_ephemeral
from experiment_ollama_fever import prompt as original_prompt,SCHEMA as ORIGINAL_SCHEMA,validate_response
from fetch_wikipedia_articles import atomic_json,sha
from local_llm_experiments import call,connect
from retrieve_title_aware import retrieve

ROOT=Path(__file__).resolve().parents[1]


def load(path):
    with Path(path).open() as stream:return [json.loads(line) for line in stream if line.strip()]

def normalize(text):return ' '.join(text.casefold().split())
def now():return datetime.now(timezone.utc).isoformat()


def sample_rows(gold,excluded,seed=20261029):
    rng=random.Random(seed);chosen=[];seen=set(excluded)
    for label,n in zip(LABELS,[4,3,3]):
        pool=[]
        for row in gold:
            text=normalize(row['claim'])
            if row['label']==label and text not in seen:pool.append(row);seen.add(text)
        if len(pool)<n:raise ValueError('Insufficient distinct claims')
        chosen.extend(rng.sample(pool,n))
    rng.shuffle(chosen)
    return chosen


def prepare(out):
    if out.exists():raise FileExistsError('Choose a new pilot folder')
    paused=ROOT/'reports/part2/ollama_full_dev_20261028'
    if json.loads((paused/'batch_status.json').read_text())['status']!='stopped':raise ValueError('Full benchmark must remain stopped')
    gold_path=ROOT/'data/official_fever/shared_task_dev.jsonl';gold=load(gold_path)
    prior_inputs=load(paused/'inputs.jsonl');by_id={r['id']:r['claim'] for r in prior_inputs}
    prior_results=load(paused/'A_original_responses.jsonl')
    inspected={normalize(by_id[r['id']]) for r in prior_results}
    inspected.update(normalize(r['claim']) for r in load(ROOT/'reports/part2/ollama_comparison_20261028/sample.jsonl'))
    inspected.update(normalize(r['claim']) for r in load(ROOT/'reports/part2/llm_configurations_20261028/inputs.jsonl'))
    inspected.update(map(normalize,['Jack Falahee was born.','Telemundo is a English-language television network.','Psych debuted in 2004.']))
    training_path=ROOT/'data/processed/train.jsonl';training={normalize(r['claim']) for r in load(training_path)}
    selected=sample_rows(gold,inspected|training)
    connection=sqlite3.connect((ROOT/'data/processed/wiki_fts.db').resolve().as_uri()+'?mode=ro',uri=True)
    rng=random.Random(431);rows=[]
    try:
        for g in selected:
            order=['old_five_sentence','ephemeral_full_article_rag'];rng.shuffle(order)
            started=time.perf_counter();evidence=retrieve(connection,g['claim'],5);elapsed=time.perf_counter()-started
            rows.append(dict(id=g['id'],claim=g['claim'],gold=g['label'],original_evidence=evidence,original_retrieval_seconds=elapsed,condition_order=order))
    finally:connection.close()
    session,model,version=connect('llama3.1:8b');session.close()
    out.mkdir(parents=True,exist_ok=False);(out/'traces').mkdir();(out/'raw').mkdir();(out/'temporary').mkdir()
    (out/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    sources=[Path(__file__).resolve(),ROOT/'src/ephemeral_rag.py',ROOT/'src/fetch_wikipedia_articles.py',ROOT/'src/retrieve_title_aware.py',
        ROOT/'src/local_llm_experiments.py',ROOT/'src/experiment_ollama_fever.py',ROOT/'src/rag_embedding_worker.py',out/'inputs.jsonl',gold_path,training_path,MODEL_DIR/'download_manifest.json']
    protocol=dict(status='frozen_before_inference',created_utc=now(),n=10,class_counts=dict(Counter(r['gold'] for r in rows)),sampling_seed=20261029,
        sampling='Unique official-dev claims excluding normalized processed training texts and previously inspected LLM/prototype texts; labels used only for 4/3/3 stratification, never in prompts.',
        model=model,ollama=version,temperature=0,seed=42,num_ctx=4096,num_predict=256,
        conditions=dict(old_five_sentence='Original retrieved five sentence texts and original direct prompt, newly evaluated on these ten claims.',
            ephemeral_full_article_rag='Original title ranking top-ten/top-five, live full articles, CPU MiniLM embeddings, per-claim FAISS+SQLite, broader passages, explicit excerpt citations.'),
        new_instruction=INSTRUCTION,new_schema=SCHEMA,baseline_schema=ORIGINAL_SCHEMA,
        settings=dict(chunk_target_tokens=200,embedding_max_tokens=256,vector_candidates=30,context_character_budget=7600,max_cited_excerpts=5,embedding_device='cpu',embedding_process='separate CPU worker; one Torch thread'),
        timing_note='Original includes recorded local retrieval time plus generation/parse. RAG includes fresh network article fetching, CPU embedding worker startup/inference, vector search, generation and workspace deletion. Shared tokenizer setup is recorded separately. This is not an equal-cost retrieval benchmark.',
        embedding_model=json.loads((MODEL_DIR/'download_manifest.json').read_text()),
        retention='Retain prediction, source metadata, supplied/cited passages, exact prompts/responses. Delete all freshly downloaded full articles and vector databases after each claim.',
        source_sha256={str(p):sha(p.read_bytes()) for p in sources},
        limitations=['Ten exploratory development claims; not a reliable population improvement estimate or a blind test.',
            'Corpus date, evidence retrieval, input context and citation prompt differ together; this does not isolate the effect of a vector database.',
            'No strict FEVER score is computed for live-Wikipedia citations; current revision/block offsets are not historical FEVER sentence IDs.',
            'Citation validation proves only that quoted excerpts came from the supplied context, not that they logically establish the label.',
            'Original title ranking can retain irrelevant/disambiguation pages. No retuning from these ten outcomes.',
            'Baseline uses a 256-token output cap in this pilot, versus 128 in the earlier saved original experiment. Both current conditions have the same generation limits.',
            'One local Llama model and embedding model; temperature zero and a fixed seed do not guarantee cross-runtime determinism.'])
    atomic_json(out/'protocol.json',protocol)
    for p in sources[:7]:(out/('source_'+p.name)).write_bytes(p.read_bytes())
    atomic_json(out/'progress.json',dict(status='prepared',completed=0,total=10))
    print('Frozen 10-claim pilot:',dict(Counter(r['gold'] for r in rows)),flush=True)


def old_prediction(row,generate):
    started=time.perf_counter();result=dict(claim=row['claim'],label='INVALID',supplied_evidence=row['original_evidence'])
    text=original_prompt(dict(claim=row['claim'],sentences=row['original_evidence']));result['prompt']=text
    try:
        result['raw']=generate(text);result.update(validate_response(result['raw']['response']))
        body=result['raw']['response']
        if body.get('prompt_eval_count',0)+body.get('eval_count',0)>=4096:raise ValueError('Context budget may have been exhausted')
    except Exception as exc:result.update(label='INVALID',error=type(exc).__name__+': '+str(exc))
    result['retrieval_seconds']=row.get('original_retrieval_seconds',0.)
    result['wall_seconds']=time.perf_counter()-started+result['retrieval_seconds']
    return result


def condition_metrics(rows,condition):
    matrix=[[0]*4 for _ in LABELS]
    for r in rows:
        prediction=r['results'][condition]['label'];matrix[LABELS.index(r['gold'])][LABELS.index(prediction) if prediction in LABELS else 3]+=1
    per_class={}
    for i,label in enumerate(LABELS):
        tp=matrix[i][i];fn=sum(matrix[i])-tp;fp=sum(row[i] for row in matrix)-tp
        precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
        per_class[label]=dict(precision=precision,recall=recall,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return dict(n=len(rows),correct=sum(matrix[i][i] for i in range(3)),accuracy=sum(matrix[i][i] for i in range(3))/len(rows),
        macro_f1=sum(v['f1'] for v in per_class.values())/3,confusion_matrix=matrix,per_class=per_class,
        invalid=sum(row[3] for row in matrix),mean_wall_seconds=sum(r['results'][condition]['wall_seconds'] for r in rows)/len(rows))


def analyze(out,rows,protocol):
    metrics={c:condition_metrics(rows,c) for c in ['old_five_sentence','ephemeral_full_article_rag']}
    fixed=sum(r['results']['old_five_sentence']['label']!=r['gold'] and r['results']['ephemeral_full_article_rag']['label']==r['gold'] for r in rows)
    introduced=sum(r['results']['old_five_sentence']['label']==r['gold'] and r['results']['ephemeral_full_article_rag']['label']!=r['gold'] for r in rows)
    cleanup=[r['results']['ephemeral_full_article_rag']['cleanup']['articles_and_database_deleted'] for r in rows]
    if not all(cleanup) or any((out/'temporary').iterdir()):raise RuntimeError('Temporary workspace cleanup failed; inspect before proceeding')
    summary=dict(status='complete',n=len(rows),conditions=metrics,fixed=fixed,introduced=introduced,all_workspaces_deleted=all(cleanup),
        limitations=protocol['limitations'],embedding_setup_seconds=rows[0].get('embedding_setup_seconds'),no_full_articles_retained=True)
    atomic_json(out/'summary.json',summary)
    lines=['# Temporary per-claim RAG: ten-claim paired pilot','','Same ten official-development claims, same installed Llama 3.1 8B, temperature 0, seed 42. This is a small system comparison; no broad improvement claim.','',
        '| System | Correct | Accuracy | Macro-F1 | Invalid | Mean pipeline time |','|---|---:|---:|---:|---:|---:|']
    for c,m in metrics.items():lines.append('| %s | %d/%d | %.1f%% | %.4f | %d | %.2f s |'%(c,m['correct'],m['n'],m['accuracy']*100,m['macro_f1'],m['invalid'],m['mean_wall_seconds']))
    lines+=['','Errors fixed: %d. New errors introduced: %d.'%(fixed,introduced),'','All ten per-claim workspaces deleted, including full articles, chunk metadata and vectors. Retained files contain only the review artifacts approved by the user. The reusable embedding model remains installed.','',
        '| Claim ID | Claim | FEVER reference | Original five-sentence Llama | Full-article RAG Llama |','|---|---|---|---|---|']
    for r in rows:lines.append('| %s | %s | %s | %s | %s |'%(r['id'],r['claim'].replace('|','\\|'),r['gold'],r['results']['old_five_sentence']['label'],r['results']['ephemeral_full_article_rag']['label']))
    lines+=['','## Limits','',*['- '+s for s in protocol['limitations']],'','Per-class scores and confusion matrices are in summary.json. Each trace retains the exact context and response. An available or valid citation is not independent evidence of a sound explanation.','']
    (out/'REPORT.md').write_text('\n'.join(lines))


def run(out):
    protocol=json.loads((out/'protocol.json').read_text());rows=load(out/'inputs.jsonl')
    if (out/'results.jsonl').exists():raise FileExistsError('Pilot already started; existing results preserved')
    if json.loads((ROOT/'reports/part2/ollama_full_dev_20261028/batch_status.json').read_text())['status']!='stopped':raise ValueError('Full benchmark must remain stopped')
    for path,expected in protocol['source_sha256'].items():
        if sha(Path(path).read_bytes())!=expected:raise ValueError('Frozen input/source changed: '+path)
    session,model,version=connect(protocol['model']['name'])
    if model['digest']!=protocol['model']['digest'] or version!=protocol['ollama']:raise ValueError('Model or runtime changed')
    started=time.perf_counter();embedder=MiniLMEmbedder();setup=time.perf_counter()-started;results=[]
    try:
        with (out/'results.jsonl').open('x') as stream:
            for row in rows:
                combined={k:row[k] for k in ['id','claim','gold']};combined['results']={}
                for condition in row['condition_order']:
                    schema=ORIGINAL_SCHEMA if condition=='old_five_sentence' else SCHEMA
                    def generate(prompt):
                        raw=call(session,model['name'],prompt,42,schema,0,256)
                        with (out/'raw'/('%s_%s.json'%(row['id'],condition))).open('x') as raw_stream:json.dump(raw,raw_stream,ensure_ascii=False,indent=2)
                        return raw
                    if condition=='old_five_sentence':trace=old_prediction(row,generate)
                    else:trace=classify_ephemeral(row['claim'],embedder,generate,out/'temporary')
                    combined['results'][condition]=dict(label=trace['label'],wall_seconds=trace['wall_seconds'],
                        **({'cleanup':trace['cleanup'],'cited_evidence':trace.get('cited_evidence',[]),'error':trace.get('error')} if condition!='old_five_sentence' else {}))
                    with (out/'traces'/('%s_%s.json'%(row['id'],condition))).open('x') as trace_stream:json.dump(trace,trace_stream,ensure_ascii=False,indent=2)
                    print('SAVED',row['id'],condition,trace['label'],'cleanup',trace.get('cleanup'),flush=True)
                    if condition!='old_five_sentence' and not trace['cleanup']['articles_and_database_deleted']:raise RuntimeError('Claim workspace not deleted')
                if not results:combined['embedding_setup_seconds']=setup
                stream.write(json.dumps(combined,ensure_ascii=False)+'\n');stream.flush();results.append(combined)
                atomic_json(out/'progress.json',dict(status='running',completed=len(results),total=10,latest_id=row['id'],updated_utc=now()))
        check,after,after_version=connect(model['name']);check.close()
        if after['digest']!=model['digest'] or after_version!=version:raise ValueError('Model/runtime changed during pilot')
        analyze(out,results,protocol);atomic_json(out/'progress.json',dict(status='complete',completed=10,total=10,updated_utc=now()))
    except BaseException as exc:
        atomic_json(out/'progress.json',dict(status='stopped',completed=len(results),total=10,error=type(exc).__name__+': '+str(exc),updated_utc=now()));raise
    finally:session.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','run']);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output).resolve()
    def stop(signum,frame):raise KeyboardInterrupt('Pilot interrupted; current workspace cleaned')
    signal.signal(signal.SIGTERM,stop)
    prepare(out) if args.mode=='prepare' else run(out)

if __name__=='__main__':main()
