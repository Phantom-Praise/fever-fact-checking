"""Final judgment for all ten full-scan claims using verified model token counts.

Uses the installed Ollama llama-server tokenizer, with observed chat-wrapper
calibration and a strict post-generation prompt-count check. No new reading,
downloads, embeddings or source filtering. Retain earlier budget failures.
"""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time

from experiment_exhaustive_reader import ROOT,OUT as SOURCE,generate,validate_final,now
from fetch_wikipedia_articles import atomic_json,sha
from local_llm_experiments import connect
from replay_claim_parts import INSTRUCTION,SCHEMA

OUT=ROOT/'reports/part2/exhaustive_token_verified_20261029'
BLOB='/Users/praiseben/.ollama/models/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29'


def token_count(session,text):
    process_list=subprocess.check_output(['ps','-Ao','pid=,command='],text=True)
    pattern=r'^\s*(\d+)\s+/Applications/Ollama\.app/Contents/Resources/llama-server --model '+re.escape(BLOB)+r' --port (\d+) --host 127\.0\.0\.1(?: |$)'
    matches=[re.match(pattern,line) for line in process_list.splitlines()];matches=[m for m in matches if m]
    if len(matches)!=1:raise ValueError('Need exactly one active tokenizer for the pinned local model')
    url='http://127.0.0.1:'+matches[0].group(2)+'/tokenize'
    payload=dict(content=text,add_special=False,parse_special=True)
    response=session.post(url,json=payload,timeout=30);response.raise_for_status();tokens=response.json()['tokens']
    if not isinstance(tokens,list) or not all(isinstance(t,int) for t in tokens):raise ValueError('Invalid tokenizer response')
    return dict(raw_prompt_tokens=len(tokens),token_ids_sha256=sha(json.dumps(tokens).encode()),prompt_sha256=sha(text.encode()),tokenizer_process=int(matches[0].group(1)),tokenizer_url=url,tokenizer_options={k:v for k,v in payload.items() if k!='content'})


def context_for(tokens,wrapper):
    context=max(8192,math.ceil((tokens+wrapper+3072+512)/4096)*4096)
    if context>32768:raise ValueError('Measured token count exceeds final 32768-token protocol limit; no evidence discarded')
    return context


def final_payload(trace):
    # Preserve the complete set, order and exact text of first-pass findings.
    byid=trace['cited_sources']
    return dict(claim=trace['claim'],evidence=[dict(id=f['evidence_id'],title=byid[f['evidence_id']]['title'],section=byid[f['evidence_id']]['section_path'],text=f['quote'],reader_note=f['relevance']) for f in trace['findings']])


def prompt_for(trace):
    return INSTRUCTION+'\n\nAll article batches have been read. The following exact quotes were collected across all batches. Reader notes are fallible suggestions, not evidence. Judge the quotes themselves.\n\nINPUT JSON:\n'+json.dumps(final_payload(trace),ensure_ascii=False)


def prepare():
    if OUT.exists():raise FileExistsError(OUT)
    original=json.loads((SOURCE/'protocol.json').read_text());session,model,version=connect(original['model']['name'])
    if model['digest']!=original['model']['digest'] or version!=original['ollama']:raise ValueError('Model/runtime changed')
    calibration=[]
    try:
        files=[SOURCE/'traces'/str(i)/'batch_001_attempt_1.json' for i in original['ids'][:5]]
        files += [SOURCE/'traces/40442/final_attempt_1.json',SOURCE/'traces/2147/final_attempt_1.json']
        for file in files:
            raw=json.loads(file.read_text());count=token_count(session,raw['request']['messages'][0]['content']);count.update(file=str(file),reported_prompt_tokens=raw['response']['prompt_eval_count']);count['wrapper_tokens']=count['reported_prompt_tokens']-count['raw_prompt_tokens'];calibration.append(count)
        offsets={c['wrapper_tokens'] for c in calibration}
        if len(offsets)!=1 or not 0<=next(iter(offsets))<=128:raise ValueError('Chat wrapper token calibration is not stable')
        wrapper=next(iter(offsets))
    finally:session.close()
    OUT.mkdir();(OUT/'traces').mkdir();shutil.copyfile(Path(__file__),OUT/'source_finalize_exhaustive_tokens.py')
    atomic_json(OUT/'protocol.json',dict(status='frozen_before_final_inference',created_utc=now(),model=model,ollama=version,ids=original['ids'],source_code_sha256=sha(Path(__file__).read_bytes()),original_protocol_sha256=sha((SOURCE/'protocol.json').read_bytes()),wrapper_tokens=wrapper,calibration=calibration,temperature=0,seed=42,output_caps=[1536,3072],maximum_context=32768,instruction=INSTRUCTION,schema=SCHEMA,
        evidence='All selected chunks from each original exhaustive reader trace, unchanged. No new retrieval, source filtering, sentence selection or reference labels in prompts. Every original article-body chunk was already processed.',
        context='Count raw prompt tokens using the tokenizer of the active pinned model; add calibrated chat-wrapper tokens, 3072 output reserve and 512 safety tokens; round up to 4096, minimum8192, maximum32768. Require reported prompt_eval_count to equal expected token count after inference. Otherwise INVALID.',
        retry='At most two identical-prompt attempts for completion/validation failure, with output caps1536 then3072. No retry based on reference labels.',
        retention='No new articles, vectors or database. Full original batch prompts and final prompts/responses remain. Original temporary download/database cleanup audited separately.',
        limitations=['Development repair after observing overly conservative byte-budget failures; not a new held-out estimate.','Final judge receives all first-pass selected chunks, while reader batches collectively received every original body chunk. Relevance selection can still miss evidence.','Workflow and context/output budgets differ from earlier top-k and essential-parts runs.','Both original failed attempts and stopped sentence-refinement attempt are preserved separately.']))
    atomic_json(OUT/'progress.json',dict(status='prepared',completed=0,total=10))
    print('Prepared measured-token final judgments; calibrated wrapper tokens:',wrapper,flush=True)


def run():
    p=json.loads((OUT/'protocol.json').read_text())
    if sha(Path(__file__).read_bytes())!=p['source_code_sha256']:raise ValueError('Frozen source changed')
    if (OUT/'results.jsonl').exists():raise FileExistsError('Preserve previous results')
    session,model,version=connect(p['model']['name'])
    if model['digest']!=p['model']['digest'] or version!=p['ollama']:raise ValueError('Model/runtime changed')
    rows=[json.loads(l) for l in (SOURCE/'results.jsonl').read_text().splitlines()];results=[]
    try:
        with (OUT/'results.jsonl').open('x') as stream:
            for row in rows:
                start=time.perf_counter();ident=row['id'];folder=OUT/'traces'/str(ident);folder.mkdir();file=SOURCE/'traces'/str(ident)/'trace.json';original=json.loads(file.read_text())
                trace=dict(id=ident,claim=row['claim'],status='running',token_verified=True,model_label='INVALID',derived_label='INVALID',coverage=original['coverage'],cleanup=original['cleanup'],source_trace_sha256=sha(file.read_bytes()),articles=original['articles'],final_attempts=[],parts=[])
                atomic_json(OUT/'progress.json',dict(status='running',completed=len(results),total=10,current_id=ident,updated_utc=now()))
                try:
                    if original['coverage']['processed']!=original['coverage']['total'] or not original['coverage']['total']:raise ValueError('Original exhaustive scan incomplete')
                    trace['input']=final_payload(original);trace['supplied_evidence']={f['evidence_id']:dict(text=f['quote'],source=original['cited_sources'][f['evidence_id']]) for f in original['findings']}
                    prompt=prompt_for(original);count=token_count(session,prompt);expected=count['raw_prompt_tokens']+p['wrapper_tokens'];context=context_for(count['raw_prompt_tokens'],p['wrapper_tokens']);trace.update(tokenization=count,expected_prompt_tokens=expected,context=context)
                    parsed=None
                    for attempt,cap in enumerate(p['output_caps'],1):
                        output=folder/('final_attempt_%d.json'%attempt)
                        try:
                            body=generate(session,model['name'],prompt,SCHEMA,context,cap,output)
                            if body.get('prompt_eval_count')!=expected:raise ValueError('Prompt token count mismatch; possible formatting change or truncation')
                            parsed=validate_final(body,set(trace['supplied_evidence']),context);trace['final_attempts'].append(dict(file=output.name,valid=True,reported_prompt_tokens=body['prompt_eval_count']));break
                        except Exception as exc:parsed=None;trace['final_attempts'].append(dict(file=output.name,valid=False,error=type(exc).__name__+': '+str(exc)))
                    if parsed is None:raise ValueError('Final generation/validation failed')
                    trace.update(parsed,status='complete');trace['consistent']=trace['model_label']==trace['derived_label']
                except Exception as exc:trace.update(status='failed',error=type(exc).__name__+': '+str(exc))
                trace['wall_seconds']=time.perf_counter()-start;atomic_json(folder/'trace.json',trace)
                result={k:trace[k] for k in ['id','claim','status','model_label','derived_label','coverage','cleanup']};result.update(gold=row['gold'],error=trace.get('error'))
                stream.write(json.dumps(result)+'\n');stream.flush();results.append(result)
                print('FINAL',ident,trace['model_label'],trace['derived_label'],'tokens',trace.get('expected_prompt_tokens'),'context',trace.get('context'),trace.get('error'),flush=True)
                atomic_json(OUT/'progress.json',dict(status='running',completed=len(results),total=10,updated_utc=now()))
        check,after,after_version=connect(model['name']);check.close()
        if after['digest']!=model['digest'] or after_version!=version:raise ValueError('Model/runtime changed during final judgments')
        atomic_json(OUT/'summary.json',dict(status='complete',n=10,model_correct=sum(r['model_label']==r['gold'] for r in results),derived_correct=sum(r['derived_label']==r['gold'] for r in results),invalid=sum(r['model_label']=='INVALID' for r in results),coverage_chunks=sum(r['coverage']['processed'] for r in results),total_chunks=sum(r['coverage']['total'] for r in results)))
        atomic_json(OUT/'progress.json',dict(status='complete',completed=10,total=10,updated_utc=now()))
    finally:session.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);a=parser.parse_args();prepare() if a.mode=='prepare' else run()
