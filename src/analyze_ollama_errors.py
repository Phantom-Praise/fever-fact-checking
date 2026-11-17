"""Analyze the saved paired pilot and run a bounded gold-evidence diagnostic.

No training or production changes. Oracle inputs deliberately use FEVER gold
annotations for diagnosis only, never as a claimed end-to-end system result.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import random
import sqlite3

from evaluate_retrieval import gold_evidence_groups
from experiment_ollama_fever import LABELS, SCHEMA, prompt, validate_response, load
from experiment_symbolic_v2 import digest
from local_llm_experiments import connect, call

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'reports/part2/ollama_comparison_20261028'
DEFAULT=ROOT/'reports/part2/ollama_error_analysis_20261028'


def coverage(row):
    if row['gold']['label']=='NOT ENOUGH INFO':return 'NEI'
    found=set(map(tuple,row['predicted_evidence']))
    return 'complete_annotated_group' if any(g<=found for g in gold_evidence_groups(row['gold'])) else 'no_complete_annotated_group'


def prepare(out):
    if out.exists():raise FileExistsError(out)
    rows=load(PILOT/'sample.jsonl');responses=load(PILOT/'responses.jsonl')
    if len(rows)!=300 or [r['id'] for r in rows]!=[r['id'] for r in responses]:raise ValueError('Incomplete paired sample')
    groups=defaultdict(Counter);transitions=defaultdict(Counter);class_changes=defaultdict(Counter)
    for r,p in zip(rows,responses):
        g=r['gold']['label'];old=LABELS[r['baselines']['combined_512']];new=p['label'];bucket=coverage(r)
        c=groups[bucket];c['n']+=1;c['combined_correct']+=old==g;c['llm_correct']+=new==g;c['llm_NEI']+=new==LABELS[2]
        c['new_errors']+=old==g and new!=g;c['errors_fixed']+=old!=g and new==g
        transitions[g][new]+=1
        c=class_changes[g];c['combined_correct']+=old==g;c['llm_correct']+=new==g;c['net_correct_change']+=(new==g)-(old==g)
    summary=json.loads((PILOT/'summary.json').read_text())
    findings=dict(n=300,by_annotation_coverage=groups,llm_gold_to_prediction=transitions,correct_by_gold_label=class_changes,
        predicted_label_counts=dict(llm=dict(Counter(p['label'] for p in responses)),combined_512=dict(Counter(LABELS[r['baselines']['combined_512']] for r in rows))),
        comparable_metrics={name:summary['conditions'][name] for name in ['evidence_512','combined_512','local_llm']},
        paired_comparisons=summary['comparisons'],
        integrity=dict(invalid_responses=sum(p['label']=='INVALID' for p in responses),
            incomplete_responses=sum(not p.get('raw',{}).get('response',{}).get('done',False) for p in responses),
            max_prompt_plus_output_tokens=max(p['raw']['response']['prompt_eval_count']+p['raw']['response']['eval_count'] for p in responses),
            context_limit=4096,roberta512_truncated_inputs=sum(r['roberta_full_input_tokens']>512 for r in rows)),
        caveats=['Annotation coverage is observable, not proof that all unannotated evidence is insufficient.',
            'The combined baseline uses claim-only and evidence models plus frozen calibration; Llama is a single prompted model.',
            'No aggregate count reveals a model\'s hidden reasoning. Examples illustrate specific errors, not a complete manual taxonomy.',
            'Gold evidence interventions are oracle diagnostics, not achievable retrieval or production accuracy.',
            'This is the existing 300-claim development pilot, not a new untouched test.'])
    conn=sqlite3.connect((ROOT/'data/processed/wiki_fts.db').as_uri()+'?mode=ro',uri=True)
    eligible=[];missing=[]
    for r,p in zip(rows,responses):
        if coverage(r)!='no_complete_annotated_group':continue
        options=[]
        for group in gold_evidence_groups(r['gold']):
            sentences=[]
            for page,sid in sorted(group):
                found=conn.execute('SELECT text FROM page_sentences WHERE page_id=? AND sentence_id=?',(page,sid)).fetchone()
                if found is None:break
                sentences.append(dict(page_id=page,sentence_id=sid,text=found[0]))
            else:options.append(sentences)
        if not options:missing.append(r['id']);continue
        selected=min(options,key=lambda s:(len(s),[(x['page_id'],x['sentence_id']) for x in s]))
        eligible.append(dict(id=r['id'],claim=r['claim'],gold_label=r['gold']['label'],original_llm_label=p['label'],
            original_reason=p['reason'],original_sentences=r['sentences'],sentences=selected))
    conn.close()
    chosen=random.Random(42).sample(eligible,20)
    out.mkdir(parents=True,exist_ok=False)
    (out/'analysis.json').write_text(json.dumps(findings,indent=2))
    (out/'oracle_inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in chosen))
    protocol=json.loads((PILOT/'protocol.json').read_text())
    manifest=dict(model=protocol['model'],ollama=protocol['ollama'],seed=42,n=20,
        selection='Random.sample(seed=42) of 20 verifiable pilot claims lacking a complete annotated group in original top five, restricted to cases with a complete gold group in the local index. No selection on LLM correctness.',
        incomplete_group_population=79,eligible=len(eligible),missing_gold_from_index=missing,ids=[r['id'] for r in chosen],
        intervention='Keep claim/model/prompt/decoding fixed; replace original evidence texts with the shortest available complete FEVER-annotated group (lexicographic reference tie-break).',
        settings=dict(temperature=0,seed=42,num_predict=128,num_ctx=4096),oracle=True,
        input_sha256={str(p):digest(p) for p in [Path(__file__),PILOT/'sample.jsonl',PILOT/'responses.jsonl',out/'oracle_inputs.jsonl']})
    (out/'protocol.json').write_text(json.dumps(manifest,indent=2));(out/'runner_source.py').write_text(Path(__file__).read_text())
    print('Prepared oracle diagnostic',json.dumps(dict(n=20,eligible=len(eligible),missing_index=missing)),flush=True)


def run(out):
    protocol=json.loads((out/'protocol.json').read_text())
    for path,expected in protocol['input_sha256'].items():
        if digest(path)!=expected:raise ValueError('Inputs changed')
    session,model,version=connect(protocol['model']['name'])
    if model['digest']!=protocol['model']['digest'] or version!=protocol['ollama']:raise ValueError('Model/runtime changed')
    rows=load(out/'oracle_inputs.jsonl');results=[]
    with (out/'oracle_responses.jsonl').open('x') as stream:
        for row in rows:
            result=dict(id=row['id'],label='INVALID',reason='')
            try:
                raw=call(session,model['name'],prompt(row),42,SCHEMA,0,128)
                result['raw']=raw;result.update(validate_response(raw['response']))
            except Exception as exc:result['error']=type(exc).__name__+': '+str(exc)
            stream.write(json.dumps(result)+'\n');stream.flush();results.append(result)
            print('Oracle completed',len(results),'/20',flush=True)
    _,after,after_version=connect(model['name'])
    if after['digest']!=model['digest'] or after_version!=version:raise ValueError('Model/runtime changed')
    old=[r['original_llm_label']==r['gold_label'] for r in rows]
    new=[r['label']==s['gold_label'] for r,s in zip(results,rows)]
    summary=dict(n=20,original_correct=sum(old),oracle_correct=sum(new),errors_fixed=sum(not a and b for a,b in zip(old,new)),
        newly_wrong=sum(a and not b for a,b in zip(old,new)),invalid=sum(r['label']=='INVALID' for r in results),
        oracle_only=True,warning='Selected incomplete-annotation subgroup; artificial access to gold evidence; not a replacement benchmark or a production improvement.')
    (out/'oracle_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


def report(out):
    a=json.loads((out/'analysis.json').read_text());oracle=json.loads((out/'oracle_summary.json').read_text())
    rows=load(PILOT/'sample.jsonl');responses=load(PILOT/'responses.jsonl');by_id={r['id']:(r,p) for r,p in zip(rows,responses)}
    lines=['# Why the local LLM scored lower on the paired pilot','',
        'All measurements below use the same 300 official-development claims and the saved original predictions. No benchmark labels or predictions were changed.','',
        '| System | Correct / 300 | Accuracy |','|---|---:|---:|']
    for name in ['evidence_512','combined_512','local_llm']:
        m=a['comparable_metrics'][name];lines.append('| %s | %d | %.2f%% |'%(name,round(300*m['accuracy']),100*m['accuracy']))
    lines+=['','The 60.33% comparator is a two-model calibrated combination. Evidence-only RoBERTa is 52.33%, versus Llama 51.67%; the paired accuracy-difference interval for that comparison includes zero.','',
        '| True FEVER label | Combined correct / 100 | LLM correct / 100 | Net change |','|---|---:|---:|---:|']
    for label,v in a['correct_by_gold_label'].items():lines.append('| %s | %d | %d | %+d |'%(label,v['combined_correct'],v['llm_correct'],v['net_correct_change']))
    lines+=['','Llama predicted NEI 163 times, versus 72 for the combined system. It called 37 SUPPORTS and 60 REFUTES claims NEI. The extra 24 correct NEI cases did not offset losing 23 SUPPORTS and 27 REFUTES correct cases: net -26 correct answers.','',
        '| Annotated evidence status | Claims | LLM correct | Combined correct | LLM predicted NEI |','|---|---:|---:|---:|---:|']
    for bucket,v in a['by_annotation_coverage'].items():lines.append('| %s | %d | %d | %d | %d |'%(bucket,v['n'],v['llm_correct'],v['combined_correct'],v['llm_NEI']))
    lines+=['','Missing a complete annotated group is not proof that every retrieved sentence is insufficient. The table measures annotation coverage, not a definitive causal attribution.','',
        '## Examples from the saved calls','']
    comments={146231:'Retrieval failure is visible here: none of the five texts concerns Juventus uniforms. The evidence-only NEI response is reasonable for the supplied texts, while FEVER expects REFUTES from the full Wikipedia annotation.',
        149173:'The gold sentence giving a 2006 debut is present. Treating 2006 as close enough to 2004 is an observable classification/explanation error.',
        202040:'The supplied evidence about a brother does not establish a sister. This is an unsupported inference in the saved answer.'}
    for ident,comment in comments.items():
        r,p=by_id[ident];lines+=['### #'+str(ident),r['claim'],'','Gold: '+r['gold']['label']+'; LLM: '+p['label']+'.','', 'Saved explanation: '+p['reason'],'',comment,'']
    lines+=['## Bounded oracle diagnostic','',
        'On 20 randomly sampled cases from the incomplete-annotation subgroup, replacing the retrieved texts with a complete FEVER-annotated evidence group changed correct labels from %d/20 to %d/20 (%d fixed, %d introduced).'%(oracle['original_correct'],oracle['oracle_correct'],oracle['errors_fixed'],oracle['newly_wrong']),
        'Claims, model, prompt and decoding settings stayed fixed. This measures a gold-evidence intervention on a selected subgroup. It is not end-to-end accuracy, and the intervention changes both relevance and evidence length/amount.','',
        '## What the current records rule out','',
        'No invalid outputs; all 300 responses completed. Maximum prompt plus generated token usage was %d, below the 4096 context budget. All supplied evidence fit in the matched RoBERTa 512-token inputs.'%a['integrity']['max_prompt_plus_output_tokens'],
        'These checks do not reveal hidden model reasoning or prove a single universal cause.','',
        '## Next controlled experiment','',
        'Improve candidate-page/entity matching on separate validation data, then compare the same judges with the resulting frozen evidence. Separately test date/negation/relationship counterexamples and an LLM hybrid selected on calibration data. Do not optimize prompts or fusion on this evaluated 300-claim sample.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    esc=html.escape
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FEVER · Why the LLM score dropped</title><style>body{font:16px/1.65 system-ui;max-width:1050px;margin:30px auto;padding:0 22px;color:#193047;background:#f4f7fa}section{background:white;border:1px solid #dce6ed;border-radius:14px;padding:24px;margin:22px 0}h1{font-size:34px;line-height:1.2;letter-spacing:-1px}h2{font-size:22px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:12px 8px;border-bottom:1px solid #e2e9ef;text-align:left}a{color:#126f73}small{color:#526c7b}.scroll{overflow:auto}blockquote{border-left:3px solid #9ecac7;margin:15px 0;padding-left:15px}.number{font-size:28px;font-weight:700}</style>',
        '<a href="/llm-comparison">← LLM comparison</a><h1>Why did the accuracy drop?</h1><p>Analysis of the same 300 claims, saved evidence and actual model responses.</p><section><h2>1. The 60% system combined two models</h2><div class="scroll"><table><tr><th>System</th><th>Correct / 300</th><th>Accuracy</th></tr>']
    names={'evidence_512':'Evidence-only RoBERTa','combined_512':'Combined RoBERTa system','local_llm':'Llama 3.1 8B'}
    for name in names:
        m=a['comparable_metrics'][name];page.append('<tr><td>%s</td><td>%d</td><td>%.2f%%</td></tr>'%(names[name],round(m['accuracy']*300),m['accuracy']*100))
    page+=['</table></div><p>The combined system merges claim-only and evidence model log scores with frozen calibration. Llama was used as a single prompted classifier. Its accuracy was close to evidence-only RoBERTa; that paired difference is uncertain on this sample.</p></section><section><h2>2. Llama answered NEI much more often</h2><p><span class="number">163 vs 72</span> NEI predictions, Llama versus combined system.</p><div class="scroll"><table><tr><th>True label</th><th>Combined correct / 100</th><th>LLM correct / 100</th><th>Net change</th></tr>']
    for label,v in a['correct_by_gold_label'].items():page.append('<tr><td>%s</td><td>%d</td><td>%d</td><td>%+d</td></tr>'%(esc(label),v['combined_correct'],v['llm_correct'],v['net_correct_change']))
    page+=['</table></div><p>It gained 24 correct NEI answers, but lost 23 SUPPORTS and 27 REFUTES correct answers: 26 fewer correct labels overall.</p></section><section><h2>3. Some retrieved evidence missed the subject</h2><p>79 of the 200 verifiable claims lacked a complete FEVER-annotated group in the retrieved five sentences. Llama answered NEI on 61 of those 79.</p><p>Coverage does not prove every unannotated alternative is insufficient. The Juventus case below is a concrete retrieval failure.</p></section>']
    for ident,comment in comments.items():
        r,p=by_id[ident];page+=['<section><h2>'+esc(r['claim'])+'</h2><p>FEVER: <b>'+esc(r['gold']['label'])+'</b> · Llama: <b>'+esc(p['label'])+'</b></p><blockquote>'+esc(p['reason'])+'</blockquote><p>'+esc(comment)+'</p><a href="/ollama-traces#'+str(ident)+'">Inspect the exact prompt and evidence →</a></section>']
    page+=['<section><h2>4. What happened when we supplied annotated evidence?</h2><p>A separate 20-claim oracle diagnostic sampled cases with incomplete original annotation coverage, without selecting on whether Llama was right.</p><p class="number">%d / 20 → %d / 20 correct</p><p>Only the evidence changed to a complete FEVER-annotated group. %d errors were fixed and %d new errors appeared.</p><small>This is an artificial gold-evidence intervention on a selected subgroup, not a production result or an improved benchmark score. Relevance and evidence length/amount change together.</small></section>'%(oracle['original_correct'],oracle['oracle_correct'],oracle['errors_fixed'],oracle['newly_wrong']),
        '<section><h2>5. Some errors remain with annotated evidence present</h2><p>In the original pilot, 36 of 121 verifiable claims with a complete annotated evidence group were still labeled NEI by Llama. Correct labels on this subgroup: Llama 73/121; combined system 90/121.</p><p>The Psych example shows an explicit date mismatch being accepted. Better retrieval alone does not explain all the errors.</p></section>',
        '<section><h2>Checks and interpretation</h2><p>No invalid responses. All 300 responses completed. Maximum prompt plus generated tokens: %d / 4096. All evidence fit fully in the matched RoBERTa 512-token input.</p><p>Strict FEVER moved differently from label accuracy: Llama 46.33%% versus combined 44.00%%. The paired interval for that strict-score difference includes zero.</p><p>These observations describe this local model, prompt and pilot. They do not establish that all LLMs perform worse or reveal a model\'s hidden reasoning.</p><p>Next: test retrieval/entity matching and date/relationship checks on separate validation data. Freeze changes before using a new evaluation sample.</p><small>FEVER supplied the reference labels and evidence annotations. Analysis and viewer added with AI assistance. Original experiment results remain unchanged.</small></section></html>'%a['integrity']['max_prompt_plus_output_tokens']]
    (out/'index.html').write_text(''.join(page))
    print('Wrote analysis report and page',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','report']);p.add_argument('--output',type=Path,default=DEFAULT);args=p.parse_args()
    {'prepare':prepare,'run':run,'report':report}[args.mode](args.output)


if __name__=='__main__':main()
