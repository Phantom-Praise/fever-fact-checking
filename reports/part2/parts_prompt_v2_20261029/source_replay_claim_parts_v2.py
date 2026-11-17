"""Approved three-addition prompt test on the same ten frozen evidence inputs."""
import argparse
import json
from pathlib import Path
import shutil

import replay_claim_parts as base
from fetch_wikipedia_articles import atomic_json,sha

ROOT=base.ROOT
PREVIOUS=ROOT/'reports/part2/parts_prompt_replay_20261029'
OUT=ROOT/'reports/part2/parts_prompt_v2_20261029'
ADDITIONS=[
 ('An indivisible claim may have one part.', 'Write each part as a complete factual assertion with its subject and relationship. Do not return isolated topics or duplicate the same assertion across multiple parts.'),
 ('Missing information alone is not a contradiction.', 'Before assigning REFUTES, identify the specific statement in the supplied evidence that is incompatible with the part. A different attribute, activity, or example is not a contradiction if both statements could be true.'),
 ('NOT ENOUGH INFO may cite none.', 'For SUPPORTS or REFUTES, explain how the cited text establishes or contradicts the complete part. An excerpt discussing the same subject is insufficient by itself.')]


def revised_prompt(before):
    after=before
    for anchor,addition in ADDITIONS:
        if after.count(anchor)!=1:raise ValueError('Prompt anchor is not unique')
        after=after.replace(anchor,anchor+'\n\n'+addition,1)
    return after


def prepare(out):
    before=(PREVIOUS/'USER_PROMPT.txt').read_text().rstrip('\n')
    if before!=base.INSTRUCTION:raise ValueError('Previous prompt source changed')
    after=revised_prompt(before);base.INSTRUCTION=after;base.prepare(out)
    (out/'BEFORE.txt').write_text(before+'\n');(out/'AFTER.txt').write_text(after+'\n')
    shutil.copyfile(Path(__file__),out/'source_replay_claim_parts_v2.py')
    protocol=json.loads((out/'protocol.json').read_text())
    paths=[Path(__file__).resolve(),PREVIOUS/'protocol.json',PREVIOUS/'summary.json']
    inputs=[]
    for ident in protocol['ids']:
        old_path=PREVIOUS/'traces'/('%s.json'%ident);paths.append(old_path)
        old=json.loads(old_path.read_text());original=json.loads((base.SOURCE/'traces'/('%s_ephemeral_full_article_rag.json'%ident)).read_text())
        if old['prompt'].split(base.MARKER,1)[1]!=original['prompt'].split(base.MARKER,1)[1]:raise ValueError('Previous evidence input changed')
        inputs.append(dict(id=ident,input_json_sha256=sha(old['prompt'].split(base.MARKER,1)[1].encode())))
    protocol.update(previous_condition=str(PREVIOUS),additions=[a for _,a in ADDITIONS],paired_inputs=inputs,
        change='Only the three approved general instruction additions. Same JSON schema, evidence suffix, excerpt IDs, model, seed, temperature, context/output caps and claim order as the earlier parts prompt.',
        limitations=['Ten already inspected development claims; not an unseen or statistically reliable improvement estimate.',
            'Three additions tested together; this run cannot isolate which addition caused a change.',
            'Same 256-token output cap. Truncated or invalid results remain INVALID and count as errors; no retries or silent repairs.',
            'Citation validity and label aggregation do not establish semantic correctness. No claim-specific vocabulary, gold labels or example answers are added to prompts.'])
    protocol['source_sha256'].update({str(p):sha(p.read_bytes()) for p in paths})
    atomic_json(out/'protocol.json',protocol)


def audit_and_report(out):
    protocol=json.loads((out/'protocol.json').read_text());rows=[json.loads(l) for l in (out/'results.jsonl').read_text().splitlines()]
    gold={r['id']:r['gold'] for r in map(json.loads,(base.SOURCE/'results.jsonl').read_text().splitlines())}
    comparisons=[];checks=[]
    for row in rows:
        ident=row['id'];old=json.loads((PREVIOUS/'traces'/('%s.json'%ident)).read_text());new=json.loads((out/'traces'/('%s.json'%ident)).read_text())
        same_input=old['prompt'].split(base.MARKER,1)[1]==new['prompt'].split(base.MARKER,1)[1]
        same_evidence=old['supplied_evidence']==new['supplied_evidence']
        assert same_input and same_evidence
        assert new['prompt'].split(base.MARKER,1)[0]==revised_prompt(old['prompt'].split(base.MARKER,1)[0])
        if 'raw' in new:
            assert old['raw']['request']['options']==new['raw']['request']['options']
            assert old['raw']['request']['format']==new['raw']['request']['format']
            assert old['raw']['request']['model']==new['raw']['request']['model']
            assert new['raw']['request']['messages']==[dict(role='user',content=new['prompt'])]
        checks.append(dict(id=ident,evidence_byte_identical=same_input,citation_metadata_identical=same_evidence,settings_and_schema_unchanged='raw' in new))
        comparisons.append(dict(id=ident,claim=row['claim'],gold=gold[ident],before_model=old['model_label'],before_derived=old['derived_label'],after_model=new['model_label'],after_derived=new['derived_label'],consistent=new['consistent'],error=new.get('error')))
    assert len(rows)==10 and len(set(r['id'] for r in rows))==10
    assert all(sha(Path(p).read_bytes())==h for p,h in protocol['source_sha256'].items())
    summary=json.loads((out/'summary.json').read_text())
    summary.update(before_model_correct=sum(r['before_model']==r['gold'] for r in comparisons),before_derived_correct=sum(r['before_derived']==r['gold'] for r in comparisons),
        derived_fixed=[r['id'] for r in comparisons if r['before_derived']!=r['gold'] and r['after_derived']==r['gold']],
        derived_regressed=[r['id'] for r in comparisons if r['before_derived']==r['gold'] and r['after_derived']!=r['gold']])
    atomic_json(out/'summary.json',summary);atomic_json(out/'comparison.json',comparisons);atomic_json(out/'audit.json',dict(n=10,checks=checks,all_source_hashes_match=True))
    lines=['# Three-addition parts-prompt test','', 'Same ten claim IDs and byte-identical saved evidence. No new retrieval, article downloads, embeddings, training or full-development evaluation. Only the three approved instruction additions changed.','',
        '| Scoring | Previous prompt | New prompt |','|---|---:|---:|',
        '| Model overall label | %s/10 | %s/10 |'%(summary['before_model_correct'],summary['model_label_correct']),
        '| Code-derived label | %s/10 | %s/10 |'%(summary['before_derived_correct'],summary['derived_label_correct']),'',
        'Invalid outputs: %s. Model/derived disagreements: %s.'%(summary['invalid'],summary['inconsistent']),'',
        '| ID | Reference | Before model | Before derived | After model | After derived |','|---|---|---|---|---|---|']
    for r in comparisons:lines.append('| %s | %s | %s | %s | %s | %s |'%(r['id'],r['gold'],r['before_model'],r['before_derived'],r['after_model'],r['after_derived']))
    lines+=['','## Limits','',*['- '+s for s in protocol['limitations']],'']
    (out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','report']);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();out=a.output.resolve()
    if a.mode=='prepare':prepare(out)
    elif a.mode=='run':
        base.INSTRUCTION=revised_prompt((PREVIOUS/'USER_PROMPT.txt').read_text().rstrip('\n'));base.run(out);audit_and_report(out)
    else:audit_and_report(out)

if __name__=='__main__':main()
