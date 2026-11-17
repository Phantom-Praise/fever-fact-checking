"""Audit the conditional precision follow-up without changing model results."""
import json
from pathlib import Path
from refine_exhaustive_evidence import OUT,SOURCE,ROOT
from fetch_wikipedia_articles import sha,atomic_json
from replay_claim_parts import LABELS


def main():
    protocol=json.loads((OUT/'protocol.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    assert summary['status']=='complete' and summary['n']==10
    from local_llm_experiments import connect
    session,model,version=connect(protocol['model']['name']);session.close()
    assert model['digest']==protocol['model']['digest'] and version==protocol['ollama']
    assert sha((ROOT/'src/refine_exhaustive_evidence.py').read_bytes())==protocol['source_code_sha256']
    assert sha((SOURCE/'protocol.json').read_bytes())==protocol['source_protocol_sha256']
    rows=[json.loads(l) for l in (OUT/'results.jsonl').read_text().splitlines()];comparison=json.loads((SOURCE/'comparison.json').read_text());metrics=json.loads((SOURCE/'metrics.json').read_text())
    byid={r['id']:r for r in rows};audits=[]
    for old in comparison:
        ident=old['id'];folder=OUT/'traces'/str(ident);trace=json.loads((folder/'trace.json').read_text());original_file=SOURCE/'traces'/str(ident)/'trace.json';original=json.loads(original_file.read_text())
        assert sha(original_file.read_bytes())==trace['original_trace_sha256']
        assert trace['coverage']['processed']==trace['coverage']['total'] and trace['cleanup']['temporary_files_deleted']
        audit=dict(id=ident,refined=trace['refined'],full_scan_complete=True)
        if trace['refined']:
            source_texts={f['evidence_id']:f['quote'] for f in original['findings']};processed=[];sentences={}
            for batch in trace['batches']:
                valid=[a for a in batch['attempts'] if a['valid']]
                if not valid:continue
                raw=json.loads((folder/valid[-1]['file']).read_text());payload=json.loads(raw['request']['messages'][0]['content'].split('\n\nINPUT JSON:\n',1)[1]);assert set(payload)=={'claim','chunks'} and payload['claim']==trace['claim']
                for c in payload['chunks']:
                    processed.append(c['chunk_id']);text=source_texts[c['chunk_id']]
                    for s in c['sentences']:
                        assert s['parent_id']==c['chunk_id']
                        assert text[s['local_char_start']:s['local_char_end']]==s['text'];sentences[s['id']]=s
                for f in batch['findings']:assert f['quote']==sentences[f['evidence_id']]['text']
            if trace['status']=='complete':
                assert len(processed)==len(set(processed))==len(source_texts) and set(processed)==set(source_texts)
                final=[a for a in trace['final_attempts'] if a['valid']][-1]
                raw=json.loads((folder/final['file']).read_text());payload=json.loads(raw['request']['messages'][0]['content'].split('\n\nINPUT JSON:\n',1)[1]);assert set(payload)=={'claim','evidence'}
                for e in payload['evidence']:assert e['text']==sentences[e['id']]['text']
                assert {e['id'] for e in payload['evidence']}=={f['evidence_id'] for f in trace['findings']}
                audit.update(selected_chunks_refined=len(processed),selected_sentences=len(trace['findings']))
        else:
            assert trace['model_label']==original['model_label'] and trace['derived_label']==original['derived_label']
            assert len(list(folder.glob('*.json')))==1
        audits.append(audit);old.update(precise_model=trace['model_label'],precise_derived=trace['derived_label'])
    for condition in ['precise_model','precise_derived']:
        matrix=[[sum(r['gold']==a and r[condition]==b for r in comparison) for b in LABELS+['INVALID']] for a in LABELS];per_class={}
        for label in LABELS:
            tp=sum(r['gold']==label and r[condition]==label for r in comparison);fp=sum(r['gold']!=label and r[condition]==label for r in comparison);fn=sum(r['gold']==label and r[condition]!=label for r in comparison)
            p=tp/(tp+fp) if tp+fp else 0;rec=tp/(tp+fn) if tp+fn else 0
            per_class[label]=dict(precision=p,recall=rec,f1=2*p*rec/(p+rec) if p+rec else 0)
        metrics[condition]=dict(correct=sum(r[condition]==r['gold'] for r in comparison),n=10,macro_f1=sum(v['f1'] for v in per_class.values())/3,confusion_matrix=matrix,columns=LABELS+['INVALID'],rows=LABELS,per_class=per_class)
    assert all(r['model_label']!='INVALID' or r.get('error') for r in rows)
    audit=dict(n=10,claims=audits,source_hashes_match=True,all_original_chunks_processed=True,no_new_articles_or_database=True)
    atomic_json(OUT/'audit.json',audit);atomic_json(OUT/'comparison.json',comparison);atomic_json(OUT/'metrics.json',metrics)
    lines=['# Full reading with precise-citation follow-up','',
        'The full reader processed every eligible body chunk in the same ten claims’ previously successful Wikipedia revisions. The first frozen run is preserved. Only cases blocked by its conservative final-prompt budget received a second selection pass, choosing exact sentence IDs after the broader reading. Other judgments were reused without additional model calls.','',
        '| Condition | Correct / 10 | Accuracy | Macro F1 |','|---|---:|---:|---:|']
    for c,m in metrics.items():lines.append('| %s | %s | %.0f%% | %.3f |'%(c,m['correct'],m['correct']*10,m['macro_f1']))
    lines+=['','The model’s overall label and the label derived from its reported part verdicts are shown separately. INVALID attempts remain errors in the denominator.','',
        '| ID | Reference | Prior parts derived | Original full scan derived | Precise model | Precise derived |','|---|---|---|---|---|---|']
    for r in comparison:lines.append('| %s | %s | %s | %s | %s | %s |'%(r['id'],r['gold'],r['parts_derived'],r['exhaustive_derived'],r['precise_model'],r['precise_derived']))
    lines+=['','## Coverage and limits','',f"- {summary['coverage_chunks']}/{summary['total_chunks']} original body chunks processed across ten claims.",f"- {summary['refined']} cases received precise-sentence refinement. Remaining cases reused their original full-scan judgments.",f"- Invalid final outputs: {summary['invalid']}.",'- Original downloads and temporary chunk databases were deleted. Full batch prompts still retain supplied article text; this is not full-text erasure.','- Sentence text is copied directly from recorded source spans; model-selected IDs and final citations were audited. This verifies provenance, not the correctness of the verdicts.','- The original final-budget failures were caused by a conservative UTF-8 byte bound. They do not demonstrate measured exhaustion of Llama’s token context.','- This fallback was developed after observing those implementation failures. These are inspected development claims, not an unseen improvement estimate.','- Final judgment sees selected evidence gathered across all batches, not all raw article text simultaneously. Both selection stages can miss evidence or retain irrelevant passages.','- This workflow changes context budgets, selection and number of calls; the comparison does not isolate one causal factor.','- FEVER reference labels are retained unchanged. Current article revisions and annotation issues can disagree with an otherwise plausible evidence-grounded answer.','']
    lines+=['## Exact final part judgments','']
    for r in rows:
        t=json.loads((OUT/'traces'/str(r['id'])/'trace.json').read_text());lines += ['### #'+str(r['id'])+' — '+r['claim'],'','Reference: '+r['gold']+'. Model: '+r['model_label']+'. Derived: '+r['derived_label']+'.','']
        for p in t.get('parts',[]):lines.append('- '+p['verdict']+' — '+p['part']+' — '+p['justification'])
        if t.get('error'):lines.append('- '+t['error'])
        lines.append('')
    (OUT/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(dict(summary=summary,scores={k:v['correct'] for k,v in metrics.items()},audit=audit),indent=2))

if __name__=='__main__':main()
