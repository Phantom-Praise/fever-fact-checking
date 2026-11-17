"""Audit and report the ten measured-token exhaustive-reader final judgments."""
import json
from pathlib import Path
from finalize_exhaustive_tokens import OUT,SOURCE,ROOT,final_payload,prompt_for
from fetch_wikipedia_articles import atomic_json,sha
from replay_claim_parts import LABELS


def main():
    p=json.loads((OUT/'protocol.json').read_text());summary=json.loads((OUT/'summary.json').read_text());assert summary['status']=='complete' and summary['n']==10
    assert sha((ROOT/'src/finalize_exhaustive_tokens.py').read_bytes())==p['source_code_sha256']
    assert sha((SOURCE/'protocol.json').read_bytes())==p['original_protocol_sha256']
    source_audit=json.loads((SOURCE/'audit.json').read_text());assert source_audit['full_coverage_claims']==10 and source_audit['temporary_folder_empty'] and source_audit['article_text_matches_prior']
    rows=[json.loads(l) for l in (OUT/'results.jsonl').read_text().splitlines()];comparison=json.loads((SOURCE/'comparison.json').read_text());metrics=json.loads((SOURCE/'metrics.json').read_text());byid={r['id']:r for r in rows};checks=[]
    for r in comparison:
        ident=r['id'];folder=OUT/'traces'/str(ident);t=json.loads((folder/'trace.json').read_text());file=SOURCE/'traces'/str(ident)/'trace.json';original=json.loads(file.read_text())
        assert sha(file.read_bytes())==t['source_trace_sha256'];assert t['coverage']==original['coverage']
        assert t['input']==final_payload(original)
        assert t['tokenization']['prompt_sha256']==sha(prompt_for(original).encode())
        valid=[a for a in t['final_attempts'] if a['valid']]
        if t['status']=='complete':
            raw=json.loads((folder/valid[-1]['file']).read_text());assert raw['request']['messages'][0]['content']==prompt_for(original)
            assert raw['response']['prompt_eval_count']==t['expected_prompt_tokens']==t['tokenization']['raw_prompt_tokens']+p['wrapper_tokens']
            assert raw['response']['prompt_eval_count']+raw['response']['eval_count']<t['context']
            assert raw['response']['done'] and raw['response'].get('done_reason') in ('stop',None)
            assert set(t['supplied_evidence'])=={f['evidence_id'] for f in original['findings']}
            for f in original['findings']:assert t['supplied_evidence'][f['evidence_id']]['text']==f['quote']
        token_attempts=[]
        for attempt in t['final_attempts']:
            raw=json.loads((folder/attempt['file']).read_text())
            token_attempts.append(dict(file=attempt['file'],reported_prompt_tokens=raw.get('response',{}).get('prompt_eval_count'),matches_expected=raw.get('response',{}).get('prompt_eval_count')==t['expected_prompt_tokens'],valid=attempt['valid'],error=attempt.get('error')))
        checks.append(dict(id=ident,status=t['status'],all_selected_evidence_preserved=True,raw_prompt_tokens=t['tokenization']['raw_prompt_tokens'],expected_prompt_tokens=t['expected_prompt_tokens'],reported_prompt_tokens=token_attempts[-1]['reported_prompt_tokens'] if token_attempts else None,context=t['context'],model_derived_consistent=t.get('consistent'),attempts=token_attempts))
        r.update(full_reader_model=byid[ident]['model_label'],full_reader_derived=byid[ident]['derived_label'])
    for condition in ['full_reader_model','full_reader_derived']:
        matrix=[[sum(r['gold']==a and r[condition]==b for r in comparison) for b in LABELS+['INVALID']] for a in LABELS];per_class={}
        for label in LABELS:
            tp=sum(r['gold']==label and r[condition]==label for r in comparison);fp=sum(r['gold']!=label and r[condition]==label for r in comparison);fn=sum(r['gold']==label and r[condition]!=label for r in comparison)
            precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
            per_class[label]=dict(precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0)
        metrics[condition]=dict(correct=sum(r[condition]==r['gold'] for r in comparison),n=10,accuracy=sum(r[condition]==r['gold'] for r in comparison)/10,macro_f1=sum(v['f1'] for v in per_class.values())/3,confusion_matrix=matrix,columns=LABELS+['INVALID'],rows=LABELS,per_class=per_class)
    audit=dict(n=10,all_original_chunks_processed=True,chunks=summary['coverage_chunks'],all_selected_evidence_preserved=True,all_token_counts_match=all(c['attempts'] and all(a['matches_expected'] for a in c['attempts']) for c in checks),all_articles_match_prior_text=source_audit['article_text_matches_prior'],all_temporary_downloads_and_databases_deleted=True,checks=checks)
    reader_calls=[];initial_final_calls=[];verified_calls=[]
    for row in rows:
        ident=row['id'];source_folder=SOURCE/'traces'/str(ident)
        reader_calls += [json.loads(f.read_text()) for f in source_folder.glob('batch_*_attempt_*.json')]
        initial_final_calls += [json.loads(f.read_text()) for f in source_folder.glob('final_attempt_*.json')]
        verified_calls += [json.loads(f.read_text()) for f in (OUT/'traces'/str(ident)).glob('final_attempt_*.json')]
    runtime=dict(reader_calls=len(reader_calls),initial_guard_attempt_final_calls=len(initial_final_calls),verified_final_calls=len(verified_calls),reader_call_seconds=sum(r['wall_seconds'] for r in reader_calls),verified_final_call_seconds=sum(r['wall_seconds'] for r in verified_calls),note='Recorded call times for separate stages, not a single-pass latency benchmark. Stopped quotation and sentence-refinement development attempts are preserved separately and excluded here.')
    atomic_json(OUT/'runtime.json',runtime)
    atomic_json(OUT/'audit.json',audit);atomic_json(OUT/'comparison.json',comparison);atomic_json(OUT/'metrics.json',metrics)
    lines=['# Every-chunk reader — measured-token final results','',
        'Same ten development claims, same Llama 3.1 8B digest, and the same previously successful Wikipedia article revisions. The reader processed all 2,129 eligible body-chunk instances. The final judge received every chunk selected across those reader batches, without additional sentence filtering.','',
        '| Condition | Correct / 10 | Accuracy | Macro F1 |','|---|---:|---:|---:|']
    for c in ['old_five_sentence','top_k_rag','parts_model','parts_derived','full_reader_model','full_reader_derived']:
        m=metrics[c];lines.append('| %s | %s | %.0f%% | %.3f |'%(c,m['correct'],10*m['correct'],m['macro_f1']))
    lines+=['','Raw overall labels and code-derived labels remain separate. Derivation is any REFUTES part → REFUTES; otherwise all SUPPORTS → SUPPORTS; otherwise NEI. Neither consistency nor a valid citation ID proves the evidence reasoning is correct.','',
        '| ID | FEVER reference | Prior parts: model / derived | Full reader: model / derived | Input tokens (with wrapper) |','|---|---|---|---|---:|']
    for r,c in zip(comparison,checks):lines.append('| %s | %s | %s / %s | %s / %s | %s |'%(r['id'],r['gold'],r['parts_model'],r['parts_derived'],r['full_reader_model'],r['full_reader_derived'],c['expected_prompt_tokens']))
    lines+=['','## Integrity and development corrections','',f"- Coverage: {summary['coverage_chunks']}/{summary['total_chunks']} body chunks across ten claims. Full coverage means each was supplied in a completed, validated reader batch.",f"- Final invalid outputs: {summary['invalid']}. All reported token counts match: {audit['all_token_counts_match']}.",'- All fetched article-text hashes matched the prior pilot. One previously unavailable article remains unavailable; that claim uses the same four successfully fetched pages.','- Original temporary articles and SQLite databases were deleted. Full saved reader prompts still retain supplied article text.','- Initial final judgments used a conservative UTF-8 byte bound; seven cases were rejected before inference. Those are implementation-budget failures, not measured token-context exhaustion. Their original results remain in `../exhaustive_reader_20261029/`.','- An intermediate precise-sentence follow-up was stopped after encountering the same conservative byte-bound issue. Its partial artifacts remain in `../exhaustive_precise_20261029/`. It is not included as a completed accuracy condition.','- The final correction counts tokens with the active model tokenizer. Seven saved map/final requests calibrated a stable ten-token wrapper. Each final response must report exactly the expected full prompt token count; a mismatch is INVALID.','- All ten final judgments were run under this corrected policy. No new article retrieval, embeddings, evidence filtering, training, or full-19K evaluation was performed.','',
        '## Interpretation and limits','',
        'These ten claims were already inspected during development. Treat changes as development observations, not a generalization estimate or statistical evidence of improvement. Reader selection, number of calls, prompts and context/output budgets differ from the earlier top-k condition. Reading all chunks does not guarantee complete understanding or perfect relevance selection. The final judge receives selected source chunks, not all article text simultaneously. FEVER reference labels are retained unchanged; historical evidence and current revisions can differ.','',
        '## Qualitative caution','', 'For #130325, the final part verdict is REFUTES, but its justification says dark matter particles are different from neutrinos—the same relationship asserted by the claim. It also cites many unrelated passages, including biological dark matter. The reference label matches while the explanation is inconsistent. Citation-ID validity and aggregation consistency do not validate semantic reasoning.', '',
        'For #47172 (Yvonne Strahovski), the model returned REFUTES because the evidence did not establish silent-film acting and mentioned sound productions. It separately returned NEI for a fragment called silent film. Appearing in sound productions does not itself exclude silent-film acting; missing support was incorrectly treated as contradiction. Broader reading did not resolve this reasoning error.', '',
        '## Exact final part judgments','']
    for r in rows:
        t=json.loads((OUT/'traces'/str(r['id'])/'trace.json').read_text());lines+=['### #'+str(r['id'])+' — '+r['claim'],'','Reference: '+r['gold']+'. Model: '+r['model_label']+'. Derived: '+r['derived_label']+'.','']
        for part in t['parts']:lines.append('- '+part['verdict']+' — '+part['part']+' — '+part['justification'])
        if t.get('error'):lines.append('- '+t['error'])
        lines.append('')
    (OUT/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(dict(summary=summary,scores={k:v['correct'] for k,v in metrics.items()},audit=audit),indent=2))

if __name__=='__main__':main()
