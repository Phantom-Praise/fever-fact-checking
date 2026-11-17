"""Audit exhaustive-reader coverage and compare the fixed ten development claims."""
from collections import Counter
import json
from pathlib import Path
from experiment_exhaustive_reader import OUT, SOURCE, ROOT, LABELS
from fetch_wikipedia_articles import atomic_json, sha


def main():
    rows=[json.loads(l) for l in (OUT/'results.jsonl').read_text().splitlines()]
    if len(rows)!=10 or json.loads((OUT/'progress.json').read_text())['status']!='complete':raise ValueError('Wait until all ten attempts finish')
    old={r['id']:r for r in map(json.loads,(SOURCE/'results.jsonl').read_text().splitlines())}
    replay={r['id']:r for r in map(json.loads,(ROOT/'reports/part2/parts_prompt_replay_20261029/results.jsonl').read_text().splitlines())}
    checks=[];comparison=[]
    for r in rows:
        folder=OUT/'traces'/str(r['id']);t=json.loads((folder/'trace.json').read_text());manifest={c['id']:c for c in t.get('chunk_manifest',[])};processed=[]
        for b in t['batches']:
            valid=[a for a in b['attempts'] if a['valid']]
            if not valid:continue
            raw=json.loads((folder/valid[-1]['file']).read_text());payload=json.loads(raw['request']['messages'][0]['content'].split('\n\nINPUT JSON:\n',1)[1])
            assert payload['claim']==r['claim'] and set(payload)=={'claim','chunks'}
            assert [c['id'] for c in payload['chunks']]==b['chunk_ids']
            assert raw['response']['done'] and raw['response'].get('done_reason') in ('stop',None)
            for c in payload['chunks']:assert sha(c['text'].encode())==manifest[c['id']]['text_sha256']
            byid={c['id']:c for c in payload['chunks']}
            for f in b['findings']:assert f['quote']==byid[f['evidence_id']]['text']
            processed.extend(b['chunk_ids'])
        assert len(processed)==len(set(processed))==t['coverage']['processed']
        assert r['coverage']==t['coverage']
        full=set(processed)==set(manifest) and len(manifest)==t['coverage']['total'] and bool(manifest)
        if t['status']=='complete':assert full
        assert t['cleanup']['temporary_files_deleted']
        checks.append(dict(id=r['id'],chunks=len(manifest),valid_processed=len(processed),all_chunks_processed=full,article_text_matches_prior=all(a['matches_previous_text'] for a in t['articles']),status=t['status'],error=t.get('error')))
        comparison.append(dict(id=r['id'],claim=r['claim'],gold=r['gold'],old_five_sentence=old[r['id']]['results']['old_five_sentence']['label'],top_k_rag=old[r['id']]['results']['ephemeral_full_article_rag']['label'],parts_model=replay[r['id']]['model_label'],parts_derived=replay[r['id']]['derived_label'],exhaustive_model=r['model_label'],exhaustive_derived=r['derived_label']))
    protocol=json.loads((OUT/'protocol.json').read_text())
    source_unchanged=all(sha(Path(p).read_bytes())==h for p,h in protocol['source_sha256'].items());assert source_unchanged
    assert not any((OUT/'temporary').iterdir())
    audit=dict(claims=checks,source_hashes_unchanged=source_unchanged,temporary_folder_empty=True,full_coverage_claims=sum(c['all_chunks_processed'] for c in checks),article_text_matches_prior=all(c['article_text_matches_prior'] for c in checks),retention=protocol['retention'])
    atomic_json(OUT/'audit.json',audit);atomic_json(OUT/'comparison.json',comparison)
    metrics={}
    for condition in ['old_five_sentence','top_k_rag','parts_model','parts_derived','exhaustive_model','exhaustive_derived']:
        matrix=[[sum(r['gold']==a and r[condition]==b for r in comparison) for b in LABELS+['INVALID']] for a in LABELS]
        per_class={}
        for label in LABELS:
            tp=sum(r['gold']==label and r[condition]==label for r in comparison);fp=sum(r['gold']!=label and r[condition]==label for r in comparison);fn=sum(r['gold']==label and r[condition]!=label for r in comparison)
            p=tp/(tp+fp) if tp+fp else 0;rec=tp/(tp+fn) if tp+fn else 0
            per_class[label]=dict(precision=p,recall=rec,f1=2*p*rec/(p+rec) if p+rec else 0)
        metrics[condition]=dict(correct=sum(r[condition]==r['gold'] for r in comparison),n=10,confusion_matrix=matrix,columns=LABELS+['INVALID'],rows=LABELS,per_class=per_class,macro_f1=sum(v['f1'] for v in per_class.values())/3)
    atomic_json(OUT/'metrics.json',metrics)
    lines=['# Every-chunk reading pilot — ten development claims','',
        'The same ten claim IDs and previously successful Wikipedia revisions were used. Each eligible article-body chunk entered exactly one reader batch. The final judge received the original text of the chunks selected across those batches. This is a multi-call full scan, not a top-k vector search and not a single prompt containing all articles.','',
        '| Condition | Correct / 10 | Accuracy | Macro F1 |','|---|---:|---:|---:|']
    for c,m in metrics.items():lines.append('| %s | %s | %.0f%% | %.3f |'%(c,m['correct'],10*m['correct'],m['macro_f1']))
    lines+=['','Model overall labels and code-derived labels are separate measurements. INVALID attempts remain errors in the denominator.','',
        '| Claim ID | FEVER reference | Previous top-k | Parts derived | Every-chunk model | Every-chunk derived | Chunks processed |','|---|---|---|---|---|---|---|']
    for r,c in zip(comparison,checks):lines.append('| %s | %s | %s | %s | %s | %s | %s/%s |'%(r['id'],r['gold'],r['top_k_rag'],r['parts_derived'],r['exhaustive_model'],r['exhaustive_derived'],c['valid_processed'],c['chunks']))
    lines+=['','## Coverage and integrity','',f"- Claims with every chunk processed: {audit['full_coverage_claims']}/10.",f"- Extracted article text matches prior pilot: {audit['article_text_matches_prior']}.",'- Temporary article downloads and SQLite databases were deleted. Full batch prompts retain supplied article text.',f"- Total attempt time: {sum(r['wall_seconds'] for r in rows)/60:.1f} minutes.",'- The earlier quote-extraction attempt is preserved in `../exhaustive_reader_quote_attempt_20261029`; it was stopped after non-verbatim quotations failed validation. It is not folded into this protocol’s scores.','',
        '## Failures and limits','']
    for c in checks:
        if c['error']:lines.append('- #%s: %s'%(c['id'],c['error']))
    lines+=['- Final context sizing uses a conservative UTF-8 byte upper bound, not a measured Llama token count. A prompt rejected by this bound is an implementation-budget failure; it is not a measured model context exhaustion.']
    lines += ['- '+v for v in protocol['limitations']]
    lines+=['','A full scan proves delivery coverage, not reasoning quality. Relevance selection may still miss evidence or retain distracting background. A correct reference label can still have an unsupported explanation. No general accuracy improvement can be established from these ten inspected cases.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(dict(metrics={k:v['correct'] for k,v in metrics.items()},audit=audit),indent=2))

if __name__=='__main__':main()
