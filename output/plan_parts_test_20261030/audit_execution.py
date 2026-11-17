"""Inspect completed traces without rerunning either classifier."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def describe(values):
    return dict(n=len(values), minimum=min(values), median=statistics.median(values),
                mean=statistics.mean(values), maximum=max(values)) if values else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();out=args.run.resolve()
    summary=json.loads((out/'summary.json').read_text())
    traces=[json.loads(p.read_text()) for p in (out/'traces/parts_rag').glob('*.json')]
    baseline=[json.loads(p.read_text()) for p in (out/'traces/evidence_roberta').glob('*.json')]
    assert len(traces)==len(baseline)==summary['n']
    errors=Counter(t.get('error','none') for t in traces if t['label']=='INVALID')
    mismatches=[t['id'] for t in traces if 'raw' in t and t['raw']['response'].get('prompt_eval_count')!=t['token_preflight']['expected_prompt_tokens']]
    repeated_pages=[t['id'] for t in traces if len({a['page_id'] for a in t['articles']})<len(t['articles'])]
    audit=dict(n=summary['n'],token_count_mismatch_ids=mismatches,
               all_completed_workspaces_deleted=all(t['cleanup']['articles_and_database_deleted'] for t in traces),
               temporary_root_empty=not any((out/'temporary').iterdir()),invalid_errors=dict(errors),
               claims_with_fetch_errors=sum(bool(t['fetch_errors']) for t in traces),
               fetch_error_total=sum(len(t['fetch_errors']) for t in traces),
               claims_with_duplicate_resolved_pages=repeated_pages,
               main_predicted_counts=dict(Counter(t['label'] for t in traces)),
               baseline_predicted_counts=dict(Counter(t['label'] for t in baseline)),
               baseline_truncated_inputs=sum(t['token_trace']['truncated'] for t in baseline),
               pipeline_seconds=describe([t['wall_seconds'] for t in traces]),
               fetch_seconds=describe([t['fetch_seconds'] for t in traces if 'fetch_seconds' in t]),
               embedding_seconds=describe([t['embedding_seconds'] for t in traces if 'embedding_seconds' in t]),
               generation_seconds=describe([t['raw']['wall_seconds'] for t in traces if 'raw' in t]),
               chunks=describe([t['database']['chunks'] for t in traces if 'database' in t]),
               prompt_tokens=describe([t['token_preflight']['expected_prompt_tokens'] for t in traces if 'token_preflight' in t]),
               peak_per_claim_storage_bytes=describe([t['temporary_storage_bytes'] for t in traces if 'temporary_storage_bytes' in t]))
    with (out/'execution_audit.json').open('x') as stream:json.dump(audit,stream,indent=2)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
