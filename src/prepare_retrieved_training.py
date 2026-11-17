"""Prepare a bounded, provenance-preserving retrieved-evidence training pilot.

NEI uses real retrieved text instead of a label-revealing empty string. Human
labels are retained; verifiable claims without a complete annotated group in
the input are skipped, never relabeled. Does not train or overwrite anything.
"""
import argparse
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

from evidence_pipeline_v2 import evidence_text, retrieve_bm25
from evaluate_retrieval import gold_evidence_groups
from experiment_symbolic_v2 import digest


def training_eligibility(record, evidence):
    if not evidence:
        return False, 'empty_retrieval'
    if int(record['label_id']) == 2:
        return True, 'human_NEI_with_nonempty_retrieval'
    found = {(r['page_id'], int(r['sentence_id'])) for r in evidence}
    if any(group <= found for group in gold_evidence_groups(record)):
        return True, 'complete_annotated_group_present'
    return False, 'verifiable_claim_missing_complete_group'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', default='data/processed/train.jsonl')
    p.add_argument('--db', default='data/processed/wiki_fts.db')
    p.add_argument('--per-class', type=int, default=32)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if not 1 <= args.per_class <= 1000:
        p.error('Use 1-1000 candidate claims per class; full training preparation is a separate explicit operation')
    records = [json.loads(line) for line in Path(args.train).read_text().splitlines() if line.strip()]
    selected = []
    rng = random.Random(42)
    for label in range(3):
        candidates = [r for r in records if int(r['label_id']) == label]
        rng.shuffle(candidates)
        selected += candidates[:args.per_class]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    connection = sqlite3.connect(Path(args.db).resolve().as_uri() + '?mode=ro', uri=True)
    accepted, audit = [], []
    try:
        for record in selected:
            evidence = retrieve_bm25(connection, record['claim'], 5, link_subject=True)
            eligible, reason = training_eligibility(record, evidence)
            row = {'id': record['id'], 'claim': record['claim'], 'label_id': int(record['label_id']),
                   'evidence_text': evidence_text(evidence, include_titles=True), 'retrieved_evidence': evidence,
                   'eligible': eligible, 'reason': reason}
            audit.append(row)
            if eligible: accepted.append(row)
    finally:
        connection.close()
    (out / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in accepted))
    (out / 'candidate_audit.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in audit))
    (out / 'manifest.json').write_text(json.dumps({'candidate_n': len(selected), 'accepted_n': len(accepted),
        'accepted_class_counts': dict(Counter(r['label_id'] for r in accepted)), 'reasons': dict(Counter(r['reason'] for r in audit)),
        'seed': 42, 'input_sha256': {p: digest(p) for p in (args.train, __file__, 'src/evidence_pipeline_v2.py')},
        'policy': ['Only the original training partition is sampled.', 'No label is inferred from retrieval failure.',
                   'Nonempty NEI evidence removes the empty-input shortcut opportunity.',
                   'Filtering verifiable claims creates a retrieval-success bias; report this selection explicitly.',
                   'This is a data-format pilot, not a trained or validated replacement model.']}, indent=2))
    print('Prepared', len(accepted), 'eligible training examples from', len(selected), 'candidates; no training performed')


if __name__ == '__main__':
    main()
