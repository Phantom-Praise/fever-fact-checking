"""Frozen-checkpoint evidence-depth and paired-ablation development experiments.

Select a validation slice by offset; never reads the test partition or trains.
All evidence depths share one top-10 retrieval and identical IDs/checkpoints.
"""
import argparse
import csv
import itertools
import json
import random
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from analyze_controlled import metrics, paired
from evaluate_retrieval import gold_evidence_groups
from experiment_paired_ablation import synchronize
from experiment_symbolic_v2 import digest
from retrieve_title_aware import retrieve


def read_slice(path, offset, count, excluded_ids=(), excluded_claims=()):
    selected = []
    seen_ids, seen_claims = set(excluded_ids), set(excluded_claims)
    with open(path, encoding='utf-8') as stream:
        for source_index, record in enumerate(json.loads(line) for line in stream if line.strip()):
            if source_index < offset:
                continue
            key, text = str(record['id']), record['claim'].casefold().strip()
            if key in seen_ids or text in seen_claims:
                continue
            seen_ids.add(key)
            seen_claims.add(text)
            selected.append(record)
            if len(selected) == count:
                break
    if len(selected) != count or len({str(r['id']) for r in selected}) != count:
        raise ValueError('Requested slice unavailable or has duplicate IDs')
    if len({r['claim'].casefold().strip() for r in selected}) != count:
        raise ValueError('Repeated claim text in evaluation slice')
    return selected


def retrieval_metrics(records, k):
    eligible = [r for r in records if int(r['label_id']) != 2 and gold_evidence_groups(r)]
    any_hits = complete_hits = 0
    for r in eligible:
        found = {(e['page_id'], int(e['sentence_id'])) for e in r['retrieved_evidence'][:k]}
        groups = gold_evidence_groups(r)
        any_hits += any(found & g for g in groups)
        complete_hits += any(g <= found for g in groups)
    return {'k': k, 'eligible': len(eligible), 'any_hits': any_hits, 'complete_hits': complete_hits,
            'any_recall': any_hits / len(eligible) if eligible else None,
            'complete_recall': complete_hits / len(eligible) if eligible else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation', default='data/processed/validation.jsonl')
    p.add_argument('--db', default='data/processed/wiki_fts.db')
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--claim-checkpoint', default='checkpoints/tuning_weighted/H2_seed42/best')
    p.add_argument('--evidence-checkpoint', default='checkpoints/roberta_evidence/epoch_1')
    p.add_argument('--gold-evidence', default='data/processed/evidence/validation.jsonl')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--reference', help='Optional top-5 artifact; exact IDs/text/evidence prefix required')
    p.add_argument('--exclude', help='Optional artifact whose IDs and claim texts must be disjoint')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.offset < 0 or args.count < 2 or args.batch_size < 1:
        p.error('offset >= 0, count >= 2, batch-size >= 1 required')
    previous = []
    if args.exclude:
        previous = [json.loads(line) for line in Path(args.exclude).read_text().splitlines() if line.strip()]
    records = read_slice(args.validation, args.offset, args.count,
                         [str(r['id']) for r in previous], [r['claim'].casefold().strip() for r in previous])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'runner_source.py').write_text(Path(__file__).read_text())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    conn = sqlite3.connect(Path(args.db).resolve().as_uri() + '?mode=ro', uri=True)
    start = time.perf_counter()
    try:
        for r in records:
            r['retrieved_evidence'] = retrieve(conn, r['claim'], 10)
    finally:
        conn.close()
    retrieval_seconds = time.perf_counter() - start
    if args.reference:
        reference = [json.loads(line) for line in Path(args.reference).read_text().splitlines() if line.strip()]
        by_id = {str(r['id']): r for r in reference}
        if len(reference) != len(records) or set(by_id) != {str(r['id']) for r in records}:
            raise ValueError('Reference IDs differ')
        for r in records:
            saved = by_id[str(r['id'])]
            if (saved['claim'] != r['claim'] or int(saved['label_id']) != int(r['label_id'])
                    or saved['retrieved_evidence'] != r['retrieved_evidence'][:5]):
                raise ValueError('Reference evidence differs; cannot reuse paired comparison')
    (out / 'retrieved_top10.jsonl').write_text(''.join(json.dumps({**r, 'evidence_text': ' '.join(e['text'] for e in r['retrieved_evidence'])}) + '\n' for r in records))
    gold_by_id = {str(r['id']): r for r in map(json.loads, Path(args.gold_evidence).read_text().splitlines())}
    oracle_count = sum(str(r['id']) in gold_by_id for r in records)
    gold = [int(r['label_id']) for r in records]
    inputs = {f'retrieved_k{k}': [' '.join(e['text'] for e in r['retrieved_evidence'][:k]) for r in records] for k in (1, 5, 10)}
    inputs['empty_evidence'] = [''] * len(records)
    inputs['wrong_claim_evidence'] = inputs['retrieved_k5'][1:] + inputs['retrieved_k5'][:1]
    if oracle_count == len(records):
        inputs['gold_evidence_oracle'] = [gold_by_id[str(r['id'])]['evidence_text'] for r in records]
    else:
        print(f'Skipping oracle: evidence available for {oracle_count}/{len(records)} claims', flush=True)
    conditions = [('claim_only_H2', args.claim_checkpoint, None)] + [(c, args.evidence_checkpoint, e) for c, e in inputs.items()]
    predictions, summaries, prediction_rows = {}, [], []
    active_checkpoint = None
    for name, checkpoint, evidence in conditions:
        if checkpoint != active_checkpoint:
            if active_checkpoint is not None:
                del model
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
            if model.config.num_labels != 3:
                raise ValueError('Expected three FEVER labels')
            active_checkpoint = checkpoint
        predicted = []
        synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            for index in range(0, len(records), args.batch_size):
                claims = [r['claim'] for r in records[index:index + args.batch_size]]
                batch_evidence = None if evidence is None else evidence[index:index + args.batch_size]
                encoded = tokenizer(claims, text_pair=batch_evidence, padding='max_length', truncation=True,
                                    max_length=256, return_tensors='pt')
                predicted.extend(model(**{k: v.to(device) for k, v in encoded.items()}).logits.argmax(-1).cpu().tolist())
        synchronize(device)
        seconds = time.perf_counter() - start
        precision, recall, f1, support = precision_recall_fscore_support(gold, predicted, labels=[0, 1, 2], zero_division=0)
        summary = {'condition': name, 'checkpoint': checkpoint, 'n': len(records),
                   **metrics(gold, predicted), 'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()),
                   'per_class': {str(i): {'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i])} for i in range(3)},
                   'inference_seconds': seconds, 'prediction_counts': dict(Counter(predicted))}
        summaries.append(summary)
        predictions[name] = predicted
        for r, pred in zip(records, predicted):
            prediction_rows.append({'condition': name, 'id': r['id'], 'gold_label_id': r['label_id'],
                                    'predicted_label_id': pred, 'correct': int(pred == int(r['label_id']))})
        print(name, 'accuracy=', summary['accuracy'], 'macro_f1=', summary['macro_f1'], flush=True)
    with (out / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    comparisons = {a + '__vs__' + b: paired(gold, predictions[a], predictions[b], args.seed)
                   for a, b in itertools.combinations(predictions, 2)}
    files = [args.validation, args.gold_evidence, __file__, 'src/retrieve_title_aware.py']
    files += [p for p in (args.reference, args.exclude) if p]
    summary = {'n': len(records), 'offset': args.offset, 'seed': args.seed, 'device': str(device),
               'batch_size': args.batch_size, 'max_length': 256, 'conditions': summaries,
               'retrieval': [retrieval_metrics(records, k) for k in (1, 5, 10)],
               'retrieval_top10_seconds': retrieval_seconds, 'paired_comparisons': comparisons,
               'oracle_evidence_available': oracle_count,
               'input_sha256': {p: digest(p) for p in files},
               'checkpoint_config_sha256': {p: digest(Path(p) / 'config.json') for p in (args.claim_checkpoint, args.evidence_checkpoint)},
               'db': {'path': args.db, 'bytes': Path(args.db).stat().st_size, 'mtime_ns': Path(args.db).stat().st_mtime_ns},
               'runtime': {'python': sys.version, 'torch': torch.__version__, 'transformers': transformers.__version__},
               'sampling': 'Scan from raw validation offset, skip excluded IDs/text and within-sample duplicate casefolded claim text, retain next count records.',
               'limitations': ['Development validation, including the tuning pool; not a held-out final test.',
                               'Frozen checkpoint inference is not an independent training-seed repeat.',
                               'Oracle may leak labels through empty NEI evidence.',
                               'Claim-only versus evidence conditions differ in training/checkpoint.',
                               'Timing excludes model loading and is not an end-to-end latency benchmark.',
                               'Bootstrap and unadjusted multiple p-values are exploratory.']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
