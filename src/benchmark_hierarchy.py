"""RQ4 local, warm-cache pipeline benchmark on read-only SQLite indexes.

Repeat --db for ALREADY BUILT corpus sizes. Never constructs or modifies indexes.
Symbolic decisions are replayed from a frozen run; this is not live KB latency.
"""
import argparse
import csv
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from analyze_controlled import metrics
from experiment_paired_ablation import synchronize
from experiment_symbolic_v2 import digest, load_inputs
from retrieve_title_aware import retrieve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', action='append', required=True)
    p.add_argument('--symbolic', required=True)
    p.add_argument('--retrieved', default='data/processed/retrieved/validation_first_100_top5_title_fast.jsonl')
    p.add_argument('--predictions', default='reports/part2/paired/predictions.csv')
    p.add_argument('--checkpoint', default='checkpoints/roberta_evidence/epoch_1')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.repeats < 2:
        p.error('Use at least two repeats')
    records, _ = load_inputs(args.retrieved, args.predictions)
    with open(args.symbolic, newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['condition'] == 'v2']
    route = {str(r['id']): r for r in rows}
    if len(rows) != len(records) or set(route) != {str(r['id']) for r in records}:
        raise ValueError('Symbolic IDs differ')
    for r in records:
        saved = route[str(r['id'])]
        if saved['claim'] != r['claim'] or int(saved['gold_label_id']) != int(r['label_id']):
            raise ValueError('Symbolic claim/label mismatch')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, local_files_only=True).to(device).eval()

    def classify(claim, evidence):
        inputs = tokenizer(claim, evidence, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
        with torch.inference_mode():
            pred = model(**{k: v.to(device) for k, v in inputs.items()}).logits.argmax(-1).item()
        synchronize(device)
        return pred

    classify(records[0]['claim'], records[0]['evidence_text'])
    samples, summaries = [], []
    for db in args.db:
        conn = sqlite3.connect(Path(db).resolve().as_uri() + '?mode=ro', uri=True)
        # Warm both the retrieval path and the classifier before timed trials.
        retrieve(conn, records[0]['claim'], 5)
        for repeat in range(args.repeats):
            order = ('baseline', 'cached_symbolic') if repeat % 2 == 0 else ('cached_symbolic', 'baseline')
            for condition in order:
                predictions, elapsed_times = [], []
                avoided = 0
                for r in records:
                    start = time.perf_counter()
                    decision = route[str(r['id'])]['symbolic_prediction'] if condition == 'cached_symbolic' else ''
                    if decision != '':
                        pred = int(decision)
                        avoided += 1
                    else:
                        evidence = retrieve(conn, r['claim'], 5)
                        pred = classify(r['claim'], ' '.join(e['text'] for e in evidence))
                    elapsed = time.perf_counter() - start
                    elapsed_times.append(elapsed)
                    predictions.append(pred)
                    samples.append({'db': db, 'repeat': repeat, 'condition': condition,
                                    'id': r['id'], 'prediction': pred, 'seconds': elapsed})
                summary = {'db': db, 'db_bytes': Path(db).stat().st_size, 'db_mtime_ns': Path(db).stat().st_mtime_ns,
                           'repeat': repeat, 'condition': condition, 'n': len(records),
                           'retrieval_calls': len(records) - avoided, 'transformer_calls': len(records) - avoided,
                           'total_seconds': sum(elapsed_times), 'mean_seconds': float(np.mean(elapsed_times)),
                           'p50_seconds': float(np.quantile(elapsed_times, .5)),
                           'p95_seconds': float(np.quantile(elapsed_times, .95)),
                           'claims_per_second': len(records) / sum(elapsed_times),
                           **metrics([int(r['label_id']) for r in records], predictions)}
                summaries.append(summary)
                print(db, repeat, condition, summary['mean_seconds'], flush=True)
        conn.close()
    with (out / 'samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    (out / 'summary.json').write_text(json.dumps({
        'device': str(device), 'checkpoint': args.checkpoint, 'max_length': 256, 'batch_size': 1,
        'repeats': args.repeats, 'results': summaries,
        'input_sha256': {p: digest(p) for p in (args.retrieved, args.symbolic, __file__)},
        'limitations': ['Warm local cache, serialized per-claim inference; model loading excluded.',
                       'Symbolic route is a saved decision lookup: excludes KB network and rule execution.',
                       'One corpus size cannot establish scalability. Repeat with independently built nested indexes.',
                       'No monetary cost or live network hybrid latency is measured.',
                       'Condition order alternates; OS cache and thermal effects remain.']}, indent=2))


if __name__ == '__main__':
    main()
