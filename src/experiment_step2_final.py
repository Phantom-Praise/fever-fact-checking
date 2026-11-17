"""Frozen held-out comparison and actual online symbolic pipeline timing.

Writes its protocol and label-blind sample before any inference. No training,
rule tuning, gold-evidence input, model downloads, or test-driven selection.
"""
import argparse
import csv
import itertools
import json
import platform
import random
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from analyze_controlled import metrics, paired
from experiment_evidence_topk import retrieval_metrics
from experiment_paired_ablation import synchronize
from experiment_symbolic_v2 import SnapshotVerifier, digest
from retrieve_title_aware import retrieve

CHECKPOINTS = {'unweighted_reference': 'checkpoints/roberta_epoch_1',
               'claim_only_H2': 'checkpoints/tuning_weighted/H2_seed42/best',
               'retrieved_top5': 'checkpoints/roberta_evidence/epoch_1'}


def claim_key(text):
    return ' '.join(text.casefold().split())


def sample_records(records, count, seed, development):
    excluded_ids = {str(r['id']) for r in development}
    excluded_text = {claim_key(r['claim']) for r in development}
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    chosen, seen = [], set()
    for index in order:
        r = records[index]
        key = claim_key(r['claim'])
        if str(r['id']) in excluded_ids or key in excluded_text:
            raise ValueError('Held-out partition overlaps training/validation by ID or normalized text')
        if key not in seen:
            seen.add(key)
            chosen.append(r)
            if len(chosen) == count:
                break
    if len(chosen) != count:
        raise ValueError('Insufficient unique held-out claims')
    return chosen


def read_jsonl(path):
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def metric_summary(gold, predictions):
    precision, recall, f1, support = precision_recall_fscore_support(gold, predictions, labels=[0, 1, 2], zero_division=0)
    return {**metrics(gold, predictions), 'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()),
            'per_class': {str(i): {'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i])} for i in range(3)}}


def run_pipeline(claim, classify, retrieve_fn, verify_fn=None):
    started = time.perf_counter()
    decision, reason, failure = None, 'not_requested', ''
    symbolic_seconds = retrieval_seconds = inference_seconds = 0.0
    if verify_fn is not None:
        phase = time.perf_counter()
        try:
            decision, reason = verify_fn(claim)
        except RuntimeError as exc:
            failure = str(exc)
            reason = 'symbolic_service_failure_fallback'
        symbolic_seconds = time.perf_counter() - phase
    evidence = []
    if decision is None:
        phase = time.perf_counter()
        evidence = retrieve_fn(claim)
        retrieval_seconds = time.perf_counter() - phase
        phase = time.perf_counter()
        prediction = classify(claim, ' '.join(e['text'] for e in evidence))
        inference_seconds = time.perf_counter() - phase
    else:
        prediction = decision
    return {'prediction': prediction, 'symbolic_prediction': decision, 'reason': reason,
            'service_failure': failure, 'symbolic_seconds': symbolic_seconds,
            'retrieval_seconds': retrieval_seconds, 'inference_seconds': inference_seconds,
            'total_seconds': time.perf_counter() - started, 'retrieval_called': int(decision is None),
            'model_called': int(decision is None), 'evidence': evidence}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test', default='data/processed/test.jsonl')
    p.add_argument('--train', default='data/processed/train.jsonl')
    p.add_argument('--validation', default='data/processed/validation.jsonl')
    p.add_argument('--db', default='data/processed/wiki_fts.db')
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--seed', type=int, default=874)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if not 2 <= args.count <= 500:
        p.error('This bounded final study requires 2-500 claims')
    records = sample_records(read_jsonl(args.test), args.count, args.seed,
                             read_jsonl(args.train) + read_jsonl(args.validation))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    code = [__file__, 'src/retrieve_title_aware.py', 'src/experiment_symbolic_v2.py', 'src/experiment_symbolic_hybrid.py']
    protocol = {'status': 'frozen_before_inference', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'n': len(records), 'seed': args.seed, 'sample_ids': [r['id'] for r in records],
        'selection': 'Seeded shuffle of the existing held-out split; unique normalized claim text. Labels not used for sampling.',
        'conditions': ['unweighted_reference', 'claim_only_H2', 'retrieved_top5', 'hybrid_v2'],
        'checkpoints': CHECKPOINTS, 'max_length': 256, 'batch_size': 1, 'device': str(device),
        'pipeline_passes': ['initial_online', 'warm_snapshot_1', 'warm_snapshot_2'],
        'policy': 'One new snapshot for this study, initially empty. Initial-online pass includes live request overhead and within-pass caching. Later passes reuse responses.',
        'failure_policy': 'Record service failures explicitly, then fall back to retrieval/model. Never treat an outage as a factual abstention.',
        'primary_comparisons': ['retrieved_top5 versus hybrid_v2', 'claim_only_H2 versus retrieved_top5'],
        'source_sha256': {path: digest(path) for path in [args.test, args.train, args.validation, *code]},
        'checkpoint_config_sha256': {path: digest(Path(path) / 'config.json') for path in CHECKPOINTS.values()},
        'index': {'path': args.db, 'bytes': Path(args.db).stat().st_size, 'mtime_ns': Path(args.db).stat().st_mtime_ns},
        'runtime': {'python': sys.version, 'torch': torch.__version__, 'transformers': transformers.__version__, 'platform': platform.platform()},
        'notes': ['Small held-out sample, not full test-partition evaluation.', 'All choices fixed from earlier development results; do not tune on this sample.']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    (out / 'runner_source.py').write_text(Path(__file__).read_text())
    (out / 'claims_without_labels.jsonl').write_text(''.join(json.dumps({'id': r['id'], 'claim': r['claim']}) + '\n' for r in records))
    connection = sqlite3.connect(Path(args.db).resolve().as_uri() + '?mode=ro', uri=True)
    gold = [int(r['label_id']) for r in records]
    predictions, summaries, output_rows = {}, {}, []

    def load(checkpoint):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
        if model.config.num_labels != 3:
            raise ValueError('Expected three label logits')
        def classify(claim, evidence=None):
            tokens = tokenizer(claim, text_pair=evidence, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
            with torch.inference_mode():
                result = model(**{k: v.to(device) for k, v in tokens.items()}).logits.argmax(-1).item()
            synchronize(device)
            return result
        return model, classify

    try:
        for name in ('unweighted_reference', 'claim_only_H2'):
            model, classify = load(CHECKPOINTS[name])
            classify(records[0]['claim'])
            values = []
            for r in records:
                start = time.perf_counter()
                pred = classify(r['claim'])
                values.append(pred)
                output_rows.append({'pass': 'classification', 'condition': name, 'id': r['id'], 'gold_label_id': r['label_id'],
                                    'predicted_label_id': pred, 'total_seconds': time.perf_counter() - start})
            predictions[name] = values
            summaries[name] = metric_summary(gold, values)
            print(name, summaries[name]['accuracy'], summaries[name]['macro_f1'], flush=True)
            del model, classify
        model, classify = load(CHECKPOINTS['retrieved_top5'])
        classify(records[0]['claim'], '')
        retrieve_fn = lambda claim: retrieve(connection, claim, 5)
        verifier = SnapshotVerifier(out / 'wikidata_snapshot', timeout=8)
        trials = []
        initial_retrieval = {}
        phase_rows = []
        with (out / 'pipeline_events.jsonl').open('x') as events:
            for pass_index, pass_name in enumerate(protocol['pipeline_passes']):
                counts_before = (verifier.calls, verifier.failures, verifier.hits)
                values = {'retrieved_top5': [], 'hybrid_v2': []}
                for index, r in enumerate(records):
                    # Alternate which pipeline is timed first for each paired claim.
                    order = ('retrieved_top5', 'hybrid_v2') if (index + pass_index) % 2 == 0 else ('hybrid_v2', 'retrieved_top5')
                    for condition in order:
                        result = run_pipeline(r['claim'], classify, retrieve_fn,
                                              verifier.verify if condition == 'hybrid_v2' else None)
                        values[condition].append((index, result['prediction']))
                        event = {'pass': pass_name, 'condition': condition, 'id': r['id'], 'gold_label_id': r['label_id'], **result}
                        events.write(json.dumps(event) + '\n'); events.flush()
                        phase_rows.append(event)
                        output_rows.append({'pass': pass_name, 'condition': condition, 'id': r['id'], 'gold_label_id': r['label_id'],
                                            'predicted_label_id': result['prediction'], 'total_seconds': result['total_seconds']})
                        if pass_index == 0 and condition == 'retrieved_top5':
                            initial_retrieval[str(r['id'])] = result['evidence']
                    if (index + 1) % 10 == 0:
                        print(pass_name, index + 1, '/', len(records), 'HTTP', verifier.calls, 'failures', verifier.failures, flush=True)
                for condition in ('retrieved_top5', 'hybrid_v2'):
                    pred = [v for _, v in sorted(values[condition])]
                    if pass_index == 0:
                        predictions[condition] = pred
                        summaries[condition] = metric_summary(gold, pred)
                    selected = [r for r in phase_rows if r['pass'] == pass_name and r['condition'] == condition]
                    times = [r['total_seconds'] for r in selected]
                    trials.append({'pass': pass_name, 'condition': condition, **metric_summary(gold, pred),
                        'mean_seconds': float(np.mean(times)), 'p50_seconds': float(np.quantile(times, .5)),
                        'p95_seconds': float(np.quantile(times, .95)), 'total_seconds': sum(times),
                        'claims_per_second': len(records) / sum(times),
                        'symbolic_seconds': sum(r['symbolic_seconds'] for r in selected),
                        'retrieval_seconds': sum(r['retrieval_seconds'] for r in selected),
                        'inference_seconds': sum(r['inference_seconds'] for r in selected),
                        'resolved': sum(r['symbolic_prediction'] is not None for r in selected),
                        'retrieval_calls': sum(r['retrieval_called'] for r in selected),
                        'model_calls': sum(r['model_called'] for r in selected),
                        'service_failure_claims': sum(bool(r['service_failure']) for r in selected),
                        'http_attempts_in_pass': verifier.calls - counts_before[0],
                        'http_failures_in_pass': verifier.failures - counts_before[1],
                        'snapshot_hits_in_pass': verifier.hits - counts_before[2]})
        retrieved = [{**r, 'retrieved_evidence': initial_retrieval[str(r['id'])]} for r in records]
        (out / 'retrieved_with_gold.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in retrieved))
        with (out / 'predictions.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output_rows[0])); writer.writeheader(); writer.writerows(output_rows)
        summary = {'n': len(records), 'label_counts': dict(Counter(gold)), 'conditions': summaries,
            'trials': trials, 'retrieval': [retrieval_metrics(retrieved, k) for k in (1, 5)],
            'paired_comparisons': {a + '__vs__' + b: paired(gold, predictions[a], predictions[b], args.seed)
                                   for a, b in itertools.combinations(predictions, 2)},
            'protocol_sha256': digest(out / 'protocol.json'),
            'snapshot_sha256': {p.name: digest(p) for p in sorted((out / 'wikidata_snapshot').glob('*.json'))},
            'limitations': ['Only a bounded held-out sample; no full test-partition claim.',
                'No rule/model selection changes after seeing these outputs.',
                'Initial online timing includes networking/backoff; model loading is excluded.',
                'Shared OS caches, repeated claims across passes, and only three passes limit timing inference.',
                'Symbolic service failures are explicitly counted operational fallbacks, not factual abstentions.',
                'Wikidata facts may differ from historical FEVER labels.',
                'Unadjusted paired p-values and row bootstrap intervals remain exploratory.']}
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        (out / 'COMPLETE.json').write_text(json.dumps({'status': 'complete', 'summary_sha256': digest(out / 'summary.json'), 'finished_utc': datetime.now(timezone.utc).isoformat()}, indent=2))
        print(json.dumps({'conditions': summaries, 'trials': trials}, indent=2))
    finally:
        connection.close()


if __name__ == '__main__':
    main()
