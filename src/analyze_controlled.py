"""Analyze paired predictions and retrieval artifacts without model inference."""
import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from evaluate_retrieval import gold_evidence_groups
from experiment_symbolic_v2 import digest, load_inputs


def metrics(gold, predictions):
    return {'accuracy': float(accuracy_score(gold, predictions)),
            'macro_f1': float(f1_score(gold, predictions, labels=[0, 1, 2], average='macro', zero_division=0)),
            'confusion_matrix': confusion_matrix(gold, predictions, labels=[0, 1, 2]).tolist()}


def paired(gold, left, right, seed=42, samples=10000):
    delta = (np.asarray(right) == gold).astype(float) - (np.asarray(left) == gold).astype(float)
    rng = np.random.default_rng(seed)
    distribution = delta[rng.integers(0, len(gold), (samples, len(gold)))].mean(axis=1)
    wins, losses = int(sum(delta > 0)), int(sum(delta < 0))
    return {'right_minus_left_accuracy': float(delta.mean()),
            'paired_bootstrap_accuracy_ci95': np.quantile(distribution, [0.025, 0.975]).tolist(),
            'right_only_correct': wins, 'left_only_correct': losses,
            'mcnemar_exact_p_unadjusted': float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0,
            'bootstrap_seed': seed, 'bootstrap_samples': samples}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--retrieved', default='data/processed/retrieved/validation_first_100_top5_title_fast.jsonl')
    p.add_argument('--predictions', default='reports/part2/paired/predictions.csv')
    p.add_argument('--symbolic', required=True)
    p.add_argument('--gold', default='data/processed/validation.jsonl')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    records, _ = load_inputs(args.retrieved, args.predictions)
    ids = [str(r['id']) for r in records]
    gold = [int(r['label_id']) for r in records]
    with open(args.predictions, newline='') as stream:
        saved = list(csv.DictReader(stream))
    conditions = {}
    for condition in sorted({r['condition'] for r in saved}):
        rows = [r for r in saved if r['condition'] == condition]
        by_id = {str(r['id']): r for r in rows}
        if len(rows) != len(ids) or set(by_id) != set(ids):
            raise ValueError('Unpaired condition: ' + condition)
        if any(int(by_id[key]['gold_label_id']) != label for key, label in zip(ids, gold)):
            raise ValueError('Gold mismatch: ' + condition)
        conditions[condition] = [int(by_id[key]['predicted_label_id']) for key in ids]
    with open(args.symbolic, newline='') as stream:
        symbolic = list(csv.DictReader(stream))
    coverage = {}
    for condition in sorted({r['condition'] for r in symbolic}):
        rows = [r for r in symbolic if r['condition'] == condition]
        by_id = {str(r['id']): r for r in rows}
        if len(rows) != len(ids) or set(by_id) != set(ids):
            raise ValueError('Unpaired symbolic condition')
        for index, record in enumerate(records):
            row = by_id[str(record['id'])]
            if (row['claim'] != record['claim'] or int(row['gold_label_id']) != int(record['label_id'])
                    or int(row['baseline_prediction']) != conditions['retrieved_evidence'][index]):
                raise ValueError('Symbolic input mismatch')
        conditions['hybrid_' + condition] = [int(by_id[key]['hybrid_prediction']) for key in ids]
        resolved = [r for r in rows if r['symbolic_prediction'] != '']
        correct = sum(int(r['symbolic_prediction']) == int(r['gold_label_id']) for r in resolved)
        coverage[condition] = {'resolved': len(resolved), 'coverage': len(resolved) / len(ids),
                               'correct': correct, 'resolved_accuracy': correct / len(resolved) if resolved else None,
                               'counterfactual_calls_avoided': len(resolved),
                               'reason_counts': dict(Counter(r['reason'] for r in rows)), 'resolved_cases': resolved}
    comparisons = {a + '__vs__' + b: paired(gold, conditions[a], conditions[b])
                   for a, b in itertools.combinations(conditions, 2)}
    gold_records = {str(r['id']): r for r in map(json.loads, Path(args.gold).read_text().splitlines())}
    retrieval = {}
    for path in sorted(Path(args.retrieved).parent.glob('validation_first_100_top5_*.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != len(ids) or {str(r['id']) for r in rows} != set(ids):
            raise ValueError('Retrieval strategy IDs differ: ' + str(path))
        eligible = [r for r in rows if int(gold_records[str(r['id'])]['label_id']) != 2
                    and gold_evidence_groups(gold_records[str(r['id'])])]
        retrieval[path.stem] = {}
        for k in (1, 5):
            any_hits = complete_hits = 0
            for r in eligible:
                groups = gold_evidence_groups(gold_records[str(r['id'])])
                found = {(v['page_id'], int(v['sentence_id'])) for v in r['retrieved_evidence'][:k]}
                any_hits += any(found & group for group in groups)
                complete_hits += any(group <= found for group in groups)
            retrieval[path.stem][str(k)] = {'eligible': len(eligible), 'any_hits': any_hits,
                                          'complete_hits': complete_hits,
                                          'any_recall': any_hits / len(eligible) if eligible else None,
                                          'complete_recall': complete_hits / len(eligible) if eligible else None}
    diagnostic = Counter()
    for r, pred in zip(records, conditions['retrieved_evidence']):
        if int(r['label_id']) == 2:
            diagnostic['NEI_correct' if pred == 2 else 'NEI_wrong'] += 1
            continue
        found = {(v['page_id'], int(v['sentence_id'])) for v in r['retrieved_evidence']}
        groups = gold_evidence_groups(gold_records[str(r['id'])])
        complete = any(group <= found for group in groups)
        diagnostic[('complete_evidence' if complete else 'incomplete_evidence') +
                   ('_correct' if pred == int(r['label_id']) else '_wrong')] += 1
    report = {'n': len(ids), 'metrics': {c: metrics(gold, pred) for c, pred in conditions.items()},
              'paired_comparisons': comparisons, 'symbolic': coverage,
              'retrieval': retrieval, 'retrieval_classification_errors': dict(diagnostic),
              'input_sha256': {p: digest(p) for p in (args.retrieved, args.predictions, args.symbolic, args.gold)},
              'limitations': ['Exploratory development sample; no held-out or population improvement claim.',
                              'Bootstrap assumes independent rows; repeated claim text can violate this assumption.',
                              'Unadjusted p-values are descriptive; comparisons are multiple and not preregistered.',
                              'Top-5 artifacts cannot measure Recall@10.',
                              'Oracle empty NEI evidence can expose labels.',
                              'Different checkpoints/training inputs confound claim-only versus evidence comparisons.',
                              'Counterfactual avoided calls do not measure latency or cost savings.']}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({'metrics': report['metrics'], 'errors': dict(diagnostic)}, indent=2))


if __name__ == '__main__':
    main()
