"""Calibrate cheap pipeline improvements on validation, then assess a separate slice.

Does not read test.jsonl, train Transformer weights, overwrite original results,
or represent the lightweight ranker as an implementation of GEAR/KGAT.
"""
import argparse
import csv
import itertools
import json
import random
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from analyze_controlled import metrics, paired
from evidence_pipeline_v2 import evidence_text, fuse_probabilities, retrieve_bm25
from experiment_evidence_topk import retrieval_metrics
from experiment_paired_ablation import synchronize
from experiment_symbolic_v2 import digest
from retrieve_title_aware import retrieve

VARIANTS = ['original', 'local_bm25', 'subject_bm25', 'subject_bm25_titles']


def split_validation(records, calibration_count, assessment_count):
    calibration, assessment = [], []
    seen = set()
    # All earlier first-2000 tuning claims are excluded from assessment by text.
    prior = {' '.join(r['claim'].casefold().split()) for r in records[:2000]}
    for r in records[:2000]:
        key = ' '.join(r['claim'].casefold().split())
        if key not in seen and len(calibration) < calibration_count:
            calibration.append(r); seen.add(key)
    for r in records[2000:]:
        key = ' '.join(r['claim'].casefold().split())
        if key not in prior and key not in seen and len(assessment) < assessment_count:
            assessment.append(r); seen.add(key)
    if len(calibration) != calibration_count or len(assessment) != assessment_count:
        raise ValueError('Insufficient disjoint validation claims')
    return calibration, assessment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation', default='data/processed/validation.jsonl')
    p.add_argument('--db', default='data/processed/wiki_fts.db')
    p.add_argument('--calibration-count', type=int, default=300)
    p.add_argument('--assessment-count', type=int, default=200)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if min(args.calibration_count, args.assessment_count) < 1 or args.calibration_count > 1000 or args.assessment_count > 1000:
        p.error('Use 1-1000 claims for each bounded validation phase')
    records = [json.loads(line) for line in Path(args.validation).read_text().splitlines() if line.strip()]
    calibration, assessment = split_validation(records, args.calibration_count, args.assessment_count)
    all_records = calibration + assessment
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    claim_checkpoint = 'checkpoints/tuning_weighted/H2_seed42/best'
    evidence_checkpoint = 'checkpoints/roberta_evidence/epoch_1'
    protocol = {'calibration_ids': [r['id'] for r in calibration], 'assessment_ids': [r['id'] for r in assessment],
        'selection': 'First unique calibration claims within validation first 2000; assessment scans beyond raw offset 2000 and excludes all prior normalized texts.',
        'variants': VARIANTS, 'claim_weights': [0, .25, .5, .75, 1], 'nei_biases': [0, 1, 2, 3],
        'selection_metric': 'calibration macro-F1, then accuracy; ties prefer smaller absolute NEI bias and fewer changes from original retrieval.',
        'seed': 42, 'batch_size': 8, 'max_length': 256, 'device': str(device), 'test_partition_read': False,
        'checkpoint_paths': [claim_checkpoint, evidence_checkpoint],
        'input_sha256': {path: digest(path) for path in [args.validation, __file__, 'src/evidence_pipeline_v2.py', 'src/retrieve_title_aware.py']},
        'literature': {'FEVER': 'https://aclanthology.org/N18-1074/', 'GEAR': 'https://aclanthology.org/P19-1085/',
                       'KGAT': 'https://aclanthology.org/2020.acl-main.655/'},
        'notes': ['Evidence relevance motivates ranker ablations; no graph network is trained or reproduced.',
                  'NEI score calibration is a new engineering hypothesis, not a reported GEAR/KGAT algorithm.',
                  'Assessment metrics are computed only after selected_config.json is written.']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    (out / 'runner_source.py').write_text(Path(__file__).read_text())
    (out / 'evidence_pipeline_source.py').write_text(Path('src/evidence_pipeline_v2.py').read_text())
    connection = sqlite3.connect(Path(args.db).resolve().as_uri() + '?mode=ro', uri=True)
    evidence = {v: [] for v in VARIANTS}; retrieval_by_variant = {v: [] for v in VARIANTS}
    start = time.perf_counter()
    try:
        with (out / 'retrieval.jsonl').open('x') as stream:
            for index, r in enumerate(all_records, 1):
                original = retrieve(connection, r['claim'], 10)
                local = retrieve_bm25(connection, r['claim'], 10)
                subject = retrieve_bm25(connection, r['claim'], 10, link_subject=True)
                choices = {'original': original, 'local_bm25': local, 'subject_bm25': subject, 'subject_bm25_titles': subject}
                for variant, rows in choices.items():
                    retrieval_by_variant[variant].append(rows)
                    evidence[variant].append(evidence_text(rows[:5], include_titles=variant.endswith('_titles')))
                stream.write(json.dumps({'id': r['id'], 'claim': r['claim'], 'choices': choices}) + '\n')
                if index % 100 == 0: print('Retrieved', index, '/', len(all_records), flush=True)
    finally:
        connection.close()
    retrieval_seconds = time.perf_counter() - start

    def infer(checkpoint, inputs):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
        predictions = {}
        for name, text in inputs.items():
            values = []
            with torch.inference_mode():
                for index in range(0, len(all_records), 8):
                    claims = [r['claim'] for r in all_records[index:index + 8]]
                    pairs = None if text is None else text[index:index + 8]
                    encoded = tokenizer(claims, text_pair=pairs, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
                    logits = model(**{k: v.to(device) for k, v in encoded.items()}).logits
                    values.extend(torch.softmax(logits, -1).cpu().tolist())
            synchronize(device)
            predictions[name] = np.array(values)
            print('Inferred', name, flush=True)
        del model
        return predictions

    claim = infer(claim_checkpoint, {'claim_only': None})['claim_only']
    probability = infer(evidence_checkpoint, evidence)
    n = len(calibration); cal_gold = np.array([int(r['label_id']) for r in calibration])
    candidates = []
    for vi, variant in enumerate(VARIANTS):
        for weight, bias in itertools.product(protocol['claim_weights'], protocol['nei_biases']):
            pred = fuse_probabilities(claim[:n], probability[variant][:n], weight, bias)
            score = metrics(cal_gold, pred)
            candidates.append({'variant': variant, 'claim_weight': weight, 'nei_bias': bias, 'accuracy': score['accuracy'],
                               'macro_f1': score['macro_f1'], 'variant_index': vi})
    chosen = max(candidates, key=lambda r: (r['macro_f1'], r['accuracy'], -abs(r['nei_bias']), -r['variant_index']))
    selected = {**chosen, 'status': 'frozen_before_assessment_scoring', 'calibration_n': n}
    (out / 'selected_config.json').write_text(json.dumps(selected, indent=2))
    (out / 'calibration_candidates.json').write_text(json.dumps(candidates, indent=2))
    np.savez_compressed(out / 'probabilities.npz', claim=claim, **probability)
    # No assessment labels are used above this line for configuration selection.
    gold = np.array([int(r['label_id']) for r in assessment])
    conditions = {'claim_only_H2': claim[n:].argmax(1)}
    for variant in VARIANTS: conditions['evidence_' + variant] = probability[variant][n:].argmax(1)
    conditions['selected_fusion'] = fuse_probabilities(claim[n:], probability[chosen['variant']][n:], chosen['claim_weight'], chosen['nei_bias'])
    # Ablations isolate NEI calibration from fusion under the selected evidence condition.
    conditions['selected_weight_without_nei_bias'] = fuse_probabilities(claim[n:], probability[chosen['variant']][n:], chosen['claim_weight'], 0)
    conditions['selected_bias_without_claim_fusion'] = fuse_probabilities(claim[n:], probability[chosen['variant']][n:], 0, chosen['nei_bias'])
    summaries = {}
    for name, pred in conditions.items():
        precision, recall, f1, support = precision_recall_fscore_support(gold, pred, labels=[0, 1, 2], zero_division=0)
        summaries[name] = {**metrics(gold, pred), 'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()),
                           'nei_recall': float(recall[2]), 'prediction_counts': {str(k): v for k, v in Counter(pred.tolist()).items()},
                           'per_class': {str(i): {'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i])} for i in range(3)}}
        print(name, summaries[name]['accuracy'], summaries[name]['macro_f1'], 'NEI recall', summaries[name]['nei_recall'], flush=True)
    with (out / 'assessment_predictions.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['condition', 'id', 'gold_label_id', 'predicted_label_id']); writer.writeheader()
        for name, pred in conditions.items():
            writer.writerows({'condition': name, 'id': r['id'], 'gold_label_id': int(r['label_id']), 'predicted_label_id': int(value)} for r, value in zip(assessment, pred))
    recalls = {}
    for variant in VARIANTS:
        eval_records = [{**r, 'retrieved_evidence': rows} for r, rows in zip(assessment, retrieval_by_variant[variant][n:])]
        recalls[variant] = [retrieval_metrics(eval_records, k) for k in (1, 5, 10)]
    result = {'assessment_n': len(assessment), 'selected_config': selected, 'conditions': summaries, 'retrieval': recalls,
        'paired': {baseline + '__vs__selected_fusion': paired(gold, conditions[baseline], conditions['selected_fusion']) for baseline in ('claim_only_H2', 'evidence_original')},
        'retrieval_seconds_all_variants': retrieval_seconds,
        'limitations': ['Validation calibration and separate assessment, not a new untouched test result.',
                       'The calibration population lies within earlier model-selection validation data.',
                       '80 small decision/ranker combinations can still overfit calibration; assessment is not used for selection.',
                       'Fusion runs two models and is not automatically an efficiency improvement.',
                       'NEI score calibration does not repair the underlying evidence-training mismatch.',
                       'Retrieval-only gains do not establish verification gains.',
                       'No retraining, graph model, dense retrieval model, or symbolic rule changes were made.']}
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    print('Selected', json.dumps(selected), flush=True)


if __name__ == '__main__':
    main()
