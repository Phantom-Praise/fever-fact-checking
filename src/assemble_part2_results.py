"""Assemble source-linked Part 2 tables without mixing evaluation populations."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from analyze_controlled import metrics, paired
from experiment_symbolic_v2 import digest


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_predictions(path, scope):
    saved = read_csv(path)
    rows = []
    for name in dict.fromkeys(r['condition'] for r in saved):
        selected = [r for r in saved if r['condition'] == name]
        gold = [int(r['gold_label_id']) for r in selected]
        pred = [int(r.get('predicted_label_id', r.get('hybrid_prediction'))) for r in selected]
        precision, recall, _, _ = precision_recall_fscore_support(gold, pred, labels=[0, 1, 2], zero_division=0)
        rows.append({'scope': scope, 'condition': name, 'n': len(selected),
                     'accuracy': metrics(gold, pred)['accuracy'], 'macro_precision': float(precision.mean()),
                     'macro_recall': float(recall.mean()), 'macro_f1': metrics(gold, pred)['macro_f1'],
                     'source': str(path)})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    sources, runs = [], []
    for config in ['H1', 'H2', 'H3', 'H4']:
        path = Path(f'reports/part2/tuning/{config}_seed42.csv')
        sources.append(path)
        for row in read_csv(path):
            runs.append({'family': 'claim_only_tuned', 'run': config + '_seed42', 'epoch': int(row['epoch']),
                         'seed': row['seed'], 'learning_rate': row['learning_rate'], 'weight_decay': row['weight_decay'],
                         'batch_size': row['batch_size'], 'max_length': 256, 'train_fraction': .2,
                         'evaluation': 'first_2000_validation_claims', 'n': 2000,
                         'accuracy': float(row['val_accuracy']), 'macro_f1': float(row['val_macro_f1']),
                         'selected': config == 'H2' and int(row['epoch']) == 2,
                         'source': str(path)})
    for family, path, chosen_epoch, wd in [
            ('claim_only_weighted_legacy', Path('results/weighted_training_log.csv'), None, .01),
            ('gold_evidence_roberta', Path('results/main_training_log.csv'), 1, .01)]:
        sources.append(path)
        for row in read_csv(path):
            runs.append({'family': family, 'run': family + '_seed42', 'epoch': int(row['epoch']), 'seed': 42,
                         'learning_rate': 2e-5, 'weight_decay': wd, 'batch_size': 8, 'max_length': 256,
                         'train_fraction': .2, 'evaluation': 'first_2000_validation_' + ('gold_evidence' if family.startswith('gold') else 'claims'),
                         'n': 2000, 'accuracy': float(row['val_accuracy']), 'macro_f1': float(row['val_macro_f1']),
                         'selected': int(row['epoch']) == chosen_epoch, 'source': str(path)})
    path = Path('results/training_log.csv')
    sources.append(path)
    for index, row in enumerate(read_csv(path), 1):
        runs.append({'family': 'unweighted_legacy', 'run': 'legacy_log_observation_' + str(index),
                     'epoch': int(row['epoch']), 'seed': '', 'learning_rate': '', 'weight_decay': '', 'batch_size': '',
                     'max_length': '', 'train_fraction': '', 'evaluation': 'legacy_validation_no_run_manifest', 'n': '',
                     'accuracy': float(row['validation_accuracy']), 'macro_f1': float(row['validation_macro_f1']),
                     'selected': False, 'source': str(path)})
    write_csv(out / 'all_training_runs.csv', runs)
    evaluation_sources = [
        (Path('reports/part2/paired/predictions.csv'), 'validation_first100_saved'),
        (Path('reports/part2/symbolic_v2/run_20261023_retry/predictions.csv'), 'validation_first100_symbolic'),
        (Path('reports/part2/evidence_depth_20261023/predictions.csv'), 'validation_first100_topk'),
        (Path('reports/part2/validation_extension_20261023_complete/predictions.csv'), 'validation_extension100_disjoint_text'),
        (Path('reports/part2/local_llm_20261023/pilot_baseline_predictions.csv'), 'balanced30_judge_pilot')]
    evaluated = []
    for path, scope in evaluation_sources:
        sources.append(path)
        evaluated.extend(summarize_predictions(path, scope))
    reference_path = Path('reports/part2/reference_checkpoints_20261023/summary.json')
    if reference_path.exists():
        sources.append(reference_path)
        for r in json.loads(reference_path.read_text())['conditions']:
            evaluated.append({'scope': 'validation_' + r['scope'] + '_reference', 'condition': r['condition'],
                              **{key: r[key] for key in ('n', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1')},
                              'source': str(reference_path)})
    write_csv(out / 'all_classifier_evaluations.csv', evaluated)
    judge_path = Path('reports/part2/local_llm_20261023/judge_scores.json')
    if judge_path.exists():
        sources.append(judge_path)
        scores = json.loads(judge_path.read_text())
        judge_rows = []
        for model, model_score in scores['models'].items():
            for condition, summary in model_score['conditions'].items():
                for repeat, result in summary['per_repeat'].items():
                    matrix = np.array(result['confusion_matrix'], dtype=float)
                    diagonal = matrix.diagonal()
                    precision = np.divide(diagonal, matrix.sum(0), out=np.zeros(3), where=matrix.sum(0) != 0).mean()
                    recall = np.divide(diagonal, matrix.sum(1), out=np.zeros(3), where=matrix.sum(1) != 0).mean()
                    judge_rows.append({'scope': 'balanced30_judge_pilot', 'model': model, 'condition': condition,
                                       'repeat': repeat, 'n': summary['n_unique_claims'], 'accuracy': result['accuracy'],
                                       'macro_f1': result['macro_f1'], 'macro_precision': precision, 'macro_recall': recall,
                                       'all_repeats_agree_fraction': summary['all_repeats_agree_fraction'], 'source': str(judge_path)})
        write_csv(out / 'all_judge_runs.csv', judge_rows)
    retrieval_rows = []
    for name in ('evidence_depth_20261023', 'validation_extension_20261023_complete'):
        path = Path('reports/part2') / name / 'summary.json'
        sources.append(path)
        for row in json.loads(path.read_text())['retrieval']:
            retrieval_rows.append({'scope': name, **row, 'source': str(path)})
    corpus_recall = Path('reports/part2/corpus_growth_20261023/evaluation_recall_corrected/recall.json')
    if corpus_recall.exists():
        sources.append(corpus_recall)
        for corpus in json.loads(corpus_recall.read_text()):
            for row in corpus['recall']:
                retrieval_rows.append({'scope': 'corpus_growth_first100', 'pages': corpus['pages'],
                                       'sentences': corpus['sentences'], **row, 'source': str(corpus_recall)})
    write_csv(out / 'all_retrieval_runs.csv', retrieval_rows)
    latency_rows = []
    for path in (Path('reports/part2/latency_20261023/summary.json'), Path('reports/part2/corpus_growth_20261023/evaluation/timing/summary.json')):
        if path.exists():
            sources.append(path)
            for row in json.loads(path.read_text())['results']:
                latency_rows.append({**{k: v for k, v in row.items() if k != 'confusion_matrix'}, 'source': str(path)})
    write_csv(out / 'all_latency_runs.csv', latency_rows)
    choices = [
        {'family': 'claim_only', 'checkpoint': 'checkpoints/tuning_weighted/H2_seed42/best', 'selected': True,
         'reason': 'H2 epoch 2 has highest validation macro-F1 (0.620249) among H1-H4 and highest accuracy (0.6775) on the same 2000 claims.'},
        {'family': 'evidence_roberta', 'checkpoint': 'checkpoints/roberta_evidence/epoch_1', 'selected': True,
         'reason': 'Epoch 1 exceeds epoch 2 in gold-evidence validation macro-F1 (0.933485 vs 0.931286). This selection does not validate retrieved-evidence transfer.'},
        {'family': 'retrieval', 'checkpoint': 'title_fast_top5', 'selected': True,
         'reason': 'Top-5 reproduces the existing paired condition; top-10 ties first100 accuracy but has lower macro-F1. Selection is provisional on development data.'},
        {'family': 'symbolic_hybrid', 'checkpoint': 'v2_experimental_abstaining', 'selected': False,
         'reason': 'V1 and V2 resolve zero first100 claims. Keep v2 as a reported variant; no accuracy/efficiency case for selecting it.'},
        {'family': 'llm_judge', 'checkpoint': 'llama3.2:latest (digest recorded in local pilot manifest)', 'selected': False,
         'reason': 'Pilot comparator chosen for already installed local availability, not tuned or selected by performance. No final LLM selection from this small pilot.'}]
    write_csv(out / 'configuration_choices.csv', choices)
    report = {'training_rows': len(runs), 'classifier_evaluation_rows': len(evaluated), 'choices': choices,
              'input_sha256': {str(path): digest(path) for path in sources},
              'script_sha256': digest(__file__),
              'notes': ['Training legacy logs do not retain precision/recall or full run manifests; unavailable values are not invented.',
                        'Different evaluation scopes are separate populations; do not rank their metrics together.',
                        'Two identical legacy epoch-1 log entries are observations of uncertain provenance, not demonstrated independent seeds.',
                        'Seeds 42/43/44 in judge requests are inference samples, not Transformer retraining repeats.',
                        'Development selections are candidates for later final testing; test partition remains untouched.']}
    (out / 'manifest.json').write_text(json.dumps(report, indent=2))
    print('Assembled', len(runs), 'training rows and', len(evaluated), 'classifier rows')


if __name__ == '__main__':
    main()
