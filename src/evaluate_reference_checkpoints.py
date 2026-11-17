"""Reevaluate historical claim-only checkpoints on the current paired samples."""
import argparse
import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import precision_recall_fscore_support

from analyze_controlled import metrics
from experiment_paired_ablation import predict
from experiment_symbolic_v2 import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    sources = {'first100': 'data/processed/retrieved/validation_first_100_top5_title_fast.jsonl',
               'extension100': 'reports/part2/validation_extension_20261023_complete/retrieved_top10.jsonl'}
    checkpoints = {'unweighted_reference': 'checkpoints/roberta_epoch_1',
                   'weighted_reference_epoch2': 'checkpoints/roberta_weighted/epoch_2'}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    summaries, saved = [], []
    for scope, source in sources.items():
        records = [json.loads(line) for line in Path(source).read_text().splitlines()]
        gold = [int(r['label_id']) for r in records]
        for condition, checkpoint in checkpoints.items():
            predictions, seconds = predict(checkpoint, records, None, device, 8, 256)
            precision, recall, _, _ = precision_recall_fscore_support(gold, predictions, labels=[0, 1, 2], zero_division=0)
            summaries.append({'scope': scope, 'condition': condition, 'checkpoint': checkpoint, 'n': len(records),
                              **metrics(gold, predictions), 'macro_precision': float(precision.mean()),
                              'macro_recall': float(recall.mean()), 'inference_seconds': seconds})
            for row, pred in zip(records, predictions):
                saved.append({'scope': scope, 'condition': condition, 'id': row['id'],
                              'gold_label_id': row['label_id'], 'predicted_label_id': pred})
            print(scope, condition, summaries[-1]['accuracy'], summaries[-1]['macro_f1'], flush=True)
    with (out / 'predictions.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(saved[0]))
        writer.writeheader()
        writer.writerows(saved)
    (out / 'summary.json').write_text(json.dumps({'conditions': summaries, 'device': str(device), 'seed': 42,
        'batch_size': 8, 'max_length': 256, 'input_sha256': {p: digest(p) for p in [*sources.values(), __file__]},
        'checkpoint_config_sha256': {p: digest(Path(p) / 'config.json') for p in checkpoints.values()},
        'limitations': ['Frozen inference, not retraining or independent seeds.', 'Validation development samples; not final testing.',
                        'Historical training samples/configurations differ, so these comparisons are descriptive.']}, indent=2))


if __name__ == '__main__':
    main()
