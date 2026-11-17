"""Read-only eligibility audit. No fetching, inference, training or model loading."""
from pathlib import Path
from collections import Counter
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def records(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def norm(text):
    return ' '.join(text.casefold().split())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    train = ROOT / 'data/processed/train.jsonl'
    validation = ROOT / 'data/processed/validation.jsonl'
    test = ROOT / 'data/processed/test.jsonl'
    training_text = {norm(r['claim']) for r in records(train)}
    validation_text = {norm(r['claim']) for r in records(validation)}
    previous = sorted(p for p in (ROOT / 'reports').rglob('*.jsonl')
                      if p.name in {'inputs.jsonl', 'sample.jsonl', 'claims_without_labels.jsonl',
                                    'results.jsonl', 'outputs.jsonl'}
                      and 'source_snapshots' not in p.parts and 'submission' not in str(p))
    prior_text = set()
    for path in previous:
        for row in records(path):
            if isinstance(row.get('claim'), str):
                prior_text.add(norm(row['claim']))
    seen = set()
    reasons = Counter()
    eligible, excluded = [], []
    labels, eligible_labels = Counter(), Counter()
    for row in records(test):
        text = norm(row['claim'])
        labels[row['label']] += 1
        why = []
        if text in training_text:
            why.append('normalized_training_overlap')
        if text in validation_text:
            why.append('normalized_validation_overlap')
        if text in prior_text:
            why.append('prior_experiment_input_or_output')
        if text in seen:
            why.append('duplicate_within_test')
        seen.add(text)
        if why:
            reasons.update(why)
            excluded.append({'id': row['id'], 'reasons': why})
        else:
            eligible.append({'id': row['id'], 'claim': row['claim']})
            eligible_labels[row['label']] += 1
    audit = dict(status='planning_only_no_inference', full_test_rows=sum(labels.values()),
                 full_test_class_counts=dict(labels), unique_normalized_test_claims=len(seen),
                 eligible_unique_rows=len(eligible), eligible_class_counts=dict(eligible_labels),
                 excluded_rows=len(excluded), exclusion_counts_overlapping=dict(reasons),
                 policy='Exclude normalized train/validation overlap, prior saved experiment inputs/outputs, '
                        'and duplicate test texts. Preserve original files and labels. Prior inputs are '
                        'conservatively considered exposed even when inference did not finish.',
                 limits='This audit checks the listed saved artifacts. It cannot prove no manual inspection '
                        'ever occurred. Existing test-sample analyses mean this is not an untouched project-wide test.',
                 prior_artifacts=[str(p.relative_to(ROOT)) for p in previous],
                 sha256={str(p.relative_to(ROOT)): digest(p) for p in [train, validation, test, *previous]})
    for name, data in [('population_audit.json', audit), ('excluded_ids.json', excluded)]:
        with (OUT / name).open('x') as stream:
            json.dump(data, stream, indent=2)
    with (OUT / 'eligible_claims_without_labels.jsonl').open('x') as stream:
        stream.writelines(json.dumps(row) + '\n' for row in eligible)
    print(json.dumps({k: v for k, v in audit.items() if k not in {'sha256', 'prior_artifacts'}}, indent=2))


if __name__ == '__main__':
    main()
