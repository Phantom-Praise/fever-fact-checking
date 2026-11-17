"""Score completed independent human annotations; never invent missing labels."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score
from experiment_symbolic_v2 import digest

LABELS = {'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO', 'UNRATEABLE'}
CLAIM_TYPES = {'PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 'NUMBER', 'RELATION', 'OTHER'}
ERROR_TYPES = {'NONE', 'ENTITY', 'DATE', 'NUMBER', 'RELATION', 'MISSING_CONTEXT', 'UNSUPPORTED', 'UNRATEABLE'}


def load_annotations(path, key):
    with open(path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    by_id = {r['id']: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != set(key):
        raise ValueError('Annotation IDs are missing, duplicated, or unexpected')
    for identity, row in by_id.items():
        if row['claim'] != key[identity]['claim']:
            raise ValueError('Annotation claim differs from blinded source')
        if row['human_label'] not in LABELS or row['atomic'] not in ('YES', 'NO'):
            raise ValueError('Incomplete human label/atomicity: ' + identity)
        if any(row.get(field) not in ('YES', 'NO') for field in ('temporal', 'context_required')):
            raise ValueError('Incomplete temporal/context judgment: ' + identity)
        if row.get('claim_type') not in CLAIM_TYPES or row.get('error_type') not in ERROR_TYPES:
            raise ValueError('Unknown claim or error category: ' + identity)
        if not row['claim_type'] or not row['error_type'] or not row['notes']:
            raise ValueError('Claim type, error type and review notes are required: ' + identity)
        if row['human_label'] != 'UNRATEABLE' and not row['evidence_source']:
            raise ValueError('A factual judgment needs an evidence source: ' + identity)
    return by_id


def evaluate(first, second, key):
    ids = sorted(key)
    left = [first[i]['human_label'] for i in ids]
    right = [second[i]['human_label'] for i in ids]
    agreed = [i for i in ids if first[i]['human_label'] == second[i]['human_label']]
    differences = [{'id': i, 'claim': key[i]['claim'], 'label_A': first[i]['human_label'],
                    'label_B': second[i]['human_label'], 'adjudicated_label': '', 'adjudication_source': '', 'adjudication_notes': ''}
                   for i in ids if first[i]['human_label'] != second[i]['human_label']
                   or first[i]['claim_type'] != second[i]['claim_type']
                   or first[i]['error_type'] != second[i]['error_type']
                   or first[i]['atomic'] != second[i]['atomic']]
    return {'n': len(ids), 'label_agreement': len(agreed) / len(ids),
            'label_cohen_kappa': float(cohen_kappa_score(left, right)) if len(set(left + right)) > 1 else None,
            'atomicity_agreement': sum(first[i]['atomic'] == second[i]['atomic'] for i in ids) / len(ids),
            'claim_type_agreement': sum(first[i]['claim_type'] == second[i]['claim_type'] for i in ids) / len(ids),
            'error_type_agreement': sum(first[i]['error_type'] == second[i]['error_type'] for i in ids) / len(ids),
            'agreed_labels_by_source': {source: dict(Counter(first[i]['human_label'] for i in agreed if key[i]['source'] == source))
                                       for source in sorted({r['source'] for r in key.values()})},
            'requires_adjudication': differences,
            'limitations': ['Agreement is not label correctness; disagreements require independent adjudication.',
                            'Agreed-only counts are incomplete and must not substitute for an adjudicated population distribution.',
                            'Small retrieval-seeded generation pilot; no general representativeness conclusion.']}


def finalize_annotations(first, second, key, adjudications):
    """Require an explicit reviewed resolution for every disputed study field."""
    fields = ('human_label', 'atomic', 'claim_type', 'error_type', 'temporal', 'context_required')
    disputed = {identity for identity in key
                if any(first[identity].get(field, '') != second[identity].get(field, '') for field in fields)}
    by_id = {r['id']: r for r in adjudications}
    if len(by_id) != len(adjudications) or set(by_id) != disputed:
        raise ValueError('Adjudications must contain exactly the disputed unique IDs')
    final = []
    for identity, source in key.items():
        if identity in disputed:
            resolution = by_id[identity]
            if not resolution.get('adjudication_source') or not resolution.get('adjudication_notes'):
                raise ValueError('Adjudication requires a source and explanation: ' + identity)
            row = {field: resolution.get('adjudicated_' + field, '') for field in fields}
            if row['human_label'] not in LABELS or any(row[field] not in ('YES', 'NO') for field in ('atomic', 'temporal', 'context_required')):
                raise ValueError('Incomplete adjudicated labels: ' + identity)
            if row['claim_type'] not in CLAIM_TYPES or row['error_type'] not in ERROR_TYPES:
                raise ValueError('Incomplete adjudicated categories: ' + identity)
            row.update(evidence_source=resolution['adjudication_source'], notes=resolution['adjudication_notes'], resolution='adjudicated')
        else:
            row = {field: first[identity].get(field, '') for field in fields}
            row.update(evidence_source=first[identity].get('evidence_source', ''), notes=first[identity].get('notes', ''), resolution='reviewers_agreed')
        final.append({'id': identity, 'source': source['source'], 'claim': source['claim'], **row})
    distributions = {}
    for source in sorted({r['source'] for r in final}):
        rows = [r for r in final if r['source'] == source]
        distributions[source] = {'n': len(rows), 'unrateable': sum(r['human_label'] == 'UNRATEABLE' for r in rows),
            **{field: dict(Counter(r[field] for r in rows)) for field in fields}}
    return final, distributions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--key', required=True)
    p.add_argument('--reviewer-a', required=True)
    p.add_argument('--reviewer-b', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--adjudications', help='Completed adjudication CSV; enables final source distributions')
    args = p.parse_args()
    records = json.loads(Path(args.key).read_text())
    key = {r['review_id']: r for r in records}
    if len(key) != len(records) or not key:
        raise ValueError('Empty or duplicate source key')
    first, second = load_annotations(args.reviewer_a, key), load_annotations(args.reviewer_b, key)
    report = evaluate(first, second, key)
    # Include disagreements in temporal/context fields in the adjudication workflow.
    disputed_ids = {r['id'] for r in report['requires_adjudication']}
    for identity in key:
        if identity not in disputed_ids and any(first[identity].get(f, '') != second[identity].get(f, '') for f in ('temporal', 'context_required')):
            report['requires_adjudication'].append({'id': identity, 'claim': key[identity]['claim'],
                'label_A': first[identity]['human_label'], 'label_B': second[identity]['human_label']})
    for row in report['requires_adjudication']:
        row.pop('adjudicated_label', None)
        for field in ('human_label', 'atomic', 'claim_type', 'error_type', 'temporal', 'context_required'):
            row.setdefault('adjudicated_' + field, '')
        for field in ('adjudication_source', 'adjudication_notes'):
            row.setdefault(field, '')
    final = None
    report['final_distributions_status'] = 'pending_adjudication_or_explicit_finalization'
    if args.adjudications:
        with open(args.adjudications, newline='') as stream:
            resolutions = list(csv.DictReader(stream))
        final, distributions = finalize_annotations(first, second, key, resolutions)
        report['final_distributions_by_source'] = distributions
        report['final_distributions_status'] = 'complete_for_supplied_reviews'
    report['input_sha256'] = {p: digest(p) for p in (args.key, args.reviewer_a, args.reviewer_b)}
    if args.adjudications:
        report['input_sha256'][args.adjudications] = digest(args.adjudications)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    if report['requires_adjudication']:
        with (out / 'adjudication_needed.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report['requires_adjudication'][0]))
            writer.writeheader()
            writer.writerows(report['requires_adjudication'])
    if final is not None:
        with (out / 'final_annotations.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(final[0]))
            writer.writeheader()
            writer.writerows(final)


if __name__ == '__main__':
    main()
