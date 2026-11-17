"""Audit local judge results and prepare evidence packs for human claim review."""
import argparse
import csv
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_controlled import paired
from experiment_symbolic_v2 import digest
from retrieve_title_aware import retrieve


def text_profile(records):
    lengths = [len(re.findall(r'\b\w+\b', r['claim'])) for r in records]
    return {'n': len(records), 'unique_claims': len({r['claim'].casefold().strip() for r in records}),
            'mean_words': float(np.mean(lengths)), 'median_words': float(np.median(lengths)),
            'digit_fraction': sum(bool(re.search(r'\d', r['claim'])) for r in records) / len(records),
            'explicit_year_fraction': sum(bool(re.search(r'\b(?:18|19|20)\d{2}\b', r['claim'])) for r in records) / len(records),
            'negation_token_fraction': sum(bool(re.search(r'\b(?:not|never|no)\b', r['claim'], re.I)) for r in records) / len(records)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', default='reports/part2/local_llm_20261023')
    p.add_argument('--db', default='data/processed/wiki_fts.db')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, out = Path(args.pilot), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    generated = [json.loads(line) for line in (root / 'generated_claims.jsonl').read_text().splitlines()]
    fever = [json.loads(line) for line in (root / 'pilot_retrieved.jsonl').read_text().splitlines()]
    with (root / 'pilot_baseline_predictions.csv').open(newline='') as stream:
        baseline = list(csv.DictReader(stream))
    with (root / 'judge_responses.csv').open(newline='') as stream:
        responses = {r['task_id']: r for r in csv.DictReader(stream)}
    gold_by_id = {str(r['id']): int(r['label_id']) for r in fever}
    baseline_by_condition = {}
    for row in baseline:
        baseline_by_condition.setdefault(row['condition'], {})[row['id']] = int(row['predicted_label_id'])
    scores = json.loads((root / 'judge_scores.json').read_text())
    comparisons, condition_summaries = {}, {}
    for model, result in scores['models'].items():
        for condition, summary in result['conditions'].items():
            condition_summaries[condition] = {'mean_accuracy': summary['mean_accuracy'], 'mean_macro_f1': summary['mean_macro_f1'],
                'sample_std_accuracy_across_repeats': float(np.std([r['accuracy'] for r in summary['per_repeat'].values()], ddof=1)),
                'sample_std_macro_f1_across_repeats': float(np.std([r['macro_f1'] for r in summary['per_repeat'].values()], ddof=1)),
                'all_repeats_agree_fraction': summary['all_repeats_agree_fraction']}
            evidence, prompt = condition.split(':')
            for repeat in summary['per_repeat']:
                ids = list(gold_by_id)
                judge = [int(responses[f'{key}:{evidence}:{prompt}:{repeat}']['predicted_label_id']) for key in ids]
                for name in ('claim_only_H2', 'retrieved_evidence'):
                    comparisons[f'{condition}:repeat{repeat}:vs:{name}'] = paired(
                        [gold_by_id[key] for key in ids], [baseline_by_condition[name][key] for key in ids], judge)
    connection = sqlite3.connect(Path(args.db).resolve().as_uri() + '?mode=ro', uri=True)
    evidence_rows = []
    try:
        for r in generated:
            evidence_rows.append({'id': r['id'], 'claim': r['claim'], 'candidate_evidence': retrieve(connection, r['claim'], 5),
                                  'human_label': None, 'label_status': 'unreviewed'})
    finally:
        connection.close()
    (out / 'generated_claim_evidence.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in evidence_rows))
    import random
    all_claims = [{'id': str(r['id']), 'claim': r['claim'], 'source': 'FEVER'} for r in fever] + [dict(r, source='generated') for r in generated]
    random.Random(42).shuffle(all_claims)
    blinded = [{'review_id': f'review_{i:03}', 'original_id': r['id'], 'source': r['source'], 'claim': r['claim']}
               for i, r in enumerate(all_claims)]
    (out / 'analysis_only_source_key.json').write_text(json.dumps(blinded, indent=2))
    by_generated_id = {r['id']: r['candidate_evidence'] for r in evidence_rows}
    by_fever_id = {str(r['id']): r['retrieved_evidence'] for r in fever}
    (out / 'blinded_review_evidence.jsonl').write_text(''.join(json.dumps({
        'id': r['review_id'], 'claim': r['claim'],
        'candidate_evidence': by_fever_id[r['original_id']] if r['source'] == 'FEVER' else by_generated_id[r['original_id']]
    }) + '\n' for r in blinded))
    for annotator in ('A', 'B'):
        with (out / f'human_review_{annotator}.csv').open('x', newline='') as stream:
            fields = ['id', 'source', 'claim', 'annotator', 'atomic', 'claim_type', 'temporal', 'context_required',
                      'error_type', 'evidence_source', 'human_label', 'notes']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({'id': r['review_id'], 'claim': r['claim'], 'annotator': annotator} for r in blinded)
    report = {'fever_profile': text_profile(fever), 'generated_profile': text_profile(generated),
              'generated_exact_copies_of_fever': sum(r['claim'].casefold().strip() in {f['claim'].casefold().strip() for f in fever} for r in generated),
              'human_review': {'claims_per_annotator': len(fever) + len(generated), 'annotators': 2, 'completed': 0},
              'judge_condition_summaries': condition_summaries, 'judge_vs_baseline_paired': comparisons,
              'input_sha256': {str(root / p): digest(root / p) for p in ('generated_claims.jsonl', 'judge_responses.csv', 'judge_scores.json', 'pilot_retrieved.jsonl')},
              'limitations': ['Profiles are descriptive properties, not evidence that FEVER represents LLM factual errors.',
                              'Generated topics are retrieval-seeded, not a natural-use sample.',
                              'Human reviewers must verify truth and atomicity independently; reference labels are deliberately blank.',
                              'Candidate retrieval is an annotation aid, not ground truth or exhaustive evidence.',
                              'Annotator IDs distinguish source to the analysis code; reviewers should not inspect source files.',
                              'Only 30 independent FEVER claims; repeat judgments are not 90 independent claims.',
                              'Local quantized model availability drove judge selection; conclusions do not extend to all LLMs.',
                              'Paired p-values are exploratory, unadjusted, and repeated across dependent conditions.']}
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'fever_profile': report['fever_profile'], 'generated_profile': report['generated_profile']}, indent=2))


if __name__ == '__main__':
    main()
