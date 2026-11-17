"""Prepare blinded RQ1/RQ2 inputs, or score externally collected judge outputs.

No API calls. A judge response file must have exactly one row per task, plus
model and predicted_label_id (0/1/2). Invalid or missing responses are errors.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score
from analyze_controlled import metrics
from experiment_symbolic_v2 import digest, load_inputs

PROMPTS = {
    'direct': 'Classify the factual claim using only the supplied evidence. Missing evidence is insufficient. Return JSON with label SUPPORTS, REFUTES, or NOT ENOUGH INFO.',
    'evidence_first': 'Assess whether the supplied evidence establishes the claim, contradicts it, or is insufficient. Use no outside knowledge. Return JSON with label SUPPORTS, REFUTES, or NOT ENOUGH INFO.',
}


def prepare(args):
    records, _ = load_inputs(args.retrieved, args.predictions)
    if args.repeats < 2:
        raise ValueError('At least two repeats are needed for consistency')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    tasks = []
    for row in records:
        for condition in ('retrieved', 'empty'):
            for prompt, instruction in PROMPTS.items():
                for repeat in range(args.repeats):
                    tasks.append({'task_id': f"{row['id']}:{condition}:{prompt}:{repeat}",
                                  'id': str(row['id']), 'evidence_condition': condition,
                                  'prompt': prompt, 'repeat': repeat, 'instruction': instruction,
                                  'claim': row['claim'],
                                  'evidence': row['evidence_text'] if condition == 'retrieved' else ''})
    (out / 'judge_tasks.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    with (out / 'rq1_annotation.csv').open('w', newline='') as stream:
        fields = ['id', 'source', 'claim', 'annotator', 'atomic', 'claim_type',
                  'temporal', 'context_required', 'error_type', 'evidence_source', 'human_label', 'notes']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({'id': r['id'], 'source': 'FEVER_validation', 'claim': r['claim']} for r in records)
    (out / 'manifest.json').write_text(json.dumps({
        'claims': len(records), 'tasks': len(tasks), 'repeats': args.repeats,
        'input_sha256': {p: digest(p) for p in (args.retrieved, args.predictions)},
        'protocol': ['No gold labels or gold evidence exposed in judge tasks.',
                     'Choose and record an exact model version, temperature, seed policy, token limit and date before execution.',
                     'Randomize task order using a recorded seed; isolate each request from previous messages.',
                     'Model is held fixed across prompt/evidence/repeat conditions.',
                     'Evidence-constrained empty condition is an abstention control, not a closed-book factual baseline.',
                     'Record usage, latency, raw response, errors and retries; do not silently drop failed tasks.',
                     'For RQ1 add a separately sampled LLM-generated claim corpus with prompt/model provenance.',
                     'Use two blinded annotators and adjudicate disagreements. No human labels are filled automatically.',
                     'Do not assume a synthetic perturbation is false; verify it independently.',
                     'This preparation alone does not answer RQ1 or RQ2.']}, indent=2))
    print('Prepared', len(tasks), 'judge tasks; no requests executed')


def score(args):
    tasks = [json.loads(line) for line in Path(args.tasks).read_text().splitlines() if line.strip()]
    expected = {t['task_id']: t for t in tasks}
    if len(expected) != len(tasks) or not tasks:
        raise ValueError('Empty or duplicate tasks')
    records, _ = load_inputs(args.retrieved, args.predictions)
    gold = {str(r['id']): int(r['label_id']) for r in records}
    records_by_id = {str(r['id']): r for r in records}
    cells = defaultdict(set)
    for task in tasks:
        record = records_by_id.get(task['id'])
        if record is None or task['claim'] != record['claim']:
            raise ValueError('Judge task claim differs from evaluation input')
        if task['evidence_condition'] not in ('retrieved', 'empty') or task['prompt'] not in PROMPTS:
            raise ValueError('Unknown judge condition')
        evidence = record['evidence_text'] if task['evidence_condition'] == 'retrieved' else ''
        if task['evidence'] != evidence or task['instruction'] != PROMPTS[task['prompt']]:
            raise ValueError('Judge task evidence or instruction differs')
        repeat = int(task['repeat'])
        if repeat < 0 or task['task_id'] != f"{task['id']}:{task['evidence_condition']}:{task['prompt']}:{repeat}":
            raise ValueError('Malformed judge task ID')
        cells[(task['id'], task['evidence_condition'], task['prompt'])].add(repeat)
    repetitions = next(iter(cells.values()))
    if (len(cells) != len(records) * 4 or len(repetitions) < 2
            or any(repeats != repetitions for repeats in cells.values())):
        raise ValueError('Incomplete paired judge factorial design')
    with open(args.responses, newline='') as stream:
        responses = list(csv.DictReader(stream))
    models = {r['model'] for r in responses}
    if not models or '' in models:
        raise ValueError('Missing model version')
    results = {}
    for model in sorted(models):
        rows = [r for r in responses if r['model'] == model]
        by_task = {r['task_id']: r for r in rows}
        if len(by_task) != len(rows) or set(by_task) != set(expected):
            raise ValueError('Each model requires exactly one response per task')
        groups = defaultdict(list)
        for tid, task in expected.items():
            pred = int(by_task[tid]['predicted_label_id'])
            if pred not in (0, 1, 2):
                raise ValueError('Invalid judge label')
            groups[(task['evidence_condition'], task['prompt'])].append((task, pred))
        result = {}
        for (condition, prompt), pairs in groups.items():
            repeated = defaultdict(list)
            per_repeat = defaultdict(list)
            for task, pred in pairs:
                repeated[task['id']].append(pred)
                per_repeat[int(task['repeat'])].append((task['id'], pred))
            repeat_metrics = {}
            for repeat, values in per_repeat.items():
                repeat_metrics[str(repeat)] = metrics([gold[key] for key, _ in values], [pred for _, pred in values])
            result[condition + ':' + prompt] = {
                'per_repeat': repeat_metrics,
                'mean_accuracy': float(np.mean([m['accuracy'] for m in repeat_metrics.values()])),
                'mean_macro_f1': float(np.mean([m['macro_f1'] for m in repeat_metrics.values()])),
                'all_repeats_agree_fraction': sum(len(set(v)) == 1 for v in repeated.values()) / len(repeated),
                'pairwise_repeat_agreement': float(np.mean([a == b for v in repeated.values() for i, a in enumerate(v) for b in v[i+1:]])),
                'n_unique_claims': len(repeated)}
        # Prompt consistency at the same evidence condition and repeat.
        prompt_agreement = {}
        for condition in sorted({t['evidence_condition'] for t in tasks}):
            direct, alternative = [], []
            for t in tasks:
                if t['evidence_condition'] == condition and t['prompt'] == 'direct':
                    other = f"{t['id']}:{condition}:evidence_first:{t['repeat']}"
                    direct.append(int(by_task[t['task_id']]['predicted_label_id']))
                    alternative.append(int(by_task[other]['predicted_label_id']))
            prompt_agreement[condition] = {'agreement': float(np.mean(np.array(direct) == alternative)),
                                           'cohen_kappa': float(cohen_kappa_score(direct, alternative)) if len(set(direct + alternative)) > 1 else None}
        results[model] = {'conditions': result, 'prompt_agreement': prompt_agreement}
    report = {'models': results, 'input_sha256': {p: digest(p) for p in (args.tasks, args.responses, args.retrieved)},
              'limitations': ['Repeated judgments are not independent claims.',
                              'Development sample; no final-test conclusion.',
                              'Scorer validates supplied labels; retain raw responses and parsing logs for audit.']}
    with Path(args.output).open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'score'])
    parser.add_argument('--retrieved', default='data/processed/retrieved/validation_first_100_top5_title_fast.jsonl')
    parser.add_argument('--predictions', default='reports/part2/paired/predictions.csv')
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--tasks')
    parser.add_argument('--responses')
    args = parser.parse_args()
    if args.mode == 'score' and (not args.tasks or not args.responses):
        parser.error('score requires --tasks and --responses')
    (prepare if args.mode == 'prepare' else score)(args)


if __name__ == '__main__':
    main()
