"""Summarize the bounded final Step 2 experiments without changing prior results."""
import argparse
import csv
import json
from pathlib import Path

from experiment_symbolic_v2 import digest


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join('---' for _ in headers) + ' |\n' + ''.join('| ' + ' | '.join(str(v) for v in row) + ' |\n' for row in rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', default='reports/part2/step2_final_20261023')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, out = Path(args.study), Path(args.output)
    heldout = json.loads((root / 'heldout100/summary.json').read_text())
    machine = json.loads((root / 'rq1_machine_pilot/summary.json').read_text())
    protocol = json.loads((root / 'heldout100/protocol.json').read_text())
    for directory in ('heldout100', 'rq1_machine_pilot'):
        complete = json.loads((root / directory / 'COMPLETE.json').read_text())
        if digest(root / directory / 'summary.json') != complete['summary_sha256']:
            raise ValueError('Incomplete or modified study summary')
    out.mkdir(parents=True, exist_ok=False)
    names = {'unweighted_reference': 'Original unweighted reference', 'claim_only_H2': 'Frozen claim-only H2',
             'retrieved_top5': 'Frozen retrieval plus RoBERTa', 'hybrid_v2': 'Live symbolic v2 hybrid'}
    conditions = [{'condition': name, 'n': heldout['n'], **{k: value[k] for k in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1')}}
                  for name, value in heldout['conditions'].items()]
    write_csv(out / 'heldout_conditions.csv', conditions)
    timing = []
    for row in heldout['trials']:
        timing.append({k: v for k, v in row.items() if k not in ('confusion_matrix', 'per_class')} | {
            'condition_http_attempts': row['http_attempts_in_pass'] if row['condition'] == 'hybrid_v2' else 0,
            'note': 'The source *_in_pass counters describe the shared paired pass; condition_http_attempts isolates the actual caller.'})
    write_csv(out / 'heldout_timing.csv', timing)
    machine_rows = []
    for model, sources in machine['source_distributions'].items():
        for source, counts in sources.items():
            machine_rows.append({'model': model, 'source': source, 'n': counts['n'], 'valid': counts['valid'], 'invalid': counts['invalid'],
                                 **{label: counts['label'].get(label, 0) for label in ('SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO', 'UNRATEABLE')},
                                 'atomic_yes': counts['atomic'].get('YES', 0), 'atomic_no': counts['atomic'].get('NO', 0),
                                 'label_origin': 'machine_evidence_relative_not_human_truth'})
    write_csv(out / 'rq1_machine_distributions.csv', machine_rows)
    quality = []
    for model in machine['source_distributions']:
        annotations = json.loads((root / 'rq1_machine_pilot' / (model.replace(':', '_') + '_annotations.json')).read_text())
        valid = [r['annotation'] for r in annotations.values() if r['valid']]
        inconsistent = []
        for identity, result in annotations.items():
            if not result['valid']:
                continue
            r = result['annotation']
            expected = {'SUPPORTS': 'NONE', 'NOT ENOUGH INFO': 'UNSUPPORTED', 'UNRATEABLE': 'UNRATEABLE'}.get(r['label'])
            if (expected is not None and r['error_type'] != expected) or (r['label'] == 'REFUTES' and r['error_type'] == 'NONE'):
                inconsistent.append(identity)
        quality.append({'model': model, 'structurally_valid': len(valid), 'semantic_protocol_inconsistencies': len(inconsistent),
                        'inconsistent_ids': inconsistent, 'unique_error_type_values': sorted({r['error_type'] for r in valid})})
    (out / 'rq1_protocol_quality_audit.json').write_text(json.dumps({
        'audit_type': 'posthoc_consistency_check_not_new_annotation', 'models': quality,
        'policy': 'Preserve original model judgments. Do not convert them into corrected labels or human annotations.'}, indent=2))
    rows = [[names[r['condition']], f"{r['accuracy']:.4f}", f"{r['macro_precision']:.4f}",
             f"{r['macro_recall']:.4f}", f"{r['macro_f1']:.4f}"] for r in conditions]
    paired = heldout['paired_comparisons']['claim_only_H2__vs__retrieved_top5']
    report = '''# Step 2 experiment update

This update adds a frozen held-out comparison, actual online symbolic-stage timing, and a separate automated RQ1 pilot. Earlier reports and measurements are preserved. No models were retrained or rules tuned, and the experiment branch was not merged into main.

## Status

- RQ2 already has 360 measured local-judge requests, prompt/evidence conditions, and inference repeats.
- RQ3 now also has a prespecified 100-claim held-out comparison.
- RQ4 now includes actual online Wikidata overhead, in addition to the earlier corpus-growth measurements.
- RQ1 now has a two-model evidence-conditioned automated coding experiment. Independent human review remains pending, as confirmed by the user. The generated-claim labels below are not established human ground truth.

## Held-out comparison

Seed 874 shuffles the existing test partition; the first 100 unique normalized claims are retained. Selection does not inspect labels. Selected IDs and normalized claim text are checked against training and validation. The protocol, sample IDs, and code snapshot were written before inference. The selected checkpoints, rules, top-5 retrieval, length 256, and batch size 1 are fixed. This is a bounded sample of the held-out split, not the full test partition.

'''
    report += md_table(['Condition', 'Accuracy', 'Macro precision', 'Macro recall', 'Macro F1'], rows)
    report += '\nDevice for this new comparison: ' + protocol['device'] + '. Do not directly compare these timings with the earlier CPU corpus-growth run.\n'
    report += f'''\nRetrieval-minus-H2 accuracy is {paired['right_minus_left_accuracy']:+.2f}; paired bootstrap 95% CI {paired['paired_bootstrap_accuracy_ci95']}, unadjusted exact McNemar p={paired['mcnemar_exact_p_unadjusted']:.4f}. The small accuracy difference is not a demonstrated general improvement. H2 has higher macro-F1; both retrieval and hybrid misclassify all 22 NEI claims. Do not change model selection based on these test outputs.

The hybrid and retrieval predictions are identical. Symbolic v2 resolves zero claims and therefore avoids zero retrieval or Transformer calls. A negative result still answers whether the implemented symbolic stage helped this measured sample: it did not.

'''
    report += '## Actual pipeline latency\n\n'
    report += md_table(['Pass', 'Condition', 'Mean ms per claim', 'Symbolic seconds', 'HTTP attempts', 'Retrieval calls', 'Model calls'], [
        [r['pass'], r['condition'], f"{r['mean_seconds']*1000:.2f}", f"{r['symbolic_seconds']:.4f}", r['condition_http_attempts'], r['retrieval_calls'], r['model_calls']]
        for r in timing])
    report += '''
The initial online pass starts with an empty study snapshot and includes actual request latency, pacing/backoff if any, rule execution, retrieval, tokenization, and inference. Model loading and warmup are excluded. Within-pass query caching is allowed. The two later passes reuse the saved Wikidata responses. Each claim alternates which pipeline is run first across passes. Raw events record each phase, the decision, call counts, and service failures.

Seven HTTP requests were made with zero HTTP failures. No service-outage fallbacks occurred. The initial hybrid pass averaged 106.57 ms per claim versus 66.35 ms for retrieval alone. Warm-cache differences fluctuate without any avoided calls, so they do not establish a symbolic computational saving. Only three passes, shared OS caches, and changing network latency limit conclusions. No monetary cost saving is claimed.

## RQ1 automated experiment

Two different already-installed local models see the same 60 source-blinded claims and the same five candidate evidence sentences: 30 FEVER claims and 30 generated claims. Neither source identity nor gold FEVER labels enters the prompt. Settings are fixed: seed 874, temperature 0, context 4096, output budget 256 tokens. The coding schema includes label, atomicity, claim type, primary error type, cited sentence indices, and a short reason. Both models are run regardless of their scores; invalid outputs are retained and counted.

'''
    report += md_table(['Model', 'Source', 'N', 'Valid', 'Invalid', 'Supports', 'Refutes', 'NEI', 'Unrateable'], [
        [r['model'], r['source'], r['n'], r['valid'], r['invalid'], r['SUPPORTS'], r['REFUTES'], r['NOT ENOUGH INFO'], r['UNRATEABLE']]
        for r in machine_rows])
    report += '\nThe method is calibrated against the original human FEVER labels on the 30 FEVER claims:\n\n'
    report += md_table(['Model', 'FEVER calibration N', 'Accuracy', 'Macro F1'], [
        [model, r['n'], f"{r['accuracy']:.4f}", f"{r['macro_f1']:.4f}"] for model, r in machine['fever_calibration'].items()])
    report += '\nMachine agreement is a reliability diagnostic, not proof that either model is correct:\n\n'
    report += md_table(['Field', 'Jointly valid N', 'Agreement fraction', 'Cohen kappa'], [
        [field, r['joint_valid_n'], 'undefined' if r['fraction'] is None else f"{r['fraction']:.4f}", 'undefined' if r['cohen_kappa'] is None else f"{r['cohen_kappa']:.4f}"]
        for field, r in machine['machine_agreement'].items()])
    report += '\nA posthoc protocol-consistency check found the following issues; structural validity alone is insufficient:\n\n'
    report += md_table(['Model', 'Structurally valid', 'Error category inconsistencies', 'Distinct error categories returned'], [
        [r['model'], r['structurally_valid'], r['semantic_protocol_inconsistencies'], ', '.join(r['unique_error_type_values'])] for r in quality])
    report += '\nBoth models achieved only 0.30 accuracy against FEVER references. Their constant error-category outputs and weak calibration make this automated coding method unsuitable for establishing generated-claim factual-error rates. RQ1 is inconclusive under this method; independent review remains necessary. No errors are repaired by inventing labels.\n'
    report += '''
Generated label counts are evidence-relative model decisions, not verified factual-error rates. Missing or historical evidence can lead to NEI for a true current-world claim. Citation indices are checked for presence, not independently established entailment. Retrieval-seeded topics, a small sample, quantization, and subjective coding restrict generalization. The experiment tests an automated comparison method; it cannot settle how accurately FEVER represents real LLM errors without reliable independent labels.

The human-review sheets are unchanged and blank. The human scorer now imports explicit adjudication decisions, rejects missing/disputed IDs or unresolved fields, and exports final annotations and source distributions only when completed reviews and decisions are supplied. No model output is copied into the human files. Use the supplied machine-disagreement file only for a separate assisted-review workflow; do not show it to reviewers who must remain independent.

## What remains

The requested held-out and live-timing experiments have been executed. RQ1's automated pilot is executed, but independent human validation remains unperformed. Additional Transformer training seeds belong to the optional reproducibility strengthening in Step 3, not to this Step 2 run. No further tuning on the held-out sample is justified by these results.

## Files

- `heldout_conditions.csv`: four frozen conditions on identical held-out IDs.
- `heldout_timing.csv`: initial online and two warm-snapshot paired passes.
- `rq1_machine_distributions.csv`: explicitly machine-coded source distributions.
- `../heldout100/`: frozen protocol, raw phase events, predictions, retrieval, and Wikidata snapshot.
- `../rq1_machine_pilot/`: model digests/settings, complete raw responses, validation failures, and disagreements.
'''
    (out / 'STEP2_RESULTS.md').write_text(report)
    (out / 'manifest.json').write_text(json.dumps({'input_sha256': {str(path): digest(path) for path in (
        root / 'heldout100/summary.json', root / 'heldout100/protocol.json', root / 'rq1_machine_pilot/summary.json', root / 'rq1_machine_pilot/protocol.json', Path(__file__))},
        'status': 'experiments_executed_human_validation_pending', 'human_labels_created': 0}, indent=2))
    print(out / 'STEP2_RESULTS.md')


if __name__ == '__main__':
    main()
