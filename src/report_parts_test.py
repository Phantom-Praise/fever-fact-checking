"""Assignment tables/figures from completed saved paired predictions; no inference."""
import csv
import json
from pathlib import Path


def build(out):
    out = Path(out)
    summary = json.loads((out / 'summary.json').read_text())
    protocol = json.loads((out / 'protocol.json').read_text())
    cases = json.loads((out / 'scored_cases.json').read_text())
    names = {'evidence_roberta': 'Evidence RoBERTa (baseline)', 'parts_rag': 'Wikipedia RAG + Llama parts (main)',
             'parts_model_label': "Llama's overall label (diagnostic)"}
    columns = ['System', 'N', 'Accuracy', 'Macro-F1', 'INVALID', 'Mean seconds']
    table = []
    per_class = []
    for condition, name in names.items():
        m = summary['conditions'][condition]
        table.append([name, m['n'], f"{100*m['accuracy']:.2f}%", f"{m['macro_f1']:.4f}",
                      m['invalid'], f"{m['mean_wall_seconds']:.2f}" if 'mean_wall_seconds' in m else 'same call as main'])
        for label, values in m['per_class'].items():
            per_class.append({'System': name, 'Label': label, **values})
    with (out / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(table)
    with (out / 'per_class.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)
    with (out / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'claim', 'gold', 'baseline', 'main', 'model_label', 'error'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(cases)
    b, m = (summary['conditions'][c] for c in ['evidence_roberta', 'parts_rag'])
    da, df = m['accuracy']-b['accuracy'], m['macro_f1']-b['macro_f1']
    lo, hi = summary['paired_accuracy_95_percentile_interval']
    verdict = 'supported descriptively on this sample' if da > 0 and df > 0 else 'partially supported descriptively' if da > 0 or df > 0 else 'not supported on this sample'
    lines = ['# Result Analysis: Wikipedia RAG with essential-part verdicts', '',
             '## 1. Matched results', '',
             f"Both systems evaluated the same {summary['n']} fresh eligible internal test claims, balanced across the three FEVER labels. "
             'This is a small system comparison, not full test-partition performance.', '',
             '| ' + ' | '.join(columns) + ' |', '|---|---:|---:|---:|---:|---:|']
    lines += ['| ' + ' | '.join(map(str, row)) + ' |' for row in table]
    lines += ['', '![Matched accuracy and macro-F1](comparison.png)', '',
              'The main prediction is derived from the essential-part verdicts. The model-provided overall label is shown separately. '
              'Neither system receives reference labels. INVALID outputs remain incorrect and stay in the denominator.', '',
              '## 2. Quantitative analysis', '',
              f"The observed accuracy difference is {100*da:+.2f} percentage points; macro-F1 difference is {df:+.4f}. "
              f"The paired claim-bootstrap 95% accuracy-difference interval is [{100*lo:+.2f}, {100*hi:+.2f}] percentage points. "
              + ('The interval includes zero, so a reliable accuracy gain is not established.' if lo <= 0 <= hi else
                 'This interval summarizes uncertainty for this small sampled comparison, not variation across models or training seeds.'), '',
              f"The main system corrected {summary['fixed']} baseline mistakes and introduced {summary['introduced']} new mistakes. "
              f"Model/derived-label disagreements: {summary['inconsistent']}.", '',
              '![Confusion matrices including invalid outputs](confusion.png)', '',
              '| System | Label | Precision | Recall | F1 |', '|---|---|---:|---:|---:|']
    for row in per_class:
        lines.append('| %s | %s | %.3f | %.3f | %.3f |' % tuple(row[k] for k in ['System', 'Label', 'precision', 'recall', 'f1']))
    lines += ['', '## 3. Qualitative evidence', '',
              'The examples below are selected by paired correctness, not a representative error-rate sample. '
              'A citation ID passing validation proves source membership, not sound reasoning. '
              'Failure attribution needs inspection of the supplied excerpts; absence of support alone is not refutation.', '']
    groups = [('Main correct, baseline wrong', lambda r: r['main']==r['gold'] and r['baseline']!=r['gold']),
              ('Baseline correct, main wrong', lambda r: r['main']!=r['gold'] and r['baseline']==r['gold']),
              ('Both wrong', lambda r: r['main']!=r['gold'] and r['baseline']!=r['gold']),
              ('Both correct', lambda r: r['main']==r['gold'] and r['baseline']==r['gold'])]
    for title, predicate in groups:
        lines += ['### ' + title, '']
        selected = [r for r in cases if predicate(r)][:2]
        if not selected:
            lines += ['No cases in this group.', '']
        for row in selected:
            lines += [f"**Claim {row['id']}:** {row['claim']}", '',
                      f"Reference: {row['gold']}. Baseline: {row['baseline']}. Main: {row['main']}.", '']
            for part in row['parts']:
                lines += [f"- Part: {part['part']} → **{part['verdict']}**. {part['justification']} "
                          f"Citations: {', '.join(part['evidence_ids']) or 'none'}."]
            for excerpt in row['cited_evidence'][:3]:
                lines += ['', f"Evidence {excerpt['id']} — [{excerpt['title']}]({excerpt['revision_url']}):", '',
                          '> ' + excerpt['text'].replace('\n', '\n> ')]
            if row['error']:
                lines += ['', 'Recorded failure: ' + row['error']]
            lines += ['', 'Full input and response: `traces/parts_rag/%s.json`.' % row['id'], '']
    lines += ['## 4. Hypothesis', '', protocol['hypothesis'], '',
              '**Verdict: ' + verdict + '.** The numbers above determine this description. '
              'It is not a general claim of superiority, a preregistered project-wide hypothesis, or proof of explanation faithfulness.', '',
              '## 5. Limitations and literature context', '', *['- ' + x for x in protocol['limitations']],
              '- The balanced sample has different class proportions from the full test population.',
              '- Context/output limits, one model/quantization, one execution, network failures and article date can affect outcomes.',
              '- The whole pipeline differs: a gain/loss cannot be attributed solely to embeddings or the parts prompt.',
              '- GEAR/KGAT published official-development or test scores concern different populations; their strict FEVER scores are not label accuracy. '
              'Their graph aggregation provides methodological context, not a matched numerical leaderboard here.',
              '- The hybrid paper reports F1 on 1,000 sampled SUPPORTS/REFUTES claims with NEI excluded. FActScore measures supported atomic-fact precision. '
              'These are distinct evaluation setups.', '',
              'Sources: [GEAR](https://aclanthology.org/P19-1085/), [KGAT Table 2](https://aclanthology.org/2020.acl-main.655.pdf), '
              '[hybrid paper](https://aclanthology.org/2025.winlp-main.19.pdf), [FActScore](https://aclanthology.org/2023.emnlp-main.741/).', '',
              'Full protocols, immutable traces, per-class CSV and paired predictions accompany this report. '
              'The earlier inspected 7/10 development result is not pooled into this test result.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, metric, title in zip(axes, ['accuracy', 'macro_f1'], ['Label accuracy', 'Macro-F1']):
        values = [summary['conditions'][c][metric] for c in ['evidence_roberta', 'parts_rag']]
        ax.bar(['Evidence RoBERTa', 'RAG + Llama parts'], values, color=['#64748b', '#0f766e'])
        ax.set_ylim(0, 1)
        ax.set_title(title + f" · same {summary['n']} claims")
        for i, value in enumerate(values):
            ax.text(i, value+.02, f'{value:.3f}', ha='center')
    fig.savefig(out / 'comparison.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, condition in zip(axes, ['evidence_roberta', 'parts_rag']):
        matrix = np.array(summary['conditions'][condition]['confusion_matrix'])
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=max(1, matrix.max()))
        ax.set_xticks(range(4), ['SUPPORTS', 'REFUTES', 'NEI', 'INVALID'])
        ax.set_yticks(range(3), ['SUPPORTS', 'REFUTES', 'NEI'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Reference')
        ax.set_title(names[condition])
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha='center', va='center', color='white' if value > matrix.max()/2 else 'black')
    fig.savefig(out / 'confusion.png', dpi=180)
    plt.close(fig)
