"""Validate and explain a completed, frozen official-development comparison."""
import argparse
import csv
import json
import sqlite3
from pathlib import Path

from evaluate_official_fever import CONDITIONS, normalize, official_metrics
from experiment_symbolic_v2 import digest
from vendor.fever_scorer import is_strictly_correct


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    args=parser.parse_args()
    out=Path(args.run)
    summary=json.loads((out/'summary.json').read_text())
    protocol=summary['protocol']
    for path,expected in protocol['input_sha256'].items():
        if digest(path)!=expected:
            raise ValueError('Protocol source/checkpoint changed: '+path)
    data_path=next(path for path in protocol['input_sha256'] if path.endswith('shared_task_dev.jsonl'))
    gold=[json.loads(line) for line in Path(data_path).read_text().splitlines()]
    original_pool={normalize(r['claim']) for r in map(json.loads,Path('data/raw/train.jsonl').read_text().splitlines())}
    retained=[i for i,r in enumerate(gold) if normalize(r['claim']) not in original_pool]
    sensitivity={}; examples=[]
    for condition in CONDITIONS:
        predictions=[json.loads(line) for line in (out/(condition+'.jsonl')).read_text().splitlines()]
        score=official_metrics(gold,predictions)
        for name,expected in summary['conditions'][condition].items():
            if score[name]!=expected: raise ValueError('Saved score does not reproduce: '+condition+' '+name)
        if any(len(r['predicted_evidence'])>5 for r in predictions): raise ValueError('Too many submitted sentences')
        sensitivity[condition]=official_metrics([gold[i] for i in retained],[predictions[i] for i in retained])
        if condition=='calibrated_ensemble':
            categories={}
            for g,p in zip(gold,predictions):
                strict=is_strictly_correct({**g,**p},5)
                category=('strict_correct' if strict else 'correct_label_missing_evidence' if g['label']==p['predicted_label'] else 'wrong_label')
                if category not in categories: categories[category]={'category':category,'id':g['id'],'claim':g['claim'],
                    'gold_label':g['label'],'predicted_label':p['predicted_label'],'predicted_evidence':p['predicted_evidence'],
                    'annotated_groups':[[[e[2],e[3]] for e in group if e[2] is not None and e[3] is not None] for group in g['evidence']]}
                if len(categories)==3: break
            examples=list(categories.values())
    (out/'nonoverlap_sensitivity.json').write_text(json.dumps({'n':len(retained),'removed':len(gold)-len(retained),
        'note':'Supplementary conservative text-overlap exclusion, not the official full-population comparison.',
        'conditions':sensitivity},indent=2))
    connection=sqlite3.connect(Path('data/processed/wiki_fts.db').resolve().as_uri()+'?mode=ro',uri=True)
    try:
        for example in examples:
            example['retrieved_text']=[]
            for page,line in example['predicted_evidence']:
                found=connection.execute('SELECT text FROM page_sentences WHERE page_id=? AND sentence_id=?',(page,line)).fetchone()
                example['retrieved_text'].append({'page_id':page,'sentence_id':line,'text':found[0] if found else None})
    finally: connection.close()
    (out/'illustrative_cases.json').write_text(json.dumps(examples,indent=2))
    names={'claim_only_H2':'Our frozen claim-only H2','retrieved_roberta':'Our frozen retrieved RoBERTa','calibrated_ensemble':'Our frozen calibrated ensemble'}
    text=['# Same-population, same-scoring FEVER comparison','',
          'Completed evaluation on **all 19,998 official FEVER development claims** (6,666 per label). No retraining, parameter selection, or rule changes were performed. Previous results remain preserved.','',
          '| System | Origin | Label accuracy | Strict FEVER score |',
          '|---|---|---:|---:|']
    for name,value in summary['conditions'].items():
        text.append(f"| {names[name]} | Measured locally | {100*value['label_accuracy']:.2f}% | {100*value['fever_score']:.2f}% |")
    for name,value in protocol['comparison']['rows'].items():
        text.append(f"| {name} | Published development reference; not rerun | {100*value['label_accuracy']:.2f}% | {100*value['fever_score']:.2f}% |")
    ensemble=summary['conditions']['calibrated_ensemble']
    evidence=summary['conditions']['retrieved_roberta']
    claim=summary['conditions']['claim_only_H2']
    corpus=json.loads((out/'corpus_annotation_availability.json').read_text())
    paired_claim=summary['paired_comparisons']['claim_only_H2__vs__ensemble_label_accuracy']
    ci=paired_claim['paired_bootstrap_accuracy_ci95']
    text += ['', '## What the percentages mean','',
             'Label accuracy checks the predicted class. Strict FEVER score also requires a complete annotated evidence group among the first five submitted sentences for SUPPORTS/REFUTES. Correct NEI predictions need no evidence. Macro-F1 for classification and evidence F1 are different metrics and are not substituted for FEVER score.','',
             f"The ensemble correctly labels {round(ensemble['label_accuracy']*len(gold)):,} of {len(gold):,} claims; {round(ensemble['fever_score']*len(gold)):,} satisfy the strict label-and-evidence rule. Its label accuracy differs from evidence-only by {100*(ensemble['label_accuracy']-evidence['label_accuracy']):+.2f} percentage points and claim-only by {100*(ensemble['label_accuracy']-claim['label_accuracy']):+.2f} points.",'',
             f"Against claim-only H2, the paired claim-bootstrap 95% interval for the ensemble's accuracy gain is [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] percentage points (2,000 resamples, seed 42). This supports a measured accuracy gain on this benchmark. Resampling is by claim, not by shared evidence page; related claims can be correlated. Local paired tests are exploratory and unadjusted for multiple comparisons. No such paired test against the papers is claimed.",'',
             'The claim-only reference submits no evidence. Its strict score therefore comes only from correctly predicted NEI claims; it is not presented as a complete evidence-producing fact-checker. The evidence and ensemble methods submit the same original top-five retrieved sentences.','',
             '## Exactly what now matches the papers','',
             '- The complete official development population, not a 100/200-claim internal sample.',
             '- The three original FEVER labels and published human evidence annotations.',
             '- The unmodified official scorer with max_evidence=5.',
             '- Development results are compared with development results; published test numbers (GEAR 67.10, KGAT 70.38 FEVER) are not mixed into this table.','',
             'The published reference values come from [KGAT Table 2, Dev columns](https://aclanthology.org/2020.acl-main.655.pdf). Its Table 1 confirms the official development class counts. [GEAR](https://aclanthology.org/P19-1085.pdf) is the original source for the GEAR architecture/results. [Official scoring source](https://github.com/sheffieldnlp/fever-scorer) is vendored at commit 4801615100fbf6327f8e99b5dbaefe5dd890e869, unchanged, with its license/provenance.','',
             '## What remains different','',
             'We have not rerun the authors\' trained models locally. Their rows are published reference measurements; no paired significance test against them is possible without their per-claim predictions. Our base models, training subsets, retrieval architecture, and input formatting differ from their systems. Thus matching population and scoring enables a benchmark comparison, not a controlled causal comparison of architectures or a reproduction of their full training pipeline.','',
             'Our Wikipedia index has 5,396,106 pages; KGAT describes a 5,416,537-document background collection. Exact document filtering/corpus membership has not been matched. This is an additional retrieval-resource difference, not a matched-corpus ablation.','',
             f"The annotation-availability audit found {corpus['available_in_index']:,} of {corpus['annotated_sentence_references']:,} distinct development evidence references in our index. {len(corpus['verifiable_claims_without_any_complete_group_in_index']):,} verifiable claims have no complete annotated group available in that index. They remain in the full official denominator; none is silently dropped or given oracle evidence. This audit does not feed gold evidence to the model.",'',
             'The hybrid-paper binary F1 and FActScore atomic-fact precision are different tasks/metrics and are intentionally excluded from this comparison. Matching these two FEVER papers does not make every literature-review result comparable.','',
             '## Leakage and reproducibility audit','',
             f"Official IDs overlap with the original training pool: {protocol['overlap_audit']['id_overlap_with_original_training_pool']}. Normalized claim-text overlaps: {protocol['overlap_audit']['normalized_text_overlap_with_original_training_pool']}. This audit is conservative against the entire original training pool, not just claims actually sampled for fitting. All official rows are retained for population matching. The secondary no-text-overlap sensitivity check scores {len(retained):,} claims separately and must not be labeled the full official benchmark.",'',
             'A supplementary reconstruction of each model\'s seed-42, 20% training subset found zero exact normalized-text overlaps for either model. That reconstruction follows the current training scripts; historical per-step training manifests are incomplete, so the broader three-text audit is retained as well. See training_subset_reconstruction_audit.json.','',
             'The ensemble configuration was frozen from the earlier internal-validation study: original top-5 retrieval, claim log-probability weight .75, NEI bias 2. Seed 42, MPS, inference batch 8, maximum length 256. No gold labels/evidence enter retrieval or neural inference. Do not retune on this evaluation and then describe it as an untouched test. Official development is public validation data, not the blind test set.','',
             'Thirty-five unit tests passed before evaluation, including complete-group matching, the five-sentence cap, correct NEI handling, sample-order rejection, and separation of label versus strict scores. This report re-scores all saved submissions and verifies exact reproduction of saved metrics and protocol source/checkpoint hashes.','',
             '## Files and reproduction','',
             '- `comparison.csv`: local measured results plus clearly labeled published references.',
             '- `claim_only_H2.jsonl`, `retrieved_roberta.jsonl`, `calibrated_ensemble.jsonl`: complete standard FEVER predictions in original official ID order.',
             '- `outputs.jsonl`: component probabilities and retrieved sentence references.',
             '- `protocol.json`, `runner_source.py`, `summary.json`: frozen choices, source/checkpoint hashes, official metrics, and paired local comparisons.',
             '- `normalized_overlap_audit.json`, `nonoverlap_sensitivity.json`, `illustrative_cases.json`: conservative overlap audit, secondary scoring, and concrete examples.','',
             '```bash','.venv/bin/python src/evaluate_official_fever.py --output reports/part2/official_dev_NEW',
             '.venv/bin/python src/report_official_fever.py --run reports/part2/official_dev_NEW','```','',
             'Use a new output directory. Interrupted inference can resume with --resume only when protocol inputs still match; partial JSON writes are rejected rather than silently removed. No old experiment or checkpoint is overwritten.']
    text += ['', '## Per-class recall (same official population)','',
             '| Local condition | SUPPORTS recall | REFUTES recall | NEI recall | Classification macro-F1 |',
             '|---|---:|---:|---:|---:|']
    for name,value in summary['conditions'].items():
        matrix=value['confusion_matrix']
        recall=[100*matrix[i][i]/sum(matrix[i]) for i in range(3)]
        text.append(f"| {names[name]} | {recall[0]:.2f}% | {recall[1]:.2f}% | {recall[2]:.2f}% | {value['classification_macro_f1']:.4f} |")
    (out/'OFFICIAL_COMPARISON.md').write_text('\n'.join(text)+'\n')
    manifest={'status':'verified_complete','n':len(gold),'source_and_checkpoint_hashes_verified':True,
              'official_metrics_reproduced':True,'publication_reference_source':protocol['comparison']['source'],
              'report_source_sha256':digest(__file__),
              'output_sha256':{str(p.name):digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='report_manifest.json'}}
    (out/'report_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Verified full official-development comparison and wrote report.')


if __name__=='__main__':main()
