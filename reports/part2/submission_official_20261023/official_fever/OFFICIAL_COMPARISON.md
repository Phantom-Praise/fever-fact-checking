# Same-population, same-scoring FEVER comparison

Completed evaluation on **all 19,998 official FEVER development claims** (6,666 per label). No retraining, parameter selection, or rule changes were performed. Previous results remain preserved.

| System | Origin | Label accuracy | Strict FEVER score |
|---|---|---:|---:|
| Our frozen claim-only H2 | Measured locally | 58.78% | 12.23% |
| Our frozen retrieved RoBERTa | Measured locally | 52.37% | 35.07% |
| Our frozen calibrated ensemble | Measured locally | 61.70% | 45.33% |
| GEAR | Published development reference; not rerun | 74.84% | 70.69% |
| KGAT_BERT_Base_ESIM | Published development reference; not rerun | 75.51% | 71.61% |
| KGAT_BERT_Base | Published development reference; not rerun | 78.02% | 75.88% |
| KGAT_RoBERTa_Large | Published development reference; not rerun | 78.29% | 76.11% |

## What the percentages mean

Label accuracy checks the predicted class. Strict FEVER score also requires a complete annotated evidence group among the first five submitted sentences for SUPPORTS/REFUTES. Correct NEI predictions need no evidence. Macro-F1 for classification and evidence F1 are different metrics and are not substituted for FEVER score.

The ensemble correctly labels 12,338 of 19,998 claims; 9,065 satisfy the strict label-and-evidence rule. Its label accuracy differs from evidence-only by +9.33 percentage points and claim-only by +2.92 points.

Against claim-only H2, the paired claim-bootstrap 95% interval for the ensemble's accuracy gain is [+2.45, +3.37] percentage points (2,000 resamples, seed 42). This supports a measured accuracy gain on this benchmark. Resampling is by claim, not by shared evidence page; related claims can be correlated. Local paired tests are exploratory and unadjusted for multiple comparisons. No such paired test against the papers is claimed.

The claim-only reference submits no evidence. Its strict score therefore comes only from correctly predicted NEI claims; it is not presented as a complete evidence-producing fact-checker. The evidence and ensemble methods submit the same original top-five retrieved sentences.

## Exactly what now matches the papers

- The complete official development population, not a 100/200-claim internal sample.
- The three original FEVER labels and published human evidence annotations.
- The unmodified official scorer with max_evidence=5.
- Development results are compared with development results; published test numbers (GEAR 67.10, KGAT 70.38 FEVER) are not mixed into this table.

The published reference values come from [KGAT Table 2, Dev columns](https://aclanthology.org/2020.acl-main.655.pdf). Its Table 1 confirms the official development class counts. [GEAR](https://aclanthology.org/P19-1085.pdf) is the original source for the GEAR architecture/results. [Official scoring source](https://github.com/sheffieldnlp/fever-scorer) is vendored at commit 4801615100fbf6327f8e99b5dbaefe5dd890e869, unchanged, with its license/provenance.

## What remains different

We have not rerun the authors' trained models locally. Their rows are published reference measurements; no paired significance test against them is possible without their per-claim predictions. Our base models, training subsets, retrieval architecture, and input formatting differ from their systems. Thus matching population and scoring enables a benchmark comparison, not a controlled causal comparison of architectures or a reproduction of their full training pipeline.

Our Wikipedia index has 5,396,106 pages; KGAT describes a 5,416,537-document background collection. Exact document filtering/corpus membership has not been matched. This is an additional retrieval-resource difference, not a matched-corpus ablation.

The annotation-availability audit found 7,028 of 7,100 distinct development evidence references in our index. 130 verifiable claims have no complete annotated group available in that index. They remain in the full official denominator; none is silently dropped or given oracle evidence. This audit does not feed gold evidence to the model.

The hybrid-paper binary F1 and FActScore atomic-fact precision are different tasks/metrics and are intentionally excluded from this comparison. Matching these two FEVER papers does not make every literature-review result comparable.

## Leakage and reproducibility audit

Official IDs overlap with the original training pool: 0. Normalized claim-text overlaps: 3. This audit is conservative against the entire original training pool, not just claims actually sampled for fitting. All official rows are retained for population matching. The secondary no-text-overlap sensitivity check scores 19,995 claims separately and must not be labeled the full official benchmark.

A supplementary reconstruction of each model's seed-42, 20% training subset found zero exact normalized-text overlaps for either model. That reconstruction follows the current training scripts; historical per-step training manifests are incomplete, so the broader three-text audit is retained as well. See training_subset_reconstruction_audit.json.

The ensemble configuration was frozen from the earlier internal-validation study: original top-5 retrieval, claim log-probability weight .75, NEI bias 2. Seed 42, MPS, inference batch 8, maximum length 256. No gold labels/evidence enter retrieval or neural inference. Do not retune on this evaluation and then describe it as an untouched test. Official development is public validation data, not the blind test set.

Thirty-five unit tests passed before evaluation, including complete-group matching, the five-sentence cap, correct NEI handling, sample-order rejection, and separation of label versus strict scores. This report re-scores all saved submissions and verifies exact reproduction of saved metrics and protocol source/checkpoint hashes.

## Files and reproduction

- `comparison.csv`: local measured results plus clearly labeled published references.
- `claim_only_H2.jsonl`, `retrieved_roberta.jsonl`, `calibrated_ensemble.jsonl`: complete standard FEVER predictions in original official ID order.
- `outputs.jsonl`: component probabilities and retrieved sentence references.
- `protocol.json`, `runner_source.py`, `summary.json`: frozen choices, source/checkpoint hashes, official metrics, and paired local comparisons.
- `normalized_overlap_audit.json`, `nonoverlap_sensitivity.json`, `illustrative_cases.json`: conservative overlap audit, secondary scoring, and concrete examples.

```bash
.venv/bin/python src/evaluate_official_fever.py --output reports/part2/official_dev_NEW
.venv/bin/python src/report_official_fever.py --run reports/part2/official_dev_NEW
```

Use a new output directory. Interrupted inference can resume with --resume only when protocol inputs still match; partial JSON writes are rejected rather than silently removed. No old experiment or checkpoint is overwritten.

## Per-class recall (same official population)

| Local condition | SUPPORTS recall | REFUTES recall | NEI recall | Classification macro-F1 |
|---|---:|---:|---:|---:|
| Our frozen claim-only H2 | 82.16% | 57.50% | 36.68% | 0.5775 |
| Our frozen retrieved RoBERTa | 89.96% | 67.15% | 0.00% | 0.4175 |
| Our frozen calibrated ensemble | 86.42% | 56.81% | 41.85% | 0.6078 |
