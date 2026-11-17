# Literature-informed code improvements — October 23, 2026

The new calibrated ensemble reached **71% label accuracy and 0.6655 macro-F1** on a 200-claim development assessment. Evidence-only reached 61% and claim-only H2 reached 67% on those same claims. This is a promising development result; superiority over H2 is not established by this sample. No new test-partition evaluation or neural retraining was performed.

## What the literature suggests

The project's `PraiseBen_LitReview.xlsx`, sheet `Literature Comparison`, connects the project to FEVER, GEAR, KGAT, FactKG, HoVer, FActScore, a hybrid system, and GraphCheck. The proposal and data-collection documents were also reviewed to connect these changes to RQ1–RQ4.

- [FEVER](https://aclanthology.org/N18-1074/) treats evidence retrieval and label prediction as parts of verification. That motivates measuring evidence recall separately from label accuracy.
- [GEAR](https://aclanthology.org/P19-1085/) and [KGAT](https://aclanthology.org/2020.acl-main.655/) motivate attention to relationships and relevance among evidence sentences. We tested cheap sentence-ranking changes; **we did not implement either trained graph model**.
- [FActScore](https://aclanthology.org/2023.emnlp-main.741/) motivates checking atomic facts against evidence. Our generated-claim human reviews remain unfinished; code changes cannot replace those reviews.
- [FactKG](https://aclanthology.org/2023.acl-long.895/) and [GraphCheck](https://arxiv.org/abs/2502.16514) are possible future graph-oriented directions. They were not implemented or evaluated in this update.

Two literature-review details need care in the eventual paper. The GEAR/KGAT FEVER scores are not directly comparable with our label accuracy. The review's hybrid-paper link points to [Kolli et al., WiNLP 2025](https://aclanthology.org/2025.winlp-main.19/), rather than the Zhang/EMNLP-main attribution in the workbook. The workbook itself was preserved.

## A concrete training-data problem

The existing evidence preparation supplies no evidence text for NEI claims. The audit found:

| Partition | SUPPORTS with text | REFUTES with text | NEI with empty text | NEI with text |
|---|---:|---:|---:|---:|
| Evidence training | 63,470 | 23,653 | 28,517 | 0 |
| Evidence validation | 7,890 | 2,958 | 3,595 | 0 |

Live retrieval usually supplies text even for NEI. This creates an input mismatch and an opportunity for an empty-input shortcut. It does not prove which internal features the model learned. On the new assessment, every evidence-only variant had zero NEI recall. See `training_input_audit.json` for the measured counts.

## Changes implemented and run

1. `src/evidence_pipeline_v2.py` adds local BM25 sentence ranking, optional subject-page linking, optional page titles, and a validated log-probability fusion rule. Original retrieval stays available.
2. `src/experiment_literature_improvements.py` compares four retrieval/text conditions and 80 decision configurations using frozen local checkpoints. It saves a protocol, calibration candidates, probabilities, predictions, paired uncertainty estimates, retrieval recall, and source snapshots.
3. `src/verify_calibrated.py` makes the selected decision rule usable on new claims. It returns the predicted label, retrieved evidence, component probabilities, and decision settings. It is an experimental neural fallback, not a new symbolic verifier.
4. `src/prepare_retrieved_training.py` prepares a bounded pilot with retrieved text for all classes. It never changes a human label because retrieval failed. It is preparation for a future training run, not a trained replacement.

All research execution used `.venv/bin/python`. Existing results, model weights, submission snapshots, and symbolic rules were preserved. Work remains on the experiment branch; nothing was merged into main.

## Selection and results

Calibration used 300 unique claims within the earlier first-2,000 validation tuning population. Assessment used 200 unique claims scanned after offset 2,000, excluding every normalized claim in that earlier population. Checkpoints, top-five neural evidence inputs, maximum length 256, batch size 8, and seed 42 were fixed. There was no test-partition read.

The configuration was selected by calibration macro-F1, then accuracy, with the declared tie breaks. The selected configuration file was written before assessment scoring. Its calibration accuracy was 70.33% and macro-F1 was 0.6493.

The selected rule keeps **original retrieval**, uses weight 0.75 for claim-only log probabilities and 0.25 for evidence-model log probabilities, and adds 2 to the NEI score before taking the maximum. These scores are not calibrated posterior probabilities. The NEI adjustment is an engineering hypothesis motivated by measured failures, not an algorithm attributed to the reviewed papers.

| Assessment condition | Accuracy | Macro-F1 | NEI recall |
|---|---:|---:|---:|
| Claim-only H2 | 67.0% | 0.6262 | 36.2% |
| Original evidence-only | 61.0% | 0.4471 | 0.0% |
| Local BM25 evidence-only | 61.5% | 0.4471 | 0.0% |
| Subject-linked BM25 evidence-only | 62.5% | 0.4695 | 0.0% |
| Subject-linked BM25 with titles | 62.5% | 0.4706 | 0.0% |
| **Selected calibrated ensemble** | **71.0%** | **0.6655** | **43.1%** |
| Selected weights, no NEI adjustment | 63.5% | 0.4954 | 3.4% |
| Selected NEI adjustment, evidence-only | 61.0% | 0.4471 | 0.0% |

Across the 142 verifiable assessment claims, complete evidence-group recall at five was 58.45% for original retrieval, 52.82% for local BM25, and 55.63% for subject-linked BM25. Therefore the new ranking alternatives were **not promoted**. The gain came from the selected decision combination, not a demonstrated retrieval improvement.

The ensemble's paired accuracy difference versus evidence-only was +10 percentage points, with bootstrap 95% interval [+3.5, +16.5] and unadjusted exact McNemar p=0.0045. Versus H2 it was +4 points, with interval [-1, +9] and p=0.1686. These are exploratory development comparisons without multiplicity correction. The H2 comparison does not establish a reliable gain. All conditions and per-class results are retained in `summary.json` and `results.csv`.

An initial exploratory run exposed a parser bug: generic trailing-character stripping could damage names ending in `s`. The parser was fixed, regression tests were added, and the same protocol/grid was rerun. The original run remains in the sibling directory with `SUPERSEDED.json`. The repeated assessment is the same sample, not another independent replication. No configuration was changed based on the assessment scores.

## Training preparation, verification, and remaining work

The pilot sampled 32 original training claims per class (96 candidates, seed 42). It retained 77: 25 SUPPORTS, 20 REFUTES, and 32 NEI. All retained examples have retrieved text; the 45 verifiable examples contain a complete annotated group. Nineteen verifiable claims were skipped rather than relabeled. The pilot uses subject-linked retrieval with titles as a data-format demonstration, not the selected inference configuration. Future training must deliberately match and evaluate the chosen retrieval/text format.

Files are in `data/processed/retrieved_training_pilot_20261023`; a manifest is copied into this report folder. Filtering introduces retrieval-success bias, and FEVER evidence annotations are not exhaustive. The 77 examples are not sufficient for a final replacement model. A larger controlled training set and comparison against the original evidence model would require a separately approved retraining run. Keep maximum length, data budget, seeds, and evaluation inputs matched, and report class retention and retrieval failures.

Thirty unit tests passed, including probability validation, name preservation, subject linking, assessment separation, and preventing inappropriate training-label changes. Source compilation passed. The additional `inference_integration_check.json` records whether the reusable verifier reproduces all 200 saved ensemble predictions and fingerprints the local checkpoints. This replay is a software integration check, not new statistical evidence.

Run the experimental classifier from the repository root:

```bash
.venv/bin/python src/verify_calibrated.py --claim "The Eiffel Tower is in Paris."
```

To repeat the experiment, use a fresh output directory:

```bash
.venv/bin/python src/experiment_literature_improvements.py --output reports/part2/literature_repeat_NEW
```

Fusion calls two models, so no latency/cost improvement is claimed. Its latency, symbolic-first integration, generalization to a new reserved evaluation population, and sensitivity across seeds remain unmeasured here. The earlier held-out results remain historical results for their original configurations. This update advances RQ3's decision ablation; it does not complete RQ1 human review or establish the RQ4 efficiency hypothesis.
