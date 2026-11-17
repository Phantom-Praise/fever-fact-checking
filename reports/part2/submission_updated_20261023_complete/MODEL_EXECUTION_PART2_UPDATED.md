# Model Execution - Part 2

COSC 874 - Deep Learning II | Praise Ben | Updated October 23, 2026

FEVER fact-checking: hyperparameter tuning, controlled comparisons, reproducibility, and selected results. Branch: experiment/part2-paired-ablation. This report updates the earlier submission snapshot with the held-out comparison and calibrated ensemble.

## Progress against the assignment

| Step | Status | What was done / what remains |
| --- | --- | --- |
| 1. Tune | Completed | Four controlled configurations; H2 selected by validation macro-F1. |
| 2. Experiments | Measured; RQ1 partial | Paired ablations, judge pilot, symbolic v2, corpus growth, held-out comparison, and ensemble calibration. Human review remains pending. |
| 3. Reproducibility | Completed; optional repeats pending | Fixed seeds and saved artifacts. No extra Transformer training seeds; judge/inference repeats are distinct. |
| 4. Results | Updated | Configuration choices and measured results are consolidated here and in the CSV appendix. |

## Step 1 - A small, reasoned tuning study

Part 1 showed majority-class collapse in the original baseline and evidence-model overfitting after epoch 1. The tuning study varied learning rate around 2e-5 and tested removing weight decay. Architecture, batch 8, length 256, seed 42, the same 20% training subset, two epochs, and the first 2,000 validation claims were held fixed.

| Configuration | LR | Weight decay | Epoch | Accuracy | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| H1 | 2e-05 | 0.01 | 2 | 0.6465 | 0.6103 |
| H2 - SELECTED | 1e-05 | 0.01 | 2 | 0.6775 | 0.6202 |
| H3 | 3e-05 | 0.01 | 2 | 0.6370 | 0.5952 |
| H4 | 2e-05 | 0.0 | 2 | 0.6575 | 0.6165 |

H2 epoch 2 is selected because it has the highest macro-F1 (0.6202) and accuracy (0.6775) among H1-H4. The full appendix includes both epochs of each configuration, rather than only the best rows.

The evidence model retains epoch 1: gold-evidence validation accuracy/F1 = 0.9415/0.9335, versus 0.9365/0.9313 at epoch 2. These are oracle-input diagnostics. All NEI examples had empty gold evidence, so 94.15% must not be called end-to-end accuracy.

# Step 2 - Controlled model comparisons

Metrics are label accuracy and macro precision, recall, and F1 across SUPPORTS / REFUTES / NEI. Evidence recall and latency are reported separately. Never rank rows from different samples as though they came from one shared test.

## Paired development sample: identical 100 validation claims

| Condition | Accuracy | Macro P | Macro R | Macro-F1 |
| --- | --- | --- | --- | --- |
| Unweighted reference | 0.4700 | 0.1567 | 0.3333 | 0.2132 |
| Earlier weighted epoch 2 | 0.6000 | 0.5826 | 0.5895 | 0.5766 |
| H2 claim-only - SELECTED | 0.6200 | 0.6121 | 0.5721 | 0.5761 |
| Retrieved top-1 | 0.5100 | 0.3303 | 0.4772 | 0.3876 |
| Retrieved top-5 - SELECTED | 0.5500 | 0.3559 | 0.5124 | 0.4163 |
| Retrieved top-10 | 0.5500 | 0.3534 | 0.5056 | 0.4152 |
| Empty evidence | 0.4500 | 0.5140 | 0.4796 | 0.4513 |
| Wrong-claim evidence | 0.3700 | 0.2772 | 0.3848 | 0.2897 |
| Gold evidence - ORACLE | 0.8400 | 0.8639 | 0.8173 | 0.8315 |
| Symbolic hybrid v1 | 0.5500 | 0.3559 | 0.5124 | 0.4163 |
| Symbolic hybrid v2 | 0.5500 | 0.3559 | 0.5124 | 0.4163 |

Every displayed condition uses the same claims. Frozen inference uses batch 8, length 256, seed 42. Evidence depth, empty input, and wrong-claim input vary within the same evidence checkpoint. Claim-only versus evidence models also differ in training data and checkpoints; that comparison does not isolate the causal benefit of evidence.

## Frozen held-out sample: identical 100 test claims

| Condition | Accuracy | Macro P | Macro R | Macro-F1 |
| --- | --- | --- | --- | --- |
| Unweighted reference | 0.5200 | 0.1733 | 0.3333 | 0.2281 |
| Frozen H2 | 0.6300 | 0.5890 | 0.5746 | 0.5802 |
| Frozen retrieval + RoBERTa | 0.6600 | 0.4206 | 0.5385 | 0.4723 |
| Live symbolic v2 hybrid | 0.6600 | 0.4206 | 0.5385 | 0.4723 |

Seed 874 selects unique test claims without consulting labels; overlap with train/validation is rejected. These frozen comparisons use MPS, batch 1, and length 256. Retrieval versus H2 accuracy difference is +3 points, paired 95% CI [-9, +14], p=0.7283. H2 has higher macro-F1; retrieval/hybrid miss all 22 NEI claims. These test outputs were not used to tune the new ensemble.

# New development experiment - calibrated ensemble

The training-input audit confirmed 28,517 empty-evidence NEI training records and no nonempty-evidence NEI records. Live retrieval supplies text regardless of label. This creates a shortcut opportunity and an input mismatch; it does not prove the internal cause of the model failures.

## Selection protocol and final development choice

Calibration: 300 unique validation claims within the earlier first-2,000 tuning population. Assessment: 200 claims beyond offset 2,000, excluding all earlier normalized claim texts. Four retrieval/text variants x five claim weights x four NEI biases = 80 configurations. Frozen checkpoint weights, length 256, batch 8, seed 42. Selection used calibration macro-F1, then accuracy; no assessment labels were used to select the configuration.

PROVISIONAL ensemble choice: original top-5 retrieval; 0.75 x log P(claim-only) + 0.25 x log P(evidence model); add 2 to the NEI score, then choose the maximum. Calibration accuracy/F1: 0.7033/0.6493. The score adjustment is not a calibrated posterior probability.

| 200-claim assessment | Accuracy | Macro P | Macro R | Macro-F1 |
| --- | --- | --- | --- | --- |
| H2 claim-only | 0.6700 | 0.6646 | 0.6262 | 0.6262 |
| Original evidence-only | 0.6100 | 0.3815 | 0.5402 | 0.4471 |
| Local BM25 evidence-only | 0.6150 | 0.3889 | 0.5297 | 0.4471 |
| Subject-linked BM25 | 0.6250 | 0.4022 | 0.5640 | 0.4695 |
| Subject BM25 with titles | 0.6250 | 0.4039 | 0.5640 | 0.4706 |
| Ensemble - PROVISIONAL | 0.7100 | 0.6979 | 0.6580 | 0.6655 |
| Ensemble, no NEI adjustment | 0.6350 | 0.7590 | 0.5570 | 0.4954 |
| NEI adjustment, evidence-only | 0.6100 | 0.3815 | 0.5402 | 0.4471 |

Ensemble NEI recall is 43.1%, versus 36.2% for H2 and 0% for original evidence-only. Accuracy gain versus evidence-only: +10 points, paired bootstrap CI [+3.5, +16.5], p=0.0045. Versus H2: +4 points, CI [-1, +9], p=0.1686. Superiority over H2 is not established. Statistics are exploratory and unadjusted for multiple comparisons.

Complete evidence-group recall@5 on the same 142 verifiable assessment claims: original retrieval 58.45%; local BM25 52.82%; subject-linked BM25 55.63%. The new rankers were not selected. A corrected parser run repeats the same protocol/sample; the earlier run is preserved as superseded, not counted as an independent replication.

## Prepared for future training - not a trained improvement

A 96-candidate training-data pilot retained 77 nonempty-evidence records: 25 SUPPORTS, 20 REFUTES, 32 NEI. Verifiable records must contain a complete gold group; 19 were skipped rather than relabeled. This filtering introduces retrieval-success bias. A larger, matched training comparison requires approval; no retraining was run.

# RQ1, RQ2 and RQ4 - bounded findings

## RQ1 - Generated claims: human review still pending

Thirty local-LLM claims and source-blinded review sheets for 60 claims were prepared. No human reviews have been completed. Two automated coding models each achieved only 30% accuracy on the 30 FEVER reference claims and returned constant NONE error categories. Their labels cannot establish generated-claim factual-error rates. RQ1 remains incomplete.

## RQ2 - Local LLM judge accuracy and consistency

Same balanced 30 FEVER claims, two evidence conditions, two prompt wordings, three repeats = 360 actual local requests. Installed Llama 3.2 was chosen for availability. Temperature .2; seeds 42/43/44. Values are mean +/- sample SD across inference repeats, not training repeats.

| Evidence / prompt | Accuracy | Macro-F1 | 3 repeats agree |
| --- | --- | --- | --- |
| retrieved:direct | 0.3778 +/- 0.0192 | 0.3655 +/- 0.0220 | 0.8333 |
| retrieved:evidence_first | 0.4000 +/- 0.0000 | 0.3850 +/- 0.0091 | 0.8333 |
| empty:direct | 0.3333 +/- 0.0000 | 0.1667 +/- 0.0000 | 1.0000 |
| empty:evidence_first | 0.3333 +/- 0.0000 | 0.1667 +/- 0.0000 | 1.0000 |

On these same 30 claims: H2 accuracy/F1 = .5000/.4818; retrieved RoBERTa = .5333/.4274. Empty-evidence judge prompts are abstention controls and consistently return NEI. High consistency does not imply correctness. This small local-model pilot does not justify a universal claim about LLM judges.

## RQ4 - Nested corpus growth and latency

| Pages | Complete R@5 | Accuracy | Baseline ms | Cached route ms |
| --- | --- | --- | --- | --- |
| 2,000 | .6056 | .5400 | 61.94 | 67.69 |
| 10,000 | .6056 | .5500 | 58.67 | 57.68 |
| 50,000 | .6197 | .5400 | 64.80 | 58.00 |
| 5,396,106 | .6197 | .5500 | 106.26 | 113.21 |

Same 100 claims, nested query-aware indexes, top-5, frozen model, CPU, batch 1; two alternating-order repeats. Every route still makes 100 retrieval and 100 model calls. Cached routing excludes live Wikidata latency. Query-aware core retention, caches, and fixed corpus-size order limit generalization.

| Held-out MPS pass | Retrieval ms/claim | Hybrid ms/claim |
| --- | --- | --- |
| Initial online | 66.35 | 106.57 |
| Warm snapshot 1 | 58.89 | 60.26 |
| Warm snapshot 2 | 56.97 | 50.78 |

Initial hybrid timing includes seven real HTTP attempts with no failures. Symbolic v2 resolves zero claims and avoids zero model/retrieval calls. Warm differences fluctuate without avoided work. Neither a live hybrid speedup nor monetary cost savings is established. Ensemble latency was not benchmarked.

# Selected configurations and reproducibility

| Component | Chosen reporting configuration | Reason |
| --- | --- | --- |
| Unweighted baseline | Original epoch 1; reference | Retained for comparison; majority-class behavior. |
| Claim-only RoBERTa | SELECTED: H2 epoch 2 | LR 1e-5, WD .01, batch 8; best tuning macro-F1 .6202. |
| Evidence RoBERTa | SELECTED: epoch 1 | LR 2e-5, WD .01, batch 8; best oracle-validation macro-F1 .9335. |
| Retrieval | SELECTED: title-aware top-5 | Frozen paired reference; new rankers did not improve recall. |
| Symbolic hybrid | V1/v2 reported; no winner | Zero resolved claims; no measured avoided calls. |
| Neural ensemble | PROVISIONAL: w=.75; bias=2 | Best calibration macro-F1; original retrieval; no new test evaluation. |
| LLM judge | Llama 3.2 local pilot | Availability-based comparator; not a performance-selected final judge. |

## Step 3 - Reproducibility and honest limits

Research commands used the existing .venv. Training/evaluation seed: 42; held-out selection seed: 874; judge repeats: 42/43/44. Reports retain predictions, prompts, raw responses, selection protocols, model digests, source hashes, and index memberships where recorded. The calibrated verifier reproduced all 200 saved ensemble predictions. Thirty unit tests passed.

No extra Transformer training seeds were run. Epochs, repeated judge calls, inference timing passes, and duplicated historical log rows are not independent training replications. Two additional H2 seeds remain optional if time/compute permits and require approval. The new ensemble has development evidence only; testing on a newly reserved population remains future work.

## Step 4 - What is included

The PDF summarizes selected model configurations and all principal controlled conditions. The accompanying appendix preserves all 14 historical training-log observations, every saved classifier evaluation separated by population, all 12 judge condition/repeat rows, original retrieval and latency tables, all 80 calibration candidates, and configuration-selection rationales. Two identical legacy epoch-1 log rows remain observations, not two runs.

A 100-claim validation extension is also retained in the classifier appendix: H2 .7000 accuracy / .5788 F1; top-5 .5700 / .3707; top-10 .6000 / .3961. It is disjoint from the first 100 by ID/text but still belongs to the validation tuning population. Its incomplete oracle condition was omitted rather than scored on a different N.

## Overall completion status

The tuning, reproducible artifacts, and updated results table are completed for the measured study. RQ2, RQ3, and RQ4 have bounded measured experiments, including negative findings. RQ1 cannot be completed until independent human reviews and adjudication are supplied. The symbolic-efficiency hypothesis is unsupported by the measured implementation. Prepared training data is not a trained fix.

## Sources, access and disclosure

Continuation source: user-supplied Model_Execution_Part_1.pdf. Experiment sources: reports/part2/submission_20261023_final; step2_final_20261023/report; literature_improvements_20261023_corrected. Source hashes and attachment inventory are in manifest.json. No earlier report or result was overwritten, and nothing was merged into main.

Run the experimental ensemble from the repository root: .venv/bin/python src/verify_calibrated.py --claim "The Eiffel Tower is in Paris."

AI assisted code, planning, analysis, and writing. Measurements come from saved local runs; human labels were not fabricated. Review this disclosure against the course AI-use policy.
