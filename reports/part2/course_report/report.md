# Model Execution - Part 2

Praise Ben | COSC 874: Deep Learning II

This stage tests how training settings, retrieved evidence, and a symbolic lookup affect the FEVER fact-checking system. The task has three labels: SUPPORTS, REFUTES, and NOT ENOUGH INFO (NEI). Accuracy and macro-F1 are the main classification measures; macro-F1 gives each class equal weight.

## 1. Hyperparameter tuning

Part 1 showed that the original baseline predicted only SUPPORTS. Class-weighted loss produced predictions for all three classes, so that loss was retained for tuning. The search compared a lower and a higher learning rate with the original 2e-5 setting, then tested whether removing weight decay helped.

All four configurations used RoBERTa-base, batch size 8, maximum length 256, two epochs, and seed 42. They shared the same 20% training subset and the first 2,000 validation claims. The table shows the best epoch from each run; the attached training log includes both epochs.

| Run | Learning rate | Weight decay | Epoch | Accuracy | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| H1 | 2e-05 | 0.01 | 2 | 64.65% | 0.6103 |
| H2 (chosen) | 1e-05 | 0.01 | 2 | 67.75% | 0.6202 |
| H3 | 3e-05 | 0.01 | 2 | 63.70% | 0.5952 |
| H4 | 2e-05 | 0.0 | 2 | 65.75% | 0.6165 |

H2 was chosen because it had the highest validation macro-F1 (0.6202), as well as the highest accuracy. Its learning rate is 1e-5 and its weight decay is 0.01. These results support keeping the lower learning rate for the claim-only model.

For the evidence model, epoch 1 was retained. Its gold-evidence validation macro-F1 was 0.9335, compared with 0.9313 at epoch 2. Validation loss also increased at epoch 2, despite lower training loss. This supported stopping at the earlier checkpoint.

The high gold-evidence score is a diagnostic result, not end-to-end accuracy. The gold inputs give NEI claims empty evidence, while SUPPORTS and REFUTES receive text. This creates a shortcut that does not match live retrieval. The retrieved-evidence evaluations below avoid using gold evidence as model input.

## 2. Experiments

### Full FEVER development evaluation

The main evaluation used all 19,998 official FEVER development claims, with 6,666 examples per class. The three systems below used the same claims. Their checkpoints and decision settings were fixed before this evaluation; none was trained or tuned on these claims.

| System | Accuracy | Macro P | Macro R | Macro-F1 | FEVER score |
| --- | --- | --- | --- | --- | --- |
| Claim-only H2 | 58.78% | 0.6211 | 0.5878 | 0.5775 | 12.23% |
| Retrieved evidence + RoBERTa | 52.37% | 0.3524 | 0.5237 | 0.4175 | 35.07% |
| Combined model (chosen) | 61.70% | 0.6480 | 0.6170 | 0.6078 | 45.33% |

The combined model reached 61.70% accuracy and 0.6078 macro-F1. It improved accuracy by 2.92 percentage points over claim-only H2. A paired bootstrap interval for this difference was +2.45 to +3.37 points. This is a measured gain on this benchmark, although related claims may share evidence and are not fully independent.

The official FEVER score also checks evidence: a correct SUPPORTS or REFUTES label needs a complete annotated evidence group among the first five submitted sentences. Correct NEI labels do not need evidence. The claim-only model supplies no evidence, so its FEVER score comes only from correct NEI predictions.

### Evidence ablations on a shared 100-claim validation sample

| Condition | Accuracy | Macro-F1 |
| --- | --- | --- |
| Original unweighted baseline | 47.00% | 0.2132 |
| Earlier weighted baseline | 60.00% | 0.5766 |
| Claim-only H2 | 62.00% | 0.5761 |
| Top-1 retrieved sentence | 51.00% | 0.3876 |
| Top-5 retrieved sentences | 55.00% | 0.4163 |
| Top-10 retrieved sentences | 55.00% | 0.4152 |
| Empty evidence | 45.00% | 0.4513 |
| Evidence from another claim | 37.00% | 0.2897 |
| Gold evidence (diagnostic) | 84.00% | 0.8315 |
| Symbolic v1 + top-5 | 55.00% | 0.4163 |
| Symbolic v2 + top-5 | 55.00% | 0.4163 |

Top-10 tied top-5 accuracy and had slightly lower macro-F1, so top-5 was kept. Wrong-claim evidence reduced performance. Empty evidence sometimes scored better than retrieved evidence, showing that retrieval alone did not solve the classification problem. The symbolic stages resolved none of these claims and did not change the predictions.

These are development diagnostics, separate from the full benchmark above. The evidence ablations share one checkpoint, but claim-only and evidence models were trained on different subsets. Their comparison does not isolate the effect of adding evidence.

## Additional experiments

### Combining the claim-only and evidence models

A separate calibration experiment used 300 internal validation claims to choose a decision rule, then assessed it on 200 different claims beyond the earlier tuning sample. The chosen rule weights claim-only log probabilities by 0.75 and evidence-model log probabilities by 0.25, then adds 2 to the NEI score before choosing a label.

| 200-claim assessment | Accuracy | Macro-F1 |
| --- | --- | --- |
| Claim-only H2 | 67.00% | 0.6262 |
| Evidence model | 61.00% | 0.4471 |
| Combined model | 71.00% | 0.6655 |
| Combined, without NEI adjustment | 63.50% | 0.4954 |
| NEI adjustment, without claim model | 61.00% | 0.4471 |

The combined model scored 71% on this small assessment. The 61.70% reported on the previous page comes from the much larger official set; the two percentages describe different populations. Alternative BM25 sentence-ranking methods did not improve complete evidence recall@5 over the original retriever, so the original retrieval method was retained.

### Local LLM judge

The judge experiment used the same balanced 30 FEVER claims for two prompts, two evidence conditions, and three repeats: 360 local Llama 3.2 requests. The table gives mean +/- sample standard deviation across repeats.

| Condition | Accuracy | Macro-F1 |
| --- | --- | --- |
| Retrieved evidence, direct prompt | 37.78% +/- 1.92 | 0.3655 +/- 0.0220 |
| Retrieved evidence, evidence-first prompt | 40.00% +/- 0.00 | 0.3850 +/- 0.0091 |
| Empty evidence, direct prompt | 33.33% +/- 0.00 | 0.1667 +/- 0.0000 |
| Empty evidence, evidence-first prompt | 33.33% +/- 0.00 | 0.1667 +/- 0.0000 |

On those same 30 claims, claim-only H2 reached 50% accuracy and retrieved RoBERTa reached 53.33%. The judge returned NEI consistently without evidence. That consistency did not make it accurate on a balanced three-class sample.

### Corpus size and symbolic lookup

| Pages | Accuracy | Retrieval ms/claim | Cached route ms/claim |
| --- | --- | --- | --- |
| 2,000 | 54% | 61.94 | 67.69 |
| 10,000 | 55% | 58.67 | 57.68 |
| 50,000 | 54% | 64.80 | 58.00 |
| 5,396,106 | 55% | 106.26 | 113.21 |

These runs used the same 100 claims, nested indexes, CPU inference, and two timing repeats. The indexes retained a common set of candidate pages, so they test distractor growth rather than random evidence availability. Every route still called retrieval and RoBERTa for all 100 claims. In a separate live MPS evaluation, the initial symbolic route took 106.57 ms per claim versus 66.35 ms without it. These experiments did not demonstrate an efficiency benefit.

## 3. Reproducibility

Training and model evaluation used seed 42. The 100-claim internal test sample used selection seed 874, and the LLM judge repeats used seeds 42, 43, and 44. Checkpoints, settings, predictions, source hashes, and raw judge responses were saved. The official FEVER scorer was used unchanged, with a five-sentence evidence limit.

The Transformer training runs were not repeated with additional seeds. Judge repeats and timing repeats therefore measure inference variation, not variation between independently trained models. Additional training seeds would strengthen the result, but were not run in this study.

The official development IDs did not overlap the original training pool. Three claim texts appeared in both pools; reconstruction of the sampled training subsets found none of those texts in either model's training subset. A separate analysis excluded the three texts, while the main benchmark retained the complete official set. The index also lacked a complete annotated evidence group for 130 verifiable claims; these claims remained in the evaluation.

## 4. Final configuration choices

| Model / component | Chosen configuration | Reason |
| --- | --- | --- |
| Unweighted baseline | Original epoch 1 | Reference for the majority-class failure in Part 1. |
| Claim-only model | H2, epoch 2; LR 1e-5; weight decay 0.01 | Highest validation macro-F1 among H1-H4. |
| Evidence model | Epoch 1; LR 2e-5; weight decay 0.01 | Better gold-validation macro-F1 than epoch 2; evaluated separately with retrieved text. |
| Retrieval | Original title-aware retrieval, top-5 | Top-10 gave no consistent benefit; new rankers did not improve recall. |
| Combined model | Claim weight 0.75; NEI adjustment 2 | Selected on internal calibration before the official evaluation; strongest local full-benchmark result. |
| Symbolic stage | Experimental comparison only | No resolved claims or avoided model calls in the measured samples. |

The combined model is the main configuration for reporting. Its full-benchmark result is 61.70% accuracy, 0.6078 macro-F1, and 45.33% FEVER score. For context, GEAR reported 74.84% accuracy / 70.69% FEVER score on the official development set, and KGAT with RoBERTa-large reported 78.29% / 76.11% [1]. Their models were not rerun here, and their training budgets, retrieval systems, and corpus filtering differ.

The largest unresolved training issue is the empty-evidence NEI input. A small replacement-data pilot was prepared, but no replacement model was trained. The planned comparison of FEVER and generated-claim error types also remains incomplete: 30 generated claims still need independent human review. The automated coding pilot was too inaccurate to supply reliable labels.

The attached CSV files contain all 14 historical training-log observations, 44 classifier evaluations grouped by sample, 12 judge condition/repeat rows, and the retrieval and timing results. The calibration file records all 80 decision/ranker combinations. Duplicate historical log entries are retained as observations, not counted as independent runs.

[1] Liu et al. (2020). Fine-grained Fact Verification with Kernel Graph Attention Network, Table 2. https://aclanthology.org/2020.acl-main.655/

AI assistance was used for coding and writing. The reported measurements come from the saved local experiment outputs.
