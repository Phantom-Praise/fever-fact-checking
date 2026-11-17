# Result Analysis

Evidence Inputs and Decision Rules in a Local FEVER Fact-Checker

Praise Ben | COSC 874: Deep Learning II

## Adopted question and scope

How do evidence inputs and decision rules affect label accuracy and evidence-grounded verification in a local FEVER fact-checker?

This narrower scope is explicitly adopted after reviewing results. The revised questions and working hypotheses are retrospective, not preregistered. Original proposals, numerical results and earlier reports are preserved.

The original question about representativeness of LLM-generated factual errors remains unanswered and moves to future work because independent human review is incomplete. The following Revised RQ numbering does not imply completion of that original question.

## Revised questions and working hypotheses

| ID | Question | Working hypothesis |
| --- | --- | --- |
| Revised RQ1 | How do evidence relevance and input length affect verification? | H1a: claim-related evidence outperforms deliberately mismatched evidence. H1b: doubling the evidence-model input limit produces a reliable accuracy gain. |
| Revised RQ2 | Does frozen RoBERTa fusion outperform its individual components on the official development set? | H2: validation-selected fusion improves label accuracy and macro-F1 over each individual model. |
| Revised RQ3 | Can a selected local LLM pipeline configuration improve over its original setup on disjoint assessment claims? | H3: the calibration-selected configuration improves accuracy and macro-F1 on the separate assessment sample. |
| Revised RQ4 | Does the implemented symbolic-first route avoid model calls and reduce latency? | H4: symbolic routing avoids expensive fallback calls and reduces mean inference-pipeline time without lowering label accuracy. |

## Hypothesis verdicts with evidence

Verdicts are limited to the named measured conditions. A retrospective supported finding is not a preregistered confirmation. Macro-F1 below means three-class classification F1, not evidence-retrieval F1.

| Hypothesis | Verdict | Measured evidence | Interpretation and boundary | Source |
| --- | --- | --- | --- | --- |
| H1a · evidence relevance | Supported within the measured diagnostic | Same 100 claims/checkpoint: own top-5 evidence 55% accuracy / 0.4163 F1; another claim’s evidence 37% / 0.2897. Accuracy difference +18 points; paired 95% interval +8 to +28. | Evidence selection matters in this controlled development slice. It does not prove that any retrieved text is better than no evidence. | E2 |
| H1b · longer input | No convincing support | Same 19,998 claims: combined accuracy 61.696% → 61.706%; 14 errors fixed, 12 introduced. Evidence-only: 7 fixed, 6 introduced. | Only two additional correct combined labels. The interval includes no gain, and Telemundo remained wrong. | E3 |
| H2 · frozen fusion | Supported on this benchmark | 19,998 claims: fusion 61.70% / 0.6078 F1, claim-only 58.78% / 0.5775, evidence-only 52.37% / 0.4175. Fusion-minus-claim accuracy +2.92 points; interval +2.45 to +3.37. | Measured improvement for these frozen outputs and this development population. It does not establish new-data or retraining-seed generality. | E1 |
| H3 · selected LLM pipeline | Partially supported; observed gain remains uncertain | Select on 60 calibration claims; assess on 120 disjoint claims. Original 51.67% / 0.4973 F1 → selected hybrid 59.17% / 0.5821. 25 errors fixed, 16 introduced; accuracy interval −2.50 to +18.33 points. | Promising direction, not a reliable general improvement yet. The selected system is a hybrid, and NEI recall fell. | E5, E5a–E5d |
| H4 · symbolic efficiency | Not supported by the implemented route | Matched 100-claim online run: zero resolved claims, zero avoided model calls; 100 model calls in both routes. Mean 106.57 ms with routing versus 66.35 ms without. | The implemented rules added work without resolving claims. This is a bounded negative result, not a verdict on all symbolic systems. | E8 |

## Populations: which scores may be compared?

| Population | Size and source | Class composition | Use | Comparison limit |
| --- | --- | --- | --- | --- |
| P1 | 19,998 official FEVER development claims | 6,666 per class | Main frozen-model comparison and 256/512 input comparison. | Do not compare directly with a small-sample percentage. |
| P2 | 100 internal-validation claims | Original slice; not an untouched test | Evidence relevance/depth ablations with the same checkpoint. | Development diagnostic includes the earlier tuning pool. |
| P3 | 300 official-development claims | 100 per class | Original Llama 3.1 8B pilot versus neural references. | One local-model run; Telemundo kept separate. |
| P4 | 60 internal-validation calibration claims | 20 per class | Select among five predeclared configurations. | Selection data; not an unbiased final score. |
| P5 | 120 new LLM assessment claims | 40 per class | Original LLM versus the frozen selected pipeline. | Disjoint from P4 and the earlier P3 texts; still official development, not a blind test. |
| P6 | 20 selected cases from P3 | Verifiable cases lacking a complete original annotated group | Artificial replacement with a complete gold-evidence group. | Oracle diagnostic, not production accuracy; not an independent full sample. |
| P7 | 100 internal held-out claims | 52 SUPPORTS / 26 REFUTES / 22 NEI | Symbolic comparison and online pipeline timing. | Separate population/device conditions; never pool with P1–P6. |

## Main benchmark and literature context

P1 uses all 19,998 official development claims and the unchanged official scorer. Label accuracy checks the class; strict FEVER additionally requires a complete annotated evidence group among the first five submitted sentences for SUPPORTS/REFUTES. Correct NEI answers do not require evidence.

The published rows use the development label-accuracy and FEVER columns of KGAT Table 2, not its test scores. They are numerical benchmark references, not equal-resource reproductions. Architectures, retrieval, training data/budgets and model sizes differ.

GEAR and KGAT learn evidence-graph aggregation. This project uses lexical retrieval, concatenated evidence and classifier-score fusion; it did not reproduce those graph architectures. The local result remains below the published development references.

Sources: FEVER — https://aclanthology.org/N18-1074/ ; GEAR — https://aclanthology.org/P19-1085/ ; KGAT Table 2, PDF page 6 — https://aclanthology.org/2020.acl-main.655.pdf

| System | Label accuracy | Classification macro-F1 | Strict FEVER | Origin |
| --- | --- | --- | --- | --- |
| Claim-only H2 | 58.78% | 0.5775 | 12.23% | Measured locally; P1 |
| Evidence RoBERTa | 52.37% | 0.4175 | 35.07% | Measured locally; P1 |
| Frozen fusion — primary local result | 61.70% | 0.6078 | 45.33% | Measured locally; P1 |
| GEAR development reference | 74.84% | Not reported here | 70.69% | Published in KGAT Table 2; not rerun |
| KGAT RoBERTa-large development reference | 78.29% | Not reported here | 76.11% | Published in KGAT Table 2; not rerun |

## Per-class performance on all 19,998 claims

The development set is balanced, but performance is not. Evidence RoBERTa has zero NEI recall. Fusion restores some NEI predictions, yet NEI remains its weakest class. Its 61.70% accuracy therefore does not mean equally strong verification across labels.

Claim-only supplies no evidence; its strict FEVER score is earned only by correct NEI labels. An accurate label alone does not demonstrate a complete evidence-backed verification.

| System | Class | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Claim-only | SUPPORTS | 50.37% | 82.16% | 0.6245 |
| Claim-only | REFUTES | 74.12% | 57.50% | 0.6476 |
| Claim-only | NOT ENOUGH INFO | 61.85% | 36.68% | 0.4605 |
| Evidence | SUPPORTS | 50.31% | 89.96% | 0.6453 |
| Evidence | REFUTES | 55.42% | 67.15% | 0.6072 |
| Evidence | NOT ENOUGH INFO | 0.00% | 0.00% | 0.0000 |
| Fusion | SUPPORTS | 54.03% | 86.42% | 0.6649 |
| Fusion | REFUTES | 78.60% | 56.81% | 0.6595 |
| Fusion | NOT ENOUGH INFO | 61.77% | 41.85% | 0.4990 |

## Fusion confusion matrix: actual rows, predicted columns

All rows contain 6,666 claims. For example, 1,773 actual REFUTES claims were predicted SUPPORTS. The model is strongest on SUPPORTS and frequently misses NEI.

| Actual label | Predicted SUPPORTS | Predicted REFUTES | Predicted NEI |
| --- | --- | --- | --- |
| SUPPORTS | 5761 | 284 | 621 |
| REFUTES | 1773 | 3787 | 1106 |
| NOT ENOUGH INFO | 3129 | 747 | 2790 |

## Evidence ablations: the same 100 development claims

Own retrieved evidence beats deliberately mismatched evidence, but more sentences do not consistently help. Top-10 ties top-5 accuracy and has slightly lower macro-F1. Empty evidence has lower accuracy than top-5 but higher macro-F1, so the result depends on the metric.

The 84% gold-input result is not an attainable end-to-end result: NEI receives empty gold evidence, creating a shortcut opportunity. These runs use a development slice and cannot be substituted for the main benchmark.

| Evidence input | Accuracy | Macro-F1 |
| --- | --- | --- |
| Top-1 | 51.00% | 0.3876 |
| Top-5 | 55.00% | 0.4163 |
| Top-10 | 55.00% | 0.4152 |
| Empty evidence | 45.00% | 0.4513 |
| Other-claim evidence | 37.00% | 0.2897 |
| Gold-evidence diagnostic | 84.00% | 0.8315 |

## Token-budget control: the same 19,998 claims

The claim-only branch stayed fixed. Only the evidence branch changed from 256 to 512 tokens; model weights, retrieved references and fusion parameters stayed unchanged. At 256, 1,738 inputs were truncated; only four exceeded 512.

Seeing more of the same retrieved text produced almost no overall label gain. Telemundo still predicted SUPPORTS after all 347 input tokens fit. This does not show that context limits never matter; it shows little benefit from this particular setting change.

| System | 256-token accuracy | 512-token accuracy | Errors fixed | New errors | Net change |
| --- | --- | --- | --- | --- | --- |
| Evidence RoBERTa | 52.37% | 52.38% | 7 | 6 | +0.005 points |
| Fusion | 61.70% | 61.71% | 14 | 12 | +0.010 points |

## Local LLM pilot: the same 300 claims

Llama used the same five evidence texts and a fixed evidence-constrained prompt. It scored close to evidence-only RoBERTa, but below the combined system in label accuracy. Comparing a single LLM with a two-model fusion is a system comparison, not an isolated architecture or size experiment.

Llama predicted NEI 163 times, versus 72 for fusion. It correctly identified 66/100 NEI cases versus 42/100, but only 61/100 SUPPORTS and 28/100 REFUTES versus 84/100 and 55/100. The net result was 26 fewer correct labels.

Strict FEVER moved differently: Llama 46.33% versus fusion 44.00%. Its paired strict-score interval was −3.67 to +8.33 points, so the pilot does not establish a reliable strict-score gain.

| System | Accuracy | Macro-F1 | Strict FEVER |
| --- | --- | --- | --- |
| Evidence RoBERTa | 52.33% | 0.4178 | 35.33% |
| Combined RoBERTa | 60.33% | 0.5937 | 44.00% |
| Llama 3.1 8B | 51.67% | 0.5069 | 46.33% |

## Configuration selection: 60 calibration claims only

All four unique LLM prompt/retrieval conditions used the same 60 claims. A fifth configuration reused the fourth condition’s raw LLM outputs with a fixed fallback: if Llama answered NEI, accept claim-only SUPPORTS/REFUTES only when its softmax was at least 0.80. That softmax is not verified factual confidence.

The highest calibration macro-F1 selected the hybrid. Selection and its hash were written before assessment. The settings were not changed in response to assessment labels.

| Configuration | Calibration accuracy | Calibration macro-F1 | Calibration strict FEVER |
| --- | --- | --- | --- |
| Original setup | 46.67% | 0.4627 | 40.00% |
| Comparison prompt only | 53.33% | 0.5258 | 43.33% |
| Entity/BM25 retrieval only | 55.00% | 0.5528 | 48.33% |
| New prompt + retrieval | 60.00% | 0.5902 | 51.67% |
| Both + fixed NEI fallback — selected | 63.33% | 0.6182 | 53.33% |

## Configuration assessment: 120 disjoint claims

This sample has 40 claims per class and excludes the earlier 300-claim LLM pilot and calibration texts. The selected hybrid gains nine correct labels: 25 errors fixed and 16 introduced. The accuracy-difference interval is −2.50 to +18.33 percentage points; macro-F1 difference interval −0.0107 to +0.1809.

Strict FEVER improves by 4.17 points, with interval −5.83 to +14.17. These intervals include no gain. The result is promising but inconclusive, and it does not beat the saved combined RoBERTa reference on these same claims.

The raw new-prompt/retrieval LLM component scored 57.50%. The fixed fallback changed four labels, fixing three and introducing one, for a final 59.17%. Raw LLM and final hybrid outputs remain separately recorded.

| Configuration | Assessment accuracy | Assessment macro-F1 | Assessment strict FEVER |
| --- | --- | --- | --- |
| Original setup | 51.67% | 0.4973 | 43.33% |
| Both + fixed NEI fallback — selected | 59.17% | 0.5821 | 47.50% |
| Existing combined RoBERTa reference | 60.83% | 0.5917 | Not recomputed in this reference row |

## Assessment trade-offs: TP, FP, FN and TN

These are one-versus-rest counts for each label; a TN can still confuse the other two labels. REFUTES recall rose from 22.5% to 75%, while NEI recall fell from 65% to 37.5%. The selected system improved average performance by trading some correct abstentions for more REFUTES decisions, including more false REFUTES decisions.

| System | Positive label | TP | FP | FN | TN | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| Original LLM | SUPPORTS | 27 | 15 | 13 | 65 | 67.50% |
| Original LLM | REFUTES | 9 | 5 | 31 | 75 | 22.50% |
| Original LLM | NOT ENOUGH INFO | 26 | 38 | 14 | 42 | 65.00% |
| Selected hybrid | SUPPORTS | 26 | 16 | 14 | 64 | 65.00% |
| Selected hybrid | REFUTES | 30 | 17 | 10 | 63 | 75.00% |
| Selected hybrid | NOT ENOUGH INFO | 15 | 16 | 25 | 64 | 37.50% |

## Evidence-coverage subsets and a separate oracle check

In the 300-claim pilot, Llama correctly labeled 73/121 verifiable claims with a complete annotated group and 16/79 without one. This is an association: coverage, difficulty and other factors differ. Missing an annotation does not prove that all alternative text is insufficient.

A random 20-case diagnostic from the incomplete-annotation subgroup changed evidence to one complete gold group while keeping claim, model, prompt and decoding fixed. Accuracy changed from 7/20 to 13/20, with six errors fixed and none introduced. This is a gold-evidence intervention on a selected subgroup, not a new production score. It changes relevance and evidence amount together.

| Original evidence status | Claims | LLM correct | Fusion correct | LLM NEI answers |
| --- | --- | --- | --- | --- |
| Complete annotated group | 121 | 73 | 90 | 36 |
| No complete annotated group | 79 | 16 | 49 | 61 |
| NEI; no required group | 100 | 66 | 42 | 66 |

## Symbolic route: a negative efficiency finding

In the matched 100-claim online run, symbolic v2 resolved no claims. Both conditions called retrieval and RoBERTa 100 times and produced identical labels. The initial symbolic route added about 40.22 ms per claim on average.

Later cached timing passes fluctuated without avoiding calls. Those fluctuations do not establish symbolic savings. The result concerns these restricted rules, inputs, device and network/cache conditions, not every neuro-symbolic architecture.

| Condition | Accuracy | Macro-F1 | Claims resolved | Model calls | Mean ms/claim |
| --- | --- | --- | --- | --- | --- |
| retrieved_top5 | 66.00% | 0.4723 | 0 | 100 | 66.35 |
| hybrid_v2 | 66.00% | 0.4723 | 0 | 100 | 106.57 |

## Qualitative examples: fixes, failures and explanation limits

These examples were selected to illustrate patterns and are not a random prevalence estimate. The saved request/response excerpts are bundled in evidence_cases.json. Generated explanations are not hidden reasoning traces. A correct final label does not certify its explanation.

| Case | Claim | Observed result | Specific evidence or output | What it shows | Trace source |
| --- | --- | --- | --- | --- | --- |
| Q1 / #111897 | Telemundo is a English-language television network. | Gold REFUTES; evidence RoBERTa SUPPORTS; local LLM REFUTES. | Telemundo is described as Spanish-language. The annotated sentence reached the 256-token RoBERTa input completely; 512 tokens still did not fix that prediction. | Relevant evidence can be retrieved and visible while the classifier still gets the label wrong. | E3, E9; /ollama-traces#111897 |
| Q2 / #146231 | Juventus F.C. rejected their traditional black-and-white-striped home uniform in 1903. | Gold REFUTES; LLM NOT ENOUGH INFO. | The evidence does not mention Juventus F.C. or their uniforms, so it does not support or contradict the claim. | The original candidate search missed Juventus; the supplied texts about black/white topics do not establish or contradict the uniform claim. | E4, E6; /ollama-traces#146231 |
| Q3 / #149173 | Psych is a drama and comedy that aired in 2004. | Gold REFUTES; LLM SUPPORTS. | The evidence confirms that Psych is a comedy-drama series and provides its airing year as 2006, which is close to the claimed 2004. | The 2006 debut sentence is present; accepting it as close enough to a 2004 claim is an explicit date-comparison error. | E4, E6; /ollama-traces#149173 |
| Q4 / #75152 | Tool has not performed worldwide tours. | Gold REFUTES; original LLM NOT ENOUGH INFO → selected pipeline REFUTES. | The evidence states that Tool has toured worldwide extensively, contradicting the claim that they have not performed worldwide tours. | A useful fix: selected evidence explicitly says the band toured worldwide extensively, contradicting the negated claim. | E10; /llm-configurations |
| Q5 / #52731 | Duff McKagan was born in Vietnam. | Gold NOT ENOUGH INFO; original LLM NOT ENOUGH INFO → selected pipeline REFUTES. | The evidence states Duff McKagan was born in 1964, which is not Vietnam. | A regression: a birth date does not contradict a birthplace. The final REFUTES label is wrong against FEVER’s NEI reference. | E10; /llm-configurations |
| Q6 / #121517 | Halsey's debut EP is titled The Simpsons. | Gold REFUTES; original LLM NOT ENOUGH INFO → selected pipeline REFUTES. | The evidence mentions a racehorse named Titled Hero, which contradicts the claim that Halsey's debut EP is titled The Simpsons. | Correct label, poor justification: the racehorse text does not establish a contradiction about Halsey’s EP. Accuracy credit does not validate the explanation. | E10; /llm-configurations |

## Limitations and boundaries

| Limitation | Evidence or condition | Consequence for the claim |
| --- | --- | --- |
| Retrospective revision | Questions/hypotheses were revised after results. | Treat this as transparent exploratory analysis, not preregistered confirmation. |
| Development reuse and sample sizes | P1 is official development; newer LLM assessment is a disjoint subset, not a blind official test. Other samples range from 20 to 300. | Keep denominators separate. Do not present calibration or oracle scores as final benchmark performance. |
| Single trained checkpoints / bounded repeats | No independent additional Transformer training seeds; current 8B configurations use one generation per claim. | Bootstrap intervals measure claim-sampling uncertainty, not between-training-run or repeated-generation variance. |
| Model identity and resources | Llama 3.1 8B Q4_K_M was held fixed; published graph models use different training/retrieval resources. | No model-only or quantization-only cause is established. Pretraining exposure to FEVER/Wikipedia is unknown. |
| Retrieval and annotation coverage | Some annotated groups are missed or unavailable locally; complete annotation coverage does not guarantee correct inference. | Do not equate overlap with semantic sufficiency, or oracle access with a deployable retriever. |
| Input/training mismatch | The evidence checkpoint was trained with gold-style evidence; empty NEI inputs create a shortcut opportunity. It predicts no NEI on the full retrieved-evidence benchmark. | A controlled retraining experiment is needed to isolate the cause; no retraining fix is claimed. |
| Bundled changes and class trade-offs | Entity linking and BM25 change together; prompt rules and examples change together; the selected hybrid has lower NEI recall. | Do not attribute the entire gain to one clause, retrieval subcomponent or universally better reasoning. |
| Explanation validity | Correct labels can accompany irrelevant or illogical explanations, as in the Halsey case. | No faithfulness or human-rated explanation-quality claim is made. |
| Symbolic timing limits | Zero coverage, only a few timing passes, and cache/network variation. | No avoided-call, cost-saving or general speedup claim is supported. |
| Original generated-error question | Independent review/adjudication of generated claims remains incomplete. | Original RQ1 remains unanswered and is explicitly outside the adopted central scope. |

## Conclusion and current configuration choice

For the measured main benchmark, frozen RoBERTa fusion remains the primary local result: 61.70% accuracy, 0.6078 macro-F1 and 45.33% strict FEVER on all 19,998 development claims. It improves on its individual components, while remaining below the published development references.

Evidence relevance matters in the paired diagnostics; simply increasing context length gave almost no gain. The selected LLM hybrid improved the point estimates on a disjoint 120-claim assessment, but the intervals and NEI trade-off prevent a strong general improvement claim. It remains an experimental candidate, not a new default.

The implemented symbolic stage did not establish an efficiency benefit. The contribution is an evidence-linked empirical account of what helped, what did not, and where verification still fails. Original generated-claim representativeness is future work, not a completed finding.

## Pending work, separated from completed analysis

| Item | Status | What would resolve it |
| --- | --- | --- |
| Tabular Result Analysis and adopted narrower scope | Completed in this artifact | No new model training was needed. |
| General LLM model/quantization comparison | Not performed in the latest configuration search | Hold pipeline, claims and scoring fixed; compare model variants on separate data. |
| Reliable improvement for the selected LLM hybrid | Not established by this small assessment | Use a larger new evaluation sample and planned repeats; avoid further tuning on these 120 claims. |
| Generated-claim truth/error representativeness | Unanswered; moved to future work | Independent blinded human review and adjudication. |
| Training-distribution fix and multi-seed training variance | Not run | Design separately; expensive retraining requires prior approval. |

## Evidence index

Each source ID below maps to a repository artifact and its SHA-256 in evidence_manifest.json. The submission bundle contains compact source snapshots, all displayed CSV tables, selected case excerpts and the adopted-scope document. Raw full-run outputs and checkpoints remain preserved in the repository.

| ID | Artifact | What it supports |
| --- | --- | --- |
| E1 | reports/part2/official_fever_dev_20261023/summary.json | Full 19,998-claim benchmark and paired intervals |
| E2 | reports/part2/evidence_depth_20261023/summary.json | 100-claim evidence-input ablations |
| E3 | reports/part2/token_budget_20261028/summary.json | Frozen evidence-input 256 versus 512 comparison |
| E4 | reports/part2/ollama_comparison_20261028/summary.json | Matched 300-claim local LLM pilot |
| E5 | reports/part2/llm_configurations_20261028/assessment_summary.json | 120-claim assessment after calibration-only selection |
| E5a | reports/part2/llm_configurations_20261028/calibration_summary.json | Five configurations on 60 calibration claims |
| E5b | reports/part2/llm_configurations_20261028/selected_config.json | Selection frozen before assessment |
| E5c | reports/part2/llm_configurations_20261028/assessment_start.json | Pre-assessment selection hash and timestamp |
| E5d | reports/part2/llm_configurations_20261028/assessment_uncertainty.json | Paired macro-F1 and strict-score intervals |
| E6 | reports/part2/ollama_error_analysis_20261028/analysis.json | 300-claim error and annotation-coverage breakdown |
| E7 | reports/part2/ollama_error_analysis_20261028/oracle_summary.json | Separate 20-claim gold-evidence intervention |
| E8 | reports/part2/step2_final_20261023/heldout100/summary.json | Matched symbolic routing, calls and online timing |
| E9 | reports/part2/ollama_comparison_20261028/telemundo_response.json | Separate saved Telemundo LLM diagnostic |
| E10 | reports/part2/llm_configurations_20261028/final_decisions.jsonl | Raw LLM versus final hybrid decisions |
| E11 | reports/part2/llm_configurations_20261028/protocol.json | Sampling, exclusion, prompts, model identity and fixed rules |
