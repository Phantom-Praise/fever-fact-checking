# Paper review and next experiment

## Writing reviewed

- `Praise_DeepLearning2.pdf`: original two-page FEVER proposal, including RQ1–RQ4, H1–H3 and the validation plan.
- `PraiseBen_DataCollection.pdf`: data choice, label balance, duplicate/noise concerns and limits of FEVER as a proxy for modern LLM errors.
- `PraiseBen_LitReview.xlsx`, Literature Comparison, rows 3–10: FEVER, FactKG, GEAR, KGAT, hybrid verification, HoVer, FActScore and GraphCheck.
- Downloads `Model_Execution_Part_1.pdf`: initial execution results and the gold-evidence / empty-NEI shortcut limitations.
- `reports/result_analysis_20261028/ADOPTED_SCOPE.md` and `RESULT_ANALYSIS.md`: the later adopted research scope and measured comparisons.
- October 29 saved RAG, essential-parts, exhaustive-reader and revised-prompt results and qualitative traces.

The repository's writing consists of the proposal, literature matrix and staged reports. A newer full FEVER manuscript was not identified in this review. No manuscript or workbook was edited, and no experiments or retraining were started.

## What the original proposal actually asks

The original study combines accuracy, evidence retrieval, judge reliability and efficiency. It explicitly plans to analyze cases where retrieval succeeds but classification fails, alongside cases where relevant evidence is missing. It also explicitly allows the symbolic/efficiency hypotheses to be unsupported.

The House of the Spirits trace directly fits that planned analysis: the model received E8 saying the film won awards, yet returned NEI. More article content alone cannot be assumed to resolve that observed decision error.

The original generated-claim representativeness question remains unfinished without the planned independent review. The adopted narrower scope already records that limitation. Do not present the newer prompt pilots as completing all original questions.

## What needs alignment in the writing

1. Use the adopted local FEVER question as the central scope. Claims about validated LLM-generated errors, a successful symbolic shortcut, and general scalability exceed the measured evidence.
2. Respect the user's selected evidence-RoBERTa baseline. Some older report text still calls fusion the primary system. Retain fusion measurements as a named comparison instead of deleting them or presenting an unselected system as the main baseline.
3. Keep the ten-claim prompt experiments in an exploratory/development subsection. The original parts prompt scored 5/10 using validated overall model labels and 7/10 using code-derived labels. The full-reader final condition scored 6/10 with two INVALID outputs. The three-addition prompt scored 4/10 with three INVALID outputs. These are inspected claims, not independent improvement estimates.
4. Separate label correctness from explanation faithfulness. A structurally consistent aggregation can still aggregate incorrect part verdicts, rewritten assertions or irrelevant citations.
5. Separate historical FEVER evidence from newer live Wikipedia renders. Content/version differences complicate attribution of failures. Flag the neutrino reference-label inconsistency for review; retain the original benchmark label and recorded score.
6. Preserve the Part 1 distinction between oracle evidence and end-to-end retrieval. Empty evidence for NEI can expose labels through formatting; a high oracle score cannot be presented as ordinary pipeline accuracy.

## Literature connections checked

- FEVER's original paper evaluates a retrieval/verification pipeline and oracle conditions. This supports diagnosing retrieval separately from verification: https://aclanthology.org/N18-1074.pdf .
- FActScore decomposes long-form generations into atomic facts and measures supported-fact precision. It motivates fine-grained checking, but its metric is not three-class FEVER accuracy: https://aclanthology.org/2023.emnlp-main.741/ . Our part-verdict prompt is inspired by this general idea, not a reproduction of FActScore.
- The hybrid paper linked in workbook K7 is Kolli et al., WiNLP 2025, rather than the Zhang/EMNLP attribution in C7/E7. Its reported 0.93 F1 concerns the SUPPORTS/REFUTES subset, so it must not be compared directly with our three-class accuracy: https://aclanthology.org/2025.winlp-main.19/ .

The workbook was preserved. Published metrics from different tasks, datasets, splits or label sets should remain literature context rather than a single ranked performance table.

## Recommended next experiment: freeze the prompt and locate the bottleneck

Use a preselected fresh sample for prompt development, excluding training overlaps, normalized duplicates and previously inspected prompt-evaluation cases. Record eligibility and sampling before running. It is fresh for the prompt study, not automatically an untouched test of the whole project.

On the same verifiable claims, compare:

| Condition | Input | Purpose |
|---|---|---|
| Retrieved evidence | Evidence selected by the fixed retrieval pipeline from the FEVER Wikipedia snapshot | Measure the pipeline under realistic retrieval. |
| Annotated evidence diagnostic | A complete FEVER-annotated evidence group from that same snapshot | Check whether the frozen classifier can use the supplied reference evidence. This is an oracle diagnostic, not an end-to-end score. |

Keep the original parts prompt, model digest, decoding, output schema and aggregation unchanged. Keep all conditions' claim IDs paired. Record actual token visibility and complete/incomplete outputs rather than assuming the evidence reached the model.

FEVER supplies no sufficient annotated evidence group for NEI. Evaluate NEI using the normal retrieved-evidence condition and report it separately; do not create an empty-input label cue and pool that into an oracle headline score.

An earlier oracle pilot already exists for the older LLM setup. The proposed work extends that diagnostic to the selected parts-based configuration on fresh cases; it should not be described as the project's first oracle experiment.

Score only after predictions are saved. Report accuracy, macro-F1, per-class recall, invalid counts and stage timing. Categorize errors as missing evidence, evidence present but misinterpreted, claim changed during decomposition, citation/output failure, or potential annotation ambiguity. These categories may overlap; define a counting policy before reporting category rates. Researcher inspection is qualitative analysis, not independent human validation.

Keep a separate final holdout for the selected configuration. Any sample used to change the prompt becomes development data. Continue to preserve negative results. The immediate objective is an interpretable measurement of the bottleneck, not making every familiar example pass.
