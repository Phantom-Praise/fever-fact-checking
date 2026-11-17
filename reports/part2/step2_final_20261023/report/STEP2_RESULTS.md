# Step 2 experiment update

This update adds a frozen held-out comparison, actual online symbolic-stage timing, and a separate automated RQ1 pilot. Earlier reports and measurements are preserved. No models were retrained or rules tuned, and the experiment branch was not merged into main.

## Status

- RQ2 already has 360 measured local-judge requests, prompt/evidence conditions, and inference repeats.
- RQ3 now also has a prespecified 100-claim held-out comparison.
- RQ4 now includes actual online Wikidata overhead, in addition to the earlier corpus-growth measurements.
- RQ1 now has a two-model evidence-conditioned automated coding experiment. Independent human review remains pending, as confirmed by the user. The generated-claim labels below are not established human ground truth.

## Held-out comparison

Seed 874 shuffles the existing test partition; the first 100 unique normalized claims are retained. Selection does not inspect labels. Selected IDs and normalized claim text are checked against training and validation. The protocol, sample IDs, and code snapshot were written before inference. The selected checkpoints, rules, top-5 retrieval, length 256, and batch size 1 are fixed. This is a bounded sample of the held-out split, not the full test partition.

| Condition | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | --- | --- | --- |
| Original unweighted reference | 0.5200 | 0.1733 | 0.3333 | 0.2281 |
| Frozen claim-only H2 | 0.6300 | 0.5890 | 0.5746 | 0.5802 |
| Frozen retrieval plus RoBERTa | 0.6600 | 0.4206 | 0.5385 | 0.4723 |
| Live symbolic v2 hybrid | 0.6600 | 0.4206 | 0.5385 | 0.4723 |

Device for this new comparison: mps. Do not directly compare these timings with the earlier CPU corpus-growth run.

Retrieval-minus-H2 accuracy is +0.03; paired bootstrap 95% CI [-0.09, 0.14], unadjusted exact McNemar p=0.7283. The small accuracy difference is not a demonstrated general improvement. H2 has higher macro-F1; both retrieval and hybrid misclassify all 22 NEI claims. Do not change model selection based on these test outputs.

The hybrid and retrieval predictions are identical. Symbolic v2 resolves zero claims and therefore avoids zero retrieval or Transformer calls. A negative result still answers whether the implemented symbolic stage helped this measured sample: it did not.

## Actual pipeline latency

| Pass | Condition | Mean ms per claim | Symbolic seconds | HTTP attempts | Retrieval calls | Model calls |
| --- | --- | --- | --- | --- | --- | --- |
| initial_online | retrieved_top5 | 66.35 | 0.0000 | 0 | 100 | 100 |
| initial_online | hybrid_v2 | 106.57 | 3.9507 | 7 | 100 | 100 |
| warm_snapshot_1 | retrieved_top5 | 58.89 | 0.0000 | 0 | 100 | 100 |
| warm_snapshot_1 | hybrid_v2 | 60.26 | 0.0065 | 0 | 100 | 100 |
| warm_snapshot_2 | retrieved_top5 | 56.97 | 0.0000 | 0 | 100 | 100 |
| warm_snapshot_2 | hybrid_v2 | 50.78 | 0.0064 | 0 | 100 | 100 |

The initial online pass starts with an empty study snapshot and includes actual request latency, pacing/backoff if any, rule execution, retrieval, tokenization, and inference. Model loading and warmup are excluded. Within-pass query caching is allowed. The two later passes reuse the saved Wikidata responses. Each claim alternates which pipeline is run first across passes. Raw events record each phase, the decision, call counts, and service failures.

Seven HTTP requests were made with zero HTTP failures. No service-outage fallbacks occurred. The initial hybrid pass averaged 106.57 ms per claim versus 66.35 ms for retrieval alone. Warm-cache differences fluctuate without any avoided calls, so they do not establish a symbolic computational saving. Only three passes, shared OS caches, and changing network latency limit conclusions. No monetary cost saving is claimed.

## RQ1 automated experiment

Two different already-installed local models see the same 60 source-blinded claims and the same five candidate evidence sentences: 30 FEVER claims and 30 generated claims. Neither source identity nor gold FEVER labels enters the prompt. Settings are fixed: seed 874, temperature 0, context 4096, output budget 256 tokens. The coding schema includes label, atomicity, claim type, primary error type, cited sentence indices, and a short reason. Both models are run regardless of their scores; invalid outputs are retained and counted.

| Model | Source | N | Valid | Invalid | Supports | Refutes | NEI | Unrateable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3.1:8b | FEVER | 30 | 30 | 0 | 23 | 0 | 7 | 0 |
| llama3.1:8b | generated | 30 | 30 | 0 | 24 | 0 | 5 | 1 |
| gemma3:1B | FEVER | 30 | 30 | 0 | 23 | 0 | 0 | 7 |
| gemma3:1B | generated | 30 | 30 | 0 | 30 | 0 | 0 | 0 |

The method is calibrated against the original human FEVER labels on the 30 FEVER claims:

| Model | FEVER calibration N | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| llama3.1:8b | 30 | 0.3000 | 0.2008 |
| gemma3:1B | 30 | 0.3000 | 0.1818 |

Machine agreement is a reliability diagnostic, not proof that either model is correct:

| Field | Jointly valid N | Agreement fraction | Cohen kappa |
| --- | --- | --- | --- |
| label | 60 | 0.6833 | -0.0345 |
| atomic | 60 | 0.7333 | 0.0400 |
| claim_type | 60 | 0.4833 | 0.3430 |
| error_type | 60 | 1.0000 | undefined |

A posthoc protocol-consistency check found the following issues; structural validity alone is insufficient:

| Model | Structurally valid | Error category inconsistencies | Distinct error categories returned |
| --- | --- | --- | --- |
| llama3.1:8b | 60 | 13 | NONE |
| gemma3:1B | 60 | 7 | NONE |

Both models achieved only 0.30 accuracy against FEVER references. Their constant error-category outputs and weak calibration make this automated coding method unsuitable for establishing generated-claim factual-error rates. RQ1 is inconclusive under this method; independent review remains necessary. No errors are repaired by inventing labels.

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
