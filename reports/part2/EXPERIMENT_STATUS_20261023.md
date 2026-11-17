# Controlled FEVER experiments — October 23, 2026

Continuation update: the [consolidated Part 2 submission report](submission_20261023_final/PART2_RESULTS.md)
adds measured top-10 retrieval, a disjoint validation extension, a 360-request local
LLM judge pilot, generated claims for human review, and four corpus-size conditions.
The report below preserves the findings and pending-work status at the earlier checkpoint.

Work continues on `experiment/part2-paired-ablation`. No training, merge, index modification, or paid judge requests were performed. All experiment execution used `.venv/bin/python`. Existing reports, checkpoints, data and untracked work were preserved. New output paths refuse to overwrite existing runs.

## Measured results

All classification conditions below use the same 100 validation claim IDs. These are development results: rules were informed by these claims, hyperparameters were selected on validation data, and the sample is small. They do not establish final-test or population improvements.

| Condition | Accuracy | Macro-F1 |
|---|---:|---:|
| Claim-only H2, saved paired run | 0.6200 | 0.5761 |
| Retrieved evidence, saved paired run | 0.5500 | 0.4163 |
| Empty evidence, saved paired run | 0.4500 | 0.4513 |
| Wrong-claim evidence, saved paired run | 0.3700 | 0.2897 |
| Gold-evidence oracle, saved paired run | 0.8400 | 0.8315 |
| Hybrid symbolic v1, new snapshot control | 0.5500 | 0.4163 |
| Hybrid symbolic v2 | 0.5500 | 0.4163 |

V1 and V2 both resolved **0/100** claims. Resolved accuracy is undefined, rather than 0%. Neither saved any retrieval or Transformer calls. There is **no measured accuracy or efficiency improvement** from v2 on this sample.

The retrieved-minus-claim-only accuracy difference is -7 percentage points. A paired row bootstrap with seed 42 and 10,000 resamples gives a 95% interval of [-17, +4] points; exact McNemar p = 0.2649 (unadjusted). Retrieved evidence wins on 11 discordant claims and loses on 18. This comparison also changes checkpoints and training conditions, so it does not isolate evidence use alone. All multiple-comparison statistics in the JSON are exploratory. The bootstrap assumes independent rows.

The oracle is a diagnostic, not an end-to-end baseline: empty gold evidence for NEI exposes a label-related cue. Its 84% accuracy is not an achievable retrieval-system result.

## Symbolic v2 implementation and audit

`src/experiment_symbolic_v2.py` leaves v1 intact and compares both versions using the same saved retrieval predictions. V2 adds:

- Exact, full-clause occupation rules using direct P106 labels/aliases and a human P31 constraint.
- Numeric birth/death years and decades using P569/P570, calendar and precision checks.
- Exact organization-membership claims using P463.
- Abstention for missing, ambiguous, truncated-search, qualified, deprecated or unmatched new-rule facts. No open-world REFUTES inference.

The inherited birthplace/deathplace behavior is retained as a control fallback, with temporal-expression guards. New rules exclude qualified statements; inherited v1 behavior retains its original handling. V2 does not implement occupation subclass reasoning or resolve spelled-out decades. Human typing can filter candidates for occupation/date claims; it does not establish that every remaining name refers to the FEVER entity. Successful rule matches would still require manual factual audit.

V2 recognizes eight candidate claims, all of which abstain:

| Abstention | Count | Examples |
|---|---:|---|
| Unsupported claim pattern | 92 | Rules do not cover most FEVER clauses |
| Ambiguous/unresolved human | 4 | Grace Jones, Sophie Turner, Marilyn Monroe, Prakash Jha |
| Direct occupation not matched | 2 | Wyatt Earp as actor; David Koepp as artist |
| Ambiguous/unresolved subject | 1 | Haiti membership claim |
| Unsupported temporal expression | 1 | Jackie Robinson died “in the seventies” |

These reasons describe rule behavior; they are not factual judgments on the claims. For example, failure to find a direct occupation match does not establish REFUTES.

The initial live attempt stopped at HTTP 429 and is marked unscored in `symbolic_v2/run_20261023/failure.json`. Request pacing and bounded 429/503 backoff were added. The completed attempt made 40 HTTP requests, including seven transient failures handled by retry, and reused previously saved successful responses. It stopped treating network failures as ordinary abstentions.

Successful API responses, query parameters and fetch timestamps are preserved in `symbolic_v2/wikidata_snapshot/`. The completed run records input/source/snapshot SHA-256 hashes. An offline replay reproduced all 200 condition rows exactly except timings, with zero HTTP calls. Timings in the symbolic run mix live and cached queries and cannot be compared as latency savings. Present-day Wikidata may differ from historical FEVER annotations.

Artifacts:

- `symbolic_v2/run_20261023_retry/predictions.csv` and `manifest.json`: completed experiment.
- `symbolic_v2/replay_20261023/`: offline reproduction.
- `symbolic_v2/analysis_20261023.json`: classification metrics, paired statistics, retrieval recall and error decomposition.

## Retrieval and classification diagnosis

Recall denominators are the same 71 verifiable claims with annotated evidence. “Any” means at least one annotated sentence; “complete” means at least one complete annotated evidence group.

| Retrieval strategy | Any R@1 | Complete R@1 | Any R@5 | Complete R@5 |
|---|---:|---:|---:|---:|
| AND | 5/71 | 5/71 | 17/71 | 17/71 |
| OR | 10/71 | 10/71 | 21/71 | 19/71 |
| Hybrid lexical | 6/71 | 6/71 | 21/71 | 19/71 |
| Title | 13/71 | 13/71 | 30/71 | 25/71 |
| Title fast | 35/71 | 32/71 | 46/71 | 44/71 |

Title-fast complete R@5 is 61.97%. These are measured differences on existing development artifacts, not held-out gains. Top-5 artifacts cannot measure R@10. Title-fast uses title FTS plus local term-overlap ranking, so describing this condition as pure full-corpus sentence BM25 would be inaccurate.

For title-fast retrieval plus RoBERTa:

- 38 correct and 6 incorrect verifiable claims have a complete retrieved evidence group.
- 17 correct and 10 incorrect verifiable claims lack a complete retrieved group.
- All 29 NEI claims are misclassified; the model predicts no NEI labels in this condition.

The gold-evidence training / retrieved-evidence evaluation mismatch and NEI behavior deserve controlled follow-up. These counts do not prove the mechanism behind the errors. Retraining to address them requires approval.

## RQ4 measured timing control

`src/benchmark_hierarchy.py` performs actual title-aware retrieval and local RoBERTa inference against read-only existing indexes. It alternates condition order over three repeats, uses batch size 1 and maximum length 256, and warms retrieval and model paths before timing. This environment exposed CPU, not MPS. Model loading is excluded.

| Condition | Mean across 300 timed claims | Retrieval calls per repeat | Model calls per repeat |
|---|---:|---:|---:|
| Baseline | 104.43 ms/claim | 100 | 100 |
| Cached symbolic route | 106.10 ms/claim | 100 | 100 |

All six trials reproduce accuracy 0.55 and macro-F1 0.4163. With zero symbolic resolutions, both execute the same retrieval/model work; the observed timing difference is not evidence of a symbolic benefit. Raw per-claim timings, predictions, p50/p95 latency and throughput are in `latency_20261023/`.

This is a warm local **cached-decision routing** benchmark. It excludes live KB queries and symbolic rule computation, and measures only one existing corpus size. It cannot establish live hybrid latency, monetary savings, or scaling behavior. The runner accepts repeated `--db` arguments for independently built nested indexes; it never constructs or modifies them. A later growth experiment must record page/sentence counts, keep claim IDs, rules, model, retrieval settings and hardware fixed, and report recall/accuracy as well as latency. Building new indexes was not performed here.

## RQ1 and RQ2 controlled work prepared

`src/controlled_judges.py` prepared `controlled_inputs_20261023/`:

- A 100-claim RQ1 annotation CSV without prefilled human labels. Fields include atomicity, claim type, temporal/context requirements, error type, evidence source, annotator and label.
- 1,200 RQ2 tasks: 100 identical IDs × 2 evidence conditions × 2 prompt wordings × 3 repeats. Neither gold labels nor gold evidence are sent to the judge. The empty-evidence condition is an evidence-constrained abstention control, not a closed-book knowledge test.
- A protocol manifest requiring an exact judge model/version, generation settings, randomized request order, raw response/usage/latency/error logs and isolated requests. No judge requests were executed.

The scorer checks paired task/input identity and complete responses for each model. It reports accuracy and macro-F1 by repeat, repeat consistency, and prompt agreement/kappa. Response CSV fields are `task_id,model,predicted_label_id`, with labels 0=SUPPORTS, 1=REFUTES, 2=NOT ENOUGH INFO. Retain raw responses and parsing records separately; this scorer trusts the supplied parsed labels. Missing responses and invalid labels fail instead of being silently dropped.

RQ1 remains unanswered until a separately sampled, provenance-preserving LLM-generated claim corpus is collected and reviewed by two blinded annotators with adjudication. FEVER alone cannot measure representativeness of LLM errors. Synthetic edits require independent verification; changing a date or entity does not automatically make a claim false. RQ2 remains unanswered until real judge responses are collected and scored. There are no fabricated measurements for either question.

## Reproduction

Use new output names: the following analysis/runner commands refuse to overwrite existing paths. No dependency changes are needed.

```sh
.venv/bin/python -m unittest discover -s tests -v

.venv/bin/python src/experiment_symbolic_v2.py --offline \
  --snapshot reports/part2/symbolic_v2/wikidata_snapshot \
  --output reports/part2/symbolic_v2/replay_NEW

.venv/bin/python src/analyze_controlled.py \
  --symbolic reports/part2/symbolic_v2/run_20261023_retry/predictions.csv \
  --output reports/part2/symbolic_v2/analysis_NEW.json

.venv/bin/python src/benchmark_hierarchy.py \
  --db data/processed/wiki_fts.db \
  --symbolic reports/part2/symbolic_v2/run_20261023_retry/predictions.csv \
  --output reports/part2/latency_NEW

.venv/bin/python src/controlled_judges.py prepare \
  --output reports/part2/controlled_inputs_NEW

# Only after independently collecting real judge responses:
.venv/bin/python src/controlled_judges.py score \
  --tasks reports/part2/controlled_inputs_20261023/judge_tasks.jsonl \
  --responses PATH_TO_REAL_RESPONSES.csv \
  --output reports/part2/judge_scores_NEW.json
```

Ten tests cover conservative rule behavior, ambiguous and truncated entity searches, date precision/calendar constraints, exclusion of qualified/deprecated facts, offline-cache failures, duplicate prediction rejection, paired difference direction, and factorial judge scoring/missing-response rejection. Offline replay equality and the six measured timing trials provide additional integration checks.

Next experiments should freeze rules before evaluating a disjoint validation slice, measure R@10 with newly retrieved top-10 candidates, collect human-reviewed LLM claims/judge outputs, and compare nested corpus indexes. Keep the held-out test set untouched until choices are frozen. Seek approval before expensive retraining or destructive data/index changes.
