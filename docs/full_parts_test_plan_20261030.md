# Planned test evaluation: temporary Wikipedia RAG with essential-part verdicts

**Scope update:** the user subsequently approved a 60-claim matched test study
to fit a 3–4 hour project window. See `parts_test60_20261030.md` for the active
study. The full-partition plan below is retained as a planning record and is
not the evaluation currently being executed.

Status: planning only. No new article fetching, embeddings, model inference,
training or evaluation has started. The old 19K Llama evaluation remains stopped.

## Question and comparison

On the same eligible internal FEVER test claims, how does the proposed temporary
full-article RAG + Llama parts classifier compare with evidence RoBERTa in label
accuracy, macro-F1, per-class performance and end-to-end cost?

The proposed RAG classifier is the main model. Evidence RoBERTa is the baseline.
Run both on the exact same claims. Do not reuse baseline percentages measured on
the official development set. This comparison changes evidence corpus, retrieval,
context, classifier and decision method together; it measures complete systems,
not the isolated benefit of a vector database or claim decomposition.

## Population audit

| Item | Count |
|---|---:|
| Original internal test rows | 14,548 |
| Unique case/whitespace-normalized test claims | 13,555 |
| Eligible unique claims after exclusions | 13,448 |
| Excluded rows, union | 1,100 |
| Additional duplicate rows within test | 993 |
| Rows matching prior saved experiment inputs/outputs | 140 |
| Rows matching processed training claims | 6 |
| Rows matching processed validation claims | 0 |

Reason counts overlap and must not be added to compute the union. Eligibility
retains the first occurrence of a normalized test text and excludes training,
validation and prior-experiment overlaps. Conservatively, prepared prior inputs
count as exposed even when inference did not finish. No labels or files were
changed. Eligible class counts: SUPPORTS 7,406; REFUTES 2,756; NEI 3,286. Preserve
this natural distribution rather than balancing the full population.

The audit examines 16 listed saved artifacts. It cannot prove that no other
manual inspection occurred. Earlier research already evaluated a small sample
from this partition. Describe the new study as evaluation on the remaining
eligible test claims, not an entirely untouched project-wide test. All results
are one final evaluation: do not tune on them afterward and retain the same test
claim as a claimed fresh holdout.

Audit artifacts:

- `output/plan_parts_test_20261030/population_audit.json`: counts, policies and input hashes.
- `output/plan_parts_test_20261030/excluded_ids.json`: every excluded row and reasons.
- `output/plan_parts_test_20261030/eligible_claims_without_labels.jsonl`: candidate evaluation inputs with no labels.
- `output/plan_parts_test_20261030/audit_population.py`: reproducible read-only audit.

This is a candidate population, not a launch authorization or final frozen run
protocol. Recheck source/data fingerprints at preparation and before execution.

## Main-model condition to freeze

Use `docs/working_parts_prompt.json` and its original essential-parts instruction,
schema and deterministic aggregation. Do not substitute the failed prompt-v2 or
exhaustive-reader condition.

1. Existing local title retrieval finds ten article candidates and retains the
   top five, with its existing URL deduplication.
2. Fetch actual full Wikipedia article bodies into the per-claim workspace.
3. Chunk with the installed MiniLM tokenizer (target 200 tokens), embed locally,
   and create the temporary FAISS/SQLite store. Search the entire index; retain
   up to 30 vector candidates and pack whole passages within the original
   7,600-character context budget.
4. Pass the claim and packed article passages to installed `llama3.1:8b` using
   the selected model digest, temperature 0, seed 42, context 4,096, output cap
   256, and the original essential-parts schema.
5. Validate every part verdict and citation. Primary label: any refuted part
   gives REFUTES; all supported gives SUPPORTS; otherwise NEI. Report the model's
   own overall label separately and count inconsistencies.
6. Save the exact request/response, supplied/cited passages, URLs, revisions,
   selection metadata, stage timings, errors and cleanup outcome. Delete full
   articles, all article chunks not retained as supplied evidence, and vectors
   after each claim, including failures.

Llama receives selected passages, not every article chunk. Live Wikipedia is a
different corpus from historical FEVER. A trace records what was actually used;
deleting the article corpus limits exact reproduction of vector retrieval later.
Replaying the saved classification prompt remains possible.

Keep the original 256-token output cap to evaluate the selected configuration.
It can truncate part arrays. Such cases remain INVALID, never silently repaired
after checking labels. Any budget/configuration change needs a separate version
fixed before test execution, with engineering checks on development fixtures.

## Baseline condition

Use the frozen evidence RoBERTa checkpoint
`checkpoints/roberta_evidence/epoch_1`, original five-sentence retrieval and
256-token combined input limit. Classify using its evidence probabilities alone;
do not apply claim-only fusion, NEI bias or Llama fallback. Record input token
visibility and checkpoint/source fingerprints. Disclose its historical
gold-evidence/empty-NEI training limitation. No retraining is planned.

## Engineering work required before launch

| Order | Deliverable | Planning allowance |
|---|---|---|
| 1 | General parts-based RAG adapter; reuse existing fetching/vector components without editing frozen experiment sources | 2–4 hours |
| 2 | Exact rendered-prompt token preflight with the pinned tokenizer; reject over-budget inputs; validate returned token counts; preserve raw responses before parsing | 1–3 hours |
| 3 | Resumable paired runner, separate labels/scoring, atomic per-claim results, trace viewer and storage/timing accounting | 2–4 hours |
| 4 | Offline tests and crash/resume/cleanup checks; verify identical frozen IDs/settings; engineering smoke checks on old development fixtures only | 1–2 hours |
| 5 | Freeze the run protocol and confirm compute window before launch | Under 1 hour |

These are estimates, not completed implementation work. Existing parts code
replays ten saved contexts; it is not yet a full test runner. The new runner must
not call the old direct-label RAG prompt accidentally.

## Execution and failure rules

- Freeze all input IDs and a seeded execution order, both model conditions,
  prompts, schemas, model/embedding hashes, retrieval settings, software versions
  and scoring before generation. Retain test labels outside inference inputs.
- Save both predictions before scoring. Do not show running accuracy or gold
  labels during execution. A progress dashboard may show counts, saved inputs,
  predictions, trace responses, timing and errors.
- Checkpoint after each completed claim/condition. Resume verifies hashes and
  skips completed work; it must not rerun successful cases or overwrite traces.
  A crash during an unfinished claim may refetch its articles; record attempts
  and revisions. Successful paired results are never silently replaced.
- Use one claim workspace at a time and Wikipedia-friendly request pacing with
  bounded transient-error backoff. Log missing pages and partial fetch coverage.
  Do not use snippet fallback. If no usable article context exists, record INVALID.
- Freeze bounded network retry rules before starting. Do not retry a completed
  classification because it is wrong. Schema, citation and output-budget errors
  count as INVALID under the selected condition.
- Keep INVALID and missing-output cases in the denominator. Aborted runs report
  partial coverage and cannot be presented as completed full evaluation.
- Abort safely on failed workspace cleanup, model/source drift, critically low
  disk space or repeated infrastructure failure. Preserve predictions and logs.
- Laptop shutdown stops local work. Resume later from saved progress. No promise
  that local inference continues while the machine is off or asleep.

## Runtime estimate

The earlier ten-claim full-article pilot averaged 41.50 seconds of preparation
per claim. The selected parts replay averaged 12.43 seconds of generation and
validation. Adding those different pilot measurements estimates 53.93 seconds
per claim, not a measured throughput for the new integrated runner.

For 13,448 eligible claims, that projects to **201.45 compute hours (8.4 days
continuously)** for the main pipeline alone. A practical planning window is
**8–12+ days**, depending on article sizes, API throttling, outages and machine
availability. This estimate is weak because it comes from only ten claims.
Baseline inference, engineering checks and final analysis add time. Do not
extrapolate the older simpler-pipeline 26-hour estimate to this architecture.

At five selected article URLs per claim, there are up to 67,240 claim/article
fetches; actual API request counts can be higher. Per-claim deletion means pages
shared across claims can be fetched and embedded repeatedly. This honors the
agreed retention policy but costs more than persistent caching. No persistent
full-article cache or parallel crawler is introduced by this plan.

## Assignment deliverables

After the run finishes, allow roughly 2–4 hours for computation and presentation
of results, plus time for careful qualitative review:

- Matched main-model/baseline table: N, accuracy, macro-F1, invalid count, latency
  and completed coverage; model-provided versus derived labels shown separately.
- Per-class precision/recall/F1, a 3-by-4 confusion matrix including INVALID,
  and paired accuracy difference with a bootstrap interval. Do not describe
  claim-bootstrap intervals as repeated training runs.
- Paired examples: baseline right/main wrong, main right/baseline wrong, both
  wrong, and successes; exact article passages, citations and responses. Separate
  retrieval gaps from evidence misinterpretation and decomposition/output errors.
- Explicit hypothesis verdict supported by the actual measurement, including
  negative outcomes. Keep the inspected 7/10 development result separate.
- Literature comparisons that name dataset, split, labels and metric. Do not
  compute strict FEVER scores from current article block IDs: they are not
  historical FEVER sentence IDs. Label accuracy/macro-F1 support our direct
  paired comparison; published strict FEVER results remain qualified context.

The original official-dev benchmark and the newly evaluated eligible internal
test population are different. GEAR/KGAT official development scores cannot be
placed beside these internal-test numbers as a matched leaderboard. A future
official evaluation would be a separate, explicitly authorized run.
