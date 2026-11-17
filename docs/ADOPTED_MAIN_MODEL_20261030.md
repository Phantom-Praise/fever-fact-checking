# Main model adopted for the current assignment

The user selected the temporary Wikipedia RAG + Llama essential-parts method as
the main model on October 30, 2026. Evidence RoBERTa is a comparison baseline.
Earlier fusion, symbolic and direct-Llama results remain historical experiments;
their percentages are not substituted for the new main model's test results.

## Current question

Does the frozen temporary full-article RAG + Llama parts-verdict pipeline improve
label accuracy and macro-F1 over evidence RoBERTa on the same fresh eligible
FEVER test claims?

Hypothesis: the main pipeline improves both metrics. This is a narrowed question
chosen after earlier development work; the 60-claim evaluation protocol fixes
this hypothesis before those new predictions are scored. It is not an original
preregistered question or completion of every question from the proposal.

## System

Claim → existing retrieval's ten article candidates → top five article URLs →
full article fetching → chunks and local embeddings → per-claim temporary
FAISS/SQLite database → selected relevant passages → Llama's essential-part
verdicts with citations → deterministic SUPPORTS / REFUTES / NEI label.

The database is searched across all indexed chunks, but Llama reads only the
packed retrieved passages. The original parts prompt is preserved. All refuted
parts produce REFUTES; all supported parts produce SUPPORTS; otherwise NEI.
Invalid responses are errors. Model-provided overall labels are diagnostic and
reported separately. Full articles and vectors are deleted after each claim;
inputs, responses, supplied/cited passages and revision metadata remain.

## Current measurement

Run directory: `reports/part2/parts_test60_20261030_v2`.
Sample: 60 eligible internal test claims, 20 per label, fixed before inference.
RoBERTa uses the same claim IDs but its original historical evidence pipeline.
This is a system comparison, not an isolated test of the prompt or embeddings.

The sample was chosen to meet the user's 3–4 hour completion window. It is not
the proposed full-partition evaluation. The actual status and measured results
belong to this run's progress/summary files; this adoption document does not
declare an unfinished evaluation successful.

## How this relates to the proposal

The comparison and error traces support an empirical analysis of evidence use
and decision quality. Evidence insufficiency, evidence misinterpretation,
decomposition mistakes and output failures can be discussed with examples.
Original generated-claim representativeness and independent human-review work
remain unanswered. Zero symbolic coverage and earlier negative prompt/full-read
results remain relevant limitations and development history, not achievements
of the adopted main model.

Earlier adopted scopes/reports are preserved as dated records. This document
supersedes their main-model designation for the current assignment only.
