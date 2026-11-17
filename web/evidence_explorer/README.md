# FEVER evidence explorer

From the repository root:

```bash
.venv/bin/python src/evidence_explorer.py
```

Open http://127.0.0.1:8765. Use `--port 8766` if that port is already occupied.
Keep the terminal running while browsing; Ctrl+C stops the local server.

The viewer covers all 19,998 claims in the saved official development evaluation.
Search text/IDs, paginate, filter by annotation coverage or model correctness,
and click a claim. Each detail view reconstructs the original retrieval ranking
using the existing read-only Wikipedia database and the unchanged `retrieve()`
function. No neural model is loaded or called.

## Predictions and error tracing

Open http://127.0.0.1:8765/?view=errors for the confusion matrix. Choose the
combined model, claim-only H2, or evidence RoBERTa. Rows are human FEVER labels;
columns are saved predictions. Click a cell to filter to its exact claims.

TP/FP/TN/FN are **one-versus-rest for the selected positive class**. A true
negative for SUPPORTS can still be a REFUTES-versus-NEI classification error.
Class precision/recall/F1 and counts are recomputed from saved predictions;
they reproduce the original official benchmark summaries. The four diagnostic
buttons separate wrong labels with complete retrieval, wrong labels with
incomplete retrieval, NEI errors, and correct labels lacking complete annotated
evidence. These are observed patterns, not causal labels.

Open a claim's prediction trace to compare the two model predictions and the
combined result, inspect the combination scores, and trace annotated evidence
through article search, sentence selection, and the 256-token input limit.
The existing local evidence tokenizer is loaded lazily; neural weights are not
loaded. Offset mappings replay the actual pair tokenization and identify which
selected sentence tokens were retained. Losing part of an annotated sentence
does not establish that its missing words caused the wrong label. Missing or
changed retrieval snapshots are flagged; annotation-based diagnosis cannot
identify a retrieval cause for NEI. Claim-only did not consume the evidence.

The page can download confusion tables as CSV or an individual trace as JSON.
These downloads are generated in the browser from the displayed data.
The page's provenance section explains that FEVER supplied the claims and
human labels, while the viewer was added with AI assistance around the existing
research code and saved results. The localhost address is not a public link
that a professor can access from a different computer.

- **Top 5:** sentences supplied to the classifier in the saved experiment, when
  the current reconstruction matches the saved references.
- **All ranked:** every indexed sentence from the up-to-ten candidate pages,
  in the retriever's order. This is not every related article on Wikipedia;
  pages outside the candidate set were never searched by this function.
- **Annotated matches:** sentences within that same candidate set that match
  FEVER's annotation references. Original ranks are preserved.
- **Candidate articles:** original BM25 title rank and raw title-search score.
  Lower title BM25 scores are better. Sentence scores instead rank higher first:
  distinct overlapping words plus `0.1 / (one-based article rank)`.
- **FEVER annotations:** alternative complete evidence groups, including
  references that the retriever missed or that the local index does not contain.
- Click an article title to read all its indexed local sentences. External
  Wikipedia links lead to today's article, which may differ from the snapshot.

The aggregate statistics and model labels describe the saved evaluation.
Current detail rankings are computed on demand; differences from saved top-five
references are flagged explicitly. NEI has no required evidence group and is
excluded from annotated evidence recall. Missing an annotation is not proof that
a retrieved sentence is useless or that the claim is false.

The server binds only to 127.0.0.1, exposes read-only routes, does not serve
arbitrary files, and opens SQLite in read-only mode. Original datasets,
predictions, checkpoints, retriever code, and experiment reports are not changed.

Run meaningful ranking/snapshot and coverage checks with:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_evidence_explorer.py'
```

## Controlled token-budget comparison

Open `http://127.0.0.1:8765/token-budget` for progress and final results of the
256-versus-512 evidence-input experiment. The page reads only the fixed local
`reports/part2/token_budget_20261028/` directory. It does not launch inference.
The main explorer continues to display the original 256-token benchmark.

The new experiment uses the same 19,998 official development claims, saved
five-sentence references, sentence order, local checkpoint, label order and
fusion parameters. Only the evidence branch changes to 512 tokens. Claim-only
probabilities remain frozen at their original 256-token values in both
conditions. Original 256-token probabilities are reused after a 64-claim replay
check. This is exploratory development-set analysis after inspecting errors,
not an untouched final test or model-selection claim.

Commands, from the repository root using the existing environment:

```bash
.venv/bin/python src/experiment_token_budget.py --smoke --output reports/part2/token_budget_smoke_NEW
.venv/bin/python src/experiment_token_budget.py --output reports/part2/token_budget_NEW
.venv/bin/python src/experiment_token_budget.py --resume --output reports/part2/token_budget_NEW
```

Use a new directory for each experiment. Resume validates source/checkpoint/data
hashes, runtime, database metadata and the saved claim-ID prefix. Do not edit the
runner midway through a run. A partial final JSON line is rejected rather than
silently discarded. The web comparison page is pinned to the dated experiment;
other runs remain available as files in their own directories.

Artifacts include `protocol.json`, an exact runner snapshot, per-claim
`outputs.jsonl`, `summary.json`, `per_class.csv`, `changed_predictions.csv` and a
plain-language `REPORT.md`. TP/FP/TN/FN are one-versus-rest for each FEVER label.
Paired changes count both repaired errors and newly introduced errors. Bootstrap
intervals use the empirical paired accuracy-difference distribution; McNemar
p-values are unadjusted. Inference timing excludes loading, tokenization and I/O.
The smoke timing is a diagnostic, not an end-to-end latency benchmark.

The first smoke directory has a documented report-formatting failure; use
`token_budget_20261028_smoke_verified/` for the completed verification record.
No original predictions, checkpoints or production token settings are changed.

## Local LLM comparison and reusable verifier

`http://127.0.0.1:8765/llm-comparison` shows progress and then the completed
Ollama-versus-RoBERTa pilot from
`reports/part2/ollama_comparison_20261028/`. The explorer server only reads these
artifacts; opening a page never starts an LLM request.

The pilot freezes a random seed-42 sample of 100 claims per official FEVER label
before inference. The already inspected Telemundo case is excluded from the
sample and tested separately. The default is the already-installed
`llama3.1:8b`, with its Ollama digest and runtime version recorded. The exact
five retrieved sentence texts and order are shared with both RoBERTa conditions.
All 300 examples fit fully within the RoBERTa 512-token input. The LLM receives a
fixed classification instruction and JSON evidence array, with no gold labels
or annotations. Its brief explanations are generated text, not a faithful
internal reasoning trace or independent human review.

New experiments require new directories:

```bash
.venv/bin/python src/experiment_ollama_fever.py prepare --output reports/part2/ollama_comparison_NEW
.venv/bin/python src/experiment_ollama_fever.py run --output reports/part2/ollama_comparison_NEW
.venv/bin/python src/experiment_ollama_fever.py run --resume --output reports/part2/ollama_comparison_NEW
```

Preparation checks that the requested model is already installed and never
pulls a model. Run uses the local Ollama endpoint, temperature 0, seed 42,
context limit 4096 and output limit 128. The same prepared prompt and model are
used throughout. Completed requests include their exact payload, raw response,
timing and parsed label. Invalid, incomplete or failed responses remain in the
sample denominator as INVALID, rather than being relabeled NEI or dropped.
Three consecutive failures stop the run for inspection. Resumption checks input
hashes, the model digest/runtime version and the completed ID prefix.

`sample.jsonl` contains the evaluation gold data for scoring. Only `claim` and
`sentences[].text` are passed to the prompt builder. `protocol.json` records
selection and settings; `responses.jsonl` preserves request/response traces;
`predictions.csv`, `per_class.csv`, `summary.json` and `REPORT.md` contain paired
results. Strict FEVER uses the original five retrieved references, rather than
scoring the quality of generated explanations. Paired bootstrap intervals and
unadjusted McNemar tests are exploratory. A 300-claim development pilot does not
establish full-dataset performance or control for possible pretraining exposure.

To check new claims with the same retriever and local LLM:

```bash
.venv/bin/python src/verify_ollama.py --claim 'Telemundo is an English-language television network.' --trace-file output/telemundo_local_llm_NEW.json
```

The trace filename must be new. This command queries the existing read-only
Wikipedia index, retrieves five sentences with the existing ranking function,
and returns the LLM label, short explanation and source sentences. It keeps the
original calibrated RoBERTa verifier available and does not change its defaults.

The dated pilot has a documented report-only correction: the claim-only
baseline submits no evidence for strict FEVER scoring. Its initial reports are
preserved in `initial_scoring/`; `runner_source.py` preserves the actual
inference code. `analysis_source.py`, `finalizer_source.py` and
`analysis_manifest.json` record the corrected scoring/rendering code. All LLM
labels, requests, responses and paired accuracy comparisons are unchanged.
The corrected report also includes paired uncertainty for the strict FEVER
comparison. Its interval includes zero. Source changes correctly prevent an
old frozen inference run from resuming under a different runner.

## Inspect the actual Ollama calls

Open `http://127.0.0.1:8765/ollama-traces#111897`. This read-only inspector
uses the saved 300 evaluation calls and separate Telemundo diagnostic. Search
claims or IDs, filter correct/wrong labels and page through all 301 records.
Opening a trace never invokes Ollama or changes experiment results.

Each trace shows the actual Python call helper with a source line, the exact
saved request (including prompt, evidence and JSON schema), the unmodified
`message.content`, the full raw API response, and the parsed label/explanation.
Token counts and timing are reported separately from predictions. No
per-class probabilities or hidden reasoning are inferred from these fields.
Gold labels appear for evaluation and are absent from the model request.
The call helper's source hash is checked against the saved run; a change is
flagged instead of silently presenting new code as the original helper. Prompt
and parser snippets come from the frozen inference source snapshot.

The full trace can be downloaded as JSON. The HTTP server serves only fixed
routes and trace IDs; it does not expose arbitrary file paths. Rendering uses
DOM text content for model responses, claims, code and evidence.

## Analyze the local LLM accuracy difference

`http://127.0.0.1:8765/llm-error-analysis` presents the measured class and
annotation-coverage breakdown of the same 300-claim pilot, with links to exact
saved traces. Artifacts are separate in
`reports/part2/ollama_error_analysis_20261028/`; original predictions and metrics
are unchanged.

The analysis also includes a deliberately separate, bounded oracle diagnostic:
20 seed-42 random claims from the 79 verifiable cases lacking a complete
annotated group in their original evidence, chosen without conditioning on LLM
correctness. Claim/model/prompt/decoding stay fixed while evidence is replaced
by one complete annotated group available in the local index. This uses gold
annotations and changes evidence relevance and amount together. It is a
diagnostic intervention, not an end-to-end result or a new production score.
All oracle inputs, exact requests/responses and settings are retained.

## Configuration search with separate selection and assessment

`http://127.0.0.1:8765/llm-configurations` follows the five predeclared pipeline
configurations in `reports/part2/llm_configurations_20261028/`. A/B/C/D cross two
prompts with two retrieval strategies. E reuses D's model responses and applies
a fixed claim-only fallback solely for LLM NEI answers (claim-only SUPPORTS or
REFUTES softmax at least 0.80). That score is not a factual-confidence guarantee.
Raw LLM labels and final hybrid labels are preserved separately.

The selection sample has 60 internal-validation claims (20 per class), from
raw offsets >=8000, excluding normalized training/official-dev text. The 120
assessment claims (40 per class) are disjoint from calibration and the prior
300-claim LLM pilot. All IDs and settings are frozen before new LLM inference.
Select by calibration macro-F1, then accuracy, then fewer changes on ties. Only
baseline A and the selected pipeline are run on assessment. Official dev has
previously been evaluated with RoBERTa, so this is exploratory assessment, not a
blind official test. Configuration ideas were motivated by previously inspected
errors; those earlier LLM-pilot texts are excluded from assessment.

```bash
.venv/bin/python src/experiment_llm_configurations.py prepare --output reports/part2/llm_configs_NEW
.venv/bin/python src/experiment_llm_configurations.py run --phase calibration --output reports/part2/llm_configs_NEW
.venv/bin/python src/experiment_llm_configurations.py run --phase assessment --output reports/part2/llm_configs_NEW
.venv/bin/python src/experiment_llm_configurations.py report --output reports/part2/llm_configs_NEW
```

New preparations require new directories. `--resume` verifies the saved prefix,
source and input hashes and local model/runtime before continuing either phase.
An OS file lock prevents concurrent runs of the same experiment directory.
No model pulls, training, paid APIs or default changes occur.
