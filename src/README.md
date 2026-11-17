# Source map

The current main-model study is the temporary full-article RAG + Llama parts
classifier in `parts_rag_classifier.py`, with paired execution in
`experiment_parts_test.py`. Evidence RoBERTa is its baseline. See
`docs/parts_test60_20261030.md` for explicit run/resume/scoring commands and the
read-only `serve_parts_test.py` dashboard. The frozen ten-case replay below is
the historical prompt-selection record, not this new test evaluation.

Start from the repository root, using the existing Python environment:

```sh
.venv/bin/python src/project.py
.venv/bin/python src/project.py prompt
.venv/bin/python src/project.py serve --port 8765
```

`prompt` verifies and prints the selected original parts-verdict instruction.
`serve` displays saved results; it does not classify new claims. The selected
condition scored 7/10 with code-derived labels on inspected development cases,
and 5/10 using the model's own overall labels. Selection is recorded in
`docs/working_parts_prompt.json`; it is not a global deployment switch.

## Retrieval and verification components

| File | Responsibility |
|---|---|
| `retrieve_title_aware.py` | Original article candidate search and sentence ranking |
| `retrieve_entity_anchored.py`, `evidence_pipeline_v2.py` | Separately measured retrieval/fusion alternatives |
| `fetch_wikipedia_articles.py` | Full article fetching, source revisions and cache provenance |
| `ephemeral_rag.py` | Temporary chunk/vector database, evidence packing and original direct classifier |
| `rag_embedding_worker.py` | Isolated local embedding computation |
| `local_llm_experiments.py` | Local model connection and request transport |
| `replay_claim_parts.py` | Frozen original parts prompt, validation, aggregation and ten-input replay |
| `verify_calibrated.py` | Existing neural comparison using frozen local checkpoints |
| `evaluate_retrieval.py`, `evaluate_official_fever.py` | Retrieval and official FEVER scoring |
| `evidence_explorer.py`, `live_ollama_viewer.py`, `ollama_trace_viewer.py` | Saved-result viewers |
| `preprocess.py`, `prepare_retrieved_training.py`, `tune_weighted.py` | Data preparation and explicit training workflows |

The current parts implementation is a replay runner for the saved ten contexts.
`verify_ollama.py`, `verify_article_rag.py`, and `ephemeral_rag.py` are distinct
experimental interfaces with their own prompts/evidence rules; they must not be
presented as automatically running the selected parts condition on new claims.

## Experiment history

Existing `experiment_*.py` files and source-hashed replay/finalization modules
remain at their original locations. Saved protocols refer to their paths and
hashes. Moving or editing them would break those reproducibility checks.
Full-reading and prompt-v2 experiments remain available as negative results;
their existence does not make them the selected configuration. The 19K Llama
evaluation remains stopped and is never started by `project.py`.

## Report tools

HTML, document and packaging generators live in `reporting/`. For the current
trace pages, these aliases rebuild reports from saved results without inference:

```sh
.venv/bin/python src/project.py report parts
.venv/bin/python src/project.py report parts-v2
.venv/bin/python src/project.py report rag
.venv/bin/python src/project.py report references
```

Other moved tools can be invoked as modules, for example:

```sh
.venv/bin/python -m src.reporting.report_exhaustive_tokens
```

Report generation can rewrite derived HTML/Markdown. It is an explicit command,
separate from the inference path. Some older generators remain at top level
because their source is fingerprinted in saved manifests.

## Historical prototypes

`legacy/` contains the early five-file retriever demo, model-initialization demos,
old trainers, gold-evidence preparation, and superseded retrieval/ablation scripts.
Their source bytes are preserved. They are not imported by the current pipeline.
Some old trainers overwrite original checkpoint/log paths; they are retained for
historical inspection, not recommended for a new run. New training or evaluation
must use distinct output locations and an explicit protocol.

The move mapping and pre-cleanup hashes are recorded in
`output/code_review_20261029/cleanup_manifest.json`. Existing datasets,
checkpoints, predictions, labels and recorded metrics were not changed.
