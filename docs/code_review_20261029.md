# Source review and cleanup — October 29, 2026

## Outcome

The source folder now distinguishes the retrieval/classification components,
frozen experiment runners, report generators, and historical prototypes.
This is a maintenance change, not a measured model improvement.

| Change | Status | Scope |
|---|---|---|
| Move report/document generators into `src/reporting/` | Completed | 12 files; repository-root paths repaired |
| Move historical demos and superseded scripts into `src/legacy/` | Completed | 13 files; source bytes preserved |
| Add `src/project.py` and `src/README.md` | Completed | Safe help, selected-prompt inspection, saved-results viewer, four report aliases |
| Group normalized duplicate claims during preprocessing | Completed for future runs | Case/whitespace variants share a partition; original claim text retained |
| Prevent preprocessing from overwriting existing datasets | Completed | Check all target paths first; exclusive file creation |
| Change existing datasets, train models, resume 19K evaluation | Not performed | Existing artifacts retained |

There were 81 top-level Python files before cleanup. There are now 57, including
the new entry point. No unique source file was deleted. Thirty-six sources
referenced or fingerprinted by saved protocols remain at their original paths
with identical hashes. Moving those would break historical provenance checks.

The full move mapping and original hashes are in
`output/code_review_20261029/cleanup_manifest.json`.

## Findings that still need separate work

| Priority | Finding and evidence | Implication / next action |
|---|---|---|
| P1 | The historical evidence-training preparation assigns empty evidence when no gold group exists (`src/legacy/prepare_evidence.py:115`). All 28,517 training NEI rows and all 3,595 validation NEI rows in those evidence files have empty evidence; the other classes have none empty. | The old evidence checkpoint can learn an empty-input shortcut that differs from retrieved-evidence evaluation. Keep its results with this limitation. A new controlled training experiment must use retrieved evidence for every class, distinct outputs, and explicit training approval. The existing 77-example retrieved-training pilot is not a replacement checkpoint. |
| P2 | Parts validation checks only returned token counts (`src/replay_claim_parts.py:73`). Missing counts default to zero, and a server that truncates the input before processing can return counts below the limit. The original RAG context packer uses a character budget. | These checks do not prove the entire submitted prompt was processed. A new version should preflight exact rendered prompt tokens and validate complete request/response metadata. This is not proof that any particular ten-claim error resulted from truncation. |
| P2 | Official evaluation resumes inference from a saved prefix but opens condition exports and the comparison CSV with exclusive creation (`src/evaluate_official_fever.py:127`, `:143`). | Resuming after a partial export, or resuming a completed run, can fail at finalization with `FileExistsError`. Add an idempotent, content-checked finalization path in a new version; preserve the frozen runner. This finding is from code inspection, not a rerun of the full evaluator. |
| P2 | The selected parts prompt lives in a ten-input replay runner. Other verifier interfaces retain their separately measured prompts. | Selection is a recorded research condition, not a global switch. Before testing fresh claims, implement an explicit general parts-verdict runner with frozen retrieval/model settings and separate outputs. Do not silently change old conditions. |
| P3 | `src/finalize_exhaustive_tokens.py` depends on an absolute local model-blob path and discovery of the running macOS Ollama server. | Exact-token preparation is machine-specific. Parameterize the model/tokenizer connection before porting that experiment. |

## Split isolation audit

The existing processed partitions contain six normalized claim groups shared
between train and validation, and six between train and test. Validation/test
overlap is zero. Exact-string checks missed these case/whitespace variants.

Example: “Abraham Lincoln was the 16th President.” and “Abraham Lincoln was the
16th president.” were assigned to different partitions.

The checked deterministic 20% training subset and first 2,000 validation examples
have zero normalized overlap. The checked 120-claim LLM assessment also has zero
normalized training overlap. Therefore, this review does not establish that the
six groups inflated those reported results. The effect on other measurements has
not been quantified.

The preprocessing fix applies only when deliberately generating a new dataset
directory. Existing partitions, checkpoint identities, and benchmark scores were
not regenerated or relabeled.

## What to use now

```sh
.venv/bin/python src/project.py
.venv/bin/python src/project.py prompt
.venv/bin/python src/project.py serve --port 8765
```

See `src/README.md` for the component map and report commands. The website serves
saved predictions and traces. It does not start model inference. Historical
trainers remain in the archive for inspection; some overwrite fixed output
locations if explicitly run, so they are not recommended execution paths.

## Verification and limits

- All 124 tests passed using the existing `.venv`; nine tests were added for
  entry-point behavior, prompt fingerprints, split grouping, and overwrite guards.
- All Python source files compiled without executing historical trainers.
- All 36 protected source hashes match the pre-cleanup inventory. All 13 archived
  historical files are byte-identical to their original versions.
- The saved RAG, parts, and parts-v2 HTML files retained their pre-check hashes
  after the integration tests regenerated the RAG page.
- No model training, paid API, model download, large-scale evaluation, or merge
  was performed. Local integration tests use existing artifacts; they do not
  establish new accuracy or statistical improvements.
- This review inspected source dependencies and the core data, retrieval,
  verification, evaluation, and viewer paths. It is not a claim that every
  historical experimental file has undergone a complete line-by-line audit.
