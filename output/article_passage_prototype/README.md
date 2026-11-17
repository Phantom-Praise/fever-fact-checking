# Automatic article-passage classifier — unmeasured prototype

Purpose: a claim goes in; SUPPORTS, REFUTES or NOT ENOUGH INFO comes out, with selected sentence references. No user follow-up or claim-only override is required.

Current state: evidence preparation and six offline control tests passed. No real Llama calls have been made for this prototype, so no classification improvement is claimed.

## Flow

1. Look for an exact leading subject title. If found, search all stored sentences in that article first; otherwise search the existing candidate articles.
2. Construct overlapping three-sentence passages and rank them lexically with BM25. Retain source page and sentence IDs.
3. Pack relevant passages with unique E1/E2 sentence IDs under a character budget. This budget is not a verified tokenizer limit; returned token/context limits are checked separately.
4. Ask the existing local Llama model for a three-class label, brief explanation and up to five supplied sentence IDs.
5. If the first answer is NEI, automatically expand retrieval to additional candidate pages once. No interactive user question is involved. If no extra context fits, keep the existing decision.
6. Reject incomplete responses, invented citations or unsupported output formats as INVALID. Citation existence does not independently validate entailment.

## Evidence preview

`evidence_preview_verified.json` contains the prepared inputs and provenance hashes, without predictions:

- Jack Falahee: searches both stored Jack_Falahee sentences, excluding unrelated Jack pages in the initial pass.
- Telemundo: searches seven stored sentences in Telemundo.
- Psych: searches eleven stored sentences; the selected evidence includes the July 7, 2006 debut sentence.

These are illustrative inspected cases, not a random evaluation or performance measurement.

## Research boundaries

The source is the existing FEVER Wikipedia sentence snapshot, not complete current Wikipedia HTML. Infoboxes, tables, sections or article content absent from that snapshot are not recovered. An exact title prefix is a heuristic and can select an ambiguous page such as Tool. BM25 remains lexical retrieval, not semantic reranking. Expansion is limited to the existing title candidates and two total inference rounds.

This mode changes retrieval, context, prompt, citation output and potentially request count together. Treat it as a pipeline experiment; gains cannot be assigned to one component without separate controls. Citation sentence references preserve the official FEVER limit of at most five submitted references, but evidence adequacy and three-class performance remain unmeasured.

## Reproduction

Use the existing environment:

```sh
.venv/bin/python src/verify_article_rag.py --claim 'Jack Falahee was born.' --output output/NEW_PREVIEW.json
```

Choose a new artifact path. Add `--run` for real local inference only after the active full-development benchmark finishes. The CLI refuses concurrent prototype inference while that batch is marked running. It uses the installed model only; no model download, training, paid API or corpus modification.

The prototype is in `src/verify_article_rag.py`; offline controls are in `tests/test_article_rag.py`. The approved A_original/D_both benchmark sources, prompts, outputs and protocol are unchanged.
