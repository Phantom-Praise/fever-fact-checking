# Temporary per-claim RAG

This classifier creates a fresh local knowledge base for one claim, then deletes it. No per-claim vector databases or downloaded full articles accumulate across claims.

## Architecture

Claim → existing title search (ten ranked candidates) → retain the first five articles → fetch full Wikipedia article renders into a temporary directory → paragraph/table chunks → local MiniLM embeddings → temporary FAISS vector index plus SQLite passage metadata → retrieve relevant context → local Llama classification with excerpt citations → delete the entire workspace.

The learned RoBERTa classifiers are preserved as separate baselines; the existing lexical title retriever remains unchanged. The new classifier has no claim-only override.

## What is retained

- Classification, generated explanation and errors.
- Exact supplied passages and cited excerpts.
- Wikipedia URLs, page/revision IDs, timestamps, block IDs and character offsets.
- Full prompt, API request/response and timing/token metadata.
- Source hashes, retrieval parameters, number of indexed chunks, workspace sizes and cleanup verification.

Full fetched HTML, full cleaned articles, complete chunk metadata tables, embeddings and FAISS indexes are created only beneath the per-claim temporary workspace and removed on completion or handled exceptions. Retained prompts necessarily contain the subset of article text sent to Llama. Existing article-fetch demonstrations and earlier research artifacts are preserved; this new mode does not read from or add articles to that permanent demonstration cache.

The reusable embedding-model installation stays on disk. It is not a per-claim database. The model is `sentence-transformers/all-MiniLM-L6-v2` pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; the verified weights/config/tokenizer total 91,567,205 bytes. It uses attention-mask mean pooling and L2 normalization on CPU. All model loads are local-only with no remote code.

## Implementation

- `src/ephemeral_rag.py`: chunking, per-claim FAISS/SQLite database, evidence packing, citations and cleanup.
- `src/rag_embedding_worker.py`: isolated CPU embedding subprocess. This avoids a verified FAISS/Torch OpenMP conflict in this Mac's existing environment.
- `src/experiment_ephemeral_rag.py`: reproducible, frozen ten-claim comparison.
- `requirements-ephemeral-rag.txt`: Python-3.9-compatible FAISS version. No existing package was upgraded.

The temporary database is a local composite store: FAISS indexes normalized 384-dimensional vectors; SQLite holds passage text/provenance. No separate database server is needed. There is no conversation state or shared article index between claims.

## Evidence and context controls

Full article body content is available for indexing; there is no five-sentence input cap. Paragraph/table windows are limited to roughly 200 embedding tokens, with exact block offsets preserved. Embedding inputs are checked against 256 tokens rather than silently truncated.

Vector search returns up to thirty candidates. Whole passages are packed under a 7,600-character evidence-JSON budget for the existing 4,096-token Llama context. Character count is not an exact tokenizer count: saved Ollama usage/completion metadata is checked, and exhausted/incomplete responses become INVALID. The output may cite at most five precise supplied excerpts, rather than treating five evidence sentences as an input restriction.

Citation validation checks IDs and exact supplied text/source spans. It does not establish logical entailment, claim truth or explanation faithfulness. IDs are internal excerpt IDs, not historical FEVER sentence IDs. No strict FEVER score is assigned to live-Wikipedia references.

Cleanup runs through Python context managers and an explicit database close on ordinary success/failure and handled termination. A forced OS kill or power loss cannot execute Python cleanup; any abandoned workspace would need inspection/cleanup before resuming. The ten-claim pilot verifies deletion after every processed claim.

## Pilot protocol

A deterministic sample of ten distinct official-development claims uses 4 SUPPORTS, 3 REFUTES and 3 NEI labels. Sampling excludes normalized processed training texts and previous inspected LLM/prototype texts. Gold labels are used for sampling/scoring only; neither classifier receives them.

Both conditions use the same installed Llama 3.1 8B model, seed 42, temperature 0, context 4,096 and output cap 256. Condition order within each claim is frozen/randomized before inference. The original condition uses its direct prompt and original five sentence strings. The RAG condition uses fresh Wikipedia content, vector retrieval, broader context and an excerpt-citation prompt.

This is a pipeline comparison: corpus age, evidence retrieval, context and prompt differ together. Ten outcomes cannot establish a general improvement or isolate a vector-database effect. The output cap also differs from the earlier 128-token original experiment, so those earlier scores are not substituted for fresh pilot predictions.

The previously approved 19K evaluation remains stopped and is not resumed by this pilot.

## Run commands

Use a new output directory; existing results are rejected:

```sh
.venv/bin/python src/experiment_ephemeral_rag.py prepare --output reports/part2/NEW_PILOT
.venv/bin/python src/experiment_ephemeral_rag.py run --output reports/part2/NEW_PILOT
```

The prepared protocol and input/source hashes must remain unchanged before running. Model files remain installed; every claim's freshly fetched corpus and vectors are deleted.
