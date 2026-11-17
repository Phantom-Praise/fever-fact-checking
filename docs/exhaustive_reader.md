# Reading every article chunk

This is a separate experiment. It leaves the original retriever, classifier,
previous pilot results, and stopped full-development evaluation intact.

The ten claim IDs and successful article revisions come from the earlier
temporary RAG pilot. The selected pages are not re-ranked or replaced. One
previously missing Wikipedia page remains recorded as unavailable.

For each claim:

1. Fetch the pinned revisions into a temporary folder and compare article-text
   hashes with the earlier pilot.
2. Use the existing paragraph/table chunker. It excludes headings, reference
   sections, external-link lists, and extraction/navigation clutter.
3. Store all eligible chunks in a temporary SQLite database. Read every row in
   article order. Embeddings and a similarity index are unnecessary for a full
   scan; there is no top-k filtering in this mode.
4. Pack every chunk into exactly one bounded reader batch. Llama selects source
   IDs. Code copies their original text, preventing fabricated quotations.
5. After every batch validates, pass all selected chunks to the user-approved
   essential-parts prompt. Keep the returned overall label and the separately
   derived label visible. Do not silently substitute one for the other.
6. Reject the judgment if a batch fails, citations are invented, generation is
   incomplete, or the conservative final-prompt budget cannot hold all findings.
   Keep INVALID as an error in the ten-claim denominator.
7. Delete downloaded articles and SQLite after each claim. Retain exact prompts,
   raw responses, copied evidence, revision URLs, hashes, and a coverage audit.

**Retention:** Since every batch prompt is saved, article body text survives
across the trace files. Deleting temporary downloads does not erase that text.

**Coverage:** All chunks supplied in valid completed calls proves delivery
coverage. It does not prove comprehension or perfect relevance selection. The
final judge sees selected evidence from all batches, rather than all raw text in
one prompt.

The protocol is frozen before the run. The first quote-extraction attempt was
stopped because Llama paraphrased purported exact quotes; its raw artifacts are
preserved separately. The revised reader selects IDs only. Neither version adds
claim-specific keywords or reference labels to prompts.

Run outputs: `reports/part2/exhaustive_reader_20261029/`.
Live local page: <http://127.0.0.1:8765/exhaustive-reader>.
After completion, run `.venv/bin/python -m src.reporting.report_exhaustive_reader` for
coverage auditing, comparison metrics, and the written report.

These ten claims were inspected during development. Their scores cannot support
a general improvement claim. Context budgets, multi-call workflow, prompts, and
evidence selection differ from the previous top-k condition.

## Conditional precise-citation follow-up

`src/refine_exhaustive_evidence.py` preserves that first protocol and handles only
its conservative final-budget failures. It reads all first-pass selected chunks
again, with exact sentence IDs. The model selects IDs; code copies the original
sentence spans. All selected sentences are passed to the final parts judge.
Judgments that already completed are reused without additional Llama calls.
The follow-up refuses to discard evidence silently if its budget is still too
small. Original and follow-up scores remain separate.

The follow-up protocol was frozen before its own inference, after observing the
first implementation's budget failures. It is a development repair, not a new
held-out test. Artifacts live in
`reports/part2/exhaustive_precise_20261029/`. After completion, run the original
report/audit first, then `.venv/bin/python -m src.reporting.report_precise_exhaustive`.

## Measured-token finalization (current final results)

The UTF-8 byte upper bound rejected seven original final prompts before inference.
The intermediate sentence-selection attempt also hit that guard and was stopped,
with its partial artifacts preserved. These were implementation-budget failures;
they did not measure actual token-context exhaustion.

`src/finalize_exhaustive_tokens.py` corrects this by counting tokens with the
active pinned model's llama-server `/tokenize` endpoint. It validates the model
blob/process and calibrates chat-wrapper overhead against seven saved requests
(the observed overhead is ten tokens). It then runs **all ten final judgments**
from the full original selected-chunk sets, with no further evidence filtering.
Each returned prompt token count must match the expected count exactly.
A mismatch, incomplete output, invented citation or actual protocol-limit excess
remains INVALID. Maximum context is 32,768 tokens, with output and safety reserve.

The tokenizer endpoint is documented in the official llama.cpp server reference:
<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>.
The runner discovers only the active loopback server for the exact pinned local
model blob. This integration is specific to the installed Ollama runtime.

Final artifacts: `reports/part2/exhaustive_token_verified_20261029/`.
Audit/report: `.venv/bin/python -m src.reporting.report_exhaustive_tokens` after completion.
Original full-scan and stopped follow-up artifacts remain separate.
