# Evaluation evidence source correction

The live-Wikipedia run was too slow: the inspected saved status had 88 paired
claims after about 88 minutes. Its recent timing extrapolated to about 332 hours
remaining, with long individual article requests. GPU embedding on one inspected
claim took 0.44 seconds; article fetching dominated that claim's preparation.
These observations do not establish classifier accuracy or a general speedup.

At the user's request, the new evaluation uses the Wikipedia text distributed
with FEVER as the RAG source. Existing top-ten claim-only rankings and their
first-five page IDs are unchanged. The union contains 24,319 distinct records
across 19,998 claims. All selected records were found in the original raw dump
and all have text. No label file or annotated-evidence selection was used during
export. Full selected raw `text` and `lines` fields, with source-file/line and
raw-record hashes, are packaged in a read-only SQLite corpus. This is all text
supplied by those FEVER records, not the full current Wikipedia website.

Each claim reads its selected article bodies locally, creates 200-token chunks,
embeds them with the same MiniLM model, builds a temporary FAISS/SQLite database,
retrieves 30 candidates and packs the same 7,600-character context budget. Both
unchanged RoBERTa models receive identical retained input with a 512-token limit.
Working articles, chunks and vectors are deleted after each claim; the static
input corpus remains as dataset storage. Traces retain only supplied passages
and provenance. No Wikipedia HTTP call, live fallback or gold-evidence fallback
is made. Missing preparation remains an execution error rather than NEI.

FEVER raw records lack revision IDs. Citations therefore retain snapshot database,
raw record and body hashes plus exact character spans, without fabricated oldid
links. Metrics remain label accuracy, macro-F1, per-class precision/recall/F1,
confusion matrices and paired corrections/new errors. An official strict FEVER
evidence score is not claimed.

The original live notebook, predictions and trained weights are preserved.
Snapshot conditions use a separate `roberta_paired_snapshot_dev_v1` run folder.
Source/protocol checks reject mixing the two evidence sources.

## Measurements and checks completed

- The 14.25 MiB source ZIP contains every selected page, without model weights.
- Local CPU evidence preparation succeeded for the first 20 official claims:
  mean 0.09853 seconds/claim; zero Wikipedia HTTP requests; exact citation spans
  checked against original record text; temporary databases deleted in all cases.
- This measures preparation only, excluding classifier inference, Colab hardware,
  model loading, transfer and Google Drive I/O. It is not an end-to-end ETA.
- 29 offline checks passed, including full-text coverage, no-network behavior,
  cleanup, citation spans, selection integrity, paired-input equality and resume.
- The actual ZIP import, checksums, notebook syntax and embedded source bytes were
  verified. No retraining or local classifier evaluation was performed.
- The new notebook performs a 10-pair execution check, then continues toward all
  19,998 claims within its time budget. No final accuracy is available yet.

Artifacts: `output/fever_snapshot_rag_20261106/`,
`output/FEVER_Project/FEVER_Evaluate_Snapshot_19998.ipynb`,
`colab/export_snapshot.py`, `colab/snapshot_rag.py`.
