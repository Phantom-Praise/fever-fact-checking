# Climate-FEVER migration — November 17, 2026

The next independent-model experiment uses the published Climate-FEVER
collection instead of 19,998 FEVER development claims. Existing FEVER data,
checkpoints, notebooks, protocols, and measurements remain unchanged. This is
a new transfer evaluation, not a replacement measurement of the old benchmark.

## Dataset and scope

Source: https://github.com/tdiggelm/climate-fever-dataset

Pinned commit: `03de61617b10a5c1935f8e08bb0e8ac1ee775356`.
Raw dataset SHA-256: `8a4b9032d861be482ffb49dddfd283ffa6089e654f1e968040011882c5eb6e0b`.
Download size: 3,152,039 bytes. This download contains data, not model weights.

The downloaded release has **1,535 claims**: Supports 654, Refutes 253,
Not Enough Info 474, Disputed 154. These counts come from the pinned data;
they differ slightly from counts in the original paper. All 1,535 are retained.
This release is one evaluation collection. We do not invent an official
train/validation/test split or train on its reference answers.

`colab/climate_fever.py` validates the source and creates:

- `inputs.json`: ID and claim only, for inference.
- `labels.json`: four-class answers, for reporting only.
- `references.json`: original evidence annotations and votes, for analysis only.
- `manifest.json`: counts, checksums, and scope.

## Models and evidence

Five separate selectors/classifiers remain the intended comparison: Qwen 2.5
7B, Llama 3.1 8B, Mistral 7B v0.3, Gemma 3 4B, and the existing fine-tuned
RoBERTa. Only one model is loaded at a time. Llama/Gemma access may require the
user accepting the provider's license and setting HF_TOKEN in Colab Secrets.

Reuse the existing **full historical FEVER Wikipedia snapshot**, not the five
annotated sentences. Each LLM generates its own title-search query. Existing
BM25 title search returns ten pages, full article sentence text is processed
in batches, and that model recursively selects up to three final excerpts.
RoBERTa searches with the claim and ranks excerpts by its maximum Supports/
Refutes probability; this is an explicit heuristic, not a trained relevance head.
No gold article titles, evidence sentences, votes or labels guide retrieval.

This is still constrained retrieval, not reading all Wikipedia for each claim.
The corpus predates Climate-FEVER, so temporal mismatch and missing evidence
must be reported. The smaller claim set reduces inference volume, but the
full corpus/index remains large. Source articles/index persist as a shared
corpus; selected per-claim evidence and all prompts/responses remain in traces.

The generative classifier retains essential-part verdicts and introduces
Disputed explicitly. Code derives Disputed for a disputed part or a mixture
of supported and refuted essential parts, then Refutes, all-Supports, or NEI.
This operational rule is frozen in the protocol; it is not guaranteed to match
all annotation disagreements. Citations are validated by ID, not semantically
verified by code. The new prompt has not yet been measured.

The existing RoBERTa has **three output classes**. It is evaluated honestly as
a FEVER-trained transfer baseline and cannot predict Disputed. Four-class
macro-F1 includes its zero Disputed recall. A secondary, explicitly restricted
non-Disputed comparison contains **1,381 claims** once fully evaluated. Neither
Disputed nor INVALID is silently relabelled NEI. Building a four-class trained
RoBERTa requires separate approved training data/work; it has not been done.

## Execution and preservation

Open `output/Climate_FEVER_Project/Climate_FEVER_Independent_1535.ipynb` in Colab.
The notebook embeds the verified small dataset and runtime; a ZIP upload is
optional. It automatically creates `MyDrive/Climate_FEVER_Project`. It does
not modify `MyDrive/FEVER_Project` results or notebooks.

1. Run setup. This does not start inference or download model weights/Wikipedia.
2. Choose `RUN_MODE='pilot'` for twelve source-order engineering claims and
   enable model downloads after reviewing the printed sizes. No runtime is
   promised until this measures real speed, errors and VRAM.
3. Inspect all invalid outputs and full traces. If used to revise the prompt,
   these claims become development examples and their exposure must be disclosed.
4. Choose `RUN_MODE='full'` for all 1,535 claims, with a separate output folder.
   Start Qwen first; enable the remaining model queue after its diagnostic.

Every completed claim is immutable and resumable. INVALID counts as an error,
including malformed JSON, unknown citations and incomplete generation. Three
consecutive invalid outputs stop execution. Out-of-memory and I/O errors stop
immediately, retaining an error trace. Failed records are never overwritten;
a corrected protocol or deliberate retry requires a new named run. A 30-minute
session budget is checked between claims, not a hard interrupt within a claim.
Colab can still disconnect; checkpointing reduces lost work, not disconnections.

Reports include four-class accuracy, macro-F1, per-class percentage precision/
recall/F1, an INVALID confusion-matrix column, secondary non-Disputed accuracy,
and mean runtime. Partial scores are marked partial and are not valid rankings
when models have processed different claim sets. Strict FEVER evidence scoring
is not implemented for this different dataset. Output ZIP export is included.

No Climate-FEVER model scores, improvements, new training or completed
five-model comparison are claimed. The professor's previous 19K requirement
still needs a scope discussion; migrating code does not establish approval.

## Local preparation and tests

```bash
.venv/bin/python colab/climate_fever.py --source data/climate_fever/source/climate-fever.jsonl --output data/climate_fever/prepared
.venv/bin/python -m unittest discover -s tests -p 'test_climate_fever.py'
.venv/bin/python colab/build_climate_notebook.py
```

Dataset citation: Diggelmann, T., Boyd-Graber, J., Bulian, J., Ciaramita, M.,
and Leippold, M. (2020). CLIMATE-FEVER: A Dataset for Verification of Real-World
Climate Claims. https://arxiv.org/abs/2012.00614
