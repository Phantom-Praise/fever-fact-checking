# Retrieval-Augmented Fact-Checking with Transformer Models

## COSC 874 — Deep Learning II

## New experiment: Climate-FEVER

The next five-system evaluation now targets **all 1,535 Climate-FEVER claims**.
The [migration protocol](docs/CLIMATE_FEVER_MIGRATION.md) documents the pinned
release, four labels (including Disputed), separate answer files, existing
full-article corpus, and the three-class RoBERTa transfer limitation.
Start with `output/Climate_FEVER_Project/Climate_FEVER_Independent_1535.ipynb`.
It embeds the small data bundle and creates its own Drive folder. Setup is
enabled; inference and model downloads require explicit settings. No new scores
are claimed. All FEVER results and original experiment code remain preserved.
The sections below describe the earlier FEVER work.

For local VS Code work, see [the local runner](docs/CLIMATE_FEVER_LOCAL.md).
Run `.venv/bin/python src/climate_local.py inspect` for a read-only setup check.
The existing Ollama Llama model can run a resumable 12-claim diagnostic without
Colab. Missing models are reported, never downloaded or substituted silently.

The current main-model study uses temporary full-article Wikipedia RAG with
Llama essential-part verdicts. Evidence RoBERTa is its baseline. See the
[adopted model and question](docs/ADOPTED_MAIN_MODEL_20261030.md) and the
[60-claim test study](docs/parts_test60_20261030.md). Earlier RoBERTa/fusion
results remain historical experiments rather than results for this new method.

## Working source layout

Start with `.venv/bin/python src/project.py` for the current commands and
[source map](src/README.md). Use `src/project.py prompt` to inspect the frozen
original parts prompt, and `src/project.py serve` to display saved results.
Report generators are grouped in `src/reporting/`; early Part 1 prototypes are
preserved in `src/legacy/`. Experiment sources pinned by saved protocols remain
at their original paths. This organization does not change measured results or
automatically replace the distinct historical classifier entry points.

## Adopted scope and Result Analysis

The current working question is: **How do evidence inputs and decision rules
affect label accuracy and evidence-grounded verification in a local FEVER
fact-checker?** The user explicitly adopted this narrower scope on October 28,
2026, after reviewing results. It is retrospective framing, not a preregistered
hypothesis. Original proposals, reports and measurements remain preserved.

The [adopted scope](reports/result_analysis_20261028/ADOPTED_SCOPE.md) and
[tabular Result Analysis](reports/result_analysis_20261028/RESULT_ANALYSIS.md)
link the revised questions to measured evidence, uncertainty and negative
findings. The original generated-error representativeness question remains
unanswered and moves to future work pending independent human review.

## Current experiment status

The [latest Part 2 submission with the official benchmark](reports/part2/submission_official_20261023/MODEL_EXECUTION_PART2_UPDATED.md)
is available as `output/pdf/Model_Execution_Part_2_Official_20261023.pdf`, with a ZIP
and complete results appendix in `reports/part2/submission_official_20261023`.

The [full official FEVER development comparison](reports/part2/official_fever_dev_20261023/OFFICIAL_COMPARISON.md)
now evaluates all 19,998 official claims using the pinned, unmodified official scorer.
The frozen ensemble scored **61.70% label accuracy / 45.33% strict FEVER score**,
versus 58.78% accuracy for claim-only H2 and 52.37% / 35.07% for retrieved RoBERTa.
The table uses GEAR/KGAT's published **development** references on the same population;
their models were not rerun locally, and architecture/training/corpus resources differ.
No retraining or tuning on the official development set was performed. The earlier
71% result remains a separate 200-claim validation result. Earlier submission PDFs
are snapshots from before this full official-benchmark evaluation.

The [updated Part 2 submission](reports/part2/submission_updated_20261023_complete/MODEL_EXECUTION_PART2_UPDATED.md)
combines the measured study, frozen held-out comparison, and new development ensemble.
The PDF is at `output/pdf/Model_Execution_Part_2_Updated_20261023.pdf`; the updated ZIP
and complete results CSVs are in `reports/part2/submission_updated_20261023_complete`.
RQ1 independent human review remains incomplete; extra training seeds remain optional and unrun.

The [literature-informed code update](reports/part2/literature_improvements_20261023_corrected/LITERATURE_TO_CODE.md)
adds an experimental calibrated ensemble using existing checkpoints. On a separate
200-claim development assessment it reached 71% accuracy, versus 61% evidence-only
and 67% claim-only. The gain over claim-only remains uncertain. New sentence-ranking
variants did not improve evidence recall and were not selected. No retraining was run.
Try it from the repository root with
`.venv/bin/python src/verify_calibrated.py --claim "The Eiffel Tower is in Paris."`.
This command runs the experimental neural fallback; existing symbolic experiment
results and the original demo remain preserved.

The [latest Step 2 experiment update](reports/part2/step2_final_20261023/report/STEP2_RESULTS.md)
adds a frozen held-out sample, actual online symbolic verification timing, and a
separate two-model automated RQ1 pilot. The earlier submission documents are a
snapshot from before these additional experiments. RQ1 human validation remains pending.

Work continues on `experiment/part2-paired-ablation`; it has not been merged into main.
The [Part 2 submission results and selected configurations](reports/part2/submission_20261023_final/PART2_RESULTS.md)
consolidate training runs, paired baselines, top-1/5/10 ablations, a separate validation
sample, 360 local LLM judge requests, and controlled corpus growth. A generated
30-claim corpus and independent review sheets are available; its human labels remain pending.
See [the October 23 controlled experiment report](reports/part2/EXPERIMENT_STATUS_20261023.md)
for measured paired ablations, symbolic verifier v2, retrieval diagnostics, local timing,
and RQ1/RQ2 evaluation preparation. Symbolic v2 resolved 0 of 100 development claims;
the hybrid remained at 55% accuracy and 0.4163 macro-F1. No improvement is established.
Run new experiments with `.venv/bin/python` and distinct output paths. Existing results
and checkpoints must be preserved; expensive retraining requires approval.

The overview and next-steps sections below describe the original project plan;
the dated report records the current implementation and outstanding work.

### Project Overview

This project implements a fact-checking pipeline for verifying factual claims using the FEVER dataset. The planned system combines symbolic knowledge-base verification, information retrieval, and Transformer-based classification.

The final architecture is:

Claim → Symbolic Knowledge Base → BM25 Retrieval → RoBERTa → Classification

The classification labels are:

- SUPPORTS
- REFUTES
- NOT ENOUGH INFO

## Dataset

The project uses the FEVER fact-verification dataset.

The downloaded training dataset contains 145,449 examples:

- SUPPORTS: 80,035 (55.03%)
- REFUTES: 29,775 (20.47%)
- NOT ENOUGH INFO: 35,639 (24.50%)

Each example contains a claim, label, verifiability information, and evidence annotations.

## Preprocessing

The preprocessing pipeline is implemented in:

`src/preprocess.py`

It performs:

1. Dataset loading and JSON validation
2. Required-field validation
3. Label encoding
4. Class-distribution analysis
5. Group-aware train/validation/test splitting
6. Claim-leakage verification
7. Processed dataset generation

Because repeated claim text was discovered during exploratory analysis, identical claims are kept in the same partition to reduce data leakage.

The resulting split is:

| Split | Examples | Percentage |
|---|---:|---:|
| Training | 116,356 | 80% |
| Validation | 14,545 | 10% |
| Test | 14,548 | 10% |

The original exact-string claim-overlap check reported zero overlap. The later
code review found six case/whitespace-normalized claim groups shared between
train and validation, and six between train and test. Future preprocessing now
groups these variants together and refuses to overwrite existing split files.
The historical datasets and checkpoints have not been regenerated.

## Transformer Baseline

The baseline model uses `FacebookAI/roberta-base`.

The implementation is located in:

`src/legacy/baseline.py`

The baseline currently verifies that:

- the RoBERTa tokenizer loads successfully,
- FEVER claims can be tokenized,
- RoBERTa can process the tokenized input, and
- the classification head produces three output logits corresponding to the three FEVER classes.

## Training Configuration

Training initialization is implemented in:

`src/legacy/train.py`

Current configuration:

- Model: RoBERTa-base
- Number of classes: 3
- Maximum sequence length: 256
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 3
- Device support: Apple MPS, CUDA, or CPU

The current development machine successfully detects Apple MPS acceleration.

## Project Structure

fever-fact-checking/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── test_data.py
│   ├── preprocess.py
│   ├── baseline.py
│   └── train.py
├── results/
├── checkpoints/
├── requirements.txt
└── README.md

## Next Steps

The next implementation stages are:

1. Complete RoBERTa fine-tuning on the FEVER training split.
2. Evaluate the classifier using accuracy, precision, recall, and macro-F1.
3. Implement BM25 evidence retrieval.
4. Measure evidence retrieval using Recall@1, Recall@5, and Recall@10.
5. Integrate Wikidata as the symbolic knowledge-base stage.
6. Compare the Transformer-only, retrieval-augmented, LLM-as-judge, and hybrid approaches.
7. Measure latency and the percentage of claims resolved by the symbolic stage.

## Reproducibility

Install the required Python packages with:

pip install -r requirements.txt

Run preprocessing with:

python src/preprocess.py

Test the RoBERTa baseline with:

python src/legacy/baseline.py

Test training initialization with:

python src/legacy/train.py
