# Model Execution — Part 2: results and configuration choices

COSC 874 Deep Learning II · October 23, 2026  
Branch: `experiment/part2-paired-ablation`

The selected development configurations are **H2 epoch 2 for claim-only RoBERTa**, **epoch 1 for evidence-conditioned RoBERTa**, and **title-aware top-5 retrieval as the frozen reference condition**. Symbolic v2 remains an experimental variant: it resolved no claims in the first-100 sample and demonstrated no accuracy or efficiency benefit. No model was retrained during this continuation, and nothing was merged into main.

These selections identify the configurations used for reporting and later testing. They are not held-out final-test results. Training/validation populations and evidence conditions are kept separate below.

## Coverage of the assignment requirements

| Requirement | Status | Evidence or remaining work |
| --- | --- | --- |
| 1. Tune hyperparameters | Complete for this development study | Four controlled configurations; H2 selected using validation macro-F1 and accuracy. |
| 2. Run research-question experiments | Substantially complete; RQ1 remains partial | Paired model/component ablations, repeated LLM judge pilot, and corpus-growth experiments are measured. Generated-claim truth/error annotations still require independent human review and adjudication. |
| 3. Keep runs reproducible | Seeds and saved artifacts complete; optional training repeats not run | Seeds, predictions, raw judge responses, snapshots, manifests, and index membership are retained. Judge/inference repeats are not presented as repeated Transformer training runs. |
| 4. Collect selected results | Complete for the measured development experiments | Consolidated tables below and accompanying CSVs identify reporting configurations and justify choices. No held-out final-test result is claimed. |

## 1. Hyperparameter tuning

RoBERTa-base architecture and token length 256 were held fixed. The four configurations test learning rates around the initial 2e-5 setting, plus removing weight decay. Every configuration uses batch size 8, two epochs, seed 42, a fixed 20% training subset, and the same first 2,000 validation claims. This is a small, reasoned search rather than an exhaustive sweep. Checkpoint selection uses validation macro-F1, with accuracy as a tie-breaker.

| Configuration | Learning rate | Weight decay | Batch | Best epoch | Accuracy | Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | 2e-05 | 0.01 | 8 | 2 | 0.6465 | 0.6103 |
| **H2 — selected** | 1e-05 | 0.01 | 8 | 2 | 0.6775 | 0.6202 |
| H3 | 3e-05 | 0.01 | 8 | 2 | 0.6370 | 0.5952 |
| H4 | 2e-05 | 0.0 | 8 | 2 | 0.6575 | 0.6165 |

**Choice:** H2 has the highest macro-F1 and accuracy among H1–H4. Its lower learning rate is therefore selected on measured validation performance, not an assumed benefit. The original weighted run reached accuracy 0.6325 / macro-F1 0.5982 at epoch 2; the original unweighted log recorded 0.5465 / 0.2356. Those historical runs differ in sampling/provenance, so they do not by themselves isolate the effect of tuning. Two identical epoch-1 entries in the original log are retained as observations, not presented as independent seed repeats.

**Evidence-model choice:** epoch 1 achieved gold-evidence validation accuracy 0.9415 and macro-F1 0.9335 on 2,000 examples, versus epoch 2 at 0.9365 / 0.9313. Epoch 1 is retained by macro-F1. Gold-evidence performance does not establish retrieved-evidence performance; empty gold evidence for NEI can expose label-related information.

All 14 historical training-log rows are in [all_training_runs.csv](all_training_runs.csv). Missing legacy precision, recall, seed, and run-manifest information is left unavailable rather than reconstructed without evidence.

## 2. Paired baseline, main-model and ablation results

The table uses the identical original 100 validation IDs. All fresh model evaluations use frozen checkpoints, seed 42, batch size 8, and maximum token length 256. P/R/F1 below are macro averages across the three FEVER classes. Earlier unweighted and weighted checkpoints were reevaluated on this same sample so they can be compared within this table.

| Condition | N | Accuracy | Macro precision | Macro recall | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| Original unweighted reference | 100 | 0.4700 | 0.1567 | 0.3333 | 0.2132 |
| Earlier weighted epoch 2 | 100 | 0.6000 | 0.5826 | 0.5895 | 0.5766 |
| **Selected claim-only H2** | 100 | 0.6200 | 0.6121 | 0.5721 | 0.5761 |
| retrieved_k1 | 100 | 0.5100 | 0.3303 | 0.4772 | 0.3876 |
| **Selected retrieval: top-5** | 100 | 0.5500 | 0.3559 | 0.5124 | 0.4163 |
| retrieved_k10 | 100 | 0.5500 | 0.3534 | 0.5056 | 0.4152 |
| empty_evidence | 100 | 0.4500 | 0.5140 | 0.4796 | 0.4513 |
| wrong_claim_evidence | 100 | 0.3700 | 0.2772 | 0.3848 | 0.2897 |
| gold_evidence_oracle | 100 | 0.8400 | 0.8639 | 0.8173 | 0.8315 |
| Hybrid v1 | 100 | 0.5500 | 0.3559 | 0.5124 | 0.4163 |
| Hybrid v2 | 100 | 0.5500 | 0.3559 | 0.5124 | 0.4163 |

Top-1, top-5, and top-10 share the same ranked retrieval and model checkpoint; only evidence depth changes. Fresh top-5 evidence exactly matches the saved artifact. Top-10 adds no complete evidence-group hits on this sample, ties top-5 accuracy, and has slightly lower macro-F1. Top-5 remains the frozen reference rather than claiming a universally optimal depth. Empty, shifted wrong-claim, and oracle inputs are diagnostics, not independently trained models. The oracle can expose NEI through empty gold evidence and is not an achievable end-to-end baseline.

The retrieved-minus-H2 accuracy difference is -0.07; paired bootstrap 95% CI [-0.17, +0.04], exact McNemar p=0.2649. These are exploratory unadjusted statistics. Different training/checkpoints confound the claim-only versus evidence-model comparison; within-checkpoint evidence ablations provide the tighter control.

**Symbolic result:** v1 and v2 both resolved 0/100. Resolved accuracy is undefined, and zero calls were avoided. The saved Wikidata snapshot reproduced all 200 symbolic condition rows offline except timings. V2 is retained for analysis, not selected as an improved system.

## 3. Separate validation extension and Recall@10

The next 100 eligible validation claims were selected by scanning from raw offset 100 while excluding prior IDs, casefolded claim text, and within-slice duplicates. No labels were used to choose or exclude claims. This slice is disjoint from the first sample by ID and normalized text, but it is still validation data from the tuning pool, not an independent held-out test.

| Condition | N | Accuracy | Macro precision | Macro recall | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| Original unweighted reference | 100 | 0.5700 | 0.1900 | 0.3333 | 0.2420 |
| Earlier weighted epoch 2 | 100 | 0.6600 | 0.5954 | 0.6422 | 0.5851 |
| claim_only_H2 | 100 | 0.7000 | 0.6063 | 0.5831 | 0.5788 |
| retrieved_k1 | 100 | 0.5700 | 0.3259 | 0.5532 | 0.3868 |
| retrieved_k5 | 100 | 0.5700 | 0.3067 | 0.4982 | 0.3707 |
| retrieved_k10 | 100 | 0.6000 | 0.3250 | 0.5158 | 0.3961 |
| empty_evidence | 100 | 0.4600 | 0.4385 | 0.4598 | 0.4286 |
| wrong_claim_evidence | 100 | 0.3300 | 0.2556 | 0.3854 | 0.2454 |

The extension had oracle evidence for only 99/100 IDs; the oracle condition was omitted for the entire slice so no condition silently used a smaller population. The failed preflight output is retained and marked unscored.

Top-5 versus H2 has an accuracy difference of -0.13 (paired bootstrap 95% CI [-0.22, -0.04], unadjusted exact McNemar p=0.0106). Top-10 versus top-5 changes 3 errors into correct predictions, but its exact paired p=0.25; this small development sample does not establish a general improvement. H2 remains selected using its original 2,000-example tuning criterion even though the earlier weighted checkpoint has slightly higher macro-F1 on these small slices.

| Sample | K | Verifiable N | Any-sentence recall | Complete-group recall |
| --- | --- | --- | --- | --- |
| Original 100 | 1 | 71 | 0.4930 | 0.4507 |
| Original 100 | 5 | 71 | 0.6479 | 0.6197 |
| Original 100 | 10 | 71 | 0.6479 | 0.6197 |
| Extension 100 | 1 | 67 | 0.5672 | 0.5224 |
| Extension 100 | 5 | 67 | 0.7164 | 0.6716 |
| Extension 100 | 10 | 67 | 0.7313 | 0.6866 |

Recall is computed only over verifiable claims with annotated evidence, using either any matching sentence or at least one complete evidence group. NEI claims are excluded from recall denominators.

## 4. RQ2: local LLM judge pilot

A balanced pilot takes the first 10 claims of each FEVER class from the original sample. Each of the same 30 claims receives two evidence conditions × two prompt wordings × three independently requested seed settings: **360 actual local requests**, all parsed successfully. No gold labels or oracle evidence were supplied. Each request begins with a fresh message context; execution order was shuffled with seed 42. Temperature is 0.2, repeat seeds 42/43/44, context 4096 and output limit 32 tokens.

The already installed **Llama 3.2 3.2B Q4_K_M** model was chosen for local availability, not by its measured score. Exact digest: `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`. Settings, server version, prompts, token counters, timestamps, durations, raw outputs and parsed labels are saved. No paid API, model download or retraining was needed. The request format follows the [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md).

Values below are mean ± sample SD across three inference repeats; these remain 30 unique claims, not 90 independent examples.

| Evidence : prompt | Accuracy mean ± SD | Macro-F1 mean ± SD | All 3 repeats agree |
| --- | --- | --- | --- |
| retrieved:direct | 0.3778 ± 0.0192 | 0.3655 ± 0.0220 | 0.8333 |
| retrieved:evidence_first | 0.4000 ± 0.0000 | 0.3850 ± 0.0091 | 0.8333 |
| empty:direct | 0.3333 ± 0.0000 | 0.1667 ± 0.0000 | 1.0000 |
| empty:evidence_first | 0.3333 ± 0.0000 | 0.1667 ± 0.0000 | 1.0000 |

On these **same 30 IDs**, claim-only H2 scores 0.5000 accuracy / 0.4818 macro-F1, and retrieved-evidence RoBERTa scores 0.5333 / 0.4274. Llama's evidence-first condition scores 0.4000 / 0.3850 on average. This pilot supports a descriptive comparison with FEVER human ground truth; it does not establish that all LLM judges underperform Transformers.

Retrieved-evidence prompt agreement is 0.7889, Cohen's kappa 0.6821. For each retrieved prompt, all three repeats agree on 25/30 claims. Empty-evidence requests consistently return NEI: accuracy 1/3 on the balanced dataset is expected for this **evidence-constrained abstention control**, not a closed-book factual-knowledge test. High consistency alone does not imply factual correctness.

Every judge repeat is in [all_judge_runs.csv](all_judge_runs.csv); the balanced sample must not be ranked directly against the 100- or 2,000-example tables. Paired exploratory comparisons are in the local pilot analysis.

## 5. RQ1: generated claims and annotation materials

The local LLM generated **30 distinct claims**, with model digest, topic prompt, seed 1000–1029, raw response and timing recorded. Topics come from retrieved page titles without labels or FEVER claim text, so this is a retrieval-seeded pilot rather than a natural-use sample. Generation temperature was 0.7 and output limit 128 tokens. Original outputs, including apparently incomplete or multi-assertion sentences, were retained.

| Descriptive property | FEVER balanced 30 | Generated 30 |
| --- | --- | --- |
| Mean words | 8.7000 | 22.8667 |
| Median words | 8.0000 | 24.5000 |
| Contains digits | 0.1333 | 0.6667 |
| Contains an explicit year | 0.1000 | 0.6333 |

These differences describe this small, prompt-dependent sample; they do not establish representativeness of factual errors. **RQ1 is not fully answered:** generated claims still need independent truth, atomicity and error-type annotations. There are no invented human labels.

Two source-blinded sheets each contain the same shuffled 60 claims (30 FEVER + 30 generated), with neutral review IDs. Give reviewers only the sheets, review instructions and blinded candidate-evidence pack; keep the source key and existing FEVER labels with the analysis code. Reviewers may need additional sources because retrieval is neither exhaustive nor ground truth. [Review materials](../local_llm_20261023/analysis_blinded/human_review_A.csv), the matching reviewer-B CSV, [instructions](../local_llm_20261023/analysis_blinded/REVIEW_INSTRUCTIONS.md), and `src/score_rq1_annotations.py` are ready.

## 6. RQ4: controlled corpus growth and efficiency

Nested indexes contain a common 923-page core selected only by claim-text title search, plus seeded random distractor pages. No gold labels/evidence select the core. Each larger subset includes the smaller subset; the full original index is opened read-only. This query-aware design deliberately retains the candidate-page core and tests distractor/index growth, not a representative random-corpus availability curve. Full-index title statistics and candidate ranking may still change with size.

Every size uses the same 100 claims, top-5 title-aware retrieval, frozen evidence checkpoint, length 256 and batch size 1. Two repeats alternate baseline/cached-route order. Runs use CPU and warm caches, with model loading excluded. Latencies are averages across 200 timed claims per row.

| Pages | Sentences | Complete R@5 | Accuracy | Macro-F1 | Baseline ms/claim | Cached-route ms/claim |
| --- | --- | --- | --- | --- | --- | --- |
| 2,000 | 10,835 | 0.6056 | 0.5400 | 0.4071 | 61.94 | 67.69 |
| 10,000 | 50,124 | 0.6056 | 0.5500 | 0.4134 | 58.67 | 57.68 |
| 50,000 | 239,351 | 0.6197 | 0.5400 | 0.4066 | 64.80 | 58.00 |
| 5,396,106 | 25,247,888 | 0.6197 | 0.5500 | 0.4163 | 106.26 | 113.21 |

All corpus sizes make 100 retrieval and 100 Transformer calls in each condition and repeat because symbolic coverage is zero. Cached routing has no demonstrated computational saving. The full-index pipeline is slower than the smaller-index conditions in this run, but small-index latency is non-monotonic, the size order is fixed, and OS cache/thermal effects remain. No strong scaling law is inferred from four sizes and two repeats.

The route replays saved decisions; it excludes live Wikidata and symbolic-rule execution. Consequently this experiment does not establish live hybrid latency or monetary cost savings. No monetary estimate is fabricated. The original unrelated sentence-FTS structures make full-database byte size unsuitable for a direct storage-efficiency comparison with the minimal nested indexes.

The first corpus-recall export omitted gold references and reported no eligible rows. It is preserved but **superseded** by `evaluation_recall_corrected/recall.json`, which joins and validates gold claim IDs/text/labels. A regression test now rejects missing gold annotations. Classification and timing outputs were unaffected and were not rerun for that correction.

## 7. Reproducibility, selected configurations and remaining work

- All Python execution uses the existing `.venv`. No package installation was needed.
- Training selection uses seed 42; frozen evaluations also use seed 42. Judge repeat seeds are 42/43/44. Index sampling and request-order shuffle use seed 42.
- Predictions, raw responses, manifests, software versions where recorded, index membership lists, exact model digest, and input hashes support audit. Files refuse to overwrite existing outputs. The tuning runner now also refuses an existing seed/run path.
- Seventeen tests pass, including duplicate-text exclusion, nested corpus membership, recall annotations, independent judge request context and the earlier symbolic/scoring tests.
- Transformer training was **not** repeated with additional seeds. Inference repeats, epochs and duplicate legacy log entries are not relabeled as independent training runs. Two more H2 training runs would require approval; the course rubric makes these optional when compute permits.
- No held-out test claims were evaluated. Final testing should follow frozen choices and any required human annotation/adjudication.

| Model/component | Chosen reporting configuration | Justification |
| --- | --- | --- |
| Original unweighted baseline | Existing epoch-1 checkpoint, reference only | Reevaluated on identical IDs; historical majority-class failure is retained transparently. |
| Claim-only RoBERTa | **H2, epoch 2; LR 1e-5; WD 0.01; batch 8; seed 42** | Highest macro-F1 and accuracy among the four controlled tuning configurations. |
| Evidence RoBERTa | **Epoch 1; LR 2e-5; WD 0.01; batch 8; seed 42** | Higher gold-validation macro-F1 than epoch 2; retrieved transfer is reported separately. |
| Retrieval condition | **Title-aware top-5**, frozen reference | Retains the original paired condition; no universal depth optimum is established. |
| Hybrid | V1 and V2 are reported variants; **no improved hybrid selected** | Zero symbolic coverage and no avoided calls in this sample. |
| LLM judge | Local Llama 3.2 quantized pilot comparator | Availability-based choice; not a performance-selected final judge. |

The rubric's consolidated results and configuration rationale are assembled. RQ2 now has actual repeated measurements, and RQ4 has a bounded corpus-growth experiment. RQ1 still requires two independent human reviews and adjudication before any claim about error representativeness. Additional training seeds are optional and unrun; neither the final test nor a live-network hybrid speedup has been established.

### Submission files

- [All training configurations and epochs](all_training_runs.csv)
- [All classifier evaluations, separated by sample](all_classifier_evaluations.csv)
- [Every judge condition and repeat](all_judge_runs.csv)
- [Retrieval Recall@1/@5/@10 and corpus variants](all_retrieval_runs.csv)
- [Individual latency trials](all_latency_runs.csv)
- [Configuration choices and rationale](configuration_choices.csv)
- [Source hashes and limitations](manifest.json)

### Reproduce with new output paths

```sh
.venv/bin/python -m unittest discover -s tests -v

.venv/bin/python src/experiment_evidence_topk.py \
  --reference data/processed/retrieved/validation_first_100_top5_title_fast.jsonl \
  --output reports/part2/topk_NEW

.venv/bin/python src/experiment_evidence_topk.py --offset 100 \
  --exclude data/processed/retrieved/validation_first_100_top5_title_fast.jsonl \
  --output reports/part2/extension_NEW

.venv/bin/python src/local_llm_experiments.py pilot --output reports/part2/local_NEW
.venv/bin/python src/local_llm_experiments.py judge --output reports/part2/local_NEW
.venv/bin/python src/local_llm_experiments.py generate --output reports/part2/local_NEW

.venv/bin/python src/build_controlled_corpora.py \
  --output data/processed/corpora_NEW \
  --manifest reports/part2/corpora_NEW/index_manifest.json
.venv/bin/python src/experiment_corpus_growth.py \
  --indexes reports/part2/corpora_NEW/index_manifest.json \
  --output reports/part2/corpora_NEW/evaluation

.venv/bin/python src/assemble_part2_results.py --output reports/part2/submission_NEW
```

The assembler uses the dated source paths documented in its code; rerunning experiments into different paths requires updating those source paths before assembling a different study. The original dated artifacts remain preserved.
