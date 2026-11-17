import argparse,csv,json,statistics
from pathlib import Path
base=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description='Render the source-linked Part 2 results Markdown from assembled tables.')
parser.add_argument('--tables', default='reports/part2/submission_20261023_final')
parser.add_argument('--output', required=True)
args=parser.parse_args()
out=Path(args.tables)
def csvrows(name):
 with (out/name).open(newline='') as f:return list(csv.DictReader(f))
def f(value):return f'{float(value):.4f}'
def table(headers,rows):
 return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
evaluations=csvrows('all_classifier_evaluations.csv')
by={(r['scope'],r['condition']):r for r in evaluations}
def evalrow(scope,condition,label=None):
 r=by[(scope,condition)]
 return [label or condition,r['n'],f(r['accuracy']),f(r['macro_precision']),f(r['macro_recall']),f(r['macro_f1'])]
first=[evalrow('validation_first100_reference','unweighted_reference','Original unweighted reference'),
 evalrow('validation_first100_reference','weighted_reference_epoch2','Earlier weighted epoch 2'),
 evalrow('validation_first100_saved','claim_only_H2','**Selected claim-only H2**')]
for condition in ['retrieved_k1','retrieved_k5','retrieved_k10','empty_evidence','wrong_claim_evidence','gold_evidence_oracle']:
 first.append(evalrow('validation_first100_topk',condition,'**Selected retrieval: top-5**' if condition=='retrieved_k5' else condition))
for condition in ['v1','v2']:first.append(evalrow('validation_first100_symbolic',condition,'Hybrid '+condition))
extension=[evalrow('validation_extension100_reference','unweighted_reference','Original unweighted reference'),
 evalrow('validation_extension100_reference','weighted_reference_epoch2','Earlier weighted epoch 2')]
for condition in ['claim_only_H2','retrieved_k1','retrieved_k5','retrieved_k10','empty_evidence','wrong_claim_evidence']:
 extension.append(evalrow('validation_extension100_disjoint_text',condition))
local=json.loads((base/'reports/part2/local_llm_20261023/analysis_blinded/summary.json').read_text())
judge=[]
for name,s in local['judge_condition_summaries'].items():
 judge.append([name,f"{s['mean_accuracy']:.4f} ± {s['sample_std_accuracy_across_repeats']:.4f}",f"{s['mean_macro_f1']:.4f} ± {s['sample_std_macro_f1_across_repeats']:.4f}",f(s['all_repeats_agree_fraction'])])
training=csvrows('all_training_runs.csv')
configrows=[]
for name in ['H1','H2','H3','H4']:
 row=next(r for r in training if r['run']==name+'_seed42' and r['epoch']=='2')
 configrows.append(['**H2 — selected**' if name=='H2' else name,row['learning_rate'],row['weight_decay'],row['batch_size'],row['epoch'],f(row['accuracy']),f(row['macro_f1'])])
recalls=json.loads((base/'reports/part2/corpus_growth_20261023/evaluation_recall_corrected/recall.json').read_text())
timings=json.loads((base/'reports/part2/corpus_growth_20261023/evaluation/timing/summary.json').read_text())['results']
corpus=[]
for r in recalls:
 b=[x for x in timings if x['db']==r['db'] and x['condition']=='baseline']
 h=[x for x in timings if x['db']==r['db'] and x['condition']=='cached_symbolic']
 corpus.append([f"{r['pages']:,}",f"{r['sentences']:,}",f(r['recall'][1]['complete_recall']),f(b[0]['accuracy']),f(b[0]['macro_f1']),f"{statistics.mean(x['mean_seconds'] for x in b)*1000:.2f}",f"{statistics.mean(x['mean_seconds'] for x in h)*1000:.2f}"])
content='''# Model Execution — Part 2: results and configuration choices

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

'''+table(['Configuration','Learning rate','Weight decay','Batch','Best epoch','Accuracy','Macro-F1'],configrows)+'''
**Choice:** H2 has the highest macro-F1 and accuracy among H1–H4. Its lower learning rate is therefore selected on measured validation performance, not an assumed benefit. The original weighted run reached accuracy 0.6325 / macro-F1 0.5982 at epoch 2; the original unweighted log recorded 0.5465 / 0.2356. Those historical runs differ in sampling/provenance, so they do not by themselves isolate the effect of tuning. Two identical epoch-1 entries in the original log are retained as observations, not presented as independent seed repeats.

**Evidence-model choice:** epoch 1 achieved gold-evidence validation accuracy 0.9415 and macro-F1 0.9335 on 2,000 examples, versus epoch 2 at 0.9365 / 0.9313. Epoch 1 is retained by macro-F1. Gold-evidence performance does not establish retrieved-evidence performance; empty gold evidence for NEI can expose label-related information.

All 14 historical training-log rows are in [all_training_runs.csv](all_training_runs.csv). Missing legacy precision, recall, seed, and run-manifest information is left unavailable rather than reconstructed without evidence.

## 2. Paired baseline, main-model and ablation results

The table uses the identical original 100 validation IDs. All fresh model evaluations use frozen checkpoints, seed 42, batch size 8, and maximum token length 256. P/R/F1 below are macro averages across the three FEVER classes. Earlier unweighted and weighted checkpoints were reevaluated on this same sample so they can be compared within this table.

'''+table(['Condition','N','Accuracy','Macro precision','Macro recall','Macro-F1'],first)+'''
Top-1, top-5, and top-10 share the same ranked retrieval and model checkpoint; only evidence depth changes. Fresh top-5 evidence exactly matches the saved artifact. Top-10 adds no complete evidence-group hits on this sample, ties top-5 accuracy, and has slightly lower macro-F1. Top-5 remains the frozen reference rather than claiming a universally optimal depth. Empty, shifted wrong-claim, and oracle inputs are diagnostics, not independently trained models. The oracle can expose NEI through empty gold evidence and is not an achievable end-to-end baseline.

The retrieved-minus-H2 accuracy difference is -0.07; paired bootstrap 95% CI [-0.17, +0.04], exact McNemar p=0.2649. These are exploratory unadjusted statistics. Different training/checkpoints confound the claim-only versus evidence-model comparison; within-checkpoint evidence ablations provide the tighter control.

**Symbolic result:** v1 and v2 both resolved 0/100. Resolved accuracy is undefined, and zero calls were avoided. The saved Wikidata snapshot reproduced all 200 symbolic condition rows offline except timings. V2 is retained for analysis, not selected as an improved system.

## 3. Separate validation extension and Recall@10

The next 100 eligible validation claims were selected by scanning from raw offset 100 while excluding prior IDs, casefolded claim text, and within-slice duplicates. No labels were used to choose or exclude claims. This slice is disjoint from the first sample by ID and normalized text, but it is still validation data from the tuning pool, not an independent held-out test.

'''+table(['Condition','N','Accuracy','Macro precision','Macro recall','Macro-F1'],extension)+'''
The extension had oracle evidence for only 99/100 IDs; the oracle condition was omitted for the entire slice so no condition silently used a smaller population. The failed preflight output is retained and marked unscored.

Top-5 versus H2 has an accuracy difference of -0.13 (paired bootstrap 95% CI [-0.22, -0.04], unadjusted exact McNemar p=0.0106). Top-10 versus top-5 changes 3 errors into correct predictions, but its exact paired p=0.25; this small development sample does not establish a general improvement. H2 remains selected using its original 2,000-example tuning criterion even though the earlier weighted checkpoint has slightly higher macro-F1 on these small slices.

'''
retrievalrows=[]
for name,scope in [('evidence_depth_20261023','Original 100'),('validation_extension_20261023_complete','Extension 100')]:
 for row in json.loads((base/'reports/part2'/name/'summary.json').read_text())['retrieval']:
  retrievalrows.append([scope,row['k'],row['eligible'],f(row['any_recall']),f(row['complete_recall'])])
content+=table(['Sample','K','Verifiable N','Any-sentence recall','Complete-group recall'],retrievalrows)+'''
Recall is computed only over verifiable claims with annotated evidence, using either any matching sentence or at least one complete evidence group. NEI claims are excluded from recall denominators.

## 4. RQ2: local LLM judge pilot

A balanced pilot takes the first 10 claims of each FEVER class from the original sample. Each of the same 30 claims receives two evidence conditions × two prompt wordings × three independently requested seed settings: **360 actual local requests**, all parsed successfully. No gold labels or oracle evidence were supplied. Each request begins with a fresh message context; execution order was shuffled with seed 42. Temperature is 0.2, repeat seeds 42/43/44, context 4096 and output limit 32 tokens.

The already installed **Llama 3.2 3.2B Q4_K_M** model was chosen for local availability, not by its measured score. Exact digest: `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`. Settings, server version, prompts, token counters, timestamps, durations, raw outputs and parsed labels are saved. No paid API, model download or retraining was needed. The request format follows the [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md).

Values below are mean ± sample SD across three inference repeats; these remain 30 unique claims, not 90 independent examples.

'''+table(['Evidence : prompt','Accuracy mean ± SD','Macro-F1 mean ± SD','All 3 repeats agree'],judge)+'''
On these **same 30 IDs**, claim-only H2 scores 0.5000 accuracy / 0.4818 macro-F1, and retrieved-evidence RoBERTa scores 0.5333 / 0.4274. Llama's evidence-first condition scores 0.4000 / 0.3850 on average. This pilot supports a descriptive comparison with FEVER human ground truth; it does not establish that all LLM judges underperform Transformers.

Retrieved-evidence prompt agreement is 0.7889, Cohen's kappa 0.6821. For each retrieved prompt, all three repeats agree on 25/30 claims. Empty-evidence requests consistently return NEI: accuracy 1/3 on the balanced dataset is expected for this **evidence-constrained abstention control**, not a closed-book factual-knowledge test. High consistency alone does not imply factual correctness.

Every judge repeat is in [all_judge_runs.csv](all_judge_runs.csv); the balanced sample must not be ranked directly against the 100- or 2,000-example tables. Paired exploratory comparisons are in the local pilot analysis.

## 5. RQ1: generated claims and annotation materials

The local LLM generated **30 distinct claims**, with model digest, topic prompt, seed 1000–1029, raw response and timing recorded. Topics come from retrieved page titles without labels or FEVER claim text, so this is a retrieval-seeded pilot rather than a natural-use sample. Generation temperature was 0.7 and output limit 128 tokens. Original outputs, including apparently incomplete or multi-assertion sentences, were retained.

'''+table(['Descriptive property','FEVER balanced 30','Generated 30'],[
 ['Mean words',f(local['fever_profile']['mean_words']),f(local['generated_profile']['mean_words'])],
 ['Median words',f(local['fever_profile']['median_words']),f(local['generated_profile']['median_words'])],
 ['Contains digits',f(local['fever_profile']['digit_fraction']),f(local['generated_profile']['digit_fraction'])],
 ['Contains an explicit year',f(local['fever_profile']['explicit_year_fraction']),f(local['generated_profile']['explicit_year_fraction'])]])+'''
These differences describe this small, prompt-dependent sample; they do not establish representativeness of factual errors. **RQ1 is not fully answered:** generated claims still need independent truth, atomicity and error-type annotations. There are no invented human labels.

Two source-blinded sheets each contain the same shuffled 60 claims (30 FEVER + 30 generated), with neutral review IDs. Give reviewers only the sheets, review instructions and blinded candidate-evidence pack; keep the source key and existing FEVER labels with the analysis code. Reviewers may need additional sources because retrieval is neither exhaustive nor ground truth. [Review materials](../local_llm_20261023/analysis_blinded/human_review_A.csv) and the matching reviewer-B CSV are ready.

## 6. RQ4: controlled corpus growth and efficiency

Nested indexes contain a common 923-page core selected only by claim-text title search, plus seeded random distractor pages. No gold labels/evidence select the core. Each larger subset includes the smaller subset; the full original index is opened read-only. This query-aware design deliberately retains the candidate-page core and tests distractor/index growth, not a representative random-corpus availability curve. Full-index title statistics and candidate ranking may still change with size.

Every size uses the same 100 claims, top-5 title-aware retrieval, frozen evidence checkpoint, length 256 and batch size 1. Two repeats alternate baseline/cached-route order. Runs use CPU and warm caches, with model loading excluded. Latencies are averages across 200 timed claims per row.

'''+table(['Pages','Sentences','Complete R@5','Accuracy','Macro-F1','Baseline ms/claim','Cached-route ms/claim'],corpus)+'''
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

'''
choices=csvrows('configuration_choices.csv')
content+=table(['Model/component','Chosen reporting configuration','Justification'],[
 ['Original unweighted baseline','Existing epoch-1 checkpoint, reference only','Reevaluated on identical IDs; historical majority-class failure is retained transparently.'],
 ['Claim-only RoBERTa','**H2, epoch 2; LR 1e-5; WD 0.01; batch 8; seed 42**','Highest macro-F1 and accuracy among the four controlled tuning configurations.'],
 ['Evidence RoBERTa','**Epoch 1; LR 2e-5; WD 0.01; batch 8; seed 42**','Higher gold-validation macro-F1 than epoch 2; retrieved transfer is reported separately.'],
 ['Retrieval condition','**Title-aware top-5**, frozen reference','Retains the original paired condition; no universal depth optimum is established.'],
 ['Hybrid','V1 and V2 are reported variants; **no improved hybrid selected**','Zero symbolic coverage and no avoided calls in this sample.'],
 ['LLM judge','Local Llama 3.2 quantized pilot comparator','Availability-based choice; not a performance-selected final judge.']])
content+='''
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

.venv/bin/python src/experiment_evidence_topk.py \\
  --reference data/processed/retrieved/validation_first_100_top5_title_fast.jsonl \\
  --output reports/part2/topk_NEW

.venv/bin/python src/experiment_evidence_topk.py --offset 100 \\
  --exclude data/processed/retrieved/validation_first_100_top5_title_fast.jsonl \\
  --output reports/part2/extension_NEW

.venv/bin/python src/local_llm_experiments.py pilot --output reports/part2/local_NEW
.venv/bin/python src/local_llm_experiments.py judge --output reports/part2/local_NEW
.venv/bin/python src/local_llm_experiments.py generate --output reports/part2/local_NEW

.venv/bin/python src/build_controlled_corpora.py \\
  --output data/processed/corpora_NEW \\
  --manifest reports/part2/corpora_NEW/index_manifest.json
.venv/bin/python src/experiment_corpus_growth.py \\
  --indexes reports/part2/corpora_NEW/index_manifest.json \\
  --output reports/part2/corpora_NEW/evaluation

.venv/bin/python src/assemble_part2_results.py --output reports/part2/submission_NEW
```

The assembler uses the dated source paths documented in its code; rerunning experiments into different paths requires updating those source paths before assembling a different study. The original dated artifacts remain preserved.
'''
with Path(args.output).open('x') as stream:stream.write(content)
print('Wrote',args.output)
