# Proposed research revision - discussion draft

**Archive status:** this discussion draft is superseded for the current analysis
by the [explicitly adopted October 28 scope](../result_analysis_20261028/ADOPTED_SCOPE.md).
The historical text below is preserved. The adopted questions remain a
retrospective revision, not original preregistered hypotheses.

This draft uses the professor's permission to narrow the project. It preserves the original proposal, hypotheses, literature workbook, measurements, and submission reports. These revisions were formulated after viewing existing results; they must not be presented as original, preregistered hypotheses. A negative result remains a valid finding.

## What the authors did and what this project did

The sources below are the fact-checking papers in `PraiseBen_LitReview.xlsx`, sheet `Literature Comparison`, rows 3-10. Row 2 is the supplied intrusion-detection example, not part of this comparison.

| Paper | Their approach | Similarity and difference in this project |
|---|---|---|
| [FEVER - Thorne et al., 2018](https://aclanthology.org/N18-1074/) | Created human-annotated claims and evidence; evaluated a retrieval-and-verification pipeline and oracle conditions. | We reuse FEVER claims and Wikipedia evidence, with RoBERTa and title-aware retrieval. We did not create FEVER or reproduce the official benchmark evaluation; our sampled internal splits differ. |
| [FactKG - Kim et al., 2023](https://aclanthology.org/2023.acl-long.895/) | Created a knowledge-graph verification dataset covering five reasoning types, including multi-hop and negation. | Both investigate structured knowledge. Our symbolic stage uses restricted Wikidata rules; we did not train or evaluate a FactKG reasoning model. |
| [GEAR - Zhou et al., 2019](https://aclanthology.org/P19-1085/) | Uses BERT representations and a learned evidence graph to pass information among sentences and aggregate evidence. | Both classify claims using retrieved sentences. Our evidence classifier concatenates sentences; it does not learn an evidence graph. Combining two classifiers' scores is a different operation. |
| [KGAT - Liu et al., 2020](https://aclanthology.org/2020.acl-main.655/) | Uses graph attention with node and edge kernels to identify relevant evidence and propagate information. | We test relevance using sentence-ranking ablations. Our lexical scoring and subject-page linking are not KGAT attention and did not improve recall in the new assessment. |
| [Hybrid fact-checking - Kolli et al., 2025](https://aclanthology.org/2025.winlp-main.19/) | Combines DBpedia lookup, prompted language-model classification, and search-agent fallback when coverage is insufficient. | Our architecture also tries structured lookup before a fallback, but uses Wikidata rules and a local Wikipedia/RoBERTa fallback. Our rules resolved zero evaluated claims, so no model calls were avoided. |
| [HoVer - Jiang et al., 2020](https://aclanthology.org/2020.findings-emnlp.309/) | Tests evidence retrieval and verification requiring up to four Wikipedia articles. | Both concern retrieval difficulty. We changed corpus size and evidence depth on FEVER; we did not evaluate HoVer or control the number of reasoning hops. More pages is not the same as more reasoning hops. |
| [FActScore - Min et al., 2023](https://aclanthology.org/2023.emnlp-main.741/) | Splits generated text into atomic facts, measures the fraction supported by sources, and evaluates automated scoring against human judgments. | We generated a small claim pilot and tested local judges against FEVER references. We have not completed independent reviews of generated claims, automatic long-form decomposition, or a FActScore evaluation. |
| [GraphCheck - Chen et al., 2025](https://arxiv.org/abs/2502.16514) | Extracts graphs from text, processes them with graph neural networks, and supplies a soft prompt to an LLM for verification. Its single inference call does not remove all preparation cost. | Both address evidence use and efficiency. We implemented neither graph extraction/GNN soft prompts nor GraphCheck. Our symbolic lookup and two-model ensemble are different designs. |

Example, not an executed benchmark claim: “The author of Book X was born in Country Y.” One evidence sentence may identify the author, while another gives their birthplace. GEAR/KGAT explicitly learn relationships among evidence sentences. Our current classifier receives those sentences together and must infer the relationship through its existing Transformer representation.

Do not compare our 71% development label accuracy with GEAR's 67.10% or KGAT's 70.38% FEVER scores. Their metric also requires appropriate evidence, and populations/protocols differ. Likewise, the hybrid paper's reported .93 F1 concerns the Supported/Refuted split; our principal classification task includes NEI.

Two workbook corrections to make separately: the hybrid paper is Kolli et al. at WiNLP, not Zhang et al. at EMNLP main; GEAR/KGAT do not inherently require the multiple LLM-call workflow described in the GraphCheck suggestion. They are trained evidence-graph models. The workbook itself has not been edited.

## Suggested narrower title

**Evidence Quality and Decision Calibration for FEVER Fact-Checking with RoBERTa**

Symbolic routing and local LLM judges become controlled comparison experiments. This title describes the measured core without suggesting that end-to-end LLM-generated-claim validation is finished.

## Four revised questions and hypotheses

| Revised question | Corresponding hypothesis to investigate | Existing evidence and remaining limit |
|---|---|---|
| RQ1. How do evidence relevance and evidence depth affect three-class FEVER verification? | Relevant retrieved evidence will yield higher macro-F1 than deliberately mismatched evidence; increasing depth alone will not necessarily improve performance. | Top-1/5/10, wrong-claim, empty, and oracle conditions are measured on shared claims. Oracle NEI inputs have a shortcut and are not an attainable system. |
| RQ2. Can validation-calibrated fusion of claim-only and evidence-conditioned RoBERTa improve macro-F1 and NEI recall over each model alone? | The selected fusion will improve both outcomes on a newly reserved evaluation sample. | Existing development assessment: F1 .6655 versus .6262 claim-only and .4471 evidence-only; NEI recall 43.1% versus 36.2% and 0%. This is an exploratory, result-informed hypothesis; its new-sample prediction remains untested. |
| RQ3. How do evidence availability and prompt wording affect a local LLM judge's accuracy and repeat agreement against human FEVER labels? | Retrieved-evidence prompts will have higher macro-F1 than empty-evidence prompts; high repeat agreement alone will not ensure high accuracy. | 360 requests on 30 unique claims cover two evidence conditions, two prompts, and three inference repeats. Findings apply to this model/pilot, not all LLM judges. |
| RQ4. Does a symbolic-first route reduce model calls and end-to-end latency, and how does performance change with evidence-corpus size? | Symbolic-first routing will reduce model calls and mean latency without reducing accuracy on matched inputs. | Existing coverage is zero, so the efficiency prediction is unsupported. Corpus-growth and online-latency experiments have cache/network/design limits. Full online corpus-size-by-route replication remains unrun. |

RQ1 and RQ2 above are revised numbering. Original RQ2 maps to revised RQ3. Original symbolic/comparison and efficiency questions map mainly to revised RQ4, with the model comparison extended by revised RQ2. Original RQ1, about whether FEVER represents LLM error types, moves to future work; it was not answered by changing its wording.

## How to explain the contribution

“I evaluated how evidence inputs and decision rules affect a practical RoBERTa fact-checker. I compared conditions on shared claims, identified an empty-evidence NEI shortcut opportunity, tested a calibrated ensemble, and measured whether a symbolic stage actually avoided work. The ensemble showed promising development results; the symbolic stage showed no efficiency benefit in the measured samples.”

This is an empirical evaluation contribution. It does not claim a new graph architecture, reproduction of the reviewed models, superiority over those papers, or completed human validation of generated claims. The drafted questions can be adopted after discussion; official proposal and submission documents remain unchanged.
