# Pipeline input audit — after stopping full Llama evaluation

## Operational status

The user requested a stop before architecture changes. The batch supervisor and inference client were stopped. A_original has 8,232 complete saved responses (one INVALID); D_both never started. No response file was deleted or truncated. USER_STOP.json in the run folder records the stop. Do not resume without further user authorization.

## Verified answer

Evidence RoBERTa and Ollama already consume actual Wikipedia sentence text. Article titles are used for candidate-page retrieval; they are not the only classification input. The current pipeline selects five sentences across up to ten candidate pages. It does not select five full articles for URL scraping, does not ingest their complete current contents, and does not use an embedding/vector store. The earlier article-passage prototype also uses the local snapshot and does not implement the requested full-URL ingestion architecture.

| Stage | Verified current implementation | Source |
|---|---|---|
| Candidate retrieval | SQLite FTS5/BM25 over article titles; up to ten pages | src/retrieve_title_aware.py:24 |
| Sentence ranking | Read page_sentences.text; score word overlap plus a small title-rank bonus; keep five sentences | src/retrieve_title_aware.py:54 |
| Evidence model input | Join sentence text; tokenize claim paired with that text | src/evidence_pipeline_v2.py:64; src/verify_calibrated.py:58 |
| Claim-only branch | Claim without an evidence text pair | src/verify_calibrated.py:65 |
| Fusion | 0.75 log P(claim) + 0.25 log P(evidence), plus a fixed NEI offset of 2 | official protocol; src/evidence_pipeline_v2.py:79 |
| Ollama | Instructions plus JSON containing the claim and five sentence strings | src/experiment_ollama_fever.py:40; exact saved request |

The 61.70% benchmark number belongs to frozen fusion; evidence-only RoBERTa scores 52.37%. The original evidence input limit was 256 tokens; the separate 512-token diagnostic retains the same evidence and weights.

## Exact inspected inputs

Full sentence lists, concatenated RoBERTa inputs, tokenizer lengths, actual saved Ollama request/response bodies, dataset reference labels and stored article sentence counts are in exact_inputs.json. No model weights were loaded and no new predictions were generated for this audit. The tokenizer was loaded locally to reconstruct input length. All full-run source/input hashes still match their frozen protocol.

### Claim 111897: Telemundo is a English-language television network.

FEVER reference: REFUTES. Full RoBERTa claim/evidence pair: 347 tokens.

| Rank | Article / sentence | Actual supplied text |
|---|---|---|
| 1 | Noticias_Telemundo / 8 | Noticias Telemundo maintains bureaus located at many of the network 's television stations across the United States -LRB- particularly those owned by parent subsidiary Telemundo Station Group , that are owned-and-operated stations of the network -RRB- and throughout Latin America , and uses video content from English language sister network NBC 's news division NBC News . |
| 2 | Noticiero_Telemundo / 0 | Noticiero Telemundo -LRB- -LSB- notiˈsjeɾo teleˈmundo -RSB- ; English translation : Telemundo News -RRB- is the flagship daily evening television news program of Noticias Telemundo , the news division of the American Spanish language broadcast television network Telemundo . |
| 3 | Telemundo / 0 | Telemundo -LRB- -LSB- teleˈmundo -RSB- -RRB- is an American Spanish-language terrestrial television network owned by Comcast through the NBCUniversal division NBCUniversal Telemundo Enterprises . |
| 4 | Noticias_Telemundo / 0 | Noticias Telemundo -LRB- -LSB- noˈtisjas teleˈmundo -RSB- , Telemundo News -RRB- is the news division of Telemundo , an American Spanish language broadcast television network that is owned by NBCUniversal Hispanic Enterprises and Content , a subsidiary of the NBCUniversal Television Group division of NBCUniversal . |
| 5 | Telemundo_Deportes / 0 | Telemundo Deportes is the programming division of the NBC Sports Group , owned by the NBCUniversal Television Group division of NBCUniversal , that is responsible for the production of sports events and magazine programs that air on NBCUniversal 's Spanish language television networks Telemundo and Universo . |

Actual saved Llama response:

```json
{
  "label": "REFUTES",
  "reason": "The evidence states that Telemundo is an American Spanish-language broadcast television network, contradicting the claim that it is an English-language television network."
}
```

### Claim 81703: Jack Falahee was born.

FEVER reference: SUPPORTS. Full RoBERTa claim/evidence pair: 184 tokens.

| Rank | Article / sentence | Actual supplied text |
|---|---|---|
| 1 | Jack_Falahee / 0 | Jack Ryan Falahee -LRB- born February 20 , 1989 -RRB- is an American actor . |
| 2 | Jack_&_Jack / 0 | Jack & Jack is an American pop-rap duo from Omaha , Nebraska , consisting of Jack Johnson and Jack Gilinsky , who are now based in Los Angeles , California . |
| 3 | Jack-Jack_Attack / 0 | Jack-Jack Attack is a 2005 computer animated short film produced by Pixar based on their film The Incredibles , and directed by Brad Bird . |
| 4 | Jack-Jack_Attack / 5 | The short is based on the baby , Jack-Jack , and takes place at around the same time as the events of the main film . |
| 5 | Jack-Jack_Attack / 6 | From The Incredibles , the audience knows that Jack-Jack 's babysitter Kari McKeen started experiencing difficulty with him shortly after hanging up the phone with his mother , Helen Parr -LRB- also known as Elastigirl or Mrs. Incredible -RRB- . |

Actual saved Llama response:

```json
{
  "label": "NOT ENOUGH INFO",
  "reason": "The evidence does not mention Jack Falahee's birth, only his birthdate is mentioned in a different context."
}
```

## Bottlenecks supported by existing measurements

1. Retrieval and selection: top-five complete annotated evidence-group recall is 63.37% on 13,332 verifiable full-development claims. Failure to cover an annotated group is not proof that all selected evidence is useless.
2. Article identity and distracting evidence: the Jack claim receives one Jack Falahee sentence plus four sentences about other Jack entities. The initial title-query is an OR of up to six claim keywords; sentence selection is lexical overlap, not an entailment model.
3. Classification errors even with relevant evidence: Jack Falahee claim 81703 has its birthdate sentence fully inside a 184-token input, yet the saved Llama answer is NEI and claims the birth information is insufficient. Telemundo contains the Spanish-language fact, yet saved evidence RoBERTa predictions are SUPPORTS at both 256 and 512 tokens; Llama correctly returns REFUTES. These observations show that retrieval/context length alone cannot explain every error.
4. Evidence-model training mismatch: the existing training-input audit records 28,517 empty NEI evidence texts and zero nonempty NEI texts in the processed training corpus. The evidence training dataset pairs claim and evidence_text (src/train_main.py:46). On all 19,998 retrieved-evidence evaluation claims, evidence RoBERTa predicts NEI zero times. This is consistent with an empty-evidence shortcut/distribution mismatch; it is not a causal proof without a controlled training experiment. No retraining was performed.
5. Context length: doubling 256 to 512 tokens fixed 14 and introduced 12 fusion label errors on the full set: only two additional correct answers. This does not support treating the token limit as the sole bottleneck.

## Requested architecture, preserved as the next separate system

Claim → existing candidate article retrieval (top 10) → retain/rank the top 5 articles → fetch their full available article contents by URL/revision → clean and chunk text while preserving provenance → embed and store in a vector database → retrieve claim-relevant evidence → Ollama → SUPPORTS / REFUTES / NOT ENOUGH INFO.

Current learned RoBERTa models are classifiers, not the existing article retriever or sentence ranker. Preserve them as baselines or optional components; retain the existing lexical article-retrieval stage. Keeping the current ranking literally means taking the first five title-ranked pages. A separate article reranker would be a new measured component.

Use a separate corpus and namespace for fetched live Wikipedia articles. Record URL, page title, revision and retrieval time so the inputs are auditable. Live articles can differ from FEVER annotations and cannot automatically be treated as the original FEVER evidence corpus. Preserve the old index and all previous outputs. Begin with inspected input checks and a small fixed paired study before requesting another full-scale run. The expanded-article/vector architecture has not yet been implemented or evaluated.

## Supporting saved measurements

- reports/part2/official_fever_dev_20261023/summary.json
- reports/part2/literature_improvements_20261023_corrected/training_input_audit.json
- reports/part2/token_budget_20261028/summary.json
- reports/part2/ollama_full_dev_20261028/USER_STOP.json
- reports/part2/ollama_full_dev_20261028/A_original_responses.jsonl
