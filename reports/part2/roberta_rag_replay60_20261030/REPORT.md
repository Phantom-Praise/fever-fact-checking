# Exploratory RoBERTa with saved RAG evidence

Same frozen checkpoint, same 60 already inspected claims. No retraining or new retrieval.

| Evidence | Token limit | Correct | Accuracy | Macro-F1 | Inputs truncated |
|---|---:|---:|---:|---:|---:|
| five_sentence_256 | 256 | 34/60 | 56.67% | 0.4548 | 2 |
| rag_256 | 256 | 32/60 | 53.33% | 0.4264 | 60 |
| five_sentence_512 | 512 | 34/60 | 56.67% | 0.4548 | 0 |
| rag_512 | 512 | 32/60 | 53.33% | 0.4264 | 56 |

At 256 tokens, the retrieval-evidence change is compared under an equal input budget. The two 512-token conditions separately explore input length. These changes do not repair the old training-data mismatch.

The original 256-token baseline labels and probabilities were replayed and checked against the saved run (tolerance 1e-5). All supplied RAG passages are used in order; model reasoning, selected citations and gold labels are excluded from the classifier input. The larger RAG text is still truncated to the stated budget.

Paired differences and intervals are in summary.json. These cases have already been inspected, so this is exploratory analysis rather than fresh held-out confirmation. No claim that the best-looking condition generalizes.

Inference-only timing excludes the earlier RAG article fetching and embedding cost. A live RAG+RoBERTa deployment would still pay those costs.

## Interpretation

Replacing the five-sentence evidence with the saved RAG passages did not improve this frozen classifier: accuracy changed from 34/60 (56.67%) to 32/60 (53.33%), at either token limit. The equal-budget 256-token paired difference is -3.33 points, with an exploratory interval of -11.67 to +5.00 points. A reliable performance difference is not established.

All 60 RAG inputs exceeded 256 tokens, and 56 still exceeded 512. Doubling the RAG input limit fixed two cases and introduced two new errors, leaving headline accuracy unchanged. This does not test RoBERTa reading every database chunk. Longer text does not repair the existing training mismatch, and this experiment does not establish why individual decisions changed.

All original results, checkpoints and the separate main-model report remain unchanged.
