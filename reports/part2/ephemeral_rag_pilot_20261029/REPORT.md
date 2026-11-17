# Temporary per-claim RAG: ten-claim paired pilot

Same ten official-development claims, same installed Llama 3.1 8B, temperature 0, seed 42. This is a small system comparison; no broad improvement claim.

| System | Correct | Accuracy | Macro-F1 | Invalid | Mean pipeline time |
|---|---:|---:|---:|---:|---:|
| old_five_sentence | 4/10 | 40.0% | 0.3333 | 0 | 3.08 s |
| ephemeral_full_article_rag | 4/10 | 40.0% | 0.3152 | 0 | 52.37 s |

Errors fixed: 2. New errors introduced: 2.

All ten per-claim workspaces deleted, including full articles, chunk metadata and vectors. Retained files contain only the review artifacts approved by the user. The reusable embedding model remains installed.

| Claim ID | Claim | FEVER reference | Original five-sentence Llama | Full-article RAG Llama |
|---|---|---|---|---|
| 130325 | Dark matter is different from neutrinos. | REFUTES | SUPPORTS | REFUTES |
| 40442 | The House of the Spirits won awards. | SUPPORTS | NOT ENOUGH INFO | SUPPORTS |
| 4807 | Dark matter is distinguishable from dark energy. | SUPPORTS | REFUTES | REFUTES |
| 219203 | Species distribution can be understood through factors like river capture. | NOT ENOUGH INFO | SUPPORTS | SUPPORTS |
| 178210 | Move (Little Mix song) was only released on an album called Nightmare Revisited. | REFUTES | NOT ENOUGH INFO | SUPPORTS |
| 2147 | Elizabeth Olsen was born. | SUPPORTS | SUPPORTS | SUPPORTS |
| 47172 | Yvonne Strahovski is an Australian silent film actress. | NOT ENOUGH INFO | NOT ENOUGH INFO | SUPPORTS |
| 226843 | Sejong the Great was born on May 15, 1397. | SUPPORTS | SUPPORTS | SUPPORTS |
| 118664 | Laura Robson is a volleyball player. | REFUTES | NOT ENOUGH INFO | NOT ENOUGH INFO |
| 61564 | House of Balloons is reviewed by The Weeknd. | NOT ENOUGH INFO | NOT ENOUGH INFO | SUPPORTS |

## Limits

- Ten exploratory development claims; not a reliable population improvement estimate or a blind test.
- Corpus date, evidence retrieval, input context and citation prompt differ together; this does not isolate the effect of a vector database.
- No strict FEVER score is computed for live-Wikipedia citations; current revision/block offsets are not historical FEVER sentence IDs.
- Citation validation proves only that quoted excerpts came from the supplied context, not that they logically establish the label.
- Original title ranking can retain irrelevant/disambiguation pages. No retuning from these ten outcomes.
- Baseline uses a 256-token output cap in this pilot, versus 128 in the earlier saved original experiment. Both current conditions have the same generation limits.
- One local Llama model and embedding model; temperature zero and a fixed seed do not guarantee cross-runtime determinism.

Per-class scores and confusion matrices are in summary.json. Each trace retains the exact context and response. An available or valid citation is not independent evidence of a sound explanation.
