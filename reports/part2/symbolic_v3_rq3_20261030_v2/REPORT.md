# Repaired dynamic symbolic stage: matched RQ3 study

New run; no old predictions or labels overwritten. Same claims, fixed models, randomized condition order.

| Condition | Accuracy | Macro-F1 | Mean seconds | Model calls |
|---|---:|---:|---:|---:|
| claim_only | 71.00% | 0.6320 | 0.059 | 100 |
| retrieval_roberta | 71.00% | 0.5158 | 0.104 | 100 |
| symbolic_first | 71.00% | 0.5158 | 0.207 | 99 |

Symbolic decisions: 1/100; correct against FEVER: 1. Baseline errors fixed: 0; new errors: 0.

Claims matching the declared grammar: 4/100. Operational service failures: 2. These are not the same as absent knowledge.

Wikidata facts and complete statements, including qualifiers and references, are stored in knowledge_base/knowledge.sqlite. Raw responses and fetch times are retained in knowledge_base/snapshots. Every trace records the parsed relationship, subject resolution, compared facts and fallback.

This is a limited-grammar symbolic verifier, not a general fact checker. Coverage and correctness must both be reported. No claim of general knowledge-graph superiority follows from this small experiment.

- Small single-run study; many FEVER claims lie outside the declared grammar.
- Wikidata can contain errors, missing facts, ambiguity and temporal differences from FEVER.
- Original fallback training mismatch remains. No graph model or broad semantic claim parser is reproduced.
- Development repairs were checked on earlier inspected examples and synthetic controls; this new sample must not be used for retuning.
- Live network latency and shared caches limit efficiency conclusions. API failures are separately recorded fallbacks.

## Interpretation and remaining limitations

Adding the symbolic stage did not change accuracy or macro-F1 compared with retrieval plus the same RoBERTa checkpoint. All 100 final labels were identical. One correct symbolic decision replaced a neural decision that was already correct, so no errors were fixed and no new classification errors were introduced. This is a negative RQ3 result for this limited implementation, not proof that symbolic methods generally cannot help.

Only four inputs were accepted by the parser. Two Wikidata request chains failed operationally; another claim had insufficient structured evidence under the conservative rules. The word-decade phrase "the seventies" was still routed as a place expression, an additional parser limitation uncovered in this study. No post-score rule changes were applied to this test run. The nine offline checks and known-data controls validate selected behaviors, not every possible claim type.

The knowledge base contains 293 imported structured statements with source information. These are Wikidata assertions, not 293 independently human-validated facts.
