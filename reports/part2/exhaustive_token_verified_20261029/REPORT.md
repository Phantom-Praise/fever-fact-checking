# Every-chunk reader — measured-token final results

Same ten development claims, same Llama 3.1 8B digest, and the same previously successful Wikipedia article revisions. The reader processed all 2,129 eligible body-chunk instances. The final judge received every chunk selected across those reader batches, without additional sentence filtering.

| Condition | Correct / 10 | Accuracy | Macro F1 |
|---|---:|---:|---:|
| old_five_sentence | 4 | 40% | 0.333 |
| top_k_rag | 4 | 40% | 0.315 |
| parts_model | 5 | 50% | 0.500 |
| parts_derived | 7 | 70% | 0.707 |
| full_reader_model | 6 | 60% | 0.641 |
| full_reader_derived | 6 | 60% | 0.641 |

Raw overall labels and code-derived labels remain separate. Derivation is any REFUTES part → REFUTES; otherwise all SUPPORTS → SUPPORTS; otherwise NEI. Neither consistency nor a valid citation ID proves the evidence reasoning is correct.

| ID | FEVER reference | Prior parts: model / derived | Full reader: model / derived | Input tokens (with wrapper) |
|---|---|---|---|---:|
| 130325 | REFUTES | SUPPORTS / SUPPORTS | REFUTES / REFUTES | 12775 |
| 40442 | SUPPORTS | NOT ENOUGH INFO / NOT ENOUGH INFO | NOT ENOUGH INFO / NOT ENOUGH INFO | 5873 |
| 4807 | SUPPORTS | NOT ENOUGH INFO / NOT ENOUGH INFO | INVALID / INVALID | 17941 |
| 219203 | NOT ENOUGH INFO | NOT ENOUGH INFO / NOT ENOUGH INFO | NOT ENOUGH INFO / NOT ENOUGH INFO | 3028 |
| 178210 | REFUTES | REFUTES / REFUTES | REFUTES / REFUTES | 15288 |
| 2147 | SUPPORTS | SUPPORTS / SUPPORTS | SUPPORTS / SUPPORTS | 4407 |
| 47172 | NOT ENOUGH INFO | SUPPORTS / NOT ENOUGH INFO | REFUTES / REFUTES | 13865 |
| 226843 | SUPPORTS | SUPPORTS / SUPPORTS | SUPPORTS / SUPPORTS | 18414 |
| 118664 | REFUTES | REFUTES / REFUTES | REFUTES / REFUTES | 15354 |
| 61564 | NOT ENOUGH INFO | REFUTES / NOT ENOUGH INFO | INVALID / INVALID | 23093 |

## Integrity and development corrections

- Coverage: 2129/2129 body chunks across ten claims. Full coverage means each was supplied in a completed, validated reader batch.
- Final invalid outputs: 2. All reported token counts match: True.
- All fetched article-text hashes matched the prior pilot. One previously unavailable article remains unavailable; that claim uses the same four successfully fetched pages.
- Original temporary articles and SQLite databases were deleted. Full saved reader prompts still retain supplied article text.
- Initial final judgments used a conservative UTF-8 byte bound; seven cases were rejected before inference. Those are implementation-budget failures, not measured token-context exhaustion. Their original results remain in `../exhaustive_reader_20261029/`.
- An intermediate precise-sentence follow-up was stopped after encountering the same conservative byte-bound issue. Its partial artifacts remain in `../exhaustive_precise_20261029/`. It is not included as a completed accuracy condition.
- The final correction counts tokens with the active model tokenizer. Seven saved map/final requests calibrated a stable ten-token wrapper. Each final response must report exactly the expected full prompt token count; a mismatch is INVALID.
- All ten final judgments were run under this corrected policy. No new article retrieval, embeddings, evidence filtering, training, or full-19K evaluation was performed.

## Interpretation and limits

These ten claims were already inspected during development. Treat changes as development observations, not a generalization estimate or statistical evidence of improvement. Reader selection, number of calls, prompts and context/output budgets differ from the earlier top-k condition. Reading all chunks does not guarantee complete understanding or perfect relevance selection. The final judge receives selected source chunks, not all article text simultaneously. FEVER reference labels are retained unchanged; historical evidence and current revisions can differ.

## Qualitative caution

For #130325, the final part verdict is REFUTES, but its justification says dark matter particles are different from neutrinos—the same relationship asserted by the claim. It also cites many unrelated passages, including biological dark matter. The reference label matches while the explanation is inconsistent. Citation-ID validity and aggregation consistency do not validate semantic reasoning.

For #47172 (Yvonne Strahovski), the model returned REFUTES because the evidence did not establish silent-film acting and mentioned sound productions. It separately returned NEI for a fragment called silent film. Appearing in sound productions does not itself exclude silent-film acting; missing support was incorrectly treated as contradiction. Broader reading did not resolve this reasoning error.

## Exact final part judgments

### #130325 — Dark matter is different from neutrinos.

Reference: REFUTES. Model: REFUTES. Derived: REFUTES.

- REFUTES — Dark matter is different from neutrinos. — The evidence shows that dark matter is a type of matter that does not interact with electromagnetic radiation, has a different scaling factor than radiation, and is composed of non-baryonic particles, which are different from neutrinos.

### #40442 — The House of the Spirits won awards.

Reference: SUPPORTS. Model: NOT ENOUGH INFO. Derived: NOT ENOUGH INFO.

- NOT ENOUGH INFO — The House of the Spirits — No information about the book's awards is provided.
- NOT ENOUGH INFO — won awards — No information about the book's awards is provided.

### #4807 — Dark matter is distinguishable from dark energy.

Reference: SUPPORTS. Model: INVALID. Derived: INVALID.

- ValueError: Final generation/validation failed

### #219203 — Species distribution can be understood through factors like river capture.

Reference: NOT ENOUGH INFO. Model: NOT ENOUGH INFO. Derived: NOT ENOUGH INFO.

- NOT ENOUGH INFO — Species distribution can be understood through factors like river capture. — The claim mentions river capture as a factor, but the provided evidence does not directly support or contradict this assertion.

### #178210 — Move (Little Mix song) was only released on an album called Nightmare Revisited.

Reference: REFUTES. Model: REFUTES. Derived: REFUTES.

- REFUTES — Move (Little Mix song) was only released on an album called Nightmare Revisited. — The evidence shows that Move (Little Mix song) was released on various albums, including Salute (2013), Get Weird (2015), Glory Days (2016), LM5 (2018), and Confetti (2020), but not Nightmare Revisited.

### #2147 — Elizabeth Olsen was born.

Reference: SUPPORTS. Model: SUPPORTS. Derived: SUPPORTS.

- SUPPORTS — Elizabeth Olsen was born. — The evidence directly states Elizabeth Olsen's birthdate and location.

### #47172 — Yvonne Strahovski is an Australian silent film actress.

Reference: NOT ENOUGH INFO. Model: REFUTES. Derived: REFUTES.

- REFUTES — Yvonne Strahovski is an Australian silent film actress. — The evidence shows that Yvonne Strahovski is an Australian actress, but it does not indicate that she is a silent film actress. In fact, the evidence suggests that she has appeared in a variety of films and television shows, including some with synchronized sound.
- NOT ENOUGH INFO — silent film — The evidence does not provide any information about silent films or silent film actresses.

### #226843 — Sejong the Great was born on May 15, 1397.

Reference: SUPPORTS. Model: SUPPORTS. Derived: SUPPORTS.

- SUPPORTS — Sejong the Great was born on May 15, 1397. — The text explicitly states that Sejong the Great was born on May 15, 1397.

### #118664 — Laura Robson is a volleyball player.

Reference: REFUTES. Model: REFUTES. Derived: REFUTES.

- REFUTES — Laura Robson is a volleyball player. — None of the provided excerpts mention Laura Robson as a volleyball player.

### #61564 — House of Balloons is reviewed by The Weeknd.

Reference: NOT ENOUGH INFO. Model: INVALID. Derived: INVALID.

- ValueError: Final generation/validation failed
