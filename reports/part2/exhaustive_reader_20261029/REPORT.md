# Every-chunk reading pilot — ten development claims

The same ten claim IDs and previously successful Wikipedia revisions were used. Each eligible article-body chunk entered exactly one reader batch. The final judge received the original text of the chunks selected across those batches. This is a multi-call full scan, not a top-k vector search and not a single prompt containing all articles.

| Condition | Correct / 10 | Accuracy | Macro F1 |
|---|---:|---:|---:|
| old_five_sentence | 4 | 40% | 0.333 |
| top_k_rag | 4 | 40% | 0.315 |
| parts_model | 5 | 50% | 0.500 |
| parts_derived | 7 | 70% | 0.707 |
| exhaustive_model | 2 | 20% | 0.267 |
| exhaustive_derived | 2 | 20% | 0.267 |

Model overall labels and code-derived labels are separate measurements. INVALID attempts remain errors in the denominator.

| Claim ID | FEVER reference | Previous top-k | Parts derived | Every-chunk model | Every-chunk derived | Chunks processed |
|---|---|---|---|---|---|---|
| 130325 | REFUTES | REFUTES | SUPPORTS | INVALID | INVALID | 206/206 |
| 40442 | SUPPORTS | SUPPORTS | NOT ENOUGH INFO | NOT ENOUGH INFO | NOT ENOUGH INFO | 102/102 |
| 4807 | SUPPORTS | REFUTES | NOT ENOUGH INFO | INVALID | INVALID | 269/269 |
| 219203 | NOT ENOUGH INFO | SUPPORTS | NOT ENOUGH INFO | NOT ENOUGH INFO | NOT ENOUGH INFO | 63/63 |
| 178210 | REFUTES | SUPPORTS | REFUTES | INVALID | INVALID | 238/238 |
| 2147 | SUPPORTS | SUPPORTS | SUPPORTS | SUPPORTS | SUPPORTS | 57/57 |
| 47172 | NOT ENOUGH INFO | SUPPORTS | NOT ENOUGH INFO | INVALID | INVALID | 414/414 |
| 226843 | SUPPORTS | SUPPORTS | SUPPORTS | INVALID | INVALID | 302/302 |
| 118664 | REFUTES | NOT ENOUGH INFO | REFUTES | INVALID | INVALID | 187/187 |
| 61564 | NOT ENOUGH INFO | SUPPORTS | NOT ENOUGH INFO | INVALID | INVALID | 291/291 |

## Coverage and integrity

- Claims with every chunk processed: 10/10.
- Extracted article text matches prior pilot: True.
- Temporary article downloads and SQLite databases were deleted. Full batch prompts retain supplied article text.
- Total attempt time: 20.4 minutes.
- The earlier quote-extraction attempt is preserved in `../exhaustive_reader_quote_attempt_20261029`; it was stopped after non-verbatim quotations failed validation. It is not folded into this protocol’s scores.

## Failures and limits

- #130325: ValueError: All findings exceed final context; none discarded
- #4807: ValueError: All findings exceed final context; none discarded
- #178210: ValueError: All findings exceed final context; none discarded
- #47172: ValueError: All findings exceed final context; none discarded
- #226843: ValueError: All findings exceed final context; none discarded
- #118664: ValueError: All findings exceed final context; none discarded
- #61564: ValueError: All findings exceed final context; none discarded
- Final context sizing uses a conservative UTF-8 byte upper bound, not a measured Llama token count. A prompt rejected by this bound is an implementation-budget failure; it is not a measured model context exhaustion.
- Ten previously inspected development claims; not an unseen accuracy estimate.
- All chunks delivered in completed validated calls does not prove the model understood them or extracted all relevant evidence.
- Final judge reads exact quotes selected across all reader batches, not all raw article text at once.
- Compared with earlier runs, batching, evidence selection, context/output budgets and prompt workflow change together.
- Reference labels originate in historical FEVER; live article revisions and some reference inconsistencies can affect interpretation.
- Headings, references/external links and extractor navigation exclusions follow the existing article chunker.

A full scan proves delivery coverage, not reasoning quality. Relevance selection may still miss evidence or retain distracting background. A correct reference label can still have an unsupported explanation. No general accuracy improvement can be established from these ten inspected cases.
