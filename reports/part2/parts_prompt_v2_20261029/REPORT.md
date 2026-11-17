# Three-addition parts-prompt test

Same ten claim IDs and byte-identical saved evidence. No new retrieval, article downloads, embeddings, training or full-development evaluation. Only the three approved instruction additions changed.

| Scoring | Previous prompt | New prompt |
|---|---:|---:|
| Model overall label | 5/10 | 4/10 |
| Code-derived label | 7/10 | 4/10 |

Invalid outputs: 3. Model/derived disagreements: 0.

| ID | Reference | Before model | Before derived | After model | After derived |
|---|---|---|---|---|---|
| 130325 | REFUTES | SUPPORTS | SUPPORTS | SUPPORTS | SUPPORTS |
| 40442 | SUPPORTS | NOT ENOUGH INFO | NOT ENOUGH INFO | INVALID | INVALID |
| 4807 | SUPPORTS | NOT ENOUGH INFO | NOT ENOUGH INFO | REFUTES | REFUTES |
| 219203 | NOT ENOUGH INFO | NOT ENOUGH INFO | NOT ENOUGH INFO | INVALID | INVALID |
| 178210 | REFUTES | REFUTES | REFUTES | SUPPORTS | SUPPORTS |
| 2147 | SUPPORTS | SUPPORTS | SUPPORTS | SUPPORTS | SUPPORTS |
| 47172 | NOT ENOUGH INFO | SUPPORTS | NOT ENOUGH INFO | INVALID | INVALID |
| 226843 | SUPPORTS | SUPPORTS | SUPPORTS | SUPPORTS | SUPPORTS |
| 118664 | REFUTES | REFUTES | REFUTES | REFUTES | REFUTES |
| 61564 | NOT ENOUGH INFO | REFUTES | NOT ENOUGH INFO | NOT ENOUGH INFO | NOT ENOUGH INFO |

## Limits

- Ten already inspected development claims; not an unseen or statistically reliable improvement estimate.
- Three additions tested together; this run cannot isolate which addition caused a change.
- Same 256-token output cap. Truncated or invalid results remain INVALID and count as errors; no retries or silent repairs.
- Citation validity and label aggregation do not establish semantic correctness. No claim-specific vocabulary, gold labels or example answers are added to prompts.

## Observed failures

The new prompt rewrote the Move claim, omitted two justifications, and exceeded the output cap on Yvonne. See ANALYSIS.md and the full raw traces. Keep the earlier prompt as the working research condition; this combined revision is not promoted.
