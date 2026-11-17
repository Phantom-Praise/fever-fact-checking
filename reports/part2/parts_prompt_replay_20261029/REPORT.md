# Essential-parts prompt replay — ten inspected claims

User wording, same model/settings and byte-identical saved evidence JSON. Overall model label and code-derived label remain separate.

| Condition | Correct / 10 |
|---|---:|
| Previous RAG | 4 |
| Updated prompt: model label | 5 |
| Code-derived label | 7 |

Invalid outputs: 0. Model/derived-label disagreements: 2.

| ID | FEVER reference | Previous RAG | Updated model | Code-derived | Consistent |
|---|---|---|---|---|
| 130325 | REFUTES | REFUTES | SUPPORTS | SUPPORTS | True |
| 40442 | SUPPORTS | SUPPORTS | NOT ENOUGH INFO | NOT ENOUGH INFO | True |
| 4807 | SUPPORTS | REFUTES | NOT ENOUGH INFO | NOT ENOUGH INFO | True |
| 219203 | NOT ENOUGH INFO | SUPPORTS | NOT ENOUGH INFO | NOT ENOUGH INFO | True |
| 178210 | REFUTES | SUPPORTS | REFUTES | REFUTES | True |
| 2147 | SUPPORTS | SUPPORTS | SUPPORTS | SUPPORTS | True |
| 47172 | NOT ENOUGH INFO | SUPPORTS | SUPPORTS | NOT ENOUGH INFO | False |
| 226843 | SUPPORTS | SUPPORTS | SUPPORTS | SUPPORTS | True |
| 118664 | REFUTES | NOT ENOUGH INFO | REFUTES | REFUTES | True |
| 61564 | NOT ENOUGH INFO | SUPPORTS | REFUTES | NOT ENOUGH INFO | False |

## Limits

- Retrospective development replay on ten already inspected claims; not a new held-out improvement estimate.
- Prompt and structured output schema change together. Same 256-token output cap may truncate longer part arrays; those failures remain INVALID, with no silent retry.
- A consistent label does not prove that decomposition, part verdicts or cited evidence are correct.
- No example-specific vocabulary, claim-specific rules or gold labels are added to the prompt.
