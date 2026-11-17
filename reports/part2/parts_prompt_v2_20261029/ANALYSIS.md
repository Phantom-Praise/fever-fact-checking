# Observed failure patterns

The three additions were tested together on the same ten saved evidence inputs, with the same model, JSON schema, seed, temperature and 256-token output cap. They did not improve this development sample: validated model labels fell from 5/10 to 4/10, and code-derived labels fell from 7/10 to 4/10. Three previously correct derived predictions regressed; none were fixed.

| Claim | What happened | Why it matters |
|---|---|---|
| #219203: species distribution / river capture | Returned NEI with an empty justification; previously a valid correct NEI answer. | Same validation policy rejects an incomplete explanation. The raw NEI label is retained, but INVALID remains an error. |
| #178210: Move / Nightmare Revisited | Replaced the assertion with “only released on an album called Salute” and “was not released on ... Nightmare Revisited,” then marked both SUPPORTS. | The model evaluated rewritten claims instead of preserving the original claim's meaning. ID validity and label aggregation cannot catch this semantic failure. |
| #47172: Yvonne Strahovski | Output exceeded the unchanged 256-token cap. It also treated sound-production credits as refuting silent-film acting and began adding theatre work not asserted by the claim. | Both instruction faithfulness and output length remain problems. The partial response is retained and counted INVALID. |
| #40442: The House of the Spirits | Returned NEI with an empty justification. | Another validation failure; the previous prediction was already wrong, so it does not account for an additional regression in accuracy. |

These observations do not isolate which of the three additions caused the changes. Do not promote this combined revision over the previous prompt based on these results. These are ten inspected development claims, not a held-out or statistically reliable estimate. The earlier prompt and results were preserved. No output-cap increase, retries, evidence changes, source fetching or retraining were performed.
