# Why the local LLM scored lower on the paired pilot

All measurements below use the same 300 official-development claims and the saved original predictions. No benchmark labels or predictions were changed.

| System | Correct / 300 | Accuracy |
|---|---:|---:|
| evidence_512 | 157 | 52.33% |
| combined_512 | 181 | 60.33% |
| local_llm | 155 | 51.67% |

The 60.33% comparator is a two-model calibrated combination. Evidence-only RoBERTa is 52.33%, versus Llama 51.67%; the paired accuracy-difference interval for that comparison includes zero.

| True FEVER label | Combined correct / 100 | LLM correct / 100 | Net change |
|---|---:|---:|---:|
| NOT ENOUGH INFO | 42 | 66 | +24 |
| REFUTES | 55 | 28 | -27 |
| SUPPORTS | 84 | 61 | -23 |

Llama predicted NEI 163 times, versus 72 for the combined system. It called 37 SUPPORTS and 60 REFUTES claims NEI. The extra 24 correct NEI cases did not offset losing 23 SUPPORTS and 27 REFUTES correct cases: net -26 correct answers.

| Annotated evidence status | Claims | LLM correct | Combined correct | LLM predicted NEI |
|---|---:|---:|---:|---:|
| NEI | 100 | 66 | 42 | 66 |
| no_complete_annotated_group | 79 | 16 | 49 | 61 |
| complete_annotated_group | 121 | 73 | 90 | 36 |

Missing a complete annotated group is not proof that every retrieved sentence is insufficient. The table measures annotation coverage, not a definitive causal attribution.

## Examples from the saved calls

### #146231
Juventus F.C. rejected their traditional black-and-white-striped home uniform in 1903.

Gold: REFUTES; LLM: NOT ENOUGH INFO.

Saved explanation: The evidence does not mention Juventus F.C. or their uniforms, so it does not support or contradict the claim.

Retrieval failure is visible here: none of the five texts concerns Juventus uniforms. The evidence-only NEI response is reasonable for the supplied texts, while FEVER expects REFUTES from the full Wikipedia annotation.

### #149173
Psych is a drama and comedy that aired in 2004.

Gold: REFUTES; LLM: SUPPORTS.

Saved explanation: The evidence confirms that Psych is a comedy-drama series and provides its airing year as 2006, which is close to the claimed 2004.

The gold sentence giving a 2006 debut is present. Treating 2006 as close enough to 2004 is an observable classification/explanation error.

### #202040
Tamerlan Tsarnaev had a sister.

Gold: NOT ENOUGH INFO; LLM: SUPPORTS.

Saved explanation: The evidence mentions Tamerlan Tsarnaev's brother, Dzhokhar Tsarnaev, which supports the claim that Tamerlan Tsarnaev had a sister.

The supplied evidence about a brother does not establish a sister. This is an unsupported inference in the saved answer.

## Bounded oracle diagnostic

On 20 randomly sampled cases from the incomplete-annotation subgroup, replacing the retrieved texts with a complete FEVER-annotated evidence group changed correct labels from 7/20 to 13/20 (6 fixed, 0 introduced).
Claims, model, prompt and decoding settings stayed fixed. This measures a gold-evidence intervention on a selected subgroup. It is not end-to-end accuracy, and the intervention changes both relevance and evidence length/amount.

## What the current records rule out

No invalid outputs; all 300 responses completed. Maximum prompt plus generated token usage was 594, below the 4096 context budget. All supplied evidence fit in the matched RoBERTa 512-token inputs.
These checks do not reveal hidden model reasoning or prove a single universal cause.

## Next controlled experiment

Improve candidate-page/entity matching on separate validation data, then compare the same judges with the resulting frozen evidence. Separately test date/negation/relationship counterexamples and an LLM hybrid selected on calibration data. Do not optimize prompts or fusion on this evaluated 300-claim sample.

Juventus title-query trace: "juventus" OR "rejected" OR "their" OR "traditional" OR "black" OR "white". The top ten candidate pages excluded Juventus_F.C.; the actual query/ranking function is unchanged. See retrieval_example.json.
