# Inspecting the frozen parts prompt

Working condition: original essential-parts prompt, with the final label derived
from its part verdicts. The observed score is 7/10 on inspected development
claims; the model's own overall label scored 5/10. This is not held-out accuracy.
The prompt, model/settings, schema and aggregation are recorded by hash in
`working_parts_prompt.json`. Earlier results and failed revisions are preserved.
This selection records the research condition; it does not change other model
baselines or deploy an unmeasured classifier.

## Inspect these ten without tuning to their answers

For each trace, check:

1. Did the returned parts preserve exactly what the claim asserted?
2. Does each cited passage establish or contradict the entire corresponding part?
3. Did the model treat missing information as a contradiction?
4. Does the explanation agree with its part verdict?
5. Does the code-derived label follow the frozen aggregation rule?

Record problems as general failure patterns. Do not add expected answers,
claim-specific vocabulary or special-case overrides. A correct reference label
with bad reasoning is still a reasoning problem. This inspection is exploratory;
it is not independent human-review validation.

## Keep the next evaluation honest

Keep the prompt unchanged while inspecting these examples. Before testing any
future revision, freeze the sampling, evidence pipeline, settings, scoring and
INVALID policy. Compare conditions on the same fresh claims, excluding prior
prompt-inspection/evaluation claims, training overlaps and normalized duplicates.
Save predictions before revealing reference labels. Report every selected case,
including invalid outputs and negative results.

Fresh claims drawn from the same development corpus are a new prompt-validation
sample, not automatically an untouched test set for the whole research project.
Once their outcomes influence a revision, they also become development data.
Reserve a separate final holdout and evaluate it only after choosing the prompt.

No new Llama evaluation, downloads or retraining were started by this selection.
Inspect the original traces at http://127.0.0.1:8765/claim-parts .
