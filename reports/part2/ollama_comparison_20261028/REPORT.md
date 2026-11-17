# Local Ollama versus frozen RoBERTa: paired pilot

300 randomly selected official-dev claims: 100 per label. Telemundo excluded and tested separately.
Same original five evidence sentence texts. No training, retrieval changes, model downloads or cloud calls.

| Model | Accuracy | Macro-F1 | Strict FEVER |
|---|---:|---:|---:|
| claim_only_256 | 56.67% | 0.5534 | 11.00% |
| evidence_256 | 52.00% | 0.4152 | 35.00% |
| combined_256 | 60.00% | 0.5910 | 43.67% |
| evidence_512 | 52.33% | 0.4178 | 35.33% |
| combined_512 | 60.33% | 0.5937 | 44.00% |
| local_llm | 51.67% | 0.5069 | 46.33% |

LLM versus combined_512: 52 errors fixed; 78 previously correct claims became wrong. Difference: -8.67 percentage points.
Paired bootstrap 95% interval: -16.00 to -1.33 percentage points.

Invalid responses: 0 / 300. All remain in the accuracy denominator.

Model: llama3.1:8b @ 46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e

- Exploratory balanced 300-claim development sample; not full 19998 evaluation or a blind test.
- Model pretraining exposure to FEVER/Wikipedia is unknown.
- LLM explanations are model-generated, not faithful internal reasoning or human annotations.
- LLM and RoBERTa have different model sizes, tokenizers, training and instruction formatting.
- The fixed prompt was written after inspecting Telemundo; Telemundo is separate from the random sample.
- One run per claim, temperature zero and fixed seed; determinism across runtimes is not guaranteed.
- All response failures stay in the denominator as INVALID, never silently relabeled as NEI.
- Strict FEVER uses the unchanged retrieved five references, not independent assessment of generated explanations.
- Exact sample pairing does not match training budget or establish generalization beyond this pilot.

Strict FEVER, LLM minus combined_512: +2.33 percentage points; paired 95% interval -3.67 to 8.33. The interval includes zero; this pilot does not establish a reliable strict-score improvement.

Reporting audit corrected only the claim-only strict-score condition to submit no evidence; original inference and comparison labels are unchanged. Initial reports are retained in initial_scoring/.
