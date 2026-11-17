# Frozen model: 256 versus 512 evidence-input tokens

Same claims, saved top-five evidence, checkpoint and fusion parameters. Claim-only probabilities remain unchanged.

This is an exploratory development-set experiment prompted by inspection of Telemundo. It is not a blind test or training-seed repeat.

| Condition | Accuracy | Macro-F1 | Strict FEVER |
|---|---:|---:|---:|
| retrieved_roberta_256 | 51.56% | 0.3992 | 42.19% |
| retrieved_roberta_512 | 51.56% | 0.3992 | 42.19% |
| calibrated_ensemble_256 | 68.75% | 0.6871 | 54.69% |
| calibrated_ensemble_512 | 68.75% | 0.6871 | 54.69% |

retrieved_roberta: 0 errors fixed, 0 previously correct claims became wrong; accuracy change +0.000 percentage points.

calibrated_ensemble: 0 errors fixed, 0 previously correct claims became wrong; accuracy change +0.000 percentage points.

Inputs longer than 256: 4. Still longer than 512: 0.

Full per-class TP/FP/FN/TN counts: per_class.csv. Per-claim flips: changed_predictions.csv. Raw probabilities: outputs.jsonl.

Length alone does not establish a human-readable cause. Larger inputs can add useful or distracting text. Retrieval recall cannot change because references are fixed. The evidence checkpoint was trained with a 256-token limit. No retraining or production setting change was made.
