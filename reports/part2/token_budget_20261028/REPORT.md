# Frozen model: 256 versus 512 evidence-input tokens

Same claims, saved top-five evidence, checkpoint and fusion parameters. Claim-only probabilities remain unchanged.

This is an exploratory development-set experiment prompted by inspection of Telemundo. It is not a blind test or training-seed repeat.

| Condition | Accuracy | Macro-F1 | Strict FEVER |
|---|---:|---:|---:|
| retrieved_roberta_256 | 52.37% | 0.4175 | 35.07% |
| retrieved_roberta_512 | 52.38% | 0.4175 | 35.07% |
| calibrated_ensemble_256 | 61.70% | 0.6078 | 45.33% |
| calibrated_ensemble_512 | 61.71% | 0.6078 | 45.33% |

retrieved_roberta: 7 errors fixed, 6 previously correct claims became wrong; accuracy change +0.005 percentage points.

calibrated_ensemble: 14 errors fixed, 12 previously correct claims became wrong; accuracy change +0.010 percentage points.

Inputs longer than 256: 1738. Still longer than 512: 4.

Full per-class TP/FP/FN/TN counts: per_class.csv. Per-claim flips: changed_predictions.csv. Raw probabilities: outputs.jsonl.

Length alone does not establish a human-readable cause. Larger inputs can add useful or distracting text. Retrieval recall cannot change because references are fixed. The evidence checkpoint was trained with a 256-token limit. No retraining or production setting change was made.
