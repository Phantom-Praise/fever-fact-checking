# Full RoBERTa training: checkpoint selection

All conditions completed 14,546 optimizer steps over 116,356 original-training
claims. Checkpoint selection used 14,534 separate original-validation claims.
Normalized training/validation/official-development overlaps were excluded in
data preparation. The comparison uses seed 42; no multi-seed uncertainty is measured.

| Unfrozen final blocks | Best validation macro-F1 | Validation loss at best F1 | Selected step |
|---|---:|---:|---:|
| 1 | 66.2971% | 0.680918 | 11,000 |
| 2 | 68.2425% | 0.649087 | 11,000 |
| **3 — selected** | **69.4235%** | **0.634561** | **11,000** |

These are validation macro-F1 scores, not accuracy or official-development
evaluation results. The last-3 model is selected before the new full evaluation.
All three training protocols match after excluding unfrozen-block settings and
their associated trainable-parameter counts. Each condition started from the
same existing task-trained checkpoint, with two further epochs, learning rate
2e-5, seed 42, batch 4 with accumulation 4, and a 512-token budget. Validation and
checkpointing ran every 1,000 optimizer steps. The final saved model is intended
to contain the validation-selected weights, not simply the last training step.

Sources: the downloaded `last1/2/3_trainer_state.json`, corresponding protocols
and completion metadata in this directory. `validation_selection.json` records
the selected condition and hashes of all three validation histories.

## Prepared full evaluation

`FEVER_Evaluate_19998.ipynb` compares the original checkpoint with the selected
last-3 model on the exact 19,998 official FEVER development claims. The notebook
checks selection histories again and checks that final weights match the
selected checkpoint. It performs inference only, with no retraining.

Both models receive one shared live-article RAG context per claim and an identical
512-token budget. Retained input equality is checked before publishing paired
predictions. Models are loaded once per session. The same MiniLM encoder uses
FP32 CUDA execution instead of the old CPU execution; potential floating-point
retrieval differences from earlier experiments are explicitly recorded. No
speedup or model improvement is claimed before measurements.

Full article/vector workspaces remain temporary. Saved supplied passages and
revision metadata support review. Prior training and diagnostic outputs are
preserved in their original folders. Final label metrics require both conditions
to cover all 19,998 IDs, with invalid executions retained as errors. This live
evidence setup does not yield the official strict FEVER evidence score. The
previously inspected official development set is not an untouched hidden test.

Validation selection and 24 offline checks completed. Full GPU evaluation has
not been started by the assistant; it requires running the uploaded notebook in
the user's authenticated Colab session. Its duration depends on live fetching
and GPU availability and must be estimated from measured throughput.
