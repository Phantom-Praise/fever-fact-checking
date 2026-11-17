# Proposed full-development evaluation

Status: inputs are being prepared; Llama inference has not started.

Population: all 19,998 official FEVER development claims, in official order.

Primary baseline: existing frozen evidence RoBERTa, with 512-token evidence inputs. Keep the earlier 256-token baseline as a separate reference.

A_original: existing Llama 3.1 8B Q4_K_M, original prompt and the exact original five evidence sentences. This is the direct comparison with evidence RoBERTa.

D_both: the same Llama model with the previously tested comparison prompt and entity-based retrieval. This is a comparison of pipelines; it changes both prompt and evidence.

E_claim_fallback: calculate from D's saved answers and existing claim-only scores. No additional Llama calls. Clearly label this result as hybrid.

Estimated inference time: approximately 13 hours for either A or D, 26 hours for both, extrapolated from earlier local request timings. Input preparation also takes time. These estimates are not guarantees.

Execution: local-only Ollama, sequential requests, seed 42, temperature 0, 4096-token context, maximum 128 generated tokens. No retraining, new model downloads, paid APIs or merging.

Output controls: exact prompts, requests and responses; all invalid answers remain errors; stop after three consecutive invalid responses; resume requires matching source/input hashes and response IDs/order. Earlier results remain intact.

Metrics: accuracy, classification macro-F1, per-class precision/recall/F1, confusion matrix and official strict FEVER. Report paired accuracy differences with uncertainty against evidence RoBERTa. Never use a partial-run percentage as a final 19,998-claim score.

Limitations: this is an expanded development evaluation, not a new blind test. Earlier pilot/assessment claims and the inspected Telemundo case are included. The uninspected-text complement is a sensitivity analysis, not an untouched test set. No equal-resource architecture claim.
