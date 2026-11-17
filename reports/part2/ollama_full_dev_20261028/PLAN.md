# Full FEVER development evaluation — prepared, not started

Baseline: frozen evidence RoBERTa on all 19,998 claims.

A_original: Llama with the identical original evidence (controlled system comparison).

D_both: Llama with the previously fixed comparison prompt and entity-based evidence (pipeline comparison).

E_claim_fallback: derive from D without another Llama request; label as hybrid.

Approximately 12.6 hours per Llama condition, 25.1 hours for both, based on saved timings. No retraining, downloads or cloud calls.

Every claim/request/response is saved. Resume uses an unchanged ordered prefix. Three consecutive invalid responses stop the run. Earlier experiments are untouched.

All earlier inspected pilot/assessment texts remain included in the full benchmark; report their uninspected-text complement separately as a sensitivity analysis, not a blind test.
