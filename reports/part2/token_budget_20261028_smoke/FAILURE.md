This initial 64-claim smoke run completed inference and reproduced the original 256-token predictions exactly. Report generation then stopped because the code used an incorrect accuracy-field name. No checkpoint or benchmark file changed.

The field-name issue was fixed before the full experiment. The successfully completed repeat is in ../token_budget_20261028_smoke_verified/. Preserve this directory as the initial run record; do not use it as the final smoke-check artifact.
