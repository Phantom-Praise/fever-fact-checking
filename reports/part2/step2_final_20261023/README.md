# Latest Step 2 experiment update

Start with `report/STEP2_RESULTS.md`. Its CSV tables report the new measured comparisons.

- `heldout100/`: frozen sample/protocol, actual baseline/hybrid phase events, predictions, and Wikidata responses.
- `rq1_machine_pilot/`: 120 real local-model coding requests on 60 identical source-blinded claims, with raw outputs and model identities.
- `report/`: measured tables, limitations, and a posthoc protocol-quality audit.

The new sample is from the held-out partition and was selected without labels. The older Part 2 PDF/Word package predates this evaluation; its statements about the test split being untouched describe that earlier checkpoint.

The automated RQ1 coders scored only 30% against existing human FEVER labels and did not provide reliable error categorization. Their predictions are not human ground truth. As confirmed by the user, there are no completed human reviews. The review sheets remain blank. The human scorer supports importing completed independent reviews and explicit adjudication later.

All new model execution uses the existing .venv. No training, model downloads, or merge into main occurred. Test results must not be used for more configuration selection. The archive omits checkpoints, databases, and full datasets; raw measurements and source snapshots are retained for audit.
