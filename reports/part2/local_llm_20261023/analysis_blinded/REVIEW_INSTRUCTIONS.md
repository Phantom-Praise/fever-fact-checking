# Independent RQ1 human review

Give reviewer A `human_review_A.csv` and reviewer B `human_review_B.csv`. Both receive `blinded_review_evidence.jsonl` and these instructions. Each sheet has the same 60 claims in a fixed shuffled order, with neutral review IDs. Reviewers should work independently and should not inspect the analysis-only source key, generation logs, FEVER labels, or each other's decisions. Source clues in wording cannot be completely hidden.

The older `analysis/` folder is a preliminary unblinded export. Use only `analysis_blinded/` for independent annotation.

For every claim, enter:

- `atomic`: YES for one separately checkable assertion, NO for multiple or incomplete assertions.
- `claim_type`: one primary category, PERSON, ORGANIZATION, LOCATION, DATE, NUMBER, RELATION, or OTHER. If categories overlap, choose the property being asserted and explain in notes.
- `temporal`: YES if truth depends on a time/date or historical period, otherwise NO.
- `context_required`: YES if missing referents, dates, scope, or other context prevents standalone interpretation, otherwise NO.
- `human_label`: SUPPORTS, REFUTES, NOT ENOUGH INFO, or UNRATEABLE.
- `error_type`: NONE, ENTITY, DATE, NUMBER, RELATION, MISSING_CONTEXT, UNSUPPORTED, or UNRATEABLE. Choose the main issue; record secondary issues in notes. Use NONE for a supported assertion.
- `evidence_source`: local page/sentence references and any additional authoritative URL or document used. Record retrieval/access dates for time-sensitive facts.
- `notes`: explain the decision and any ambiguity or compound assertion. Do not guess a label from writing style.

Use SUPPORTS only if the available evidence supports every factual part of the claim. Use REFUTES only with evidence contradicting a factual assertion, rather than because a search returned nothing. Use NOT ENOUGH INFO when a well-formed claim cannot be established or contradicted from the documented evidence. Use UNRATEABLE for incomplete or uninterpretable claims; do not silently discard them or automatically turn them into NEI. Preserve all original model outputs and report unrated counts.

The candidate evidence is from the project's historical FEVER Wikipedia corpus. It may be irrelevant, incomplete, or outdated. Seek additional sources where needed and document whether the judgment concerns historical corpus evidence or current-world truth. Prefer the same evidence policy for both sources; record unresolved temporal conflicts for adjudication. Independently reconstructed human labels need not match historical FEVER labels in every case.

After both reviewers finish, run:

```sh
.venv/bin/python src/score_rq1_annotations.py \
  --key reports/part2/local_llm_20261023/analysis_blinded/analysis_only_source_key.json \
  --reviewer-a PATH_TO_COMPLETED_A.csv \
  --reviewer-b PATH_TO_COMPLETED_B.csv \
  --output reports/part2/rq1_review_NEW
```

The scorer refuses incomplete or unreferenced factual judgments. It reports agreement and a disagreement sheet. A third reviewer or an explicit joint adjudication must resolve disagreements before comparing final truth/error distributions by source. Agreement is not proof that a label is correct. Do not use agreed-only counts as the final distribution.

The current generated sample has no completed human reviews. The separate FEVER judge experiment uses existing FEVER human labels; that does not supply labels for the new generated claims.

## Final adjudication import

The agreement scorer now exports all disputed study fields, including temporal and context judgments. In `adjudication_needed.csv`, complete `adjudicated_human_label`, `adjudicated_atomic`, `adjudicated_claim_type`, `adjudicated_error_type`, `adjudicated_temporal`, `adjudicated_context_required`, `adjudication_source`, and `adjudication_notes`. Do not use an AI response as an independent human review.

Rerun the scorer with the two completed reviews and `--adjudications PATH_TO_COMPLETED_ADJUDICATION.csv`, using a new output directory. It requires exactly the disputed IDs, refuses unresolved fields, and writes `final_annotations.csv` and final source distributions. If the two reviews agree in all fields, supply an empty CSV with an `id` header for explicit finalization. Original FEVER labels remain unchanged.
