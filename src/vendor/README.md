# Official FEVER scorer

`fever_scorer.py` is an unmodified copy of `src/fever/scorer.py` from
https://github.com/sheffieldnlp/fever-scorer at commit
`4801615100fbf6327f8e99b5dbaefe5dd890e869`.

The upstream Apache 2.0 license is preserved in `FEVER_SCORER_LICENSE`.
`FEVER_SCORER_PROVENANCE.json` records original URLs, file hashes, and the
official development-data download. The scorer depends on `six==1.17.0`,
pinned separately in the repository's `requirements-fever-eval.txt`.

Use the existing project virtual environment. No Transformer dependency or
checkpoint is replaced. Evaluation uses `max_evidence=5`. Correct NEI labels do
not require evidence; other labels require a complete annotated evidence group.
The upstream source, including its comments, is retained exactly as published;
the regression tests check the actual scoring behavior rather than relying on
potentially confusing comments.
