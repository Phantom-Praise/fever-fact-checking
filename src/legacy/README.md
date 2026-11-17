# Historical Part 1 code

These prototypes and superseded workflows were moved intact from `src/`.
See the source map in `../README.md` for the current research components.

- `main_model.py`, `retriever.py`: original small-corpus retrieval demo.
- `baseline.py`, `test_data.py`, `symbolic_kb.py`: early execution checks.
- `train.py`, `train_main.py`, `train_weighted.py`: early training workflows.
- `prepare_evidence.py`: gold-evidence preparation with empty NEI inputs.
- `retrieve_fever.py`, `evidence_ablation.py`, `diagnose_baseline.py`,
  `analyze_symbolic_patterns.py`: earlier retrieval/diagnostic utilities.

The source is retained for inspection and historical reproducibility. Some
trainers overwrite fixed checkpoint and log paths. Do not run them against
existing results without reviewing those paths. No training was performed during
the cleanup, and no dataset or checkpoint was changed.
