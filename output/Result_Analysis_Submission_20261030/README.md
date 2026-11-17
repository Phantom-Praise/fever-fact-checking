# Result Analysis - FEVER

Start with **Result_Analysis.pdf**. Full written tables, figures and paired
predictions are under `study/`. The 60 balanced test claims are a small matched
study, not a full test-partition benchmark. Negative and invalid results remain.

## Inspect every saved prompt, answer and evidence passage

No model installation or download is required to view the saved traces.
Unzip this folder, open a terminal in it, then run with Python 3.9 or newer:

```sh
python3 viewer/src/serve_parts_test.py --run study --port 8766
```

Open http://127.0.0.1:8766/ on that same computer. If port 8766 is occupied,
choose 8767 in the command and URL. The viewer binds only to this computer and
does not run inference, contact Ollama, fetch Wikipedia or change predictions.

## Evidence and reproducibility

The main model is full-article Wikipedia retrieval with a per-claim temporary
vector database, selected passages, a frozen Llama essential-parts prompt and
deterministic label aggregation. Evidence RoBERTa is the baseline. Source
snapshots and model/data fingerprints accompany the evaluated protocol. Model
weights and full training/test datasets are not bundled. Some recorded source
paths refer to the original research computer; the portable viewer does not
use them. Full Wikipedia articles and vectors were deleted after each claim.
Supplied/cited passages, source revisions, exact calls and all recorded failures
remain. Exact model-call inputs can be replayed, but retrieval over the deleted
full article corpus is not exactly reproducible from this package alone.

The native OpenMP startup failure was repaired before any predictions were
saved; the evaluated v2 uses the same fixed sample and model settings. The
original failed preparation remains in the repository, not in this submission.
The study does not complete the original generated-claim human-review study.
