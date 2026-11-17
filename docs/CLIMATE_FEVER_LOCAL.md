# Climate-FEVER in VS Code

Open this repository in VS Code. Its workspace configuration selects the
existing `.venv/bin/python`. No new environment, model download, training,
corpus rebuild, or paid service is required by setup.

## Start with the read-only check

In VS Code's terminal:

```bash
.venv/bin/python src/climate_local.py inspect
```

Or choose **Run and Debug → Climate-FEVER: inspect setup (no inference)**.
This checks the 1,535 input records, reads the existing Wikipedia title index,
and lists which requested Ollama models are installed. It does not load an LLM.

The local config is `configs/climate_local.json`. Desired models are Qwen 2.5
7B, Llama 3.1 8B, Mistral 7B and Gemma 3 4B. Missing models cause a clear error;
installed smaller/different models are never silently substituted. No automatic
model pulls are implemented. The RoBERTa path must point to the verified
last-three-block checkpoint; no replacement checkpoint is downloaded.

## Run a bounded diagnostic, then resume

Llama 3.1 8B is already installed on this Mac. When ready:

```bash
.venv/bin/python src/climate_local.py pilot --family llama --minutes 30
.venv/bin/python src/climate_local.py trace --family llama --claim-id climate-0
.venv/bin/python src/climate_local.py report --family llama --scope pilot
```

The pilot targets the first twelve source-order claims. VS Code also has a
launch configuration and tasks for the pilot and latest trace. These actions
call Ollama and may keep the laptop busy. No diagnostic score is claimed until
the run is executed and reviewed. Use the same command to resume completed
records without overwriting them. Stop with Ctrl+C; a partly processed claim
may need to run again, while completed claim files remain intact.

For all 1,535 claims after checking the diagnostic:

```bash
.venv/bin/python src/climate_local.py run --family llama --minutes 30
.venv/bin/python src/climate_local.py report --family llama --scope full
```

Each model runs separately; replace `llama` with `qwen`, `mistral`, `gemma`, or
`roberta` only after it is available/configured. The session limit is checked
between claims, so one long claim can exceed it. Local execution is independent
of Colab quotas. Laptop sleep/shutdown can still interrupt work; VS Code does
not run while the machine is powered off.

## Evidence, output and comparison limits

Reuse `data/processed/wiki_fts.db` read-only. Search its full title index, then
read **all indexed sentences** from each model's top ten articles. This uses
the existing historical FEVER cleaned article text; it is not current live
Wikipedia and does not include all rendered HTML/media. No per-claim annotated
article title or evidence sentence is used to retrieve/select evidence.

The LLM selects its own evidence in batches and recursively reduces it to at
most three final excerpts. Local LLM excerpts have a maximum of 600 characters,
with full original sentences and offsets retained for inspection. RoBERTa
uses its existing 160-token excerpt splitting and heuristic scoring. The local
backend uses Ollama with an 8,192-token context, greedy/temperature-zero output,
seed 42, JSON format, and a conservative byte-budget guard. Actual token counts
and raw API responses are saved; the guard is not exact tokenizer preflight.
Token-limited/incomplete output remains INVALID, never NEI.

Local and Colab backends have different quantization/runtime, chunking and
context limits. Their runs have **separate protocols and output folders** and
must not be combined or called identical reproductions. No speed or accuracy
improvement has been measured. A smaller claim set reduces evaluation volume;
the existing corpus is still about 11 GiB and model memory needs remain large.

Outputs go to `reports/climate_fever/local_ollama_v1/`:

- `pilot12_v1/<family>/predictions/<claim-id>.json`: full prompts, raw responses,
  searched pages, selected passages, errors, and timing.
- `all1535_v1/<family>/predictions/`: full-scope records, separate from the pilot.
- `<scope>/<family>/monitor/status.json` and session JSONL logs: live progress.
- `<scope>/report_*.json`: metrics, class precision/recall/F1, confusion matrix
  including INVALID, and the secondary non-Disputed comparison.

Protocols pin model digest, Ollama version, adapter hashes, database hash,
input identities, prompt, and settings. Changed settings require a new output
directory. All 154 Disputed cases remain; the old three-label RoBERTa cannot
predict Disputed and is labelled as a transfer baseline. Reading saved traces
or reports does not run inference. Pilot-driven prompt changes must be disclosed
as development exposure in any later all-claim analysis.
