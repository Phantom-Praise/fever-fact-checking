"""Build a plain nbformat-4 notebook; no execution, model calls or downloads."""
import json,textwrap,base64,zlib,hashlib,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':textwrap.dedent(s).strip().splitlines(keepends=True)})
def code(s):cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':textwrap.dedent(s).strip().splitlines(keepends=True)})
md('''# FEVER on Google Colab
This version includes live stage logs and saved debug traces. Reuse the original `FEVER_Colab_Bundle.zip`; only the notebook needs updating.\n\nStart with a GPU setup check. Then run **RoBERTa, Qwen or Llama** against the full **19,998-claim official FEVER development set**, with resumable outputs in Google Drive.

**This notebook contains no new results.** The included small training/validation sets only test the software. They are not final fine-tuning experiments. This benchmark has previously been inspected; do not call it an untouched test set.

1. Open this file in [Google Colab](https://colab.research.google.com/), then select **Runtime → Change runtime type → T4 GPU**.
2. In Google Drive, create a folder named **FEVER_Project** and upload both this notebook and `FEVER_Colab_Bundle.zip` into it. Leave the ZIP zipped. The notebook finds it automatically; you do not enter data or a file path.
3. Run sections 1–5. They verify the GPU, uploaded data and source hashes. They do not start training or the full evaluation.
4. Downloads and the 90-claim development diagnostic (section 6b) are enabled. The one-claim check is optional. Run sections 1–6b, then inspect its report. Full evaluation and training stay off. The first Qwen download is approximately 5.75 GiB plus 0.08 GiB for embeddings.
5. Enable a long run only after inspecting that check. Free Colab sessions may end early. Drive checkpoints allow resuming, but free Colab does **not** guarantee background execution after closing your laptop.

Sources: [Colab limits](https://research.google.com/colaboratory/faq.html), [Qwen 3B](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [Llama 3.2 3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
''')
md('''## 1. Check your GPU
This prints the allocated GPU and available memory. It stops if the runtime is CPU-only.''')
code('''import sys, subprocess
from pathlib import Path
import torch
print("Python:", sys.version.split()[0])
if not torch.cuda.is_available():
    raise RuntimeError("Select Runtime > Change runtime type > T4 GPU, then run this cell again.")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GiB")
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"], check=True)
''')
md('''## 2. Install the Colab dependencies
This installs Python libraries in the **Colab runtime**, not on your laptop. It does not replace Colab's CUDA-enabled PyTorch. Model weights download later, after the size estimate.''')
code('''import subprocess, sys, os
os.environ["USE_TF"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HOME"] = "/content/fever_hf_cache"
packages = ["transformers==4.57.6", "accelerate>=1.10,<2", "bitsandbytes>=0.48,<0.49",
            "faiss-cpu==1.12.0", "scikit-learn>=1.6,<2", "requests>=2.32,<3", "librosa>=0.11,<0.12"]
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *packages])
print("Libraries installed. If Colab requests a runtime restart, restart and rerun from section 1.")
''')
md('''## 3. Connect Drive — find the uploaded data automatically
Mounting Drive asks you to authenticate. The notebook never prints account tokens. Keep the bundle private; it does not need to be published anywhere.

Use **My Drive / FEVER_Project** for the two uploaded files. This cell creates that folder if necessary and finds the ZIP automatically. It also recognizes the earlier My Drive root location. The working extraction is temporary; new experiment results are saved under **FEVER_Project / runs**.''')
code('''from google.colab import drive
from pathlib import Path
import zipfile, json, sys

drive.mount("/content/drive")
DRIVE_PROJECT = Path("/content/drive/MyDrive/FEVER_Project")
DRIVE_PROJECT.mkdir(parents=True, exist_ok=True)
locations = [DRIVE_PROJECT / "FEVER_Colab_Bundle.zip",
             Path("/content/drive/MyDrive/FEVER_Colab_Bundle.zip")]
archive = next((p for p in locations if p.is_file()), None)
if archive is None:
    raise FileNotFoundError("The FEVER_Project folder is ready in My Drive. Upload FEVER_Colab_Bundle.zip into it, wait for the upload to finish, then rerun this cell. No path needs typing.")
print("Found your data:", archive)
PROJECT = Path("/content/fever_colab_project")
PROJECT.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
    # Refuse traversal and links in an uploaded ZIP.
    for info in z.infolist():
        target = (PROJECT / info.filename).resolve()
        target.relative_to(PROJECT.resolve())
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("Archive symlink rejected")
    z.extractall(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))
import fever_colab_runtime as fever
EVAL_ROWS, EVAL_LABELS = fever.verify_bundle(PROJECT)
print("Project ready:", PROJECT)
''')
md('''## 4. Experiment settings
These settings are recorded and checked on resume. Changing the model, prompt, token budget, source code or package/GPU identity requires a **new experiment name**.

- **Qwen:** Qwen2.5-3B-Instruct in 4-bit NF4, the first setup test.
- **Llama:** Llama-3.2-3B-Instruct in 4-bit NF4. You must obtain access on its model page yourself and add `HF_TOKEN` in Colab Secrets. This is a new 3B model, **not** the earlier Ollama Llama 3.1 8B.
- **RoBERTa:** the exact existing project checkpoint, with 512-token inputs.
- Only one classifier is loaded at a time. RAG contexts are shared between conditions; different tokenizers and RoBERTa's shorter context remain limitations.
- This HF generator uses ordinary greedy decoding, not Ollama's JSON-schema constrained decoder. The parser accepts plain JSON or one complete Markdown JSON fence. It derives the overall label from validated parts, records any missing model label, and preserves the exact response. Malformed, truncated or invalid-citation outputs remain errors; no content is repaired. The 512-token output cap is a declared new setting.
''')
code('''EXPERIMENT_NAME = "colab_full_dev_debug_v5" # @param {type:"string"}
MODEL_FAMILY = "qwen" # @param ["qwen", "llama", "roberta"]
ALLOW_MODEL_DOWNLOADS = True # @param {type:"boolean"}
SESSION_MINUTES = 45 # @param {type:"integer"}
MAX_NEW_TOKENS = 512 # @param {type:"integer"}
ROBERTA_MAX_LENGTH = 512 # @param [256, 512] {type:"raw"}
# Leave empty for the included original checkpoint. A completed fine-tuned model may be selected later.
ROBERTA_CHECKPOINT_OVERRIDE = "" # @param {type:"string"}
RUN_ROOT = DRIVE_PROJECT / "runs" / EXPERIMENT_NAME
previous_run = Path("/content/drive/MyDrive/FEVER_runs") / EXPERIMENT_NAME
if previous_run.is_dir() and not RUN_ROOT.exists():
    RUN_ROOT = previous_run  # Resume an existing run without moving or replacing it.
RUN_ROOT.mkdir(parents=True, exist_ok=True)
if not 1 <= SESSION_MINUTES <= 300:
    raise ValueError("Use a bounded session between 1 and 300 minutes.")
if not 1 <= MAX_NEW_TOKENS <= 1024:
    raise ValueError("Use 1–1024 output tokens; freeze the choice before the full evaluation.")
if ROBERTA_MAX_LENGTH not in (256, 512):
    raise ValueError("RoBERTa supports 256 or 512 here, not arbitrarily long article input.")
print("Results saved to:", RUN_ROOT)
''')
md('''## 5. Check model access and report download sizes
This reads model metadata; it does not download model weights. The listed weight sizes are disk/network downloads **before** runtime quantization. Four-bit loading lowers GPU memory requirements, not necessarily network transfer.

If Llama access fails, use Qwen for the setup check while requesting Llama access separately. No token is written to the experiment protocol.''')
code('''EMBEDDING = fever.remote_metadata("sentence-transformers/all-MiniLM-L6-v2")
if MODEL_FAMILY == "roberta":
    MODEL_METADATA = {"path": ROBERTA_CHECKPOINT_OVERRIDE or str(PROJECT / "checkpoint/roberta_evidence")}
    CONDITION = "roberta_" + ("selected_checkpoint" if ROBERTA_CHECKPOINT_OVERRIDE else "existing") + "_rag_" + str(ROBERTA_MAX_LENGTH)
    print("RoBERTa uses the uploaded checkpoint; no new classifier weights download.")
else:
    MODEL_METADATA = fever.remote_metadata(fever.MODELS[MODEL_FAMILY])
    CONDITION = MODEL_FAMILY + "_3b_nf4_parts"
print("Condition:", CONDITION)
print("Full evaluation target:", len(EVAL_ROWS), "claims. No evaluation has started.")
''')
md('''## 6. Run one engineering check first
The one-claim check is optional and off because it has already been completed. Section 6b runs the approved 90-claim diagnostic. This check uses **one training-pilot claim**, not a reduced test set. Its article URLs come from that example's retrieved pages; full evaluation uses the exported original top-ten ranking.

Inspect the full prompt, actual article passages and raw output. A successful check verifies execution, not model accuracy. Nothing from this check is pooled into the 19,998-claim result.''')
code('''RUN_SMOKE_CHECK = False # @param {type:"boolean"}
if not RUN_SMOKE_CHECK:
    print("Not started. Enable RUN_SMOKE_CHECK after setup.")
elif not ALLOW_MODEL_DOWNLOADS:
    print("Read section 5, enable ALLOW_MODEL_DOWNLOADS, rerun section 4 and section 5, then rerun this cell.")
else:
    from urllib.parse import quote
    import time
    example = fever.read_rows(PROJECT / "data/train_pilot.jsonl")[0]
    pages = list(dict.fromkeys(e["page_id"] for e in example["evidence"]))
    from fetch_wikipedia_articles import page_url
    example = {"id": example["id"], "claim": example["claim"],
               "ranked_articles": [{"rank": i + 1, "url": page_url(p)} for i, p in enumerate(pages)]}
    encoder = fever.ResidentEmbedder(EMBEDDING)
    verifier = None
    try:
        context = fever.context_for(example, PROJECT, RUN_ROOT / "software_check", encoder, EMBEDDING)
        print("Claim:", example["claim"])
        print("Evidence status:", context["status"])
        if context["status"] != "READY":
            raise RuntimeError("Wikipedia preparation failed. Inspect saved error before any full run: " + str(context.get("error")))
        verifier = fever.RobertaVerifier(MODEL_METADATA["path"], ROBERTA_MAX_LENGTH) if MODEL_FAMILY == "roberta" else fever.CausalVerifier(MODEL_METADATA, PROJECT)
        result = verifier.predict(example, context, max_new_tokens=MAX_NEW_TOKENS)
        fever.write_once(RUN_ROOT / "software_check" / (CONDITION + "_" + str(time.time_ns()) + ".json"), result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("Supplied evidence:", json.dumps(context["supplied_evidence"], indent=2, ensure_ascii=False))
    finally:
        if verifier is not None:
            verifier.close()
        del encoder
        fever.release()
''')
diagnostic_source = ROOT/'output/colab_diagnostic_training_20261105'
diagnostic_manifest = json.loads((diagnostic_source/'diagnostic.json').read_text())
diagnostic_files = {name:(diagnostic_source/'data'/name).read_text() for name in diagnostic_manifest['files']}
md('''## 6b. Diagnose 90 development claims — enabled
This fixed sample has 30 claims per class, selected from the original validation partition, excluding original-training, official-development and software-pilot text overlaps. It is a development diagnostic, not the final 19,998-claim evaluation.

The model receives only claims and ranked article URLs, then the unchanged full-article RAG context. Reference labels and FEVER annotations are used only for scoring/review. No prompt tuning, example-specific rules or model training occurs here.

Watch the live progress. Outputs resume from Drive after a session pauses. The report includes accuracy, macro-F1, per-class precision/recall/F1, confusion matrix and a CSV linking every prediction to its exact prompt/response and supplied evidence. Incorrect labels are marked for human review: a wrong answer alone cannot distinguish inadequate evidence from faulty reasoning. FEVER evidence references are historical; live Wikipedia may differ.''')
code('''RUN_DEVELOPMENT_DIAGNOSTIC = True # @param {type:"boolean"}
if RUN_DEVELOPMENT_DIAGNOSTIC:
    DIAGNOSTIC_ROOT = PROJECT / "development90"
    (DIAGNOSTIC_ROOT / "data").mkdir(parents=True, exist_ok=True)
    for name, content in DIAGNOSTIC_FILES.items():
        path = DIAGNOSTIC_ROOT / "data" / name
        if path.exists() and path.read_text() != content:
            raise ValueError("Existing diagnostic data differ; do not mix samples.")
        path.write_text(content)
    fever.write_once(DIAGNOSTIC_ROOT / "diagnostic.json", DIAGNOSTIC_MANIFEST)
    for name in ["src", "checkpoint"]:
        target = DIAGNOSTIC_ROOT / name
        if not target.exists(): target.symlink_to(PROJECT / name, target_is_directory=True)
    import shutil
    shutil.copy2(PROJECT / "parts_prompt.txt", DIAGNOSTIC_ROOT / "parts_prompt.txt")
    DIAGNOSTIC_ROWS = fever.verify_diagnostic(DIAGNOSTIC_ROOT)
    DIAGNOSTIC_OUT = RUN_ROOT / "development90"
    import time
    diagnostic_started = time.monotonic()
    fever.run_condition(DIAGNOSTIC_ROWS, DIAGNOSTIC_ROOT, DIAGNOSTIC_OUT, CONDITION,
        MODEL_FAMILY, MODEL_METADATA, EMBEDDING, minutes=SESSION_MINUTES,
        max_new_tokens=MAX_NEW_TOKENS, max_length=ROBERTA_MAX_LENGTH, scope="development90")
    completed = sum(p.name != "protocol.json" for p in (DIAGNOSTIC_OUT / "predictions" / CONDITION).glob("*.json"))
    remaining = SESSION_MINUTES - (time.monotonic() - diagnostic_started)/60
    if MODEL_FAMILY != "roberta" and completed == 90 and remaining >= 1:
        print("Running the existing RoBERTa baseline on the SAME 90 claims and saved RAG passages.")
        fever.run_condition(DIAGNOSTIC_ROWS, DIAGNOSTIC_ROOT, DIAGNOSTIC_OUT,
            "roberta_existing_rag_" + str(ROBERTA_MAX_LENGTH), "roberta",
            {"path": str(PROJECT / "checkpoint/roberta_evidence")}, EMBEDDING,
            minutes=remaining, max_length=ROBERTA_MAX_LENGTH, scope="development90")
else:
    print("Development diagnostic disabled.")
''')
# Embed only the small fixed diagnostic input/annotation files, not model weights.
cells[-1]['source'] = ("DIAGNOSTIC_MANIFEST = " + repr(diagnostic_manifest) + "\nDIAGNOSTIC_FILES = " + repr(diagnostic_files) + "\n" + ''.join(cells[-1]['source'])).splitlines(keepends=True)
md('''### Inspect the diagnostic results
Run this after the diagnostic cell finishes or pauses. Partial results remain clearly marked; missing predictions are not counted as completed claims. The confusion-matrix rows are reference labels; columns are model predictions. INVALID is an execution/output failure, not a fourth FEVER class.''')
code('''if "DIAGNOSTIC_OUT" in globals():
    import pandas as pd
    diagnostic_conditions = sorted(p.name for p in (DIAGNOSTIC_OUT / "predictions").glob("*") if p.is_dir())
    diagnostic_metrics = fever.diagnostic_report(DIAGNOSTIC_ROOT, DIAGNOSTIC_OUT, diagnostic_conditions)
    for name, metrics in diagnostic_metrics.items():
        display(pd.DataFrame([{"System": name, "Status": metrics["status"],
            "Completed / 90": metrics["n"], "Accuracy (%)": 100*metrics["accuracy"],
            "Macro-F1 (%)": 100*metrics["macro_f1"], "Invalid": metrics["invalid"]}]).round(2))
        display(pd.DataFrame([{"Class": label, **{key+" (%)": 100*value for key,value in values.items()}}
            for label,values in metrics["per_class"].items()]).round(2))
        display(pd.DataFrame(metrics["confusion_matrix"], index=fever.LABELS,
            columns=fever.LABELS+["INVALID"]))
    fever.monitor_status(DIAGNOSTIC_OUT, CONDITION, last_events=5)
    fever.inspect_saved(DIAGNOSTIC_OUT, CONDITION, show_reference=True, root=DIAGNOSTIC_ROOT, show_full=True)
else:
    print("Run section 6b first.")
''')
md('''## 7. Run / resume the full 19,998-claim evaluation
This is the long operation. It is **off by default**. It attempts the whole official set in its original order and pauses after the session budget. Rerun this cell to resume; do not delete the Drive folder.

Run one model family at a time, then change `MODEL_FAMILY` in section 4 and rerun sections 4–5 and this section. Each family reuses the same saved per-claim RAG passages. Newly fetched full articles and vectors are deleted after each claim; supplied passages, revisions and exact model calls remain.

Failed evidence preparation is an **INVALID execution**, not NEI. Completed invalid predictions remain errors; there is no silent retry. No strict FEVER evidence score is claimed for live article passages. Free Colab may stop early, and GPU availability is not guaranteed.''')
code('''RUN_FULL_EVALUATION = False # @param {type:"boolean"}
MAX_NEW_CLAIMS_THIS_SESSION = 0 # @param {type:"integer"}
# 0 means continue until the time budget; a positive number pauses after that many NEW claims.
if not RUN_FULL_EVALUATION:
    print("Full run is off. Target remains all 19,998 claims.")
elif not ALLOW_MODEL_DOWNLOADS:
    print("Model downloads are disabled; inspect section 5 before enabling them.")
else:
    fever.run_condition(EVAL_ROWS, PROJECT, RUN_ROOT, CONDITION, MODEL_FAMILY,
                        MODEL_METADATA, EMBEDDING, minutes=SESSION_MINUTES,
                        max_new_tokens=MAX_NEW_TOKENS, max_length=ROBERTA_MAX_LENGTH,
                        max_new_claims=MAX_NEW_CLAIMS_THIS_SESSION or None)
''')
md('''## 8. Check progress and score complete conditions
Partial runs show progress only. A final accuracy table is produced only when a condition covers **all 19,998 IDs**. INVALID remains in the denominator. Metrics are label accuracy, macro-F1, per-class precision/recall/F1 and confusion matrices.

A complete official-development result is still a development result. The benchmark has already been inspected. The historical audit found three normalized-text overlaps with the original checkpoint’s training pool; all 19,998 benchmark rows remain in evaluation. This is not an untouched final test.''')
code('''conditions = sorted(p.name for p in (RUN_ROOT / "predictions").glob("*") if p.is_dir())
for name in conditions:
    folder = RUN_ROOT / "predictions" / name
    n = sum(p.name != "protocol.json" for p in folder.glob("*.json"))
    print(name, n, "/", len(EVAL_ROWS))
METRICS = fever.score(PROJECT, RUN_ROOT, conditions, require_full=True)
if METRICS:
    import pandas as pd
    display(pd.DataFrame([{"System": name, "Claims": m["n"], "Accuracy (%)": 100*m["accuracy"],
                           "Macro-F1": m["macro_f1"], "Invalid": m["invalid"]} for name, m in METRICS.items()]))
    import matplotlib.pyplot as plt
    for name, m in METRICS.items():
        display(pd.DataFrame([{"Class": label, "Precision (%)": 100*v["precision"],
                              "Recall (%)": 100*v["recall"], "F1 (%)": 100*v["f1"]}
                             for label, v in m["per_class"].items()]))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.imshow(m["confusion_matrix"], cmap="Blues")
        ax.set_xticks(range(4), ["SUPPORTS", "REFUTES", "NEI", "INVALID"])
        ax.set_yticks(range(3), ["SUPPORTS", "REFUTES", "NEI"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name + " — all 19,998 claims")
        for i, row in enumerate(m["confusion_matrix"]):
            for j, value in enumerate(row): ax.text(j, i, str(value), ha="center", va="center")
        plt.show()
''')
md('''## 8b. Connect the full training supplement
This cell looks for `FEVER_Training_Data.zip` in your Drive `FEVER_Project` folder. Preparation is separate from the original bundle, so that bundle stays unchanged. If the supplement is not ready yet, this prints a message and does not start training.

We preserve the original split: 116,356 training claims and 14,534 validation claims after excluding normalized training/official-development text overlaps from validation. The 90 diagnostic claims are validation data. The original internal test remains unused. We do not reshuffle data that the initial checkpoint has already seen into a new validation split.''')
code('''training_archive = DRIVE_PROJECT / "FEVER_Training_Data.zip"
FULL_TRAINING_READY = False
if not training_archive.is_file():
    print("Full training supplement is not uploaded yet. No training started.")
else:
    with zipfile.ZipFile(training_archive) as archive:
        ready = json.loads(archive.read("training_ready.json"))
        expected_names = {"data/train_full.jsonl", "data/validation_full.jsonl", "training_ready.json", "preparation_protocol.json"}
        if set(archive.namelist()) != expected_names:
            raise ValueError("Unexpected training supplement contents")
        for name in ["train_full.jsonl", "validation_full.jsonl"]:
            content = archive.read("data/" + name)
            if hashlib.sha256(content).hexdigest() != ready["files"][name]:
                raise ValueError("Training supplement checksum mismatch")
            target = PROJECT / "data" / name
            if target.exists() and target.read_bytes() != content:
                raise ValueError("Existing training data differ; do not overwrite")
            target.write_bytes(content)
    fever.write_once(PROJECT / "training_ready.json", ready)
    FULL_TRAINING_READY = True
    print("Full data ready:", ready["train_n"], "training;", ready["validation_n"], "validation.")
''')
md('''## 9. Optional RoBERTa layer-unfreezing check
This is a **training operation**, off by default. Run it only when you choose to start training.

- `pilot`: included **96 training + 96 validation examples**, used only to verify that training and checkpoints work.
- `full`: requires `train_full.jsonl` and `validation_full.jsonl` from the optional full-training export. The starter bundle intentionally does not prepare this hours-long local retrieval job.
- All conditions start from the **same original task-trained RoBERTa checkpoint**. This is continued fine-tuning, not starting from raw pretrained RoBERTa.
- Freeze all weights, then unfreeze the classification head and the last **1, 2 or 3 Transformer blocks**. Earlier blocks remain frozen.
- Fixed seed 42, 2 epochs, learning rate 2e-5, batch size 4, gradient accumulation 4, maximum 512 tokens. Validation selects the checkpoint; official development labels never train or select it.
- Training uses historical retrieved sentence text for every class, not the new live-article RAG corpus. Report that distribution difference. The pilot includes nonempty NEI evidence to avoid the old empty-input shortcut.

Change only the number of unfrozen layers between controlled conditions. Llama/Qwen here are inference baselines: **this notebook does not implement their fine-tuning**. Quantized LLM adapter training would be a separate experiment and is not equivalent to unfreezing their original last layers.''')
code('''RUN_ROBERTA_TRAINING = False # @param {type:"boolean"}
UNFREEZE_LAST_N = 1 # @param [1, 2, 3] {type:"raw"}
TRAIN_SCOPE = "full" # @param ["pilot", "full"]
if not RUN_ROBERTA_TRAINING:
    print("Training is off. Pilot training is a software check, not a completed research experiment.")
else:
    final_checkpoint = fever.finetune_roberta(PROJECT, RUN_ROOT,
                    last_layers=UNFREEZE_LAST_N, scope=TRAIN_SCOPE, minutes=SESSION_MINUTES)
    if final_checkpoint:
        print("To evaluate this model, set ROBERTA_CHECKPOINT_OVERRIDE in section 4 to:")
        print(str(final_checkpoint))
        print("Use a distinct condition name for each trained checkpoint; see the next cell.")
''')
md('''### Run the three controlled full-training conditions
This is off until the diagnostic has been reviewed and the full supplement is ready. All three conditions use the same initial task-trained checkpoint, training/validation files, seed and hyperparameters. Only the last 1, 2 or 3 unfrozen blocks differ. A session stops when its budget is reached; rerunning resumes incomplete training and skips completed conditions. This trains RoBERTa only; it does not train Qwen/Llama.

These training files use historical retrieved sentences, while the diagnostic/evaluation uses live-article RAG. This distribution difference remains a limitation and must be reported.''')
code('''RUN_CONTROLLED_FULL_TRAINING = False # @param {type:"boolean"}
if RUN_CONTROLLED_FULL_TRAINING:
    if not FULL_TRAINING_READY:
        raise RuntimeError("Connect the full training supplement in section 8b first.")
    import time
    started = time.monotonic()
    for layers in [1, 2, 3]:
        remaining = SESSION_MINUTES - (time.monotonic() - started)/60
        if remaining < 1:
            print("Session budget reached. Rerun to resume remaining conditions.")
            break
        checkpoint = fever.finetune_roberta(PROJECT, RUN_ROOT, last_layers=layers,
                                           scope="full", minutes=remaining)
        if checkpoint is None:
            print("Condition paused:", layers, "blocks. Rerun this cell to resume.")
            break
else:
    print("Full training is off pending diagnostic review. No model has been trained by this cell.")
''')
md('''### Compare completed training conditions on the same 90 development claims
This runs only when controlled full training is enabled. Completed checkpoints are compared with the existing baseline on the exact diagnostic claims and saved RAG passages. Partial training conditions are skipped. These development results help assess changes; they do not replace the final official-development evaluation.''')
code('''if RUN_CONTROLLED_FULL_TRAINING and "DIAGNOSTIC_ROOT" in globals():
    import time
    comparison_started = time.monotonic()
    for layers in [1, 2, 3]:
        trained = RUN_ROOT / "training" / ("roberta_full_last" + str(layers))
        if not (trained / "complete.json").is_file():
            print("Not yet complete:", layers, "blocks")
            continue
        remaining = SESSION_MINUTES - (time.monotonic()-comparison_started)/60
        if remaining < 1:
            print("Comparison session budget reached. Rerun to resume.")
            break
        name = "roberta_full_last" + str(layers) + "_rag_" + str(ROBERTA_MAX_LENGTH)
        fever.run_condition(DIAGNOSTIC_ROWS, DIAGNOSTIC_ROOT, DIAGNOSTIC_OUT, name,
            "roberta", {"path": str(trained / "final")}, EMBEDDING, minutes=remaining,
            max_length=ROBERTA_MAX_LENGTH, scope="development90")
    conditions = sorted(p.name for p in (DIAGNOSTIC_OUT / "predictions").glob("*") if p.is_dir())
    report = fever.diagnostic_report(DIAGNOSTIC_ROOT, DIAGNOSTIC_OUT, conditions)
    import pandas as pd
    display(pd.DataFrame([{"Condition": name, "Status": m["status"], "Claims": m["n"],
        "Accuracy (%)": 100*m["accuracy"], "Macro-F1 (%)": 100*m["macro_f1"], "Invalid": m["invalid"]}
        for name,m in report.items()]).round(2))
else:
    print("Before/after comparison waits for enabled full training and the diagnostic inputs.")
''')
md('''### Evaluate a completed fine-tuned checkpoint
Use the full 19,998 inputs again. The condition name includes its layer count and training scope. A pilot-trained condition is explicitly named pilot and must not be called full-training performance.''')
code('''RUN_FINETUNED_EVALUATION = False # @param {type:"boolean"}
if not RUN_FINETUNED_EVALUATION:
    print("Fine-tuned evaluation is off.")
else:
    trained = RUN_ROOT / "training" / ("roberta_" + TRAIN_SCOPE + "_last" + str(UNFREEZE_LAST_N))
    if not (trained / "complete.json").is_file():
        raise RuntimeError("Training has not completed. Resume training before evaluating its final checkpoint.")
    name = "roberta_" + TRAIN_SCOPE + "_last" + str(UNFREEZE_LAST_N) + "_rag_" + str(ROBERTA_MAX_LENGTH)
    fever.run_condition(EVAL_ROWS, PROJECT, RUN_ROOT, name, "roberta",
        {"path": str(trained / "final")}, EMBEDDING, minutes=SESSION_MINUTES,
        max_new_tokens=MAX_NEW_TOKENS, max_length=ROBERTA_MAX_LENGTH)
''')
md('''## 10. Download a debug ZIP to send for review
Run this after pausing the evaluation/training cell. The ZIP contains the latest status and events, recent exact prompts/responses with their evidence, condition protocols, available metrics, and training-loss history. It excludes model weights, optimizer files, tokens and caches.

Leave `EXPORT_ALL_PREDICTIONS_CSV` off for a quick debug package. Turn it on after a full run if you also want compact CSVs of every saved prediction; scanning many Drive files can take time. The export does not run inference or change saved results. The ZIP is saved in **FEVER_Project / exports** and offered as a browser download. Upload that downloaded ZIP in our chat.''')
code('''EXPORT_DEBUG_ZIP = False # @param {type:"boolean"}
EXPORT_ALL_PREDICTIONS_CSV = False # @param {type:"boolean"}
if not EXPORT_DEBUG_ZIP:
    print("Enable EXPORT_DEBUG_ZIP and run this cell after a session pauses or fails.")
else:
    import csv, io, zipfile, platform
    from collections import deque
    from datetime import datetime, timezone
    from google.colab import files
    export_dir = DRIVE_PROJECT / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    export_path = export_dir / ("FEVER_debug_" + stamp + ".zip")
    added = set()
    recent_predictions = []
    def add_file(z, path, archive_name):
        path = Path(path)
        if path.is_file() and archive_name not in added:
            z.write(path, archive_name)
            added.add(archive_name)
    with zipfile.ZipFile(export_path, "x", zipfile.ZIP_DEFLATED) as z:
        z.writestr("export_info.json", json.dumps({"created_utc": stamp, "experiment": EXPERIMENT_NAME,
            "condition_selected": CONDITION, "python": platform.python_version(),
            "scope": "debug subset; recent full traces, not every model response",
            "all_predictions_csv_requested": EXPORT_ALL_PREDICTIONS_CSV}, indent=2))
        add_file(z, PROJECT / "manifest.json", "project_manifest.json")
        for status_path in (RUN_ROOT / "monitor").glob("*/status.json"):
            name = status_path.parent.name
            add_file(z, status_path, "monitor/" + name + "/status.json")
            logs = sorted(status_path.parent.glob("session_*.jsonl"))
            if logs:
                with logs[-1].open() as stream:
                    tail = list(deque(stream, maxlen=2000))
                z.writestr("monitor/" + name + "/recent_events.jsonl", "".join(tail))
                for line in reversed(tail):
                    event = json.loads(line)
                    if event.get("event") == "claim_saved" and event.get("claim_id") is not None:
                        key = (name, str(event["claim_id"]))
                        if key not in recent_predictions:
                            recent_predictions.append(key)
                            if sum(n == name for n, _ in recent_predictions) >= 10:
                                break
        for condition_dir in (RUN_ROOT / "predictions").glob("*"):
            if not condition_dir.is_dir():
                continue
            add_file(z, condition_dir / "protocol.json", "conditions/" + condition_dir.name + "/protocol.json")
            if EXPORT_ALL_PREDICTIONS_CSV:
                print("Exporting compact predictions:", condition_dir.name, flush=True)
                buffer = io.StringIO()
                columns = ["id", "claim", "label", "model_label", "evidence_status", "input_tokens", "output_tokens", "wall_seconds", "error"]
                writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                for path in condition_dir.glob("*.json"):
                    if path.name == "protocol.json":
                        continue
                    writer.writerow(json.loads(path.read_text()))
                z.writestr("conditions/" + condition_dir.name + "/predictions.csv", buffer.getvalue())
        for name, claim_id in recent_predictions:
            add_file(z, RUN_ROOT / "predictions" / name / (claim_id + ".json"), "traces/" + name + "/" + claim_id + ".json")
            add_file(z, RUN_ROOT / "evidence" / ("claim_" + claim_id + ".json"), "evidence/" + claim_id + ".json")
        for path in RUN_ROOT.glob("metrics_*.json"):
            add_file(z, path, "metrics/" + path.name)
        for pattern in ["training/*/protocol.json", "training/*/complete.json",
                        "training/*/monitor/status.json", "training/*/checkpoints/checkpoint-*/trainer_state.json"]:
            for path in RUN_ROOT.glob(pattern):
                add_file(z, path, str(path.relative_to(RUN_ROOT)))
        for path in RUN_ROOT.glob("training/*/monitor/session_*.jsonl"):
            with path.open() as stream:
                tail = list(deque(stream, maxlen=300))
            z.writestr(str(path.relative_to(RUN_ROOT)), "".join(tail))
        # Include the small engineering check too, if no long evaluation has run yet.
        for path in (RUN_ROOT / "software_check").glob("*.json"):
            add_file(z, path, "software_check/" + path.name)
        for path in (RUN_ROOT / "software_check" / "evidence").glob("*.json"):
            add_file(z, path, "software_check/evidence/" + path.name)
    print("Saved debug ZIP:", export_path)
    print("Size:", round(export_path.stat().st_size / 2**20, 2), "MiB")
    print("If your browser blocks the download, get this ZIP from FEVER_Project / exports in Drive.")
    files.download(str(export_path))
''')

md('''## What this notebook does not claim
- No GPU run has been executed by the notebook author on your Colab account. Local tests cover data guards, response validation and layer selection, not cloud throughput or final accuracy.
- The 3B NF4 models and HF decoder differ from the earlier 8B Ollama experiment. New results need their own names and must not replace the old results.
- The three model families have different architectures and effective context budgets. Same passages do not imply identical retained evidence.
- This does not implement a new symbolic architecture or a Reddit study. Your professor's intended Reddit task still needs clarification.
- Free Colab is time-limited. Save Drive outputs, resume batches, and measure throughput before estimating a full-run completion date. Do not use keep-alive tricks to bypass service limits.

All source code used by a condition is in the uploaded bundle under `src/`, with SHA-256 hashes in `manifest.json`. Exact prompts, responses, retained inputs and evidence revisions are saved under your Drive run folder.
''')
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'FEVER_Colab_Starter.ipynb','provenance':[]},'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'cells':cells}
for i,c in enumerate(cells):c['id']='fever-cell-'+str(i+1).zfill(2)
folder=ROOT/'output/colab_20261104';folder.mkdir(exist_ok=True)
# Preserve the original large ZIP. Only this notebook needs to be downloaded again.
with zipfile.ZipFile(folder/'FEVER_Colab_Bundle.zip') as z:
 original_manifest=json.loads(z.read('manifest.json'))
old_hash=original_manifest['files']['src/fever_colab_runtime.py']
source=(ROOT/'colab/fever_colab_runtime.py').read_bytes();new_hash=hashlib.sha256(source).hexdigest()
payload=base64.b64encode(zlib.compress(source)).decode()
patch_text=(
 "# Install the debug runtime embedded in this notebook; the original Drive ZIP is unchanged.\n"
 "import base64, zlib, hashlib, importlib\n"
 "patch_source = zlib.decompress(base64.b64decode("+repr(payload)+"))\n"
 "expected_runtime_sha256 = "+repr(new_hash)+"\n"
 "assert hashlib.sha256(patch_source).hexdigest() == expected_runtime_sha256\n"
 "manifest_path = PROJECT / 'manifest.json'\n"
 "manifest = json.loads(manifest_path.read_text())\n"
 "if manifest['files']['src/fever_colab_runtime.py'] not in "+repr([old_hash,new_hash])+":\n"
 "    raise RuntimeError('Unexpected bundle revision; use the original supplied ZIP.')\n"
 "(PROJECT / 'src/fever_colab_runtime.py').write_bytes(patch_source)\n"
 "manifest['files']['src/fever_colab_runtime.py'] = expected_runtime_sha256\n"
 "manifest['notebook_patch'] = {'revision': 'colab-debug-v5', 'runtime_sha256': expected_runtime_sha256}\n"
 "manifest_path.write_text(json.dumps(manifest, indent=2))\n"
 "importlib.invalidate_caches()\n"
 "fever = importlib.reload(fever)\n"
 "assert fever.RUNTIME_REVISION == 'colab-debug-v5'\n"
 "EVAL_ROWS, EVAL_LABELS = fever.verify_bundle(PROJECT)\n"
 "print('Debug runtime ready. The original ZIP and earlier experiment results were preserved.')\n"
)
# Add the patch at the end of the unpack/verify cell; weights and data were checked first.
for cell in cells:
 if cell['cell_type']=='code' and 'z.extractall(PROJECT)' in ''.join(cell['source']):
  cell['source']=(''.join(cell['source'])+'\n'+patch_text).splitlines(keepends=True)
for cell in cells:
 if cell['cell_type']=='markdown' and '## 7. Run / resume' in ''.join(cell['source']):
  cell['source']+=('\n**Live output:** one progress bar, the current claim and stage, article successes/failures, token counts, prediction, invalid count, GPU memory in the status file, and a short raw-response preview. During a long stage, a heartbeat prints its elapsed time every 20 seconds. A rough ETA starts after three saved claims; it is not a guarantee.\n\nLogs and status are saved in `RUN_ROOT/monitor/CONDITION/`. Repeated infrastructure failures or ten consecutive invalid outputs pause the run for inspection. Saved failures remain errors. These messages do not bypass Colab timeouts.\n').splitlines(keepends=True)
# Place inspection controls immediately after the long-run cell.
index=next(i for i,c in enumerate(cells) if c['cell_type']=='markdown' and '## 8. Check progress' in ''.join(c['source']))
cells[index:index]=[
 {'cell_type':'markdown','metadata':{},'source':[
  '### Inspect a saved claim or recent events\n',
  'While the evaluation cell runs, watch its live output. After pausing, stopping or a failure, run this cell to inspect Drive logs and the latest saved prediction. Choose a claim ID for its exact prompt, raw response and supplied passages. Reference labels are read only after a prediction has been saved; they never enter the model prompt.\n']},
 {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':[
  'INSPECT_CLAIM_ID = "" # @param {type:"string"}\n',
  'SHOW_FULL_TRACE = False # @param {type:"boolean"}\n',
  'SHOW_REFERENCE_LABEL = False # @param {type:"boolean"}\n',
  'fever.monitor_status(RUN_ROOT, CONDITION, last_events=8)\n',
  'try:\n',
  '    fever.inspect_saved(RUN_ROOT, CONDITION, claim_id=int(INSPECT_CLAIM_ID) if INSPECT_CLAIM_ID.strip() else None,\n',
  '        show_reference=SHOW_REFERENCE_LABEL, root=PROJECT, show_full=SHOW_FULL_TRACE)\n',
  'except FileNotFoundError:\n',
  '    print("That claim has no saved prediction yet. Check the active stage in the status above.")\n']}
]
nb['metadata']['colab']['name']='FEVER_Colab_Debug.ipynb'
for i,c in enumerate(cells):c['id']='fever-cell-'+str(i+1).zfill(2)
out=folder/'FEVER_Colab_Debug.ipynb';out.write_text(json.dumps(nb,indent=1,ensure_ascii=False))
print('Created',out,'with',len(cells),'cells')
