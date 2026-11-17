"""Build a paired evidence diagnostic using the completed snapshot evaluation."""
import ast
import hashlib
import json
import sqlite3
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/FEVER_Project"
INPUTS = ROOT / "output/colab_20261104/data/eval_inputs.jsonl"
GOLD = ROOT / "data/official_fever/shared_task_dev.jsonl"
rows = [json.loads(line) for line in INPUTS.open()]
references = {r["id"]: r for r in (json.loads(line) for line in GOLD.open())}
connection = sqlite3.connect((ROOT / "data/processed/wiki_fts.db").as_uri() + "?mode=ro", uri=True)
records = []
for row in rows:
    ref = references[row["id"]]
    if ref["label"] == "NOT ENOUGH INFO":
        continue
    groups = []
    for group in ref["evidence"]:
        keys = list(dict.fromkeys((unicodedata.normalize('NFC', item[2]), item[3]) for item in group))
        sentences = []
        for page, sid in keys:
            sentence = connection.execute(
                "SELECT text FROM page_sentences WHERE page_id=? AND sentence_id=?", (page, sid)
            ).fetchone()
            if sentence is None or not sentence[0].strip():
                raise ValueError(f"Missing annotated sentence for {row['id']}: {page}, {sid}")
            sentences.append(dict(page_id=page, sentence_id=sid, text=sentence[0]))
        if sentences not in groups:
            groups.append(sentences)
    if not groups or not groups[0]:
        raise ValueError("No annotated evidence")
    records.append(dict(id=row["id"], claim=row["claim"], label=ref["label"],
                        groups=groups, ranked_articles=row["ranked_articles"]))
connection.close()
assert len(records) == 13332
data = OUT / "FEVER_Annotated_Evidence_Diagnostic.jsonl"
data.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
digest = hashlib.sha256(data.read_bytes()).hexdigest()
base = json.loads((OUT / "FEVER_Train_Last6.ipynb").read_text())
cells = []
def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.splitlines(keepends=True)))
def code(text):
    ast.parse(text)
    cells.append(dict(cell_type="code", metadata={}, source=text.splitlines(keepends=True),
                      execution_count=None, outputs=[]))
md("""# FEVER evidence diagnostic
Select T4 GPU and Runtime > Run all. Inference is enabled.

Compare the same fine-tuned last-3 model on all 13,332 Supports/Refutes claims
from the completed 19,998-claim snapshot run. Reuse its saved RAG predictions
and supply the first complete annotation group as an oracle-evidence input.
The ordering rule is fixed before inspecting predictions.

NEI has no annotated supporting/refuting evidence, so it is outside this
diagnostic. This measures the effect of providing an annotated evidence set
instead of retrieved passages, including differences in evidence length.
It is an oracle diagnostic, not a deployable retrieval method.

Use the same 512-token budget and checkpoint. Traces retain the claim,
annotated sentences, both predictions, token counts, probabilities and exact
retained inputs. Save each completed claim to Drive and resume after a pause.
""")
for cell in base["cells"]:
    if cell["cell_type"] == "code":
        code("".join(cell["source"]))
        if len([c for c in cells if c["cell_type"] == "code"]) == 3:
            break
md("## Verify inputs and the completed run\nThe diagnostic data file is alongside this notebook in FEVER_Project.")
code(f'''import time, math
from IPython.display import display
import pandas as pd
from tqdm.auto import tqdm

data_path = DRIVE_PROJECT / "FEVER_Annotated_Evidence_Diagnostic.jsonl"
if fever.sha_file(data_path) != {digest!r}:
    raise ValueError("Annotated evidence file hash mismatch")
DIAGNOSTIC_ROWS = fever.read_rows(data_path)
assert len(DIAGNOSTIC_ROWS) == 13332
expected_claims = {{row["id"]: row["claim"] for row in EVAL_ROWS}}
if any(row["claim"] != expected_claims.get(row["id"])
       or row["label"] != EVAL_LABELS.get(str(row["id"])) for row in DIAGNOSTIC_ROWS):
    raise ValueError("Diagnostic claims or reference labels differ from the evaluation bundle")
prior = DRIVE_PROJECT / "runs/roberta_paired_snapshot_dev_v1"
condition = "roberta_last3_rag512"
selection = json.loads((prior / "validation_selection.json").read_text())
winner = selection["winner"]
assert winner["last_blocks"] == 3
model_path = Path(winner["final_path"])
best_path = Path(winner["best_checkpoint_path"])
model_identity = fever.checkpoint_identity(model_path)
best_identity = fever.checkpoint_identity(best_path)
weight_names = [name for name in model_identity if name.endswith((".safetensors", ".bin"))]
if not weight_names or any(model_identity[name] != best_identity.get(name) for name in weight_names):
    raise ValueError("Selected model weights differ from selected checkpoint")
original_protocol = json.loads((prior / "pairs/protocol.json").read_text())
if original_protocol["models"][condition] != model_identity:
    raise ValueError("Model differs from the completed RAG evaluation")
if original_protocol["max_length"] != 512:
    raise ValueError("Prior token budget differs")
prior_model_protocol = json.loads((prior / "predictions" / condition / "protocol.json").read_text())
if prior_model_protocol != {{"paired_protocol_identity": fever.signature(original_protocol), "condition": condition}}:
    raise ValueError("Prior prediction protocol differs")
prior_prediction_identity = fever.signature(prior_model_protocol)
out = DRIVE_PROJECT / "runs/roberta_evidence_diagnostic_v1"
protocol = {{
    "diagnostic_version": 1,
    "diagnostic": "first annotated group versus previously saved snapshot RAG prediction",
    "data_sha256": fever.sha_file(data_path), "model": model_identity,
    "prior_protocol_sha256": fever.sha_file(prior / "pairs/protocol.json"),
    "max_length": 512, "claims": 13332, "environment": fever.environment(),
    "runtime_sha256": fever.sha_file(PROJECT / "src/fever_colab_runtime.py"),
}}
identity = fever.freeze(out, protocol)
SESSION_MINUTES = 180
print("Target:", len(DIAGNOSTIC_ROWS), "claims. Results:", out)
print("Fixed model:", model_path)
''')
md("## Run or resume\nProgress shows throughput and ETA. Errors remain visible and prevent a final paired report.")
code('''started = time.monotonic()
model = None
try:
    results = out / "predictions"
    results.mkdir(exist_ok=True)
    pending = []
    for row in DIAGNOSTIC_ROWS:
        path = results / (str(row["id"]) + ".json")
        if path.is_file():
            saved = json.loads(path.read_text())
            if saved["identity"] != identity or saved["id"] != row["id"] or saved["claim"] != row["claim"]:
                raise ValueError("Cached diagnostic result differs")
        else:
            pending.append(row)
    print("Completed:", len(DIAGNOSTIC_ROWS) - len(pending), "Pending:", len(pending), flush=True)
    if pending:
        model = fever.RobertaVerifier(model_path, max_length=512)
    for index, row in enumerate(tqdm(pending, desc="Annotated evidence inference", unit="claim"), 1):
        if time.monotonic() - started >= SESSION_MINUTES * 60:
            print("Session budget reached. Run all to resume.", flush=True)
            break
        path = results / (str(row["id"]) + ".json")
        try:
            old = json.loads((prior / "predictions" / condition / path.name).read_text())
            if (old["id"] != row["id"] or old["claim"] != row["claim"]
                    or old.get("protocol_identity") != prior_prediction_identity
                    or old["label"] not in fever.LABELS):
                raise ValueError("Prior prediction is invalid or claim differs")
            group = row["groups"][0]
            context = {"status": "READY", "payload": {"passages": [{"evidence": group}]}}
            oracle = model.predict(row, context)
            import unicodedata
            selected_pages = {unicodedata.normalize("NFC", p["page_id"]) for p in row["ranked_articles"][:5]}
            page_hit = any(all(s["page_id"] in selected_pages for s in g) for g in row["groups"])
            trace = dict(id=row["id"], claim=row["claim"], gold_label=row["label"],
                         identity=identity, annotated_sentences=group,
                         any_complete_gold_group_in_top5_pages=page_hit,
                         original_rag=old, oracle=oracle)
            fever.write_once(path, trace)
        except Exception as exc:
            fever.write_once(out / "errors" / (str(row["id"]) + "_" + str(time.time_ns()) + ".json"),
                             dict(id=row["id"], error=repr(exc), created_utc=fever.now()))
            raise
        if index <= 3 or index % 250 == 0:
            print(row["id"], "| FEVER", row["label"], "| RAG", old["label"],
                  "| annotated", oracle["label"], "| truncated", oracle["truncated"], flush=True)
finally:
    if model is not None:
        model.close()
''')
md("""## Paired results and traces
Accuracy and macro-F1 below cover only Supports/Refutes reference claims.
The model may still predict NEI, counted as an error. The paired outcome
counts show where replacing retrieved passages helped or hurt.
A remaining error with annotated evidence may reflect classification,
truncation or annotation limitations. Page coverage alone cannot establish
sentence coverage. Final statistics require all 13,332 saved results.
""")
code('''from sklearn.metrics import accuracy_score, precision_recall_fscore_support
traces = []
for row in DIAGNOSTIC_ROWS:
    path = out / "predictions" / (str(row["id"]) + ".json")
    if path.is_file():
        trace = json.loads(path.read_text())
        if trace["identity"] != identity:
            raise ValueError("Mixed diagnostic protocols")
        traces.append(trace)
complete = len(traces) == len(DIAGNOSTIC_ROWS)
print("FINAL" if complete else "PARTIAL", len(traces), "/", len(DIAGNOSTIC_ROWS))
if traces:
    gold = [r["gold_label"] for r in traces]
    rows = []
    for key, title in [("original_rag", "Saved RAG"), ("oracle", "Annotated evidence")]:
        pred = [r[key]["label"] for r in traces]
        precision, recall, f1, support = precision_recall_fscore_support(
            gold, pred, labels=["SUPPORTS", "REFUTES"], zero_division=0)
        rows.append({"Input": title, "Accuracy (%)": 100 * accuracy_score(gold, pred),
                     "Macro-F1 on S/R (%)": 100 * f1.mean(),
                     "Supports recall (%)": 100 * recall[0], "Refutes recall (%)": 100 * recall[1]})
    display(pd.DataFrame(rows))
    paired = {}
    for trace in traces:
        a = trace["original_rag"]["label"] == trace["gold_label"]
        b = trace["oracle"]["label"] == trace["gold_label"]
        name = {(True, True): "Both correct", (False, True): "Only annotated correct",
                (True, False): "Only RAG correct", (False, False): "Both wrong"}[a, b]
        paired[name] = paired.get(name, 0) + 1
    print("Paired outcomes:", paired)
    print("Annotated input truncated:", sum(t["oracle"]["truncated"] for t in traces))
    print("At least one complete annotated group has its pages in top 5:",
          sum(t["any_complete_gold_group_in_top5_pages"] for t in traces))
    if complete:
        fever.write_once(out / "final_summary.json", dict(n=len(traces), protocol_identity=identity,
                         metrics=rows, paired=paired))
    for trace in traces[:3]:
        print(json.dumps(trace, indent=2, ensure_ascii=False))
''')
for index, cell in enumerate(cells):
    cell["id"] = f"evidence-diagnostic-{index}"
notebook = dict(nbformat=4, nbformat_minor=5, metadata=base["metadata"], cells=cells)
notebook["metadata"]["colab"]["name"] = "FEVER_Evidence_Diagnostic.ipynb"
target = OUT / "FEVER_Evidence_Diagnostic.ipynb"
target.write_text(json.dumps(notebook, indent=1))
print("Created", target, "and", data, "claims", len(records), "bytes", data.stat().st_size)
