"""Build the full paired evaluation notebook; no training or inference here."""
import ast,base64,hashlib,json,runpy,sys,textwrap,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT_MODE='--snapshot' in sys.argv
runpy.run_path(str(ROOT/'colab/build_notebook.py'))
base=json.loads((ROOT/'output/colab_20261104/FEVER_Colab_Debug.ipynb').read_text());cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':textwrap.dedent(s).strip().splitlines(keepends=True)})
def code(s):
 s=textwrap.dedent(s).strip();ast.parse(s)
 cells.append({'cell_type':'code','metadata':{},'source':s.splitlines(keepends=True),'execution_count':None,'outputs':[]})
def reuse(snippet):code(''.join(next(c['source'] for c in base['cells'] if c['cell_type']=='code' and snippet in ''.join(c['source']))))
analysis=ROOT/'reports/part2/roberta_full_training_analysis_20261106'
selection=json.loads((analysis/'validation_selection.json').read_text())
expected={str(r['last_blocks']):r['state_sha256'] for r in selection['candidates']}
source=(ROOT/'colab/paired_roberta_evaluation.py').read_bytes()
payload=base64.b64encode(zlib.compress(source)).decode();digest=hashlib.sha256(source).hexdigest()
md('''# FEVER — original versus trained RoBERTa on all 19,998 claims
**Select T4 GPU, then Runtime → Run all. Evaluation is enabled; no training runs.**

The last-3-block model was chosen using the separate training-pool validation
set: macro-F1 **69.42%**, versus 68.24% (last 2) and 66.30% (last 1). Those are
validation F1 scores, not benchmark accuracy. This notebook rechecks selection
and verifies that the final weights match the selected checkpoint at step 11,000.

For every official FEVER development claim, prepare one temporary full-article
RAG database and pass the same retrieved passages to the original and selected
RoBERTa models. Both use the same 512-token limit, and retained input text is
checked for equality. Top 10 existing ranked Wikipedia URLs → first 5 articles
→ 200-token chunks → MiniLM embeddings → top 30 vector candidates → 7,600-character
context. Full articles and vectors are deleted after each claim. Only source
metadata, supplied passages and model traces are retained.

MiniLM embeddings run in FP32 on the GPU to reduce the previous CPU bottleneck.
This is a separately recorded evaluation protocol; floating-point differences
from previous CPU embeddings are possible. Both classifier conditions share the
new evidence exactly. No speedup is claimed before measuring it.

The target is all **19,998**, not the 90-claim diagnostic. The run can require
multiple sessions because of live article fetching. A measured ETA appears
after several completed claims. Use Run all again to resume the same folder.
The previously inspected official development set is not an untouched hidden test.
''')
md('''## 1. Check GPU''');reuse('print("Python:", sys.version.split()[0])')
md('''## 2. Install runtime libraries''');reuse('packages = ["transformers')
md('''## 3. Connect Drive and verify the original bundle
Your training notebook, model checkpoints and earlier predictions are preserved.''');reuse('z.extractall(PROJECT)')
code("# Install the checksummed paired-evaluation adapter.\n"+
 "import base64, hashlib, importlib, zlib\n"+
 "paired_source = zlib.decompress(base64.b64decode("+repr(payload)+"))\n"+
 "assert hashlib.sha256(paired_source).hexdigest() == "+repr(digest)+"\n"+
 "(PROJECT / 'src/paired_roberta_evaluation.py').write_bytes(paired_source)\n"+
 "import paired_roberta_evaluation as paired\npaired = importlib.reload(paired)\nprint('Paired evaluation adapter verified.')")
md('''## 4. Recheck validation selection and set evaluation output
Selection reads only the training runs' validation histories. The expected file
hashes were fixed when we selected the model, before this full evaluation.''')
code('EXPECTED_STATE_HASHES = '+repr(expected)+'\n'+textwrap.dedent('''
TRAINING_ROOT = DRIVE_PROJECT / "runs/roberta_full_unfreeze_v1/training"
SELECTION = paired.select_validation(TRAINING_ROOT)
for candidate in SELECTION["candidates"]:
    if candidate["state_sha256"] != EXPECTED_STATE_HASHES[str(candidate["last_blocks"])]:
        raise ValueError("Training history changed since model selection; inspect before evaluating")
assert SELECTION["winner"]["last_blocks"] == 3
import pandas as pd
display(pd.DataFrame([{"Unfrozen blocks": r["last_blocks"],
    "Validation macro-F1 (%)": 100*r["validation_macro_f1"],
    "Validation loss": r["validation_loss"], "Selected step": r["selected_step"]}
    for r in SELECTION["candidates"]]).round(4))
EVALUATION_NAME = "roberta_paired_official_dev_v1" # @param {type:"string"}
SESSION_MINUTES = 180 # @param {type:"integer"}
RUN_PAIRED_EVALUATION = True # @param {type:"boolean"}
if not 1 <= SESSION_MINUTES <= 300: raise ValueError("Use 1–300 minutes")
EVALUATION_OUT = DRIVE_PROJECT / "runs" / EVALUATION_NAME
EVALUATION_OUT.mkdir(parents=True, exist_ok=True)
print("Selected model: last 3 blocks; validation macro-F1:", round(100*SELECTION["winner"]["validation_macro_f1"],2), "%")
print("Target: all", len(EVAL_ROWS), "claims, TWO models, same retained RAG input.")
print("Results:", EVALUATION_OUT)
EMBEDDING = fever.remote_metadata("sentence-transformers/all-MiniLM-L6-v2",
    requested_revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
print("Only this ~0.08 GiB embedding model may download; both RoBERTa checkpoints already exist.")
'''))
md('''## 5. Run / resume both models — enabled
Live logs show the claim, article fetching, embedding, both predictions, invalid
pair count, progress and measured ETA. There is no JSON-generating LLM call here.
Evidence failures remain INVALID for both models; they are not converted to NEI.
Three consecutive evidence failures pause the run for review.

Initial model-hash verification reads the saved weights from Drive and can take
a little time. The 180-minute budget pauses after finishing the current pair.
Completed predictions stay immutable; a new session resumes only pending pairs.''')
code('''CONDITIONS = ["roberta_original_rag512", "roberta_last3_rag512"]
if RUN_PAIRED_EVALUATION:
    print("Verifying selected weights and loading the paired models…", flush=True)
    CONDITIONS = paired.run_pairs(EVAL_ROWS, PROJECT, EVALUATION_OUT, SELECTION,
        EMBEDDING, minutes=SESSION_MINUTES)
else:
    print("Evaluation disabled.")
''')
md('''## 6. Progress and final comparison
Final metrics appear only when both systems cover all 19,998 IDs. INVALID stays
in the denominator. Metrics are label accuracy, macro-F1, per-class precision,
recall and F1, confusion matrices, and paired counts of fixes versus new errors.
Live revision citations do not constitute an official strict FEVER evidence score.
''')
code('''for name in CONDITIONS:
    folder = EVALUATION_OUT / "predictions" / name
    count = sum(p.name != "protocol.json" for p in folder.glob("*.json"))
    print(name, count, "/", len(EVAL_ROWS))
METRICS = paired.report(PROJECT, EVALUATION_OUT, CONDITIONS)
if len(METRICS) == 2:
    display(pd.DataFrame([{"System": name, "Claims": m["n"],
        "Accuracy (%)": 100*m["accuracy"], "Macro-F1 (%)": 100*m["macro_f1"],
        "Invalid": m["invalid"]} for name,m in METRICS.items()]).round(2))
    for name,m in METRICS.items():
        print(name)
        display(pd.DataFrame([{"Class": label, **{k+" (%)": 100*v for k,v in values.items()}}
            for label,values in m["per_class"].items()]).round(2))
        display(pd.DataFrame(m["confusion_matrix"], index=fever.LABELS,
            columns=fever.LABELS+["INVALID"]))
fever.monitor_status(EVALUATION_OUT, "paired_roberta", last_events=5)
''')
md('''## 7. Inspect a saved pair
An empty ID selects the most recently written prediction. Reference labels are
shown only after the model prediction has been saved. These labels never enter
either model's input. RoBERTa has probabilities and retained tokenized input,
rather than a generated textual explanation.''')
code('''INSPECT_CLAIM_ID = "" # @param {type:"string"}
for name in CONDITIONS:
    print("\\n", name)
    fever.inspect_saved(EVALUATION_OUT, name,
        claim_id=int(INSPECT_CLAIM_ID) if INSPECT_CLAIM_ID.strip() else None,
        show_reference=True, root=PROJECT, show_full=True)
''')
filename='FEVER_Evaluate_19998.ipynb'
if SNAPSHOT_MODE:
 filename='FEVER_Evaluate_Snapshot_19998.ipynb'
 snapshot_folder=ROOT/'output/fever_snapshot_rag_20261106'
 manifest_bytes=(snapshot_folder/'snapshot_manifest.json').read_bytes()
 snapshot_manifest=json.loads(manifest_bytes)
 snapshot_source=(ROOT/'colab/snapshot_rag.py').read_bytes()
 snapshot_payload=base64.b64encode(zlib.compress(snapshot_source)).decode()
 cells[0]['source']=textwrap.dedent('''# FEVER — snapshot RAG evaluation on all 19,998 claims
 **Stop the old live-download evaluation. Open this notebook with T4 GPU and choose Runtime → Run all. Evaluation is enabled. No retraining.**

 The article source is now the Wikipedia text distributed with FEVER. Existing
 claim-only retrieval rankings select the same top five page IDs; no reference
 label or annotated evidence is used to select text. All available article text
 in those snapshot records is read, chunked and embedded into a temporary RAG
 database for each claim. This uses the dataset's supplied text, not the full
 current Wikipedia website with its later edits, tables and metadata.

 The source package contains all 24,319 selected pages, with none missing or
 empty. It is already uploaded to FEVER_Project. Article preparation makes zero
 Wikipedia HTTP calls. The unchanged MiniLM embedding model may download once
 (~0.08 GiB) if not cached. Static source data remain available; per-claim working
 articles, chunks and vector databases are deleted after use. Saved traces retain
 supplied passages, source-record hashes and exact character spans. The FEVER
 dump does not supply per-page revision IDs, so no revision IDs are invented.

 Both the original RoBERTa and the validation-selected last-3 model receive the
 exact same retrieved context and 512-token budget. The last-3 model's 69.42% was
 validation macro-F1, not final accuracy. Model weights and previous results are
 unchanged. This evidence-source change uses a new results folder and starts a
 separate comparison; it does not reuse live-Wikipedia predictions.

 A local 20-claim check verified evidence preparation and citation spans with no
 network calls. This notebook first runs a resumable 10-pair execution check, then
 continues toward all 19,998 within the session budget. These checks do not prove
 accuracy or completion speed. Actual Colab progress and ETA are measured during
 the run. Use Run all again after a pause to resume saved snapshot results.
 ''').strip().splitlines(keepends=True)
 setup=(
  "import base64, hashlib, importlib, zlib, zipfile, json\n"
  "snapshot_source = zlib.decompress(base64.b64decode("+repr(snapshot_payload)+"))\n"
  "assert hashlib.sha256(snapshot_source).hexdigest() == "+repr(hashlib.sha256(snapshot_source).hexdigest())+"\n"
  "(PROJECT / 'src/snapshot_rag.py').write_bytes(snapshot_source)\n"
  "import snapshot_rag\nsnapshot_rag = importlib.reload(snapshot_rag)\n"
  "SNAPSHOT_FOLDER = PROJECT / 'fever_snapshot'\nSNAPSHOT_FOLDER.mkdir(exist_ok=True)\n"
  "snapshot_archive = DRIVE_PROJECT / 'FEVER_Wikipedia_Snapshot.zip'\n"
  "if not snapshot_archive.is_file(): raise FileNotFoundError('FEVER_Wikipedia_Snapshot.zip must be in FEVER_Project')\n"
  "with zipfile.ZipFile(snapshot_archive) as archive:\n"
  "    if set(archive.namelist()) != {'snapshot.sqlite','snapshot_manifest.json'}: raise ValueError('Unexpected snapshot ZIP contents')\n"
  "    manifest_bytes = archive.read('snapshot_manifest.json')\n"
  "    if hashlib.sha256(manifest_bytes).hexdigest() != "+repr(hashlib.sha256(manifest_bytes).hexdigest())+": raise ValueError('Snapshot manifest differs from the verified export')\n"
  "    for name in ['snapshot.sqlite','snapshot_manifest.json']:\n"
  "        content = archive.read(name)\n        target = SNAPSHOT_FOLDER / name\n"
  "        if target.exists() and target.read_bytes() != content: raise ValueError('Existing source snapshot differs')\n"
  "        if not target.exists(): target.write_bytes(content)\n"
  "probe = snapshot_rag.SnapshotCorpus(SNAPSHOT_FOLDER, fever.sha_file(PROJECT / 'data/eval_inputs.jsonl'))\n"
  "try: print('Snapshot verified:',probe.manifest['packaged_pages'],'selected full records; zero Wikipedia downloads per claim.')\n"
  "finally: probe.close()\n")
 insertion=next(i for i,c in enumerate(cells) if c['cell_type']=='markdown' and '## 4.' in ''.join(c['source']))
 cells[insertion:insertion]=[
  {'cell_type':'markdown','metadata':{},'source':['## 3b. Verify and connect FEVER Wikipedia source data\nThe ZIP is already in Drive; no paths need typing. This verifies source and retrieval-input hashes before use.\n']},
  {'cell_type':'code','metadata':{},'source':setup.splitlines(keepends=True),'execution_count':None,'outputs':[]}]
 for cell in cells:
  text=''.join(cell['source'])
  if cell['cell_type']=='code' and 'EVALUATION_NAME =' in text:
   text=text.replace('roberta_paired_official_dev_v1','roberta_paired_snapshot_dev_v1')
  if cell['cell_type']=='code' and 'CONDITIONS = [' in text and 'paired.run_pairs(' in text:
   text='''CONDITIONS = ["roberta_original_rag512", "roberta_last3_rag512"]
if RUN_PAIRED_EVALUATION:
    import time
    started = time.monotonic()
    corpus = snapshot_rag.SnapshotCorpus(SNAPSHOT_FOLDER, fever.sha_file(PROJECT / "data/eval_inputs.jsonl"))
    try:
        counts = [sum(p.name != "protocol.json" for p in (EVALUATION_OUT / "predictions" / name).glob("*.json")) for name in CONDITIONS]
        if min(counts) < 10:
            print("Execution check: prepare snapshot RAG and evaluate the first 10 pairs; target remains all 19,998.", flush=True)
            paired.run_pairs(EVAL_ROWS, PROJECT, EVALUATION_OUT, SELECTION, EMBEDDING,
                minutes=SESSION_MINUTES, max_new_claims=10-min(counts), corpus=corpus)
            print("Execution check saved. These are not final accuracy results.", flush=True)
        remaining = SESSION_MINUTES - (time.monotonic()-started)/60
        if remaining >= 1:
            CONDITIONS = paired.run_pairs(EVAL_ROWS, PROJECT, EVALUATION_OUT, SELECTION,
                EMBEDDING, minutes=remaining, corpus=corpus)
        else:
            print("Session budget reached. Run all again to resume.")
    finally:
        corpus.close()
else:
    print("Evaluation disabled.")
'''
  if cell['cell_type']=='markdown' and '## 5.' in text:
   text='''## 5. Run / resume snapshot RAG evaluation — enabled
Reads the selected full FEVER article records locally, builds/deletes a temporary
RAG database per claim and evaluates both unchanged models on identical retained
input. No live article fetching or LLM generation occurs. Progress, source reading,
embedding, predictions and rough ETA are logged. The first 10 paired outputs form
an execution check, then the same experiment continues toward all 19,998.
Three consecutive preparation failures stop for review; failures are not relabeled NEI.
Initial model hashing/loading can take time; later sessions skip completed pairs.
'''
  if cell['cell_type']=='markdown' and '## 6.' in text:
   text=text.replace('Live revision citations do not constitute an official strict FEVER evidence score.',
    'Snapshot record/character-span citations are retained; this reports label metrics, not an official strict FEVER evidence score.')
  cell['source']=text.splitlines(keepends=True)
for i,c in enumerate(cells):c['id']='fever-paired-eval-'+str(i+1).zfill(2)
for c in cells:
 if c['cell_type']=='code':ast.parse(''.join(c['source']))
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':filename,'provenance':[],'gpuType':'T4'},
 'accelerator':'GPU','kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'cells':cells}
out=ROOT/'output/FEVER_Project'/filename;out.write_text(json.dumps(nb,indent=1,ensure_ascii=False))
print('Created:',out,'cells:',len(cells))
