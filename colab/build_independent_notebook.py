"""Package independent evidence selection and prediction for full FEVER dev."""
import ast
import base64
import hashlib
import json
import textwrap
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/FEVER_Project"
manifest = json.loads((ROOT / "output/independent_systems_source/manifest.json").read_text())
source = (ROOT / "colab/independent_systems.py").read_bytes()
base = json.loads((OUT / "FEVER_Train_Last6.ipynb").read_text())
baseline = json.loads((ROOT / "reports/part2/roberta_full_training_analysis_20261106/last3_protocol.json").read_text())
cells = []
def md(text):
    cells.append(dict(cell_type="markdown",metadata={},source=text.splitlines(keepends=True)))
def code(text):
    text=textwrap.dedent(text).strip()
    ast.parse(text)
    cells.append(dict(cell_type="code",metadata={},source=text.splitlines(keepends=True),execution_count=None,outputs=[]))
md("""# FEVER - independent systems on all 19,998 claims
Select T4 GPU, then Runtime > Run all. Inference is enabled.

Each system performs evidence selection and classification independently.
Qwen, Llama, Mistral and Gemma generate their own Wikipedia search query,
search the full FEVER snapshot title index, read all sentence text from
their top 10 search results in batches, and select excerpts themselves.
Each batch selects up to 3 excerpts. The same model recursively reduces
these selections to at most 3 final excerpts, then returns claim-part verdicts.
Prompts, responses, searches and selected evidence are saved per claim.

RoBERTa uses the claim as its title-search query and ranks excerpts with its
largest Supports/Refutes probability. This is a heuristic using its trained
classification head, not a separately trained relevance model. Its selected
excerpts enter the same frozen 512-token classifier. Truncation is recorded.

Search candidates still depend on title search and its top-10 limit. Each
model selects its own evidence within its search results. The full snapshot
is available to search; models do not read every Wikipedia article per claim.

The model queue starts with Qwen. A session runs for up to 180 minutes, then
saves progress. Run all again to continue. Completed claims and systems are
skipped. Three consecutive invalid outputs pause execution for debugging.
The full scope remains 19,998 per model. Small partial scores are marked partial.

Source data require about 1.6 GB compressed, plus several GB for the SQLite
index and model weights in Colab. Model metadata prints weight download sizes
before loading. We load one classifier at a time in 4-bit NF4.
Llama and Gemma may require accepting their licenses on Hugging Face and
providing HF_TOKEN in Colab Secrets. No retraining runs.
""")
for cell in base["cells"]:
    if cell["cell_type"]=="code":
        code("".join(cell["source"]))
        if len([c for c in cells if c["cell_type"]=="code"])==3:break
md("## Connect the independent runtime")
code("import base64, importlib, shutil, time, zlib\n"
     "independent_source = zlib.decompress(base64.b64decode("+repr(base64.b64encode(zlib.compress(source)).decode())+"))\n"
     "assert hashlib.sha256(independent_source).hexdigest() == "+repr(hashlib.sha256(source).hexdigest())+"\n"
     "(PROJECT / 'src/independent_systems.py').write_bytes(independent_source)\n"
     "import independent_systems as independent\nindependent = importlib.reload(independent)\n"
     "ROWS = [{'id': r['id'], 'claim': r['claim']} for r in EVAL_ROWS]\n"
     "print('Independent selection enabled. Target:',len(ROWS),'claims per system.')")
md("## Download and index the full Wikipedia archive\nThe original FEVER archive downloads directly from fever.ai and is cached in Drive. No manual upload is needed. Indexing logs each source shard.")
code("SOURCE_MANIFEST = "+repr({k:manifest[k] for k in ('source','archive_bytes','archive_sha256')})+"\n"+textwrap.dedent('''
import requests
from tqdm.auto import tqdm
archive = Path("/content/fever_full_wikipedia.zip")
cached = DRIVE_PROJECT / "FEVER_Full_Wikipedia.zip"
if not archive.exists() and cached.is_file():
    print("Copying cached Wikipedia archive from Drive.", flush=True)
    shutil.copyfile(cached, archive)
if not archive.exists():
    temporary = archive.with_suffix(".pending")
    offset = temporary.stat().st_size if temporary.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    url = "https://fever.ai/download/fever/wiki-pages.zip"
    print("Downloading the full FEVER Wikipedia archive, about 1.6 GiB.", flush=True)
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if offset and (response.status_code != 206 or
                       not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
            raise ValueError("Archive server did not honor resumable download")
        with temporary.open("ab" if offset else "wb") as destination:
            with tqdm(total=SOURCE_MANIFEST["archive_bytes"], initial=offset,
                      unit="B", unit_scale=True, desc="FEVER Wikipedia download") as bar:
                for block in response.iter_content(2**20):
                    if block:
                        destination.write(block)
                        bar.update(len(block))
    if temporary.stat().st_size != SOURCE_MANIFEST["archive_bytes"]:
        raise ValueError("Incomplete Wikipedia archive; run again to resume")
    temporary.rename(archive)
if fever.sha_file(archive) != SOURCE_MANIFEST["archive_sha256"]:
    raise ValueError("Full Wikipedia archive hash mismatch")
if not cached.exists():
    print("Caching the verified source archive in Drive.", flush=True)
    staging = cached.with_name(cached.name + "." + str(time.time_ns()) + ".pending")
    shutil.copyfile(archive, staging)
    staging.replace(cached)
DATABASE = independent.build_corpus(archive, PROJECT / "full_wikipedia_index",
                                    SOURCE_MANIFEST["archive_sha256"])
'''))
md("## Run or resume the model queue\nEach system has separate evidence and prediction files. Measured throughput and ETA appear in the live logs. LLM downloads happen in Colab.")
code("EXPECTED_ROBERTA = "+repr(baseline)+"\n"+textwrap.dedent('''
MODEL_QUEUE = ["qwen", "llama", "mistral", "gemma", "roberta"]
SESSION_MINUTES = 180
OUTPUT = DRIVE_PROJECT / "runs/independent_systems_dev19998_v1"
CHECKPOINT = (DRIVE_PROJECT / "runs/roberta_full_unfreeze_v1/training/roberta_full_last3/final")
training_protocol = json.loads((CHECKPOINT.parent / "protocol.json").read_text())
if training_protocol != EXPECTED_ROBERTA:
    raise ValueError("RoBERTa training protocol differs from the verified last-3 baseline")
if fever.checkpoint_identity(CHECKPOINT).get("model.safetensors") is None:
    raise ValueError("The trained last-3 RoBERTa checkpoint is missing")
best = DRIVE_PROJECT / "runs/roberta_full_unfreeze_v1/training/roberta_full_last3/checkpoints/checkpoint-11000"
final_identity = fever.checkpoint_identity(CHECKPOINT)
best_identity = fever.checkpoint_identity(best)
weights = [name for name in final_identity if name.endswith((".safetensors", ".bin"))]
if not weights or any(final_identity[name] != best_identity.get(name) for name in weights):
    raise ValueError("RoBERTa final weights do not match the selected checkpoint")
started = time.monotonic()
for family in MODEL_QUEUE:
    remaining = SESSION_MINUTES - (time.monotonic() - started) / 60
    if remaining < 1:
        print("Session budget reached. Run all again to resume.", flush=True)
        break
    protocol_path = OUTPUT / family / "protocol.json"
    if protocol_path.is_file():
        old_protocol = json.loads(protocol_path.read_text())
        metadata = old_protocol["model"] if family != "roberta" else None
    else:
        metadata = fever.remote_metadata(independent.MODELS[family]) if family != "roberta" else None
    print("STARTING / RESUMING:", family, "| target 19,998", flush=True)
    status = independent.run(ROWS, PROJECT, OUTPUT, DATABASE,
        SOURCE_MANIFEST["archive_sha256"], family, metadata, CHECKPOINT, minutes=remaining)
    print("System status:", family, status, flush=True)
    if status != "complete":
        break
'''))
md("""## Saved results
Only systems with 19,998 records have final scores. Invalid outputs count as
errors. Full prompts and responses remain in each model's prediction files.
Reference labels enter only the reporting step below.
""")
code('''from IPython.display import display
import pandas as pd
summary = independent.report(ROWS, EVAL_LABELS, OUTPUT, MODEL_QUEUE)
display(pd.DataFrame([{k: v for k, v in item.items() if k != "per_class"} for item in summary]))
print("Saved reports and full traces:", OUTPUT)
''')
for i,cell in enumerate(cells):cell["id"]=f"independent-{i+1}"
metadata=base["metadata"]
metadata["colab"]["name"]="FEVER_Independent_Systems_19998.ipynb"
target=OUT/"FEVER_Independent_Systems_19998.ipynb"
target.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,metadata=metadata,cells=cells),indent=1))
print("Created",target,len(cells),"cells")
