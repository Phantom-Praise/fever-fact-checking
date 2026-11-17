"""Build a self-contained Climate-FEVER notebook and a small upload bundle."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/Climate_FEVER_Project"


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("climate_fever.py", "independent_systems.py", "fever_colab_runtime.py"):
        files["src/" + name] = (ROOT / "colab" / name).read_bytes()
    for name in ("inputs.json", "labels.json", "references.json", "manifest.json"):
        files["data/" + name] = (ROOT / "data/climate_fever/prepared" / name).read_bytes()
    files["source/climate-fever.jsonl"] = (ROOT / "data/climate_fever/source/climate-fever.jsonl").read_bytes()
    files["source/provenance.json"] = (ROOT / "data/climate_fever/source/provenance.json").read_bytes()
    files["parts_prompt.txt"] = b"Climate-FEVER classification uses climate_fever.PROMPT.\n"
    files["README.md"] = (ROOT / "docs/CLIMATE_FEVER_MIGRATION.md").read_bytes()
    files["bundle_manifest.json"] = json.dumps({name: hashlib.sha256(data).hexdigest()
                                               for name, data in files.items()}, indent=2).encode()
    archive = OUT / "Climate_FEVER_Bundle.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for name, data in files.items():
            zipped.writestr(name, data)
    cells = []

    def md(text):
        cells.append(dict(cell_type="markdown", metadata={}, source=text.splitlines(keepends=True)))

    def code(text):
        text = textwrap.dedent(text).strip()
        ast.parse(text)
        cells.append(dict(cell_type="code", metadata={}, source=text.splitlines(keepends=True),
                          execution_count=None, outputs=[]))

    md("""# Climate-FEVER — five independent systems
1,535 claims. Separate results from every historical FEVER experiment.

**Start here:** select T4 GPU and Run all. The default performs setup only:
no model inference, training or large Wikipedia download. The small dataset and
runtime are embedded in this notebook; uploading the ZIP separately is optional.
Everything is saved in MyDrive/Climate_FEVER_Project.

After setup, use RUN_MODE='pilot' for the first 12 claims with Qwen. Inspect its
saved prompts, responses, invalid count and measured speed. RUN_MODE='full'
targets all 1,535 claims with the identical settings in a separate run folder.
Partial runs are never presented as final scores. No guarantee of Colab availability.

Four labels are retained: Supports 654, Refutes 253, NEI 474, Disputed 154.
RoBERTa is the existing frozen three-label FEVER checkpoint and cannot output
Disputed; it is explicitly a transfer baseline, not a four-class trained model.
No Climate-FEVER fine-tuning or training split is created.

The evidence corpus is the **full historical FEVER Wikipedia snapshot**, reused
from your earlier work. This is a transfer experiment, not a reproduction of
Climate-FEVER's original evidence collection. Its older articles may lack facts.
No per-claim annotated sentences or article IDs enter retrieval or prompts.
Each model independently searches top-10 titles and selects up to 3 excerpts
after reading all returned article sentences in batches. No model reads the
entire Wikipedia corpus. Limits match the existing independent runner.
""")
    code('''
        from pathlib import Path
        import json, sys, time, hashlib, shutil, zipfile, io, base64
        from google.colab import drive
        drive.mount('/content/drive')
        DRIVE_PROJECT = Path('/content/drive/MyDrive/Climate_FEVER_Project')
        DRIVE_PROJECT.mkdir(parents=True, exist_ok=True)
        # Keep console text as well as structured claim traces. Colab's rich
        # displays are also retained when the user saves the notebook.
        class ConsoleTee:
            _climate_console_tee = True
            def __init__(self, original, path):
                self.original = original
                self.log = path.open('a', encoding='utf-8', buffering=1)
            def write(self, value):
                self.original.write(value)
                self.log.write(value)
                return len(value)
            def flush(self):
                self.original.flush()
                self.log.flush()
            def __getattr__(self, name):
                return getattr(self.original, name)
        while getattr(sys.stdout, '_climate_console_tee', False):
            sys.stdout = sys.stdout.original
        while getattr(sys.stderr, '_climate_console_tee', False):
            sys.stderr = sys.stderr.original
        console_path = DRIVE_PROJECT / ('console_' + str(time.time_ns()) + '.log')
        sys.stdout = ConsoleTee(sys.stdout, console_path)
        sys.stderr = ConsoleTee(sys.stderr, console_path)
        print('Console text log:', console_path, flush=True)
        PROJECT = Path('/content/climate_fever_runtime')
        PROJECT.mkdir(parents=True, exist_ok=True)
        RUN_MODE = 'setup'  # 'setup', 'pilot', or 'full'
        MODEL_QUEUE = ['qwen']  # then ['qwen', 'llama', 'mistral', 'gemma', 'roberta']
        SESSION_MINUTES = 30  # stops BETWEEN claims; a long claim may exceed this
        ALLOW_MODEL_DOWNLOADS = False  # prints weight sizes first; enable for inference
        ALLOW_WIKIPEDIA_DOWNLOAD = False  # only needed if old archive cache is absent
        CHECKPOINT = Path('/content/drive/MyDrive/FEVER_Project/runs/roberta_full_unfreeze_v1/training/roberta_full_last3/final')
        print('Mode:', RUN_MODE, '| Models:', MODEL_QUEUE, '| Outputs:', DRIVE_PROJECT, flush=True)
    ''')
    payload = base64.b64encode(archive.read_bytes()).decode()
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    code(f"payload = base64.b64decode({payload!r})\n"
         f"assert hashlib.sha256(payload).hexdigest() == {digest!r}\n"
         "with zipfile.ZipFile(io.BytesIO(payload)) as bundle:\n"
         "    bundle.extractall(PROJECT)\n"
         "manifest = json.loads((PROJECT / 'bundle_manifest.json').read_text())\n"
         "for name, expected in manifest.items():\n"
         "    assert hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == expected, name\n"
         "sys.path.insert(0, str(PROJECT / 'src'))\n"
         "for name in ('data', 'source'):\n"
         "    shutil.copytree(PROJECT / name, DRIVE_PROJECT / name, dirs_exist_ok=True)\n"
         "print('Verified embedded dataset and code; dataset copied to Drive.', flush=True)")
    code('''
        import subprocess
        install = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
            'transformers==4.57.6', 'accelerate==1.11.0', 'bitsandbytes==0.48.2',
            'scikit-learn', 'faiss-cpu', 'requests', 'tqdm', 'sentencepiece'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(install.stdout)
        install.check_returncode()
        import climate_fever as climate
        import independent_systems as independent
        import fever_colab_runtime as core
        ROWS, DATA_MANIFEST = climate.verify_data(PROJECT / 'data')
        print('Verified', len(ROWS), 'claims. Counts:', DATA_MANIFEST['label_counts'])
        assert RUN_MODE in ('setup', 'pilot', 'full')
        assert MODEL_QUEUE and len(set(MODEL_QUEUE)) == len(MODEL_QUEUE)
        assert all(f in list(independent.MODELS) + ['roberta'] for f in MODEL_QUEUE)
        if RUN_MODE != 'setup':
            core.check_cuda()  # never falls back to a multi-month CPU run
    ''')
    md("""## Evidence corpus — cached full articles, not supplied answer sentences
Reuse the 1.6 GiB archive already saved under FEVER_Project. A local SQLite
index is rebuilt after runtime loss; indexing logs each shard. This remains
a large corpus even though the claim set is much smaller. No models are loaded
while indexing. The archive and its hash are unchanged from the earlier study.
""")
    code('''
        WIKI_SHA = '4b06d95da6adf7fe02d2796176c670dacccb21348da89cba4c50676ab99665f2'
        WIKI_BYTES = 1713485474
        DATABASE = None
        if RUN_MODE != 'setup':
            archive = Path('/content/climate_wikipedia.zip')
            cache = Path('/content/drive/MyDrive/FEVER_Project/FEVER_Full_Wikipedia.zip')
            if not archive.exists() and cache.exists():
                print('Restoring full Wikipedia archive from Drive.', flush=True)
                shutil.copyfile(cache, archive)
            if not archive.exists():
                if not ALLOW_WIKIPEDIA_DOWNLOAD:
                    raise RuntimeError('Cached Wikipedia archive missing. Enable ALLOW_WIKIPEDIA_DOWNLOAD for the 1.6 GiB download.')
                import requests
                from tqdm.auto import tqdm
                staging = archive.with_suffix('.pending')
                with requests.get('https://fever.ai/download/fever/wiki-pages.zip', stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    with staging.open('wb') as stream, tqdm(total=WIKI_BYTES, unit='B', unit_scale=True, desc='Wikipedia') as bar:
                        for block in response.iter_content(2**20):
                            if block:
                                stream.write(block)
                                bar.update(len(block))
                if staging.stat().st_size != WIKI_BYTES or core.sha_file(staging) != WIKI_SHA:
                    raise ValueError('Wikipedia download incomplete or checksum mismatch')
                staging.replace(archive)
                if not cache.exists():
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    temporary_cache = cache.with_name(cache.name + '.pending')
                    shutil.copyfile(archive, temporary_cache)
                    temporary_cache.replace(cache)
            if core.sha_file(archive) != WIKI_SHA:
                raise ValueError('Wikipedia archive hash mismatch')
            DATABASE = independent.build_corpus(archive, PROJECT / 'wikipedia_index', WIKI_SHA)
        else:
            print('Setup only: corpus indexing and all inference skipped.')
    ''')
    md("""## Run one model at a time, with durable progress
The pilot uses source-order claims 0–11, not claims chosen using their labels.
If it is used to change prompts, it becomes development data; a later all-claim
score must disclose that exposure. Existing prediction records, including INVALID,
are never overwritten. A protocol change requires a new run folder. The code
stops after three consecutive invalid outputs or a system error. It does not
automatically retry or silently replace failed outputs with NEI.
""")
    code('''
        import torch
        TARGET_ROWS = ROWS[:12] if RUN_MODE == 'pilot' else ROWS
        OUTPUT = DRIVE_PROJECT / 'runs' / ('independent_pilot12_v1' if RUN_MODE == 'pilot' else 'independent_all1535_v1')
        if RUN_MODE != 'setup':
            if not ALLOW_MODEL_DOWNLOADS:
                for family in MODEL_QUEUE:
                    if family != 'roberta':
                        core.remote_metadata(independent.MODELS[family])
                raise RuntimeError('Download sizes printed. Set ALLOW_MODEL_DOWNLOADS=True to run inference. No weights downloaded yet.')
            started = time.monotonic()
            for family in MODEL_QUEUE:
                remaining = SESSION_MINUTES - (time.monotonic() - started) / 60
                if remaining <= 0:
                    break
                previous = OUTPUT / family / 'protocol.json'
                metadata = None
                if family != 'roberta':
                    metadata = (json.loads(previous.read_text())['model'] if previous.exists()
                                else core.remote_metadata(independent.MODELS[family]))
                elif not (CHECKPOINT / 'config.json').exists():
                    raise FileNotFoundError('Existing RoBERTa checkpoint not found: ' + str(CHECKPOINT))
                print('START / RESUME', family, '| target:', len(TARGET_ROWS), flush=True)
                status = climate.run(TARGET_ROWS, PROJECT, OUTPUT, DATABASE,
                    {'dataset_sha256': climate.SOURCE_SHA256, 'wikipedia_archive_sha256': WIKI_SHA},
                    family, metadata, CHECKPOINT, minutes=remaining)
                print('STATUS:', family, status, flush=True)
                if status != 'complete':
                    break
        else:
            print('No evaluation started. Choose pilot or full in the settings cell when ready.')
    ''')
    code('''
        import pandas as pd
        from IPython.display import display
        if RUN_MODE != 'setup':
            # Labels are opened only for reporting, after inference.
            all_labels = json.loads((PROJECT / 'data/labels.json').read_text())
            labels = {row['id']: all_labels[row['id']] for row in TARGET_ROWS}
            results = climate.report(TARGET_ROWS, labels, OUTPUT, MODEL_QUEUE)
            display(pd.DataFrame([{k: v for k, v in r.items() if k not in
                ('per_class', 'confusion_matrix', 'confusion_rows', 'confusion_columns')} for r in results]))
            print('Full prompts/responses:', OUTPUT, flush=True)
            for r in results:
                if 'confusion_matrix' in r:
                    print(r['model'], '| actual rows, predicted columns')
                    display(pd.DataFrame(r['confusion_matrix'], index=r['confusion_rows'], columns=r['confusion_columns']))
    ''')
    md("""## Inspect a full trace / export outputs
The trace includes every search, selection and classification prompt/response,
selected evidence, probabilities (RoBERTa), errors and timings. Opening a trace
does not call the model again. Cell outputs also remain in the saved notebook.
""")
    code('''
        if RUN_MODE != 'setup':
            paths = sorted(OUTPUT.glob('*/predictions/*.json'))
            print('Saved prediction records:', len(paths))
            if paths:
                TRACE_PATH = paths[-1]  # choose any saved path here
                print('TRACE:', TRACE_PATH)
                print(TRACE_PATH.read_text())
            # Re-run this cell whenever you want an up-to-date portable export.
            exported = shutil.make_archive(str(DRIVE_PROJECT / ('outputs_' + str(time.time_ns()))), 'zip', OUTPUT)
            sys.stdout.flush()
            sys.stderr.flush()
            with zipfile.ZipFile(exported, 'a', zipfile.ZIP_DEFLATED) as exported_zip:
                for log in DRIVE_PROJECT.glob('console_*.log'):
                    exported_zip.write(log, 'console/' + log.name)
            print('Output archive saved:', exported)
    ''')
    for i, cell in enumerate(cells):
        cell["id"] = f"climate-{i + 1}"
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
        "metadata": {"colab": {"name": "Climate_FEVER_Independent_1535.ipynb"},
                     "accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}}}
    target = OUT / "Climate_FEVER_Independent_1535.ipynb"
    target.write_text(json.dumps(notebook, indent=1))
    print("Notebook:", target, "bytes:", target.stat().st_size)
    print("Bundle:", archive, "bytes:", archive.stat().st_size)


if __name__ == "__main__":
    build()
