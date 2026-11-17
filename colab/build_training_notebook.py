"""Build a training-only Colab notebook. No model execution or downloads here."""
import argparse,ast,json,runpy,textwrap
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
followups=parser.add_mutually_exclusive_group()
followups.add_argument('--last6',action='store_true',help='Build the six-block follow-up in a separate notebook')
followups.add_argument('--sweep-4-10',action='store_true',help='Build the seven-condition follow-up')
args=parser.parse_args()
filename='FEVER_Train_Full.ipynb'
if args.last6:
 filename='FEVER_Train_Last6.ipynb'
elif args.sweep_4_10:
 filename='FEVER_Train_Layers_4_to_10.ipynb'
followup=args.last6 or args.sweep_4_10
# Reuse the same verified setup/runtime patch rather than a second implementation.
if not followup:
 runpy.run_path(str(ROOT/'colab/build_notebook.py'))
base=json.loads((ROOT/'output/colab_20261104/FEVER_Colab_Debug.ipynb').read_text())
cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':textwrap.dedent(s).strip().splitlines(keepends=True)})
def code(s):
 s=textwrap.dedent(s).strip();ast.parse(s)
 cells.append({'cell_type':'code','metadata':{},'source':s.splitlines(keepends=True),'execution_count':None,'outputs':[]})
def reuse(snippet):
 found=next(c for c in base['cells'] if c['cell_type']=='code' and snippet in ''.join(c['source']))
 code(''.join(found['source']))
md('''# FEVER — full RoBERTa training
**Select T4 GPU, then Runtime → Run all. Training is already enabled.**

This notebook continues fine-tuning our existing FEVER RoBERTa checkpoint on
116,356 training claims. It uses 14,534 separate validation claims for checkpoint
selection. The 19,998 official-development claims never enter training or checkpoint selection.

Three controlled conditions unfreeze the classification head and the last 1, 2,
or 3 Transformer blocks. Every condition starts from the same task-trained checkpoint;
only the number of unfrozen blocks differs. This notebook trains RoBERTa, not Qwen or Llama.

Training is substantial and may require multiple Colab sessions. After a session
pauses, use Run all again to resume; completed conditions are skipped. Drive stores
optimizer checkpoints, progress logs, validation metrics and final models.

The original bundle and `FEVER_Training_Data.zip` are already in your FEVER_Project
Drive folder. No manual path entry or dataset upload is needed.
''')
md('''## 1. Check GPU''');reuse('print("Python:", sys.version.split()[0])')
md('''## 2. Install runtime libraries
Libraries install only in Colab. The original RoBERTa weights are in the uploaded bundle;
this notebook does not download Qwen, Llama or embedding models.''');reuse('packages = ["transformers')
md('''## 3. Connect Drive and verify the original bundle
Approve the normal Drive sign-in prompt. The original files/results stay intact.
The updated training runtime is embedded in this notebook and verified before use.''');reuse('z.extractall(PROJECT)')
md('''## 4. Training settings — already enabled
Two epochs per condition; learning rate 0.00002; weight decay 0.01; batch size 4
with 4-step gradient accumulation (effective batch 16); maximum 512 tokens;
seed 42; FP16. Validate and save every 1,000 optimizer steps, keeping the best
validation macro-F1 checkpoint and the latest resumable checkpoint. The session
budget is 180 minutes across conditions. No full benchmark evaluation runs here.

Training evidence is historical retrieved top-five sentence text for all labels,
including NEI. It is not newly fetched full-article RAG training data. Nonempty
retrieval does not guarantee that the evidence is sufficient. No improvement is claimed yet.''')
code('''EXPERIMENT_NAME = "roberta_full_unfreeze_v1" # @param {type:"string"}
SESSION_MINUTES = 180 # @param {type:"integer"}
RUN_FULL_TRAINING = True # @param {type:"boolean"}
RUN_ROOT = DRIVE_PROJECT / "runs" / EXPERIMENT_NAME
RUN_ROOT.mkdir(parents=True, exist_ok=True)
if not 1 <= SESSION_MINUTES <= 300: raise ValueError("Use a session budget between 1 and 300 minutes")
print("Training results and checkpoints:", RUN_ROOT)
print("Conditions: last 1, 2 and 3 blocks; classification head trainable in each.")
''')
md('''## 5. Verify the full training supplement
This checks file hashes, actual counts, all labels and normalized-text separation
between training, validation and official development before any training starts.''')
ready=json.loads((ROOT/'output/colab_diagnostic_training_20261105/training_ready.json').read_text())
code('EXPECTED_TRAINING = '+repr(ready)+'\n'+textwrap.dedent('''
training_archive = DRIVE_PROJECT / "FEVER_Training_Data.zip"
if not training_archive.is_file():
    raise FileNotFoundError("FEVER_Training_Data.zip is missing from FEVER_Project. Do not substitute pilot data.")
with zipfile.ZipFile(training_archive) as archive:
    expected_names = {"data/train_full.jsonl", "data/validation_full.jsonl", "training_ready.json", "preparation_protocol.json"}
    if set(archive.namelist()) != expected_names: raise ValueError("Unexpected supplement contents")
    ready = json.loads(archive.read("training_ready.json"))
    if ready != EXPECTED_TRAINING: raise ValueError("Supplement differs from the verified full training export")
    for name in ["train_full.jsonl", "validation_full.jsonl"]:
        content = archive.read("data/" + name)
        if hashlib.sha256(content).hexdigest() != ready["files"][name]:
            raise ValueError("Training file hash mismatch")
        target = PROJECT / "data" / name
        if target.exists() and target.read_bytes() != content:
            raise ValueError("Different full training data already exist")
        target.write_bytes(content)
train = fever.read_rows(PROJECT / "data/train_full.jsonl")
validation = fever.read_rows(PROJECT / "data/validation_full.jsonl")
assert len(train) == 116356 and len(validation) == 14534
assert len({r["id"] for r in train}) == len(train)
assert len({r["id"] for r in validation}) == len(validation)
assert all(r["label"] in fever.LABELS and isinstance(r["evidence_text"], str) for r in train+validation)
train_text = {fever.normalize(r["claim"]) for r in train}
validation_text = {fever.normalize(r["claim"]) for r in validation}
official_text = {fever.normalize(r["claim"]) for r in EVAL_ROWS}
if train_text & validation_text or (train_text | validation_text) & official_text:
    raise ValueError("Training/validation/official-development text overlap")
from collections import Counter
for label, records in [("Training",train),("Validation",validation)]:
    print(label, len(records), "claims; classes:", dict(Counter(r["label"] for r in records)))
    print("Empty evidence:", sum(not r["evidence_text"] for r in records))
print("PASS: all hashes/counts verified; zero normalized-text overlap across partitions.")
fever.write_once(RUN_ROOT / "data_verification.json", ready)
del train, validation, train_text, validation_text, official_text
'''))
md('''## 6. Train / resume — enabled
The live bar shows optimizer steps. Logs include training loss, learning rate,
validation loss, validation macro-F1, GPU memory and checkpoint locations.
Every 20 seconds a heartbeat reports the current stage. A paused run is not a
completed experiment. Rerun this notebook to resume unchanged conditions.
''')
code('''if not RUN_FULL_TRAINING:
    print("Training disabled.")
else:
    import time
    started = time.monotonic()
    for layers in [1, 2, 3]:
        remaining = SESSION_MINUTES - (time.monotonic() - started)/60
        if remaining < 1:
            print("Session budget reached. Run all again to resume remaining conditions.")
            break
        print("\\nSTARTING / RESUMING: last", layers, "Transformer block(s).", flush=True)
        checkpoint = fever.finetune_roberta(PROJECT, RUN_ROOT, last_layers=layers,
            scope="full", minutes=remaining, checkpoint_steps=1000)
        if checkpoint is None:
            print("Paused and checkpointed. Run all again to resume.")
            break
        print("Completed checkpoint:", checkpoint, flush=True)
''')
md('''## 7. Saved progress and completion
This summarizes saved status after training finishes or pauses. Validation scores
choose checkpoints; they are not final test accuracy. Later we evaluate trained
checkpoints and the unchanged baseline on identical claims and evidence.
''')
code('''for layers in [1, 2, 3]:
    name = "roberta_full_last" + str(layers)
    folder = RUN_ROOT / "training" / name
    status_path = folder / "monitor" / "status.json"
    print("\\n", name)
    if status_path.is_file():
        status = json.loads(status_path.read_text())
        print(json.dumps({key:status.get(key) for key in ["status","stage","completed","total","reason","updated_utc"]}, indent=2))
    else:
        print("Not started yet.")
    if (folder / "complete.json").is_file():
        print("FINAL MODEL:", folder / "final")
        print((folder / "complete.json").read_text())
print("All losses and full logs are saved in Drive; the notebook's cell outputs also save there.")
''')
if followup:
 baseline_dir=ROOT/'reports/part2/roberta_full_training_analysis_20261106'
 baseline=json.loads((baseline_dir/'last3_protocol.json').read_text())
 best=json.loads((baseline_dir/'last3_trainer_state.json').read_text())['best_metric']
 layer_counts=list(range(4,11)) if args.sweep_4_10 else [6]
 layer_description='4 through 10' if args.sweep_4_10 else '6'
 run_suffix='layers4_to10' if args.sweep_4_10 else 'last6'
 for cell in cells:
  source=''.join(cell['source'])
  source=source.replace('# FEVER — full RoBERTa training',f'# FEVER — RoBERTa last {layer_description} blocks')
  source=source.replace('Three controlled conditions unfreeze the classification head and the last 1, 2,\nor 3 Transformer blocks. Every condition starts from the same task-trained checkpoint;\nonly the number of unfrozen blocks differs. This notebook trains RoBERTa, not Qwen or Llama.',
   f'This notebook tests {layer_description} unfrozen final Transformer blocks, with the classification head trainable in each condition.\nEach condition starts from the same checkpoint as the last-3 run ({best:.2%} validation macro-F1).')
  source=source.replace('roberta_full_unfreeze_v1',f'roberta_full_unfreeze_{run_suffix}_v1')
  source=source.replace('last 1, 2 and 3 blocks; classification head trainable in each.',
   f'last {layer_description} blocks plus the classification head.')
  source=source.replace('for layers in [1, 2, 3]:',f'for layers in {layer_counts}:')
  if source.startswith('EXPECTED_TRAINING ='):
   source+= '\nBASELINE = '+repr(baseline)+'\n'+textwrap.dedent('''
   for name, key in [("train_full.jsonl", "train_sha256"), ("validation_full.jsonl", "validation_sha256")]:
       if ready["files"][name] != BASELINE[key]:
           raise ValueError("Data do not match the last-3 baseline")
   if fever.sha_file(PROJECT / "src/fever_colab_runtime.py") != BASELINE["runtime_sha256"]:
       raise ValueError("Training runtime differs from the last-3 baseline")
   if fever.checkpoint_identity(PROJECT / "checkpoint/roberta_evidence") != BASELINE["initial_checkpoint"]:
       raise ValueError("Starting checkpoint differs from the last-3 baseline")
   current_env = fever.environment()
   if current_env != BASELINE["environment"]:
       print("Colab environment differs from the last-3 run. Both environments are recorded.")
       print("Last-3 environment:", BASELINE["environment"])
       print("Current environment:", current_env)
   ''').strip()
  if source.startswith(f'for layers in {layer_counts}:'):
   source=textwrap.dedent(f'''
   print("Last 3 validation macro-F1: {best:.2%} (baseline)")
   for layers in {layer_counts}:
       folder = RUN_ROOT / "training" / f"roberta_full_last{{layers}}"
       complete_path = folder / "complete.json"
       if not complete_path.is_file():
           status_path = folder / "monitor" / "status.json"
           status = json.loads(status_path.read_text()) if status_path.is_file() else {{}}
           print(f"Last {{layers}}: {{status.get('status', 'not started')}} "
                 f"({{status.get('completed', 0)}} optimizer steps)")
           continue
       complete = json.loads(complete_path.read_text())
       best_path = Path(complete["best_validation_checkpoint"])
       state = json.loads((best_path / "trainer_state.json").read_text())
       score = state["best_metric"]
       delta = 100 * (score - {best!r})
       print(f"Last {{layers}}: {{score:.2%}} validation macro-F1 ({{delta:+.2f}} pp vs. last 3) "
             f"| {{complete['steps']}} steps | selected {{best_path.name}}")
   ''').strip()
  if cell['cell_type']=='code':ast.parse(source)
  cell['source']=source.splitlines(keepends=True)
for i,c in enumerate(cells):c['id']='fever-training-'+str(i+1).zfill(2)
nb={'nbformat':4,'nbformat_minor':5,'metadata':{
 'colab':{'name':filename,'provenance':[],'gpuType':'T4'},
 'accelerator':'GPU','kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
 'language_info':{'name':'python'}},'cells':cells}
out=ROOT/'output/FEVER_Project'/filename;out.write_text(json.dumps(nb,indent=1,ensure_ascii=False))
print('Created training-only notebook:',out,'cells:',len(cells))
