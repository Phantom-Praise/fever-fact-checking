"""Package completed evidence and a portable standard-library trace viewer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    summary = json.loads((run/'summary.json').read_text())
    assert summary['status'] == 'complete' and summary['n'] == 60
    assert not any((run/'temporary').iterdir())
    protocol = json.loads((run/'protocol.json').read_text())
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.pdf, out/'Result_Analysis.pdf')
    study = out/'study'
    study.mkdir()
    keep = ['protocol.json','inputs.jsonl','reference_labels.json','progress.json','summary.json','scored_cases.json',
            'REPORT.md','REVIEWED_ANALYSIS.md','comparison.csv','per_class.csv','predictions.csv','comparison.png','confusion.png',
            'qualitative_review.json','plot_runtime.json','execution_audit.json','literature_context_sources.json']
    for name in keep:
        shutil.copy2(run/name, study/name)
    for name in ['traces','attempts']:
        shutil.copytree(run/name, study/name)
    viewer = out/'viewer'
    (viewer/'src').mkdir(parents=True)
    (viewer/'web/evidence_explorer').mkdir(parents=True)
    shutil.copy2(ROOT/'src/serve_parts_test.py',viewer/'src/serve_parts_test.py')
    page = (ROOT/'web/evidence_explorer/parts-test.html').read_text()
    page = page.replace('<a href="http://127.0.0.1:8765/">← Earlier evidence explorer</a>', '<a href="/">Study dashboard</a>')
    (viewer/'web/evidence_explorer/parts-test.html').write_text(page)
    source = out/'source_snapshots'
    source.mkdir()
    for filename, expected in protocol['sha256'].items():
        path = Path(filename)
        if path.suffix == '.py' and path.parent == ROOT/'src':
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError('Source differs from evaluated version: '+filename)
            shutil.copy2(path,source/path.name)
    shutil.copy2(ROOT/'docs/working_parts_prompt.json',source/'working_parts_prompt.json')
    shutil.copy2(ROOT/'docs/parts_test60_20261030.md',source/'study_notes.md')
    shutil.copy2(ROOT/'output/plan_parts_test_20261030/population_audit.json',source/'population_audit.json')
    (out/'README.md').write_text('''# Result Analysis - FEVER

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
''')
    manifest = {str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.rglob('*') if p.is_file()}
    (out/'file_hashes.json').write_text(json.dumps(manifest,indent=2))
    zip_path = out.with_suffix('.zip')
    with zipfile.ZipFile(zip_path, 'x', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                archive.write(p,str(Path(out.name)/p.relative_to(out)))
    print('Packaged',zip_path,'bytes',zip_path.stat().st_size)


if __name__=='__main__':
    main()
