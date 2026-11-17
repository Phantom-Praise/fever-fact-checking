"""Build a bounded submission archive from explicit research-artifact paths.

Includes measured outputs, source code, selected small inputs, and a hash audit.
Excludes model weights, virtual environments, and all SQLite indexes.
"""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


VERIFY = '''"""Verify an extracted Part 2 package using only Python's standard library."""
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parent
manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text())
for name, expected in manifest["files_sha256"].items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit("Missing or changed package file: " + name)
print("Verified", len(manifest["files_sha256"]), "files")
'''

README = '''# COSC 874 Model Execution Part 2

Start with Model_Execution_Part_2.pdf or the editable Word version. The six
root-level CSV tables contain the complete saved training/configuration,
classifier, judge, retrieval, latency, and selection records. The detailed
Markdown report is under reports/part2/submission_20261023_final/.

These are development results. Independent human labels for generated claims
remain blank; the report explicitly marks RQ1 as partial. Additional Transformer
training seeds and held-out final testing were not performed. The symbolic
variant has no demonstrated benefit on the measured sample.

The archive contains measured raw outputs, selected small FEVER inputs, source
scripts, tests, and manifests. It does not include checkpoints, SQLite indexes,
model downloads, or the Python environment. Full inference reproduction needs
the original project environment, checkpoints and indexes. Saved metrics can
be audited without running inference. Research commands use .venv/bin/python;
the Word builder uses the Codex bundled document runtime for formatting only.

To verify an extracted package using standard Python:

    python VERIFY_PACKAGE.py

This checks file integrity, not independent scientific replication. The package
manifest records the source file hashes, intended exclusions, and unresolved
work. The existing project was not merged into main or destructively modified.

For independent RQ1 review, give reviewers only human_review_A.csv or
human_review_B.csv, the blinded evidence pack, and REVIEW_INSTRUCTIONS.md under
reports/part2/local_llm_20261023/analysis_blinded/. Keep the analysis-only source
key and existing FEVER labels with the analysis code. Do not substitute model
judgments for human labels.
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    base = Path(__file__).resolve().parents[2]
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    tables = base / 'reports/part2/submission_20261023_final'
    manifest = json.loads((tables / 'manifest.json').read_text())
    files = {}

    def add(path, name=None):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix in ('.db', '.safetensors', '.bin'):
            raise ValueError('Large model/index artifacts are forbidden in this archive')
        data = path.read_bytes()
        if len(data) > 15_000_000:
            raise ValueError('Unexpectedly large artifact: ' + str(path))
        target = name or path.relative_to(base).as_posix()
        if target in files and files[target] != data:
            raise ValueError('Conflicting archive member: ' + target)
        files[target] = data

    for name, expected in manifest['input_sha256'].items():
        path = base / name
        if digest(path.read_bytes()) != expected:
            raise ValueError('Measured source differs from saved manifest: ' + name)
        add(path)
    for name in ('Model_Execution_Part_2.docx', 'Model_Execution_Part_2.pdf'):
        add(tables / 'delivery' / name, name)
    for path in tables.iterdir():
        if path.is_file():
            add(path)
            if path.suffix == '.csv':
                add(path, path.name)
    for directory in ('paired', 'tuning', 'symbolic_v2', 'evidence_depth_20261023',
                      'validation_extension_20261023_complete', 'reference_checkpoints_20261023',
                      'local_llm_20261023', 'corpus_growth_20261023', 'latency_20261023'):
        for path in (base / 'reports/part2' / directory).rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.csv', '.md', '.py'):
                if directory == 'local_llm_20261023' and 'analysis' in path.relative_to(base / 'reports/part2/local_llm_20261023').parts:
                    continue
                add(path)
    for path in (base / 'src').glob('*.py'):
        add(path)
    for path in (base / 'tests').glob('*.py'):
        add(path)
    for name in ('README.md', 'requirements.txt',
                 'data/processed/validation.jsonl',
                 'data/processed/evidence/validation.jsonl',
                 'data/processed/retrieved/validation_first_100_top5_title_fast.jsonl'):
        add(base / name)
    plan = json.loads((base / 'reports/part2/corpus_growth_20261023/index_manifest.json').read_text())
    for index in plan['indexes']:
        add(base / index['membership_file'])
    files['VERIFY_PACKAGE.py'] = VERIFY.encode()
    files['START_HERE.md'] = README.encode()
    package_manifest = {'branch': 'experiment/part2-paired-ablation',
        'files_sha256': {name: digest(data) for name, data in sorted(files.items())},
        'exclusions': ['Model weights', 'SQLite indexes', 'Python virtual environment', 'Full training/test partitions', 'Preliminary unblinded review export'],
        'unresolved': ['Independent RQ1 human review and adjudication', 'Optional additional Transformer training seeds', 'Held-out final testing'],
        'note': 'Packaging does not add measurements or claim a hybrid improvement.'}
    files['PACKAGE_MANIFEST.json'] = json.dumps(package_manifest, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'x', ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
        saved = json.loads(archive.read('PACKAGE_MANIFEST.json'))
        for name, expected in saved['files_sha256'].items():
            if digest(archive.read(name)) != expected:
                raise ValueError('Archive checksum mismatch: ' + name)
    print(json.dumps({'archive': str(output), 'files': len(files), 'bytes': output.stat().st_size,
                      'sha256': digest(output.read_bytes())}, indent=2))


if __name__ == '__main__':
    main()
