"""Prepare, run, and score a paired test study. Each command is explicit.

Uses only installed local models. Default/help never executes inference.
Gold labels are read by prepare/score, never passed to either classifier.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
CONDITIONS = ['evidence_roberta', 'parts_rag']
AUDIT = ROOT / 'output/plan_parts_test_20261030'
CHECKPOINT = ROOT / 'checkpoints/roberta_evidence/epoch_1'


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    """Atomic and exclusive: a crash cannot leave a partial completed trace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix='.pending-', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # fails if completed artifact exists
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def select_sample(eligible, gold, count=60, seed=20261030):
    if count < 3 or count % 3:
        raise ValueError('Use a positive multiple of three for balanced sampling')
    rng = random.Random(seed)
    chosen = []
    seen = set()
    pools = {label: [] for label in LABELS}
    for row in eligible:
        reference = gold[row['id']]
        if reference['claim'] != row['claim'] or set(row) != {'id', 'claim'}:
            raise ValueError('Eligible claim differs from original test data')
        text = ' '.join(row['claim'].casefold().split())
        if text in seen:
            raise ValueError('Duplicate eligible claim')
        seen.add(text)
        pools[reference['label']].append(row)
    for label in LABELS:
        chosen.extend(rng.sample(pools[label], count // 3))
    rng.shuffle(chosen)
    return chosen


def prepare(out, count=60):
    from project import load_working_prompt
    from replay_claim_parts import INSTRUCTION, SCHEMA
    selected, text = load_working_prompt()
    if text.rstrip('\n') != INSTRUCTION:
        raise ValueError('Adapter instruction differs from selected prompt')
    audit = load(AUDIT / 'population_audit.json')
    for name, expected in audit['sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Population audit source changed: ' + name)
    eligible_path = AUDIT / 'eligible_claims_without_labels.jsonl'
    gold = {r['id']: r for r in rows(ROOT / 'data/processed/test.jsonl')}
    chosen = select_sample(rows(eligible_path), gold, count)
    if load(ROOT / 'reports/part2/ollama_full_dev_20261028/batch_status.json')['status'] != 'stopped':
        raise ValueError('Old full-dev evaluation must stay stopped')
    out.mkdir(parents=True, exist_ok=False)
    for condition in CONDITIONS:
        (out / 'traces' / condition).mkdir(parents=True)
    (out / 'attempts').mkdir()
    (out / 'temporary').mkdir()
    inputs = out / 'inputs.jsonl'
    with inputs.open('x') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in chosen)
    write_new(out / 'reference_labels.json', {str(r['id']): gold[r['id']]['label'] for r in chosen})
    sources = ['experiment_parts_test.py', 'parts_rag_classifier.py', 'ephemeral_rag.py',
               'rag_embedding_worker.py', 'fetch_wikipedia_articles.py', 'retrieve_title_aware.py',
               'replay_claim_parts.py', 'local_llm_experiments.py', 'finalize_exhaustive_tokens.py',
               'project.py', 'report_parts_test.py', 'parts_baseline_worker.py']
    files = [ROOT / 'src' / name for name in sources]
    files += list(CHECKPOINT.iterdir())
    from ephemeral_rag import MODEL_DIR
    files += [MODEL_DIR / r['name'] for r in load(MODEL_DIR / 'download_manifest.json')['files']]
    files += [MODEL_DIR / 'download_manifest.json', ROOT / 'docs/working_parts_prompt.json',
              ROOT / selected['prompt_file'], eligible_path, AUDIT / 'population_audit.json',
              inputs, out / 'reference_labels.json', ROOT / 'data/processed/test.jsonl']
    index = ROOT / 'data/processed/wiki_fts.db'
    stat = index.stat()
    import importlib.metadata
    versions = {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'numpy', 'faiss-cpu', 'requests']}
    protocol = dict(status='frozen_before_inference', created_utc=now(), n=count, sampling_seed=20261030,
                    ids=[r['id'] for r in chosen], class_counts=dict(Counter(gold[r['id']]['label'] for r in chosen)),
                    selection='Seeded 20-per-class sample from audited eligible internal test claims; '
                              'no reference labels supplied to classifiers. Balanced sample differs from population prevalence.',
                    main='parts_rag', baseline='evidence_roberta', primary_label='code-derived essential-parts label',
                    hypothesis='The frozen main pipeline improves both accuracy and macro-F1 over evidence RoBERTa on these paired claims.',
                    model=selected['model'], ollama=selected['ollama'], settings=selected['settings'],
                    prompt=INSTRUCTION, schema=SCHEMA, versions=versions,
                    evidence=dict(articles=5, candidates=10, chunk_tokens=200, vector_candidates=30,
                                  context_characters=7600, baseline_max_length=256),
                    tokenizer='Pinned installed Llama tokenizer; verify historical development calibration and every returned prompt count.',
                    failure_policy='No classification retries. INVALID counts as incorrect. Bounded existing fetch retries (2). '
                                   'Stop after five consecutive infrastructure/preparation failures or failed cleanup. '
                                   'Resume unfinished attempts only; completed files are immutable.',
                    timing='Per-condition wall time includes preparation, generation and cleanup; shared model setup recorded separately. '
                           'Baseline CPU, embeddings CPU, Llama local Ollama. Conditions differ in evidence corpus and cost.',
                    checkpoint=str(CHECKPOINT.relative_to(ROOT)),
                    index=dict(path=str(index.relative_to(ROOT)), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
                    sha256={str(p.resolve()): digest(p) for p in files if p.is_file()},
                    limitations=['60-claim single-run balanced study, not full test-set performance.',
                                 'Selected prompt was developed on ten other inspected examples.',
                                 'Live Wikipedia versions differ from the historical FEVER evidence corpus.',
                                 'Baseline trained on annotated evidence with empty NEI inputs; no retraining.',
                                 'No strict FEVER score for live article block citations.',
                                 'Full article text and vectors are deleted; exact supplied prompts can be replayed.'])
    write_new(out / 'protocol.json', protocol)
    from fetch_wikipedia_articles import atomic_json
    atomic_json(out / 'progress.json', dict(status='prepared', total=count, completed_pairs=0, conditions={c: 0 for c in CONDITIONS}))
    print('Prepared', count, 'paired test inputs; no inference executed.', flush=True)


def verify_frozen(out):
    p = load(out / 'protocol.json')
    for name, expected in p['sha256'].items():
        if digest(name) != expected:
            raise ValueError('Frozen input/source changed: ' + name)
    index = (ROOT / p['index']['path']).stat()
    if (index.st_size, index.st_mtime_ns) != (p['index']['bytes'], p['index']['mtime_ns']):
        raise ValueError('Wikipedia index changed')
    import importlib.metadata
    if any(importlib.metadata.version(k) != v for k, v in p['versions'].items()):
        raise ValueError('Python dependency version changed')
    return p


def completed(out, condition, ident):
    path = out / 'traces' / condition / (str(ident) + '.json')
    if not path.exists():
        return None
    data = load(path)
    if data['id'] != ident or data['label'] not in LABELS + ['INVALID']:
        raise ValueError('Invalid completed trace: ' + str(path))
    return data


def progress(out, inputs, status, **extra):
    from fetch_wikipedia_articles import atomic_json
    counts = {c: sum(completed(out, c, r['id']) is not None for r in inputs) for c in CONDITIONS}
    pairs = sum(all(completed(out, c, r['id']) is not None for c in CONDITIONS) for r in inputs)
    atomic_json(out / 'progress.json', dict(status=status, total=len(inputs), completed_pairs=pairs,
                                          conditions=counts, updated_utc=now(), **extra))


class EvidenceBaseline:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        torch.set_num_threads(4)
        torch.manual_seed(42)
        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, local_files_only=True).cpu().eval()

    def __call__(self, claim):
        import torch
        from retrieve_title_aware import retrieve
        start = time.perf_counter()
        conn = sqlite3.connect((ROOT / 'data/processed/wiki_fts.db').as_uri() + '?mode=ro', uri=True)
        try:
            evidence = retrieve(conn, claim, 5)
        finally:
            conn.close()
        text = ' '.join(e['text'] for e in evidence)
        full = self.tokenizer(claim, text, truncation=False, verbose=False)
        encoded = self.tokenizer(claim, text, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
        with torch.inference_mode():
            logits = self.model(**encoded).logits[0]
            probabilities = torch.softmax(logits, -1).tolist()
        return dict(claim=claim, label=LABELS[int(logits.argmax())], label_order=LABELS,
                    probabilities=probabilities, logits=logits.tolist(), evidence=evidence,
                    token_trace=dict(tokens_before=len(full['input_ids']), tokens_retained=int(encoded['attention_mask'].sum()),
                                     truncated=len(full['input_ids']) > 256, max_length=256,
                                     decoded_retained_input=self.tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=False)),
                    status='complete', device='cpu', wall_seconds=time.perf_counter() - start)


class PartsBackend:
    def __init__(self, protocol):
        from local_llm_experiments import connect, URL
        from finalize_exhaustive_tokens import token_count
        self.session, self.model, version = connect(protocol['model']['name'])
        if self.model['digest'] != protocol['model']['digest'] or version != protocol['ollama']:
            raise ValueError('Llama model/runtime differs from frozen selection')
        response = self.session.post(URL + '/api/generate', json={'model': self.model['name'], 'keep_alive': '5m'}, timeout=180)
        response.raise_for_status()  # load the installed model, no claim prompt
        calibration = load(ROOT / 'reports/part2/exhaustive_token_verified_20261029/protocol.json')
        self.wrapper = calibration['wrapper_tokens']
        if calibration['model']['digest'] != self.model['digest'] or calibration['ollama'] != version:
            raise ValueError('Historical tokenizer calibration does not match this runtime')
        self.calibration = []
        for sample in calibration['calibration'][:2]:
            saved = load(sample['file'])
            count = token_count(self.session, saved['request']['messages'][0]['content'])
            if count['raw_prompt_tokens'] + self.wrapper != sample['reported_prompt_tokens'] or count['token_ids_sha256'] != sample['token_ids_sha256']:
                raise ValueError('Pinned tokenizer/chat-wrapper calibration changed')
            self.calibration.append(count)
        self.count = lambda prompt: token_count(self.session, prompt)

    def generate(self, prompt):
        from local_llm_experiments import call, connect
        from replay_claim_parts import SCHEMA
        check, current, version = connect(self.model['name'])
        check.close()
        if current['digest'] != self.model['digest'] or version != load(ROOT / 'docs/working_parts_prompt.json')['ollama']:
            raise RuntimeError('Model/runtime changed during run')
        return call(self.session, self.model['name'], prompt, 42, SCHEMA, 0, 256)

    def close(self):
        self.session.close()


def baseline_phase(out, pending, inputs):
    """Keep PyTorch's baseline thread runtime separate from FAISS/OpenMP."""
    env = dict(os.environ, OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    process = subprocess.run([sys.executable, str(ROOT / 'src/parts_baseline_worker.py'),
                              '--run', str(out)], env=env)
    if process.returncode:
        raise RuntimeError('Isolated baseline worker exited with code ' + str(process.returncode))


def execute(out):
    from ephemeral_rag import MiniLMEmbedder
    from parts_rag_classifier import classify
    if not out.is_dir():
        raise ValueError('Prepare a new run first')
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = verify_frozen(out)
        inputs = rows(out / 'inputs.jsonl')  # deliberately never read reference_labels here
        if [r['id'] for r in inputs] != protocol['ids'] or any(set(r) != {'id', 'claim'} for r in inputs):
            raise ValueError('Inference input schema or order changed')
        if load(ROOT / 'reports/part2/ollama_full_dev_20261028/batch_status.json')['status'] != 'stopped':
            raise ValueError('Old large evaluation must stay stopped')
        backend = None
        try:
            progress(out, inputs, 'running_baseline')
            pending = [r for r in inputs if completed(out, 'evidence_roberta', r['id']) is None]
            if pending:
                baseline_phase(out, pending, inputs)
            pending = [r for r in inputs if completed(out, 'parts_rag', r['id']) is None]
            if pending:
                # Only orphan workspaces inside this run's own temporary root are eligible for removal.
                for folder in (out / 'temporary').iterdir():
                    if folder.is_symlink() or not folder.is_dir() or not folder.name.startswith('claim-'):
                        raise ValueError('Unexpected temporary workspace; manual inspection required')
                    shutil.rmtree(folder)
                start = time.perf_counter()
                backend = PartsBackend(protocol)
                embedder = MiniLMEmbedder()
                write_new(out / ('rag_setup_' + str(time.time_ns()) + '.json'),
                          dict(seconds=time.perf_counter() - start, calibration=backend.calibration))
                streak = 0
                for row in pending:
                    if shutil.disk_usage(out).free < 2 * 1024**3:
                        raise RuntimeError('Less than 2 GiB disk space remains')
                    progress(out, inputs, 'running_parts_rag', latest_id=row['id'])
                    attempt = out / 'attempts' / str(row['id']) / str(time.time_ns())
                    attempt.mkdir(parents=True)
                    def generate(prompt):
                        raw = backend.generate(prompt)
                        write_new(attempt / 'raw.json', raw)  # preserve before parsing
                        return raw
                    trace = classify(row['claim'], embedder, generate, backend.count, backend.wrapper,
                                     out / 'temporary', save_prepared=lambda trace: write_new(attempt / 'prepared.json', trace))
                    trace['attempt'] = str(attempt.relative_to(out))
                    trace['id'] = row['id']
                    write_new(out / 'traces/parts_rag' / (str(row['id']) + '.json'), trace)
                    print('SAVED', row['id'], trace['label'], trace.get('error', ''), flush=True)
                    if not trace['cleanup']['articles_and_database_deleted'] or any((out / 'temporary').iterdir()):
                        raise RuntimeError('Temporary workspace cleanup failed')
                    infra = trace['status'] != 'complete' and trace['stage'] not in {'validation', 'token_preflight'}
                    streak = streak + 1 if infra else 0
                    if streak >= 5:
                        raise RuntimeError('Five consecutive infrastructure/preparation failures; stopping')
                backend.close()
                backend = None
            verify_frozen(out)
            progress(out, inputs, 'complete_unscored')
        except BaseException as exc:
            progress(out, inputs, 'stopped', error=type(exc).__name__ + ': ' + str(exc))
            raise
        finally:
            if backend is not None:
                backend.close()


def metrics(gold, predictions):
    matrix = [[0] * 4 for _ in LABELS]
    for g, p in zip(gold, predictions):
        matrix[LABELS.index(g)][LABELS.index(p) if p in LABELS else 3] += 1
    per_class = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        fn = sum(matrix[i]) - tp
        fp = sum(row[i] for row in matrix) - tp
        per_class[label] = dict(TP=tp, FP=fp, FN=fn, TN=len(gold)-tp-fp-fn,
                                precision=tp/(tp+fp) if tp+fp else 0,
                                recall=tp/(tp+fn) if tp+fn else 0,
                                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return dict(n=len(gold), accuracy=sum(g == p for g, p in zip(gold, predictions))/len(gold),
                macro_f1=sum(v['f1'] for v in per_class.values())/3, confusion_matrix=matrix,
                prediction_columns=LABELS+['INVALID'], per_class=per_class,
                invalid=sum(p not in LABELS for p in predictions))


def score(out):
    protocol = verify_frozen(out)
    inputs = rows(out / 'inputs.jsonl')
    gold_by_id = load(out / 'reference_labels.json')
    gold = [gold_by_id[str(r['id'])] for r in inputs]
    traces = {c: [completed(out, c, r['id']) for r in inputs] for c in CONDITIONS}
    if any(t is None for ts in traces.values() for t in ts):
        raise ValueError('Both conditions must finish on every frozen claim before scoring')
    predictions = {c: [t['label'] for t in ts] for c, ts in traces.items()}
    summary = {c: dict(**metrics(gold, predictions[c]), mean_wall_seconds=sum(t['wall_seconds'] for t in traces[c])/len(gold)) for c in CONDITIONS}
    summary['parts_model_label'] = metrics(gold, [t['model_label'] for t in traces['parts_rag']])
    differences = [int(p == g)-int(b == g) for g, p, b in zip(gold, predictions['parts_rag'], predictions['evidence_roberta'])]
    rng = random.Random(42)
    boot = sorted(sum(rng.choices(differences, k=len(gold)))/len(gold) for _ in range(5000))
    result = dict(status='complete', n=len(gold), conditions=summary,
                  paired_accuracy_difference=sum(differences)/len(gold),
                  paired_accuracy_95_percentile_interval=[boot[124], boot[4874]], bootstrap_seed=42,
                  fixed=sum(v == 1 for v in differences), introduced=sum(v == -1 for v in differences),
                  inconsistent=sum(t.get('consistent') is False for t in traces['parts_rag']),
                  all_workspaces_deleted=not any((out / 'temporary').iterdir()), limitations=protocol['limitations'])
    def save_analysis(name, data):
        path = out / name
        if path.exists():
            if load(path) != data:
                raise ValueError('Existing analysis differs; preserve it and investigate')
        else:
            write_new(path, data)
    save_analysis('summary.json', result)
    save_analysis('scored_cases.json', [dict(id=r['id'], claim=r['claim'], gold=g,
              baseline=b['label'], main=m['label'], model_label=m['model_label'], parts=m.get('parts', []),
              cited_evidence=m.get('cited_evidence', []), error=m.get('error'))
              for r, g, b, m in zip(inputs, gold, traces['evidence_roberta'], traces['parts_rag'])])
    from report_parts_test import build
    build(out)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'score'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=60, help='Preparation only; balanced sample size')
    args = parser.parse_args()
    def stop(signum, frame):
        raise KeyboardInterrupt('Stop requested; resume skips completed traces')
    signal.signal(signal.SIGTERM, stop)
    out = args.output.resolve()
    if args.mode == 'prepare':
        prepare(out, args.count)
    elif args.mode == 'run':
        execute(out)
    else:
        score(out)


if __name__ == '__main__':
    main()
