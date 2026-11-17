"""Paired 256-vs-512 evidence-input diagnostic on frozen official-dev outputs.

Reuses saved 256-token probabilities and exact saved top-five references. The
claim-only branch stays at its original 256-token inputs; only the evidence
branch changes. No training, retrieval changes, calibration, or test-set access.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import sqlite3
import sys
import time

import numpy as np
from scipy.stats import binomtest
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from analyze_controlled import metrics
from evaluate_official_fever import official_metrics
from evidence_pipeline_v2 import fuse_probabilities
from experiment_paired_ablation import synchronize
from experiment_symbolic_v2 import digest

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
MODELS = ['retrieved_roberta', 'calibrated_ensemble']


def read_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def comparison(gold, left, right):
    delta = (np.asarray(right) == gold).astype(int) - (np.asarray(left) == gold).astype(int)
    wins = int((delta == 1).sum()); losses = int((delta == -1).sum())
    # Equivalent IID paired bootstrap of the per-claim accuracy differences,
    # using their three-category empirical distribution, without a huge N x B array.
    draws = np.random.default_rng(42).multinomial(len(delta),
        [losses / len(delta), 1 - (wins + losses) / len(delta), wins / len(delta)], size=10000)
    ci = np.quantile((draws[:, 2] - draws[:, 0]) / len(delta), [.025, .975]).tolist()
    return dict(n=len(delta), fixed_errors=wins, newly_wrong=losses,
        both_correct=int(((np.asarray(left) == gold) & (np.asarray(right) == gold)).sum()),
        both_wrong=int(((np.asarray(left) != gold) & (np.asarray(right) != gold)).sum()),
        accuracy_delta=float(delta.mean()), paired_accuracy_ci95=ci,
        mcnemar_exact_p_unadjusted=float(binomtest(wins, wins + losses, .5).pvalue) if wins + losses else 1.,
        bootstrap_samples=10000, bootstrap_seed=42)


def per_class(matrix):
    n = sum(map(sum, matrix)); result = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]; fn = sum(matrix[i]) - tp
        fp = sum(row[i] for row in matrix) - tp; tn = n - tp - fp - fn
        result[label] = dict(TP=tp, FP=fp, FN=fn, TN=tn,
            precision=tp / (tp + fp) if tp + fp else 0.,
            recall=tp / (tp + fn) if tp + fn else 0.,
            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    return result


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2)); temp.replace(path)


def analyze(out, gold, saved, rows, protocol):
    targets = np.array([LABELS.index(r['label']) for r in gold])
    results = {}; comparisons = {}; csv_rows = []
    for model in MODELS:
        left = [s['predictions'][model] for s in saved]
        right = [r['predictions'][model] for r in rows]
        for budget, predictions in ((256, left), (512, right)):
            submission = [dict(id=r['id'], predicted_label=LABELS[p], predicted_evidence=s['predicted_evidence'])
                          for r, s, p in zip(gold, saved, predictions)]
            score = official_metrics(gold, submission)
            score['per_class'] = per_class(score['confusion_matrix'])
            results[model + '_' + str(budget)] = score
            for label, values in score['per_class'].items():
                csv_rows.append(dict(model=model, evidence_token_limit=budget, label=label, **values))
        comparisons[model] = comparison(targets, left, right)
        mask = np.array([r['full_input_tokens'] > 256 for r in rows])
        comparisons[model]['truncated_at_256'] = comparison(targets[mask], np.asarray(left)[mask], np.asarray(right)[mask]) if mask.any() else None
        mask = ~mask
        comparisons[model]['already_fit_256'] = comparison(targets[mask], np.asarray(left)[mask], np.asarray(right)[mask]) if mask.any() else None
    summary = dict(status='complete', n=len(gold), protocol=protocol,
        truncation=dict(over_256=sum(r['full_input_tokens'] > 256 for r in rows),
            over_512=sum(r['full_input_tokens'] > 512 for r in rows),
            fully_fit_at_512=sum(r['full_input_tokens'] <= 512 for r in rows)),
        conditions=results, comparisons=comparisons,
        inference_512_seconds=sum(r['batch_inference_seconds'] / r['batch_n'] for r in rows),
        telemundo=next((r for r in rows if r['id'] == 111897), None))
    write_json(out / 'summary.json', summary)
    with (out / 'per_class.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    changed = []
    for g, s, r in zip(gold, saved, rows):
        for model in MODELS:
            a = s['predictions'][model]; b = r['predictions'][model]
            if a != b:
                changed.append(dict(id=g['id'], claim=g['claim'], gold=g['label'], model=model,
                    prediction_256=LABELS[a], prediction_512=LABELS[b],
                    change='fixed' if b == LABELS.index(g['label']) else 'new_error' if a == LABELS.index(g['label']) else 'wrong_to_wrong',
                    full_input_tokens=r['full_input_tokens']))
    with (out / 'changed_predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id','claim','gold','model','prediction_256','prediction_512','change','full_input_tokens'])
        writer.writeheader(); writer.writerows(changed)
    lines = ['# Frozen model: 256 versus 512 evidence-input tokens', '',
        'Same claims, saved top-five evidence, checkpoint and fusion parameters. Claim-only probabilities remain unchanged.', '',
        'This is an exploratory development-set experiment prompted by inspection of Telemundo. It is not a blind test or training-seed repeat.', '',
        '| Condition | Accuracy | Macro-F1 | Strict FEVER |', '|---|---:|---:|---:|']
    for name, score in results.items():
        lines.append('| %s | %.2f%% | %.4f | %.2f%% |' % (name, score['accuracy'] * 100, score['macro_f1'], score['fever_score'] * 100))
    for model, value in comparisons.items():
        lines.extend(['', '%s: %d errors fixed, %d previously correct claims became wrong; accuracy change %+.3f percentage points.' %
                      (model, value['fixed_errors'], value['newly_wrong'], 100 * value['accuracy_delta'])])
    lines.extend(['', 'Inputs longer than 256: %d. Still longer than 512: %d.' %
                  (summary['truncation']['over_256'], summary['truncation']['over_512']), '',
                  'Full per-class TP/FP/FN/TN counts: per_class.csv. Per-claim flips: changed_predictions.csv. Raw probabilities: outputs.jsonl.', '',
                  'Length alone does not establish a human-readable cause. Larger inputs can add useful or distracting text. Retrieval recall cannot change because references are fixed. The evidence checkpoint was trained with a 256-token limit. No retraining or production setting change was made.'])
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    write_json(out / 'progress.json', dict(status='complete', completed=len(rows), total=len(gold)))
    print(json.dumps(dict(n=len(gold), conditions={k:dict(accuracy=v['accuracy'], macro_f1=v['macro_f1']) for k,v in results.items()}, comparisons=comparisons)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--smoke', action='store_true', help='First 64 official claims plus Telemundo, including a 256-token replay')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', choices=['auto', 'mps', 'cpu', 'cuda'], default='auto')
    args = p.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    if device.type == 'cpu': torch.set_num_threads(4)
    benchmark = ROOT / 'reports/part2/official_fever_dev_20261023'
    gold_path = ROOT / 'data/official_fever/shared_task_dev.jsonl'
    saved_path = benchmark / 'outputs.jsonl'
    checkpoint = ROOT / 'checkpoints/roberta_evidence/epoch_1'
    db = ROOT / 'data/processed/wiki_fts.db'
    original_protocol = json.loads((benchmark / 'protocol.json').read_text())
    gold = read_jsonl(gold_path); saved = read_jsonl(saved_path)
    if len(gold) != 19998 or [r['id'] for r in gold] != [r['id'] for r in saved]:
        raise ValueError('Expected matching full official dev IDs in original order')
    if any(g['claim'] != s['claim'] for g,s in zip(gold,saved)):
        raise ValueError('Claim mismatch')
    if args.smoke:
        positions = list(range(64)); target = next(i for i,g in enumerate(gold) if g['id']==111897)
        if target not in positions: positions.append(target)
        gold = [gold[i] for i in positions]; saved = [saved[i] for i in positions]
    paths = [gold_path, saved_path, Path(__file__), benchmark / 'protocol.json', ROOT / 'src/evidence_pipeline_v2.py', ROOT / 'src/vendor/fever_scorer.py']
    paths += sorted(f for f in checkpoint.iterdir() if f.is_file())
    hashes = {str(f.relative_to(ROOT)): digest(f) for f in paths}
    for f in checkpoint.iterdir():
        key = str(f.relative_to(ROOT))
        if f.is_file() and original_protocol['input_sha256'].get(key) != hashes[key]:
            raise ValueError('Checkpoint differs from saved benchmark: ' + key)
    protocol = dict(experiment='evidence_input_256_vs_512', n=len(gold), smoke=args.smoke,
        population='First 64 official dev claims plus Telemundo' if args.smoke else 'All 19998 official FEVER development claims in original order',
        labels=LABELS, evidence_lengths=[256,512], batch_size=8, device=str(device), seed=42,
        original_device='mps', claim_branch='Frozen saved 256-token probabilities, unchanged in both conditions',
        evidence_branch='Same checkpoint; exact original top-five references and text order; max_length only changes to 512',
        fusion=original_protocol['frozen_calibration_config'], input_sha256=hashes,
        db=dict(path=str(db.relative_to(ROOT)), bytes=db.stat().st_size, mtime_ns=db.stat().st_mtime_ns),
        runtime=dict(python=sys.version, torch=torch.__version__, transformers=transformers.__version__),
        limitations=['Exploratory dev-set analysis after inspection of Telemundo; not a new untouched test.',
                     'No tuning, training, or claim-only input-length change.',
                     'Baseline probabilities reused; smoke check independently reproduces 256-token inference.',
                     '512 can still truncate long inputs; training used 256.',
                     'Comparisons include multiple model branches; p-values are unadjusted.'])
    out = Path(args.output)
    if args.resume:
        if json.loads((out/'protocol.json').read_text()) != protocol: raise ValueError('Resume protocol or input mismatch')
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/'protocol.json',protocol)
        (out/'runner_source.py').write_text(Path(__file__).read_text())
    path = out/'outputs.jsonl'
    rows = read_jsonl(path) if args.resume and path.exists() else []
    if [r['id'] for r in rows] != [g['id'] for g in gold[:len(rows)]] or len(rows)>len(gold):
        raise ValueError('Saved prefix differs')
    if len(rows)==len(gold):
        analyze(out,gold,saved,rows,protocol);return
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
    if model.config.max_position_embeddings < 514 or model.config.num_labels != 3:
        raise ValueError('Checkpoint cannot support requested 512-token input')
    conn = sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)
    def inputs(batch):
        texts = []
        for s in batch:
            texts_for_claim = []
            for page,sid in s['predicted_evidence']:
                row = conn.execute('SELECT text FROM page_sentences WHERE page_id=? AND sentence_id=?',(page,sid)).fetchone()
                if row is None: raise ValueError('Saved evidence is missing from snapshot')
                texts_for_claim.append(row[0])
            texts.append(' '.join(texts_for_claim))
        return texts
    def infer(claims,texts,length):
        encoded = tokenizer(claims,text_pair=texts,padding='max_length',truncation=True,max_length=length,return_tensors='pt')
        synchronize(device);start=time.perf_counter()
        with torch.inference_mode():
            probability=torch.softmax(model(**{k:v.to(device) for k,v in encoded.items()}).logits,-1).cpu().numpy()
        synchronize(device)
        return probability, time.perf_counter()-start, encoded
    # Same warm-up for 512 to exclude startup from the main per-batch timing.
    infer([gold[0]['claim']],inputs(saved[:1]),512)
    if args.smoke:
        maximum_difference=0.; mismatches=0;seconds=0.
        for index in range(0,len(gold),8):
            g=gold[index:index+8];s=saved[index:index+8]
            probability,elapsed,_=infer([r['claim'] for r in g],inputs(s),256)
            reference=np.asarray([r['model_probabilities']['evidence'] for r in s])
            maximum_difference=max(maximum_difference,float(abs(probability-reference).max()))
            mismatches+=int((probability.argmax(1)!=reference.argmax(1)).sum());seconds+=elapsed
        replay=dict(n=len(gold),max_probability_absolute_difference=maximum_difference,label_mismatches=mismatches,inference_seconds=seconds)
        write_json(out/'baseline_replay.json',replay)
        print('256 replay',json.dumps(replay),flush=True)
        if mismatches or maximum_difference>1e-3:raise ValueError('Saved baseline replay failed')
    started=time.perf_counter();initial=len(rows)
    with path.open('a' if args.resume else 'x') as stream:
        for index in range(initial,len(gold),8):
            g=gold[index:index+8];s=saved[index:index+8];texts=inputs(s)
            claims=[r['claim'] for r in g]
            probability,seconds,encoded=infer(claims,texts,512)
            full=tokenizer(claims,text_pair=texts,truncation=False,padding=False,verbose=False)
            combined=fuse_probabilities([r['model_probabilities']['claim'] for r in s],probability,
                protocol['fusion']['claim_weight'],protocol['fusion']['nei_bias'])
            for j,(record,old) in enumerate(zip(g,s)):
                result=dict(id=record['id'],claim=record['claim'],gold_label=record['label'],
                    predicted_evidence=old['predicted_evidence'],evidence_text_sha256=__import__('hashlib').sha256(texts[j].encode()).hexdigest(),
                    model_probabilities=dict(claim=old['model_probabilities']['claim'],evidence=probability[j].tolist()),
                    predictions=dict(retrieved_roberta=int(probability[j].argmax()),calibrated_ensemble=int(combined[j])),
                    baseline_predictions=old['predictions'],full_input_tokens=len(full['input_ids'][j]),
                    retained_tokens_512=int(encoded['attention_mask'][j].sum()),batch_inference_seconds=seconds,batch_n=len(g))
                stream.write(json.dumps(result)+'\n');rows.append(result)
            stream.flush()
            if len(rows)%128==0 or len(rows)==len(gold):
                elapsed=time.perf_counter()-started
                progress=dict(status='inference_running',completed=len(rows),total=len(gold),session_seconds=elapsed,
                    session_claims=len(rows)-initial,estimated_remaining_seconds=(len(gold)-len(rows))*elapsed/(len(rows)-initial))
                write_json(out/'progress.json',progress)
                print('PROGRESS',json.dumps(progress),flush=True)
    conn.close()
    analyze(out,gold,saved,rows,protocol)


if __name__=='__main__':main()
