"""Re-score existing Ollama outputs without inference; preserve initial reports.

Corrects claim-only strict FEVER reporting (claim-only submits no evidence),
adds paired strict-score uncertainty, and improves display labels. The frozen
inference protocol and exact original runner snapshot remain unchanged.
"""
import json
from pathlib import Path
import shutil

import numpy as np

from evaluate_retrieval import gold_evidence_groups
from experiment_ollama_fever import LABELS, analyze, load
from experiment_symbolic_v2 import digest
from experiment_token_budget import comparison

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'reports/part2/ollama_comparison_20261028'


def main():
    protocol=json.loads((RUN/'protocol.json').read_text())
    runner=str(ROOT/'src/experiment_ollama_fever.py')
    # The inference source is preserved exactly; reporting corrections do not
    # rewrite the provenance of requests that have already been executed.
    for path,expected in protocol['input_sha256'].items():
        target=RUN/'runner_source.py' if path==runner else Path(path)
        if digest(target)!=expected:raise ValueError('Frozen inference input changed: '+path)
    archive=RUN/'initial_scoring'
    archive.mkdir(exist_ok=False)
    for name in ['summary.json','REPORT.md','index.html','per_class.csv','predictions.csv','audit.json']:
        if (RUN/name).exists():shutil.copy2(RUN/name,archive/name)
    rows=load(RUN/'sample.jsonl');responses=load(RUN/'responses.jsonl')
    if len(rows)!=300 or [r['id'] for r in rows]!=[r['id'] for r in responses]:raise ValueError('Incomplete or unpaired outputs')
    analyze(RUN,rows,responses,protocol)
    summary=json.loads((RUN/'summary.json').read_text())
    def strict_flags(predictions):
        flags=[]
        for r,p in zip(rows,predictions):
            found=set(map(tuple,r['predicted_evidence']))
            flags.append(int(p==r['gold']['label'] and (p=='NOT ENOUGH INFO' or any(group<=found for group in gold_evidence_groups(r['gold'])))))
        return flags
    left=strict_flags([LABELS[r['baselines']['combined_512']] for r in rows])
    right=strict_flags([r['label'] for r in responses])
    result=comparison(np.ones(300,dtype=int),left,right)
    summary['strict_comparison_vs_combined_512']=result
    summary['reporting_correction']='Claim-only strict FEVER uses no submitted evidence. Label metrics and LLM-versus-evidence/combined comparisons unchanged. See analysis_manifest.json and initial_scoring/.'
    (RUN/'summary.json').write_text(json.dumps(summary,indent=2))
    with (RUN/'REPORT.md').open('a') as stream:
        stream.write('\nStrict FEVER, LLM minus combined_512: %+.2f percentage points; paired 95%% interval %.2f to %.2f. The interval includes zero; this pilot does not establish a reliable strict-score improvement.\n'%(100*result['accuracy_delta'],100*result['paired_accuracy_ci95'][0],100*result['paired_accuracy_ci95'][1]))
        stream.write('\nReporting audit corrected only the claim-only strict-score condition to submit no evidence; original inference and comparison labels are unchanged. Initial reports are retained in initial_scoring/.\n')
    p=RUN/'index.html';page=p.read_text()
    names={'claim_only_256':'Claim-only RoBERTa · 256','evidence_256':'Evidence RoBERTa · 256','combined_256':'Combined RoBERTa · 256','evidence_512':'Evidence RoBERTa · 512','combined_512':'Combined RoBERTa · 512','local_llm':'Llama 3.1 8B · Ollama'}
    for name,label in names.items():page=page.replace(name,label)
    page=page.replace('<h2>Telemundo: separate diagnostic</h2>','<h2>Telemundo: separate diagnostic</h2>')
    note='<section><h2>Two metrics, different outcomes</h2><p>The LLM has lower label accuracy and macro-F1 on this sample, but a higher strict FEVER score. Strict FEVER credits correct NEI answers directly and requires complete evidence for verifiable claims. The LLM correctly identified 66 NEI cases versus 42 for the combined model, but identified fewer SUPPORTS and REFUTES cases correctly.</p><p>Strict-score difference: %+.2f percentage points. Paired bootstrap 95%% interval: %.2f to %.2f points; this includes zero.</p></section>'%(100*result['accuracy_delta'],100*result['paired_accuracy_ci95'][0],100*result['paired_accuracy_ci95'][1])
    page=page.replace('<section><h2>Telemundo: separate diagnostic</h2>',note+'<section><h2>Telemundo: separate diagnostic</h2>')
    page=page.replace('<h2>What changed compared with Combined RoBERTa · 512?</h2>','<h2>What changed compared with Combined RoBERTa · 512?</h2><small>Claim links open the original 256-token trace. The comparison labels below are the saved 512-token condition.</small>')
    p.write_text(page)
    manifest=dict(reason='Correct claim-only strict-score evidence submission; add strict-score uncertainty and human-readable display names. No new LLM inference or prediction changes.',
        frozen_inference_source_sha256=digest(RUN/'runner_source.py'),analysis_source_sha256=digest(ROOT/'src/experiment_ollama_fever.py'),
        finalizer_source_sha256=digest(Path(__file__)),preserved_initial_reports='initial_scoring/')
    (RUN/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2))
    shutil.copy2(ROOT/'src/experiment_ollama_fever.py',RUN/'analysis_source.py')
    shutil.copy2(Path(__file__),RUN/'finalizer_source.py')
    print('Finalized reports without new inference.',json.dumps(result))


if __name__=='__main__':main()
