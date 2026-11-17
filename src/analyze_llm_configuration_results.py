"""Export final decisions and per-class counts from a completed configuration run.

No inference or experiment selection. Raw model outputs remain unchanged; hybrid
labels are explicitly distinguished from raw LLM labels in every exported row.
"""
import csv
import html
import json
from pathlib import Path
import shutil

import numpy as np

from evaluate_retrieval import gold_evidence_groups
from experiment_llm_configurations import OUT, LABELS, CONFIGS, derive, load, score
from experiment_symbolic_v2 import digest
from experiment_token_budget import per_class, comparison


def paired_macro_f1_interval(gold, left, right, samples=10000, seed=42):
    """Paired IID claim bootstrap via the empirical joint outcome distribution."""
    joint=np.zeros((3,4,4),dtype=int)
    for g,a,b in zip(gold,left,right):joint[g,a,b]+=1
    draws=np.random.default_rng(seed).multinomial(len(gold),joint.ravel()/len(gold),size=samples).reshape(samples,3,4,4)
    def macro(matrix):
        value=np.zeros(samples)
        for i in range(3):
            tp=matrix[:,i,i];den=matrix[:,i,:].sum(1)+matrix[:,:,i].sum(1)
            value+=np.divide(2*tp,den,out=np.zeros(samples),where=den!=0)/3
        return value
    return np.quantile(macro(draws.sum(axis=2))-macro(draws.sum(axis=3)),[.025,.975]).tolist()


def main():
    out=OUT;summary=json.loads((out/'assessment_summary.json').read_text())
    all_rows=load(out/'inputs.jsonl');records=[];counts=[];lookups={}
    for phase in ['calibration','assessment']:
        rows=[r for r in all_rows if r['split']==phase]
        responses=load(out/(phase+'_responses.jsonl'))
        lookup={(r['config'],r['id']):r for r in responses};lookups[phase]=lookup
        conditions=list(CONFIGS) if phase=='calibration' else list(dict.fromkeys(['A_original',summary['selected']]))
        metrics=json.loads((out/('calibration_summary.json' if phase=='calibration' else 'assessment_summary.json')).read_text())
        if phase=='assessment':metrics=metrics['conditions']
        for config in conditions:
            labels,override_count=derive(rows,responses,config)
            raw_config='D_both' if config=='E_claim_fallback' else config
            for row,label in zip(rows,labels):
                raw=lookup[(raw_config,row['id'])];claim_index=int(np.argmax(row['claim_probabilities']))
                evidence=row['evidence'][CONFIGS[config]['retrieval']]
                found={(r['page_id'],r['sentence_id']) for r in evidence}
                complete=any(g<=found for g in gold_evidence_groups(row['gold'])) if row['gold']['label']!=LABELS[2] else None
                records.append(dict(phase=phase,config=config,id=row['id'],claim=row['claim'],gold=row['gold']['label'],
                    raw_llm_label=raw['label'],raw_llm_reason=raw.get('reason',''),final_label=label,
                    claim_override=label!=raw['label'],claim_only_label=LABELS[claim_index],claim_max_softmax=max(row['claim_probabilities']),
                    correct=label==row['gold']['label'],complete_annotated_group=complete,
                    submitted_evidence=[[r['page_id'],r['sentence_id']] for r in evidence],
                    raw_response_file=phase+'_responses.jsonl',raw_response_config=raw_config))
            assert sum(r['claim_override'] for r in records if r['phase']==phase and r['config']==config)==override_count
            for label,values in per_class(metrics[config]['confusion_matrix']).items():
                counts.append(dict(phase=phase,config=config,label=label,**values))
    (out/'final_decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    with (out/'per_class.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(counts[0]));writer.writeheader();writer.writerows(counts)
    selected=summary['selected'];rows=[r for r in all_rows if r['split']=='assessment'];lookup=lookups['assessment']
    baseline={r['id']:r for r in records if r['phase']=='assessment' and r['config']=='A_original'}
    final={r['id']:r for r in records if r['phase']=='assessment' and r['config']==selected}
    fixed=[r for r in rows if not baseline[r['id']]['correct'] and final[r['id']]['correct']]
    regressed=[r for r in rows if baseline[r['id']]['correct'] and not final[r['id']]['correct']]
    y=[LABELS.index(r['gold']['label']) for r in rows]
    left=[LABELS.index(baseline[r['id']]['final_label']) if baseline[r['id']]['final_label'] in LABELS else 3 for r in rows]
    right=[LABELS.index(final[r['id']]['final_label']) if final[r['id']]['final_label'] in LABELS else 3 for r in rows]
    def strict_flags(decisions):
        return [int(decisions[r['id']]['correct'] and (r['gold']['label']==LABELS[2] or decisions[r['id']]['complete_annotated_group'])) for r in rows]
    strict=comparison(np.ones(len(rows),dtype=int),strict_flags(baseline),strict_flags(final))
    uncertainty=dict(paired_macro_f1_ci95=paired_macro_f1_interval(y,left,right),paired_strict_score=strict,bootstrap_seed=42,bootstrap_samples=10000,
        interpretation='Paired claim bootstrap with frozen predictions; not training-seed or generation-repeat variance. Exploratory secondary intervals, no multiplicity adjustment.')
    (out/'assessment_uncertainty.json').write_text(json.dumps(uncertainty,indent=2))
    component=None
    if selected=='E_claim_fallback':
        raw_values=[final[r['id']]['raw_llm_label'] for r in rows]
        component=dict(raw_llm_component=score(rows,raw_values,'D_both'),
            overrides=sum(final[r['id']]['claim_override'] for r in rows),
            fixed=sum(raw_values[i]!=r['gold']['label'] and final[r['id']]['correct'] for i,r in enumerate(rows)),
            introduced=sum(raw_values[i]==r['gold']['label'] and not final[r['id']]['correct'] for i,r in enumerate(rows)),
            interpretation='Component audit of the preselected hybrid using its already-generated LLM outputs. No new requests or assessment-driven selection changes.')
        (out/'assessment_component_diagnostic.json').write_text(json.dumps(component,indent=2))
    with (out/'REPORT.md').open('a') as stream:
        stream.write('\nPaired macro-F1 difference 95%% interval: %+.4f to %+.4f.\n'%(uncertainty['paired_macro_f1_ci95'][0],uncertainty['paired_macro_f1_ci95'][1]))
        stream.write('Paired strict-FEVER difference: %+.2f percentage points; 95%% interval %+.2f to %+.2f points.\n'%(100*strict['accuracy_delta'],100*strict['paired_accuracy_ci95'][0],100*strict['paired_accuracy_ci95'][1]))
        if component:stream.write('Selected hybrid component audit: raw selected LLM accuracy %.2f%%; fallback changed %d labels, fixing %d errors and introducing %d. This does not change the frozen selection.\n'%(100*component['raw_llm_component']['accuracy'],component['overrides'],component['fixed'],component['introduced']))
    esc=html.escape;parts=['<!-- configuration-details --><section><h2>Examples: fixes and regressions</h2><p>First two cases of each type in assessment order. These illustrate changes; the aggregate results above include every assessment claim.</p>']
    for kind,examples in [('Fixed',fixed[:2]),('New error',regressed[:2])]:
        for row in examples:
            old=baseline[row['id']];new=final[row['id']]
            parts.append('<h3>'+kind+' · '+esc(row['claim'])+'</h3><p>Gold: <b>'+esc(row['gold']['label'])+'</b> · original LLM: '+esc(old['final_label'])+' → selected pipeline: '+esc(new['final_label'])+'</p>')
            if new['claim_override']:
                parts.append('<p>The raw selected LLM answered '+esc(new['raw_llm_label'])+'. The fixed claim-only fallback changed the final label. Its generated explanation below belongs to the raw LLM answer, not the fallback.</p>')
            parts.append('<p><b>Original LLM explanation:</b> '+esc(old['raw_llm_reason'])+'<br><b>Selected LLM explanation:</b> '+esc(new['raw_llm_reason'])+'</p>')
            a=lookup[('A_original',row['id'])];b=lookup[('D_both' if selected=='E_claim_fallback' else selected,row['id'])]
            trace=dict(id=row['id'],gold=row['gold']['label'],baseline=a,selected_llm=b,final_decision=new)
            parts.append('<details><summary>Exact saved requests, responses and final decision</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px">'+esc(json.dumps(trace,indent=2))+'</pre></details>')
    parts.append('</section><section><h2>Per-class outcomes on the 120 assessment claims</h2><div class="scroll"><table><tr><th>Configuration</th><th>Positive label</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th></tr>')
    for row in counts:
        if row['phase']=='assessment':parts.append('<tr><td>%s</td><td>%s</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td></tr>'%(esc(row['config']),esc(row['label']),row['TP'],row['FP'],row['FN'],row['TN']))
    parts.append('</table></div><small>One-versus-rest counts. A TN can still confuse the two other FEVER labels.</small></section>')
    p=out/'index.html';page=p.read_text()
    if '<!-- configuration-details -->' in page:raise ValueError('Details already appended; regenerate base report before rerunning')
    extra='<section><h2>Paired uncertainty on the assessment sample</h2><p>Macro-F1 difference, 95%% interval: %+.4f to %+.4f.</p><p>Strict FEVER difference: %+.2f percentage points; 95%% interval: %+.2f to %+.2f points.</p><small>Paired resampling of claims with predictions fixed, not a generation or training repeat. Exploratory intervals without multiplicity adjustment.</small></section>'%(uncertainty['paired_macro_f1_ci95'][0],uncertainty['paired_macro_f1_ci95'][1],100*strict['accuracy_delta'],100*strict['paired_accuracy_ci95'][0],100*strict['paired_accuracy_ci95'][1])
    if component:extra+='<section><h2>How much did the fallback contribute?</h2><p>The selected hybrid’s raw LLM component scored %.2f%% accuracy. The fixed fallback changed %d labels: %d errors fixed and %d introduced.</p><small>This audits a component of the preselected hybrid using existing responses. It does not change the selection or create another chosen model after seeing assessment results.</small></section>'%(100*component['raw_llm_component']['accuracy'],component['overrides'],component['fixed'],component['introduced'])
    page=page.replace('</html>',extra+''.join(parts)+'</html>');p.write_text(page)
    manifest=dict(source_sha256=digest(Path(__file__)),final_decisions_sha256=digest(out/'final_decisions.jsonl'),per_class_sha256=digest(out/'per_class.csv'),
        note='Additional exports and presentation only; no model calls, selection changes or prediction changes.')
    (out/'detail_manifest.json').write_text(json.dumps(manifest,indent=2));shutil.copy2(Path(__file__),out/'details_source.py')
    print('Exported',len(records),'final decision records and per-class counts.')


if __name__=='__main__':main()
