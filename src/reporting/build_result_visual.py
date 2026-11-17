"""Build a portable, chart-first view of existing research results. No inference."""
import json
from pathlib import Path
import shutil
import zipfile
from ollama_trace_viewer import source_function

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/result_analysis_20261028'


def read(path):return json.loads((ROOT/path).read_text())
def rows(path):return [json.loads(line) for line in (ROOT/path).read_text().splitlines() if line.strip()]


def main():
    full=read('reports/part2/official_fever_dev_20261023/summary.json')
    assess=read('reports/part2/llm_configurations_20261028/assessment_summary.json')
    component=read('reports/part2/llm_configurations_20261028/assessment_component_diagnostic.json')
    pilot=read('reports/part2/ollama_comparison_20261028/summary.json')
    neural_ref=read('reports/result_analysis_20261028/derived_assessment_neural_reference.json')
    cases=read('reports/result_analysis_20261028/evidence_cases.json')
    def standard(s):return dict(accuracy=s.get('accuracy',s.get('label_accuracy')),macro_f1=s.get('macro_f1',s.get('classification_macro_f1')),fever_score=s['fever_score'],matrix=s['confusion_matrix'])
    main_models=[]
    for key,label,color in [('claim_only_H2','Claim-only H2','#96a9b7'),('retrieved_roberta','Evidence RoBERTa','#527da9'),('calibrated_ensemble','Combined model','#167f7b')]:
        main_models.append(dict(key=key,label=label,color=color,**standard(full['conditions'][key])))
    assessment=[]
    for key,label,color,s in [('A_original','Original LLM','#96a9b7',assess['conditions']['A_original']),('raw_component','New prompt + retrieval','#527da9',component['raw_llm_component']),('E_claim_fallback','Selected hybrid','#167f7b',assess['conditions']['E_claim_fallback'])]:
        assessment.append(dict(key=key,label=label,color=color,**standard(s)))
    nn=assess['neural_reference']['neural_combined_512']
    assessment.append(dict(key='neural_reference',label='Existing combined RoBERTa',color='#3d5669',accuracy=nn['accuracy'],macro_f1=nn['macro_f1'],fever_score=neural_ref['fever_score'],matrix=nn['confusion_matrix']))
    inputs={r['id']:r for r in rows('reports/part2/llm_configurations_20261028/inputs.jsonl') if r['split']=='assessment'}
    raw={(r['config'],r['id']):r for r in rows('reports/part2/llm_configurations_20261028/assessment_responses.jsonl')}
    decisions={(r['config'],r['id']):r for r in rows('reports/part2/llm_configurations_20261028/final_decisions.jsonl') if r['phase']=='assessment'}
    waffle=[]
    token_records={r['id']:r for r in rows('reports/part2/token_budget_20261028/outputs.jsonl') if r['id'] in inputs}
    for ident,r in inputs.items():
        a=decisions[('A_original',ident)];b=decisions[('E_claim_fallback',ident)]
        status='both-correct' if a['correct'] and b['correct'] else 'fixed' if b['correct'] else 'new-error' if a['correct'] else 'both-wrong'
        waffle.append(dict(id=ident,claim=r['claim'],gold=r['gold']['label'],original=a['final_label'],selected=b['final_label'],
            raw_selected=b['raw_llm_label'],override=b['claim_override'],original_reason=a['raw_llm_reason'],selected_reason=b['raw_llm_reason'],status=status,
            original_evidence=r['evidence']['original'],selected_evidence=r['evidence']['entity'],
            original_response=raw[('A_original',ident)].get('raw',{}).get('response'),selected_response=raw[('D_both',ident)].get('raw',{}).get('response'),
            original_request=raw[('A_original',ident)]['raw']['request'],selected_request=raw[('D_both',ident)]['raw']['request'],
            neural=token_records[ident],neural_label=['SUPPORTS','REFUTES','NOT ENOUGH INFO'][token_records[ident]['predictions']['calibrated_ensemble']],
            claim_probabilities=r['claim_probabilities']))
    from collections import Counter
    assert Counter(r['status'] for r in waffle)=={'both-correct':46,'fixed':25,'new-error':16,'both-wrong':33}
    descriptions=[
      (111897,'Telemundo · language','English-language','Spanish-language','≠','Claim language','Wikipedia language','REFUTES','SUPPORTS','REFUTES','The contradiction was visible. RoBERTa still missed it.',True),
      (146231,'Juventus · wrong topic','Juventus uniforms','Black/White pages','↛','Claim subject','Retrieved topics','REFUTES','NOT ENOUGH INFO',None,'The search missed the Juventus article.',False),
      (149173,'Psych · wrong year','2004','2006','≠','Claim year','Wikipedia debut','REFUTES','SUPPORTS',None,'The model called two different years “close”.',True),
      (75152,'Tool · a useful fix','No world tours','Worldwide tours','≠','Claim','Selected evidence','REFUTES','NOT ENOUGH INFO','REFUTES','The selected evidence supports the corrected label.',True),
      (121517,'Halsey · right label, bad reason','Halsey’s EP','A racehorse','?','Claim subject','Retrieved topic','REFUTES','NOT ENOUGH INFO','REFUTES','A correct label does not validate this explanation.',False),
      (52731,'Duff McKagan · different properties','Vietnam','1964','?','Claim birthplace','Evidence birth year','NOT ENOUGH INFO','NOT ENOUGH INFO','REFUTES','A birth year does not answer a birthplace claim.',True)]
    examples=[]
    for ident,title,left,right,relation,left_label,right_label,gold,old,new,message,relevant in descriptions:
        c=next(c for c in cases if c['id'].endswith('#'+str(ident)))
        examples.append(dict(id=ident,title=title,left=left,right=right,relation=relation,left_label=left_label,right_label=right_label,gold=gold,
            original=old,selected=new,claim=c['claim'],message=message,relevant=relevant,reason=c.get('raw_llm_reason',c['evidence']),sentences=c['sentences'],source=c['source']))
    pilot_pair=[dict(key='combined_512',label='Combined RoBERTa',**standard(pilot['conditions']['combined_512'])),dict(key='local_llm',label='Llama 3.1 8B',**standard(pilot['conditions']['local_llm']))]
    labels=['SUPPORTS','REFUTES','NOT ENOUGH INFO']
    catalog=[]
    gold={r['id']:r['label'] for r in rows('data/official_fever/shared_task_dev.jsonl')}
    for r in rows('reports/part2/official_fever_dev_20261023/outputs.jsonl'):
        catalog.append(dict(id=r['id'],claim=r['claim'],gold=gold[r['id']],predictions={k:labels[v] for k,v in r['predictions'].items()}))
    pilot_inputs=rows('reports/part2/ollama_comparison_20261028/sample.jsonl')
    pilot_raw={r['id']:r for r in rows('reports/part2/ollama_comparison_20261028/responses.jsonl')}
    pilot_catalog=[dict(id=r['id'],claim=r['claim'],gold=r['gold']['label'],predictions=dict(combined_512=labels[r['baselines']['combined_512']],local_llm=pilot_raw[r['id']]['label']),
                        evidence=r['sentences'],llm=pilot_raw[r['id']],neural=token_records.get(r['id'])) for r in pilot_inputs]
    # Pilot and assessment are disjoint. Include the exact saved 512-token neural records for the pilot too.
    pilot_ids={r['id'] for r in pilot_catalog}
    pilot_neural={r['id']:r for r in rows('reports/part2/token_budget_20261028/outputs.jsonl') if r['id'] in pilot_ids}
    for r in pilot_catalog:r['neural']=pilot_neural[r['id']]
    data=dict(main=main_models,assessment=assessment,comparison=assess['paired_selected_vs_baseline'],waffle=waffle,examples=examples,catalog=catalog,pilot_catalog=pilot_catalog,
        call_source=source_function(ROOT/'src/local_llm_experiments.py','call'),
        heatmap_pairs=dict(assessment=[assessment[0],assessment[2]],official=[main_models[0],main_models[2]],official_evidence=[main_models[1],main_models[2]],pilot=pilot_pair),
        cohort_sizes=dict(assessment=120,official=19998,official_evidence=19998,pilot=300),
        latency=[dict(label='Retrieval + RoBERTa',value=66.35285959,color='#527da9'),dict(label='Symbolic route + fallback',value=106.5699938,color='#c27826')],
        literature=[dict(label='Our combined model',accuracy=main_models[2]['accuracy'],fever_score=main_models[2]['fever_score'],published=False),dict(label='GEAR · published',accuracy=.7484,fever_score=.7069,published=True),dict(label='KGAT RoBERTa-large · published',accuracy=.7829,fever_score=.7611,published=True)],
        provenance=dict(model='llama3.1:8b',quantization='Q4_K_M',new_inference=False,scope='Retrospective adopted scope; not preregistered',source_manifest='../evidence_manifest.json'))
    target=OUT/'visual';target.mkdir(exist_ok=True)
    for name in ['index.html','visual.css','visual.js']:shutil.copy2(ROOT/'web/evidence_explorer/result_visual'/name,target/name)
    (target/'data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    (target/'data.js').write_text('window.RESULT_DATA = '+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n')
    (target/'README.txt').write_text('Open index.html in a browser. Charts, all claim catalogs, and the saved pilot/assessment traces use bundled data.js. Full benchmark Wikipedia/token details require the local research server. No model requests are made. Saved measurements, cohort sizes and uncertainty are preserved. The full written analysis is included.\n')
    shutil.copy2(Path(__file__),target/'build_source.py')
    archive=OUT/'Result_Analysis_Visual.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(target.iterdir()):
            if p.is_file() and p.name!='index.html':z.write(p,p.name)
        offline=(target/'index.html').read_text().replace('href="../result-analysis-details"','href="RESULT_ANALYSIS.md"').replace('href="../result-analysis.md"','href="RESULT_ANALYSIS.md"').replace('href="../result-analysis.zip"','href="Result_Analysis_Evidence.zip"')
        offline=offline.replace('<a href="../">← Evidence explorer</a>','<a href="RESULT_ANALYSIS.md">Written analysis</a>').replace('<a href="../result-analysis-visual.zip">Download visuals ↓</a>','<span>Offline visual summary</span>')
        z.writestr('index.html',offline)
        for name in ['RESULT_ANALYSIS.md','ADOPTED_SCOPE.md','Result_Analysis_Evidence.zip']:z.write(OUT/name,name)
    print('Visual data:',len((target/'data.js').read_bytes()),'bytes; standalone bundle:',archive.stat().st_size,'bytes.')


if __name__=='__main__':main()
