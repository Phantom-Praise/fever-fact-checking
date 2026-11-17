"""Build the adopted scope and evidence-linked tabular Result Analysis.

Reads existing artifacts only. No inference, training, new hypotheses disguised
as preregistered predictions, or modifications to prior numerical results.
"""
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

from experiment_token_budget import per_class
from vendor.fever_scorer import fever_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/result_analysis_20261028'
LABELS=['SUPPORTS','REFUTES','NOT ENOUGH INFO']
TITLE='Evidence Inputs and Decision Rules in a Local FEVER Fact-Checker'
QUESTION='How do evidence inputs and decision rules affect label accuracy and evidence-grounded verification in a local FEVER fact-checker?'


def read(path):return json.loads((ROOT/path).read_text())
def rows(path):return [json.loads(line) for line in (ROOT/path).read_text().splitlines() if line.strip()]
def pct(value):return f'{100*value:.2f}%'
def f1(value):return f'{value:.4f}'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists():raise FileExistsError(OUT)
    official=read('reports/part2/official_fever_dev_20261023/summary.json')
    ablation=read('reports/part2/evidence_depth_20261023/summary.json')
    token=read('reports/part2/token_budget_20261028/summary.json')
    pilot=read('reports/part2/ollama_comparison_20261028/summary.json')
    config=read('reports/part2/llm_configurations_20261028/assessment_summary.json')
    cal=read('reports/part2/llm_configurations_20261028/calibration_summary.json')
    uncertainty=read('reports/part2/llm_configurations_20261028/assessment_uncertainty.json')
    error=read('reports/part2/ollama_error_analysis_20261028/analysis.json')
    oracle=read('reports/part2/ollama_error_analysis_20261028/oracle_summary.json')
    symbolic=read('reports/part2/step2_final_20261023/heldout100/summary.json')
    protocols=read('reports/part2/llm_configurations_20261028/protocol.json')
    fused=official['conditions']['calibrated_ensemble']
    assessment_inputs=[r for r in rows('reports/part2/llm_configurations_20261028/inputs.jsonl') if r['split']=='assessment']
    neural_submission=[dict(id=r['id'],predicted_label=LABELS[r['neural_combined_512']],predicted_evidence=[[e['page_id'],e['sentence_id']] for e in r['evidence']['original']]) for r in assessment_inputs]
    neural_strict,neural_accuracy,_,_,_=fever_score(neural_submission,[r['gold'] for r in assessment_inputs],max_evidence=5)
    assert neural_accuracy==config['neural_reference']['neural_combined_512']['accuracy']

    source_specs=[
      ('E1','reports/part2/official_fever_dev_20261023/summary.json','Full 19,998-claim benchmark and paired intervals'),
      ('E2','reports/part2/evidence_depth_20261023/summary.json','100-claim evidence-input ablations'),
      ('E3','reports/part2/token_budget_20261028/summary.json','Frozen evidence-input 256 versus 512 comparison'),
      ('E4','reports/part2/ollama_comparison_20261028/summary.json','Matched 300-claim local LLM pilot'),
      ('E5','reports/part2/llm_configurations_20261028/assessment_summary.json','120-claim assessment after calibration-only selection'),
      ('E5a','reports/part2/llm_configurations_20261028/calibration_summary.json','Five configurations on 60 calibration claims'),
      ('E5b','reports/part2/llm_configurations_20261028/selected_config.json','Selection frozen before assessment'),
      ('E5c','reports/part2/llm_configurations_20261028/assessment_start.json','Pre-assessment selection hash and timestamp'),
      ('E5d','reports/part2/llm_configurations_20261028/assessment_uncertainty.json','Paired macro-F1 and strict-score intervals'),
      ('E6','reports/part2/ollama_error_analysis_20261028/analysis.json','300-claim error and annotation-coverage breakdown'),
      ('E7','reports/part2/ollama_error_analysis_20261028/oracle_summary.json','Separate 20-claim gold-evidence intervention'),
      ('E8','reports/part2/step2_final_20261023/heldout100/summary.json','Matched symbolic routing, calls and online timing'),
      ('E9','reports/part2/ollama_comparison_20261028/telemundo_response.json','Separate saved Telemundo LLM diagnostic'),
      ('E10','reports/part2/llm_configurations_20261028/final_decisions.jsonl','Raw LLM versus final hybrid decisions'),
      ('E11','reports/part2/llm_configurations_20261028/protocol.json','Sampling, exclusion, prompts, model identity and fixed rules'),
      ('E12','reports/part2/llm_configurations_20261028/inputs.jsonl','Saved neural labels and original evidence used to derive the matched 120-claim neural strict score')]
    manifest=[dict(id=k,path=p,sha256=sha(ROOT/p),description=d) for k,p,d in source_specs]
    papers=[dict(id='L1',title='FEVER: a Large-scale Dataset for Fact Extraction and VERification',url='https://aclanthology.org/N18-1074/',use='Dataset/task reference; not a matched numerical row here.'),
      dict(id='L2',title='GEAR: Graph-based Evidence Aggregating and Reasoning for Fact Verification',url='https://aclanthology.org/P19-1085/',use='Evidence graph method; its development reference below is sourced from KGAT Table 2.'),
      dict(id='L3',title='Fine-grained Fact Verification with Kernel Graph Attention Network',url='https://aclanthology.org/2020.acl-main.655.pdf',use='Table 2, Dev LA and FEVER columns, printed page 7347 (PDF page 6). Verified October 28, 2026.')]
    OUT.mkdir(parents=True,exist_ok=False)
    scope='''# Adopted narrower research scope

Status: **Adopted on October 28, 2026 for this analysis.** This supersedes the October 23 discussion draft for the current analysis; it does not rewrite the original proposal or earlier reports.

## Title

'''+TITLE+'\n\n## Main research question\n\n'+QUESTION+'''

## Scope and timing of the revision

The study is restricted to three-label FEVER verification with the existing Wikipedia snapshot, frozen RoBERTa models, and the measured local Llama configurations. It is an empirical evaluation of evidence inputs and decision rules, not a new graph architecture or a general evaluation of every LLM.

The revised questions and working hypotheses below were written after inspecting results. They are retrospective research framing, not original or preregistered predictions. Earlier protocols remain as recorded; in particular, the latest five-configuration selection was frozen before its disjoint assessment. That within-experiment ordering does not make this later research reframing prospective.

## What happens to the original questions?

| Earlier scope | Decision in this adopted scope | What may be claimed |
|---|---|---|
| Original RQ1: whether FEVER represents LLM-generated factual-error types | Move to future work, explicitly unanswered | Generated-claim preparation and automated pilots exist; independent human validation is incomplete. Renaming the question does not complete that work. |
| Original local-judge question | Narrow to the measured local model, evidence and configuration conditions | Discuss model-specific classification/error patterns; do not generalize to every LLM or claim a completed human study. |
| Earlier neural/symbolic comparison and efficiency questions | Analyze frozen fusion and the implemented symbolic route as separate comparisons | Report measured gains and the zero-coverage negative result. No claim that symbolic methods generally fail. |
| Earlier corpus-size, generated-claim and graph-system ambitions | Preserve their records; exclude broad unsupported claims from the central contribution | Do not claim graph-model reproduction, a full blind benchmark, corpus-scale generality, or verified generated-claim truth labels. |

The new numbering always uses **Revised RQ1–RQ4**. It must not be read as a claim that all original questions were completed.
'''
    rq_rows=[
      ['Revised RQ1','How do evidence relevance and input length affect verification?','H1a: claim-related evidence outperforms deliberately mismatched evidence. H1b: doubling the evidence-model input limit produces a reliable accuracy gain.'],
      ['Revised RQ2','Does frozen RoBERTa fusion outperform its individual components on the official development set?','H2: validation-selected fusion improves label accuracy and macro-F1 over each individual model.'],
      ['Revised RQ3','Can a selected local LLM pipeline configuration improve over its original setup on disjoint assessment claims?','H3: the calibration-selected configuration improves accuracy and macro-F1 on the separate assessment sample.'],
      ['Revised RQ4','Does the implemented symbolic-first route avoid model calls and reduce latency?','H4: symbolic routing avoids expensive fallback calls and reduces mean inference-pipeline time without lowering label accuracy.']]
    scope+='\n'+md_table(['Question','Narrowed question','Retrospective working hypothesis'],rq_rows)+'\n'
    (OUT/'ADOPTED_SCOPE.md').write_text(scope)
    doc=['# Result Analysis\n\n'+TITLE+'\n\nPraise Ben | COSC 874: Deep Learning II\n']
    pages=[]
    def section(title,paragraphs=(),headers=None,data=None,open_section=False):
        doc.append('## '+title+'\n')
        for p in paragraphs:doc.append(p+'\n')
        if headers:doc.append(md_table(headers,data)+'\n')
        content=''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs)
        if headers:content+='<div class="scroll">'+html_table(headers,data)+'</div>'
        if open_section:pages.append('<section class="card"><h2>'+html.escape(title)+'</h2>'+content+'</section>')
        else:pages.append('<details class="card"><summary>'+html.escape(title)+'</summary>'+content+'</details>')
    section('Adopted question and scope',[
      QUESTION,
      'This narrower scope is explicitly adopted after reviewing results. The revised questions and working hypotheses are retrospective, not preregistered. Original proposals, numerical results and earlier reports are preserved.',
      'The original question about representativeness of LLM-generated factual errors remains unanswered and moves to future work because independent human review is incomplete. The following Revised RQ numbering does not imply completion of that original question.'],open_section=True)
    section('Revised questions and working hypotheses',headers=['ID','Question','Working hypothesis'],data=rq_rows)
    verdicts=[
      ['H1a · evidence relevance','Supported within the measured diagnostic','Same 100 claims/checkpoint: own top-5 evidence 55% accuracy / 0.4163 F1; another claim’s evidence 37% / 0.2897. Accuracy difference +18 points; paired 95% interval +8 to +28.','Evidence selection matters in this controlled development slice. It does not prove that any retrieved text is better than no evidence.','E2'],
      ['H1b · longer input','No convincing support','Same 19,998 claims: combined accuracy 61.696% → 61.706%; 14 errors fixed, 12 introduced. Evidence-only: 7 fixed, 6 introduced.','Only two additional correct combined labels. The interval includes no gain, and Telemundo remained wrong.','E3'],
      ['H2 · frozen fusion','Supported on this benchmark','19,998 claims: fusion 61.70% / 0.6078 F1, claim-only 58.78% / 0.5775, evidence-only 52.37% / 0.4175. Fusion-minus-claim accuracy +2.92 points; interval +2.45 to +3.37.','Measured improvement for these frozen outputs and this development population. It does not establish new-data or retraining-seed generality.','E1'],
      ['H3 · selected LLM pipeline','Partially supported; observed gain remains uncertain','Select on 60 calibration claims; assess on 120 disjoint claims. Original 51.67% / 0.4973 F1 → selected hybrid 59.17% / 0.5821. 25 errors fixed, 16 introduced; accuracy interval −2.50 to +18.33 points.','Promising direction, not a reliable general improvement yet. The selected system is a hybrid, and NEI recall fell.','E5, E5a–E5d'],
      ['H4 · symbolic efficiency','Not supported by the implemented route','Matched 100-claim online run: zero resolved claims, zero avoided model calls; 100 model calls in both routes. Mean 106.57 ms with routing versus 66.35 ms without.','The implemented rules added work without resolving claims. This is a bounded negative result, not a verdict on all symbolic systems.','E8']]
    quick=[
      ['Relevant versus mismatched evidence','Supported in the diagnostic','100 claims: 55% versus 37% accuracy.'],
      ['Longer input alone','No convincing support','19,998 claims: only two additional correct fusion labels.'],
      ['RoBERTa fusion','Supported on this benchmark','19,998 claims: 58.78% claim-only → 61.70% fusion.'],
      ['Selected LLM pipeline','Partially supported; uncertain','120 assessment claims: 51.67% → 59.17%; interval includes no gain.'],
      ['Symbolic speedup','Not supported','100 online claims: zero calls avoided; 106.57 versus 66.35 ms/claim.']]
    section('At a glance: what the evidence supports',headers=['Tested idea','Verdict','Key evidence'],data=quick,open_section=True)
    section('Hypothesis verdicts with evidence',[
      'Verdicts are limited to the named measured conditions. A retrospective supported finding is not a preregistered confirmation. Macro-F1 below means three-class classification F1, not evidence-retrieval F1.'],
      ['Hypothesis','Verdict','Measured evidence','Interpretation and boundary','Source'],verdicts)
    populations=[
      ['P1','19,998 official FEVER development claims','6,666 per class','Main frozen-model comparison and 256/512 input comparison.','Do not compare directly with a small-sample percentage.'],
      ['P2','100 internal-validation claims','Original slice; not an untouched test','Evidence relevance/depth ablations with the same checkpoint.','Development diagnostic includes the earlier tuning pool.'],
      ['P3','300 official-development claims','100 per class','Original Llama 3.1 8B pilot versus neural references.','One local-model run; Telemundo kept separate.'],
      ['P4','60 internal-validation calibration claims','20 per class','Select among five predeclared configurations.','Selection data; not an unbiased final score.'],
      ['P5','120 new LLM assessment claims','40 per class','Original LLM versus the frozen selected pipeline.','Disjoint from P4 and the earlier P3 texts; still official development, not a blind test.'],
      ['P6','20 selected cases from P3','Verifiable cases lacking a complete original annotated group','Artificial replacement with a complete gold-evidence group.','Oracle diagnostic, not production accuracy; not an independent full sample.'],
      ['P7','100 internal held-out claims','52 SUPPORTS / 26 REFUTES / 22 NEI','Symbolic comparison and online pipeline timing.','Separate population/device conditions; never pool with P1–P6.']]
    section('Populations: which scores may be compared?',headers=['Population','Size and source','Class composition','Use','Comparison limit'],data=populations)
    main_metrics=[]
    for name,title in [('claim_only_H2','Claim-only H2'),('retrieved_roberta','Evidence RoBERTa'),('calibrated_ensemble','Frozen fusion — primary local result')]:
        s=official['conditions'][name];main_metrics.append([title,pct(s['label_accuracy']),f1(s['classification_macro_f1']),pct(s['fever_score']),'Measured locally; P1'])
    main_metrics += [['GEAR development reference','74.84%','Not reported here','70.69%','Published in KGAT Table 2; not rerun'],['KGAT RoBERTa-large development reference','78.29%','Not reported here','76.11%','Published in KGAT Table 2; not rerun']]
    section('Main benchmark and literature context',[
      'P1 uses all 19,998 official development claims and the unchanged official scorer. Label accuracy checks the class; strict FEVER additionally requires a complete annotated evidence group among the first five submitted sentences for SUPPORTS/REFUTES. Correct NEI answers do not require evidence.',
      'The published rows use the development label-accuracy and FEVER columns of KGAT Table 2, not its test scores. They are numerical benchmark references, not equal-resource reproductions. Architectures, retrieval, training data/budgets and model sizes differ.',
      'GEAR and KGAT learn evidence-graph aggregation. This project uses lexical retrieval, concatenated evidence and classifier-score fusion; it did not reproduce those graph architectures. The local result remains below the published development references.',
      'Sources: FEVER — https://aclanthology.org/N18-1074/ ; GEAR — https://aclanthology.org/P19-1085/ ; KGAT Table 2, PDF page 6 — https://aclanthology.org/2020.acl-main.655.pdf'],
      ['System','Label accuracy','Classification macro-F1','Strict FEVER','Origin'],main_metrics)
    pc=[]
    for model,title in [('claim_only_H2','Claim-only'),('retrieved_roberta','Evidence'),('calibrated_ensemble','Fusion')]:
        values=per_class(official['conditions'][model]['confusion_matrix'])
        for label,v in values.items():pc.append([title,label,pct(v['precision']),pct(v['recall']),f1(v['f1'])])
    section('Per-class performance on all 19,998 claims',[
      'The development set is balanced, but performance is not. Evidence RoBERTa has zero NEI recall. Fusion restores some NEI predictions, yet NEI remains its weakest class. Its 61.70% accuracy therefore does not mean equally strong verification across labels.',
      'Claim-only supplies no evidence; its strict FEVER score is earned only by correct NEI labels. An accurate label alone does not demonstrate a complete evidence-backed verification.'],
      ['System','Class','Precision','Recall','F1'],pc)
    matrix=official['conditions']['calibrated_ensemble']['confusion_matrix']
    section('Fusion confusion matrix: actual rows, predicted columns',[
      'All rows contain 6,666 claims. For example, 1,773 actual REFUTES claims were predicted SUPPORTS. The model is strongest on SUPPORTS and frequently misses NEI.'],
      ['Actual label','Predicted SUPPORTS','Predicted REFUTES','Predicted NEI'],[[label,*r] for label,r in zip(LABELS,matrix)])
    depth={r['condition']:r for r in ablation['conditions']}
    evidence_rows=[]
    for name,title in [('retrieved_k1','Top-1'),('retrieved_k5','Top-5'),('retrieved_k10','Top-10'),('empty_evidence','Empty evidence'),('wrong_claim_evidence','Other-claim evidence'),('gold_evidence_oracle','Gold-evidence diagnostic')]:
        s=depth[name];evidence_rows.append([title,pct(s['accuracy']),f1(s['macro_f1'])])
    section('Evidence ablations: the same 100 development claims',[
      'Own retrieved evidence beats deliberately mismatched evidence, but more sentences do not consistently help. Top-10 ties top-5 accuracy and has slightly lower macro-F1. Empty evidence has lower accuracy than top-5 but higher macro-F1, so the result depends on the metric.',
      'The 84% gold-input result is not an attainable end-to-end result: NEI receives empty gold evidence, creating a shortcut opportunity. These runs use a development slice and cannot be substituted for the main benchmark.'],
      ['Evidence input','Accuracy','Macro-F1'],evidence_rows)
    budget=[]
    for model,title in [('retrieved_roberta','Evidence RoBERTa'),('calibrated_ensemble','Fusion')]:
        a=token['conditions'][model+'_256'];b=token['conditions'][model+'_512'];c=token['comparisons'][model]
        budget.append([title,pct(a['accuracy']),pct(b['accuracy']),str(c['fixed_errors']),str(c['newly_wrong']),f"{100*c['accuracy_delta']:+.3f} points"])
    section('Token-budget control: the same 19,998 claims',[
      'The claim-only branch stayed fixed. Only the evidence branch changed from 256 to 512 tokens; model weights, retrieved references and fusion parameters stayed unchanged. At 256, 1,738 inputs were truncated; only four exceeded 512.',
      'Seeing more of the same retrieved text produced almost no overall label gain. Telemundo still predicted SUPPORTS after all 347 input tokens fit. This does not show that context limits never matter; it shows little benefit from this particular setting change.'],
      ['System','256-token accuracy','512-token accuracy','Errors fixed','New errors','Net change'],budget)
    pilot_rows=[]
    for name,title in [('evidence_512','Evidence RoBERTa'),('combined_512','Combined RoBERTa'),('local_llm','Llama 3.1 8B')]:
        s=pilot['conditions'][name];pilot_rows.append([title,pct(s['accuracy']),f1(s['macro_f1']),pct(s['fever_score'])])
    section('Local LLM pilot: the same 300 claims',[
      'Llama used the same five evidence texts and a fixed evidence-constrained prompt. It scored close to evidence-only RoBERTa, but below the combined system in label accuracy. Comparing a single LLM with a two-model fusion is a system comparison, not an isolated architecture or size experiment.',
      'Llama predicted NEI 163 times, versus 72 for fusion. It correctly identified 66/100 NEI cases versus 42/100, but only 61/100 SUPPORTS and 28/100 REFUTES versus 84/100 and 55/100. The net result was 26 fewer correct labels.',
      'Strict FEVER moved differently: Llama 46.33% versus fusion 44.00%. Its paired strict-score interval was −3.67 to +8.33 points, so the pilot does not establish a reliable strict-score gain.'],
      ['System','Accuracy','Macro-F1','Strict FEVER'],pilot_rows)
    cfg_names={'A_original':'Original setup','B_prompt':'Comparison prompt only','C_retrieval':'Entity/BM25 retrieval only','D_both':'New prompt + retrieval','E_claim_fallback':'Both + fixed NEI fallback — selected'}
    cal_rows=[[cfg_names[name],pct(s['accuracy']),f1(s['macro_f1']),pct(s['fever_score'])] for name,s in cal.items()]
    section('Configuration selection: 60 calibration claims only',[
      'All four unique LLM prompt/retrieval conditions used the same 60 claims. A fifth configuration reused the fourth condition’s raw LLM outputs with a fixed fallback: if Llama answered NEI, accept claim-only SUPPORTS/REFUTES only when its softmax was at least 0.80. That softmax is not verified factual confidence.',
      'The highest calibration macro-F1 selected the hybrid. Selection and its hash were written before assessment. The settings were not changed in response to assessment labels.'],
      ['Configuration','Calibration accuracy','Calibration macro-F1','Calibration strict FEVER'],cal_rows)
    assess_rows=[[cfg_names[name],pct(s['accuracy']),f1(s['macro_f1']),pct(s['fever_score'])] for name,s in config['conditions'].items()]
    assess_rows.append(['Existing combined RoBERTa reference',pct(config['neural_reference']['neural_combined_512']['accuracy']),f1(config['neural_reference']['neural_combined_512']['macro_f1']),pct(neural_strict)])
    section('Configuration assessment: 120 disjoint claims',[
      'This sample has 40 claims per class and excludes the earlier 300-claim LLM pilot and calibration texts. The selected hybrid gains nine correct labels: 25 errors fixed and 16 introduced. The accuracy-difference interval is −2.50 to +18.33 percentage points; macro-F1 difference interval −0.0107 to +0.1809.',
      'Strict FEVER improves by 4.17 points, with interval −5.83 to +14.17. These intervals include no gain. The result is promising but inconclusive, and it remains below the saved combined RoBERTa reference in label accuracy and macro-F1 on these same claims. Its strict FEVER score is higher: 47.50% versus the recomputed neural reference of 41.67%. These are different metrics, not a universal winner.',
      'The raw new-prompt/retrieval LLM component scored 57.50%. The fixed fallback changed four labels, fixing three and introducing one, for a final 59.17%. Raw LLM and final hybrid outputs remain separately recorded.'],
      ['Configuration','Assessment accuracy','Assessment macro-F1','Assessment strict FEVER'],assess_rows)
    trade=[]
    for name,title in [('A_original','Original LLM'),('E_claim_fallback','Selected hybrid')]:
        counts=per_class(config['conditions'][name]['confusion_matrix'])
        for label,v in counts.items():trade.append([title,label,str(v['TP']),str(v['FP']),str(v['FN']),str(v['TN']),pct(v['recall'])])
    section('Assessment trade-offs: TP, FP, FN and TN',[
      'These are one-versus-rest counts for each label; a TN can still confuse the other two labels. REFUTES recall rose from 22.5% to 75%, while NEI recall fell from 65% to 37.5%. The selected system improved average performance by trading some correct abstentions for more REFUTES decisions, including more false REFUTES decisions.'],
      ['System','Positive label','TP','FP','FN','TN','Recall'],trade)
    coverage_rows=[]
    for bucket,name in [('complete_annotated_group','Complete annotated group'),('no_complete_annotated_group','No complete annotated group'),('NEI','NEI; no required group')]:
        r=error['by_annotation_coverage'][bucket];coverage_rows.append([name,str(r['n']),str(r['llm_correct']),str(r['combined_correct']),str(r['llm_NEI'])])
    section('Evidence-coverage subsets and a separate oracle check',[
      'In the 300-claim pilot, Llama correctly labeled 73/121 verifiable claims with a complete annotated group and 16/79 without one. This is an association: coverage, difficulty and other factors differ. Missing an annotation does not prove that all alternative text is insufficient.',
      'A random 20-case diagnostic from the incomplete-annotation subgroup changed evidence to one complete gold group while keeping claim, model, prompt and decoding fixed. Accuracy changed from 7/20 to 13/20, with six errors fixed and none introduced. This is a gold-evidence intervention on a selected subgroup, not a new production score. It changes relevance and evidence amount together.'],
      ['Original evidence status','Claims','LLM correct','Fusion correct','LLM NEI answers'],coverage_rows)
    initial=[r for r in symbolic['trials'] if r['pass']=='initial_online']
    timing=[[r['condition'],pct(r['accuracy']),f1(r['macro_f1']),str(r['resolved']),str(r['model_calls']),f"{1000*r['mean_seconds']:.2f}"] for r in initial]
    section('Symbolic route: a negative efficiency finding',[
      'In the matched 100-claim online run, symbolic v2 resolved no claims. Both conditions called retrieval and RoBERTa 100 times and produced identical labels. The initial symbolic route added about 40.22 ms per claim on average.',
      'Later cached timing passes fluctuated without avoiding calls. Those fluctuations do not establish symbolic savings. The result concerns these restricted rules, inputs, device and network/cache conditions, not every neuro-symbolic architecture.'],
      ['Condition','Accuracy','Macro-F1','Claims resolved','Model calls','Mean ms/claim'],timing)
    cases=build_cases()
    case_rows=[[c['id'],c['claim'],c['observed'],c['evidence'],c['interpretation'],c['source']] for c in cases]
    section('Qualitative examples: fixes, failures and explanation limits',[
      'These examples were selected to illustrate patterns and are not a random prevalence estimate. The saved request/response excerpts are bundled in evidence_cases.json. Generated explanations are not hidden reasoning traces. A correct final label does not certify its explanation.'],
      ['Case','Claim','Observed result','Specific evidence or output','What it shows','Trace source'],case_rows)
    limits=[
      ['Retrospective revision','Questions/hypotheses were revised after results.','Treat this as transparent exploratory analysis, not preregistered confirmation.'],
      ['Development reuse and sample sizes','P1 is official development; newer LLM assessment is a disjoint subset, not a blind official test. Other samples range from 20 to 300.','Keep denominators separate. Do not present calibration or oracle scores as final benchmark performance.'],
      ['Single trained checkpoints / bounded repeats','No independent additional Transformer training seeds; current 8B configurations use one generation per claim.','Bootstrap intervals measure claim-sampling uncertainty, not between-training-run or repeated-generation variance.'],
      ['Model identity and resources','Llama 3.1 8B Q4_K_M was held fixed; published graph models use different training/retrieval resources.','No model-only or quantization-only cause is established. Pretraining exposure to FEVER/Wikipedia is unknown.'],
      ['Retrieval and annotation coverage','Some annotated groups are missed or unavailable locally; complete annotation coverage does not guarantee correct inference.','Do not equate overlap with semantic sufficiency, or oracle access with a deployable retriever.'],
      ['Input/training mismatch','The evidence checkpoint was trained with gold-style evidence; empty NEI inputs create a shortcut opportunity. It predicts no NEI on the full retrieved-evidence benchmark.','A controlled retraining experiment is needed to isolate the cause; no retraining fix is claimed.'],
      ['Bundled changes and class trade-offs','Entity linking and BM25 change together; prompt rules and examples change together; the selected hybrid has lower NEI recall.','Do not attribute the entire gain to one clause, retrieval subcomponent or universally better reasoning.'],
      ['Explanation validity','Correct labels can accompany irrelevant or illogical explanations, as in the Halsey case.','No faithfulness or human-rated explanation-quality claim is made.'],
      ['Symbolic timing limits','Zero coverage, only a few timing passes, and cache/network variation.','No avoided-call, cost-saving or general speedup claim is supported.'],
      ['Original generated-error question','Independent review/adjudication of generated claims remains incomplete.','Original RQ1 remains unanswered and is explicitly outside the adopted central scope.']]
    section('Limitations and boundaries',headers=['Limitation','Evidence or condition','Consequence for the claim'],data=limits)
    section('Conclusion and current configuration choice',[
      'For the measured main benchmark, frozen RoBERTa fusion remains the primary local result: 61.70% accuracy, 0.6078 macro-F1 and 45.33% strict FEVER on all 19,998 development claims. It improves on its individual components, while remaining below the published development references.',
      'Evidence relevance matters in the paired diagnostics; simply increasing context length gave almost no gain. The selected LLM hybrid improved the point estimates on a disjoint 120-claim assessment, but the intervals and NEI trade-off prevent a strong general improvement claim. It remains an experimental candidate, not a new default.',
      'The implemented symbolic stage did not establish an efficiency benefit. The contribution is an evidence-linked empirical account of what helped, what did not, and where verification still fails. Original generated-claim representativeness is future work, not a completed finding.'],open_section=True)
    section('Pending work, separated from completed analysis',headers=['Item','Status','What would resolve it'],data=[
      ['Tabular Result Analysis and adopted narrower scope','Completed in this artifact','No new model training was needed.'],
      ['General LLM model/quantization comparison','Not performed in the latest configuration search','Hold pipeline, claims and scoring fixed; compare model variants on separate data.'],
      ['Reliable improvement for the selected LLM hybrid','Not established by this small assessment','Use a larger new evaluation sample and planned repeats; avoid further tuning on these 120 claims.'],
      ['Generated-claim truth/error representativeness','Unanswered; moved to future work','Independent blinded human review and adjudication.'],
      ['Training-distribution fix and multi-seed training variance','Not run','Requires a separate training study and additional compute; none was run.']])
    section('Evidence index',[
      'Each source ID below maps to a repository artifact and its SHA-256 in evidence_manifest.json. The submission bundle contains compact source snapshots, all displayed CSV tables, selected case excerpts and the adopted-scope document. Raw full-run outputs and checkpoints remain preserved in the repository.'],
      ['ID','Artifact','What it supports'],[[m['id'],m['path'],m['description']] for m in manifest])
    (OUT/'derived_assessment_neural_reference.json').write_text(json.dumps(dict(n=120,condition='neural_combined_512',accuracy=neural_accuracy,fever_score=neural_strict,evidence='Original saved top-five references; not the entity-retrieval variant',source='E12'),indent=2))
    (OUT/'RESULT_ANALYSIS.md').write_text('\n'.join(doc))
    (OUT/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2));(OUT/'literature_sources.json').write_text(json.dumps(papers,indent=2));(OUT/'evidence_cases.json').write_text(json.dumps(cases,indent=2))
    export_csv('hypothesis_evidence.csv',['Hypothesis','Verdict','Measured evidence','Interpretation and boundary','Source'],verdicts)
    export_csv('main_benchmark.csv',['System','Accuracy','Macro-F1','Strict FEVER','Origin'],main_metrics)
    export_csv('official_per_class.csv',['System','Class','Precision','Recall','F1'],pc)
    export_csv('assessment_per_class.csv',['System','Positive label','TP','FP','FN','TN','Recall'],trade)
    export_csv('populations.csv',['Population','Size and source','Class composition','Use','Comparison limit'],populations)
    css='''body{font:16px/1.65 system-ui;max-width:1250px;margin:30px auto;padding:0 22px;color:#183047;background:#f4f7fa}h1{font-size:34px;line-height:1.2;letter-spacing:-1px}h2{font-size:22px;line-height:1.4}a{color:#126f73}.card{background:white;border:1px solid #dbe5ed;border-radius:14px;padding:22px;margin:20px 0}summary{cursor:pointer;font-weight:700;font-size:21px;line-height:1.4}p{max-width:100ch}.scroll{overflow:auto;margin-top:16px}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.6}td,th{padding:12px 10px;text-align:left;border-bottom:1px solid #e0e8ee;vertical-align:top}th{background:#eff5f8;color:#38586b}td{min-width:90px}table tr:nth-child(even){background:#f9fbfc}.actions{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}.actions a,button{font:inherit;border:1px solid #b8cbd5;border-radius:8px;background:white;color:#126f73;padding:9px 13px;cursor:pointer}.note{background:#e9f4f1;border-left:4px solid #159087;padding:14px 18px;border-radius:7px}small{color:#536d7b}@media(max-width:750px){body{padding:0 14px}.card{padding:16px}h1{font-size:29px}td,th{padding:9px;min-width:85px}}@media print{body{max-width:none;background:white;color:black;font:10pt/1.4 Arial;padding:0}.actions,.nav{display:none}.card{border:0;padding:0;margin:12pt 0}details:not([open])>*:not(summary){display:block!important}summary{list-style:none;font-size:14pt}.scroll{overflow:visible}table{font-size:8pt;table-layout:auto}td,th{min-width:0;padding:5pt;overflow-wrap:anywhere}tr{break-inside:avoid}h1,h2,summary{break-after:avoid}a{color:black}thead{display:table-header-group}}'''
    refs='<section class="card"><h2>Primary literature sources</h2><ul>'+''.join('<li><a href="'+html.escape(p['url'],quote=True)+'">'+html.escape(p['title'])+'</a> — '+html.escape(p['use'])+'</li>' for p in papers)+'</ul></section>'
    page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>COSC 874 · Result Analysis</title><style>'+css+'</style></head><body><p class="nav"><a href="/">← Evidence explorer</a> · <a href="/llm-configurations">Configuration results</a></p><h1>Result Analysis</h1><p>'+html.escape(TITLE)+'</p><p class="note"><b>Scope explicitly adopted.</b> Revised after results were reviewed; not preregistered. Negative and inconclusive findings are retained. Original reports and measurements are preserved.</p><div class="actions"><a href="/result-analysis.md">Download written analysis</a><a href="/result-analysis.zip">Download analysis + evidence tables</a><button id="print">Print / Save PDF</button><button id="expand">Expand all evidence tables</button></div>'+''.join(pages)+refs+'<footer><small>FEVER supplied the claims, labels and evidence annotations. Analysis and viewer prepared with AI assistance. Generated explanations are not human reviews or faithful internal reasoning traces. Local page, accessible on this computer.</small></footer><script>document.getElementById("print").onclick=()=>{document.querySelectorAll("details").forEach(d=>d.open=true);window.print();};document.getElementById("expand").onclick=()=>document.querySelectorAll("details").forEach(d=>d.open=true);</script></body></html>'
    (OUT/'index.html').write_text(page)
    sources=OUT/'source_snapshots';sources.mkdir()
    for m in manifest:
        p=ROOT/m['path']
        if p.suffix=='.json':shutil.copy2(p,sources/(m['id']+'_'+p.name))
    shutil.copy2(Path(__file__),OUT/'analysis_builder.py')
    (OUT/'README.md').write_text('Result Analysis submission bundle\n\nStart with RESULT_ANALYSIS.md. ADOPTED_SCOPE.md records the explicitly revised scope and the unanswered original question. CSVs hold the displayed evidence tables; evidence_cases.json preserves selected case excerpts; evidence_manifest.json records repository paths and SHA-256 hashes; source_snapshots/ contains compact source JSON artifacts. The analysis is retrospective and exploratory. Keep study populations separate. The HTML page is also available in the repository/local viewer; no new model calls or training were performed by this builder.\n')
    zip_path=OUT/'Result_Analysis_Evidence.zip'
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p!=zip_path:archive.write(p,p.relative_to(OUT))
    print('Wrote adopted scope, tabular analysis, CSV tables, evidence snapshots and bundle:',OUT)


def md_table(headers,values):
    def cell(value):return str(value).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(cell(v) for v in row)+' |' for row in values])


def html_table(headers,values):
    esc=lambda v:html.escape(str(v))
    return '<table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in values)+'</tbody></table>'


def export_csv(name,headers,values):
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(headers);writer.writerows(values)


def build_cases():
    old_inputs={r['id']:r for r in rows('reports/part2/ollama_comparison_20261028/sample.jsonl')}
    old_responses={r['id']:r for r in rows('reports/part2/ollama_comparison_20261028/responses.jsonl')}
    config_inputs={r['id']:r for r in rows('reports/part2/llm_configurations_20261028/inputs.jsonl')}
    decisions=rows('reports/part2/llm_configurations_20261028/final_decisions.jsonl')
    tele_input=read('reports/part2/ollama_comparison_20261028/telemundo_input.json');tele=read('reports/part2/ollama_comparison_20261028/telemundo_response.json')
    cases=[dict(id='Q1 / #111897',claim=tele_input['claim'],observed='Gold REFUTES; evidence RoBERTa SUPPORTS; local LLM REFUTES.',
        evidence='Telemundo is described as Spanish-language. The annotated sentence reached the 256-token RoBERTa input completely; 512 tokens still did not fix that prediction.',
        interpretation='Relevant evidence can be retrieved and visible while the classifier still gets the label wrong.',source='E3, E9; /ollama-traces#111897',
        raw_llm_reason=tele['reason'],sentences=tele_input['sentences'])]
    for ident,comment in [(146231,'The original candidate search missed Juventus; the supplied texts about black/white topics do not establish or contradict the uniform claim.'),(149173,'The 2006 debut sentence is present; accepting it as close enough to a 2004 claim is an explicit date-comparison error.')]:
        r=old_inputs[ident];p=old_responses[ident]
        cases.append(dict(id='Q'+str(len(cases)+1)+' / #'+str(ident),claim=r['claim'],observed='Gold '+r['gold']['label']+'; LLM '+p['label']+'.',evidence=p['reason'],interpretation=comment,source='E4, E6; /ollama-traces#'+str(ident),sentences=r['sentences']))
    notes={75152:'A useful fix: selected evidence explicitly says the band toured worldwide extensively, contradicting the negated claim.',
           52731:'A regression: a birth date does not contradict a birthplace. The final REFUTES label is wrong against FEVER’s NEI reference.',
           121517:'Correct label, poor justification: the racehorse text does not establish a contradiction about Halsey’s EP. Accuracy credit does not validate the explanation.'}
    for ident,note in notes.items():
        r=config_inputs[ident];a=next(d for d in decisions if d['id']==ident and d['phase']=='assessment' and d['config']=='A_original');b=next(d for d in decisions if d['id']==ident and d['phase']=='assessment' and d['config']=='E_claim_fallback')
        cases.append(dict(id='Q'+str(len(cases)+1)+' / #'+str(ident),claim=r['claim'],observed='Gold '+r['gold']['label']+'; original LLM '+a['final_label']+' → selected pipeline '+b['final_label']+'.',
            evidence=b['raw_llm_reason'],interpretation=note,source='E10; /llm-configurations',claim_override=b['claim_override'],sentences=r['evidence']['entity']))
    return cases


if __name__=='__main__':main()
