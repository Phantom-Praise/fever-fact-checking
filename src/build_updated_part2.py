"""Assemble the measured Part 2 results without rerunning or replacing experiments.

Document-only command: use the bundled artifact Python (ReportLab installed).
Research execution remains in the repository virtual environment.
"""
import csv
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from statistics import mean, stdev
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'reports/part2/submission_20261023_final'
STEP2 = ROOT / 'reports/part2/step2_final_20261023/report'
NEW = ROOT / 'reports/part2/literature_improvements_20261023_corrected'
OUT = ROOT / 'reports/part2/submission_updated_20261023_complete'
PDF = ROOT / 'output/pdf/Model_Execution_Part_2_Updated_20261023.pdf'


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def main():
    global OUT, PDF
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', help='Rebuild only this generated submission; preserve experiment inputs.')
    parser.add_argument('--include-official', action='store_true', help='Create a separate submission including the complete official FEVER development evaluation.')
    args = parser.parse_args()
    refresh = args.refresh
    official_dir = ROOT / 'reports/part2/official_fever_dev_20261023'
    official = None
    if args.include_official:
        OUT = ROOT / 'reports/part2/submission_official_20261023'
        PDF = ROOT / 'output/pdf/Model_Execution_Part_2_Official_20261023.pdf'
        audit = json.loads((official_dir / 'report_manifest.json').read_text())
        assert audit['status'] == 'verified_complete' and audit['n'] == 19998
        for name, expected in audit['output_sha256'].items():
            assert sha(official_dir / name) == expected, name
        official = json.loads((official_dir / 'summary.json').read_text())
    if refresh:
        previous = json.loads((OUT / 'manifest.json').read_text())
        if previous['pdf'] != str(PDF):
            raise ValueError('Output is not the expected generated submission')
    OUT.mkdir(parents=True, exist_ok=refresh)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    if PDF.exists() and not refresh:
        raise FileExistsError(PDF)
    training = rows(OLD / 'all_training_runs.csv')
    classifiers = rows(OLD / 'all_classifier_evaluations.csv')
    heldout = rows(STEP2 / 'heldout_conditions.csv')
    judges = rows(OLD / 'all_judge_runs.csv')
    summary = json.loads((NEW / 'summary.json').read_text())
    assert summary['assessment_n'] == 200
    assert json.loads((NEW / 'inference_integration_check.json').read_text())['matches_saved_predictions']
    all_classifiers = list(classifiers)
    all_classifiers += [dict(r, scope='frozen_test100', source=str((STEP2 / 'heldout_conditions.csv').relative_to(ROOT))) for r in heldout]
    all_classifiers += [dict(v, condition=k, scope='new_validation_assessment200', n=200, source=str((NEW / 'summary.json').relative_to(ROOT))) for k,v in summary['conditions'].items()]
    if official:
        for name, value in official['conditions'].items():
            matrix = value['confusion_matrix']
            precision = [matrix[i][i] / max(1, sum(row[i] for row in matrix)) for i in range(3)]
            recall = [matrix[i][i] / sum(matrix[i]) for i in range(3)]
            all_classifiers.append({'scope':'official_fever_development19998','condition':name,'n':value['n'],
                'accuracy':value['label_accuracy'],'macro_precision':mean(precision),'macro_recall':mean(recall),
                'macro_f1':value['classification_macro_f1'],'fever_score':value['fever_score'],
                'evidence_precision':value['evidence_macro_precision'],'evidence_recall':value['evidence_macro_recall'],
                'evidence_f1':value['evidence_f1'],'source':str((official_dir / 'summary.json').relative_to(ROOT))})
    columns = ['scope', 'condition', 'n', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'fever_score', 'evidence_precision', 'evidence_recall', 'evidence_f1', 'source']
    with (OUT / 'all_classifier_evaluations_updated.csv').open('w' if refresh else 'x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(all_classifiers)
    sources = [OLD / f for f in ('all_training_runs.csv', 'all_judge_runs.csv', 'all_latency_runs.csv', 'all_retrieval_runs.csv')]
    sources += [STEP2 / f for f in ('heldout_timing.csv', 'rq1_machine_distributions.csv', 'rq1_protocol_quality_audit.json')]
    sources += [NEW / f for f in ('summary.json', 'protocol.json', 'selected_config.json', 'calibration_candidates.json', 'training_input_audit.json', 'training_pilot_manifest.json', 'inference_integration_check.json', 'LITERATURE_TO_CODE.md')]
    for source in sources:
        shutil.copy2(source, OUT / source.name)
    if official:
        target = OUT / 'official_fever'
        target.mkdir(exist_ok=refresh)
        for source in sorted(official_dir.iterdir()):
            if source.is_file():
                shutil.copy2(source, target / source.name)
                sources.append(source)
    choices = [
        ['Unweighted baseline', 'Original epoch 1; reference', 'Retained for comparison; majority-class behavior.'],
        ['Claim-only RoBERTa', 'SELECTED: H2 epoch 2', 'LR 1e-5, WD .01, batch 8; best tuning macro-F1 .6202.'],
        ['Evidence RoBERTa', 'SELECTED: epoch 1', 'LR 2e-5, WD .01, batch 8; best oracle-validation macro-F1 .9335.'],
        ['Retrieval', 'SELECTED: title-aware top-5', 'Frozen paired reference; new rankers did not improve recall.'],
        ['Symbolic hybrid', 'V1/v2 reported; no winner', 'Zero resolved claims; no measured avoided calls.'],
        ['Neural ensemble', 'SELECTED REPORTING: w=.75; bias=2' if official else 'PROVISIONAL: w=.75; bias=2', 'Original top-5; selected on calibration before official evaluation.' if official else 'Best calibration macro-F1; original retrieval; no new test evaluation.'],
        ['LLM judge', 'Llama 3.2 local pilot', 'Availability-based comparator; not a performance-selected final judge.'],
    ]
    with (OUT / 'configuration_choices_updated.csv').open('w' if refresh else 'x', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['component','choice','reason']); writer.writerows(choices)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Title2', fontName='Helvetica-Bold', fontSize=21, leading=25, textColor=colors.HexColor('#17324d'), spaceAfter=8))
    styles.add(ParagraphStyle(name='Section2', fontName='Helvetica-Bold', fontSize=12, leading=15, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor('#17324d')))
    styles.add(ParagraphStyle(name='Body2', fontName='Helvetica', fontSize=9.2, leading=12.5, spaceAfter=7))
    styles.add(ParagraphStyle(name='Small2', fontName='Helvetica', fontSize=8, leading=10.5, spaceAfter=6))
    styles.add(ParagraphStyle(name='Cell2', fontName='Helvetica', fontSize=8.1, leading=10.5, alignment=TA_LEFT))
    story, markdown = [], []

    def title(text):
        story.append(Paragraph(escape(text), styles['Title2'])); markdown.append('# ' + text + '\n')

    def heading(text):
        story.append(Paragraph(escape(text), styles['Section2'])); markdown.append('## ' + text + '\n')

    def para(text, small=False):
        story.append(Paragraph(escape(text), styles['Small2' if small else 'Body2'])); markdown.append(text + '\n')

    def table(headers, values, widths):
        rendered = [[Paragraph(escape(str(cell)), styles['Cell2']) for cell in row] for row in [headers] + values]
        tab = Table(rendered, colWidths=widths, repeatRows=1, hAlign='LEFT')
        tab.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e5edf5')),
            ('VALIGN',(0,0),(-1,-1),'TOP'), ('LEFTPADDING',(0,0),(-1,-1),6),
            ('RIGHTPADDING',(0,0),(-1,-1),6), ('TOPPADDING',(0,0),(-1,-1),5),
            ('BOTTOMPADDING',(0,0),(-1,-1),5),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')]),
            ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#a7bacb')),
        ]))
        story.extend([tab, Spacer(1,7)])
        markdown.append('| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |')
        markdown.extend('| ' + ' | '.join(map(str,row)) + ' |' for row in values)
        markdown.append('')

    def next_page():
        story.append(PageBreak())

    def fmt(value): return f'{float(value):.4f}'
    def metric_rows(records, names):
        return [[names.get(r['condition'], r['condition'])] + [fmt(r[k]) for k in ('accuracy','macro_precision','macro_recall','macro_f1')] for r in records]

    title('Model Execution - Part 2')
    para('COSC 874 - Deep Learning II | Praise Ben')
    para('FEVER fact-checking: hyperparameter tuning, controlled comparisons, reproducibility, and selected results. Branch: experiment/part2-paired-ablation. This report updates the earlier submission snapshot with the held-out comparison and calibrated ensemble.')
    if official:
        para('LATEST BENCHMARK: frozen ensemble on all 19,998 official development claims: 61.70% label accuracy, 0.6078 macro-F1, 45.33% strict FEVER score. The earlier 71% result is a separate 200-claim development assessment.', small=True)
    heading('Progress against the assignment')
    table(['Step','Status','What was done / what remains'], [
        ['1. Tune','Completed','Four controlled configurations; H2 selected by validation macro-F1.'],
        ['2. Experiments','Measured; RQ1 partial','Paired ablations, judge pilot, symbolic v2, corpus growth, held-out comparison, and ensemble calibration. Human review remains pending.'],
        ['3. Reproducibility','Completed; optional repeats pending','Fixed seeds and saved artifacts. No extra Transformer training seeds; judge/inference repeats are distinct.'],
        ['4. Results','Updated','Configuration choices and measured results are consolidated here and in the CSV appendix.'],
    ], [65,105,334])
    heading('Step 1 - A small, reasoned tuning study')
    para('Part 1 showed majority-class collapse in the original baseline and evidence-model overfitting after epoch 1. The tuning study varied learning rate around 2e-5 and tested removing weight decay. Architecture, batch 8, length 256, seed 42, the same 20% training subset, two epochs, and the first 2,000 validation claims were held fixed.')
    tuned = [r for r in training if r['family']=='claim_only_tuned' and r['epoch']=='2']
    table(['Configuration','LR','Weight decay','Epoch','Accuracy','Macro-F1'], [[r['run'].replace('_seed42','') + (' - SELECTED' if r['run']=='H2_seed42' else ''),r['learning_rate'],r['weight_decay'],r['epoch'],fmt(r['accuracy']),fmt(r['macro_f1'])] for r in tuned], [123,69,79,49,92,92])
    para('H2 epoch 2 is selected because it has the highest macro-F1 (0.6202) and accuracy (0.6775) among H1-H4. The full appendix includes both epochs of each configuration, rather than only the best rows.')
    para('The evidence model retains epoch 1: gold-evidence validation accuracy/F1 = 0.9415/0.9335, versus 0.9365/0.9313 at epoch 2. These are oracle-input diagnostics. All NEI examples had empty gold evidence, so 94.15% must not be called end-to-end accuracy.', small=True)

    next_page(); title('Step 2 - Controlled model comparisons')
    para('Metrics are label accuracy and macro precision, recall, and F1 across SUPPORTS / REFUTES / NEI. Evidence recall and latency are reported separately. Never rank rows from different samples as though they came from one shared test.')
    heading('Paired development sample: identical 100 validation claims')
    # The source export intentionally retains scopes. Select one observation per displayed condition.
    first = {}
    for r in classifiers:
        if 'first100' in r['scope'] and r['condition'] not in first:
            first[r['condition']] = r
    display_conditions = [('unweighted_reference','Unweighted reference'),('weighted_epoch2_reference','Earlier weighted epoch 2'),('claim_only_H2','H2 claim-only - SELECTED'),('retrieved_k1','Retrieved top-1'),('retrieved_k5','Retrieved top-5 - SELECTED'),('retrieved_k10','Retrieved top-10'),('empty_evidence','Empty evidence'),('wrong_claim_evidence','Wrong-claim evidence'),('gold_evidence_oracle','Gold evidence - ORACLE'),('hybrid_v1','Symbolic hybrid v1'),('hybrid_v2','Symbolic hybrid v2')]
    # Condition spellings in the historical source are validated below by explicit aliases.
    aliases = {'unweighted_reference':['unweighted_reference','original_unweighted'], 'weighted_epoch2_reference':['weighted_reference_epoch2'], 'retrieved_k5':['retrieved_k5','retrieved_evidence'], 'hybrid_v1':['v1'], 'hybrid_v2':['v2']}
    chosen_rows = []
    for condition, label in display_conditions:
        found = next((first[k] for k in aliases.get(condition,[condition]) if k in first), None)
        if found is None:
            raise ValueError('Missing paired condition: '+condition+'; available '+str(sorted(first)))
        chosen_rows.append([label]+[fmt(found[k]) for k in ('accuracy','macro_precision','macro_recall','macro_f1')])
    table(['Condition','Accuracy','Macro P','Macro R','Macro-F1'],chosen_rows,[212,73,73,73,73])
    para('Every displayed condition uses the same claims. Frozen inference uses batch 8, length 256, seed 42. Evidence depth, empty input, and wrong-claim input vary within the same evidence checkpoint. Claim-only versus evidence models also differ in training data and checkpoints; that comparison does not isolate the causal benefit of evidence.', small=True)
    heading('Frozen held-out sample: identical 100 test claims')
    table(['Condition','Accuracy','Macro P','Macro R','Macro-F1'], metric_rows(heldout, {'unweighted_reference':'Unweighted reference','claim_only_H2':'Frozen H2','retrieved_top5':'Frozen retrieval + RoBERTa','hybrid_v2':'Live symbolic v2 hybrid'}),[212,73,73,73,73])
    para('Seed 874 selects unique test claims without consulting labels; overlap with train/validation is rejected. These frozen comparisons use MPS, batch 1, and length 256. Retrieval versus H2 accuracy difference is +3 points, paired 95% CI [-9, +14], p=0.7283. H2 has higher macro-F1; retrieval/hybrid miss all 22 NEI claims. These test outputs were not used to tune the new ensemble.',small=True)

    next_page(); title('New development experiment - calibrated ensemble')
    para('The training-input audit confirmed 28,517 empty-evidence NEI training records and no nonempty-evidence NEI records. Live retrieval supplies text regardless of label. This creates a shortcut opportunity and an input mismatch; it does not prove the internal cause of the model failures.')
    heading('Selection protocol and final development choice')
    para('Calibration: 300 unique validation claims within the earlier first-2,000 tuning population. Assessment: 200 claims beyond offset 2,000, excluding all earlier normalized claim texts. Four retrieval/text variants x five claim weights x four NEI biases = 80 configurations. Frozen checkpoint weights, length 256, batch 8, seed 42. Selection used calibration macro-F1, then accuracy; no assessment labels were used to select the configuration.')
    para('PROVISIONAL ensemble choice: original top-5 retrieval; 0.75 x log P(claim-only) + 0.25 x log P(evidence model); add 2 to the NEI score, then choose the maximum. Calibration accuracy/F1: 0.7033/0.6493. The score adjustment is not a calibrated posterior probability.')
    names = {'claim_only_H2':'H2 claim-only','evidence_original':'Original evidence-only','evidence_local_bm25':'Local BM25 evidence-only','evidence_subject_bm25':'Subject-linked BM25','evidence_subject_bm25_titles':'Subject BM25 with titles','selected_fusion':'Ensemble - PROVISIONAL','selected_weight_without_nei_bias':'Ensemble, no NEI adjustment','selected_bias_without_claim_fusion':'NEI adjustment, evidence-only'}
    table(['200-claim assessment','Accuracy','Macro P','Macro R','Macro-F1'], [[names[k]]+[fmt(v[m]) for m in ('accuracy','macro_precision','macro_recall','macro_f1')] for k,v in summary['conditions'].items()],[212,73,73,73,73])
    para('Ensemble NEI recall is 43.1%, versus 36.2% for H2 and 0% for original evidence-only. Accuracy gain versus evidence-only: +10 points, paired bootstrap CI [+3.5, +16.5], p=0.0045. Versus H2: +4 points, CI [-1, +9], p=0.1686. Superiority over H2 is not established. Statistics are exploratory and unadjusted for multiple comparisons.')
    para('Complete evidence-group recall@5 on the same 142 verifiable assessment claims: original retrieval 58.45%; local BM25 52.82%; subject-linked BM25 55.63%. The new rankers were not selected. A corrected parser run repeats the same protocol/sample; the earlier run is preserved as superseded, not counted as an independent replication.',small=True)
    heading('Prepared for future training - not a trained improvement')
    para('A 96-candidate training-data pilot retained 77 nonempty-evidence records: 25 SUPPORTS, 20 REFUTES, 32 NEI. Verifiable records must contain a complete gold group; 19 were skipped rather than relabeled. This filtering introduces retrieval-success bias. A larger, matched training comparison requires approval; no retraining was run.')

    if official:
        next_page(); title('Full official FEVER benchmark')
        para('All 19,998 official development claims are evaluated in their original order: 6,666 per class. Models and ensemble settings were frozen before inference. No gold evidence was supplied to the models; no tuning or retraining used this evaluation. All local conditions share these same claims, length 256, batch 8, seed 42, and MPS.')
        heading('Primary final reporting table - measured locally')
        table(['Configuration','Accuracy','Macro P','Macro R','Macro-F1','FEVER score'],
              [[{'claim_only_H2':'Selected H2 claim-only','retrieved_roberta':'Selected evidence epoch 1 + top-5','calibrated_ensemble':'Selected calibrated ensemble'}[r['condition']]] +
               [fmt(r[k]) for k in ('accuracy','macro_precision','macro_recall','macro_f1','fever_score')]
               for r in all_classifiers if r['scope']=='official_fever_development19998'], [160,69,69,69,69,68])
        para('Accuracy counts correct labels. Strict FEVER score also requires a complete annotated evidence group among the first five submitted sentences for SUPPORTS/REFUTES; correct NEI labels need no evidence. Claim-only submits no evidence, so its strict score comes only from correct NEI predictions. Classification macro-F1 and evidence F1 are different metrics.')
        heading('Published development references - not rerun locally')
        table(['Published system','Label accuracy','FEVER score'],
              [[name,fmt(value['label_accuracy']),fmt(value['fever_score'])] for name,value in official['protocol']['comparison']['rows'].items()], [252,126,126])
        para('Source: KGAT paper, Table 2, Dev columns: https://aclanthology.org/2020.acl-main.655.pdf. These references use the same official development population and scoring rules. Their architectures, training budgets, retrieval systems, and corpus filtering differ. The authors\' models were not rerun, so this is a benchmark comparison, not a controlled reproduction.')
        heading('What this establishes')
        para('The ensemble reaches 61.70% accuracy versus H2 at 58.78% and evidence-only at 52.37%. Its gain over H2 is +2.92 percentage points (paired claim-bootstrap 95% interval +2.45 to +3.37). Strict FEVER rises from 35.07% evidence-only to 45.33% ensemble. Our models remain below the published GEAR/KGAT references. Claim correlations and exploratory multiple comparisons limit statistical interpretation.')
        para('Official IDs have zero overlap with the original training pool. Three normalized claim texts overlap that broader pool; reconstruction of the actual sampled training subsets found zero overlaps. The primary benchmark retains all 19,998 claims; a separate 19,995-claim sensitivity check excludes those three texts. Our index lacks a complete annotated group for 130 verifiable claims; they remain in the denominator.', small=True)
        para('The unmodified official scorer is pinned and licensed. All 59,994 condition predictions cover identical IDs; rescoring reproduces the results. Per-class metrics, probabilities, evidence references, environment details, source hashes, and overlap/corpus audits are retained in the official_fever appendix.', small=True)

    next_page(); title('RQ1, RQ2 and RQ4 - bounded findings')
    heading('RQ1 - Generated claims: human review still pending')
    para('Thirty local-LLM claims and source-blinded review sheets for 60 claims were prepared. No human reviews have been completed. Two automated coding models each achieved only 30% accuracy on the 30 FEVER reference claims and returned constant NONE error categories. Their labels cannot establish generated-claim factual-error rates. RQ1 remains incomplete.')
    heading('RQ2 - Local LLM judge accuracy and consistency')
    para('Same balanced 30 FEVER claims, two evidence conditions, two prompt wordings, three repeats = 360 actual local requests. Installed Llama 3.2 was chosen for availability. Temperature .2; seeds 42/43/44. Values are mean +/- sample SD across inference repeats, not training repeats.')
    judge_values=[]
    for condition in ['retrieved:direct','retrieved:evidence_first','empty:direct','empty:evidence_first']:
        subset=[r for r in judges if r['condition']==condition]
        acc=[float(r['accuracy']) for r in subset]; f1=[float(r['macro_f1']) for r in subset]
        judge_values.append([condition,f'{mean(acc):.4f} +/- {stdev(acc):.4f}',f'{mean(f1):.4f} +/- {stdev(f1):.4f}',fmt(subset[0]['all_repeats_agree_fraction'])])
    table(['Evidence / prompt','Accuracy','Macro-F1','3 repeats agree'],judge_values,[155,118,118,113])
    para('On these same 30 claims: H2 accuracy/F1 = .5000/.4818; retrieved RoBERTa = .5333/.4274. Empty-evidence judge prompts are abstention controls and consistently return NEI. High consistency does not imply correctness. This small local-model pilot does not justify a universal claim about LLM judges.',small=True)
    heading('RQ4 - Nested corpus growth and latency')
    table(['Pages','Complete R@5','Accuracy','Baseline ms','Cached route ms'],[
        ['2,000','.6056','.5400','61.94','67.69'],['10,000','.6056','.5500','58.67','57.68'],
        ['50,000','.6197','.5400','64.80','58.00'],['5,396,106','.6197','.5500','106.26','113.21']], [112,105,80,100,107])
    para('Same 100 claims, nested query-aware indexes, top-5, frozen model, CPU, batch 1; two alternating-order repeats. Every route still makes 100 retrieval and 100 model calls. Cached routing excludes live Wikidata latency. Query-aware core retention, caches, and fixed corpus-size order limit generalization.',small=True)
    table(['Held-out MPS pass','Retrieval ms/claim','Hybrid ms/claim'],[['Initial online','66.35','106.57'],['Warm snapshot 1','58.89','60.26'],['Warm snapshot 2','56.97','50.78']],[226,139,139])
    para('Initial hybrid timing includes seven real HTTP attempts with no failures. Symbolic v2 resolves zero claims and avoids zero model/retrieval calls. Warm differences fluctuate without avoided work. Neither a live hybrid speedup nor monetary cost savings is established. Ensemble latency was not benchmarked.',small=True)

    next_page(); title('Selected configurations and reproducibility')
    table(['Component','Chosen reporting configuration','Reason'], choices,[108,164,232])
    heading('Step 3 - Reproducibility and honest limits')
    para('Research used the existing .venv. Training/evaluation seed: 42; held-out sample seed: 874; judge repeats: 42/43/44. Saved predictions, prompts, responses, protocols, hashes, and index memberships support audit. The reusable verifier reproduced all 200 saved ensemble predictions. ' + ('Thirty-five unit tests passed; all full-benchmark submissions were rescored successfully.' if official else 'Thirty unit tests passed.'))
    para('Extra Transformer training seeds remain optional and unrun. Epochs, judge repeats, and duplicated log entries are not independent training replications. Additional training requires approval. ' + ('The ensemble now has full official-development results; blind-test or new-dataset generalization remains unmeasured.' if official else 'The ensemble has development evidence only; testing on a new reserved population remains future work.'))
    heading('Step 4 - What is included')
    para('The PDF summarizes selected model configurations and all principal controlled conditions. The accompanying appendix preserves all 14 historical training-log observations, every saved classifier evaluation separated by population, all 12 judge condition/repeat rows, original retrieval and latency tables, all 80 calibration candidates, and configuration-selection rationales. Two identical legacy epoch-1 log rows remain observations, not two runs.')
    para('A 100-claim validation extension is also retained in the classifier appendix: H2 .7000 accuracy / .5788 F1; top-5 .5700 / .3707; top-10 .6000 / .3961. It is disjoint from the first 100 by ID/text but still belongs to the validation tuning population. Its incomplete oracle condition was omitted rather than scored on a different N.',small=True)
    heading('Overall completion status')
    para('The tuning, reproducible artifacts, and updated results table are completed for the measured study. RQ2, RQ3, and RQ4 have bounded measured experiments, including negative findings. RQ1 cannot be completed until independent human reviews and adjudication are supplied. The symbolic-efficiency hypothesis is unsupported by the measured implementation. Prepared training data is not a trained fix.')
    heading('Sources, access and disclosure')
    para('Sources: user-supplied Part 1 PDF and dated experiment reports. Source hashes and inventory are in manifest.json. ' + ('The official_fever folder contains the full benchmark and published-reference comparison. ' if official else '') + 'Earlier submissions/results remain preserved. Nothing was merged into main.',small=True)
    para('Run the experimental ensemble from the repository root: .venv/bin/python src/verify_calibrated.py --claim "The Eiffel Tower is in Paris."',small=True)
    para('AI assisted code, planning, analysis, and writing. Measurements come from saved local runs; human labels were not fabricated. Review this disclosure against the course AI-use policy.',small=True)

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#d5dfe8')); canvas.line(54,43,558,43)
        canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#516474'))
        canvas.drawString(54,30,'COSC 874 | Model Execution - Part 2 | October 23, 2026')
        canvas.drawRightString(558,30,str(doc.page))

    doc = SimpleDocTemplate(str(PDF), pagesize=letter, rightMargin=54,leftMargin=54,topMargin=43,bottomMargin=55,
                            title='Model Execution - Part 2: Updated Results', author='Praise Ben')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (OUT / 'MODEL_EXECUTION_PART2_UPDATED.md').write_text('\n'.join(markdown))
    input_sources = sources + [OLD/'all_classifier_evaluations.csv', STEP2/'heldout_conditions.csv', Path('/Users/praiseben/Downloads/Model_Execution_Part_1.pdf'),Path(__file__)]
    manifest = {'status':'updated_measured_study_RQ1_human_review_pending', 'research_branch':'experiment/part2-paired-ablation',
                'new_retraining':False, 'additional_training_seeds':False, 'new_test_evaluation':False,
                'full_official_development_evaluated':bool(official),
                'source_sha256':{str(p):sha(p) for p in input_sources},
                'classifier_rows':len(all_classifiers), 'training_log_rows':len(training), 'judge_run_rows':len(judges),
                'pdf':str(PDF), 'limitations':['RQ1 human review incomplete.','New ensemble is development-only.','Additional training seeds unrun.','No established symbolic accuracy/efficiency benefit.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    bundle=OUT/'COSC874_Part2_Updated.zip'
    with ZipFile(bundle,'w' if refresh else 'x',ZIP_DEFLATED) as archive:
        archive.write(PDF,PDF.name)
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path != bundle: archive.write(path,str(path.relative_to(OUT)))
    print(json.dumps({'pdf':str(PDF),'appendix':str(OUT),'bundle':str(bundle),'classifier_rows':len(all_classifiers)},indent=2))


if __name__=='__main__':
    main()
