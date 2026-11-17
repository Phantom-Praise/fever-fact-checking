"""Write a concise course report from saved measurements; no model execution."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
from statistics import mean, stdev
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/part2/submission_official_20261023'
OFFICIAL=ROOT/'reports/part2/official_fever_dev_20261023'
OUT=ROOT/'reports/part2/course_report'
PDF=ROOT/'output/pdf/Model_Execution_Part_2_Report.pdf'


def read_csv(path):
    with path.open(newline='') as stream: return list(csv.DictReader(stream))


def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    if PDF.exists(): raise FileExistsError(PDF)
    training=read_csv(SOURCE/'all_training_runs.csv')
    classifiers=read_csv(SOURCE/'all_classifier_evaluations_updated.csv')
    judges=read_csv(SOURCE/'all_judge_runs.csv')
    official=json.loads((OFFICIAL/'summary.json').read_text())
    assert len(classifiers)==44 and len(training)==14
    assert official['conditions']['calibrated_ensemble']['n']==19998
    normal=ParagraphStyle('Body',fontName='Times-Roman',fontSize=10.5,leading=13.5,spaceAfter=7)
    small=ParagraphStyle('Small',parent=normal,fontSize=9,leading=11.5,spaceAfter=6)
    heading_style=ParagraphStyle('Section',fontName='Times-Bold',fontSize=12,leading=15,spaceBefore=10,spaceAfter=7)
    sub_style=ParagraphStyle('Subsection',fontName='Times-Bold',fontSize=10.5,leading=13.5,spaceBefore=7,spaceAfter=5)
    cell=ParagraphStyle('Cell',fontName='Times-Roman',fontSize=9.2,leading=11.5)
    story=[]; markdown=[]

    def p(text,compact=False):
        story.append(Paragraph(escape(text),small if compact else normal));markdown.extend([text,''])
    def h(text,sub=False):
        story.append(Paragraph(escape(text),sub_style if sub else heading_style));markdown.extend([('### ' if sub else '## ')+text,''])
    def table(headers,values,widths,bold_rows=()):
        content=[]
        for i,row in enumerate([headers]+values):
            content.append([Paragraph(('<b>'+escape(str(v))+'</b>') if i==0 or i-1 in bold_rows else escape(str(v)),cell) for v in row])
        tab=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),
            ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
            ('LINEABOVE',(0,0),(-1,0),.65,colors.black),('LINEBELOW',(0,0),(-1,0),.4,colors.black),
            ('LINEBELOW',(0,-1),(-1,-1),.4,colors.black)]))
        story.extend([tab,Spacer(1,7)])
        markdown.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
        markdown.extend('| '+' | '.join(str(v) for v in row)+' |' for row in values);markdown.append('')
    def page():story.append(PageBreak())
    def pct(x):return f'{float(x)*100:.2f}%'
    def f1(x):return f'{float(x):.4f}'
    def find(scope,condition):return next(r for r in classifiers if r['scope']==scope and r['condition']==condition)

    story.append(Paragraph('Model Execution - Part 2',ParagraphStyle('Title',fontName='Times-Bold',fontSize=18,leading=22,spaceAfter=8)))
    markdown.extend(['# Model Execution - Part 2',''])
    p('Praise Ben | COSC 874: Deep Learning II')
    p('This stage tests how training settings, retrieved evidence, and a symbolic lookup affect the FEVER fact-checking system. The task has three labels: SUPPORTS, REFUTES, and NOT ENOUGH INFO (NEI). Accuracy and macro-F1 are the main classification measures; macro-F1 gives each class equal weight.')
    h('1. Hyperparameter tuning')
    p('Part 1 showed that the original baseline predicted only SUPPORTS. Class-weighted loss produced predictions for all three classes, so that loss was retained for tuning. The search compared a lower and a higher learning rate with the original 2e-5 setting, then tested whether removing weight decay helped.')
    p('All four configurations used RoBERTa-base, batch size 8, maximum length 256, two epochs, and seed 42. They shared the same 20% training subset and the first 2,000 validation claims. The table shows the best epoch from each run; the attached training log includes both epochs.')
    tuned=[r for r in training if r['family']=='claim_only_tuned' and r['epoch']=='2']
    table(['Run','Learning rate','Weight decay','Epoch','Accuracy','Macro-F1'],
          [[r['run'].replace('_seed42','')+(' (chosen)' if r['run']=='H2_seed42' else ''),r['learning_rate'],r['weight_decay'],r['epoch'],pct(r['accuracy']),f1(r['macro_f1'])] for r in tuned],
          [100,85,80,48,96,95],bold_rows=(1,))
    p('H2 was chosen because it had the highest validation macro-F1 (0.6202), as well as the highest accuracy. Its learning rate is 1e-5 and its weight decay is 0.01. These results support keeping the lower learning rate for the claim-only model.')
    p('For the evidence model, epoch 1 was retained. Its gold-evidence validation macro-F1 was 0.9335, compared with 0.9313 at epoch 2. Validation loss also increased at epoch 2, despite lower training loss. This supported stopping at the earlier checkpoint.')
    p('The high gold-evidence score is a diagnostic result, not end-to-end accuracy. The gold inputs give NEI claims empty evidence, while SUPPORTS and REFUTES receive text. This creates a shortcut that does not match live retrieval. The retrieved-evidence evaluations below avoid using gold evidence as model input.')

    page();h('2. Experiments')
    h('Full FEVER development evaluation',True)
    p('The main evaluation used all 19,998 official FEVER development claims, with 6,666 examples per class. The three systems below used the same claims. Their checkpoints and decision settings were fixed before this evaluation; none was trained or tuned on these claims.')
    names={'claim_only_H2':'Claim-only H2','retrieved_roberta':'Retrieved evidence + RoBERTa','calibrated_ensemble':'Combined model (chosen)'}
    final=[]
    for condition in names:
        r=find('official_fever_development19998',condition)
        final.append([names[condition],pct(r['accuracy']),f1(r['macro_precision']),f1(r['macro_recall']),f1(r['macro_f1']),pct(r['fever_score'])])
    table(['System','Accuracy','Macro P','Macro R','Macro-F1','FEVER score'],final,[169,70,64,64,68,69],bold_rows=(2,))
    p('The combined model reached 61.70% accuracy and 0.6078 macro-F1. It improved accuracy by 2.92 percentage points over claim-only H2. A paired bootstrap interval for this difference was +2.45 to +3.37 points. This is a measured gain on this benchmark, although related claims may share evidence and are not fully independent.',True)
    p('The official FEVER score also checks evidence: a correct SUPPORTS or REFUTES label needs a complete annotated evidence group among the first five submitted sentences. Correct NEI labels do not need evidence. The claim-only model supplies no evidence, so its FEVER score comes only from correct NEI predictions.',True)
    h('Evidence ablations on a shared 100-claim validation sample',True)
    pairs=[('validation_first100_reference','unweighted_reference','Original unweighted baseline'),
           ('validation_first100_reference','weighted_reference_epoch2','Earlier weighted baseline'),
           ('validation_first100_saved','claim_only_H2','Claim-only H2'),
           ('validation_first100_topk','retrieved_k1','Top-1 retrieved sentence'),
           ('validation_first100_topk','retrieved_k5','Top-5 retrieved sentences'),
           ('validation_first100_topk','retrieved_k10','Top-10 retrieved sentences'),
           ('validation_first100_saved','empty_evidence','Empty evidence'),
           ('validation_first100_saved','wrong_claim_evidence','Evidence from another claim'),
           ('validation_first100_saved','gold_evidence_oracle','Gold evidence (diagnostic)'),
           ('validation_first100_symbolic','v1','Symbolic v1 + top-5'),
           ('validation_first100_symbolic','v2','Symbolic v2 + top-5')]
    table(['Condition','Accuracy','Macro-F1'],[[label,pct(find(scope,condition)['accuracy']),f1(find(scope,condition)['macro_f1'])] for scope,condition,label in pairs],[306,100,98])
    p('Top-10 tied top-5 accuracy and had slightly lower macro-F1, so top-5 was kept. Wrong-claim evidence reduced performance. Empty evidence sometimes scored better than retrieved evidence, showing that retrieval alone did not solve the classification problem. The symbolic stages resolved none of these claims and did not change the predictions.',True)
    p('These are development diagnostics, separate from the full benchmark above. The evidence ablations share one checkpoint, but claim-only and evidence models were trained on different subsets. Their comparison does not isolate the effect of adding evidence.',True)

    page();h('Additional experiments')
    h('Combining the claim-only and evidence models',True)
    p('A separate calibration experiment used 300 internal validation claims to choose a decision rule, then assessed it on 200 different claims beyond the earlier tuning sample. The chosen rule weights claim-only log probabilities by 0.75 and evidence-model log probabilities by 0.25, then adds 2 to the NEI score before choosing a label.')
    cal=[('claim_only_H2','Claim-only H2'),('evidence_original','Evidence model'),('selected_fusion','Combined model'),
         ('selected_weight_without_nei_bias','Combined, without NEI adjustment'),('selected_bias_without_claim_fusion','NEI adjustment, without claim model')]
    table(['200-claim assessment','Accuracy','Macro-F1'],[[label,pct(find('new_validation_assessment200',key)['accuracy']),f1(find('new_validation_assessment200',key)['macro_f1'])] for key,label in cal],[306,100,98])
    p('The combined model scored 71% on this small assessment. The 61.70% reported on the previous page comes from the much larger official set; the two percentages describe different populations. Alternative BM25 sentence-ranking methods did not improve complete evidence recall@5 over the original retriever, so the original retrieval method was retained.',True)
    h('Local LLM judge',True)
    p('The judge experiment used the same balanced 30 FEVER claims for two prompts, two evidence conditions, and three repeats: 360 local Llama 3.2 requests. The table gives mean +/- sample standard deviation across repeats.')
    judge_rows=[]
    for condition,label in [('retrieved:direct','Retrieved evidence, direct prompt'),('retrieved:evidence_first','Retrieved evidence, evidence-first prompt'),('empty:direct','Empty evidence, direct prompt'),('empty:evidence_first','Empty evidence, evidence-first prompt')]:
        subset=[r for r in judges if r['condition']==condition]
        a=[float(r['accuracy']) for r in subset]; f=[float(r['macro_f1']) for r in subset]
        judge_rows.append([label,f'{100*mean(a):.2f}% +/- {100*stdev(a):.2f}',f'{mean(f):.4f} +/- {stdev(f):.4f}'])
    table(['Condition','Accuracy','Macro-F1'],judge_rows,[262,120,122])
    p('On those same 30 claims, claim-only H2 reached 50% accuracy and retrieved RoBERTa reached 53.33%. The judge returned NEI consistently without evidence. That consistency did not make it accurate on a balanced three-class sample.',True)
    h('Corpus size and symbolic lookup',True)
    table(['Pages','Accuracy','Retrieval ms/claim','Cached route ms/claim'],[
        ['2,000','54%','61.94','67.69'],['10,000','55%','58.67','57.68'],
        ['50,000','54%','64.80','58.00'],['5,396,106','55%','106.26','113.21']],[120,78,153,153])
    p('These runs used the same 100 claims, nested indexes, CPU inference, and two timing repeats. The indexes retained a common set of candidate pages, so they test distractor growth rather than random evidence availability. Every route still called retrieval and RoBERTa for all 100 claims. In a separate live MPS evaluation, the initial symbolic route took 106.57 ms per claim versus 66.35 ms without it. These experiments did not demonstrate an efficiency benefit.',True)

    page();h('3. Reproducibility')
    p('Training and model evaluation used seed 42. The 100-claim internal test sample used selection seed 874, and the LLM judge repeats used seeds 42, 43, and 44. Checkpoints, settings, predictions, source hashes, and raw judge responses were saved. The official FEVER scorer was used unchanged, with a five-sentence evidence limit.')
    p('The Transformer training runs were not repeated with additional seeds. Judge repeats and timing repeats therefore measure inference variation, not variation between independently trained models. Additional training seeds would strengthen the result, but were not run in this study.')
    p('The official development IDs did not overlap the original training pool. Three claim texts appeared in both pools; reconstruction of the sampled training subsets found none of those texts in either model\'s training subset. A separate analysis excluded the three texts, while the main benchmark retained the complete official set. The index also lacked a complete annotated evidence group for 130 verifiable claims; these claims remained in the evaluation.')
    h('4. Final configuration choices')
    choices=[['Unweighted baseline','Original epoch 1','Reference for the majority-class failure in Part 1.'],
             ['Claim-only model','H2, epoch 2; LR 1e-5; weight decay 0.01','Highest validation macro-F1 among H1-H4.'],
             ['Evidence model','Epoch 1; LR 2e-5; weight decay 0.01','Better gold-validation macro-F1 than epoch 2; evaluated separately with retrieved text.'],
             ['Retrieval','Original title-aware retrieval, top-5','Top-10 gave no consistent benefit; new rankers did not improve recall.'],
             ['Combined model','Claim weight 0.75; NEI adjustment 2','Selected on internal calibration before the official evaluation; strongest local full-benchmark result.'],
             ['Symbolic stage','Experimental comparison only','No resolved claims or avoided model calls in the measured samples.']]
    table(['Model / component','Chosen configuration','Reason'],choices,[105,173,226])
    p('The combined model is the main configuration for reporting. Its full-benchmark result is 61.70% accuracy, 0.6078 macro-F1, and 45.33% FEVER score. For context, GEAR reported 74.84% accuracy / 70.69% FEVER score on the official development set, and KGAT with RoBERTa-large reported 78.29% / 76.11% [1]. Their models were not rerun here, and their training budgets, retrieval systems, and corpus filtering differ.')
    p('The largest unresolved training issue is the empty-evidence NEI input. A small replacement-data pilot was prepared, but no replacement model was trained. The planned comparison of FEVER and generated-claim error types also remains incomplete: 30 generated claims still need independent human review. The automated coding pilot was too inaccurate to supply reliable labels.')
    p('The attached CSV files contain all 14 historical training-log observations, 44 classifier evaluations grouped by sample, 12 judge condition/repeat rows, and the retrieval and timing results. The calibration file records all 80 decision/ranker combinations. Duplicate historical log entries are retained as observations, not counted as independent runs.',True)
    p('[1] Liu et al. (2020). Fine-grained Fact Verification with Kernel Graph Attention Network, Table 2. https://aclanthology.org/2020.acl-main.655/',True)
    p('AI assistance was used for coding and writing. The reported measurements come from the saved local experiment outputs.',True)

    def footer(canvas,doc):
        canvas.setFont('Times-Roman',9);canvas.drawCentredString(letter[0]/2,30,str(doc.page))
    PDF.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(PDF),pagesize=letter,leftMargin=54,rightMargin=54,topMargin=43,bottomMargin=48,
                      title='Model Execution - Part 2',author='Praise Ben').build(story,onFirstPage=footer,onLaterPages=footer)
    (OUT/'report.md').write_text('\n'.join(markdown))
    sources=[SOURCE/name for name in ('all_training_runs.csv','all_classifier_evaluations_updated.csv','all_judge_runs.csv','all_retrieval_runs.csv','all_latency_runs.csv')]
    sources += [ROOT/'reports/part2/literature_improvements_20261023_corrected/calibration_candidates.json',OFFICIAL/'comparison.csv',OFFICIAL/'summary.json',OFFICIAL/'protocol.json']
    for source in sources: shutil.copy2(source,OUT/source.name)
    with (OUT/'chosen_configurations.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['component','configuration','reason']);writer.writerows(choices)
    (OUT/'manifest.json').write_text(json.dumps({'source_sha256':{str(p):digest(p) for p in sources},'report_source_sha256':digest(Path(__file__)),
        'pdf_sha256':digest(PDF),'new_experiments':False,'human_review_pending':True,'additional_training_seeds_run':False},indent=2))
    with ZipFile(OUT/'Model_Execution_Part_2_Submission.zip','x',ZIP_DEFLATED) as archive:
        archive.write(PDF,PDF.name)
        for path in sorted(OUT.iterdir()):
            if path.is_file() and path.suffix!='.zip':archive.write(path,path.name)
    print('Created',PDF)


if __name__=='__main__': main()
