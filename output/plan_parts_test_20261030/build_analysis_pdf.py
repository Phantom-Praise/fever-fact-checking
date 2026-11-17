"""Create the final analysis PDF from scored outputs and reviewed examples only.

Run with the bundled document Python (reportlab), after final scoring.
"""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                               Image, PageBreak, KeepTogether)

ROOT = Path(__file__).resolve().parents[2]
INK = colors.HexColor('#193044')
TEAL = colors.HexColor('#0f766e')
GRAY = colors.HexColor('#536477')
PALE = colors.HexColor('#edf4f6')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = args.run.resolve()
    summary = json.loads((folder/'summary.json').read_text())
    protocol = json.loads((folder/'protocol.json').read_text())
    review = json.loads((folder/'qualitative_review.json').read_text())
    cases = {str(r['id']): r for r in json.loads((folder/'scored_cases.json').read_text())}
    fonts = next((ROOT/'.venv').glob('lib/python*/site-packages/matplotlib/mpl-data/fonts/ttf'))
    pdfmetrics.registerFont(TTFont('DV', str(fonts/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DV-Bold', str(fonts/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('DV', normal='DV', bold='DV-Bold', italic='DV', boldItalic='DV-Bold')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('BodyDV', fontName='DV', fontSize=9.25, leading=13.5, textColor=INK, spaceAfter=8))
    styles.add(ParagraphStyle('SmallDV', parent=styles['BodyDV'], fontSize=8, leading=11, textColor=GRAY, spaceAfter=6))
    styles.add(ParagraphStyle('HeadDV', fontName='DV-Bold', fontSize=18, leading=24, textColor=INK, spaceAfter=12))
    styles.add(ParagraphStyle('SubDV', fontName='DV-Bold', fontSize=11, leading=15, textColor=TEAL, spaceBefore=8, spaceAfter=7))
    styles.add(ParagraphStyle('EyebrowDV', fontName='DV-Bold', fontSize=8, leading=12, textColor=TEAL, spaceAfter=10))
    styles.add(ParagraphStyle('CellDV', parent=styles['BodyDV'], fontSize=8.1, leading=11, spaceAfter=0))
    styles.add(ParagraphStyle('QuoteDV', parent=styles['BodyDV'], fontSize=8.5, leading=12, leftIndent=10, borderPadding=6,
                              borderColor=TEAL, borderWidth=.5, backColor=PALE, spaceAfter=8))
    def p(text, style='BodyDV'):
        return Paragraph(text, styles[style])
    def plain(text, style='BodyDV'):
        return p(escape(str(text)), style)
    def table(headers, rows, widths):
        data = [[p('<b>'+escape(str(x))+'</b>', 'CellDV') for x in headers]]
        data += [[plain(x, 'CellDV') for x in row] for row in rows]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE), ('VALIGN',(0,0),(-1,-1),'TOP'),
                              ('LEFTPADDING',(0,0),(-1,-1),8), ('RIGHTPADDING',(0,0),(-1,-1),8),
                              ('TOPPADDING',(0,0),(-1,-1),8), ('BOTTOMPADDING',(0,0),(-1,-1),8),
                              ('LINEBELOW',(0,0),(-1,0),.8,TEAL), ('LINEBELOW',(0,1),(-1,-1),.35,colors.HexColor('#dce5eb'))]))
        return t
    width = 7.25*inch
    b, m = (summary['conditions'][c] for c in ['evidence_roberta','parts_rag'])
    da, df = m['accuracy']-b['accuracy'], m['macro_f1']-b['macro_f1']
    lo, hi = summary['paired_accuracy_95_percentile_interval']
    story = [plain('COSC 874 | DEEP LEARNING II', 'EyebrowDV'),
             plain('Result Analysis', 'HeadDV'),
             plain('Temporary Wikipedia RAG with essential-part verdicts', 'SubDV'),
             plain('Research question: Does the frozen RAG + Llama pipeline improve label accuracy and macro-F1 over evidence RoBERTa on the same eligible FEVER test claims?'),
             plain('Study: 60 claims, 20 per label, selected from the audited internal test partition before inference. '
                   'The prompt was developed on other examples. This study does not evaluate the full test partition.'),
             table(['System','Accuracy','Macro-F1','Invalid'],[
                 ['Evidence RoBERTa (baseline)',f"{b['accuracy']:.2%}",f"{b['macro_f1']:.4f}",b['invalid']],
                 ['RAG + Llama parts (main)',f"{m['accuracy']:.2%}",f"{m['macro_f1']:.4f}",m['invalid']],
                 ["Llama's overall label (diagnostic)",f"{summary['conditions']['parts_model_label']['accuracy']:.2%}",
                  f"{summary['conditions']['parts_model_label']['macro_f1']:.4f}",summary['conditions']['parts_model_label']['invalid']]],
                   [235,100,100,width-435]), Spacer(1,10),
             Image(str(folder/'comparison.png'),width=width,height=width*.4),
             plain(f"Observed accuracy difference: {100*da:+.2f} percentage points. Macro-F1 difference: {df:+.4f}. "
                   f"The paired claim-bootstrap 95% accuracy-difference interval is [{100*lo:+.2f}, {100*hi:+.2f}] points."),
             plain('Accuracy measures correct labels; macro-F1 weights the three classes equally. Invalid answers count as errors. '
                   'The main label is derived from the part verdicts; the model-generated overall label remains a separate diagnostic.', 'SmallDV'),
             PageBreak(), plain('Quantitative findings', 'HeadDV'),
             plain('Confusion matrices show the FEVER reference label by row and the prediction by column. '
                   'INVALID is retained as a fourth prediction column.'),
             Image(str(folder/'confusion.png'),width=width,height=width/3),
             plain('Per-class performance', 'SubDV')]
    data = []
    for condition, name in [('evidence_roberta','RoBERTa'),('parts_rag','RAG + Llama')]:
        for label, values in summary['conditions'][condition]['per_class'].items():
            data.append([name,label,f"{values['precision']:.3f}",f"{values['recall']:.3f}",f"{values['f1']:.3f}"])
    story += [table(['System','Label','Precision','Recall','F1'],data,[105,155,90,85,width-435]),Spacer(1,10),
              plain(f"The main system corrected {summary['fixed']} baseline mistakes and introduced {summary['introduced']} new mistakes. "
                    f"Its model-provided and derived labels disagreed on {summary['inconsistent']} cases."),
              plain('Cost and execution', 'SubDV'),
              plain(f"Mean recorded pipeline time: RoBERTa {b['mean_wall_seconds']:.2f} seconds per claim; "
                    f"RAG + Llama {m['mean_wall_seconds']:.2f} seconds. Model setup is recorded separately. "
                    'RAG time includes online article fetching, local embeddings, vector search, generation and temporary-workspace deletion. '
                    'The conditions have different evidence and computation budgets.'),
              plain(review['quantitative_interpretation']),
              PageBreak(), plain('Qualitative evidence', 'HeadDV'),
              plain('These examples explain specific observed outcomes. They are selected by paired correctness and are not estimates of failure-category prevalence.', 'SmallDV')]
    for example in review['examples'][:3]:
        row = cases[str(example['id'])]
        trace = json.loads((folder/'traces/parts_rag'/(str(row['id'])+'.json')).read_text())
        block = [plain(example['title']+' | claim '+str(row['id']), 'SubDV'), plain(row['claim']),
                 plain('Reference: '+row['gold']+' | RoBERTa: '+row['baseline']+' | Main: '+row['main'], 'SmallDV'),
                 plain(example['interpretation'])]
        for ident in example.get('excerpt_ids', [])[:1]:
            excerpt = trace['supplied_evidence'][ident]
            quote = excerpt['text']
            if len(quote)>520:
                quote=quote[:520].rsplit(' ',1)[0]+' ... [excerpt shortened for display]'
            block += [plain(quote,'QuoteDV'),
                      p('Evidence '+escape(ident)+' | '+escape(excerpt['title'])+' | '
                        '<link href="'+escape(excerpt['revision_url'])+'" color="#0f766e">Wikipedia revision '+str(excerpt['revision_id'])+'</link>', 'SmallDV')]
        if example.get('model_excerpt'):
            block.append(plain('Model output: '+example['model_excerpt'], 'SmallDV'))
        story.append(KeepTogether(block))
    story.append(plain(review['faithfulness_example']['observation'], 'SmallDV'))
    story += [plain('Exact complete prompts, unedited responses, all supplied excerpts, source revisions, token counts and '
                    'RoBERTa retained inputs accompany the report in the saved trace files and local dashboard. '
                    'Generated justifications are explanations, not verified hidden reasoning.', 'SmallDV'),
              PageBreak(), plain('Hypothesis, literature and limits', 'HeadDV'),
              plain('Hypothesis verdict', 'SubDV'), plain(review['hypothesis_verdict']),
              plain('Literature context', 'SubDV'),
              plain('The table below uses the same full official development population and metric definitions for the historical '
                    'baseline and published references. The new 60-claim main-model study is excluded from this table.', 'SmallDV'),
              table(['System / official development set','Label accuracy','Strict FEVER'],[
                  ['Historical evidence RoBERTa (256 tokens)', '52.37%', '35.07%'],
                  ['GEAR - published reference [1, 2]', '74.84%', '70.69%'],
                  ['KGAT RoBERTa-large - published [2]', '78.29%', '76.11%']], [270,126,width-396]), Spacer(1,8),
              plain('All rows concern 19,998 official development claims. Published models were not rerun; architectures, '
                    'retrieval, model size and training budgets differ. Our observed full-benchmark baseline is lower on both '
                    'metrics, but this does not measure the new RAG main model.', 'SmallDV'),
              plain('The workbook headline values, GEAR 67.10% and KGAT 70.38%, are test FEVER scores. '
                    'The table above instead uses the development columns of KGAT Table 2 to match our historical evaluation.', 'SmallDV'),
              plain('Methodological context: GEAR/KGAT learn evidence graphs; our method uses a temporary vector database '
                    'and prompted part verdicts. The hybrid result of 0.931 F1 excludes NEI [3]. '
                    'The FActScore value of 58% for ChatGPT measures supported atomic-fact precision [4]. '
                    'Those two percentages are not directly comparable with three-class accuracy.', 'SmallDV'),
              plain('Main limitations', 'SubDV')]
    limitations = ['Small balanced sample, one model/quantization and one execution; uncertainty remains substantial.',
                   'Live Wikipedia differs from the historical FEVER snapshot. Article block citations cannot support a strict FEVER score.',
                   'RoBERTa learned from annotated evidence and empty NEI inputs; its training mismatch limits baseline interpretation.',
                   'Retrieval, corpus, context, classifier and aggregation change together. This study cannot isolate which change caused a difference.',
                   'The 256-token generation cap can cause incomplete outputs; errors are preserved. Citation validation does not establish logical correctness.',
                   'Full articles and vectors are deleted. Saved supplied prompts can be replayed, but exact retrieval over the deleted corpus cannot.']
    for text in limitations:
        story.append(plain('• '+text,'SmallDV'))
    story += [plain('References', 'SubDV')]
    refs = [('[1] Zhou et al. (2019), GEAR.', 'https://aclanthology.org/P19-1085/'),
            ('[2] Liu et al. (2020), KGAT, Table 2.', 'https://aclanthology.org/2020.acl-main.655/'),
            ('[3] Kolli et al. (2025), Hybrid Fact-Checking, Table 1.', 'https://aclanthology.org/2025.winlp-main.19/'),
            ('[4] Min et al. (2023), FActScore.', 'https://aclanthology.org/2023.emnlp-main.741/')]
    for title, url in refs:
        story.append(p(escape(title)+' <link href="'+url+'" color="#0f766e">Paper</link>', 'SmallDV'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#dce5eb'))
        canvas.line(45,36,letter[0]-45,36)
        canvas.setFont('DV',8)
        canvas.setFillColor(GRAY)
        canvas.drawString(45,23,'COSC 874 | FEVER result analysis')
        canvas.drawRightString(letter[0]-45,23,str(doc.page))
    doc = SimpleDocTemplate(str(args.output),pagesize=letter,leftMargin=45,rightMargin=45,
                            topMargin=39,bottomMargin=47,title='Result Analysis: Wikipedia RAG with Essential-Part Verdicts')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print('Created',args.output)


if __name__=='__main__':
    main()
