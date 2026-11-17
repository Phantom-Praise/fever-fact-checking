"""Create the course submission DOCX from verified saved experiment tables.

Artifact formatting uses the Codex bundled document runtime. Research execution
continues to use the project's existing .venv. No inference or training occurs.
"""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    return f'{float(value):.4f}'


def paragraph(document, text, bold_lead=None):
    p = document.add_paragraph()
    if bold_lead and text.startswith(bold_lead):
        p.add_run(bold_lead).bold = True
        p.add_run(text[len(bold_lead):])
    else:
        p.add_run(text)
    return p


def table(document, headers, rows, widths, selected=()):
    result = document.add_table(rows=1, cols=len(headers))
    result.autofit = False
    for col, width in zip(result.columns, widths):
        col.width = Inches(width)
    properties = result._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        border = OxmlElement('w:' + side)
        for key, value in [('val', 'single'), ('sz', '5'), ('color', 'D9D9D9')]:
            border.set(qn('w:' + key), value)
        borders.append(border)
    properties.append(borders)
    header = result.rows[0]
    repeat = OxmlElement('w:tblHeader')
    header._tr.get_or_add_trPr().append(repeat)
    for i, label in enumerate(headers):
        header.cells[i].text = str(label)
    for row in rows:
        cells = result.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = str(text)
    for row_index, row in enumerate(result.rows):
        no_split = OxmlElement('w:cantSplit')
        row._tr.get_or_add_trPr().append(no_split)
        for i, cell in enumerate(row.cells):
            cell.width = Inches(widths[i])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tc_pr = cell._tc.get_or_add_tcPr()
            margins = OxmlElement('w:tcMar')
            for side, value in [('top', '80'), ('bottom', '80'), ('left', '90'), ('right', '90')]:
                item = OxmlElement('w:' + side)
                item.set(qn('w:w'), value)
                item.set(qn('w:type'), 'dxa')
                margins.append(item)
            tc_pr.append(margins)
            if row_index == 0 or row_index - 1 in selected:
                fill = OxmlElement('w:shd')
                fill.set(qn('w:fill'), 'DCE6F1' if row_index == 0 else 'F0F3F6')
                tc_pr.append(fill)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.05
                for run in p.runs:
                    run.font.size = Pt(9.5)
                    run.font.color.rgb = RGBColor(0, 0, 0)
                    if row_index == 0 or row_index - 1 in selected:
                        run.bold = True
    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(1)
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.line_spacing = 0.5
    spacer.add_run().font.size = Pt(2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', default='reports/part2/submission_20261023_final')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[2]
    source = Path(args.tables)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source / 'manifest.json').read_text())
    for path, expected in manifest['input_sha256'].items():
        if sha(base / path) != expected:
            raise ValueError('Measured source changed: ' + path)
    evaluations = {(r['scope'], r['condition']): r for r in read_csv(source / 'all_classifier_evaluations.csv')}
    training = read_csv(source / 'all_training_runs.csv')
    local = json.loads((base / 'reports/part2/local_llm_20261023/analysis_blinded/summary.json').read_text())
    recalls = json.loads((base / 'reports/part2/corpus_growth_20261023/evaluation_recall_corrected/recall.json').read_text())
    timings = json.loads((base / 'reports/part2/corpus_growth_20261023/evaluation/timing/summary.json').read_text())['results']

    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin = section.bottom_margin = Inches(.7)
    section.left_margin = section.right_margin = Inches(.7)
    section.footer_distance = Inches(.3)
    for name in ('Normal', 'Title', 'Subtitle', 'Heading 1', 'Heading 2', 'Caption'):
        style = doc.styles[name]
        style.font.name = 'Times New Roman'
        fonts = style.element.get_or_add_rPr().find(qn('w:rFonts'))
        if fonts is not None:
            for attribute in ('asciiTheme', 'hAnsiTheme', 'eastAsiaTheme', 'cstheme', 'csTheme'):
                fonts.attrib.pop(qn('w:' + attribute), None)
            for attribute in ('ascii', 'hAnsi', 'eastAsia', 'cs'):
                fonts.set(qn('w:' + attribute), 'Times New Roman')
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.underline = False
        style.paragraph_format.space_after = Pt(6)
    for border in list(doc.styles.element.iter(qn('w:pBdr'))):
        border.getparent().remove(border)
    doc.styles['Normal'].font.size = Pt(11)
    doc.styles['Normal'].paragraph_format.line_spacing = 1.08
    doc.styles['Normal'].paragraph_format.widow_control = True
    doc.styles['Title'].font.size = Pt(22)
    doc.styles['Title'].paragraph_format.space_after = Pt(5)
    doc.styles['Subtitle'].font.size = Pt(13)
    for name, size in [('Heading 1', 13), ('Heading 2', 11.5)]:
        doc.styles[name].font.size = Pt(size)
        doc.styles[name].font.bold = True
        doc.styles[name].paragraph_format.space_before = Pt(10)
        doc.styles[name].paragraph_format.space_after = Pt(5)
    doc.core_properties.title = 'Model Execution Part 2'
    doc.core_properties.subject = 'FEVER fact checking experiments and selected configurations'
    doc.core_properties.author = 'COSC 874 Research Project'
    doc.core_properties.keywords = 'FEVER, RoBERTa, ablation, reproducibility'
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run('Page ')
    for field, after in [('PAGE', ' of '), ('NUMPAGES', '')]:
        run = footer.add_run()
        element = OxmlElement('w:fldSimple')
        element.set(qn('w:instr'), field)
        run._r.addnext(element)
        footer.add_run(after)
    for run in footer.runs:
        run.font.size = Pt(9)
    doc.add_paragraph('Model Execution Part 2', 'Title')
    doc.add_paragraph('FEVER Fact Checking Experiments', 'Subtitle')
    paragraph(doc, 'COSC 874 Deep Learning II | October 23, 2026')
    paragraph(doc, 'Four tuning configurations select H2 epoch 2 for claim-only RoBERTa. The evidence model retains epoch 1, and title-aware top-5 retrieval remains the reporting reference. Paired ablations, a local LLM judge pilot, and corpus-growth experiments are complete. Symbolic verifier v2 resolved no claims in the original sample and showed no measured benefit. RQ1 generated-claim labels still require independent review.')
    paragraph(doc, 'Scope. All reported scores are development-validation results. The test partition remains untouched. Different evaluation samples and gold versus retrieved evidence are reported separately.', 'Scope.')
    doc.add_heading('Hyperparameter tuning', 1)
    paragraph(doc, 'RoBERTa-base architecture, batch size 8, token length 256, seed 42, a fixed 20% training subset, and the same 2,000 validation claims were held constant. Learning rates bracket the initial 2e-5 setting; H4 tests removing weight decay. Checkpoints are selected by validation macro-F1, with accuracy as the tie-breaker.')
    rows = []
    for name in ('H1', 'H2', 'H3', 'H4'):
        r = next(r for r in training if r['run'] == name + '_seed42' and r['epoch'] == '2')
        rows.append([name + (' selected' if name == 'H2' else ''), r['learning_rate'], r['weight_decay'], r['epoch'], number(r['accuracy']), number(r['macro_f1'])])
    table(doc, ['Config', 'Learning rate', 'Weight decay', 'Epoch', 'Accuracy', 'Macro F1'], rows, [1.3, 1.25, 1.2, .7, 1.0, 1.15], selected=(1,))
    paragraph(doc, 'H2 has the highest accuracy (0.6775) and macro-F1 (0.6202) among H1-H4. Its selected settings are learning rate 1e-5, weight decay 0.01, two epochs, batch size 8, and seed 42. This small search is sufficient to document a reasoned tuning decision; it does not establish a global optimum.')
    doc.add_heading('Evidence model selection', 2)
    table(doc, ['Gold evidence checkpoint', 'Validation N', 'Accuracy', 'Macro F1'], [
        ['Epoch 1 selected', '2,000', '0.9415', '0.9335'], ['Epoch 2', '2,000', '0.9365', '0.9313']], [2.8, 1.2, 1.3, 1.3], selected=(0,))
    paragraph(doc, 'Epoch 1 is retained by macro-F1. These gold-evidence scores do not establish end-to-end retrieval performance. Empty gold evidence for NOT ENOUGH INFO can expose a label-related cue. All 14 historical training observations, including legacy runs with incomplete provenance, remain in the accompanying CSV.')

    doc.add_page_break()
    doc.add_heading('Paired baseline and ablation results', 1)
    paragraph(doc, 'All conditions below use the original 100 validation IDs. Fresh model evaluations use frozen checkpoints, batch size 8, seed 42, and token length 256. Precision, recall, and F1 are macro averages across SUPPORTS, REFUTES, and NOT ENOUGH INFO. The earlier checkpoints were reevaluated on these same IDs.')
    rows = []
    specifications = [
        ('validation_first100_reference', 'unweighted_reference', 'Original unweighted reference'),
        ('validation_first100_reference', 'weighted_reference_epoch2', 'Earlier weighted epoch 2'),
        ('validation_first100_saved', 'claim_only_H2', 'Selected claim only H2'),
        ('validation_first100_topk', 'retrieved_k1', 'Retrieved top 1'),
        ('validation_first100_topk', 'retrieved_k5', 'Selected retrieval top 5'),
        ('validation_first100_topk', 'retrieved_k10', 'Retrieved top 10'),
        ('validation_first100_topk', 'empty_evidence', 'Empty evidence'),
        ('validation_first100_topk', 'wrong_claim_evidence', 'Wrong claim evidence'),
        ('validation_first100_topk', 'gold_evidence_oracle', 'Gold evidence oracle'),
        ('validation_first100_symbolic', 'v1', 'Hybrid symbolic v1'),
        ('validation_first100_symbolic', 'v2', 'Hybrid symbolic v2')]
    for scope, condition, label in specifications:
        r = evaluations[(scope, condition)]
        rows.append([label] + [number(r[key]) for key in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1')])
    table(doc, ['Condition', 'Accuracy', 'Macro precision', 'Macro recall', 'Macro F1'], rows, [2.5, .95, 1.1, 1.05, 1.0], selected=(2, 4))
    paragraph(doc, 'Within-checkpoint ablations change only the evidence input. Top-1, top-5, and top-10 use prefixes of one ranked retrieval. Wrong-claim evidence is a deterministic cyclic shift. The oracle is diagnostic and can leak NEI through empty evidence; it is not an end-to-end baseline.')
    doc.add_heading('Retrieval recall on the original sample', 2)
    first = json.loads((base / 'reports/part2/evidence_depth_20261023/summary.json').read_text())
    table(doc, ['Retrieved K', 'Verifiable N', 'Any sentence recall', 'Complete group recall'], [
        [r['k'], r['eligible'], number(r['any_recall']), number(r['complete_recall'])] for r in first['retrieval']], [1.15, 1.25, 2.15, 2.05])
    paragraph(doc, 'Top-10 adds no complete evidence-group hits on this sample and ties top-5 accuracy with slightly lower macro-F1. Top-5 remains a frozen reference. The retrieved-minus-H2 accuracy difference is -0.07, with paired bootstrap 95% CI [-0.17, +0.04] and exact McNemar p=0.2649. These exploratory, unadjusted comparisons also change the training/checkpoint between claim-only and evidence models.')
    paragraph(doc, 'Symbolic v1 and v2 both resolved 0/100 claims; conditional accuracy is undefined, and no retrieval or Transformer calls were avoided. Offline replay reproduced all symbolic predictions and abstention reasons. None of these measurements supports a hybrid improvement.')

    doc.add_page_break()
    doc.add_heading('Separate validation extension', 1)
    paragraph(doc, 'The next 100 eligible claims were selected by scanning validation from raw offset 100, excluding prior IDs, casefolded claim text, and within-slice duplicates. Selection did not use labels. This slice is disjoint from the original sample but remains part of the validation tuning pool; it is not a held-out final test.')
    rows = []
    for scope, condition, label in [
        ('validation_extension100_reference', 'unweighted_reference', 'Original unweighted reference'),
        ('validation_extension100_reference', 'weighted_reference_epoch2', 'Earlier weighted epoch 2'),
        ('validation_extension100_disjoint_text', 'claim_only_H2', 'Selected claim only H2'),
        ('validation_extension100_disjoint_text', 'retrieved_k1', 'Retrieved top 1'),
        ('validation_extension100_disjoint_text', 'retrieved_k5', 'Frozen retrieval top 5'),
        ('validation_extension100_disjoint_text', 'retrieved_k10', 'Retrieved top 10'),
        ('validation_extension100_disjoint_text', 'empty_evidence', 'Empty evidence'),
        ('validation_extension100_disjoint_text', 'wrong_claim_evidence', 'Wrong claim evidence')]:
        r = evaluations[(scope, condition)]
        rows.append([label] + [number(r[key]) for key in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1')])
    table(doc, ['Condition', 'Accuracy', 'Macro precision', 'Macro recall', 'Macro F1'], rows, [2.5, .95, 1.1, 1.05, 1.0], selected=(2,))
    extension = json.loads((base / 'reports/part2/validation_extension_20261023_complete/summary.json').read_text())
    doc.add_heading('Retrieval recall on the extension', 2)
    table(doc, ['Retrieved K', 'Verifiable N', 'Any sentence recall', 'Complete group recall'], [
        [r['k'], r['eligible'], number(r['any_recall']), number(r['complete_recall'])] for r in extension['retrieval']], [1.15, 1.25, 2.15, 2.05])
    paragraph(doc, 'Oracle evidence was available for only 99/100 extension IDs. The entire oracle condition was omitted instead of silently evaluating a different population. NEI claims are excluded from recall denominators; complete-group recall requires at least one full annotated evidence group.')
    paragraph(doc, 'Top-5 versus H2 has an accuracy difference of -0.13, paired bootstrap 95% CI [-0.22, -0.04], and unadjusted McNemar p=0.0106. Top-10 corrects three top-5 errors, but its paired p=0.25 does not establish a general advantage. H2 remains selected using the original 2,000-example tuning criterion, even though the earlier weighted checkpoint has slightly higher macro-F1 on the small paired slices.')
    doc.add_heading('Error analysis', 2)
    paragraph(doc, 'In the original 100-claim retrieved-evidence condition, all 29 NEI claims are misclassified. Among verifiable claims, 38 correct and 6 incorrect predictions have a complete evidence group; 17 correct and 10 incorrect predictions lack a complete group. Evidence availability and classifier behavior both warrant investigation. These counts do not prove why an error occurred.')

    doc.add_page_break()
    doc.add_heading('LLM judge accuracy and consistency', 1)
    paragraph(doc, 'The RQ2 pilot uses the first 10 claims of each FEVER class from the original sample. Each of the same 30 claims receives two evidence conditions, two prompt wordings, and three fresh requests: 360 actual local judgments. No gold label or oracle evidence is supplied. Conditions share one installed Llama 3.2 3.2B Q4_K_M model, temperature 0.2, context 4096, output limit 32 tokens, and repeat seeds 42/43/44. Request order is shuffled with seed 42.')
    rows = []
    for name, summary in local['judge_condition_summaries'].items():
        rows.append([name.replace(':', ' / ').replace('evidence_first', 'evidence first'),
                     f"{summary['mean_accuracy']:.4f} +/- {summary['sample_std_accuracy_across_repeats']:.4f}",
                     f"{summary['mean_macro_f1']:.4f} +/- {summary['sample_std_macro_f1_across_repeats']:.4f}",
                     number(summary['all_repeats_agree_fraction'])])
    table(doc, ['Evidence and prompt', 'Accuracy mean and SD', 'Macro F1 mean and SD', 'All repeats agree'], rows, [2.35, 1.5, 1.6, 1.15])
    paragraph(doc, 'On the same 30 IDs, H2 scores accuracy 0.5000 / macro-F1 0.4818 and retrieved-evidence RoBERTa scores 0.5333 / 0.4274. The LLM evidence-first condition averages 0.4000 / 0.3850. This small pilot permits a descriptive FEVER comparison, not a conclusion about all LLM judges. Repeat judgments remain 30 unique claims, not 90 independent examples.')
    paragraph(doc, 'Retrieved-evidence prompt agreement is 0.7889 with Cohen\'s kappa 0.6821. Each retrieved prompt has complete repeat agreement on 25/30 claims. Empty-evidence requests consistently return NEI, giving one-third accuracy on this balanced sample. This is an evidence-constrained abstention control, not a closed-book knowledge test. Raw responses and the exact model digest are retained; all 360 outputs parsed successfully.')
    doc.add_heading('Generated claims for RQ1', 1)
    paragraph(doc, 'The local model generated 30 distinct claims using retrieval-derived topic titles, temperature 0.7, seeds 1000-1029, and a 128-token output limit. All original outputs were retained, including apparently incomplete or compound sentences. Labels and FEVER claim text were not supplied to generation. The topic sampling introduces bias.')
    rows = []
    for label, key in [('Mean words', 'mean_words'), ('Median words', 'median_words'), ('Contains digits', 'digit_fraction'), ('Contains an explicit year', 'explicit_year_fraction')]:
        rows.append([label, number(local['fever_profile'][key]), number(local['generated_profile'][key])])
    table(doc, ['Descriptive property', 'FEVER balanced 30', 'Generated 30'], rows, [3.1, 1.75, 1.75])
    paragraph(doc, 'These are prompt-dependent text properties, not measured factual-error rates. Two blinded 60-claim review sheets, candidate evidence, and an agreement/adjudication scorer are prepared. Generated-claim truth, atomicity, and error labels remain unreviewed. RQ1 representativeness therefore remains partial; no human annotations are fabricated.')

    doc.add_page_break()
    doc.add_heading('Corpus growth and verification efficiency', 1)
    paragraph(doc, 'RQ4 uses nested indexes with a common 923-page core selected by claim-text title search, plus seeded random distractors. No gold labels or evidence select the core. This controlled design preserves candidate-page availability and measures distractor/index growth; it is not a representative random-corpus study. The original full index remains read-only.')
    rows = []
    for r in recalls:
        baseline = [x for x in timings if x['db'] == r['db'] and x['condition'] == 'baseline']
        cached = [x for x in timings if x['db'] == r['db'] and x['condition'] == 'cached_symbolic']
        rows.append([f"{r['pages']:,}", number(r['recall'][1]['complete_recall']), number(baseline[0]['accuracy']), number(baseline[0]['macro_f1']),
                     f"{statistics.mean(x['mean_seconds'] for x in baseline)*1000:.2f}",
                     f"{statistics.mean(x['mean_seconds'] for x in cached)*1000:.2f}"])
    table(doc, ['Pages', 'Complete R at 5', 'Accuracy', 'Macro F1', 'Baseline ms', 'Cached route ms'], rows, [1.25, 1.2, 1.0, 1.0, 1.2, .95])
    paragraph(doc, 'Latency is milliseconds per claim, averaged over 200 claims per row. Each size uses the same 100 claims, frozen evidence checkpoint, top-5 retrieval, length 256, batch size 1, CPU, and two repeats with alternating condition order. The full index contains 5,396,106 pages and 25,247,888 sentences. Model loading is excluded.')
    paragraph(doc, 'All conditions make 100 retrieval and 100 model calls per repeat. Zero symbolic coverage provides no work to skip. Cached routing excludes live Wikidata requests and rule execution, so it cannot establish live hybrid savings or monetary cost reduction. Small-index timing is non-monotonic, corpus size order is fixed, and cache/thermal effects limit interpretation. No scaling law is inferred.')
    doc.add_heading('Reproducibility and experiment records', 1)
    paragraph(doc, 'Research runs use the existing Python virtual environment. Seeds, source hashes, predictions, raw judge responses, model digest, Wikidata responses, index membership lists, and corrected recall artifacts are saved. Seventeen tests passed, including duplicate-text exclusion, nested membership, missing-gold rejection, independent judge request context, and conservative symbolic rules. The first corpus-recall export lacked gold references and is superseded by the validated corrected export; saved timing and classification trials were unaffected.')
    paragraph(doc, 'Independent Transformer training seeds were not rerun. Epochs, duplicate legacy log entries, and inference repeats are not reported as independent training runs. Additional training requires approval. Existing results are preserved, and the experiment branch has not been merged into main. The test partition remains untouched.')
    doc.add_heading('Submission materials and conclusion', 1)
    paragraph(doc, 'The supplementary tables contain all training observations, classifier conditions, judge repeats, retrieval runs, latency trials, and configuration choices. Their manifest links each source to a SHA-256 hash. The full Markdown report provides commands and detailed limitations; raw outputs support reanalysis without repeating inference.')
    paragraph(doc, 'The selected reporting configurations are H2 epoch 2 for claim-only RoBERTa, evidence epoch 1, and title-aware top-5 as the frozen reference. Symbolic v2 is an experimental variant with no demonstrated benefit. RQ2 and the bounded RQ4 pilot have measurements; independent annotation and adjudication are still required before drawing RQ1 factual-error representativeness conclusions.')
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    print(output)


if __name__ == '__main__':
    main()
