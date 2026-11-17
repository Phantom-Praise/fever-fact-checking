from pathlib import Path
import json
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

ROOT = Path("/Users/praiseben/fever-fact-checking")
WORK = ROOT / "tmp/fever_first_draft"
OUT = ROOT / "output/doc/FEVER_First_Draft.docx"
data = json.loads((WORK / "sources.json").read_text())
scores = data["scorecard"]
metrics = scores["model_metrics"]
baseline = metrics["roberta_original_rag512"]
tuned = metrics["roberta_last3_rag512"]
assert scores["n"] == 19998
assert all(m["n"] == 19998 and m["invalid"] == 0 for m in metrics.values())
assert scores["paired_counts"] == dict(both_correct=8909, original_only_correct=1311,
                                       selected_only_correct=3532, both_wrong=6246)
examples = data["examples"]
for i in range(4):
    assert examples[i]["retained_input"] == examples[i+4]["retained_input"]
    assert examples[i]["evidence_identity"] == examples[i+4]["evidence_identity"]

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin = sec.bottom_margin = Inches(.8)
sec.left_margin = sec.right_margin = Inches(1)
sec.footer_distance = Inches(.35)
normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(11.5)
normal.paragraph_format.line_spacing = 1.08
normal.paragraph_format.space_after = Pt(4)
for name, size in [("Title", 17), ("Subtitle", 11), ("Heading 1", 13), ("Heading 2", 11.5)]:
    style = doc.styles[name]
    style.font.name = "Times New Roman"
    style.font.size = Pt(size)
    style.font.color.rgb = RGBColor(0,0,0)
    style.font.bold = name != "Subtitle"
    style.paragraph_format.space_before = Pt(7 if name.startswith("Heading") else 0)
    style.paragraph_format.space_after = Pt(5)
    if name.startswith("Heading"):
        style.paragraph_format.keep_with_next = True
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
field = OxmlElement("w:fldSimple")
field.set(qn("w:instr"), "PAGE")
footer._p.append(field)
doc.core_properties.title = "Evidence Retrieval and Fine Tuning for FEVER Fact Verification"
doc.core_properties.author = "Praise Ben"
doc.core_properties.subject = "COSC 874 Deep Learning II first research paper draft"

def p(text, style=None):
    paragraph = doc.add_paragraph(text, style)
    paragraph.paragraph_format.widow_control = True
    return paragraph

next_heading_page = False
def h(text, level=1):
    global next_heading_page
    paragraph = doc.add_heading(text, level=level)
    if next_heading_page:
        paragraph.paragraph_format.page_break_before = True
        next_heading_page = False
    return paragraph

def page():
    global next_heading_page
    next_heading_page = True

def caption(text):
    paragraph = p(text)
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.runs[0].italic = True
    paragraph.runs[0].font.size = Pt(10.5)

def table(headers, rows, widths, shade=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for col, width in zip(t.columns, widths):
        col.width = Inches(width)
    for cell, text in zip(t.rows[0].cells, headers):
        cell.text = text
    for row in rows:
        cells = t.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)
    for ri, row in enumerate(t.rows):
        cant = OxmlElement("w:cantSplit")
        row._tr.get_or_add_trPr().append(cant)
        for ci, cell in enumerate(row.cells):
            cell.width = Inches(widths[ci])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for para in cell.paragraphs:
                para.paragraph_format.space_after = Pt(3)
                para.paragraph_format.space_before = Pt(3)
                para.paragraph_format.line_spacing = 1
                for run in para.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(10.5)
                    run.bold = ri == 0
            if ri == 0 or (shade and (ri,ci) in shade):
                shading = OxmlElement("w:shd")
                shading.set(qn("w:fill"), "E7E7E7" if ri == 0 else "F2F2F2")
                cell._tc.get_or_add_tcPr().append(shading)
    repeat = OxmlElement("w:tblHeader")
    t.rows[0]._tr.get_or_add_trPr().append(repeat)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement("w:" + edge)
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:color"), "BBBBBB")
        borders.append(border)
    t._tbl.tblPr.append(borders)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.line_spacing = Pt(2)
    spacer.add_run("").font.size = Pt(1)
    spacer.paragraph_format.keep_with_next = True
    return t

def link(paragraph, text, url):
    ident = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    el = OxmlElement("w:hyperlink")
    el.set(qn("r:id"), ident)
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "000000")
    props.append(color)
    run.append(props)
    text_el = OxmlElement("w:t")
    text_el.text = text
    run.append(text_el)
    el.append(run)
    paragraph._p.append(el)

# PAGE 1
p("Evidence Retrieval and Fine Tuning for FEVER Fact Verification", "Title")
p("Praise Ben | COSC 874 Deep Learning II | First draft", "Subtitle")
h("Abstract")
abstract = (
"This project examines how evidence inputs and model adaptation affect three-class fact verification on FEVER. "
"The central experiment compares an existing RoBERTa checkpoint with a continued fine-tuned version on all 19,998 "
"official development claims. Both models receive identical passages retrieved from the historical FEVER Wikipedia "
"snapshot and the same 512-token input budget. Following instructor feedback, I fine-tuned the classifier head "
"and the final one, two, or three Transformer blocks on 116,356 training claims. A separate set of 14,534 claims "
"selected the best configuration by validation macro-F1. The three-block configuration reached 69.42% validation "
"macro-F1. On the full development set, label accuracy increased from 51.11% to 62.21%, and macro-F1 increased "
"from 40.84% to 61.12%. The fine-tuned model correctly identified 2,676 Not Enough Info claims, compared with "
"zero for the original checkpoint, while recall declined for Supports and Refutes. A separate symbolic pilot "
"provided no accuracy improvement and increased average latency. These findings support continued fine-tuning "
"under the tested settings, but the proposed hybrid accuracy-and-efficiency hypothesis remains unsupported by "
"the symbolic pilot. Full independent-model LLM comparisons remain incomplete."
)
assert 150 <= len(abstract.split()) <= 250, len(abstract.split())
p(abstract)
h("1 Introduction")
p("Fact-checking systems must find relevant information and decide whether it establishes, contradicts, or leaves a claim unresolved. "
  "A fluent answer or a related Wikipedia page can still fail to establish the claim. My original proposal investigated a symbolic "
  "knowledge-base stage followed by retrieval and Transformer classification for checking LLM-generated claims.")
p("The proposal posed four research questions. RQ1 asks whether FEVER represents modern LLM factual errors. RQ2 asks how accurately "
  "and consistently LLM judges match human labels. RQ3 asks whether symbolic verification improves retrieval and Transformer baselines. "
  "RQ4 asks whether hierarchical verification reduces latency and cost as the corpus grows.")
p("This draft centers the completed 19,998-claim experiment. Following my professor's recommendation, I added a focused comparison of "
  "continued fine-tuning with different numbers of unfrozen layers. The follow-up question asks whether this improves verification "
  "with the same retrieved evidence. The measured improvement mainly concerns Not Enough Info (NEI), while retrieval quality, "
  "truncation, and incomplete symbolic coverage remain open issues.")

# PAGE 2
page()
h("2 Related Work")
h("Benchmark tasks and evidence requirements", 2)
p("Thorne et al. (2018) introduced FEVER, a benchmark of 185,445 Wikipedia-derived claims with Supports, Refutes, and NEI labels. "
  "The dataset includes evidence annotations for verifiable claims and makes it possible to distinguish label correctness from "
  "evidence-supported verification. Jiang et al. (2020) introduced HoVer, which requires combining information across as many as "
  "four Wikipedia articles. Both benchmarks motivate examining whether retrieval actually supplies the information needed for "
  "a claim. The current experiments use FEVER; HoVer remains a possible extension.")
h("Graph reasoning and structured evidence", 2)
p("GEAR represents evidence as a graph and passes information between evidence nodes before aggregating a decision "
  "(Zhou et al., 2019). KGAT adds kernel attention to weight evidence and propagate fine-grained matching signals "
  "(Liu et al., 2020). These papers motivate reasoning across multiple passages instead of assuming that concatenated "
  "related text is sufficient. My completed model comparison uses a RoBERTa encoder over selected passage text, "
  "without a learned evidence graph.")
p("FactKG provides claims designed around several forms of reasoning over knowledge graphs (Kim et al., 2023). "
  "GraphCheck processes extracted knowledge graphs as soft prompts for an LLM (Chen et al., 2025). These approaches "
  "inform the project's symbolic objective. My implemented symbolic pilot instead used restricted claim patterns "
  "and external Wikidata statements. Its low coverage limits what its outcome says about graph-based verification.")
h("LLM verification and claim decomposition", 2)
p("FActScore decomposes generated text into atomic facts and estimates the proportion supported by a knowledge source "
  "(Min et al., 2023). It motivates checking essential assertions individually, although its factual-precision metric "
  "differs from FEVER's three-class labels. Kolli et al. (2025) combine DBpedia lookups, LLM classification, and search "
  "fallback. Their reported F1 of 0.93 concerns the Supports/Refutes split. NEI and the differing evaluation population "
  "prevent treating that score as directly comparable to this project's three-class results.")
h("Position of this study", 2)
p("The central contribution is an instrumented comparison within an existing FEVER pipeline. It holds evaluation passages "
  "and input budgets fixed while changing the classifier's continued training. The trace analysis identifies both corrections "
  "and new errors. The symbolic and LLM pilots contribute implementation observations, while their broader proposed comparisons "
  "still require complete evaluation.")
p("Published development-set label accuracies are included in Table 5. GEAR and KGAT provide external reference points. "
  "Their different architectures, retrieval methods, and training procedures mean the comparison cannot isolate the cause "
  "of their higher scores.")

# PAGE 3
page()
h("3 Methodology")
h("Dataset and partitions", 2)
p("The completed evaluation uses all 19,998 official FEVER development claims, with 6,666 in each class. The project retained "
  "its existing training and validation partitions and removed normalized claim-text overlaps with official development. "
  "Continued training used 116,356 claims and validation used 14,534. Normalization case-folded text and collapsed whitespace. "
  "The official development set had already been inspected during project development.")
h("Evidence retrieval and model input", 2)
p("Title-based retrieval ranks ten Wikipedia page IDs per claim and reads the first five. Their union contains 24,319 "
  "historical FEVER records. The retrieval-augmented generation (RAG) infrastructure prepares evidence for classification. "
  "It embeds 200-token chunks with all-MiniLM-L6-v2, builds a temporary FAISS/SQLite database, retrieves 30 vector "
  "candidates, and packs a 7,600-character context.")
p("Both checkpoints receive the same claim and passage text. The tokenizer preserves the claim and truncates the evidence "
  "to fit 512 tokens. The evaluation checks identical retained inputs for each pair. The temporary article/vector workspace "
  "is deleted after each claim, while supplied passages, source-record hashes, token counts, probabilities, and decoded "
  "retained inputs are saved. No gold evidence is used to choose the evaluation passages.")
h("Baseline and continued fine tuning", 2)
p("The baseline is the existing Part 1 roberta-base FEVER checkpoint. Its earlier training used oracle-style evidence and "
  "empty NEI inputs, creating an input mismatch for retrieved evidence. Each new training condition starts from that same "
  "checkpoint. The classifier head and final one, two, or three blocks are trainable. All other parameters are frozen.")
p("The new training inputs contain the top five retrieved sentences for every class, with reference labels retained even "
  "when retrieval misses. They differ from the full-article RAG evaluation inputs. All conditions completed 14,546 optimizer "
  "steps. Validation macro-F1 selected checkpoint 11,000 in every condition.")
caption("Table 1  Fixed settings for the three continued-training conditions")
table(["Setting", "Value"], [
    ("Epochs and seed", "2 epochs; seed 42"),
    ("Learning rate and weight decay", "0.00002; 0.01"),
    ("Batch and gradient accumulation", "4 examples; 4 steps (effective batch 16)"),
    ("Input budget and precision", "512 tokens; FP16 training"),
    ("Validation and checkpoint interval", "Every 1,000 optimizer steps"),
    ("Compute and software", "Tesla T4; PyTorch 2.11.0; Transformers 4.57.6"),
], [2.7,3.8])
h("Metrics and reproducibility", 2)
p("I report label accuracy, macro-F1, per-class precision/recall/F1, confusion matrices, and paired corrections. "
  "Both conditions completed with zero execution failures. Strict FEVER evidence scoring remains unmeasured. "
  "The run folder roberta_paired_snapshot_dev_v1 retains model, data, and code hashes.")

# PAGE 4
page()
h("4 Results")
caption("Table 2  Completed paired evaluation on all 19,998 FEVER development claims")
table(["Condition", "Correct", "Accuracy (%)", "Macro F1 (%)"], [
    ("Original RoBERTa with snapshot RAG", "10,220", f"{100*baseline['accuracy']:.2f}", f"{100*baseline['macro_f1']:.2f}"),
    ("Fine-tuned RoBERTa with snapshot RAG", "12,441", f"{100*tuned['accuracy']:.2f}", f"{100*tuned['macro_f1']:.2f}"),
], [2.8,1.0,1.3,1.4])
p("Accuracy increased by 11.11 percentage points, corresponding to 2,221 additional correct labels. The fine-tuned "
  "model corrected 3,532 baseline errors and introduced 1,311 errors on previously correct claims. Both models were "
  "correct on 8,909 claims and wrong on 6,246.")
caption("Table 3  Model selection on the separate 14,534-claim validation partition")
selection = json.loads((ROOT / "reports/part2/roberta_full_training_analysis_20261106/validation_selection.json").read_text())
table(["Unfrozen blocks", "Trainable parameters", "Macro F1 (%)", "Validation loss"], [
    (str(n) + (" (selected)" if n == 3 else ""), f"{count:,}", f"{100*r['validation_macro_f1']:.2f}", f"{r['validation_loss']:.4f}")
    for n,count,r in zip([1,2,3],[7680771,14768643,21856515],selection["candidates"])
], [1.5,1.9,1.5,1.6])
p("The final three-block configuration had the highest validation macro-F1. Its 69.42% validation macro-F1 and "
  "61.12% development macro-F1 describe different evaluation partitions and evidence inputs.")
caption("Table 4  Per-class development performance in percentages")
class_rows = []
for name, model in [("Original",baseline),("Fine-tuned",tuned)]:
    for label, short in [("SUPPORTS","Supports"),("REFUTES","Refutes"),("NOT ENOUGH INFO","NEI")]:
        m=model["per_class"][label]
        class_rows.append((name,short,f"{100*m['precision']:.2f}",f"{100*m['recall']:.2f}",f"{100*m['f1']:.2f}"))
table(["Model", "Class", "Precision", "Recall", "F1"], class_rows, [1.3,1.3,1.3,1.3,1.3])
p("NEI accounts for the net gain. Correct NEI predictions increased from 0 to 2,676. Correct Supports predictions "
  "decreased by 166 and correct Refutes predictions decreased by 289. Refutes precision rose from 56.88% to 73.63%, "
  "while recall fell from 65.41% to 61.07%. The updated model made fewer Refutes predictions and a larger proportion "
  "of those predictions were correct.")
p("The original model produced no NEI predictions. Its undefined NEI precision is displayed as zero, matching "
  "the scorer's zero-division convention. Every class contains 6,666 reference claims.")

# PAGE 5
page()
h("4 Results continued")
caption("Table 5  Published FEVER development references")
table(["System", "Label accuracy (%)", "Strict FEVER score (%)"], [
    ("This study, original RoBERTa", "51.11", "Not measured"),
    ("This study, fine-tuned RoBERTa", "62.21", "Not measured"),
    ("GEAR, published", "74.84", "70.69"),
    ("KGAT RoBERTa-large, published", "78.29", "76.11"),
], [3.1,1.6,1.8])
p("Published values are the development columns of Liu et al. (2020), Table 2. The workbook's 67.10% for GEAR and "
  "70.38% for KGAT are test FEVER scores. The table above aligns the split and label metric, while model size and "
  "retrieval still differ. Strict FEVER scoring requires sufficient annotated evidence for verifiable claims in "
  "addition to a correct label.")
caption("Table 6  Confusion matrices with actual labels in rows")
confusion_rows=[]
for name, model in [("Original",baseline),("Fine-tuned",tuned)]:
    for label,row in zip(["Supports","Refutes","NEI"],model["confusion_matrix"]):
        confusion_rows.append((name,label,*[f"{v:,}" for v in row[:3]]))
table(["Model", "Actual", "Predicted S", "Predicted R", "Predicted NEI"], confusion_rows,
      [1.2,1.25,1.25,1.25,1.55], shade={(r,2+(r-1)%3) for r in range(1,7)})
h("Examples from the full development run", 2)
caption("Table 7  Researcher inspection of saved paired traces")
table(["Claim and reference label", "Original / updated", "Visible evidence and interpretation"], [
    ("Theresa May worked the docks.\nReference NEI\nID 111816", "Refutes / NEI",
     "The retained text describes banking and political work. It does not settle whether she worked at docks. Fine-tuning corrected the verdict."),
    ("Michael Hutchence died on a boat.\nReference Refutes\nID 145641", "Refutes / NEI",
     "Both retained inputs state that he was found dead in a Sydney hotel room. The updated model missed a contradiction that was visible in its input."),
    ("Trouble with the Curve is a television show.\nReference Refutes\nID 81957", "Refutes / Refutes",
     "The retained evidence identifies it as a 2012 sports-drama film. Both models returned the reference label."),
], [2.1,1.1,3.3])
p("These cases were selected after reviewing results and illustrate possible error mechanisms. They are not an "
  "independently annotated distribution of error types. The paired traces confirm identical visible inputs; "
  "RoBERTa's internal reasoning cannot be inferred from class probabilities alone.")

# PAGE 6
page()
h("5 Discussion")
h("Interpretation and hypothesis verdicts", 2)
p("NEI gains outweighed lower recall for the two verifiable classes. The added fine-tuning question is supported "
  "under one seed and the tested settings. This comparison followed instructor feedback. The original symbolic "
  "and LLM hypotheses require their own evidence.")
caption("Table 8  Status of the original hypotheses and the added fine-tuning question")
table(["Question or hypothesis", "Assessment"], [
    ("H1  A symbolic hybrid improves verification while reducing latency",
     "Unsupported in the 100-claim symbolic pilot. Accuracy stayed at 71%; mean latency rose from 0.104 to 0.207 seconds."),
    ("H2  LLM judges are less reliable or consistent than supervised classifiers",
     "Unresolved at full-benchmark scale. An earlier 60-claim pilot favored Llama in accuracy; repeated-run consistency remains unmeasured."),
    ("H3  Symbolic routing reduces expensive verification work",
     "One neural call was avoided in 100 claims, but total latency increased. The broader efficiency claim was unsupported in that pilot."),
    ("Follow-up  Does continued fine-tuning improve the fixed-evidence classifier",
     "Supported in the completed comparison. Three unfrozen blocks selected by validation achieved 62.21% development accuracy."),
], [2.65,3.85])
p("Symbolic coverage was limited. Of 100 claims, 96 missed the supported patterns, two encountered service "
  "failures, one lacked usable evidence, and one was verified. The verified case was already correct under "
  "the neural fallback. Broader knowledge-graph methods remain untested.")
h("Limitations and pending work", 2)
p("The development set was inspected throughout the project. An earlier audit found three text matches with "
  "the original training pool. Reconstructing the historical sampled subset found no matches, although a "
  "complete historical per-step manifest was unavailable. Continued-training inputs excluded development overlaps.")
p("Other limits include one seed, three measured layer configurations, a weak starting checkpoint, sentence-based "
  "training versus RAG evaluation, and a 512-token limit. Remaining errors have not been systematically assigned "
  "to retrieval, truncation, classification, or annotation problems. The annotated-evidence diagnostic awaits "
  "a completed, reviewed comparison.")
p("Full independent Qwen, Llama, Mistral, Gemma, and RoBERTa selection results remain pending at 19,998 claims each. "
  "An earlier 60-claim pilot gave Llama 63.33% accuracy and RAG RoBERTa 53.33%, with different effective input budgets. "
  "RQ1 needs independent review of generated claims, and RQ4 needs controlled corpus-scaling measurements.")

h("6 Conclusion")
p("On all 19,998 FEVER development claims, continued fine-tuning improved the existing RoBERTa pipeline by 11.11 "
  "accuracy points under identical retrieved inputs. The class-level results show substantial NEI gains and "
  "reduced recall for Supports and Refutes. The symbolic pilot failed to improve accuracy or latency. "
  "The next steps are to finish the annotated-evidence diagnostic, complete the independent-model evaluations, "
  "and test whether improved retrieval or classifier adaptation addresses the remaining errors.")

# PAGE 7
page()
h("References")
references = [
("Chen, Y., Liu, H., Liu, Y., Xie, J., Yang, R., Yuan, H., Fu, Y., Zhou, P. Y., Chen, Q., Caverlee, J., & Li, I. (2025). "
 "GraphCheck: Breaking long-term text barriers with extracted knowledge graph-powered fact-checking. arXiv preprint arXiv:2502.16514. ",
 "https://doi.org/10.48550/arXiv.2502.16514"),
("Jiang, Y., Bordia, S., Zhong, Z., Dognin, C., Singh, M., & Bansal, M. (2020). HoVer: A dataset for many-hop fact extraction "
 "and claim verification. Findings of the Association for Computational Linguistics: EMNLP 2020, 3441-3460. ",
 "https://doi.org/10.18653/v1/2020.findings-emnlp.309"),
("Kim, J., Park, S., Kwon, Y., Jo, Y., Thorne, J., & Choi, E. (2023). FactKG: Fact verification via reasoning on knowledge graphs. "
 "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 16190-16206. ",
 "https://doi.org/10.18653/v1/2023.acl-long.895"),
("Kolli, S., Rosenbaum, R., Cavelius, T., Strothe, L., Lata, A., & Diesner, J. (2025). Hybrid fact-checking that integrates "
 "knowledge graphs, large language models, and search-based retrieval agents improves interpretable claim verification. "
 "Proceedings of the 9th Widening NLP Workshop, 106-115. ",
 "https://doi.org/10.18653/v1/2025.winlp-main.19"),
("Liu, Z., Xiong, C., Sun, M., & Liu, Z. (2020). Fine-grained fact verification with kernel graph attention network. "
 "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 7342-7351. ",
 "https://doi.org/10.18653/v1/2020.acl-main.655"),
("Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W.-t., Koh, P., Iyyer, M., Zettlemoyer, L., & Hajishirzi, H. (2023). "
 "FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. Proceedings of the 2023 "
 "Conference on Empirical Methods in Natural Language Processing, 12076-12100. ",
 "https://doi.org/10.18653/v1/2023.emnlp-main.741"),
("Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). FEVER: A large-scale dataset for fact extraction "
 "and verification. Proceedings of NAACL-HLT 2018, Volume 1 (Long Papers), 809-819. ",
 "https://doi.org/10.18653/v1/N18-1074"),
("Zhou, J., Han, X., Yang, C., Liu, Z., Wang, L., Li, C., & Sun, M. (2019). GEAR: Graph-based evidence aggregating "
 "and reasoning for fact verification. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, "
 "892-901. ", "https://doi.org/10.18653/v1/P19-1085"),
]
for text,url in references:
    para=p(text)
    para.paragraph_format.left_indent=Inches(.25)
    para.paragraph_format.first_line_indent=Inches(-.25)
    para.paragraph_format.space_after=Pt(8)
    para.paragraph_format.keep_together=True
    for run in para.runs:run.font.size=Pt(10.5)
    link(para,url,url)
h("Project result records", 2)
p("The MiniLM revision is 1110a243fdf4706b3f48f1d95db1a4f5529b4d41. The initial RoBERTa weight SHA-256 "
  "is 7a4a377205db4adf4af1671345e0ddbed2b164b78f1f4c1aaa13666b3b07e7f2.")
p("The full paired scorecard records the 19,998-claim metrics, confusion matrices, and paired corrections. "
  "Training protocols and validation histories are retained under roberta_full_unfreeze_v1. "
  "Relevant implementation entry points are colab/prepare_experiments.py, colab/fever_colab_runtime.py, "
  "colab/snapshot_rag.py, and colab/paired_roberta_evaluation.py.")
para=p("Saved paired scorecard  ")
link(para,"Google Drive result record",
     "https://drive.google.com/file/d/1HCj1eL5qP61MbJMCJKlhxEsNWruSi97d/view")

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("Created",OUT)
print("Abstract words",len(abstract.split()))
print("Paragraph words",sum(len(para.text.split()) for para in doc.paragraphs))
