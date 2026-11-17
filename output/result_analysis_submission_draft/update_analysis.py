from pathlib import Path
import json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];out=Path(__file__).parent
main=root/'reports/part2/parts_test60_20261030_v2';replay=root/'reports/part2/roberta_rag_replay60_20261030'
a=json.loads((main/'summary.json').read_text());b=json.loads((replay/'summary.json').read_text())
path=out/'Result_Analysis.md'
backup=out/'Result_Analysis_before_rag_baseline_revision.md'
if not backup.exists():shutil.copy2(path,backup)
t=backup.read_text()
refs=json.loads((main/'reference_labels.json').read_text())
print('reference format',type(refs).__name__)
# The file is a mapping of claim ID to reference label.
if 'labels' in refs:refs=refs['labels']
print('reference keys',list(refs)[:3])
rows=[]
for p in sorted((replay/'traces/rag_256').glob('*.json')):
 r=json.loads(p.read_text());m=json.loads((main/'traces/parts_rag'/p.name).read_text());gold=refs[str(r['id'])]
 if isinstance(gold,dict):gold=gold.get('label',gold.get('gold'))
 rows.append((r['id'],r['label']==gold,m['label']==gold))
d=np.array([int(m)-int(r) for _,r,m in rows]);rng=np.random.default_rng(42)
ci=np.quantile(d[rng.integers(0,60,(10000,60))].mean(axis=1),[.025,.975])
paired={'n':60,'comparison':'saved RAG Llama vs saved RAG RoBERTa 256; exploratory','accuracy_difference':float(d.mean()),'fixed':sum(m and not r for _,r,m in rows),'introduced':sum(r and not m for _,r,m in rows),'bootstrap_seed':42,'bootstrap_repetitions':10000,'paired_percentile_95_interval':ci.tolist()}
(out/'rag_paired_analysis.json').write_text(json.dumps(paired,indent=2))
print(paired)
conditions=[('RAG + RoBERTa (256)',b['conditions']['rag_256']),('RAG + Llama parts',a['conditions']['parts_rag']),('Five sentences + RoBERTa',a['conditions']['evidence_roberta'])]
intro='''The report foregrounds the requested **RAG + RoBERTa versus RAG + Llama** comparison. The original five-sentence RoBERTa remains a reference condition. This change in presentation is retrospective: the RoBERTa RAG replay took place after inspection of the 60 claims and is **exploratory**, not a newly frozen confirmatory study.

Both RAG classifiers received evidence drawn from the same saved passages in the same order. This controls evidence retrieval better than the original comparison, but does not give identical effective context: RoBERTa retained at most 256 or 512 tokens, while Llama used a different tokenizer and larger prompt budget. RoBERTa was not retrained on RAG evidence. None of these conditions reads every database chunk.

'''
t=t.replace('### Main comparison: the same 60 claims','### Main comparison: the same 60 claims\n\n'+intro)
t=t.replace('The baseline retrieves five Wikipedia evidence sentences and classifies the claim/evidence pair with a frozen RoBERTa checkpoint, using a 256-token input budget.','The RAG RoBERTa baseline concatenates the saved retrieved article passages and classifies them with the claim using a frozen RoBERTa checkpoint. Both 256- and 512-token budgets were tested. The original five-sentence baseline is retained as an additional reference.') if False else t
# Replace complete architecture paragraph and table area while preserving following literature section.
start=t.index('The baseline retrieves five Wikipedia evidence sentences')
end=t.index('### Relation to the literature review')
t=t[:start]+'''The RAG RoBERTa baseline concatenates the saved retrieved passages and classifies the claim/evidence pair with a frozen checkpoint. Its original training is unchanged. The main system retrieves Wikipedia articles, chunks and embeds their contents in a temporary per-claim vector store, retrieves passages, and supplies them to local Llama 3.1 8B. Llama returns essential-part verdicts; code derives REFUTES if any part is refuted, SUPPORTS if all are supported, and NEI otherwise. Invalid responses count as incorrect.

| Condition on the same 60 claims | Correct | Accuracy | Macro-F1 | Invalid | Truncated RoBERTa inputs |
|---|---:|---:|---:|---:|---:|
| RAG + RoBERTa, 256 tokens — comparison baseline | 32/60 | 53.33% | 0.4264 | 0 | 60/60 |
| RAG + RoBERTa, 512 tokens — length variant | 32/60 | 53.33% | 0.4264 | 0 | 56/60 |
| RAG + Llama part verdicts — main | 38/60 | 63.33% | 0.6583 | 4 | Not comparable across tokenizers |
| Original five sentences + RoBERTa, 256 tokens — reference | 34/60 | 56.67% | 0.4548 | 0 | 2/60 |
| Original five sentences + RoBERTa, 512 tokens — reference variant | 34/60 | 56.67% | 0.4548 | 0 | 0/60 |

![Figure 1. Same-claim accuracy and macro-F1, including the RAG RoBERTa baseline.](comparison.png)

The main model answered six more claims correctly than RAG RoBERTa, and four more than the original five-sentence reference. Giving the frozen RoBERTa more RAG evidence did not improve its accuracy. At 512 tokens, two predictions improved and two worsened compared with RAG at 256, leaving its score unchanged. Matching headline scores does not imply identical predictions.

Mean end-to-end measured time was 58.76 seconds for RAG + Llama and 0.14 seconds for the original five-sentence reference, excluding setup. RoBERTa RAG replay measured classifier inference only, reusing previously fetched and embedded evidence. It is therefore not an end-to-end RAG timing comparison; we do not assign that pipeline a falsely low total runtime.

'''+t[end:]
# Insert accurate scope statement before literature.
idx=t.index('### Relation to the literature review')
scope='''### Why 60 claims, and what the larger dataset numbers mean

The 60 claims were a practical sample chosen to fit the approved 3–4 hour project window, not the full FEVER test set and not the entire dataset. Our internal test partition contained 14,548 rows; the earlier audit identified 13,448 eligible unique claims after exclusions. This internal split differs from the official benchmark split used in published work.

FEVER's official release contains 185,445 claims. KGAT Table 1 lists **145,449 training claims** and **19,998 development and 19,998 test claims**. Training size is not the number of examples behind an evaluation percentage. The hybrid paper by Kolli et al. evaluated **1,000 sampled SUPPORTS/REFUTES claims**, excluding NEI; it did not report its headline F1 on every FEVER claim. We should not describe all literature results as evaluations of “120,000 claims.”

At the measured RAG + Llama mean of 58.76 seconds per claim, a simple serial extrapolation to 13,448 eligible claims is about **219.5 hours (9.1 continuous days)**. Hypothetically, 120,000 claims would take about **1,958.6 hours (81.6 days)** at that same rate. These are rough projections, not completed runs or promised runtimes; caching, article lengths, network behavior, and hardware can change them. They estimate this pipeline's inference time, not RoBERTa training time. The earlier pilot-based plan estimated about 8.4 days for 13,448 claims; the estimate here uses the completed 60-claim mean instead.

Sources: [official FEVER dataset](https://fever.ai/dataset/fever.html), [KGAT Tables 1–2](https://aclanthology.org/2020.acl-main.655.pdf), and [hybrid paper, Section 4](https://aclanthology.org/2025.winlp-main.19.pdf).

'''
t=t[:idx]+scope+t[idx:]
# Replace per-class table block and interpretation before symbolic.
start=t.index('### Performance by class');end=t.index('### Separate symbolic experiment')
section='''### Performance by class

Precision asks: when the system predicts a class, how often is it correct? Recall asks: of the claims belonging to that class, how many did it find? F1 balances both. Macro-F1 averages the three class F1 scores equally.

| System | Class | Precision | Recall | F1 | Correct / class total |
|---|---|---:|---:|---:|---:|
'''
for name,c in conditions:
 for label,m in c['per_class'].items():
  section+=f"| {name} | {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['TP']}/20 |\n"
section+='''
Neither RoBERTa evidence condition predicted NEI. Its undefined NEI precision is represented as zero by the scoring convention. The main model found 12/20 NEI claims, but correctly classified fewer SUPPORTS and REFUTES claims than either RoBERTa condition. The improvement in overall accuracy therefore reflects a trade-off, not better performance on every class. The known empty-evidence NEI training mismatch is a plausible contributor to RoBERTa's NEI weakness, not a proven causal explanation.

![Figure 2. Confusion matrices for all three conditions.](confusion.png)

Rows below are actual FEVER labels; columns are predictions. INVALID remains an error, not a fourth ground-truth class.

'''
for name,c in conditions:
 section+='**'+name+'**\n\n| Actual \\ Predicted | SUPPORTS | REFUTES | NEI | INVALID |\n|---|---:|---:|---:|---:|\n'
 for label,values in zip(['SUPPORTS','REFUTES','NEI'],c['confusion_matrix']):section+='| '+label+' | '+' | '.join(map(str,values))+' |\n'
 section+='\n'
section+='''Because this is a three-class task, TP/FP/FN/TN are calculated separately for each class, treating that class as positive and all others as negative.

| System | Positive class | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|
'''
for name,c in conditions:
 for label,m in c['per_class'].items():section+='| '+name+' | '+label+' | '+' | '.join(str(m[k]) for k in ['TP','FP','FN','TN'])+' |\n'
section+=f'''
Against RAG RoBERTa at 256 tokens, Llama fixed **{paired['fixed']}** errors and introduced **{paired['introduced']}**, for a net **+10.00 percentage points**. An exploratory paired bootstrap (10,000 resamples, seed 42) gives a 95% percentile interval of **{ci[0]*100:.2f} to +{ci[1]*100:.2f} percentage points**. This resamples the 60 claims; it does not measure run-to-run variation or remove retrospective-selection bias. Against the original five-sentence reference, Llama fixed 14 and introduced 10; the earlier paired interval was −10.00 to +21.67 percentage points. Neither interval establishes a reliable population accuracy advantage.

Four Llama responses hit the output cap and were invalid. Model-provided and code-derived labels disagreed on four claims, but both scored 38/60; aggregation did not improve accuracy on this sample.

'''
t=t[:start]+section+t[end:]
# Add symbolic diagnostic causes with evidence distinctions.
idx=t.index('## 3. Qualitative analysis')
causes='''### Why the symbolic stage did not help

The 293 stored statements are a local record of externally retrieved Wikidata assertions, including identifiers and metadata. They are not 293 claims verified, and they were not generated as true facts from the input claims.

| Observed issue | Evidence from this run | Interpretation |
|---|---|---|
| Narrow claim parser | 96/100 claims rejected before lookup | The dominant measured bottleneck. A useful fact may exist but is never requested. |
| Operational failures | Two accepted claim lookup chains failed | These are service/request failures, not evidence that Wikidata lacked facts; the saved generic error does not establish a more specific cause. |
| Conservative comparison | One accepted claim lacked sufficient evidence under the rules | Qualified, ambiguous, missing, or insufficiently precise facts can trigger abstention; guessing would not be a valid repair. |
| Parser defect | “The seventies” was routed toward a place lookup | An implementation bug, not a limitation of symbolic knowledge in principle. |
| Limited opportunity for gains | The one resolved claim was already correct under RoBERTa | One avoided neural call, zero baseline errors repaired. |
| Network overhead | Mean time increased from 0.104 to 0.207 seconds | Lookups outweighed the saving from one avoided call in this measured setting. |

The literature's hybrid pipeline links entities broadly, retrieves one-hop DBpedia triples, ranks them semantically, and gives selected facts to an LLM or DeBERTa before web fallback. Our implementation requires a few wording patterns and uses handwritten value comparisons. It does not reproduce that reasoning architecture. The measured failure is therefore **low coverage and no benefit from this particular implementation**, not a demonstration that knowledge graphs cannot improve fact-checking. Broader entity linking, fact ranking, and model-based evidence comparison are plausible next experiments, not measured fixes in this report.

'''
t=t[:idx]+causes+t[idx:]
# Qualitative examples now identify both baselines.
t=t.replace('| Claim / trace ID | Reference | Baseline | Main | What the saved evidence and response show |','| Claim / trace ID | Reference | Five-sentence RoBERTa | RAG RoBERTa, 256 | Main | What the saved evidence and response show |').replace('|---|---|---|---|---|\n| “Saif','|---|---|---|---|---|---|\n| “Saif')
t=t.replace('| NEI | SUPPORTS | NEI | A passage','| NEI | SUPPORTS | SUPPORTS | NEI | A passage').replace('| SUPPORTS | SUPPORTS | REFUTES | The main','| SUPPORTS | SUPPORTS | SUPPORTS | REFUTES | The main').replace('| SUPPORTS | REFUTES | NEI | Selected','| SUPPORTS | REFUTES | REFUTES | NEI | Selected')
t=t.replace('The predicted accuracy direction was not observed in the 60-claim comparison:** Llama RAG scored 38/60 versus 34/60.','The predicted accuracy direction was not observed in the 60-claim comparison:** Llama RAG scored 38/60 versus RAG RoBERTa 32/60 and the original reference 34/60.')
t=t.replace('The pipelines differ and the interval includes zero','The effective contexts differ, the RAG replay is exploratory, and the interval includes zero')
t=t.replace('Measured results: `reports/part2/parts_test60_20261030_v2/summary.json`','Measured results: `reports/part2/roberta_rag_replay60_20261030/summary.json`, per-claim saved RAG replay traces; `reports/part2/parts_test60_20261030_v2/summary.json`')
path.write_text(t)
# Updated figures, from saved summary metrics only.
fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
colors=['#8096b0','#21756b','#b98b46'];names=['RAG RoBERTa\n256 tokens','RAG Llama\nparts','Five-sentence\nRoBERTa']
for ax,metric,title in zip(axs,['accuracy','macro_f1'],['Accuracy','Macro-F1']):
 vals=[c[metric] for _,c in conditions];ax.bar(names,vals,color=colors);ax.set_ylim(0,1);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
 for i,v in enumerate(vals):ax.text(i,v+.02,f'{v*100:.2f}%' if metric=='accuracy' else f'{v:.4f}',ha='center')
fig.suptitle('Same 60 claims — RAG RoBERTa replay is exploratory');fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
for ax,(name,c) in zip(axs,conditions):
 arr=np.array(c['confusion_matrix']);ax.imshow(arr,cmap='Blues',vmin=0,vmax=20);ax.set_xticks(range(4),['S','R','NEI','Invalid']);ax.set_yticks(range(3),['S','R','NEI']);ax.set_title(name,fontsize=11);ax.set_xlabel('Predicted');ax.set_ylabel('Actual')
 for i in range(3):
  for j in range(4):ax.text(j,i,str(arr[i,j]),ha='center',va='center',color='white' if arr[i,j]>10 else 'black')
fig.suptitle('Confusion matrices: 20 actual claims per class');fig.savefig(out/'confusion.png',dpi=180);plt.close(fig)
print('Updated draft and both figures')
