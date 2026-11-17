"""Exploratory frozen-RoBERTa replay: saved five-sentence vs saved RAG evidence.

No FAISS, article fetching, Llama calls or training. All 60 prior cases remain,
including cases where Llama's generation was invalid. Gold is read only to score.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

from experiment_parts_test import LABELS, metrics, write_new

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/part2/parts_test60_20261030_v2'
CHECKPOINT = ROOT / 'checkpoints/roberta_evidence/epoch_1'
CONDITIONS = [('five_sentence_256', 256, 'five_sentences'), ('rag_256', 256, 'rag_excerpts'),
              ('five_sentence_512', 512, 'five_sentences'), ('rag_512', 512, 'rag_excerpts')]


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rag_excerpts(trace):
    """Use ALL supplied passages in their original order, never just Llama citations."""
    if trace['input']['claim'] != trace['claim']:
        raise ValueError('Claim mismatch in saved RAG context')
    selected = []
    for passage in trace['input']['passages']:
        for excerpt in passage['evidence']:
            metadata = trace['supplied_evidence'][excerpt['id']]
            if metadata['text'] != excerpt['text']:
                raise ValueError('Saved passage and citation map differ')
            selected.append(dict(id=excerpt['id'], **metadata))
    if not selected:
        raise ValueError('Missing supplied RAG passages')
    return selected


def evidence_text(excerpts):
    return ' '.join(e['text'] for e in excerpts)


def prepare(out):
    if out.exists():
        raise FileExistsError('Choose a new experiment directory')
    original = load(SOURCE / 'protocol.json')
    if load(SOURCE / 'summary.json')['n'] != 60:
        raise ValueError('Expected the completed 60-claim comparison')
    source_inputs = [json.loads(line) for line in (SOURCE/'inputs.jsonl').read_text().splitlines()]
    inputs = []
    files = [Path(__file__), ROOT/'src/experiment_parts_test.py', SOURCE/'protocol.json',
             SOURCE/'inputs.jsonl', SOURCE/'reference_labels.json']
    for row in source_inputs:
        ident = str(row['id'])
        baseline_path = SOURCE / 'traces/evidence_roberta' / (ident+'.json')
        rag_path = SOURCE / 'traces/parts_rag' / (ident+'.json')
        old, rag = load(baseline_path), load(rag_path)
        if old['claim'] != row['claim'] or rag['claim'] != row['claim']:
            raise ValueError('Paired claim mismatch')
        inputs.append(dict(id=row['id'], claim=row['claim'], five_sentences=old['evidence'],
                           rag_excerpts=rag_excerpts(rag)))
        files += [baseline_path, rag_path]
    files += [p for p in CHECKPOINT.iterdir() if p.is_file()]
    out.mkdir(parents=True, exist_ok=False)
    (out/'traces').mkdir()
    write_new(out/'inputs.json', inputs)
    files.append(out/'inputs.json')
    import importlib.metadata
    protocol = dict(status='frozen_before_inference', n=len(inputs), ids=[r['id'] for r in inputs],
                    conditions=[dict(name=n, max_length=l, evidence=e) for n,l,e in CONDITIONS],
                    checkpoint=str(CHECKPOINT), device='cpu', seed=42, torch_threads=4,
                    role='Exploratory follow-up on already inspected claims; not a new held-out improvement estimate.',
                    primary_comparison='rag_256 versus five_sentence_256; equal checkpoint, claim IDs, input budget and text-only formatting.',
                    secondary='Both evidence conditions at 512 tokens; never pool this with the 256-token retrieval effect.',
                    evidence='Reuse exact saved article-derived excerpts offered to Llama, in original order. '
                             'No Llama labels, reasons, part verdicts, reference labels, titles or instructions enter RoBERTa.',
                    original_protocol_sha256=digest(SOURCE/'protocol.json'),
                    sha256={str(p.resolve()):digest(p) for p in files},
                    versions={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy']},
                    limitations=['No retraining; annotated-evidence/empty-NEI training mismatch persists.',
                                 'RAG changes passage selection and corpus date together; not an isolated vector-store effect.',
                                 'Even 512 tokens cannot necessarily contain all saved RAG passages.',
                                 'Inference time below excludes the earlier RAG fetching/embedding cost.',
                                 'All 60 already reviewed claims are exploratory; do not choose a configuration then claim fresh test accuracy.'])
    write_new(out/'protocol.json', protocol)
    print('Prepared 60 claims and four frozen input conditions; no inference yet.')


def verify(out):
    protocol = load(out/'protocol.json')
    for path, expected in protocol['sha256'].items():
        if digest(path) != expected:
            raise ValueError('Frozen source changed: '+path)
    return protocol


def predict(model, tokenizer, claim, excerpts, max_length):
    import torch
    text = evidence_text(excerpts)
    full = tokenizer(claim, text, truncation=False, return_offsets_mapping=True, verbose=False)
    limited = tokenizer(claim, text, truncation=True, max_length=max_length,
                        padding='max_length', return_offsets_mapping=True, verbose=False)
    from collections import Counter
    all_tokens = Counter((s,e,t) for (s,e),t,seq in zip(full['offset_mapping'], full['input_ids'], full.sequence_ids()) if seq==1 and e>s)
    kept_tokens = Counter((s,e,t) for (s,e),t,seq in zip(limited['offset_mapping'], limited['input_ids'], limited.sequence_ids()) if seq==1 and e>s)
    position = 0
    visibility = []
    for rank, excerpt in enumerate(excerpts,1):
        end = position + len(excerpt['text'])
        tokens = Counter({key:value for key,value in all_tokens.items() if key[0]<end and key[1]>position})
        total, kept = sum(tokens.values()), sum((tokens & kept_tokens).values())
        visibility.append(dict(rank=rank,id=excerpt.get('id'),page_id=excerpt.get('page_id'),
                               tokens_total=total,tokens_kept=kept,visibility='full' if total and kept==total else 'partial' if kept else 'none'))
        position = end+1
    encoded = {k:torch.tensor([limited[k]],dtype=torch.long) for k in ['input_ids','attention_mask']}
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(**encoded).logits[0]
        probabilities = torch.softmax(logits,-1).tolist()
    return dict(label=LABELS[int(logits.argmax())], probabilities=probabilities, logits=logits.tolist(),
                inference_seconds=time.perf_counter()-started, evidence_text=text,
                token_trace=dict(max_length=max_length,tokens_before=len(full['input_ids']),
                                 tokens_retained=sum(limited['attention_mask']),truncated=len(full['input_ids'])>max_length,
                                 evidence_tokens_total=sum(all_tokens.values()), evidence_tokens_retained=sum(kept_tokens.values()),
                                 excerpts=visibility,decoded_retained_input=tokenizer.decode(limited['input_ids'],skip_special_tokens=False)))


def run(out):
    verify(out)
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT,local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT,local_files_only=True).cpu().eval()
    if model.config.max_position_embeddings < 514:
        raise ValueError('Checkpoint does not support the planned 512-token input')
    inputs = load(out/'inputs.json')
    for name, length, field in CONDITIONS:
        folder=out/'traces'/name
        folder.mkdir(exist_ok=True)
        for row in inputs:
            target=folder/(str(row['id'])+'.json')
            if target.exists():
                continue
            result=predict(model,tokenizer,row['claim'],row[field],length)
            if name=='five_sentence_256':
                saved=load(SOURCE/'traces/evidence_roberta'/(str(row['id'])+'.json'))
                difference=max(abs(a-b) for a,b in zip(result['probabilities'],saved['probabilities']))
                if result['label']!=saved['label'] or difference>1e-5:
                    raise ValueError('Original baseline does not reproduce; investigate before interpreting changes')
                result['saved_probability_max_difference']=difference
            write_new(target,dict(id=row['id'],claim=row['claim'],condition=name,**result))
        print('SAVED',name,len(inputs),'predictions',flush=True)
    verify(out)


def score(out):
    verify(out)
    inputs=load(out/'inputs.json')
    references=load(SOURCE/'reference_labels.json')
    gold=[references[str(r['id'])] for r in inputs]
    results={}
    for name,_,_ in CONDITIONS:
        traces=[load(out/'traces'/name/(str(r['id'])+'.json')) for r in inputs]
        results[name]=dict(**metrics(gold,[r['label'] for r in traces]),
                          correct=sum(r['label']==g for r,g in zip(traces,gold)),
                          truncated_inputs=sum(r['token_trace']['truncated'] for r in traces),
                          mean_evidence_tokens_retained=sum(r['token_trace']['evidence_tokens_retained'] for r in traces)/len(inputs))
    comparisons={}
    import random
    for base,new in [('five_sentence_256','rag_256'),('five_sentence_512','rag_512'),
                     ('five_sentence_256','five_sentence_512'),('rag_256','rag_512')]:
        a=[load(out/'traces'/base/(str(r['id'])+'.json'))['label'] for r in inputs]
        b=[load(out/'traces'/new/(str(r['id'])+'.json'))['label'] for r in inputs]
        diff=[int(y==g)-int(x==g) for x,y,g in zip(a,b,gold)]
        rng=random.Random(42)
        bootstrap=sorted(sum(rng.choices(diff,k=len(diff)))/len(diff) for _ in range(5000))
        comparisons[new+'__vs__'+base]=dict(accuracy_difference=sum(diff)/len(diff),fixed=diff.count(1),
                                          introduced=diff.count(-1),paired_95_interval=[bootstrap[124],bootstrap[4874]])
    summary=dict(n=len(inputs),status='complete_exploratory',conditions=results,comparisons=comparisons,
                 limitations=load(out/'protocol.json')['limitations'])
    write_new(out/'summary.json',summary)
    with (out/'comparison.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['condition','correct','n','accuracy','macro_f1','truncated_inputs'])
        for name,m in results.items():writer.writerow([name,m['correct'],m['n'],m['accuracy'],m['macro_f1'],m['truncated_inputs']])
    lines=['# Exploratory RoBERTa with saved RAG evidence','','Same frozen checkpoint, same 60 already inspected claims. No retraining or new retrieval.','',
           '| Evidence | Token limit | Correct | Accuracy | Macro-F1 | Inputs truncated |','|---|---:|---:|---:|---:|---:|']
    for name,length,_ in CONDITIONS:
        m=results[name];lines.append('| %s | %d | %d/60 | %.2f%% | %.4f | %d |'%(name,length,m['correct'],100*m['accuracy'],m['macro_f1'],m['truncated_inputs']))
    lines+=['','At 256 tokens, the retrieval-evidence change is compared under an equal input budget. The two 512-token conditions separately explore input length. These changes do not repair the old training-data mismatch.','',
            'The original 256-token baseline labels and probabilities were replayed and checked against the saved run (tolerance 1e-5). All supplied RAG passages are used in order; model reasoning, selected citations and gold labels are excluded from the classifier input. The larger RAG text is still truncated to the stated budget.','',
            'Paired differences and intervals are in summary.json. These cases have already been inspected, so this is exploratory analysis rather than fresh held-out confirmation. No claim that the best-looking condition generalizes.','',
            'Inference-only timing excludes the earlier RAG article fetching and embedding cost. A live RAG+RoBERTa deployment would still pay those costs.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run','score'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    {'prepare':prepare,'run':run,'score':score}[args.mode](out)


if __name__=='__main__':main()
