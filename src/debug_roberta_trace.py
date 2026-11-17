"""Inspect a frozen evidence classifier on one saved FEVER example.

Local CPU inference only; new diagnostic files, never overwrite benchmark results.
The individual-sentence and edited-claim probes are exploratory, not benchmark runs.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sqlite3

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--claim-id', type=int, default=111897)
    parser.add_argument('--output', required=True, help='New directory; existing directories are rejected')
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    benchmark = ROOT / 'reports/part2/official_fever_dev_20261023'
    saved = next(json.loads(line) for line in (benchmark / 'outputs.jsonl').open()
                 if json.loads(line)['id'] == args.claim_id)
    gold = next(json.loads(line) for line in (ROOT / 'data/official_fever/shared_task_dev.jsonl').open()
                if json.loads(line)['id'] == args.claim_id)
    checkpoint = ROOT / 'checkpoints/roberta_evidence/epoch_1'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).cpu().eval()
    torch.set_num_threads(4)
    conn = sqlite3.connect((ROOT / 'data/processed/wiki_fts.db').as_uri() + '?mode=ro', uri=True)
    sentences = []
    for rank, (page, sid) in enumerate(saved['predicted_evidence'], 1):
        row = conn.execute('SELECT text FROM page_sentences WHERE page_id=? AND sentence_id=?', (page, sid)).fetchone()
        if row is None:
            raise ValueError('Missing saved evidence')
        sentences.append(dict(rank=rank, page_id=page, sentence_id=sid, text=row[0]))
    conn.close()
    claim = gold['claim']
    cases = [('original_top5', claim, ' '.join(s['text'] for s in sentences))]
    cases += [('sentence_%s_only' % s['rank'], claim, s['text']) for s in sentences]
    if args.claim_id == 111897:
        cases.append(('spanish_claim_telemundo_only', claim.replace('English-language', 'Spanish-language'), sentences[2]['text']))
    traces = []
    with torch.inference_mode():
        for name, text, evidence in cases:
            encoded = tokenizer(text, text_pair=evidence, padding='max_length', truncation=True,
                                max_length=256, return_tensors='pt')
            result = model(**encoded, output_hidden_states=True)
            # RoBERTa classification head: first position -> dense -> tanh -> output.
            # Dropout is inactive in eval(). Recompute the actual head, not log(probability).
            cls = result.hidden_states[-1][:, 0, :]
            dense = model.classifier.dense(cls)
            activation = torch.tanh(dense)
            contributions = activation[0].unsqueeze(0) * model.classifier.out_proj.weight
            replay = contributions.sum(dim=1) + model.classifier.out_proj.bias
            error = float((replay - result.logits[0]).abs().max())
            if not torch.allclose(replay, result.logits[0], atol=1e-5, rtol=1e-5):
                raise AssertionError('Classifier-head replay mismatch')
            logits = result.logits[0]
            probabilities = torch.softmax(logits, dim=-1)
            mask = encoded['attention_mask'][0].bool()
            layer_summaries = []
            for index, state in enumerate(result.hidden_states):
                layer_summaries.append(dict(layer=index, kind='embeddings' if index == 0 else 'transformer',
                    shape=list(state.shape), cls_l2=float(state[0, 0].norm()),
                    retained_token_mean_l2=float(state[0, mask].norm(dim=-1).mean())))
            ids = encoded['input_ids'][0].tolist()
            traces.append(dict(name=name, claim=text, evidence=evidence,
                prediction=LABELS[int(logits.argmax())], logits=logits.tolist(), probabilities=probabilities.tolist(),
                retained_tokens=int(mask.sum()), input_ids=ids, attention_mask=encoded['attention_mask'][0].tolist(),
                token_strings=tokenizer.convert_ids_to_tokens(ids), decoded_retained_input=tokenizer.decode(ids, skip_special_tokens=False),
                layer_summaries=layer_summaries, final_cls_vector=cls[0].tolist(), dense_output=dense[0].tolist(),
                tanh_activation=activation[0].tolist(), output_bias=model.classifier.out_proj.bias.tolist(),
                output_contributions_by_hidden_dimension=contributions.tolist(), replayed_logits=replay.tolist(),
                head_replay_max_absolute_error=error))
            print(name, json.dumps(dict(zip(LABELS, probabilities.tolist()))), flush=True)
    difference = max(abs(a-b) for a,b in zip(traces[0]['probabilities'], saved['model_probabilities']['evidence']))
    data = dict(claim_id=args.claim_id, claim=claim, gold_label=gold['label'], selected_sentences=sentences,
        checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=file_digest(checkpoint/'model.safetensors'),
        torch_version=torch.__version__, device='cpu', eval_mode=True, max_length=256,
        label_order=LABELS, saved_evidence_probabilities=saved['model_probabilities']['evidence'],
        original_probability_max_absolute_difference=difference,
        interpretation='Diagnostic inference only. Probes do not change benchmark results. Layer norms and head contributions are numerical traces, not human-readable reasons or causal explanations. Single-sentence and edited-claim inputs are exploratory; gold annotations are not used to change the deployed retriever.',
        computation='token IDs -> embeddings + positions -> 12 contextual transformer layers -> first-position vector -> dense -> tanh -> output linear layer -> 3 logits -> softmax -> argmax',
        traces=traces)
    out.mkdir(parents=True, exist_ok=False)
    (out/'trace.json').write_text(json.dumps(data,indent=2))
    esc = html.escape
    parts = ['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>RoBERTa debug trace</title><style>body{font:17px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:0 20px;color:#17283a;background:#f7fafc}section{background:white;padding:22px;border:1px solid #dae3eb;border-radius:12px;margin:18px 0}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}small{color:#526477}</style>',
        '<h1>Inside the frozen RoBERTa classifier</h1><p><b>Claim:</b> '+esc(claim)+'</p><p><b>FEVER label:</b> '+esc(gold['label'])+'</p>',
        '<section><h2>What calculates the score?</h2><p>'+esc(data['computation'])+'</p><p>Tokenization only converts text to IDs. Learned neural layers calculate context-dependent vectors. The classifier head converts one vector into three raw scores (logits). Softmax turns those into numbers that sum to one. The largest wins.</p><pre>h = final_hidden_state[first_position]\na = tanh(W_dense @ h + b_dense)\nlogits = W_output @ a + b_output\np[i] = exp(logit[i] - max(logits)) / sum(exp(logits - max(logits)))</pre></section>',
        '<section><h2>The five saved sentences, in order</h2>']
    for s in sentences:
        parts.append('<h3>#%d · %s / sentence %d</h3><p>%s</p>' % (s['rank'],esc(s['page_id']),s['sentence_id'],esc(s['text'])))
    parts.append('</section><section><h2>Measured diagnostic probes</h2><small>New local inference; these are not new benchmark scores.</small><table><tr><th>Input</th><th>SUPPORTS</th><th>REFUTES</th><th>NEI</th></tr>')
    for t in traces:
        parts.append('<tr><td>'+esc(t['name'])+'</td>'+''.join('<td>%.2f%%</td>'%(p*100) for p in t['probabilities'])+'</tr>')
    parts.append('</table><p>Original-input maximum probability difference from saved run: %.9f.</p></section>' % difference)
    for t in traces:
        parts.append('<section><details><summary>'+esc(t['name'])+' · raw scores, tokens and layer summaries</summary><pre>'+esc(json.dumps({k:t[k] for k in ('claim','evidence','logits','probabilities','retained_tokens','head_replay_max_absolute_error','layer_summaries','token_strings')},indent=2))+'</pre></details></section>')
    parts.append('<p>'+esc(data['interpretation'])+'</p><p>The JSON file next to this page includes full first-position and classifier-head vectors and per-dimension output contributions.</p>')
    (out/'index.html').write_text(''.join(parts))
    print('Wrote', out)


if __name__ == '__main__':
    main()
