"""Read-only local browser for every claim in the official FEVER evaluation.

Run: .venv/bin/python src/evidence_explorer.py
No training, new predictions, external services, or experiment-file changes.
"""
import argparse
import csv
from collections import Counter
from functools import lru_cache
import hashlib
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import re
from pathlib import Path
import sqlite3
import threading
from urllib.parse import parse_qs, quote, urlparse

from evaluate_retrieval import gold_evidence_groups
from retrieve_title_aware import STOPWORDS, find_titles, retrieve, tokenize

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'web/evidence_explorer'
LABELS = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
MODELS = ['claim_only_H2', 'retrieved_roberta', 'calibrated_ensemble']
DIAGNOSTICS = ['correct_complete', 'correct_incomplete', 'wrong_complete', 'wrong_incomplete', 'correct_nei', 'wrong_nei']


def binary_outcome(actual, predicted, positive):
    return ('T' if actual == predicted == positive or (actual != positive and predicted != positive) else 'F') + ('P' if predicted == positive else 'N')


def diagnostic_bucket(actual, predicted, quality):
    return ('correct_' if actual == predicted else 'wrong_') + ('nei' if actual == LABELS[2] else 'complete' if quality == 'complete' else 'incomplete')


def confusion_summary(records, model):
    matrix = [[0] * 3 for _ in LABELS]
    buckets = Counter()
    strict = 0
    for r in records:
        predicted = r['predictions'][model]
        matrix[LABELS.index(r['label'])][LABELS.index(predicted)] += 1
        buckets[diagnostic_bucket(r['label'], predicted, r['quality'])] += 1
        if predicted == r['label'] and (r['label'] == LABELS[2] or model != 'claim_only_H2' and r['quality'] == 'complete'):
            strict += 1
    n = len(records)
    per_class = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]; fn = sum(matrix[i]) - tp
        fp = sum(row[i] for row in matrix) - tp; tn = n - tp - fp - fn
        per_class[label] = {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'precision': tp / (tp + fp) if tp + fp else 0,
            'recall': tp / (tp + fn) if tp + fn else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}
    return {'model': model, 'labels': LABELS, 'n': n, 'matrix': matrix, 'per_class': per_class,
            'accuracy': sum(matrix[i][i] for i in range(3)) / n,
            'macro_f1': sum(v['f1'] for v in per_class.values()) / 3,
            'strict_fever_score': strict / n,
            'diagnostics': {key: buckets[key] for key in DIAGNOSTICS}}


def title_text(page):
    for token, value in [('-LRB-', '('), ('-RRB-', ')'), ('-LSB-', '['), ('-RSB-', ']'), ('-COLON-', ':')]:
        page = page.replace(token, value)
    return page.replace('_', ' ')


def page_info(page):
    title = title_text(page)
    return {'page_id': page, 'title': title, 'url': 'https://en.wikipedia.org/wiki/' + quote(title.replace(' ', '_'), safe='')}


def unique_groups(record):
    return sorted({tuple(sorted(group)) for group in gold_evidence_groups(record)})


def evidence_quality(record, references):
    if record['label'] == 'NOT ENOUGH INFO':
        return 'nei'
    found = set(map(tuple, references))
    groups = unique_groups(record)
    if any(set(group) <= found for group in groups):
        return 'complete'
    return 'partial' if any(found & set(group) for group in groups) else 'miss'


class Explorer:
    def __init__(self, data, outputs, db, protocol):
        self.db = Path(db)
        gold = [json.loads(line) for line in Path(data).read_text().splitlines() if line.strip()]
        saved = [json.loads(line) for line in Path(outputs).read_text().splitlines() if line.strip()]
        if [r['id'] for r in gold] != [r['id'] for r in saved]:
            raise ValueError('Saved outputs and gold claims must have identical IDs and order')
        if any(r['claim'] != p['claim'] for r, p in zip(gold, saved)):
            raise ValueError('Claim text mismatch')
        self.gold = {str(r['id']): r for r in gold}
        self.saved = {str(r['id']): r for r in saved}
        self.listing = []
        for r, p in zip(gold, saved):
            prediction = LABELS[p['predictions']['calibrated_ensemble']]
            self.listing.append({'id': r['id'], 'claim': r['claim'], 'label': r['label'],
                'prediction': prediction, 'correct': prediction == r['label'],
                'predictions': {name: LABELS[p['predictions'][name]] for name in MODELS},
                'quality': evidence_quality(r, p['predicted_evidence'])})
        self.counts = Counter(r['quality'] for r in self.listing)
        protocol_data = json.loads(Path(protocol).read_text())
        self.fusion_config = protocol_data['frozen_calibration_config']
        self.max_length = protocol_data['max_length']
        self.analysis = {model: confusion_summary(self.listing, model) for model in MODELS}
        self.tokenizer = None
        self.tokenizer_lock = threading.Lock()
        digest = hashlib.sha256((ROOT / 'src/retrieve_title_aware.py').read_bytes()).hexdigest()
        self.source_matches = digest == protocol_data['input_sha256']['src/retrieve_title_aware.py']

    def connect(self):
        return sqlite3.connect(self.db.resolve().as_uri() + '?mode=ro', uri=True)

    def metadata(self, model='calibrated_ensemble'):
        if model not in MODELS: raise ValueError('Unknown model')
        eligible = len(self.listing) - self.counts['nei']
        return {'total': len(self.listing), 'verifiable': eligible, 'counts': dict(self.counts),
            'complete_recall_at5': self.counts['complete'] / eligible if eligible else None,
            'any_recall_at5': (self.counts['complete'] + self.counts['partial']) / eligible if eligible else None,
            'accuracy': self.analysis[model]['accuracy'], 'analysis': self.analysis[model],
            'retriever_source_matches_saved_run': self.source_matches,
            'source': 'Full official FEVER development evaluation', 'snapshot': 'Local FEVER Wikipedia snapshot'}

    def claims(self, query='', quality='all', correctness='all', label='all', page=1, limit=40,
               model='calibrated_ensemble', prediction='all', outcome='all', positive='SUPPORTS', diagnostic='all'):
        if quality not in ('all', 'complete', 'partial', 'miss', 'nei'):
            raise ValueError('Unknown retrieval filter')
        if correctness not in ('all', 'correct', 'wrong') or label not in ['all'] + LABELS:
            raise ValueError('Unknown label filter')
        if page < 1 or not 1 <= limit <= 100:
            raise ValueError('Invalid page')
        if model not in MODELS or prediction not in ['all'] + LABELS or positive not in LABELS:
            raise ValueError('Unknown model or prediction')
        if outcome not in ('all', 'TP', 'FP', 'TN', 'FN') or diagnostic not in ['all'] + DIAGNOSTICS:
            raise ValueError('Unknown outcome')
        query = query.casefold().strip()
        rows = [r for r in self.listing if (not query or query in r['claim'].casefold() or query == str(r['id']))
                and (quality == 'all' or r['quality'] == quality)
                and (correctness == 'all' or (r['predictions'][model] == r['label']) == (correctness == 'correct'))
                and (label == 'all' or r['label'] == label)
                and (prediction == 'all' or r['predictions'][model] == prediction)
                and (outcome == 'all' or binary_outcome(r['label'], r['predictions'][model], positive) == outcome)
                and (diagnostic == 'all' or diagnostic_bucket(r['label'], r['predictions'][model], r['quality']) == diagnostic)]
        start = (page - 1) * limit
        items = [{**r, 'prediction': r['predictions'][model], 'correct': r['label'] == r['predictions'][model]} for r in rows[start:start + limit]]
        return {'total': len(rows), 'page': page, 'limit': limit, 'items': items}

    def token_visibility(self, claim, selected):
        """Inspect actual tokenizer truncation, without loading neural model weights."""
        text = ' '.join(row['text'] for row in selected)
        try:
            with self.tokenizer_lock:
                if self.tokenizer is None:
                    from transformers import AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(ROOT / 'checkpoints/roberta_evidence/epoch_1', local_files_only=True)
                full = self.tokenizer(claim, text, return_offsets_mapping=True, truncation=False, verbose=False)
                encoded = self.tokenizer(claim, text, return_offsets_mapping=True, truncation=True,
                                         max_length=self.max_length, padding='max_length')
                full_seq, kept_seq = full.sequence_ids(), encoded.sequence_ids()
            kept = Counter((s, e, token) for (s, e), token, seq in zip(encoded['offset_mapping'], encoded['input_ids'], kept_seq) if seq == 1 and e > s)
            evidence_tokens = [(s, e, token) for (s, e), token, seq in zip(full['offset_mapping'], full['input_ids'], full_seq) if seq == 1 and e > s]
            rows = []; position = 0
            for row in selected:
                end = position + len(row['text'])
                tokens = Counter(t for t in evidence_tokens if t[0] < end and t[1] > position)
                retained = sum((tokens & kept).values()); total = sum(tokens.values())
                rows.append({'page_id': row['page_id'], 'sentence_id': row['sentence_id'], 'rank': row['rank'],
                    'tokens_total': total, 'tokens_kept': retained,
                    'visibility': 'full' if tokens and retained == total else 'partial' if retained else 'none'})
                position = end + 1
            return {'available': True, 'max_length': self.max_length, 'tokens_before': len(full['input_ids']),
                    'tokens_after': sum(encoded['attention_mask']), 'truncated': len(full['input_ids']) > self.max_length,
                    'claim_tokens_before': full_seq.count(0), 'claim_tokens_after': kept_seq.count(0),
                    'claim_truncated': full_seq.count(0) > kept_seq.count(0),
                    'sentences': rows}
        except (OSError, ImportError, ValueError, RuntimeError) as exc:
            return {'available': False, 'reason': type(exc).__name__, 'sentences': []}

    @lru_cache(maxsize=96)
    def detail(self, claim_id, model='calibrated_ensemble'):
        if model not in MODELS: raise ValueError('Unknown model')
        r = self.gold[claim_id]
        saved = self.saved[claim_id]
        conn = self.connect()
        try:
            titles = find_titles(conn, r['claim'], limit=10)
            # This is the original ranking function, with only the output cap expanded.
            ranked = retrieve(conn, r['claim'], top_k=None)
            keywords = {w for w in tokenize(r['claim']) if w not in STOPWORDS and len(w) > 1}
            first_keywords = [w for w in tokenize(r['claim']) if w not in STOPWORDS and len(w) > 1][:6]
            title_query = ' OR '.join('"' + word + '"' for word in dict.fromkeys(first_keywords))
            title_scores = conn.execute('SELECT page_id,bm25(title_fts) FROM title_fts WHERE title_fts MATCH ? ORDER BY bm25(title_fts) LIMIT 10', (title_query,)).fetchall() if title_query else []
            if [row[0] for row in title_scores] != titles:
                raise ValueError('Title ranking changed between queries')
            positions = {page: i for i, page in enumerate(titles)}
            groups = unique_groups(r)
            gold_keys = set().union(*(set(group) for group in groups)) if groups else set()
            current_keys = [(row['page_id'], row['sentence_id']) for row in ranked[:5]]
            saved_keys = list(map(tuple, saved['predicted_evidence']))
            for rank, row in enumerate(ranked, 1):
                row.update(page_info(row['page_id']))
                row['rank'] = rank
                row['article_rank'] = positions[row['page_id']] + 1
                row['matched_terms'] = sorted(keywords & set(tokenize(row['text'])))
                row['overlap'] = len(row['matched_terms'])
                row['title_bonus'] = .1 / row['article_rank']
                row['selected'] = rank <= 5
                row['annotated'] = (row['page_id'], row['sentence_id']) in gold_keys
            articles = []
            counts = Counter(row['page_id'] for row in ranked)
            for index, (page, score) in enumerate(title_scores, 1):
                articles.append({**page_info(page), 'rank': index, 'bm25_score': score, 'sentence_count': counts[page]})
            references = []
            candidate_keys = {(row['page_id'], row['sentence_id']) for row in ranked}
            for group in groups:
                refs = []
                for page, sentence in group:
                    text = conn.execute('SELECT text FROM page_sentences WHERE page_id=? AND sentence_id=?', (page, sentence)).fetchone()
                    refs.append({**page_info(page), 'sentence_id': sentence, 'text': text[0] if text else None,
                                 'selected': (page, sentence) in current_keys, 'candidate': (page, sentence) in candidate_keys})
                references.append(refs)
            visibility = self.token_visibility(r['claim'], ranked[:5])
            exposed = {(x['page_id'], x['sentence_id']) for x in visibility['sentences'] if x['visibility'] == 'full'}
            full_gold_visible = any(set(group) <= exposed for group in groups) if visibility['available'] and groups else None
            exposure = {(x['page_id'], x['sentence_id']): x for x in visibility['sentences']}
            for row in ranked[:5]: row['token_exposure'] = exposure.get((row['page_id'], row['sentence_id']))
            quality = evidence_quality(r, current_keys)
            if not groups:
                bottleneck = 'nei'
            elif quality == 'complete':
                bottleneck = 'claim_truncation' if visibility.get('claim_truncated') else 'input_truncation' if full_gold_visible is False else 'evidence_reached_input' if full_gold_visible else 'selected_unconfirmed_input'
            elif any(set(group) <= candidate_keys for group in groups):
                bottleneck = 'sentence_ranking'
            elif any(all(ref['text'] is not None for ref in group) for group in references):
                bottleneck = 'article_search'
            else:
                bottleneck = 'index_coverage'
            prediction = LABELS[saved['predictions'][model]]
            probability = saved['model_probabilities']
            weight = self.fusion_config['claim_weight']; bias = self.fusion_config['nei_bias']
            fusion = []
            for i, label in enumerate(LABELS):
                a = weight * math.log(max(probability['claim'][i], 1e-12))
                b = (1 - weight) * math.log(max(probability['evidence'][i], 1e-12))
                offset = bias if i == 2 else 0
                fusion.append({'label': label, 'claim_contribution': a, 'evidence_contribution': b,
                               'nei_adjustment': offset, 'score': a + b + offset})
            saved_quality = evidence_quality(r, saved_keys)
            diagnostic = diagnostic_bucket(r['label'], prediction, saved_quality)
            strict = prediction == r['label'] and (r['label'] == LABELS[2] or model != 'claim_only_H2' and saved_quality == 'complete')
            return {'id': r['id'], 'claim': r['claim'], 'label': r['label'],
                'prediction': prediction, 'model': model,
                'model_predictions': {name: LABELS[saved['predictions'][name]] for name in MODELS},
                'model_probabilities': probability, 'fusion_scores': fusion, 'fusion_config': self.fusion_config,
                'fusion_reproduced': LABELS[max(range(3), key=lambda i: fusion[i]['score'])] == LABELS[saved['predictions']['calibrated_ensemble']],
                'diagnostic': diagnostic, 'strict_correct': strict,
                'outcomes': {label: binary_outcome(r['label'], prediction, label) for label in LABELS},
                'bottleneck': bottleneck, 'token_trace': visibility, 'complete_gold_visible_in_input': full_gold_visible,
                'quality': evidence_quality(r, saved['predicted_evidence']),
                'current_quality': evidence_quality(r, current_keys), 'keywords': sorted(keywords),
                'ignored_terms': sorted(set(tokenize(r['claim'])) - keywords), 'title_query': title_query,
                'articles': articles, 'sentences': ranked, 'gold_groups': references,
                'saved_top5_equal': current_keys == saved_keys, 'saved_top5': saved['predicted_evidence'],
                'candidate_count': len(ranked)}
        finally:
            conn.close()

    def article(self, page):
        if not page or len(page) > 500:
            raise ValueError('Invalid article')
        conn = self.connect()
        try:
            sentences = [{'sentence_id': row[0], 'text': row[1]} for row in conn.execute(
                'SELECT sentence_id,text FROM page_sentences WHERE page_id=? ORDER BY sentence_id', (page,))]
            if not sentences:
                raise KeyError('Article is not in the local index')
            return {**page_info(page), 'sentences': sentences}
        finally:
            conn.close()


def make_handler(explorer, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get('Host') not in ('127.0.0.1:' + str(port), 'localhost:' + str(port)):
                return self.send_json({'error': 'Local requests only'}, 403)
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            def value(key, default=''):
                return qs.get(key, [default])[0]
            try:
                if parsed.path == '/exhaustive-reader':
                    return self.send_html((ROOT/'web/evidence_explorer/exhaustive-reader.html').read_bytes())
                if parsed.path == '/api/exhaustive-reader':
                    folder=ROOT/'reports/part2/exhaustive_reader_20261029'
                    original=ROOT/'reports/part2/ephemeral_rag_pilot_20261029'
                    data={'progress':json.loads((folder/'progress.json').read_text()),'cases':[]}
                    replay={r['id']:r for r in map(json.loads,(ROOT/'reports/part2/parts_prompt_replay_20261029/results.jsonl').read_text().splitlines())}
                    for line in (original/'results.jsonl').read_text().splitlines():
                        row=json.loads(line);trace=folder/'traces'/str(row['id'])/'trace.json'
                        data['cases'].append(dict(id=row['id'],claim=row['claim'],gold=row['gold'],previous_rag=row['results']['ephemeral_full_article_rag']['label'],old_five_sentence=row['results']['old_five_sentence']['label'],parts_model=replay[row['id']]['model_label'],parts_derived=replay[row['id']]['derived_label'],trace=json.loads(trace.read_text()) if trace.exists() else None))
                    if (folder/'summary.json').exists():data['summary']=json.loads((folder/'summary.json').read_text())
                    verified=ROOT/'reports/part2/exhaustive_token_verified_20261029'
                    if (verified/'progress.json').exists():data['verified_progress']=json.loads((verified/'progress.json').read_text())
                    if (verified/'summary.json').exists():data['verified_summary']=json.loads((verified/'summary.json').read_text())
                    for case in data['cases']:
                        file=verified/'traces'/str(case['id'])/'trace.json'
                        if file.exists():case['verified']=json.loads(file.read_text())
                    return self.send_json(data)
                if parsed.path.startswith('/api/exhaustive-verified/'):
                    match=re.fullmatch(r'/api/exhaustive-verified/(\d+)/(trace\.json|batch_\d{3}_attempt_[12]\.json|final_attempt_[12]\.json)',parsed.path)
                    if not match:return self.send_json({'error':'Unknown trace'},404)
                    file=ROOT/'reports/part2/exhaustive_token_verified_20261029/traces'/match.group(1)/match.group(2)
                    if not file.exists():return self.send_json({'error':'Trace not saved yet'},404)
                    return self.send_json(json.loads(file.read_text()))
                if parsed.path.startswith('/api/exhaustive-reader/'):
                    match=re.fullmatch(r'/api/exhaustive-reader/(\d+)/(trace\.json|batch_\d{3}_attempt_[12]\.json|final_attempt_[12]\.json)',parsed.path)
                    if not match:return self.send_json({'error':'Unknown trace'},404)
                    file=ROOT/'reports/part2/exhaustive_reader_20261029/traces'/match.group(1)/match.group(2)
                    if not file.exists():return self.send_json({'error':'Trace not saved yet'},404)
                    return self.send_json(json.loads(file.read_text()))
                if parsed.path == '/fever-reference-review':
                    return self.send_html((ROOT/'reports/part2/parts_prompt_reference_review_20261029/index.html').read_bytes())
                if parsed.path == '/claim-parts-v2':
                    return self.send_html((ROOT/'reports/part2/parts_prompt_v2_20261029/index.html').read_bytes())
                if parsed.path == '/claim-parts':
                    return self.send_html((ROOT/'reports/part2/parts_prompt_replay_20261029/index.html').read_bytes())
                if parsed.path == '/ephemeral-rag':
                    folder=ROOT/'reports/part2/ephemeral_rag_pilot_20261029'
                    if (folder/'results.jsonl').exists() and (folder/'results.jsonl').stat().st_size:
                        from reporting.build_ephemeral_rag_report import main as build_pilot_report
                        build_pilot_report()
                    if (folder/'index.html').exists():return self.send_html((folder/'index.html').read_bytes())
                    progress=json.loads((folder/'progress.json').read_text()) if (folder/'progress.json').exists() else {'status':'not_started','completed':0,'total':10}
                    body=('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="10"><title>FEVER · Temporary RAG pilot</title><style>body{font:18px/1.7 system-ui;max-width:900px;margin:50px auto;padding:0 20px;color:#18364a}progress{width:100%%;accent-color:#157c75}</style><a href="/live-llama">← Paused full benchmark</a><h1>Temporary per-claim RAG pilot</h1><p>Status: %s · %d / %d paired claims completed.</p><progress value="%d" max="%d"></progress><p>Each claim gets its own full-article vector database. Articles and vectors are deleted afterward. Saved predictions, supplied/cited evidence and prompts/responses remain reviewable.</p><p>The complete comparison report appears here after all ten claims finish. The 19K evaluation remains stopped.</p></html>'%(html.escape(progress['status']),progress['completed'],progress['total'],progress['completed'],progress['total'])).encode()
                    return self.send_html(body)
                if parsed.path == '/api/ephemeral-rag':
                    folder=ROOT/'reports/part2/ephemeral_rag_pilot_20261029'
                    data={'progress':json.loads((folder/'progress.json').read_text())}
                    if (folder/'summary.json').exists():data['summary']=json.loads((folder/'summary.json').read_text())
                    return self.send_json(data)
                if parsed.path == '/api/live-llama/status':
                    from live_ollama_viewer import live_store
                    return self.send_json(live_store().status())
                if parsed.path == '/api/live-llama/claims':
                    from live_ollama_viewer import live_store
                    return self.send_json(live_store().listing(config=value('config','A_original'),query=value('q'),
                        outcome=value('outcome','all'),gold=value('gold','all'),prediction=value('prediction','all'),
                        show=value('show','completed'),order=value('order','latest'),page=int(value('page','1'))))
                if parsed.path.startswith('/api/live-llama/claim/'):
                    from live_ollama_viewer import live_store
                    return self.send_json(live_store().detail(parsed.path.rsplit('/',1)[1],value('config','A_original')))
                if parsed.path == '/api/meta':
                    return self.send_json(explorer.metadata(value('model', 'calibrated_ensemble')))
                if parsed.path == '/api/claims':
                    return self.send_json(explorer.claims(value('q'), value('quality', 'all'), value('correctness', 'all'),
                        value('label', 'all'), int(value('page', '1')), model=value('model', 'calibrated_ensemble'),
                        prediction=value('prediction', 'all'), outcome=value('outcome', 'all'),
                        positive=value('positive', 'SUPPORTS'), diagnostic=value('diagnostic', 'all')))
                if parsed.path.startswith('/api/claim/'):
                    return self.send_json(explorer.detail(parsed.path.rsplit('/', 1)[1], value('model', 'calibrated_ensemble')))
                if parsed.path == '/api/article':
                    return self.send_json(explorer.article(value('page_id')))
                if parsed.path == '/result-analysis':
                    self.send_response(302)
                    self.send_header('Location', '/result-analysis/')
                    self.end_headers()
                    return
                if parsed.path == '/result-analysis/':
                    return self.send_html((ROOT / 'reports/result_analysis_20261028/visual/index.html').read_bytes())
                visual_assets = {'/result-analysis/visual.css': ('visual.css', 'text/css'),
                                 '/result-analysis/visual.js': ('visual.js', 'text/javascript'),
                                 '/result-analysis/data.js': ('data.js', 'text/javascript'),
                                 '/result-analysis/data.json': ('data.json', 'application/json')}
                if parsed.path in visual_assets:
                    name, mime = visual_assets[parsed.path]
                    raw = (ROOT / 'reports/result_analysis_20261028/visual' / name).read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type', mime + '; charset=utf-8')
                    self.send_header('Content-Length', str(len(raw)))
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    self.end_headers(); self.wfile.write(raw)
                    return
                if parsed.path in ('/result-analysis-details', '/result-analysis.md', '/result-analysis.zip', '/result-analysis-visual.zip'):
                    folder = ROOT / 'reports/result_analysis_20261028'
                    if parsed.path == '/result-analysis-details': return self.send_html((folder / 'index.html').read_bytes())
                    name, mime = ('RESULT_ANALYSIS.md', 'text/markdown') if parsed.path.endswith('.md') else ('Result_Analysis_Visual.zip' if parsed.path == '/result-analysis-visual.zip' else 'Result_Analysis_Evidence.zip', 'application/zip')
                    raw = (folder / name).read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type', mime)
                    self.send_header('Content-Length', str(len(raw)))
                    self.send_header('Content-Disposition', 'attachment; filename="' + name + '"')
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    self.end_headers(); self.wfile.write(raw)
                    return
                if parsed.path in ('/llm-configurations', '/api/llm-configurations'):
                    folder = ROOT / 'reports/part2/llm_configurations_20261028'
                    progress = json.loads((folder / 'progress.json').read_text())
                    if parsed.path == '/api/llm-configurations':
                        payload = {'progress': progress}
                        for name in ('calibration_summary', 'selected_config', 'assessment_summary', 'assessment_uncertainty', 'assessment_component_diagnostic'):
                            if (folder / (name + '.json')).exists(): payload[name] = json.loads((folder / (name + '.json')).read_text())
                        return self.send_json(payload)
                    if (folder / 'index.html').exists(): return self.send_html((folder / 'index.html').read_bytes())
                    selected = json.loads((folder / 'selected_config.json').read_text())['config'] if (folder / 'selected_config.json').exists() else 'Not selected yet'
                    raw = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="20"><title>FEVER · Configuration experiments</title>'
                        '<style>body{font:18px/1.7 system-ui;max-width:900px;margin:45px auto;padding:0 22px;color:#183047;background:#f4f7fa}section{padding:24px;background:white;border:1px solid #dce6ed;border-radius:14px}progress{width:100%%;accent-color:#147c80}li{margin:6px 0}</style>'
                        '<a href="/llm-error-analysis">← Error analysis</a><h1>Testing configurations</h1><p>Same installed Llama 3.1 8B. Separate calibration and assessment claims. No retraining.</p><section><h2>%s</h2><p>%d / %d requests completed in this phase.</p><progress value="%d" max="%d"></progress><p>Current configuration: %s<br>Frozen selection: %s</p><ol><li>Original prompt and original retrieval.</li><li>Comparison prompt with date, negation and relationship examples.</li><li>Exact leading entity-page match plus BM25 sentence retrieval.</li><li>Both changes together.</li><li>Both changes plus a fixed claim-only fallback for NEI.</li></ol><small>Four unique LLM conditions on 60 calibration claims; fifth configuration reuses existing responses. Only the baseline and frozen selection are assessed on 120 disjoint claims. This page refreshes every 20 seconds.</small></section></html>' % (html.escape(progress['phase'].title()),progress['completed'],progress['total'],progress['completed'],progress['total'],html.escape(progress.get('config','—')),html.escape(selected))).encode()
                    return self.send_html(raw)
                if parsed.path == '/llm-error-analysis':
                    return self.send_html((ROOT / 'reports/part2/ollama_error_analysis_20261028/index.html').read_bytes())
                if parsed.path == '/api/llm-error-analysis':
                    folder = ROOT / 'reports/part2/ollama_error_analysis_20261028'
                    return self.send_json({'analysis': json.loads((folder / 'analysis.json').read_text()),
                                           'oracle': json.loads((folder / 'oracle_summary.json').read_text())})
                if parsed.path == '/api/ollama-traces':
                    from ollama_trace_viewer import trace_store
                    return self.send_json(trace_store().listing(value('q'), value('outcome', 'all'), int(value('page', '1'))))
                if parsed.path.startswith('/api/ollama-trace/'):
                    from ollama_trace_viewer import trace_store
                    return self.send_json(trace_store().detail(parsed.path.rsplit('/', 1)[1]))
                if parsed.path in ('/api/llm-comparison', '/llm-comparison'):
                    folder = ROOT / 'reports/part2/ollama_comparison_20261028'
                    progress = json.loads((folder / 'progress.json').read_text()) if (folder / 'progress.json').exists() else {'status': 'not_started', 'completed': 0, 'total': 300}
                    if parsed.path == '/api/llm-comparison':
                        payload = {'status': progress['status'], 'progress': progress}
                        if progress['status'] == 'complete': payload['summary'] = json.loads((folder / 'summary.json').read_text())
                        return self.send_json(payload)
                    if progress['status'] == 'complete' and (folder / 'index.html').exists():
                        return self.send_html((folder / 'index.html').read_bytes())
                    diagnostic = json.loads((folder / 'telemundo_response.json').read_text()) if (folder / 'telemundo_response.json').exists() else None
                    diagnostic_text = ('Telemundo was classified ' + diagnostic['label'] + ' in the separate diagnostic. It is excluded from the random comparison sample.') if diagnostic else 'The separate Telemundo diagnostic is pending.'
                    raw = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="20"><title>FEVER · Local LLM experiment</title>'
                        '<style>body{font:18px/1.7 system-ui;max-width:850px;margin:60px auto;padding:0 24px;background:#f4f7fa;color:#183047}section{background:white;border:1px solid #dae4ec;border-radius:14px;padding:24px}progress{width:100%%;accent-color:#147c80}</style>'
                        '<a href="/">← Evidence explorer</a><h1>Local LLM versus RoBERTa</h1><p>Llama 3.1 8B through Ollama · same five evidence sentences · fixed 300-claim sample.</p><section><h2>Evaluation in progress</h2>'
                        '<p>%d / 300 claims completed.</p><progress max="300" value="%d"></progress><p>%s</p>'
                        '<small>This page refreshes every 20 seconds. Final results appear after all claims finish.</small></section></html>' % (progress['completed'], progress['completed'], html.escape(diagnostic_text))).encode()
                    return self.send_html(raw)
                if parsed.path == '/api/token-budget':
                    folder = ROOT / 'reports/part2/token_budget_20261028'
                    if not (folder / 'summary.json').exists():
                        progress = json.loads((folder / 'progress.json').read_text()) if (folder / 'progress.json').exists() else {'status': 'not_started', 'completed': 0, 'total': 19998}
                        return self.send_json({'status': progress['status'], 'progress': progress})
                    summary = json.loads((folder / 'summary.json').read_text())
                    with (folder / 'changed_predictions.csv').open() as stream:
                        examples = list(csv.DictReader(stream))[:30]
                    return self.send_json({'status': 'complete', 'summary': summary, 'examples': examples})
                assets = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'), '/style.css': ('style.css', 'text/css'), '/diagnostics.css': ('diagnostics.css', 'text/css')}
                assets['/token-budget'] = ('token-budget.html', 'text/html')
                assets['/ollama-traces'] = ('ollama-traces.html', 'text/html')
                assets['/ollama-traces.js'] = ('ollama-traces.js', 'text/javascript')
                assets['/ollama-traces.css'] = ('ollama-traces.css', 'text/css')
                assets['/live-llama'] = ('live-llama.html', 'text/html')
                assets['/live-llama.js'] = ('live-llama.js', 'text/javascript')
                assets['/live-llama.css'] = ('live-llama.css', 'text/css')
                if parsed.path in assets:
                    filename, mime = assets[parsed.path]
                    raw = (ASSETS / filename).read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type', mime + '; charset=utf-8')
                    self.send_header('Content-Length', str(len(raw)))
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    self.end_headers(); self.wfile.write(raw)
                    return
                self.send_json({'error': 'Not found'}, 404)
            except (ValueError, TypeError):
                self.send_json({'error': 'Invalid request'}, 400)
            except KeyError:
                self.send_json({'error': 'Claim or article not found'}, 404)
            except (sqlite3.Error, OSError):
                self.send_json({'error': 'Could not read local evidence'}, 500)

        def send_json(self, value, status=200):
            raw = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers(); self.wfile.write(raw)

        def send_html(self, raw):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers(); self.wfile.write(raw)

        def log_message(self, *_):
            pass
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--data', default=str(ROOT / 'data/official_fever/shared_task_dev.jsonl'))
    p.add_argument('--outputs', default=str(ROOT / 'reports/part2/official_fever_dev_20261023/outputs.jsonl'))
    p.add_argument('--db', default=str(ROOT / 'data/processed/wiki_fts.db'))
    p.add_argument('--protocol', default=str(ROOT / 'reports/part2/official_fever_dev_20261023/protocol.json'))
    args = p.parse_args()
    if not 1024 <= args.port <= 65535:
        p.error('Choose a port from 1024 to 65535')
    explorer = Explorer(args.data, args.outputs, args.db, args.protocol)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(explorer, args.port))
    print(f'Evidence explorer: http://127.0.0.1:{args.port} ({len(explorer.listing):,} claims)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
