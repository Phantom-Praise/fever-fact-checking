"""Lightweight evidence ranking and calibrated decision fusion.

Inspired by evidence selection in FEVER/GEAR/KGAT, not a reproduction of their
trained graph models. Original retrieval/checkpoint behavior stays available.
"""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from retrieve_title_aware import STOPWORDS, find_titles, tokenize

PREDICATE = re.compile(r"\s+(?:is|are|was|were|has|have|had|was born|died|won|worked|played|wrote|released|contains|includes|became|makes|made)\b", re.I)


def display_title(page_id):
    result = page_id.replace('_', ' ')
    for code, value in [('-LRB-', '('), ('-RRB-', ')'), ('-COLON-', ':'), ('-LSB-', '['), ('-RSB-', ']')]:
        result = result.replace(code, value)
    return result


def subject_prefix(claim):
    match = PREDICATE.search(claim)
    if not match:
        return None
    name = claim[:match.start()].strip()
    if name.endswith("'s"):
        name = name[:-2]
    return name if 2 <= len(name) <= 100 and len(name.split()) <= 12 else None


def retrieve_bm25(connection, claim, top_k=5, link_subject=False):
    titles = find_titles(connection, claim, limit=20 if link_subject else 10)
    anchors = []
    subject = subject_prefix(claim) if link_subject else None
    if subject:
        page_name = subject.replace(' ', '_')
        candidates = [page_name, page_name.replace('(', '-LRB-').replace(')', '-RRB-').replace(':', '-COLON-')]
        placeholders = ','.join('?' for _ in candidates)
        anchors = [r[0] for r in connection.execute('SELECT page_id FROM wiki_titles WHERE page_id IN (' + placeholders + ') ORDER BY page_id', candidates)]
    titles = list(dict.fromkeys(anchors + titles))[:20 if link_subject else 10]
    if not titles:
        return []
    rows = connection.execute('SELECT page_id,sentence_id,text FROM page_sentences WHERE page_id IN (' + ','.join('?' for _ in titles) + ')', titles).fetchall()
    if not rows:
        return []
    stopwords = STOPWORDS - {'not'}
    def terms(text):
        return [word for word in tokenize(text) if word not in stopwords and len(word) > 1]
    corpus = [terms(row[2]) for row in rows]
    # BM25 requires at least one token overall; punctuation-only pages safely score zero.
    scores = BM25Okapi(corpus).get_scores(terms(claim)) if any(corpus) else np.zeros(len(rows))
    ranks = {title: i for i, title in enumerate(titles)}
    ranked = []
    for (page, sentence, text), score in zip(rows, scores):
        value = float(score) + (2.0 if page in anchors else 0.0) + .05 / (1 + ranks[page])
        ranked.append({'page_id': page, 'sentence_id': int(sentence), 'text': text,
                       'score': value, 'subject_page_match': page in anchors})
    ranked.sort(key=lambda r: (-r['score'], ranks[r['page_id']], r['sentence_id']))
    return ranked[:top_k]


def evidence_text(rows, include_titles=False):
    return ' '.join((display_title(r['page_id']) + ': ' if include_titles else '') + r['text'] for r in rows)


def fuse_probabilities(claim_probabilities, evidence_probabilities, claim_weight, nei_bias):
    claim = np.asarray(claim_probabilities, dtype=float)
    evidence = np.asarray(evidence_probabilities, dtype=float)
    if claim.shape != evidence.shape or claim.ndim != 2 or claim.shape[1] != 3:
        raise ValueError('Expected paired N by 3 probability arrays')
    if not 0 <= claim_weight <= 1 or not np.isfinite(nei_bias):
        raise ValueError('Invalid fusion parameters')
    if (not np.isfinite(claim).all() or not np.isfinite(evidence).all()
            or (claim < 0).any() or (evidence < 0).any()
            or not np.allclose(claim.sum(1), 1) or not np.allclose(evidence.sum(1), 1)):
        raise ValueError('Inputs must be normalized nonnegative finite probabilities')
    scores = claim_weight * np.log(np.maximum(claim, 1e-12)) + (1 - claim_weight) * np.log(np.maximum(evidence, 1e-12))
    scores[:, 2] += nei_bias
    return scores.argmax(axis=1)
