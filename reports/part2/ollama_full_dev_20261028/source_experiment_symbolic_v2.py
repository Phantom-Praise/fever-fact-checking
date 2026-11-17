"""Frozen-prediction, snapshot-backed symbolic rule ablation; never trains models.

V2 is deliberately support-only. Non-matches are not evidence of REFUTES.
Rules are development hypotheses, not tuned using gold labels.
"""
import argparse
import csv
import hashlib
import json
import re
import time
import requests
from datetime import datetime, timezone
from pathlib import Path

from experiment_symbolic_hybrid import API, HEADERS, WikidataVerifier, normalize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def values(entity, prop):
    return [s['mainsnak']['datavalue']['value']
            for s in entity.get('claims', {}).get(prop, [])
            if s.get('rank') != 'deprecated' and not s.get('qualifiers')
            and 'datavalue' in s.get('mainsnak', {})]


class SnapshotVerifier(WikidataVerifier):
    def __init__(self, snapshot, offline=False, timeout=8):
        super().__init__(timeout)
        self.snapshot = Path(snapshot)
        self.snapshot.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.hits = 0
        self.last_request = 0

    def query(self, **params):
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        path = self.snapshot / (key + '.json')
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['params'] != params:
                raise ValueError('Snapshot query mismatch')
            self.hits += 1
            return saved['response']
        if self.offline:
            raise RuntimeError('Missing offline snapshot query: ' + str(params))
        for attempt in range(3):
            time.sleep(max(0, 1.0 - (time.monotonic() - self.last_request)))
            self.calls += 1
            try:
                response = self.session.get(API, params={**params, 'format': 'json'},
                                            headers=HEADERS, timeout=self.timeout)
                self.last_request = time.monotonic()
                if response.status_code in (429, 503) and attempt < 2:
                    self.failures += 1
                    delay = response.headers.get('Retry-After', '15')
                    time.sleep(min(30, max(1, int(delay) if delay.isdigit() else 15)))
                    continue
                response.raise_for_status()
                data = response.json()
                if not data or 'error' in data:
                    raise ValueError('Invalid Wikidata response')
                break
            except (requests.RequestException, ValueError):
                self.failures += 1
                raise RuntimeError('Wikidata request failed; aborting rather than scoring an outage')
        path.write_text(json.dumps({'params': params, 'response': data,
                                   'fetched_utc': datetime.now(timezone.utc).isoformat()},
                                  indent=2, ensure_ascii=False))
        return data

    def subject(self, name, human=False):
        data = self.query(action='wbsearchentities', search=name, language='en', limit=50)
        # A truncated candidate list cannot establish uniqueness.
        if 'search-continue' in data:
            return None
        candidates = [item['id'] for item in data.get('search', [])
                      if normalize(item.get('label', '')) == normalize(name)]
        if human:
            candidates = [qid for qid in candidates
                          if any(isinstance(v, dict) and v.get('id') == 'Q5'
                                 for v in values(self.entity(qid), 'P31'))]
        return self.entity(candidates[0]) if len(set(candidates)) == 1 else None

    def named_match(self, qid, text):
        entity = self.entity(qid)
        names = [entity.get('labels', {}).get('en', {}).get('value', '')]
        names += [v['value'] for v in entity.get('aliases', {}).get('en', [])]
        return normalize(text) in {normalize(n) for n in names if n}

    def verify(self, claim):
        claim = claim.strip().rstrip('.').strip()
        # Full clauses only: modifiers, conjunctions, and negation do not match.
        match = re.fullmatch(r'(.+?) (?:is|was) an? (actor|actress|singer|writer|politician|director|footballer|musician|scientist|artist|dancer)', claim, re.I)
        if match:
            entity = self.subject(match[1], human=True)
            if entity is None:
                return None, 'v2_ambiguous_or_unresolved_human'
            for value in values(entity, 'P106'):
                if isinstance(value, dict) and value.get('id') and self.named_match(value['id'], match[2]):
                    return 0, 'v2_P106_exact_occupation'
            return None, 'v2_occupation_not_matched'
        match = re.fullmatch(r'(.+?) is a member of (.+)', claim, re.I)
        if match:
            entity = self.subject(match[1])
            if entity is None:
                return None, 'v2_ambiguous_or_unresolved_subject'
            for value in values(entity, 'P463'):
                if isinstance(value, dict) and value.get('id') and self.named_match(value['id'], match[2]):
                    return 0, 'v2_P463_exact_membership'
            return None, 'v2_membership_not_matched'
        match = re.fullmatch(r'(.+?) (was born|died) in (?:the )?(\d{4})(s)?', claim, re.I)
        if match:
            entity = self.subject(match[1], human=True)
            if entity is None:
                return None, 'v2_ambiguous_or_unresolved_human'
            prop = 'P569' if match[2].lower() == 'was born' else 'P570'
            year, decade = int(match[3]), bool(match[4])
            if decade and year % 10:
                return None, 'v2_invalid_decade'
            dates = values(entity, prop)
            if len(dates) != 1 or not isinstance(dates[0], dict):
                return None, 'v2_missing_or_multiple_dates'
            value = dates[0]
            date = re.fullmatch(r'\+(\d+)-\d{2}-\d{2}T00:00:00Z', value.get('time', ''))
            if (date and value.get('precision', 0) >= (8 if decade else 9)
                    and value.get('before', 0) == value.get('after', 0) == 0
                    and value.get('calendarmodel', '').endswith('/Q1985727')):
                actual = int(date[1])
                if year <= actual <= year + (9 if decade else 0):
                    return 0, 'v2_' + prop + '_date_match'
            return None, 'v2_date_not_matched'
        # Avoid sending temporal expressions to place rules.
        if re.search(r'\b(?:born|died) in (?:the )?(?:\d|\w+ties\b)', claim, re.I):
            return None, 'v2_unsupported_temporal_expression'
        return super().verify(claim)


def load_inputs(retrieved, predictions):
    records = [json.loads(line) for line in Path(retrieved).read_text().splitlines() if line.strip()]
    ids = [str(r['id']) for r in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicate claim IDs')
    with open(predictions, newline='') as stream:
        saved = [r for r in csv.DictReader(stream) if r['condition'] == 'retrieved_evidence']
    by_id = {str(r['id']): r for r in saved}
    if len(by_id) != len(saved) or set(ids) != set(by_id):
        raise ValueError('Saved predictions must contain exactly the same unique IDs')
    for r in records:
        row = by_id[str(r['id'])]
        if int(r['label_id']) not in (0, 1, 2) or int(row['gold_label_id']) != int(r['label_id']):
            raise ValueError('Gold label mismatch')
        if int(row['predicted_label_id']) not in (0, 1, 2):
            raise ValueError('Invalid prediction')
    return records, by_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrieved', default='data/processed/retrieved/validation_first_100_top5_title_fast.jsonl')
    parser.add_argument('--predictions', default='reports/part2/paired/predictions.csv')
    parser.add_argument('--output', required=True, help='New directory; existing paths are refused')
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    records, baseline = load_inputs(args.retrieved, args.predictions)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    verifier = SnapshotVerifier(args.snapshot, args.offline)
    rows = []
    for condition in ('v1', 'v2'):
        for r in records:
            start = time.perf_counter()
            pred, reason = (WikidataVerifier.verify(verifier, r['claim']) if condition == 'v1'
                            else verifier.verify(r['claim']))
            rows.append({'condition': condition, 'id': r['id'], 'claim': r['claim'],
                         'gold_label_id': r['label_id'],
                         'baseline_prediction': int(baseline[str(r['id'])]['predicted_label_id']),
                         'symbolic_prediction': '' if pred is None else pred,
                         'hybrid_prediction': int(baseline[str(r['id'])]['predicted_label_id']) if pred is None else pred,
                         'reason': reason, 'symbolic_seconds': time.perf_counter() - start})
        print(condition, 'resolved', sum(r['symbolic_prediction'] != '' for r in rows if r['condition'] == condition), flush=True)
    with (out / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'inputs': {p: digest(p) for p in (args.retrieved, args.predictions, __file__, 'src/experiment_symbolic_hybrid.py')},
                'snapshot': str(Path(args.snapshot).resolve()), 'offline': args.offline,
                'http_calls': verifier.calls, 'http_failures': verifier.failures,
                'snapshot_hits': verifier.hits,
                'snapshot_files': {p.name: digest(p) for p in sorted(Path(args.snapshot).glob('*.json'))},
                'notes': ['Development rules informed by this sample; not confirmatory testing.',
                          'V1 and V2 share a query snapshot; V2 additionally uses human type constraints.',
                          'Timing mixes network and cache; cannot establish latency savings.',
                          'Only exact direct occupations, no subclass expansion.',
                          'Qualified/deprecated statements excluded in new V2 rules.',
                          'Present-day Wikidata can disagree with historical FEVER ground truth.']}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
