"""Evaluate recall and cached-routing pipeline timing on nested corpus indexes."""
import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from experiment_evidence_topk import retrieval_metrics
from experiment_symbolic_v2 import digest
from retrieve_title_aware import retrieve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--indexes', required=True)
    p.add_argument('--full-db', default='data/processed/wiki_fts.db')
    p.add_argument('--claims', default='data/processed/retrieved/validation_first_100_top5_title_fast.jsonl')
    p.add_argument('--symbolic', default='reports/part2/symbolic_v2/run_20261023_retry/predictions.csv')
    p.add_argument('--gold', default='data/processed/validation.jsonl')
    p.add_argument('--recall-only', action='store_true', help='Recompute recall without repeating valid timing trials')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    indexes = json.loads(Path(args.indexes).read_text())
    if digest(args.claims) != indexes['claim_sha256']:
        raise ValueError('Corpus core and evaluation claims differ')
    for item in indexes['indexes']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Nested corpus hash mismatch')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    dbs = [item['path'] for item in indexes['indexes']] + [args.full_db]
    originals = [json.loads(line) for line in Path(args.claims).read_text().splitlines() if line.strip()]
    with open(args.gold) as stream:
        gold_by_id = {str(r['id']): r for r in map(json.loads, stream)}
    for record in originals:
        gold = gold_by_id[str(record['id'])]
        if gold['claim'] != record['claim'] or int(gold['label_id']) != int(record['label_id']):
            raise ValueError('Corpus-growth gold claim/label mismatch')
        record['evidence'] = gold['evidence']
    recalls = []
    for db in dbs:
        conn = sqlite3.connect(Path(db).resolve().as_uri() + '?mode=ro', uri=True)
        records = []
        try:
            pages = conn.execute('SELECT COUNT(*) FROM wiki_titles').fetchone()[0]
            sentences = conn.execute('SELECT COUNT(*) FROM page_sentences').fetchone()[0]
            for r in originals:
                records.append({**r, 'retrieved_evidence': retrieve(conn, r['claim'], 5)})
        finally:
            conn.close()
        recalls.append({'db': db, 'pages': pages, 'sentences': sentences,
                        'recall': [retrieval_metrics(records, k) for k in (1, 5)]})
        print('Corpus', pages, 'pages; recall computed', flush=True)
    (out / 'recall.json').write_text(json.dumps(recalls, indent=2))
    command = [sys.executable, str(Path(__file__).with_name('benchmark_hierarchy.py')),
               '--retrieved', args.claims, '--symbolic', args.symbolic,
               '--repeats', str(args.repeats), '--output', str(out / 'timing')]
    for db in dbs:
        command.extend(['--db', db])
    if not args.recall_only:
        subprocess.run(command, check=True)
    (out / 'manifest.json').write_text(json.dumps({
        'input_sha256': {p: digest(p) for p in (args.indexes, args.claims, args.gold, args.symbolic, __file__, str(Path(__file__).with_name('benchmark_hierarchy.py')))},
        'command': None if args.recall_only else command, 'recall_only': args.recall_only,
        'limitations': indexes['limitations'] + ['Two inference repeats are not independent training repeats.',
                                                'Cached symbolic routing excludes live KB costs; zero coverage cannot save model/retrieval calls.',
                                                'Fixed growth order and warm caches limit latency interpretation.']}, indent=2))


if __name__ == '__main__':
    main()
