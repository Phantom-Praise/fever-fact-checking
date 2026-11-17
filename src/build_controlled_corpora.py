"""Build bounded nested, query-aware distractor indexes without modifying source.

A common core uses only claim-text title search, never labels/gold evidence.
Random additional pages are nested via one seeded sample of lexical row indices.
This isolates distractor growth, not random-corpus evidence availability.
"""
import argparse
import hashlib
import json
import random
import sqlite3
import time
from pathlib import Path

from experiment_symbolic_v2 import digest
from retrieve_title_aware import find_titles


def nested_memberships(core, additional, sizes):
    core = sorted(set(core))
    core_set = set(core)
    additional = list(dict.fromkeys(page for page in additional if page not in core_set))
    if min(sizes) < len(core) or max(sizes) > len(core) + len(additional):
        raise ValueError('Invalid nested corpus sizes')
    return {n: core + additional[:n - len(core)] for n in sorted(set(sizes))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='data/processed/wiki_fts.db')
    p.add_argument('--claims', default='data/processed/retrieved/validation_first_100_top5_title_fast.jsonl')
    p.add_argument('--sizes', type=int, nargs='+', default=[2000, 10000, 50000])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', required=True)
    p.add_argument('--manifest', required=True)
    args = p.parse_args()
    if min(args.sizes) < 1 or max(args.sizes) > 50000:
        p.error('Bounded build requires page counts between 1 and 50000')
    out = Path(args.output)
    if out.exists() or Path(args.manifest).exists():
        raise FileExistsError('Output/manifest already exists')
    records = [json.loads(line) for line in Path(args.claims).read_text().splitlines() if line.strip()]
    source = sqlite3.connect(Path(args.source).resolve().as_uri() + '?mode=ro', uri=True)
    before = Path(args.source).stat()
    start = time.perf_counter()
    core = {page for r in records for page in find_titles(source, r['claim'], 10)}
    count = source.execute('SELECT COUNT(*) FROM wiki_titles').fetchone()[0]
    if min(args.sizes) < len(core) or max(args.sizes) > count:
        raise ValueError('Page counts incompatible with source/core')
    # Uniform sample without replacement of source row indices, independent of gold.
    wanted = max(args.sizes) - len(core)
    sampled = random.Random(args.seed).sample(range(count), min(count, wanted + len(core)))
    sampled_set = set(sampled)
    picked = {}
    core_titles = {}
    for i, (page, title) in enumerate(source.execute('SELECT page_id,title FROM wiki_titles ORDER BY page_id')):
        if page in core:
            core_titles[page] = title
        if i in sampled_set and page not in core:
            picked[i] = (page, title)
    additional_rows = [picked[i] for i in sampled if i in picked][:wanted]
    if len(additional_rows) < wanted:
        raise ValueError('Insufficient sampled distractors')
    memberships = nested_memberships(core, [row[0] for row in additional_rows], args.sizes)
    title_lookup = {**core_titles, **dict(additional_rows)}
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for size, pages in memberships.items():
        path = out / f'pages_{size}.db'
        conn = sqlite3.connect(path)
        sentences = 0
        try:
            conn.execute('CREATE TABLE wiki_titles(page_id TEXT PRIMARY KEY, title TEXT NOT NULL)')
            conn.execute("CREATE VIRTUAL TABLE title_fts USING fts5(page_id UNINDEXED,title,tokenize='unicode61')")
            conn.execute('CREATE TABLE page_sentences(page_id TEXT NOT NULL,sentence_id INTEGER NOT NULL,text TEXT NOT NULL,PRIMARY KEY(page_id,sentence_id)) WITHOUT ROWID')
            conn.executemany('INSERT INTO wiki_titles VALUES (?,?)', ((page, title_lookup[page]) for page in pages))
            conn.executemany('INSERT INTO title_fts VALUES (?,?)', ((page, title_lookup[page]) for page in pages))
            for i in range(0, len(pages), 400):
                batch = pages[i:i + 400]
                rows = source.execute('SELECT page_id,sentence_id,text FROM page_sentences WHERE page_id IN (' + ','.join('?' for _ in batch) + ')', batch).fetchall()
                conn.executemany('INSERT INTO page_sentences VALUES (?,?,?)', rows)
                sentences += len(rows)
            conn.commit()
            assert conn.execute('SELECT COUNT(*) FROM wiki_titles').fetchone()[0] == size
        finally:
            conn.close()
        membership_path = out / f'pages_{size}.json'
        membership_path.write_text(json.dumps(pages))
        results.append({'path': str(path), 'pages': size, 'sentences': sentences,
                        'bytes': path.stat().st_size, 'sha256': digest(path),
                        'membership_file': str(membership_path), 'membership_sha256': digest(membership_path)})
        print('Built', size, 'pages', sentences, 'sentences', flush=True)
    source.close()
    after = Path(args.source).stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Source index changed during experiment preparation')
    manifest = {'source': args.source, 'source_pages': count, 'source_bytes': before.st_size,
                'source_mtime_ns': before.st_mtime_ns, 'claim_sha256': digest(args.claims),
                'script_sha256': digest(__file__), 'core_pages': len(core), 'core_page_ids': sorted(core),
                'seed': args.seed, 'sampling': 'Common claim-only title-search core plus nested seeded uniform distractor sample.',
                'indexes': results, 'build_seconds': time.perf_counter() - start,
                'limitations': ['Query-aware core selected using the full title index; not a representative random corpus sample.',
                                'Core uses no gold labels or gold evidence.',
                                'Growth experiment isolates added distractors and index size; evidence availability is deliberately preserved.',
                                'Index schema contains only structures used by title-aware retrieval; full source also stores unrelated indexes.']}
    target = Path(args.manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        json.dump(manifest, stream, indent=2)


if __name__ == '__main__':
    main()
