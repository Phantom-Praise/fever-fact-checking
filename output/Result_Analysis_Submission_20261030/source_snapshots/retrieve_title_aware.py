
import argparse
import json
import re
import sqlite3
import time
from pathlib import Path

DB_PATH = Path("data/processed/wiki_fts.db")
INPUT_PATH = Path("data/processed/validation.jsonl")
OUTPUT_DIR = Path("data/processed/retrieved")

STOPWORDS = {
    "a", "an", "the", "is", "was", "were", "are",
    "of", "in", "on", "at", "to", "for", "and",
    "or", "by", "with", "from", "as", "that",
    "this", "it", "its", "has", "have", "had",
    "be", "been", "being", "not"
}


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def find_titles(conn, claim, limit=10):
    keywords = [
        word for word in tokenize(claim)
        if word not in STOPWORDS and len(word) > 1
    ]

    if not keywords:
        return []

    # Prefer more specific title matches.
    query = " OR ".join(
        f'"{word}"' for word in dict.fromkeys(keywords[:6])
    )

    rows = conn.execute(
        """
        SELECT page_id, title, bm25(title_fts) AS score
        FROM title_fts
        WHERE title_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()

    return [row[0] for row in rows]


def retrieve(conn, claim, top_k=5):
    titles = find_titles(conn, claim, limit=10)

    if not titles:
        return []

    # Retrieve sentences directly from candidate pages.
    # No full-corpus sentence FTS query.
    placeholders = ",".join("?" for _ in titles)

    rows = conn.execute(
        f"""
        SELECT page_id, sentence_id, text
        FROM page_sentences
        WHERE page_id IN ({placeholders})
        """,
        titles,
    ).fetchall()

    if not rows:
        return []

    # Rank candidate sentences locally using claim terms.
    keywords = {
        word for word in tokenize(claim)
        if word not in STOPWORDS and len(word) > 1
    }

    title_rank = {
        page_id: rank
        for rank, page_id in enumerate(titles)
    }

    scored = []

    for page_id, sentence_id, text in rows:
        sentence_words = set(tokenize(text))

        overlap = len(keywords & sentence_words)

        # Favor sentences matching more claim terms.
        # Small title bonus breaks close ties.
        score = (
            overlap
            + 0.1 / (1 + title_rank[page_id])
        )

        scored.append(
            (
                score,
                page_id,
                sentence_id,
                text,
            )
        )

    scored.sort(
        key=lambda item: (
            -item[0],
            title_rank[item[1]],
            int(item[2]),
        )
    )

    return [
        {
            "page_id": page_id,
            "sentence_id": sentence_id,
            "text": text,
            "score": float(score),
        }
        for score, page_id, sentence_id, text
        in scored[:top_k]
    ]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / (
        f"validation_first_{args.limit}_"
        f"top{args.top_k}_title_fast.jsonl"
    )

    conn = sqlite3.connect(
        f"file:{DB_PATH.resolve()}?mode=ro",
        uri=True,
    )

    total = 0
    empty = 0
    start = time.time()

    try:
        with INPUT_PATH.open(encoding="utf-8") as source, \
             output_path.open("w", encoding="utf-8") as output:

            for line in source:
                if total >= args.limit:
                    break

                example = json.loads(line)
                evidence = retrieve(
                    conn, example["claim"], args.top_k
                )

                if not evidence:
                    empty += 1

                result = {
                    "id": example["id"],
                    "claim": example["claim"],
                    "label_id": example["label_id"],
                    "retrieved_evidence": evidence,
                    "evidence_text": " ".join(
                        item["text"] for item in evidence
                    ),
                }

                output.write(
                    json.dumps(result, ensure_ascii=False)
                    + "\n"
                )

                total += 1

                if total % 10 == 0:
                    print(
                        f"Processed {total}/{args.limit} | "
                        f"Empty: {empty} | "
                        f"Elapsed: {time.time()-start:.1f}s",
                        flush=True,
                    )

    finally:
        conn.close()

    print(f"\nSaved: {output_path}")
    print(f"Claims processed: {total}")
    print(f"Empty searches: {empty}")


if __name__ == "__main__":
    main()
