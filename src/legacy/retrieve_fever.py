
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



def make_query(claim, strategy="and"):
    tokens = re.findall(r"\b\w+\b", claim.lower())

    keywords = []
    for token in tokens:
        if token not in STOPWORDS and len(token) > 1:
            if token not in keywords:
                keywords.append(token)

    if not keywords:
        return None

    # Use up to five terms for flexible search.
    # Strict search keeps the original three-term behavior.
    selected = (
        keywords[:3] if strategy == "and"
        else keywords[:5]
    )

    operator = " AND " if strategy == "and" else " OR "

    return operator.join(f'"{term}"' for term in selected)



def retrieve(conn, claim, top_k=5, strategy="and"):
    def search(query, limit):
        if not query:
            return []

        rows = conn.execute(
            """
            SELECT page_id, sentence_id, text,
                   bm25(wiki) AS score
            FROM wiki
            WHERE wiki MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()

        return [
            {
                "page_id": page_id,
                "sentence_id": sentence_id,
                "text": text,
                "score": score,
            }
            for page_id, sentence_id, text, score in rows
        ]

    if strategy != "hybrid":
        return search(
            make_query(claim, strategy=strategy),
            top_k,
        )

    # First attempt: fast, restrictive search.
    results = search(
        make_query(claim, strategy="and"),
        top_k,
    )

    if len(results) >= top_k:
        return results

    # Fallback: broader search when AND gives
    # fewer than the requested number of results.
    fallback = search(
        make_query(claim, strategy="or"),
        top_k,
    )

    seen = {
        (item["page_id"], item["sentence_id"])
        for item in results
    }

    for item in fallback:
        key = (item["page_id"], item["sentence_id"])

        if key not in seen:
            results.append(item)
            seen.add(key)

        if len(results) >= top_k:
            break

    return results



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--strategy",
        choices=["and", "or", "hybrid"],
        default="and",
    )
    args = parser.parse_args()

    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / (
        f"validation_first_{args.limit}_"
        f"top{args.top_k}_{args.strategy}.jsonl"
    )

    start = time.time()
    total = 0
    empty = 0

    conn = sqlite3.connect(
        f"file:{DB_PATH.resolve()}?mode=ro",
        uri=True,
    )

    try:
        with INPUT_PATH.open(encoding="utf-8") as source, \
             output_path.open("w", encoding="utf-8") as output:

            for line in source:
                if total >= args.limit:
                    break

                example = json.loads(line)
                claim = example["claim"]

                evidence = retrieve(
                    conn,
                    claim,
                    top_k=args.top_k,
                    strategy=args.strategy,
                )

                if not evidence:
                    empty += 1

                result = {
                    "id": example.get("id"),
                    "claim": claim,
                    "label_id": example.get("label_id"),
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
    print(f"Claims with no retrieved evidence: {empty}")


if __name__ == "__main__":
    main()
