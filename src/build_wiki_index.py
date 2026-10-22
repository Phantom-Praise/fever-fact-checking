
import json
import sqlite3
import time
from pathlib import Path

WIKI_DIR = Path("data/raw/wiki-pages")
DB_PATH = Path("data/processed/wiki_fts.db")
BATCH_SIZE = 5000


def parse_sentences(lines):
    for line in lines.split("\n"):
        parts = line.split("\t", 2)

        if len(parts) < 2:
            continue

        try:
            sentence_id = int(parts[0])
        except ValueError:
            continue

        sentence = parts[1].strip()

        if sentence:
            yield sentence_id, sentence


def build_index():
    files = sorted(WIKI_DIR.glob("wiki-*.jsonl"))

    if not files:
        raise FileNotFoundError(
            f"No Wikipedia files found in {WIKI_DIR}"
        )

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        raise FileExistsError(
            f"{DB_PATH} already exists. "
            "Move or delete it before rebuilding."
        )

    print(f"Found {len(files)} Wikipedia files.")
    print(f"Creating index at {DB_PATH}")

    connection = sqlite3.connect(DB_PATH)

    try:
        connection.execute(
            """
            CREATE VIRTUAL TABLE wiki USING fts5(
                page_id UNINDEXED,
                sentence_id UNINDEXED,
                text,
                tokenize='unicode61'
            )
            """
        )

        total = 0
        start = time.time()
        batch = []

        connection.execute("BEGIN")

        for file_number, file_path in enumerate(files, 1):
            with file_path.open(
                "r", encoding="utf-8"
            ) as file:
                for line in file:
                    record = json.loads(line)
                    page_id = record.get("id", "")
                    lines = record.get("lines", "")

                    for sentence_id, sentence in parse_sentences(lines):
                        batch.append(
                            (page_id, sentence_id, sentence)
                        )

                        if len(batch) >= BATCH_SIZE:
                            connection.executemany(
                                """
                                INSERT INTO wiki(
                                    page_id, sentence_id, text
                                )
                                VALUES (?, ?, ?)
                                """,
                                batch,
                            )
                            total += len(batch)
                            batch.clear()
                            connection.commit()
                            connection.execute("BEGIN")

            if batch:
                connection.executemany(
                    """
                    INSERT INTO wiki(
                        page_id, sentence_id, text
                    )
                    VALUES (?, ?, ?)
                    """,
                    batch,
                )
                total += len(batch)
                batch.clear()
                connection.commit()
                connection.execute("BEGIN")

            elapsed = (time.time() - start) / 60
            print(
                f"[{file_number}/{len(files)}] "
                f"{file_path.name} | "
                f"{total:,} sentences | "
                f"{elapsed:.1f} minutes",
                flush=True,
            )

        connection.commit()

        print("\nOptimizing search index...")
        connection.execute(
            "INSERT INTO wiki(wiki) VALUES ('optimize')"
        )
        connection.commit()

        print(f"\nFinished indexing {total:,} sentences.")
        print(f"Database: {DB_PATH}")

    finally:
        connection.close()


if __name__ == "__main__":
    build_index()
