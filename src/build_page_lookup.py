
import sqlite3
import time
from pathlib import Path

DB_PATH = Path("data/processed/wiki_fts.db")


def main():
    start = time.time()
    conn = sqlite3.connect(DB_PATH)

    try:
        print("Creating page-to-sentence lookup table...", flush=True)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS page_sentences (
                page_id TEXT NOT NULL,
                sentence_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                PRIMARY KEY (page_id, sentence_id)
            ) WITHOUT ROWID
        """)

        conn.execute("""
            INSERT OR IGNORE INTO page_sentences
            SELECT page_id, sentence_id, text
            FROM wiki
        """)

        conn.commit()

        count = conn.execute(
            "SELECT COUNT(*) FROM page_sentences"
        ).fetchone()[0]

        print(f"Indexed {count:,} page sentences.")
        print(
            f"Elapsed: {(time.time() - start) / 60:.1f} minutes"
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
