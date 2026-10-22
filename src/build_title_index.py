
import sqlite3
import time
from pathlib import Path

DB_PATH = Path("data/processed/wiki_fts.db")


def main():
    start = time.time()

    conn = sqlite3.connect(DB_PATH)

    try:
        print("Building article title table...", flush=True)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS wiki_titles (
                page_id TEXT PRIMARY KEY,
                title TEXT NOT NULL
            )
        """)

        conn.execute("""
            INSERT OR IGNORE INTO wiki_titles(page_id, title)
            SELECT page_id, REPLACE(page_id, '_', ' ')
            FROM wiki
            GROUP BY page_id
        """)

        conn.commit()

        print("Building title search index...", flush=True)

        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS title_fts
            USING fts5(
                page_id UNINDEXED,
                title,
                tokenize='unicode61'
            )
        """)

        conn.execute("DELETE FROM title_fts")

        conn.execute("""
            INSERT INTO title_fts(page_id, title)
            SELECT page_id, title FROM wiki_titles
        """)

        conn.commit()

        print("Adding page lookup index...", flush=True)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_wiki_titles_page
            ON wiki_titles(page_id)
        """)

        conn.commit()

        count = conn.execute(
            "SELECT COUNT(*) FROM wiki_titles"
        ).fetchone()[0]

        print(f"Indexed {count:,} Wikipedia titles.")
        print(f"Elapsed: {(time.time()-start)/60:.1f} minutes")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
