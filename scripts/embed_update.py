#!/usr/bin/env python3
"""Generate embeddings using mxbai-embed-large from raw_text."""

import psycopg2

from core.config import DB_CONFIG
from core.embedding import get_embedding


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Find rows where embedding is NULL
    cur.execute("SELECT id, raw_text FROM invoices WHERE embedding IS NULL")
    rows = cur.fetchall()

    for row_id, raw_text in rows:
        if not raw_text or not raw_text.strip():
            continue
        text_to_embed = raw_text.strip()[:4096]  # truncate to embedding model context
        print(f"Embedding row {row_id} ({len(text_to_embed)} chars)...")
        vec = get_embedding(text_to_embed)
        cur.execute("UPDATE invoices SET embedding = %s WHERE id = %s", (vec, row_id))

    conn.commit()
    cur.close()
    conn.close()
    print(f"Updated {len(rows)} rows.")

if __name__ == "__main__":
    main()