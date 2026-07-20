#!/usr/bin/env python3
"""Generate embeddings using mxbai-embed-large from structured fields only."""

import requests
import psycopg2

OLLAMA_EMBED = "http://127.0.0.1:11434/api/embed"
EMBED_MODEL = "mxbai-embed-large"

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "user": "ocr", "password": "***REMOVED***", "dbname": "invoices"
}

def get_embedding(text: str) -> list[float]:
    resp = requests.post(OLLAMA_EMBED, json={"model": EMBED_MODEL, "input": text})
    resp.raise_for_status()
    return resp.json()["embeddings"][0]

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Find rows where embedding is NULL
    cur.execute("SELECT id, invoice_number, date, vendor_name, total_amount, currency FROM invoices WHERE embedding IS NULL")
    rows = cur.fetchall()

    for row_id, inv_num, date, vendor, amount, currency in rows:
        # Build a clean string from the five fields
        parts = [str(p) for p in [inv_num, date, vendor, amount, currency] if p]
        if not parts:
            continue
        text_to_embed = " ".join(parts)
        print(f"Embedding row {row_id}: {text_to_embed}")
        vec = get_embedding(text_to_embed)
        cur.execute("UPDATE invoices SET embedding = %s WHERE id = %s", (vec, row_id))

    conn.commit()
    cur.close()
    conn.close()
    print(f"Updated {len(rows)} rows.")

if __name__ == "__main__":
    main()