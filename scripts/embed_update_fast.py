#!/usr/bin/env python3
"""
Batch‑update embeddings for all invoices (fast version).
Processes up to 100 texts per Ollama call.
"""

import requests, psycopg2, time

OLLAMA_EMBED = "http://127.0.0.1:11434/api/embed"
MODEL = "mxbai-embed-large"          # or "nomic-embed-text" for even faster
BATCH_SIZE = 100

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "user": "ocr", "password": "***REMOVED***", "dbname": "invoices"
}

def get_embeddings(texts):
    """Return a list of vectors for the given texts."""
    resp = requests.post(OLLAMA_EMBED, json={
        "model": MODEL,
        "input": texts
    })
    resp.raise_for_status()
    return resp.json()["embeddings"]   # list of lists

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Select rows that need embedding – using structured fields as before
    cur.execute("""
        SELECT id, invoice_number, date, vendor_name, total_amount, currency
        FROM invoices
        WHERE embedding IS NULL
    """)
    rows = cur.fetchall()
    total = len(rows)
    print(f"Found {total} rows to embed.")

    # Build the text for each row (same field concatenation as before)
    row_texts = []
    for row in rows:
        parts = [str(p) for p in row[1:6] if p]   # inv_num, date, vendor, amount, currency
        text = " ".join(parts) if parts else ""
        row_texts.append((row[0], text))   # keep id

    # Process in batches
    for i in range(0, total, BATCH_SIZE):
        batch = row_texts[i:i+BATCH_SIZE]
        ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]

        # Retry logic in case of temporary Ollama issues
        for attempt in range(3):
            try:
                embeddings = get_embeddings(texts)
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                print(f"  Retrying batch {i//BATCH_SIZE+1} ({e})...")
                time.sleep(2)

        # Update the table
        for row_id, vec in zip(ids, embeddings):
            cur.execute(
                "UPDATE invoices SET embedding = %s::vector WHERE id = %s",
                (str(vec), row_id)
            )
        conn.commit()
        print(f"  Batch {i//BATCH_SIZE+1}: {min(i+BATCH_SIZE, total)}/{total}")

    cur.close()
    conn.close()
    print("Embeddings updated.")

if __name__ == "__main__":
    main()