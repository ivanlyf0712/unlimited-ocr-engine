#!/usr/bin/env python3
"""
Batch-update embeddings for invoices or messages (fast version).
Processes up to 100 texts per Ollama call.

Usage:
    python3 scripts/embed_update_fast.py --table invoices
    python3 scripts/embed_update_fast.py --table messages
"""
import sys, os, time
import psycopg2
import requests

# Allow imports from the project root (for core.*)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.config import DB_CONFIG, OLLAMA_URL

MODEL = "mxbai-embed-large"
BATCH_SIZE = 100


def get_embeddings(texts):
    """Return a list of vectors for the given texts."""
    resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
        "model": MODEL,
        "input": texts
    })
    resp.raise_for_status()
    return resp.json()["embeddings"]


def embed_invoices(cur):
    """Generate embeddings for invoice rows that have raw_text but no embedding."""
    cur.execute("""
        SELECT id, raw_text
        FROM invoices
        WHERE embedding IS NULL
    """)
    rows = cur.fetchall()
    return [(r[0], (r[1] or "").strip()[:4096]) for r in rows], "invoices"


def embed_messages(cur):
    """Generate embeddings for message rows that have no embedding.

    Builds a rich text string from the message + contact info:
      "Customer John Smith (ABC Corp) says: ... [label: normal]"
      "Agent Zhang San (XYZ Ltd) says: ... [label: scam_crypto]"
    """
    # Fetch messages needing embedding
    cur.execute("""
        SELECT id, external_userid, servicer_userid, origin, content, label
        FROM messages
        WHERE embedding IS NULL
    """)
    msg_rows = cur.fetchall()

    if not msg_rows:
        return [], "messages"

    # Fetch all contacts in one query for fast lookup
    cur.execute("SELECT userid, full_name, company FROM contacts")
    contacts = {}
    for r in cur.fetchall():
        contacts[r[0]] = (r[1] or r[0], r[2] or "")

    row_texts = []
    for row in msg_rows:
        msg_id, ext_uid, srv_uid, origin, content, label = row

        if origin == 3:
            role = "Customer"
            userid = ext_uid
        else:
            role = "Agent"
            userid = srv_uid or ext_uid

        name, company = contacts.get(userid, (userid or "Unknown", ""))
        company_str = f" ({company})" if company else ""

        text = f"{role} {name}{company_str} says: {content or ''}"
        if label:
            text += f" [label: {label}]"

        row_texts.append((msg_id, text[:4096]))

    return row_texts, "messages"


def main():
    if "--table" not in sys.argv:
        print("Usage: python3 embed_update_fast.py --table <invoices|messages>")
        sys.exit(1)

    table = sys.argv[sys.argv.index("--table") + 1]
    if table not in ("invoices", "messages"):
        print(f"Unknown table: {table}. Use 'invoices' or 'messages'.")
        sys.exit(1)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    if table == "invoices":
        row_texts, name = embed_invoices(cur)
    else:
        row_texts, name = embed_messages(cur)

    total = len(row_texts)
    print(f"Found {total} {name} rows to embed.")

    if total == 0:
        cur.close()
        conn.close()
        print("Nothing to do.")
        return

    for i in range(0, total, BATCH_SIZE):
        batch = row_texts[i:i + BATCH_SIZE]
        ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]

        for attempt in range(3):
            try:
                embeddings = get_embeddings(texts)
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                print(f"  Retrying batch {i // BATCH_SIZE + 1} ({e})...")
                time.sleep(2)

        for row_id, vec in zip(ids, embeddings):
            cur.execute(
                f"UPDATE {name} SET embedding = %s::vector WHERE id = %s",
                (str(vec), row_id)
            )
        conn.commit()
        print(f"  Batch {i // BATCH_SIZE + 1}: {min(i + BATCH_SIZE, total)}/{total}")

    cur.close()
    conn.close()
    print("Embeddings updated.")


if __name__ == "__main__":
    main()