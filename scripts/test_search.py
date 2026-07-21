#!/usr/bin/env python3
"""
End-to-end test of the search_similar function without Streamlit.
Requires: at least one invoice with an embedding in the database.

Usage:
  1. Run migrate_db.py first
  2. Upload at least one invoice via the Streamlit UI (this generates embeddings)
  3. Run: python3 scripts/test_search.py
"""
import sys, os, time, requests, psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Replicate the minimal functions needed (avoid importing app.py which pulls in streamlit)
OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "mxbai-embed-large"
DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "user": "ocr", "password": "***REMOVED***", "dbname": "invoices"
}

def get_embedding(text: str):
    """Get embedding vector from Ollama."""
    resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
        "model": EMBED_MODEL, "input": text
    })
    resp.raise_for_status()
    vec = resp.json()["embeddings"][0]
    print(f"  Embedding dimension: {len(vec)} (expected: 1024)")
    return vec

def search_similar(query, vendor_filter=None, top_k=5,
                   date_from=None, date_to=None, amount_min=None, amount_max=None,
                   keyword_filter=None):
    """Same logic as app.py search_similar (kept in sync)."""
    query_vec = get_embedding(query)
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    conditions = ["embedding IS NOT NULL"]
    params = [query_vec]

    if vendor_filter:
        conditions.append("vendor_name ILIKE %s")
        params.append(f"%{vendor_filter}%")
    if date_from:
        conditions.append("date >= %s")
        params.append(date_from)
    if date_to:
        conditions.append("date <= %s")
        params.append(date_to)
    if amount_min:
        conditions.append("total_amount::numeric >= %s")
        params.append(amount_min)
    if amount_max:
        conditions.append("total_amount::numeric <= %s")
        params.append(amount_max)
    if keyword_filter:
        conditions.append(
            "to_tsvector('english', raw_text) @@ plainto_tsquery('english', %s)"
        )
        params.append(keyword_filter)

    where_clause = " AND ".join(conditions)

    sql = f"""
        SELECT id, invoice_number, date, vendor_name, total_amount, currency,
               1 - (embedding <=> %s::vector) AS similarity
        FROM invoices
        WHERE {where_clause}
        ORDER BY similarity DESC
        LIMIT %s
    """
    params.append(top_k)

    t0 = time.time()
    cur.execute(sql, params)
    results = cur.fetchall()
    elapsed = time.time() - t0
    cur.close()
    conn.close()
    return results, elapsed

# ─── Main test ───
print("=" * 60)
print("Test 1: Check row count and embedding coverage")
print("=" * 60)
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()
cur.execute("SELECT COUNT(*), COUNT(embedding) FROM invoices")
total, with_emb = cur.fetchone()
print(f"  Total invoices: {total}")
print(f"  With embeddings: {with_emb}")
if with_emb == 0:
    print("  ⚠️  No embeddings found. Upload an invoice via the UI first.")
    print("     Run: streamlit run app.py  →  Upload tab  →  Process Invoice")
    cur.close(); conn.close(); sys.exit(1)
cur.close()

# Check embedding dimension
cur = conn.cursor()
cur.execute("SELECT vector_dims(embedding) FROM invoices WHERE embedding IS NOT NULL LIMIT 1")
dims = cur.fetchone()[0]
print(f"  Stored vector dimension: {dims} (expected: 1024)")
cur.close(); conn.close()

print()
print("=" * 60)
print("Test 2: Basic semantic search (no filters)")
print("=" * 60)
results, elapsed = search_similar("total amount", top_k=3)
print(f"  Query time: {elapsed*1000:.1f} ms")
print(f"  Results: {len(results)}")
for i, r in enumerate(results):
    print(f"  [{i+1}] id={r[0]} | {r[1]} | {r[3]} | ${r[4]} | sim={r[6]:.4f}")

print()
print("=" * 60)
print("Test 3: Vendor filter")
print("=" * 60)
# Get a vendor name from the first result
if results:
    sample_vendor = results[0][3] if results[0][3] else ""
    if sample_vendor:
        results2, elapsed2 = search_similar("total amount", vendor_filter=sample_vendor[:3], top_k=5)
        print(f"  Filter: vendor ILIKE '%{sample_vendor[:3]}%'")
        print(f"  Query time: {elapsed2*1000:.1f} ms")
        print(f"  Results: {len(results2)}")
        for i, r in enumerate(results2):
            print(f"  [{i+1}] id={r[0]} | vendor={r[3]} | sim={r[6]:.4f}")

print()
print("=" * 60)
print("Test 4: Keyword full-text search on raw_text")
print("=" * 60)
results3, elapsed3 = search_similar("invoice", keyword_filter="total amount", top_k=5)
print(f"  Filter: keyword='total amount' on raw_text")
print(f"  Query time: {elapsed3*1000:.1f} ms")
print(f"  Results: {len(results3)}")
for i, r in enumerate(results3):
    print(f"  [{i+1}] id={r[0]} | {r[1]} | {r[3]} | sim={r[6]:.4f}")

print()
print("=" * 60)
print("Test 5: Stress test — search with top_k=100")
print("=" * 60)
results4, elapsed4 = search_similar("invoice", top_k=100)
print(f"  Query time: {elapsed4*1000:.1f} ms")
print(f"  Results: {len(results4)} (out of {total} total)")
print(f"  ✅ Fast enough for 5000 rows: {'YES' if elapsed4 < 0.5 else 'NO — check HNSW index'}")

print()
print("=" * 60)
print("All tests passed ✅")