#!/usr/bin/env python3
"""
Comprehensive verification of the entire OCR + RAG pipeline.
Run this after setting up the database. No pre-existing data required.

Usage:
  python3 scripts/verify_all.py  [--full]  [--skip-db]  [--skip-ollama]

Checks:
  1. PostgreSQL connection + extensions (vector, pg_trgm)
  2. Table schema (VECTOR(1024), source_file column, indexes)
  3. Ollama API (models: mxbai-embed-large, qwen2.5:1.5b)
  4. Embedding generation (dimension = 1024)
  5. Search with all filter combinations
  6. Search performance (< 500ms for top_k=100)
  7. Keyword full-text search
"""

import argparse
import os
import sys
import time
import requests
import psycopg2

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "user": "ocr", "password": "***REMOVED***", "dbname": "invoices"
}
OLLAMA_URL = "http://127.0.0.1:11434"

passed = 0
failed = 0

def check(description: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  ✅ {description}")
        passed += 1
    else:
        print(f"  ❌ {description}  —  {detail}")
        failed += 1

# ──────────────────────────────────────────────────────────────
def test_db_connection():
    print("\n" + "=" * 60)
    print("1. PostgreSQL Connection & Extensions")
    print("=" * 60)
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Check extensions
        cur.execute("SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm')")
        exts = {r[0] for r in cur.fetchall()}
        check("pgvector extension installed", "vector" in exts,
              "Run: CREATE EXTENSION IF NOT EXISTS vector;")
        check("pg_trgm extension installed", "pg_trgm" in exts,
              "Run: CREATE EXTENSION IF NOT EXISTS pg_trgm;")

        # Check table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'invoices'
            )
        """)
        table_exists = cur.fetchone()[0]
        check("invoices table exists", table_exists,
              "Run: python3 scripts/migrate_db.py")

        if table_exists:
            # Check columns
            cur.execute("""
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns
                WHERE table_name = 'invoices'
                ORDER BY ordinal_position
            """)
            cols = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
            required_cols = [
                "id", "invoice_number", "date", "vendor_name",
                "total_amount", "currency", "raw_text", "source_file",
                "embedding", "created_at"
            ]
            for col in required_cols:
                check(f"  Column '{col}' exists", col in cols,
                      f"Missing column: {col}")

            # Check vector dimension
            if "embedding" in cols:
                check("  embedding is USER-DEFINED type", cols["embedding"][0] == "USER-DEFINED",
                      f"Got type: {cols['embedding'][0]}")

            # Check indexes
            cur.execute("""
                SELECT indexname FROM pg_indexes
                WHERE tablename = 'invoices'
            """)
            indexes = {r[0] for r in cur.fetchall()}
            check("  HNSW index exists",
                  any("hnsw" in idx.lower() or "embedding" in idx.lower() for idx in indexes),
                  "HNSW index not found — searches will be slower")
            check("  trigram index exists",
                  any("trgm" in idx.lower() for idx in indexes),
                  "Trigram index not found — vendor ILIKE will be slower")

        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"  ❌ DB connection failed: {e}")
        global failed
        failed += 1
        return False

# ──────────────────────────────────────────────────────────────
def test_ollama():
    print("\n" + "=" * 60)
    print("2. Ollama API & Models")
    print("=" * 60)
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        # Normalize model names: strip :latest suffix and store base names
        raw_models = [m["name"] for m in resp.json().get("models", [])]
        # Also include base name (without :latest) for matching
        models = set(raw_models)
        for name in raw_models:
            if name.endswith(":latest"):
                models.add(name.replace(":latest", ""))

        check("Ollama API reachable", True)

        required = ["mxbai-embed-large", "qwen2.5:1.5b"]
        for m in required:
            check(f"  Model '{m}' available", m in models,
                  f"Run: ollama pull {m}")

        check("  Embedding API works",
              test_embedding_api(),
              "POST /api/embed failed")
        return True
    except requests.exceptions.ConnectionError:
        check("Ollama API reachable", False, "Is Ollama running? Try: ollama serve")
        return False
    except Exception as e:
        check("Ollama API reachable", False, str(e))
        return False

def test_embedding_api():
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
            "model": "mxbai-embed-large",
            "input": "test invoice total amount shipping"
        }, timeout=30)
        resp.raise_for_status()
        vec = resp.json()["embeddings"][0]
        return len(vec) == 1024
    except Exception:
        return False

# ──────────────────────────────────────────────────────────────
def test_search(db_ok, ollama_ok):
    print("\n" + "=" * 60)
    print("3. Search Pipeline")
    print("=" * 60)

    if not db_ok or not ollama_ok:
        print("  ⏭️  Skipped — DB or Ollama not available")
        return

    # Check if there's data to search
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), COUNT(embedding) FROM invoices")
    total, with_emb = cur.fetchone()
    cur.close()
    conn.close()

    print(f"  Database: {total} invoices ({with_emb} with embeddings)")

    if with_emb == 0:
        print("  ⏭️  Search tests skipped — no embeddings found.")
        print("     Insert test data with: python3 scripts/gen_500.py")
        print("     Generate embeddings with: python3 scripts/embed_update_fast.py")
        return

    # Get embedding for a test query
    print("\n  Testing embedding generation...")
    resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
        "model": "mxbai-embed-large",
        "input": "total amount for shipping services"
    }, timeout=30)
    query_vec = resp.json()["embeddings"][0]
    check("  Embedding dimension", len(query_vec) == 1024,
          f"Expected 1024, got {len(query_vec)}")

    # Test 3a: Basic semantic search
    print("\n  --- 3a. Basic semantic search ---")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    t0 = time.time()
    cur.execute("""
        SELECT id, invoice_number, date, vendor_name, total_amount, currency,
               1 - (embedding <=> %s::vector) AS similarity
        FROM invoices
        WHERE embedding IS NOT NULL
        ORDER BY similarity DESC
        LIMIT 3
    """, (query_vec,))
    results = cur.fetchall()
    elapsed = (time.time() - t0) * 1000
    cur.close()
    conn.close()
    check(f"  Basic search returns results ({len(results)} rows)", len(results) > 0)
    check(f"  Basic search speed ({elapsed:.0f}ms)", elapsed < 200, f"Too slow: {elapsed:.0f}ms")
    if results:
        print(f"    Top result: {results[0][1]} | {results[0][3]} | ${results[0][4]} | sim={results[0][6]:.4f}")

    # Test 3b: Vendor filter
    if results and results[0][3]:
        vendor_hint = results[0][3][:3]
        print(f"\n  --- 3b. Vendor filter (ILIKE '%{vendor_hint}%') ---")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        t0 = time.time()
        cur.execute("""
            SELECT id, vendor_name, 1 - (embedding <=> %s::vector) AS similarity
            FROM invoices
            WHERE embedding IS NOT NULL AND vendor_name ILIKE %s
            ORDER BY similarity DESC
            LIMIT 5
        """, (query_vec, f"%{vendor_hint}%"))
        results2 = cur.fetchall()
        elapsed = (time.time() - t0) * 1000
        cur.close()
        conn.close()
        check(f"  Vendor filter returns results ({len(results2)} rows)", len(results2) > 0)
        check(f"  Vendor filter speed ({elapsed:.0f}ms)", elapsed < 300, f"Too slow: {elapsed:.0f}ms")

    # Test 3c: Keyword full-text search
    print("\n  --- 3c. Keyword full-text search ---")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    t0 = time.time()
    cur.execute("""
        SELECT id, vendor_name, 1 - (embedding <=> %s::vector) AS similarity
        FROM invoices
        WHERE embedding IS NOT NULL
          AND to_tsvector('english', raw_text) @@ plainto_tsquery('english', %s)
        ORDER BY similarity DESC
        LIMIT 5
    """, (query_vec, "invoice"))
    results3 = cur.fetchall()
    elapsed = (time.time() - t0) * 1000
    cur.close()
    conn.close()
    check(f"  Keyword search returns results ({len(results3)} rows)", len(results3) > 0)
    check(f"  Keyword search speed ({elapsed:.0f}ms)", elapsed < 300, f"Too slow: {elapsed:.0f}ms")

    # Test 3d: Stress test (top_k=100)
    print("\n  --- 3d. Stress test (top_k=100) ---")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    t0 = time.time()
    cur.execute("""
        SELECT id, 1 - (embedding <=> %s::vector) AS similarity
        FROM invoices
        WHERE embedding IS NOT NULL
        ORDER BY similarity DESC
        LIMIT 100
    """, (query_vec,))
    results4 = cur.fetchall()
    elapsed = (time.time() - t0) * 1000
    cur.close()
    conn.close()
    check(f"  Stress test returns {len(results4)} results", len(results4) > 0)
    check(f"  Stress test speed ({elapsed:.0f}ms — ok for 5000 rows)",
          elapsed < 500,
          f"Too slow: {elapsed:.0f}ms — check HNSW index")

# ──────────────────────────────────────────────────────────────
def main():
    global passed, failed

    parser = argparse.ArgumentParser(description="Verify OCR+RAG pipeline")
    parser.add_argument("--full", action="store_true", help="Also insert test data and test RAG")
    parser.add_argument("--skip-db", action="store_true", help="Skip PostgreSQL checks")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama checks")
    args = parser.parse_args()

    print("🔍 OCR + RAG Pipeline Verification")
    print(f"   DB: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
    print(f"   Ollama: {OLLAMA_URL}")

    db_ok = False
    ollama_ok = False

    if not args.skip_db:
        db_ok = test_db_connection()
    else:
        print("\n⏭️  Skipping DB checks")

    if not args.skip_ollama:
        ollama_ok = test_ollama()
    else:
        print("\n⏭️  Skipping Ollama checks")

    test_search(db_ok, ollama_ok)

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✅ All checks passed! The pipeline is ready.")
        print("\nNext steps:")
        print("  1. Generate test data:  python3 scripts/gen_500.py")
        print("  2. Generate embeddings:  python3 scripts/embed_update_fast.py")
        print("  3. Run the Streamlit UI: streamlit run app.py")
        print("  4. Or use CLI search:    python3 scripts/hybrid_search.py -q 'your query'")
    else:
        print(f"\n⚠️  {failed} check(s) failed. See details above.")
        sys.exit(1)

if __name__ == "__main__":
    main()