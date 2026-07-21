#!/usr/bin/env python3
"""Check that all indexes exist and are valid (not INVALID)."""
import psycopg2

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "user": "ocr", "password": "***REMOVED***", "dbname": "invoices"
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

print("=" * 60)
print("Index Status Check")
print("=" * 60)

# Get all indexes on invoices table with details
cur.execute("""
    SELECT
        i.indexname,
        i.indexdef,
        am.amname AS index_method,
        pg_size_pretty(pg_relation_size(idx.indexrelid)) AS size,
        idx.indisvalid
    FROM pg_indexes i
    JOIN pg_class c ON c.relname = i.indexname
    JOIN pg_am am ON am.oid = c.relam
    JOIN pg_index idx ON idx.indexrelid = c.oid
    WHERE i.tablename = 'invoices'
    ORDER BY i.indexname
""")

rows = cur.fetchall()
if not rows:
    print("\n❌ NO INDEXES FOUND on 'invoices' table!")
    print("   Run: python3 scripts/migrate_db.py")
else:
    print(f"\nFound {len(rows)} index(es):\n")
    for name, defn, method, size, valid in rows:
        status = "✅ VALID" if valid else "❌ INVALID"
        print(f"  {status}  {name}")
        print(f"          Method: {method}  |  Size: {size}")
        # Show just the USING clause
        parts = defn.split("USING")
        if len(parts) > 1:
            using_part = parts[1].strip().split("(")[0].strip()
            print(f"          Using: {using_part}")
        print()

# Also check table row count
cur.execute("SELECT COUNT(*), COUNT(embedding) FROM invoices")
total, with_emb = cur.fetchone()
print(f"Table: {total} rows, {with_emb} with embeddings")

cur.close()
conn.close()