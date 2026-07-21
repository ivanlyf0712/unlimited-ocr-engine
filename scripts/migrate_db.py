#!/usr/bin/env python3
"""
Drop and recreate the invoices table with the corrected schema (VECTOR(1024)).
WARNING: This DELETES all existing invoice data. Run setup_db.py afterward.
"""
import psycopg2

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "user": "ocr", "password": "***REMOVED***", "dbname": "invoices"
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Enable required extensions
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

# Drop existing table (if any)
cur.execute("DROP TABLE IF EXISTS invoices CASCADE;")

# Create table with correct vector dimension
cur.execute("""
    CREATE TABLE invoices (
        id SERIAL PRIMARY KEY,
        invoice_number VARCHAR(100),
        date VARCHAR(50),
        vendor_name VARCHAR(200),
        total_amount VARCHAR(50),
        currency VARCHAR(10),
        raw_text TEXT,
        source_file VARCHAR(300),
        embedding VECTOR(1024),
        created_at TIMESTAMP DEFAULT NOW()
    );
""")

# HNSW index for fast ANN search
cur.execute("""
    CREATE INDEX invoices_embedding_hnsw_idx
    ON invoices USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
""")

# Trigram index for vendor ILIKE filters
cur.execute("""
    CREATE INDEX invoices_vendor_name_trgm_idx
    ON invoices USING gin (vendor_name gin_trgm_ops);
""")

conn.commit()
cur.close()
conn.close()
print("Migration complete. Table recreated with VECTOR(1024), HNSW index, trigram index.")
print("Now re-upload your invoices via the Streamlit UI — embeddings will be generated from raw_text.")