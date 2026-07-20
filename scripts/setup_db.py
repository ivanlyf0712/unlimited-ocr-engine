#!/usr/bin/env python3
"""Create the invoices table with pgvector support."""
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    user="ocr",
    password="***REMOVED***",
    dbname="invoices"
)
cur = conn.cursor()

# Enable the vector extension
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Create the table
cur.execute("""
    CREATE TABLE IF NOT EXISTS invoices (
        id SERIAL PRIMARY KEY,
        invoice_number VARCHAR(100),
        date VARCHAR(50),
        vendor_name VARCHAR(200),
        total_amount VARCHAR(50),
        currency VARCHAR(10),
        raw_text TEXT,
        embedding VECTOR(1536),
        created_at TIMESTAMP DEFAULT NOW()
    );
""")

conn.commit()
cur.close()
conn.close()
print("Database table ready.")