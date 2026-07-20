#!/usr/bin/env python3
"""
Hybrid search over invoices – command‑line interface.
Usage:
  python3 hybrid_search.py "your question" [vendor_filter] [limit]
Examples:
  python3 hybrid_search.py "total amount for Alibaba" Alibaba
  python3 hybrid_search.py "what is the date?" "" 10
"""

import sys, requests, psycopg2

OLLAMA_EMBED = "http://127.0.0.1:11434/api/embed"
MODEL = "mxbai-embed-large"

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "user": "ocr", "password": "***REMOVED***", "dbname": "invoices"
}

def get_embedding(text):
    resp = requests.post(OLLAMA_EMBED, json={"model": MODEL, "input": text})
    resp.raise_for_status()
    return resp.json()["embeddings"][0]

def hybrid_query(question, vendor=None, limit=3):
    query_vec = get_embedding(question)
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    if vendor:
        cur.execute("""
            SELECT invoice_number, date, vendor_name, total_amount, currency,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM invoices
            WHERE vendor_name ILIKE %s AND embedding IS NOT NULL
            ORDER BY similarity DESC
            LIMIT %s
        """, (query_vec, f"%{vendor}%", limit))
    else:
        cur.execute("""
            SELECT invoice_number, date, vendor_name, total_amount, currency,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM invoices
            WHERE embedding IS NOT NULL
            ORDER BY similarity DESC
            LIMIT %s
        """, (query_vec, limit))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    query = sys.argv[1]
    vendor = sys.argv[2] if len(sys.argv) > 2 else None
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    print(f"Query: {query}")
    print(f"Vendor filter: {vendor if vendor else 'none'}")
    print(f"Limit: {limit}\n")

    results = hybrid_query(query, vendor, limit)
    if not results:
        print("No matching invoices found.")
    else:
        print(f"{'Inv #':<20} {'Date':<12} {'Vendor':<25} {'Amount':<15} {'Similarity':<10}")
        print("-" * 85)
        for inv_num, date, vendor_name, amount, currency, sim in results:
            print(f"{inv_num:<20} {date:<12} {vendor_name:<25} {amount:<15} {sim:.4f}")