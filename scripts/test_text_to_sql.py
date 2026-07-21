#!/usr/bin/env python3
"""
Test: Direct text-to-SQL using LLM (no hardcoded patterns).
Compares LLM-generated SQL against the existing pattern-based SQL.

Usage:
  python3 scripts/test_text_to_sql.py
"""

import re
import json
import requests
import psycopg2

# ─── Config ───
OLLAMA_URL = "http://127.0.0.1:11434"
TEXT_MODEL = "qwen2.5:1.5b"

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "user": "ocr", "password": "***REMOVED***", "dbname": "invoices"
}

DB_SCHEMA = """
Table: invoices
Columns:
- id (SERIAL PRIMARY KEY)
- invoice_number (VARCHAR)
- date (VARCHAR, format YYYY-MM-DD)
- vendor_name (VARCHAR)
- total_amount (VARCHAR, numeric stored as string, cast with ::numeric)
- currency (VARCHAR(3))
- raw_text (TEXT)
- source_file (VARCHAR)
- created_at (TIMESTAMPTZ)
- embedding (VECTOR(1024))
"""

TEXT_TO_SQL_PROMPT = """You are a PostgreSQL expert. Given the schema below and a natural language question, write a single valid SQL query.

Schema:
{schema}

Important rules:
- total_amount is stored as VARCHAR. Cast it to numeric: total_amount::numeric
- Use ILIKE for case-insensitive text matching
- For GROUP BY queries, include relevant columns
- Wrap numeric results with ::numeric(12,2) for clean output
- Only use SELECT statements (read-only)
- Do NOT use markdown code fences. Output ONLY the raw SQL.

Question: {question}
SQL:"""


def text_to_sql(question: str) -> str:
    """Use LLM to generate SQL directly from natural language."""
    prompt = TEXT_TO_SQL_PROMPT.format(schema=DB_SCHEMA, question=question)
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": TEXT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 512}
            },
            timeout=30
        )
        resp.raise_for_status()
        content = resp.json().get("response", "").strip()
        # Remove any lingering markdown fences
        content = re.sub(r'^```sql\s*|```$', '', content, flags=re.MULTILINE).strip()
        return content
    except Exception as e:
        return f"-- ERROR: {e}"


def run_sql(sql: str) -> list:
    """Execute SQL and return results as list of tuples."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        cur.execute(sql)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        return cols, rows
    except Exception as e:
        return None, [str(e)]
    finally:
        cur.close()
        conn.close()


# ─── Test queries ───
TEST_QUERIES = [
    "Total amount for Alibaba in Q1 2024",
    "Which vendor has the highest total invoice amount?",
    "How many invoices were issued in March 2024?",
    "Show me the 5 biggest invoices",
    "Average invoice amount by currency",
    "List all invoices from last month",
    "Count invoices by vendor",
    "What was the largest invoice from EuroTrade GmbH?",
]


def main():
    print("=" * 80)
    print("TEXT-TO-SQL DEMO (LLM generates SQL directly)")
    print(f"  Model: {TEXT_MODEL}")
    print("=" * 80)

    for query in TEST_QUERIES:
        print(f"\n{'─' * 80}")
        print(f"Query: \"{query}\"")
        sql = text_to_sql(query)
        print(f"\nGenerated SQL:\n{sql}")

        cols, rows = run_sql(sql)
        if cols is None:
            print(f"\n❌ SQL Error: {rows[0]}")
        elif not rows:
            print("\n📭 No results returned.")
        else:
            print(f"\n✅ Results ({len(rows)} row(s)):")
            print(f"   Columns: {', '.join(cols)}")
            for row in rows[:5]:  # show max 5
                print(f"   {row}")

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()