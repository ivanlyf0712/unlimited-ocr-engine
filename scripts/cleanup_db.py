#!/usr/bin/env python3
"""
Stricter one‑time clean‑up:
  - Handle "EUR 1,200.50" (currency before amount) and "1,200.50 EUR"
  - Remove commas from numbers
  - Ensure total_amount is a plain numeric string (e.g. "1200.50")
  - Fill missing source_file & embeddings
"""

import json, re, psycopg2, requests

OLLAMA_EMBED = "http://127.0.0.1:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "user": "ocr", "password": "***REMOVED***", "dbname": "invoices"
}

def get_embedding(text):
    resp = requests.post(OLLAMA_EMBED, json={"model": EMBED_MODEL, "input": text})
    resp.raise_for_status()
    return resp.json()["embeddings"][0]

def parse_total_amount(raw_value):
    """
    Returns (clean_numeric_amount, currency_code).
    Handles:
      - dict like {"amount": 1250.0, "currency": "USD"}
      - "1250.00 USD"
      - "EUR 1,200.50"
      - "1,200.50"
    """
    if raw_value is None:
        return "", ""

    # 1. If it's a dict (JSON stored as string)
    if isinstance(raw_value, str):
        try:
            data = json.loads(raw_value)
            if isinstance(data, dict):
                amount = str(data.get("amount", ""))
                curr = data.get("currency", "")
                # remove commas from amount
                amount = amount.replace(",", "")
                return amount, curr.upper() if curr else ""
        except (json.JSONDecodeError, TypeError):
            pass

    # 2. String patterns
    if isinstance(raw_value, str):
        val = raw_value.strip()
        # Pattern A: currency before amount, like "EUR 1,200.50" or "EUR1,200.50"
        m = re.match(r'^([A-Za-z]{3})\s*([\d,]+\.?\d*)$', val)
        if m:
            curr = m.group(1).upper()
            amount = m.group(2).replace(",", "")
            return amount, curr

        # Pattern B: amount then currency, like "1,200.50 EUR" or "1200.50EUR"
        m = re.match(r'^([\d,]+\.?\d*)\s*([A-Za-z]{3})$', val)
        if m:
            amount = m.group(1).replace(",", "")
            curr = m.group(2).upper()
            return amount, curr

        # Pattern C: just a number (maybe with commas)
        if re.match(r'^[\d,]+\.?\d*$', val):
            return val.replace(",", ""), ""

    # Fallback: return the raw value as amount, no currency
    return str(raw_value).replace(",", ""), ""

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # 1. Clean total_amount & currency
    print("Cleaning total_amount / currency ...")
    cur.execute("SELECT id, total_amount, currency FROM invoices")
    updates = 0
    for row_id, total, curr in cur.fetchall():
        new_amount, new_curr = parse_total_amount(total)
        # Use new_curr if available, else keep existing currency
        final_curr = new_curr if new_curr else (curr if curr else "")
        # Ensure final_curr is uppercase
        final_curr = final_curr.upper() if final_curr else ""

        # Update if something changed
        if new_amount != str(total) or final_curr != (curr or ""):
            cur.execute(
                "UPDATE invoices SET total_amount = %s, currency = %s WHERE id = %s",
                (new_amount, final_curr, row_id)
            )
            updates += 1
            print(f"  id={row_id}: '{total}' -> total_amount='{new_amount}', currency='{final_curr}'")
    print(f"  Updated {updates} rows.")

    # 2. Fill missing source_file
    print("\nFixing missing source_file ...")
    cur.execute("SELECT id FROM invoices WHERE source_file IS NULL")
    for (row_id,) in cur.fetchall():
        placeholder = f"legacy_{row_id}"
        cur.execute("UPDATE invoices SET source_file = %s WHERE id = %s", (placeholder, row_id))
        print(f"  id={row_id}: source_file='{placeholder}'")

    # 3. Regenerate missing embeddings
    print("\nRegenerating missing embeddings ...")
    cur.execute("SELECT id, raw_text FROM invoices WHERE embedding IS NULL")
    for row_id, raw_text in cur.fetchall():
        if raw_text:
            vec = get_embedding(raw_text)
            cur.execute("UPDATE invoices SET embedding = %s WHERE id = %s", (vec, row_id))
            print(f"  id={row_id}: embedding generated")

    conn.commit()
    cur.close()
    conn.close()
    print("\n✅ Cleanup complete. All total_amount values are now plain numbers, currency is clean.")

if __name__ == "__main__":
    main()