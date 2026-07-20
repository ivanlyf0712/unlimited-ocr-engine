#!/usr/bin/env python3
"""
Week 2 Pipeline: Invoice image → Qwen2.5‑VL JSON extraction → PostgreSQL insert.
Usage: python3 pipeline.py <invoice_image.jpg>
"""
import sys
import base64
import json
import time
import io
import requests
import psycopg2
from PIL import Image

# ─── Configuration ───────────────────────────────────────────
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5vl:3b"            # update if you renamed
MAX_LONG_EDGE = 512               # image resize target
JPEG_QUALITY = 60                 # compression level

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "ocr",
    "password": "***REMOVED***",
    "dbname": "invoices"
}

PROMPT = """Extract the following fields from this invoice and return ONLY a valid JSON object:
- invoice_number
- date
- vendor_name
- total_amount
- currency

If a field is missing, use an empty string.
Example output:
{"invoice_number":"INV-001","date":"2024-01-15","vendor_name":"ABC Corp","total_amount":"1250.00","currency":"USD"}"""

# ─── Image preprocessing ─────────────────────────────────────
def preprocess(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_LONG_EDGE:
        ratio = MAX_LONG_EDGE / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"  Resized: {w}×{h} → {new_w}×{new_h}")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()

# ─── OCR & JSON extraction ───────────────────────────────────
def extract_json(image_path: str):
    """Returns (dict_or_None, elapsed_seconds, raw_text)."""
    b64 = preprocess(image_path)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
        "stream": False,
        "options": {"temperature": 0}
    }

    t0 = time.time()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    elapsed = time.time() - t0
    data = resp.json()
    content = data.get("message", {}).get("content", "")

    # Extract JSON object from the reply
    try:
        start = content.index('{')
        end = content.rindex('}') + 1
        json_str = content[start:end]
        return json.loads(json_str), elapsed, content
    except (ValueError, json.JSONDecodeError):
        return None, elapsed, content

# ─── Database insertion ──────────────────────────────────────
def insert_into_db(fields: dict, raw_text: str):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO invoices (invoice_number, date, vendor_name,
                              total_amount, currency, raw_text)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        fields.get("invoice_number", ""),
        fields.get("date", ""),
        fields.get("vendor_name", ""),
        fields.get("total_amount", ""),
        fields.get("currency", ""),
        raw_text
    ))
    conn.commit()
    cur.close()
    conn.close()
    print("  ✅ Inserted into PostgreSQL.")

# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 pipeline.py <invoice_image.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"📄 Processing: {image_path}")

    # 1. OCR + JSON extraction
    data, elapsed, raw = extract_json(image_path)
    print(f"  ⏱️  OCR completed in {elapsed:.1f}s")

    if data is None:
        print("  ❌ Could not parse JSON. Raw response saved to DB.")
        data = {}   # still insert raw text

    print("  📊 Extracted fields:", json.dumps(data, indent=2))

    # 2. Save to database
    insert_into_db(data, raw)

    print("🎉 Pipeline finished.")