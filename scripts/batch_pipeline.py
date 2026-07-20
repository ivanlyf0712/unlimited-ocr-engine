#!/usr/bin/env python3
"""
Batch pipeline: process all invoice images in a directory (or a single file),
extract structured JSON with Qwen2.5‑VL, and insert into PostgreSQL.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
import requests
import psycopg2
from PIL import Image

# --------------------------- CONFIGURATION ---------------------------
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5vl:3b"            # adjust if you renamed
MAX_LONG_EDGE = 512               # resize target (px)
JPEG_QUALITY = 60

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "ocr",
    "password": "***REMOVED***",
    "dbname": "invoices"
}

# The same prompt used for single‑file extraction
PROMPT = """Extract the following fields from this invoice and return ONLY a valid JSON object:
- invoice_number
- date
- vendor_name
- total_amount
- currency

If a field is missing, use an empty string.
Example output:
{"invoice_number":"INV-001","date":"2024-01-15","vendor_name":"ABC Corp","total_amount":"1250.00","currency":"`USD"`}"""

# --------------------------- PREPROCESSING ---------------------------
def preprocess(image_path: str) -> str:
    """Resize and compress an invoice image, return base64 string."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_LONG_EDGE:
        ratio = MAX_LONG_EDGE / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()

# --------------------------- OCR EXTRACTION ---------------------------
def extract_json(image_path: str):
    """Returns (dict or None, elapsed_seconds, raw_text)."""
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

# --------------------------- DATABASE INSERT ---------------------------
def insert_into_db(fields: dict, raw_text: str, source_file: str = ""):
    """Insert one invoice record. Skips embedding (NULL)."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
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
        print(f"  ✅ Inserted: {source_file}")
    except Exception as e:
        conn.rollback()
        print(f"  ❌ DB insert error for {source_file}: {e}")
    finally:
        cur.close()
        conn.close()

# --------------------------- BATCH PROCESSING ---------------------------
def process_image(image_path):
    """Process a single image file."""
    print(f"\n📄 Processing: {image_path}")
    data, elapsed, raw = extract_json(image_path)
    print(f"  ⏱  {elapsed:.1f}s")
    if data is None:
        print(f"  ⚠️  Could not parse JSON – raw text will still be inserted.")
        data = {}
    else:
        print(f"  📊 Fields: {json.dumps(data, indent=2)}")
    insert_into_db(data, raw, source_file=os.path.basename(image_path))

def process_directory(dir_path):
    """Recursively process all common image files in a directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    for root, _, files in os.walk(dir_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                process_image(os.path.join(root, f))

# --------------------------- MAIN ---------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Batch OCR + DB insert for invoices."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", help="Single invoice image file")
    group.add_argument("-d", "--dir", help="Directory containing invoice images")

    args = parser.parse_args()

    if args.file:
        if not os.path.isfile(args.file):
            print(f"Error: {args.file} is not a valid file.")
            sys.exit(1)
        process_image(args.file)
    else:
        if not os.path.isdir(args.dir):
            print(f"Error: {args.dir} is not a valid directory.")
            sys.exit(1)
        process_directory(args.dir)

if __name__ == "__main__":
    main()