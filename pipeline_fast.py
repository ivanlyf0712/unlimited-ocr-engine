#!/usr/bin/env python3
"""
Fast two‑stage invoice OCR pipeline with PDF support.
(No JSON schema – relies on robust post‑processing cleaning.)

Usage:
  python3 pipeline_fast.py -f invoice.jpg
  python3 pipeline_fast.py -f batch_of_invoices.pdf   # each page = own invoice
  python3 pipeline_fast.py -f single_invoice_3pages.pdf --multi-page
  python3 pipeline_fast.py -d ./input_folder/
"""

import argparse, json, os, sys, time
import requests, psycopg2

# ── Core modules ──
from core.config import (
    OLLAMA_URL, DB_CONFIG, LLAMA_SERVER_URL
)
from core.ocr import run_ocr, clean_grounding_tags
from core.extraction import text_to_json, clean_invoice_data
from core.db import get_embedding
from core.pdf import pdf_to_images_path, pdf_pages_to_base64_list

# ═══════════════════════════════════════════════════════════════════
# DATABASE INSERT (with inline embedding)
# ═══════════════════════════════════════════════════════════════════
def insert_into_db(fields: dict, raw_text: str, source_file: str):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO invoices (invoice_number, date, vendor_name,
                                  total_amount, currency, raw_text, source_file)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            fields.get("invoice_number", ""),
            fields.get("date", ""),
            fields.get("vendor_name", ""),
            fields.get("total_amount", ""),
            fields.get("currency", ""),
            raw_text,
            source_file
        ))
        new_id = cur.fetchone()[0]
        # Generate embedding from raw_text (not structured fields)
        if raw_text and raw_text.strip():
            vec = get_embedding(raw_text.strip())
            cur.execute("UPDATE invoices SET embedding = %s WHERE id = %s", (vec, new_id))
        conn.commit()
        print(f"  ✅ Inserted: {source_file} (id={new_id}, embedding generated)")
    except Exception as e:
        conn.rollback()
        print(f"  ❌ DB error: {e}")
    finally:
        cur.close()
        conn.close()

# ═══════════════════════════════════════════════════════════════════
# PROCESS SINGLE IMAGE (OR PDF PAGE)
# ═══════════════════════════════════════════════════════════════════
def process_single_image(image_path: str, source_file: str = None):
    if source_file is None:
        source_file = os.path.basename(image_path)

    fname = source_file
    print(f"\n📄 {fname}")
    t0 = time.time()

    raw_text = run_ocr(image_path)
    raw_text = clean_grounding_tags(raw_text)
    print("==============OCR Result===================")
    print(raw_text)
    t1 = time.time()
    print(f"  ⏱  OCR: {t1-t0:.1f}s")
    print("===========================================")

    data = text_to_json(raw_text)
    t2 = time.time()
    print(f"  ⏱  JSON parse: {t2-t1:.1f}s")

    if data is None:
        print("  ⚠️  JSON extraction failed – inserting raw text only.")
        data = {}
    else:
        print(f"  📊 Fields: {json.dumps(data, indent=2)}")

    empty_count = 0
    for key in ("invoice_number", "date", "vendor_name", "currency"):
        if not data.get(key):
            empty_count += 1
    ta = data.get("total_amount")
    if ta is None or ta == 0 or ta == 0.0 or ta == "":
        empty_count += 1
    if empty_count > 2:
        print(f"  🚫 Rejected: {empty_count}/5 fields empty – inserting with fields cleared.")
        data = {"invoice_number": "", "date": "", "vendor_name": "", "total_amount": 0.0, "currency": ""}

    insert_into_db(data, raw_text, fname)
    print(f"  🕐 Total: {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════
# MULTI-PAGE (SINGLE DOCUMENT) MODE
# ═══════════════════════════════════════════════════════════════════
def multi_page_ocr(pdf_path: str) -> str:
    b64_images = pdf_pages_to_base64_list(pdf_path)
    content_parts = [{"type": "text", "text": "document parsing."}]
    for b64 in b64_images:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    payload = {
        "model": "unlimited-ocr",
        "messages": [{"role": "user", "content": content_parts}],
        "temperature": 0,
        "max_tokens": 4096
    }
    resp = requests.post(LLAMA_SERVER_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# ═══════════════════════════════════════════════════════════════════
# MAIN DISPATCHER
# ═══════════════════════════════════════════════════════════════════
def process_image(image_path: str, multi_page: bool = False):
    fname = os.path.basename(image_path)

    if image_path.lower().endswith(".pdf"):
        if multi_page:
            print(f"\n📑 PDF (multi‑page, single document): {fname}")
            raw_text = multi_page_ocr(image_path)
            print("  ⏱  Multi‑page OCR done. Extracting JSON...")
            data = text_to_json(raw_text)
            if data is None:
                data = {}
            else:
                print(f"  📊 Fields: {json.dumps(data, indent=2)}")
            insert_into_db(data, raw_text, fname)
            return

        print(f"\n📑 PDF detected: {fname} (page‑by‑page mode)")
        page_paths = pdf_to_images_path(image_path)
        total_pages = len(page_paths)
        print(f"   → {total_pages} pages.")
        for i, page_path in enumerate(page_paths):
            source = f"{fname}_page_{i}"
            try:
                process_single_image(page_path, source)
            except Exception as e:
                print(f"  ❌ Page {i} failed: {e}")
            os.remove(page_path)
            print(f"  [{i+1}/{total_pages}] pages done.")
        print(f"  🎉 PDF processing complete ({total_pages} pages).")
        return

    process_single_image(image_path)

def main():
    parser = argparse.ArgumentParser(
        description="Fast two‑stage OCR pipeline (Unlimited‑OCR + Qwen2.5‑1.5B).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -f invoice.jpg
  %(prog)s -f batch.pdf
  %(prog)s -f multi_page_invoice.pdf --multi-page
  %(prog)s -d ~/invoices/
        """
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", help="Single invoice image or PDF")
    group.add_argument("-d", "--dir", help="Directory containing invoice images/PDFs")
    parser.add_argument("--multi-page", action="store_true",
                        help="Process multi‑page PDF as a single document")
    args = parser.parse_args()

    if args.file:
        try:
            process_image(args.file, args.multi_page)
        except Exception as e:
            print(f"  ❌ Failed to process {args.file}: {e}")
            sys.exit(1)
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pdf"}
        skipped = []
        for root, _, files in os.walk(args.dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    filepath = os.path.join(root, f)
                    try:
                        process_image(filepath, args.multi_page)
                    except Exception as e:
                        print(f"  ❌ Skipped {filepath}: {e}")
                        skipped.append(filepath)
        if skipped:
            print(f"\n⚠️  Skipped {len(skipped)} file(s) due to errors:")
            for s in skipped:
                print(f"    - {s}")

if __name__ == "__main__":
    main()