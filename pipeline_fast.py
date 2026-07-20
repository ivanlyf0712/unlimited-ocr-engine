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

import argparse, base64, io, json, os, re, subprocess, sys, time
import requests, psycopg2
import fitz                          # PyMuPDF
from PIL import Image

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-mtmd-cli")
UOCR_MODEL = os.path.expanduser("~/uocr/Unlimited-OCR-Q4_K_M.gguf")
UOCR_MMPROJ = os.path.expanduser("~/uocr/mmproj-Unlimited-OCR-F16.gguf")

MAX_LONG_EDGE = 512
JPEG_QUALITY = 60

OLLAMA_URL = "http://127.0.0.1:11434"
TEXT_MODEL = "qwen2.5:1.5b"       # larger model for cleaner JSON

# llama-server for multi-page mode (start separately)
LLAMA_SERVER_URL = "http://127.0.0.1:8081/v1/chat/completions"

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "ocr",
    "password": "***REMOVED***",
    "dbname": "invoices"
}

JSON_PROMPT = """You are an invoice data extractor. Read the invoice text below and output a single JSON object.

Required keys and their types:
- "invoice_number": string — Always look for "Invoice #", "Invoice No.", "INV-#", etc.
- "date": string — The invoice date in YYYY-MM-DD format (e.g. "2024-07-15").
- "vendor_name": string — The company that issued the invoice.
- "total_amount": string — A plain numeric string like "1250.00", without currency symbol.
- "currency": string — Three‑letter currency code, e.g. "USD".

Important:
- Output ONLY the JSON object.
- If a field is missing, use "".

Invoice text:
___RAW_TEXT___

JSON:"""

# ═══════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING & PDF CONVERSION
# ═══════════════════════════════════════════════════════════════════
def preprocess_image(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_LONG_EDGE:
        ratio = MAX_LONG_EDGE / max(w, h)
        img = img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
    tmp = "/tmp/ocr_fast.jpg"
    img.save(tmp, "JPEG", quality=JPEG_QUALITY)
    return tmp

def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[str]:
    doc = fitz.open(pdf_path)
    page_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        w, h = img.size
        if max(w, h) > MAX_LONG_EDGE:
            ratio = MAX_LONG_EDGE / max(w, h)
            img = img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
        tmp = f"/tmp/pdf_page_{os.path.basename(pdf_path)}_{i}.jpg"
        img.save(tmp, "JPEG", quality=JPEG_QUALITY)
        page_paths.append(tmp)
    doc.close()
    return page_paths

def pdf_pages_to_base64_list(pdf_path: str) -> list[str]:
    b64_list = []
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        w, h = img.size
        if max(w, h) > MAX_LONG_EDGE:
            ratio = MAX_LONG_EDGE / max(w, h)
            img = img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        b64_list.append(base64.b64encode(buf.getvalue()).decode())
    doc.close()
    return b64_list

# ═══════════════════════════════════════════════════════════════════
# STAGE 1: UNLIMITED-OCR (CLI)
# ═══════════════════════════════════════════════════════════════════
def run_ocr(image_path: str) -> str:
    img = preprocess_image(image_path)
    cmd = [
        LLAMA_CLI,
        "-m", UOCR_MODEL,
        "--mmproj", UOCR_MMPROJ,
        "--image", img,
        "-p", "Free OCR.",
        "--chat-template", "deepseek-ocr",
        "--temp", "0",
        "-c", "2048",
        "-ngl", "0",
        "--threads", "4",
        "-n", "384"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    stdout = result.stdout.decode("utf-8", errors="replace")
    skip = [
        "llama_model_loader", "llama_model_load", "encode_image",
        "system_info", "main:", "init:", "build:", "start:",
        "clip_model", "ggml_", "warming up", "srv", "slot", "kv_cache"
    ]
    lines = [l.strip() for l in stdout.split("\n") if l.strip() and not any(s in l for s in skip)]
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════
# STAGE 2: TEXT → JSON (prompt + aggressive cleaning)
# ═══════════════════════════════════════════════════════════════════
def _extract_json(prompt_template: str, raw_text: str):
    prompt = prompt_template.replace("___RAW_TEXT___", raw_text)
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": TEXT_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=60
    )
    content = resp.json().get("response", "")
    try:
        start = content.index('{')
        end = content.rindex('}') + 1
        data = json.loads(content[start:end])
        # Clean everything (no schema needed)
        data = clean_invoice_data(data)
        return data
    except (ValueError, json.JSONDecodeError):
        return None

def clean_invoice_data(data: dict) -> dict:
    """Force all fields into the correct format."""
    # total_amount → plain numeric string
    ta = data.get("total_amount")
    if isinstance(ta, dict):
        amount = ta.get("amount", "")
        curr = ta.get("currency", "")
        data["total_amount"] = f"{float(amount):.2f}" if amount else ""
        if curr and not data.get("currency"):
            data["currency"] = curr
    elif isinstance(ta, str):
        cleaned = re.sub(r'[^\d.]', '', ta.replace(',', '').replace(' ', ''))
        data["total_amount"] = f"{float(cleaned):.2f}" if cleaned else ""
    elif isinstance(ta, (int, float)):
        data["total_amount"] = f"{float(ta):.2f}"
    else:
        data["total_amount"] = ""

    # currency → 3 uppercase letters
    curr = data.get("currency", "")
    if isinstance(curr, str):
        curr = curr.strip().upper()
        match = re.match(r'^([A-Z]{3})', curr)
        data["currency"] = match.group(1) if match else ""
    else:
        data["currency"] = ""

    # date → YYYY-MM-DD if possible
    date_val = data.get("date", "")
    if date_val:
        data["date"] = _normalise_date(str(date_val).strip())
    else:
        data["date"] = ""

    for key in ["invoice_number", "date", "vendor_name", "total_amount", "currency"]:
        if key not in data:
            data[key] = ""

    return data

def _normalise_date(date_str: str) -> str:
    if not date_str:
        return ""
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', date_str)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{b:02d}-{a:02d}" if a > 12 else f"{y:04d}-{a:02d}-{b:02d}"
    m = re.match(r'^(\d{1,2})[- ]([A-Za-z]{3})[- ](\d{4})$', date_str)
    if m:
        months = {"jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06",
                  "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12"}
        mm = months.get(m.group(2).lower())
        if mm:
            return f"{int(m.group(3)):04d}-{mm}-{int(m.group(1)):02d}"
    m = re.match(r'^(\d{4})/(\d{1,2})/(\d{1,2})$', date_str)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.match(r'^(\d{1,2})\.(\d{1,2})\.(\d{4})$', date_str)
    if m:
        return f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    return date_str

def text_to_json(raw_text: str):
    return _extract_json(JSON_PROMPT, raw_text)

# ═══════════════════════════════════════════════════════════════════
# DATABASE INSERT
# ═══════════════════════════════════════════════════════════════════
def insert_into_db(fields: dict, raw_text: str, source_file: str):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO invoices (invoice_number, date, vendor_name,
                                  total_amount, currency, raw_text, source_file)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            fields.get("invoice_number", ""),
            fields.get("date", ""),
            fields.get("vendor_name", ""),
            fields.get("total_amount", ""),
            fields.get("currency", ""),
            raw_text,
            source_file
        ))
        conn.commit()
        print(f"  ✅ Inserted: {source_file}")
    except Exception as e:
        conn.rollback()
        print(f"  ❌ DB error: {e}")
    finally:
        cur.close()
        conn.close()

# ═══════════════════════════════════════════════════════════════════
# PROCESS SINGLE IMAGE (OR PDF PAGE)
# ═══════════════════════════════════════════════════════════════════
def clean_grounding_tags(text: str) -> str:
    """Remove <|det|> ... <|/det|> and other grounding markers, keep only the text."""
    # Remove entire <|det|> block, leaving the text that follows
    cleaned = re.sub(r'<\|det\|>.*?<\|/det\|>', '', text)
    # Remove any leftover grounding special tokens (just in case)
    cleaned = re.sub(r'<\|(?:grounding|ref|det|/det|/ref)\|>', '', cleaned)
    # Collapse multiple blank lines
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    # print(cleaned.strip())
    return cleaned.strip()

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
        page_paths = pdf_to_images(image_path)
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