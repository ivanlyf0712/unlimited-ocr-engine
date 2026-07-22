#!/usr/bin/env python3
"""
Benchmark the llama-server OCR pipeline (HTTP requests) on images/PDFs.
Usage:
  python3 benchmark_server.py -f invoice.jpg
  python3 benchmark_server.py -d ./my_folder/
  python3 benchmark_server.py -f multi_page.pdf -o results.txt
"""

import argparse, base64, io, os, sys, time, requests
from PIL import Image
import fitz  # PyMuPDF

# ── Server configuration ──
URL = "http://127.0.0.1:8081/v1/chat/completions"
MODEL = "Unlimited-OCR"
PROMPT = "Please OCR the text in this image."
TEMPERATURE = 0.1
MAX_TOKENS = 2048
REPEAT_PENALTY = 1.1
STREAM = False

# ── Image preprocessing (resize to reduce transfer time) ──
MAX_LONG_EDGE = 1024   # or 512 if you want faster, but server can handle 1024
JPEG_QUALITY = 85

def preprocess_image(image_path):
    """Resize and compress, return path to temporary JPEG."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_LONG_EDGE:
        ratio = MAX_LONG_EDGE / max(w, h)
        img = img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
    tmp = "/tmp/ocr_bench_server.jpg"
    img.save(tmp, "JPEG", quality=JPEG_QUALITY)
    return tmp

def ocr_single_image(image_path):
    """Send an image to the server, return (text, elapsed_seconds)."""
    # Preprocess and encode
    tmp_path = preprocess_image(image_path)
    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    os.unlink(tmp_path)

    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "repeat_penalty": REPEAT_PENALTY,
        "stream": STREAM
    }

    t0 = time.time()
    resp = requests.post(URL, json=payload, timeout=180)
    elapsed = time.time() - t0

    if resp.status_code != 200:
        return f"ERROR {resp.status_code}: {resp.text}", elapsed

    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content, elapsed

def process_file(file_path):
    """Yield (source_name, text, elapsed) for each page/image."""
    fname = os.path.basename(file_path)
    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # Resize if needed
            w, h = img.size
            if max(w, h) > MAX_LONG_EDGE:
                ratio = MAX_LONG_EDGE / max(w, h)
                img = img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
            tmp = "/tmp/ocr_bench_server_page.jpg"
            img.save(tmp, "JPEG", quality=JPEG_QUALITY)
            text, elapsed = ocr_single_image(tmp)
            os.unlink(tmp)
            yield f"{fname}_page_{i}", text, elapsed
        doc.close()
    else:
        text, elapsed = ocr_single_image(file_path)
        yield fname, text, elapsed

def main():
    parser = argparse.ArgumentParser(description="Benchmark llama-server OCR")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", help="Single image or PDF")
    group.add_argument("-d", "--dir", help="Directory containing images/PDFs")
    parser.add_argument("-o", "--output", help="Save full OCR text to file (optional)")
    args = parser.parse_args()

    out_fh = open(args.output, "w") if args.output else None

    def handle_path(path):
        for source, text, elapsed in process_file(path):
            print(f"📄 {source}")
            print(f"   ⏱  {elapsed:.1f}s")
            print(f"   OCR: {text}\n")
            if out_fh:
                out_fh.write(f"=== {source} ===\n{text}\n\n")

    if args.file:
        handle_path(args.file)
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pdf"}
        for root, _, files in os.walk(args.dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    handle_path(os.path.join(root, f))

    if out_fh:
        out_fh.close()
        print(f"Full OCR text saved to {args.output}")

if __name__ == "__main__":
    main()