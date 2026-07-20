#!/usr/bin/env python3
"""
Process a PDF in chunks using llama-mtmd-cli (page by page within each chunk).
Combines all OCR output into a single file.

Usage:
  python3 ocr_chunks_cli.py huge.pdf
  python3 ocr_chunks_cli.py huge.pdf -o output.txt --chunk-size 10
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
import fitz          # PyMuPDF
from PIL import Image

# --------------------------- CONFIGURATION ---------------------------
LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-mtmd-cli")
MODEL = os.path.expanduser("~/uocr/Unlimited-OCR-Q4_K_M.gguf")
MMPROJ = os.path.expanduser("~/uocr/mmproj-Unlimited-OCR-F16.gguf")

MAX_LONG_EDGE = 512          # resize pages to this max dimension (px)
JPEG_QUALITY = 60

# CLI flags (edit as needed)
CLI_FLAGS = [
    "--chat-template", "deepseek-ocr",
    "--temp", "0",
    "-c", "2048",
    "-ngl", "0",
    "--threads", "4",
    "-n", "384"
]

# --------------------------------------------------------------------

def page_to_temp_image(page, page_num, max_long_edge=MAX_LONG_EDGE, quality=JPEG_QUALITY):
    """Render a PyMuPDF page, resize, save to a temporary JPEG, return path."""
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    w, h = img.size
    if max(w, h) > max_long_edge:
        ratio = max_long_edge / max(w, h)
        img = img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp, "JPEG", quality=quality)
        return tmp.name

def run_ocr_on_image(image_path: str) -> str:
    """Run llama-mtmd-cli on a single image, return cleaned text."""
    cmd = [
        LLAMA_CLI,
        "-m", MODEL,
        "--mmproj", MMPROJ,
        "--image", image_path,
        "-p", "Free OCR.",
    ] + CLI_FLAGS

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    stdout = result.stdout.decode("utf-8", errors="replace")

    # Remove log lines
    skip = [
        "llama_model_loader", "llama_model_load", "encode_image",
        "system_info", "main:", "init:", "build:", "start:",
        "clip_model", "ggml_", "warming up", "srv", "slot", "kv_cache"
    ]
    lines = [l.strip() for l in stdout.split("\n") if l.strip() and not any(s in l for s in skip)]
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="CLI‑based chunked PDF OCR")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("-o", "--output", default="output.txt", help="Output text file (default: output.txt)")
    parser.add_argument("--chunk-size", type=int, default=10, help="Number of pages per chunk (default: 10)")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"Error: file not found: {args.pdf}")
        sys.exit(1)

    doc = fitz.open(args.pdf)
    total_pages = doc.page_count
    chunk_size = args.chunk_size
    total_chunks = (total_pages + chunk_size - 1) // chunk_size

    print(f"📄 PDF: {args.pdf}  →  {total_pages} pages  →  {total_chunks} chunks of up to {chunk_size} pages")
    print(f"   Mode: CLI (one image per call, sequential within chunk)")
    print()

    with open(args.output, "w", encoding="utf-8") as outfile:
        for chunk_idx in range(total_chunks):
            start_page = chunk_idx * chunk_size
            end_page = min(start_page + chunk_size, total_pages)
            pages_in_chunk = end_page - start_page

            print(f"Chunk {chunk_idx+1}/{total_chunks} (pages {start_page+1}-{end_page})")
            chunk_start_time = time.time()

            for page_idx in range(pages_in_chunk):
                global_page_num = start_page + page_idx
                page = doc.load_page(global_page_num)

                # Render & save temp image
                tmp_img = page_to_temp_image(page, global_page_num)

                # OCR
                t0 = time.time()
                text = run_ocr_on_image(tmp_img)
                elapsed = time.time() - t0
                os.unlink(tmp_img)   # immediately delete temp file

                # Write to output
                outfile.write(f"--- Page {global_page_num+1} ---\n{text}\n\n")
                print(f"   Page {global_page_num+1:4d}: {len(text):5d} chars in {elapsed:.1f}s")

            chunk_elapsed = time.time() - chunk_start_time
            print(f"   Chunk completed in {chunk_elapsed:.1f}s\n")

    doc.close()
    print(f"✅ All pages saved to {args.output}")

if __name__ == "__main__":
    main()