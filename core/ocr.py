# -------------------- OCR Module --------------------
"""
OCR engine supporting two backends:
  - Server mode (default): sends images to llama-server via HTTP
  - CLI mode (fallback): calls llama-mtmd-cli via subprocess

Usage:
    from core import run_ocr
    text = run_ocr("image.jpg")

    # or from the command line:
    python3 -m core.ocr image.jpg
    python3 -m core.ocr document.pdf -o output.md
"""
import argparse
import base64
import os
import re
import subprocess
import sys

import requests
from PIL import Image

from core.config import (
    OCR_MODE,
    OCR_SERVER_URL, OCR_SERVER_MODEL, OCR_SERVER_PROMPT,
    OCR_SERVER_TEMPERATURE, OCR_SERVER_MAX_TOKENS, OCR_SERVER_REPEAT_PENALTY,
    LLAMA_CLI, UOCR_MODEL, UOCR_MMPROJ,
    MAX_LONG_EDGE, JPEG_QUALITY,
)
from core.pdf import pdf_to_images_path

# Log lines emitted by llama.cpp that should be stripped from CLI output
_LLAMA_LOG_MARKERS = (
    "llama_model_loader", "llama_model_load", "encode_image",
    "system_info", "main:", "init:", "build:", "start:",
    "clip_model", "ggml_", "warming up", "srv", "slot", "kv_cache",
)


# -- Image preprocessing --

def _preprocess_image(image_path: str) -> str:
    """Resize and compress an image, return path to a temporary JPEG."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_LONG_EDGE:
        ratio = MAX_LONG_EDGE / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    tmp = "/tmp/ocr_preprocessed.jpg"
    img.save(tmp, "JPEG", quality=JPEG_QUALITY)
    return tmp


# -- Output cleaning --

def clean_grounding_tags(text: str) -> str:
    """Remove grounding markers left by CLI-mode output."""
    cleaned = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text)
    cleaned = re.sub(r"<\|(?:grounding|ref|det|/det|/ref)\|>", "", cleaned)
    cleaned = re.sub(r"\n\s*\n", "\n", cleaned)
    return cleaned.strip()


def _clean_server_text(text: str) -> str:
    """Remove coordinate annotations from server-mode output.

    Server returns lines like:  "title [59, 228, 224, 263]INVOICE"
    We keep just:               "INVOICE"
    """
    cleaned = re.sub(r"\b(?:title|text|para|line|block)\s*\[[\d,\s]+\]\s*", "", text)
    cleaned = re.sub(r"\n\s*\n", "\n", cleaned)
    return cleaned.strip()


# -- Server-mode OCR (primary) --

def run_ocr_server(image_path: str) -> str:
    """Send an image to llama-server, return cleaned OCR text."""
    tmp = _preprocess_image(image_path)
    with open(tmp, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    os.unlink(tmp)

    payload = {
        "model": OCR_SERVER_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_SERVER_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }],
        "cache_prompt": False,
        "temperature": OCR_SERVER_TEMPERATURE,
        "max_tokens": OCR_SERVER_MAX_TOKENS,
        "repeat_penalty": OCR_SERVER_REPEAT_PENALTY,
        "stream": False,
    }

    resp = requests.post(OCR_SERVER_URL, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# -- CLI-mode OCR (fallback) --

def run_ocr_cli(image_path: str) -> str:
    """Run llama-mtmd-cli on an image (for environments without llama-server)."""
    processed = _preprocess_image(image_path)
    cmd = [
        LLAMA_CLI, "-m", UOCR_MODEL, "--mmproj", UOCR_MMPROJ,
        "--image", processed, "-p", "Free OCR.",
        "--chat-template", "deepseek-ocr",
        "--temp", "0", "-c", "2048", "-ngl", "0",
        "--threads", "4", "-n", "384",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=180)
    stdout = result.stdout.decode("utf-8", errors="replace")
    lines = [l.strip() for l in stdout.split("\n")
             if l.strip() and not any(s in l for s in _LLAMA_LOG_MARKERS)]
    return "\n".join(lines)


# -- Public API --

def run_ocr(image_path: str) -> str:
    """OCR an image using the active backend (see OCR_MODE in core/config.py)."""
    if OCR_MODE == "cli":
        return clean_grounding_tags(run_ocr_cli(image_path))
    return _clean_server_text(run_ocr_server(image_path))


def ocr_pdf(pdf_path: str, output_path: str = None, dpi: int = 200) -> str:
    """OCR a PDF page by page, return the combined text."""
    image_paths = pdf_to_images_path(pdf_path, dpi=dpi)
    total = len(image_paths)
    print(f"OCRing {total} page(s) via {OCR_MODE} backend...")

    sections = []
    for idx, img_path in enumerate(image_paths):
        print(f"  page {idx + 1}/{total}...", end=" ", flush=True)
        try:
            page_text = run_ocr(img_path)
            sections.append(f"## Page {idx + 1}\n\n{page_text}\n")
            print(f"{len(page_text)} chars")
        except Exception as e:
            sections.append(f"## Page {idx + 1}\n\n*OCR failed: {e}*\n")
            print(f"FAILED: {e}")
        finally:
            if os.path.exists(img_path):
                os.unlink(img_path)

    full_text = "\n---\n\n".join(sections)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Saved to {output_path}")
    return full_text


# -- CLI entry point --

def main():
    parser = argparse.ArgumentParser(description="OCR an image or PDF with Unlimited-OCR")
    parser.add_argument("input", help="Image (jpg/png) or PDF file")
    parser.add_argument("-o", "--output", help="Write result to this file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Error: file not found: {args.input}")

    if args.input.lower().endswith(".pdf"):
        text = ocr_pdf(args.input, output_path=args.output)
    else:
        text = run_ocr(args.input)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)

    if not args.output:
        print(text)


if __name__ == "__main__":
    main()
