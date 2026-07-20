#!/usr/bin/env python3
"""
Unlimited-OCR wrapper using llama-mtmd-cli
Phase 1: Optimized CLI parameters
Phase 2: Image preprocessing for speed
"""

import subprocess
import sys
import os
import re
import time
from PIL import Image

# ─── Configuration ───────────────────────────────────────────
LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-mtmd-cli")
MODEL = os.path.expanduser("~/uocr/Unlimited-OCR-Q4_K_M.gguf")
MMPROJ = os.path.expanduser("~/uocr/mmproj-Unlimited-OCR-F16.gguf")

THREADS = 4
CONTEXT = 2048
MAX_TOKENS = 384
MAX_LONG_EDGE = 512
QUALITY = 60

# ─── Phase 2: Image Preprocessing ────────────────────────────
def preprocess_image(image_path, max_long_edge=MAX_LONG_EDGE):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    if max(w, h) > max_long_edge:
        ratio = max_long_edge / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"  Resized: {w}×{h} → {new_w}×{new_h}")
    else:
        print(f"  Size OK: {w}×{h} (no resize)")

    temp_path = "/tmp/ocr_fast.jpg"
    img.save(temp_path, "JPEG", quality=QUALITY)
    return temp_path


# ─── Core OCR Function ───────────────────────────────────────
def ocr(image_path, prompt="Free OCR."):
    cmd = [
        LLAMA_CLI,
        "-m", MODEL,
        "--mmproj", MMPROJ,
        "--image", image_path,
        "-p", prompt,
        "--chat-template", "deepseek-ocr",
        "--temp", "0",
        "-c", str(CONTEXT),
        "-ngl", "0",
        "--threads", str(THREADS),
        "-n", str(MAX_TOKENS),
        "--mlock",
        "--no-mmap",
    ]

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    elapsed = time.time() - start_time

    stdout = result.stdout.decode("utf-8", errors="replace")
    stdout = re.sub(r"\x1b\[[0-9;]*m", "", stdout)
    stderr = result.stderr.decode("utf-8", errors="replace")
    stderr = re.sub(r"\x1b\[[0-9;]*m", "", stderr)

    # ─── Parse model load time (from stderr, where llama.cpp logs it) ─
    load_time = None
    prompt_eval_time = None

    # llama.cpp outputs timing to stderr; search both streams to be safe
    for line in (stderr + "\n" + stdout).split("\n"):
        line_lower = line.lower()
        # Match patterns like "load time = 1234.56 ms" or "load time: 1234.56 ms"
        if "load time" in line_lower and "model load time" not in line_lower:
            match = re.search(r'(\d+\.?\d*)\s*ms', line)
            if match:
                load_time = float(match.group(1)) / 1000
        # Match "prompt eval time = 123.45 ms" or similar
        if "prompt eval time" in line_lower:
            match = re.search(r'(\d+\.?\d*)\s*ms', line)
            if match:
                prompt_eval_time = float(match.group(1)) / 1000
        # Alternative: match "total time = X.XX ms" for model loading
        if load_time is None and "load time" in line_lower and "total" in line_lower:
            match = re.search(r'(\d+\.?\d*)\s*ms', line)
            if match:
                load_time = float(match.group(1)) / 1000

    # ─── Extract real output lines ───────────────────────────
    skip_patterns = [
        "llama_model_loader", "llama_model_load", "encode_image",
        "system_info", "main:", "init:", "build:", "start:",
        "clip_model", "ggml_", "warming up", "srv",
        "slot", "kv_cache", "prompt", "token",
    ]

    output_lines = []
    for line in stdout.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if any(pattern in stripped for pattern in skip_patterns):
            continue
        output_lines.append(stripped)

    return "\n".join(output_lines), elapsed, load_time, prompt_eval_time


# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_ocr.py <image_path> [output_file]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(image_path):
        print(f"ERROR: File not found: {image_path}")
        sys.exit(1)

    t_total_start = time.time()

    print("=" * 55)
    print("  Unlimited-OCR (CPU) — Optimized")
    print("=" * 55)
    print(f"  Model:    {os.path.basename(MODEL)}")
    print(f"  Threads:  {THREADS}")
    print(f"  Context:  {CONTEXT}")
    print(f"  Max out:  {MAX_TOKENS} tokens")
    print(f"  Resize:   {MAX_LONG_EDGE}px")
    print("-" * 55)

    image_path = preprocess_image(image_path, MAX_LONG_EDGE)

    print(f"  Prompt:   \"Free OCR.\"")
    print(f"  Running OCR...")

    text, inference_time, load_time, eval_time = ocr(image_path, "Free OCR.")
    t_total = time.time() - t_total_start

    print()
    print("=" * 55)
    print("  OCR RESULT")
    print("=" * 55)
    print(text if text else "(no text extracted)")
    print("=" * 55)
    print()
    print("⏱️   Timing")
    print(f"  Model load:   {load_time:.1f}s" if load_time else "  Model load:   N/A")
    print(f"  Image encode: {eval_time:.1f}s" if eval_time else "  Image encode: N/A")
    print(f"  Inference:    {inference_time:.1f}s")
    print(f"  Total:        {t_total:.1f}s")
    if load_time:
        print(f"  (Inference - load = {inference_time - load_time:.1f}s)")
    print("=" * 55)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text if text else "")
        print(f"\nSaved OCR result to {output_file}")

