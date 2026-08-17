# ──────────────────── OCR Module ────────────────────
"""
OCR engine supporting two backends:
  - Server mode (default): sends images to llama-server via HTTP
  - CLI mode (fallback): calls llama-mtmd-cli via subprocess

Import run_ocr() to get text from the active backend.
"""
import base64
import os
import re
import subprocess

import requests
from PIL import Image
import fitz  # PyMuPDF  ← 唯一新增的匯入（用於 PDF 轉圖片）

from core.config import (
    OCR_MODE,
    OCR_SERVER_URL, OCR_SERVER_MODEL, OCR_SERVER_PROMPT,
    OCR_SERVER_TEMPERATURE, OCR_SERVER_MAX_TOKENS, OCR_SERVER_REPEAT_PENALTY,
    LLAMA_CLI, UOCR_MODEL, UOCR_MMPROJ,
    MAX_LONG_EDGE, JPEG_QUALITY,
)


# ── Image preprocessing ──────────────────────────────────

def _preprocess_image(image_path: str) -> str:
    """Resize and compress an image, return path to temporary JPEG."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_LONG_EDGE:
        ratio = MAX_LONG_EDGE / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    tmp = "/tmp/ocr_server.jpg"
    img.save(tmp, "JPEG", quality=JPEG_QUALITY)
    return tmp


# ── Text cleaning ────────────────────────────────────────

def clean_grounding_tags(text: str) -> str:
    """Remove grounding markers left by CLI-mode."""
    cleaned = re.sub(r'<\|det\|>.*?<\|/det\|>', '', text)
    cleaned = re.sub(r'<\|(?:grounding|ref|det|/det|/ref)\|>', '', cleaned)
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    return cleaned.strip()


def _clean_server_text(text: str) -> str:
    """Remove coordinate annotations from server-mode OCR output.
    Server returns lines like: "title [59, 228, 224, 263]INVOICE"
    We want just: "INVOICE"
    """
    cleaned = re.sub(r'\b(?:title|text|para|line|block)\s*\[[\d,\s]+\]\s*', '', text)
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    return cleaned.strip()


# ── Server-mode OCR (primary) ────────────────────────────

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
        "stream": False
    }

    resp = requests.post(OCR_SERVER_URL, json=payload, timeout=180)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return content.strip()


# ── CLI-mode OCR (fallback) ──────────────────────────────

def run_ocr_cli(image_path: str) -> str:
    """Run Unlimited-OCR via subprocess (legacy, kept for environments
       where llama-server is not running)."""
    processed = _preprocess_image(image_path)
    cmd = [LLAMA_CLI, "-m", UOCR_MODEL, "--mmproj", UOCR_MMPROJ,
           "--image", processed, "-p", "Free OCR.",
           "--chat-template", "deepseek-ocr",
           "--temp", "0", "-c", "2048", "-ngl", "0",
           "--threads", "4", "-n", "384"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=180)
    stdout = result.stdout.decode("utf-8", errors="replace")
    skip = ["llama_model_loader", "llama_model_load", "encode_image",
            "system_info", "main:", "init:", "build:", "start:",
            "clip_model", "ggml_", "warming up", "srv", "slot", "kv_cache"]
    lines = [l.strip() for l in stdout.split("\n")
             if l.strip() and not any(s in l for s in skip)]
    return "\n".join(lines)


# ── Public API ───────────────────────────────────────────

def run_ocr(image_path: str) -> str:
    """OCR an image using the active backend (configured via OCR_MODE)."""
    if OCR_MODE == "cli":
        text = run_ocr_cli(image_path)
        return clean_grounding_tags(text)
    else:
        return _clean_server_text(run_ocr_server(image_path))


# ════════════════════════════════════════════════════════════════
#  以下為【新增】的 PDF 逐頁處理功能（完全相容上方介面）
# ════════════════════════════════════════════════════════════════

def pdf_to_images(pdf_path: str, dpi: int = 300) -> list:
    """
    將 PDF 每一頁轉為 PNG 圖片，存放在當前目錄的 'tmp-image' 資料夾。
    回傳圖片路徑清單。
    """
    doc = fitz.open(pdf_path)
    out_dir = "tmp-image"
    os.makedirs(out_dir, exist_ok=True)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(out_dir, f"page_{i+1:04d}.png")
        pix = page.get_pixmap(matrix=mat)
        pix.save(out)
        paths.append(out)
    doc.close()
    print(f"✅ 已轉換 {len(paths)} 頁至 {os.path.abspath(out_dir)}")
    return paths


def ocr_pdf(pdf_path: str, output_path: str = "./results", dpi: int = 300) -> str:
    """
    將 PDF 逐頁 OCR（使用 run_ocr 統一介面），合併所有頁面文字後儲存。
    這是官方 GGUF 建議的「page-by-page」標準實作。
    """
    image_paths = pdf_to_images(pdf_path, dpi=dpi)
    total = len(image_paths)
    print(f"開始逐頁 OCR（共 {total} 頁）...")

    all_text = []
    for idx, img_path in enumerate(image_paths):
        print(f"  處理第 {idx+1}/{total} 頁...")
        try:
            # 直接使用你原有的 run_ocr()，自動根據 OCR_MODE 切換 server/cli
            page_text = run_ocr(img_path)
            all_text.append(f"## 第 {idx+1} 頁\n\n{page_text}\n\n---\n\n")
        except Exception as e:
            print(f"  ❌ 第 {idx+1} 頁失敗：{e}")
            all_text.append(f"## 第 {idx+1} 頁\n\n*OCR 失敗*\n\n---\n\n")

    full_text = "".join(all_text)

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        out_file = os.path.join(output_path, "pdf_output.md")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"✅ 結果已儲存至 {out_file}")

    return full_text