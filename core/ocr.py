# ──────────────────── OCR Module ────────────────────
import subprocess
import re

from PIL import Image

from core.config import LLAMA_CLI, UOCR_MODEL, UOCR_MMPROJ, MAX_LONG_EDGE, JPEG_QUALITY


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


def preprocess_image(image_path: str) -> str:
    """Resize and compress an image for faster OCR. Returns path to temp file."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_LONG_EDGE:
        ratio = MAX_LONG_EDGE / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    tmp = "/tmp/ocr_fast.jpg"
    img.save(tmp, "JPEG", quality=JPEG_QUALITY)
    return tmp


def run_ocr(image_path: str) -> str:
    """Preprocess image, then run Unlimited-OCR and return cleaned text."""
    processed_path = preprocess_image(image_path)
    cmd = [LLAMA_CLI, "-m", UOCR_MODEL, "--mmproj", UOCR_MMPROJ,
           "--image", processed_path, "-p", "Free OCR.",
           "--chat-template", "deepseek-ocr",
           "--temp", "0", "-c", "2048", "-ngl", "0",
           "--threads", "4", "-n", "384"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    stdout = result.stdout.decode("utf-8", errors="replace")
    skip = ["llama_model_loader","llama_model_load","encode_image",
            "system_info","main:","init:","build:","start:",
            "clip_model","ggml_","warming up","srv","slot","kv_cache"]
    lines = [l.strip() for l in stdout.split("\n") if l.strip() and not any(s in l for s in skip)]
    return "\n".join(lines)
