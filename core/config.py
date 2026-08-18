# ──────────────────── Configuration ────────────────────
import os

# ── OCR: Server mode (primary, faster) ──
OCR_MODE = os.getenv("OCR_MODE", "server")   # "server" | "cli"
OCR_SERVER_URL = os.getenv("OCR_SERVER_URL", "http://127.0.0.1:8081/v1/chat/completions")
OCR_SERVER_MODEL = os.getenv("OCR_SERVER_MODEL", "Unlimited-OCR")
OCR_SERVER_PROMPT = "Please OCR the text in this image."
OCR_SERVER_TEMPERATURE = 0.0
OCR_SERVER_MAX_TOKENS = 32768
OCR_SERVER_REPEAT_PENALTY = 1.1

# ── OCR: CLI mode (fallback, uses subprocess) ──
LLAMA_CLI = os.path.expanduser(os.getenv("LLAMA_CLI", "~/llama.cpp/build/bin/llama-mtmd-cli"))
UOCR_MODEL = os.path.expanduser(os.getenv("UOCR_MODEL", "~/uocr/Unlimited-OCR-Q4_K_M.gguf"))
UOCR_MMPROJ = os.path.expanduser(os.getenv("UOCR_MMPROJ", "~/uocr/mmproj-Unlimited-OCR-F16.gguf"))

# ── Image preprocessing ──
MAX_LONG_EDGE = int(os.getenv("MAX_LONG_EDGE", "512"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "60"))