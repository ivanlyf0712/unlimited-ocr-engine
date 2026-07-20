#!/usr/bin/env python3
"""Extract structured JSON from invoices using Qwen2.5-VL via Ollama."""
import sys, base64, json, time, io, requests
from PIL import Image

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5vl:3b"          # or "qwen-ocr:latest" if you renamed it
MAX_LONG_EDGE = 512             # resize images to this max dimension
QUALITY = 60                    # JPEG quality (smaller = faster)

PROMPT = """Extract the following fields from this invoice and return ONLY a valid JSON object:
- invoice_number
- date
- vendor_name
- total_amount
- currency

If a field is missing, use an empty string.
Example output:
{"invoice_number":"INV-001","date":"2024-01-15","vendor_name":"ABC Corp","total_amount":"1250.00","currency":"USD"}"""

def preprocess(image_path: str) -> str:
    """Resize and compress image, return base64 string."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_LONG_EDGE:
        ratio = MAX_LONG_EDGE / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"  Resized: {w}×{h} → {new_w}×{new_h}")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=QUALITY)
    return base64.b64encode(buf.getvalue()).decode()

def extract_json(image_path: str):
    b64_image = preprocess(image_path)

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [b64_image]
            }
        ],
        "stream": False,
        "options": {"temperature": 0}
    }

    t0 = time.time()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    elapsed = time.time() - t0
    data = resp.json()

    content = data.get("message", {}).get("content", "")
    # Try to extract JSON from the response
    try:
        start = content.index('{')
        end = content.rindex('}') + 1
        json_str = content[start:end]
        return json.loads(json_str), elapsed, content
    except (ValueError, json.JSONDecodeError):
        return None, elapsed, content

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ocr_json_qwen.py <image>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Processing: {image_path}")

    data, elapsed, raw = extract_json(image_path)
    print(f"\nTime: {elapsed:.1f}s")
    print(f"Raw response (first 300 chars):\n{raw[:300]}\n")
    if data:
        print("Parsed JSON:")
        print(json.dumps(data, indent=2))
    else:
        print("❌ Could not parse JSON. Check the raw response above.")