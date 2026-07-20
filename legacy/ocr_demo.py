import base64
import io
import sys
import time
import argparse
import requests
from PIL import Image

# ========== Token 節約 ==========
# 448*448 = 200,704 pixels → faster, fewer vision tokens
MAX_PIXELS = 448 * 448
JPEG_QUALITY = 60
OLLAMA_URL = "http://localhost:11434/api/chat"


def encode_image_to_base64(image_path, max_pixels=None):
    limit = max_pixels or MAX_PIXELS
    t0 = time.time()
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    if w * h > limit:
        ratio = (limit / (w * h)) ** 0.5
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    elapsed = time.time() - t0
    print(f"Image preprocessed ({w}x{h} -> {img.size[0]}x{img.size[1]}) in {elapsed:.2f}s")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def main(image_path, prompt=None, max_pixels=None):
    # 1. Encode image
    try:
        b64_image = encode_image_to_base64(image_path, max_pixels=max_pixels)
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)

    # 2. Call Ollama API
    user_prompt = prompt or "Extract all text visible in this image. Output only the text, nothing else."
    payload = {
        "model": "qwen2.5vl:3b",
        "messages": [{
            "role": "user",
            "content": user_prompt,
            "images": [b64_image]
        }],
        "options": {"temperature": 0, "num_predict": 512},
        "stream": False
    }

    t0 = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        elapsed = time.time() - t0
        print(f"Inference completed in {elapsed:.2f}s")
        print("---")
        print(resp.json()["message"]["content"])
    except requests.RequestException as e:
        print(f"Ollama API error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast OCR using Ollama + Qwen2.5-VL")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--prompt", "-p", help="Custom prompt for the model", default=None)
    parser.add_argument("--max-pixels", "-m", type=int, help="Override MAX_PIXELS limit", default=None)
    args = parser.parse_args()
    main(args.image_path, prompt=args.prompt, max_pixels=args.max_pixels)