#!/usr/bin/env python3
"""Simple test: send an image to llama-server, print OCR text & timing."""

import sys, base64, time, requests

SERVER = "http://127.0.0.1:8081/v1/chat/completions"
IMAGE = sys.argv[1] if len(sys.argv) > 1 else "test_small.jpg"

with open(IMAGE, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {
    "model": "unlimited-ocr",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Free OCR."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]
    }],
    "temperature": 0,
    "max_tokens": 256
}

print(f"Sending {IMAGE} ({len(b64)} chars) …")
t0 = time.time()
resp = requests.post(SERVER, json=payload, timeout=120)
elapsed = time.time() - t0

if resp.status_code == 200:
    text = resp.json()["choices"][0]["message"]["content"]
    print(f"\n✅ Response in {elapsed:.1f}s\n")
    print("OCR TEXT:\n" + text)
else:
    print(f"❌ Server error {resp.status_code}:\n{resp.text}")