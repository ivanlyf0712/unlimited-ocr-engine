#!/usr/bin/env python3
"""
Benchmark OCR across different image resolutions and collect token counts.
Usage:
  python3 benchmark_resolutions.py
"""
import argparse
import base64
import json
import os
import time
import requests
from PIL import Image

URL = "http://127.0.0.1:8081/v1/chat/completions"
MODEL = "Unlimited-OCR"
PROMPT = "Please OCR the text in this image."
TEMPERATURE = 0.1
MAX_TOKENS = 2048
REPEAT_PENALTY = 1.1
JPEG_QUALITY = 85
IMAGE_PATH = "samples/sample_invoice.jpg"


def benchmark_resolution(image_path, max_edge):
    """Run OCR at a given max_edge resolution, return (elapsed_s, prompt_tokens, completion_tokens, text_preview)."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    original_size = (w, h)
    if max(w, h) > max_edge:
        ratio = max_edge / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    new_size = img.size

    tmp = "/tmp/ocr_bench_res.jpg"
    img.save(tmp, "JPEG", quality=JPEG_QUALITY)

    with open(tmp, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    os.unlink(tmp)

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
        "stream": False,
        "__verbose": True,
    }

    t0 = time.time()
    resp = requests.post(URL, json=payload, timeout=300)
    elapsed = time.time() - t0

    if resp.status_code != 200:
        print(f"  ERROR {resp.status_code}: {resp.text[:500]}")
        return elapsed, 0, 0, "ERROR", new_size

    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
    completion_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0

    return elapsed, prompt_tokens, completion_tokens, content[:120], new_size


def main():
    resolutions = [256, 384, 512, 800]
    results = []

    print("=" * 80)
    print("OCR BENCHMARK - Multiple Resolutions")
    print(f"Image: {IMAGE_PATH}")
    print(f"Server: {URL}")
    print("=" * 80)

    for res in resolutions:
        print(f"\n--- Resolution: max_edge={res} ---")
        elapsed, prompt_tokens, completion_tokens, text_preview, img_size = benchmark_resolution(
            IMAGE_PATH, res
        )
        print(f"  Image size: {img_size[0]}x{img_size[1]}")
        print(f"  Elapsed: {elapsed:.2f}s")
        print(f"  Prompt tokens: {prompt_tokens}")
        print(f"  Completion tokens: {completion_tokens}")
        print(f"  Text preview: {text_preview}")

        results.append({
            "resolution": res,
            "img_width": img_size[0],
            "img_height": img_size[1],
            "elapsed_s": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        })

    # ── Summary calculations ──
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    GPT4O_INPUT_PER_1M = 5.00
    GPT4O_OUTPUT_PER_1M = 15.00
    AZURE_PRICE_PER_PAGE = 0.01
    LOCAL_COST = 2000.00

    print()
    print("| Resolution | Time (s) | Prompt Tok | Comp Tok | Img/hr | Img/8h | Img/yr | GPT-4o Cost/img | Azure Cost/img |")
    print("|-----------|----------|------------|----------|--------|--------|--------|-----------------|---------------|")

    for r in results:
        t = r["elapsed_s"]
        img_hr = 3600 / t if t > 0 else 0
        img_8h = img_hr * 8
        img_yr = img_hr * 8 * 365

        gpt_cost = (r["prompt_tokens"] / 1_000_000) * GPT4O_INPUT_PER_1M + \
                   (r["completion_tokens"] / 1_000_000) * GPT4O_OUTPUT_PER_1M

        print(f"| {r['resolution']} | {t:.1f} | {r['prompt_tokens']} | {r['completion_tokens']} | {img_hr:.0f} | {img_8h:.0f} | {img_yr:,.0f} | ${gpt_cost:.4f} | ${AZURE_PRICE_PER_PAGE:.4f} |")

    print()

    # ── ROI Analysis ──
    print("=" * 80)
    print("ROI ANALYSIS (Local Server: $2,000 one-time)")
    print("=" * 80)

    # Use the slowest (most realistic/conservative) resolution for ROI
    best = results[-1]  # 800 resolution is the highest quality

    for r in results:
        t = r["elapsed_s"]
        gpt_cost = (r["prompt_tokens"] / 1_000_000) * GPT4O_INPUT_PER_1M + \
                   (r["completion_tokens"] / 1_000_000) * GPT4O_OUTPUT_PER_1M

        # Break-even vs GPT-4o
        if gpt_cost > 0:
            breakeven_gpt = LOCAL_COST / gpt_cost
        else:
            breakeven_gpt = float("inf")

        # Break-even vs Azure
        breakeven_azure = LOCAL_COST / AZURE_PRICE_PER_PAGE

        # Annual savings for 100k/month = 1.2M/year
        monthly = 100_000
        annual = monthly * 12
        annual_gpt_cost = annual * gpt_cost
        annual_azure_cost = annual * AZURE_PRICE_PER_PAGE
        savings_gpt = annual_gpt_cost - LOCAL_COST  # first year
        savings_azure = annual_azure_cost - LOCAL_COST

        print(f"\nResolution {r['resolution']} ({r['img_width']}x{r['img_height']}, {t:.1f}s):")
        print(f"  Break-even vs GPT-4o: {breakeven_gpt:,.0f} images")
        print(f"  Break-even vs Azure:  {breakeven_azure:,.0f} images")
        print(f"  Annual GPT-4o cost (100k/mo): ${annual_gpt_cost:,.2f}")
        print(f"  Annual Azure cost (100k/mo):  ${annual_azure_cost:,.2f}")
        print(f"  Year-1 savings vs GPT-4o:     ${savings_gpt:,.2f}")
        print(f"  Year-1 savings vs Azure:      ${savings_azure:,.2f}")

        # Annual images processed locally
        imgs_per_year = (3600 / t) * 8 * 365
        print(f"  Max images/year (8h nights):   {imgs_per_year:,.0f}")


if __name__ == "__main__":
    main()