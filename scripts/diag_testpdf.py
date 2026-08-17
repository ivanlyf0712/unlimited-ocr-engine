#!/usr/bin/env python3
"""Check what the actual test_multipage.pdf looks like after pdf_pages_to_base64_list processing."""
import io, base64, json
import fitz
from PIL import Image

MAX_LONG_EDGE = 1024
JPEG_QUALITY = 85

doc = fitz.open('/home/ivanleeyf/ocr/test_multipage.pdf')
print(f'Pages: {doc.page_count}')
print()

b64_list = []
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=200)
    w, h = pix.width, pix.height
    max_edge = max(w, h)
    ratio = 1024 / max_edge if max_edge > 1024 else 1.0
    print(f'Page {i}: pixmap={w}x{h} ({w*h:,} px), max_edge={max_edge}, resize_ratio={ratio:.4f}')

    img = Image.frombytes('RGB', [w, h], pix.samples)
    if max_edge > MAX_LONG_EDGE:
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f'  -> resized to: {new_w}x{new_h} ({new_w*new_h:,} px)')

    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode()
    b64_list.append(b64)
    print(f'  JPEG bytes: {buf.tell():,}, b64 chars: {len(b64):,}')
    print()

doc.close()

# Now compute the total payload size for multi-page mode
total_b64 = sum(len(b) for b in b64_list)
print(f'{"="*60}')
print(f'Total b64 across all pages: {total_b64:,} chars ({total_b64/1024:.1f} KB)')
print(f'Estimated JSON payload size: ~{total_b64 + 500:,} chars')

# Also compute what a single-page payload looks like
single_payload = {
    "model": "unlimited-ocr",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Free OCR."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_list[0]}"}}
    ]}],
    "temperature": 0,
    "max_tokens": 16384,
    "stream": False
}
single_size = len(json.dumps(single_payload))
print(f'Single-page payload size: {single_size:,} chars ({single_size/1024:.1f} KB)')

multi_payload = {
    "model": "unlimited-ocr",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "document parsing."}
    ] + [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
        for b in b64_list
    ]}],
    "temperature": 0,
    "max_tokens": 16384,
    "stream": False
}
multi_size = len(json.dumps(multi_payload))
print(f'Multi-page payload size:  {multi_size:,} chars ({multi_size/1024:.1f} KB)')
print(f'Ratio (multi/single):     {multi_size/single_size:.1f}x')
print()
print(f'Note: llama-server typically has limits on total context/payload size.')
print(f'If multi-page payload is too large, the request may fail or hang.')

# Also do a quick sanity check: what does the first page base64 look like?
print(f'\nFirst page b64 prefix: {b64_list[0][:80]}...')
print(f'First page b64 looks like valid JPEG base64: {b64_list[0].startswith("/9j/")}')