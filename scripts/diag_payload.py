#!/usr/bin/env python3
"""Compare single-image vs multi-page payload sizes."""
import json

# Simulate single image payload (ocr.py run_ocr_server)
single = {
    "model": "unlimited-ocr",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Free OCR."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + ("X" * 22000)}}
    ]}],
    "temperature": 0,
    "max_tokens": 16384
}

# Simulate 6-page multi payload (pipeline_fast.py multi_page_ocr)
# Each page ~160K base64 chars after DPI=200 render + resize to 1024px
multi = {
    "model": "unlimited-ocr",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "document parsing."}
    ] + [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + ("X" * 160000)}}
        for _ in range(6)
    ]}],
    "temperature": 0,
    "max_tokens": 16384
}

print(f'Single image payload: {len(json.dumps(single)):>10,} chars ({len(json.dumps(single))/1024:.1f} KB)')
print(f'6-page multi payload: {len(json.dumps(multi)):>10,} chars ({len(json.dumps(multi))/1024:.1f} KB)')
print(f'Ratio:                {len(json.dumps(multi))/len(json.dumps(single)):>10.1f}x')
print()
print('KEY FINDINGS:')
print('1. A PDF page rendered at 200 DPI produces ~160K chars base64 per page')
print('   (A4 = 1653x2339 pixels -> resize to 723x1024 -> ~110-120KB JPEG)')
print('2. A typical uploaded JPG produces ~22K chars base64')
print('3. A 6-page PDF produces ~777 KB JSON payload vs ~22 KB for single image')
print('4. This is 35x LARGER - the multi-page payload may hit server body size limits')
print()
print('ROOT CAUSE: The pdf_pages_to_base64_list uses DPI=200 which creates large pixmaps.')
print('The MAX_LONG_EDGE=1024 resize helps but each page is still ~110-120KB JPEG.')
print('The single-page path (run_ocr_server) also uses _preprocess_image which')
print('applies the same MAX_LONG_EDGE=1024 resize, so a pre-existing JPEG would be smaller.')
print()
print('POSSIBLE FIXES:')
print('- Reduce DPI from 200 to something lower (e.g., 150 or 144)')
print('- Reduce MAX_LONG_EDGE from 1024 to 768 or 512')
print('- Increase JPEG compression (lower JPEG_QUALITY from 85 to 60)')
print('- Split the multi-page request into multiple single-page requests')