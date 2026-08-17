#!/usr/bin/env python3
"""Diagnostic: compare base64 output of a sample image vs an A4 PDF page."""
import io, os, base64
from PIL import Image, ImageDraw
import fitz

# ── Create a sample image (simulating a typical upload) ──
sample_img = Image.new('RGB', (800, 600), color='white')
draw = ImageDraw.Draw(sample_img)
draw.rectangle([50, 50, 750, 550], outline='black', width=3)
draw.text((100, 100), 'Sample Image Test', fill='black')
draw.text((100, 200), 'Line 2: Hello World', fill='black')
draw.text((100, 300), 'Line 3: OCR Test 123', fill='black')
sample_img.save('/tmp/sample_image.jpg', 'JPEG', quality=85)

# ── Create an A4 PDF page ──
a4_w, a4_h = 595, 842  # A4 in points
doc = fitz.open()
page = doc.new_page(width=a4_w, height=a4_h)
page.insert_text((50, 50), 'A4 Page Test', fontsize=24, color=(0,0,0))
page.insert_text((50, 150), 'Line 2: Hello World from PDF', fontsize=18, color=(0,0,0))
page.insert_text((50, 250), 'Line 3: OCR Test 456', fontsize=18, color=(0,0,0))
page.insert_text((50, 350), 'Line 4: More content here', fontsize=18, color=(0,0,0))
doc.save('/tmp/test_a4_page.pdf')
doc.close()

# ── Config (mirrors core/config.py) ──
MAX_LONG_EDGE = 1024
JPEG_QUALITY = 85

# Path A: process the sample image the way pdf_pages_to_base64_list would
img = Image.open('/tmp/sample_image.jpg').convert('RGB')
w, h = img.size
print(f'Sample image original size: {w}x{h}')
if max(w, h) > MAX_LONG_EDGE:
    ratio = MAX_LONG_EDGE / max(w, h)
    img = img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
    print(f'  -> resized to: {img.size}')
buf = io.BytesIO()
img.save(buf, format='JPEG', quality=JPEG_QUALITY)
sample_b64 = base64.b64encode(buf.getvalue()).decode()
sample_bytes_raw = buf.tell()
print(f'Sample JPEG bytes: {sample_bytes_raw}')

# Path B: process the A4 PDF page via same logic
doc2 = fitz.open('/tmp/test_a4_page.pdf')
page = doc2[0]
pix = page.get_pixmap(dpi=200)
print(f'\nA4 PDF page pixmap size: {pix.width}x{pix.height} (at 200 DPI)')
print(f'  Pixels total: {pix.width * pix.height:,}')
img2 = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
w2, h2 = img2.size
print(f'A4 PDF page PIL image size: {w2}x{h2}')
if max(w2, h2) > MAX_LONG_EDGE:
    ratio2 = MAX_LONG_EDGE / max(w2, h2)
    new_w, new_h = int(w2*ratio2), int(h2*ratio2)
    img2 = img2.resize((new_w, new_h), Image.LANCZOS)
    print(f'  -> resized to: {img2.size} (ratio={ratio2:.4f})')
buf2 = io.BytesIO()
img2.save(buf2, format='JPEG', quality=JPEG_QUALITY)
a4_b64 = base64.b64encode(buf2.getvalue()).decode()
a4_bytes_raw = buf2.tell()
doc2.close()
print(f'A4 PDF page JPEG bytes: {a4_bytes_raw}')

print(f'\n{"="*60}')
print(f'SIZE COMPARISON')
print(f'{"="*60}')
print(f'Sample image b64: {len(sample_b64):>8,} chars ({len(sample_b64)/1024:>7.1f} KB b64, ~{len(sample_b64)*3/4/1024:>7.1f} KB raw)')
print(f'A4 PDF page b64:  {len(a4_b64):>8,} chars ({len(a4_b64)/1024:>7.1f} KB b64, ~{len(a4_b64)*3/4/1024:>7.1f} KB raw)')
print(f'Ratio (A4/sample): {len(a4_b64)/len(sample_b64):.1f}x')

# Also check what the ocr.py _preprocess_image produces
print(f'\n{"="*60}')
print(f'OCR.PY _PREPROCESS_IMAGE (for comparison)')
print(f'{"="*60}')
img3 = Image.open('/tmp/sample_image.jpg').convert('RGB')
w3, h3 = img3.size
print(f'Original: {w3}x{h3}')
if max(w3, h3) > MAX_LONG_EDGE:
    ratio3 = MAX_LONG_EDGE / max(w3, h3)
    img3 = img3.resize((int(w3*ratio3), int(h3*ratio3)), Image.LANCZOS)
    print(f'  -> resized to: {img3.size}')
tmp3 = '/tmp/ocr_server_test.jpg'
img3.save(tmp3, 'JPEG', quality=JPEG_QUALITY)
with open(tmp3, 'rb') as f:
    data3 = f.read()
ocr_b64 = base64.b64encode(data3).decode()
print(f'ocr.py preprocess b64 length: {len(ocr_b64)} chars')
print(f'Same as pdf_pages_to_base64_list sample? {sample_b64 == ocr_b64}')

# Now: what about the "image_url" format difference?
print(f'\n{"="*60}')
print(f'API PAYLOAD FORMAT COMPARISON')
print(f'{"="*60}')
# In run_ocr_server (ocr.py), the image_url format is:
#   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
# In multi_page_ocr (pipeline_fast.py), the image_url format is identical:
#   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
print('Both use the same format: data:image/jpeg;base64,<b64>')
print('No functional difference in the image_url structure.')