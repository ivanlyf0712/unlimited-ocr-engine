# ──────────────────── PDF Module ────────────────────
import io
import os
import base64
import fitz  # PyMuPDF
from PIL import Image

from core.config import MAX_LONG_EDGE, JPEG_QUALITY


def pdf_to_images_bytes(pdf_bytes: bytes, source_filename: str):
    """Convert a PDF (from bytes, e.g. Streamlit upload) to a list of (temp_path, source_name) tuples."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_list = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        w, h = img.size
        if max(w, h) > MAX_LONG_EDGE:
            ratio = MAX_LONG_EDGE / max(w, h)
            img = img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
        tmp = f"/tmp/pdf_{source_filename}_page_{i}.jpg"
        img.save(tmp, "JPEG", quality=JPEG_QUALITY)
        page_list.append((tmp, f"{source_filename}_page_{i}"))
    doc.close()
    return page_list


def pdf_to_images_path(pdf_path: str, dpi: int = 200) -> list:
    """Convert a PDF (from file path) to a list of temp image paths."""
    doc = fitz.open(pdf_path)
    page_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        w, h = img.size
        if max(w, h) > MAX_LONG_EDGE:
            ratio = MAX_LONG_EDGE / max(w, h)
            img = img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
        tmp = f"/tmp/pdf_page_{os.path.basename(pdf_path)}_{i}.jpg"
        img.save(tmp, "JPEG", quality=JPEG_QUALITY)
        page_paths.append(tmp)
    doc.close()
    return page_paths


def pdf_pages_to_base64_list(pdf_path: str) -> list:
    """Convert each page of a PDF to a base64-encoded JPEG string."""
    b64_list = []
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        w, h = img.size
        if max(w, h) > MAX_LONG_EDGE:
            ratio = MAX_LONG_EDGE / max(w, h)
            img = img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        b64_list.append(base64.b64encode(buf.getvalue()).decode())
    doc.close()
    return b64_list