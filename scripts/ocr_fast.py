import os
import base64
import requests
import fitz  # PyMuPDF

# ---------- 設定 ----------
LLAMA_SERVER_URL = "http://localhost:8081/v1/chat/completions"
MODEL_NAME = "unlimited-ocr"

# ---------- 單張圖片 OCR ----------
def ocr_image(image_path: str, prompt: str = "document parsing.") -> str:
    """對單張圖片進行 OCR，回傳文字。"""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    url = f"data:{mime};base64,{b64}"

    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url}}
            ]
        }],
        "temperature": 0,
        "max_tokens": 32768,
        "cache_prompt": False,
        "repeat_penalty": 1.1,
        "stream": False
    }
    resp = requests.post(LLAMA_SERVER_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# ---------- PDF 轉圖片 ----------
def pdf_to_images(pdf_path: str, dpi: int = 300) -> list:
    """將 PDF 每頁轉為 PNG，存在 tmp-image/ 下，回傳路徑清單。"""
    doc = fitz.open(pdf_path)
    out_dir = "tmp-image"
    os.makedirs(out_dir, exist_ok=True)
    mat = fitz.Matrix(dpi/72, dpi/72)
    paths = []
    for i, page in enumerate(doc):
        path = os.path.join(out_dir, f"page_{i+1:04d}.png")
        page.get_pixmap(matrix=mat).save(path)
        paths.append(path)
    doc.close()
    print(f"✅ 已轉換 {len(paths)} 頁至 {os.path.abspath(out_dir)}")
    return paths

# ---------- PDF OCR（逐頁順序處理） ----------
def ocr_pdf(pdf_path: str, output_path: str = None, dpi: int = 300) -> str:
    """將 PDF 逐頁 OCR，結果合併後儲存（可選）。"""
    image_paths = pdf_to_images(pdf_path, dpi)
    print(f"開始逐頁 OCR（共 {len(image_paths)} 頁）...")
    all_text = []
    for i, img in enumerate(image_paths):
        print(f"  處理第 {i+1}/{len(image_paths)} 頁...")
        try:
            text = ocr_image(img)
            all_text.append(f"## 第 {i+1} 頁\n\n{text}\n\n---\n\n")
        except Exception as e:
            print(f"  ❌ 第 {i+1} 頁失敗：{e}")
            all_text.append(f"## 第 {i+1} 頁\n\n*OCR 失敗*\n\n---\n\n")
    full_text = "".join(all_text)

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        out_file = os.path.join(output_path, "output.md")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"✅ 結果已儲存至 {out_file}")
    return full_text

# ---------- 使用範例 ----------
if __name__ == "__main__":
    # 單張圖片
    # result = ocr_image("test.jpg")
    # print(result)

    # PDF
    ocr_pdf("test_multipage2.pdf", output_path="./results", dpi=300)