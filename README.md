# Unlimited-OCR Engine

A **pure-CPU, fully on-premise** OCR toolkit built on the [Unlimited-OCR GGUF](https://huggingface.co/sahilchachra/Unlimited-OCR-GGUF) vision model running on [llama.cpp](https://github.com/ggml-org/llama.cpp) (PR #24975, `llama-mtmd-cli` / `llama-server`).

Extracts raw text from images and multi-page PDFs — no GPU, no cloud, no per-page cost.

> Domain applications built on this engine (invoice extraction, chat intelligence) live in separate repositories. This repo is the reusable OCR core only.

---

## Features

- **Two backends**
  - *Server mode* (default, fast): HTTP calls to a persistent `llama-server` — no model reload per image
  - *CLI mode* (fallback): `llama-mtmd-cli` via subprocess
- **Image preprocessing** — resizes to a max long edge and re-encodes as JPEG to minimise tokens and latency
- **PDF support** — renders pages with PyMuPDF, OCRs page by page, merges output
- **Output cleaning** — strips grounding tags / model artifacts from raw output

## Project Structure

```
core/
  ocr.py        # run_ocr() — server/CLI backends, preprocessing, PDF handling
  pdf.py        # PDF → image helpers (PyMuPDF)
  config.py     # env-driven configuration (server URL, model paths, image settings)
scripts/
  batch_ocr.py              # OCR large PDFs page-by-page via CLI
  ocr_fast.py               # quick single-image / PDF OCR via server
  benchmark_server.py       # server-mode timing
  benchmark_resolutions.py  # accuracy/speed trade-off across resolutions
  test_llama_server.sh      # smoke-test a running llama-server
tests/
  test_ocr.py     # CLI wrapper test with preprocessing
samples/          # sample images & PDFs
```

## Setup

1. **Download the model** (~2 GB):

```bash
mkdir -p ~/uocr
huggingface-cli download sahilchachra/Unlimited-OCR-GGUF \
  Unlimited-OCR-Q4_K_M.gguf mmproj-Unlimited-OCR-F16.gguf \
  --local-dir ~/uocr
```

2. **Build llama.cpp** with DeepSeek-OCR support:

```bash
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp
git fetch origin pull/24975/head:pr24975 && git checkout pr24975
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j $(nproc) --target llama-mtmd-cli llama-server
```

3. **Python deps**:

```bash
pip install -r requirements.txt
```

## Usage

**Start the server** (persistent, recommended):

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/uocr/Unlimited-OCR-Q4_K_M.gguf \
  --mmproj ~/uocr/mmproj-Unlimited-OCR-F16.gguf \
  --chat-template deepseek-ocr --port 8081
```

**OCR an image or PDF from Python:**

```python
from core import run_ocr

text = run_ocr("samples/sample_invoice.jpg")
print(text)
```

**Batch PDF via CLI:**

```bash
python3 scripts/batch_ocr.py document.pdf -o output.txt --chunk-size 10
```

## Configuration

All settings are environment-variable driven (see `core/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `OCR_MODE` | `server` | `server` or `cli` |
| `OCR_SERVER_URL` | `http://127.0.0.1:8081/v1/chat/completions` | llama-server endpoint |
| `OCR_SERVER_MODEL` | `Unlimited-OCR` | model name |
| `LLAMA_CLI` | `~/llama.cpp/build/bin/llama-mtmd-cli` | CLI binary path |
| `UOCR_MODEL` / `UOCR_MMPROJ` | `~/uocr/...` | GGUF model files |
| `MAX_LONG_EDGE` | `512` | resize target (px) |
| `JPEG_QUALITY` | `60` | preprocessing quality |

## Performance

| Metric | Value |
|---|---|
| OCR per image (warm server) | ~12 s |
| OCR per image (cold CLI) | ~30–60 s |
| Throughput (single CPU) | 2,400+ images / 8 h |

## License

MIT
