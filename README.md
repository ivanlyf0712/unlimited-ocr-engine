```markdown
# Unlimited‑OCR & RAG – On‑Premise Invoice Processing Pipeline

A fully local, CPU‑only pipeline that:
- Extracts structured data from invoice images (and PDFs)
- Stores them in a PostgreSQL database with vector embeddings
- Enables hybrid search (hard filter + semantic similarity)
- Provides a **RAG (Retrieval‑Augmented Generation)** web interface to answer natural‑language questions about your invoices

All processing stays on‑premise – **zero cloud cost, zero data leakage**.

---

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## ✨ Features

- **Pure CPU OCR** – Baidu’s Unlimited‑OCR (GGUF, 4‑bit quantized) via a custom `llama.cpp` CLI
- **Structured JSON extraction** – A tiny LLM (`qwen2.5:1.5b`) converts raw OCR text to clean fields
- **Post‑processing cleaning** – Automatically normalises amounts, currencies, and dates
- **PostgreSQL + pgvector** – Stores relational fields and 768‑/1024‑dim embeddings side by side
- **Hybrid search** – Hard vendor filter (`ILIKE`) + cosine similarity (`<=>`)
- **Graph‑boosted ranking** – Uses `NetworkX` to reward invoices connected by vendor, date, or amount
- **Streamlit web app** – Upload, process, view, search, and ask questions in natural language
- **Batch processing** – Handle single images, entire directories, or multi‑page PDFs
- **Duplicate‑safe** – Tracks source file; can optionally enforce uniqueness

---

## 🧰 Tech Stack

| Component            | Technology |
|----------------------|------------|
| OCR Engine           | [Baidu Unlimited‑OCR GGUF](https://huggingface.co/sahilchachra/Unlimited-OCR-GGUF) |
| OCR Runtime          | llama.cpp (PR #24975) – `llama‑mtmd‑cli` |
| Text‑to‑JSON LLM     | Qwen2.5‑1.5B (served by [Ollama](https://ollama.com)) |
| Embedding Model      | mxbai‑embed‑large (1024‑dim) |
| Database             | PostgreSQL 16 + [pgvector](https://github.com/pgvector/pgvector) |
| Graph (optional)     | NetworkX |
| Web UI               | Streamlit |
| Containerisation     | Docker Compose (Ollama + Postgres) |
| Language             | Python 3.12+ |

---

## 📦 Prerequisites

- **Docker Desktop** (with at least 8 GB RAM allocated)
- **Python 3.12+** and `pip`
- **Git**
- **CMake** and a C++ compiler (for building llama.cpp)
- **huggingface‑hub** (for downloading GGUF models)
- **macOS or Windows/WSL2** – both are fully supported

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ocr
```

### 2. Create and activate a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Start the backing services (Ollama + PostgreSQL)

```bash
docker compose up -d ollama postgres
```

### 4. Pull the required LLMs into Ollama

```bash
docker exec -it ollama ollama pull qwen2.5:1.5b
docker exec -it ollama ollama pull mxbai-embed-large
docker exec -it ollama ollama pull qwen2.5:3b   # for RAG answers
```

### 5. Download the Unlimited‑OCR GGUF model files

```bash
mkdir -p ~/uocr
cd ~/uocr
huggingface-cli download sahilchachra/Unlimited-OCR-GGUF \
  Unlimited-OCR-Q4_K_M.gguf mmproj-Unlimited-OCR-F16.gguf \
  --local-dir ./
```

### 6. Build llama.cpp with DeepSeek‑OCR support

```bash
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp
git fetch origin pull/24975/head:pr24975 && git checkout pr24975
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j $(nproc) --target llama-mtmd-cli
```

On macOS, replace `$(nproc)` with `$(sysctl -n hw.ncpu)`.

### 7. Create the database table

```bash
cd ~/ocr
source venv/bin/activate
python3 scripts/setup_db.py
```

### 8. (Optional) Process sample invoices

Place some test images in `samples/` and run:

```bash
./reset.sh -all samples/
```

---

## 🖥️ Usage

### Streamlit Web App

```bash
streamlit run app.py
# Open http://localhost:8501
```

- **View Database** – browse all processed invoices
- **Upload Invoice** – drag & drop an image or PDF to run the full pipeline
- **Search** – enter a natural language query and optionally filter by vendor

### Command Line

#### Process a single image

```bash
python3 pipeline_fast.py -f invoice.jpg
```

#### Process a directory

```bash
python3 pipeline_fast.py -d ~/invoices/
```

#### Hybrid search

```bash
python3 scripts/hybrid_search.py "total amount for Alibaba" Alibaba 5
```

#### Full reset (clear DB, re‑process, re‑embed)

```bash
./reset.sh -all samples/
```

---

## 📁 Project Structure

```
ocr/
├── app.py                     # Streamlit web UI
├── pipeline_fast.py           # Main OCR → JSON → DB pipeline
├── pipeline.py                # Legacy pipeline (optional)
├── docker-compose.yml         # Docker services
├── reset.sh                   # One‑command reset + reprocess
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── scripts/                   # Utility scripts
│   ├── setup_db.py
│   ├── embed_update.py
│   ├── embed_update_fast.py
│   ├── cleanup_db.py
│   ├── hybrid_search.py
│   ├── batch_pipeline.py
│   ├── batch_ocr.py
│   └── gen_invoices.py
├── tests/                     # Test / debugging scripts
│   ├── test_ocr.py
│   └── test_json_extraction.py
├── samples/                   # Sample invoice images
│   ├── sample_invoice.jpg
│   └── ...
├── legacy/                    # Old scripts kept for reference
│   ├── ocr_demo.py
│   └── ...
└── data/                      # Generated large files (gitignored)
    ├── output.txt
    └── random_500_invoices.pdf
```

---

## ⚡ Performance

- **OCR speed**: ~20 s per page (CPU only, image resized to 384 px)
- **JSON extraction**: <1 s (tiny LLM)
- **Embedding generation**: <0.5 s per invoice (batch mode)
- **Total pipeline**: ~21 s per invoice
- Scales linearly with additional CPU cores (parallel mode available)

---

## 🔧 Troubleshooting

- **`model not found` in Ollama** – Restart the container with `docker compose restart ollama`
- **`KeyError` / `ValueError` in pipeline** – The post‑processing cleaner handles most malformed responses
- **Missing `llama-mtmd-cli`** – Ensure you built llama.cpp from PR #24975 and the binary is at `~/llama.cpp/build/bin/llama-mtmd-cli`
- **Out of memory** – Reduce `MAX_LONG_EDGE` to 256, or use the smaller `Q3_K_M` model
- **Swap usage high** – Stop other memory‑heavy processes; use `--parallel 1` if needed

---

## 📜 License

This project is licensed under the MIT License.  
Model weights are subject to their respective licenses (Baidu Unlimited‑OCR: MIT; Qwen2.5: Apache 2.0).
```

---

Save this file as `README.md` in your project root (`~/ocr/README.md`). It will be automatically included when you push to Git.

The README covers:
- A clear, professional description of the project
- All prerequisites and setup steps
- How to use the web UI and CLI
- The reorganised project structure
- Performance expectations
- Common troubleshooting tips

If you want me to also generate a **Chinese version** or a shorter **summary for the presentation**, let me know.