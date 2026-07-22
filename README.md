# Unlimited‑OCR & RAG Platform

A **fully on‑premise, pure‑CPU** pipeline that combines **Invoice OCR** with **Corporate Chat Intelligence**, using a shared RAG (Retrieval‑Augmented Generation) engine.

- **Invoice OCR** – extracts structured fields from images/PDFs, stores them with vector embeddings, and provides hybrid search + natural‑language Q&A.
- **CorpChat Intelligence** – generates or ingests WeChat Work‑style chat logs, links them to business‑card contacts, and enables semantic search, fraud detection, and relationship analysis.

All processing happens **locally** – zero cloud costs, zero data leakage.

---

## 🧰 Tech Stack

| Component           | Technology                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| OCR Engine          | [Baidu Unlimited‑OCR GGUF](https://huggingface.co/sahilchachra/Unlimited-OCR-GGUF) |
| OCR Runtime         | llama.cpp (PR #24975) – `llama-mtmd-cli` (CLI) or `llama-server`          |
| JSON Extraction     | Qwen2.5‑1.5B (Ollama) – tiny, fast, accurate with post‑processing         |
| Embedding Model     | mxbai‑embed‑large (1024‑dim)                                               |
| RAG Answer Model    | Qwen2.5‑1.5B (or 3B for higher quality)                                   |
| Database            | PostgreSQL 16 + [pgvector](https://github.com/pgvector/pgvector)           |
| Graph (optional)    | NetworkX – graph‑boosted ranking                                           |
| Containerisation    | Docker Compose (Ollama + Postgres)                                         |
| Web UI              | Streamlit                                                                   |
| Language            | Python 3.12+                                                               |

---

## ✨ Features

### Invoice OCR & RAG
- **Pure CPU OCR** – resizes images to minimise tokens, supports both CLI and persistent server mode.
- **Structured JSON extraction** – with robust post‑processing that normalises amounts, dates, and currencies.
- **Hybrid search** – hard vendor filter (`ILIKE`) + cosine similarity, optionally boosted by a graph of vendor/date/amount relationships.
- **Two‑path query router** – automatically chooses between **aggregation SQL** (sum, count, average, etc.) and **semantic search + RAG**, based on a hybrid classifier (regex + LLM).
- **Streamlit dashboard** – upload invoices, browse the database, search, and get AI‑generated answers.

### CorpChat Intelligence
- **Business‑card contacts** – stores names, companies, phones, emails (from Faker or real OCR).
- **WeChat Work‑compatible messages** – synthetic or real chat logs in the official API JSON format.
- **Identity‑aware embeddings** – each message vector includes sender name, company, and conversation label.
- **Conversation viewer** – WhatsApp‑style UI: select a conversation to see the full chat history.
- **Semantic search** – find messages by meaning (e.g., “crypto scam discussions”) and generate RAG answers.

---

## 📁 Project Structure
ocr/
├── core/ # Shared library (used by both apps)
│ ├── config.py # DB config, model names, paths
│ ├── db.py # DB operations (connect, insert, update, fetch)
│ ├── ocr.py # OCR engine (CLI or server)
│ ├── extraction.py # Text → JSON extraction + cleaning
│ ├── embedding.py # Embedding generation (get_embedding)
│ ├── classifier.py # Hybrid intent classifier (regex + LLM)
│ └── pdf.py # PDF → image conversion
│
├── apps/
│ ├── invoice/ # Invoice OCR & RAG application
│ │ ├── app.py # Streamlit UI
│ │ ├── pipeline_fast.py # Batch OCR pipeline
│ │ ├── agg_engine.py # SQL aggregation engine + LLM rephrasing
│ │ ├── init_invoice.sql # Database schema for invoices
│ │ └── reset.sh # Reset + reprocess invoices
│ │
│ └── corpchat/ # CorpChat Intelligence application
│ ├── app.py # Streamlit UI
│ ├── pipeline.py # Data generation + insertion
│ ├── gen_fake_data.py # Faker + agent chat generator
│ └── init_chat.sql # Database schema for contacts + messages
│
├── scripts/ # Utility and evaluation scripts
│ ├── eval_pipeline.py # Performance benchmark
│ ├── embed_update_fast.py # Batch embedding (invoices or messages)
│ ├── benchmark_server.py # Server‑mode OCR timing
│ ├── test_llama_server.sh # Quick server test
│ └── ...
│
├── samples/ # Sample invoice images
├── data/ # Large generated files (gitignored)
├── legacy/ # Archived code
├── requirements.txt
├── README.md
└── venv/ # Virtual environment (gitignored)

text

---

## 📦 Prerequisites

- **Docker Desktop** (with ≥ 8 GB RAM allocated)
- **Python 3.12+** and `pip`
- **Git**
- **CMake** and a C++ compiler (for building llama.cpp)
- **huggingface‑hub** (for downloading GGUF models)
- Works on **macOS** and **Windows/WSL2**

---

## 🚀 Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd ocr
2. Create and activate virtual environment
bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
3. Start backing services (Ollama + PostgreSQL)
bash
docker compose up -d ollama postgres
4. Pull the required LLMs into Ollama
bash
docker exec -it ollama ollama pull qwen2.5:1.5b
docker exec -it ollama ollama pull mxbai-embed-large
# optional for better RAG answers:
docker exec -it ollama ollama pull qwen2.5:3b
5. Download the Unlimited‑OCR GGUF model files
bash
mkdir -p ~/uocr
cd ~/uocr
huggingface-cli download sahilchachra/Unlimited-OCR-GGUF \
  Unlimited-OCR-Q4_K_M.gguf mmproj-Unlimited-OCR-F16.gguf \
  --local-dir ./
6. Build llama.cpp with DeepSeek‑OCR support
bash
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp
git fetch origin pull/24975/head:pr24975 && git checkout pr24975
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j $(nproc) --target llama-mtmd-cli
On macOS, replace $(nproc) with $(sysctl -n hw.ncpu).

7. Initialise the databases
bash
# Create all tables (invoices, contacts, messages)
docker exec -i postgres psql -U ocr -d invoices < init.sql
If you only need invoices, use apps/invoice/init_invoice.sql.

8. (Optional) Process sample invoices
bash
cd apps/invoice
./reset.sh -all ../../samples/
🖥️ Usage
Invoice OCR & RAG
bash
streamlit run apps/invoice/app.py
# Open http://localhost:8501
Tabs:

📋 View Database – browse all invoices, check embedding status.

📤 Upload Invoice – upload an image or PDF; the pipeline runs OCR, extracts JSON, inserts into DB, and generates the embedding.

🔍 Search – type a natural‑language question. The system automatically routes to:

Path A (SQL) – for aggregation queries like totals, averages, counts.

Path B (Semantic + RAG) – for open‑ended questions, returning top‑3 invoices and an AI‑generated answer.

CorpChat Intelligence
bash
streamlit run apps/corpchat/app.py
# Open http://localhost:8501
Tabs:

📋 Contacts – view the business‑card contacts.

💬 Messages – browse the last 500 messages.

📊 Overview – database statistics and label distribution.

💬 Chat Viewer – WhatsApp‑style conversation browser (filter by label, search participant).

🔍 Search – semantic search over messages, with RAG answer generation.

Command Line
bash
# Process a single invoice
python3 apps/invoice/pipeline_fast.py -f invoice.jpg

# Process all images/PDFs in a directory
python3 apps/invoice/pipeline_fast.py -d ./my_invoices/

# Hybrid search (CLI)
python3 scripts/hybrid_search.py "total amount for Alibaba" Alibaba 5
⚡ Performance
Metric	Value
OCR per image (warm server)	~12 s
JSON extraction	<1 s
Embedding generation	<0.5 s per invoice (batch mode)
Total pipeline	~15 s per invoice
Throughput (single CPU)	2,400+ invoices per 8‑hour night shift
Scalability	Linear with more CPU cores (parallel mode)
🧪 Evaluation
Run the performance evaluation script to measure OCR, classifier, SQL, and RAG timings on your actual hardware:

bash
python3 scripts/eval_pipeline.py
This script re‑uses your production code, so the numbers are realistic. Use the results to build your cost‑savings analysis and presentation slides.

🔧 Troubleshooting
“model not found” in Ollama → restart the container: docker compose restart ollama

Missing llama‑mtmd‑cli → rebuild llama.cpp from PR #24975 (see setup step 6)

Out of memory → reduce MAX_LONG_EDGE in core/config.py, or use the smaller Q3_K_M model

Import errors → always run scripts from the project root (~/ocr); the path setup handles the rest

Timeouts in RAG → increase num_predict or switch to qwen2.5:1.5b for faster responses

📜 License
This project is licensed under the MIT License.
Model weights are subject to their respective licenses (Baidu Unlimited‑OCR: MIT; Qwen2.5: Apache 2.0).
