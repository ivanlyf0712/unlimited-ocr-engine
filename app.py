#!/usr/bin/env python3
"""
Localhost Streamlit App for Invoice OCR & RAG – full pipeline with RAG answering.
(No JSON schema – robust cleaning only)
"""

import streamlit as st
import subprocess, os, time, io, json, base64, re, requests
import psycopg2
from PIL import Image
import pandas as pd
import networkx as nx
from datetime import datetime
import fitz          # PyMuPDF

# ──────────────────── Configuration ────────────────────
LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-mtmd-cli")
UOCR_MODEL = os.path.expanduser("~/uocr/Unlimited-OCR-Q3_K_M.gguf")
UOCR_MMPROJ = os.path.expanduser("~/uocr/mmproj-Unlimited-OCR-F16.gguf")
MAX_LONG_EDGE = 384
JPEG_QUALITY = 50

OLLAMA_URL = "http://127.0.0.1:11434"
TEXT_MODEL = "qwen2.5:1.5b"
EMBED_MODEL = "mxbai-embed-large"
RAG_MODEL = "qwen2.5:3b"

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "user": "ocr", "password": "***REMOVED***", "dbname": "invoices"
}

JSON_PROMPT = """Return a single JSON object with these keys:
"invoice_number", "date", "vendor_name", "total_amount", "currency".

Rules:
- Use the exact text from the invoice. Do NOT invent or guess any values.
- If a field is missing, set it to "".
- "total_amount" must contain only the number (e.g. "1250.00"), without currency symbol.
- "currency" must be the three‑letter currency code (e.g. "USD").
- Do NOT use nested objects.

Invoice text:
___RAW_TEXT___

JSON:"""

FALLBACK_PROMPT = """Extract these fields from the invoice text.
Do NOT use any of the following words: value, text, string, example, placeholder, xxxx.
Return ONLY a valid JSON object with the keys:
"invoice_number", "date", "vendor_name", "total_amount", "currency".
"total_amount" must be a plain number (e.g. "1250.00").
"currency" must be a three‑letter code (e.g. "USD").
If a field is truly missing, leave it as "".

Invoice text:
___RAW_TEXT___

JSON:"""

# ──────────────────── Helper Functions ────────────────────
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_LONG_EDGE:
        ratio = MAX_LONG_EDGE / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()

def clean_grounding_tags(text: str) -> str:
    """Remove <|det|> ... <|/det|> and other grounding markers, keep only the text."""
    # Remove entire <|det|> block, leaving the text that follows
    cleaned = re.sub(r'<\|det\|>.*?<\|/det\|>', '', text)
    # Remove any leftover grounding special tokens (just in case)
    cleaned = re.sub(r'<\|(?:grounding|ref|det|/det|/ref)\|>', '', cleaned)
    # Collapse multiple blank lines
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    # print(cleaned.strip())
    return cleaned.strip()

def run_uocr(image_path):
    cmd = [LLAMA_CLI, "-m", UOCR_MODEL, "--mmproj", UOCR_MMPROJ,
           "--image", image_path, "-p", "Free OCR.",
           "--chat-template", "deepseek-ocr",
           "--temp", "0", "-c", "2048", "-ngl", "0",
           "--threads", "4", "-n", "384"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    stdout = result.stdout.decode("utf-8", errors="replace")
    skip = ["llama_model_loader","llama_model_load","encode_image",
            "system_info","main:","init:","build:","start:",
            "clip_model","ggml_","warming up","srv","slot","kv_cache"]
    lines = [l.strip() for l in stdout.split("\n") if l.strip() and not any(s in l for s in skip)]
    return "\n".join(lines)


def clean_invoice_data(data: dict) -> dict:
    """Force all fields into the correct format."""
    ta = data.get("total_amount")
    if isinstance(ta, dict):
        amount = ta.get("amount", "")
        curr = ta.get("currency", "")
        data["total_amount"] = f"{float(amount):.2f}" if amount else ""
        if curr and not data.get("currency"):
            data["currency"] = curr
    elif isinstance(ta, str):
        cleaned = re.sub(r'[^\d.]', '', ta.replace(',', '').replace(' ', ''))
        data["total_amount"] = f"{float(cleaned):.2f}" if cleaned else ""
    elif isinstance(ta, (int, float)):
        data["total_amount"] = f"{float(ta):.2f}"
    else:
        data["total_amount"] = ""

    curr = data.get("currency", "")
    if isinstance(curr, str):
        curr = curr.strip().upper()
        match = re.match(r'^([A-Z]{3})', curr)
        data["currency"] = match.group(1) if match else ""
    else:
        data["currency"] = ""

    date_val = data.get("date", "")
    if date_val:
        data["date"] = _normalise_date(str(date_val).strip())
    else:
        data["date"] = ""

    for key in ["invoice_number", "date", "vendor_name", "total_amount", "currency"]:
        if key not in data:
            data[key] = ""
    return data

def _extract_json(prompt_template: str, raw_text: str):
    prompt = prompt_template.replace("___RAW_TEXT___", raw_text)
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": TEXT_MODEL, "prompt": prompt,
        "stream": False, "options": {"temperature": 0}
    }, timeout=60)
    content = resp.json().get("response", "")
    try:
        start = content.index('{')
        end = content.rindex('}') + 1
        data = json.loads(content[start:end])
        data = clean_invoice_data(data)
        return data
    except (ValueError, json.JSONDecodeError):
        return None

def _normalise_date(date_str: str) -> str:
    if not date_str:
        return ""
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', date_str)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{b:02d}-{a:02d}" if a > 12 else f"{y:04d}-{a:02d}-{b:02d}"
    m = re.match(r'^(\d{1,2})[- ]([A-Za-z]{3})[- ](\d{4})$', date_str)
    if m:
        months = {"jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06",
                  "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12"}
        mm = months.get(m.group(2).lower())
        if mm:
            return f"{int(m.group(3)):04d}-{mm}-{int(m.group(1)):02d}"
    m = re.match(r'^(\d{4})/(\d{1,2})/(\d{1,2})$', date_str)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.match(r'^(\d{1,2})\.(\d{1,2})\.(\d{4})$', date_str)
    if m:
        return f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    return date_str

def text_to_json(raw_text):
    return _extract_json(JSON_PROMPT, raw_text)

def text_to_json_fallback(raw_text):
    return _extract_json(FALLBACK_PROMPT, raw_text)

def is_likely_fake(data):
    suspicious = {"value", "text", "string", "example", "placeholder", "xxxx"}
    for field in ["invoice_number", "date", "vendor_name", "total_amount"]:
        val = (data.get(field) or "").strip().lower()
        if val in suspicious:
            return True
    return False

def insert_invoice(fields, raw_text, source_file):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO invoices (invoice_number, date, vendor_name,
                                  total_amount, currency, raw_text, source_file)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            fields.get("invoice_number", ""),
            fields.get("date", ""),
            fields.get("vendor_name", ""),
            fields.get("total_amount", ""),
            fields.get("currency", ""),
            raw_text,
            source_file
        ))
        new_id = cur.fetchone()[0]
        conn.commit()
        return new_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def get_embedding(text: str):
    resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
        "model": EMBED_MODEL, "input": text
    })
    resp.raise_for_status()
    return resp.json()["embeddings"][0]

def update_embedding(row_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT invoice_number, date, vendor_name, total_amount, currency
        FROM invoices WHERE id = %s AND embedding IS NULL
    """, (row_id,))
    row = cur.fetchone()
    if row:
        parts = [str(p) for p in row if p]
        if parts:
            text_to_embed = " ".join(parts)
            vec = get_embedding(text_to_embed)
            cur.execute("UPDATE invoices SET embedding = %s WHERE id = %s", (vec, row_id))
            conn.commit()
    cur.close()
    conn.close()

def fetch_all_invoices():
    conn = get_db_connection()
    df = pd.read_sql("SELECT id, invoice_number, date, vendor_name, total_amount, currency, source_file, created_at FROM invoices ORDER BY created_at DESC", conn)
    conn.close()
    return df

# ──────────────────── Graph Boost ────────────────────
def build_invoice_graph():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, vendor_name, date, total_amount FROM invoices WHERE vendor_name != ''")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    G = nx.Graph()
    for inv_id, vendor, date_str, amount in rows:
        try:
            amt = float(amount) if amount else 0.0
        except ValueError:
            amt = 0.0
        G.add_node(inv_id, vendor=vendor, date=date_str, amount=amt)

    vendor_groups = {}
    for inv_id, vendor, _, _ in rows:
        vendor_groups.setdefault(vendor, []).append(inv_id)
    for ids in vendor_groups.values():
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                G.add_edge(ids[i], ids[j], weight=0.8, rel="same_vendor")

    for i, (id1, _, date1, _) in enumerate(rows):
        try:
            d1 = datetime.strptime(date1, "%Y-%m-%d")
        except (ValueError, TypeError):
            continue
        for id2, _, date2, _ in rows[i+1:]:
            try:
                d2 = datetime.strptime(date2, "%Y-%m-%d")
            except (ValueError, TypeError):
                continue
            if abs((d1 - d2).days) <= 7:
                G.add_edge(id1, id2, weight=0.4, rel="date_proximity")

    for i, (id1, _, _, amt1_raw) in enumerate(rows):
        try:
            amt1 = float(amt1_raw) if amt1_raw else 0.0
        except (ValueError, TypeError):
            amt1 = 0.0
        if amt1 == 0.0:
            continue
        for id2, _, _, amt2_raw in rows[i+1:]:
            try:
                amt2 = float(amt2_raw) if amt2_raw else 0.0
            except (ValueError, TypeError):
                amt2 = 0.0
            if amt2 == 0.0:
                continue
            max_amt = max(amt1, amt2)
            if max_amt == 0:
                continue
            diff = abs(amt1 - amt2) / max_amt
            if diff <= 0.1:
                G.add_edge(id1, id2, weight=0.3, rel="amount_similarity")

    return G

def search_similar(query, vendor_filter=None, top_k=5):
    query_vec = get_embedding(query)
    conn = get_db_connection()
    cur = conn.cursor()

    if vendor_filter:
        cur.execute("""
            SELECT id, invoice_number, date, vendor_name, total_amount, currency,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM invoices
            WHERE vendor_name ILIKE %s AND embedding IS NOT NULL
            ORDER BY similarity DESC
            LIMIT %s
        """, (query_vec, f"%{vendor_filter}%", top_k * 3))
    else:
        cur.execute("""
            SELECT id, invoice_number, date, vendor_name, total_amount, currency,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM invoices
            WHERE embedding IS NOT NULL
            ORDER BY similarity DESC
            LIMIT %s
        """, (query_vec, top_k * 3))

    vector_results = cur.fetchall()
    cur.close()
    conn.close()

    if not vector_results:
        return []

    G = build_invoice_graph()
    ids = [r[0] for r in vector_results]
    boosted = []
    for r in vector_results:
        inv_id = r[0]
        sim = r[-1]
        boost = 0.0
        count = 0
        for other_id in ids:
            if other_id == inv_id:
                continue
            try:
                if G.has_node(inv_id) and G.has_node(other_id):
                    dist = nx.shortest_path_length(G, source=inv_id, target=other_id)
                    boost += 1.0 / (dist + 1)
                    count += 1
            except nx.NetworkXNoPath:
                pass
        if count > 0:
            boost /= count
        combined = 0.7 * sim + 0.3 * boost
        boosted.append((combined, r))

    boosted.sort(key=lambda x: x[0], reverse=True)

    final = []
    for _, r in boosted[:top_k]:
        final.append(r)   # keep id as first element
    return final

# ──────────────────── PDF Support ────────────────────
def pdf_to_images(pdf_bytes, source_filename):
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

# ──────────────────── Streamlit UI ────────────────────
st.set_page_config(page_title="Invoice OCR & RAG", layout="wide")
st.title("📄 Unlimited‑OCR & RAG Dashboard")

tab1, tab2, tab3 = st.tabs(["📋 View Database", "📤 Upload Invoice", "🔍 Search"])

with tab1:
    st.subheader("Invoices Table")
    df = fetch_all_invoices()
    st.dataframe(df, use_container_width=True)

with tab2:
    st.subheader("Upload Invoice Image or PDF")
    uploaded_file = st.file_uploader("Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"])
    if uploaded_file is not None:
        st.write(f"File: **{uploaded_file.name}**")
        if st.button("Process Invoice"):
            with st.spinner("Running OCR pipeline (this may take ~15 seconds per page)..."):
                if uploaded_file.name.lower().endswith(".pdf"):
                    pdf_bytes = uploaded_file.getvalue()
                    pages = pdf_to_images(pdf_bytes, uploaded_file.name)
                    st.write(f"PDF has {len(pages)} pages. Processing each page...")
                    for img_path, source_name in pages:
                        raw_text = run_uocr(img_path)
                        data = text_to_json(raw_text)
                        if data and is_likely_fake(data):
                            data = text_to_json_fallback(raw_text)
                        if data is None or is_likely_fake(data):
                            data = {}
                        new_id = insert_invoice(data, raw_text, source_name)
                        update_embedding(new_id)
                        os.remove(img_path)
                    st.success(f"Processed {len(pages)} pages from PDF.")
                else:
                    tmp_path = f"/tmp/{uploaded_file.name}"
                    with open(tmp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    t0 = time.time()
                    raw_text = run_uocr(tmp_path)
                    t1 = time.time()
                    st.write(f"⏱ OCR: {t1-t0:.1f}s")

                    data = text_to_json(raw_text)
                    if data and is_likely_fake(data):
                        st.warning("First attempt returned placeholder data – retrying with stricter prompt...")
                        data = text_to_json_fallback(raw_text)

                    t2 = time.time()
                    st.write(f"⏱ JSON parse: {t2-t1:.1f}s")

                    if data is None or is_likely_fake(data):
                        st.error("Could not extract valid JSON – raw text will be stored.")
                        data = {}
                    else:
                        st.json(data)

                    new_id = insert_invoice(data, raw_text, uploaded_file.name)
                    st.success(f"Inserted with ID {new_id}")

                    with st.spinner("Generating embedding..."):
                        update_embedding(new_id)
                    st.success("Embedding generated.")

                    st.write(f"🕐 Total: {time.time()-t0:.1f}s")
                    os.remove(tmp_path)

with tab3:
    st.subheader("Semantic Search over Invoices")

    # ──────── Initialize session state for search/answer persistence ────────
    if 'search_history' not in st.session_state:
        st.session_state['search_history'] = []            # list of dicts: {query, answer, timestamp}
    if 'current_answer' not in st.session_state:
        st.session_state['current_answer'] = None          # answer for the current search results

    query = st.text_input("Search query", placeholder="e.g., total amount for Alibaba invoices")
    vendor_filter = st.text_input("Vendor filter (optional)", placeholder="Alibaba")
    top_k = st.slider("Number of results", 1, 20, 5)

    col1, col2 = st.columns([1, 1])
    with col1:
        search_clicked = st.button("Search")
    with col2:
        generate_clicked = st.button("💬 Generate Answer from Top Results", key="rag_answer_btn")

    if search_clicked:
        if not query:
            st.warning("Please enter a query.")
        else:
            st.session_state['current_answer'] = None   # reset answer when new search is done
            with st.spinner("Searching..."):
                results = search_similar(query, vendor_filter if vendor_filter else None, top_k)
                st.session_state['search_results'] = results
                st.session_state['search_query'] = query

    # ──────── Display search results (persistent via session state) ────────
    if 'search_results' in st.session_state and st.session_state['search_results'] is not None:
        results = st.session_state['search_results']
        if results:
            st.write(f"Top {len(results)} results for: **{st.session_state.get('search_query', '')}**")
            df_res = pd.DataFrame(
                [r[1:] for r in results],
                columns=["Invoice Number", "Date", "Vendor", "Amount", "Currency", "Similarity"]
            )
            df_res["Similarity"] = df_res["Similarity"].apply(lambda x: f"{x:.4f}")
            st.table(df_res)
        else:
            st.info("No matching invoices found.")

    # ──────── Generate Answer (always visible, persists across re-searches via history) ────────
    if generate_clicked:
        if 'search_results' not in st.session_state or not st.session_state.get('search_results'):
            st.warning("Please run a search first before generating an answer.")
        else:
            results = st.session_state['search_results']
            query_text = st.session_state.get('search_query', '')
            context_parts = []
            with get_db_connection() as conn:
                cur = conn.cursor()
                for r in results[:3]:
                    inv_id = r[0]
                    cur.execute("SELECT raw_text FROM invoices WHERE id = %s", (inv_id,))
                    raw_row = cur.fetchone()
                    if raw_row and raw_row[0]:
                        context_parts.append(raw_row[0])
                cur.close()
            if not context_parts:
                st.warning("No raw text available for these invoices.")
            else:
                context = "\n\n".join(context_parts)
                rag_prompt = f"""請根據以下發票內容回答問題。如果無法回答，請說「資料不足」。
問題：{query_text}
發票內容：{context}
答案："""

                with st.spinner("🧠 Generating answer... (may take 15‑30 seconds)"):
                    try:
                        t_start = time.time()
                        resp = requests.post(
                            f"{OLLAMA_URL}/api/generate",
                            json={
                                "model": RAG_MODEL,
                                "prompt": rag_prompt,
                                "stream": False,
                                "options": {"temperature": 0.1, "num_predict": 256}
                            },
                            timeout=120
                        )
                        elapsed = time.time() - t_start
                        if resp.status_code == 200:
                            answer = resp.json().get("response", "").strip()
                            if answer:
                                st.session_state['current_answer'] = answer
                                # Add to persistent search history log
                                st.session_state['search_history'].insert(0, {
                                    "query": query_text,
                                    "answer": answer,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                })
                            else:
                                st.error("The model returned an empty response.")
                        else:
                            st.error(f"RAG model error (HTTP {resp.status_code})")
                            st.write(resp.text[:500])
                    except requests.exceptions.Timeout:
                        st.error("Request timed out – the model may be overloaded or the prompt too long.")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

    # ──────── Display current answer (persists even after new searches) ────────
    if st.session_state.get('current_answer'):
        st.divider()
        st.subheader("💬 Current Answer")
        st.success(st.session_state['current_answer'])

    # ──────── Search History Log ────────
    if st.session_state['search_history']:
        st.divider()
        st.subheader("📜 Search & Answer History")
        for idx, entry in enumerate(st.session_state['search_history']):
            with st.expander(f"Q{idx+1}: {entry['query']}  —  {entry['timestamp']}"):
                st.markdown(f"**Answer:**\n\n{entry['answer']}")
        if st.button("Clear History"):
            st.session_state['search_history'] = []
            st.session_state['current_answer'] = None
            st.rerun()
