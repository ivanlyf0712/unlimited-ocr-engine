#!/usr/bin/env python3
"""
Performance Evaluation Script – Invoice OCR & RAG Pipeline

Measures:
  1. OCR + JSON extraction speed on sample images
  2. Hybrid classifier latency
  3. Path A (aggregation SQL) latency
  4. Path B (semantic search + RAG) latency
  5. Overall response time for a set of representative queries

Requirements:
  - All services running (ollama, postgres, llama-server if using server‑mode OCR)
  - Sample invoice images in samples/ folder
"""

import sys
import os
import time
import json
import statistics
from typing import List, Dict, Tuple

# Ensure the project root is on the path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Also add the invoice app directory, where agg_engine.py sits
INVOICE_DIR = os.path.join(ROOT_DIR, "apps", "invoice")
if INVOICE_DIR not in sys.path:
    sys.path.insert(0, INVOICE_DIR)
    
# ── Core imports (mirrors the app) ──
from core.ocr import run_ocr, clean_grounding_tags
from core.extraction import text_to_json, text_to_json_fallback, is_likely_fake
from core.config import (
    OLLAMA_URL, RAG_MODEL, EMBED_MODEL, DB_CONFIG
)
from core.db import get_db_connection, search_similar, get_embedding
from core.classifier import classify_hybrid
from agg_engine import generate_aggregation_sql, handle_aggregation, _run_sql

# ── Test configuration ──
TEST_IMAGES = [
    "samples/sample_invoice.jpg",
    "samples/sample_INV-2024-0720.jpg",
    "samples/sample_INV-2024-0721.jpg",
    # add more if desired
]

# Representative queries: (query, vendor_filter, expected_intent)
AGGREGATION_QUERIES = [
    ("What is the total amount for Alibaba invoices?", "Alibaba"),
    ("Which vendor has the highest total invoice amount?", None),
    ("How many invoices are in the database?", None),
]

SEMANTIC_QUERIES = [
    ("Find invoices about consulting services", None),
    ("Show invoices related to software licenses", None),
]

RAG_QUERIES = [
    ("What was the total amount of the Alibaba invoice INV-2024-0720?", "Alibaba"),
]

# ── Helper to time a callable ──
def timed_call(fn, *args, **kwargs) -> Tuple[float, any]:
    start = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - start
    return elapsed, result

# ═══════════════════════════════════════════════════════════
# 1. OCR + JSON Extraction Benchmark
# ═══════════════════════════════════════════════════════════
def benchmark_ocr():
    print("\n" + "="*60)
    print("OCR + JSON Extraction Benchmark")
    print("="*60)
    ocr_times = []
    json_times = []
    for img in TEST_IMAGES:
        if not os.path.exists(img):
            print(f"  SKIP: {img} not found")
            continue
        # OCR
        elapsed, raw = timed_call(run_ocr, img)
        raw = clean_grounding_tags(raw)
        ocr_times.append(elapsed)
        # JSON extraction
        e2, data = timed_call(text_to_json, raw)
        json_times.append(e2)
        valid = "✓" if data and not is_likely_fake(data) else "✗"
        print(f"  {img:40s} OCR: {elapsed:.1f}s  JSON: {e2:.1f}s  valid={valid}")
    if ocr_times:
        print(f"\n  Avg OCR: {statistics.mean(ocr_times):.1f}s  "
              f"Avg JSON: {statistics.mean(json_times):.1f}s")

# ═══════════════════════════════════════════════════════════
# 2. Classifier Benchmark
# ═══════════════════════════════════════════════════════════
def benchmark_classifier():
    print("\n" + "="*60)
    print("Hybrid Classifier Benchmark")
    print("="*60)
    all_queries = AGGREGATION_QUERIES + SEMANTIC_QUERIES
    times = []
    for q, vendor in all_queries:
        e, cls = timed_call(classify_hybrid, q)
        times.append(e)
        intent = cls.get("intent", "?")
        method = cls.get("method", "?")
        print(f"  {q[:50]:50s} {e:.2f}s  intent={intent:12s} method={method}")
    if times:
        print(f"\n  Avg classify time: {statistics.mean(times):.2f}s")

# ═══════════════════════════════════════════════════════════
# 3. Path A – Aggregation SQL Benchmark
# ═══════════════════════════════════════════════════════════
def benchmark_path_a():
    print("\n" + "="*60)
    print("Path A: Aggregation SQL Benchmark")
    print("="*60)
    for q, vendor in AGGREGATION_QUERIES:
        # Time SQL generation + execution
        e_sql, result = timed_call(
            generate_aggregation_sql, q,
            vendor_filter=vendor, date_from=None, date_to=None
        )
        if result is None:
            print(f"  {q[:50]:50s}  No template matched (fallback to semantic)")
            continue
        sql, params, desc = result
        e_run, (rows, cols) = timed_call(_run_sql, sql, params)
        # Time LLM rephrasing
        e_llm, answer = timed_call(
            handle_aggregation, q,
            vendor_filter=vendor, date_from=None, date_to=None
        )
        print(f"  {q[:50]:50s}  SQL gen: {e_sql:.2f}s  exec: {e_run:.2f}s  rephrase: {e_llm:.2f}s  "
              f"rows={len(rows)}  answer='{answer[:60]}...'")
    # also show a full timed call from handle_aggregation (which does all steps)
    print("\n  Full handle_aggregation timings:")
    for q, vendor in AGGREGATION_QUERIES[:1]:   # just one example
        e_full, answer = timed_call(
            handle_aggregation, q,
            vendor_filter=vendor, date_from=None, date_to=None
        )
        print(f"    {q[:50]:50s}  total={e_full:.2f}s")

# ═══════════════════════════════════════════════════════════
# 4. Path B – Semantic Search + RAG Benchmark
# ═══════════════════════════════════════════════════════════
def benchmark_path_b():
    print("\n" + "="*60)
    print("Path B: Semantic Search + RAG Benchmark")
    print("="*60)
    # Helper to run RAG answer (copy from app logic)
    def rag_answer(query, results):
        if not results:
            return "No results"
        context_parts = []
        with get_db_connection() as conn:
            cur = conn.cursor()
            for r in results[:3]:
                cur.execute("SELECT raw_text FROM invoices WHERE id = %s", (r[0],))
                raw_row = cur.fetchone()
                if raw_row and raw_row[0]:
                    raw = raw_row[0][:2500]  # truncate
                    context_parts.append(raw)
            cur.close()
        if not context_parts:
            return "No OCR text"
        context = "\n\n".join(context_parts)
        prompt = f"""請根據以下發票內容回答問題。如果無法回答，請說「資料不足」。
問題：{query}
發票內容：{context}
答案："""
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": RAG_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 256}
            },
            timeout=120
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        return f"Error {resp.status_code}"

    for q, vendor in RAG_QUERIES + SEMANTIC_QUERIES:
        # Step 1: Search
        e_search, results = timed_call(
            search_similar, q,
            vendor_filter=vendor, top_k=5
        )
        # Step 2: RAG answer
        if results:
            e_rag, answer = timed_call(rag_answer, q, results)
            print(f"  {q[:50]:50s}  search: {e_search:.2f}s  RAG: {e_rag:.2f}s  "
                  f"answer='{answer[:60]}...'")
        else:
            print(f"  {q[:50]:50s}  no results found")

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import requests   # needed for RAG path
    print("🔧 Invoice OCR & RAG Performance Evaluation")
    print("  Make sure ollama, postgres, and OCR server are running.")
    benchmark_ocr()
    benchmark_classifier()
    benchmark_path_a()
    benchmark_path_b()
    print("\n✅ Evaluation complete.")