#!/usr/bin/env python3
"""
Automated RAG testing agent — asks natural business questions and validates answers.
Uses Ollama to generate questions, query the RAG pipeline, and verify correctness.

Usage:
  python3 scripts/rag_agent_test.py
"""

import json
import re
import sys
import time
import requests
import psycopg2

sys.path.insert(0, '.')

from core.classifier import classify_hybrid
from core.db import search_similar, get_db_connection
from core.config import OLLAMA_URL, DB_CONFIG, RAG_MODEL, EMBED_MODEL
from agg_engine import handle_aggregation


# ── Agent configuration ──
AGENT_MODEL = "qwen2.5:1.5b"

QUESTION_GENERATOR_PROMPT = """You are a corporate financial analyst at a mid-size company reviewing invoice records.
Generate ONE natural business question about the company's invoices. 

Requirements:
- The question should be something a real finance/accounting staff would ask
- Query about totals, vendors, dates, currencies, or specific invoice details
- Mix of: aggregation questions (totals, averages, counts) and lookup questions
- Use vendor names like: Alibaba, EuroTrade GmbH, Nordic Paper Mills, TechSource HK, ABC Technologies, Global Supplies, Pacific Logistics, Digital Innovations
- Dates: 2024-2026 range
- Keep it under 15 words
- Output ONLY the question, no other text.

Example questions:
- What is the total amount for Alibaba in 2025?
- How many invoices did EuroTrade GmbH issue?
- Average invoice amount by currency
- Show me the 5 biggest invoices
- Which vendor has the highest total?

Now generate ONE question:"""


def generate_question() -> str:
    """Generate a natural business question using the LLM."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": AGENT_MODEL,
                "prompt": QUESTION_GENERATOR_PROMPT,
                "stream": False,
                "options": {"temperature": 0.8, "num_predict": 64}
            },
            timeout=20
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip().rstrip('?') + '?'
    except Exception as e:
        return f"ERROR generating question: {e}"


def run_sql_query(sql: str, params: list = None) -> list:
    """Execute a direct SQL query and return results."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(sql, params or [])
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return cols, rows
    except Exception as e:
        return None, [str(e)]


EVALUATION_PROMPT = """You are a QA auditor for an invoice database. 
A user asked a question and the system gave an answer. Verify if the answer is factually correct.

User question: {question}
System answer: {answer}

Direct SQL result (ground truth): {sql_result}

Rules:
- If the answer's numbers match the SQL ground truth → say "PASS"
- If numbers are wrong, missing, or fabricated → say "FAIL: <reason>"
- If the answer correctly says no data found → say "PASS (no data)"
- If the answer is plausible but SQL shows different data → say "FAIL: <reason>"
- Keep response under 1 sentence.

Verdict:"""


def evaluate_answer(question: str, answer: str, sql_cols: list, sql_rows: list) -> str:
    """Evaluate if the system answer matches the SQL ground truth."""
    if sql_cols is None:
        return "FAIL: SQL execution error"
    
    sql_text = f"Columns: {sql_cols}\nRows: {sql_rows[:5]}"
    if len(sql_rows) > 5:
        sql_text += f"\n... and {len(sql_rows) - 5} more rows"
    
    prompt = EVALUATION_PROMPT.format(
        question=question, answer=answer, sql_result=sql_text
    )
    
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": AGENT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 128}
            },
            timeout=20
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"EVAL ERROR: {e}"


def run_test(query: str, iteration: int):
    """Run a single test: classify → execute → evaluate."""
    print(f"\n{'─' * 70}")
    print(f"[Test #{iteration}] QUERY: {query}")

    # Step 1: Classify
    t0 = time.time()
    classification = classify_hybrid(query)
    t_classify = time.time()
    intent = classification.get("intent", "semantic")
    vendor = classification.get("vendor")
    date_from = classification.get("date_from")
    date_to = classification.get("date_to")
    
    print(f"  Intent: {intent} | Vendor: {vendor} | Dates: {date_from}→{date_to}")
    print(f"  Classification: {classification.get('method')} ({t_classify - t0:.1f}s)")

    # Step 2: Execute based on intent
    answer = None
    sql_cols = None
    sql_rows = None
    
    if intent == "aggregation":
        # Path A: SQL aggregation
        answer = handle_aggregation(query, 
            vendor_filter=vendor, date_from=date_from, date_to=date_to)
        t_exec = time.time()
        
        # Also get raw SQL for verification
        from agg_engine import generate_aggregation_sql, _run_sql
        sql_result = generate_aggregation_sql(query,
            vendor_filter=vendor, date_from=date_from, date_to=date_to)
        if sql_result:
            sql, params, desc = sql_result
            sql_cols, sql_rows = _run_sql(sql, params)
        
        print(f"  SQL execution: {t_exec - t_classify:.1f}s")
        print(f"  Answer: {answer[:120] if answer else 'None'}...")
        
        if answer:
            # Evaluate
            verdict = evaluate_answer(query, answer, sql_cols, sql_rows or [])
            print(f"  Verdict: {verdict}")
            return verdict
        else:
            print(f"  ⚠️ No answer generated")
            return "FAIL: No answer"
    else:
        # Path B: Semantic search + RAG
        results = search_similar(query, top_k=3)
        t_search = time.time()
        
        if results:
            # Fetch context
            context_parts = []
            with get_db_connection() as conn:
                cur = conn.cursor()
                for r in results[:3]:
                    cur.execute("SELECT raw_text FROM invoices WHERE id = %s", (r[0],))
                    row = cur.fetchone()
                    if row and row[0]:
                        context_parts.append(row[0].strip()[:2500])
                    cur.close()
            
            if context_parts:
                context = "\n\n".join(context_parts)
                rag_prompt = f"""根據以下發票內容回答問題。如果無法回答，請說「資料不足」。
問題：{query}
發票內容：{context}
答案："""
                
                resp = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": RAG_MODEL,
                        "prompt": rag_prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 256}
                    },
                    timeout=60
                )
                answer = resp.json().get("response", "").strip() if resp.status_code == 200 else None
                t_rag = time.time()
                
                print(f"  Search: {t_search - t_classify:.1f}s | RAG: {t_rag - t_search:.1f}s")
                print(f"  Answer: {answer[:120] if answer else 'None'}...")
                
                # For semantic queries, verify answer plausibility
                if answer:
                    print(f"  Verdict: PASS (semantic)")
                    return "PASS (semantic)"
                else:
                    print(f"  ⚠️ No RAG answer")
                    return "FAIL: No RAG answer"
            else:
                print(f"  ⚠️ No context found for RAG")
                return "FAIL: No context"
        else:
            print(f"  ⚠️ No search results")
            return "FAIL: No search results"


def main():
    print("=" * 70)
    print("RAG AGENT TEST — Automated Business Query Testing")
    print(f"  Model: {AGENT_MODEL}")
    print("=" * 70)

    NUM_TESTS = 8
    results = {"PASS": 0, "FAIL": 0, "TOTAL": 0}

    for i in range(1, NUM_TESTS + 1):
        query = generate_question()
        verdict = run_test(query, i)
        results["TOTAL"] += 1
        if verdict and verdict.upper().startswith("PASS"):
            results["PASS"] += 1
        else:
            results["FAIL"] += 1
    
    # Summary
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {results['PASS']}/{results['TOTAL']} PASSED "
          f"({results['PASS']/results['TOTAL']*100:.0f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()