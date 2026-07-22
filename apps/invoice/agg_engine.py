#!/usr/bin/env python3
"""
Aggregation Engine — converts natural-language questions into SQL queries,
runs them against PostgreSQL, and returns LLM-rephrased answers.

Used by app.py as Path A of the hybrid query router.
Classification and parameter extraction are handled by core.classifier.
"""
import os, sys, re
import textwrap
from typing import Optional, Tuple

import requests
import psycopg2

# Allow imports from the project root (for core.*)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from core.classifier import is_aggregation_query, extract_vendor_from_query, extract_date_range_from_query
from core.config import OLLAMA_URL, DB_CONFIG

# ── Local config ──
RAG_MODEL = "qwen2.5:1.5b"


# ────────────────────────────────────────────────
# 1. WHERE clause builder
# ────────────────────────────────────────────────

def _validate_date(date_str: str) -> bool:
    """Check that a string is valid YYYY-MM-DD."""
    try:
        from datetime import datetime
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False


def _build_where(vendor_filter=None, date_from=None, date_to=None):
    """Build a SQL WHERE clause from optional filter parameters."""
    conditions = []
    params = []
    if vendor_filter:
        conditions.append("vendor_name ILIKE %s")
        params.append(f"%{vendor_filter}%")
    if date_from and _validate_date(date_from):
        conditions.append("date >= %s")
        params.append(date_from)
    if date_to and _validate_date(date_to):
        conditions.append("date <= %s")
        params.append(date_to)
    return (" AND ".join(conditions) if conditions else "TRUE"), params


# ────────────────────────────────────────────────
# 2. SQL generator — pattern-based routing
# ────────────────────────────────────────────────

def generate_aggregation_sql(
    query: str,
    vendor_filter=None,
    date_from=None,
    date_to=None,
) -> Optional[Tuple[str, list, str]]:
    """Match a natural-language question to a predefined SQL template.

    Returns (sql, params, description) or None if no template matches.
    """
    # Auto-extract filters if not explicitly provided
    if vendor_filter is None:
        vendor_filter = extract_vendor_from_query(query)
    if date_from is None and date_to is None:
        date_from, date_to = extract_date_range_from_query(query)

    where_sql, where_params = _build_where(vendor_filter, date_from, date_to)
    q = query.lower()

    # ── Pattern 0: top N largest ──
    m = re.search(r"top\s+(\d+)\s+(?:largest|biggest)|(\d+)\s+(?:largest|biggest)|(?:largest|biggest)\s+(\d+)", q)
    if m:
        limit = int(m.group(1) or m.group(2))
        sql = f"""
            SELECT invoice_number, date, vendor_name,
                   NULLIF(total_amount, '')::numeric(12,2) AS amount, currency
            FROM invoices WHERE {where_sql}
            ORDER BY NULLIF(total_amount, '')::numeric DESC LIMIT %s
        """
        return sql, where_params + [limit], f"top {limit} largest invoices"

    # ── Pattern 1: highest-spending vendor ──
    if re.search(r"(?:which|what).*vendor.*(?:highest|largest|most)|(?:highest|largest|most).*vendor", q):
        if "total" in q or "amount" in q or "sum" in q:
            sql = f"""
                SELECT vendor_name, SUM(NULLIF(total_amount, '')::numeric)::numeric(12,2) AS total
                FROM invoices WHERE {where_sql}
                GROUP BY vendor_name ORDER BY total DESC LIMIT 1
            """
            return sql, where_params, "highest total amount by vendor"

    # ── Pattern 2: lowest-spending vendor ──
    if re.search(r"(?:which|what).*vendor.*(?:lowest|least|smallest|minimum)|(?:lowest|least|smallest|minimum).*vendor", q):
        if "total" in q or "amount" in q or "sum" in q:
            sql = f"""
                SELECT vendor_name, SUM(NULLIF(total_amount, '')::numeric)::numeric(12,2) AS total
                FROM invoices WHERE {where_sql}
                GROUP BY vendor_name ORDER BY total ASC LIMIT 1
            """
            return sql, where_params, "lowest total amount by vendor"

    # ── Pattern 3: single largest invoice ──
    if re.search(r"(?:largest|biggest|highest)\s+(?:value|amount|total)?\s*invoice\b|"
                 r"\binvoice\b.*(?:largest|biggest|highest)\s+(?:value|amount|total)?", q):
        sql = f"""
            SELECT invoice_number, date, vendor_name,
                   NULLIF(total_amount, '')::numeric(12,2) AS amount, currency
            FROM invoices WHERE {where_sql}
            ORDER BY NULLIF(total_amount, '')::numeric DESC LIMIT 1
        """
        return sql, where_params, "largest invoice"

    # ── Pattern 4: total sum ──
    if re.search(r"\btotal\b|\bsum\b|\bhow much\b", q) and ("total amount" in q or "sum" in q):
        sql = f"""
            SELECT COALESCE(SUM(NULLIF(total_amount, '')::numeric), 0)::numeric(12,2) AS total,
                   COUNT(*) AS invoice_count
            FROM invoices WHERE {where_sql}
        """
        return sql, where_params, "total sum"

    # ── Pattern 5: count ──
    if re.search(r"\bcount\b|\bhow many\b", q):
        if re.search(r"vendor|by vendor|each vendor|per vendor", q):
            sql = f"""
                SELECT vendor_name, COUNT(*) AS cnt
                FROM invoices WHERE {where_sql}
                GROUP BY vendor_name ORDER BY cnt DESC
            """
            return sql, where_params, "count by vendor"
        else:
            sql = f"""
                SELECT COUNT(*) AS total_invoices,
                       COUNT(DISTINCT vendor_name) AS unique_vendors,
                       COALESCE(SUM(NULLIF(total_amount, '')::numeric), 0)::numeric(12,2) AS total_amount
                FROM invoices WHERE {where_sql}
            """
            return sql, where_params, "count summary"

    # ── Pattern 6: average by currency ──
    if re.search(r"average|avg", q) and re.search(r"currency|by currency|per currency", q):
        sql = f"""
            SELECT currency, COUNT(*) AS cnt,
                   AVG(NULLIF(total_amount, '')::numeric)::numeric(12,2) AS avg_amount,
                   SUM(NULLIF(total_amount, '')::numeric)::numeric(12,2) AS total
            FROM invoices WHERE {where_sql}
            GROUP BY currency ORDER BY total DESC
        """
        return sql, where_params, "average by currency"

    # ── Pattern 7: average (general) ──
    if re.search(r"average|avg", q):
        sql = f"""
            SELECT AVG(NULLIF(total_amount, '')::numeric)::numeric(12,2) AS avg_amount,
                   COUNT(*) AS invoice_count
            FROM invoices WHERE {where_sql}
        """
        return sql, where_params, "average amount"

    # ── Pattern 8: summarize / list ──
    if re.search(r"summarize|summary|list|show|breakdown|all\s+(invoice|payment)", q) \
       and not re.search(r"(?:top|largest|biggest|smallest|highest|lowest)\s+\d+|\d+\s+(?:largest|biggest)", q):
        sql = f"""
            SELECT invoice_number, date, vendor_name,
                   NULLIF(total_amount, '')::numeric(12,2) AS amount, currency
            FROM invoices WHERE {where_sql}
            ORDER BY date DESC LIMIT 50
        """
        return sql, where_params, "summarize invoices"

    # ── Fallback: generic listing for any aggregation-typed query ──
    if is_aggregation_query(query):
        sql = f"""
            SELECT invoice_number, date, vendor_name,
                   NULLIF(total_amount, '')::numeric(12,2) AS amount, currency
            FROM invoices WHERE {where_sql}
            ORDER BY date DESC LIMIT 10
        """
        return sql, where_params, "summary (fallback)"

    return None


# ────────────────────────────────────────────────
# 3. Execution + formatting
# ────────────────────────────────────────────────

def _run_sql(sql: str, params: list) -> Tuple[list, list]:
    """Execute a parameterised SQL query and return (rows, column_names)."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()
    return rows, colnames


def _format_result(rows: list, colnames: list, desc: str) -> str:
    """Format SQL result rows into a human-readable summary for the LLM."""
    if not rows:
        return "No matching invoices found in the database."
    if len(rows) == 1 and len(colnames) <= 3:
        parts = [f"{colnames[i]}: {rows[0][i]}" for i in range(len(colnames)) if rows[0][i] is not None]
        return f"[{desc}] " + ", ".join(parts)
    display_rows = rows[:10]
    lines = [f"[{desc}] {len(rows)} results (showing top 10):"]
    for row in display_rows:
        lines.append("  " + " | ".join(str(v) if v is not None else "N/A" for v in row))
    if len(rows) > 10:
        lines.append(f"  ... and {len(rows) - 10} more rows")
    return "\n".join(lines)


# ────────────────────────────────────────────────
# 4. LLM rephraser
# ────────────────────────────────────────────────

def _rephrase_with_llm(question: str, structured_summary: str) -> str:
    """Ask the LLM to turn raw SQL output into a natural-language answer."""
    prompt = textwrap.dedent(f"""\
        You are a helpful assistant. Based on the data below, answer the user's question
        in one or two sentences.

        CRITICAL: Copy all numbers EXACTLY as they appear — do not round, truncate,
        or change digits. Write amounts with commas and two decimals, e.g. 49,965.77.

        User question: {question}

        Data from database:
        {structured_summary}

        Answer:""")

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": RAG_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 128}
            },
            timeout=60
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"(LLM error: {e})\nRaw data:\n{structured_summary}"


# ────────────────────────────────────────────────
# 5. Public API
# ────────────────────────────────────────────────

def handle_aggregation(
    query: str,
    vendor_filter=None,
    date_from=None,
    date_to=None,
) -> Optional[str]:
    """Main entry point for app.py.

    Generates SQL, runs it, formats the result, and rephrases with LLM.
    Returns a natural-language answer string, or None if no SQL template
    matched (caller should fall back to semantic search).
    """
    result = generate_aggregation_sql(query, vendor_filter, date_from, date_to)
    if result is None:
        return None

    sql, params, desc = result
    rows, colnames = _run_sql(sql, params)
    summary = _format_result(rows, colnames, desc)
    return _rephrase_with_llm(query, summary)