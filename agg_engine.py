#!/usr/bin/env python3
"""
Aggregation Engine — converts natural language questions into SQL queries,
runs them against PostgreSQL, and returns LLM-rephrased answers.

Import into app.py for the hybrid query router.

Usage from app.py:
    from agg_engine import is_aggregation_query, handle_aggregation

    if is_aggregation_query(query):
        answer = handle_aggregation(query, vendor_filter, date_from, date_to)
    else:
        # fall through to existing vector search + RAG pipeline
"""
import re
import textwrap
import calendar
from datetime import datetime
from typing import Optional, Tuple, List

import requests
import psycopg2

# ── Config ──
OLLAMA_URL = "http://127.0.0.1:11434"
RAG_MODEL = "qwen2.5:1.5b"

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "user": "ocr", "password": "***REMOVED***", "dbname": "invoices"
}

# ────────────────────────────────────────────────
# 1. INTENT CLASSIFIER
# ────────────────────────────────────────────────
AGGREGATION_KEYWORDS = [
    r"\btotal\b", r"\bsum\b", r"\baverage\b", r"\bavg\b",
    r"\bhighest\b", r"\blowest\b", r"\bmaximum\b", r"\bminimum\b",
    r"\bcount\b", r"\bhow many\b", r"\bhow much\b",
    r"\bgroup by\b", r"\bper\b", r"\beach\b",
    r"\bsummarize\b", r"\bsummary\b", r"\bbreakdown\b",
    r"\blargest\b", r"\bsmallest\b", r"\btop\b",
    r"总金额", r"平均", r"最高", r"最低", r"最多", r"最少",
    r"汇总", r"统计", r"数量", r"多少", r"有几个", r"排名",
]

AGG_CONTEXT = re.compile(
    r"(amount|invoice|payment|vendor|currency|total|sum|avg|count)",
    re.IGNORECASE
)

AGG_PATTERN = re.compile("|".join(AGGREGATION_KEYWORDS), re.IGNORECASE)


def is_aggregation_query(query: str) -> bool:
    """Return True if the query likely needs SQL aggregation instead of vector search."""
    if not AGG_PATTERN.search(query):
        return False
    return bool(AGG_CONTEXT.search(query))


# ────────────────────────────────────────────────
# 2. EXTRACTORS
# ────────────────────────────────────────────────
_vendor_cache: Optional[List[str]] = None


def _get_known_vendors() -> List[str]:
    global _vendor_cache
    if _vendor_cache is not None:
        return _vendor_cache
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT vendor_name FROM invoices WHERE vendor_name != ''")
        _vendor_cache = [r[0] for r in cur.fetchall()]
        cur.close()
        conn.close()
    except Exception:
        _vendor_cache = []
    return _vendor_cache


def extract_vendor_from_query(query: str) -> Optional[str]:
    known = _get_known_vendors()
    if not known:
        return None
    candidates = []
    m = re.search(r'(?:for|to|from)\s+([A-Za-z0-9\s\.\-]+?)(?:\s+in\s|\s+Q\d|\s+\d{4}|\s*$)', query, re.IGNORECASE)
    if m:
        candidates.append(m.group(1).strip())
    m = re.search(r'([A-Za-z0-9\s\.\-]+?)\s+(?:invoices|payments|transactions)', query, re.IGNORECASE)
    if m:
        candidates.append(m.group(1).strip())
    for candidate in candidates:
        if len(candidate) < 3:
            continue
        for db_vendor in known:
            if candidate.lower() in db_vendor.lower():
                return db_vendor
    return None


def extract_date_range_from_query(query: str) -> Tuple[Optional[str], Optional[str]]:
    q = query.lower()
    m = re.search(r'between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})', q)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r'(?:in|of)\s+(\d{4})', q)
    if m:
        year = m.group(1)
        return f"{year}-01-01", f"{year}-12-31"
    m = re.search(r'q([1-4])\s*(\d{4})', q)
    if m:
        quarter, year = int(m.group(1)), m.group(2)
        month_start = (quarter - 1) * 3 + 1
        month_end = month_start + 2
        day_end = calendar.monthrange(int(year), month_end)[1]
        return (f"{year}-{month_start:02d}-01", f"{year}-{month_end:02d}-{day_end}")
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    m = re.search(r'(' + '|'.join(months.keys()) + r')\s*(\d{4})', q)
    if m:
        month_name, year = m.group(1), m.group(2)
        month = months[month_name]
        day_end = calendar.monthrange(int(year), month)[1]
        return (f"{year}-{month:02d}-01", f"{year}-{month:02d}-{day_end}")
    if "last year" in q:
        this_year = datetime.now().year
        year = str(this_year - 1)
        return f"{year}-01-01", f"{year}-12-31"
    return None, None


def _validate_date(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


# ────────────────────────────────────────────────
# 3. SQL GENERATOR
# ────────────────────────────────────────────────
def _build_where(vendor_filter=None, date_from=None, date_to=None):
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


def generate_aggregation_sql(query: str, vendor_filter=None,
                              date_from=None, date_to=None) -> Optional[Tuple[str, list, str]]:
    if vendor_filter is None:
        vendor_filter = extract_vendor_from_query(query)
    if date_from is None and date_to is None:
        date_from, date_to = extract_date_range_from_query(query)

    where_sql, where_params = _build_where(vendor_filter, date_from, date_to)
    q = query.lower()

    # Pattern 8 (first): top N largest
    m = re.search(r"top\s+(\d+)\s+(?:largest|biggest)|(\d+)\s+(?:largest|biggest)|(?:largest|biggest)\s+(\d+)", q)
    if m:
        limit = int(m.group(1) or m.group(2))
        sql = f"""
            SELECT invoice_number, date, vendor_name,
                   total_amount::numeric::numeric(12,2) AS amount, currency
            FROM invoices WHERE {where_sql}
            ORDER BY total_amount::numeric DESC LIMIT %s
        """
        return sql, where_params + [limit], f"top {limit} largest invoices"

    # Pattern 1: highest vendor
    if re.search(r"(?:which|what).*vendor.*(?:highest|largest|most)|(?:highest|largest|most).*vendor", q):
        if "total" in q or "amount" in q or "sum" in q:
            sql = f"""
                SELECT vendor_name, SUM(total_amount::numeric)::numeric(12,2) AS total
                FROM invoices WHERE {where_sql}
                GROUP BY vendor_name ORDER BY total DESC LIMIT 1
            """
            return sql, where_params, "highest total amount by vendor"

    # Pattern 2: lowest vendor
    if re.search(r"(?:which|what).*vendor.*(?:lowest|least|smallest|minimum)|(?:lowest|least|smallest|minimum).*vendor", q):
        if "total" in q or "amount" in q or "sum" in q:
            sql = f"""
                SELECT vendor_name, SUM(total_amount::numeric)::numeric(12,2) AS total
                FROM invoices WHERE {where_sql}
                GROUP BY vendor_name ORDER BY total ASC LIMIT 1
            """
            return sql, where_params, "lowest total amount by vendor"

    # Pattern 2b: largest / highest invoice (single invoice, not aggregated)
    # Matches: "largest invoice", "highest value invoice", "biggest invoice from X", etc.
    if re.search(r"(?:largest|biggest|highest)\s+(?:value|amount|total)?\s*invoice\b|"
                 r"\binvoice\b.*(?:largest|biggest|highest)\s+(?:value|amount|total)?", q):
        sql = f"""
            SELECT invoice_number, date, vendor_name,
                   total_amount::numeric::numeric(12,2) AS amount, currency
            FROM invoices WHERE {where_sql}
            ORDER BY total_amount::numeric DESC LIMIT 1
        """
        return sql, where_params, "largest invoice"

    # Pattern 3: total sum
    if re.search(r"\btotal\b|\bsum\b|\bhow much\b", q) and ("total amount" in q or "sum" in q):
        sql = f"""
            SELECT COALESCE(SUM(total_amount::numeric), 0)::numeric(12,2) AS total,
                   COUNT(*) AS invoice_count
            FROM invoices WHERE {where_sql}
        """
        return sql, where_params, "total sum"

    # Pattern 4: count
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
                       COALESCE(SUM(total_amount::numeric), 0)::numeric(12,2) AS total_amount
                FROM invoices WHERE {where_sql}
            """
            return sql, where_params, "count summary"

    # Pattern 5: average by currency
    if re.search(r"average|avg", q) and re.search(r"currency|by currency|per currency", q):
        sql = f"""
            SELECT currency, COUNT(*) AS cnt,
                   AVG(total_amount::numeric)::numeric(12,2) AS avg_amount,
                   SUM(total_amount::numeric)::numeric(12,2) AS total
            FROM invoices WHERE {where_sql}
            GROUP BY currency ORDER BY total DESC
        """
        return sql, where_params, "average by currency"

    # Pattern 6: average (general)
    if re.search(r"average|avg", q):
        sql = f"""
            SELECT AVG(total_amount::numeric)::numeric(12,2) AS avg_amount,
                   COUNT(*) AS invoice_count
            FROM invoices WHERE {where_sql}
        """
        return sql, where_params, "average amount"

    # Pattern 7: summarize / list
    if re.search(r"summarize|summary|list|show|breakdown|all\s+(invoice|payment)", q) \
       and not re.search(r"(?:top|largest|biggest|smallest|highest|lowest)\s+\d+|\d+\s+(?:largest|biggest)", q):
        sql = f"""
            SELECT invoice_number, date, vendor_name,
                   total_amount::numeric::numeric(12,2) AS amount, currency
            FROM invoices WHERE {where_sql}
            ORDER BY date DESC LIMIT 50
        """
        return sql, where_params, "summarize invoices"

    # Fallback
    if is_aggregation_query(query):
        sql = f"""
            SELECT invoice_number, date, vendor_name,
                   total_amount::numeric::numeric(12,2) AS amount, currency
            FROM invoices WHERE {where_sql}
            ORDER BY date DESC LIMIT 10
        """
        return sql, where_params, "summary (fallback)"

    return None


# ────────────────────────────────────────────────
# 4. EXECUTION + FORMATTING
# ────────────────────────────────────────────────
def _run_sql(sql: str, params: list) -> Tuple[list, list]:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()
    return rows, colnames


def _format_result(rows: list, colnames: list, desc: str) -> str:
    if not rows:
        return "No matching invoices found in the database."
    if len(rows) == 1 and len(colnames) <= 3:
        parts = [f"{colnames[i]}: {rows[0][i]}" for i in range(len(colnames)) if rows[0][i] is not None]
        return f"[{desc}] " + ", ".join(parts)
    lines = [f"[{desc}] {len(rows)} results:"]
    for row in rows:
        lines.append("  " + " | ".join(str(v) if v is not None else "N/A" for v in row))
    return "\n".join(lines)


# ────────────────────────────────────────────────
# 5. LLM REPHRASER
# ────────────────────────────────────────────────
def _rephrase_with_llm(question: str, structured_summary: str) -> str:
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
# 6. PUBLIC API — the one function app.py calls
# ────────────────────────────────────────────────
def handle_aggregation(query: str, vendor_filter=None,
                       date_from=None, date_to=None) -> str:
    """
    Main entry point for app.py.
    Detects intent, generates SQL, runs it, formats result, rephrases with LLM.
    Returns a natural language answer string.
    """
    result = generate_aggregation_sql(query, vendor_filter, date_from, date_to)
    if result is None:
        return None  # caller should fall back to vector search

    sql, params, desc = result
    rows, colnames = _run_sql(sql, params)
    summary = _format_result(rows, colnames, desc)
    return _rephrase_with_llm(query, summary)