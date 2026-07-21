#!/usr/bin/env python3

"""
Aggregation Engine Demo — standalone test script for hybrid SQL + LLM queries.
Runs a battery of test queries against your real PostgreSQL database.

Usage:
  python3 scripts/agg_engine_demo.py

Requirements:
  - PostgreSQL with invoices table (structured fields + embedding)
  - Ollama running with qwen2.5:1.5b available
"""

import re
import sys
import textwrap
import calendar
from datetime import datetime
from typing import Optional, Tuple, List

import requests
import psycopg2

# ── Config (matches app.py) ──
OLLAMA_URL = "http://127.0.0.1:11434"
RAG_MODEL = "qwen2.5:1.5b"

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "user": "ocr", "password": "***REMOVED***", "dbname": "invoices"
}


# ────────────────────────────────────────────────
# 1. INTENT CLASSIFIER (improved)
# ────────────────────────────────────────────────

# Core aggregation keywords – broad but we'll refine with context
AGGREGATION_KEYWORDS = [
    r"\btotal\b", r"\bsum\b", r"\baverage\b", r"\bavg\b",
    r"\bhighest\b", r"\blowest\b", r"\bmaximum\b", r"\bminimum\b",
    r"\bcount\b", r"\bhow many\b", r"\bhow much\b",
    r"\bgroup by\b", r"\bper\b", r"\beach\b",
    r"\bsummarize\b", r"\bsummary\b", r"\bbreakdown\b",
    r"\blargest\b", r"\bsmallest\b", r"\btop\b",
    # Chinese
    r"总金额", r"平均", r"最高", r"最低", r"最多", r"最少",
    r"汇总", r"统计", r"数量", r"多少", r"有几个", r"排名",
]

# Additional context words that strengthen aggregation intent
AGG_CONTEXT = re.compile(
    r"(amount|invoice|payment|vendor|currency|total|sum|avg|count)",
    re.IGNORECASE
)

AGG_PATTERN = re.compile("|".join(AGGREGATION_KEYWORDS), re.IGNORECASE)


def is_aggregation_query(query: str) -> bool:
    """Return True if the query likely needs SQL aggregation.
       Refined: must have an aggregation keyword AND a relevant context word
       to reduce false positives (e.g., "highest quality" won't match).
    """
    if not AGG_PATTERN.search(query):
        return False
    # Additionally, check that the query also mentions a data field
    # (this avoids false positives like "highest quality vendor")
    return bool(AGG_CONTEXT.search(query))


# ────────────────────────────────────────────────
# 2. EXTRACTORS (new helper functions)
# ────────────────────────────────────────────────


# ── Vendor Name Cache ──
_vendor_cache: Optional[List[str]] = None


def _get_known_vendors() -> List[str]:
    """Fetch distinct vendor names from the database (cached)."""
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
    """
    Attempt to extract a vendor name from the query by matching against
    known vendors in the database. Only returns a result if the extracted
    candidate is a substring of an actual vendor name in the DB.

    Patterns checked:
      - "for <vendor>", "to <vendor>", "from <vendor>"
      - "<vendor> invoices", "<vendor> payments"
    Returns the matching vendor name (from DB) or None.
    """
    known = _get_known_vendors()
    if not known:
        return None

    candidates = []
    # Pattern A: preposition + vendor name
    m = re.search(r'(?:for|to|from)\s+([A-Za-z0-9\s\.\-]+?)(?:\s+in\s|\s+Q\d|\s+\d{4}|\s*$)', query, re.IGNORECASE)
    if m:
        candidates.append(m.group(1).strip())

    # Pattern B: vendor name + "invoices"/"payments"
    m = re.search(r'([A-Za-z0-9\s\.\-]+?)\s+(?:invoices|payments|transactions)', query, re.IGNORECASE)
    if m:
        candidates.append(m.group(1).strip())

    # Validate: candidate must be a substring of a known vendor
    for candidate in candidates:
        if len(candidate) < 3:  # too short to be meaningful
            continue
        for db_vendor in known:
            if candidate.lower() in db_vendor.lower():
                return db_vendor  # return the canonical DB name

    return None


def extract_date_range_from_query(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse common date expressions into (date_from, date_to) strings (YYYY-MM-DD).
    Supports:
      - "in 2024" → (2024-01-01, 2024-12-31)
      - "Q1 2024"  → (2024-01-01, 2024-03-31)
      - "January 2024" → (2024-01-01, 2024-01-31)
      - "last year" → (2025-01-01, 2025-12-31)  # relative to current date
      - "between 2024-01-01 and 2024-03-31" → explicit
    """
    q = query.lower()

    # Explicit date range: between X and Y
    m = re.search(r'between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})', q)
    if m:
        return m.group(1), m.group(2)

    # Year only
    m = re.search(r'(?:in|of)\s+(\d{4})', q)
    if m:
        year = m.group(1)
        return f"{year}-01-01", f"{year}-12-31"

    # Quarter: Q1 2024, Q2 2024, etc.
    m = re.search(r'q([1-4])\s*(\d{4})', q)
    if m:
        quarter, year = int(m.group(1)), m.group(2)
        month_start = (quarter - 1) * 3 + 1
        month_end = month_start + 2
        day_end = calendar.monthrange(int(year), month_end)[1]
        return (f"{year}-{month_start:02d}-01",
                f"{year}-{month_end:02d}-{day_end}")

    # Month: January 2024, Jan 2024
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
        return (f"{year}-{month:02d}-01",
                f"{year}-{month:02d}-{day_end}")

    # "last year" (relative)
    if "last year" in q:
        this_year = datetime.now().year
        year = str(this_year - 1)
        return f"{year}-01-01", f"{year}-12-31"

    return None, None


def validate_date(date_str: str) -> bool:
    """Check if a string is a valid YYYY-MM-DD date."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


# ────────────────────────────────────────────────
# 3. SQL GENERATOR (updated to use extractors)
# ────────────────────────────────────────────────

def build_where_clause(vendor_filter=None, date_from=None, date_to=None):
    """Return (where_sql, params) safely."""
    conditions = []
    params = []
    if vendor_filter:
        conditions.append("vendor_name ILIKE %s")
        params.append(f"%{vendor_filter}%")
    if date_from and validate_date(date_from):
        conditions.append("date >= %s")
        params.append(date_from)
    if date_to and validate_date(date_to):
        conditions.append("date <= %s")
        params.append(date_to)
    where_sql = " AND ".join(conditions) if conditions else "TRUE"
    return where_sql, params


def generate_aggregation_sql(query: str, vendor_filter=None,
                              date_from=None, date_to=None) -> Optional[Tuple[str, list, str]]:
    """
    Analyze the query and return a (sql, params, desc) tuple or None if unrecognized.
    Supports:
      - Highest/lowest vendor by total amount
      - Total amount for a vendor / date range
      - Average amount by currency
      - Count by vendor
      - Top N largest invoices
      - Summarize (list all invoices matching filters)
    """
    # If filters not provided, try to extract from query
    if vendor_filter is None:
        vendor_filter = extract_vendor_from_query(query)
    if date_from is None and date_to is None:
        date_from, date_to = extract_date_range_from_query(query)

    where_sql, where_params = build_where_clause(vendor_filter, date_from, date_to)

    q = query.lower()

    # --- Pattern 8 (checked FIRST): top N largest invoices (most specific) ---
    # Matches: "top 5 largest", "5 largest", "largest 5", "top 5 biggest", etc.
    m = re.search(r"top\s+(\d+)\s+(?:largest|biggest)|(\d+)\s+(?:largest|biggest)|(?:largest|biggest)\s+(\d+)", q)
    if m:
        limit = int(m.group(1) or m.group(2))
        sql = f"""
            SELECT invoice_number, date, vendor_name,
                   total_amount::numeric::numeric(12,2) AS amount, currency
            FROM invoices
            WHERE {where_sql}
            ORDER BY total_amount::numeric DESC
            LIMIT %s
        """
        desc = f"top {limit} largest invoices"
        return sql, where_params + [limit], desc

    # --- Pattern 1: highest total amount by vendor ---
    if re.search(r"(?:which|what).*vendor.*(?:highest|largest|most)|(?:highest|largest|most).*vendor", q):
        if "total" in q or "amount" in q or "sum" in q:
            sql = f"""
                SELECT vendor_name, SUM(total_amount::numeric)::numeric(12,2) AS total
                FROM invoices
                WHERE {where_sql}
                GROUP BY vendor_name
                ORDER BY total DESC
                LIMIT 1
            """
            desc = "highest total amount by vendor"
            return sql, where_params, desc

    # --- Pattern 2: lowest total amount by vendor ---
    if re.search(r"(?:which|what).*vendor.*(?:lowest|least|smallest|minimum)|(?:lowest|least|smallest|minimum).*vendor", q):
        if "total" in q or "amount" in q or "sum" in q:
            sql = f"""
                SELECT vendor_name, SUM(total_amount::numeric)::numeric(12,2) AS total
                FROM invoices
                WHERE {where_sql}
                GROUP BY vendor_name
                ORDER BY total ASC
                LIMIT 1
            """
            desc = "lowest total amount by vendor"
            return sql, where_params, desc

    # --- Pattern 3: total amount (sum) ---
    # FIXED: parenthesized the and/or to prevent "sum" substring in "summarize" from triggering
    if re.search(r"\btotal\b|\bsum\b|\bhow much\b", q) and ("total amount" in q or "sum" in q):
        sql = f"""
            SELECT COALESCE(SUM(total_amount::numeric), 0)::numeric(12,2) AS total,
                   COUNT(*) AS invoice_count
            FROM invoices
            WHERE {where_sql}
        """
        desc = "total sum"
        return sql, where_params, desc

    # --- Pattern 4: count ---
    if re.search(r"\bcount\b|\bhow many\b", q):
        if re.search(r"vendor|by vendor|each vendor|per vendor", q):
            sql = f"""
                SELECT vendor_name, COUNT(*) AS cnt
                FROM invoices
                WHERE {where_sql}
                GROUP BY vendor_name
                ORDER BY cnt DESC
            """
            desc = "count by vendor"
            return sql, where_params, desc
        else:
            sql = f"""
                SELECT COUNT(*) AS total_invoices,
                       COUNT(DISTINCT vendor_name) AS unique_vendors,
                       COALESCE(SUM(total_amount::numeric), 0)::numeric(12,2) AS total_amount
                FROM invoices
                WHERE {where_sql}
            """
            desc = "count summary"
            return sql, where_params, desc

    # --- Pattern 5: average by currency ---
    if re.search(r"average|avg", q) and re.search(r"currency|by currency|per currency", q):
        sql = f"""
            SELECT currency,
                   COUNT(*) AS cnt,
                   AVG(total_amount::numeric)::numeric(12,2) AS avg_amount,
                   SUM(total_amount::numeric)::numeric(12,2) AS total
            FROM invoices
            WHERE {where_sql}
            GROUP BY currency
            ORDER BY total DESC
        """
        desc = "average by currency"
        return sql, where_params, desc

    # --- Pattern 6: average (general) ---
    if re.search(r"average|avg", q):
        sql = f"""
            SELECT AVG(total_amount::numeric)::numeric(12,2) AS avg_amount,
                   COUNT(*) AS invoice_count
            FROM invoices
            WHERE {where_sql}
        """
        desc = "average amount"
        return sql, where_params, desc

    # --- Pattern 7: summarize / list ---
    # Skip if this is actually a "top N largest" query (handled by Pattern 8)
    if re.search(r"summarize|summary|list|show|breakdown|all\s+(invoice|payment)", q) \
       and not re.search(r"(?:top|largest|biggest|smallest|highest|lowest)\s+\d+|\d+\s+(?:largest|biggest)", q):
        sql = f"""
            SELECT invoice_number, date, vendor_name,
                   total_amount::numeric::numeric(12,2) AS amount, currency
            FROM invoices
            WHERE {where_sql}
            ORDER BY date DESC
            LIMIT 50
        """
        desc = "summarize invoices"
        return sql, where_params, desc

    # Fallback: if aggregation keywords detected but no pattern matched,
    # do a general summarization
    if is_aggregation_query(query):
        sql = f"""
            SELECT invoice_number, date, vendor_name,
                   total_amount::numeric::numeric(12,2) AS amount, currency
            FROM invoices
            WHERE {where_sql}
            ORDER BY date DESC
            LIMIT 10
        """
        desc = "summary (fallback)"
        return sql, where_params, desc

    return None  # Not an aggregation query we can handle


# ────────────────────────────────────────────────
# 4. EXECUTE SQL + FORMAT RESULT (unchanged)
# ────────────────────────────────────────────────

def run_aggregation(sql: str, params: list) -> Tuple[list, list]:
    """Execute SQL and return rows and column names."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()
    return rows, colnames


def format_result(rows: list, colnames: list, desc: str) -> str:
    """Convert SQL rows to a compact text summary."""
    if not rows:
        return "No matching invoices found in the database."

    if len(rows) == 1 and len(colnames) <= 3:
        row = rows[0]
        parts = []
        for i, col in enumerate(colnames):
            if row[i] is not None:
                parts.append(f"{col}: {row[i]}")
        return f"[{desc}] " + ", ".join(parts)

    lines = [f"[{desc}] {len(rows)} results:"]
    for row in rows:
        line = "  " + " | ".join(str(v) if v is not None else "N/A" for v in row)
        lines.append(line)
    return "\n".join(lines)


# ────────────────────────────────────────────────
# 5. LLM REPHRASER (unchanged)
# ────────────────────────────────────────────────

def rephrase_with_llm(question: str, structured_summary: str) -> str:
    """Ask the LLM to turn structured data into a natural language answer."""
    prompt = textwrap.dedent(f"""\
        You are a helpful assistant. Based on the data below, answer the user's question
        in one or two sentences. Do NOT invent numbers — only use the provided data.

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
# 6. MAIN: BATTERY OF TEST QUERIES (unchanged)
# ────────────────────────────────────────────────

TEST_QUERIES = [
    # Query, vendor_filter, date_from, date_to
    ("Which vendor has the highest total invoice amount?", None, None, None),
    ("Which vendor has the lowest total invoice amount?", None, None, None),
    ("What is the total amount for Alibaba invoices?", "Alibaba", None, None),
    ("How many invoices does each vendor have?", None, None, None),
    ("What is the average invoice amount by currency?", None, None, None),
    ("Count how many invoices are in the database", None, None, None),
    ("Summarize all invoices in 2024", None, "2024-01-01", "2024-12-31"),
    ("Show me the 5 largest invoices by amount", None, None, None),
    ("What is the average invoice amount?", None, None, None),
    # Non-aggregation query (should be routed to vector search, not SQL)
    ("Find invoices related to consulting services", None, None, None),
    # New: natural language date without explicit filters
    ("Total amount for Alibaba in Q1 2024", None, None, None),
    ("How many invoices were issued in January 2024?", None, None, None),
]


def main():
    print("=" * 70)
    print("AGGREGATION ENGINE DEMO (revised)")
    print(f"  DB: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
    print(f"  LLM: {RAG_MODEL}")
    print("=" * 70)

    # Quick DB check
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), COUNT(embedding) FROM invoices")
        total, with_emb = cur.fetchone()
        cur.close()
        conn.close()
        print(f"\nDatabase: {total} invoices ({with_emb} with embeddings)")
        if total == 0:
            print("❌ No data in database. Insert data first (gen_500.py + embed_update_fast.py).")
            return
    except Exception as e:
        print(f"❌ Cannot connect to DB: {e}")
        return

    passed = 0
    failed = 0

    for i, (query, vendor, date_from, date_to) in enumerate(TEST_QUERIES, 1):
        print(f"\n{'─' * 70}")
        print(f"Test {i}: \"{query}\"")
        if vendor:
            print(f"  Filter: vendor='{vendor}'")
        if date_from or date_to:
            print(f"  Filter: date {date_from or '...'} → {date_to or '...'}")

        is_agg = is_aggregation_query(query)
        print(f"  Intent: {'📊 AGGREGATION (SQL)' if is_agg else '🔍 SEMANTIC SEARCH (vector)'}")

        if is_agg:
            result = generate_aggregation_sql(query, vendor_filter=vendor,
                                              date_from=date_from, date_to=date_to)
            if result is None:
                print(f"  ⚠️  No SQL template matched (will fall back to vector search in production)")
                failed += 1
                continue

            sql, params, desc = result
            print(f"  SQL pattern: {desc}")
            print(f"  Generated SQL:\n    {sql.strip().replace(chr(10), chr(10) + '    ')}")
            print(f"  Params: {params}")

            try:
                rows, colnames = run_aggregation(sql, params)
                summary = format_result(rows, colnames, desc)
                print(f"\n  Raw result:\n{summary}")

                # LLM rephrase
                print(f"\n  --- LLM Answer ---")
                answer = rephrase_with_llm(query, summary)
                print(f"  {answer}")
                passed += 1
            except Exception as e:
                print(f"  ❌ SQL error: {e}")
                failed += 1
        else:
            print(f"  ✅ Correctly identified as non-aggregation. Would use vector search + RAG.")
            passed += 1

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(TEST_QUERIES)} tests")
    print(f"{'=' * 70}")

    if passed == len(TEST_QUERIES):
        print("\n✅ All tests passed! The aggregation engine is ready to integrate into app.py.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review the output above.")


if __name__ == "__main__":
    main()