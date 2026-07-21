#!/usr/bin/env python3
"""
Demo: Hybrid query classification using LLM + regex fallback.
Tests a battery of queries and prints the extracted intent and parameters.

Usage:
  python3 scripts/test_hybrid_classifier.py
"""

import json
import re
import sys
import calendar
from datetime import datetime
from typing import Optional, Tuple, List

import requests

# ─── Configuration (mirrors app.py) ───
OLLAMA_URL = "http://127.0.0.1:11434"
CLASSIFIER_MODEL = "qwen2.5:1.5b"   # small, fast

# ─── LLM Classifier ───────────────────────────────────────

CLASSIFIER_PROMPT = """You are a query analyzer for an invoice database.
Analyze the user's question and return a JSON object with these fields:

- "intent": either "aggregation" or "semantic"
- "vendor": the vendor name if mentioned, else null
- "date_from": start date in YYYY-MM-DD if a specific date range is mentioned, else null
- "date_to": end date in YYYY-MM-DD if mentioned, else null
- "amount_min": minimum amount if mentioned, else null
- "amount_max": maximum amount if mentioned, else null
- "aggregation_type": if intent is "aggregation", one of:
    "sum", "count", "average", "max", "min", "list", "top_n", "unknown"

Rules:
- Dates can be expressed as "last month", "Q1 2024", "2024", "January 2024", etc.
- "aggregation_type" should be:
  "sum" if the question asks for total/sum,
  "count" for how many,
  "average" for average/mean,
  "max" for highest/largest/most/biggest,
  "min" for lowest/smallest/least,
  "list" if it asks to show/display/list invoices (without aggregation math — e.g. "List invoices...", "Show me...", "Find invoices..."),
  "top_n" if it asks for top N largest/smallest/ranked items,
  "unknown" if the intent is aggregation but no specific type is clear.
- Questions like "List invoices...", "Show me...", "Find invoices..." that ask to display/retrieve records (not compute math) should have intent="aggregation" with aggregation_type="list".
- If the question is a simple lookup, open‑ended, or truly semantic search (not asking to list/find/show specific records from the DB), set "intent" to "semantic" and leave aggregation_type null.

Return ONLY the JSON object, no other text.

Question: {question}
JSON:"""

def classify_with_llm(question: str) -> Optional[dict]:
    """Return structured classification from LLM, or None if it fails."""
    prompt = CLASSIFIER_PROMPT.format(question=question)
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": CLASSIFIER_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 256}
            },
            timeout=15
        )
        resp.raise_for_status()
        content = resp.json().get("response", "")
        # Find JSON object
        start = content.index('{')
        end = content.rindex('}') + 1
        return json.loads(content[start:end])
    except Exception:
        return None

# ─── Regex‑based fallback (copied from your existing engine) ───

AGGREGATION_KEYWORDS = [
    r"\btotal\b", r"\bsum\b", r"\baverage\b", r"\bavg\b",
    r"\bhighest\b", r"\blowest\b", r"\bmaximum\b", r"\bminimum\b",
    r"\bcount\b", r"\bhow many\b", r"\bhow much\b",
    r"\bgroup by\b", r"\bper\b", r"\beach\b",
    r"\bsummarize\b", r"\bsummary\b", r"\bbreakdown\b",
    r"\blargest\b", r"\bsmallest\b", r"\btop\b",
    r"\bbiggest\b", r"\bshow me\b", r"\blist\b", r"\bdisplay\b",
    r"\bfind\b", r"\bshow\b",
    # Chinese
    r"总金额", r"平均", r"最高", r"最低", r"最多", r"最少",
    r"汇总", r"统计", r"数量", r"多少", r"有几个", r"排名",
]

AGG_CONTEXT = re.compile(
    r"(amount|invoice|payment|vendor|currency|total|sum|avg|count)",
    re.IGNORECASE
)

# Patterns that indicate a semantic/open-ended search, overriding aggregation keywords
_SEMANTIC_CONTEXT = re.compile(
    r"\b(about|related to|regarding|like|similar to|containing|description|details)\b",
    re.IGNORECASE
)

AGG_PATTERN = re.compile("|".join(AGGREGATION_KEYWORDS), re.IGNORECASE)

# Patterns to detect aggregation_type from regex fallback
_AGG_TYPE_PATTERNS = [
    (r"\bhow many\b|\bcount\b|\bnumber of\b", "count"),
    (r"\b(average|avg|mean)\b", "average"),
    (r"\b(highest|largest|biggest|maximum|max|most)\b", "max"),
    (r"\b(lowest|smallest|minimum|min|least)\b", "min"),
    (r"\b(total|sum|amount)\b", "sum"),
    (r"\btop\s*\d+\b", "top_n"),
    (r"\b(list|show me|display|find|show)\b", "list"),
]

def _detect_aggregation_type(query: str) -> str:
    """Detect aggregation_type from regex patterns on the query."""
    q = query.lower()
    for pattern, agg_type in _AGG_TYPE_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return agg_type
    return "unknown"

def is_aggregation_query(query: str) -> bool:
    """Regex‑based aggregation intent detection."""
    if not AGG_PATTERN.search(query):
        return False
    # Check for semantic context override (e.g. "Find invoices about X")
    if _SEMANTIC_CONTEXT.search(query):
        return False
    return bool(AGG_CONTEXT.search(query))

def extract_vendor_from_query(query: str) -> Optional[str]:
    """Extract vendor name using regex patterns (fallback)."""
    # Pattern 1: "for/to/from VENDOR_NAME"
    m = re.search(
        r'(?:for|to|from)\s+([A-Za-z0-9][A-Za-z0-9\s\.\-&]{1,40}?)(?:\s+in\s|\s+Q\d|\s+\d{4}|\s*$)',
        query, re.IGNORECASE
    )
    if m:
        vendor = m.group(1).strip().rstrip('.')
        if len(vendor) >= 3 and not re.match(r'^\d+$', vendor):
            return vendor

    # Pattern 2: "VENDOR_NAME invoices/payments"
    m = re.search(
        r'([A-Z][A-Za-z0-9\s\.\-&]{2,40}?)\s+(?:invoices|payments|transactions|invoice)\b',
        query, re.IGNORECASE
    )
    if m:
        vendor = m.group(1).strip().rstrip('.')
        if len(vendor) >= 3:
            return vendor

    # Pattern 3: "made to VENDOR"
    m = re.search(r'made\s+to\s+([A-Za-z0-9][A-Za-z0-9\s\.\-&]{1,40}?)(?:\s+in\s|\s+\d{4}|\s*$)', query, re.IGNORECASE)
    if m:
        vendor = m.group(1).strip().rstrip('.')
        if len(vendor) >= 3:
            return vendor

    return None

def extract_date_range_from_query(query: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse date expressions like 'in 2024', 'Q1 2024', 'January 2024'."""
    q = query.lower()
    # Year only
    m = re.search(r'(?:in|of)\s+(\d{4})', q)
    if m:
        year = m.group(1)
        return f"{year}-01-01", f"{year}-12-31"
    # Quarter: Q1 2024
    m = re.search(r'q([1-4])\s*(\d{4})', q)
    if m:
        quarter, year = int(m.group(1)), m.group(2)
        month_start = (quarter - 1) * 3 + 1
        month_end = month_start + 2
        day_end = calendar.monthrange(int(year), month_end)[1]
        return (f"{year}-{month_start:02d}-01", f"{year}-{month_end:02d}-{day_end}")
    # Month: January 2024
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
    return None, None

# ─── Hybrid classifier ────────────────────────────────────

def classify_hybrid(query: str) -> dict:
    """Return a dict with intent, vendor, date_from, date_to, method.

    Regex is the final gatekeeper:
    - If regex says NOT aggregation → semantic (LLM is never consulted).
    - If regex says aggregation → try LLM for richer params; fall back to regex.
    """
    # 1. Always run regex first – deterministic & fast
    regex_agg = is_aggregation_query(query)
    date_from, date_to = extract_date_range_from_query(query)
    vendor = extract_vendor_from_query(query)

    # 2. If regex says semantic, we stop here – LLM is not consulted
    if not regex_agg:
        return {
            "method": "Regex",
            "intent": "semantic",
            "vendor": None,
            "date_from": None,
            "date_to": None,
            "amount_min": None,
            "amount_max": None,
            "aggregation_type": None,
        }

    # 3. Regex says it's aggregation → try LLM for better parameters
    llm_result = classify_with_llm(query)
    if llm_result and llm_result.get("intent") == "aggregation":
        return {
            "method": "LLM (regex confirmed)",
            "intent": "aggregation",
            "vendor": llm_result.get("vendor") or vendor,
            "date_from": llm_result.get("date_from") or date_from,
            "date_to": llm_result.get("date_to") or date_to,
            "amount_min": llm_result.get("amount_min"),
            "amount_max": llm_result.get("amount_max"),
            "aggregation_type": llm_result.get("aggregation_type"),
        }

    # 4. LLM failed / returned semantic → use regex‑extracted parameters
    return {
        "method": "Regex (LLM unavailable)",
        "intent": "aggregation",
        "vendor": vendor,
        "date_from": date_from,
        "date_to": date_to,
        "amount_min": None,
        "amount_max": None,
        "aggregation_type": "unknown",
    }


def _non_null(value):
    """Return value if it's a valid non-null, non-'unknown' string, else None."""
    if not value:
        return None
    s = str(value).strip().lower()
    if s in ("none", "null", "unknown", ""):
        return None
    # Reject values that look like query fragments rather than real vendor names
    # (e.g. LLM hallucinating "How many" or "Show me the 5 biggest" as vendor)
    noise_patterns = [
        r"\bhow many\b", r"\bshow me\b", r"\bwhich vendor\b", r"\blist\b",
        r"\bfind\b", r"\btotal\b", r"\bsum\b", r"\bhighest\b", r"\blargest\b",
        r"\bbiggest\b", r"\baverage\b", r"\blast month\b", r"\bwhat is\b",
        r"^\d+$",  # pure numbers
    ]
    for pat in noise_patterns:
        if re.search(pat, s, re.IGNORECASE):
            return None
    return value


def _non_null_agg(value):
    """Return agg_type if valid, else None."""
    valid = {"sum", "count", "average", "max", "min", "list", "top_n"}
    if value and str(value).lower() in valid:
        return str(value).lower()
    return None

# ─── Demo: battery of test queries ───────────────────────

TEST_QUERIES = [
    "Total amount for Alibaba in Q1 2024",
    "Which vendor has the highest total invoice amount?",
    "List all invoices from last month",                # LLM should understand "last month"
    "Find invoices about consulting services",          # semantic
    "How many invoices were issued in March 2024?",
    "Show me the 5 biggest invoices",
    "What is the capital of France?",                   # out of domain
    "Average invoice amount by currency",
    "Summarize all payments made to Alibaba in 2024",
    "What was the largest invoice from EuroTrade GmbH?",
]

def main():
    print("=" * 80)
    print("HYBRID QUERY CLASSIFIER DEMO (LLM + Regex Fallback)")
    print(f"  LLM model: {CLASSIFIER_MODEL}")
    print("=" * 80)
    for query in TEST_QUERIES:
        print(f"\nQuery: \"{query}\"")
        result = classify_hybrid(query)
        print(f"  Method:       {result['method']}")
        print(f"  Intent:       {result['intent']}")
        if result['intent'] == 'aggregation':
            print(f"  Vendor:       {result['vendor']}")
            print(f"  Date from:    {result['date_from']}")
            print(f"  Date to:      {result['date_to']}")
            print(f"  Amount min:   {result['amount_min']}")
            print(f"  Amount max:   {result['amount_max']}")
            print(f"  Agg type:     {result['aggregation_type']}")
    print("\n" + "=" * 80)
    print("Done.")

if __name__ == "__main__":
    main()