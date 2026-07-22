# ──────────────────── Hybrid Query Classifier ────────────────────
"""
Hybrid query classification with two tiers:
1. Regex is the fast path — if it matches a known aggregation keyword,
   the query is routed to SQL immediately (no LLM needed).
2. LLM is the fallback — only consulted when regex says "semantic",
   in case it's an aggregation that regex didn't recognize.
"""
import json
import re
import calendar
from datetime import datetime, timedelta
from typing import Optional, Tuple

import requests

from core.config import OLLAMA_URL

CLASSIFIER_MODEL = "qwen2.5:1.5b"

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
  "sum" for total/sum, "count" for how many, "average" for average/mean,
  "max" for highest/largest/most/biggest, "min" for lowest/smallest/least,
  "list" for show/display/list, "top_n" for top N ranked,
  "unknown" if the intent is aggregation but no specific type is clear.
- Questions like "List invoices...", "Show me...", "Find invoices..." are aggregation (type="list").
- True semantic search (open-ended questions, not asking to list/find/show specific records) has intent="semantic".

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
            timeout=30
        )
        resp.raise_for_status()
        content = resp.json().get("response", "")
        start = content.index('{')
        end = content.rindex('}') + 1
        return json.loads(content[start:end])
    except Exception:
        return None


# ─── Regex-based aggregation detection ──────────────────

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

_SEMANTIC_CONTEXT = re.compile(
    r"\b(about|related to|regarding|like|similar to|containing|description|details)\b",
    re.IGNORECASE
)

AGG_PATTERN = re.compile("|".join(AGGREGATION_KEYWORDS), re.IGNORECASE)


def is_aggregation_query(query: str) -> bool:
    """Regex‑based aggregation intent detection."""
    if not AGG_PATTERN.search(query):
        return False
    if _SEMANTIC_CONTEXT.search(query):
        return False
    return bool(AGG_CONTEXT.search(query))


def extract_vendor_from_query(query: str) -> Optional[str]:
    """Extract vendor name using regex patterns (fallback)."""
    # Pattern 1: "for/to/from VENDOR_NAME"
    m = re.search(
        r'(?:for|to|from)\s+([A-Za-z0-9][A-Za-z0-9\s\.\-&]{2,40}?)(?:\s+in\s|\s+Q\d|\s+\d{4}|\s*\?$|\s*$)',
        query, re.IGNORECASE
    )
    if m:
        vendor = _clean_vendor(m.group(1))
        if vendor:
            return vendor

    # Pattern 2: "VENDOR_NAME invoices/payments"
    m = re.search(
        r'([A-Z][A-Za-z0-9\s\.\-&]{2,40}?)\s+(?:invoices|payments|transactions|invoice)\b',
        query, re.IGNORECASE
    )
    if m:
        vendor = _clean_vendor(m.group(1))
        if vendor:
            return vendor

    # Pattern 3: "made to VENDOR"
    m = re.search(r'made\s+to\s+([A-Za-z0-9][A-Za-z0-9\s\.\-&]{1,40}?)(?:\s+in\s|\s+\d{4}|\s*$)',
                  query, re.IGNORECASE)
    if m:
        vendor = _clean_vendor(m.group(1))
        if vendor:
            return vendor

    return None


def _clean_vendor(raw: str) -> Optional[str]:
    """Clean and validate a vendor name candidate."""
    vendor = raw.strip().rstrip('.')
    if len(vendor) < 3:
        return None
    if re.match(r'^\d+$', vendor):
        return None
    # Reject values that look like query fragments, not real vendor names
    noise = re.compile(
        r'\b(?:how many|show me|which vendor|list|find|total|sum|highest|l(?:argest|owest)|'
        r'biggest|average|last month|what is|how much|count|display|'
        r'summarize|summary|breakdown|all|every|each|per)\b',
        re.IGNORECASE
    )
    if noise.search(vendor):
        return None
    # Strip leading question words: "Which Alibaba" → "Alibaba"
    vendor = re.sub(r'^(?:which|what|show|find|list|display|how|who|where|when)\s+', '', vendor, flags=re.IGNORECASE).strip()
    # Strip trailing noise words: "Alibaba invoices" → "Alibaba"
    vendor = re.sub(r'\s+(?:invoices?|payments?|transactions?)\s*$', '', vendor, flags=re.IGNORECASE).strip()
    if len(vendor) < 3:
        return None
    return vendor


def extract_date_range_from_query(query: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse date expressions like 'in 2024', 'Q1 2024', '2024-03', 'March 2024'.

    Matches:
      - "in 2024", "of 2024", "from 2024", "for 2024", "during 2024"
      - Bare year: "Alibaba 2024", "invoices 2024" (with word boundary)
      - "Q1 2024", "Q2 2024", etc.
      - "2024-05" or "2024/05" → whole month
      - "2024-05-12" or "2024/05/12" → exact day (single-day range)
      - "May 2024", "January 2024"
      - "2024 January" (year-first)
      - "last month", "this month", "last year" (relative)
    """
    q = query.lower()

    # ── Pattern 0: "between YYYY-MM-DD and YYYY-MM-DD" (MUST be before single date) ──
    m = re.search(r'between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})', q)
    if m:
        return m.group(1), m.group(2)

    # ── Pattern 0b: Full date YYYY-MM-DD or YYYY/MM/DD ──
    m = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', q)
    if m:
        yr, mo, dy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= dy <= 31:
            date_str = f"{yr:04d}-{mo:02d}-{dy:02d}"
            return date_str, date_str

    # ── Pattern 0c: Year-month YYYY-MM or YYYY/MM (no day) ──
    m = re.search(r'(\d{4})[-/](\d{1,2})\b(?![-/]\d)', q)
    if m:
        yr, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            day_end = calendar.monthrange(yr, mo)[1]
            return (f"{yr:04d}-{mo:02d}-01", f"{yr:04d}-{mo:02d}-{day_end}")

    # ── Pattern 1: Year with preposition "in/of/from/for/during 2024" ──
    m = re.search(r'(?:in|of|from|for|during)\s+(\d{4})(?:\s|$)', q)
    if m:
        year = m.group(1)
        return f"{year}-01-01", f"{year}-12-31"

    # ── Pattern 2: "Q1 2024", "Q2 2024" ──
    m = re.search(r'q([1-4])\s*(\d{4})', q)
    if m:
        quarter, year = int(m.group(1)), m.group(2)
        month_start = (quarter - 1) * 3 + 1
        month_end = month_start + 2
        day_end = calendar.monthrange(int(year), month_end)[1]
        return (f"{year}-{month_start:02d}-01", f"{year}-{month_end:02d}-{day_end}")

    # ── Pattern 3: "last month" (relative) ──
    if "last month" in q:
        today = datetime.now()
        first_of_this = today.replace(day=1)
        last_month_end = first_of_this - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return last_month_start.strftime("%Y-%m-%d"), last_month_end.strftime("%Y-%m-%d")

    # ── Pattern 4: "this month" (relative) ──
    if "this month" in q:
        today = datetime.now()
        first = today.replace(day=1)
        return first.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    # ── Pattern 5: "last year" (relative) ──
    if "last year" in q:
        this_year = datetime.now().year
        year = str(this_year - 1)
        return f"{year}-01-01", f"{year}-12-31"

    # ── Pattern 6: Month-name + year "January 2024" ──
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        # Abbreviations
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    }
    m = re.search(r'(' + '|'.join(months.keys()) + r')\s*(\d{4})', q)
    if m:
        month_name, year = m.group(1), m.group(2)
        month = months[month_name]
        day_end = calendar.monthrange(int(year), month)[1]
        return (f"{year}-{month:02d}-01", f"{year}-{month:02d}-{day_end}")

    # ── Pattern 7: Year-first "2024 January" ──
    m = re.search(r'(\d{4})\s*(' + '|'.join(months.keys()) + r')', q)
    if m:
        year, month_name = m.group(1), m.group(2)
        month = months[month_name]
        day_end = calendar.monthrange(int(year), month)[1]
        return (f"{year}-{month:02d}-01", f"{year}-{month:02d}-{day_end}")

    # ── Pattern 8 (LAST): Bare year fallback ──
    # Only matches if no more specific pattern caught it.
    # e.g., "Alibaba 2024", "invoices 2024", "total 2024"
    m = re.search(r'(?:^|\D)(\d{4})(?:\D|$)', q)
    if m:
        yr_val = int(m.group(1))
        if 2000 <= yr_val <= 2099:
            year = m.group(1)
            return f"{year}-01-01", f"{year}-12-31"

    return None, None


# ─── Hybrid classifier ──────────────────────────────────

def classify_hybrid(query: str) -> dict:
    """Return a dict with intent, vendor, date_from, date_to, method.

    Two‑tier classification:
    1. Fast regex check first. If regex matches a known aggregation pattern
       (total, sum, average, count, etc.), return aggregation immediately —
       no LLM needed, the SQL template will handle it.
    2. If regex says semantic, ask LLM to verify — it may be an aggregation
       query that doesn't fit the predefined templates (e.g. unusual phrasing).
    """
    # 1. Always run regex first – deterministic & fast
    regex_agg = is_aggregation_query(query)
    date_from, date_to = extract_date_range_from_query(query)
    vendor = extract_vendor_from_query(query)

    # 2. Regex says aggregation → return immediately, no LLM needed.
    #    The SQL templates in generate_aggregation_sql() will handle matching.
    if regex_agg:
        return {
            "method": "Regex",
            "intent": "aggregation",
            "vendor": vendor,
            "date_from": date_from,
            "date_to": date_to,
            "amount_min": None,
            "amount_max": None,
            "aggregation_type": "unknown",
        }

    # 3. Regex says semantic → consult LLM to verify if it might actually
    #    be an aggregation that doesn't match the predefined regex patterns.
    llm_result = classify_with_llm(query)
    if llm_result and llm_result.get("intent") == "aggregation":
        llm_vendor = _clean_vendor(llm_result.get("vendor") or "") if llm_result.get("vendor") else None
        return {
            "method": "LLM (regex didn't match)",
            "intent": "aggregation",
            "vendor": llm_vendor or vendor,
            "date_from": llm_result.get("date_from") or date_from,
            "date_to": llm_result.get("date_to") or date_to,
            "amount_min": llm_result.get("amount_min"),
            "amount_max": llm_result.get("amount_max"),
            "aggregation_type": llm_result.get("aggregation_type"),
        }

    # 4. LLM also says semantic (or LLM unavailable) → true semantic search
    return {
        "method": "Regex + LLM" if llm_result else "Regex (LLM unavailable)",
        "intent": "semantic",
        "vendor": vendor,
        "date_from": date_from,
        "date_to": date_to,
        "amount_min": None,
        "amount_max": None,
        "aggregation_type": None,
    }
