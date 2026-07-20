#!/usr/bin/env python3
"""
Compare different schema strictness levels for JSON extraction.
Reads raw text from output.txt (one invoice text per run).
"""

import json, re, requests, sys

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
TEXT_MODEL = "qwen2.5:0.5b"

# ---------- Prompts ----------
PROMPT = """You are an invoice data extractor. 
- "invoice_number"
- "date"
- "vendor_name"
- "total_amount"
- "currency"

Invoice text:
___RAW_TEXT___

JSON:"""

# ---------- Schemas ----------
# 1. No schema (just prompt)
NO_SCHEMA = None

# 2. Balanced schema – correct types, no pattern, allows flexibility
BALANCED_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "date": {"type": "string"},                     # no pattern – cleaner handles format
        "vendor_name": {"type": "string"},
        "total_amount": {"type": "string"},             # no pattern – cleaner ensures "XX.XX"
        "currency": {"type": "string"}                  # no pattern – cleaner forces 3 uppercase
    },
    "required": ["invoice_number", "date", "vendor_name", "total_amount", "currency"]
}

# 3. Strict schema (original, causing empty output)
STRICT_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "date": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
        "vendor_name": {"type": "string"},
        "total_amount": {"type": "string", "pattern": "^\\d+\\.\\d{2}$"},
        "currency": {"type": "string", "pattern": "^[A-Z]{3}$"}
    },
    "required": ["invoice_number", "date", "vendor_name", "total_amount", "currency"]
}

# ---------- Cleaner (same as pipeline) ----------
def clean_invoice_data(data):
    # total_amount → plain numeric string
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

    # currency → 3 uppercase letters
    curr = data.get("currency", "")
    if isinstance(curr, str):
        curr = curr.strip().upper()
        match = re.match(r'^([A-Z]{3})', curr)
        data["currency"] = match.group(1) if match else ""
    else:
        data["currency"] = ""

    # date → YYYY-MM-DD if possible
    date_val = data.get("date", "")
    if date_val:
        from datetime import datetime
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%d-%b-%Y", "%Y/%m/%d", "%d.%m.%Y"):
            try:
                data["date"] = datetime.strptime(date_val, fmt).strftime("%Y-%m-%d")
                break
            except:
                pass
    else:
        data["date"] = ""

    for key in ["invoice_number", "date", "vendor_name", "total_amount", "currency"]:
        data.setdefault(key, "")
    return data

# ---------- Test function ----------
def test_config(name, schema, raw_text):
    payload = {
        "model": TEXT_MODEL,
        "prompt": PROMPT.replace("___RAW_TEXT___", raw_text),
        "stream": False,
        "options": {"temperature": 0}
    }
    if schema:
        payload["format"] = {
            "type": "json_schema",
            "json_schema": {"name": "invoice", "schema": schema}
        }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    content = resp.json().get("response", "")
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print("Raw response:", content[:500])

    # Try to extract JSON
    data = None
    try:
        start = content.index('{')
        end = content.rindex('}') + 1
        data = json.loads(content[start:end])
        print("Parsed JSON:", json.dumps(data, indent=2))
        # Apply cleaner
        data = clean_invoice_data(data)
        print("After cleaning:", json.dumps(data, indent=2))
    except (ValueError, json.JSONDecodeError) as e:
        print("JSON parse error:", e)

    if data and all(data.get(k) for k in ("invoice_number", "vendor_name")):
        print("✅ Extraction successful and fields look plausible.")
    else:
        print("⚠️  Extraction may have failed or fields are empty.")

# ---------- Main ----------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            raw_text = f.read()
    else:
        raw_text = input("Paste raw invoice text (or provide file as argument): ")

    test_config("1. No schema (prompt only)", NO_SCHEMA, raw_text)
    test_config("2. Balanced schema (types only, no patterns)", BALANCED_SCHEMA, raw_text)
    test_config("3. Strict schema (patterns enforced)", STRICT_SCHEMA, raw_text)