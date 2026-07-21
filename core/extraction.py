# ──────────────────── Extraction Module ────────────────────
import json
import re
import requests

from core.config import OLLAMA_URL, TEXT_MODEL, JSON_PROMPT, FALLBACK_PROMPT


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


def text_to_json(raw_text: str):
    return _extract_json(JSON_PROMPT, raw_text)


def text_to_json_fallback(raw_text: str):
    return _extract_json(FALLBACK_PROMPT, raw_text)


def is_likely_fake(data: dict) -> bool:
    suspicious = {"value", "text", "string", "example", "placeholder", "xxxx"}
    for field in ["invoice_number", "date", "vendor_name", "total_amount"]:
        val = (data.get(field) or "").strip().lower()
        if val in suspicious:
            return True
    return False