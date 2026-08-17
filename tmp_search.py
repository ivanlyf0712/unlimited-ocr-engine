import requests, json

API_KEY = ""

# Test search with a keyword from the message
resp = requests.post(
    "http://localhost:3000/api/search",
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={"query": "最新產品的價格", "search_type": "keyword", "max_results": 10}
)
print("Status:", resp.status_code)
try:
    data = resp.json()
    print("Results:")
    if "results" in data and len(data["results"]) > 0:
        for r in data["results"]:
            print(f"  - document_id: {r.get('document_id')}, title: {r.get('title')}")
    else:
        print("  No results found")
        if "error" in data:
            print(f"  Error: {data['error']}")
except:
    print("Raw response:", resp.text[:500])