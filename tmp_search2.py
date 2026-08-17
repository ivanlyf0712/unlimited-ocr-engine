import requests, json

API_KEY = ""

# Search for the test_msg document specifically
resp = requests.post(
    "http://localhost:3000/api/search",
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={"query": "測試消息", "search_type": "keyword", "max_results": 10}
)
print("Status:", resp.status_code)
try:
    data = resp.json()
    print("Results:")
    if "results" in data and len(data["results"]) > 0:
        found_test = False
        for r in data["results"]:
            title = r.get('title', '')
            doc_id = r.get('document_id', '')
            print(f"  - document_id: {doc_id}, title: {title}")
            if 'test_msg' in title or '测试消息' in title:
                found_test = True
        if not found_test:
            print("  (no test_msg documents found in results)")
    else:
        print("  No results found")
        if "error" in data:
            print(f"  Error: {data['error']}")
except Exception as e:
    print(f"Error parsing: {e}")
    print("Raw response:", resp.text[:500])