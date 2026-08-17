import requests, json, time

API_KEY = ""

# Test 1: Quick search to see if Agent-style query works
# The Agent uses the /api/search endpoint internally but filters by persona/documents
# Let's try to use the /api/persona/{id}/search endpoint if it exists
print("=== Test 1: Search with persona_id parameter ===")
r = requests.post("http://localhost:3000/api/search", 
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={
        "query": "產品價格",
        "search_type": "keyword",
        "max_results": 5,
        "persona_id": 1
    },
    timeout=15
)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    print(f"Results count: {len(data.get('results', []))}")
    for res in data.get("results", []):
        print(f"  - title: {res.get('title')}, source: {res.get('source_type')}")
else:
    print(f"Error: {r.text[:300]}")

print()
print("=== Test 2: Direct /api/stream-query (non-streaming) ===")
r = requests.post("http://localhost:3000/api/query", 
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={
        "query": "告訴我有關產品價格的信息",
        "persona_id": 1,
        "search_doc_ids": [],
        "use_agentic_search": False,
        "return_citations": True,
        "skip_rerank": False
    },
    timeout=30
)
print(f"Status: {r.status_code}")
print(f"Response: {r.text[:800]}")