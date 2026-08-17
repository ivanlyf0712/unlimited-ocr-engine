import requests, json

# Use HTTPS with admin auth
OPENSEARCH_URL = "https://onyx-opensearch-1:9200"
AUTH = ("admin", "StrongPassword123!")

# Update all ingestion_api chunks: set document_sets to ["Postgres"]
payload = {
    "script": {
        "source": "ctx._source.document_sets = params.doc_sets",
        "lang": "painless",
        "params": {
            "doc_sets": ["Postgres"]
        }
    },
    "query": {
        "bool": {
            "must_not": [
                {"term": {"source_type": "user_file"}}
            ]
        }
    }
}

resp = requests.post(
    f"{OPENSEARCH_URL}/danswer_chunk_nomic_ai_nomic_embed_text_v1/_update_by_query?pretty&conflicts=proceed",
    auth=AUTH,
    headers={"Content-Type": "application/json"},
    json=payload,
    verify=False,
    timeout=60
)
print(f"Status: {resp.status_code}")
data = resp.json()
print(f"Updated: {data.get('updated', 'N/A')}")
print(f"Failures: {data.get('failures', [])}")

# Also update user_file chunks with document_sets
payload2 = {
    "script": {
        "source": "ctx._source.document_sets = params.doc_sets",
        "lang": "painless",
        "params": {
            "doc_sets": ["Postgres"]
        }
    },
    "query": {
        "term": {"source_type": "user_file"}
    }
}

resp2 = requests.post(
    f"{OPENSEARCH_URL}/danswer_chunk_nomic_ai_nomic_embed_text_v1/_update_by_query?pretty&conflicts=proceed",
    auth=AUTH,
    headers={"Content-Type": "application/json"},
    json=payload2,
    verify=False,
    timeout=60
)
print(f"Status: {resp2.status_code}")
data2 = resp2.json()
print(f"Updated: {data2.get('updated', 'N/A')}")
print(f"Failures: {data2.get('failures', [])}")

# Verify
check = requests.get(
    f"{OPENSEARCH_URL}/danswer_chunk_nomic_ai_nomic_embed_text_v1/_search?pretty&size=1&_source=document_id,document_sets,source_type",
    auth=AUTH,
    headers={"Content-Type": "application/json"},
    json={
        "query": {"bool": {"must_not": [{"term": {"source_type": "user_file"}}]}}
    },
    verify=False,
    timeout=30
)
print(f"\nVerification - ingestion_api chunk:")
print(check.json()["hits"]["hits"][0]["_source"])

check2 = requests.get(
    f"{OPENSEARCH_URL}/danswer_chunk_nomic_ai_nomic_embed_text_v1/_search?pretty&size=1&_source=document_id,document_sets,source_type",
    auth=AUTH,
    headers={"Content-Type": "application/json"},
    json={
        "query": {"term": {"source_type": "user_file"}}
    },
    verify=False,
    timeout=30
)
print(f"Verification - user_file chunk:")
print(check2.json()["hits"]["hits"][0]["_source"])