import requests
import psycopg2
import json

API_KEY = ""
DB = {"host":"localhost","port":5432,"database":"invoices","user":"ocr","password":"***REMOVED***"}

conn = psycopg2.connect(**DB)
cur = conn.cursor()
cur.execute("SELECT msgid, content FROM messages WHERE content IS NOT NULL LIMIT 1;")
msgid, content = cur.fetchone()
cur.close()
conn.close()

print(f"Pushing msgid={msgid}, content[:50]={content[:50]}")

payload = {
    "document": {
        "id": f"test_public_{msgid}",
        "semantic_identifier": f"公开测试消息 {msgid}",
        "sections": [{"text": content}],
        "metadata": {"source": "test_public"}
    },
    "cc_pair_id": 1,
    "public_doc": True
}

resp = requests.post("http://localhost:3000/api/onyx-api/ingestion",
                    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                    json=payload)
print("Status:", resp.status_code)
print("Body:", resp.text)