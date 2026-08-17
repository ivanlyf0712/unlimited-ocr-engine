#!/usr/bin/env python3
"""
深入诊断: 检查 tags JSON 的原始内容和 txtai 的返回格式
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from apps.corpchat.search import DEFAULT_INDEX_PATH, load_index

idx = load_index(DEFAULT_INDEX_PATH)
print(f"Index: {idx.count()} chunks")

# Check: what format does txtai search return?
print("\n=== RAW search for '诈骗' (first result) ===")
raw = idx.search("诈骗", limit=3)
for i, r in enumerate(raw):
    print(f"\n--- Result {i+1} ---")
    print(f"  type={type(r).__name__}")
    if isinstance(r, dict):
        for k, v in r.items():
            v_str = str(v)[:200]
            print(f"  [{k}] = {v_str}")
    elif isinstance(r, tuple):
        print(f"  tuple length={len(r)}")
        for j, val in enumerate(r):
            print(f"  [{j}] = {str(val)[:200]}")

# Check: search by document ID directly
print("\n=== Direct ID search ===")
id_results = idx.search("id:msg_kf_old_friend_reconnect_29_0000__chunk0", limit=1)
if id_results:
    r = id_results[0]
    if isinstance(r, dict):
        print(f"  tags_raw={r.get('tags','NO_TAGS')[:300]}")
        print(f"  text={r.get('text','')[:200]}")
        try:
            tags = json.loads(r.get("tags","{}"))
            print(f"  parsed tags keys={list(tags.keys())}")
            print(f"  label={tags.get('label','???')}")
        except Exception as e:
            print(f"  json parse error: {e}")

# Check format of second channel output - the "text" field
print("\n=== Verify text field contains label info ===")
for query in ["msg_kf_old_friend_reconnect_29_0000", "msg_kf_investment_opportunity_30_0000"]:
    r = idx.search(query, limit=1)
    if r and isinstance(r[0], dict):
        print(f"\n  query={query}")
        print(f"  text begins with: {str(r[0].get('text',''))[:100]}")
        print(f"  in tags: {str(r[0].get('tags',''))[:100]}")