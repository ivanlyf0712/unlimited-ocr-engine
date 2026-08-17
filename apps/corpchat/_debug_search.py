#!/usr/bin/env python3
"""
诊断脚本: 探测 "诈骗" 查询为什么匹配不上 "old_friend_reconnect"
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from apps.corpchat.search import DEFAULT_INDEX_PATH, load_index, Searcher

idx = load_index(DEFAULT_INDEX_PATH)
print(f"Index: {idx.count()} chunks, graph={bool(idx.graph)}")

# 1. 直接搜索 "诈骗"
print("\n=== 1. Query='诈骗' expand=False (原生 txtai hybrid) ===")
results = idx.search("诈骗", limit=20)
for r in results:
    if isinstance(r, dict):
        tags = json.loads(r.get("tags","{}"))
        print(f"  score={r.get('score',0):.4f}  label={tags.get('label','?')[:30]}  id={r.get('id','')[:40]}")
    elif isinstance(r, tuple):
        print(f"  tuple={r}")

# 2. 查看 "old_friend_reconnect" 标签的所有块
print("\n=== 2. Search by 'old_friend_reconnect' label ===")
results2 = idx.search("old_friend_reconnect", limit=20)
for r in results2:
    if isinstance(r, dict):
        tags = json.loads(r.get("tags","{}"))
        text = r.get("text","")[:150]
        print(f"  score={r.get('score',0):.4f}  label={tags.get('label','?')}  text={text}")

# 3. 查看 "investment_opportunity" 标签
print("\n=== 3. Search by 'investment_opportunity' label ===")
results3 = idx.search("investment_opportunity", limit=20)
for r in results3:
    if isinstance(r, dict):
        tags = json.loads(r.get("tags","{}"))
        text = r.get("text","")[:150]
        print(f"  score={r.get('score',0):.4f}  label={tags.get('label','?')}  text={text}")

# 4. 查看所有不同的 label
print("\n=== 4. All labels in index ===")
all_labels = set()
results4 = idx.search("*", limit=200)
for r in results4:
    if isinstance(r, dict):
        tags = json.loads(r.get("tags","{}"))
        lbl = tags.get("label", "?")
        all_labels.add(lbl)
print(f"  Labels found: {sorted(all_labels)}")

# 5. 检查 enriched text 的样子
print("\n=== 5. Sample enriched texts from different labels ===")
for label in ["old_friend_reconnect", "investment_opportunity", "product_inquiry"]:
    r5 = idx.search(label, limit=1)
    if r5:
        r = r5[0]
        if isinstance(r, dict):
            tags = json.loads(r.get("tags","{}"))
            if tags.get("label") == label:
                text = r.get("text","")[:300]
                print(f"\n  [{label}] (enriched text first 300 chars):")
                print(f"  {text}")