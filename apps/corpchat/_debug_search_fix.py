#!/usr/bin/env python3
"""
验证修复方案: 从 enriched text 中提取 metadata
"""
import sys, os, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from apps.corpchat.search import DEFAULT_INDEX_PATH, load_index

idx = load_index(DEFAULT_INDEX_PATH)

def extract_metadata_from_text(text: str) -> dict:
    """从 enriched text 格式中提取 metadata dict.
    
    格式: [title]\n---\n[content]\n---\nMetadata: [key=value; key=value; ...]
    """
    meta = {}
    # 找最后一个 "---\nMetadata:" 分隔符
    marker = "\n---\nMetadata: "
    if marker not in text:
        return meta
    meta_str = text.split(marker)[-1]
    # 解析 key=value; key=value; ...
    parts = [p.strip() for p in meta_str.split(";")]
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            meta[k.strip()] = v.strip()
    return meta

# 测试
query = "msg_kf_old_friend_reconnect_29_0000"
results = idx.search(query, limit=3)
for r in results:
    if isinstance(r, dict):
        text = r.get("text", "")
        meta = extract_metadata_from_text(text)
        print(f"  id={r.get('id','')[:50]}")
        print(f"  extracted label={meta.get('label','???')}")
        print(f"  extracted customer_name={meta.get('customer_name','???')}")
        print(f"  extracted send_time={meta.get('send_time','???')}")
        print()

# 测试 "诈骗" 搜索 - 看看旧_friend_reconnect 的实际 score
print("=== Test '诈骗' with label filter ===")
r = idx.search("诈骗", limit=20)
results = []
for item in r:
    if isinstance(item, dict):
        text = item.get("text", "")
        meta = extract_metadata_from_text(text)
        results.append(("诈骗", item.get("score",0), meta.get("label","?"), meta.get("customer_name","?"), text[:60]))
# 显示 top 5
results.sort(key=lambda x: -x[1])
for score, label, name, preview in [(r[1], r[2], r[3], r[4]) for r in results[:10]]:
    print(f"  score={score:.4f}  label={label[:25]}  name={name[:12]}  text={preview}")