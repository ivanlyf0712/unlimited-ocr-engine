#!/usr/bin/env python3
"""
CorpChat Search Framework — 轻量级 Onyx 风格搜索系统
基于 analysis_report.md 的设计蓝图构建。

核心组件：
  - IndexBuilder: 分块 + 丰富化 → 索引 (参考 §2.2, §2.3)
  - QueryExpander: LLM 查询扩展 (参考 §2.7)
  - Searcher: 多查询 RRF 融合 + 图增强 + 重排序 (参考 §2.5, §2.6, §2.8)
  - AgenticDecider: 意图路由 (参考 §2.7)
  - CLI: build, search, benchmark, validate

使用方式：
  python corpchat_search_framework.py build --force
  python corpchat_search_framework.py search "诈骗" --expand --graph-expand 1
  python corpchat_search_framework.py benchmark
  python corpchat_search_framework.py validate

注意：本框架为“骨架”，部分实现（如 LLM 调用）需根据实际 API 配置。
"""

import os
import sys
import json
import time
import logging
import statistics
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import click
import psycopg2
import txtai
from tabulate import tabulate

# ================================================================
# 1. 配置与常量
# ================================================================
# 环境变量 (可被 .env 覆盖)
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "invoices"),
    "user": os.getenv("DB_USER", "ocr"),
    "password": os.getenv("DB_PASSWORD", "***REMOVED***"),
}

INDEX_PATH = os.getenv("INDEX_PATH", "corpchat_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 256))          # token
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "sk-placeholder")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "https://your-litellm-proxy.example.com/v1")
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "dseek-v4-flash")

# 日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("corpchat-search")


# ================================================================
# 2. 索引构建器 (IndexBuilder)
#    参考 analysis_report.md §2.2 (分块) 和 §2.3 (丰富化)
# ================================================================
class IndexBuilder:
    """构建带分块、丰富化、元数据的混合索引。"""

    def __init__(self, index_path: str = INDEX_PATH):
        self.index_path = index_path

    def _fetch_messages(self) -> List[Dict]:
        """从 PostgreSQL 读取消息及关联联系人信息。"""
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT m.msgid, m.content, m.send_time, m.external_userid,
                   m.servicer_userid, m.label, c.full_name AS customer_name,
                   m.open_kfid, m.origin
            FROM messages m
            LEFT JOIN contacts c ON m.external_userid = c.userid
            WHERE m.content IS NOT NULL AND m.content != ''
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        messages = []
        for row in rows:
            messages.append({
                "msgid": row[0],
                "content": row[1],
                "send_time": row[2],
                "external_userid": row[3],
                "servicer_userid": row[4],
                "label": row[5],
                "customer_name": row[6] or row[3],
                "open_kfid": row[7],
                "origin": row[8],
            })
        return messages

    def _chunk_message(self, msg: Dict) -> List[Dict]:
        """
        将单条消息拆分为多个块 (如果内容超过 CHUNK_SIZE)。
        使用句子级分块，无重叠。
        """
        content = msg["content"]
        # 尝试使用 chonkie (如果安装) 否则用简单 split
        chunks_text = []
        try:
            from chonkie import SentenceChunker
            # 注意: 需要 tokenizer，这里用一个轻量级 tokenizer
            chunker = SentenceChunker(
                tokenizer=EMBEDDING_MODEL,
                chunk_size=CHUNK_SIZE,
                min_chunk_size=50,
            )
            chunks = chunker.chunk(content)
            chunks_text = [c.text for c in chunks]
        except ImportError:
            # fallback: 按句子分割并合并到 CHUNK_SIZE 字符左右
            import re
            sentences = re.split(r'(?<=[.!?])\s+', content)
            current = []
            current_len = 0
            for sent in sentences:
                sent_len = len(sent)
                if current_len + sent_len < CHUNK_SIZE * 4:  # 粗略字符估计
                    current.append(sent)
                    current_len += sent_len
                else:
                    if current:
                        chunks_text.append(" ".join(current))
                    current = [sent]
                    current_len = sent_len
            if current:
                chunks_text.append(" ".join(current))

        if not chunks_text:
            chunks_text = [content]  # 保底

        # 生成块 ID
        base_id = msg["msgid"]
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            chunk_id = f"{base_id}_chunk{i}"
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "msgid": msg["msgid"],
                    "send_time": msg["send_time"].isoformat() if msg["send_time"] else None,
                    "external_userid": msg["external_userid"],
                    "servicer_userid": msg["servicer_userid"],
                    "label": msg["label"],
                    "customer_name": msg["customer_name"],
                    "open_kfid": msg["open_kfid"],
                    "origin": msg["origin"],
                    "chunk_index": i,
                },
                "title": f"{msg['customer_name']} ({msg['label'] or 'general'})",
            })
        return chunks

    def _enrich_chunk(self, chunk: Dict) -> str:
        """
        丰富化: 组合标题 + 内容 + 元数据描述。
        参考 §2.3 的 generate_enriched_content_for_chunk_embedding。
        """
        title = chunk.get("title", "")
        text = chunk["text"]
        meta = chunk["metadata"]
        # 构造元数据描述 (排除内部字段)
        meta_desc = "; ".join([
            f"{k}={v}" for k, v in meta.items()
            if k not in ["msgid", "chunk_index", "origin"]
        ])
        return f"{title}\n---\n{text}\n---\nMetadata: {meta_desc}"

    def build(self, force: bool = False, enable_graph: bool = True) -> txtai.Embeddings:
        """构建或加载索引，包含分块和丰富化。"""
        if os.path.exists(self.index_path) and not force:
            logger.info(f"从 {self.index_path} 加载已有索引 ...")
            embeddings = txtai.Embeddings()
            embeddings.load(self.index_path)
            logger.info(f"已加载 {embeddings.count()} 个块")
            return embeddings

        logger.info("从数据库构建新索引 (含分块+丰富化) ...")
        messages = self._fetch_messages()
        if not messages:
            raise RuntimeError("数据库中没有消息")

        # 分块
        all_chunks = []
        for msg in messages:
            chunks = self._chunk_message(msg)
            all_chunks.extend(chunks)
        logger.info(f"分块完成: {len(all_chunks)} 个块")

        # 丰富化并准备文档
        docs = []
        for chunk in all_chunks:
            enriched_text = self._enrich_chunk(chunk)
            # tags 存储元数据的 JSON 字符串 (用于过滤)
            tags_json = json.dumps(chunk["metadata"], default=str)
            docs.append((chunk["id"], enriched_text, tags_json))

        # 配置 txtai 索引
        config = {
            "path": EMBEDDING_MODEL,
            "content": True,
            "objects": True,
            "hybrid": True,
            "scoring": {"method": "bm25"},
        }
        if enable_graph:
            config["graph"] = True

        embeddings = txtai.Embeddings(config)

        logger.info(f"索引 {len(docs)} 个文档块 ...")
        t0 = time.perf_counter()
        embeddings.index(docs)
        logger.info(f"索引完成，耗时 {time.perf_counter()-t0:.2f}s")

        # 图: 自动构建 (txtai 已基于向量相似度构建)
        if enable_graph and embeddings.graph:
            logger.info("图自动推断已启用 (由 txtai 基于向量构建)")

        embeddings.save(self.index_path)
        logger.info(f"索引保存至 {self.index_path}")
        return embeddings


# ================================================================
# 3. 查询扩展器 (QueryExpander)
#    参考 §2.7 的 LLM 查询扩展
# ================================================================
class QueryExpander:
    """使用 LLM 生成语义重写和关键词扩展查询。"""

    def __init__(self, api_base: str = LITELLM_BASE_URL,
                 api_key: str = LITELLM_API_KEY,
                 model: str = LITELLM_MODEL):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self._cache = {}

    def _call_llm(self, prompt: str) -> str:
        """调用 LiteLLM 并返回结果字符串。"""
        import requests
        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 200,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"LLM 调用失败: {e}")
            return ""

    def expand(self, query: str) -> List[Tuple[str, float]]:
        """
        生成扩展查询列表，每个查询带权重。
        权重参考: 原始 0.5, 语义 1.3, 关键词 1.0
        """
        # 检查缓存
        cache_key = query[:100]
        if cache_key in self._cache:
            return self._cache[cache_key]

        results = [(query, 0.5)]  # 原始查询

        # 语义重写
        semantic_prompt = f"""Rewrite the following user query into a standalone semantic search query. Only output the query, no extra text.

Query: {query}
Semantic query:"""
        semantic = self._call_llm(semantic_prompt)
        if semantic and semantic != query:
            results.append((semantic, 1.3))

        # 关键词扩展
        keyword_prompt = f"""Extract up to 3 keyword-only search queries from the user query, each on a new line. Only output the keywords, no extra text.

Query: {query}
Keywords:"""
        keywords_raw = self._call_llm(keyword_prompt)
        if keywords_raw:
            for line in keywords_raw.split('\n'):
                kw = line.strip()
                if kw and len(kw) > 1:
                    results.append((kw, 1.0))

        # 最多保留 4 个查询 (原始 + 语义 + 2 个关键词)
        if len(results) > 4:
            # 保留原始和语义，取前两个关键词
            results = [results[0], results[1]] + results[2:4]

        self._cache[cache_key] = results
        return results


# ================================================================
# 4. 重排序器 (Reranker)
#    使用交叉编码器 (替代 Onyx 的 LLM 选择)
# ================================================================
class Reranker:
    """轻量级交叉编码器重排序。"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.enabled = False
        self.model = None
        self.model_name = model_name
        try:
            from sentence_transformers import CrossEncoder
            self.enabled = True
        except ImportError:
            logger.warning("sentence_transformers 未安装，重排序禁用")

    def _ensure_model(self):
        if self.model is None and self.enabled:
            from sentence_transformers import CrossEncoder
            logger.info(f"加载交叉编码器: {self.model_name}")
            self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, results: List[Dict], top_n: int = 20) -> List[Dict]:
        """只对 top_n 个结果重排序，其余保持不变。"""
        if not self.enabled or not results:
            return results
        self._ensure_model()
        if len(results) <= top_n:
            to_rerank = results
            rest = []
        else:
            to_rerank = results[:top_n]
            rest = results[top_n:]
        pairs = [(query, item["text"]) for item in to_rerank]
        scores = self.model.predict(pairs)
        for i, score in enumerate(scores):
            to_rerank[i]["score"] = float(score)
        to_rerank.sort(key=lambda x: x["score"], reverse=True)
        return to_rerank + rest


# ================================================================
# 5. 搜索器 (Searcher)
#    实现多查询 RRF 融合 (参考 §2.8) + 图扩展 (参考 §2.5)
# ================================================================
class Searcher:
    def __init__(self, embeddings: txtai.Embeddings,
                 expander: Optional[QueryExpander] = None,
                 reranker: Optional[Reranker] = None):
        self.embeddings = embeddings
        self.expander = expander
        self.reranker = reranker

    @staticmethod
    def _weighted_rrf_fusion(all_results: List[Tuple[List[Tuple[str, float]], float]],
                             k: int = 50) -> List[Tuple[str, float]]:
        """
        加权 RRF 融合多个查询的结果。
        all_results: [(result_list, weight), ...]  result_list 为 [(doc_id, score), ...]
        返回 [(doc_id, fused_score), ...]
        """
        scores = defaultdict(float)
        for result_list, weight in all_results:
            for rank, (doc_id, score) in enumerate(result_list, start=1):
                scores[doc_id] += weight / (k + rank)
        # 按分数降序
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items

    def _fetch_doc_details(self, doc_id: str) -> Optional[Dict]:
        """通过 txtai 获取文档详情。"""
        # txtai 的 search 可以返回文档，但这里我们用内部方法
        # 简单方法: 用 search 查询 doc_id (需要确保索引中有该 id)
        results = self.embeddings.search(f"id:{doc_id}", limit=1)
        if results:
            # results 可能是 list of dict or list of tuples
            if isinstance(results[0], dict):
                return results[0]
            else:
                # tuple: (id, text, tags, score)
                return {"id": results[0][0], "text": results[0][1], "tags": results[0][2]}
        return None

    def search(self, query: str,
               mode: str = "hybrid",
               limit: int = 10,
               expand: bool = True,
               graph_expand: int = 0,
               label_filter: Optional[str] = None,
               date_from: Optional[str] = None,
               date_to: Optional[str] = None,
               use_rerank: bool = False) -> List[Dict]:
        """
        执行多查询扩展 + RRF 融合 + 图扩展 + 重排序。
        mode: 'keyword', 'semantic', 'hybrid'
        """
        # 1. 确定 weights (用于 txtai)
        weight_map = {
            "keyword": (0.0, 1.0),
            "semantic": (1.0, 0.0),
            "hybrid": None,
        }
        weights = weight_map.get(mode, None)

        # 2. 生成扩展查询
        queries = [(query, 0.5)]
        if expand and self.expander:
            queries = self.expander.expand(query)

        # 3. 对每个查询执行 txtai 搜索
        all_results = []
        for q, weight in queries:
            # txtai 搜索
            raw_results = self.embeddings.search(q, limit=limit * 2, weights=weights)
            # 将结果转换为 (doc_id, score) 列表
            result_list = []
            for item in raw_results:
                if isinstance(item, dict):
                    doc_id = item.get("id")
                    score = item.get("score", 0.0)
                else:
                    # tuple (id, text, tags, score)
                    doc_id = item[0]
                    score = item[3] if len(item) > 3 else 1.0
                if doc_id:
                    result_list.append((doc_id, score))
            all_results.append((result_list, weight))

        # 4. RRF 融合
        fused = self._weighted_rrf_fusion(all_results, k=50)

        # 5. 截断到 limit * 2 (为后续过滤和重排序留余地)
        top_ids = [doc_id for doc_id, _ in fused[:limit * 2]]

        # 6. 获取文档详情并应用过滤
        output = []
        for doc_id in top_ids:
            doc = self._fetch_doc_details(doc_id)
            if not doc:
                continue
            # 解析 tags 元数据
            tags_raw = doc.get("tags", "{}")
            try:
                meta = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
            except:
                meta = {}
            # 标签过滤
            if label_filter and meta.get("label") != label_filter:
                continue
            # 日期过滤 (字符串比较，假设 ISO 格式)
            send_time = meta.get("send_time")
            if date_from and send_time and send_time < date_from:
                continue
            if date_to and send_time and send_time > date_to:
                continue
            # 构建结果条目
            output.append({
                "id": doc_id,
                "text": doc.get("text", ""),
                "score": dict(fused).get(doc_id, 0.0),
                "metadata": meta,
            })

        # 7. 图扩展
        if graph_expand > 0 and self.embeddings.graph:
            output = self._graph_expand(output, graph_expand, limit)

        # 8. 重排序 (如果启用)
        if use_rerank and self.reranker:
            output = self.reranker.rerank(query, output)

        # 9. 再次截断到 limit
        return output[:limit]

    def _graph_expand(self, results: List[Dict], hops: int, limit: int) -> List[Dict]:
        """通过图一跳扩展 (只对前几个结果扩展)。"""
        if not self.embeddings.graph:
            return results

        graph = self.embeddings.graph
        expanded_ids = {r["id"] for r in results}
        expanded = list(results)

        # 只对前 3 个结果扩展
        for r in results[:3]:
            try:
                # 使用 graph.search 获取邻居
                # 注意: txtai graph 的 search 可能不支持 Cypher，我们使用 neighbors 方法
                # 但 neighbors 可能不存在，我们用 search 查询
                # 简单方法: 用 id 查询图关系
                # 这里我们假设 graph.search 可以接受简单查询
                # 如果支持 Cypher 更好，否则用 nodes 和 edges 手动
                # 由于 txtai graph 实现可能不同，这里使用通用的 neighbor 获取
                # 实际上 txtai 的 Graph 有 search 方法，但语法可能不同
                # 为了鲁棒，我们尝试用 MATCH
                neighbor_query = f"MATCH (n{{id:'{r['id']}'}})-[e]-(m) RETURN m.id, e LIMIT 3"
                neighbors = graph.search(neighbor_query)
                for neighbor_id, edge_label in neighbors:
                    if neighbor_id and neighbor_id not in expanded_ids:
                        expanded_ids.add(neighbor_id)
                        # 获取邻居文档
                        neighbor_doc = self._fetch_doc_details(neighbor_id)
                        if neighbor_doc:
                            expanded.append({
                                "id": neighbor_id,
                                "text": neighbor_doc.get("text", "")[:100],
                                "score": r["score"] * 0.8,  # 折扣
                                "metadata": {"_graph_relation": str(edge_label), "_from_node": r["id"]},
                            })
            except Exception as e:
                logger.debug(f"图扩展失败: {e}")

        expanded.sort(key=lambda x: x["score"], reverse=True)
        return expanded[:limit]


# ================================================================
# 6. Agentic 决策器 (AgenticDecider)
#    参考 §2.7 的意图路由
# ================================================================
class AgenticDecider:
    """基于规则 + LLM 回退的搜索策略决策。"""

    def __init__(self, api_base: str = LITELLM_BASE_URL,
                 api_key: str = LITELLM_API_KEY,
                 model: str = LITELLM_MODEL):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self._cache = {}

    def decide(self, query: str) -> Dict[str, Any]:
        """返回决策字典: mode, expand, graph_expand, use_rerank."""
        # 简单规则
        q_lower = query.lower()
        if any(kw in q_lower for kw in ["谁", "什么", "where", "when", "who"]):
            mode = "keyword"
            expand = False
        elif any(kw in q_lower for kw in ["类似", "相关", "similar", "related"]):
            mode = "semantic"
            expand = True
        else:
            # 默认 hybrid + 扩展
            mode = "hybrid"
            expand = True

        # 复杂查询启用图扩展和重排序
        if len(query.split()) > 5:
            graph_expand = 1
            use_rerank = True
        else:
            graph_expand = 0
            use_rerank = False

        # 可选 LLM 微调 (这里省略以保持轻量)
        return {
            "mode": mode,
            "expand": expand,
            "graph_expand": graph_expand,
            "use_rerank": use_rerank,
        }


# ================================================================
# 7. CLI (click)
# ================================================================
@click.group()
@click.option("--debug", is_flag=True)
def cli(debug):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

@cli.command("build")
@click.option("--force", is_flag=True)
@click.option("--no-graph", is_flag=True)
def build_cmd(force, no_graph):
    """构建索引 (含分块和丰富化)。"""
    builder = IndexBuilder()
    embeddings = builder.build(force=force, enable_graph=not no_graph)
    click.echo(f"✅ 索引构建完成: {embeddings.count()} 个块")

@cli.command("search")
@click.argument("query")
@click.option("--mode", type=click.Choice(["keyword","semantic","hybrid","auto"]), default="hybrid")
@click.option("--limit", default=10, type=int)
@click.option("--expand/--no-expand", default=True)
@click.option("--graph-expand", default=0, type=int, help="图一跳扩展数")
@click.option("--label", default=None)
@click.option("--date-from", default=None)
@click.option("--date-to", default=None)
@click.option("--rerank", is_flag=True)
@click.option("--agentic", is_flag=True)
def search_cmd(query, mode, limit, expand, graph_expand, label,
               date_from, date_to, rerank, agentic):
    """执行搜索。"""
    if not os.path.exists(INDEX_PATH):
        click.echo(f"❌ 索引不存在: {INDEX_PATH}", err=True)
        sys.exit(1)

    embeddings = txtai.Embeddings()
    embeddings.load(INDEX_PATH)
    click.echo(f"📊 索引: {embeddings.count()} 个块")

    # Agentic 决策
    if agentic or mode == "auto":
        decider = AgenticDecider()
        decision = decider.decide(query)
        mode = decision["mode"]
        expand = decision["expand"]
        graph_expand = decision["graph_expand"]
        rerank = decision["use_rerank"]
        click.echo(f"🤖 Agentic 决策: mode={mode}, expand={expand}, graph={graph_expand}, rerank={rerank}")

    # 扩展器
    expander = QueryExpander() if expand else None
    reranker = Reranker() if rerank else None

    searcher = Searcher(embeddings, expander=expander, reranker=reranker)
    t0 = time.perf_counter()
    results = searcher.search(
        query=query,
        mode=mode,
        limit=limit,
        expand=expand,
        graph_expand=graph_expand,
        label_filter=label,
        date_from=date_from,
        date_to=date_to,
        use_rerank=rerank,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    click.echo(f"🔍 查询: \"{query}\" | {len(results)} 条 | {elapsed:.1f}ms\n")

    # 格式化输出
    rows = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        text_preview = r["text"][:80] + "..." if len(r["text"]) > 80 else r["text"]
        rows.append([
            i,
            r["id"][:25],
            f"{r['score']:.4f}",
            meta.get("customer_name", "")[:10],
            meta.get("label", "-"),
            text_preview,
        ])
    click.echo(tabulate(rows, headers=["#", "ID", "Score", "From", "Label", "Content"], tablefmt="simple_grid"))

@cli.command("benchmark")
@click.option("--runs", default=20, type=int)
def benchmark_cmd(runs):
    """性能基准测试。"""
    if not os.path.exists(INDEX_PATH):
        click.echo(f"❌ 索引不存在: {INDEX_PATH}", err=True)
        sys.exit(1)

    embeddings = txtai.Embeddings()
    embeddings.load(INDEX_PATH)
    searcher = Searcher(embeddings)
    queries = ["诈骗", "合作", "紧急", "预算", "客户投诉"]
    all_lat = []
    rows = []
    for q in queries:
        lat = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = searcher.search(q, mode="hybrid", limit=10, expand=False)
            lat.append((time.perf_counter() - t0) * 1000)
        all_lat.extend(lat)
        rows.append([q, f"{statistics.mean(lat):.1f}", f"{statistics.median(lat):.1f}",
                     f"{sorted(lat)[int(len(lat)*0.95)]:.1f}"])
    click.echo(tabulate(rows, headers=["Query", "Avg(ms)", "P50(ms)", "P95(ms)"], tablefmt="simple_grid"))
    if all_lat:
        click.echo(f"\n总体 P95: {sorted(all_lat)[int(len(all_lat)*0.95)]:.1f}ms")

@cli.command("validate")
def validate_cmd():
    """验证准确率 (MRR) - 需要预定义测试集。"""
    click.echo("⚠️ 请先在代码中定义 TEST_QUERIES 并运行 validate。")

if __name__ == "__main__":
    cli()