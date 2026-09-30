#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""知识库数据加载与多模式检索引擎。

复用 generate-data.js 产出的 search-data.js（包含 TF-IDF 向量、词表、别名表），
在 Python 端实现关键词 / 实体 / 语义三种检索能力，作为 Agent 的工具。
"""
import json
import math
import os
import re
from typing import Any

# 数据文件路径：search-data.js 位于 Wiki 根目录（app/core/ 向上两级）
_WIKI_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_JS = os.path.join(_WIKI_ROOT, "search-data.js")


def _load_data() -> dict:
    """读取 search-data.js，解析出 window.WIKI_DATA 的 JSON 对象。"""
    with open(_DATA_JS, "r", encoding="utf-8") as f:
        text = f.read()
    # 匹配 window.WIKI_DATA = {...};
    m = re.search(r"window\.WIKI_DATA\s*=\s*(\{.*?\});\s*$", text, re.DOTALL)
    if not m:
        # 退化匹配：取等号后第一个完整 JSON
        m = re.search(r"window\.WIKI_DATA\s*=\s*(\{.*\})\s*;?\s*$", text, re.DOTALL)
    if not m:
        raise RuntimeError("无法从 search-data.js 解析 WIKI_DATA，请先运行 node generate-data.js")
    return json.loads(m.group(1))


# 与 generate-data.js 中 STOP_CHARS / tokenize 保持一致
_STOP_CHARS_RE = re.compile(
    r"[\s\n\r\t，。、；：！？“”‘’（）【】《》「」\[\]|·\-—…~`!@#$%^&*()_+=\-{}\|;:'\",.<>/?\\0-9]"
)


def _tokenize(text: str) -> list[str]:
    """中文 bigram + 英文单词 + 中文单字，与 JS 端分词保持一致。"""
    if not text:
        return []
    tokens: list[str] = []
    segs = [s for s in _STOP_CHARS_RE.split(text) if s]
    for seg in segs:
        if re.fullmatch(r"[a-zA-Z]+", seg):
            tokens.append(seg.lower())
            continue
        chars = list(seg)
        for i in range(len(chars) - 1):
            if not _STOP_CHARS_RE.match(chars[i]) and not _STOP_CHARS_RE.match(chars[i + 1]):
                tokens.append(chars[i] + chars[i + 1])
        for c in chars:
            if not _STOP_CHARS_RE.match(c):
                tokens.append(c)
    return tokens


class WikiSearchEngine:
    """知识库检索引擎：封装关键词 / 实体 / 语义三种检索。"""

    def __init__(self) -> None:
        self.data: dict = _load_data()
        self.concepts: list[dict] = self.data.get("concepts", [])
        self.entities: list[dict] = self.data.get("entities", [])
        self.sources: list[dict] = self.data.get("sources", [])
        self.aliases: dict = {k.lower(): v for k, v in self.data.get("aliases", {}).items()}
        self.vocab: list[str] = self.data.get("vocab", [])
        self.idf: list[float] = self.data.get("idf", [])
        # 构建 token -> index 反查表，加速查询向量的稀疏构建
        self.vocab_index: dict = {t: i for i, t in enumerate(self.vocab)}
        # 所有条目合并列表
        self.all_items: list[dict] = []
        for x in self.concepts:
            x.setdefault("type", "concept")
            self.all_items.append(x)
        for x in self.entities:
            x.setdefault("type", "entity")
            self.all_items.append(x)
        for x in self.sources:
            x.setdefault("type", "source")
            self.all_items.append(x)
        # 标题 -> 条目 反查表
        self.by_title: dict = {x["title"]: x for x in self.all_items}
        # 把每个条目的 title 也作为自身别名加入
        for x in self.all_items:
            self.aliases.setdefault(x["title"].lower(), x["title"])

    # ---- 工具函数 ----
    def expand_keyword(self, kw: str) -> list[str]:
        """别名扩展：返回可能的规范标题集合。"""
        if not kw:
            return []
        lower = kw.lower().strip()
        result: list[str] = []
        if lower in self.aliases:
            result.append(self.aliases[lower])
        for a, target in self.aliases.items():
            if lower in a and a != lower and target not in result:
                result.append(target)
        return result

    def _doc_text(self, item: dict) -> str:
        """拼接用于检索的全文（与 JS 端 docText 一致）。"""
        parts: list[str] = [item.get("title", "")]
        for k in ("description", "summary", "keyPoints"):
            v = item.get(k)
            if v:
                parts.append(v)
        for k in ("keywords", "relatedEntities", "relatedConcepts", "links"):
            v = item.get(k)
            if v:
                parts.append(" ".join(v))
        if item.get("category"):
            parts.append(item["category"])
        body = item.get("body", "")
        if body:
            parts.append(re.sub(r"\[\[|\]\]", "", body))
        return " ".join(parts)

    # ---- 三种检索方式 ----
    def keyword_search(self, query: str, type_filter: str = None, limit: int = 20) -> list[dict]:
        """关键词搜索：子串匹配 + 别名扩展，按标题精确/前缀/引用数排序。"""
        kw = (query or "").strip().lower()
        expanded = self.expand_keyword(kw)
        result: list[tuple[dict, int, int, float]] = []
        for item in self.all_items:
            if type_filter and item.get("type") != type_filter:
                continue
            title = (item.get("title") or "").lower()
            desc = (item.get("description") or item.get("summary") or "").lower()
            kp = (item.get("keyPoints") or "").lower()
            body = (item.get("body") or "").lower()
            keywords = " ".join(item.get("keywords") or []).lower()
            hit = (
                kw in title or kw in desc or kw in kp or kw in body or kw in keywords
                or any(title == e.lower() for e in expanded)
            )
            if not hit:
                continue
            exact = 0 if (title == kw or item["title"] in expanded) else 1
            start = 0 if title.startswith(kw) else 1
            score = (item.get("refCount", 0)) + self._interaction_score(item)
            result.append((item, exact, start, score))
        # 排序
        result.sort(key=lambda r: (r[1], r[2], -r[3]))
        return [self._slim(r[0]) for r in result[:limit]]

    def entity_search(self, query: str, subgroup: str = "all", limit: int = 30) -> list[dict]:
        """实体搜索：限定 entity 类型，按子分类过滤。"""
        subgroups = {
            "training": {"训练方案", "训练方法", "居家训练"},
            "scenario": {"冲突场景", "行为场景", "校园场景"},
            "method": {"干预方法", "应对策略", "陪读方案"},
        }
        kw = (query or "").strip().lower()
        expanded = self.expand_keyword(kw)
        result: list[dict] = []
        for item in self.entities:
            cat = item.get("category")
            if subgroup != "all":
                allowed = subgroups.get(subgroup, set())
                if cat not in allowed:
                    continue
            if kw:
                title = (item.get("title") or "").lower()
                desc = (item.get("description") or "").lower()
                hit = (kw in title or kw in desc or title in [e.lower() for e in expanded])
                if not hit:
                    continue
            result.append(self._slim(item))
            if len(result) >= limit:
                break
        return result

    def semantic_search(self, query: str, type_filter: str = None, limit: int = 20) -> list[dict]:
        """语义搜索：基于 TF-IDF 向量的余弦相似度排序。"""
        qv = self._build_query_vector(query)
        if not qv:
            return []
        scored: list[tuple[dict, float]] = []
        for item in self.all_items:
            if type_filter and item.get("type") != type_filter:
                continue
            sim = self._cosine(qv, item.get("tfidf", {}))
            if sim > 0.01:
                scored.append((item, sim))
        scored.sort(key=lambda r: -r[1])
        out = []
        for item, sim in scored[:limit]:
            x = self._slim(item)
            x["similarity"] = round(sim, 4)
            out.append(x)
        return out

    def get_detail(self, title: str) -> dict:
        """获取条目详情，包含反向引用列表。"""
        clean = re.sub(r"\[\[|\]\]", "", title or "").strip()
        item = self.by_title.get(clean)
        if not item:
            target = self.aliases.get(clean.lower())
            if target:
                item = self.by_title.get(target)
        if not item:
            return {"error": f"未找到条目：{title}"}
        detail = self._slim(item)
        # 反向引用
        if item.get("type") in ("concept", "entity"):
            back_refs = [s["title"] for s in self.sources if item["title"] in (s.get("links") or [])]
            detail["backRefs"] = back_refs
        return detail

    # ---- 内部辅助 ----
    def _interaction_score(self, item: dict) -> float:
        stats = item.get("stats") or {}
        s = 0
        try:
            s += int(stats.get("点赞", 0))
            s += int(stats.get("收藏", 0)) * 2
        except (ValueError, TypeError):
            pass
        return s / 100

    def _slim(self, item: dict) -> dict:
        """生成精简版条目（去掉 tfidf 等大字段），用于工具返回。"""
        out = {
            "id": item.get("id"),
            "type": item.get("type"),
            "title": item.get("title"),
            "category": item.get("category"),
            "description": item.get("description") or item.get("summary") or "",
            "keyPoints": item.get("keyPoints", ""),
            "refCount": item.get("refCount", 0),
            "stats": item.get("stats", {}),
            "links": item.get("links", []),
            "keywords": item.get("keywords", []),
            "relatedEntities": item.get("relatedEntities", []),
            "relatedConcepts": item.get("relatedConcepts", []),
            "file": item.get("file", ""),
        }
        return out

    def _build_query_vector(self, query: str) -> dict:
        """构建查询的 TF-IDF 归一化向量（稀疏 dict: token_index -> weight）。"""
        tokens = _tokenize(query or "")
        if not tokens:
            return {}
        tf: dict = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec: dict = {}
        norm = 0.0
        for t, freq in tf.items():
            idx = self.vocab_index.get(t)
            if idx is None:
                continue
            w = freq * self.idf[idx]
            vec[idx] = w
            norm += w * w
        if norm == 0:
            return {}
        norm = math.sqrt(norm)
        for k in vec:
            vec[k] = vec[k] / norm
        return vec

    def _cosine(self, qv: dict, dv: dict) -> float:
        """稀疏向量余弦相似度（输入已归一化）。

        注意：qv 的 key 是 int（来自 vocab_index），dv 的 key 是 str
        （来自 JSON 反序列化）。这里统一转 int 查找。
        """
        if not qv or not dv:
            return 0.0
        # 让较短的向量做迭代，减少遍历次数
        a, b = (qv, dv) if len(qv) < len(dv) else (dv, qv)
        # 预处理 b：把字符串 key 转成 int，加速查找
        b_int = {}
        for k, v in b.items():
            b_int[int(k)] = v
        return sum(w * b_int.get(int(k), 0.0) for k, w in a.items())

    # ---- 供 Agent 使用的工具描述 ----
    def tool_specs(self) -> list[dict]:
        """返回 OpenAI / dashscope function-calling 兼容的工具描述。"""
        return [
            {
                "name": "keyword_search",
                "description": "关键词搜索知识库（训练方案、行为场景、概念、实战资料）。"
                              "适用于明确知道名称或关键词的查询。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词，如 '脱敏训练'、'社交冲突'、'冲动控制'"},
                        "type_filter": {
                            "type": "string",
                            "enum": ["concept", "entity", "source"],
                            "description": "可选：限定类型。concept=概念，entity=实体，source=原始资料。不传则全部类型。",
                        },
                        "limit": {"type": "integer", "description": "返回条数上限，默认 20", "default": 20},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "entity_search",
                "description": "实体搜索：限定为训练方案/冲突场景/干预方法。可按子分类过滤。"
                              "适用于查训练名称、特定行为场景等明确实体类型的查询。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "可选的过滤关键词"},
                        "subgroup": {
                            "type": "string",
                            "enum": ["all", "training", "scenario", "method"],
                            "description": "子分类：all=全部，training=训练方案，scenario=冲突场景，method=干预方法。默认 all",
                            "default": "all",
                        },
                        "limit": {"type": "integer", "description": "返回条数上限，默认 30", "default": 30},
                    },
                },
            },
            {
                "name": "semantic_search",
                "description": "语义搜索：基于 TF-IDF 向量余弦相似度排序，召回语义相近的条目。"
                              "适用于模糊描述、自然语言提问，如 '哪支球队是夺冠热门'。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "自然语言查询，如 '夺冠热门球队有哪些'"},
                        "type_filter": {
                            "type": "string",
                            "enum": ["concept", "entity", "source"],
                            "description": "可选：限定类型。",
                        },
                        "limit": {"type": "integer", "description": "返回条数上限，默认 20", "default": 20},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_detail",
                "description": "获取某个条目的完整详情，包括简介、关键要点、相关实体/概念、反向引用。"
                              "当用户追问某个条目的细节时使用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "条目标题，如 '梅西'、'世界杯'、'大力神杯'"},
                    },
                    "required": ["title"],
                },
            },
        ]

    def call_tool(self, name: str, arguments: dict) -> str:
        """统一工具调用入口，返回 JSON 字符串。"""
        try:
            if name == "keyword_search":
                result = self.keyword_search(
                    arguments.get("query", ""),
                    arguments.get("type_filter"),
                    arguments.get("limit", 20),
                )
            elif name == "entity_search":
                result = self.entity_search(
                    arguments.get("query", ""),
                    arguments.get("subgroup", "all"),
                    arguments.get("limit", 30),
                )
            elif name == "semantic_search":
                result = self.semantic_search(
                    arguments.get("query", ""),
                    arguments.get("type_filter"),
                    arguments.get("limit", 20),
                )
            elif name == "get_detail":
                result = self.get_detail(arguments.get("title", ""))
            else:
                result = {"error": f"未知工具：{name}"}
        except Exception as e:  # noqa: BLE001
            result = {"error": f"工具执行异常：{e}"}
        return json.dumps(result, ensure_ascii=False)