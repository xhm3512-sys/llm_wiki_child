#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agentic Search Agent：基于 Qwen 大模型的推理决策搜索。

通过 dashscope 调用 Qwen，使用 function-calling 让模型自主决定调用
哪种检索工具（关键词 / 实体 / 语义 / 详情），最多迭代 max_rounds 轮，
最终汇总生成回答。

通过 search_stream() 生成器逐步产出推理过程，便于前端可视化。
"""
import json
import os
from datetime import datetime
from typing import Any

import dashscope
from dashscope import Generation

from .search_engine import WikiSearchEngine

# 从环境变量读取 API Key（DASHSCOPE_API_KEY 是阿里云百炼的默认变量）
_API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
if _API_KEY:
    dashscope.api_key = _API_KEY

# 模型名称：Qwen-Plus 兼具能力与速度
_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")
_MAX_ROUNDS = int(os.environ.get("AGENT_MAX_ROUNDS", "6"))


SYSTEM_PROMPT_TEMPLATE = """你是儿童社交与冲突行为训练知识库的智能搜索助手，当前知识库覆盖儿童被动冲突动手、社交脱敏训练、情绪管理、校园行为引导等相关的概念、训练方案与实战资料。

当前时间：{timestamp}

你可以调用以下工具来检索知识库：
1. keyword_search —— 关键词搜索：明确知道名称或关键词时使用（如训练名称、行为术语、概念词）
2. entity_search   —— 实体搜索：限定具体训练方案、行为场景、干预方法等实体条目，可按子分类过滤
3. semantic_search —— 语义搜索：用自然语言描述问题时使用（如"孩子为什么在学校容易跟人打架"）
4. get_detail      —— 获取条目详情：用户追问某训练方案或场景的细节时使用

搜索策略：
- 对于模糊/开放性问题（如"孩子打人怎么办"），优先用 semantic_search 召回，再用 get_detail 补充关键训练方案细节
- 对于明确术语或方案查询，用 keyword_search 或 entity_search
- 可以多轮调用不同工具组合检索，确保回答全面准确
- 工具返回的 JSON 中，每条结果包含 title / type / category / description / keyPoints 等字段

回答要求：
- 用中文回答，结构清晰，重要信息用列表呈现
- 引用知识库中的具体条目名称（用【】标注，如【无意触碰脱敏训练】）
- 若知识库中找不到相关信息，明确告知用户并建议调整问题
"""


class AgenticSearchAgent:
    """Agent 主类：协调 LLM 推理与工具调用。"""

    def __init__(self, max_rounds: int = _MAX_ROUNDS) -> None:
        self.engine = WikiSearchEngine()
        self.max_rounds = max_rounds
        self.model = _MODEL

    def _build_system_prompt(self) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return SYSTEM_PROMPT_TEMPLATE.format(timestamp=ts)

    def search_stream(self, question: str):
        """流式产出推理步骤，供前端逐步渲染。

        yield 的每个 step 是 dict，包含 step 字段标识阶段：
        - init          初始化，含 system_prompt / question / timestamp
        - tool_call     一轮工具调用，含 round / tool_name / arguments / result / llm 原始响应
        - final         最终回答，含 final_answer / 对话上下文 / LLM 原始响应
        - fallback      API 不可用，降级到本地搜索
        - error         异常
        """
        if not _API_KEY:
            # 无 API Key，降级到本地语义搜索
            yield {
                "step": "fallback",
                "intent": "semantic",
                "reason": "未检测到 DASHSCOPE_API_KEY，降级为本地语义检索",
            }
            yield from self._local_fallback(question)
            return

        system_prompt = self._build_system_prompt()
        yield {
            "step": "init",
            "system_prompt": system_prompt,
            "question": question,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 构建 messages（OpenAI / dashscope 兼容格式）
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        tools = self.engine.tool_specs()
        round_idx = 0

        try:
            while round_idx < self.max_rounds:
                round_idx += 1
                # 调用 Qwen，开启 function calling
                response = Generation.call(
                    model=self.model,
                    messages=messages,
                    tools=[{"type": "function", "function": t} for t in tools],
                    result_format="message",
                )

                # 把原始响应保存
                raw_response = self._safe_dumps(response)

                if response.status_code != 200:
                    yield {
                        "step": "fallback",
                        "intent": "semantic",
                        "reason": f"Qwen API 异常: {response.code} {response.message}",
                    }
                    yield from self._local_fallback(question)
                    return

                # 取 assistant 消息
                choice = response.output.choices[0]
                assistant_msg = choice.message
                # 兼容 tool_calls 字段名
                tool_calls = assistant_msg.get("tool_calls") or []

                # 记录 assistant 推理（含可能的 tool_calls）
                llm_text = assistant_msg.get("content", "") or ""
                # 把 assistant 消息加入历史（保留 tool_calls 让模型知道自己的决策）
                messages.append(self._normalize_assistant_msg(assistant_msg))

                if not tool_calls:
                    # 模型直接给出最终回答（无工具调用）
                    yield {
                        "step": "final",
                        "round": round_idx,
                        "final_answer": llm_text,
                        "final_prompt_type": "模型直接生成（无需工具调用）",
                        "llm_raw_response": raw_response,
                        "final_messages_context": self._index_messages(messages),
                    }
                    return

                # 有工具调用：逐个执行
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}

                    result = self.engine.call_tool(tool_name, args)

                    yield {
                        "step": "tool_call",
                        "round": round_idx,
                        "tool_name": tool_name,
                        "arguments": args,
                        "result": result,
                        "llm_text_content": llm_text,
                        "llm_raw_response": raw_response,
                    }

                    # 把工具结果作为 tool 消息追加
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "name": tool_name,
                    })

            # 达到最大轮数后强制让模型生成最终回答
            final_resp = Generation.call(
                model=self.model,
                messages=messages,
                result_format="message",
            )
            raw_final = self._safe_dumps(final_resp)
            if final_resp.status_code == 200:
                final_text = final_resp.output.choices[0].message.content or ""
            else:
                final_text = "（已达最大搜索轮数，未能生成最终回答）"

            yield {
                "step": "final",
                "round": round_idx,
                "final_answer": final_text,
                "final_prompt_type": "达到最大轮数后强制生成最终回答",
                "llm_raw_response": raw_final,
                "final_messages_context": self._index_messages(messages),
            }

        except Exception as e:  # noqa: BLE001
            yield {
                "step": "error",
                "error": str(e),
                "round": round_idx,
            }

    # ---- 本地降级：当 API 不可用时 ----
    def _local_fallback(self, question: str):
        """无 LLM 时，用语义搜索 + 关键词搜索做降级回答。"""
        sem = self.engine.semantic_search(question, limit=8)
        kw = self.engine.keyword_search(question, limit=5)
        # 合并去重
        seen = set()
        merged: list[dict] = []
        for x in sem + kw:
            if x["title"] not in seen:
                seen.add(x["title"])
                merged.append(x)

        lines = [f"（本地降级模式）根据问题「{question}」检索到以下相关条目：", ""]
        if not merged:
            lines.append("未找到相关内容，请尝试更换关键词。")
        else:
            for i, x in enumerate(merged, 1):
                sim = x.get("similarity")
                sim_str = f"（相似度 {sim:.2%}）" if sim else ""
                desc = x.get("description") or x.get("keyPoints", "")[:80] or "暂无简介"
                lines.append(
                    f"{i}. 【{x['title']}】（{self._type_label(x['type'])}）{sim_str}\n   {desc}"
                )

        answer = "\n".join(lines)
        yield {
            "step": "final",
            "round": 0,
            "final_answer": answer,
            "final_prompt_type": "本地降级：语义检索 + 关键词检索",
            "llm_raw_response": "",
            "final_messages_context": [],
            "local_results": merged,
        }

    # ---- 辅助 ----
    @staticmethod
    def _type_label(t: str) -> str:
        return {"concept": "概念", "entity": "实体", "source": "资料"}.get(t, t)

    @staticmethod
    def _safe_dumps(obj: Any) -> str:
        """把 dashscope 响应对象安全转成 JSON 字符串。"""
        try:
            # dashscope 响应有 to_dict / __dict__
            if hasattr(obj, "to_dict"):
                return json.dumps(obj.to_dict(), ensure_ascii=False, default=str)
            if hasattr(obj, "__dict__"):
                return json.dumps(obj.__dict__, ensure_ascii=False, default=str)
            return json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            return str(obj)

    @staticmethod
    def _normalize_assistant_msg(msg: dict) -> dict:
        """归一化 assistant 消息：保留 content 和 tool_calls，去掉空字段。"""
        out: dict = {"role": "assistant"}
        if msg.get("content"):
            out["content"] = msg["content"]
        if msg.get("tool_calls"):
            # 转成简化结构，方便序列化展示
            out["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", "{}"),
                    },
                }
                for tc in msg["tool_calls"]
            ]
        return out

    @staticmethod
    def _index_messages(messages: list[dict]) -> list[dict]:
        """给每条消息加 index 字段并精简，供前端展示对话上下文。"""
        out: list[dict] = []
        for i, m in enumerate(messages):
            item: dict = {"index": i, "role": m.get("role", "?")}
            content = m.get("content", "")
            if isinstance(content, (dict, list)):
                content = json.dumps(content, ensure_ascii=False)
            item["content"] = content or ""
            if m.get("tool_calls"):
                item["tool_calls"] = [
                    {
                        "tool_name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", {}),
                    }
                    for tc in m["tool_calls"]
                ]
            if m.get("name"):
                item["name"] = m["name"]
            out.append(item)
        return out