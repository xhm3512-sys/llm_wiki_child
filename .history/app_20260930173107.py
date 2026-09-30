#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""儿童社交训练知识库 Agentic Search - 智能搜索可视化界面。

基于 Qwen 大模型推理决策，复用 search-data.js 的 TF-IDF 向量做语义检索，
通过 function-calling 让 Agent 自主选择关键词/实体/语义/详情工具，
并完整可视化每一轮推理与工具调用过程。
"""
import json
import os
import sys
from datetime import datetime

import streamlit as st

# 把 app 子目录加入 sys.path（core 包在 app/core 下，而非 Wiki 根目录）
_APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from core.agent import AgenticSearchAgent  # noqa: E402
from core.search_engine import WikiSearchEngine  # noqa: E402


st.set_page_config(
    page_title="儿童社交训练知识库 Agentic Search",
    page_icon="🧒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    @media (max-width: 768px) {
        .main-header { font-size: 1.6rem; }
        .sub-header { font-size: 0.95rem; margin-bottom: 1rem; }
    }
    .source-badge {
        display: inline-block;
        background-color: #E3F2FD;
        color: #1565C0;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-right: 0.5rem;
    }
    .trace-label {
        display: inline-block;
        padding: 0.1rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 0.3rem;
    }
    .trace-llm { background: #E8F5E9; color: #2E7D32; }
    .trace-tool { background: #FFF3E0; color: #E65100; }
    .trace-result { background: #E3F2FD; color: #1565C0; }
    .trace-warn { background: #FFEBEE; color: #C62828; }
    .trace-prompt { background: #F3E5F5; color: #6A1B9A; }
    .trace-final { background: #E8F5E9; color: #1B5E20; font-size: 1rem; }
    .status-running { color: #FB8C00; font-weight: bold; }
    .status-complete { color: #43A047; font-weight: bold; }
    .type-concept { color: #1565C0; }
    .type-entity { color: #C2185B; }
    .type-source { color: #F57F17; }
    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        div[data-testid="stHorizontalBlock"] > div {
            width: 100% !important;
        }
        .stButton button {
            width: 100%;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_agent():
    return AgenticSearchAgent()


@st.cache_resource
def get_engine():
    return WikiSearchEngine()


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_searching" not in st.session_state:
        st.session_state.is_searching = False
    if "history_records" not in st.session_state:
        st.session_state.history_records = []
    if "active_history_idx" not in st.session_state:
        st.session_state.active_history_idx = None


def perform_search(agent, question):
    """执行 Agent 搜索，流式渲染推理过程。"""
    status_area = st.empty()
    prompt_area = st.empty()
    rounds_area = st.empty()
    final_area = st.empty()

    all_tool_calls = []
    round_idx = 0

    try:
        for step in agent.search_stream(question):
            if step["step"] == "init":
                status_area.info("🔄 Agent 推理决策进行中...")
                prompt_text = step.get("system_prompt", "")
                with prompt_area.container():
                    st.markdown("##### 📤 用户问题")
                    st.info(step.get("question", question))
                    st.markdown(
                        "##### <span class='trace-label trace-prompt'>⚙️ 1. System Prompt 生成</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"当前时间: {step.get('timestamp', 'N/A')}  |  共 {len(prompt_text)} 字符"
                    )
                    with st.expander("📋 查看完整 System Prompt", expanded=False):
                        st.code(prompt_text, language="markdown")

            elif step["step"] == "tool_call":
                all_tool_calls.append(step)
                round_idx = step["round"]

                with rounds_area.container():
                    for i, tc in enumerate(all_tool_calls):
                        # 同一轮的第一个 tool_call 才显示轮次标题
                        is_new_round = i == 0 or all_tool_calls[i - 1]["round"] != tc["round"]
                        if is_new_round:
                            st.markdown("---")
                            st.markdown(
                                f"##### <span class='trace-label trace-llm'>🧠 2.{tc['round']} "
                                f"第 {tc['round']} 轮 — Qwen 模型推理</span>",
                                unsafe_allow_html=True,
                            )
                            llm_text = tc.get("llm_text_content", "")
                            if llm_text:
                                st.caption(f"LLM 推理说明: {llm_text[:150]}")
                            with st.expander("📡 Qwen API 原始响应 JSON"):
                                st.code(tc.get("llm_raw_response", ""), language="json")

                        st.markdown(
                            f"<span class='trace-label trace-tool'>🔧 执行工具: "
                            f"**{tc['tool_name']}**</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("**调用参数:**")
                        st.json(tc.get("arguments", {}))
                        st.markdown("**执行结果:**")
                        result_raw = tc.get("result", "")
                        try:
                            result_json = json.loads(result_raw)
                            if isinstance(result_json, list):
                                # 检索结果列表：用表格呈现
                                st.caption(f"共召回 {len(result_json)} 条")
                                _render_result_table(result_json)
                            elif isinstance(result_json, dict) and "error" in result_json:
                                st.warning(result_json["error"])
                            elif isinstance(result_json, dict):
                                # 单条详情
                                _render_detail(result_json)
                            else:
                                st.code(result_raw, language="json")
                        except (json.JSONDecodeError, TypeError):
                            st.text(result_raw[:500])

            elif step["step"] == "fallback":
                with rounds_area.container():
                    st.markdown(
                        f"<span class='trace-label trace-warn'>⚠️ API 不可用，"
                        f"降级到本地 {step.get('intent', '')} 检索</span>",
                        unsafe_allow_html=True,
                    )
                    if step.get("reason"):
                        st.caption(step["reason"])

            elif step["step"] == "final":
                status_area.success(
                    f"✅ 搜索完成（共 {step.get('round', 0)} 轮）"
                )
                with final_area.container():
                    st.markdown("---")
                    st.markdown(
                        "### <span class='trace-label trace-final'>🎯 3. 最终回答生成</span>",
                        unsafe_allow_html=True,
                    )

                    prompt_type = step.get("final_prompt_type", "模型生成最终回答")
                    st.info(f"**生成方式**: {prompt_type}")

                    if step.get("llm_raw_response"):
                        st.markdown("##### 📡 Qwen API 最终响应（生成最终回答的完整 API 返回）")
                        with st.expander("查看 Qwen API 原始响应 JSON", expanded=False):
                            st.code(step.get("llm_raw_response", ""), language="json")

                    ctx = step.get("final_messages_context", [])
                    if ctx:
                        st.markdown(
                            "##### 📋 生成最终回答时的完整对话上下文（发送给 LLM 的 messages 数组）"
                        )
                        with st.expander("查看完整对话消息上下文", expanded=False):
                            for m in ctx:
                                idx = m.get("index", "?")
                                role = m.get("role", "?")
                                role_emoji = {
                                    "system": "⚙️",
                                    "user": "👤",
                                    "assistant": "🤖",
                                    "tool": "🔧",
                                }.get(role, "❓")
                                content = m.get("content", "")
                                if role == "assistant" and m.get("tool_calls"):
                                    st.markdown(
                                        f"**{role_emoji} [{idx}] {role} — 工具调用决策**"
                                    )
                                    if content:
                                        st.code(content, language="json")
                                    for tc in m["tool_calls"]:
                                        args = tc.get("arguments", {})
                                        if isinstance(args, str):
                                            try:
                                                args = json.loads(args)
                                            except json.JSONDecodeError:
                                                pass
                                        st.caption(
                                            f"     🔨 调用工具: **{tc.get('tool_name', '')}** "
                                            f"| 参数: {json.dumps(args, ensure_ascii=False)}"
                                        )
                                elif role == "tool":
                                    st.markdown(
                                        f"**{role_emoji} [{idx}] {role} — 工具返回数据**"
                                        + (f"（{m.get('name', '')}）" if m.get("name") else "")
                                    )
                                    try:
                                        parsed = json.loads(content)
                                        if isinstance(parsed, list):
                                            st.caption(f"共 {len(parsed)} 条结果")
                                            _render_result_table(parsed)
                                        else:
                                            st.json(parsed)
                                    except (json.JSONDecodeError, TypeError):
                                        st.code(content[:1000], language="json")
                                else:
                                    st.markdown(f"**{role_emoji} [{idx}] {role}**")
                                    st.code(content, language="text")

                    # 本地降级时展示检索结果
                    local_results = step.get("local_results", [])
                    if local_results:
                        st.markdown("##### 📚 本地检索召回结果")
                        _render_result_table(local_results)

                    st.markdown("---")
                    st.markdown("### 📝 最终回答")
                    st.markdown(step.get("final_answer", ""))
                return step.get("final_answer", "")

            elif step["step"] == "error":
                status_area.error(f"搜索异常: {step.get('error', '')}")
                return f"搜索过程中出现错误：{step.get('error', '')}"

    except Exception as e:  # noqa: BLE001
        status_area.error(f"搜索异常: {e}")
        return f"搜索过程中出现错误：{str(e)}"

    status_area.warning("⚠️ 达到最大搜索轮数")
    return "搜索已达到最大轮数，请尝试更精确的问题。"


def _render_result_table(items):
    """以表格形式渲染检索结果列表。"""
    if not items:
        st.caption("无结果")
        return
    rows = []
    for x in items:
        desc = (x.get("description") or x.get("keyPoints", "")[:80] or "暂无简介")
        sim = x.get("similarity")
        rows.append(
            {
                "标题": x.get("title", ""),
                "类型": _type_label(x.get("type", "")),
                "分类": x.get("category", ""),
                "简介": desc[:60] + ("..." if len(desc) > 60 else ""),
                "引用": x.get("refCount", 0),
                "相似度": f"{sim:.2%}" if sim else "",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_detail(detail: dict):
    """渲染单条详情。"""
    title = detail.get("title", "")
    type_str = _type_label(detail.get("type", ""))
    st.markdown(f"#### 【{title}】 <span class='type-{detail.get('type','')}'>{type_str}</span>",
                unsafe_allow_html=True)
    if detail.get("category"):
        st.caption(f"分类: {detail['category']}")
    if detail.get("description"):
        st.markdown(f"**简介:** {detail['description']}")
    if detail.get("keyPoints"):
        with st.expander("关键要点", expanded=False):
            st.markdown(detail["keyPoints"])
    stats = detail.get("stats", {})
    if stats:
        cols = st.columns(len(stats))
        for col, (k, v) in zip(cols, stats.items()):
            col.metric(k, v)
    for field, label in [
        ("keywords", "关键词"),
        ("relatedEntities", "相关实体"),
        ("relatedConcepts", "关联概念"),
        ("backRefs", "被以下资料引用"),
    ]:
        vals = detail.get(field, [])
        if vals:
            with st.expander(f"{label}（{len(vals)}）", expanded=False):
                st.write("、".join(vals))


def _type_label(t: str) -> str:
    return {"concept": "概念", "entity": "训练方案", "source": "实战资料"}.get(t, t)


# ===================== 主流程 =====================
init_session_state()
agent = get_agent()
engine = get_engine()

st.markdown('<p class="main-header">🧒 儿童社交训练知识库 Agentic Search</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">基于 Qwen 大模型推理决策的智能搜索 '
    "— 训练方案 / 冲突场景 / 语义向量多模式检索，完整过程可视化</p>",
    unsafe_allow_html=True,
)

pending_question = st.session_state.pop("pending_question", None)
if pending_question:
    st.session_state.question_input = pending_question

main_col, hist_col = st.columns([4, 1])

with main_col:
    with st.form("search_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input(
                "请输入您的问题：",
                placeholder="例如：孩子被同学碰一下就动手怎么办？如何做社交冲突脱敏训练？",
                label_visibility="collapsed",
                key="question_input",
            )
        with col2:
            search_button = st.form_submit_button("🔍 搜索", use_container_width=True)

    st.markdown("**💡 示例问题（点击试用）:**")
    example_cols = st.columns(4)
    examples = [
        "孩子在学校容易跟人打架怎么办？",
        "如何做无意触碰脱敏训练？",
        "社交冲突三选一替代训练具体怎么做？",
        "排队拥挤推人的核心原因是什么？",
    ]
    for c, ex in zip(example_cols, examples):
        if c.button(ex, key=f"ex_{ex}"):
            st.session_state.pending_question = ex

    run_question = None
    if search_button and question and question.strip():
        run_question = question.strip()
    elif pending_question:
        run_question = pending_question

    if run_question:
        st.session_state.is_searching = True
        final_answer = perform_search(agent, run_question)
        st.session_state.messages.append({"role": "user", "content": run_question})
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        st.session_state.history_records.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": run_question,
            "answer": final_answer,
        })
        st.session_state.is_searching = False

    st.divider()
    st.markdown("### 💬 对话历史")
    if st.session_state.messages:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg["content"])
    else:
        st.info("👆 请在上方输入问题开始搜索")

with hist_col:
    st.markdown("### 🕐 搜索历史")
    records = list(reversed(st.session_state.history_records))

    if not records:
        st.info("暂无历史记录")
    else:
        st.caption(f"共 {len(records)} 条记录")
        for i, rec in enumerate(records):
            real_idx = len(records) - 1 - i
            ts = rec["timestamp"]
            q_preview = rec["question"][:40] + ("..." if len(rec["question"]) > 40 else "")
            is_active = st.session_state.active_history_idx == real_idx

            with st.container():
                col_btn, col_del = st.columns([4, 1])
                with col_btn:
                    label = f"**{ts.split(' ')[1]}**  {q_preview}"
                    if st.button(label, key=f"hist_{real_idx}", use_container_width=True):
                        if is_active:
                            st.session_state.active_history_idx = None
                        else:
                            st.session_state.active_history_idx = real_idx
                        st.rerun()
                with col_del:
                    if st.button("🗑️", key=f"del_{real_idx}"):
                        st.session_state.history_records.pop(real_idx)
                        if st.session_state.active_history_idx == real_idx:
                            st.session_state.active_history_idx = None
                        elif st.session_state.active_history_idx is not None and st.session_state.active_history_idx > real_idx:
                            st.session_state.active_history_idx -= 1
                        st.rerun()

                if is_active:
                    st.markdown("---")
                    st.markdown(f"**⏰ {rec['timestamp']}**")
                    st.markdown("**👤 问题:**")
                    st.markdown(rec["question"])
                    st.markdown("**🤖 回答:**")
                    st.markdown(rec["answer"])
                    st.markdown("---")

        if st.button("🗑️ 清空全部历史", use_container_width=True):
            st.session_state.history_records = []
            st.session_state.messages = []
            st.session_state.active_history_idx = None
            st.rerun()

# ===================== 侧边栏 =====================
with st.sidebar:
    st.markdown("### 📚 数据源")
    st.markdown(
        """
    <span class="source-badge">📖 concept/ 概念</span>
    <span class="source-badge">🎯 entity/ 训练方案</span>
    <span class="source-badge">📰 source/ 实战资料</span>
    <span class="source-badge">🔢 TF-IDF 向量</span>
    <span class="source-badge">🔗 Wiki 链接图</span>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("### ℹ️ 系统信息")
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    st.markdown(
        f"""
    - **模型**: Qwen (qwen-plus)
    - **API Key**: {'✅ 已配置' if api_key else '❌ 未配置（将降级本地搜索）'}
    - **最大搜索轮次**: {agent.max_rounds}
    - **词表规模**: {len(engine.vocab)} tokens
    - **知识库**: 儿童社交冲突训练 & 居家干预方案
    - **降级模式**: 本地 TF-IDF 语义检索（API 不可用时）
    """
    )

    st.markdown("### 🔧 Agent 工具")
    for t in engine.tool_specs():
        st.markdown(f"**`{t['name']}`**")
        st.caption(t["description"])

    st.markdown("### 📊 知识库统计")
    st.metric("概念", len(engine.concepts))
    st.metric("实体", len(engine.entities))
    st.metric("原始资料", len(engine.sources))
    st.metric("别名映射", len(engine.aliases))
    st.metric("历史记录", len(st.session_state.history_records))

    st.markdown("### 🔧 状态")
    if st.session_state.is_searching:
        st.markdown('<p class="status-running">🔄 搜索中...</p>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-complete">✅ 就绪</p>',
                    unsafe_allow_html=True)