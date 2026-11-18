"""
第四步：Streamlit 网页界面。
启动方式：在项目目录运行 streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from step3_ask import ask_deepseek, get_api_key, load_resources, retrieve
from retrieval_utils import validate_citations

st.set_page_config(
    page_title="六级备考 RAG 助手",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_resources():
    """模型和 ChromaDB 只加载一次，避免每次提问都重新初始化。"""
    return load_resources()


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------

try:
    api_key = get_api_key()
    model, collection = get_resources()
except SystemExit as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"初始化失败：{exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📚 我的六级资料库")
    st.metric("已索引 chunk", collection.count())
    st.caption("Embedding：BAAI/bge-small-zh-v1.5")
    st.caption("LLM：deepseek-chat")

    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**可以这样问：**")
    st.markdown(
        "- 六级翻译中促进经济怎么翻最地道？\n"
        "- 总结一下翻译错题里最常见的冠词错误\n"
        "- 帮我分析这段阅读长难句的语法"
    )


# ---------------------------------------------------------------------------
# 页面主体
# ---------------------------------------------------------------------------

st.title("📚 六级备考 RAG 助手")
st.caption(
    "基于你的高频词汇、阅读长难句分析和翻译错题本回答问题。"
    "回答只依据检索到的个人笔记，并标注出处。"
)

if not st.session_state.messages:
    st.info(
        "在下方输入你的问题。例如：**六级翻译中促进经济怎么翻最地道？**"
    )


def render_sources(sources: list[dict]) -> None:
    with st.expander("查看检索来源"):
        for index, source in enumerate(sources, start=1):
            st.markdown(
                f"**[{index}] {source['source']} / {source['section']}**"
            )
            st.caption(f"distance = {source['distance']:.4f}")
            st.text(source["document"][:600])
            if index < len(sources):
                st.divider()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])


question = st.chat_input("输入你的六级问题……")

if question:
    question = question.strip()
    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("正在检索笔记并生成回答……"):
                try:
                    hits = retrieve(model, collection, question)
                    answer = ask_deepseek(api_key, question, hits)

                    sources = [
                        {
                            "source": hit["metadata"].get("source", ""),
                            "section": hit["metadata"].get("section", ""),
                            "distance": hit["distance"],
                            "document": hit["document"],
                        }
                        for hit in hits
                    ]

                    st.markdown(answer)

                    citation = validate_citations(answer, len(hits))
                    if citation["invalid"]:
                        st.warning(
                            "引用校验：回答引用了不存在的资料编号 "
                            + ", ".join(
                                f"[{n}]" for n in citation["invalid"]
                            )
                        )
                    elif citation["has_citation"]:
                        st.caption(
                            "引用校验通过："
                            + ", ".join(
                                f"[{n}]" for n in citation["cited"]
                            )
                        )
                    else:
                        st.info("引用校验：回答没有标注来源，请核对原文。")

                    render_sources(sources)
                    st.caption(
                        "以上回答只依据你的个人笔记；资料不足时请补充对应内容。"
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                except Exception as exc:
                    st.error(f"生成失败：{exc}")
