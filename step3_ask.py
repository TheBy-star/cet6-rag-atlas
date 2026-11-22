"""
第三步：检索 ChromaDB，把检索结果交给 DeepSeek，生成带出处的回答。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import chromadb
import requests
from fastembed import TextEmbedding

from local_index import LocalIndex
from retrieval_utils import validate_citations

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
CHUNKS_FILE = BASE_DIR / "data" / "chunks" / "chunks.jsonl"
MODEL_DIR = BASE_DIR / "models"

COLLECTION_NAME = "cet6_notes"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEEPSEEK_MODEL = "deepseek-chat"
API_URL = "https://api.deepseek.com/chat/completions"
TOP_K = 3
RERANK_CANDIDATES = 10

SYSTEM_PROMPT = """你是专攻大学英语六级备考的助教。
你只能依据【检索资料】回答，不要使用资料之外的知识编造答案。
如果资料不足以回答，就直接说“你的资料库里没有足够依据”，并建议用户补充哪类笔记。
回答要面向六级考试，尽量给出：
1. 直接结论；
2. 地道表达或语法分析；
3. 易错点；
4. 引用资料编号，例如 [1]。
如果资料之间冲突，指出冲突，不要强行统一。"""


def load_env() -> None:
    """读取项目根目录的 .env，避免额外安装 python-dotenv。"""
    if not ENV_FILE.exists():
        return

    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_api_key() -> str:
    load_env()
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key or "粘贴" in key or key.startswith("sk-在此"):
        raise SystemExit(
            "还没有配置 DeepSeek API Key。\n"
            f"请用记事本打开：{ENV_FILE}\n"
            "把 DEEPSEEK_API_KEY= 后面替换成你自己的 Key，保存后重试。"
        )
    return key


def load_resources():
    print("加载 BGE 模型……")
    model = TextEmbedding(
        model_name=EMBEDDING_MODEL,
        cache_dir=str(MODEL_DIR),
    )

    try:
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR) + os.sep,
        )
        collection = client.get_collection(COLLECTION_NAME)

        expected_count = 0
        if CHUNKS_FILE.exists():
            expected_count = sum(
                1
                for line in CHUNKS_FILE.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            )

        if collection.count() != expected_count:
            raise RuntimeError(
                "ChromaDB 数量与 chunks.jsonl 不一致："
                f"{collection.count()} vs {expected_count}"
            )
        if collection.count() == 0:
            raise RuntimeError("ChromaDB collection 为空")

        print(f"ChromaDB 已加载：{collection.count()} 条\n")
        return model, collection
    except Exception as exc:
        print(f"ChromaDB 不可用：{exc}")
        print("切换到本地 NumPy 索引……")

        from local_index import LocalIndex

        local_index = LocalIndex(model)
        print(f"本地索引已加载：{local_index.count()} 条\n")
        return model, local_index


def retrieve(model, collection, question: str):
    query_vector = list(model.query_embed([question]))[0].tolist()
    enable_reranker = os.getenv("ENABLE_RERANKER", "0") == "1"
    candidate_k = (
        RERANK_CANDIDATES if enable_reranker else TOP_K
    )
    candidate_k = min(candidate_k, collection.count())

    if isinstance(collection, LocalIndex):
        results = collection.search(
            question,
            [query_vector],
            n_results=candidate_k,
        )
    else:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"],
        )

    hits = []
    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "document": document,
                "metadata": metadata,
                "distance": float(distance),
            }
        )

    if enable_reranker:
        try:
            from reranker import rerank_hits

            hits = rerank_hits(question, hits, top_k=TOP_K)
        except Exception as exc:
            print(f"Reranker 不可用，回退到混合检索：{exc}")
            hits = hits[:TOP_K]

    return hits[:TOP_K]


def build_context(hits) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        blocks.append(
            f"[{index}] 来源：{metadata.get('source')}｜"
            f"章节：{metadata.get('section')}\n"
            f"{hit['document']}"
        )
    return "\n\n".join(blocks)


def ask_deepseek(api_key: str, question: str, hits) -> str:
    context = build_context(hits)
    user_prompt = f"""下面是从我的六级资料库检索到的内容：

{context}

我的问题：{question}

请只依据上面的资料回答，并在句末用 [1]、[2] 这样的编号标注出处。
如果资料没有覆盖这个问题，请明确说资料不足，不要编造。"""

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1000,
            "stream": False,
        },
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"DeepSeek API 返回 {response.status_code}：\n"
            f"{response.text[:800]}"
        )

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def print_sources(hits) -> None:
    print("\n检索到的资料：")
    for index, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        print(
            f"  [{index}] {metadata.get('source')} / "
            f"{metadata.get('section')} "
            f"(distance={hit['distance']:.4f})"
        )


def ask_once(api_key: str, model, collection, question: str) -> None:
    hits = retrieve(model, collection, question)
    print_sources(hits)

    print("\nDeepSeek 回答：\n")
    answer = ask_deepseek(api_key, question, hits)
    print(answer)

    citation = validate_citations(answer, len(hits))
    if citation["invalid"]:
        print(
            "\n[引用校验] 发现无效引用："
            + ", ".join(f"[{n}]" for n in citation["invalid"])
        )
    elif citation["has_citation"]:
        print(
            "\n[引用校验] 通过："
            + ", ".join(f"[{n}]" for n in citation["cited"])
        )
    else:
        print("\n[引用校验] 回答未标注来源")

    print("\n" + "=" * 60)


def main() -> None:
    api_key = get_api_key()
    model, collection = load_resources()

    # 支持命令行一次性提问：
    # python step3_ask.py "六级翻译中促进经济怎么翻最地道？"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
        ask_once(api_key, model, collection, question)
        return

    print("输入问题开始提问；输入 exit 退出。")
    while True:
        question = input("\n你问：").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break
        if not question:
            continue
        try:
            ask_once(api_key, model, collection, question)
        except Exception as exc:
            print(f"\n[出错] {exc}")


if __name__ == "__main__":
    main()
