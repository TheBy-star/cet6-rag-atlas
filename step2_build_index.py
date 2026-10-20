"""
第二步：读取 chunks.jsonl，用 BGE 生成向量，写入 ChromaDB，并做一次检索测试。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import chromadb
from fastembed import TextEmbedding

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "data" / "chunks" / "chunks.jsonl"
DB_DIR = BASE_DIR / "data" / "chroma_db"
MODEL_DIR = BASE_DIR / "models"

COLLECTION_NAME = "cet6_notes"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY = "六级翻译中促进经济怎么翻最地道？"


def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        raise SystemExit(
            f"找不到 {CHUNKS_FILE}\n请先运行第一步生成 chunks.jsonl。"
        )

    records = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise SystemExit("chunks.jsonl 是空的，请先运行第一步。")

    return records


def embed_documents(model: TextEmbedding, documents: list[str]) -> list[list[float]]:
    if hasattr(model, "passage_embed"):
        vectors = model.passage_embed(documents, batch_size=8)
    else:
        vectors = model.embed(documents, batch_size=8)
    return [vector.tolist() for vector in vectors]


def embed_query(model: TextEmbedding, query: str) -> list[float]:
    if hasattr(model, "query_embed"):
        vectors = model.query_embed([query])
    else:
        vectors = model.embed([query])
    return list(vectors)[0].tolist()


def main() -> None:
    records = load_chunks()
    print(f"读取 {len(records)} 个 chunk")

    print(f"\n加载模型：{MODEL_NAME}")
    print("第一次运行会自动下载模型，约 100 MB，请耐心等待……")

    model = TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=str(MODEL_DIR),
    )

    documents = [record["page_content"] for record in records]
    metadatas = [record["metadata"] for record in records]
    ids = [record["metadata"]["chunk_id"] for record in records]

    print("正在生成文档向量……")
    document_vectors = embed_documents(model, documents)
    print(f"向量维度：{len(document_vectors[0])}")

    # 不删除整个数据库目录，避免 Windows 上文件句柄未释放导致
    # “unable to open database file”。每次只重建 collection。
    DB_DIR.mkdir(parents=True, exist_ok=True)

    # 旧版本曾在 data 根目录错误生成过 sqlite 文件，先清掉它。
    stale_db = DB_DIR.parent / "chroma.sqlite3"
    if stale_db.exists():
        try:
            stale_db.unlink()
        except OSError:
            pass

    client = chromadb.PersistentClient(path=str(DB_DIR) + os.sep)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    collection.add(
        ids=ids,
        embeddings=document_vectors,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"已写入 ChromaDB：{collection.count()} 条")
    print(f"数据库目录：{DB_DIR}")

    print("\n" + "=" * 60)
    print(f"检索测试：{QUERY}")
    print("=" * 60)

    query_vector = embed_query(model, QUERY)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(3, len(records)),
        include=["documents", "metadatas", "distances"],
    )

    for index, (document, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        print(f"\nTop {index} | distance={float(distance):.4f}")
        print(f"来源：{metadata.get('source')}")
        print(f"章节：{metadata.get('section')}")
        print("-" * 60)
        print(document[:400])
        print("-" * 60)

    print("\n第二步完成：向量库已建好，并且检索测试已跑通。")
    print("下一步可以接 DeepSeek，把检索结果交给大模型生成回答。")


if __name__ == "__main__":
    main()
