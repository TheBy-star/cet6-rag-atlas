"""
本地 NumPy 向量索引：当 ChromaDB 数据库不可用时作为兜底。
接口与 Chroma collection 的 count() / query() 保持兼容。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from retrieval_utils import BM25, reciprocal_rank_fusion

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "data" / "chunks" / "chunks.jsonl"
INDEX_DIR = BASE_DIR / "data" / "local_index"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
ITEMS_FILE = INDEX_DIR / "items.json"
META_FILE = INDEX_DIR / "meta.json"

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MODEL_DIR = BASE_DIR / "models"


def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        raise RuntimeError(
            f"找不到 {CHUNKS_FILE}，请先运行 step1_chunk.py"
        )

    records = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise RuntimeError("chunks.jsonl 是空的")
    return records


def build_index(model: TextEmbedding | None = None) -> None:
    records = load_chunks()
    model = model or TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=str(MODEL_DIR),
    )

    texts = [record["page_content"] for record in records]
    print(f"构建本地索引：{len(texts)} 个 chunk……")

    vectors = [
        vector.tolist()
        for vector in model.passage_embed(texts, batch_size=8)
    ]
    matrix = np.asarray(vectors, dtype=np.float32)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, matrix)

    items = [
        {
            "document": record["page_content"],
            "metadata": record["metadata"],
        }
        for record in records
    ]
    ITEMS_FILE.write_text(
        json.dumps(items, ensure_ascii=False),
        encoding="utf-8",
    )
    META_FILE.write_text(
        json.dumps(
            {"chunks_mtime_ns": CHUNKS_FILE.stat().st_mtime_ns},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"本地索引已保存：{INDEX_DIR}")


class LocalIndex:
    def __init__(self, model: TextEmbedding | None = None) -> None:
        self.model = model or TextEmbedding(
            model_name=MODEL_NAME,
            cache_dir=str(MODEL_DIR),
        )

        need_build = not (
            EMBEDDINGS_FILE.exists()
            and ITEMS_FILE.exists()
            and META_FILE.exists()
        )

        if not need_build:
            meta = json.loads(META_FILE.read_text(encoding="utf-8"))
            need_build = (
                meta.get("chunks_mtime_ns")
                != CHUNKS_FILE.stat().st_mtime_ns
            )

        if need_build:
            build_index(self.model)

        self.embeddings = np.load(EMBEDDINGS_FILE)
        self.items = json.loads(
            ITEMS_FILE.read_text(encoding="utf-8")
        )
        self.norms = np.linalg.norm(self.embeddings, axis=1)
        self.bm25 = BM25([item["document"] for item in self.items])

    def count(self) -> int:
        return len(self.items)

    def query(
        self,
        query_embeddings,
        n_results: int = 3,
        include=None,
    ) -> dict:
        query = np.asarray(query_embeddings[0], dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            raise ValueError("query embedding 不能是零向量")

        similarities = (
            self.embeddings @ query
            / (self.norms * query_norm + 1e-12)
        )
        top_indices = np.argsort(-similarities)[:n_results]

        documents = []
        metadatas = []
        distances = []

        for index in top_indices:
            item = self.items[int(index)]
            documents.append(item["document"])
            metadatas.append(item["metadata"])
            distances.append(float(1.0 - similarities[int(index)]))

        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def search(
        self,
        question: str,
        query_embeddings,
        n_results: int = 3,
    ) -> dict:
        """Hybrid retrieval: dense cosine similarity + BM25 via RRF."""
        query = np.asarray(query_embeddings[0], dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            raise ValueError("query embedding 不能是零向量")

        dense_scores = (
            self.embeddings @ query
            / (self.norms * query_norm + 1e-12)
        )
        dense_ranking = list(
            np.argsort(-dense_scores)[: min(50, len(self.items))]
        )

        bm25_scores = np.asarray(self.bm25.scores(question))
        bm25_ranking = list(
            np.argsort(-bm25_scores)[: min(50, len(self.items))]
        )

        fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking])
        top_indices = fused[:n_results]

        documents = []
        metadatas = []
        distances = []

        for index in top_indices:
            item = self.items[int(index)]
            documents.append(item["document"])
            metadatas.append(item["metadata"])
            distances.append(float(1.0 - dense_scores[int(index)]))

        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }


if __name__ == "__main__":
    build_index()
