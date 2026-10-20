"""
读取指定目录下所有 txt，按 300 字切分，用 BGE 生成向量，直接写入 ChromaDB。

默认：
    数据目录：data/raw
    向量库：  data/chroma_db
    collection：cet6_notes
    切分：    300 字，重叠 50 字

示例：
    python ingest_all_txt.py
    python ingest_all_txt.py --data-dir "D:\我的六级笔记" --reset
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path

import chromadb
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "raw"
DEFAULT_DB_DIR = BASE_DIR / "data" / "chroma_db"
MODEL_DIR = BASE_DIR / "models"

DEFAULT_COLLECTION = "cet6_notes"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_CHUNK_SIZE = 300
DEFAULT_OVERLAP = 50
BATCH_SIZE = 8

TYPE_LABEL = {
    "vocabulary": "六级高频词汇",
    "long_sentences": "阅读长难句",
    "translation": "翻译错题本",
    "other": "其他资料",
}

FOLDER_ALIASES = {
    "vocabulary": "vocabulary",
    "词汇": "vocabulary",
    "高频词汇": "vocabulary",
    "long_sentences": "long_sentences",
    "长难句": "long_sentences",
    "阅读长难句": "long_sentences",
    "translation": "translation",
    "翻译": "translation",
    "翻译错题本": "translation",
}


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_text(path: Path) -> str:
    errors = []
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = clean_text(path.read_text(encoding=encoding))
            if text:
                return text
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")

    raise RuntimeError(
        f"无法读取 {path}，请另存为 UTF-8。\n" + "\n".join(errors)
    )


def detect_doc_type(path: Path, data_dir: Path) -> str:
    relative_path = path.relative_to(data_dir)

    for part in relative_path.parts[:-1]:
        key = part.lower()
        if key in FOLDER_ALIASES:
            return FOLDER_ALIASES[key]

    name = path.name.lower()
    if any(keyword in name for keyword in ("翻译", "translation", "错题")):
        return "translation"
    if any(keyword in name for keyword in ("长难句", "sentence", "阅读")):
        return "long_sentences"
    if any(keyword in name for keyword in ("词汇", "单词", "vocab", "word")):
        return "vocabulary"
    return "other"


def extract_section(text: str) -> str:
    for line in text.splitlines()[:30]:
        line = line.strip()
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if heading:
                return heading[:60]

    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    return first_line[:60] if first_line else "未命名资料"


def make_chunk_id(source: str, index: int, text: str) -> str:
    key = f"{source}|{index}|{text}"
    return "chk_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def build_splitter(chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0，且小于 chunk_size")

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        keep_separator=True,
        separators=[
            "\n\n", "\n",
            "。", "！", "？", "；", "……",
            ". ", "! ", "? ", "; ",
            "，", ", ",
            " ", "",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="读取 txt 并写入 ChromaDB")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="txt 所在目录")
    parser.add_argument("--db-dir", default=str(DEFAULT_DB_DIR), help="ChromaDB 持久化目录")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="collection 名称")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="先删除同名 collection 再重建；不传则增量 upsert",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    db_dir = Path(args.db_dir).expanduser().resolve()

    if not data_dir.exists():
        raise SystemExit(f"数据目录不存在：{data_dir}")

    files = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".txt"
    )
    if not files:
        raise SystemExit(f"在 {data_dir} 下没有找到 .txt 文件")

    splitter = build_splitter(args.chunk_size, args.overlap)

    print(f"数据目录：{data_dir}")
    print(f"发现 {len(files)} 个 txt 文件")
    print(f"切分参数：{args.chunk_size} 字 / overlap {args.overlap} 字")

    print("\n加载 BGE 模型……")
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(MODEL_DIR))

    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir) + os.sep)

    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"已删除旧 collection：{args.collection}")
        except Exception:
            print(f"collection 不存在，将新建：{args.collection}")

    collection = client.get_or_create_collection(name=args.collection)
    before_count = collection.count()
    print(f"写入前 collection 数量：{before_count}\n")

    total = 0

    for path in files:
        source = path.relative_to(data_dir).as_posix()
        text = load_text(path)
        doc_type = detect_doc_type(path, data_dir)
        section = extract_section(text)
        chunks = [
            chunk.strip()
            for chunk in splitter.split_text(text)
            if chunk.strip()
        ]

        if not chunks:
            print(f"[跳过] {source}：没有可用文本")
            continue

        ids = []
        documents = []
        metadatas = []

        for index, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(source, index, chunk)
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "source": source,
                    "doc_type": doc_type,
                    "doc_type_label": TYPE_LABEL[doc_type],
                    "section": section,
                    "chunk_index": index,
                    "chunk_id": chunk_id,
                    "char_count": len(chunk),
                }
            )

        for start in range(0, len(documents), BATCH_SIZE):
            batch_docs = documents[start:start + BATCH_SIZE]
            batch_vectors = [
                vector.tolist()
                for vector in model.passage_embed(batch_docs)
            ]
            collection.upsert(
                ids=ids[start:start + BATCH_SIZE],
                embeddings=batch_vectors,
                documents=batch_docs,
                metadatas=metadatas[start:start + BATCH_SIZE],
            )

        total += len(chunks)
        print(f"[写入] {source} -> {len(chunks)} 个 chunk")

    after_count = collection.count()
    print("\n" + "=" * 60)
    print(f"本次写入：{total} 个 chunk")
    print(f"collection：{args.collection}")
    print(f"写入后数量：{after_count}")
    print(f"数据库目录：{db_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
