"""
第一步：读取本地 txt，切分成 chunk，输出 data/chunks/chunks.jsonl
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "chunks"
OUT_FILE = OUT_DIR / "chunks.jsonl"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

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

ENTRY_SEP_RE = re.compile(r"(?m)^[ \t]*---+[ \t]*$")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_txt(path: Path) -> str:
    errors = []
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = clean_text(path.read_text(encoding=encoding))
            if not text:
                raise ValueError(f"文件为空：{path}")
            return text
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError(
        f"无法解码：{path}\n请另存为 UTF-8 后重试。\n" + "\n".join(errors)
    )


def detect_doc_type(path: Path) -> str:
    rel = path.relative_to(RAW_DIR)
    for part in rel.parts[:-1]:
        if part.lower() in FOLDER_ALIASES:
            return FOLDER_ALIASES[part.lower()]

    name = path.name.lower()
    if any(k in name for k in ("翻译", "translation", "错题")):
        return "translation"
    if any(k in name for k in ("长难句", "sentence", "阅读")):
        return "long_sentences"
    if any(k in name for k in ("词汇", "单词", "vocab", "word")):
        return "vocabulary"
    return "other"


def extract_section(entry: str, title: str) -> str:
    headings = []
    for line in entry.splitlines():
        line = line.strip()
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if heading:
                headings.append(heading)

    for heading in headings:
        if heading != title:
            return heading[:60]
    if headings:
        return headings[0][:60]

    first_line = entry.splitlines()[0].strip()
    return first_line[:60] if first_line else "未命名条目"


def make_chunk_id(rel_path: str, entry_index: int, start: int, text: str) -> str:
    key = f"{rel_path}|{entry_index}|{start}|{text}"
    return "chk_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def build_chunks() -> list[Document]:
    if not RAW_DIR.exists():
        raise SystemExit(f"找不到目录：{RAW_DIR}")

    files = sorted(
        p for p in RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() == ".txt"
    )
    if not files:
        raise SystemExit(f"在 {RAW_DIR} 下没有找到 .txt 文件。")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        keep_separator=True,
        add_start_index=True,
        separators=[
            "\n\n", "\n",
            "。", "！", "？", "；", "……",
            ". ", "! ", "? ", "; ",
            "，", ", ",
            " ", "",
        ],
    )

    chunks: list[Document] = []
    print(f"发现 {len(files)} 个 txt 文件\n")

    for path in files:
        rel_path = path.relative_to(RAW_DIR).as_posix()
        text = load_txt(path)
        doc_type = detect_doc_type(path)
        label = TYPE_LABEL[doc_type]
        title = path.stem

        entries = [
            part.strip()
            for part in ENTRY_SEP_RE.split(text)
            if part.strip()
        ]

        if len(entries) == 1 and len(text) > CHUNK_SIZE * 2:
            print("  ⚠️ 没检测到 --- 分隔符，建议每条之间加一行 ---")

        print(f"[读取] {rel_path} -> {len(entries)} 条 ({label})")

        for entry_index, entry in enumerate(entries):
            section = extract_section(entry, title)
            header = (
                f"【{label}｜{section}】"
                if section and section != title
                else f"【{label}】"
            )

            sub_docs = splitter.create_documents(
                texts=[entry],
                metadatas=[{
                    "source": rel_path,
                    "doc_type": doc_type,
                    "doc_type_label": label,
                    "title": title,
                    "section": section,
                    "entry_index": entry_index,
                }],
            )

            for chunk_index, sub_doc in enumerate(sub_docs):
                content = f"{header}\n{sub_doc.page_content.strip()}"
                metadata = dict(sub_doc.metadata)
                metadata.update({
                    "chunk_id": make_chunk_id(
                        rel_path,
                        entry_index,
                        int(metadata.get("start_index") or 0),
                        content,
                    ),
                    "chunk_index": chunk_index,
                    "char_count": len(content),
                })
                chunks.append(Document(page_content=content, metadata=metadata))

    return chunks


def save_chunks(chunks: list[Document]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8", newline="\n") as file:
        for chunk in chunks:
            record = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    print("=" * 60)
    print("Step 1：读取 txt -> 切分 chunk")
    print("=" * 60)

    chunks = build_chunks()
    if not chunks:
        raise SystemExit("没有生成任何 chunk。")

    save_chunks(chunks)

    counts = Counter(c.metadata["doc_type"] for c in chunks)
    lengths = [c.metadata["char_count"] for c in chunks]

    print("\n" + "=" * 60)
    print(f"完成：共 {len(chunks)} 个 chunk")
    for doc_type, count in counts.most_common():
        print(f"  - {TYPE_LABEL[doc_type]}：{count}")
    print(
        f"长度：min={min(lengths)}，max={max(lengths)}，"
        f"avg={sum(lengths) // len(lengths)} 字符"
    )
    print(f"输出文件：{OUT_FILE}")
    print("=" * 60)

    print("\n预览前 2 个 chunk：")
    for chunk in chunks[:2]:
        print("\n" + "-" * 60)
        print(f"id: {chunk.metadata['chunk_id']}")
        print(f"metadata: {json.dumps(chunk.metadata, ensure_ascii=False)}")
        print(f"content:\n{chunk.page_content[:240]}...")

    print("\n第一步完成。下一步再做 BGE 向量化和 Chroma 入库。")


if __name__ == "__main__":
    main()
