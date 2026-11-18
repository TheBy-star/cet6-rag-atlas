"""Offline retrieval evaluation: Recall@K and MRR on a small gold set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastembed import TextEmbedding

from local_index import LocalIndex, MODEL_DIR, MODEL_NAME

BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = BASE_DIR / "eval" / "questions.jsonl"
RESULTS_DIR = BASE_DIR / "eval" / "results"


def load_questions() -> list[dict]:
    if not QUESTIONS_FILE.exists():
        raise SystemExit(f"找不到评测集：{QUESTIONS_FILE}")

    questions = []
    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if not questions:
        raise SystemExit("评测集为空")

    return questions


def find_rank(hits: list[dict], expected_source: str, expected_section: str) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        if (
            metadata.get("source") == expected_source
            and metadata.get("section") == expected_section
        ):
            return rank
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("dense", "hybrid"),
        default="hybrid",
        help="dense = 纯向量；hybrid = 向量 + BM25",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    questions = load_questions()
    model = TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=str(MODEL_DIR),
    )
    index = LocalIndex(model)

    details = []
    ranks = []

    for item in questions:
        query_vector = list(model.query_embed([item["question"]]))[0].tolist()

        if args.mode == "hybrid":
            results = index.search(
                item["question"],
                [query_vector],
                n_results=args.top_k,
            )
        else:
            results = index.query(
                [query_vector],
                n_results=args.top_k,
            )

        hits = [
            {"document": document, "metadata": metadata, "distance": float(distance)}
            for document, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

        rank = find_rank(
            hits,
            item["expected_source"],
            item["expected_section"],
        )
        ranks.append(rank)
        details.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_source": item["expected_source"],
                "expected_section": item["expected_section"],
                "rank": rank,
                "top_sources": [
                    {
                        "source": hit["metadata"].get("source"),
                        "section": hit["metadata"].get("section"),
                        "distance": hit["distance"],
                    }
                    for hit in hits
                ],
            }
        )

    total = len(ranks)
    recall_at = {
        k: sum(1 for rank in ranks if rank is not None and rank <= k) / total
        for k in (1, 3, 5)
    }
    mrr = sum(
        1.0 / rank for rank in ranks if rank is not None
    ) / total

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_file = RESULTS_DIR / f"{args.mode}.json"
    result_file.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "total": total,
                "recall_at_1": recall_at[1],
                "recall_at_3": recall_at[3],
                "recall_at_5": recall_at[5],
                "mrr": mrr,
                "details": details,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"mode={args.mode} total={total}")
    print(f"Recall@1={recall_at[1]:.2%}")
    print(f"Recall@3={recall_at[3]:.2%}")
    print(f"Recall@5={recall_at[5]:.2%}")
    print(f"MRR={mrr:.3f}")
    print(f"结果：{result_file}")

    misses = [item for item in details if item["rank"] is None]
    if misses:
        print("\n未命中：")
        for item in misses:
            print(f"  {item['id']} {item['question']}")


if __name__ == "__main__":
    main()
