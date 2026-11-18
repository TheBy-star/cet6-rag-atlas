"""Optional cross-encoder reranker for the local hybrid retriever."""

from __future__ import annotations

from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_NAME = "BAAI/bge-reranker-v2-m3-int8"

_model: TextCrossEncoder | None = None


def _get_model() -> TextCrossEncoder:
    global _model
    if _model is None:
        _model = TextCrossEncoder(
            model_name=MODEL_NAME,
            cache_dir=str(MODEL_DIR),
        )
    return _model


def rerank_hits(question: str, hits: list[dict], top_k: int = 3) -> list[dict]:
    """Rerank candidate hits and return the best top_k."""
    if not hits:
        return []

    model = _get_model()
    documents = [hit["document"] for hit in hits]
    scores = list(model.rerank(question, documents, batch_size=8))

    ranked = sorted(
        zip(hits, scores),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )

    results = []
    for hit, score in ranked[:top_k]:
        enriched = dict(hit)
        enriched["rerank_score"] = float(score)
        results.append(enriched)

    return results
