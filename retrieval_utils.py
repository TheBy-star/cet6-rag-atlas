"""Pure-Python retrieval utilities: tokenization, BM25 and citation checks."""

from __future__ import annotations

import math
import re
from collections import Counter

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[a-z0-9]+")
CITATION_RE = re.compile(r"\[(\d+)\]")


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text.

    English/number tokens are kept as words. Chinese text is represented by
    unigrams and bigrams, which works reasonably without a word segmenter.
    """
    lowered = text.lower()
    tokens = LATIN_RE.findall(lowered)
    chinese = CJK_RE.findall(lowered)
    tokens.extend(chinese)
    tokens.extend(a + b for a, b in zip(chinese, chinese[1:]))
    return tokens


class BM25:
    """Small, dependency-free BM25 implementation for local reranking."""

    def __init__(
        self,
        documents: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = [tokenize(document) for document in documents]
        self.k1 = k1
        self.b = b
        self.doc_count = len(self.documents)
        self.doc_lengths = [len(document) for document in self.documents]
        self.avg_length = (
            sum(self.doc_lengths) / self.doc_count
            if self.doc_count
            else 0.0
        )
        self.term_frequencies = [
            Counter(document) for document in self.documents
        ]
        self.document_frequencies: Counter[str] = Counter()
        for document in self.documents:
            self.document_frequencies.update(set(document))

        self.idf = {
            term: math.log(
                1 + (self.doc_count - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in self.document_frequencies.items()
        }

    def scores(self, query: str) -> list[float]:
        query_terms = tokenize(query)
        scores = [0.0] * self.doc_count

        for term in query_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, frequencies in enumerate(self.term_frequencies):
                tf = frequencies.get(term, 0)
                if tf == 0:
                    continue
                length = self.doc_lengths[index]
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * length / (self.avg_length or 1)
                )
                scores[index] += idf * tf * (self.k1 + 1) / denominator

        return scores


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 60,
) -> list[int]:
    """Fuse ranked index lists using Reciprocal Rank Fusion (RRF)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking):
            scores[index] = scores.get(index, 0.0) + 1.0 / (k + rank + 1)

    return [
        index
        for index, _ in sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def validate_citations(answer: str, source_count: int) -> dict:
    """Check that every [n] citation points to a retrieved source."""
    cited = sorted({int(number) for number in CITATION_RE.findall(answer)})
    invalid = [
        number
        for number in cited
        if number < 1 or number > source_count
    ]
    return {
        "cited": cited,
        "invalid": invalid,
        "has_citation": bool(cited),
        "valid": not invalid,
    }
