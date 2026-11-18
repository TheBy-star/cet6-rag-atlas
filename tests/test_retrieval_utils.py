import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_utils import (
    BM25,
    reciprocal_rank_fusion,
    tokenize,
    validate_citations,
)


class RetrievalUtilsTest(unittest.TestCase):
    def test_tokenize_mixed_text(self) -> None:
        tokens = tokenize("碳中和 carbon neutrality")
        self.assertIn("carbon", tokens)
        self.assertIn("碳", tokens)
        self.assertIn("碳中", tokens)

    def test_bm25_prefers_exact_topic(self) -> None:
        bm25 = BM25(
            [
                "碳中和 carbon neutrality",
                "文化传承 cultural inheritance",
            ]
        )
        scores = bm25.scores("碳中和")
        self.assertGreater(scores[0], scores[1])

    def test_rrf_fuses_two_rankings(self) -> None:
        fused = reciprocal_rank_fusion([[1, 2], [0, 1]])
        self.assertEqual(fused[0], 1)

    def test_citation_validation(self) -> None:
        valid = validate_citations("答案是 boost economic growth [1][2]", 3)
        self.assertTrue(valid["valid"])
        self.assertEqual(valid["cited"], [1, 2])

        invalid = validate_citations("答案见 [3]", 2)
        self.assertFalse(invalid["valid"])
        self.assertEqual(invalid["invalid"], [3])

        missing = validate_citations("资料不足", 3)
        self.assertFalse(missing["has_citation"])
        self.assertTrue(missing["valid"])


if __name__ == "__main__":
    unittest.main()
