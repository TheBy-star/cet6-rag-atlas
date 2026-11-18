import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from step1_chunk import clean_text, extract_section


class ChunkingTest(unittest.TestCase):
    def test_clean_text_normalizes_newlines(self) -> None:
        raw = "第一行\r\n\r\n\r\n第二行   \r\n"
        self.assertEqual(clean_text(raw), "第一行\n\n第二行")

    def test_extract_section_skips_file_title(self) -> None:
        entry = "# 翻译错题本\n## 促进经济\n【原文】..."
        self.assertEqual(extract_section(entry, "翻译错题本"), "促进经济")

    def test_extract_section_falls_back_to_first_line(self) -> None:
        entry = "碳中和 carbon neutrality\n【类别】翻译短语"
        self.assertEqual(
            extract_section(entry, "翻译高频短语"),
            "碳中和 carbon neutrality",
        )


if __name__ == "__main__":
    unittest.main()
