import unittest
from joseon_rag.core import Article, RuleRegenerator, answer, build_index, load_articles, query_date, retrieve

class RagTest(unittest.TestCase):
    def test_filter_budget_and_citation(self):
        idx = build_index([Article("a", "Scholars discussed education and royal study.", "Policy", 1420, 5, 12, "https://example/a"), Article("b", "A new script was published for the people.", "Script", 1446, 9, 29, "https://example/b")])
        q = RuleRegenerator({"hall": {"year": 1420, "aliases": ["scholars"]}}).rewrite("Tell me about the hall")
        result = answer(retrieve(idx, "Tell me about the hall", q, 200))
        self.assertIn("a", result["answer"]); self.assertEqual(result["trace"]["date_filter"]["year"], 1420)

    def test_full_iso_date(self):
        import json, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"; p.write_text(json.dumps({"id":"x","date":"1420-05-12","text":"record"}))
            a = load_articles(p)[0]; self.assertEqual((a.year, a.month, a.day), (1420, 5, 12))

    def test_later_joseon_years_and_explicit_date_precedence(self):
        self.assertEqual(query_date("1652년 기록"), (1652, None, None))
        self.assertEqual(query_date("1799-12-30 record"), (1799, 12, 30))
        idx = build_index([Article("old", "event", year=1652), Article("new", "event", year=1799)])
        bundle = retrieve(idx, "What happened in 1652?", "What happened in 1799?", 100)
        self.assertEqual(bundle["date_filter"]["year"], 1652)

    def test_strict_budget_skips_indivisible_article(self):
        idx = build_index([Article("large", "word " * 100, year=1652)])
        self.assertEqual(retrieve(idx, "1652 word", "1652 word", 5)["evidence"], [])

if __name__ == "__main__": unittest.main()
