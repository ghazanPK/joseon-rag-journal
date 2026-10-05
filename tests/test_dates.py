"""Date-parser regressions, including every failing input from the 2026-10-06 audit."""
import unittest

from joseon_rag.core import Article, build_index, parse_date, parse_record_date, query_date, retrieve


class DateParserTest(unittest.TestCase):
    def check(self, text, expected, year_end=None):
        spec = parse_date(text)
        self.assertEqual(spec.as_tuple(), expected, text)
        self.assertEqual(spec.year_end, year_end, text)

    def test_audit_english_month_names(self):
        self.check("What was the educational policy on May 12, 1420?", (1420, 5, 12))
        self.check("May 12 1420", (1420, 5, 12))
        self.check("12 May 1420", (1420, 5, 12))
        self.check("the 12th of May, 1420", (1420, 5, 12))
        self.check("Sept. 3rd, 1446", (1446, 9, 3))
        self.check("What was the educational policy in May 1420?", (1420, 5, None))
        self.check("in October of 1446", (1446, 10, None))

    def test_audit_bare_numbers_are_not_dates(self):
        self.check("1420 3 officials", (None, None, None))
        self.check("How did 1500 soldiers defend the northern border?", (None, None, None))
        self.check("An army of 1500 soldiers marched", (None, None, None))
        self.check("Tell me about the 28 letters", (None, None, None))

    def test_year_needs_date_context(self):
        self.check("What educational policies were implemented in 1420?", (1420, None, None))
        self.check("the policies of 1420?", (1420, None, None))
        self.check("approximate date: 1446", (1446, None, None))
        self.check("1652년 기록", (1652, None, None))
        self.check("1420 AD", (1420, None, None))

    def test_numeric_and_korean_dates(self):
        self.check("1799-12-30 record", (1799, 12, 30))
        self.check("1420/5/12", (1420, 5, 12))
        self.check("1420-05", (1420, 5, None))
        self.check("1420년 5월 12일 경연", (1420, 5, 12))
        self.check("1420년 5월", (1420, 5, None))
        self.check("1420년 13월", (1420, None, None))

    def test_reign_year_forms(self):
        self.check("세종 2년 5월 12일에 무슨 일이 있었나?", (1420, 5, 12))
        self.check("세종 28년 9월", (1446, 9, None))
        self.check("세종 즉위년 8월 10일", (1418, 8, 10))
        self.check("세종대왕 10년", (1428, None, None))
        self.check("in the 28th year of King Sejong", (1446, None, None))
        self.check("Sejong year 2", (1420, None, None))
        self.check("Sejong's accession year", (1418, None, None))
        self.check("세종 40년", (None, None, None))
        self.assertTrue(parse_date("세종 2년 윤1월 3일").leap_month)

    def test_ranges(self):
        self.check("between 1420 and 1425", (1420, None, None), 1425)
        self.check("from 1443 to 1446", (1443, None, None), 1446)
        self.check("1420–1425", (1420, None, None), 1425)
        self.check("1420년부터 1425년까지", (1420, None, None), 1425)
        self.check("세종 2년부터 5년까지", (1420, None, None), 1423)
        self.check("May 12-14, 1420", (1420, 5, None))
        self.check("from May to June 1420", (1420, None, None))

    def test_structured_record_dates(self):
        self.assertEqual(parse_record_date("1420").as_tuple(), (1420, None, None))
        self.assertEqual(parse_record_date("1420-05-29").as_tuple(), (1420, 5, 29))
        self.assertEqual(query_date("1799-12-30 record"), (1799, 12, 30))


class DateFilterTest(unittest.TestCase):
    def setUp(self):
        self.index = build_index([
            Article("a1420", "Scholars of the hall discussed royal lectures.", "Hall", 1420, 5, 12),
            Article("b1433", "Soldiers defended the northern border fortresses.", "Border", 1433, 4, 1),
            Article("c1446", "The new script was promulgated for the people.", "Script", 1446, 9, 29),
        ])

    def test_regenerated_date_not_overridden_by_spurious_original_number(self):
        original = "How did 1500 soldiers defend the northern border?"
        bundle = retrieve(self.index, original, original + " Historical context: approximate date: 1433.", 500)
        self.assertEqual((bundle["date_filter"]["year"], bundle["date_filter"]["source"]), (1433, "regenerated"))
        self.assertEqual(bundle["evidence"][0]["article"]["id"], "b1433")

    def test_explicit_original_date_takes_precedence(self):
        bundle = retrieve(self.index, "What happened in 1420?", "What happened in 1446?", 500)
        self.assertEqual((bundle["date_filter"]["year"], bundle["date_filter"]["source"]), (1420, "original"))

    def test_year_range_filter(self):
        bundle = retrieve(self.index, "What did soldiers or scholars do between 1430 and 1450?", "soldiers scholars between 1430 and 1450", 500)
        self.assertEqual({e["article"]["id"] for e in bundle["evidence"]}, {"b1433"})
        self.assertEqual(bundle["date_filter"]["mode"], "year-range")

    def test_english_full_date_filters_exact_day(self):
        bundle = retrieve(self.index, "What happened on May 12, 1420?", "What happened on May 12, 1420?", 500)
        self.assertEqual([e["article"]["id"] for e in bundle["evidence"]], ["a1420"])
        self.assertTrue(bundle["similarity_floor"]["waived"])


if __name__ == "__main__":
    unittest.main()
