"""Crawler/parser tests.

The HTML fixtures in tests/fixtures/sillok reproduce the markup of
sillok.history.go.kr month-index, day-list and article pages as observed on
2026-10-06; their text is authored for the tests and contains no Annals content.
"""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from joseon_rag import cli
from joseon_rag.core import load_articles
from joseon_rag.crawler import (BASE_URL, SAMPLE_CAP, SAMPLE_MONTH, SAMPLE_OUT, SAMPLE_REIGN_YEAR, CrawlStopped, PoliteFetcher,
                                crawl, parse_article, parse_day_list, parse_month_index, sample_plan)
from joseon_rag.importer import import_page

FIX = Path(__file__).parent / "fixtures" / "sillok"


def fixture(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


class FakeOpener:
    """Serves fixture pages and records request times (no network)."""

    def __init__(self, pages, clock):
        self.pages = pages; self.calls = []; self.clock = clock

    def __call__(self, req, timeout):
        self.calls.append((req.full_url, self.clock.now))
        url = req.full_url
        for key, (status, body) in self.pages.items():
            if url.endswith(key):
                return status, "text/html;charset=UTF-8", body.encode("utf-8")
        return 404, "text/html", b""


class FakeClock:
    def __init__(self): self.now = 0.0
    def time(self): return self.now
    def sleep(self, s): self.now += s


class ParserTest(unittest.TestCase):
    def test_month_index_keeps_chronicle_months_and_leap_flag(self):
        months = parse_month_index(fixture("month-index.html"))
        self.assertEqual([m.id for m in months], ["kda_100080", "kda_100090", "kda_102011", "kda_102050"])
        leap = months[2]
        self.assertEqual((leap.reign_year, leap.month, leap.leap, leap.count), (2, 1, True, 27))

    def test_day_list_excludes_neighbouring_months_and_duplicates(self):
        self.assertEqual(parse_day_list(fixture("day-list.html"), "kda_102050"),
                         ["kda_10205029_001", "kda_10205029_002", "kda_10205030_001"])

    def test_article_scoped_to_detail_view_with_lunar_date(self):
        rec = parse_article(fixture("article.html"), "kda_10205029_001", include_hanja=True)
        self.assertEqual((rec["year"], rec["month"], rec["day"], rec["leap_month"], rec["calendar"]), (1420, 5, 29, False, "lunar"))
        self.assertEqual(rec["date"], "1420-05-29")
        self.assertEqual(rec["reign_year"], 2); self.assertEqual(rec["ganji"], "병신")
        self.assertEqual(rec["volume"], "세종실록 8권")
        self.assertIn("세종 2년 5월 29일", rec["date_original"]); self.assertIn("1420년", rec["date_original"])
        self.assertEqual(rec["title"], "시험용 기사 제목: 학사들이 경연을 논의하다")
        self.assertIn("집현전의 학사들이", rec["text"]); self.assertIn("경연(經筵) 의 시간을", rec["text"])
        for leaked in ("조선왕조실록", "HOME", "모바일", "12)", "【분류】", "Copyright", "시험용 저작권"):
            self.assertNotIn(leaked, rec["text"])
        self.assertEqual(rec["notes"], ["경연(經筵) : 시험용 주석 문장."])
        self.assertIn("정치-행정(行政)", rec["categories"])
        self.assertIn("集賢殿", rec["hanja_text"])
        self.assertEqual(rec["source_url"], f"{BASE_URL}/id/kda_10205029_001")
        self.assertNotIn("parse_warnings", rec)

    def test_accession_year_and_leap_month_from_id(self):
        html = fixture("article.html").replace("세종 2년 5월 29일", "세종 즉위년 윤8월 3일")
        rec = parse_article(html, "kda_10008103_001")
        self.assertEqual((rec["year"], rec["month"], rec["day"], rec["leap_month"], rec["reign_year"]), (1418, 8, 3, True, 0))
        self.assertTrue(any("page year" in w for w in rec["parse_warnings"]))  # fixture still prints 1420년

    def test_records_load_into_the_corpus_contract(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "a.jsonl"
            out.write_text(json.dumps(parse_article(fixture("article.html"), "kda_10205029_001"), ensure_ascii=False) + "\n", encoding="utf-8")
            a = load_articles(out)[0]
            self.assertEqual((a.year, a.month, a.day, a.calendar), (1420, 5, 29, "lunar"))
            self.assertIn("(lunar)", a.citation)

    def test_selected_sillok_page_import_uses_article_block(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "p.html"; src.write_text(fixture("article.html"), encoding="utf-8"); out = Path(d) / "r.jsonl"
            row = import_page(f"{BASE_URL}/id/kda_10205029_001", out, source_file=src, article_id="x", date="1420-05-29")
            self.assertEqual(row["title"], "시험용 기사 제목: 학사들이 경연을 논의하다")
            self.assertNotIn("조선왕조실록", row["text"]); self.assertEqual(row["calendar"], "lunar")


class CrawlTest(unittest.TestCase):
    def pages(self):
        return {"/robots.txt": (404, "<html>not found</html>"),
                "inspectionMonthList.do?id=kda": (200, fixture("month-index.html")),
                "inspectionDayList.do?id=kda_102011&level=3": (200, "<html><body>no articles</body></html>"),
                "inspectionDayList.do?id=kda_102050&level=3": (200, fixture("day-list.html")),
                "/id/kda_10205029_001": (200, fixture("article.html")),
                "/id/kda_10205029_002": (200, fixture("article.html").replace("1/2 기사", "2/2 기사")),
                "/id/kda_10205030_001": (200, fixture("article.html").replace("5월 29일", "5월 30일"))}

    def fetcher(self, cache, pages, clock, **kw):
        opener = FakeOpener(pages, clock)
        return PoliteFetcher(cache, delay=1.5, opener=opener, sleep=clock.sleep, clock=clock.time, log=lambda m: None, **kw), opener

    def test_rate_limited_cached_and_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); clock = FakeClock(); out = root / "sejong.jsonl"
            f, opener = self.fetcher(root / "cache", self.pages(), clock)
            s = crawl(out, root / "cache", years="2", max_articles=2, fetcher=f, log=lambda m: None)
            self.assertEqual((s["written"], s["stopped"]), (2, "max-articles"))
            times = [t for _, t in opener.calls]
            self.assertTrue(all(b - a >= 1.5 for a, b in zip(times, times[1:])), times)
            self.assertIn("no robots.txt directives", s["robots"])
            # Resume: cached pages are not refetched and existing IDs are skipped.
            f2, opener2 = self.fetcher(root / "cache", self.pages(), clock)
            s2 = crawl(out, root / "cache", years="2", fetcher=f2, log=lambda m: None)
            self.assertEqual((s2["written"], s2["skipped_existing"]), (1, 2))
            self.assertEqual([u for u, _ in opener2.calls], [f"{BASE_URL}/robots.txt", f"{BASE_URL}/id/kda_10205030_001"])
            ids = [json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(ids, ["kda_10205029_001", "kda_10205029_002", "kda_10205030_001"])
            # Offline mode parses only the cache.
            f3 = PoliteFetcher(root / "cache", offline=True, log=lambda m: None)
            self.assertEqual(crawl(root / "again.jsonl", root / "cache", years="2", fetcher=f3, log=lambda m: None)["written"], 3)

    def test_robots_disallow_and_forbidden_stop_the_crawl(self):
        with tempfile.TemporaryDirectory() as d:
            pages = self.pages(); pages["/robots.txt"] = (200, "User-agent: *\nDisallow: /search/\n")
            f, _ = self.fetcher(Path(d) / "c", pages, FakeClock())
            with self.assertRaises(CrawlStopped):
                crawl(Path(d) / "o.jsonl", Path(d) / "c", fetcher=f, log=lambda m: None)
            pages = self.pages(); pages["inspectionMonthList.do?id=kda"] = (403, "")
            f, _ = self.fetcher(Path(d) / "c2", pages, FakeClock())
            with self.assertRaises(CrawlStopped):
                crawl(Path(d) / "o2.jsonl", Path(d) / "c2", fetcher=f, log=lambda m: None)

    def test_minimum_delay_is_enforced(self):
        with self.assertRaises(ValueError):
            PoliteFetcher(Path("unused"), delay=0.2)


class SampleModeTest(unittest.TestCase):
    """``crawl --sample``: one lunar month, capped, idempotent; fixtures only, no network."""

    pages = CrawlTest.pages
    fetcher = CrawlTest.fetcher

    def test_plan_defaults_and_refusals(self):
        plan = sample_plan()
        self.assertEqual((plan["years"], plan["month"], plan["leap"], plan["max_total"], plan["delay"]),
                         (str(SAMPLE_REIGN_YEAR), SAMPLE_MONTH, False, SAMPLE_CAP, 1.5))
        self.assertEqual((SAMPLE_REIGN_YEAR, SAMPLE_MONTH, SAMPLE_CAP), (2, 5, 100))
        self.assertIsNone(plan["max_articles"])
        self.assertEqual(sample_plan(years="28", month=9, max_articles=30)["max_total"], 30)
        self.assertEqual(sample_plan(years="28", month=9)["years"], "28")
        for bad in (dict(max_articles=SAMPLE_CAP + 1), dict(max_articles=0), dict(delay=1.0), dict(years="1-3"), dict(years="40")):
            with self.assertRaises(ValueError, msg=bad):
                sample_plan(**bad)

    def run_cli(self, *argv):
        captured = {}

        def fake_crawl(out, cache, **kw):
            captured.update(out=out, cache=cache, **kw); return {"written": 0}
        with mock.patch("joseon_rag.crawler.crawl", fake_crawl), mock.patch.object(sys, "argv", ["joseon-rag", "crawl", *argv]), \
                mock.patch("sys.stdout", io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            cli.main()
        return captured

    def test_cli_sample_arguments(self):
        got = self.run_cli("--sample")
        self.assertEqual((got["out"], got["years"], got["month"], got["leap"], got["max_total"], got["max_articles"], got["delay"]),
                         (SAMPLE_OUT, "2", 5, False, 100, None, 1.5))
        got = self.run_cli("--sample", "--max-articles", "12", "--out", "data/x.jsonl")
        self.assertEqual((got["out"], got["max_total"]), (Path("data/x.jsonl"), 12))
        got = self.run_cli("--out", "o.jsonl", "--years", "2", "--month", "5", "--max-articles", "7")
        self.assertEqual((got["month"], got["max_articles"]), (5, 7)); self.assertNotIn("max_total", got)
        for argv in (["--sample", "--max-articles", "500"], ["--sample", "--delay", "1.0"], ["--years", "2"]):
            with self.assertRaises(SystemExit, msg=argv):
                self.run_cli(*argv)

    def test_sample_crawl_is_capped_and_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); out = root / "sample.jsonl"; clock = FakeClock()
            plan = sample_plan(max_articles=2); plan.pop("delay")  # the test fetcher runs at the same 1.5 s
            f, opener = self.fetcher(root / "cache", self.pages(), clock)
            s = crawl(out, root / "cache", fetcher=f, log=lambda m: None, **plan)
            self.assertEqual((s["written"], s["stopped"], s["month_ids"]), (2, "max-total", ["kda_102050"]))
            self.assertNotIn("kda_102011", " ".join(u for u, _ in opener.calls))  # other months are never fetched
            times = [t for _, t in opener.calls]
            self.assertTrue(all(b - a >= 1.5 for a, b in zip(times, times[1:])), times)
            # Rerun with the same cap: everything comes from the cache, no request at all (not even robots.txt).
            f2, opener2 = self.fetcher(root / "cache", self.pages(), clock)
            s2 = crawl(out, root / "cache", fetcher=f2, log=lambda m: None, **plan)
            self.assertEqual((s2["written"], s2["skipped_existing"], s2["network_requests"]), (0, 2, 0))
            self.assertEqual(opener2.calls, []); self.assertIn("not checked", s2["robots"])
            self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 2)
            # A larger cap resumes with only the missing article.
            f3, opener3 = self.fetcher(root / "cache", self.pages(), clock)
            bigger = sample_plan(); bigger.pop("delay")
            s3 = crawl(out, root / "cache", fetcher=f3, log=lambda m: None, **bigger)
            self.assertEqual((s3["written"], s3["stopped"]), (1, "complete"))
            self.assertEqual([u for u, _ in opener3.calls], [f"{BASE_URL}/robots.txt", f"{BASE_URL}/id/kda_10205030_001"])

    def test_month_and_leap_selection(self):
        with tempfile.TemporaryDirectory() as d:
            f, opener = self.fetcher(Path(d) / "c", self.pages(), FakeClock())
            s = crawl(Path(d) / "o.jsonl", Path(d) / "c", years="2", month=1, leap=True, fetcher=f, log=lambda m: None)
            self.assertEqual((s["month_ids"], s["written"]), (["kda_102011"], 0))
            with self.assertRaises(ValueError):  # no such month in the index
                crawl(Path(d) / "o.jsonl", Path(d) / "c", years="2", month=7, fetcher=f, log=lambda m: None)
            with self.assertRaises(ValueError):
                crawl(Path(d) / "o.jsonl", Path(d) / "c", years="2", leap=True, fetcher=f, log=lambda m: None)


if __name__ == "__main__":
    unittest.main()
