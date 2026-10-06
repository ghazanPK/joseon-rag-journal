import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from joseon_rag.core import Article, build_index
from joseon_rag.server import create_handler


class BrowserAPITest(unittest.TestCase):
    def test_question_exposes_rewrite_date_citation_and_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index = root / "index.json"
            index.write_text(json.dumps(build_index([
                Article("a", "Scholars discussed education in the hall.", "Hall", 1420, 5, 12, "https://example.org/a"),
                Article("b", "Officials discussed ships.", "Ships", 1440, 1, 2, "https://example.org/b"),
            ])), encoding="utf-8")
            events = root / "events.json"
            events.write_text(json.dumps({"hall": {"year": 1420, "aliases": ["scholars"]}}), encoding="utf-8")
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(index, events))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                body = json.dumps({"query": "Tell me about the hall", "token_budget": 100}).encode()
                req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/ask", body, {"Content-Type": "application/json"})
                with urllib.request.urlopen(req) as response: result = json.load(response)
                self.assertEqual(result["trace"]["date_filter"]["year"], 1420)
                self.assertEqual(result["citations"][0]["url"], "https://example.org/a")
                self.assertEqual(result["trace"]["evidence"][0]["article"]["id"], "a")
            finally:
                server.shutdown(); server.server_close(); thread.join()


class SuggestedQuestionTest(unittest.TestCase):
    def test_corpus_reports_the_suggested_question(self):
        with tempfile.TemporaryDirectory() as folder:
            index = Path(folder) / "index.json"
            index.write_text(json.dumps(build_index([Article("a", "Text.", "Title", 1420, 5, 6, "https://example.org/a")])), encoding="utf-8")
            for suggested in ("", "세종 2년 5월 살곶이 다리 공사를 감독한 사람은 누구인가?"):
                server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(index, suggested_query=suggested))
                thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/corpus") as response:
                        self.assertEqual(json.load(response)["suggested_query"], suggested)
                finally:
                    server.shutdown(); server.server_close(); thread.join()
