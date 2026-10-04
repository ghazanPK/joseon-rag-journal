"""Import an explicitly selected public article page as one source record."""
from __future__ import annotations

import json
import re
import urllib.request
from datetime import date as calendar_date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


class _ArticleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.skip = 0
        self.parts: list[str] = []
        self.title = ""
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "footer"}:
            self.skip += 1
        if tag == "title":
            self.in_title = True
        if tag in {"article", "main"}:
            self.depth += 1
        if tag in {"p", "h1", "h2", "h3", "li", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "footer"}:
            self.skip = max(0, self.skip - 1)
        if tag == "title":
            self.in_title = False
        if tag in {"article", "main"}:
            self.depth = max(0, self.depth - 1)

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        if not self.skip:
            self.parts.append(data)


def import_page(url: str, out: Path, *, source_file: Path | None = None,
                article_id: str, date: str, title: str = "", volume: str = "") -> dict:
    if urlparse(url).scheme != "https":
        raise ValueError("source URL must use HTTPS")
    calendar_date.fromisoformat(date)
    if not article_id.strip():
        raise ValueError("article ID is required")
    if source_file:
        raw = source_file.read_bytes()
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "PaperReach educational import (single requested page)"})
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("page exceeds 2 MB; save a reduced article HTML file")
    parser = _ArticleText()
    parser.feed(raw.decode("utf-8-sig", errors="replace"))
    text = re.sub(r"[ \t]+", " ", "".join(parser.parts))
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    if len(text) < 50:
        raise ValueError("too little readable text; inspect page or provide JSONL manually")
    row = {"id": article_id, "date": date, "title": title or parser.title.strip(),
           "text": text, "source_url": url, "volume": volume,
           "import_method": "user-selected-html", "source_file": str(source_file) if source_file else "fetched-single-page"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row
