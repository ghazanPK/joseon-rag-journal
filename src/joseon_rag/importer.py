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
    """Readable text of a page, scoped to <article>/<main> when the page has them.

    The document <title> is captured separately and never enters the body text;
    head, navigation, header/footer, forms and scripts are skipped.
    """
    SKIP = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript", "template", "svg", "button", "select", "head"}
    BLOCK = {"p", "h1", "h2", "h3", "h4", "li", "br", "div", "section", "tr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.scope = 0
        self.in_title = False
        self.title = ""
        self.page: list[str] = []
        self.scoped: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.in_title = True
        elif tag in self.SKIP:
            self.skip += 1
        if tag in {"article", "main"}:
            self.scope += 1
        if tag in self.BLOCK:
            self.page.append("\n")
            if self.scope:
                self.scoped.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        if tag in {"article", "main"}:
            self.scope = max(0, self.scope - 1)

    def handle_data(self, data):
        if self.in_title:
            self.title += data
            return
        if self.skip:
            return
        self.page.append(data)
        if self.scope:
            self.scoped.append(data)

    def text(self) -> str:
        def clean(parts: list[str]) -> str:
            text = re.sub(r"[ \t\r\f\v]+", " ", "".join(parts))
            return re.sub(r"\n\s*\n*", "\n", text).strip()
        scoped = clean(self.scoped)
        return scoped if len(scoped) >= 50 else clean(self.page)


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
    html = raw.decode("utf-8-sig", errors="replace")
    extra: dict = {}
    if urlparse(url).hostname == "sillok.history.go.kr" and 'class="detail-view"' in html:
        # Annals article page: use the article-block parser instead of whole-page text.
        from .crawler import ARTICLE_ID, parse_article
        aid = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
        if ARTICLE_ID.match(aid):
            rec = parse_article(html, aid, url=url)
            text, page_title = rec["text"], rec["title"]
            extra = {k: rec[k] for k in ("calendar", "leap_month", "date_original", "notes", "categories")}
            volume = volume or rec["volume"]
        else:
            parser = _ArticleText(); parser.feed(html); text, page_title = parser.text(), parser.title.strip()
    else:
        parser = _ArticleText(); parser.feed(html); text, page_title = parser.text(), parser.title.strip()
    if len(text) < 50 and not extra:
        raise ValueError("too little readable text; inspect page or provide JSONL manually")
    row = {"id": article_id, "date": date, "title": title or page_title,
           "text": text, "source_url": url, "volume": volume, **extra,
           "import_method": "user-selected-html", "source_file": str(source_file) if source_file else "fetched-single-page"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row
