"""Rate-limited, resumable importer for the Sejong Annals (세종실록).

Source: the National Institute of Korean History's public Annals of the Joseon
Dynasty service, https://sillok.history.go.kr. The crawler walks

    month index  /search/inspectionMonthList.do?id=kda
    day list     /search/inspectionDayList.do?id=kda_1YYMML&level=3   (one per lunar month)
    article      /id/kda_1YYMMLDD_NNN

and writes one JSONL record per article in the repository's corpus contract.
Every response is cached under ``--cache``; rerunning the same command resumes
from the cache and skips article IDs already present in the output file.

Sample mode (``joseon-rag crawl --sample``) fetches one lunar month, by default
Sejong year 2, month 5 (1420), and holds at most ``SAMPLE_CAP`` records, enough
to test the pipeline and the demos without a full crawl. A rerun with a complete
cache makes no network request.

Calendar convention: Annals dates are lunar-calendar dates (음력) as recorded by
the court. ``year`` is the Western year printed with the record (세종 N년 = 1418 + N;
the accession year 즉위년 is 1418), while ``month`` and ``day`` are the lunar month
and day, not converted to the Julian or Gregorian calendar. Late lunar months can
therefore fall early in the following Western year. ``leap_month`` marks 윤달 and
``date_original`` keeps the page's own date line.

Access and reuse: on 2026-10-06 ``/robots.txt`` returned an HTML not-found page
(no crawl directives). The Korean translation carries the notice
"ⓒ 세종대왕기념사업회" and the classical-Chinese original carries a KOGL
(공공누리) mark. Keep crawled data local, follow the site's current terms, and do
not redistribute the corpus. The crawler re-checks robots.txt before the first
network request of every run and stops on HTTP 401/403/429 responses it cannot
recover from.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from datetime import date as calendar_date
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable

BASE_URL = "https://sillok.history.go.kr"
USER_AGENT = "PaperReach-joseon-rag/0.2 (educational research importer; +https://github.com/ghazanPK)"
MIN_DELAY_SECONDS = 1.0
SAMPLE_REIGN_YEAR, SAMPLE_MONTH = 2, 5  # 세종 2년 5월 (lunar), 1420
SAMPLE_CAP = 100                        # most records a sample may hold
SAMPLE_MIN_DELAY = 1.5                  # sample runs never go faster than the default rate
SAMPLE_OUT = Path("data/sejong-sample.jsonl")
SAMPLE_INDEX = Path("outputs/sejong-sample.index.json")
KING_CODES = {"kda": ("세종", 1418)}  # Sejong; reign year N = 1418 + N
MONTH_ID = re.compile(r"^kda_1(\d{2})(\d{2})([01])$")
ARTICLE_ID = re.compile(r"^kda_1(\d{2})(\d{2})([01])(\d{2})_(\d{3})$")
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


class CrawlStopped(RuntimeError):
    """The site refused access (robots.txt, 401/403 or persistent 429/5xx)."""


class NotCached(RuntimeError):
    """Offline mode and the page is not in the cache."""


# ------------------------------------------------------------------ mini DOM
class _Node:
    __slots__ = ("tag", "attrs", "children", "parent")

    def __init__(self, tag: str, attrs: dict[str, str], parent: "_Node | None"):
        self.tag = tag; self.attrs = attrs; self.children: list[_Node | str] = []; self.parent = parent

    @property
    def classes(self) -> set[str]:
        return set((self.attrs.get("class") or "").split())

    def iter(self) -> Iterable["_Node"]:
        stack = [self]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed([c for c in node.children if isinstance(c, _Node)]))

    def find_all(self, tag: str | None = None, cls: str | None = None) -> list["_Node"]:
        return [n for n in self.iter() if n is not self and (tag is None or n.tag == tag) and (cls is None or cls in n.classes)]

    def find(self, tag: str | None = None, cls: str | None = None) -> "_Node | None":
        found = self.find_all(tag, cls)
        return found[0] if found else None


class _TreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("#root", {}, None); self.cur = self.root

    def handle_starttag(self, tag, attrs):
        node = _Node(tag, {k: v or "" for k, v in attrs}, self.cur)
        self.cur.children.append(node)
        if tag not in VOID_TAGS:
            self.cur = node

    def handle_startendtag(self, tag, attrs):
        self.cur.children.append(_Node(tag, {k: v or "" for k, v in attrs}, self.cur))

    def handle_endtag(self, tag):
        node = self.cur
        while node is not None and node.tag != tag:
            node = node.parent
        if node is not None and node.parent is not None:
            self.cur = node.parent

    def handle_data(self, data):
        self.cur.children.append(data)


def _tree(html: str) -> _Node:
    builder = _TreeBuilder(); builder.feed(html); builder.close()
    return builder.root


def _text(node: _Node | None, skip: Callable[[_Node], bool] = lambda n: False) -> str:
    if node is None:
        return ""
    parts: list[str] = []

    def walk(n: _Node):
        for child in n.children:
            if isinstance(child, str):
                parts.append(child)
            elif child.tag not in {"script", "style"} and not skip(child):
                if child.tag in {"br", "p", "li", "div"}:
                    parts.append(" ")
                walk(child)
    walk(node)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _is_footnote_marker(node: _Node) -> bool:
    return node.tag == "sup"


# ------------------------------------------------------------------- parsers
@dataclass(frozen=True)
class MonthRef:
    id: str
    reign_year: int
    month: int
    leap: bool
    count: int  # number shown by the site's month index


def parse_month_index(html: str, king: str = "kda") -> list[MonthRef]:
    """Lunar-month entries of the chronicle (총서/appendix volumes are skipped)."""
    out, seen = [], set()
    for mid, cnt in re.findall(rf"search\('({king}_1\d{{5}})',\s*'(\d+)',\s*'3'\)", html):
        m = MONTH_ID.match(mid)
        if m and mid not in seen:
            seen.add(mid); out.append(MonthRef(mid, int(m.group(1)), int(m.group(2)), m.group(3) == "1", int(cnt)))
    return out


def parse_day_list(html: str, month_id: str) -> list[str]:
    """Article IDs of one lunar month, in page order. The page also lists the
    neighbouring months' articles; those are excluded by the ID prefix."""
    ids, seen = [], set()
    for aid in re.findall(r"searchView\('([a-z]{3}_\d{8}_\d{3})'\)", html):
        if aid.startswith(month_id) and aid not in seen:
            seen.add(aid); ids.append(aid)
    return ids


DATE_LINE = re.compile(r"(?P<book>\S+실록)\s*(?P<vol>\d+)\s*권\s*,\s*(?P<king>\S+?)\s*(?:(?P<accession>즉위년)|(?P<reign>\d+)\s*년)\s*"
                       r"(?P<leap>윤)?\s*(?P<month>\d+)\s*월\s*(?P<day>\d+)\s*일\s*(?P<ganji>[가-힣]{2})?\s*(?P<seq>\d+)\s*/\s*(?P<total>\d+)\s*기사"
                       r"(?:\s*/\s*(?P<year>\d{4})\s*년)?")


def parse_article(html: str, article_id: str, *, include_hanja: bool = False, url: str | None = None) -> dict:
    """Extract one article from a ``/id/<article id>`` page.

    Only the desktop ``div.detail-view`` block is read (the page repeats the
    article in a mobile block and carries site navigation). Footnote markers are
    removed from the body and footnotes are kept in ``notes``.
    """
    m = ARTICLE_ID.match(article_id)
    if not m:
        raise ValueError(f"not a Sejong Annals article ID: {article_id}")
    reign, month, leap, day = int(m.group(1)), int(m.group(2)), m.group(3) == "1", int(m.group(4))
    root = _tree(html)
    view = root.find("div", "detail-view")
    if view is None:
        raise ValueError(f"{article_id}: no article block (div.detail-view) found")
    head = view.find("div", "title-head")
    date_line = _text(head.find("p", "date") if head else None)
    title_node = head.find("h3") if head else None
    title = _text(title_node)
    items = view.find_all("div", "view-item")
    left = next((n for n in items if "left" in n.classes), None)
    right = next((n for n in items if "right" in n.classes), None)

    def paragraphs(column: _Node | None) -> str:
        box = column.find("div", "view-text") if column else None
        if box is None:
            return ""
        paras = [_text(p, _is_footnote_marker) for p in box.find_all("p", "paragraph")]
        return "\n".join(p for p in paras if p)

    text = paragraphs(left)
    if not text:
        raise ValueError(f"{article_id}: empty Korean translation")
    box = left.find("div", "view-text") if left else None
    notes = [re.sub(r"^\[註\s*\d+\]\s*", "", _text(li)) for ul in (box.find_all("ul", "ins_footnote") if box else []) for li in ul.find_all("li")]
    info = box.find("ul", "bot-info") if box else None
    categories = [_text(a) for a in info.find_all("a")] if info else []
    copies = [_text(s) for s in info.find_all("span")] if info else []
    d = DATE_LINE.search(date_line)
    year = 1418 + reign
    warnings = []
    if d:
        page = (0 if d.group("accession") else int(d.group("reign")), int(d.group("month")), int(d.group("day")), bool(d.group("leap")))
        if page != (reign, month, day, leap):
            warnings.append(f"date line {page} differs from article ID {(reign, month, day, leap)}")
        if d.group("year") and int(d.group("year")) != year:
            warnings.append(f"page year {d.group('year')} differs from 1418+{reign}")
        volume = f"{d.group('book')} {d.group('vol')}권"
    else:
        warnings.append("date line not recognised; date taken from the article ID")
        volume = ""
    record = {
        "id": article_id, "date": f"{year}-{month:02d}-{day:02d}", "year": year, "month": month, "day": day,
        "leap_month": leap, "calendar": "lunar", "reign": "세종", "reign_year": reign,
        "ganji": (d.group("ganji") or "") if d else "", "date_original": date_line,
        "title": title, "text": text, "source_url": url or f"{BASE_URL}/id/{article_id}", "volume": volume,
        "notes": notes, "categories": categories, "copies": copies,
        "import_method": "sillok-crawl", "retrieved": calendar_date.today().isoformat(),
    }
    if include_hanja:
        record["hanja_text"] = paragraphs(right)
    if warnings:
        record["parse_warnings"] = warnings
    return record


# ------------------------------------------------------------------- fetching
class PoliteFetcher:
    """Cached HTTP GET with a minimum interval between network requests."""

    def __init__(self, cache_dir: Path, *, delay: float = 1.5, offline: bool = False, timeout: int = 30, retries: int = 3,
                 opener: Callable[[urllib.request.Request, int], tuple[int, str, bytes]] | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic, log: Callable[[str], None] = print):
        if delay < MIN_DELAY_SECONDS:
            raise ValueError(f"delay must be at least {MIN_DELAY_SECONDS} s")
        self.cache = Path(cache_dir); self.delay = delay; self.offline = offline; self.timeout = timeout; self.retries = retries
        self.opener = opener or self._urlopen; self.sleep = sleep; self.clock = clock; self.log = log
        self.last: float | None = None; self.requests = 0; self.robots: urllib.robotparser.RobotFileParser | None = None
        self.robots_status = "not checked"

    @staticmethod
    def _urlopen(req: urllib.request.Request, timeout: int) -> tuple[int, str, bytes]:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read(5_000_001)
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", "") if e.headers else "", b""

    def _cache_path(self, key: str) -> Path:
        return self.cache / (re.sub(r"[^A-Za-z0-9_.=-]", "_", key) + ".html")

    def _network(self, url: str) -> tuple[int, str, bytes]:
        if self.last is not None:
            wait = self.delay - (self.clock() - self.last)
            if wait > 0:
                self.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "ko,en;q=0.5"})
        try:
            return self.opener(req, self.timeout)
        finally:
            self.last = self.clock(); self.requests += 1

    def check_robots(self) -> str:
        if self.offline:
            self.robots_status = "offline (not checked)"; return self.robots_status
        status, ctype, body = self._network(f"{BASE_URL}/robots.txt")
        text = body.decode("utf-8", "replace")
        if status == 200 and re.search(r"(?im)^\s*user-agent\s*:", text):
            parser = urllib.robotparser.RobotFileParser(); parser.parse(text.splitlines()); self.robots = parser
            self.robots_status = "robots.txt directives applied"
        elif status in (401, 403):
            raise CrawlStopped(f"robots.txt returned HTTP {status}; access is not permitted")
        else:
            self.robots_status = f"no robots.txt directives (HTTP {status}, {ctype or 'unknown type'})"
        return self.robots_status

    def get(self, url: str, key: str) -> str:
        path = self._cache_path(key)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        if self.offline:
            raise NotCached(f"{key} is not cached and --offline is set")
        if self.robots_status == "not checked":  # once per run, before the first network page
            self.log(f"robots: {self.check_robots()}")
        if self.robots is not None and not self.robots.can_fetch(USER_AGENT, url):
            raise CrawlStopped(f"robots.txt disallows {url}")
        backoff = max(self.delay, 5.0)
        for attempt in range(self.retries + 1):
            try:
                status, _, body = self._network(url)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                status, body = 0, b""; self.log(f"network error for {key}: {exc}")
            if status == 200 and body:
                if len(body) > 5_000_000:
                    raise ValueError(f"{key}: response exceeds 5 MB")
                html = body.decode("utf-8", "replace")
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".part"); tmp.write_text(html, encoding="utf-8"); tmp.replace(path)
                return html
            if status in (401, 403):
                raise CrawlStopped(f"HTTP {status} for {url}; stopping")
            if status == 404:
                raise FileNotFoundError(url)
            if attempt < self.retries:
                self.log(f"HTTP {status or 'error'} for {key}; retrying in {backoff:.0f} s")
                self.sleep(backoff); backoff = min(backoff * 2, 120)
        raise CrawlStopped(f"giving up on {url} after {self.retries + 1} attempts (last status {status})")


# ---------------------------------------------------------------------- crawl
def _parse_years(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    years: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        lo, hi = int(a), int(b or a)
        years.update(range(min(lo, hi), max(lo, hi) + 1))
    if any(not 0 <= y <= 32 for y in years):
        raise ValueError("Sejong reign years are 0 (accession year) to 32")
    return years


def sample_plan(*, years: str | None = None, month: int | None = None, leap: bool = False,
                max_articles: int | None = None, delay: float = SAMPLE_MIN_DELAY) -> dict:
    """Resolve ``--sample`` into crawl arguments: one lunar month, at most ``SAMPLE_CAP`` records.

    Defaults to Sejong year 2, month 5. ``--years`` (one reign year), ``--month``,
    ``--leap`` and a smaller ``--max-articles`` may narrow the sample; anything
    larger than the cap or faster than ``SAMPLE_MIN_DELAY`` is refused. The cap
    counts records already in the output, so reruns never grow the sample.
    """
    if years is not None:
        chosen = _parse_years(years) or set()
        if len(chosen) != 1:
            raise ValueError("--sample takes a single reign year, e.g. --years 2")
        years = str(next(iter(chosen)))
    if max_articles is not None and not 1 <= max_articles <= SAMPLE_CAP:
        raise ValueError(f"--sample holds 1-{SAMPLE_CAP} articles; drop --sample for a larger crawl")
    if delay < SAMPLE_MIN_DELAY:
        raise ValueError(f"--sample uses a delay of at least {SAMPLE_MIN_DELAY} s")
    return {"years": years or str(SAMPLE_REIGN_YEAR), "month": month or SAMPLE_MONTH, "leap": leap,
            "max_articles": None, "max_total": max_articles or SAMPLE_CAP, "delay": delay}


def crawl(out: Path, cache_dir: Path, *, years: str | None = None, month: int | None = None, leap: bool = False,
          max_articles: int | None = None, max_total: int | None = None, delay: float = 1.5,
          include_hanja: bool = False, offline: bool = False, fetcher: PoliteFetcher | None = None,
          log: Callable[[str], None] = print) -> dict:
    """Crawl the Sejong Annals into ``out`` (JSONL, appended). Resumable.

    ``month``/``leap`` select one lunar month within ``years``. ``max_articles``
    limits new records per run; ``max_total`` limits the records of the selected
    months held in ``out`` (existing ones included), as used by sample mode.
    """
    if month is not None and not 1 <= month <= 12:
        raise ValueError("lunar month must be 1-12")
    if leap and month is None:
        raise ValueError("--leap needs --month")
    out = Path(out); cache_dir = Path(cache_dir)
    fetcher = fetcher or PoliteFetcher(cache_dir, delay=delay, offline=offline, log=log)
    started = fetcher.clock()
    done: set[str] = set()
    if out.is_file():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try: done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError): pass
    wanted = _parse_years(years)
    index_html = fetcher.get(f"{BASE_URL}/search/inspectionMonthList.do?id=kda", "index-kda")
    listed = parse_month_index(index_html)
    if not listed:
        raise ValueError("no lunar months found in the month index; the page structure may have changed")
    months = [m for m in listed if (wanted is None or m.reign_year in wanted)
              and (month is None or (m.month, m.leap) == (month, leap))]
    if not months:
        raise ValueError(f"the month index lists no lunar month matching years={years!r} month={month} leap={leap}")
    out.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = failed = 0; failures = cache_dir / "failures.jsonl"

    def summary(stopped: str) -> dict:
        return _summary(out, months, written, skipped, failed, fetcher, fetcher.robots_status, stopped, fetcher.clock() - started)

    with out.open("a", encoding="utf-8") as sink:
        for ref in months:
            try:
                day_html = fetcher.get(f"{BASE_URL}/search/inspectionDayList.do?id={ref.id}&level=3", f"month-{ref.id}")
            except (FileNotFoundError, NotCached) as exc:
                failed += 1; log(f"skip month {ref.id}: {exc}"); continue
            ids = parse_day_list(day_html, ref.id)
            if not ids:
                log(f"warning: no articles listed for {ref.id}")
            for aid in ids:
                if max_total is not None and written + skipped >= max_total:
                    return summary("max-total")
                if aid in done:
                    skipped += 1; continue
                if max_articles is not None and written >= max_articles:
                    return summary("max-articles")
                url = f"{BASE_URL}/id/{aid}"
                try:
                    record = parse_article(fetcher.get(url, f"article-{aid}"), aid, include_hanja=include_hanja, url=url)
                except (ValueError, FileNotFoundError, NotCached) as exc:
                    failed += 1
                    failures.parent.mkdir(parents=True, exist_ok=True)
                    with failures.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"id": aid, "error": str(exc)}, ensure_ascii=False) + "\n")
                    log(f"skip {aid}: {exc}"); continue
                sink.write(json.dumps(record, ensure_ascii=False) + "\n"); sink.flush()
                done.add(aid); written += 1
                if written % 50 == 0:
                    log(f"{written} articles written ({aid}); {fetcher.requests} network requests")
    return summary("complete")


def _summary(out, months, written, skipped, failed, fetcher, robots, stopped, elapsed):
    if robots == "not checked":
        robots = "offline (not checked)" if fetcher.offline else "not checked (every page was cached; no network request)"
    result = {"out": str(out), "months": len(months), "written": written, "skipped_existing": skipped, "failed": failed,
              "network_requests": fetcher.requests, "robots": robots, "stopped": stopped, "elapsed_seconds": round(elapsed, 1)}
    if len(months) <= 12:
        result["month_ids"] = [m.id for m in months]
    return result
