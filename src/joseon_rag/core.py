from __future__ import annotations

import base64
import csv
import json
import math
import os
import re
import shlex
import subprocess
import urllib.request
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Protocol

# --------------------------------------------------------------------------- tokens
# Lexical retrieval tokens. Latin words/numbers stay whole words; Hangul and Hanja
# runs become overlapping character bigrams, so "집현전의 학사" and "집현전 학사"
# share 집현/현전/학사 even though their eojeol differ. Single-character runs are kept.
LEXICAL_TOKENIZER = "korean-char-bigram-v1"
_RUN = re.compile(r"[가-힣]+|[㐀-䶿一-鿿豈-﫿]+|[0-9A-Za-z]+")
_LEGACY_WORD = re.compile(r"[0-9A-Za-z가-힣]+")
_HANGUL = re.compile(r"[가-힣]")
_HANJA = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
STOPWORDS = frozenset("""a an the of and or in on at to for from by with as is are was were be been being it its
this that these those what which who whom whose when where why how did does do done me tell about please
describe explain there their they them he she his her i you we us our your can could would should will
related approximate approximately date""".split())
# Generic question words ignored when deciding whether a query is "date only".
_QUESTION_WORDS = STOPWORDS | frozenset(
    "happened happen event events record records occurred occur anything something day date year month historical context "
    "approximate approximately details during dated circa ca since until till before after around between".split()) | frozenset(
    "무엇 무슨 어떤 일 일이 일은 있었나 있었나요 있었는가 있었는지 있었니 일어났나 일어났나요 일어난 알려줘 알려주세요 기록 기록은 날 그날 당시 때 "
    "에 에서 에는 의 은 는 이 가 을 를 와 과 도 로 으로".split())


def tokenize(text: str) -> list[str]:
    """Korean-aware lexical tokens used by the default TF-IDF index."""
    out: list[str] = []
    for run in _RUN.findall(text or ""):
        if run[0].isascii():
            low = run.lower()
            if len(low) > 1 and low not in STOPWORDS:
                out.append(low)
        elif len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


def _legacy_tokenize(text: str) -> list[str]:
    """Whole-word tokens used by indexes built before the bigram tokenizer."""
    return [t.lower() for t in _LEGACY_WORD.findall(text or "") if len(t) > 1]


def _index_tokenizer(index: dict[str, Any]) -> Callable[[str], list[str]]:
    return tokenize if index.get("tokenizer") == LEXICAL_TOKENIZER else _legacy_tokenize


# ---------------------------------------------------------------- token budgets
_TIKTOKEN_CACHE: dict[str, Any] = {}


def conservative_token_estimate(text: str) -> int:
    """Upper-leaning token estimate without a tokenizer.

    Hangul syllables count as one token each, Hanja/CJK characters as two, and
    remaining non-space characters as one token per three characters. Common GPT
    tokenizers usually need fewer tokens than this for Korean/Hanja text, so the
    budget errs towards sending less rather than overflowing the model context.
    """
    text = text or ""
    hangul = len(_HANGUL.findall(text)); hanja = len(_HANJA.findall(text))
    other = len(re.sub(r"\s", "", text)) - hangul - hanja
    return max(1, hangul + 2 * hanja + math.ceil(max(0, other) / 3))


def token_counter(name: str = "auto") -> tuple[Callable[[str], int], str]:
    """Return (count_function, label). ``auto`` uses tiktoken when it is installed
    and its encoding is available locally; otherwise the conservative estimate."""
    if name not in {"auto", "tiktoken", "chars"}:
        raise ValueError("token counter must be auto, tiktoken or chars")
    if name in {"auto", "tiktoken"}:
        try:
            if "enc" not in _TIKTOKEN_CACHE:
                import tiktoken  # optional
                _TIKTOKEN_CACHE["enc"] = tiktoken.get_encoding(os.getenv("JOSEON_TIKTOKEN_ENCODING", "o200k_base"))
            enc = _TIKTOKEN_CACHE["enc"]
            return (lambda text: max(1, len(enc.encode(text or "")))), f"tiktoken:{enc.name}"
        except Exception as exc:  # missing package or encoding file
            if name == "tiktoken":
                raise RuntimeError("tiktoken is not installed or its encoding is unavailable") from exc
    return conservative_token_estimate, "conservative-char-v1"


# ------------------------------------------------------------------- articles
@dataclass
class Article:
    id: str
    text: str
    title: str = ""
    year: int | None = None
    month: int | None = None
    day: int | None = None
    source_url: str = ""
    volume: str = ""
    calendar: str = ""
    leap_month: bool = False
    date_original: str = ""

    @property
    def date_label(self) -> str:
        date = "-".join(str(x).zfill(2) for x in (self.year, self.month, self.day) if x is not None)
        if not date:
            return "date unknown"
        if self.calendar == "lunar":
            return f"{date} (lunar{' leap month' if self.leap_month else ''})"
        return date

    @property
    def citation(self) -> str:
        return f"[{self.id}] {self.date_label} {self.title}".strip()


_ARTICLE_FIELDS = {f.name for f in fields(Article)}


def _article(data: dict[str, Any]) -> Article:
    return Article(**{k: v for k, v in data.items() if k in _ARTICLE_FIELDS})


# ---------------------------------------------------------------------- dates
SEJONG_FIRST_YEAR = 1418  # 세종 즉위년 (accession year, reign year 0); year N = 1418 + N
SEJONG_LAST_REIGN_YEAR = 32
_Y = r"(1[3-8]\d{2}|19(?:0\d|10))"
_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_MN = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?(?![a-z])"
_D = r"(3[01]|[12]\d|0?[1-9])(?:st|nd|rd|th)?(?!\d)"
_RANGE_SEP = r"(?:-|–|—|~|to|until|through|and)"
_STRONG_CTX = r"(?:\b(?:in|during|year|dated|date|circa|ca\.|c\.)\s*:?\s+|\(\s*)"
_WEAK_CTX = r"\b(?:of|by|around|since|until|till|before|after|from|on)\s+"
_WEAK_END = r"(?=\s*(?:$|[?.!,;:)]|and\b|to\b|ad\b|ce\b|년))"

_DATE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("reign-ko-range", re.compile(r"세종\s*(?:대왕\s*)?(즉위년|\d{1,2}\s*년)\s*(?:부터\s*|[-–~]\s*)(\d{1,2})\s*년")),
    ("reign-ko", re.compile(r"세종\s*(?:대왕\s*)?(?:(즉위)\s*년|(\d{1,2})\s*년)(?:\s*(윤)?\s*(\d{1,2})\s*월(?:\s*(\d{1,2})\s*일)?)?")),
    ("reign-en-ordinal", re.compile(r"\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+(?:regnal\s+)?year\s+of\s+(?:king\s+)?sejong", re.I)),
    ("reign-en-accession", re.compile(r"\b(?:accession\s+year\s+of\s+(?:king\s+)?sejong|(?:king\s+)?sejong(?:'s)?\s+accession\s+year)", re.I)),
    ("reign-en", re.compile(r"\b(?:king\s+)?sejong(?:'s)?\s+(?:reign\s+)?(?:year\s+)?(\d{1,2})(?:st|nd|rd|th)?(?:\s+year)?(?!\d)", re.I)),
    ("year-range-words", re.compile(rf"\b(?:between|from)\s+{_Y}\s*년?\s*(?:and|to|until|through|-|–|—|~)\s*{_Y}(?!\d)", re.I)),
    ("year-range-ko", re.compile(rf"(?<!\d){_Y}\s*년?\s*(?:부터|~|-|–|—)\s*{_Y}\s*년")),
    ("year-range", re.compile(rf"(?<!\d){_Y}\s*(?:–|—|~|-|\bto\b)\s*{_Y}(?!\d)", re.I)),
    ("iso", re.compile(rf"(?<!\d){_Y}([-/.])(1[0-2]|0?[1-9])(?!\d)(?:\2(3[01]|[12]\d|0?[1-9])(?!\d))?")),
    ("korean", re.compile(rf"(?<!\d){_Y}\s*년(?:\s*(윤)?\s*(1[0-2]|0?[1-9])\s*월(?:\s*(3[01]|[12]\d|0?[1-9])\s*일)?)?")),
    ("en-month-range", re.compile(rf"\b{_MN}\s*{_RANGE_SEP}\s*{_MN},?\s+(?:of\s+)?{_Y}(?!\d)", re.I)),
    ("en-mdy", re.compile(rf"\b{_MN}\s+{_D}(?:\s*(?:-|–|—|to)\s*{_D})?,?\s+(?:of\s+)?{_Y}(?!\d)", re.I)),
    ("en-dmy", re.compile(rf"\b(?:the\s+)?{_D}(?:\s*(?:-|–|—|to)\s*{_D})?\s+(?:of\s+)?{_MN},?\s+{_Y}(?!\d)", re.I)),
    ("en-my", re.compile(rf"\b{_MN},?\s+(?:of\s+)?{_Y}(?!\d)", re.I)),
    ("ad", re.compile(rf"(?<!\d){_Y}\s*(?:AD\b|A\.D\.|CE\b)")),
    ("year-strong", re.compile(rf"{_STRONG_CTX}{_Y}(?!\d)", re.I)),
    ("year-weak", re.compile(rf"{_WEAK_CTX}{_Y}(?!\d){_WEAK_END}", re.I)),
]


@dataclass(frozen=True)
class DateSpec:
    """A date constraint parsed from text. ``year_end`` marks an inclusive year range."""
    year: int | None = None
    month: int | None = None
    day: int | None = None
    year_end: int | None = None
    leap_month: bool = False
    rule: str = ""
    text: str = ""

    @property
    def found(self) -> bool:
        return self.year is not None

    def as_tuple(self) -> tuple[int | None, int | None, int | None]:
        return self.year, self.month, self.day


def _reign(n: int) -> int | None:
    return SEJONG_FIRST_YEAR + n if 0 <= n <= SEJONG_LAST_REIGN_YEAR else None


def _month(name: str) -> int:
    return _MONTHS[name.lower()[:3]]


def _spec_from(rule: str, m: re.Match[str]) -> DateSpec | None:
    g = m.groups(); t = m.group(0)
    if rule == "reign-ko-range":
        a = 0 if g[0].startswith("즉위") else int(re.sub(r"\D", "", g[0]))
        a, b = _reign(a), _reign(int(g[1]))
        return DateSpec(min(a, b), year_end=max(a, b), rule=rule, text=t) if a and b and a != b else None
    if rule == "reign-ko":
        y = _reign(0 if g[0] else int(g[1]))
        if y is None:
            return None
        mo = int(g[3]) if g[3] else None; d = int(g[4]) if g[4] else None
        if mo is not None and not 1 <= mo <= 12 or d is not None and not 1 <= d <= 30:
            return DateSpec(y, rule=rule, text=t)
        return DateSpec(y, mo, d, leap_month=bool(g[2]), rule=rule, text=t)
    if rule in {"reign-en-ordinal", "reign-en"}:
        y = _reign(int(g[0]))
        return DateSpec(y, rule=rule, text=t) if y else None
    if rule == "reign-en-accession":
        return DateSpec(SEJONG_FIRST_YEAR, rule=rule, text=t)
    if rule.startswith("year-range"):
        a, b = int(g[0]), int(g[1])
        if a == b:
            return DateSpec(a, rule=rule, text=t)
        return DateSpec(min(a, b), year_end=max(a, b), rule=rule, text=t) if abs(a - b) <= 100 else None
    if rule == "iso":
        return DateSpec(int(g[0]), int(g[2]), int(g[3]) if g[3] else None, rule=rule, text=t)
    if rule == "korean":
        return DateSpec(int(g[0]), int(g[2]) if g[2] else None, int(g[3]) if g[2] and g[3] else None, leap_month=bool(g[1]), rule=rule, text=t)
    if rule == "en-month-range":
        return DateSpec(int(g[2]), rule=rule, text=t)
    if rule == "en-mdy":
        # A day range such as "May 12-14, 1420" narrows to the containing month.
        return DateSpec(int(g[3]), _month(g[0]), None if g[2] else int(g[1]), rule=rule, text=t)
    if rule == "en-dmy":
        return DateSpec(int(g[3]), _month(g[2]), None if g[1] else int(g[0]), rule=rule, text=t)
    if rule == "en-my":
        return DateSpec(int(g[1]), _month(g[0]), rule=rule, text=t)
    return DateSpec(int(g[0]), rule=rule, text=t)


_CONTEXT_RULES = {"year-strong", "year-weak", "year-range-words"}


def _date_matches(text: str) -> list[tuple[int, int, DateSpec]]:
    """(start, end, spec) per match. For rules introduced by a context word
    ("in", "date:", "from") the span starts at the year itself, so a more specific
    expression at the same position ("date: 1446-09") wins the tie."""
    found = []
    for rule, pattern in _DATE_PATTERNS:
        for m in pattern.finditer(text or ""):
            spec = _spec_from(rule, m)
            if spec and spec.year is not None:
                found.append((m.start(1) if rule in _CONTEXT_RULES else m.start(), m.end(), spec))
    return found


def parse_date(text: str) -> DateSpec:
    """Parse the first date mentioned in free text.

    Bare numbers are never dates: a year needs date context ("in 1420", "1420년",
    "May 1420", "세종 2년", "date: 1420"). Ranges become year ranges; a day range
    inside one month narrows to that month. When several expressions name the same
    year, the most specific one is kept.
    """
    matches = _date_matches(text)
    if not matches:
        return DateSpec()
    start, end, first = min(matches, key=lambda x: (x[0], -(x[1] - x[0])))
    if first.year_end is not None or first.rule == "en-month-range":
        return first
    # Another, separate mention of the same year may add the month/day.
    same = [s for a, b, s in matches if (b <= start or a >= end) and s.year == first.year and s.year_end is None
            and (first.month is None or s.month == first.month)]
    return max([first, *same], key=lambda s: (s.month is not None, s.day is not None))


def parse_record_date(value: str) -> DateSpec:
    """Lenient parser for a structured record ``date`` field (a bare year is allowed)."""
    value = str(value or "").strip()
    m = re.fullmatch(_Y, value)
    if m:
        return DateSpec(int(m.group(1)), rule="bare-year-field", text=value)
    return parse_date(value)


def strip_date_expressions(text: str) -> str:
    keep = [True] * len(text or "")
    for start, end, _ in _date_matches(text):
        keep[start:end] = [False] * (end - start)
    return "".join(c if k else " " for c, k in zip(text or "", keep))


def content_terms(text: str) -> list[str]:
    """Topical words of a query after removing date expressions and question words."""
    words = re.findall(r"[0-9A-Za-z]+|[가-힣㐀-䶿一-鿿豈-﫿]+", strip_date_expressions(text))
    return [w for w in (x.lower() for x in words) if w not in _QUESTION_WORDS and not w.isdigit() and len(w) > (1 if w.isascii() else 0)]


def query_date(query: str) -> tuple[int | None, int | None, int | None]:
    return parse_date(query).as_tuple()


def _date_from(value: str) -> tuple[int | None, int | None, int | None]:
    return parse_record_date(value).as_tuple()


def load_articles(path: Path) -> list[Article]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = []
    for i, row in enumerate(rows, 1):
        text = str(row.get("text") or row.get("content") or "").strip()
        if not text:
            raise ValueError(f"record {i} has no text/content")
        if row.get("date"):
            spec = parse_record_date(str(row["date"]))
        else:
            spec = parse_date(str(row.get("title") or ""))
            if not spec.found:
                spec = parse_date(text[:120])
        def num(key: str, fallback: int | None) -> int | None:
            return int(row[key]) if row.get(key) not in (None, "") else fallback
        leap = row.get("leap_month")
        out.append(Article(
            str(row.get("id") or f"article-{i}"), text, str(row.get("title") or ""),
            num("year", spec.year), num("month", spec.month), num("day", spec.day),
            str(row.get("source_url") or row.get("url") or ""), str(row.get("volume") or ""),
            str(row.get("calendar") or ""),
            (str(leap).lower() in {"1", "true", "yes"}) if leap not in (None, "") else spec.leap_month,
            str(row.get("date_original") or "")))
    return out


# -------------------------------------------------------------------- indexing
_DENSE_MODELS: dict[str, Any] = {}
_DENSE_MATRICES: dict[int, tuple[Any, Any]] = {}


def _dense_model(name: str):
    if name not in _DENSE_MODELS:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("install the optional 'semantic' extra") from e
        _DENSE_MODELS[name] = SentenceTransformer(name, local_files_only=True)
    return _DENSE_MODELS[name]


def _dense_matrix(index: dict[str, Any]):
    """Article vectors as one float32 numpy matrix, decoded once per loaded index."""
    import numpy as np
    key_obj = index.get("vectors_b64") or index.get("vectors")
    cached = _DENSE_MATRICES.get(id(key_obj))
    if cached is not None and cached[0] is key_obj:
        return cached[1]
    if index.get("vectors_b64"):
        matrix = np.frombuffer(base64.b64decode(index["vectors_b64"]), dtype="<f4").reshape(len(index["articles"]), int(index["dim"]))
    else:
        matrix = np.asarray(index["vectors"], dtype=np.float32)
    _DENSE_MATRICES[id(key_obj)] = (key_obj, matrix)
    return matrix


def build_index(articles: list[Article], backend: str = "tfidf", model_name: str | None = None) -> dict[str, Any]:
    if backend == "sentence-transformer":
        if not model_name:
            raise ValueError("sentence-transformer backend requires a model name or cached path")
        import numpy as np
        model = _dense_model(model_name)
        texts = [f"{a.title}\n{a.text}" for a in articles]
        vectors = np.asarray(model.encode(texts, normalize_embeddings=True), dtype="<f4")
        return {"schema": "paperreach.joseon.article-index.v2", "backend": "sentence-transformer-cache-only", "model": model_name,
                "max_seq_length": getattr(model, "max_seq_length", None), "articles": [asdict(a) for a in articles],
                "dim": int(vectors.shape[1]), "vectors_b64": base64.b64encode(vectors.tobytes()).decode("ascii")}
    docs = [tokenize(f"{a.title} {a.text}") for a in articles]
    df: dict[str, int] = {}
    for doc in docs:
        for t in set(doc):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log((1 + len(docs)) / (1 + n)) + 1 for t, n in df.items()}
    vectors = []
    for doc in docs:
        counts: dict[str, int] = {}
        for t in doc:
            counts[t] = counts.get(t, 0) + 1
        vec = {t: round(c * idf[t], 6) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1
        vectors.append({t: round(v / norm, 6) for t, v in vec.items()})
    return {"schema": "paperreach.joseon.article-index.v2", "backend": "tfidf-baseline", "tokenizer": LEXICAL_TOKENIZER,
            "idf": idf, "articles": [asdict(a) for a in articles], "vectors": vectors}


def _cosine(query: str, index: dict[str, Any], vector: dict[str, float]) -> float:
    counts: dict[str, int] = {}
    for t in _index_tokenizer(index)(query):
        counts[t] = counts.get(t, 0) + 1
    q = {t: c * index["idf"].get(t, 1.0) for t, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in q.values())) or 1
    return sum((v / norm) * vector.get(t, 0.0) for t, v in q.items())


def _dense_scores(query: str, index: dict[str, Any]) -> list[float]:
    model = _dense_model(index["model"])
    q = model.encode([query], normalize_embeddings=True)[0]
    return (_dense_matrix(index) @ q.astype("float32")).tolist()


# ---------------------------------------------------------------- prompts/LLMs
PROMPT_PROFILES: dict[str, dict[str, str]] = {
    "journal": {
        "generation": (
            "You are a historical-analysis assistant for the Annals of the Joseon Dynasty (Sejong Annals, 1418-1450). "
            "The user message contains a question and source articles retrieved up to the model's maximum context. "
            "Read all of them, then select only the articles and passages necessary for the question and ignore the rest.\n\n"
            "Write the answer in the language of the question with two sections:\n"
            "Objective facts: short statements of what the records state, each ending with the bracketed source ID it comes from, "
            "for example [kda_10205029_001]. State nothing that is absent from the sources.\n"
            "Contextual analysis: interpret the facts - background, causes, significance and links between the cited records. "
            "Keep the interpretation grounded in the cited records, cite the IDs it relies on, and mark uncertainty.\n\n"
            "Abstain instead of answering when the sources contain no record of the event, person or topic asked about "
            "(it may not exist), when the date in the question contradicts the dates in the sources, or when the sources are "
            "insufficient. To abstain, begin with 'INSUFFICIENT EVIDENCE:' and explain briefly; for a date contradiction, "
            "state the date the records give with its citation. Never invent facts, dates or source IDs. "
            "Article dates follow the Annals' lunar calendar; the year is the Western year of that lunar year."),
        "regeneration": (
            "You rewrite a user's question about the Annals of the Joseon Dynasty (Sejong reign, 1418-1450) into a more detailed "
            "search query for a vector store of article-level records. Add details about the events, people, institutions or topics "
            "the question mentions (names, Korean and Hanja terms, related institutions). If the question gives no date, add the "
            "approximate date of the event as 'approximate date: YYYY' (or YYYY-MM or YYYY-MM-DD when well established). If the "
            "question already contains a date, keep it exactly as written; do not correct it. If you do not know the event, add no "
            "details or dates. Return only the rewritten query on one line."),
    },
    "ismar": {
        "generation": (
            "You are the voice of an embodied historical agent answering questions about the Annals of the Joseon Dynasty "
            "(Sejong Annals, 1418-1450). The user message contains a question and source articles retrieved up to the model's "
            "maximum context. Review them, select only the necessary information and give a clear, reliable, well-referenced answer.\n\n"
            "Use two short sections in the language of the question:\n"
            "Objective facts: short spoken sentences, each ending with the bracketed source ID it comes from, for example "
            "[kda_10205029_001]. State nothing that is absent from the sources.\n"
            "Contextual analysis: two or three sentences interpreting the facts, citing the IDs they rely on.\n"
            "Do not use tables, markdown lists or headings beyond the two section labels; bracketed IDs are removed before speech.\n\n"
            "Abstain when the sources contain no record of the event (it may not exist), when the date in the question contradicts the "
            "sources, or when the sources are insufficient: begin with 'INSUFFICIENT EVIDENCE:' and explain in one or two sentences, "
            "giving the date the records state with its citation for a date contradiction. Never invent facts, dates or source IDs. "
            "Article dates follow the Annals' lunar calendar."),
        "regeneration": (
            "Rewrite the user's spoken question about the Annals of the Joseon Dynasty (Sejong reign, 1418-1450) into a clearer search "
            "query. Add missing context about the event, people or institutions, and when the question gives no date add the "
            "approximate date as 'approximate date: YYYY' (or YYYY-MM / YYYY-MM-DD when well established). Keep any date the user gave "
            "exactly as stated. If you do not know the event, add nothing. Return only the rewritten query on one line."),
    },
}
ABSTAIN_MARKER = "INSUFFICIENT EVIDENCE:"
CHAT_STYLES = ("auto", "standard", "reasoning")


def profile_prompts(profile: str) -> dict[str, str]:
    if profile not in PROMPT_PROFILES:
        raise ValueError(f"unknown prompt profile {profile!r}; choose {', '.join(PROMPT_PROFILES)}")
    return PROMPT_PROFILES[profile]


def is_reasoning_style(model: str, chat_style: str = "auto") -> bool:
    if chat_style not in CHAT_STYLES:
        raise ValueError(f"chat style must be one of {', '.join(CHAT_STYLES)}")
    if chat_style != "auto":
        return chat_style == "reasoning"
    return bool(re.match(r"o\d", model.rsplit("/", 1)[-1].lower()))


def chat_payload(model: str, instructions: str, user: str, chat_style: str = "auto", max_output_tokens: int | None = None) -> dict[str, Any]:
    """Chat-completions payload. Reasoning models (o1-style) take a ``developer``
    message, reject ``temperature`` and use ``max_completion_tokens``."""
    if is_reasoning_style(model, chat_style):
        body: dict[str, Any] = {"model": model, "messages": [{"role": "developer", "content": instructions}, {"role": "user", "content": user}]}
        if max_output_tokens:
            body["max_completion_tokens"] = max_output_tokens
        return body
    body = {"model": model, "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": user}], "temperature": 0}
    if max_output_tokens:
        body["max_tokens"] = max_output_tokens
    return body


Transport = Callable[[str, dict[str, Any], dict[str, str], int], dict[str, Any]]


def _http_transport(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(url, json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class _ChatEndpoint:
    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY", *, profile: str = "journal",
                 chat_style: str = "auto", max_output_tokens: int | None = None, transport: Transport | None = None):
        self.url = base_url.rstrip("/") + "/chat/completions"; self.model = model; self.key = os.getenv(api_key_env, "")
        self.prompts = profile_prompts(profile); self.profile = profile; self.chat_style = chat_style
        is_reasoning_style(model, chat_style)  # validate early
        self.max_output_tokens = max_output_tokens; self.transport = transport or _http_transport

    def _complete(self, instructions: str, user: str, timeout: int) -> str:
        payload = chat_payload(self.model, instructions, user, self.chat_style, self.max_output_tokens)
        headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {self.key}"} if self.key else {})}
        data = self.transport(self.url, payload, headers, timeout)
        return str(data["choices"][0]["message"]["content"] or "").strip()


class Regenerator(Protocol):
    def rewrite(self, query: str) -> str: ...


class RuleRegenerator:
    """Offline stand-in for the paper's GPT query regeneration: adds event aliases and
    details from a user-maintained events file and an approximate date when missing."""
    def __init__(self, events: dict[str, Any] | None = None): self.events = events or {}

    def rewrite(self, query: str) -> str:
        low = query.lower(); extra: list[str] = []; date = None
        for name, info in self.events.items():
            terms = [name, *info.get("aliases", [])]
            if any(term.lower() in low for term in terms):
                extra.extend(str(x) for x in terms if x.lower() not in low)
                extra.extend(str(x) for x in info.get("details", []) if str(x).lower() not in low)
                if date is None and info.get("year"):
                    parts = [str(info["year"])]
                    if info.get("month"):
                        parts.append(str(info["month"]).zfill(2))
                        if info.get("day"):
                            parts.append(str(info["day"]).zfill(2))
                    date = "-".join(parts)
        if not parse_date(query).found and date:
            extra.append(f"approximate date: {date}")
        return query if not extra else f"{query} | related: {'; '.join(extra)}"


class OpenAICompatibleRegenerator(_ChatEndpoint):
    def rewrite(self, query: str) -> str:
        return self._complete(self.prompts["regeneration"], query, 60) or query


class LocalCommandRegenerator:
    def __init__(self, command: str, profile: str = "journal"): self.command = shlex.split(command); self.profile = profile

    def rewrite(self, query: str) -> str:
        payload = {"query": query, "system_prompt": profile_prompts(self.profile)["regeneration"], "profile": self.profile}
        p = subprocess.run(self.command, input=json.dumps(payload, ensure_ascii=False), text=True, capture_output=True, check=True, timeout=60)
        try: return json.loads(p.stdout)["query"]
        except (json.JSONDecodeError, KeyError): return p.stdout.strip()


# ------------------------------------------------------------------- retrieval
DEFAULT_SIMILARITY_FLOOR = {"tfidf": 0.05, "dense": 0.30}


def _floor_for(index: dict[str, Any], min_similarity: float | None) -> float:
    if min_similarity is not None:
        return float(min_similarity)
    return DEFAULT_SIMILARITY_FLOOR["dense" if index["backend"].startswith("sentence-transformer") else "tfidf"]


def retrieve(index: dict[str, Any], original: str, rewritten: str, token_budget: int = 2400, min_filtered: int = 2,
             *, min_similarity: float | None = None, tokenizer: str = "auto") -> dict[str, Any]:
    """Date-filter, score, apply the similarity floor and pack whole articles.

    Packing is fit-or-stop: ranked articles are added while the next one fits the
    budget, and packing stops at the first that does not. A lower-ranked short
    article never displaces a higher-ranked long one, so the model sees a strict
    top-k prefix of the ranking (some budget may stay unused).
    """
    explicit = parse_date(original)
    inferred = parse_date(rewritten)
    spec = explicit if explicit.found else inferred
    source = "original" if explicit.found else ("regenerated" if inferred.found else "none")
    rows = [_article(a) for a in index["articles"]]

    def matches(a: Article) -> bool:
        if spec.year_end is not None:
            return a.year is not None and spec.year <= a.year <= spec.year_end
        return (spec.year is None or a.year == spec.year) and (spec.month is None or a.month == spec.month) and (spec.day is None or a.day == spec.day)

    candidates = [i for i, a in enumerate(rows) if matches(a)]
    filter_mode = ("year-range" if spec.year_end else "exact-or-partial-date") if spec.found else "all-records"
    widened = False
    if len(candidates) < min_filtered and source == "regenerated":
        candidates = list(range(len(rows))); widened = True; filter_mode = "inferred-date-fallback-all"

    floor = _floor_for(index, min_similarity)
    topical = content_terms(rewritten)
    asked_topic = content_terms(original)
    # A date-only question ("What happened on May 12, 1420?") selects by date alone.
    floor_waived = source == "original" and not asked_topic
    dense = _dense_scores(rewritten, index) if index["backend"].startswith("sentence-transformer") else None
    qterms = set(_index_tokenizer(index)(rewritten))

    def score(i: int) -> tuple[float, float, Article]:
        a = rows[i]; base = dense[i] if dense is not None else _cosine(rewritten, index, index["vectors"][i])
        title_overlap = len(qterms & set(_index_tokenizer(index)(a.title))) / max(1, len(qterms))
        in_range = spec.found and (spec.year <= (a.year or -1) <= (spec.year_end or spec.year))
        return base + .15 * title_overlap + (.08 if in_range else 0), base, a

    scored = [score(i) for i in candidates]
    kept = [s for s in scored if floor_waived or s[1] >= floor]
    kept.sort(key=lambda x: (-x[0], x[2].id))
    count, counter_label = token_counter(tokenizer)
    selected = []; used = 0; stopped_at = None
    for rerank, base, a in kept:
        cost = count(f"{a.title}\n{a.text}")
        if used + cost > token_budget:
            stopped_at = a.id; break
        selected.append({"article": asdict(a), "similarity": round(base, 6), "rerank_score": round(rerank, 6), "estimated_tokens": cost})
        used += cost
    bundle: dict[str, Any] = {
        "original_query": original, "rewritten_query": rewritten,
        "date_filter": {"year": spec.year, "month": spec.month, "day": spec.day, "year_end": spec.year_end, "mode": filter_mode,
                        "source": source, "matched_text": spec.text, "widened": widened},
        "similarity_floor": {"value": floor, "waived": floor_waived, "below_floor": len(scored) - len(kept)},
        "packing": {"policy": "fit-or-stop", "stopped_at": stopped_at},
        "token_budget": token_budget, "token_counter": counter_label, "estimated_tokens": used, "evidence": selected,
        "topical_terms": topical,
    }
    if source == "original" and not selected and stopped_at is None and asked_topic:
        # Explicit date with no relevant record: look for the topic elsewhere so the
        # answer can say which date the records actually give.
        others = [score(i) for i in range(len(rows)) if i not in set(candidates)]
        others = sorted((s for s in others if s[1] >= floor), key=lambda x: (-x[0], x[2].id))[:3]
        if others:
            bundle["date_conflict"] = {"requested": {"year": spec.year, "month": spec.month, "day": spec.day, "year_end": spec.year_end},
                                       "records": [{"id": a.id, "date": a.date_label, "title": a.title, "similarity": round(b, 6)} for _, b, a in others]}
            for rerank, base, a in others:
                cost = count(f"{a.title}\n{a.text}")
                if used + cost > token_budget:
                    break
                selected.append({"article": asdict(a), "similarity": round(base, 6), "rerank_score": round(rerank, 6),
                                 "estimated_tokens": cost, "outside_date_filter": True})
                used += cost
            bundle["estimated_tokens"] = used
    return bundle


# ------------------------------------------------------------------ generation
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")


def _abstain(bundle: dict[str, Any], reason: str, text: str, citations: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {"answer": f"{ABSTAIN_MARKER} {text}", "facts": [], "claims": [], "analysis": "", "abstained": True, "abstain_reason": reason,
            "citations": citations or [], "trace": bundle, "generation": "extractive-offline-baseline"}


def answer(bundle: dict[str, Any], max_sentences: int = 6) -> dict[str, Any]:
    """Extractive offline answer: cited source sentences plus a metadata-only context
    section. It abstains when nothing passes the similarity floor, when selected
    sentences share no topical term with the question, or when the question's date
    contradicts the dates of the matching records."""
    conflict = bundle.get("date_conflict")
    if conflict:
        req = "-".join(str(x).zfill(2) for x in (conflict["requested"]["year"], conflict["requested"]["month"], conflict["requested"]["day"]) if x is not None)
        listed = "; ".join(f"{r['title'] or r['id']} is dated {r['date']} [{r['id']}]" for r in conflict["records"])
        cites = [{"id": item["article"]["id"], "label": _article(item["article"]).citation, "url": item["article"]["source_url"]} for item in bundle["evidence"]]
        return _abstain(bundle, "date-conflict", f"No record dated {req} supports this question. The matching records give a different date: {listed}.", cites)
    if not bundle["evidence"]:
        why = ("the highest-ranked matching article does not fit the token budget" if bundle.get("packing", {}).get("stopped_at")
               else "no article passed the date filter and similarity floor; the event may not be recorded in this corpus or may not exist")
        return _abstain(bundle, "no-evidence", f"The retrieved records do not support an answer: {why}.")
    tok = tokenize
    q = set(tok(" ".join(bundle.get("topical_terms") or []))) or set(tok(bundle["rewritten_query"]))
    date_only = bundle.get("similarity_floor", {}).get("waived", False)
    candidates = []
    for rank, item in enumerate(bundle["evidence"], 1):
        a = _article(item["article"])
        for position, sentence in enumerate(s.strip() for s in _SENTENCE_SPLIT.split(a.text)):
            if not sentence:
                continue
            overlap = len(q & set(tok(sentence))) / max(1, len(q))
            if overlap == 0 and not date_only:
                continue
            candidates.append((overlap + item["rerank_score"] * .35 - position * 1e-4, rank, sentence, a))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    picked = []; seen = set()
    for score, rank, sentence, a in candidates:
        key = sentence.lower()
        if key in seen: continue
        seen.add(key); picked.append({"text": sentence, "citation_id": a.id, "source_url": a.source_url, "score": round(score, 6)})
        if len(picked) == max_sentences: break
    if not picked:
        return _abstain(bundle, "no-topical-sentence", "The retrieved records do not mention the topic of the question.")
    cited = [_article(item["article"]) for item in bundle["evidence"] if any(p["citation_id"] == item["article"]["id"] for p in picked)]
    dates = sorted({a.date_label for a in cited})
    analysis = (f"The cited records are dated {', '.join(dates)}. This extractive mode quotes source sentences without further "
                "interpretation; a grounded language-model generator adds contextual analysis.")
    facts = "\n".join(f"- {p['text']} [{p['citation_id']}]" for p in picked)
    citations = [{"id": a.id, "label": a.citation, "url": a.source_url} for a in cited]
    return {"answer": f"Objective facts:\n{facts}\n\nContextual analysis:\n{analysis}", "facts": picked, "claims": picked, "analysis": analysis,
            "abstained": False, "abstain_reason": "", "citations": citations, "trace": bundle, "generation": "extractive-offline-baseline"}


def _grounding_prompt(bundle: dict[str, Any]) -> str:
    f = bundle["date_filter"]; blocks = []
    for item in bundle["evidence"]:
        a = _article(item["article"])
        flag = "\nNOTE: outside the requested date" if item.get("outside_date_filter") else ""
        blocks.append(f"SOURCE [{a.id}]\nDATE: {a.date_label}{f' | original: {a.date_original}' if a.date_original else ''}\n"
                      f"TITLE: {a.title}\nURL: {a.source_url}{flag}\nTEXT: {a.text}")
    date = "none" if f["year"] is None else "-".join(str(x).zfill(2) for x in (f["year"], f["month"], f["day"]) if x is not None) + (f" to {f['year_end']}" if f.get("year_end") else "")
    header = (f"QUESTION: {bundle['original_query']}\nREGENERATED QUERY: {bundle['rewritten_query']}\n"
              f"DATE FILTER: {date} (from {f.get('source', 'query')}{', widened to all records' if f.get('widened') else ''})\n")
    if bundle.get("date_conflict"):
        header += "NOTE: no record matched the requested date; the sources below come from other dates.\n"
    if not blocks:
        header += "NOTE: no source passed the date filter and similarity floor.\n"
    return header + "\n" + "\n\n".join(blocks)


_CITE = re.compile(r"\[([^\[\]\s]+(?:\s*[,;]\s*[^\[\]\s]+)*)\]")


def _adapter_result(text: str, bundle: dict[str, Any], label: str) -> dict[str, Any]:
    text = text.strip(); known = {item["article"]["id"]: _article(item["article"]) for item in bundle["evidence"]}
    cited_ids = [i for group in _CITE.findall(text) for i in re.split(r"\s*[,;]\s*", group)]
    citations = [{"id": a.id, "label": a.citation, "url": a.source_url} for i, a in known.items() if i in cited_ids]
    abstained = text.upper().startswith(ABSTAIN_MARKER[:-1])
    m = re.search(r"contextual analysis\s*:?\s*\n?(.*)", text, re.I | re.S)
    return {"answer": text, "facts": [], "claims": [], "analysis": m.group(1).strip() if m else "", "abstained": abstained,
            "abstain_reason": "model-abstained" if abstained else "", "citations": citations,
            "unknown_citations": sorted({i for i in cited_ids if i not in known}), "trace": bundle, "generation": label,
            "verification": "adapter output is unverified; inspect source IDs and evidence trace"}


class OpenAICompatibleGenerator(_ChatEndpoint):
    def generate(self, bundle: dict[str, Any]) -> dict[str, Any]:
        text = self._complete(self.prompts["generation"], _grounding_prompt(bundle), 300)
        style = "reasoning" if is_reasoning_style(self.model, self.chat_style) else "standard"
        result = _adapter_result(text, bundle, "openai-compatible-grounded-unverified")
        result["prompt_profile"] = self.profile; result["chat_style"] = style
        return result


class LocalCommandGenerator:
    def __init__(self, command: str, profile: str = "journal"): self.command = shlex.split(command); self.profile = profile

    def generate(self, bundle: dict[str, Any]) -> dict[str, Any]:
        payload = {"system_prompt": profile_prompts(self.profile)["generation"], "prompt": _grounding_prompt(bundle),
                   "query": bundle["original_query"], "evidence": bundle["evidence"], "profile": self.profile}
        p = subprocess.run(self.command, input=json.dumps(payload, ensure_ascii=False), text=True, capture_output=True, check=True, timeout=300)
        try:
            obj = json.loads(p.stdout); text = obj.get("answer", obj.get("text", ""))
        except json.JSONDecodeError: text = p.stdout
        result = _adapter_result(text, bundle, "local-command-grounded-unverified"); result["prompt_profile"] = self.profile
        return result
