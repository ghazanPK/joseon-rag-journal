from __future__ import annotations

import csv
import json
import math
import os
import re
import shlex
import subprocess
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

WORD = re.compile(r"[0-9A-Za-z가-힣]+")
YEAR_VALUE = r"(?:1[3-8]\d{2}|19(?:0\d|10))"
ISO_DATE = re.compile(rf"(?<!\d)({YEAR_VALUE})[-/.년\s]+(1[0-2]|0?[1-9])(?:[-/.월\s]+(3[01]|[12]\d|0?[1-9]))?")
YEAR = re.compile(rf"(?<!\d)({YEAR_VALUE})(?:년)?(?!\d)")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in WORD.findall(text) if len(t) > 1]


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

    @property
    def citation(self) -> str:
        date = "-".join(str(x).zfill(2) for x in (self.year, self.month, self.day) if x is not None)
        return f"[{self.id}] {date or 'date unknown'} {self.title}".strip()


def _date_from(value: str) -> tuple[int | None, int | None, int | None]:
    m = ISO_DATE.search(value or "")
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else None
    m = YEAR.search(value or "")
    return (int(m.group(1)), None, None) if m else (None, None, None)


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
        y, m, d = _date_from(str(row.get("date") or row.get("title") or text[:120]))
        out.append(Article(str(row.get("id") or f"article-{i}"), text, str(row.get("title") or ""), int(row["year"]) if row.get("year") else y, int(row["month"]) if row.get("month") else m, int(row["day"]) if row.get("day") else d, str(row.get("source_url") or row.get("url") or ""), str(row.get("volume") or "")))
    return out


def build_index(articles: list[Article], backend: str = "tfidf", model_name: str | None = None) -> dict[str, Any]:
    if backend == "sentence-transformer":
        if not model_name: raise ValueError("sentence-transformer backend requires a model name or cached path")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("install the optional 'semantic' extra") from e
        model = SentenceTransformer(model_name, local_files_only=True)
        texts = [f"{a.title}\n{a.text}" for a in articles]
        vectors = model.encode(texts, normalize_embeddings=True).tolist()
        return {"schema": "paperreach.joseon.article-index.v1", "backend": "sentence-transformer-cache-only", "model": model_name, "articles": [asdict(a) for a in articles], "vectors": vectors}
    docs = [tokenize(f"{a.title} {a.text}") for a in articles]
    df: dict[str, int] = {}
    for doc in docs:
        for t in set(doc): df[t] = df.get(t, 0) + 1
    idf = {t: math.log((1 + len(docs)) / (1 + n)) + 1 for t, n in df.items()}
    vectors = []
    for doc in docs:
        counts: dict[str, int] = {}
        for t in doc: counts[t] = counts.get(t, 0) + 1
        vec = {t: round(c * idf[t], 6) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1
        vectors.append({t: round(v / norm, 6) for t, v in vec.items()})
    return {"schema": "paperreach.joseon.article-index.v1", "backend": "tfidf-baseline", "idf": idf, "articles": [asdict(a) for a in articles], "vectors": vectors}


class Regenerator(Protocol):
    def rewrite(self, query: str) -> str: ...


class RuleRegenerator:
    def __init__(self, events: dict[str, Any] | None = None): self.events = events or {}
    def rewrite(self, query: str) -> str:
        low = query.lower(); extra = []
        for name, info in self.events.items():
            terms = [name, *info.get("aliases", [])]
            if any(term.lower() in low for term in terms):
                extra.extend(str(x) for x in terms if x.lower() not in low)
                if not YEAR.search(query) and info.get("year"): extra.append(str(info["year"]))
        return query if not extra else f"{query} Historical context: {'; '.join(extra)}."


class OpenAICompatibleRegenerator:
    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY"):
        self.url = base_url.rstrip("/") + "/chat/completions"; self.model = model; self.key = os.getenv(api_key_env, "")
    def rewrite(self, query: str) -> str:
        body = json.dumps({"model": self.model, "messages": [{"role": "system", "content": "Rewrite the historical search query with explicit event terms and dates only when known. Return only the query."}, {"role": "user", "content": query}], "temperature": 0}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json", **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        with urllib.request.urlopen(req, timeout=60) as r: data = json.load(r)
        return data["choices"][0]["message"]["content"].strip()


class LocalCommandRegenerator:
    def __init__(self, command: str): self.command = shlex.split(command)
    def rewrite(self, query: str) -> str:
        p = subprocess.run(self.command, input=json.dumps({"query": query}), text=True, capture_output=True, check=True, timeout=60)
        try: return json.loads(p.stdout)["query"]
        except (json.JSONDecodeError, KeyError): return p.stdout.strip()


def query_date(query: str) -> tuple[int | None, int | None, int | None]:
    return _date_from(query)


def _cosine(query: str, index: dict[str, Any], vector: dict[str, float]) -> float:
    counts: dict[str, int] = {}
    for t in tokenize(query): counts[t] = counts.get(t, 0) + 1
    q = {t: c * index["idf"].get(t, 1.0) for t, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in q.values())) or 1
    return sum((v / norm) * vector.get(t, 0.0) for t, v in q.items())


def _dense_scores(query: str, index: dict[str, Any]) -> list[float]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError("querying this index requires the optional 'semantic' extra") from e
    model = SentenceTransformer(index["model"], local_files_only=True)
    q = model.encode([query], normalize_embeddings=True)[0]
    return [float(sum(float(a) * float(b) for a, b in zip(q, row))) for row in index["vectors"]]


def retrieve(index: dict[str, Any], original: str, rewritten: str, token_budget: int = 2400, min_filtered: int = 2) -> dict[str, Any]:
    explicit = query_date(original)
    inferred = query_date(rewritten)
    y, m, d = explicit if any(x is not None for x in explicit) else inferred
    rows = [Article(**a) for a in index["articles"]]
    def matches(a: Article) -> bool:
        return (y is None or a.year == y) and (m is None or a.month == m) and (d is None or a.day == d)
    candidates = [i for i, a in enumerate(rows) if matches(a)]
    filter_mode = "exact-or-partial-date" if any(x is not None for x in (y, m, d)) else "all-records"
    widened = False
    if len(candidates) < min_filtered and explicit == (None, None, None) and any(x is not None for x in (y, m, d)):
        candidates = list(range(len(rows))); widened = True; filter_mode = "inferred-date-fallback-all"
    qterms = set(tokenize(rewritten))
    scored = []
    dense = _dense_scores(rewritten, index) if index["backend"].startswith("sentence-transformer") else None
    for i in candidates:
        a = rows[i]; base = dense[i] if dense is not None else _cosine(rewritten, index, index["vectors"][i])
        title_overlap = len(qterms & set(tokenize(a.title))) / max(1, len(qterms))
        date_bonus = .08 if y and a.year == y else 0
        scored.append((base + .15 * title_overlap + date_bonus, base, a))
    scored.sort(key=lambda x: (-x[0], x[2].id))
    selected = []; used = 0
    for rerank, base, a in scored:
        cost = max(1, math.ceil(len(tokenize(a.text)) * 1.35))
        if used + cost > token_budget: continue
        selected.append({"article": asdict(a), "similarity": round(base, 6), "rerank_score": round(rerank, 6), "estimated_tokens": cost})
        used += cost
        if used >= token_budget: break
    return {"original_query": original, "rewritten_query": rewritten, "date_filter": {"year": y, "month": m, "day": d, "mode": filter_mode, "widened": widened}, "token_budget": token_budget, "estimated_tokens": used, "evidence": selected}


def answer(bundle: dict[str, Any], max_sentences: int = 6) -> dict[str, Any]:
    q = set(tokenize(bundle["rewritten_query"])); candidates = []
    for rank, item in enumerate(bundle["evidence"], 1):
        a = Article(**item["article"])
        for sentence in re.split(r"(?<=[.!?])\s+", a.text):
            terms = set(tokenize(sentence)); score = len(q & terms) / max(1, len(q)) + item["rerank_score"] * .35
            if sentence.strip(): candidates.append((score, rank, sentence.strip(), a))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    picked = []; seen = set()
    for score, rank, sentence, a in candidates:
        key = sentence.lower()
        if key in seen: continue
        seen.add(key); picked.append({"text": sentence, "citation_id": a.id, "source_url": a.source_url, "score": round(score, 6)})
        if len(picked) == max_sentences: break
    if not picked:
        text = "No supporting article was retrieved within the configured filter and token budget."
    else:
        text = " ".join(f"{p['text']} [{p['citation_id']}]" for p in picked)
    citations = []
    for item in bundle["evidence"]:
        a = Article(**item["article"])
        if any(p["citation_id"] == a.id for p in picked): citations.append({"id": a.id, "label": a.citation, "url": a.source_url})
    return {"answer": text, "claims": picked, "citations": citations, "trace": bundle, "generation": "extractive-offline-baseline"}


def _grounding_prompt(bundle: dict[str, Any]) -> str:
    blocks = []
    for item in bundle["evidence"]:
        a = Article(**item["article"])
        blocks.append(f"SOURCE [{a.id}]\nDATE: {a.year}-{a.month or '?'}-{a.day or '?'}\nTITLE: {a.title}\nURL: {a.source_url}\nTEXT: {a.text}")
    return f"Answer the query using only the sources below. Cite each factual claim with its exact [source-id]. If evidence is insufficient, say so.\n\nQUERY: {bundle['original_query']}\n\n" + "\n\n".join(blocks)


def _adapter_result(text: str, bundle: dict[str, Any], label: str) -> dict[str, Any]:
    citations = []
    for item in bundle["evidence"]:
        a = Article(**item["article"])
        if f"[{a.id}]" in text: citations.append({"id": a.id, "label": a.citation, "url": a.source_url})
    return {"answer": text.strip(), "claims": [], "citations": citations, "trace": bundle, "generation": label, "verification": "adapter output is unverified; inspect source IDs and evidence trace"}


class OpenAICompatibleGenerator:
    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY"):
        self.url = base_url.rstrip("/") + "/chat/completions"; self.model = model; self.key = os.getenv(api_key_env, "")
    def generate(self, bundle: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps({"model": self.model, "messages": [{"role": "system", "content": "You are an evidence-bound historical assistant. Never use unsupported facts."}, {"role": "user", "content": _grounding_prompt(bundle)}], "temperature": 0}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json", **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        with urllib.request.urlopen(req, timeout=120) as r: data = json.load(r)
        return _adapter_result(data["choices"][0]["message"]["content"], bundle, "openai-compatible-grounded-unverified")


class LocalCommandGenerator:
    def __init__(self, command: str): self.command = shlex.split(command)
    def generate(self, bundle: dict[str, Any]) -> dict[str, Any]:
        payload = {"prompt": _grounding_prompt(bundle), "query": bundle["original_query"], "evidence": bundle["evidence"]}
        p = subprocess.run(self.command, input=json.dumps(payload, ensure_ascii=False), text=True, capture_output=True, check=True, timeout=120)
        try:
            obj = json.loads(p.stdout); text = obj.get("answer", obj.get("text", ""))
        except json.JSONDecodeError: text = p.stdout
        return _adapter_result(text, bundle, "local-command-grounded-unverified")
