from __future__ import annotations
import argparse, json
from pathlib import Path
from .core import (CHAT_STYLES, PROMPT_PROFILES, LocalCommandGenerator, LocalCommandRegenerator, OpenAICompatibleGenerator,
                   OpenAICompatibleRegenerator, RuleRegenerator, answer, build_index, load_articles, retrieve)
from .importer import import_page

PROFILE = "journal"


def _llm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-url", default=""); p.add_argument("--model", default="")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable holding the endpoint key")
    p.add_argument("--chat-style", choices=CHAT_STYLES, default="auto",
                   help="reasoning = o1-compatible request (developer role, no temperature); auto detects o1/o3-style model names")
    p.add_argument("--max-output-tokens", type=int, help="max_tokens / max_completion_tokens for the endpoint")
    p.add_argument("--profile", choices=sorted(PROMPT_PROFILES), default=PROFILE, help="paper prompt profile")


def main() -> None:
    p = argparse.ArgumentParser(description="Article-preserving, evidence-first historical RAG")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("corpus", type=Path); b.add_argument("--out", type=Path, required=True); b.add_argument("--backend", choices=["tfidf", "sentence-transformer"], default="tfidf"); b.add_argument("--model", help="local cached model name/path; network downloads are disabled")
    imp = sub.add_parser("import-html"); imp.add_argument("url"); imp.add_argument("--file", type=Path); imp.add_argument("--id", required=True); imp.add_argument("--date", required=True); imp.add_argument("--title", default=""); imp.add_argument("--volume", default=""); imp.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("crawl", help="rate-limited, resumable Sejong Annals importer (sillok.history.go.kr)")
    c.add_argument("--sample", action="store_true", help="small test crawl: one lunar month (default Sejong year 2, month 5), "
                   "at most 100 articles, default --out data/sejong-sample.jsonl; reruns read the cache")
    c.add_argument("--out", type=Path, help="JSONL corpus to append to (keep under ignored data/); required without --sample")
    c.add_argument("--cache", type=Path, default=Path("data/sillok-cache"), help="raw page cache used for resuming")
    c.add_argument("--years", help="Sejong reign years, e.g. '2' or '0-5,28' (0 = accession year 1418)")
    c.add_argument("--month", type=int, help="only this lunar month (1-12) of the selected years")
    c.add_argument("--leap", action="store_true", help="with --month: the leap month (윤달)")
    c.add_argument("--max-articles", type=int, help="stop after writing this many new articles (with --sample: sample size, at most 100)")
    c.add_argument("--delay", type=float, default=1.5, help="seconds between network requests (minimum 1.0; 1.5 with --sample)")
    c.add_argument("--include-hanja", action="store_true", help="also store the classical-Chinese original as hanja_text")
    c.add_argument("--offline", action="store_true", help="parse cached pages only; never touch the network")
    serve = sub.add_parser("serve"); serve.add_argument("index", type=Path); serve.add_argument("--events", type=Path); _llm_args(serve); serve.add_argument("--host", default="127.0.0.1"); serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--question", default="", help="question the browser pre-fills instead of the bundled example question")
    q = sub.add_parser("ask"); q.add_argument("index", type=Path); q.add_argument("query"); q.add_argument("--out", type=Path); q.add_argument("--events", type=Path); q.add_argument("--token-budget", type=int, default=2400)
    q.add_argument("--tokenizer", choices=["auto", "tiktoken", "chars"], default="auto", help="token counter for the evidence budget")
    q.add_argument("--min-similarity", type=float, help="similarity floor (default 0.05 TF-IDF, 0.30 dense)")
    q.add_argument("--regenerator", choices=["rule", "none", "endpoint", "command"], default="rule"); q.add_argument("--generator", choices=["extractive", "endpoint", "command"], default="extractive"); _llm_args(q); q.add_argument("--command"); q.add_argument("--generator-command")
    a = p.parse_args()
    if a.cmd == "import-html":
        row = import_page(a.url, a.out, source_file=a.file, article_id=a.id, date=a.date, title=a.title, volume=a.volume)
        print(json.dumps({"id": row["id"], "characters": len(row["text"]), "source_url": row["source_url"], "out": str(a.out)}, ensure_ascii=False)); return
    if a.cmd == "crawl":
        from .crawler import SAMPLE_OUT, crawl, sample_plan
        try:
            plan = (sample_plan(years=a.years, month=a.month, leap=a.leap, max_articles=a.max_articles, delay=a.delay) if a.sample
                    else dict(years=a.years, month=a.month, leap=a.leap, max_articles=a.max_articles, delay=a.delay))
        except ValueError as exc:
            p.error(str(exc))
        out = a.out or (SAMPLE_OUT if a.sample else None)
        if out is None: p.error("crawl requires --out (or --sample)")
        summary = crawl(out, a.cache, include_hanja=a.include_hanja, offline=a.offline, **plan)
        print(json.dumps(summary, ensure_ascii=False, indent=2)); return
    if a.cmd == "serve":
        from http.server import ThreadingHTTPServer
        from .server import create_handler
        server = ThreadingHTTPServer((a.host, a.port), create_handler(a.index, a.events, a.base_url, a.model, chat_style=a.chat_style, profile=a.profile, api_key_env=a.api_key_env, max_output_tokens=a.max_output_tokens, suggested_query=a.question))
        print(f"Open http://{a.host}:{a.port}"); server.serve_forever(); return
    if a.cmd == "build":
        idx = build_index(load_articles(a.corpus), a.backend, a.model); a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8"); print(json.dumps({"articles": len(idx["articles"]), "backend": idx["backend"], "out": str(a.out)}, indent=2)); return
    idx = json.loads(a.index.read_text(encoding="utf-8")); events = json.loads(a.events.read_text(encoding="utf-8")) if a.events else {}
    llm = dict(api_key_env=a.api_key_env, profile=a.profile, chat_style=a.chat_style, max_output_tokens=a.max_output_tokens)
    if a.regenerator == "endpoint":
        if not a.base_url or not a.model: p.error("endpoint requires --base-url and --model")
        reg = OpenAICompatibleRegenerator(a.base_url, a.model, **llm)
    elif a.regenerator == "command":
        if not a.command: p.error("command requires --command")
        reg = LocalCommandRegenerator(a.command, a.profile)
    else: reg = RuleRegenerator(events)
    rewritten = a.query if a.regenerator == "none" else reg.rewrite(a.query)
    bundle = retrieve(idx, a.query, rewritten, a.token_budget, min_similarity=a.min_similarity, tokenizer=a.tokenizer)
    if a.generator == "endpoint":
        if not a.base_url or not a.model: p.error("endpoint generator requires --base-url and --model")
        result = OpenAICompatibleGenerator(a.base_url, a.model, **llm).generate(bundle)
    elif a.generator == "command":
        if not a.generator_command: p.error("command generator requires --generator-command")
        result = LocalCommandGenerator(a.generator_command, a.profile).generate(bundle)
    else: result = answer(bundle)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if a.out: a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(rendered, encoding="utf-8")
    print(rendered)

if __name__ == "__main__": main()
