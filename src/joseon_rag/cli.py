from __future__ import annotations
import argparse, json
from pathlib import Path
from .core import LocalCommandGenerator, LocalCommandRegenerator, OpenAICompatibleGenerator, OpenAICompatibleRegenerator, RuleRegenerator, answer, build_index, load_articles, retrieve

def main() -> None:
    p = argparse.ArgumentParser(description="Article-preserving, evidence-first historical RAG")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("corpus", type=Path); b.add_argument("--out", type=Path, required=True); b.add_argument("--backend", choices=["tfidf", "sentence-transformer"], default="tfidf"); b.add_argument("--model", help="local cached model name/path; network downloads are disabled")
    q = sub.add_parser("ask"); q.add_argument("index", type=Path); q.add_argument("query"); q.add_argument("--out", type=Path); q.add_argument("--events", type=Path); q.add_argument("--token-budget", type=int, default=2400); q.add_argument("--regenerator", choices=["rule", "none", "endpoint", "command"], default="rule"); q.add_argument("--generator", choices=["extractive", "endpoint", "command"], default="extractive"); q.add_argument("--base-url"); q.add_argument("--model"); q.add_argument("--command"); q.add_argument("--generator-command")
    a = p.parse_args()
    if a.cmd == "build":
        idx = build_index(load_articles(a.corpus), a.backend, a.model); a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8"); print(json.dumps({"articles": len(idx["articles"]), "backend": idx["backend"], "out": str(a.out)}, indent=2)); return
    idx = json.loads(a.index.read_text(encoding="utf-8")); events = json.loads(a.events.read_text(encoding="utf-8")) if a.events else {}
    if a.regenerator == "endpoint":
        if not a.base_url or not a.model: p.error("endpoint requires --base-url and --model")
        reg = OpenAICompatibleRegenerator(a.base_url, a.model)
    elif a.regenerator == "command":
        if not a.command: p.error("command requires --command")
        reg = LocalCommandRegenerator(a.command)
    else: reg = RuleRegenerator(events)
    rewritten = a.query if a.regenerator == "none" else reg.rewrite(a.query)
    bundle = retrieve(idx, a.query, rewritten, a.token_budget)
    if a.generator == "endpoint":
        if not a.base_url or not a.model: p.error("endpoint generator requires --base-url and --model")
        result = OpenAICompatibleGenerator(a.base_url, a.model).generate(bundle)
    elif a.generator == "command":
        if not a.generator_command: p.error("command generator requires --generator-command")
        result = LocalCommandGenerator(a.generator_command).generate(bundle)
    else: result = answer(bundle)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if a.out: a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(rendered, encoding="utf-8")
    print(rendered)

if __name__ == "__main__": main()
