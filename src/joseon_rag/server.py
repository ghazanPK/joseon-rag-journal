"""Local browser inspector for the article-preserving retrieval pipeline."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .core import (OpenAICompatibleGenerator, OpenAICompatibleRegenerator,
                   RuleRegenerator, answer, retrieve)
from .speech_backend import SpeechBackend, speech_route


def create_handler(index_path: Path, events_path: Path | None = None,
                   base_url: str = "", model: str = "", embodied: bool = False, *,
                   chat_style: str = "auto", profile: str = "journal", api_key_env: str = "OPENAI_API_KEY",
                   max_output_tokens: int | None = None, suggested_query: str = ""):
    index = json.loads(index_path.read_text(encoding="utf-8"))
    events = json.loads(events_path.read_text(encoding="utf-8")) if events_path else {}
    llm = dict(api_key_env=api_key_env, profile=profile, chat_style=chat_style, max_output_tokens=max_output_tokens)
    page_path = Path(__file__).parents[2] / "static" / "index.html"
    static = Path(__file__).parents[2] / "static"
    speech = SpeechBackend()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                page = page_path.read_bytes()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page))); self.end_headers(); self.wfile.write(page)
            elif self.path == "/api/corpus":
                self._json({"articles": len(index["articles"]), "backend": index["backend"],
                            "collection_label": "Authored demonstration records" if index["articles"] and all(a["title"].startswith("Authored example:") for a in index["articles"]) else "Article records",
                            "sources": [{"id": a["id"], "title": a["title"], "url": a["source_url"]} for a in index["articles"]],
                            "suggested_query": suggested_query, "speech": speech.status()})
            elif self.path in {"/static/avatar.js", "/static/speech.js", "/static/vendor/three.module.js", "/static/vendor/three.core.js"}:
                asset = static / self.path.removeprefix("/static/")
                if not asset.is_file(): self.send_error(404); return
                data = asset.read_bytes(); self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            else:
                self.send_error(404)

        def do_POST(self):
            if speech_route(self, speech): return
            if self.path != "/api/ask":
                self.send_error(404); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 100_000:
                    raise ValueError("invalid request size")
                data = json.loads(self.rfile.read(length))
                query = str(data.get("query", "")).strip()
                if not query or len(query) > 1000:
                    raise ValueError("query must contain 1–1000 characters")
                budget = int(data.get("token_budget", 2400))
                if not 50 <= budget <= 16000:
                    raise ValueError("token budget must be 50–16000")
                rewrite = data.get("rewrite", "rule")
                if rewrite not in {"rule", "none", "endpoint"}:
                    raise ValueError("unsupported rewrite mode")
                rewritten = query if rewrite == "none" else (
                    OpenAICompatibleRegenerator(base_url, model, **llm).rewrite(query) if rewrite == "endpoint" and base_url and model
                    else RuleRegenerator(events).rewrite(query))
                if rewrite == "endpoint" and not (base_url and model):
                    raise ValueError("configure --base-url and --model to use endpoint rewriting")
                bundle = retrieve(index, query, rewritten, budget)
                generation = data.get("generation", "extractive")
                if generation == "endpoint" and base_url and model:
                    result = OpenAICompatibleGenerator(base_url, model, **llm).generate(bundle)
                elif generation == "extractive":
                    result = answer(bundle)
                else:
                    raise ValueError("configure --base-url and --model to use endpoint generation")
                if embodied:
                    from .agent import agent_events
                    result["agent"] = agent_events(result)
                self._json(result)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)
            except Exception as exc:
                self._json({"error": f"query failed: {exc}"}, 502)

        def _json(self, value, status=200):
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Serve the local historical evidence inspector")
    parser.add_argument("index", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--chat-style", choices=["auto", "standard", "reasoning"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), create_handler(args.index, args.events, args.base_url, args.model, chat_style=args.chat_style))
    print(f"Open http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
