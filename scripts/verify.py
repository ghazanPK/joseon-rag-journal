"""Build an index and answer a question with the offline RAG pipeline."""
import json
from pathlib import Path

from joseon_rag.core import RuleRegenerator, answer, build_index, load_articles, retrieve

ROOT = Path(__file__).parents[1]
OUT = ROOT / "outputs" / "verify"
OUT.mkdir(parents=True, exist_ok=True)
index = build_index(load_articles(ROOT / "examples" / "articles.jsonl"))
events = json.loads((ROOT / "examples" / "events.json").read_text(encoding="utf-8"))
query = "What scholarly policy was discussed in 1420?"
rewritten = RuleRegenerator(events).rewrite(query)
result = answer(retrieve(index, query, rewritten))
(OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "answer.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
if not result["citations"] or not result["trace"]["evidence"]:
    raise RuntimeError("verify question returned no cited evidence")
print(json.dumps({"articles": len(index["articles"]), "backend": index["backend"], "citations": [item["id"] for item in result["citations"]], "outputs": ["index.json", "answer.json"]}, indent=2))

