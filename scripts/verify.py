"""Build an index, answer a question and an erroneous question with the offline RAG pipeline."""
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
erroneous = "What happened at the steam railway opening in 1420?"
abstention = answer(retrieve(index, erroneous, RuleRegenerator(events).rewrite(erroneous)))
wrong_date = "Describe the new script for the people promulgated in 1420."
conflict = answer(retrieve(index, wrong_date, RuleRegenerator(events).rewrite(wrong_date)))
abstention = {"nonexistent_event": abstention, "date_contradiction": conflict}
(OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "answer.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "abstention.json").write_text(json.dumps(abstention, ensure_ascii=False, indent=2), encoding="utf-8")
if not result["citations"] or not result["trace"]["evidence"] or result["abstained"]:
    raise RuntimeError("verify question returned no cited evidence")
if [a["abstain_reason"] for a in abstention.values()] != ["no-evidence", "date-conflict"]:
    raise RuntimeError("erroneous questions were answered instead of abstaining")
print(json.dumps({"articles": len(index["articles"]), "backend": index["backend"], "tokenizer": index["tokenizer"],
                  "citations": [item["id"] for item in result["citations"]],
                  "erroneous_questions": {k: v["abstain_reason"] for k, v in abstention.items()},
                  "outputs": ["index.json", "answer.json", "abstention.json"]}, indent=2))
