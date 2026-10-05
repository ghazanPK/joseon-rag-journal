"""Retrieval, abstention and LLM-adapter tests with stub endpoints (no network)."""
import json
import unittest

from joseon_rag.core import (ABSTAIN_MARKER, PROMPT_PROFILES, Article, OpenAICompatibleGenerator, OpenAICompatibleRegenerator,
                             RuleRegenerator, answer, build_index, chat_payload, conservative_token_estimate, retrieve, tokenize)


class StubTransport:
    def __init__(self, reply): self.reply = reply; self.calls = []

    def __call__(self, url, payload, headers, timeout):
        self.calls.append((url, payload, headers)); return {"choices": [{"message": {"content": self.reply}}]}


def corpus():
    return build_index([
        Article("hall-1", "집현전의 학사들이 경연에서 옛 제도를 논의하였다. 임금이 학문을 장려하였다.", "집현전 학사 논의", 1420, 5, 12, "https://example.org/1", calendar="lunar"),
        Article("hall-2", "Scholars of the Hall of Worthies were appointed to royal study.", "Appointment of scholars", 1420, 5, 14, "https://example.org/2"),
        Article("script", "The new script was promulgated for the people.", "Promulgation", 1446, 9, 29, "https://example.org/3"),
    ])


class RetrievalTest(unittest.TestCase):
    def test_korean_bigrams_match_across_particles(self):
        self.assertIn("집현", tokenize("집현전의 학사"))
        bundle = retrieve(corpus(), "집현전 학사", "집현전 학사", 2000)
        self.assertEqual(bundle["evidence"][0]["article"]["id"], "hall-1")
        self.assertGreater(bundle["evidence"][0]["similarity"], 0.1)

    def test_hanja_tokens(self):
        self.assertEqual(tokenize("集賢殿"), ["集賢", "賢殿"])
        self.assertEqual(tokenize("王"), ["王"])

    def test_fit_or_stop_never_skips_a_higher_ranked_article(self):
        long_text = "scholars hall lecture. " * 150
        idx = build_index([Article("long", long_text, "Scholars hall", 1420), Article("short", "scholars", "Note", 1420)])
        bundle = retrieve(idx, "scholars hall in 1420", "scholars hall in 1420", 300)
        self.assertEqual(bundle["evidence"], [])
        self.assertEqual(bundle["packing"], {"policy": "fit-or-stop", "stopped_at": "long"})
        self.assertTrue(answer(bundle)["abstained"])

    def test_conservative_estimate_counts_korean_and_hanja(self):
        self.assertEqual(conservative_token_estimate("집현전"), 3)
        self.assertEqual(conservative_token_estimate("集賢殿"), 6)
        self.assertGreaterEqual(conservative_token_estimate("word " * 30), 40)

    def test_similarity_floor_excludes_unrelated_articles(self):
        bundle = retrieve(corpus(), "Tell me about the moon landing", "Tell me about the moon landing", 2000)
        self.assertEqual(bundle["evidence"], [])
        result = answer(bundle)
        self.assertTrue(result["abstained"]); self.assertTrue(result["answer"].startswith(ABSTAIN_MARKER))
        self.assertEqual(result["abstain_reason"], "no-evidence")


class ExtractiveAnswerTest(unittest.TestCase):
    def test_cited_facts_and_context_section(self):
        q = "What happened to the scholars of the Hall of Worthies?"
        result = answer(retrieve(corpus(), q, RuleRegenerator({"hall of worthies": {"year": 1420, "aliases": ["jiphyeonjeon"]}}).rewrite(q), 2000))
        self.assertFalse(result["abstained"])
        self.assertIn("Objective facts:", result["answer"]); self.assertIn("Contextual analysis:", result["answer"])
        self.assertIn("[hall-2]", result["answer"])
        self.assertNotIn("script", [c["id"] for c in result["citations"]])

    def test_date_contradicting_evidence_abstains_and_names_the_record_date(self):
        result = answer(retrieve(corpus(), "Why was the new script promulgated in 1420?", "Why was the new script promulgated in 1420?", 2000))
        self.assertTrue(result["abstained"]); self.assertEqual(result["abstain_reason"], "date-conflict")
        self.assertIn("1446-09-29", result["answer"]); self.assertIn("[script]", result["answer"])
        self.assertTrue(result["trace"]["evidence"][0]["outside_date_filter"])

    def test_rule_regenerator_adds_details_and_approximate_date(self):
        events = {"hunminjeongeum promulgation": {"year": 1446, "month": 9, "aliases": ["new script"], "details": ["correct sounds"]}}
        text = RuleRegenerator(events).rewrite("Tell me about the court's reaction to the new script")
        self.assertIn("approximate date: 1446-09", text); self.assertIn("correct sounds", text)
        self.assertNotIn("approximate date", RuleRegenerator(events).rewrite("the new script in 1445"))


class EndpointAdapterTest(unittest.TestCase):
    def bundle(self):
        return retrieve(corpus(), "What happened to the scholars in 1420?", "What happened to the scholars in 1420?", 2000)

    def test_generation_prompt_and_reasoning_payload(self):
        stub = StubTransport("Objective facts:\nScholars were appointed [hall-2].\n\nContextual analysis:\nThis shows royal support [hall-2] [ghost].")
        gen = OpenAICompatibleGenerator("http://local/v1", "o1", profile="journal", chat_style="auto", max_output_tokens=900, transport=stub)
        result = gen.generate(self.bundle())
        url, payload, _ = stub.calls[0]
        self.assertEqual(url, "http://local/v1/chat/completions")
        self.assertEqual([m["role"] for m in payload["messages"]], ["developer", "user"])
        self.assertNotIn("temperature", payload); self.assertEqual(payload["max_completion_tokens"], 900)
        instructions = payload["messages"][0]["content"]
        for phrase in ("select only", "Objective facts", "Contextual analysis", "INSUFFICIENT EVIDENCE", "contradicts"):
            self.assertIn(phrase, instructions)
        self.assertIn("SOURCE [hall-2]", payload["messages"][1]["content"])
        self.assertEqual([c["id"] for c in result["citations"]], ["hall-2"])
        self.assertEqual(result["unknown_citations"], ["ghost"]); self.assertEqual(result["analysis"], "This shows royal support [hall-2] [ghost].")
        self.assertEqual(result["chat_style"], "reasoning"); self.assertFalse(result["abstained"])

    def test_standard_payload_and_model_abstention(self):
        stub = StubTransport("INSUFFICIENT EVIDENCE: the records do not mention a moon landing.")
        result = OpenAICompatibleGenerator("http://local/v1", "gpt-4o", profile="ismar", transport=stub).generate(self.bundle())
        payload = stub.calls[0][1]
        self.assertEqual(payload["messages"][0]["role"], "system"); self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["messages"][0]["content"], PROMPT_PROFILES["ismar"]["generation"])
        self.assertTrue(result["abstained"])
        forced = chat_payload("my-local-model", "x", "y", chat_style="reasoning")
        self.assertEqual(forced["messages"][0]["role"], "developer"); self.assertNotIn("temperature", forced)

    def test_regeneration_prompt_requests_details_and_approximate_date(self):
        stub = StubTransport("Court reaction to Hunminjeongeum promulgation approximate date: 1446-09")
        reg = OpenAICompatibleRegenerator("http://local/v1", "o3-mini", profile="journal", transport=stub)
        self.assertEqual(reg.rewrite("court reaction to Hunminjeongeum"), "Court reaction to Hunminjeongeum promulgation approximate date: 1446-09")
        prompt = stub.calls[0][1]["messages"][0]["content"]
        self.assertIn("approximate date", prompt); self.assertIn("keep it exactly as written", prompt)
        bundle = retrieve(corpus(), "court reaction to Hunminjeongeum", reg.rewrite("court reaction to Hunminjeongeum"), 2000)
        self.assertEqual((bundle["date_filter"]["year"], bundle["date_filter"]["month"]), (1446, 9))

    def test_invalid_profile_and_style_rejected(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleGenerator("http://x", "m", profile="other")
        with self.assertRaises(ValueError):
            OpenAICompatibleGenerator("http://x", "m", chat_style="fast")


if __name__ == "__main__":
    unittest.main()
