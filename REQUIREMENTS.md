# Requirements and provenance

## Paper-supported facts

The complete journal paper was read before this requirements file was written. It describes a Sejong Annals pipeline with 30,949 article-level chunks from 163 volumes (1418–1450), article-preserving dynamic chunking, year/month/day metadata, GPT-based query regeneration that adds context and approximate dates, exact or partial date metadata filtering, similarity search, and retrieval of ranked whole articles until a model-specific maximum token threshold is reached. The answer model selects necessary information from that context and returns sourced factual information plus contextual interpretation. The paper reports a 30-question manually grounded benchmark with erroneous questions and factual-accuracy, reliability, and reasonableness scores.

The paper identifies risks: whole articles can mix topics and dilute embeddings, query regeneration can infer a wrong date, event dates can differ from mention dates, processing adds latency, evaluation is manually scored, and broader/multimodal sources remain future work.

## Functional requirements

The public implementation must ingest user-downloaded JSONL or CSV records, preserve one complete source article per chunk, normalize year/month/day and stable source metadata, build an offline searchable index, optionally regenerate a query through a rule/local/HTTP adapter, filter exact and partial dates without silently excluding fallback evidence, rerank candidates, fill a configurable token budget, and emit an extractive evidence-grounded answer with stable citations. Intermediate query, filter, retrieval score, and evidence data must be auditable.

## Assumptions and substitutions

This is independent educational code, not the institute implementation. TF-IDF cosine retrieval and an extractive sentence selector are explicit offline baselines, not reproductions of the paper's embedding service or GPT-o1 answer model. The paper does not specify prompts, crawler selectors, embedding model/version, vector database, similarity thresholds, tokenizer, or exact reranker; this project therefore uses documented schemas, approximate wordpiece budgeting, lexical/date/source reranking, and widening fallback when inferred-date filtering produces too few results. Optional OpenAI-compatible HTTP and local-command adapters are generic user-configured interfaces. No claims are made that baseline scores reproduce the paper.

