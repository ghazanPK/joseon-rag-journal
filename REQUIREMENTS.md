# Requirements and provenance

## Paper-supported facts

The complete journal paper was read before this requirements file was written. It describes a Sejong Annals pipeline with 30,949 article-level chunks from 163 volumes (1418–1450), article-preserving dynamic chunking, year/month/day metadata, GPT-based query regeneration that adds context and approximate dates, exact or partial date metadata filtering, similarity search, and retrieval of ranked whole articles until a model-specific maximum token threshold is reached. The answer model selects necessary information from that context and returns sourced factual information plus contextual interpretation. The paper reports a 30-question manually grounded benchmark with erroneous questions and factual-accuracy, reliability, and reasonableness scores.

The paper identifies risks: whole articles can mix topics and dilute embeddings, query regeneration can infer a wrong date, event dates can differ from mention dates, processing adds latency, evaluation is manually scored, and broader/multimodal sources remain future work.

## Functional requirements

The public implementation must:

- crawl the Sejong Annals politely and resumably, or ingest user-prepared JSONL/CSV records;
- preserve one complete source article per chunk;
- normalize year/month/day (lunar, with the original date string) and stable source metadata;
- build an offline searchable index;
- optionally regenerate a query through a rule, local-command or HTTP adapter;
- filter exact, partial and range dates without silently excluding fallback evidence for inferred dates;
- rerank candidates and fill a configurable token budget with whole articles;
- abstain when evidence is missing or contradicts the question's date;
- emit an evidence-grounded answer with stable citations: cited facts plus a contextual-analysis section.

Intermediate query, filter, similarity floor, packing, retrieval score and evidence data must be auditable.

## Assumptions and substitutions

This is independent educational code, not the institute implementation. Korean-aware TF-IDF retrieval (Latin words plus Hangul/Hanja character bigrams) and an extractive sentence selector are explicit offline baselines. They do not reproduce the paper's embedding service or GPT-o1 answer model.

The paper does not specify the following, so this project makes its own documented choices:

- **Prompts.** The prompt profiles were written from the paper's description: select necessary information from the long context, cite objective facts, add contextual analysis, abstain for erroneous questions, and regenerate queries with event details and approximate dates.
- **Crawler.** The selectors follow the site's markup observed on 6 October 2026.
- **Small sample.** The paper used the whole Sejong reign. For testing, `joseon-rag crawl --sample` fetches one lunar month (Sejong year 2, month 5; 87 articles in 165 s on 6 October 2026). It holds at most 100 records and runs at no less than 1.5 s per request. A cached rerun makes no request. `scripts/start_demo.py --sample-crawl` serves the sample; by default the demo stays on the authored offline examples. A sample answer illustrates the pipeline and is not a benchmark result. Crawled text stays under ignored `data/` and is not redistributed. The full crawl is optional.
- **Calendar.** Dates keep the Annals' lunar month and day, with year = 1418 + reign year. No Julian or Gregorian conversion is applied.
- **Tokenizer.** `tiktoken` is used when installed; otherwise a conservative character estimate.
- **Similarity floors.** The defaults are 0.05 for TF-IDF and 0.30 for dense indexes.
- **Packing.** Packing is fit-or-stop, which preserves rank order at the cost of unused budget.
- **Reranking and fallback.** Reranking combines lexical, date and title signals. Inferred-date filtering widens to all records when it matches too few.
- **Embedding model, vector database and model version** are not reproduced.

Optional OpenAI-compatible HTTP (including an o1-compatible `developer`-role request without `temperature`) and local-command adapters are generic user-configured interfaces. No claims are made that baseline scores reproduce the paper.

