# A Retrieval‐Augmented Generation System for Accurate and Contextual Historical Analysis: AI‐Agent for the Annals of the Joseon Dynasty

**Jeong Ha Lee, Ghazanfar Ali, Jae‐In Hwang**

**Computer Animation and Virtual Worlds · 2025** · Published

[Paper / publisher](https://doi.org/10.1002/cav.70048) · [Project page](https://ghazanfarali.com/research/joseon-rag-journal/) · [BibTeX](citation.bib) · [Requirements](REQUIREMENTS.md) · [Code & setup](#implementation-and-usage)

> Article-aware retrieval grounds answers in the Annals of the Joseon Dynasty.

![Method diagram from Figure 1 of the joseon-rag-journal paper](paper-assets/method.png)

*Original method figure from the paper: Figure 1, PDF page 2. Extracted for this research introduction; the diagram describes the original system, not verification of this reimplementation.*

## Why this research

Dense historical records are difficult to search and interpret reliably. Preserving article boundaries and temporal context helps a language model answer with traceable historical evidence.

The agent preserves historical article boundaries, uses date metadata and query refinement to retrieve evidence, and supplies that evidence to a language model. Its purpose is to support factual answers and contextual interpretation with traceable sources.

## Method at a glance

**Historical question** → **Date-aware evidence retrieval** → **Answer + source citations**

| | Research system |
|---|---|
| Input | A historical question and the Annals corpus |
| Method | Article-aware chunking, date-aware retrieval, query refinement, and LLM response generation |
| Output | Source-grounded answers and contextual analysis |

## Evidence and scope

Paper reports improvements of approximately 23–50 points on its 100-point evaluation scale over the compared systems

**Attribution:** These findings describe the paper or manuscript, not results obtained with this repository's code.

**Study context:** Annals of the Joseon Dynasty.

**Limitations:** Evidence comes from the paper’s historical task and benchmark; it does not guarantee factual correctness for arbitrary questions.

## Explore the implementation

Article-preserving ingestion, date filtering, query rewriting, optional semantic retrieval, evidence traces and extractive or optional grounded LLM answers.

This repository contains independently written research code. The institute's original source, datasets and trained models are not distributed. Public-data preparation, commands, assumptions and checks are documented below and in [REQUIREMENTS.md](REQUIREMENTS.md).

## Resources and citation

Read the paper through its [publisher record](https://doi.org/10.1002/cav.70048). PDFs are hosted by publishers or preprint archives rather than stored in this repository.

Please cite the research paper when using its ideas; [download the BibTeX citation](citation.bib). The implementation has its own documented scope.

## Implementation and usage

<!-- implementation-guide -->

This repository implements an evidence-first pipeline for dense chronological records: one complete source article per chunk, date metadata, auditable query regeneration, exact/partial date filters with safe widening for inferred dates, TF-IDF retrieval, a transparent reranker, maximum-token retrieval, extractive answers, and source citations.

**Citation.** Jeong Ha Lee, Ghazanfar Ali, and Jae-In Hwang. “A Retrieval-Augmented Generation System for Accurate and Contextual Historical Analysis: AI-Agent for the Annals of the Joseon Dynasty.” *Computer Animation and Virtual Worlds* (2025), e70048. [https://doi.org/10.1002/cav.70048](https://doi.org/10.1002/cav.70048). Status: published.

This is an independent educational implementation. It is not the institute implementation and does not ship the Annals corpus, crawler output, API credentials, embedding cache, model weights, benchmark, or prompts.

### Run the offline example

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python scripts/smoke.py
```

The JSON answer includes every selected claim, citation ID/URL, rewritten query, filter behavior, retrieval/reranking scores, estimated token use, and the complete evidence trace. The included corpus is an authored artificial schema fixture, not a transcription of the Annals and not suitable for historical claims.

### Bring your own public corpus

Download records through the official National Institute of Korean History service: [Annals of the Joseon Dynasty](https://sillok.history.go.kr/). Follow the site's current access terms and robots/API guidance. Export one article per JSONL row (or CSV row) with:

```json
{"id":"stable-id","date":"YYYY-MM-DD","title":"article title","text":"complete article text","source_url":"https://...","volume":"optional"}
```

`year`, `month`, and `day` columns may replace `date`. Keep downloads under ignored `data/`. This code does not crawl the site because endpoint structure and access policy can change.

After exporting real articles with that contract, replace `examples/articles.jsonl` in the CLI command: `joseon-rag build data/articles.jsonl --out outputs/my.index.json`, then query it with `joseon-rag ask outputs/my.index.json "your question" --events examples/events.json --out outputs/my-answer.json`. The smoke script uses the same loader, indexer, retrieval, answer, and citation paths.

The default index is an honestly labeled TF-IDF baseline and requires no model. For a sentence-transformer already present in your cache, run `python -m pip install -e ".[semantic]"` and `joseon-rag build data/articles.jsonl --out outputs/semantic.index.json --backend sentence-transformer --model path-or-cached-model-name`. The loader uses `local_files_only=True` and fails rather than downloading weights. The CLI also supports a user-run OpenAI-compatible endpoint (`--regenerator endpoint --base-url ... --model ...`) or a local JSON-in/query-out command adapter. API use, models, and costs remain the user's responsibility.

Extractive generation is the default. To call a user-operated compatible model with the retrieved, source-labeled evidence, add `--generator endpoint --base-url http://localhost:8000/v1 --model your-model`. A local process can be used with `--generator command --generator-command "your-program --json"`; it receives JSON containing the grounded prompt and evidence. These adapter outputs are marked unverified in the result. Always inspect their cited source IDs and evidence trace.

This baseline cannot reproduce the reported evaluation scores. It does not know that an inferred date is historically correct, and the primary source may contain variant or conflicting accounts. Inspect citations and the trace. See [REQUIREMENTS.md](REQUIREMENTS.md) for the paper/assumption boundary.
