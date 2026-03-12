# 🔍 Cuba-Search

**Self-hosted MCP search server for AI agents.** Zero cost. Zero API keys. Zero cloud dependencies.

8 tools, 6-signal confidence scoring, semantic reranking, fact validation, and optional JS rendering — in ~48MB RAM.

## Features

| Tool | Description |
|------|-------------|
| `cuba_search` | Web search via SearXNG + DuckDuckGo fallback, with semantic reranking |
| `cuba_scrape` | Scrape URL → clean text + markdown. Optional Playwright for SPAs |
| `cuba_crawl` | Crawl website (max depth 3, max 20 pages, same-domain) |
| `cuba_extract` | Extract content from multiple URLs with token budgeting |
| `cuba_map` | Discover all URLs on a page (sitemap discovery) |
| `cuba_validate` | Cross-reference claims against 5+ sources, detect contradictions |
| `cuba_docs` | Resolve library → official docs → scrape → compress |
| `cuba_research` | Deep research pipeline: search → scrape → validate → compress |

## What Makes It Unique

- **6-signal confidence scoring** — BM25 + source tier + semantic + freshness + info density + agreement
- **Fact validation** — cross-source verification with contradiction detection
- **Source credibility tiers** — official docs (T1) > known repos (T2) > general (T3)
- **Semantic reranking** — model2vec potion-base-8M (256-dim, <0.5ms/query)
- **Bilingual synonym expansion** — 100 EN/ES term groups
- **Structured markdown output** — headings, tables, lists, code blocks preserved
- **SSRF protection** — 8 private ranges + scheme blocking (OWASP A01)
- **Circuit breaker** — SearXNG auto-recovery (5 failures → 60s cooldown → retry)

## Installation

```bash
pip install -e .
```

### Optional dependencies

```bash
# JS rendering (SPAs: React, Angular, Vue)
pip install -e ".[js]"
playwright install chromium

# PDF extraction
pip install -e ".[pdf]"
```

## Prerequisites

- **Python 3.14+**
- **SearXNG** instance (optional — falls back to DuckDuckGo)

Set `SEARXNG_URL` environment variable if not on localhost:

```bash
export SEARXNG_URL=http://localhost:8080
```

## MCP Configuration

Add to your MCP settings (Claude Desktop, Antigravity, etc.):

```json
{
  "cuba-search": {
    "command": "python3.14",
    "args": ["-m", "cuba_search.server"],
    "env": {
      "SEARXNG_URL": "http://localhost:8080"
    }
  }
}
```

## Architecture

```
server.py          Entry point (asyncio.run)
protocol.py        JSON-RPC MCP transport
handlers.py        8 tool handlers
├── retrieval.py   SearXNG + DDG backends, circuit breaker
├── query.py       Normalization, decomposition, synonym expansion
├── quality.py     Deduplication, classification, normalization
├── ranking.py     BM25, RRF, 6-signal confidence
├── semantic.py    model2vec embedding + cosine reranking
├── grounding.py   Fact validation, contradiction detection
├── scraper.py     Content extraction, SSRF protection, robots.txt
├── crawler.py     Same-domain crawling, URL canonicalization
├── markdown.py    HTML → Markdown structured conversion
├── js_render.py   Optional Playwright JS rendering
├── compression.py BM25 sentence scoring for token budgets
├── partitioning.py Token budget distribution
├── cache.py       LRU + TTL caching
├── docs.py        Library → official docs resolver
└── constants.py   Tool definitions, config
```

## Benchmarks

Run the SimpleQA accuracy benchmark:

```bash
PYTHONPATH=src python3.14 benchmarks/simpleqa.py
```

Output: accuracy score, latency per query, per-question pass/fail in `simpleqa_results.json`.

## RAM Usage

| Component | RAM |
|-----------|-----|
| Base (httpx, bs4, readability) | ~25MB |
| model2vec (potion-base-8M) | ~23MB |
| Playwright (optional) | ~140MB |
| **Total (without Playwright)** | **~48MB** |

## Security

- **SSRF protection**: 8 private IP ranges + blocked schemes (file, ftp, gopher, data, javascript)
- **robots.txt**: respected with 1-hour domain cache
- **No eval/pickle**: all deserialization via `json`
- **Input sanitization**: query normalization, URL validation

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — Free to use and modify, **not for commercial use**.
