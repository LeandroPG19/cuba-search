"""Cuba-Search MCP server — thin entry point.

All logic decomposed into:
  - constants.py: tool definitions, blacklists, tier maps
  - handlers.py:  8 tool handler functions
  - protocol.py:  JSON-RPC transport, event loop
  - query.py:     V1: decomposition, expansion, normalization
  - retrieval.py: V1: search backends, V9 resilience, V10 rate limit
  - quality.py:   V2: depuration, V3: classification, V12: normalization
  - ranking.py:   V1: RRF, BM25, V14: info density, confidence
  - compression.py: V5: BM25 sentence scoring
  - partitioning.py: V4: chunking, token budgeting
  - grounding.py: fact validation, contradiction detection
  - scraper.py:   V6: content extraction, V11: robots.txt
  - crawler.py:   V6: crawling, V11: ethical, V12: URL canonical
  - cache.py:     V8: LRU+TTL
  - docs.py:      V15: Documentation Intelligence
"""
import asyncio

from cuba_search.protocol import main


def run() -> None:
    """Entry point for the MCP server (called by pyproject.toml)."""
    asyncio.run(main())
