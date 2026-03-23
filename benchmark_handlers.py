import sys
from unittest.mock import MagicMock

# Mock missing dependencies
mock_modules = [
    "httpx",
    "bs4",
    "readability",
    "model2vec",
    "lxml",
    "pdfplumber",
    "playwright",
    "playwright.async_api",
    "numpy",
]

for mod_name in mock_modules:
    sys.modules[mod_name] = MagicMock()

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, patch
import os

# Add src to sys.path to import cuba_search
sys.path.append(os.path.abspath("src"))

# Now import handlers
from cuba_search import handlers

async def mock_search(query, **kwargs):
    await asyncio.sleep(0.1)  # Simulate network latency
    return [{"url": f"http://example.com/{query}", "content": f"content for {query}"}]

def mock_filter_and_classify(results):
    return results

def mock_bm25_rank(query, results, **kwargs):
    return results

async def benchmark():
    sub_queries = [f"query {i}" for i in range(5)]
    max_results = 10

    print(f"Benchmarking _search_sub_queries with {len(sub_queries)} sub-queries...")

    # We need to patch the functions inside the handlers module since it already imported them
    with patch("cuba_search.retrieval.search", side_effect=mock_search), \
         patch("cuba_search.quality.filter_and_classify", side_effect=mock_filter_and_classify), \
         patch("cuba_search.ranking.bm25_rank", side_effect=mock_bm25_rank), \
         patch("cuba_search.ranking.rrf_fuse", side_effect=lambda x: x[0]):

        start_time = time.perf_counter()
        results = await handlers._search_sub_queries(sub_queries, max_results)
        end_time = time.perf_counter()

        duration = end_time - start_time
        print(f"Duration: {duration:.4f} seconds")
        return duration

if __name__ == "__main__":
    asyncio.run(benchmark())
