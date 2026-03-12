"""V15: Documentation Intelligence — replaces Context7 MCP.

Self-hosted, real-time documentation retrieval.
Uses PyPI/npm/crates.io JSON APIs for URL resolution.
Scrapes official docs with readability-lxml.
CC: all functions ≤ 7.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx

from cuba_search.cache import TTLCache, normalize_cache_key
from cuba_search.compression import compress_to_budget
from cuba_search.scraper import scrape_url

logger = logging.getLogger("cuba-search.docs")

# ── Library URL mapping (top ~100 libs) ────────────────────────────
_LIBRARY_MAP_PATH = Path(__file__).parent / "data" / "library_mapping.json"
_library_map: dict[str, str] | None = None

# Documentation cache (longer TTL — docs change slowly)
_docs_cache = TTLCache(maxsize=200, ttl=3600.0)  # 1 hour

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CubaSearch/1.0; "
        "+https://github.com/cuba-search)"
    ),
    "Accept": "application/json",
}


def _load_library_map() -> dict[str, str]:
    """Load library-to-docs-URL mapping from JSON file.

    Returns:
        Dict mapping library names to documentation URLs.
    """
    global _library_map
    if _library_map is not None:
        return _library_map

    if _LIBRARY_MAP_PATH.exists():
        try:
            _library_map = json.loads(_LIBRARY_MAP_PATH.read_text("utf-8"))
            return _library_map
        except (json.JSONDecodeError, OSError):
            pass

    _library_map = {}
    return _library_map


async def _resolve_pypi(
    client: httpx.AsyncClient,
    library: str,
) -> str | None:
    """Resolve docs URL via PyPI JSON API.

    Args:
        client: HTTP client.
        library: Python package name.

    Returns:
        Documentation URL or None.
    """
    try:
        resp = await client.get(
            f"https://pypi.org/pypi/{library}/json",
            timeout=10.0,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        info = data.get("info", {})
        urls = info.get("project_urls") or {}
        # Priority: Documentation > Homepage > Project URL
        for key in ("Documentation", "Docs", "documentation", "docs"):
            if key in urls:
                return urls[key]
        return urls.get("Homepage") or info.get("home_page")
    except (httpx.HTTPError, ValueError, KeyError):
        return None


async def _resolve_npm(
    client: httpx.AsyncClient,
    library: str,
) -> str | None:
    """Resolve docs URL via npm registry API.

    Args:
        client: HTTP client.
        library: npm package name.

    Returns:
        Documentation URL or None.
    """
    try:
        resp = await client.get(
            f"https://registry.npmjs.org/{library}",
            timeout=10.0,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("homepage") or data.get("repository", {}).get("url")
    except (httpx.HTTPError, ValueError, KeyError):
        return None


async def resolve_docs_url(library: str) -> str | None:
    """Resolve documentation URL for a library.

    Priority: local mapping → PyPI → npm.

    Args:
        library: Library name.

    Returns:
        Documentation URL or None.
    """
    lib_map = _load_library_map()
    normalized = library.lower().strip().replace(" ", "-")

    # 1. Check local mapping
    if normalized in lib_map:
        return lib_map[normalized]

    # 2. Try PyPI
    async with httpx.AsyncClient(headers=_HEADERS) as client:
        url = await _resolve_pypi(client, normalized)
        if url:
            return url
        # 3. Try npm
        url = await _resolve_npm(client, normalized)
        if url:
            return url

    return None


async def query_docs(
    library: str,
    query: str,
    max_tokens: int = 1500,
) -> dict[str, Any]:
    """Query documentation for a library — single-call replacement for Context7.

    Resolves library → docs URL → scrapes → compresses → returns.

    Args:
        library: Library name (e.g., "fastapi", "react", "sqlalchemy").
        query: What to search for in the docs.
        max_tokens: Maximum response tokens.

    Returns:
        Dict with: library, docs_url, content, token_count, source.
    """
    # Check cache
    cache_key = normalize_cache_key(library, query)
    cached = _docs_cache.get(cache_key)
    if cached is not None:
        return cached

    # Resolve docs URL
    docs_url = await resolve_docs_url(library)
    if not docs_url:
        result = {
            "library": library,
            "docs_url": None,
            "content": f"Could not resolve documentation URL for '{library}'.",
            "token_count": 0,
            "source": "none",
        }
        return result

    # Scrape documentation page
    scraped = await scrape_url(docs_url)
    if scraped.get("status") != "ok" or not scraped.get("content"):
        result = {
            "library": library,
            "docs_url": docs_url,
            "content": f"Could not scrape docs at {docs_url}.",
            "token_count": 0,
            "source": "error",
        }
        return result

    # Compress to budget
    compressed = compress_to_budget(
        scraped["content"],
        query=f"{library} {query}",
        max_tokens=max_tokens,
    )

    result = {
        "library": library,
        "docs_url": docs_url,
        "title": scraped.get("title", ""),
        "content": compressed,
        "token_count": len(compressed.split()),
        "source": "official_docs",
    }

    _docs_cache.set(cache_key, result)
    return result
