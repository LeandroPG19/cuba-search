"""V1+V9+V10: Search retrieval — SearXNG + DuckDuckGo fallback.

SearXNG = primary (self-hosted, ~60 engines).
DuckDuckGo HTML = zero-config fallback when SearXNG unavailable.
V9: Retry with exponential backoff + jitter (proven pattern).
V10: Rate limiting via asyncio.Semaphore.
V16: Circuit breaker for SearXNG (CLOSED→OPEN→HALF_OPEN, §12.2).
CC: all functions ≤ 7.
"""
import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from bs4 import BeautifulSoup

from cuba_search.scraper import _ssrf_redirect_hook

logger = logging.getLogger("cuba-search.retrieval")

# ── Configuration ──────────────────────────────────────────────────
SEARXNG_URL: str = os.environ.get("SEARXNG_URL", "http://localhost:8080")

# V10: Rate limiting
_search_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent searches
_scrape_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent scrapes

# V9: Retry config
MAX_RETRIES: int = 3
RETRY_BASE_DELAY: float = 1.0
RETRY_MAX_DELAY: float = 10.0

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CubaSearch/1.0; "
        "+https://github.com/cuba-search)"
    ),
}


# ── V16: Circuit Breaker (§12.2) ──────────────────────────────────
@dataclass(slots=True)
class CircuitBreaker:
    """Three-state circuit breaker: CLOSED → OPEN → HALF_OPEN.

    Auto-recovers after recovery_timeout seconds.
    """

    threshold: int = 5
    recovery_timeout: float = 60.0
    _failures: int = field(default=0, repr=False)
    _last_failure: float = field(default=0.0, repr=False)

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (should bypass).

        Returns:
            True if failures exceeded threshold AND not in recovery window.
        """
        if self._failures < self.threshold:
            return False
        elapsed = time.monotonic() - self._last_failure
        return elapsed <= self.recovery_timeout

    def record_failure(self) -> None:
        """Record a failed attempt."""
        self._failures += 1
        self._last_failure = time.monotonic()

    def record_success(self) -> None:
        """Reset circuit on success (HALF_OPEN → CLOSED)."""
        self._failures = 0


_searxng_breaker = CircuitBreaker(threshold=5, recovery_timeout=60.0)


def _retrieval_ssrf_hook(request: httpx.Request) -> None:
    """SSRF validation hook for search backends.

    Allows SEARXNG_URL (which might be local), but validates all other targets.
    """
    url_str = str(request.url)
    if url_str.startswith(SEARXNG_URL):
        return
    _ssrf_redirect_hook(request)


async def _retry_request(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any] | None = None,
    timeout: float = 15.0,
) -> httpx.Response:
    """HTTP GET with exponential backoff + jitter.

    Args:
        client: HTTP client.
        url: Request URL.
        params: Query parameters.
        timeout: Request timeout.

    Returns:
        HTTP response.

    Raises:
        httpx.HTTPError: After all retries exhausted.
    """
    last_error: BaseException | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (httpx.HTTPError, TimeoutError) as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = min(
                    RETRY_BASE_DELAY * (2 ** attempt),
                    RETRY_MAX_DELAY,
                )
                jitter = random.uniform(0, delay * 0.3)
                await asyncio.sleep(delay + jitter)
    msg = f"All {MAX_RETRIES} retries failed: {last_error}"
    raise httpx.HTTPError(msg)


async def search_searxng(
    query: str,
    categories: str = "general",
    language: str = "auto",
    time_range: str = "",
    max_results: int = 10,
    engines: str = "",
) -> list[dict[str, Any]]:
    """Search using SearXNG instance.

    Args:
        query: Search query.
        categories: Search category (general, science, it, etc).
        language: Result language (auto, en, es, etc).
        time_range: Time range filter (day, week, month, year).
        max_results: Maximum results to return.
        engines: Specific engines comma-separated (optional).

    Returns:
        List of result dicts with url, title, content.
    """
    params: dict[str, Any] = {
        "q": query,
        "format": "json",
        "categories": categories,
    }
    if language != "auto":
        params["language"] = language
    if time_range:
        params["time_range"] = time_range
    if engines:
        params["engines"] = engines

    async with _search_semaphore, httpx.AsyncClient(
        headers=_HEADERS,
        follow_redirects=True,
        event_hooks={"request": [_retrieval_ssrf_hook]},
    ) as client:
        try:
            resp = await _retry_request(
                client,
                f"{SEARXNG_URL}/search",
                params=params,
            )
            data = resp.json()
            _searxng_breaker.record_success()
        except (httpx.HTTPError, ValueError) as e:
            _searxng_breaker.record_failure()
            logger.warning(
                "SearXNG failure #%d: %s",
                _searxng_breaker._failures, e,
            )
            return await search_duckduckgo(query, max_results)

    results = data.get("results", [])[:max_results]
    return [
        {
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content": r.get("content", ""),
            "engine": r.get("engine", "searxng"),
            "score": r.get("score", 0.0),
        }
        for r in results
        if r.get("url")
    ]


async def search_duckduckgo(
    query: str,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search using DuckDuckGo HTML scraping (zero-config fallback).

    No API key required. Parses the lite HTML version.

    Args:
        query: Search query.
        max_results: Maximum results to return.

    Returns:
        List of result dicts with url, title, content.
    """
    url = "https://lite.duckduckgo.com/lite/"
    data = {"q": query}

    async with _search_semaphore, httpx.AsyncClient(
        headers=_HEADERS,
        follow_redirects=True,
        event_hooks={"request": [_ssrf_redirect_hook]},
    ) as client:
        try:
            resp = await client.post(url, data=data, timeout=15.0)
            resp.raise_for_status()
        except (httpx.HTTPError, TimeoutError) as e:
            logger.error("DuckDuckGo search failed: %s", e)
            return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results: list[dict[str, Any]] = []

    # Parse DDG lite results
    for link in soup.find_all("a", class_="result-link"):
        href = link.get("href", "")
        title = link.get_text(strip=True)
        # Get snippet from next sibling
        snippet_td = link.find_parent("tr")
        snippet = ""
        if snippet_td:
            next_tr = snippet_td.find_next_sibling("tr")
            if next_tr:
                snippet = next_tr.get_text(strip=True)

        if href and not href.startswith("/"):
            results.append({
                "url": href,
                "title": title,
                "content": snippet,
                "engine": "duckduckgo",
                "score": 0.0,
            })

        if len(results) >= max_results:
            break

    return results


async def search(
    query: str,
    categories: str = "general",
    language: str = "auto",
    time_range: str = "",
    max_results: int = 10,
    engines: str = "",
) -> list[dict[str, Any]]:
    """Unified search interface: SearXNG → DuckDuckGo fallback.

    Args:
        query: Search query.
        categories: Search category.
        language: Result language.
        time_range: Time range filter.
        max_results: Maximum results.
        engines: Specific engines.

    Returns:
        Search results from best available backend.
    """
    if not _searxng_breaker.is_open:
        return await search_searxng(
            query, categories, language, time_range, max_results, engines,
        )
    logger.info("SearXNG circuit open — using DuckDuckGo")
    return await search_duckduckgo(query, max_results)
