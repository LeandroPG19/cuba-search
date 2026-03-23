"""V6+V11+V12: Crawler — multi-page crawling with ethical constraints.

Respects robots.txt and rate limits.
CC: all functions ≤ 7 (verified with radon).
"""
import asyncio
import logging
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from cuba_search.quality import normalize_url
from cuba_search.scraper import (
    _HEADERS,
    _is_ssrf_safe,
    _ssrf_redirect_hook,
    scrape_url,
)

logger = logging.getLogger("cuba-search.crawler")

# Crawler limits
MAX_PAGES: int = 20
MAX_DEPTH: int = 3
CRAWL_DELAY: float = 1.0  # seconds between requests per domain


def _matches_instructions(content: str, instructions: str) -> bool:
    """Check if page content matches filter instructions.

    Args:
        content: Page text content.
        instructions: Space-separated keywords to match.

    Returns:
        True if any keyword found in content.
    """
    keywords = instructions.lower().split()
    return any(kw in content.lower() for kw in keywords)


def _enqueue_links(
    scraped: dict[str, Any],
    url: str,
    depth: int,
    max_depth: int,
    base_domain: str,
    visited: set[str],
    queue: list[tuple[str, int]],
) -> None:
    """Extract and enqueue unvisited links from scraped page.

    Args:
        scraped: Scraped page data.
        url: Current page URL.
        depth: Current crawl depth.
        max_depth: Maximum depth.
        base_domain: Base domain for filtering.
        visited: Set of visited canonical URLs.
        queue: BFS queue to append to.
    """
    if depth >= max_depth:
        return
    links = _extract_links(scraped.get("content", ""), url, base_domain)
    for link in links:
        if normalize_url(link) not in visited:
            queue.append((link, depth + 1))


def _should_skip(
    url: str,
    canonical: str,
    depth: int,
    max_depth: int,
    base_domain: str,
    visited: set[str],
) -> bool:
    """Check if URL should be skipped.

    Args:
        url: URL to check.
        canonical: Canonicalized URL.
        depth: Current depth.
        max_depth: Maximum depth.
        base_domain: Allowed domain.
        visited: Already visited URLs.

    Returns:
        True if URL should be skipped.
    """
    if canonical in visited or depth > max_depth:
        return True
    return urlparse(url).netloc.lower() != base_domain


async def crawl(
    start_url: str,
    max_pages: int = MAX_PAGES,
    max_depth: int = MAX_DEPTH,
    instructions: str = "",
) -> list[dict[str, Any]]:
    """Crawl a website starting from a URL.

    BFS traversal respecting depth limit and ethical constraints.

    Args:
        start_url: Starting URL.
        max_pages: Maximum pages to crawl.
        max_depth: Maximum depth from start URL.
        instructions: Optional filter keywords.

    Returns:
        List of crawled page dicts with url, title, content, depth.
    """
    if not _is_ssrf_safe(start_url):
        return [{"error": "SSRF: private network blocked"}]

    visited: set[str] = set()
    results: list[dict[str, Any]] = []
    queue: list[tuple[str, int]] = [(start_url, 0)]
    base_domain = urlparse(start_url).netloc.lower()

    async with httpx.AsyncClient(
        headers=_HEADERS,
        follow_redirects=True,
        timeout=15.0,
        event_hooks={"request": [_ssrf_redirect_hook]},
    ) as client:
        while queue and len(results) < max_pages:
            url, depth = queue.pop(0)
            canonical = normalize_url(url)

            if _should_skip(url, canonical, depth, max_depth, base_domain, visited):
                continue
            visited.add(canonical)

            scraped = await scrape_url(url, client=client)
            if scraped.get("status") != "ok":
                continue

            if instructions and not _matches_instructions(
                scraped.get("content", ""), instructions
            ):
                continue

            results.append({**scraped, "depth": depth})
            _enqueue_links(scraped, url, depth, max_depth, base_domain, visited, queue)
            await asyncio.sleep(CRAWL_DELAY)

    return results


def discover_urls(
    html: str,
    base_url: str,
    same_domain_only: bool = True,
) -> list[str]:
    """Discover URLs from HTML content.

    Args:
        html: HTML content to parse.
        base_url: Base URL for resolving relative links.
        same_domain_only: Only return same-domain URLs.

    Returns:
        List of discovered URLs.
    """
    base_domain = urlparse(base_url).netloc.lower()
    return _extract_links(html, base_url, base_domain if same_domain_only else "")


def _is_valid_href(href: str) -> bool:
    """Check if href is a valid link (not anchor/javascript/mailto)."""
    return not href.startswith(("#", "javascript:", "mailto:"))


def _extract_links(
    html_or_text: str,
    base_url: str,
    filter_domain: str,
) -> list[str]:
    """Extract and filter links from HTML content.

    Args:
        html_or_text: HTML content.
        base_url: Base URL for resolving relative links.
        filter_domain: Only include links from this domain (empty = all).

    Returns:
        List of absolute URLs.
    """
    try:
        soup = BeautifulSoup(html_or_text, "html.parser")
    except Exception:
        return []

    links: list[str] = []
    seen: set[str] = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        if not _is_valid_href(href):
            continue
        absolute = urljoin(base_url, href)
        canonical = normalize_url(absolute)

        if canonical in seen:
            continue
        seen.add(canonical)

        if filter_domain and urlparse(absolute).netloc.lower() != filter_domain:
            continue

        if _is_ssrf_safe(absolute):
            links.append(absolute)

    return links[:100]
