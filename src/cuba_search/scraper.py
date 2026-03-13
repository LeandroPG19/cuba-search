"""V6: Content extraction + V11: Ethical crawling + SSRF protection.

Uses readability-lxml for clean content extraction.
Respects robots.txt, implements SSRF protection (OWASP A01 2025).
V17: Optional Playwright fallback for JS-rendered pages (SPAs).
V17: Markdown structured output via html_to_markdown.
CC: all functions ≤ 7 (scrape_url CC=11, justified state machine).
"""
import ipaddress
import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("cuba-search.scraper")

# ── SSRF Protection (OWASP A01 2025) ──────────────────────────────
_PRIVATE_RANGES = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

_BLOCKED_SCHEMES: frozenset[str] = frozenset({
    "file", "ftp", "gopher", "data", "javascript",
})

# ── HTTP Client Config ─────────────────────────────────────────────
_DEFAULT_TIMEOUT = 15.0
_USER_AGENT = (
    "Mozilla/5.0 (compatible; CubaSearch/1.0; "
    "+https://github.com/cuba-search)"
)
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
}

# robots.txt cache (in-memory, small)
_robots_cache: dict[str, tuple[float, bool]] = {}


def _ssrf_redirect_hook(request: httpx.Request) -> None:
    """Validate every outgoing request (including redirects).

    Args:
        request: The httpx Request object.

    Raises:
        httpx.RequestError: If the URL fails SSRF safety check.
    """
    if not _is_ssrf_safe(str(request.url)):
        msg = f"SSRF Protection: Blocked access to {request.url}"
        raise httpx.RequestError(msg)


def _is_ssrf_safe(url: str) -> bool:
    """Validate URL is not targeting private/internal networks.

    Args:
        url: URL to validate.

    Returns:
        True if URL is safe (public network).
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() in _BLOCKED_SCHEMES:
        return False
    hostname = parsed.hostname
    if not hostname:
        return False
    try:
        addr = ipaddress.ip_address(hostname)
        return not any(addr in net for net in _PRIVATE_RANGES)
    except ValueError:
        # Hostname (not IP) — check for localhost patterns
        lower = hostname.lower()
        return lower not in {"localhost", "0.0.0.0", "[::1]"}


def _parse_robots_disallowed(text: str, path: str) -> bool:
    """Parse robots.txt to check if path is disallowed.

    Args:
        text: robots.txt content (lowercased).
        path: URL path to check.

    Returns:
        True if path is disallowed by robots.txt.
    """
    in_block = False
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("user-agent:"):
            ua = line.split(":", 1)[1].strip()
            in_block = ua == "*" or "cubasearch" in ua
        elif in_block and line.startswith("disallow:"):
            blocked = line.split(":", 1)[1].strip()
            if blocked and path.startswith(blocked):
                return True
    return False


async def check_robots_txt(
    client: httpx.AsyncClient,
    url: str,
) -> bool:
    """Check if URL is allowed by robots.txt.

    Caches result for 1 hour per domain.

    Args:
        client: HTTP client.
        url: URL to check.

    Returns:
        True if URL is allowed or robots.txt is unavailable.
    """
    import time
    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"

    # Check cache
    if domain in _robots_cache:
        ts, allowed = _robots_cache[domain]
        if time.monotonic() - ts < 3600.0:
            return allowed

    robots_url = f"{domain}/robots.txt"
    try:
        resp = await client.get(robots_url, timeout=5.0)
        if resp.status_code != 200:
            _robots_cache[domain] = (time.monotonic(), True)
            return True
        path = parsed.path or "/"
        disallowed = _parse_robots_disallowed(resp.text.lower(), path)
        _robots_cache[domain] = (time.monotonic(), not disallowed)
        return not disallowed
    except (httpx.HTTPError, TimeoutError):
        _robots_cache[domain] = (time.monotonic(), True)
        return True


async def fetch_raw_html(
    url: str,
    timeout: float = _DEFAULT_TIMEOUT,
) -> str | None:
    """Fetch raw HTML from a URL without readability cleaning.

    Used by handle_map to preserve <a> tags for link extraction.
    Includes SSRF protection and robots.txt compliance.

    Args:
        url: URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Raw HTML string, or None if blocked/error.
    """
    if not _is_ssrf_safe(url):
        return None

    async with httpx.AsyncClient(
        headers=_HEADERS,
        follow_redirects=True,
        timeout=timeout,
        event_hooks={"request": [_ssrf_redirect_hook]},
    ) as client:
        try:
            allowed = await check_robots_txt(client, url)
            if not allowed:
                return None
            resp = await client.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except (httpx.HTTPError, TimeoutError):
            return None


async def scrape_url(
    url: str,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    respect_robots: bool = True,
) -> dict[str, Any]:
    """Scrape a single URL and extract clean content.

    Uses readability-lxml for main content extraction.
    Includes SSRF protection and robots.txt compliance.

    Args:
        url: URL to scrape.
        client: Optional shared HTTP client.
        timeout: Request timeout in seconds.
        respect_robots: Whether to check robots.txt.

    Returns:
        Dict with: url, title, content, status, content_type.

    Raises:
        ValueError: If URL fails SSRF check.
    """
    if not _is_ssrf_safe(url):
        return {
            "url": url, "title": "", "content": "",
            "status": "blocked", "error": "SSRF: private network",
        }

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(
            headers=_HEADERS,
            follow_redirects=True,
            timeout=timeout,
            event_hooks={"request": [_ssrf_redirect_hook]},
        )

    assert client is not None  # type narrowing for mypy

    try:
        if respect_robots:
            allowed = await check_robots_txt(client, url)
            if not allowed:
                return {
                    "url": url, "title": "", "content": "",
                    "status": "blocked", "error": "robots.txt disallowed",
                }

        resp = await client.get(url, timeout=timeout)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return {
                "url": url, "title": "", "content": resp.text[:2000],
                "markdown": "",
                "status": "ok", "content_type": content_type,
            }

        result = _extract_html(url, resp.text)

        # V17: JS rendering fallback — if content is minimal, try Playwright
        if len(result.get("content", "")) < 100:
            from cuba_search import js_render
            if js_render.is_available():
                logger.info("Minimal content — trying Playwright for %s", url)
                rendered = await js_render.render_page(url)
                if rendered.get("html"):
                    result = _extract_html(url, rendered["html"])
                    result["renderer"] = "playwright"

        return result

    except httpx.HTTPStatusError as e:
        return {
            "url": url, "title": "", "content": "",
            "status": "error", "error": f"HTTP {e.response.status_code}",
        }
    except (httpx.HTTPError, TimeoutError) as e:
        return {
            "url": url, "title": "", "content": "",
            "status": "error", "error": str(type(e).__name__),
        }
    finally:
        if own_client:
            await client.aclose()


def _extract_html(url: str, html: str) -> dict[str, Any]:
    """Extract clean content from HTML using readability + BeautifulSoup.

    Returns both plain text and markdown-structured content.

    Args:
        url: Source URL.
        html: Raw HTML content.

    Returns:
        Dict with extracted title, content (text), and markdown.
    """
    try:
        from readability import Document
        doc = Document(html)
        title = doc.title()
        summary_html = doc.summary()
    except Exception:
        title = ""
        summary_html = html

    # V17: Markdown structured output
    from cuba_search.markdown import html_to_markdown
    md = html_to_markdown(summary_html)

    soup = BeautifulSoup(summary_html, "html.parser")

    # Remove scripts, styles, nav, footer
    for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Clean up excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return {
        "url": url,
        "title": title.strip() if title else "",
        "content": text,
        "markdown": md,
        "status": "ok",
        "content_type": "text/html",
    }
