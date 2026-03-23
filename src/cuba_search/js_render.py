"""V17: Optional JavaScript rendering via Playwright.

Fallback for pages that return empty content with httpx (SPAs, React, Angular).
Playwright is an OPTIONAL dependency — not loaded unless needed.
CC: all functions ≤ 5.
"""
import logging
from typing import Any

from cuba_search.scraper import _is_ssrf_safe

logger = logging.getLogger("cuba-search.js_render")

_PLAYWRIGHT_AVAILABLE: bool | None = None  # None = not checked yet


def is_available() -> bool:
    """Check if Playwright is installed (lazy, cached).

    Returns:
        True if playwright is importable.
    """
    global _PLAYWRIGHT_AVAILABLE  # noqa: PLW0603
    if _PLAYWRIGHT_AVAILABLE is not None:
        return _PLAYWRIGHT_AVAILABLE
    try:
        import playwright  # noqa: F401
        _PLAYWRIGHT_AVAILABLE = True
    except ImportError:
        _PLAYWRIGHT_AVAILABLE = False
        logger.info("Playwright not installed — JS rendering disabled")
    return _PLAYWRIGHT_AVAILABLE


async def render_page(
    url: str,
    timeout_ms: int = 15000,
    wait_for: str = "networkidle",
) -> dict[str, Any]:
    """Render a page with Playwright and return HTML + title.

    Args:
        url: URL to render.
        timeout_ms: Navigation timeout in milliseconds.
        wait_for: Wait condition (networkidle, load, domcontentloaded).

    Returns:
        Dict with html, title, status keys.
    """
    if not is_available():
        return {"html": "", "title": "", "status": "unavailable"}

    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                # V17: SSRF Protection for Playwright
                context = await browser.new_context()
                page = await context.new_page()

                # Intercept all requests to prevent SSRF (including redirects)
                async def intercept(route: Any) -> None:
                    if _is_ssrf_safe(route.request.url):
                        await route.continue_()
                    else:
                        logger.warning("SSRF Blocked Playwright request: %s", route.request.url)
                        await route.abort("blockedbyclient")

                await page.route("**/*", intercept)

                await page.goto(
                    url,
                    timeout=timeout_ms,
                    wait_until=wait_for,  # type: ignore[arg-type]
                )
                title = await page.title()
                html = await page.content()
                return {"html": html, "title": title, "status": "ok"}
            finally:
                await browser.close()
    except Exception as e:
        logger.warning("Playwright render failed for %s: %s", url, e)
        return {"html": "", "title": "", "status": "error", "error": str(type(e).__name__)}
