"""V2: Depuration + V3: Classification + V12: Normalization.

Filters noise, classifies results, normalizes URLs.
CC: all functions ≤ 5.
"""
import re
from urllib.parse import urlparse, urlunparse
from typing import Any

# ── V2: Domain blacklist (configurable) ────────────────────────────
DEFAULT_BLACKLIST: frozenset[str] = frozenset({
    "pinterest.com", "pinterest.es",
    "quora.com",
    "facebook.com", "fb.com",
    "instagram.com",
    "tiktok.com",
    "twitter.com", "x.com",
    "linkedin.com",
    "reddit.com",  # Often low signal-to-noise for technical queries
    "medium.com",  # Paywalled + low quality
    "w3schools.com",  # Frequently inaccurate
    "geeksforgeeks.org",  # SEO farm
})

# ── V3: Source tier classification ─────────────────────────────────
TIER_1_DOMAINS: frozenset[str] = frozenset({
    # Official docs
    "docs.python.org", "react.dev", "nextjs.org", "fastapi.tiangolo.com",
    "nodejs.org", "developer.mozilla.org", "docs.rust-lang.org",
    "learn.microsoft.com", "cloud.google.com", "docs.aws.amazon.com",
    "docs.docker.com", "kubernetes.io",
    # Academic
    "arxiv.org", "scholar.google.com", "ieee.org", "acm.org",
    "dl.acm.org", "nature.com", "science.org",
    # Standards
    "rfc-editor.org", "w3.org", "ecma-international.org",
})

TIER_2_DOMAINS: frozenset[str] = frozenset({
    "stackoverflow.com", "github.com", "gitlab.com",
    "wikipedia.org", "en.wikipedia.org",
    "docs.github.com", "pypi.org", "npmjs.com",
    "crates.io", "pkg.go.dev",
})


def is_blacklisted(url: str, blacklist: frozenset[str] | None = None) -> bool:
    """Check if URL domain is in the blacklist.

    Args:
        url: URL to check.
        blacklist: Custom blacklist (uses DEFAULT_BLACKLIST if None).

    Returns:
        True if domain is blacklisted.
    """
    bl = blacklist if blacklist is not None else DEFAULT_BLACKLIST
    domain = urlparse(url).netloc.lower()
    return any(domain.endswith(b) for b in bl)


def classify_source(url: str) -> tuple[int, float]:
    """Classify source into tiers for credibility scoring.

    Args:
        url: Source URL.

    Returns:
        Tuple of (tier number 1-3, tier score 0.0-1.0).
    """
    domain = urlparse(url).netloc.lower()
    if any(domain.endswith(d) for d in TIER_1_DOMAINS):
        return 1, 1.0
    if any(domain.endswith(d) for d in TIER_2_DOMAINS):
        return 2, 0.7
    return 3, 0.4


def normalize_url(url: str) -> str:
    """Canonicalize URL for deduplication.

    Removes fragments, trailing slashes, normalizes scheme.

    Args:
        url: URL to normalize.

    Returns:
        Canonicalized URL string.
    """
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        path,
        parsed.params,
        parsed.query,
        "",  # Remove fragment
    ))


def deduplicate_results(
    results: list[dict[str, Any]],
    url_key: str = "url",
) -> list[dict[str, Any]]:
    """Remove duplicate results by normalized URL.

    Args:
        results: List of result dicts.
        url_key: Key containing the URL.

    Returns:
        Deduplicated results preserving original order.
    """
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for r in results:
        canonical = normalize_url(r.get(url_key, ""))
        if canonical not in seen:
            seen.add(canonical)
            unique.append(r)
    return unique


def filter_and_classify(
    results: list[dict[str, Any]],
    blacklist: frozenset[str] | None = None,
    url_key: str = "url",
) -> list[dict[str, Any]]:
    """Apply full quality pipeline: blacklist → dedup → classify.

    Args:
        results: Raw search results.
        blacklist: Custom domain blacklist.
        url_key: Key containing the URL.

    Returns:
        Filtered, deduplicated, and classified results.
    """
    # V2: Remove blacklisted
    filtered = [
        r for r in results
        if not is_blacklisted(r.get(url_key, ""), blacklist)
    ]
    # V12: Normalize and deduplicate
    deduped = deduplicate_results(filtered, url_key)
    # V3: Add tier classification
    classified = []
    for r in deduped:
        tier, tier_score = classify_source(r.get(url_key, ""))
        classified.append({**r, "source_tier": tier, "tier_score": tier_score})
    return classified
