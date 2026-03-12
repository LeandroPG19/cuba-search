"""8 tool handlers for cuba-search MCP.

Each handler orchestrates the pipeline stages for its tool.
Dispatch table at module level for O(1) routing.
CC: all handlers ≤ 7.
"""
import json
import logging
from typing import Any

from cuba_search import retrieval, quality, ranking, compression
from cuba_search import grounding, scraper, crawler, docs, partitioning
from cuba_search import query as query_mod
from cuba_search import semantic
from cuba_search.cache import TTLCache, normalize_cache_key

logger = logging.getLogger("cuba-search.handlers")

# Search result cache (5 min TTL)
_search_cache = TTLCache(maxsize=500, ttl=300.0)


def _serialize(data: Any) -> str:
    """Serialize response to JSON string."""
    return json.dumps(data, ensure_ascii=False, default=str)


async def handle_search(args: dict[str, Any]) -> str:
    """cuba_search: Full pipeline search.

    Pipeline: query → search → quality → rank → compress → respond.
    """
    raw_query = args.get("query", "")
    max_results = min(args.get("max_results", 10), 20)
    categories = args.get("categories", "general")
    language = args.get("language", "auto")
    time_range = args.get("time_range", "")

    # V1: Normalize query
    normalized = query_mod.normalize_query(raw_query)
    intent = query_mod.detect_intent(normalized)

    # V16: Expand with synonyms
    expanded = query_mod.expand_query(normalized)

    # Cache check
    cache_key = normalize_cache_key(normalized, categories, language, time_range)
    cached = _search_cache.get(cache_key)
    if cached is not None:
        return _serialize(cached)

    # V1: Retrieve (use expanded query for broader recall)
    results = await retrieval.search(
        expanded, categories, language, time_range, max_results * 2,
    )

    # V2+V3+V12: Quality pipeline
    results = quality.filter_and_classify(results)

    # V1+V14: Rank with BM25 (use normalized for precision)
    results = ranking.bm25_rank(normalized, results, text_key="content")

    # V16: Semantic reranking
    results = semantic.semantic_rerank(normalized, results)

    # V14: Add info density
    for r in results:
        content = r.get("content", "")
        r["info_density"] = round(ranking.information_density(content), 4)

    # Grounding: claim density + contradiction markers
    results = grounding.detect_contradictions(results)
    for r in results:
        r["claim_density"] = grounding.claim_density(r.get("content", ""))

    # Cross-source agreement
    results = grounding.cross_source_agreement(results)

    # Confidence scoring (6 signals including semantic)
    for r in results:
        conf, level = ranking.compute_confidence(
            content_relevance=min(r.get("bm25_score", 0) / 10.0, 1.0),
            source_tier=r.get("tier_score", 0.4),
            freshness=0.7,  # Default; enhanced with dates when available
            info_density=r.get("info_density", 0.5),
            cross_agreement=r.get("agreement_score", 0.0),
            semantic_score=r.get("semantic_score", 0.0),
        )
        r["confidence"] = conf
        r["confidence_level"] = level

    # Truncate to requested count
    results = results[:max_results]

    # V5: Compress content
    results = compression.compress_results(results, normalized, total_budget=3000)

    response = {
        "query": raw_query,
        "normalized_query": normalized,
        "intent": intent,
        "result_count": len(results),
        "results": results,
    }

    _search_cache.set(cache_key, response)
    return _serialize(response)


async def handle_scrape(args: dict[str, Any]) -> str:
    """cuba_scrape: Scrape single URL."""
    url = args.get("url", "")
    max_tokens = args.get("max_tokens", 2000)

    result = await scraper.scrape_url(url)

    # Compress to budget
    if result.get("content"):
        result["content"] = compression.compress_to_budget(
            result["content"], query=url, max_tokens=max_tokens,
        )
        result["token_count"] = query_mod.estimate_tokens(result["content"])

    return _serialize(result)


async def handle_crawl(args: dict[str, Any]) -> str:
    """cuba_crawl: Crawl website from starting URL."""
    url = args.get("url", "")
    max_pages = min(args.get("max_pages", 10), 20)
    max_depth = min(args.get("max_depth", 2), 3)
    instructions = args.get("instructions", "")

    results = await crawler.crawl(url, max_pages, max_depth, instructions)

    return _serialize({
        "start_url": url,
        "pages_crawled": len(results),
        "results": results,
    })


async def handle_extract(args: dict[str, Any]) -> str:
    """cuba_extract: Extract content from multiple URLs."""
    urls = args.get("urls", [])
    query = args.get("query", "")
    max_tokens = args.get("max_tokens", 3000)

    if not urls:
        return _serialize({"error": "No URLs provided"})

    results = []
    budgets = partitioning.distribute_budget(len(urls), max_tokens)

    for url, budget in zip(urls, budgets, strict=False):
        scraped = await scraper.scrape_url(url)
        if scraped.get("content") and query:
            scraped["content"] = compression.compress_to_budget(
                scraped["content"], query, max_tokens=budget,
            )
        results.append(scraped)

    return _serialize({
        "url_count": len(urls),
        "results": results,
    })


async def handle_map(args: dict[str, Any]) -> str:
    """cuba_map: Discover URLs from a page."""
    url = args.get("url", "")
    max_urls = min(args.get("max_urls", 50), 100)
    same_domain = args.get("same_domain", True)

    raw_html = await scraper.fetch_raw_html(url)
    if raw_html is None:
        return _serialize({"error": f"Could not access {url}"})

    urls = crawler.discover_urls(
        raw_html,
        url,
        same_domain_only=same_domain,
    )

    return _serialize({
        "start_url": url,
        "url_count": min(len(urls), max_urls),
        "urls": urls[:max_urls],
    })


async def handle_validate(args: dict[str, Any]) -> str:
    """cuba_validate: Cross-reference validation of a claim."""
    claim = args.get("claim", "")
    max_sources = min(args.get("max_sources", 5), 10)

    # Search for the claim
    results = await retrieval.search(claim, max_results=max_sources * 2)
    results = quality.filter_and_classify(results)
    results = ranking.bm25_rank(claim, results, text_key="content")
    results = results[:max_sources]

    # Grounding analysis
    results = grounding.detect_contradictions(results)
    results = grounding.cross_source_agreement(results)

    # Aggregate confidence
    claim_has_negation = grounding.has_negation(claim)
    total_claims = grounding.count_claims(claim)

    sources_supporting = sum(
        1 for r in results
        if not r.get("has_contradiction_markers", False)
    )

    overall_conf, conf_level = ranking.compute_confidence(
        content_relevance=min(results[0].get("bm25_score", 0) / 10.0, 1.0) if results else 0,
        source_tier=sum(r.get("tier_score", 0.4) for r in results) / max(len(results), 1),
        freshness=0.7,
        info_density=sum(ranking.information_density(r.get("content", "")) for r in results) / max(len(results), 1),
        cross_agreement=sum(r.get("agreement_score", 0) for r in results) / max(len(results), 1),
    )

    return _serialize({
        "claim": claim,
        "has_negation_markers": claim_has_negation,
        "verifiable_claims": total_claims,
        "sources_checked": len(results),
        "sources_supporting": sources_supporting,
        "overall_confidence": overall_conf,
        "confidence_level": conf_level,
        "sources": [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "source_tier": r.get("source_tier"),
                "has_contradictions": r.get("has_contradiction_markers", False),
                "agreement": r.get("agreement_score", 0),
            }
            for r in results
        ],
    })


async def handle_docs(args: dict[str, Any]) -> str:
    """cuba_docs: Documentation Intelligence — replaces Context7."""
    library = args.get("library", "")
    query = args.get("query", "")
    max_tokens = args.get("max_tokens", 1500)

    result = await docs.query_docs(library, query, max_tokens)
    return _serialize(result)


_DEPTH_CONFIG: dict[str, dict[str, int | bool]] = {
    "quick": {"max_results": 5, "scrape_top": 0},
    "standard": {"max_results": 10, "scrape_top": 3},
    "deep": {"max_results": 15, "scrape_top": 5},
}


async def _search_sub_queries(
    sub_queries: list[str],
    max_results: int,
) -> list[dict[str, Any]]:
    """Search all sub-queries and fuse with RRF.

    Args:
        sub_queries: Atomic sub-queries.
        max_results: Max results per sub-query.

    Returns:
        Fused results from all sub-queries.
    """
    all_rankings: list[list[dict[str, Any]]] = []
    for sq in sub_queries:
        results = await retrieval.search(sq, max_results=max_results)
        results = quality.filter_and_classify(results)
        results = ranking.bm25_rank(sq, results, text_key="content")
        all_rankings.append(results)

    if len(all_rankings) > 1:
        return ranking.rrf_fuse(all_rankings)
    return all_rankings[0] if all_rankings else []


async def _deep_scrape_top(
    results: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    """Deep scrape the top N results to get full content.

    Args:
        results: Ranked results.
        top_n: Number of top results to scrape.

    Returns:
        Results with 'full_content' added where available.
    """
    n = min(top_n, len(results))
    for i in range(n):
        url = results[i].get("url", "")
        if not url:
            continue
        scraped = await scraper.scrape_url(url)
        if scraped.get("content"):
            results[i]["full_content"] = scraped["content"]
    return results


def _enrich_results(
    results: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """Add grounding, info density, claim density, and confidence.

    Args:
        results: Search results to enrich.
        query: Original query.

    Returns:
        Enriched results with all quality metrics.
    """
    enriched = grounding.detect_contradictions(results, content_key="content")
    enriched = grounding.cross_source_agreement(enriched, content_key="content")

    for r in enriched:
        content = r.get("full_content", r.get("content", ""))
        r["info_density"] = round(ranking.information_density(content), 4)
        r["claim_density"] = grounding.claim_density(content)
        conf, level = ranking.compute_confidence(
            content_relevance=min(r.get("bm25_score", 0) / 10.0, 1.0),
            source_tier=r.get("tier_score", 0.4),
            freshness=0.7,
            info_density=r.get("info_density", 0.5),
            cross_agreement=r.get("agreement_score", 0.0),
        )
        r["confidence"] = conf
        r["confidence_level"] = level
    return enriched


async def handle_research(args: dict[str, Any]) -> str:
    """cuba_research: Deep research combining all pipeline stages."""
    raw_query = args.get("query", "")
    depth = args.get("depth", "standard")
    max_tokens = args.get("max_tokens", 3000)

    normalized = query_mod.normalize_query(raw_query)
    sub_queries = query_mod.decompose_query(normalized)
    config = _DEPTH_CONFIG.get(depth, _DEPTH_CONFIG["standard"])

    # Search + fuse
    fused = await _search_sub_queries(sub_queries, int(config["max_results"]))

    # Deep scrape
    scrape_n = int(config["scrape_top"])
    if scrape_n > 0:
        fused = await _deep_scrape_top(fused, scrape_n)

    # Enrich with quality metrics
    fused = _enrich_results(fused, normalized)

    # Compress and clean up
    fused = compression.compress_results(
        fused, normalized, total_budget=max_tokens,
        content_key="content",
    )
    for r in fused:
        r.pop("full_content", None)

    return _serialize({
        "query": raw_query,
        "sub_queries": sub_queries,
        "depth": depth,
        "result_count": len(fused),
        "results": fused[:int(config["max_results"])],
    })


# ── Dispatch table (O(1) routing) ─────────────────────────────────
HANDLERS: dict[str, Any] = {
    "cuba_search": handle_search,
    "cuba_scrape": handle_scrape,
    "cuba_crawl": handle_crawl,
    "cuba_extract": handle_extract,
    "cuba_map": handle_map,
    "cuba_validate": handle_validate,
    "cuba_docs": handle_docs,
    "cuba_research": handle_research,
}
