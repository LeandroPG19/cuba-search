"""8 tool handlers for cuba-search MCP.

Each handler orchestrates the pipeline stages for its tool.
Dispatch table at module level for O(1) routing.
CC: all handlers ≤ 7.
"""

import json
import logging
import re as _re
from typing import Any

# ── M3: Dynamic freshness weighting ────────────────────────────────
# Temporal queries need higher freshness weight and lower source_tier weight
# (established docs may be outdated). Remaining 0.55 is split: semantic=0.20,
# info_density=0.10, cross_agreement=0.10, plus the three variable weights = 1.0.
_TEMPORAL_RE = _re.compile(
    r"\b(?:latest|newest|current|2025|2026|recent|now|today|updated?)\b",
    _re.IGNORECASE,
)


def _confidence_weights(query: str) -> tuple[float, float, float]:
    """Return (w_content, w_tier, w_freshness) adjusted for temporal queries.

    Temporal queries boost freshness and reduce source_tier weight, since
    authoritative but old docs rank poorly for time-sensitive topics.
    """
    if _TEMPORAL_RE.search(query):
        return 0.20, 0.15, 0.25
    return 0.25, 0.20, 0.15


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
        expanded,
        categories,
        language,
        time_range,
        max_results * 2,
    )

    # V2+V3+V12: Quality pipeline
    results = quality.filter_and_classify(results)

    # V1+V14: Rank with BM25 — M2: use intent-adapted k1/b params
    results = ranking.bm25_rank(normalized, results, text_key="content", intent=intent)

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

    # M3: Dynamic confidence weights (temporal queries boost freshness)
    w_content, w_tier, w_freshness = _confidence_weights(normalized)

    # Confidence scoring (6 signals including semantic)
    for r in results:
        conf, level = ranking.compute_confidence(
            content_relevance=min(r.get("bm25_score", 0) / 10.0, 1.0),
            source_tier=r.get("tier_score", 0.4),
            freshness=0.7,  # Default; enhanced with dates when available
            info_density=r.get("info_density", 0.5),
            cross_agreement=r.get("agreement_score", 0.0),
            semantic_score=r.get("semantic_score", 0.0),
            w_content=w_content,
            w_tier=w_tier,
            w_freshness=w_freshness,
        )
        r["confidence"] = conf
        r["confidence_level"] = level

    # M6: Diversity — limit results per domain before truncating
    results = _diversify_results(results)

    # Truncate to requested count
    results = results[:max_results]

    # V5: Compress content (M1: weighted budget via distribute_budget)
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
            result["content"],
            query=url,
            max_tokens=max_tokens,
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

    return _serialize(
        {
            "start_url": url,
            "pages_crawled": len(results),
            "results": results,
        }
    )


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
                scraped["content"],
                query,
                max_tokens=budget,
            )
        results.append(scraped)

    return _serialize(
        {
            "url_count": len(urls),
            "results": results,
        }
    )


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

    return _serialize(
        {
            "start_url": url,
            "url_count": min(len(urls), max_urls),
            "urls": urls[:max_urls],
        }
    )


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

    sources_supporting = sum(1 for r in results if not r.get("has_contradiction_markers", False))

    overall_conf, conf_level = ranking.compute_confidence(
        content_relevance=min(results[0].get("bm25_score", 0) / 10.0, 1.0) if results else 0,
        source_tier=sum(r.get("tier_score", 0.4) for r in results) / max(len(results), 1),
        freshness=0.7,
        info_density=sum(ranking.information_density(r.get("content", "")) for r in results)
        / max(len(results), 1),
        cross_agreement=sum(r.get("agreement_score", 0) for r in results) / max(len(results), 1),
    )

    return _serialize(
        {
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
        }
    )


async def handle_docs(args: dict[str, Any]) -> str:
    """cuba_docs: Documentation Intelligence — replaces Context7."""
    library = args.get("library", "")
    query = args.get("query", "")
    max_tokens = args.get("max_tokens", 1500)

    result = await docs.query_docs(library, query, max_tokens)
    return _serialize(result)


# M7: Adjusted depth config — quick now scrapes 2 URLs (was 0),
# deep covers 20 results and 7 deep-scraped pages.
_DEPTH_CONFIG: dict[str, dict[str, int | bool]] = {
    "quick": {"max_results": 8, "scrape_top": 2},
    "standard": {"max_results": 12, "scrape_top": 4},
    "deep": {"max_results": 20, "scrape_top": 7},
}

# M5: Seeds for query reformulation in iterative deepening
_INTENT_SEEDS: dict[str, list[str]] = {
    "code": ["example", "tutorial", "how to"],
    "academic": ["paper", "study", "research"],
    "navigational": ["official", "docs", "guide"],
    "informational": ["guide", "overview", "explained"],
}


def _diversify_results(
    results: list[dict[str, Any]],
    max_per_domain: int = 2,
) -> list[dict[str, Any]]:
    """Limit results per domain (MMR simplified, Carbonell & Goldstein 1998).

    Tier-1 sources (official docs) get one extra slot since they are
    authoritative even if multiple pages are relevant.

    Args:
        results: Ranked results to diversify.
        max_per_domain: Max results per unique domain.

    Returns:
        Filtered results with at most max_per_domain per domain.
    """
    from urllib.parse import urlparse

    domain_counts: dict[str, int] = {}
    filtered = []
    for r in results:
        netloc = urlparse(r.get("url", "")).netloc.lower()
        limit = max_per_domain + (1 if r.get("source_tier", 3) == 1 else 0)
        if domain_counts.get(netloc, 0) < limit:
            filtered.append(r)
            domain_counts[netloc] = domain_counts.get(netloc, 0) + 1
    return filtered


# PRF stopwords (module-level constant, not re-created per call)
_PRF_STOPS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "of",
        "in",
        "to",
        "for",
        "on",
        "at",
        "by",
        "as",
        "with",
        "from",
        "that",
        "this",
        "it",
        "its",
        "they",
        "their",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
    }
)


def _score_prf_terms(
    all_doc_terms: list[list[str]],
    query_terms: set[str],
) -> dict[str, float]:
    """Compute RM3-inspired tf×idf scores for PRF expansion candidates.

    score(t) = Σ_d  (tf(t,d) / |d|) × log(N / (df(t) + 1) + 1)

    Excludes query terms, stopwords, and tokens shorter than 4 chars.
    """
    import math
    from collections import Counter

    n_docs = len(all_doc_terms)
    df: dict[str, int] = {}
    for terms in all_doc_terms:
        for t in set(terms):
            df[t] = df.get(t, 0) + 1

    term_scores: dict[str, float] = {}
    for terms in all_doc_terms:
        if not terms:
            continue
        tf_counts = Counter(terms)
        doc_len = len(terms)
        for t, tf in tf_counts.items():
            if len(t) < 4 or t in query_terms or t in _PRF_STOPS:
                continue
            idf = math.log(n_docs / (df.get(t, 0) + 1) + 1.0)
            term_scores[t] = term_scores.get(t, 0.0) + (tf / doc_len) * idf
    return term_scores


def _prf_expand_query(
    normalized: str,
    top_docs: list[dict[str, Any]],
    top_k: int = 5,
) -> str:
    """RM3-inspired PRF: expand query with top tf×idf terms from top documents.

    Math (simplified RM3, Lavrenko & Croft 2001):
        score(t) = Σ_d  (tf(t,d) / |d|) × log(N / (df(t) + 1) + 1)

    Delegates scoring to _score_prf_terms() to keep CC ≤ 7.

    Args:
        normalized: Normalized original query.
        top_docs: Top-ranked documents from first retrieval pass.
        top_k: Number of expansion terms to add.

    Returns:
        Original query + top_k PRF expansion terms.
    """
    query_terms = set(normalized.lower().split())
    all_doc_terms = [doc.get("content", "").lower().split() for doc in top_docs]
    term_scores = _score_prf_terms(all_doc_terms, query_terms)
    expansion = sorted(term_scores, key=term_scores.__getitem__, reverse=True)[:top_k]
    return f"{normalized} {' '.join(expansion)}" if expansion else normalized


async def _research_retry(
    normalized: str,
    intent: str,
    max_results: int,
    top_docs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Generate alternative sub-queries and search (one retry round).

    M16: When top_docs are provided, uses RM3-inspired PRF to extract
    expansion terms from the retrieved documents — more systematic than
    fixed seed words (arXiv:2503.14887, University of Queensland 2025).

    Falls back to intent-seed expansion when top_docs is None.

    Args:
        normalized: Normalized original query.
        intent: Detected query intent for seed selection / BM25 params.
        max_results: Max results per sub-query.
        top_docs: Top documents from first retrieval pass (for PRF).

    Returns:
        Fused results from alternative queries.
    """
    if top_docs:
        # M16: PRF — tf×idf weighted expansion from actual top documents
        prf_query = _prf_expand_query(normalized, top_docs[:3])
        return await _search_sub_queries([prf_query], max_results, intent=intent)
    # Fallback: intent-seed based expansion
    seeds = _INTENT_SEEDS.get(intent, _INTENT_SEEDS["informational"])
    alt_queries = [f"{normalized} {seed}" for seed in seeds[:2]]
    return await _search_sub_queries(alt_queries, max_results, intent=intent)


async def _search_sub_queries(
    sub_queries: list[str],
    max_results: int,
    intent: str = "informational",
) -> list[dict[str, Any]]:
    """Search all sub-queries and fuse with RRF.

    Args:
        sub_queries: Atomic sub-queries.
        max_results: Max results per sub-query.
        intent: Query intent for M2 BM25 parameter selection.

    Returns:
        Fused results from all sub-queries.
    """
    all_rankings: list[list[dict[str, Any]]] = []
    for sq in sub_queries:
        results = await retrieval.search(sq, max_results=max_results)
        results = quality.filter_and_classify(results)
        # M2: pass intent for domain-adapted BM25 params
        results = ranking.bm25_rank(sq, results, text_key="content", intent=intent)
        all_rankings.append(results)

    if len(all_rankings) > 1:
        # M15: Decay weights — first sub-query is closest to original intent.
        # w_i = 1 / (1 + 0.2·i)  → 1.0, 0.83, 0.71, 0.63, ...
        # Empirical basis: WRRF (Samuel et al. 2025) shows per-signal weighting
        # improves nDCG@10 by up to 6.4% over uniform RRF.
        decay = [1.0 / (1.0 + 0.2 * i) for i in range(len(all_rankings))]
        return ranking.rrf_fuse(all_rankings, weights=decay)
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

    # M3: Dynamic confidence weights based on temporal keywords in query
    w_content, w_tier, w_freshness = _confidence_weights(query)

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
            w_content=w_content,
            w_tier=w_tier,
            w_freshness=w_freshness,
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
    intent = query_mod.detect_intent(normalized)
    sub_queries = query_mod.decompose_query(normalized)
    config = _DEPTH_CONFIG.get(depth, _DEPTH_CONFIG["standard"])
    max_res = int(config["max_results"])

    # Search + fuse — M2: pass intent for adapted BM25 params
    fused = await _search_sub_queries(sub_queries, max_res, intent=intent)

    # Deep scrape
    scrape_n = int(config["scrape_top"])
    if scrape_n > 0:
        fused = await _deep_scrape_top(fused, scrape_n)

    # Enrich with quality metrics
    fused = _enrich_results(fused, normalized)

    # M5+M16: Iterative deepening with PRF — retry if top-3 confidence is weak.
    # M16: Pass top_docs so _research_retry uses RM3-inspired tf×idf PRF
    # instead of fixed seed words (arXiv:2503.14887).
    # M15: Weight primary results 2× over retry (they come from original query).
    top3 = fused[:3]
    if top3:
        mean_conf = sum(r.get("confidence", 0.0) for r in top3) / len(top3)
        if mean_conf < 0.45:
            retry = await _research_retry(normalized, intent, max_res, top_docs=top3)
            if retry:
                retry_enriched = _enrich_results(retry, normalized)
                fused = ranking.rrf_fuse([fused, retry_enriched], weights=[2.0, 1.0])
                fused = _enrich_results(fused, normalized)

    # M6: Diversity — limit per-domain results
    fused = _diversify_results(fused)

    # Compress and clean up (M1: weighted budget)
    fused = compression.compress_results(
        fused,
        normalized,
        total_budget=max_tokens,
        content_key="content",
    )
    for r in fused:
        r.pop("full_content", None)

    return _serialize(
        {
            "query": raw_query,
            "sub_queries": sub_queries,
            "depth": depth,
            "result_count": len(fused),
            "results": fused[:max_res],
        }
    )


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
