"""V1: RRF ranking + BM25 re-ranking + V14: info density + confidence scoring.

Hand-implemented BM25 (~40 LOC, no scikit-learn).
H/H_max from cuba-memorys/hebbian.py:79-95.
Multi-signal confidence from cuba-memorys/search.py:72-109.
CC: all functions ≤ 7.
"""
import math
from collections import Counter
from typing import Any

# ── BM25 Parameters (Robertson & Zaragoza 2009) ────────────────────
BM25_K1: float = 1.2   # Term saturation
BM25_B: float = 0.75   # Length normalization

# ── RRF Constant (Cormack et al. 2009) ─────────────────────────────
RRF_K: int = 60


def bm25_score(
    query_terms: list[str],
    doc_terms: list[str],
    doc_freq: dict[str, int],
    total_docs: int,
    avg_doc_len: float,
) -> float:
    """Compute BM25 relevance score for a document against a query.

    Args:
        query_terms: Tokenized query words.
        doc_terms: Tokenized document words.
        doc_freq: Number of documents containing each term.
        total_docs: Total number of documents in collection.
        avg_doc_len: Average document length in collection.

    Returns:
        BM25 score (higher = more relevant).
    """
    score = 0.0
    doc_len = len(doc_terms)
    term_counts = Counter(doc_terms)

    for qt in query_terms:
        tf = term_counts.get(qt, 0)
        if tf == 0:
            continue
        df = doc_freq.get(qt, 0)
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
        numerator = tf * (BM25_K1 + 1.0)
        denominator = tf + BM25_K1 * (
            1.0 - BM25_B + BM25_B * doc_len / max(avg_doc_len, 1.0)
        )
        score += idf * (numerator / denominator)
    return score


def bm25_rank(
    query: str,
    documents: list[dict[str, Any]],
    text_key: str = "content",
) -> list[dict[str, Any]]:
    """Rank documents by BM25 relevance to query.

    Args:
        query: Search query string.
        documents: List of result dicts with text content.
        text_key: Key in dict containing text to score.

    Returns:
        Documents sorted by BM25 score (descending), with 'bm25_score' added.
    """
    if not documents:
        return []

    query_terms = query.lower().split()
    all_doc_terms = [
        d.get(text_key, "").lower().split() for d in documents
    ]

    # Compute document frequencies
    doc_freq: dict[str, int] = {}
    for terms in all_doc_terms:
        for term in set(terms):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    total_docs = len(documents)
    avg_len = sum(len(t) for t in all_doc_terms) / max(total_docs, 1)

    scored = []
    for doc, terms in zip(documents, all_doc_terms, strict=True):
        s = bm25_score(query_terms, terms, doc_freq, total_docs, avg_len)
        scored.append({**doc, "bm25_score": round(s, 4)})

    scored.sort(key=lambda d: d["bm25_score"], reverse=True)
    return scored


def rrf_fuse(
    signal_rankings: list[list[dict[str, Any]]],
    id_key: str = "url",
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion across N ranked result lists.

    Proven in cuba-memorys/search.py, Cormack et al. 2009.

    Args:
        signal_rankings: Lists of ranked results from different sources.
        id_key: Key to use as unique result identifier.

    Returns:
        Fused results sorted by RRF score.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}

    for ranking in signal_rankings:
        for rank, item in enumerate(ranking):
            key = str(item.get(id_key, ""))
            scores[key] = scores.get(key, 0) + 1.0 / (RRF_K + rank + 1)
            if key not in items:
                items[key] = item

    sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)
    return [
        {**items[k], "rrf_score": round(scores[k], 4)} for k in sorted_keys
    ]


def information_density(text: str) -> float:
    """Shannon entropy ratio H/H_max — from cuba-memorys/hebbian.py:79-95.

    High = diverse information. Low = repetitive/boilerplate.

    Args:
        text: Input text to measure.

    Returns:
        Density ratio in [0, 1]. 0 = repetitive, 1 = maximally diverse.
    """
    words = text.lower().split()
    if len(words) < 2:
        return 0.0
    counts = Counter(words)
    n = len(words)
    h = -sum((c / n) * math.log2(c / n) for c in counts.values())
    h_max = math.log2(n)
    return h / h_max if h_max > 0 else 0.0


def compute_confidence(
    content_relevance: float,
    source_tier: float,
    freshness: float,
    info_density: float,
    cross_agreement: float,
    semantic_score: float = 0.0,
) -> tuple[float, str]:
    """Multi-signal weighted confidence — from cuba-memorys/search.py:72-109.

    Args:
        content_relevance: BM25/relevance score [0, 1].
        source_tier: Source credibility tier score [0, 1].
        freshness: Freshness score [0, 1] (1 = recent).
        info_density: H/H_max information density [0, 1].
        cross_agreement: Cross-source agreement [0, 1].
        semantic_score: model2vec cosine similarity [0, 1].

    Returns:
        Tuple of (confidence score, confidence level string).
    """
    score = (
        0.25 * content_relevance
        + 0.20 * source_tier
        + 0.20 * semantic_score
        + 0.15 * freshness
        + 0.10 * info_density
        + 0.10 * cross_agreement
    )
    score = max(0.0, min(1.0, score))

    if score >= 0.8:
        level = "high"
    elif score >= 0.5:
        level = "medium"
    elif score >= 0.3:
        level = "low"
    else:
        level = "unknown"

    return round(score, 4), level

