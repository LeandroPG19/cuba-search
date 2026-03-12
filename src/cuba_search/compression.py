"""V5: Compression — BM25 sentence scoring for extractive summarization.

Reuses the same BM25 algorithm from ranking.py at sentence level.
No TextRank graph — simpler and faster.
CC: all functions ≤ 5.
"""
import re
from typing import Any

from cuba_search.ranking import bm25_score

# Sentence boundary regex (handles abbreviations somewhat)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u024F])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex boundary detection.

    Args:
        text: Input text.

    Returns:
        List of sentences (non-empty, stripped).
    """
    sentences = _SENTENCE_RE.split(text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def compress_to_budget(
    text: str,
    query: str,
    max_tokens: int = 800,
    max_sentences: int = 10,
) -> str:
    """Extract most relevant sentences that fit within token budget.

    Uses BM25 at sentence level to score relevance.
    Maintains original sentence order for coherence.

    Args:
        text: Full text to compress.
        query: Search query for relevance scoring.
        max_tokens: Token budget (estimated as words × 1.3).
        max_sentences: Maximum number of sentences to include.

    Returns:
        Compressed text within token budget.
    """
    sentences = split_sentences(text)
    if not sentences:
        return text[:max_tokens * 4]  # Rough char-to-token estimate

    query_terms = query.lower().split()

    # Score each sentence with BM25
    all_terms = [s.lower().split() for s in sentences]
    doc_freq: dict[str, int] = {}
    for terms in all_terms:
        for t in set(terms):
            doc_freq[t] = doc_freq.get(t, 0) + 1

    total = len(sentences)
    avg_len = sum(len(t) for t in all_terms) / max(total, 1)

    scored: list[tuple[int, float, str]] = []
    for idx, (sentence, terms) in enumerate(zip(sentences, all_terms, strict=True)):
        s = bm25_score(query_terms, terms, doc_freq, total, avg_len)
        scored.append((idx, s, sentence))

    # Sort by score, take top N
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = scored[:max_sentences]

    # Re-sort by original position for coherence
    selected.sort(key=lambda x: x[0])

    # Build result within token budget
    result_parts: list[str] = []
    token_count = 0
    for _, _, sentence in selected:
        sentence_tokens = len(sentence.split())
        if token_count + sentence_tokens > max_tokens:
            break
        result_parts.append(sentence)
        token_count += sentence_tokens

    return " ".join(result_parts) if result_parts else sentences[0]


def compress_results(
    results: list[dict[str, Any]],
    query: str,
    total_budget: int = 3000,
    content_key: str = "content",
) -> list[dict[str, Any]]:
    """Compress a list of search results to fit total token budget.

    Distributes budget proportionally across results.

    Args:
        results: Search results with content.
        query: Search query for relevance scoring.
        total_budget: Total token budget for all results.
        content_key: Key containing text content.

    Returns:
        Results with compressed content.
    """
    if not results:
        return []

    per_result = total_budget // max(len(results), 1)
    compressed = []
    for result in results:
        content = result.get(content_key, "")
        if not content:
            compressed.append(result)
            continue
        c = compress_to_budget(content, query, max_tokens=per_result)
        compressed.append({**result, content_key: c})
    return compressed
