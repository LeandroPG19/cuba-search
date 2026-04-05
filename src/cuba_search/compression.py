"""V5: Compression — BM25 sentence scoring for extractive summarization.

Reuses the same BM25 algorithm from ranking.py at sentence level.
No TextRank graph — simpler and faster.

M14: Contextual coherence — EXIT-inspired (ACL Findings 2025, arXiv:2412.12559).
After BM25 selects top sentences, the next sentence is included when it starts
with a context connector (pronoun/demonstrative), preserving reference chains
like "FastAPI uses Pydantic. It validates all inputs automatically."
CC: all functions ≤ 5.
"""

import re
from typing import Any

from cuba_search.partitioning import distribute_budget
from cuba_search.ranking import bm25_score

# Sentence boundary regex (handles abbreviations somewhat)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u024F])")

# M14: Context connectors — sentences starting with these reference prior content.
# Including them preserves pronoun chains and demonstrative references.
_CONTEXT_CONNECTOR_RE = re.compile(
    r"^(?:It|Its|They|Their|These|This|That|Such|The same|Both|"
    r"However|Therefore|Thus|Hence|As a result|Consequently|"
    r"Additionally|Furthermore|Moreover|In addition)\b",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex boundary detection.

    Args:
        text: Input text.

    Returns:
        List of sentences (non-empty, stripped).
    """
    sentences = _SENTENCE_RE.split(text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _add_context_neighbors(
    selected: list[tuple[int, float, str]],
    sentences: list[str],
) -> list[tuple[int, float, str]]:
    """Add immediately-following context-connector sentences to the selection.

    M14: EXIT (ACL Findings 2025) shows that preserving inter-sentence
    reference chains (pronouns, demonstratives, discourse markers) improves
    extractive compression quality for QA tasks. When sentence i+1 starts
    with a connector word, it likely references sentence i — include it.

    Args:
        selected: Already-selected (idx, score, text) tuples.
        sentences: All sentences in original order.

    Returns:
        Extended selection with context neighbors added.
    """
    selected_indices = {idx for idx, _, _ in selected}
    additions: list[tuple[int, float, str]] = []
    for idx, score, _ in selected:
        next_idx = idx + 1
        if next_idx >= len(sentences) or next_idx in selected_indices:
            continue
        if _CONTEXT_CONNECTOR_RE.match(sentences[next_idx]):
            additions.append((next_idx, score * 0.9, sentences[next_idx]))
            selected_indices.add(next_idx)
    return selected + additions


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
        return text[: max_tokens * 4]  # Rough char-to-token estimate

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

    # M14: Expand selection with context connector neighbors (EXIT-inspired)
    selected = _add_context_neighbors(selected, sentences)

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

    # Weighted budget: top results (higher relevance) get more tokens.
    # Uses linear decay from partitioning.distribute_budget() — already exists.
    budgets = distribute_budget(len(results), total_budget)
    compressed = []
    for result, budget in zip(results, budgets, strict=False):
        content = result.get(content_key, "")
        if not content:
            compressed.append(result)
            continue
        c = compress_to_budget(content, query, max_tokens=budget)
        compressed.append({**result, content_key: c})
    return compressed
