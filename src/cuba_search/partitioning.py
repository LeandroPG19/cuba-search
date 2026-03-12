"""V4: Partitioning — chunking and token budget management.

Splits content into chunks respecting natural boundaries.
CC: all functions ≤ 5.
"""

_DEFAULT_CHUNK_SIZE: int = 800   # tokens per chunk
_DEFAULT_OVERLAP: int = 50      # token overlap between chunks


def chunk_text(
    text: str,
    max_tokens: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_OVERLAP,
) -> list[str]:
    """Split text into chunks at paragraph/sentence boundaries.

    Tries paragraph boundaries first, falls back to sentence,
    then word-level splitting.

    Args:
        text: Text to chunk.
        max_tokens: Maximum tokens per chunk.
        overlap: Token overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text.strip():
        return []

    # Estimate if text fits in single chunk
    words = text.split()
    est_tokens = int(len(words) * 1.3)
    if est_tokens <= max_tokens:
        return [text]

    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_words: list[str] = []
    current_count = 0

    for para in paragraphs:
        para_words = para.split()
        para_tokens = int(len(para_words) * 1.3)

        if current_count + para_tokens > max_tokens and current_words:
            chunks.append(" ".join(current_words))
            # Keep overlap
            overlap_words = current_words[-overlap:] if overlap > 0 else []
            current_words = overlap_words
            current_count = int(len(current_words) * 1.3)

        current_words.extend(para_words)
        current_count += para_tokens

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def distribute_budget(
    n_results: int,
    total_budget: int = 3000,
    min_per_result: int = 200,
) -> list[int]:
    """Distribute token budget across N results.

    Top results get more budget (linear decay).

    Args:
        n_results: Number of results.
        total_budget: Total available tokens.
        min_per_result: Minimum tokens per result.

    Returns:
        List of token budgets per result.
    """
    if n_results <= 0:
        return []
    if n_results == 1:
        return [total_budget]

    # Linear decay: result 0 gets most, result N-1 gets least
    weights = [n_results - i for i in range(n_results)]
    total_weight = sum(weights)

    budgets = []
    for w in weights:
        budget = max(min_per_result, int(total_budget * w / total_weight))
        budgets.append(budget)

    return budgets
