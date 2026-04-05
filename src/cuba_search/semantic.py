"""Semantic similarity via model2vec static embeddings.

Uses potion-base-8M (MTEB SOTA for static embeddings):
- 8M params, 256 dims, 7.5MB on disk
- 500× faster than BERT (Tulkens & van Dongen 2024)
- No PyTorch — numpy only at runtime

M4: Batch encoding + sentence-aware snippet selection.
CC: all functions ≤ 5.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("cuba-search.semantic")

# ── Lazy model singleton ───────────────────────────────────────────
_MODEL_NAME: str = "minishlab/potion-base-8M"
_model: Any = None

# Max sentences to scan when selecting best snippet (performance cap)
_SNIPPET_SCAN_LIMIT: int = 20
# Sentence window size around best match
_SNIPPET_WINDOW: int = 3


def _load_model() -> Any:
    """Load model2vec model (lazy, thread-safe via GIL).

    Returns:
        StaticModel instance.
    """
    global _model  # noqa: PLW0603
    if _model is not None:
        return _model

    from model2vec import StaticModel  # type: ignore[import-untyped,import-not-found]

    logger.info("Loading model2vec: %s", _MODEL_NAME)
    _model = StaticModel.from_pretrained(_MODEL_NAME)
    logger.info("model2vec loaded — dims=%d", _model.dim)
    return _model


def embed(text: str) -> np.ndarray:
    """Embed text into 256-dim normalized vector.

    Args:
        text: Input text to embed.

    Returns:
        Normalized numpy array of shape (256,).
    """
    model = _load_model()
    vectors = model.encode([text])
    vec = vectors[0]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two pre-normalized vectors.

    Args:
        vec_a: First vector (normalized).
        vec_b: Second vector (normalized).

    Returns:
        Similarity in [-1, 1].
    """
    return float(np.dot(vec_a, vec_b))


def _best_snippet(content: str, query_vec: np.ndarray) -> str:
    """Return the sentence window most semantically similar to the query.

    Replaces arbitrary content[:200] truncation with an informed selection.
    Scans up to _SNIPPET_SCAN_LIMIT sentences, returns ±1 sentence window
    around the best match.

    Args:
        content: Full document content.
        query_vec: Pre-computed normalized query embedding.

    Returns:
        Selected sentence window (≤ _SNIPPET_WINDOW sentences).
    """
    from cuba_search.compression import split_sentences  # avoid circular at import

    sentences = split_sentences(content)
    if len(sentences) <= _SNIPPET_WINDOW:
        return content[:500]

    model = _load_model()
    candidates = sentences[:_SNIPPET_SCAN_LIMIT]
    raw_vecs = model.encode(candidates)

    best_idx, best_sim = 0, -1.0
    for i, sv in enumerate(raw_vecs):
        norm = float(np.linalg.norm(sv))
        if norm > 0:
            sim = float(np.dot(query_vec, sv / norm))
            if sim > best_sim:
                best_sim, best_idx = sim, i

    start = max(0, best_idx - 1)
    end = min(len(sentences), best_idx + _SNIPPET_WINDOW - 1)
    return " ".join(sentences[start:end])


def semantic_rerank(
    query: str,
    results: list[dict[str, Any]],
    content_key: str = "content",
) -> list[dict[str, Any]]:
    """Add semantic similarity scores to search results.

    M4: Uses batch encoding (single model.encode() call) and sentence-aware
    snippet selection instead of arbitrary content[:200] truncation.

    Args:
        query: Search query.
        results: Search results with content.
        content_key: Key containing text content.

    Returns:
        Results with 'semantic_score' field added.
    """
    if not results:
        return []

    query_vec = embed(query)

    # Select best snippet per result (sentence-aware, not [:200])
    snippets = [
        _best_snippet(r.get(content_key, ""), query_vec) if r.get(content_key) else ""
        for r in results
    ]

    # Batch encode all snippets in a single model call
    model = _load_model()
    raw_vecs = model.encode(snippets)

    scored = []
    for r, sv in zip(results, raw_vecs, strict=True):
        if not r.get(content_key):
            scored.append({**r, "semantic_score": 0.0})
            continue
        norm = float(np.linalg.norm(sv))
        doc_vec = sv / norm if norm > 0 else sv
        sim = cosine_similarity(query_vec, doc_vec)
        scored.append({**r, "semantic_score": round(max(0.0, sim), 4)})

    return scored
