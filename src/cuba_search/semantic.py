"""Semantic similarity via model2vec static embeddings.

Uses potion-base-8M (MTEB SOTA for static embeddings):
- 8M params, 256 dims, 7.5MB on disk
- 500× faster than BERT (Tulkens & van Dongen 2024)
- No PyTorch — numpy only at runtime
CC: all functions ≤ 5.
"""
import logging
from typing import Any

import numpy as np

logger = logging.getLogger("cuba-search.semantic")

# ── Lazy model singleton ───────────────────────────────────────────
_MODEL_NAME: str = "minishlab/potion-base-8M"
_model: Any = None


def _load_model() -> Any:
    """Load model2vec model (lazy, thread-safe via GIL).

    Returns:
        StaticModel instance.
    """
    global _model  # noqa: PLW0603
    if _model is not None:
        return _model

    from model2vec import StaticModel  # type: ignore[import-untyped]

    logger.info("Loading model2vec: %s", _MODEL_NAME)
    _model = StaticModel.from_pretrained(_MODEL_NAME)
    logger.info("model2vec loaded — dims=%d", _model.dim)
    return _model


def embed(text: str) -> np.ndarray:
    """Embed text into 256-dim vector.

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


def semantic_rerank(
    query: str,
    results: list[dict[str, Any]],
    content_key: str = "content",
) -> list[dict[str, Any]]:
    """Add semantic similarity scores to search results.

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
    scored = []
    for r in results:
        content = r.get(content_key, "")
        if not content:
            scored.append({**r, "semantic_score": 0.0})
            continue
        # Use title + first 200 chars for speed
        snippet = content[:200]
        doc_vec = embed(snippet)
        sim = cosine_similarity(query_vec, doc_vec)
        scored.append({**r, "semantic_score": round(max(0.0, sim), 4)})

    return scored
