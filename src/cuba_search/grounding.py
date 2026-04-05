"""Grounding: fact validation, contradiction detection, claim density.

Adopted techniques from Cuba MCPs:
- Negation patterns: cuba-memorys/search.py:34-40
- Claim density patterns: cuba-thinking/quality-metrics.service.ts:189-199

M13: cross_source_agreement uses cosine semantic similarity instead of
     Jaccard lexical overlap — catches paraphrase agreement that Jaccard misses
     (e.g. "machine learning" vs "deep learning" cosine≈0.85, Jaccard≈0.0).
CC: all functions ≤ 5.
"""

import re
from typing import Any

# ── Negation patterns (bilingual EN/ES, from cuba-memorys) ─────────
_NEGATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bno\b|\bnot\b|\bnever\b|\bya no\b|\bno es\b", re.IGNORECASE),
    re.compile(r"\bcambió\s+de\b|\bchanged\s+from\b|\breplaced\b", re.IGNORECASE),
    re.compile(r"\ben vez de\b|\binstead of\b|\brather than\b", re.IGNORECASE),
    re.compile(r"\bremoved\b|\belimina\b|\bdeprecated\b|\bobsolete\b", re.IGNORECASE),
    re.compile(r"\bwas\b.*\bnow\b|\bantes\b.*\bahora\b", re.IGNORECASE),
]

# ── Claim patterns (ported from cuba-thinking, ~15 LOC) ────────────
_CLAIM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b\d+(?:\.\d+)?%"),  # Percentages
    re.compile(r"\b\d{2,}"),  # Numbers ≥ 2 digits (200ms, 1024, etc)
    re.compile(r"\b(?:always|never|all|none|every|must)\b", re.IGNORECASE),  # Absolutes
    re.compile(r"\b(?:proves?|confirms?|demonstrates?|shows?)\b", re.IGNORECASE),  # Causal
]

# ── Temporal change patterns (M9) ─────────────────────────────────
# More specific than has_negation() — detects deprecation/replacement signals.
_TEMPORAL_CHANGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:deprecated|obsolete|discontinued|removed)\b", re.IGNORECASE),
    re.compile(r"\breplaced\s+(?:by|with)\b", re.IGNORECASE),
    re.compile(r"\bno\s+longer\s+(?:supported|maintained|available|works?)\b", re.IGNORECASE),
    re.compile(r"\b(?:was|were)\s+(?:previously|formerly)\b", re.IGNORECASE),
]

# Max chars per document to embed for agreement scoring (speed cap)
_AGREEMENT_SNIPPET_LEN: int = 300


def has_temporal_change(text: str) -> bool:
    """Detect if text mentions deprecation, removal, or replacement of a feature.

    More precise than has_negation() — avoids false positives from common
    negation words ("not None", "does not support") by focusing on
    lifecycle-change vocabulary.

    Args:
        text: Text to check.

    Returns:
        True if temporal/lifecycle change pattern found.
    """
    return any(p.search(text) for p in _TEMPORAL_CHANGE_PATTERNS)


def has_negation(text: str) -> bool:
    """Check if text contains negation/contradiction markers.

    From cuba-memorys/search.py. Bilingual EN/ES.

    Args:
        text: Text to check.

    Returns:
        True if negation pattern found.
    """
    return any(p.search(text) for p in _NEGATION_PATTERNS)


def detect_contradictions(
    results: list[dict[str, Any]],
    content_key: str = "content",
) -> list[dict[str, Any]]:
    """Flag results that contain contradiction markers.

    Helps AI agents identify conflicting information sources.

    Args:
        results: Search results with content.
        content_key: Key containing text content.

    Returns:
        Results with 'has_contradiction_markers' flag added.
    """
    flagged = []
    for r in results:
        content = r.get(content_key, "")
        r_copy = {
            **r,
            "has_contradiction_markers": has_negation(content),
            # M9: Separate signal for deprecation/lifecycle changes — more
            # precise than has_negation for identifying outdated information.
            "has_temporal_change": has_temporal_change(content),
        }
        flagged.append(r_copy)
    return flagged


def count_claims(text: str) -> int:
    """Count verifiable claims/assertions in text.

    Ported from cuba-thinking/quality-metrics.service.ts.
    Identifies: percentages, large numbers, absolutes, causal claims.

    Args:
        text: Text to analyze.

    Returns:
        Number of verifiable claims detected.
    """
    total = 0
    for pattern in _CLAIM_PATTERNS:
        matches = pattern.findall(text)
        total += len(matches)
    return total


def claim_density(text: str) -> float:
    """Compute claims per sentence ratio.

    Higher density = more verifiable assertions = more valuable for research.

    Args:
        text: Text to analyze.

    Returns:
        Claims per sentence ratio (0.0 if no sentences).
    """
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.0
    claims = count_claims(text)
    return round(claims / len(sentences), 4)


def _build_cosine_matrix(texts: list[str]) -> "tuple[Any, list[bool]]":
    """Batch-encode texts and return L2-normalised cosine matrix + content mask.

    Returns:
        (sim_matrix [n×n ndarray], has_content [bool list])
    """
    import numpy as np

    from cuba_search.semantic import _load_model

    has_content = [bool(t.strip()) for t in texts]
    model = _load_model()
    raw_vecs = model.encode(texts)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    vecs = raw_vecs / norms
    return vecs @ vecs.T, has_content


def cross_source_agreement(
    results: list[dict[str, Any]],
    content_key: str = "content",
) -> list[dict[str, Any]]:
    """Estimate agreement between sources using semantic cosine similarity.

    M13: Replaces Jaccard lexical overlap with cosine similarity over
    model2vec embeddings (batch encoded). Cosine catches paraphrase agreement
    that Jaccard misses — e.g. "machine learning" vs "deep learning" have
    Jaccard≈0.0 but cosine≈0.85.

    Math: sim_matrix = V @ V.T where V = L2-normalized embedding matrix.
    O(n × 256) encode + O(n² × 256) multiply — fast for n ≤ 20 results.

    Args:
        results: Search results with content.
        content_key: Key containing text content.

    Returns:
        Results with 'agreement_score' added.
    """
    if len(results) < 2:
        return [{**r, "agreement_score": 0.0} for r in results]

    texts = [r.get(content_key, "")[:_AGREEMENT_SNIPPET_LEN] for r in results]
    sim_matrix, has_content = _build_cosine_matrix(texts)

    scored = []
    for i, r in enumerate(results):
        if not has_content[i]:
            scored.append({**r, "agreement_score": 0.0})
            continue
        other_sims = [
            float(sim_matrix[i, j]) for j in range(len(results)) if j != i and has_content[j]
        ]
        avg_sim = sum(other_sims) / len(other_sims) if other_sims else 0.0
        scored.append({**r, "agreement_score": round(max(0.0, avg_sim), 4)})

    return scored
