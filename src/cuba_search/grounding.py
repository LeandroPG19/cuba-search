"""Grounding: fact validation, contradiction detection, claim density.

Adopted techniques from Cuba MCPs:
- Negation patterns: cuba-memorys/search.py:34-40
- Claim density patterns: cuba-thinking/quality-metrics.service.ts:189-199
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
    re.compile(r"\b\d+(?:\.\d+)?%"),                                    # Percentages
    re.compile(r"\b\d{2,}"),                                            # Numbers ≥ 2 digits (200ms, 1024, etc)
    re.compile(r"\b(?:always|never|all|none|every|must)\b", re.IGNORECASE),  # Absolutes
    re.compile(r"\b(?:proves?|confirms?|demonstrates?|shows?)\b", re.IGNORECASE),  # Causal
]


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
        r_copy = {**r, "has_contradiction_markers": has_negation(content)}
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


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two term sets.

    Args:
        set_a: First set of terms.
        set_b: Second set of terms.

    Returns:
        Jaccard index in [0, 1].
    """
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


def _compute_avg_agreement(
    idx: int,
    term_sets: list[set[str]],
) -> float:
    """Compute average Jaccard agreement of one item vs all others.

    Args:
        idx: Index of the item.
        term_sets: All term sets.

    Returns:
        Average Jaccard index against all other items.
    """
    terms_i = term_sets[idx]
    if not terms_i:
        return 0.0
    scores = [
        _jaccard_similarity(terms_i, term_sets[j])
        for j in range(len(term_sets))
        if j != idx and term_sets[j]
    ]
    return sum(scores) / len(scores) if scores else 0.0


def cross_source_agreement(
    results: list[dict[str, Any]],
    content_key: str = "content",
) -> list[dict[str, Any]]:
    """Estimate agreement between sources using shared term overlap.

    Simple Jaccard-like metric: shared unique terms / total unique terms.

    Args:
        results: Search results with content.
        content_key: Key containing text content.

    Returns:
        Results with 'agreement_score' added.
    """
    if len(results) < 2:
        return [{**r, "agreement_score": 0.0} for r in results]

    term_sets = [
        set(r.get(content_key, "").lower().split()) for r in results
    ]

    return [
        {**r, "agreement_score": round(_compute_avg_agreement(i, term_sets), 4)}
        for i, r in enumerate(results)
    ]
