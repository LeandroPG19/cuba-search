"""V1: Query intelligence — decomposition, expansion, normalization.

Handles query preprocessing before search execution.
V16: synonym expansion for improved BM25 recall.
CC: all functions ≤ 5.
"""
import json
import re
import unicodedata
from pathlib import Path

# Operators passed directly to SearXNG
_OPERATOR_RE = re.compile(
    r"\b(?:site|filetype|intitle|inurl):\S+", re.IGNORECASE,
)

# Stopwords (minimal set — just connectors, not domain terms)
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "and", "or", "but", "not", "it", "this", "that",
    "el", "la", "los", "las", "de", "en", "un", "una", "es",
    "y", "o", "no", "por", "para", "con", "del", "al",
})

# Intent keywords for classification (V3)
_NAVIGATIONAL_PATTERNS = re.compile(
    r"\b(go to|official|homepage|login|sign in|download)\b", re.IGNORECASE,
)
_ACADEMIC_PATTERNS = re.compile(
    r"\b(paper|arxiv|doi|journal|research|study|thesis|citation)\b",
    re.IGNORECASE,
)
_CODE_PATTERNS = re.compile(
    r"\b(github|npm|pypi|crate|library|package|module|api|sdk|docs)\b",
    re.IGNORECASE,
)


def normalize_query(query: str) -> str:
    """Normalize query text: Unicode NFKC, lowercase, strip excess whitespace.

    Args:
        query: Raw user query.

    Returns:
        Normalized query string.
    """
    text = unicodedata.normalize("NFKC", query)
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)


def extract_operators(query: str) -> tuple[str, list[str]]:
    """Extract search operators (site:, filetype:) from query.

    Args:
        query: Query that may contain operators.

    Returns:
        Tuple of (clean query without operators, list of operators).
    """
    operators = _OPERATOR_RE.findall(query)
    clean = _OPERATOR_RE.sub("", query).strip()
    clean = re.sub(r"\s+", " ", clean)
    return clean, operators


def remove_stopwords(text: str) -> str:
    """Remove stopwords from text for cache key normalization.

    Args:
        text: Input text.

    Returns:
        Text with stopwords removed.
    """
    words = text.split()
    return " ".join(w for w in words if w.lower() not in _STOPWORDS)


def detect_intent(query: str) -> str:
    """Classify query intent using keyword heuristics.

    Based on Broder (2002) taxonomy: navigational/informational/academic/code.

    Args:
        query: Normalized query.

    Returns:
        Intent string: 'navigational', 'academic', 'code', or 'informational'.
    """
    if _NAVIGATIONAL_PATTERNS.search(query):
        return "navigational"
    if _ACADEMIC_PATTERNS.search(query):
        return "academic"
    if _CODE_PATTERNS.search(query):
        return "code"
    return "informational"


def decompose_query(query: str) -> list[str]:
    """Decompose complex query into atomic sub-queries.

    Splits on conjunctions and semicolons. Only decomposes if
    result has ≥ 2 meaningful sub-queries.

    Args:
        query: Normalized query.

    Returns:
        List of atomic sub-queries (or single-element list if not decomposable).
    """
    parts = re.split(r"\b(?:and|y|además|also)\b|[;]", query, flags=re.IGNORECASE)
    sub_queries = [p.strip() for p in parts if len(p.strip()) > 5]
    if len(sub_queries) >= 2:
        return sub_queries
    return [query]


def estimate_tokens(text: str) -> int:
    """Estimate token count using word-split heuristic.

    Accurate to ±10% vs tiktoken, zero dependencies.

    Args:
        text: Input text.

    Returns:
        Estimated token count (minimum 1).
    """
    return max(1, int(len(text.split()) * 1.3))


# ── V16: Synonym expansion ─────────────────────────────────────────
_SYNONYM_MAP_PATH = Path(__file__).parent / "data" / "synonyms.json"
_synonym_map: dict[str, list[str]] | None = None


def _load_synonym_map() -> dict[str, list[str]]:
    """Load synonym map from JSON file (lazy).

    Returns:
        Dict mapping terms to their synonyms.
    """
    global _synonym_map  # noqa: PLW0603
    if _synonym_map is not None:
        return _synonym_map
    try:
        _synonym_map = json.loads(_SYNONYM_MAP_PATH.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        _synonym_map = {}
    return _synonym_map


def expand_query(query: str, max_expansions: int = 2) -> str:
    """Expand query terms with synonyms for improved recall.

    Adds up to max_expansions synonym terms per original word.

    Args:
        query: Normalized query string.
        max_expansions: Max synonyms to add per term.

    Returns:
        Expanded query string.
    """
    syn_map = _load_synonym_map()
    words = query.lower().split()
    expanded: list[str] = list(words)

    for word in words:
        synonyms = syn_map.get(word, [])
        added = 0
        for syn in synonyms:
            if syn not in expanded and added < max_expansions:
                expanded.append(syn)
                added += 1

    return " ".join(expanded)

