"""V8: LRU cache with TTL — proven pattern from cuba-memorys/search.py.

Uses time.monotonic() for TTL (immune to clock adjustments).
Design: OrderedDict for O(1) LRU eviction + TTL expiry.
CC: all functions ≤ 4.
"""
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    """LRU cache with time-based expiry.

    Args:
        maxsize: Maximum number of cached entries.
        ttl: Time-to-live in seconds (default: 300s = 5 min).
    """

    __slots__ = ("_maxsize", "_ttl", "_store")

    def __init__(self, maxsize: int = 500, ttl: float = 300.0) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: OrderedDict[int, tuple[float, Any]] = OrderedDict()

    def get(self, key: int) -> Any | None:
        """Get value by key. Returns None if expired or missing.

        Args:
            key: Cache key (int hash).

        Returns:
            Cached value or None.
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: int, value: Any) -> None:
        """Set value with current timestamp.

        Args:
            key: Cache key (int hash).
            value: Value to cache.
        """
        if key in self._store:
            self._store.move_to_end(key)
        elif len(self._store) >= self._maxsize:
            self._store.popitem(last=False)
        self._store[key] = (time.monotonic(), value)

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    @property
    def size(self) -> int:
        """Current number of entries (including potentially expired)."""
        return len(self._store)


def normalize_cache_key(*parts: str) -> int:
    """Create normalized cache key from query parts.

    Lowercases, strips whitespace, sorts terms for order-independence.

    Args:
        *parts: Query components to hash.

    Returns:
        Integer hash for cache lookup.
    """
    normalized = " ".join(
        sorted(p.strip().lower() for p in parts if p.strip())
    )
    return hash(normalized)
