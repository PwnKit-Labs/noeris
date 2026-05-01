"""LRU cache for FP8 prepacked (N x K) weights."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class PrepackCacheStats:
    hits: int
    misses: int
    evictions: int
    size: int


class Fp8PrepackCache(Generic[T]):
    """Small LRU cache keyed by logical weight id.

    The cache stores already-prepacked tensors/objects (typically NxK FP8
    weights) so repeated calls can skip transpose+contiguous work.
    """

    def __init__(self, *, max_items: int = 64) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be > 0")
        self.max_items = max_items
        self._items: OrderedDict[Hashable, T] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: Hashable) -> T | None:
        value = self._items.get(key)
        if value is None:
            self._misses += 1
            return None
        self._hits += 1
        self._items.move_to_end(key)
        return value

    def get_or_create(self, key: Hashable, factory: Callable[[], T]) -> tuple[T, bool]:
        value = self.get(key)
        if value is not None:
            return value, True

        created = factory()
        self._items[key] = created
        self._items.move_to_end(key)
        if len(self._items) > self.max_items:
            self._items.popitem(last=False)
            self._evictions += 1
        return created, False

    def clear(self) -> None:
        self._items.clear()

    def stats(self) -> PrepackCacheStats:
        return PrepackCacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            size=len(self._items),
        )
