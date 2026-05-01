from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401

from research_engine.fp8_prepack_cache import Fp8PrepackCache


class Fp8PrepackCacheTests(unittest.TestCase):
    def test_get_or_create_hit_miss_counts(self) -> None:
        cache: Fp8PrepackCache[int] = Fp8PrepackCache(max_items=2)
        v1, hit1 = cache.get_or_create("a", lambda: 1)
        v2, hit2 = cache.get_or_create("a", lambda: 2)
        self.assertEqual(v1, 1)
        self.assertEqual(v2, 1)
        self.assertFalse(hit1)
        self.assertTrue(hit2)
        stats = cache.stats()
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.misses, 1)

    def test_lru_eviction(self) -> None:
        cache: Fp8PrepackCache[int] = Fp8PrepackCache(max_items=2)
        cache.get_or_create("a", lambda: 1)
        cache.get_or_create("b", lambda: 2)
        cache.get_or_create("c", lambda: 3)  # evict a
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), 2)
        self.assertEqual(cache.get("c"), 3)
        self.assertEqual(cache.stats().evictions, 1)

    def test_invalid_capacity_raises(self) -> None:
        with self.assertRaises(ValueError):
            Fp8PrepackCache(max_items=0)


if __name__ == "__main__":
    unittest.main()
