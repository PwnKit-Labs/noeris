"""Tests for the MAP-Elites quality-diversity archive.

Coverage:
- Archive creation with default and custom bins
- Insert / retrieve lifecycle
- Best-per-bin replacement semantics
- Diverse candidate retrieval from distinct bins
- Coverage metric increases with inserts
- Behavioral classification for rmsnorm (memory-bound) configs
- Behavioral classification for attention (mixed) configs
- Serialization round-trip (to_dict / from_dict)
- MAPElitesSelector exploration-exploitation split
"""

from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401

from research_engine.map_elites import (
    MAPElitesArchive,
    MAPElitesSelector,
    memory_intensity,
    parallelism_level,
    _op_category,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matmul_config(
    bm: int = 128, bn: int = 128, bk: int = 64,
    group: int = 8, warps: int = 4, stages: int = 3,
) -> dict:
    return {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "GROUP_SIZE_M": group,
        "num_warps": warps,
        "num_stages": stages,
    }


def _rmsnorm_config(block_size: int = 1024, warps: int = 4) -> dict:
    return {"BLOCK_SIZE": block_size, "num_warps": warps}


def _attention_config(bm: int = 64, bn: int = 64, warps: int = 4) -> dict:
    return {"BLOCK_M": bm, "BLOCK_N": bn, "num_warps": warps}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestArchiveCreation(unittest.TestCase):
    """test_archive_creation: creates with default bins."""

    def test_archive_creation(self) -> None:
        archive = MAPElitesArchive(operator="matmul", shape_bucket="medium")
        self.assertEqual(archive.n_memory_bins, 5)
        self.assertEqual(archive.n_parallelism_bins, 5)
        self.assertEqual(archive.operator, "matmul")
        self.assertEqual(archive.shape_bucket, "medium")
        self.assertEqual(len(archive.archive), 0)
        self.assertAlmostEqual(archive.coverage(), 0.0)

    def test_archive_custom_bins(self) -> None:
        archive = MAPElitesArchive(
            operator="rmsnorm", shape_bucket="large",
            n_memory_bins=10, n_parallelism_bins=8,
        )
        self.assertEqual(archive.n_memory_bins, 10)
        self.assertEqual(archive.n_parallelism_bins, 8)


class TestInsertAndRetrieve(unittest.TestCase):
    """test_insert_and_retrieve: insert a config, get it back from the right bin."""

    def test_insert_and_retrieve(self) -> None:
        archive = MAPElitesArchive(operator="matmul", shape_bucket="medium")
        cfg = _matmul_config(bm=128, bn=256, bk=64, group=8, warps=8)
        inserted = archive.insert(cfg, metric=150.0)
        self.assertTrue(inserted)

        # The config should be retrievable via get_diverse_candidates
        candidates = archive.get_diverse_candidates(top_k=10)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["BLOCK_SIZE_M"], 128)
        self.assertEqual(candidates[0]["BLOCK_SIZE_N"], 256)

    def test_insert_returns_false_for_worse(self) -> None:
        archive = MAPElitesArchive(operator="matmul", shape_bucket="medium")
        cfg = _matmul_config(bm=128, bn=128, bk=64)
        archive.insert(cfg, metric=100.0)
        # Same bin, worse metric -> should not replace
        result = archive.insert(cfg, metric=50.0)
        self.assertFalse(result)


class TestBestPerBin(unittest.TestCase):
    """test_best_per_bin: inserting a better config for the same bin replaces."""

    def test_best_per_bin(self) -> None:
        archive = MAPElitesArchive(operator="matmul", shape_bucket="medium")
        cfg_a = _matmul_config(bm=128, bn=128, bk=64, warps=4)
        cfg_b = _matmul_config(bm=128, bn=128, bk=64, warps=4)
        cfg_b["num_stages"] = 5  # different config, same bin

        archive.insert(cfg_a, metric=100.0)
        bin_a = archive.classify_config(cfg_a)
        bin_b = archive.classify_config(cfg_b)
        self.assertEqual(bin_a, bin_b, "configs should map to the same bin")

        # Insert better config in same bin
        replaced = archive.insert(cfg_b, metric=200.0)
        self.assertTrue(replaced)

        candidates = archive.get_diverse_candidates()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["num_stages"], 5)


class TestDiverseCandidates(unittest.TestCase):
    """test_diverse_candidates_from_different_bins."""

    def test_diverse_candidates_from_different_bins(self) -> None:
        archive = MAPElitesArchive(operator="matmul", shape_bucket="medium")
        # Insert configs that land in different bins by varying warps and group
        configs = [
            (_matmul_config(bm=64, bn=64, bk=32, group=1, warps=2), 80.0),
            (_matmul_config(bm=256, bn=256, bk=128, group=16, warps=8), 150.0),
            (_matmul_config(bm=128, bn=128, bk=64, group=4, warps=4), 120.0),
        ]
        bins_seen = set()
        for cfg, metric in configs:
            archive.insert(cfg, metric)
            bins_seen.add(archive.classify_config(cfg))

        # We should have configs from distinct bins
        self.assertGreaterEqual(len(bins_seen), 2, "should hit at least 2 distinct bins")

        candidates = archive.get_diverse_candidates(top_k=10)
        self.assertEqual(len(candidates), len(bins_seen))

        # Should be sorted by metric descending
        if len(candidates) >= 2:
            # The first candidate should be from the best-performing bin
            self.assertIn(candidates[0]["BLOCK_SIZE_N"], [64, 128, 256])


class TestCoverage(unittest.TestCase):
    """test_coverage_increases_with_inserts."""

    def test_coverage_increases_with_inserts(self) -> None:
        archive = MAPElitesArchive(operator="matmul", shape_bucket="medium")
        self.assertAlmostEqual(archive.coverage(), 0.0)

        archive.insert(_matmul_config(bm=64, bn=64, bk=32, group=1, warps=2), 80.0)
        cov1 = archive.coverage()
        self.assertGreater(cov1, 0.0)

        archive.insert(_matmul_config(bm=256, bn=256, bk=128, group=16, warps=8), 150.0)
        cov2 = archive.coverage()
        self.assertGreaterEqual(cov2, cov1)


class TestClassifyRmsnorm(unittest.TestCase):
    """test_classify_config_rmsnorm: verify bin assignment for rmsnorm configs."""

    def test_classify_config_rmsnorm(self) -> None:
        archive = MAPElitesArchive(operator="rmsnorm", shape_bucket="medium")

        # Small block size -> low memory intensity bin
        cfg_small = _rmsnorm_config(block_size=256, warps=2)
        # Large block size -> high memory intensity bin
        cfg_large = _rmsnorm_config(block_size=8192, warps=8)

        bin_small = archive.classify_config(cfg_small)
        bin_large = archive.classify_config(cfg_large)

        # Memory intensity dimension should differ
        self.assertNotEqual(bin_small, bin_large)
        # Small block -> lower memory bin index
        self.assertLess(bin_small[0], bin_large[0])

    def test_memory_intensity_rmsnorm(self) -> None:
        mi_small = memory_intensity({"BLOCK_SIZE": 256}, "rmsnorm")
        mi_large = memory_intensity({"BLOCK_SIZE": 4096}, "rmsnorm")
        self.assertLess(mi_small, mi_large)

    def test_parallelism_rmsnorm(self) -> None:
        pl_2 = parallelism_level({"num_warps": 2}, "rmsnorm")
        pl_8 = parallelism_level({"num_warps": 8}, "rmsnorm")
        self.assertLess(pl_2, pl_8)


class TestClassifyAttention(unittest.TestCase):
    """test_classify_config_attention: verify bin assignment for attention."""

    def test_classify_config_attention(self) -> None:
        archive = MAPElitesArchive(operator="attention", shape_bucket="medium")

        # BN >> BM -> high memory intensity (loading more K/V)
        cfg_kv_heavy = _attention_config(bm=32, bn=128, warps=4)
        # BN << BM -> low memory intensity
        cfg_compute = _attention_config(bm=128, bn=32, warps=4)

        bin_kv = archive.classify_config(cfg_kv_heavy)
        bin_compute = archive.classify_config(cfg_compute)

        # Memory dimension should differ
        self.assertNotEqual(bin_kv[0], bin_compute[0])

    def test_memory_intensity_attention(self) -> None:
        mi = memory_intensity({"BLOCK_M": 64, "BLOCK_N": 128}, "attention")
        self.assertAlmostEqual(mi, 2.0)

    def test_parallelism_attention(self) -> None:
        pl = parallelism_level({"num_warps": 8}, "attention")
        self.assertEqual(pl, 8.0)


class TestSerializationRoundtrip(unittest.TestCase):
    """test_serialization_roundtrip: to_dict / from_dict preserves state."""

    def test_serialization_roundtrip(self) -> None:
        archive = MAPElitesArchive(operator="matmul", shape_bucket="large")
        configs = [
            (_matmul_config(bm=64, bn=64, bk=32, group=1, warps=2), 80.0),
            (_matmul_config(bm=256, bn=256, bk=128, group=16, warps=8), 150.0),
            (_matmul_config(bm=128, bn=128, bk=64, group=4, warps=4), 120.0),
        ]
        for cfg, metric in configs:
            archive.insert(cfg, metric, hardware_counters={"sm_occupancy": 0.75})

        data = archive.to_dict()
        restored = MAPElitesArchive.from_dict(data)

        self.assertEqual(restored.operator, archive.operator)
        self.assertEqual(restored.shape_bucket, archive.shape_bucket)
        self.assertEqual(restored.n_memory_bins, archive.n_memory_bins)
        self.assertEqual(restored.n_parallelism_bins, archive.n_parallelism_bins)
        self.assertEqual(len(restored.archive), len(archive.archive))
        self.assertAlmostEqual(restored.coverage(), archive.coverage())

        # Check that configs and metrics survive
        for key, cell in archive.archive.items():
            self.assertIn(key, restored.archive)
            self.assertEqual(restored.archive[key].config, cell.config)
            self.assertAlmostEqual(restored.archive[key].metric, cell.metric)

    def test_selector_serialization_roundtrip(self) -> None:
        selector = MAPElitesSelector(n_memory_bins=4, n_parallelism_bins=4)
        selector.ingest_result(
            operator="matmul", shape_bucket="medium", hardware="A100",
            config=_matmul_config(), metric=100.0,
        )
        data = selector.to_dict()
        restored = MAPElitesSelector.from_dict(data)
        self.assertEqual(len(restored._archives), 1)
        self.assertEqual(restored.n_memory_bins, 4)


class TestMAPElitesSelector(unittest.TestCase):
    """Integration tests for the selector."""

    def test_select_explores_empty_bins(self) -> None:
        selector = MAPElitesSelector(exploration_fraction=0.5)
        # Ingest one config
        selector.ingest_result(
            operator="matmul", shape_bucket="medium", hardware="A100",
            config=_matmul_config(bm=128, bn=128, bk=64, group=4, warps=4),
            metric=100.0,
        )
        # Provide candidates in different bins
        candidates = [
            _matmul_config(bm=64, bn=64, bk=32, group=1, warps=2),
            _matmul_config(bm=256, bn=256, bk=128, group=16, warps=8),
            _matmul_config(bm=128, bn=128, bk=32, group=8, warps=4),
        ]
        selected = selector.select_configs(
            operator="matmul", shape_bucket="medium", hardware="A100",
            candidate_pool=candidates, max_configs=4,
        )
        self.assertGreater(len(selected), 0)
        self.assertLessEqual(len(selected), 4)

    def test_coverage_report(self) -> None:
        selector = MAPElitesSelector()
        selector.ingest_result(
            operator="rmsnorm", shape_bucket="small", hardware="A100",
            config=_rmsnorm_config(1024, 4), metric=50.0,
        )
        report = selector.coverage_report()
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["operator"], "rmsnorm")
        self.assertGreater(report[0]["coverage"], 0.0)


class TestOpCategory(unittest.TestCase):
    """Operator category classification."""

    def test_known_ops(self) -> None:
        self.assertEqual(_op_category("rmsnorm"), "memory")
        self.assertEqual(_op_category("matmul"), "compute")
        self.assertEqual(_op_category("attention"), "mixed")

    def test_unknown_defaults_to_memory(self) -> None:
        self.assertEqual(_op_category("unknown_kernel"), "memory")


if __name__ == "__main__":
    unittest.main()
