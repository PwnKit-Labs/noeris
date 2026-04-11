"""Tests for the Thompson-sampling BanditSelector.

Coverage:
- Posterior updates on new observations (alpha/beta accounting)
- Sampling preference for high-reward configs
- Graceful handling of an empty database (falls back to curated configs)
- Shape-bucket isolation (matmul:large results don't bleed into attention)
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests import _pathfix  # noqa: F401

from research_engine.bandit_selector import BanditSelector, _BetaArm, _PRIOR_ALPHA, _PRIOR_BETA
from research_engine.triton_kernels import (
    ConfigDatabase,
    TRITON_MATMUL_CURATED_CONFIGS,
    config_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path) -> ConfigDatabase:
    """Return a fresh, empty ConfigDatabase."""
    return ConfigDatabase(path=tmp_path / "db.json")


def _matmul_config(bm: int = 128, bn: int = 128, bk: int = 64) -> dict:
    return {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }


def _record_many(db: ConfigDatabase, configs_tflops: list[tuple[dict, float]],
                 *, operator: str = "matmul", bucket: str = "medium",
                 hardware: str = "A100") -> None:
    """Helper: record multiple (config, tflops) observations into one bucket."""
    for cfg, tflops in configs_tflops:
        db.record_result(
            shape={"M": 2048, "N": 2048, "K": 2048, "bucket": bucket},
            hardware=hardware,
            config=cfg,
            tflops=tflops,
            ms=1.0,
            correct=True,
            operator=operator,
            bucket=bucket,
            config_id_str=config_id(cfg),
        )


# ---------------------------------------------------------------------------
# _BetaArm unit tests
# ---------------------------------------------------------------------------

class BetaArmTests(unittest.TestCase):
    """Unit-test the Beta posterior arm in isolation."""

    def test_initial_prior(self) -> None:
        arm = _BetaArm(config_id="x", config={})
        self.assertEqual(arm.alpha, _PRIOR_ALPHA)
        self.assertEqual(arm.beta, _PRIOR_BETA)
        self.assertEqual(arm.total_observations, 0)

    def test_success_increments_alpha(self) -> None:
        arm = _BetaArm(config_id="x", config={})
        arm.update(success=True)
        self.assertAlmostEqual(arm.alpha, _PRIOR_ALPHA + 1.0)
        self.assertAlmostEqual(arm.beta, _PRIOR_BETA)
        self.assertEqual(arm.total_observations, 1)

    def test_failure_increments_beta(self) -> None:
        arm = _BetaArm(config_id="x", config={})
        arm.update(success=False)
        self.assertAlmostEqual(arm.alpha, _PRIOR_ALPHA)
        self.assertAlmostEqual(arm.beta, _PRIOR_BETA + 1.0)
        self.assertEqual(arm.total_observations, 1)

    def test_expected_reward_equals_alpha_over_total(self) -> None:
        arm = _BetaArm(config_id="x", config={})
        arm.update(success=True)
        arm.update(success=True)
        arm.update(success=False)
        expected = arm.alpha / (arm.alpha + arm.beta)
        self.assertAlmostEqual(arm.expected_reward, expected)

    def test_perfect_winner_has_high_expected_reward(self) -> None:
        arm = _BetaArm(config_id="x", config={})
        for _ in range(10):
            arm.update(success=True)
        self.assertGreater(arm.expected_reward, 0.8)

    def test_always_loser_has_low_expected_reward(self) -> None:
        arm = _BetaArm(config_id="x", config={})
        for _ in range(10):
            arm.update(success=False)
        self.assertLess(arm.expected_reward, 0.2)

    def test_sample_is_in_unit_interval(self) -> None:
        import numpy as np
        arm = _BetaArm(config_id="x", config={})
        arm.update(success=True)
        rng = np.random.default_rng(42)
        for _ in range(50):
            s = arm.sample(rng)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


# ---------------------------------------------------------------------------
# BanditSelector.load_from_database tests
# ---------------------------------------------------------------------------

class LoadFromDatabaseTests(unittest.TestCase):
    """Test that posteriors are built correctly from ConfigDatabase records."""

    def test_empty_database_leaves_no_arms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            selector = BanditSelector(seed=0)
            selector.load_from_database(db, operator="matmul", hardware="A100")
            self.assertEqual(len(selector._arms), 0)

    def test_single_record_creates_one_arm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            cfg = _matmul_config()
            _record_many(db, [(cfg, 150.0)])
            selector = BanditSelector(seed=0)
            selector.load_from_database(db, operator="matmul", hardware="A100")
            self.assertEqual(len(selector._arms), 1)
            cell = "matmul:medium:A100"
            self.assertIn(cell, selector._arms)
            self.assertIn(config_id(cfg), selector._arms[cell])

    def test_success_threshold_at_top_30_percent(self) -> None:
        """With 5 results and top_k=30%, only the best 1-2 should get successes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            configs = [_matmul_config(bm=32 * (i + 1)) for i in range(5)]
            tflops = [100.0, 120.0, 130.0, 160.0, 200.0]
            _record_many(db, list(zip(configs, tflops)))

            selector = BanditSelector(seed=0, top_k_percent=30.0)
            selector.load_from_database(db, operator="matmul", hardware="A100")

            cell = "matmul:medium:A100"
            arms = selector._arms[cell]

            # The best config (200 TFLOPS) should have alpha > prior+1, beta == prior
            best_cid = config_id(configs[4])
            best_arm = arms[best_cid]
            self.assertGreater(best_arm.alpha, _PRIOR_ALPHA)
            # The weakest config (100 TFLOPS) should have more beta increments
            worst_cid = config_id(configs[0])
            worst_arm = arms[worst_cid]
            self.assertLess(worst_arm.expected_reward, best_arm.expected_reward)

    def test_operator_filter_respected(self) -> None:
        """Records for 'attention' operator must not pollute 'matmul' arms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            # Insert one matmul and one attention record
            matmul_cfg = _matmul_config(bm=64)
            attn_cfg = _matmul_config(bm=32)
            _record_many(db, [(matmul_cfg, 150.0)], operator="matmul", bucket="medium")
            _record_many(db, [(attn_cfg, 200.0)], operator="attention", bucket="medium")

            selector = BanditSelector(seed=0)
            selector.load_from_database(db, operator="matmul", hardware="A100")

            # Only matmul cell should exist
            self.assertIn("matmul:medium:A100", selector._arms)
            self.assertNotIn("attention:medium:A100", selector._arms)

    def test_hardware_filter_respected(self) -> None:
        """Records for a different hardware must not create arms for queried hw."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            cfg = _matmul_config()
            _record_many(db, [(cfg, 150.0)], hardware="H100")

            selector = BanditSelector(seed=0)
            selector.load_from_database(db, operator="matmul", hardware="A100")
            self.assertEqual(len(selector._arms), 0)

    def test_incorrect_results_excluded(self) -> None:
        """Results with correct=False must not affect posteriors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            cfg = _matmul_config()
            # Insert a failed result manually
            db.record_result(
                shape={"M": 2048, "N": 2048, "K": 2048, "bucket": "medium"},
                hardware="A100",
                config=cfg,
                tflops=0.0,
                ms=0.0,
                correct=False,
                operator="matmul",
                bucket="medium",
                config_id_str=config_id(cfg),
            )
            selector = BanditSelector(seed=0)
            selector.load_from_database(db, operator="matmul", hardware="A100")
            # No correct results -> no arms created
            self.assertEqual(len(selector._arms), 0)


# ---------------------------------------------------------------------------
# Sampling preference tests
# ---------------------------------------------------------------------------

class SamplingPreferenceTests(unittest.TestCase):
    """Verify that high-reward configs are selected more often."""

    def test_winner_sampled_more_than_loser(self) -> None:
        """With enough trials, the winner's posterior dominates the loser's."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            winner = _matmul_config(bm=256)
            loser = _matmul_config(bm=32)
            # Insert many results: winner always tops 30%, loser never does
            _record_many(db, [
                (winner, 200.0),
                (loser, 80.0),
            ] * 20)

            selector = BanditSelector(seed=42, top_k_percent=30.0)
            selector.load_from_database(db, operator="matmul", hardware="A100")

            cell = "matmul:medium:A100"
            winner_arm = selector._arms[cell][config_id(winner)]
            loser_arm = selector._arms[cell][config_id(loser)]

            self.assertGreater(winner_arm.expected_reward, loser_arm.expected_reward)

    def test_ranked_configs_winner_first(self) -> None:
        """ranked_configs_for_bucket should put the likely winner near the top."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            winner = _matmul_config(bm=256)
            loser = _matmul_config(bm=32)
            _record_many(db, [(winner, 200.0)] * 15 + [(loser, 80.0)] * 15)

            selector = BanditSelector(seed=0, top_k_percent=30.0)
            selector.load_from_database(db, operator="matmul", hardware="A100")

            ranked = selector.ranked_configs_for_bucket(
                operator="matmul", bucket="medium", hardware="A100",
            )
            self.assertTrue(len(ranked) >= 2)
            # Winner should appear before loser in expectation over many seeds.
            # With seed=0 and clear separation, this holds deterministically.
            winner_idx = next(
                i for i, c in enumerate(ranked) if config_id(c) == config_id(winner)
            )
            loser_idx = next(
                i for i, c in enumerate(ranked) if config_id(c) == config_id(loser)
            )
            self.assertLess(winner_idx, loser_idx)


# ---------------------------------------------------------------------------
# select_configs with empty database
# ---------------------------------------------------------------------------

class EmptyDatabaseFallbackTests(unittest.TestCase):
    """When the database is empty the selector must return curated configs."""

    def _get_matmul_spec(self):
        from research_engine.triton_operators import REGISTRY
        return REGISTRY.get("matmul")

    def test_empty_db_returns_curated_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            spec = self._get_matmul_spec()
            selector = BanditSelector(seed=0)
            configs = selector.select_configs(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=[{"M": 2048, "N": 2048, "K": 2048}],
                max_configs=8,
            )
            self.assertGreater(len(configs), 0)
            # All returned configs should be valid parameter dicts
            for cfg in configs:
                self.assertIn("BLOCK_SIZE_M", cfg)
                self.assertIn("num_warps", cfg)

    def test_empty_db_respects_max_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            spec = self._get_matmul_spec()
            selector = BanditSelector(seed=0)
            configs = selector.select_configs(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=[{"M": 128, "N": 128, "K": 128}],
                max_configs=3,
            )
            self.assertLessEqual(len(configs), 3)

    def test_no_duplicates_in_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            spec = self._get_matmul_spec()
            selector = BanditSelector(seed=0)
            configs = selector.select_configs(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=[{"M": 2048, "N": 2048, "K": 2048}],
                max_configs=12,
            )
            ids = [config_id(c) for c in configs]
            self.assertEqual(len(ids), len(set(ids)), "Duplicate config IDs returned")


# ---------------------------------------------------------------------------
# Shape-bucket isolation test
# ---------------------------------------------------------------------------

class ShapeBucketIsolationTests(unittest.TestCase):
    """Confirm that rewards in one (operator, bucket) don't affect another."""

    def _get_matmul_spec(self):
        from research_engine.triton_operators import REGISTRY
        return REGISTRY.get("matmul")

    def test_large_matmul_bucket_does_not_pollute_tiny_bucket(self) -> None:
        """A config that wins in 'large' should not get credit in 'tiny'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            winner_in_large = _matmul_config(bm=256)
            _record_many(
                db,
                [(winner_in_large, 300.0)] * 10,
                operator="matmul",
                bucket="large",
            )

            selector = BanditSelector(seed=0)
            selector.load_from_database(db, operator="matmul", hardware="A100")

            # The 'tiny' bucket should have no arms at all
            self.assertNotIn("matmul:tiny:A100", selector._arms)

    def test_matmul_large_does_not_pollute_attention(self) -> None:
        """matmul:large results must not appear in the attention cell."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            matmul_winner = _matmul_config(bm=256)
            _record_many(
                db,
                [(matmul_winner, 300.0)] * 5,
                operator="matmul",
                bucket="large",
            )
            # Also insert one attention result
            attn_cfg = _matmul_config(bm=64)
            _record_many(
                db,
                [(attn_cfg, 100.0)] * 5,
                operator="attention",
                bucket="large",
            )

            selector = BanditSelector(seed=0)
            # Load without operator filter so all records are loaded
            selector.load_from_database(db, hardware="A100")

            matmul_cell = "matmul:large:A100"
            attention_cell = "attention:large:A100"

            # Each cell should contain only its own configs
            matmul_ids = set(selector._arms.get(matmul_cell, {}).keys())
            attention_ids = set(selector._arms.get(attention_cell, {}).keys())
            self.assertIn(config_id(matmul_winner), matmul_ids)
            self.assertIn(config_id(attn_cfg), attention_ids)
            # No cross-contamination
            self.assertNotIn(config_id(attn_cfg), matmul_ids)
            self.assertNotIn(config_id(matmul_winner), attention_ids)

    def test_same_config_id_different_buckets_independent(self) -> None:
        """The same config_id that wins in 'medium' and loses in 'small'
        should have independent Beta posteriors for the two cells."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            cfg = _matmul_config(bm=128)

            # Insert as a winner in medium
            _record_many(
                db,
                [(cfg, 200.0)] * 5 + [(_matmul_config(bm=32), 80.0)] * 5,
                operator="matmul",
                bucket="medium",
            )
            # Insert as a loser in small
            _record_many(
                db,
                [(cfg, 60.0)] * 5 + [(_matmul_config(bm=64), 180.0)] * 5,
                operator="matmul",
                bucket="small",
            )

            selector = BanditSelector(seed=0, top_k_percent=30.0)
            selector.load_from_database(db, operator="matmul", hardware="A100")

            medium_arm = selector._arms["matmul:medium:A100"][config_id(cfg)]
            small_arm = selector._arms["matmul:small:A100"][config_id(cfg)]

            # In medium it should be a winner, in small a loser
            self.assertGreater(medium_arm.expected_reward, small_arm.expected_reward)


# ---------------------------------------------------------------------------
# Posterior summary introspection
# ---------------------------------------------------------------------------

class PosteriorSummaryTests(unittest.TestCase):
    def test_summary_returns_sorted_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            configs = [_matmul_config(bm=32 * (i + 1)) for i in range(4)]
            tflops = [80.0, 100.0, 150.0, 200.0]
            _record_many(db, list(zip(configs, tflops)))

            selector = BanditSelector(seed=0, top_k_percent=30.0)
            selector.load_from_database(db, operator="matmul", hardware="A100")
            summary = selector.posterior_summary(
                operator="matmul", bucket="medium", hardware="A100", top_n=4,
            )
            self.assertEqual(len(summary), 4)
            # Should be sorted descending by expected_reward
            rewards = [s["expected_reward"] for s in summary]
            self.assertEqual(rewards, sorted(rewards, reverse=True))

    def test_empty_bucket_returns_empty_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            selector = BanditSelector(seed=0)
            selector.load_from_database(db)
            summary = selector.posterior_summary(
                operator="matmul", bucket="nonexistent", hardware="A100",
            )
            self.assertEqual(summary, [])


if __name__ == "__main__":
    unittest.main()
