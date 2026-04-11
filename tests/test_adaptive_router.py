"""Tests for the AdaptiveRouter bandit-over-selectors.

Coverage (18 tests):
1.  Uniform prior: all arms have equal expected reward initially
2.  select_arm returns one of the three valid arm constants
3.  Arm selection is deterministic for a fixed seed
4.  After many successes on bandit arm, bandit has highest expected reward
5.  After many failures on cost_model arm, cost_model has lowest expected reward
6.  Different operators have fully independent arm posteriors
7.  record_outcome success increments alpha (not beta)
8.  record_outcome failure increments beta (not alpha)
9.  record_outcome with unknown arm is a no-op (no crash, warning only)
10. load_from_database with no attributed records → uniform prior for all operators
11. load_from_database replays attributed outcomes correctly
12. load_from_database is idempotent (calling twice yields same state)
13. select_configs delegates to bandit arm when bandit arm is chosen
14. select_configs delegates to baseline when arm has no dependency available
15. select_configs falls back to baseline when ARM_BASELINE is chosen
16. select_configs returns a 2-tuple (configs, arm_name)
17. posterior_summary returns sorted list by expected_reward descending
18. posterior_summary covers all three arms, even unseen ones (prior)
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from tests import _pathfix  # noqa: F401

from research_engine.adaptive_router import (
    AdaptiveRouter,
    ARM_BANDIT,
    ARM_BASELINE,
    ARM_COST_MODEL,
    _ALL_ARMS,
    _ArmPosterior,
    _PRIOR_ALPHA,
    _PRIOR_BETA,
)
from research_engine.bandit_selector import BanditSelector
from research_engine.triton_kernels import ConfigDatabase, config_id
from research_engine.triton_operators import REGISTRY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_db(tmpdir: Path) -> ConfigDatabase:
    return ConfigDatabase(path=tmpdir / "db.json")


def _matmul_spec():
    return REGISTRY.get("matmul")


def _matmul_config(bm: int = 128, bn: int = 128, bk: int = 64) -> dict:
    return {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }


def _record_result(db: ConfigDatabase, cfg: dict, tflops: float = 100.0,
                   *, operator: str = "matmul", bucket: str = "medium",
                   hardware: str = "A100", selector: str = "",
                   selector_improved: bool | None = None) -> None:
    """Helper: insert a single benchmark result into the database.

    ConfigDatabase.record_result() does not accept extra kwargs, so we inject
    the ``selector`` and ``selector_improved`` fields directly into the last
    appended result entry after the call.
    """
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
    # Inject attribution fields into the result entry we just appended.
    key = f"{operator}:{bucket}:{hardware}"
    last_result = db.records[key].results[-1]
    if selector:
        last_result["selector"] = selector
    if selector_improved is not None:
        last_result["selector_improved"] = selector_improved


# ---------------------------------------------------------------------------
# _ArmPosterior unit tests
# ---------------------------------------------------------------------------


class ArmPosteriorTests(unittest.TestCase):
    """Unit-test the Beta posterior arm in isolation."""

    def test_initial_prior(self) -> None:
        arm = _ArmPosterior(arm=ARM_BANDIT)
        self.assertAlmostEqual(arm.alpha, _PRIOR_ALPHA)
        self.assertAlmostEqual(arm.beta, _PRIOR_BETA)
        self.assertEqual(arm.total_observations, 0)

    def test_success_increments_alpha(self) -> None:
        arm = _ArmPosterior(arm=ARM_BANDIT)
        arm.update(success=True)
        self.assertAlmostEqual(arm.alpha, _PRIOR_ALPHA + 1.0)
        self.assertAlmostEqual(arm.beta, _PRIOR_BETA)

    def test_failure_increments_beta(self) -> None:
        arm = _ArmPosterior(arm=ARM_BANDIT)
        arm.update(success=False)
        self.assertAlmostEqual(arm.alpha, _PRIOR_ALPHA)
        self.assertAlmostEqual(arm.beta, _PRIOR_BETA + 1.0)

    def test_expected_reward_formula(self) -> None:
        arm = _ArmPosterior(arm=ARM_COST_MODEL)
        arm.update(success=True)
        arm.update(success=True)
        arm.update(success=False)
        expected = arm.alpha / (arm.alpha + arm.beta)
        self.assertAlmostEqual(arm.expected_reward, expected)

    def test_sample_in_unit_interval(self) -> None:
        arm = _ArmPosterior(arm=ARM_BANDIT)
        arm.update(success=True)
        rng = np.random.default_rng(7)
        for _ in range(50):
            s = arm.sample(rng)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


# ---------------------------------------------------------------------------
# Test 1-3: uniform prior, valid arm, determinism
# ---------------------------------------------------------------------------


class ArmSelectionBasicsTests(unittest.TestCase):

    def test_uniform_prior_all_arms_equal_expected_reward(self) -> None:
        """Test 1: All arms start with equal expected reward Beta(1,1)."""
        router = AdaptiveRouter(seed=0)
        summary = router.posterior_summary("matmul")
        rewards = [row["expected_reward"] for row in summary]
        # All prior means are 0.5
        for r in rewards:
            self.assertAlmostEqual(r, 0.5)

    def test_select_arm_returns_valid_arm(self) -> None:
        """Test 2: select_arm always returns one of the three ARM_* constants."""
        router = AdaptiveRouter(seed=42)
        for _ in range(30):
            arm = router.select_arm("matmul")
            self.assertIn(arm, _ALL_ARMS)

    def test_select_arm_deterministic_for_fixed_seed(self) -> None:
        """Test 3: same seed → same arm selection sequence."""
        router1 = AdaptiveRouter(seed=99)
        router2 = AdaptiveRouter(seed=99)
        choices1 = [router1.select_arm("matmul") for _ in range(10)]
        choices2 = [router2.select_arm("matmul") for _ in range(10)]
        self.assertEqual(choices1, choices2)


# ---------------------------------------------------------------------------
# Tests 4-5: posterior update after many outcomes
# ---------------------------------------------------------------------------


class PosteriorUpdateTests(unittest.TestCase):

    def test_many_successes_on_bandit_arm_boosts_expected_reward(self) -> None:
        """Test 4: After 10 successes on bandit, bandit has highest expected reward."""
        router = AdaptiveRouter(seed=0)
        for _ in range(10):
            router.record_outcome(operator="matmul", arm=ARM_BANDIT, improvement=True)
        summary = router.posterior_summary("matmul")
        arms_by_name = {row["arm"]: row for row in summary}
        self.assertGreater(
            arms_by_name[ARM_BANDIT]["expected_reward"],
            arms_by_name[ARM_COST_MODEL]["expected_reward"],
        )
        self.assertGreater(
            arms_by_name[ARM_BANDIT]["expected_reward"],
            arms_by_name[ARM_BASELINE]["expected_reward"],
        )

    def test_many_failures_on_cost_model_lowers_expected_reward(self) -> None:
        """Test 5: After 10 failures on cost_model, cost_model has lowest expected reward."""
        router = AdaptiveRouter(seed=0)
        for _ in range(10):
            router.record_outcome(operator="matmul", arm=ARM_COST_MODEL, improvement=False)
        summary = router.posterior_summary("matmul")
        arms_by_name = {row["arm"]: row for row in summary}
        self.assertLess(
            arms_by_name[ARM_COST_MODEL]["expected_reward"],
            arms_by_name[ARM_BANDIT]["expected_reward"],
        )
        self.assertLess(
            arms_by_name[ARM_COST_MODEL]["expected_reward"],
            arms_by_name[ARM_BASELINE]["expected_reward"],
        )


# ---------------------------------------------------------------------------
# Test 6: operator independence
# ---------------------------------------------------------------------------


class OperatorIndependenceTests(unittest.TestCase):

    def test_different_operators_have_independent_posteriors(self) -> None:
        """Test 6: matmul arm updates must not affect softmax arm posteriors."""
        router = AdaptiveRouter(seed=0)
        # Give bandit many successes for matmul
        for _ in range(10):
            router.record_outcome(operator="matmul", arm=ARM_BANDIT, improvement=True)
        # softmax posteriors should still be at the uniform prior
        softmax_summary = router.posterior_summary("softmax")
        for row in softmax_summary:
            self.assertAlmostEqual(row["expected_reward"], 0.5, places=5)

    def test_two_operators_can_have_different_winning_arms(self) -> None:
        """Bandit wins matmul, cost_model wins attention — router reflects both."""
        router = AdaptiveRouter(seed=0)
        for _ in range(8):
            router.record_outcome(operator="matmul", arm=ARM_BANDIT, improvement=True)
        for _ in range(8):
            router.record_outcome(operator="attention", arm=ARM_COST_MODEL, improvement=True)

        matmul_summary = {r["arm"]: r for r in router.posterior_summary("matmul")}
        attn_summary = {r["arm"]: r for r in router.posterior_summary("attention")}

        self.assertGreater(
            matmul_summary[ARM_BANDIT]["expected_reward"],
            matmul_summary[ARM_COST_MODEL]["expected_reward"],
        )
        self.assertGreater(
            attn_summary[ARM_COST_MODEL]["expected_reward"],
            attn_summary[ARM_BANDIT]["expected_reward"],
        )


# ---------------------------------------------------------------------------
# Tests 7-9: record_outcome correctness
# ---------------------------------------------------------------------------


class RecordOutcomeTests(unittest.TestCase):

    def test_success_increments_alpha(self) -> None:
        """Test 7: record_outcome success increments alpha."""
        router = AdaptiveRouter(seed=0)
        router.record_outcome(operator="matmul", arm=ARM_BANDIT, improvement=True)
        posterior = router._posteriors["matmul"][ARM_BANDIT]
        self.assertAlmostEqual(posterior.alpha, _PRIOR_ALPHA + 1.0)
        self.assertAlmostEqual(posterior.beta, _PRIOR_BETA)

    def test_failure_increments_beta(self) -> None:
        """Test 8: record_outcome failure increments beta."""
        router = AdaptiveRouter(seed=0)
        router.record_outcome(operator="matmul", arm=ARM_BANDIT, improvement=False)
        posterior = router._posteriors["matmul"][ARM_BANDIT]
        self.assertAlmostEqual(posterior.alpha, _PRIOR_ALPHA)
        self.assertAlmostEqual(posterior.beta, _PRIOR_BETA + 1.0)

    def test_unknown_arm_does_not_crash(self) -> None:
        """Test 9: record_outcome with unknown arm name is a safe no-op."""
        router = AdaptiveRouter(seed=0)
        # Must not raise
        router.record_outcome(operator="matmul", arm="unknown_arm", improvement=True)
        # The posteriors dict should not have gained an entry for "unknown_arm"
        self.assertNotIn("unknown_arm", router._posteriors.get("matmul", {}))


# ---------------------------------------------------------------------------
# Tests 10-12: load_from_database
# ---------------------------------------------------------------------------


class LoadFromDatabaseTests(unittest.TestCase):

    def test_empty_database_leaves_uniform_prior(self) -> None:
        """Test 10: empty database → all arm expected_rewards are 0.5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            router = AdaptiveRouter(seed=0)
            router.load_from_database(db)
            # No posteriors should have been created
            self.assertEqual(len(router._posteriors), 0)
            # select_arm still works and returns a valid arm
            arm = router.select_arm("matmul")
            self.assertIn(arm, _ALL_ARMS)

    def test_attributed_outcomes_are_replayed(self) -> None:
        """Test 11: results with selector + selector_improved fields are replayed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            cfg = _matmul_config()
            # Record 5 successes for the bandit arm
            for _ in range(5):
                _record_result(
                    db, cfg,
                    selector=ARM_BANDIT, selector_improved=True,
                )
            # Record 5 failures for the cost_model arm
            for _ in range(5):
                _record_result(
                    db, _matmul_config(bm=64),
                    selector=ARM_COST_MODEL, selector_improved=False,
                )

            router = AdaptiveRouter(seed=0)
            router.load_from_database(db)

            matmul_arms = router._posteriors.get("matmul", {})
            self.assertIn(ARM_BANDIT, matmul_arms)
            self.assertIn(ARM_COST_MODEL, matmul_arms)

            bandit_arm = matmul_arms[ARM_BANDIT]
            cm_arm = matmul_arms[ARM_COST_MODEL]

            # 5 successes → alpha = prior + 5
            self.assertAlmostEqual(bandit_arm.alpha, _PRIOR_ALPHA + 5.0)
            # 5 failures → beta = prior + 5
            self.assertAlmostEqual(cm_arm.beta, _PRIOR_BETA + 5.0)

    def test_load_from_database_is_idempotent(self) -> None:
        """Test 12: calling load twice on same database gives same state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            cfg = _matmul_config()
            for _ in range(3):
                _record_result(db, cfg, selector=ARM_BANDIT, selector_improved=True)

            router = AdaptiveRouter(seed=0)
            router.load_from_database(db)
            alpha_first = router._posteriors["matmul"][ARM_BANDIT].alpha

            router.load_from_database(db)
            alpha_second = router._posteriors["matmul"][ARM_BANDIT].alpha

            self.assertAlmostEqual(alpha_first, alpha_second)

    def test_unattributed_records_are_ignored(self) -> None:
        """Records without a selector field must not affect posteriors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            cfg = _matmul_config()
            # Record without selector annotation
            _record_result(db, cfg)
            router = AdaptiveRouter(seed=0)
            router.load_from_database(db)
            # No posteriors should be created
            self.assertEqual(len(router._posteriors), 0)


# ---------------------------------------------------------------------------
# Tests 13-16: select_configs delegation
# ---------------------------------------------------------------------------


class SelectConfigsDelegationTests(unittest.TestCase):

    def _make_mock_bandit(self, configs: list[dict]) -> MagicMock:
        """Mock BanditSelector that returns a fixed config list."""
        mock = MagicMock(spec=BanditSelector)
        mock.select_configs.return_value = configs
        return mock

    def _make_mock_cost_model(self) -> MagicMock:
        cm = MagicMock()
        cm.rank_configs.return_value = []
        return cm

    def test_bandit_arm_delegates_to_bandit_selector(self) -> None:
        """Test 13: when arm=bandit is chosen, bandit.select_configs is called."""
        spec = _matmul_spec()
        configs = [_matmul_config()]
        mock_bandit = self._make_mock_bandit(configs)

        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))

            # Force the router to always choose the bandit arm by giving it
            # a deterministic mock that always returns the bandit arm.
            router = AdaptiveRouter(seed=0)
            with patch.object(router, "select_arm", return_value=ARM_BANDIT):
                result_configs, chosen_arm = router.select_configs(
                    spec=spec,
                    database=db,
                    hardware="A100",
                    shapes=[{"M": 2048, "N": 2048, "K": 2048}],
                    max_configs=4,
                    cost_model=None,
                    bandit_selector=mock_bandit,
                )

        self.assertEqual(chosen_arm, ARM_BANDIT)
        mock_bandit.select_configs.assert_called_once()
        self.assertEqual(result_configs, configs)

    def test_cost_model_arm_delegates_to_select_configs_for_operator(self) -> None:
        """Test 14: when arm=cost_model is chosen, select_configs_for_operator is called."""
        spec = _matmul_spec()
        mock_cm = self._make_mock_cost_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))

            router = AdaptiveRouter(seed=0)
            with patch.object(router, "select_arm", return_value=ARM_COST_MODEL):
                with patch(
                    "research_engine.triton_operators.select_configs_for_operator",
                    return_value=[_matmul_config()],
                ) as mock_select:
                    result_configs, chosen_arm = router.select_configs(
                        spec=spec,
                        database=db,
                        hardware="A100",
                        shapes=[{"M": 2048, "N": 2048, "K": 2048}],
                        max_configs=4,
                        cost_model=mock_cm,
                        bandit_selector=None,
                    )

        self.assertEqual(chosen_arm, ARM_COST_MODEL)
        mock_select.assert_called_once()

    def test_baseline_arm_uses_frontier_selector(self) -> None:
        """Test 15: when arm=baseline is chosen, falls back to frontier-slot selector."""
        spec = _matmul_spec()

        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))

            router = AdaptiveRouter(seed=0)
            with patch.object(router, "select_arm", return_value=ARM_BASELINE):
                with patch(
                    "research_engine.triton_operators.select_configs_for_operator",
                    return_value=[_matmul_config()],
                ) as mock_select:
                    result_configs, chosen_arm = router.select_configs(
                        spec=spec,
                        database=db,
                        hardware="A100",
                        shapes=[{"M": 2048, "N": 2048, "K": 2048}],
                        max_configs=4,
                        cost_model=None,
                        bandit_selector=None,
                    )

        self.assertEqual(chosen_arm, ARM_BASELINE)
        # Should have called the frontier selector with cost_model=None
        call_kwargs = mock_select.call_args.kwargs
        self.assertIsNone(call_kwargs.get("cost_model"))

    def test_select_configs_returns_tuple_of_configs_and_arm(self) -> None:
        """Test 16: select_configs returns (list[dict], str) tuple."""
        spec = _matmul_spec()

        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            router = AdaptiveRouter(seed=99)
            result = router.select_configs(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=[{"M": 2048, "N": 2048, "K": 2048}],
                max_configs=4,
                cost_model=None,
                bandit_selector=None,
            )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        configs, arm = result
        self.assertIsInstance(configs, list)
        self.assertIn(arm, _ALL_ARMS)

    def test_bandit_arm_chosen_but_no_bandit_falls_back_to_baseline(self) -> None:
        """Test: if bandit arm is chosen but bandit_selector is None, fall back to baseline."""
        spec = _matmul_spec()

        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            router = AdaptiveRouter(seed=0)
            with patch.object(router, "select_arm", return_value=ARM_BANDIT):
                with patch(
                    "research_engine.triton_operators.select_configs_for_operator",
                    return_value=[_matmul_config()],
                ) as mock_select:
                    result_configs, chosen_arm = router.select_configs(
                        spec=spec,
                        database=db,
                        hardware="A100",
                        shapes=[{"M": 2048, "N": 2048, "K": 2048}],
                        max_configs=4,
                        cost_model=None,
                        bandit_selector=None,  # No bandit provided
                    )

        # Falls back to baseline
        self.assertEqual(chosen_arm, ARM_BASELINE)
        mock_select.assert_called_once()


# ---------------------------------------------------------------------------
# Tests 17-18: posterior_summary
# ---------------------------------------------------------------------------


class PosteriorSummaryTests(unittest.TestCase):

    def test_summary_sorted_by_expected_reward_descending(self) -> None:
        """Test 17: posterior_summary returns rows sorted by expected_reward desc."""
        router = AdaptiveRouter(seed=0)
        # Make bandit the clear winner, baseline middle, cost_model worst
        for _ in range(8):
            router.record_outcome(operator="matmul", arm=ARM_BANDIT, improvement=True)
        for _ in range(4):
            router.record_outcome(operator="matmul", arm=ARM_BASELINE, improvement=True)
        for _ in range(8):
            router.record_outcome(operator="matmul", arm=ARM_COST_MODEL, improvement=False)

        summary = router.posterior_summary("matmul")
        rewards = [row["expected_reward"] for row in summary]
        self.assertEqual(rewards, sorted(rewards, reverse=True))

    def test_summary_covers_all_three_arms(self) -> None:
        """Test 18: posterior_summary always returns all three arms, even unseen ones."""
        router = AdaptiveRouter(seed=0)
        # Only update bandit arm
        router.record_outcome(operator="matmul", arm=ARM_BANDIT, improvement=True)
        summary = router.posterior_summary("matmul")
        arm_names = {row["arm"] for row in summary}
        self.assertEqual(arm_names, set(_ALL_ARMS))

    def test_summary_unseen_arms_show_prior_parameters(self) -> None:
        """Unseen arms in posterior_summary report prior alpha/beta values."""
        router = AdaptiveRouter(seed=0)
        # No outcomes recorded for any arm for softmax
        summary = router.posterior_summary("softmax")
        for row in summary:
            self.assertAlmostEqual(row["alpha"], _PRIOR_ALPHA)
            self.assertAlmostEqual(row["beta"], _PRIOR_BETA)


# ---------------------------------------------------------------------------
# Integration: realistic BanditSelector mock
# ---------------------------------------------------------------------------


class IntegrationTests(unittest.TestCase):
    """End-to-end tests with a real BanditSelector and real ConfigDatabase."""

    def test_select_configs_with_real_bandit_returns_valid_configs(self) -> None:
        """Integration: AdaptiveRouter with a real BanditSelector returns valid configs."""
        spec = _matmul_spec()

        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            bandit = BanditSelector(seed=0)
            router = AdaptiveRouter(seed=42)

            with patch.object(router, "select_arm", return_value=ARM_BANDIT):
                configs, chosen_arm = router.select_configs(
                    spec=spec,
                    database=db,
                    hardware="A100",
                    shapes=[{"M": 2048, "N": 2048, "K": 2048}],
                    max_configs=6,
                    cost_model=None,
                    bandit_selector=bandit,
                )

        self.assertEqual(chosen_arm, ARM_BANDIT)
        self.assertGreater(len(configs), 0)
        for cfg in configs:
            self.assertIn("BLOCK_SIZE_M", cfg)
            self.assertIn("num_warps", cfg)

    def test_full_loop_records_outcomes_and_shifts_posteriors(self) -> None:
        """Integration: N iterations where bandit always improves shift posterior toward bandit."""
        spec = _matmul_spec()

        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_db(Path(tmpdir))
            bandit = BanditSelector(seed=0)
            router = AdaptiveRouter(seed=123)

            # Simulate 5 iterations where we always force-choose bandit and it always improves.
            for _ in range(5):
                with patch.object(router, "select_arm", return_value=ARM_BANDIT):
                    configs, chosen_arm = router.select_configs(
                        spec=spec,
                        database=db,
                        hardware="A100",
                        shapes=[{"M": 2048, "N": 2048, "K": 2048}],
                        max_configs=4,
                        cost_model=None,
                        bandit_selector=bandit,
                    )
                router.record_outcome(operator="matmul", arm=chosen_arm, improvement=True)

        summary = {row["arm"]: row for row in router.posterior_summary("matmul")}
        # Bandit should have 5 successes: alpha = prior + 5
        self.assertAlmostEqual(
            summary[ARM_BANDIT]["alpha"], _PRIOR_ALPHA + 5.0
        )
        # Other arms untouched — still at prior
        self.assertAlmostEqual(summary[ARM_COST_MODEL]["alpha"], _PRIOR_ALPHA)
        self.assertAlmostEqual(summary[ARM_BASELINE]["alpha"], _PRIOR_ALPHA)


if __name__ == "__main__":
    unittest.main()
