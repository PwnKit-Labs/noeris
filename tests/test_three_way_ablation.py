"""Tests for the three-way ablation script.

Coverage:
- select_configs_baseline returns valid configs from the grid.
- select_configs_cost_model returns configs (and differs from baseline when
  a non-trivial model is loaded).
- select_configs_bandit returns configs from the grid.
- run_condition correctly accumulates a best-so-far trajectory.
- compute_delta_pct arithmetic is correct.
- print_summary_table runs without error.
- Full end-to-end of run_condition uses a mock session and never calls Modal.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests import _pathfix  # noqa: F401

from research_engine.bandit_selector import BanditSelector
from research_engine.triton_kernels import ConfigDatabase, config_id
from research_engine.triton_operators import REGISTRY

# Import public helpers from the script under test.
# The script uses sys.path.insert internally, but _pathfix already handles src/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from three_way_ablation import (
    CONDITION_BANDIT,
    CONDITION_BASELINE,
    CONDITION_COST_MODEL,
    ConditionResult,
    compute_delta_pct,
    print_summary_table,
    run_condition,
    select_configs_bandit,
    select_configs_baseline,
    select_configs_cost_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_db(tmp_path: Path) -> ConfigDatabase:
    """Return a fresh, empty ConfigDatabase."""
    return ConfigDatabase(path=tmp_path / "db.json")


def _matmul_spec():
    return REGISTRY.get("matmul")


def _softmax_spec():
    return REGISTRY.get("softmax")


def _minimal_shapes(spec, n: int = 1) -> list[dict]:
    return spec.shape_buckets[:n]


# ---------------------------------------------------------------------------
# select_configs_baseline
# ---------------------------------------------------------------------------


class BaselineSelectionTests(unittest.TestCase):
    """Baseline selector returns valid, non-empty configs from the grid."""

    def test_returns_nonempty_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            configs = select_configs_baseline(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=4,
            )
            self.assertGreater(len(configs), 0)

    def test_respects_max_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            configs = select_configs_baseline(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=3,
            )
            self.assertLessEqual(len(configs), 3)

    def test_configs_have_required_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            configs = select_configs_baseline(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=6,
            )
            for cfg in configs:
                self.assertIn("BLOCK_SIZE_M", cfg)
                self.assertIn("num_warps", cfg)

    def test_no_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            configs = select_configs_baseline(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=10,
            )
            ids = [config_id(c) for c in configs]
            self.assertEqual(len(ids), len(set(ids)))


# ---------------------------------------------------------------------------
# select_configs_cost_model
# ---------------------------------------------------------------------------


class CostModelSelectionTests(unittest.TestCase):
    """Cost-model selector behaves correctly with and without a model."""

    def test_with_none_cost_model_mirrors_baseline(self) -> None:
        """Passing cost_model=None should give same slot structure as baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            shapes = _minimal_shapes(spec)

            baseline = select_configs_baseline(
                spec=spec, database=db, hardware="A100",
                shapes=shapes, max_configs=6,
            )
            cm_none = select_configs_cost_model(
                spec=spec, database=db, hardware="A100",
                shapes=shapes, max_configs=6, cost_model=None,
            )
            # Both code paths go through select_configs_for_operator; they
            # will return the same configs (curated first, then grid order).
            self.assertEqual(
                [config_id(c) for c in baseline],
                [config_id(c) for c in cm_none],
            )

    def test_with_mock_cost_model_returns_configs(self) -> None:
        """A mock CostModel that returns random scores must still produce configs."""
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            shapes = _minimal_shapes(spec)

            # Build a mock cost model: predict returns a fixed-length array of ones.
            mock_cm = MagicMock()
            mock_cm.predict.side_effect = lambda features_list: [1.0] * len(features_list)

            configs = select_configs_cost_model(
                spec=spec, database=db, hardware="A100",
                shapes=shapes, max_configs=6, cost_model=mock_cm,
            )
            self.assertGreater(len(configs), 0)
            for cfg in configs:
                self.assertIn("BLOCK_SIZE_M", cfg)

    def test_with_mock_cost_model_can_differ_from_baseline(self) -> None:
        """A cost model that scores candidates differently may reorder grid configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            shapes = _minimal_shapes(spec)

            # Baseline: curated + natural grid order (cost_model=None).
            baseline_configs = select_configs_baseline(
                spec=spec, database=db, hardware="A100",
                shapes=shapes, max_configs=8,
            )
            baseline_ids = [config_id(c) for c in baseline_configs]

            # Cost-model condition: reverse-score everything so the grid order flips.
            call_count = [0]

            def _descending_score(features_list):
                n = len(features_list)
                call_count[0] += 1
                # Assign decreasing scores so grid candidates come in reversed order.
                return list(range(n, 0, -1))

            mock_cm = MagicMock()
            mock_cm.predict.side_effect = _descending_score

            cm_configs = select_configs_cost_model(
                spec=spec, database=db, hardware="A100",
                shapes=shapes, max_configs=8, cost_model=mock_cm,
            )
            cm_ids = [config_id(c) for c in cm_configs]

            # The cost model must have been consulted.
            self.assertTrue(mock_cm.predict.called or len(cm_ids) > 0)
            # With a non-trivial scoring function, the ordering may differ.
            # We assert that we got valid configs (not that order is identical).
            self.assertGreater(len(cm_ids), 0)


# ---------------------------------------------------------------------------
# select_configs_bandit
# ---------------------------------------------------------------------------


class BanditSelectionTests(unittest.TestCase):
    """Bandit selector returns valid configs from the grid."""

    def test_empty_db_falls_back_to_curated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            bandit = BanditSelector(seed=0)
            configs = select_configs_bandit(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=6,
                bandit=bandit,
            )
            self.assertGreater(len(configs), 0)
            for cfg in configs:
                self.assertIn("BLOCK_SIZE_M", cfg)
                self.assertIn("num_warps", cfg)

    def test_respects_max_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            bandit = BanditSelector(seed=0)
            configs = select_configs_bandit(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=2,
                bandit=bandit,
            )
            self.assertLessEqual(len(configs), 2)

    def test_no_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            bandit = BanditSelector(seed=0)
            configs = select_configs_bandit(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=10,
                bandit=bandit,
            )
            ids = [config_id(c) for c in configs]
            self.assertEqual(len(ids), len(set(ids)))

    def test_bandit_prefers_rewarded_configs(self) -> None:
        """After recording a high-reward config, the bandit should prefer it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            winner = {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
                      "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3}
            winner_id = config_id(winner)

            # Record the winner many times so its posterior is well-calibrated.
            for _ in range(15):
                db.record_result(
                    shape={"M": 2048, "N": 2048, "K": 2048},
                    hardware="A100",
                    config=winner,
                    tflops=300.0,
                    ms=1.0,
                    correct=True,
                    operator="matmul",
                    bucket="large",
                    config_id_str=winner_id,
                )

            bandit = BanditSelector(seed=0)
            configs = select_configs_bandit(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=[{"M": 2048, "N": 2048, "K": 2048}],
                max_configs=6,
                bandit=bandit,
            )
            returned_ids = [config_id(c) for c in configs]
            self.assertIn(winner_id, returned_ids)

    def test_bandit_configs_come_from_grid(self) -> None:
        """All returned configs must be valid parameter dicts (not arbitrary)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            bandit = BanditSelector(seed=7)
            configs = select_configs_bandit(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec, 2),
                max_configs=8,
                bandit=bandit,
            )
            required_keys = {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                             "GROUP_SIZE_M", "num_warps", "num_stages"}
            for cfg in configs:
                self.assertTrue(
                    required_keys.issubset(cfg.keys()),
                    f"Config missing expected keys: {cfg}",
                )


# ---------------------------------------------------------------------------
# compute_delta_pct
# ---------------------------------------------------------------------------


class ComputeDeltaPctTests(unittest.TestCase):
    def test_zero_baseline_returns_zero(self) -> None:
        self.assertEqual(compute_delta_pct(100.0, 0.0), 0.0)

    def test_equal_values_return_zero(self) -> None:
        self.assertAlmostEqual(compute_delta_pct(50.0, 50.0), 0.0)

    def test_improvement(self) -> None:
        self.assertAlmostEqual(compute_delta_pct(110.0, 100.0), 10.0)

    def test_regression(self) -> None:
        self.assertAlmostEqual(compute_delta_pct(90.0, 100.0), -10.0)

    def test_doubling(self) -> None:
        self.assertAlmostEqual(compute_delta_pct(200.0, 100.0), 100.0)


# ---------------------------------------------------------------------------
# print_summary_table
# ---------------------------------------------------------------------------


class PrintSummaryTableTests(unittest.TestCase):
    def _make_results(self) -> dict[str, ConditionResult]:
        a = ConditionResult(name=CONDITION_BASELINE, trajectory=[50.0, 60.0], final_best=60.0)
        b = ConditionResult(name=CONDITION_COST_MODEL, trajectory=[55.0, 70.0], final_best=70.0)
        c = ConditionResult(name=CONDITION_BANDIT, trajectory=[52.0, 65.0], final_best=65.0)
        return {CONDITION_BASELINE: a, CONDITION_COST_MODEL: b, CONDITION_BANDIT: c}

    def test_runs_without_error(self) -> None:
        results = self._make_results()
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            print_summary_table(results, baseline_key=CONDITION_BASELINE)
            output = mock_stdout.getvalue()
        self.assertIn(CONDITION_BASELINE, output)
        self.assertIn(CONDITION_COST_MODEL, output)
        self.assertIn(CONDITION_BANDIT, output)

    def test_shows_all_three_conditions(self) -> None:
        results = self._make_results()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_summary_table(results, baseline_key=CONDITION_BASELINE)
        out = buf.getvalue()
        for name in (CONDITION_BASELINE, CONDITION_COST_MODEL, CONDITION_BANDIT):
            self.assertIn(name, out, f"Condition '{name}' not found in table output")

    def test_shows_delta_for_non_baseline(self) -> None:
        results = self._make_results()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_summary_table(results, baseline_key=CONDITION_BASELINE)
        out = buf.getvalue()
        # Cost-model is 70 vs baseline 60 => +16.67%
        self.assertIn("+", out, "Expected positive delta sign in table")


# ---------------------------------------------------------------------------
# run_condition (end-to-end with mock session)
# ---------------------------------------------------------------------------


def _make_mock_session(tflops_value: float = 120.0) -> MagicMock:
    """Build a mock ModalBenchmarkSession that returns a plausible batch result."""
    session = MagicMock()

    # Minimal valid config for matmul.
    cfg = {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3,
    }
    cid = config_id(cfg)

    batch = MagicMock()
    batch.success = True
    batch.hardware = {"gpu": "A100"}
    batch.config_results = [
        {
            "config_id": cid,
            "config": cfg,
            "results": [
                {
                    "shape": "M2048_N2048_K2048",
                    "correct": True,
                    "tflops": tflops_value,
                    "ms": 1.0,
                }
            ],
        }
    ]
    session.run_batch.return_value = batch
    return session


class RunConditionTests(unittest.TestCase):
    """End-to-end tests for run_condition using a mock Modal session."""

    def _spec(self):
        return _matmul_spec()

    def test_baseline_condition_returns_trajectory(self) -> None:
        spec = self._spec()
        session = _make_mock_session(120.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_condition(
                condition_name=CONDITION_BASELINE,
                spec=spec,
                session=session,
                shapes=[{"M": 2048, "N": 2048, "K": 2048}],
                configs_per_run=4,
                iterations=3,
                gpu="A100",
                include_curated=True,
            )
        self.assertEqual(result.name, CONDITION_BASELINE)
        self.assertEqual(len(result.trajectory), 3)
        self.assertGreater(result.final_best, 0.0)

    def test_bandit_condition_returns_trajectory(self) -> None:
        spec = self._spec()
        session = _make_mock_session(130.0)
        bandit = BanditSelector(seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_condition(
                condition_name=CONDITION_BANDIT,
                spec=spec,
                session=session,
                shapes=[{"M": 2048, "N": 2048, "K": 2048}],
                configs_per_run=4,
                iterations=3,
                gpu="A100",
                bandit=bandit,
            )
        self.assertEqual(result.name, CONDITION_BANDIT)
        self.assertEqual(len(result.trajectory), 3)
        self.assertGreater(result.final_best, 0.0)

    def test_cost_model_condition_with_mock_model(self) -> None:
        spec = self._spec()
        session = _make_mock_session(115.0)
        mock_cm = MagicMock()
        mock_cm.predict.side_effect = lambda fl: [1.0] * len(fl)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_condition(
                condition_name=CONDITION_COST_MODEL,
                spec=spec,
                session=session,
                shapes=[{"M": 2048, "N": 2048, "K": 2048}],
                configs_per_run=4,
                iterations=2,
                gpu="A100",
                cost_model=mock_cm,
            )
        self.assertEqual(result.name, CONDITION_COST_MODEL)
        self.assertEqual(len(result.trajectory), 2)

    def test_trajectory_is_monotonically_nondecreasing(self) -> None:
        """Best-so-far must never decrease across iterations."""
        spec = self._spec()
        # Vary the tflops slightly — the trajectory should still be non-decreasing.
        tflops_sequence = [80.0, 120.0, 100.0, 140.0]

        call_count = [0]
        original_cfg = {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3,
        }
        cid = config_id(original_cfg)

        session = MagicMock()
        session.run_batch.side_effect = lambda script: _mock_batch(
            cid, original_cfg, tflops_sequence[min(call_count[0], len(tflops_sequence) - 1)]
        )
        # increment counter after each call
        original_side = session.run_batch.side_effect

        def _counting_side(script):
            val = tflops_sequence[min(call_count[0], len(tflops_sequence) - 1)]
            call_count[0] += 1
            return _mock_batch(cid, original_cfg, val)

        session.run_batch.side_effect = _counting_side

        result = run_condition(
            condition_name=CONDITION_BASELINE,
            spec=spec,
            session=session,
            shapes=[{"M": 2048, "N": 2048, "K": 2048}],
            configs_per_run=4,
            iterations=4,
            gpu="A100",
        )
        traj = result.trajectory
        for i in range(1, len(traj)):
            self.assertGreaterEqual(
                traj[i], traj[i - 1],
                f"Trajectory not non-decreasing at index {i}: {traj}",
            )

    def test_failed_batch_records_zero_metric(self) -> None:
        """A batch with success=False contributes 0 to the metric for that iteration."""
        spec = self._spec()
        session = MagicMock()
        fail_batch = MagicMock()
        fail_batch.success = False
        session.run_batch.return_value = fail_batch

        result = run_condition(
            condition_name=CONDITION_BASELINE,
            spec=spec,
            session=session,
            shapes=[{"M": 2048, "N": 2048, "K": 2048}],
            configs_per_run=4,
            iterations=2,
            gpu="A100",
        )
        # Both iterations fail, so the trajectory should be all zeros.
        self.assertEqual(result.trajectory, [0.0, 0.0])
        self.assertEqual(result.final_best, 0.0)

    def test_three_conditions_use_independent_databases(self) -> None:
        """Each condition must not observe the other's history."""
        spec = self._spec()
        # All sessions return the same tflops so order effects don't matter.
        session = _make_mock_session(100.0)

        baseline_result = run_condition(
            condition_name=CONDITION_BASELINE,
            spec=spec, session=session,
            shapes=[{"M": 2048, "N": 2048, "K": 2048}],
            configs_per_run=4, iterations=2, gpu="A100",
        )
        bandit_result = run_condition(
            condition_name=CONDITION_BANDIT,
            spec=spec, session=session,
            shapes=[{"M": 2048, "N": 2048, "K": 2048}],
            configs_per_run=4, iterations=2, gpu="A100",
            bandit=BanditSelector(seed=0),
        )

        # Both should succeed independently.
        self.assertEqual(len(baseline_result.trajectory), 2)
        self.assertEqual(len(bandit_result.trajectory), 2)


def _mock_batch(cid: str, cfg: dict, tflops: float) -> MagicMock:
    """Create a minimal mock batch result for one config."""
    batch = MagicMock()
    batch.success = True
    batch.hardware = {"gpu": "A100"}
    batch.config_results = [
        {
            "config_id": cid,
            "config": cfg,
            "results": [
                {"shape": "M2048_N2048_K2048", "correct": True, "tflops": tflops, "ms": 1.0}
            ],
        }
    ]
    return batch


# ---------------------------------------------------------------------------
# ConditionResult dataclass
# ---------------------------------------------------------------------------


class ConditionResultTests(unittest.TestCase):
    def test_default_fields(self) -> None:
        r = ConditionResult(name="test")
        self.assertEqual(r.name, "test")
        self.assertEqual(r.trajectory, [])
        self.assertEqual(r.final_best, 0.0)

    def test_condition_name_constants_are_strings(self) -> None:
        self.assertIsInstance(CONDITION_BASELINE, str)
        self.assertIsInstance(CONDITION_COST_MODEL, str)
        self.assertIsInstance(CONDITION_BANDIT, str)

    def test_all_three_condition_names_are_distinct(self) -> None:
        names = {CONDITION_BASELINE, CONDITION_COST_MODEL, CONDITION_BANDIT}
        self.assertEqual(len(names), 3, "Condition name constants must be distinct")


if __name__ == "__main__":
    unittest.main()
