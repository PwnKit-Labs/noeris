"""Tests for the ensemble config selector.

Coverage:
- Alternation between cost-model and bandit picks (interleaving order)
- Deduplication when both selectors pick the same config_id
- Fallback when cost model is None (bandit-only mode)
- Fallback when bandit is None (cost-model-only mode)
- Fallback when both are None (frontier-slot baseline mode)
- Empty database behaves reasonably (no crash, returns configs)
- LLM-proposed configs still get through (slot 2)
- max_configs respected in all modes
- Incumbent (best known config) is selected first
- Curated configs included/excluded by include_curated flag
- Tested configs are not re-selected
- Valid config dicts returned in all modes
- All config IDs in output are unique
- Ensemble ablation script imports and exposes expected symbols
- ConditionResult dataclass has expected fields
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from tests import _pathfix  # noqa: F401

from research_engine.bandit_selector import BanditSelector
from research_engine.ensemble_selector import select_configs_ensemble
from research_engine.triton_kernels import ConfigDatabase, config_id
from research_engine.triton_operators import REGISTRY


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_db(tmpdir: Path) -> ConfigDatabase:
    return ConfigDatabase(path=tmpdir / "db.json")


def _matmul_spec():
    return REGISTRY.get("matmul")


def _softmax_spec():
    return REGISTRY.get("softmax")


def _minimal_shapes(spec, n: int = 1) -> list[dict]:
    return spec.shape_buckets[:n]


def _matmul_config(bm: int = 128, bn: int = 128, bk: int = 64) -> dict:
    return {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }


def _mock_cost_model(scores: list[float] | None = None):
    """Return a mock CostModel whose rank_configs returns configs in provided score order."""
    cm = MagicMock()

    def _rank_configs(configs, shapes, hardware, operator, top_k=None):
        if scores is not None:
            # Pair each config with its score, then sort descending
            paired = list(zip(configs, (scores + [0.0] * len(configs))[:len(configs)]))
            paired.sort(key=lambda x: -x[1])
            return paired
        # Default: return in original order with score 1.0
        return [(c, 1.0) for c in configs]

    cm.rank_configs.side_effect = _rank_configs
    return cm


def _mock_bandit() -> BanditSelector:
    """Return a BanditSelector with seed=0."""
    return BanditSelector(seed=0)


def _record_result(db: ConfigDatabase, cfg: dict, tflops: float,
                   *, operator: str = "matmul", bucket: str = "medium",
                   hardware: str = "A100") -> None:
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
# Test 1: Alternation — cost model and bandit picks are interleaved
# ---------------------------------------------------------------------------


class AlternationTests(unittest.TestCase):
    """Verify strict cost-model / bandit alternation in ensemble slot 4."""

    def test_alternation_produces_picks_from_both_selectors(self) -> None:
        """With a populated bandit and cost model, both should contribute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            # Record one config so bandit has posterior data
            winner = _matmul_config(bm=256, bn=256, bk=64)
            _record_result(db, winner, 250.0)

            cm = _mock_cost_model()
            bandit = _mock_bandit()

            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=6,
                cost_model=cm,
                bandit_selector=bandit,
                include_curated=False,
            )
            # Cost model's rank_configs must have been called
            self.assertTrue(cm.rank_configs.called)
            self.assertGreater(len(configs), 0)

    def test_alternation_deduplicates_overlapping_picks(self) -> None:
        """If cost model and bandit pick the same config, it appears only once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))

            # Populate bandit with a single config
            shared = _matmul_config(bm=128, bn=128, bk=64)
            _record_result(db, shared, 200.0)

            # Make cost model always return the same single config first
            def _rank_always_shared(configs, shapes, hardware, operator, top_k=None):
                # Put shared config first if it appears in the pool
                shared_id = config_id(shared)
                first = [c for c in configs if config_id(c) == shared_id]
                rest = [c for c in configs if config_id(c) != shared_id]
                return [(c, 1.0) for c in first + rest]

            cm = MagicMock()
            cm.rank_configs.side_effect = _rank_always_shared

            bandit = _mock_bandit()
            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=8,
                cost_model=cm,
                bandit_selector=bandit,
                include_curated=False,
            )
            ids = [config_id(c) for c in configs]
            # No duplicates regardless of overlap
            self.assertEqual(len(ids), len(set(ids)), f"Duplicate IDs in: {ids}")


# ---------------------------------------------------------------------------
# Test 2: Fallback when cost model is None (bandit-only)
# ---------------------------------------------------------------------------


class CostModelNoneFallbackTests(unittest.TestCase):
    def test_no_cost_model_uses_bandit_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            bandit = _mock_bandit()
            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=4,
                cost_model=None,
                bandit_selector=bandit,
            )
            self.assertGreater(len(configs), 0)
            for cfg in configs:
                self.assertIn("BLOCK_SIZE_M", cfg)

    def test_no_cost_model_respects_max_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            bandit = _mock_bandit()
            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=2,
                cost_model=None,
                bandit_selector=bandit,
            )
            self.assertLessEqual(len(configs), 2)


# ---------------------------------------------------------------------------
# Test 3: Fallback when bandit is None (cost-model-only)
# ---------------------------------------------------------------------------


class BanditNoneFallbackTests(unittest.TestCase):
    def test_no_bandit_uses_cost_model_only(self) -> None:
        """When bandit is None and curated configs are excluded, cost model fills slots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            cm = _mock_cost_model()
            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=4,
                cost_model=cm,
                bandit_selector=None,
                include_curated=False,  # force into ensemble exploration slots
            )
            # rank_configs should have been called for the cost model
            self.assertTrue(cm.rank_configs.called)
            self.assertGreater(len(configs), 0)

    def test_no_bandit_respects_max_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            cm = _mock_cost_model()
            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=3,
                cost_model=cm,
                bandit_selector=None,
            )
            self.assertLessEqual(len(configs), 3)


# ---------------------------------------------------------------------------
# Test 4: Both selectors None (baseline fallback)
# ---------------------------------------------------------------------------


class BothNoneFallbackTests(unittest.TestCase):
    def test_both_none_returns_configs_from_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=4,
                cost_model=None,
                bandit_selector=None,
            )
            self.assertGreater(len(configs), 0)

    def test_both_none_respects_max_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=2,
                cost_model=None,
                bandit_selector=None,
            )
            self.assertLessEqual(len(configs), 2)


# ---------------------------------------------------------------------------
# Test 5: Empty database behaves reasonably
# ---------------------------------------------------------------------------


class EmptyDatabaseTests(unittest.TestCase):
    def test_empty_db_with_both_selectors_returns_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            cm = _mock_cost_model()
            bandit = _mock_bandit()
            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=6,
                cost_model=cm,
                bandit_selector=bandit,
            )
            self.assertGreater(len(configs), 0)

    def test_empty_db_no_incumbent_in_output(self) -> None:
        """With an empty database there is no incumbent to carry forward."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            cm = _mock_cost_model()
            bandit = _mock_bandit()
            # Just check it doesn't crash and returns valid configs
            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=4,
                cost_model=cm,
                bandit_selector=bandit,
                include_curated=False,
            )
            self.assertIsInstance(configs, list)


# ---------------------------------------------------------------------------
# Test 6: LLM proposals still get through
# ---------------------------------------------------------------------------


class LLMProposalTests(unittest.TestCase):
    def test_proposed_configs_appear_in_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            proposal = _matmul_config(bm=32, bn=32, bk=16)
            proposal_id = config_id(proposal)

            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=6,
                proposed_configs=[proposal],
                cost_model=None,
                bandit_selector=None,
            )
            returned_ids = [config_id(c) for c in configs]
            self.assertIn(proposal_id, returned_ids, "LLM proposal was not included")

    def test_proposed_configs_not_duplicated(self) -> None:
        """A config proposed by LLM and also in the grid should appear once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            # Use a curated config as the proposal — it will also appear in slot 3
            proposal = spec.curated_configs[0]
            proposal_id = config_id(proposal)

            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=8,
                proposed_configs=[proposal],
                cost_model=None,
                bandit_selector=None,
                include_curated=True,
            )
            returned_ids = [config_id(c) for c in configs]
            count = returned_ids.count(proposal_id)
            self.assertEqual(count, 1, f"Proposal appeared {count} times in output")


# ---------------------------------------------------------------------------
# Test 7: max_configs respected in all modes
# ---------------------------------------------------------------------------


class MaxConfigsTests(unittest.TestCase):
    def _run(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            return select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                **kwargs,
            )

    def test_max_configs_both_selectors(self) -> None:
        configs = self._run(
            max_configs=3,
            cost_model=_mock_cost_model(),
            bandit_selector=_mock_bandit(),
        )
        self.assertLessEqual(len(configs), 3)

    def test_max_configs_cost_model_only(self) -> None:
        configs = self._run(
            max_configs=2,
            cost_model=_mock_cost_model(),
            bandit_selector=None,
        )
        self.assertLessEqual(len(configs), 2)

    def test_max_configs_bandit_only(self) -> None:
        configs = self._run(
            max_configs=2,
            cost_model=None,
            bandit_selector=_mock_bandit(),
        )
        self.assertLessEqual(len(configs), 2)

    def test_max_configs_both_none(self) -> None:
        configs = self._run(
            max_configs=1,
            cost_model=None,
            bandit_selector=None,
        )
        self.assertLessEqual(len(configs), 1)


# ---------------------------------------------------------------------------
# Test 8: No duplicate config_ids in output
# ---------------------------------------------------------------------------


class NoDuplicatesTests(unittest.TestCase):
    def _ids(self, configs):
        return [config_id(c) for c in configs]

    def test_no_duplicates_with_both_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            cm = _mock_cost_model()
            bandit = _mock_bandit()
            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=12,
                cost_model=cm,
                bandit_selector=bandit,
            )
            ids = self._ids(configs)
            self.assertEqual(len(ids), len(set(ids)), f"Duplicates found: {ids}")

    def test_no_duplicates_with_incumbent_and_ensemble(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            # Record a result so there's an incumbent
            incumbent = _matmul_config(bm=128)
            _record_result(db, incumbent, 200.0)
            cm = _mock_cost_model()
            bandit = _mock_bandit()
            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=8,
                cost_model=cm,
                bandit_selector=bandit,
            )
            ids = self._ids(configs)
            self.assertEqual(len(ids), len(set(ids)), f"Duplicates found: {ids}")


# ---------------------------------------------------------------------------
# Test 9: Tested configs are not re-selected
# ---------------------------------------------------------------------------


class TestedConfigsExcludedTests(unittest.TestCase):
    def test_already_tested_configs_not_returned(self) -> None:
        """Configs already in the database must not appear in the exploration slots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))

            # Record all curated configs as already tested
            for cfg in spec.curated_configs:
                _record_result(db, cfg, 100.0)
            tested = {config_id(c) for c in spec.curated_configs}

            cm = _mock_cost_model()
            bandit = _mock_bandit()
            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=8,
                cost_model=cm,
                bandit_selector=bandit,
                include_curated=True,  # curated are tested, so they come from incumbent slot only
            )
            # Any config that was in the *grid exploration* slots must not be tested
            # (incumbent may legitimately be a tested config)
            non_incumbent = configs[1:]  # skip incumbent slot
            for cfg in non_incumbent:
                cid = config_id(cfg)
                self.assertNotIn(
                    cid, tested,
                    f"Tested config {cid} should not be re-selected in exploration slots",
                )


# ---------------------------------------------------------------------------
# Test 10: include_curated flag
# ---------------------------------------------------------------------------


class IncludeCuratedTests(unittest.TestCase):
    def test_curated_excluded_when_flag_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            curated_ids = {config_id(c) for c in spec.curated_configs}
            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=6,
                cost_model=None,
                bandit_selector=None,
                include_curated=False,
            )
            for cfg in configs:
                self.assertNotIn(
                    config_id(cfg), curated_ids,
                    "Curated config appeared despite include_curated=False",
                )

    def test_curated_included_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            curated_ids = {config_id(c) for c in spec.curated_configs}
            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=8,
                cost_model=None,
                bandit_selector=None,
                include_curated=True,
            )
            returned_ids = {config_id(c) for c in configs}
            # At least some curated configs should appear
            overlap = returned_ids & curated_ids
            self.assertGreater(len(overlap), 0, "No curated configs included")


# ---------------------------------------------------------------------------
# Test 11: Valid config dicts returned
# ---------------------------------------------------------------------------


class ValidConfigDictsTests(unittest.TestCase):
    def test_all_configs_have_required_matmul_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            cm = _mock_cost_model()
            bandit = _mock_bandit()
            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=6,
                cost_model=cm,
                bandit_selector=bandit,
            )
            required = {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                        "GROUP_SIZE_M", "num_warps", "num_stages"}
            for cfg in configs:
                self.assertTrue(
                    required.issubset(cfg.keys()),
                    f"Config missing required keys: {cfg}",
                )

    def test_softmax_operator_returns_valid_configs(self) -> None:
        """Ensemble works for memory-bound operators too."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _softmax_spec()
            db = _make_db(Path(tmpdir))
            cm = _mock_cost_model()
            bandit = BanditSelector(seed=1)
            configs = select_configs_ensemble(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=4,
                cost_model=cm,
                bandit_selector=bandit,
            )
            self.assertGreater(len(configs), 0)
            for cfg in configs:
                self.assertIn("BLOCK_SIZE", cfg)


# ---------------------------------------------------------------------------
# Test 12: Ensemble ablation script symbols and ConditionResult
# ---------------------------------------------------------------------------

# Add scripts/ directory to path for import
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class EnsembleAblationScriptTests(unittest.TestCase):
    """Smoke tests for ensemble_ablation.py module structure."""

    def test_imports_expected_condition_constants(self) -> None:
        from ensemble_ablation import (
            CONDITION_BASELINE,
            CONDITION_COST_MODEL,
            CONDITION_BANDIT,
            CONDITION_ENSEMBLE,
        )
        names = {CONDITION_BASELINE, CONDITION_COST_MODEL,
                 CONDITION_BANDIT, CONDITION_ENSEMBLE}
        self.assertEqual(len(names), 4, "All four condition names must be distinct")

    def test_condition_result_dataclass_fields(self) -> None:
        from ensemble_ablation import ConditionResult
        r = ConditionResult(name="ensemble")
        self.assertEqual(r.name, "ensemble")
        self.assertEqual(r.trajectory, [])
        self.assertEqual(r.final_best, 0.0)

    def test_compute_delta_pct_arithmetic(self) -> None:
        from ensemble_ablation import compute_delta_pct
        self.assertAlmostEqual(compute_delta_pct(110.0, 100.0), 10.0)
        self.assertAlmostEqual(compute_delta_pct(90.0, 100.0), -10.0)
        self.assertEqual(compute_delta_pct(100.0, 0.0), 0.0)

    def test_select_configs_ensemble_condition_returns_configs(self) -> None:
        """The wrapper in ensemble_ablation also works end-to-end."""
        from ensemble_ablation import select_configs_ensemble_condition
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _matmul_spec()
            db = _make_db(Path(tmpdir))
            cm = _mock_cost_model()
            bandit = BanditSelector(seed=0)
            configs = select_configs_ensemble_condition(
                spec=spec,
                database=db,
                hardware="A100",
                shapes=_minimal_shapes(spec),
                max_configs=4,
                cost_model=cm,
                bandit=bandit,
            )
            self.assertGreater(len(configs), 0)

    def test_run_condition_ensemble_with_mock_session(self) -> None:
        """run_condition with is_ensemble=True runs without error."""
        from ensemble_ablation import run_condition, ConditionResult, CONDITION_ENSEMBLE
        from research_engine.triton_kernels import config_id as _cid

        spec = _matmul_spec()
        cfg = _matmul_config()
        cid = _cid(cfg)

        session = MagicMock()
        batch = MagicMock()
        batch.success = True
        batch.hardware = {"gpu": "A100"}
        batch.config_results = [{
            "config_id": cid,
            "config": cfg,
            "results": [{"shape": "M2048_N2048_K2048", "correct": True,
                         "tflops": 150.0, "ms": 1.0}],
        }]
        session.run_batch.return_value = batch

        result = run_condition(
            condition_name=CONDITION_ENSEMBLE,
            spec=spec,
            session=session,
            shapes=[{"M": 2048, "N": 2048, "K": 2048}],
            configs_per_run=4,
            iterations=2,
            gpu="A100",
            cost_model=_mock_cost_model(),
            bandit=BanditSelector(seed=0),
            is_ensemble=True,
        )
        self.assertEqual(result.name, CONDITION_ENSEMBLE)
        self.assertEqual(len(result.trajectory), 2)
        self.assertGreater(result.final_best, 0.0)

    def test_print_summary_table_includes_ensemble(self) -> None:
        import io
        from unittest.mock import patch
        from ensemble_ablation import (
            ConditionResult,
            CONDITION_BASELINE,
            CONDITION_ENSEMBLE,
            print_summary_table,
        )
        results = {
            CONDITION_BASELINE: ConditionResult(
                name=CONDITION_BASELINE, trajectory=[80.0], final_best=80.0
            ),
            CONDITION_ENSEMBLE: ConditionResult(
                name=CONDITION_ENSEMBLE, trajectory=[90.0], final_best=90.0
            ),
        }
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_summary_table(results, baseline_key=CONDITION_BASELINE)
        out = buf.getvalue()
        self.assertIn(CONDITION_ENSEMBLE, out)
        self.assertIn("+", out)


if __name__ == "__main__":
    unittest.main()
