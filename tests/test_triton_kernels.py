from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests import _pathfix  # noqa: F401

from research_engine.triton_kernels import (
    MATMUL_FAMILY_CONFIGS,
    MATMUL_SHAPE_BUCKETS,
    TRITON_MATMUL_CURATED_CONFIGS,
    ConfigDatabase,
    config_id,
    generate_config_grid,
    generate_triton_benchmark_script,
    matmul_family_key,
    routed_matmul_config,
    select_configs_for_run,
    shape_bucket_key,
)


class ConfigIdTests(unittest.TestCase):
    def test_config_id_is_stable(self) -> None:
        config = {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_warps": 8,
            "num_stages": 3,
        }
        self.assertEqual(config_id(config), "bm128_bn256_bk64_gm8_w8_s3")

    def test_different_configs_have_different_ids(self) -> None:
        ids = {config_id(c) for c in TRITON_MATMUL_CURATED_CONFIGS}
        self.assertEqual(len(ids), len(TRITON_MATMUL_CURATED_CONFIGS))


class ShapeBucketTests(unittest.TestCase):
    def test_tiny_shape(self) -> None:
        self.assertEqual(shape_bucket_key(64, 64, 64), "tiny")

    def test_large_square(self) -> None:
        self.assertEqual(shape_bucket_key(4096, 4096, 4096), "xlarge")

    def test_deep_k(self) -> None:
        self.assertEqual(shape_bucket_key(1024, 1024, 8192), "deep_k")

    def test_tall_skinny(self) -> None:
        self.assertEqual(shape_bucket_key(8192, 1024, 1024), "tall_skinny")


class MatmulFamilyRoutingTests(unittest.TestCase):
    def test_routes_square_dense_family(self) -> None:
        self.assertEqual(matmul_family_key(4096, 4096, 4096), "square_dense")

    def test_routes_irregular_masked_family(self) -> None:
        self.assertEqual(matmul_family_key(8205, 5921, 2949), "irregular_masked")

    def test_routes_small_k_family(self) -> None:
        self.assertEqual(matmul_family_key(32768, 32768, 64), "small_k")

    def test_routes_tall_skinny_family(self) -> None:
        self.assertEqual(matmul_family_key(32768, 32, 32768), "tall_skinny")

    def test_routes_large_k_family(self) -> None:
        self.assertEqual(matmul_family_key(256, 256, 524288), "large_k")

    def test_routed_config_matches_family_defaults(self) -> None:
        cfg = routed_matmul_config(32768, 32768, 64)
        self.assertEqual(cfg, MATMUL_FAMILY_CONFIGS["small_k"])
        self.assertTrue(cfg["OUTPUT_FP32"])

    def test_large_k_family_uses_splitk(self) -> None:
        cfg = routed_matmul_config(256, 256, 524288)
        self.assertEqual(cfg, MATMUL_FAMILY_CONFIGS["large_k"])
        self.assertEqual(cfg["SPLIT_K"], 32)


class ConfigGridTests(unittest.TestCase):
    def test_grid_includes_curated_configs(self) -> None:
        grid = generate_config_grid(include_curated=True)
        grid_ids = {config_id(c) for c in grid}
        for curated in TRITON_MATMUL_CURATED_CONFIGS:
            self.assertIn(config_id(curated), grid_ids)

    def test_grid_respects_max(self) -> None:
        grid = generate_config_grid(max_configs=20)
        self.assertLessEqual(len(grid), 20)

    def test_grid_has_no_duplicates(self) -> None:
        grid = generate_config_grid(max_configs=200)
        ids = [config_id(c) for c in grid]
        self.assertEqual(len(ids), len(set(ids)))

    def test_grid_produces_substantial_configs(self) -> None:
        grid = generate_config_grid(max_configs=500)
        self.assertGreater(len(grid), 50)


class ConfigDatabaseTests(unittest.TestCase):
    def test_round_trip_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "configs.json"
            db = ConfigDatabase(path=db_path)
            db.record_result(
                shape={"M": 4096, "N": 4096, "K": 4096},
                hardware="A100",
                config=TRITON_MATMUL_CURATED_CONFIGS[0],
                tflops=312.5,
                ms=0.45,
                correct=True,
                run_id="test-001",
            )
            db.save()

            db2 = ConfigDatabase(path=db_path)
            best = db2.get_best_config(
                shape={"M": 4096, "N": 4096, "K": 4096},
                hardware="A100",
            )
            self.assertIsNotNone(best)
            self.assertEqual(best["BLOCK_SIZE_M"], 128)

    def test_new_best_returns_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ConfigDatabase(path=Path(tmpdir) / "configs.json")
            result1 = db.record_result(
                shape={"M": 2048, "N": 2048, "K": 2048},
                hardware="A100",
                config=TRITON_MATMUL_CURATED_CONFIGS[0],
                tflops=200.0,
                ms=0.8,
                correct=True,
            )
            self.assertTrue(result1)

            result2 = db.record_result(
                shape={"M": 2048, "N": 2048, "K": 2048},
                hardware="A100",
                config=TRITON_MATMUL_CURATED_CONFIGS[1],
                tflops=150.0,
                ms=1.0,
                correct=True,
            )
            self.assertFalse(result2)

            result3 = db.record_result(
                shape={"M": 2048, "N": 2048, "K": 2048},
                hardware="A100",
                config=TRITON_MATMUL_CURATED_CONFIGS[2],
                tflops=250.0,
                ms=0.6,
                correct=True,
            )
            self.assertTrue(result3)

    def test_insights_reflect_best_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ConfigDatabase(path=Path(tmpdir) / "configs.json")
            db.record_result(
                shape={"M": 4096, "N": 4096, "K": 4096},
                hardware="A100",
                config=TRITON_MATMUL_CURATED_CONFIGS[0],
                tflops=300.0,
                ms=0.5,
                correct=True,
            )
            db.record_result(
                shape={"M": 4096, "N": 4096, "K": 4096},
                hardware="A100",
                config=TRITON_MATMUL_CURATED_CONFIGS[1],
                tflops=280.0,
                ms=0.6,
                correct=True,
            )
            insights = db.get_insights(hardware="A100")
            self.assertEqual(len(insights), 1)
            self.assertEqual(insights[0]["best_tflops"], 300.0)
            self.assertEqual(len(insights[0]["top_configs"]), 2)

    def test_incorrect_result_does_not_become_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ConfigDatabase(path=Path(tmpdir) / "configs.json")
            db.record_result(
                shape={"M": 2048, "N": 2048, "K": 2048},
                hardware="A100",
                config=TRITON_MATMUL_CURATED_CONFIGS[0],
                tflops=999.0,
                ms=0.01,
                correct=False,
            )
            best = db.get_best_config(
                shape={"M": 2048, "N": 2048, "K": 2048},
                hardware="A100",
            )
            self.assertIsNone(best)


class BenchmarkScriptTests(unittest.TestCase):
    def test_script_is_valid_python(self) -> None:
        script = generate_triton_benchmark_script(
            config=TRITON_MATMUL_CURATED_CONFIGS[0],
            shapes=[{"name": "test", "M": 512, "N": 512, "K": 512}],
        )
        compile(script, "<benchmark>", "exec")

    def test_script_embeds_config(self) -> None:
        config = TRITON_MATMUL_CURATED_CONFIGS[0]
        script = generate_triton_benchmark_script(
            config=config,
            shapes=[{"name": "test", "M": 512, "N": 512, "K": 512}],
        )
        self.assertIn(str(config["BLOCK_SIZE_M"]), script)
        self.assertIn("matmul_kernel", script)
        self.assertIn("check_correctness", script)
        self.assertIn("benchmark_config", script)


class SelectConfigsTests(unittest.TestCase):
    def test_select_respects_max(self) -> None:
        configs = select_configs_for_run(
            shapes=[{"M": 2048, "N": 2048, "K": 2048}],
            max_configs=5,
        )
        self.assertLessEqual(len(configs), 5)

    def test_select_includes_proposed(self) -> None:
        proposed = [{
            "BLOCK_SIZE_M": 192,
            "BLOCK_SIZE_N": 192,
            "BLOCK_SIZE_K": 48,
            "GROUP_SIZE_M": 12,
            "num_warps": 6,
            "num_stages": 3,
        }]
        configs = select_configs_for_run(
            shapes=[{"M": 2048, "N": 2048, "K": 2048}],
            proposed_configs=proposed,
            max_configs=8,
        )
        config_ids = {config_id(c) for c in configs}
        self.assertIn(config_id(proposed[0]), config_ids)

    def test_select_prioritizes_incumbent_from_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ConfigDatabase(path=Path(tmpdir) / "configs.json")
            winner = TRITON_MATMUL_CURATED_CONFIGS[5]
            db.record_result(
                shape={"M": 2048, "N": 2048, "K": 2048},
                hardware="A100",
                config=winner,
                tflops=350.0,
                ms=0.3,
                correct=True,
            )
            configs = select_configs_for_run(
                database=db,
                hardware="A100",
                shapes=[{"M": 2048, "N": 2048, "K": 2048}],
                max_configs=3,
            )
            self.assertEqual(config_id(configs[0]), config_id(winner))


if __name__ == "__main__":
    unittest.main()
