"""Tests for kernelbench_upstream.py (Task 4).

Exercises the non-CUDA parts: problem vendoring, Model.exec materialization,
script generation, and report serialization. The Modal-backed end-to-end
path is covered by the on-device experiment, not by unit tests.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests import _pathfix  # noqa: F401

from research_engine.kernelbench_upstream import (
    UPSTREAM_PROBLEMS,
    UpstreamProblem,
    UpstreamReport,
    UpstreamResult,
    generate_kernelbench_upstream_script,
    list_problem_files,
    load_problem_source,
    materialize_problem,
    problems_dir,
)


class TestProblemVendoring(unittest.TestCase):
    def test_vendored_dir_exists(self) -> None:
        self.assertTrue(problems_dir().exists())

    def test_at_least_12_problems_vendored(self) -> None:
        files = list_problem_files()
        self.assertGreaterEqual(len(files), 12, f"got {files}")

    def test_registry_references_only_vendored_files(self) -> None:
        vendored = set(list_problem_files())
        for p in UPSTREAM_PROBLEMS:
            self.assertIn(p.problem_file, vendored)

    def test_registry_covers_all_target_operators(self) -> None:
        ops = {p.noeris_operator for p in UPSTREAM_PROBLEMS}
        for required in ("matmul", "softmax", "rmsnorm", "layernorm",
                         "cross_entropy", "attention", "geglu"):
            self.assertIn(required, ops)

    def test_load_problem_source_returns_python_code(self) -> None:
        code = load_problem_source("1_Square_matrix_multiplication_.py")
        self.assertIn("class Model", code)
        self.assertIn("def get_inputs", code)

    def test_load_problem_source_missing_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_problem_source("nonexistent_problem.py")


class TestMaterialization(unittest.TestCase):
    """Verify that each vendored problem's source contains the expected
    upstream shape constants. We can't exec() them offline because every
    file imports torch, and this sub-agent environment does not have torch
    installed (Noeris only needs it inside Modal). Shape verification via
    source-text matching is good enough for the credibility pass — the
    actual exec+bench path runs on Modal.
    """

    def test_square_matmul_source_has_N_4096(self) -> None:
        src = load_problem_source("1_Square_matrix_multiplication_.py")
        self.assertIn("N = 2048 * 2", src)  # i.e. 4096

    def test_rmsnorm_source_has_4d_shape_constants(self) -> None:
        src = load_problem_source("36_RMSNorm_.py")
        self.assertIn("batch_size = 112", src)
        self.assertIn("features = 64", src)
        self.assertIn("dim1 = 512", src)
        self.assertIn("dim2 = 512", src)

    def test_layernorm_source_has_4d_shape_constants(self) -> None:
        src = load_problem_source("40_LayerNorm.py")
        self.assertIn("batch_size = 16", src)
        self.assertIn("features = 64", src)
        self.assertIn("dim1 = 256", src)
        self.assertIn("dim2 = 256", src)

    def test_cross_entropy_source_has_32768_4096(self) -> None:
        src = load_problem_source("95_CrossEntropyLoss.py")
        self.assertIn("batch_size = 32768", src)
        self.assertIn("num_classes = 4096", src)

    def test_softmax_source_has_4096_393216(self) -> None:
        src = load_problem_source("23_Softmax.py")
        self.assertIn("batch_size = 4096", src)
        self.assertIn("dim = 393216", src)

    def test_sdpa_source_has_upstream_constants(self) -> None:
        src = load_problem_source("97_ScaledDotProductAttention.py")
        self.assertIn("batch_size = 32", src)
        self.assertIn("num_heads = 32", src)
        self.assertIn("sequence_length = 512", src)
        self.assertIn("embedding_dimension = 1024", src)


class TestScriptGeneration(unittest.TestCase):
    def test_generated_script_is_valid_python(self) -> None:
        script = generate_kernelbench_upstream_script(UPSTREAM_PROBLEMS[:3])
        try:
            compile(script, "<upstream_runner>", "exec")
        except SyntaxError as exc:
            self.fail(f"Generated script is invalid Python: {exc}")

    def test_generated_script_sets_timer(self) -> None:
        script = generate_kernelbench_upstream_script(
            UPSTREAM_PROBLEMS[:1], timer="do_bench"
        )
        self.assertIn('NOERIS_TIMER = "do_bench"', script)

    def test_generated_script_embeds_problem_sources(self) -> None:
        # The first problem is 1_Square_matrix_multiplication_.py; its
        # source should appear verbatim in the generated script.
        script = generate_kernelbench_upstream_script([UPSTREAM_PROBLEMS[0]])
        self.assertIn("class Model", script)
        self.assertIn("torch.matmul(A, B)", script)

    def test_generated_script_embeds_timing_helper(self) -> None:
        script = generate_kernelbench_upstream_script([UPSTREAM_PROBLEMS[0]])
        self.assertIn("_noeris_time_cuda_event", script)
        self.assertIn("_noeris_clear_l2_cache", script)

    def test_generated_script_defines_all_noeris_adapters(self) -> None:
        script = generate_kernelbench_upstream_script(UPSTREAM_PROBLEMS)
        for op in ("matmul", "softmax", "rmsnorm", "layernorm",
                   "cross_entropy", "attention", "geglu"):
            self.assertIn(f'"{op}"', script)

    def test_generated_script_has_allclose_fp32_tolerance(self) -> None:
        script = generate_kernelbench_upstream_script([UPSTREAM_PROBLEMS[0]])
        self.assertIn("rtol=1e-4", script)
        self.assertIn("atol=1e-4", script)


class TestReportRendering(unittest.TestCase):
    def test_report_to_dict_round_trip(self) -> None:
        r = UpstreamReport(
            metadata={"hardware": "A100", "timer": "cuda_event"},
            results=[
                UpstreamResult(
                    problem="1_Square_matrix_multiplication_.py",
                    operator="matmul",
                    upstream_ms=2.5,
                    noeris_ms=1.2,
                    speedup=2.08,
                    correct=True,
                    external_h100_ms=2.66,
                ),
            ],
        )
        d = r.to_dict()
        text = json.dumps(d)
        recovered = json.loads(text)
        self.assertEqual(recovered["results"][0]["speedup"], 2.08)

    def test_summary_text_contains_problem_row(self) -> None:
        r = UpstreamReport(
            metadata={"hardware": "H100", "timer": "cuda_event"},
            results=[
                UpstreamResult(
                    problem="36_RMSNorm_.py",
                    operator="rmsnorm",
                    upstream_ms=14.0,
                    noeris_ms=6.0,
                    speedup=2.33,
                    correct=True,
                    external_h100_ms=14.2,
                ),
            ],
        )
        text = r.summary_text()
        self.assertIn("H100", text)
        self.assertIn("36_RMSNorm_.py", text)
        self.assertIn("2.33x", text)


if __name__ == "__main__":
    unittest.main()
