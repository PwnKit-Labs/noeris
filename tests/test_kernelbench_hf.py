from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401

from research_engine.kernelbench_hf import (
    KernelBenchProblem,
    compute_coverage,
    match_operator,
    problems_to_benchmark_shapes,
    _extract_shape_from_code,
)


class OperatorMatchingTests(unittest.TestCase):
    def test_matmul_patterns(self) -> None:
        self.assertEqual(match_operator("y = torch.matmul(x, w)"), "matmul")
        self.assertEqual(match_operator("out = self.linear(x)\nlinear = torch.nn.Linear(64, 128)"), "matmul")

    def test_softmax_patterns(self) -> None:
        self.assertEqual(match_operator("y = F.softmax(x, dim=-1)"), "softmax")
        self.assertEqual(match_operator("y = torch.softmax(x, -1)"), "softmax")

    def test_layernorm_patterns(self) -> None:
        self.assertEqual(match_operator("y = F.layer_norm(x, (1024,))"), "layernorm")
        self.assertEqual(match_operator("self.norm = torch.nn.LayerNorm(1024)"), "layernorm")

    def test_attention_pattern(self) -> None:
        code = "y = F.scaled_dot_product_attention(q, k, v)"
        self.assertEqual(match_operator(code), "attention")

    def test_cross_entropy_pattern(self) -> None:
        self.assertEqual(match_operator("loss = F.cross_entropy(logits, target)"), "cross_entropy")

    def test_rmsnorm_pattern(self) -> None:
        code = "rms = torch.rsqrt(x.pow(2).mean(-1) + eps)"
        self.assertEqual(match_operator(code), "rmsnorm")

    def test_attention_takes_priority_over_matmul(self) -> None:
        # Attention code often contains matmul too — make sure attention wins
        code = """
        scores = torch.matmul(q, k.transpose(-2, -1))
        return F.scaled_dot_product_attention(q, k, v)
        """
        self.assertEqual(match_operator(code), "attention")

    def test_no_match_returns_none(self) -> None:
        self.assertIsNone(match_operator("return x + y"))


class ShapeExtractionTests(unittest.TestCase):
    def test_extract_matmul_shape(self) -> None:
        code = """
        def get_inputs():
            a = torch.randn((2048, 1024))
            b = torch.randn((1024, 512))
            return [a, b]
        """
        shape = _extract_shape_from_code(code, "matmul")
        self.assertIsNotNone(shape)
        self.assertEqual(shape["M"], 2048)
        self.assertEqual(shape["N"], 512)
        self.assertEqual(shape["K"], 1024)

    def test_extract_layernorm_shape(self) -> None:
        code = """
        def get_inputs():
            return [torch.randn((4096, 768))]
        """
        shape = _extract_shape_from_code(code, "layernorm")
        self.assertIsNotNone(shape)
        self.assertEqual(shape["n_rows"], 4096)
        self.assertEqual(shape["hidden_dim"], 768)

    def test_extract_attention_shape(self) -> None:
        code = """
        def get_inputs():
            q = torch.randn((2, 16, 2048, 64))
            k = torch.randn((2, 16, 2048, 64))
            v = torch.randn((2, 16, 2048, 64))
            return [q, k, v]
        """
        shape = _extract_shape_from_code(code, "attention")
        self.assertIsNotNone(shape)
        self.assertEqual(shape["batch"], 2)
        self.assertEqual(shape["heads"], 16)
        self.assertEqual(shape["seq_len"], 2048)
        self.assertEqual(shape["head_dim"], 64)

    def test_unextractable_shape_returns_fallback(self) -> None:
        code = "return x"
        self.assertIsNone(_extract_shape_from_code(code, "matmul"))


class CoverageReportTests(unittest.TestCase):
    def test_coverage_counts(self) -> None:
        problems = [
            KernelBenchProblem(
                problem_id="1", level=1, name="matmul_simple",
                code="torch.matmul(a, b)", matched_operator="matmul",
            ),
            KernelBenchProblem(
                problem_id="2", level=1, name="softmax_simple",
                code="F.softmax(x, -1)", matched_operator="softmax",
            ),
            KernelBenchProblem(
                problem_id="3", level=2, name="unknown_op",
                code="return x + y", matched_operator=None,
            ),
        ]
        report = compute_coverage(problems)
        self.assertEqual(report.total, 3)
        self.assertEqual(report.supported, 2)
        self.assertEqual(report.by_operator["matmul"], 1)
        self.assertEqual(report.by_operator["softmax"], 1)
        self.assertEqual(report.by_level[1], 2)
        self.assertEqual(report.by_level[2], 1)

    def test_problems_to_benchmark_shapes(self) -> None:
        problems = [
            KernelBenchProblem(
                problem_id="1", level=1, name="m1",
                code="a = torch.randn((1024, 512))\nb = torch.randn((512, 256))",
                matched_operator="matmul",
            ),
        ]
        mapped = problems_to_benchmark_shapes(problems)
        self.assertIn("matmul", mapped)
        self.assertEqual(len(mapped["matmul"]), 1)
        self.assertEqual(mapped["matmul"][0]["M"], 1024)


if __name__ == "__main__":
    unittest.main()
