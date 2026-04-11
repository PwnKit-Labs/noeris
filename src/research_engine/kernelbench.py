"""KernelBench integration for comparable evaluation.

KernelBench (arXiv:2502.10517) is the standard benchmark for LLM-driven
GPU kernel generation. It has 250 problems across 4 difficulty levels.
The standard score is `fast_p`: the fraction of problems where the
generated kernel beats PyTorch by at least p*100% (p = 1.0, 1.5, 2.0, 3.0).

This module:
- Fetches a subset of problems mapped to our supported operators
- Runs our search loop on each problem
- Computes fast_p scores
- Produces a report comparable to published numbers
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# A curated subset of KernelBench-style problems that map onto operators
# we currently support. These are public shapes from the KernelBench Level 1
# and Level 2 problem sets. We include them as test cases, not as the
# official KernelBench loader (which requires HuggingFace datasets access).
KERNELBENCH_SUBSET = {
    "matmul": [
        {"id": "kb_L1_matmul_small", "M": 512, "N": 512, "K": 512, "level": 1},
        {"id": "kb_L1_matmul_medium", "M": 1024, "N": 1024, "K": 1024, "level": 1},
        {"id": "kb_L1_matmul_large", "M": 2048, "N": 2048, "K": 2048, "level": 1},
        {"id": "kb_L1_matmul_xlarge", "M": 4096, "N": 4096, "K": 4096, "level": 1},
        {"id": "kb_L1_matmul_tall", "M": 8192, "N": 1024, "K": 1024, "level": 1},
        {"id": "kb_L1_matmul_deep", "M": 1024, "N": 1024, "K": 8192, "level": 1},
        {"id": "kb_L2_matmul_llm_qkv", "M": 4096, "N": 4096, "K": 512, "level": 2},
        {"id": "kb_L2_matmul_llm_mlp", "M": 4096, "N": 11008, "K": 4096, "level": 2},
    ],
    "rmsnorm": [
        {"id": "kb_L1_rmsnorm_small", "n_rows": 1024, "hidden_dim": 768, "level": 1},
        {"id": "kb_L1_rmsnorm_base", "n_rows": 4096, "hidden_dim": 768, "level": 1},
        {"id": "kb_L2_rmsnorm_llama7b", "n_rows": 4096, "hidden_dim": 4096, "level": 2},
        {"id": "kb_L2_rmsnorm_llama13b", "n_rows": 4096, "hidden_dim": 5120, "level": 2},
    ],
    "softmax": [
        {"id": "kb_L1_softmax_small", "n_rows": 1024, "n_cols": 512, "level": 1},
        {"id": "kb_L1_softmax_medium", "n_rows": 2048, "n_cols": 1024, "level": 1},
        {"id": "kb_L1_softmax_large", "n_rows": 4096, "n_cols": 4096, "level": 1},
        {"id": "kb_L2_softmax_vocab", "n_rows": 2048, "n_cols": 32000, "level": 2},
    ],
    "layernorm": [
        {"id": "kb_L1_layernorm_small", "n_rows": 1024, "hidden_dim": 768, "level": 1},
        {"id": "kb_L1_layernorm_bert", "n_rows": 4096, "hidden_dim": 1024, "level": 1},
        {"id": "kb_L2_layernorm_gpt", "n_rows": 4096, "hidden_dim": 1600, "level": 2},
        {"id": "kb_L2_layernorm_large", "n_rows": 8192, "hidden_dim": 4096, "level": 2},
    ],
    "cross_entropy": [
        {"id": "kb_L1_ce_gpt2", "n_rows": 1024, "n_cols": 50257, "level": 1},
        {"id": "kb_L1_ce_llama", "n_rows": 2048, "n_cols": 32000, "level": 1},
        {"id": "kb_L2_ce_long_llama", "n_rows": 4096, "n_cols": 32000, "level": 2},
        {"id": "kb_L2_ce_llama3", "n_rows": 2048, "n_cols": 128256, "level": 2},
    ],
}


FAST_P_THRESHOLDS = [1.0, 1.5, 2.0, 3.0]


@dataclass(slots=True)
class ProblemResult:
    problem_id: str
    operator: str
    level: int
    shape: dict[str, int]
    our_best_metric: float
    our_best_config_id: str
    pytorch_baseline_metric: float = 0.0
    speedup: float = 0.0
    correct: bool = False


@dataclass(slots=True)
class KernelBenchReport:
    results: list[ProblemResult] = field(default_factory=list)
    fast_p_scores: dict[float, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_fast_p(self) -> None:
        """Compute fast_p scores at each threshold, grouped by level."""
        by_level: dict[int, list[ProblemResult]] = {}
        for result in self.results:
            by_level.setdefault(result.level, []).append(result)

        scores: dict[float, dict[str, float]] = {}
        for p in FAST_P_THRESHOLDS:
            scores[p] = {}
            for level, problems in by_level.items():
                if not problems:
                    continue
                passing = sum(
                    1 for r in problems if r.correct and r.speedup >= p
                )
                scores[p][f"level_{level}"] = round(passing / len(problems), 3)
            # Overall
            all_problems = [r for r in self.results]
            if all_problems:
                passing = sum(
                    1 for r in all_problems if r.correct and r.speedup >= p
                )
                scores[p]["overall"] = round(passing / len(all_problems), 3)
        self.fast_p_scores = scores

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "fast_p_scores": self.fast_p_scores,
            "results": [
                {
                    "problem_id": r.problem_id,
                    "operator": r.operator,
                    "level": r.level,
                    "shape": r.shape,
                    "our_metric": r.our_best_metric,
                    "our_config_id": r.our_best_config_id,
                    "pytorch_baseline": r.pytorch_baseline_metric,
                    "speedup": r.speedup,
                    "correct": r.correct,
                }
                for r in self.results
            ],
        }

    def summary_text(self) -> str:
        """Render a human-readable summary."""
        lines = [
            "# KernelBench-style Evaluation",
            "",
            f"Hardware: {self.metadata.get('hardware', 'unknown')}",
            f"Problems evaluated: {len(self.results)}",
            "",
            "## fast_p Scores",
            "",
            "| Threshold | Overall | Level 1 | Level 2 |",
            "|-----------|---------|---------|---------|",
        ]
        for p in FAST_P_THRESHOLDS:
            row = self.fast_p_scores.get(p, {})
            lines.append(
                f"| fast_{p} | "
                f"{row.get('overall', 0):.1%} | "
                f"{row.get('level_1', 0):.1%} | "
                f"{row.get('level_2', 0):.1%} |"
            )
        lines.extend([
            "",
            "## Per-Problem Results",
            "",
            "| Problem | Operator | Level | Our Metric | PyTorch | Speedup | Config |",
            "|---------|----------|-------|-----------|---------|---------|--------|",
        ])
        for r in self.results:
            speedup = f"{r.speedup:.2f}x" if r.correct else "FAIL"
            lines.append(
                f"| {r.problem_id} | {r.operator} | {r.level} | "
                f"{r.our_best_metric:.1f} | {r.pytorch_baseline_metric:.1f} | "
                f"{speedup} | `{r.our_best_config_id}` |"
            )
        return "\n".join(lines) + "\n"


def build_benchmark_script_with_baseline(
    operator: str,
    problems: list[dict],
    configs: list[dict],
) -> str:
    """Generate a script that benchmarks our configs AND the PyTorch baseline.

    Returns JSON with both our results and the PyTorch baseline metric per problem.
    """
    from .triton_operators import REGISTRY
    spec = REGISTRY.get(operator)

    # Import operator-specific benchmark generation
    if operator == "matmul":
        from .modal_runner import generate_batched_benchmark_script as gen_fn
    elif operator == "rmsnorm":
        from .triton_rmsnorm import generate_rmsnorm_benchmark_script as gen_fn
    elif operator == "softmax":
        from .triton_softmax import generate_softmax_benchmark_script as gen_fn
    elif operator == "layernorm":
        from .triton_layernorm import generate_layernorm_benchmark_script as gen_fn
    elif operator == "cross_entropy":
        from .triton_cross_entropy import generate_cross_entropy_benchmark_script as gen_fn
    else:
        raise ValueError(f"Unknown operator: {operator}")

    # Reshape problems to match the script's expected shape format
    shapes = []
    for p in problems:
        shape = {k: v for k, v in p.items() if k not in ("id", "level")}
        shape["name"] = p["id"]
        shapes.append(shape)

    # Generate the normal benchmark script
    base_script = gen_fn(configs, shapes)

    # Inject a PyTorch baseline measurement step at the end
    baseline_injection = _pytorch_baseline_snippet(operator)
    # Find the main() output and patch it to include baselines
    patched = base_script.replace(
        "print(json.dumps(output, indent=2))",
        baseline_injection + "\n    print(json.dumps(output, indent=2))"
    )
    return patched


def _pytorch_baseline_snippet(operator: str) -> str:
    """Generate inline code that measures PyTorch baselines for all shapes."""
    if operator == "matmul":
        return '''
    # Measure PyTorch/cuBLAS baselines
    pytorch_baselines = []
    for shape in shapes:
        M, N, K = shape["M"], shape["N"], shape["K"]
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b), warmup=25, rep=100)
        flops = 2.0 * M * N * K
        tflops = flops / (ms * 1e-3) / 1e12
        pytorch_baselines.append({"shape_name": shape.get("name", ""), "ms": round(ms, 4), "tflops": round(tflops, 2)})
    output["pytorch_baselines"] = pytorch_baselines
'''
    elif operator == "rmsnorm":
        return '''
    # Measure PyTorch RMSNorm baseline
    pytorch_baselines = []
    for shape in shapes:
        n_rows, hidden_dim = shape["n_rows"], shape["hidden_dim"]
        x = torch.randn((n_rows, hidden_dim), device="cuda", dtype=torch.float16)
        w = torch.randn((hidden_dim,), device="cuda", dtype=torch.float16)
        def ref():
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            return (x * torch.rsqrt(variance + 1e-6) * w).to(torch.float16)
        ms = triton.testing.do_bench(ref, warmup=25, rep=100)
        bytes_moved = 2 * n_rows * hidden_dim * 2 + hidden_dim * 2
        gb_per_s = bytes_moved / (ms * 1e-3) / 1e9
        pytorch_baselines.append({"shape_name": shape.get("name", ""), "ms": round(ms, 4), "gb_per_s": round(gb_per_s, 2), "tflops": round(gb_per_s, 2)})
    output["pytorch_baselines"] = pytorch_baselines
'''
    elif operator == "softmax":
        return '''
    # Measure PyTorch softmax baseline
    pytorch_baselines = []
    for shape in shapes:
        n_rows, n_cols = shape["n_rows"], shape["n_cols"]
        x = torch.randn((n_rows, n_cols), device="cuda", dtype=torch.float16)
        ms = triton.testing.do_bench(lambda: torch.softmax(x.to(torch.float32), dim=-1).to(torch.float16), warmup=25, rep=100)
        bytes_moved = 2 * n_rows * n_cols * 2
        gb_per_s = bytes_moved / (ms * 1e-3) / 1e9
        pytorch_baselines.append({"shape_name": shape.get("name", ""), "ms": round(ms, 4), "gb_per_s": round(gb_per_s, 2), "tflops": round(gb_per_s, 2)})
    output["pytorch_baselines"] = pytorch_baselines
'''
    elif operator == "layernorm":
        return '''
    # Measure PyTorch LayerNorm baseline
    pytorch_baselines = []
    for shape in shapes:
        n_rows, hidden_dim = shape["n_rows"], shape["hidden_dim"]
        x = torch.randn((n_rows, hidden_dim), device="cuda", dtype=torch.float16)
        w = torch.randn((hidden_dim,), device="cuda", dtype=torch.float16)
        b = torch.randn((hidden_dim,), device="cuda", dtype=torch.float16)
        ms = triton.testing.do_bench(lambda: torch.nn.functional.layer_norm(x, (hidden_dim,), w, b, eps=1e-5), warmup=25, rep=100)
        bytes_moved = 2 * n_rows * hidden_dim * 2 + hidden_dim * 4
        gb_per_s = bytes_moved / (ms * 1e-3) / 1e9
        pytorch_baselines.append({"shape_name": shape.get("name", ""), "ms": round(ms, 4), "gb_per_s": round(gb_per_s, 2), "tflops": round(gb_per_s, 2)})
    output["pytorch_baselines"] = pytorch_baselines
'''
    elif operator == "cross_entropy":
        return '''
    # Measure PyTorch cross_entropy baseline
    pytorch_baselines = []
    for shape in shapes:
        n_rows, n_cols = shape["n_rows"], shape["n_cols"]
        logits = torch.randn((n_rows, n_cols), device="cuda", dtype=torch.float16)
        target = torch.randint(0, n_cols, (n_rows,), device="cuda", dtype=torch.long)
        ms = triton.testing.do_bench(lambda: torch.nn.functional.cross_entropy(logits.to(torch.float32), target, reduction="none"), warmup=25, rep=100)
        bytes_moved = n_rows * n_cols * 2 + n_rows * 8 + n_rows * 2
        gb_per_s = bytes_moved / (ms * 1e-3) / 1e9
        pytorch_baselines.append({"shape_name": shape.get("name", ""), "ms": round(ms, 4), "gb_per_s": round(gb_per_s, 2), "tflops": round(gb_per_s, 2)})
    output["pytorch_baselines"] = pytorch_baselines
'''
    return ""


def evaluate_kernelbench(
    *,
    operator: str | None = None,
    gpu: str = "A100",
    max_configs_per_problem: int = 8,
) -> KernelBenchReport:
    """Run KernelBench-style evaluation across one or all supported operators."""
    from .triton_operators import REGISTRY
    from .modal_runner import run_benchmark_batch_modal_generic, _extract_json_object

    operators = [operator] if operator else list(KERNELBENCH_SUBSET.keys())
    report = KernelBenchReport(metadata={"hardware": gpu, "operators": operators})

    for op_name in operators:
        if op_name not in KERNELBENCH_SUBSET:
            continue
        spec = REGISTRY.get(op_name)
        problems = KERNELBENCH_SUBSET[op_name]
        configs = spec.curated_configs[:max_configs_per_problem]

        # Build script with baseline injection
        script = build_benchmark_script_with_baseline(op_name, problems, configs)
        batch = run_benchmark_batch_modal_generic(benchmark_script=script, gpu=gpu)

        if not batch.success:
            continue

        # pytorch_baselines was stashed in batch.extra
        pytorch_baselines = (batch.extra or {}).get("pytorch_baselines", [])
        baselines = {
            b["shape_name"]: b.get("tflops", 0)
            for b in pytorch_baselines
        }

        # Per problem, find our best config
        for problem in problems:
            pid = problem["id"]
            level = problem["level"]
            shape = {k: v for k, v in problem.items() if k not in ("id", "level")}

            best_metric = 0.0
            best_config_id = ""
            correct = False
            for config_result in batch.config_results:
                cid = config_result.get("config_id", "")
                for sr in config_result.get("results", []):
                    if sr.get("shape_name") != pid:
                        continue
                    if sr.get("correct") and sr.get("tflops"):
                        if sr["tflops"] > best_metric:
                            best_metric = sr["tflops"]
                            best_config_id = cid
                            correct = True

            baseline = baselines.get(pid, 0)
            speedup = best_metric / baseline if baseline > 0 else 0

            report.results.append(ProblemResult(
                problem_id=pid,
                operator=op_name,
                level=level,
                shape=shape,
                our_best_metric=best_metric,
                our_best_config_id=best_config_id,
                pytorch_baseline_metric=baseline,
                speedup=speedup,
                correct=correct,
            ))

    report.compute_fast_p()
    return report
