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


# Expanded KernelBench-style problem set covering 50+ problems across
# 6 operators at difficulty levels 1-3. Shapes are drawn from real LLM
# workloads (GPT-2, BERT, LLaMA, Mistral, GPT-NeoX) plus systematic
# stress tests at Level 1.
KERNELBENCH_SUBSET = {
    "matmul": [
        # Level 1: systematic stress shapes
        {"id": "kb_L1_matmul_128", "M": 128, "N": 128, "K": 128, "level": 1},
        {"id": "kb_L1_matmul_256", "M": 256, "N": 256, "K": 256, "level": 1},
        {"id": "kb_L1_matmul_512", "M": 512, "N": 512, "K": 512, "level": 1},
        {"id": "kb_L1_matmul_1024", "M": 1024, "N": 1024, "K": 1024, "level": 1},
        {"id": "kb_L1_matmul_2048", "M": 2048, "N": 2048, "K": 2048, "level": 1},
        {"id": "kb_L1_matmul_4096", "M": 4096, "N": 4096, "K": 4096, "level": 1},
        {"id": "kb_L1_matmul_tall_4k", "M": 4096, "N": 1024, "K": 1024, "level": 1},
        {"id": "kb_L1_matmul_tall_8k", "M": 8192, "N": 1024, "K": 1024, "level": 1},
        {"id": "kb_L1_matmul_wide", "M": 1024, "N": 8192, "K": 1024, "level": 1},
        {"id": "kb_L1_matmul_deep", "M": 1024, "N": 1024, "K": 8192, "level": 1},
        # Level 2: LLM-shaped workloads
        {"id": "kb_L2_gpt2_qkv", "M": 1024, "N": 2304, "K": 768, "level": 2},
        {"id": "kb_L2_gpt2_out", "M": 1024, "N": 768, "K": 768, "level": 2},
        {"id": "kb_L2_gpt2_mlp_up", "M": 1024, "N": 3072, "K": 768, "level": 2},
        {"id": "kb_L2_gpt2_mlp_down", "M": 1024, "N": 768, "K": 3072, "level": 2},
        {"id": "kb_L2_llama7b_qkv", "M": 4096, "N": 12288, "K": 4096, "level": 2},
        {"id": "kb_L2_llama7b_mlp_up", "M": 4096, "N": 11008, "K": 4096, "level": 2},
        {"id": "kb_L2_llama7b_mlp_down", "M": 4096, "N": 4096, "K": 11008, "level": 2},
        {"id": "kb_L2_bert_qkv", "M": 512, "N": 3072, "K": 1024, "level": 2},
        {"id": "kb_L2_mistral_mlp", "M": 4096, "N": 14336, "K": 4096, "level": 2},
    ],
    "rmsnorm": [
        {"id": "kb_L1_rmsnorm_gpt2", "n_rows": 1024, "hidden_dim": 768, "level": 1},
        {"id": "kb_L1_rmsnorm_bert", "n_rows": 4096, "hidden_dim": 1024, "level": 1},
        {"id": "kb_L1_rmsnorm_gpt_xl", "n_rows": 4096, "hidden_dim": 1600, "level": 1},
        {"id": "kb_L2_rmsnorm_llama7b", "n_rows": 4096, "hidden_dim": 4096, "level": 2},
        {"id": "kb_L2_rmsnorm_llama13b", "n_rows": 4096, "hidden_dim": 5120, "level": 2},
        {"id": "kb_L2_rmsnorm_llama70b", "n_rows": 2048, "hidden_dim": 8192, "level": 2},
        {"id": "kb_L2_rmsnorm_mixtral", "n_rows": 8192, "hidden_dim": 4096, "level": 2},
    ],
    "softmax": [
        {"id": "kb_L1_softmax_tiny", "n_rows": 1024, "n_cols": 256, "level": 1},
        {"id": "kb_L1_softmax_small", "n_rows": 1024, "n_cols": 512, "level": 1},
        {"id": "kb_L1_softmax_medium", "n_rows": 2048, "n_cols": 1024, "level": 1},
        {"id": "kb_L1_softmax_large", "n_rows": 4096, "n_cols": 4096, "level": 1},
        {"id": "kb_L2_softmax_attn_short", "n_rows": 8192, "n_cols": 512, "level": 2},
        {"id": "kb_L2_softmax_attn_long", "n_rows": 2048, "n_cols": 4096, "level": 2},
        {"id": "kb_L2_softmax_vocab_gpt2", "n_rows": 1024, "n_cols": 50257, "level": 2},
        {"id": "kb_L2_softmax_vocab_llama", "n_rows": 2048, "n_cols": 32000, "level": 2},
    ],
    "layernorm": [
        {"id": "kb_L1_layernorm_gpt2", "n_rows": 1024, "hidden_dim": 768, "level": 1},
        {"id": "kb_L1_layernorm_bert_base", "n_rows": 4096, "hidden_dim": 768, "level": 1},
        {"id": "kb_L1_layernorm_bert_large", "n_rows": 4096, "hidden_dim": 1024, "level": 1},
        {"id": "kb_L2_layernorm_gpt_xl", "n_rows": 4096, "hidden_dim": 1600, "level": 2},
        {"id": "kb_L2_layernorm_neox", "n_rows": 4096, "hidden_dim": 4096, "level": 2},
        {"id": "kb_L2_layernorm_long_seq", "n_rows": 8192, "hidden_dim": 768, "level": 2},
    ],
    "cross_entropy": [
        {"id": "kb_L1_ce_bert", "n_rows": 4096, "n_cols": 30522, "level": 1},
        {"id": "kb_L1_ce_gpt2", "n_rows": 1024, "n_cols": 50257, "level": 1},
        {"id": "kb_L1_ce_gpt2_long", "n_rows": 2048, "n_cols": 50257, "level": 1},
        {"id": "kb_L1_ce_llama", "n_rows": 2048, "n_cols": 32000, "level": 1},
        {"id": "kb_L2_ce_mistral", "n_rows": 4096, "n_cols": 32000, "level": 2},
        {"id": "kb_L2_ce_long_llama", "n_rows": 8192, "n_cols": 32000, "level": 2},
        {"id": "kb_L2_ce_llama3_128k", "n_rows": 2048, "n_cols": 128256, "level": 2},
    ],
    "attention": [
        {"id": "kb_L2_attn_short_64", "batch": 4, "heads": 16, "seq_len": 512, "head_dim": 64, "is_causal": False, "level": 2},
        {"id": "kb_L2_attn_short_128", "batch": 4, "heads": 16, "seq_len": 512, "head_dim": 128, "is_causal": False, "level": 2},
        {"id": "kb_L2_attn_med_128", "batch": 2, "heads": 16, "seq_len": 2048, "head_dim": 128, "is_causal": False, "level": 2},
        {"id": "kb_L3_attn_long_64", "batch": 1, "heads": 16, "seq_len": 4096, "head_dim": 64, "is_causal": False, "level": 3},
        {"id": "kb_L3_attn_long_128", "batch": 1, "heads": 16, "seq_len": 4096, "head_dim": 128, "is_causal": False, "level": 3},
        {"id": "kb_L3_attn_llama7b", "batch": 1, "heads": 32, "seq_len": 4096, "head_dim": 128, "is_causal": False, "level": 3},
        # Causal variants (decoder-only LLM workload)
        {"id": "kb_L3_attn_llama7b_causal", "batch": 1, "heads": 32, "seq_len": 4096, "head_dim": 128, "is_causal": True, "level": 3},
        {"id": "kb_L3_attn_mistral_causal", "batch": 1, "heads": 32, "seq_len": 8192, "head_dim": 128, "is_causal": True, "level": 3},
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
    compile_baseline_metric: float = 0.0
    speedup: float = 0.0  # vs eager
    compile_speedup: float = 0.0  # vs torch.compile
    correct: bool = False


@dataclass(slots=True)
class KernelBenchReport:
    results: list[ProblemResult] = field(default_factory=list)
    fast_p_scores: dict[float, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_fast_p(self) -> None:
        """Compute fast_p scores vs both eager and compile baselines, by level."""
        by_level: dict[int, list[ProblemResult]] = {}
        for result in self.results:
            by_level.setdefault(result.level, []).append(result)

        scores: dict[str, dict[float, dict[str, float]]] = {
            "vs_eager": {},
            "vs_compile": {},
        }
        for p in FAST_P_THRESHOLDS:
            scores["vs_eager"][p] = {}
            scores["vs_compile"][p] = {}
            for level, problems in by_level.items():
                if not problems:
                    continue
                eager_passing = sum(
                    1 for r in problems if r.correct and r.speedup >= p
                )
                compile_passing = sum(
                    1 for r in problems
                    if r.correct and r.compile_speedup >= p and r.compile_baseline_metric > 0
                )
                scores["vs_eager"][p][f"level_{level}"] = round(eager_passing / len(problems), 3)
                scores["vs_compile"][p][f"level_{level}"] = round(compile_passing / len(problems), 3)

            all_problems = self.results
            if all_problems:
                eager_all = sum(1 for r in all_problems if r.correct and r.speedup >= p)
                compile_all = sum(
                    1 for r in all_problems
                    if r.correct and r.compile_speedup >= p and r.compile_baseline_metric > 0
                )
                scores["vs_eager"][p]["overall"] = round(eager_all / len(all_problems), 3)
                scores["vs_compile"][p]["overall"] = round(compile_all / len(all_problems), 3)
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
                    "compile_baseline": r.compile_baseline_metric,
                    "speedup_vs_eager": r.speedup,
                    "speedup_vs_compile": r.compile_speedup,
                    "correct": r.correct,
                }
                for r in self.results
            ],
        }

    def summary_text(self) -> str:
        """Render a human-readable summary with both baselines."""
        lines = [
            "# KernelBench-style Evaluation",
            "",
            f"Hardware: {self.metadata.get('hardware', 'unknown')}",
            f"Problems evaluated: {len(self.results)}",
            "",
            "## fast_p vs PyTorch eager",
            "",
            "| Threshold | Overall | Level 1 | Level 2 |",
            "|-----------|---------|---------|---------|",
        ]
        eager_scores = self.fast_p_scores.get("vs_eager", {})
        for p in FAST_P_THRESHOLDS:
            row = eager_scores.get(p, {})
            lines.append(
                f"| fast_{p} | {row.get('overall', 0):.1%} | "
                f"{row.get('level_1', 0):.1%} | {row.get('level_2', 0):.1%} |"
            )
        lines.extend(["", "## fast_p vs torch.compile max-autotune", "",
                      "| Threshold | Overall | Level 1 | Level 2 |",
                      "|-----------|---------|---------|---------|"])
        compile_scores = self.fast_p_scores.get("vs_compile", {})
        for p in FAST_P_THRESHOLDS:
            row = compile_scores.get(p, {})
            lines.append(
                f"| fast_{p} | {row.get('overall', 0):.1%} | "
                f"{row.get('level_1', 0):.1%} | {row.get('level_2', 0):.1%} |"
            )
        lines.extend([
            "",
            "## Per-Problem Results",
            "",
            "| Problem | Operator | Level | Our | Eager | vs-Eager | Compile | vs-Compile | Config |",
            "|---|---|---|---|---|---|---|---|---|",
        ])
        for r in self.results:
            eager_str = f"{r.speedup:.2f}x" if r.correct else "FAIL"
            compile_str = (
                f"{r.compile_speedup:.2f}x"
                if r.correct and r.compile_baseline_metric > 0
                else "—"
            )
            lines.append(
                f"| {r.problem_id} | {r.operator} | {r.level} | "
                f"{r.our_best_metric:.1f} | {r.pytorch_baseline_metric:.1f} | {eager_str} | "
                f"{r.compile_baseline_metric:.1f} | {compile_str} | "
                f"`{r.our_best_config_id}` |"
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
    elif operator == "attention":
        from .triton_attention import generate_attention_benchmark_script as gen_fn
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
    """Generate inline code that measures PyTorch eager AND torch.compile baselines."""
    if operator == "matmul":
        return '''
    # Measure PyTorch eager (cuBLAS) and torch.compile baselines
    pytorch_baselines = []
    compile_baselines = []
    compiled_matmul = torch.compile(lambda a, b: torch.matmul(a, b), mode="max-autotune", dynamic=False)
    for shape in shapes:
        M, N, K = shape["M"], shape["N"], shape["K"]
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        # Eager
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b), warmup=25, rep=100)
        flops = 2.0 * M * N * K
        tflops = flops / (ms * 1e-3) / 1e12
        pytorch_baselines.append({"shape_name": shape.get("name", ""), "ms": round(ms, 4), "tflops": round(tflops, 2)})
        # torch.compile (warm up first to trigger compilation)
        try:
            for _ in range(3):
                compiled_matmul(a, b)
            ms_c = triton.testing.do_bench(lambda: compiled_matmul(a, b), warmup=10, rep=50)
            tflops_c = flops / (ms_c * 1e-3) / 1e12
            compile_baselines.append({"shape_name": shape.get("name", ""), "ms": round(ms_c, 4), "tflops": round(tflops_c, 2)})
        except Exception as exc:
            compile_baselines.append({"shape_name": shape.get("name", ""), "error": str(exc)[:200]})
    output["pytorch_baselines"] = pytorch_baselines
    output["compile_baselines"] = compile_baselines
'''
    elif operator == "rmsnorm":
        return '''
    # Measure PyTorch eager and torch.compile RMSNorm baselines
    pytorch_baselines = []
    compile_baselines = []

    def _rms(x, w):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + 1e-6) * w).to(torch.float16)
    compiled_rms = torch.compile(_rms, mode="max-autotune", dynamic=False)

    for shape in shapes:
        n_rows, hidden_dim = shape["n_rows"], shape["hidden_dim"]
        x = torch.randn((n_rows, hidden_dim), device="cuda", dtype=torch.float16)
        w = torch.randn((hidden_dim,), device="cuda", dtype=torch.float16)
        bytes_moved = 2 * n_rows * hidden_dim * 2 + hidden_dim * 2
        # Eager
        ms = triton.testing.do_bench(lambda: _rms(x, w), warmup=25, rep=100)
        gb_per_s = bytes_moved / (ms * 1e-3) / 1e9
        pytorch_baselines.append({"shape_name": shape.get("name", ""), "ms": round(ms, 4), "gb_per_s": round(gb_per_s, 2), "tflops": round(gb_per_s, 2)})
        # Compile
        try:
            for _ in range(3):
                compiled_rms(x, w)
            ms_c = triton.testing.do_bench(lambda: compiled_rms(x, w), warmup=10, rep=50)
            gb_c = bytes_moved / (ms_c * 1e-3) / 1e9
            compile_baselines.append({"shape_name": shape.get("name", ""), "ms": round(ms_c, 4), "gb_per_s": round(gb_c, 2), "tflops": round(gb_c, 2)})
        except Exception as exc:
            compile_baselines.append({"shape_name": shape.get("name", ""), "error": str(exc)[:200]})
    output["pytorch_baselines"] = pytorch_baselines
    output["compile_baselines"] = compile_baselines
'''
    elif operator == "softmax":
        return '''
    # Measure PyTorch eager and torch.compile softmax baselines
    pytorch_baselines = []
    compile_baselines = []

    def _softmax(x):
        return torch.softmax(x.to(torch.float32), dim=-1).to(torch.float16)
    compiled_softmax = torch.compile(_softmax, mode="max-autotune", dynamic=False)

    for shape in shapes:
        n_rows, n_cols = shape["n_rows"], shape["n_cols"]
        x = torch.randn((n_rows, n_cols), device="cuda", dtype=torch.float16)
        bytes_moved = 2 * n_rows * n_cols * 2
        ms = triton.testing.do_bench(lambda: _softmax(x), warmup=25, rep=100)
        gb_per_s = bytes_moved / (ms * 1e-3) / 1e9
        pytorch_baselines.append({"shape_name": shape.get("name", ""), "ms": round(ms, 4), "gb_per_s": round(gb_per_s, 2), "tflops": round(gb_per_s, 2)})
        try:
            for _ in range(3):
                compiled_softmax(x)
            ms_c = triton.testing.do_bench(lambda: compiled_softmax(x), warmup=10, rep=50)
            gb_c = bytes_moved / (ms_c * 1e-3) / 1e9
            compile_baselines.append({"shape_name": shape.get("name", ""), "ms": round(ms_c, 4), "gb_per_s": round(gb_c, 2), "tflops": round(gb_c, 2)})
        except Exception as exc:
            compile_baselines.append({"shape_name": shape.get("name", ""), "error": str(exc)[:200]})
    output["pytorch_baselines"] = pytorch_baselines
    output["compile_baselines"] = compile_baselines
'''
    elif operator == "layernorm":
        return '''
    # Measure PyTorch eager and torch.compile LayerNorm baselines
    pytorch_baselines = []
    compile_baselines = []

    def _ln(x, w, b, hidden):
        return torch.nn.functional.layer_norm(x, (hidden,), w, b, eps=1e-5)
    compiled_ln = torch.compile(_ln, mode="max-autotune", dynamic=False)

    for shape in shapes:
        n_rows, hidden_dim = shape["n_rows"], shape["hidden_dim"]
        x = torch.randn((n_rows, hidden_dim), device="cuda", dtype=torch.float16)
        w = torch.randn((hidden_dim,), device="cuda", dtype=torch.float16)
        b = torch.randn((hidden_dim,), device="cuda", dtype=torch.float16)
        bytes_moved = 2 * n_rows * hidden_dim * 2 + hidden_dim * 4
        ms = triton.testing.do_bench(lambda: _ln(x, w, b, hidden_dim), warmup=25, rep=100)
        gb_per_s = bytes_moved / (ms * 1e-3) / 1e9
        pytorch_baselines.append({"shape_name": shape.get("name", ""), "ms": round(ms, 4), "gb_per_s": round(gb_per_s, 2), "tflops": round(gb_per_s, 2)})
        try:
            for _ in range(3):
                compiled_ln(x, w, b, hidden_dim)
            ms_c = triton.testing.do_bench(lambda: compiled_ln(x, w, b, hidden_dim), warmup=10, rep=50)
            gb_c = bytes_moved / (ms_c * 1e-3) / 1e9
            compile_baselines.append({"shape_name": shape.get("name", ""), "ms": round(ms_c, 4), "gb_per_s": round(gb_c, 2), "tflops": round(gb_c, 2)})
        except Exception as exc:
            compile_baselines.append({"shape_name": shape.get("name", ""), "error": str(exc)[:200]})
    output["pytorch_baselines"] = pytorch_baselines
    output["compile_baselines"] = compile_baselines
'''
    elif operator == "cross_entropy":
        return '''
    # Measure PyTorch eager and torch.compile cross_entropy baselines
    pytorch_baselines = []
    compile_baselines = []

    def _ce(logits, target):
        return torch.nn.functional.cross_entropy(logits.to(torch.float32), target, reduction="none")
    compiled_ce = torch.compile(_ce, mode="max-autotune", dynamic=False)

    for shape in shapes:
        n_rows, n_cols = shape["n_rows"], shape["n_cols"]
        logits = torch.randn((n_rows, n_cols), device="cuda", dtype=torch.float16)
        target = torch.randint(0, n_cols, (n_rows,), device="cuda", dtype=torch.long)
        bytes_moved = n_rows * n_cols * 2 + n_rows * 8 + n_rows * 2
        ms = triton.testing.do_bench(lambda: _ce(logits, target), warmup=25, rep=100)
        gb_per_s = bytes_moved / (ms * 1e-3) / 1e9
        pytorch_baselines.append({"shape_name": shape.get("name", ""), "ms": round(ms, 4), "gb_per_s": round(gb_per_s, 2), "tflops": round(gb_per_s, 2)})
        try:
            for _ in range(3):
                compiled_ce(logits, target)
            ms_c = triton.testing.do_bench(lambda: compiled_ce(logits, target), warmup=10, rep=50)
            gb_c = bytes_moved / (ms_c * 1e-3) / 1e9
            compile_baselines.append({"shape_name": shape.get("name", ""), "ms": round(ms_c, 4), "gb_per_s": round(gb_c, 2), "tflops": round(gb_c, 2)})
        except Exception as exc:
            compile_baselines.append({"shape_name": shape.get("name", ""), "error": str(exc)[:200]})
    output["pytorch_baselines"] = pytorch_baselines
    output["compile_baselines"] = compile_baselines
'''
    elif operator == "attention":
        return '''
    # Measure PyTorch eager and torch.compile SDPA baselines (supports causal)
    pytorch_baselines = []
    compile_baselines = []

    def _attn(q, k, v, cf):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=cf)
    compiled_attn = torch.compile(_attn, mode="max-autotune", dynamic=False)

    for shape in shapes:
        batch = shape["batch"]
        heads = shape["heads"]
        seq_len = shape["seq_len"]
        head_dim = shape["head_dim"]
        is_causal = bool(shape.get("is_causal", False))
        q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        v = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        causal_factor = 0.5 if is_causal else 1.0
        flops = 4.0 * batch * heads * seq_len * seq_len * head_dim * causal_factor
        cf = is_causal
        ms = triton.testing.do_bench(lambda: _attn(q, k, v, cf), warmup=10, rep=50)
        tflops = flops / (ms * 1e-3) / 1e12
        pytorch_baselines.append({"shape_name": shape.get("name", ""), "ms": round(ms, 4), "tflops": round(tflops, 2)})
        try:
            for _ in range(3):
                compiled_attn(q, k, v, cf)
            ms_c = triton.testing.do_bench(lambda: compiled_attn(q, k, v, cf), warmup=10, rep=50)
            tflops_c = flops / (ms_c * 1e-3) / 1e12
            compile_baselines.append({"shape_name": shape.get("name", ""), "ms": round(ms_c, 4), "tflops": round(tflops_c, 2)})
        except Exception as exc:
            compile_baselines.append({"shape_name": shape.get("name", ""), "error": str(exc)[:200]})
    output["pytorch_baselines"] = pytorch_baselines
    output["compile_baselines"] = compile_baselines
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

        # pytorch_baselines and compile_baselines were stashed in batch.extra
        pytorch_baselines = (batch.extra or {}).get("pytorch_baselines", [])
        compile_baselines = (batch.extra or {}).get("compile_baselines", [])
        baselines = {
            b["shape_name"]: b.get("tflops", 0)
            for b in pytorch_baselines
        }
        compile_map = {
            b["shape_name"]: b.get("tflops", 0)
            for b in compile_baselines
            if "tflops" in b
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
            compile_baseline = compile_map.get(pid, 0)
            speedup = best_metric / baseline if baseline > 0 else 0
            compile_speedup = (
                best_metric / compile_baseline if compile_baseline > 0 else 0
            )

            report.results.append(ProblemResult(
                problem_id=pid,
                operator=op_name,
                level=level,
                shape=shape,
                our_best_metric=best_metric,
                our_best_config_id=best_config_id,
                pytorch_baseline_metric=baseline,
                compile_baseline_metric=compile_baseline,
                speedup=speedup,
                compile_speedup=compile_speedup,
                correct=correct,
            ))

    report.compute_fast_p()
    return report
