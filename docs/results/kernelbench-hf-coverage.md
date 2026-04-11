# KernelBench HuggingFace Coverage Probe

**Dataset:** `ScalingIntelligence/KernelBench` (270 problems, Levels 1-4)

Pulled via `research_engine.kernelbench_hf.fetch_and_report_coverage()` with regex
operator matching and best-effort shape extraction.

## Coverage summary

| Level | Total problems |
|---|---|
| Level 1 | 100 |
| Level 2 | 100 |
| Level 3 | 50 |
| Level 4 | 20 |
| **Total** | **270** |

### Operator match (regex pattern in problem source code)

| Operator | Matched problems |
|---|---|
| matmul | 15 |
| softmax | 11 |
| attention | 4 |
| rmsnorm | 1 |
| cross_entropy | 1 |
| **Total** | **32 (11.85%)** |

### Problems with extractable shapes (benchmark-ready)

| Operator | Extractable shapes |
|---|---|
| softmax | 2 |
| matmul | 0 |
| attention | 0 |
| rmsnorm | 0 |
| cross_entropy | 0 |
| **Total** | **2** |

## Analysis

**The operator-match rate is 12%** — we have 7 Triton kernels and regex
patterns that catch the most common PyTorch invocations. Most KernelBench
problems use operators we don't yet cover: activation functions (ReLU,
GELU, Swish, Tanh, LeakyReLU, Sigmoid), LogSoftmax, Hinge loss, 4D matmul,
diagonal matmul, and many more. 88% of KernelBench is outside our
current kernel library.

**The shape extraction rate is 0.7%** — nearly all KernelBench problems
define inputs via `get_inputs()` functions with programmatic shape
construction (list comprehensions, config dicts, derived dims). Our regex
pattern `torch.randn((a, b))` only catches literal shapes, which miss
almost every real problem.

## What this means for evaluation

**Our 53-problem curated subset is more useful for benchmarking than a
thin slice of the HF dataset.** The curated problems are real LLM
workload shapes (GPT-2, BERT, LLaMA, Mistral) with known parameters,
designed to exercise each operator across realistic shape buckets. The
HuggingFace dataset's 2-problem overlap is not a meaningful comparison.

## To improve coverage

Two approaches:

1. **Grow the kernel library.** Add RELU, GELU, Swish, Tanh as Triton
   operators (all ~50-line memory-bound elementwise kernels). Each
   would unblock 5-10 KernelBench problems.

2. **Actually execute the problem code.** Instead of regex extraction,
   import each `KernelBenchProblem.code` as a Python module, call
   `get_inputs()`, inspect the returned tensor shapes. This is more
   reliable but requires sandboxing since we're executing adversarial
   code.

**Priority:** the kernel library approach. Adding activation kernels is
~200 lines each and lets us benchmark ~40-60 KernelBench problems
directly. Executing problem code is higher risk.

## Reproduction

```bash
pip install datasets
python3 -m research_engine.cli kernelbench-hf-coverage --levels 1 2 3 4
```

Takes ~5 seconds, loads from HuggingFace cache on subsequent runs.
