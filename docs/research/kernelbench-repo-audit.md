# KernelBench Repo Audit

_Research pass completed 2026-04-11. Repo: [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench), MIT licensed._

## Critical finding

**Noeris is not currently running KernelBench.** We have our own synthetic LLM-shape benchmark that *resembles* KernelBench's level taxonomy. The published Noeris numbers (rmsnorm 11.66×, cross-entropy 9.65×, softmax 6.38×) are **not on the same problems, dtype, or shapes** as anything on a KernelBench leaderboard. A reviewer running the upstream harness will not reproduce them.

## Repo structure

- `KernelBench/` — problem dataset, one `nn.Module` per problem
  - `level1/` — 100 single-primitive problems
  - `level2/` — 100 fusion chains (Conv+Norm+Activation)
  - `level3/` — 50 full architectures (ResNet, ViT, Mamba2, MiniGPT)
  - `level4/` — **20 HuggingFace model forward passes** (gpt2, opt-1.3b, bart-large, gpt-neo-2.7b, bigbird, electra, reformer)
- `src/kernelbench/` — eval harness: `eval.py`, `timing.py`, `score.py` (fast_p), `kernel_static_checker.py`
- `results/timing/H100_Modal/` — **pre-computed H100 Modal baselines for every problem** (`baseline_time_torch.json` + `baseline_time_torch_compile_inductor_default.json`). Same hardware Noeris uses.

Each problem exposes `Model(nn.Module)` + `get_inputs()` + `get_init_inputs()`. 270 total problems.

## Noeris overlap vs KernelBench (shape mismatch everywhere)

| Noeris op | Closest KB problem | Shape mismatch |
|---|---|---|
| matmul | L1 #1–#18 | Noeris LLM shapes vs KB 4096² fp32 |
| softmax | L1 #23 | KB (4096, 393216) fp32 vs Noeris ≤(2048, 50257) |
| rmsnorm | L1 #36 | KB **4D** (112,64,512,512) vs Noeris 2D rows×hidden |
| layernorm | L1 #40 | KB 4D (16,64,256,256) |
| cross_entropy | L1 #95 | KB (32768, 4096) fp32 vs Noeris bigger vocab |
| attention | L1 #97 | KB (32, 32, 512, 1024) non-causal fp32 vs Noeris LLaMA shapes |
| GELU/GeGLU | L1 #26, #88 | KB has plain GELU, no GeGLU |
| rotary | — | **KB has no rotary problem** |

**Realistic Noeris-addressable subset**: ~13 L1 + 20 L4 + 3 L3 = **~36 problems** with current kernel stack.

**Hard gaps in Noeris** (KB has, Noeris doesn't): reductions, scans (cumsum), pooling, conv family (38 problems!), BatchNorm/InstanceNorm/GroupNorm, loss family.

## Methodology discrepancies (Noeris vs upstream)

| | KernelBench | Noeris | Risk |
|---|---|---|---|
| Speedup threshold | strict `>` | `>=` | minor |
| Default timer | `cuda_event` 3W/10T + **L2 flush** | likely `do_bench` w/o L2 flush | **Medium** — hot-cache inflates small-tensor speedups 2–5× |
| Correctness | fp32 1e-4, fp16 1e-2 | unknown | check |
| Dtype | **fp32 default** | Noeris uses bf16/fp16 | **Major** — different benchmark |

The 11.66×/9.65×/6.38× Noeris claims may shrink under apples-to-apples timing. Better to discover internally than have a reviewer find it.

## Published results context

| System | Hardware | L1 fast_1.0 | Notes |
|---|---|---|---|
| GPT-4o greedy | L40S | low single digits | paper |
| Claude 3.5 Sonnet | L40S | ~3× on fusion outliers | paper |
| o1 | L40S | best of LLMs in paper | paper |
| Sakana AI CUDA Engineer | varies | claimed 10-100× **partially retracted** — reward-hacked kernels caught by static checker | Feb 2025 |
| METR Kevin-32B (RL-trained) | H100 | competitive on L1 | METR 2025 |

Blog post explicitly says "if you beat cudnn by more than 10%, think again" — **they treat fast_2.0+ claims with deep suspicion**.

## Integration opportunities (prioritized)

1. **Pull published H100 Modal baselines as drop-in reference** (1 hour). Instantly makes Noeris numbers comparable.
2. **Run on actual upstream problems via HF dataset** (~1 day). Only way to legitimately claim a leaderboard number.
3. **Adopt their `kernel_static_checker.py`** (low effort). Protects against reward-hacking.
4. **Switch to `cuda_event` timer with L2 flush, 3W/10T** (1 hour).
5. **Add Level 4 HF model problems** (~1 week). **Unclaimed white space** — nobody has strong L4 numbers.
6. **Change Noeris compute_fast_p `>=` → `>`** (trivial).

## Key files to modify

- `src/research_engine/kernelbench.py` — `KERNELBENCH_SUBSET`, `compute_fast_p`, baseline-loading path
- `src/research_engine/kernelbench_hf.py` — verify it actually executes upstream `Model.forward`
- Noeris timing module (wherever `speedup`/`compile_speedup` is measured) — switch to cuda_event + L2 flush

## Reference URLs

- [Repo](https://github.com/ScalingIntelligence/KernelBench) · [Paper arXiv:2502.10517](https://arxiv.org/abs/2502.10517) · [Blog](https://scalingintelligence.stanford.edu/blogs/kernelbench/)
- [HF dataset](https://huggingface.co/datasets/ScalingIntelligence/KernelBench)
- [H100 Modal eager baseline](https://github.com/ScalingIntelligence/KernelBench/blob/main/results/timing/H100_Modal/baseline_time_torch.json)
- [H100 Modal torch.compile baseline](https://github.com/ScalingIntelligence/KernelBench/blob/main/results/timing/H100_Modal/baseline_time_torch_compile_inductor_default.json)
- [eval.py](https://github.com/ScalingIntelligence/KernelBench/blob/main/src/kernelbench/eval.py)
- [score.py (fast_p)](https://github.com/ScalingIntelligence/KernelBench/blob/main/src/kernelbench/score.py)
- [kernel_static_checker.py](https://github.com/ScalingIntelligence/KernelBench/blob/main/src/kernelbench/kernel_static_checker.py)

## Bottom line (4 items to unblock leaderboard claim)

1. Load upstream H100 Modal baselines (1 hour)
2. Run Noeris kernels against actual L1 problems via HF dataset Models (~1 day)
3. Switch timer to cuda_event + L2 flush (1 hour)
4. Take a swing at L4 white space (~1 week)

Items 1+3+6 are essentially free and should land before the next paper revision.
