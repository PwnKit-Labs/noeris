<p align="center">
  <img src="docs/brand/noeris-mark.svg" alt="Noeris mark" width="96" height="96" />
</p>

<h1 align="center">Noeris</h1>

<p align="center">
  <strong>Autonomous GPU kernel optimization via parameterized Triton templates, cross-run learning, and bandit-guided search.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11+-111827?style=flat-square&logo=python&logoColor=F7C948">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-111827?style=flat-square">
  <img alt="Operators" src="https://img.shields.io/badge/operators-21-111827?style=flat-square">
  <img alt="Hardware" src="https://img.shields.io/badge/validated-T4%20%7C%20A100%20%7C%20H100-111827?style=flat-square">
</p>

---

Noeris generates GPU kernels from compact parameter tuples, stores winning configurations in a shape-indexed database, and uses a Thompson-sampling bandit with a learned cost model to converge on near-optimal configs autonomously. One command, no manual tuning. It covers 21 parameterized Triton operators across transformers and Mamba-3 SSM, validated on 19 models from 13 architecture families. Development runs on free Kaggle T4 GPUs (30 hr/week) -- no paid compute required.

## Key results

| Result | Metric |
|---|---|
| Sliding-window attention vs cuDNN FlashAttention (A100) | **6.24x faster** (8/8 shapes win) |
| Fused QK-RMSNorm+RoPE prologue (A100) | **10.2--12.9x** vs separated launches |
| Fused QK-RMSNorm+RoPE prologue (H100) | **10.4--11.9x**, peak 1628 GB/s |
| SSM selective scan validated (Mamba-3, T4) | **1.88 GB/s** -- architecture-agnostic (transformers + SSMs) |
| PLE fusion (Per-Layer Expert, T4) | **2.07x** (72.44 GB/s) |
| Operators validated on T4 | **18/20** |
| Cross-model fusion speedup (19 models, T4) | **6.5--8.9x** (19/19 pass) |
| Cross-model fusion speedup (19 models, A100) | **3.9--9.8x** (mean 340 GB/s) |
| End-to-end 26-layer Gemma 4 E2B (A100) | **1.43x** (41.4 ms -> 29.1 ms) |
| Kernel launch reduction (torch.compile vs Noeris) | 40 -> 9 -> **1 launch** |
| Bandit convergence | **98% of optimal in 1 iteration** (6 configs vs 50 exhaustive) |
| Cross-hardware zero-shot config prediction | **rho = 0.907** (A100 from T4 data) |

## Sliding-window attention vs cuDNN

Tile-pruning skips 96--98% of tiles that cuDNN FlashAttention computes densely. The winning regime is long sequences with narrow windows -- where cuDNN's dense inner loop is wasteful.

- **A100:** 8/8 shapes win, up to 6.24x
- **T4:** 4/8 shapes win, up to 3.56x

FlexAttention (PyTorch) provides a composable block-sparse API that can express similar tile-skipping. To our knowledge, we provide the first published measurement of >3x wins on narrow-window shapes across both T4 and A100.

## Compiler comparison (T4)

| Stage | Kernel launches | Prologue time |
|---|---|---|
| PyTorch eager | 40 | 3.45 ms |
| `torch.compile` | 9 (4 Triton kernels) | 0.92 ms (3.75x) |
| **Noeris** | **1** | **0.57 ms (6.08x)** |

`torch.compile` splits at the RMSNorm reduction boundary and materializes to HBM between passes. Noeris fuses the entire prologue into a single kernel launch. At the full-layer level: Noeris 1.54x vs compiler 1.23x -- a 25% gap the compiler cannot close.

## Model support

All 19 shapes pass correctness. Fusion speedup measured on T4 (Kaggle) and A100 (Modal).

| Model family | Models | Fusion speedup (T4) | Fusion speedup (A100) |
|---|---|---|---|
| Gemma | 4 E2B, 4 31B, 4 26B-A4B | 6.5--8.3x | 5.6--9.8x |
| LLaMA | 3 8B, 3 70B, 4 Scout | 8.3--8.9x | 3.9--6.3x |
| Qwen | 3 8B, 3 32B | 7.8--8.5x | 4.8--6.1x |
| Mistral / Mixtral | 7B, 8x7B | 8.3--8.9x | 5.2--6.3x |
| Phi | 3 mini, 4 mini | 8.3--8.5x | 5.2--6.1x |
| Falcon 3 | 7B | 8.3x | 5.2x |
| DBRX | 132B | 8.9x | 6.3x |
| OLMo 2 | 7B | 8.3x | 5.2x |
| InternLM 3 | 8B | 8.3x | 5.2x |
| Mamba-3 | SSM scan | 1.88 GB/s | -- |

## Operators (21 parameterized Triton templates)

| Category | Operators |
|---|---|
| Core | matmul, rmsnorm (`1+w` Gemma affine), layernorm, softmax (+ softcap), cross_entropy |
| Attention | GQA + causal + sliding-window + QK-norm + YOCO KV-share, paged-KV decode (pure Triton) |
| Fusion | fused QK-RMSNorm+RoPE (fwd + bwd), fused GeGLU, fused norm+matmul |
| Routing | RoPE (dual-base with p-RoPE), MoE router (matmul+softmax+top-k), grouped GEMM (sort-free) |
| Embedding | PLE gather (Gemma E2B/E4B per-layer), PLE fusion (2.07x), K=V attention |
| SSM | selective scan (Mamba-3) |

110+ shape buckets. 606 unit tests. 18/20 operators validated on T4; all operators pass correctness on A100 and H100.

## Search system

- **Thompson-sampling bandit** + **gradient-boosted cost model** (R^2 = 0.94) + **MAP-Elites quality-diversity**
- **Adaptive meta-bandit router** learns per-iteration which selector to trust (matches best fixed selector within 0.5%)
- **Cross-run shape-indexed config database** keyed by `(operator, shape_bucket, hardware)` -- knowledge compounds across sessions
- **Learned feasibility** -- no hardcoded shared-memory filters; the bandit learns from runtime failures
- **Cross-hardware transfer** -- A100-trained cost model rankings transfer to H100 with rho = 0.967
- ~$0.01 per iteration on Modal

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .

# Search a specific operator (A100 via Modal)
python -m research_engine.cli triton-iterate \
    --operator rmsnorm --gpu A100 --llm --configs-per-run 8

# KernelBench-style eval
python -m research_engine.cli kernelbench-eval --gpu A100
```

Requires a Modal account (`pip install modal && modal token new`). LLM proposer needs Azure OpenAI or OpenAI credentials (optional).

### Free GPU validation (no paid compute)

```bash
# Kaggle (30 hr/week free T4) or Google Colab
!git clone https://github.com/PwnKit-Labs/noeris && cd noeris
!pip install -e . numpy scikit-learn -q
!python scripts/colab_validate_all.py
```

## Honest framing

- **vLLM** has an experimental `enable_qk_norm_rope_fusion` pass (disabled by default due to H100 regression, [issue #34391](https://github.com/vllm-project/vllm/issues/34391)). We make this fusion practical with parameterized Triton + bandit autotuning.
- **Mirage** (OSDI 2025) demonstrated fused norm+matmul. We implement it independently in Triton with autotuning.
- **FlexAttention** does block-sparse tile skipping. To our knowledge, we are the first to measure >3x wins on narrow-window shapes.
- **Liger Kernel** fuses RMSNorm and RoPE backward passes separately. To our knowledge, no existing framework fuses the combined QK-RMSNorm+RoPE backward into a single kernel.
- All novelty claims use "to our knowledge" qualification.

## Related work

| System | Cross-run | Shape-indexed | Parameterized | Operators |
|---|---|---|---|---|
| **Noeris** | **Yes** | **Yes** | **Yes** | **21** |
| AutoKernel | No | No | No | 9 |
| KernelSkill (ICLR 2026) | Skill retrieval | No | No | -- |
| CUDA-L1 (ICLR 2026) | Trained model | No | No | -- |
| KernelFoundry | Within-run | No | Template-based | -- |
| Triton autotune | Cached per shape | Per-shape | Fixed list | -- |

## Repository layout

```
src/research_engine/
  triton_operators.py        operator protocol + registry
  triton_kernels.py          matmul kernel + ConfigDatabase
  triton_{rmsnorm,softmax,layernorm,cross_entropy}.py
  triton_attention.py        FlashAttention + causal + SWA + QK-norm
  triton_rotary.py           RoPE kernel
  triton_geglu.py            fused GeGLU for Gemma
  triton_qk_norm_rope.py     fused QK-RMSNorm+RoPE (fwd + bwd)
  modal_runner.py            Modal GPU execution backend
  kernelbench.py             KernelBench-style evaluation
  cost_model.py              gradient-boosted cost model
  ablation.py                cross-run learning + selector ablation
  cli.py                     CLI entry point
tests/                       606 regression tests
docs/paper/noeris.md         paper draft
docs/results/                all benchmark JSON + reports
```

## Citing

Paper draft: [`docs/paper/noeris.md`](docs/paper/noeris.md). arXiv preprint coming.

```bibtex
@misc{noeris2026,
  title   = {One Kernel Is All You Need: Architecture-Agnostic GPU Kernel Autotuning Across Transformers and State-Space Models},
  author  = {Doruk Tan Ozturk},
  year    = {2026},
  url     = {https://github.com/PwnKit-Labs/noeris}
}
```

MIT License.
