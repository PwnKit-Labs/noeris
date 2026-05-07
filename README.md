<p align="center">
  <img src="docs/brand/noeris-mark.svg" alt="Noeris mark" width="96" height="96" />
</p>

<h1 align="center">Noeris</h1>

<p align="center">
  <em>the fusions your compiler can't find</em>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11+-111827?style=flat-square&logo=python&logoColor=F7C948">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-111827?style=flat-square">
  <img alt="Operators" src="https://img.shields.io/badge/operators-21-111827?style=flat-square">
  <img alt="Hardware" src="https://img.shields.io/badge/validated-T4%20%7C%20A100%20%7C%20H100-111827?style=flat-square">
</p>

---

Noeris discovers and optimizes **cross-operation kernel fusions** that `torch.compile` and single-op libraries like Liger Kernel can't find.
The research engine contains the full operator/search stack; the current public
`noeris.patch()` API is a conservative drop-in surface for RMSNorm and gated MLP
activation patches.

## What makes Noeris different

| What | Liger Kernel | Noeris |
|---|---|---|
| Fusion level | Single-op (RMSNorm alone, RoPE alone) | **Cross-op** (RMSNorm+RoPE in 1 kernel) |
| Config tuning | Fixed | **Bandit autotuned** per shape per GPU |
| Attention | No | **Beats cuDNN 6.24x** on sliding-window |
| Architecture | Transformers only | **Transformers + SSMs** |
| Cross-hardware | No | **Zero-shot &rho;=0.907** |
| Backward pass | Individual op backward | **Fused cross-op backward** (QK-norm+RoPE) |

## Key results

| Result | Metric |
|---|---|
| Sliding-window attention vs cuDNN FlashAttention (A100) | **6.24x faster** (8/8 shapes) |
| Cross-op fusion: kernel launch reduction | 40 launches &rarr; **1** (QK-RMSNorm+RoPE) |
| End-to-end 26-layer Gemma 4 (A100) | **1.43x** (41.4 ms &rarr; 29.1 ms) |
| Gemma 4 decoder layer deeper fusion (A100) | **1.13x** (`31b_local`), **1.07x** (`31b_global`), **1.90x** (`e2b_local`) |
| Gemma 4 decoder layer deeper fusion (H100) | **1.26x** (`31b_local`), **1.17x** (`31b_global`), **2.32x** (`e2b_local`) |
| Bandit convergence | **98% of optimal in 1 iteration** (50x faster than grid search) |
| Model coverage | **19 models / 13 families** |
| Cross-hardware zero-shot config prediction | **&rho;=0.907** (A100 from free T4 data) |
| Fused QK-RMSNorm+RoPE prologue (A100) | **10.2--12.9x** vs separated launches |
| Fused QK-RMSNorm+RoPE prologue (H100) | **10.4--11.9x**, peak 1628 GB/s |

Latest layer artifacts: `docs/results/gemma4-layer-bench-deeper-fusion-a100-after-geglu-retune.json`, `docs/results/gemma4-layer-bench-deeper-fusion-h100-after-geglu-retune.json`.
Canonical results index: `docs/results/README.md`.

## Quick start

```python
import noeris
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-4-2b")
noeris.patch(model)  # Drop-in RMSNorm + gated MLP activation patches.
# QK-RMSNorm+RoPE kernels are available for custom integrations, but generic
# HuggingFace attention patching is not wired into noeris.patch() yet.
```

### Current public patch coverage

`noeris.patch()` currently wires two module-level optimizations into supported
HuggingFace-style models:

- RMSNorm module replacement, including Gemma's `(1+w)` affine mode.
- Gated MLP activation fusion for GeGLU/SwiGLU-style `gate_proj`, `up_proj`,
  `down_proj` blocks.

The project also includes lower-level Triton kernels for QK-RMSNorm+RoPE,
cross-entropy, attention, and other operators. Those kernels back the benchmark
artifacts below, but QK-RMSNorm+RoPE and cross-entropy are not generic
`noeris.patch()` hooks yet.

### CLI / search

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .

# Search a specific operator (A100 via Modal)
python -m research_engine.cli triton-iterate \
    --operator rmsnorm --gpu A100 --llm --configs-per-run 8
```

### Local CI parity runner

Run the same core checks used in GitHub CI from repo root:

```bash
./scripts/ci_local.sh
```

If your default `python3` does not have test dependencies installed, select one:

```bash
PYTHON_BIN=python3.11 ./scripts/ci_local.sh
```

This runs unit tests, public artifact reference checks, two `matmul-speedup`
benchmark runs, history export, and the history regression gate with
`--fail-on-missing`.

### Free GPU validation (no paid compute)

```bash
# Kaggle (30 hr/week free T4) or Google Colab
!git clone https://github.com/PwnKit-Labs/noeris && cd noeris
!pip install -e . numpy scikit-learn -q
!python scripts/colab_validate_all.py
```

## Works alongside Liger Kernel

Noeris is complementary to Liger Kernel. Liger optimizes individual operations (RMSNorm, RoPE, SwiGLU separately). Noeris fuses operations **across boundaries** that single-op libraries can't cross -- like combining RMSNorm + RoPE into one kernel pass. Use both for maximum performance.

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

110+ shape buckets. 606 unit tests. 18/20 operators validated on T4; all pass correctness on A100 and H100.

## Search system

- **Thompson-sampling bandit** + **gradient-boosted cost model** (R^2 = 0.94) + **MAP-Elites quality-diversity**
- **Cross-run shape-indexed config database** -- knowledge compounds across sessions
- **Cross-hardware transfer** -- A100-trained cost model rankings transfer to H100 with &rho;=0.967
- ~$0.01 per iteration on Modal

## Compiler comparison (T4)

| Stage | Kernel launches | Prologue time |
|---|---|---|
| PyTorch eager | 40 | 3.45 ms |
| `torch.compile` | 9 (4 Triton kernels) | 0.92 ms (3.75x) |
| **Noeris** | **1** | **0.57 ms (6.08x)** |

`torch.compile` splits at the RMSNorm reduction boundary and materializes to HBM between passes. Noeris fuses the entire prologue into a single kernel launch.

## Honest framing

- **vLLM** has an experimental `enable_qk_norm_rope_fusion` pass (disabled by default due to H100 regression, [issue #34391](https://github.com/vllm-project/vllm/issues/34391)). We make this fusion practical with parameterized Triton + bandit autotuning.
- **Mirage** (OSDI 2025) demonstrated fused norm+matmul. We implement it independently in Triton with autotuning.
- **FlexAttention** does block-sparse tile skipping. To our knowledge, we are the first to measure >3x wins on narrow-window shapes.
- **Liger Kernel** fuses RMSNorm and RoPE backward passes separately. To our knowledge, no existing framework fuses the combined QK-RMSNorm+RoPE backward into a single kernel.
- All novelty claims use "to our knowledge" qualification.

## Citing

Paper draft: [`docs/paper/noeris.md`](docs/paper/noeris.md). arXiv preprint coming.

```bibtex
@misc{noeris2026,
  title   = {Noeris: Architecture-Agnostic Kernel Fusion and Autotuning},
  author  = {Doruk Tan Ozturk},
  year    = {2026},
  url     = {https://github.com/PwnKit-Labs/noeris}
}
```

MIT License.
