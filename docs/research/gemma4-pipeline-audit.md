# Gemma 4 Pipeline Audit

_Research pass completed 2026-04-11. Source: agent research + authoritative upstream docs._

## 1. Authoritative architectural facts

**Family** (released 2026-04-03, Apache 2.0). Sources: [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/), [HF blog](https://huggingface.co/blog/gemma4).

| Variant | Total params | Active | Layers | Hidden | FFN | Heads (Q/KV local) | Heads (Q/KV global) | head_dim local / global | Window | Local:Global | Vocab | Ctx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E2B | 5.1B | 2.3B | 35 | 1536 | 6144 | 8 / 1 | 8 / 1 | 256 / 256 | 512 | 4:1 | 262144 | 128k |
| E4B | 8B | 4.5B | 42 | 2560 | 10240 | 8 / 2 | 8 / 2 | 256 / 256 | 512 | — | 262144 | 128k |
| 26B-A4B (MoE) | 26B | 4B | 30 | 2816 | 2112 (per expert) | 16 / 8 | 16 / 2 | 256 / 512 | 1024 | 5:1 | 262144 | 256k |
| 31B Dense | 30.7B | 30.7B | 60 | 5376 | 21504 | 32 / 16 | 32 / 4 | 256 / 512 | 1024 | 5:1 | 262144 | 256k |

_E4B / E2B / 26B-A4B / 31B rows all pulled verbatim from each HF `config.json` on 2026-04-11 (E4B: hidden=2560, ffn=10240, 42 layers, 8Q/2KV; previously listed as "—" pending verification)._

### Cross-cutting facts
- **GQA always-on**, global layers use more aggressive ratio (32:4 = 8 Q per 1 KV in 31B global).
- **Asymmetric head_dim**: local=256, **global=512** — *new in Gemma 4* (Gemma 3 used 256 throughout). [HF transformers #45201](https://github.com/huggingface/transformers/issues/45201).
- **RoPE**: local θ=10,000; global θ=1,000,000 with p-RoPE scaling (from Gemma 3).
- **QK-norm on all layers** (inherited from Gemma 3). 31B keeps a final logits softcap.
- **MLP**: GeGLU with tanh-approx GELU (`hidden_activation="gelu_pytorch_tanh"`).
- **Per-Layer Embeddings (PLE)** in small variants (E2B/E4B): second embedding table feeds a residual signal into every decoder layer. PLE dim ≈ 256.
- **Shared KV cache (YOCO-style)**: last N layers reuse KV from earlier layers. [vLLM Gemma 4 PR #38826](https://github.com/vllm-project/vllm/pull/38826).
- **MoE (26B-A4B only)**: 128 experts, top-8 + 1 shared expert; softmax-over-128 → top-k → renormalize.

## 2. What's in Noeris today

| Operator | File | Gemma-4 readiness |
|---|---|---|
| matmul | `triton_kernels.py` | Generic GEMM, no MoE grouped GEMM |
| rmsnorm | `triton_rmsnorm.py` | Plain RMSNorm; no fused QK-norm primitive |
| softmax | `triton_softmax.py` | Generic; no router top-k softmax |
| layernorm | `triton_layernorm.py` | Generic |
| cross_entropy | `triton_cross_entropy.py` | Has 256k vocab bucket |
| attention | `triton_attention.py` | Causal + sliding-window + use_qk_norm flag; **no GQA num_kv_heads**, no head_dim=512 bucket |
| rotary | `triton_rotary.py` | Has Gemma 26B head_dim=256 bucket; no dual-base (10k vs 1M), no p-RoPE |
| geglu | `triton_geglu.py` | **Fixed 2026-04-11** — see HF config.json citations above; bucket `gemma4_26b` renamed to `gemma4_26b_a4b_expert` |

## 3. Gaps by severity

### Critical (block any honest Gemma-4 claim)
1. **No GQA support in attention.** Kernel assumes Q/K/V same head count. Every Gemma shape is wrong.
2. **No asymmetric head_dim per layer.** Local=256 vs global=512 needs two distinct attention shape buckets.
3. **No MoE operators** (router + dispatch + grouped GEMM).

### High
4. **No fused QK-RMSNorm+RoPE prologue kernel.**
5. **No KV-cache-shared (YOCO) attention path.**
6. **No paged-KV decode-time attention.** All our attention buckets are training-shaped.
7. **GeGLU constants stale** — embarrassing-if-wrong bug.

### Medium
8. **No dual-base RoPE bucket** (θ=10k local vs θ=1M global).
9. Need batch-varied 256k-vocab cross-entropy buckets.
10. Per-Layer Embedding gather op missing for E2B/E4B.
11. No softcap-applied softmax (31B final logits).

## 4. Reference open-source implementations

- [vLLM gemma4](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/gemma4/) → source `vllm/model_executor/models/gemma4.py`
- [vLLM PR #38826](https://github.com/vllm-project/vllm/pull/38826) (best single read for the architecture)
- [HF transformers Gemma4 docs](https://huggingface.co/docs/transformers/model_doc/gemma4)
- [Per-layer FA2 issue #45201](https://github.com/huggingface/transformers/issues/45201)
- [google/gemma-4-31B config](https://huggingface.co/google/gemma-4-31B)
- [google/gemma-4-26B-A4B-it config.json](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/config.json)
- [Gemma 3 Tech Report arXiv:2503.19786](https://arxiv.org/html/2503.19786v1)
- [google-deepmind/gemma](https://github.com/google-deepmind/gemma) (JAX/Flax reference)
- [Kaitchup architecture breakdown](https://kaitchup.substack.com/p/gemma-4-31b-and-26b-a4b-architecture)
- [Unsloth Gemma 3 Triton patches](https://unsloth.ai/blog/gemma3) (closest prior art)

## 5. Prioritized action list

1. **Add GQA to triton_attention** — new `NUM_KV_HEADS` tl.constexpr, ~120 LOC. Gates every honest Gemma comparison. Highest impact / lowest LOC.
2. **Add Gemma-4 global-attention bucket (head_dim=512, GQA 8:1)** — ~40 LOC. head_dim=512 is the *new* thing in Gemma 4 vs Gemma 3.
3. **Fix geglu shape buckets to real Gemma-4 numbers** — ~30 LOC. Embarrassing-if-wrong.
4. **Fused QK-RMSNorm + RoPE prologue kernel** — new operator, ~250 LOC. Canonical inference fusion (vLLM/TRT-LLM both fuse).
5. **MoE router top-k softmax + dispatch** (26B-A4B) — new operator, ~300 LOC.
6. **Grouped/segmented GEMM for MoE expert FFN** — new operator, ~500 LOC. Largest gap vs vLLM FusedMoE.
7. **Dual-base RoPE buckets** — ~60 LOC.
8. **Decode-time paged-KV attention (single-query)** — new operator, ~400 LOC.

**Out of scope for near-term**: PLE gather, softcap softmax, vision/audio ops.
