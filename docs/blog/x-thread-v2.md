# X/Twitter Thread v2 — Fused QK-RMSNorm+RoPE

_Draft 2026-04-12. 10 tweets. Tone: honest, technical, inviting scrutiny._

---

**1/**

Every LLM inference engine launches ~40 CUDA kernels for the attention prologue where 1 will do.

PyTorch eager: 40 launches
torch.compile (Inductor): 9 launches
Noeris fused Triton: 1 launch

Prologue speedup: 6.77x on T4, 10-13x on A100/H100.
Full-layer speedup: 1.54x on T4.
Peak throughput: 1,627 GB/s on H100 (49% of HBM3 peak).

Here's how.

> [SCREENSHOT: side-by-side kernel launch counts — 40 / 9 / 1. Include a simple bar chart or table.]

---

**2/**

The attention prologue — RMSNorm on Q, RMSNorm on K, RoPE on Q, RoPE on K — is tiny per-op but runs every single layer. In PyTorch eager it's 40 kernel launches because each elementwise op, each reduction, each broadcast is its own dispatch.

torch.compile sees the full graph (0 graph breaks!) and gets it down to 9. But Inductor still splits at the RMSNorm reduction boundary — it emits 4 separate Triton kernels because it won't fuse across a row-wise reduce.

We fuse across the reduction in registers. 1 kernel. Load once, norm once, rotate once, store once.

> [SCREENSHOT: torch.compile graph with 0 graph breaks but 4 output kernels annotated]

---

**3/**

Measured on A100 and H100 across all Gemma 3/4 shape buckets (60 configs, 60 correct):

A100: 10.2-12.9x prologue speedup, peak 925 GB/s
H100: 10.4-11.9x prologue speedup, peak 1,627 GB/s (49% HBM3 peak)
T4: 6.77x prologue speedup, 1.54x full-layer speedup

The HBM cost model predicts ~2x (half the memory traffic). Measured 10-13x. The gap is kernel launch overhead — at sub-100 us per op, the 5-10 us CUDA dispatch latency is 10-20% of each call. Fusing 40 launches into 1 eliminates almost all of it.

> [SCREENSHOT: results table — shape, GQA ratio, GB/s, fusion speedup for A100 and H100]

---

**4/**

The ecosystem status on this fusion:

vLLM: has it (`enable_qk_norm_rope_fusion`). Disabled by default — H100 perf regression.
SGLang: has it (`fused_qknorm_rope`). Limited scope.
TensorRT-LLM: no Gemma support at all.
FlashInfer: no fusion (P0 issue filed upstream).
HuggingFace Transformers: no fusion.
llama.cpp: no fusion.

This is not "nobody thought of it." vLLM's team recognized the opportunity and built a solution. The problem is making it actually fast across hardware — which is an autotuning problem, not an algorithm problem.

---

**5/**

Why vLLM's fusion regresses on H100: their kernel assigns 1 attention head per warp. At batch size, the grid explodes — you end up with thousands of blocks fighting for SMs.

Our approach: parameterized Triton kernel where heads-per-warp is a tunable. A bandit search over the config space finds that packing multiple heads per warp is critical on H100 but not on A100. Hardware-specific configs, discovered automatically.

The bandit finds configs 22-167% better than hand-tuned starting points. On T4 it discovered that `num_warps=1` with small block sizes dominates — a config that wasn't in our curated starter list at all.

---

**6/**

Critical Gemma gotcha that bit us and will bite you:

Gemma's RMSNorm uses `y = x * rstd * (1 + weight)`, NOT the standard `y = x * rstd * weight`.

If your fused kernel uses the standard form while the trained weights assume `(1+w)`, outputs are silently wrong by ~10x in magnitude. No NaN, no crash, just confidently wrong inference. HuggingFace has `Gemma4RMSNorm` for this. vLLM has a separate `GemmaRMSNorm` CustomOp.

We handle it as a kernel parameter: one flag toggles standard vs. Gemma-mode affine. Works on LLaMA, Mistral, Phi-3, Qwen, and Gemma — not a Gemma-specific kernel.

---

**7/**

torch.compile deserves more credit than it gets here. It sees the full fusion opportunity — 0 graph breaks across the entire prologue. The issue is narrow: Inductor's fusion heuristics split at reduction boundaries. It emits 4 Triton kernels where the math allows 1.

This is a known limitation, not a bug. Inductor is optimizing for the general case. A hand-written Triton kernel can fuse across the reduction because we know the row fits in registers for these head dimensions (256 / 512). Inductor can't assume that.

---

**8/**

What this is NOT:

- Not "10x faster than vLLM end-to-end." This is prologue-only. Full model inference has 100+ layers of optimization beyond this kernel.
- Not production-deployed. Correct, fast, reproducible — but not wired into a serving stack yet.
- Not magic on all GPUs. T4 (Turing) sees 6.77x prologue fusion benefit but Triton codegen quality on SM 7.5 means absolute throughput trails PyTorch native CUDA. Honest negative result, fully documented.

---

**9/**

This kernel came out of Noeris, an autonomous kernel search system I've been building:

- 9 parameterized Triton operators
- Shape-indexed cross-run config database
- Learned cost model (R^2 = 0.94)
- Multi-armed bandit config selector with meta-bandit router
- Cross-hardware transfer correlation: Spearman 0.967 (A100 to H100)
- Backward pass fusion for training, not just inference

The fused QK-RMSNorm+RoPE kernel is the system's headline result. But the real contribution is the search infrastructure that found it.

---

**10/**

Open source, MIT license.

Reproduce in ~3 min on Modal (~$0.20):
```
git clone https://github.com/PwnKit-Labs/noeris
pip install -e . modal numpy scikit-learn
python scripts/smoke_modal.py --full --h100 --qk-only --write-results
```

Paper draft, full data, and kernel source all in the repo.

Questions welcome. Corrections especially welcome — if the numbers are wrong or the framing is unfair to prior art, I want to know.

github.com/PwnKit-Labs/noeris

> [SCREENSHOT: repo README or paper abstract]
