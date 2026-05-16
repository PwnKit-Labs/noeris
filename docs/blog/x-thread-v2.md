# X/Twitter Thread v2 — Noeris Launch

_Draft 2026-04-12. 12 tweets. Tone: confident, honest, inviting scrutiny._

---

**1/**

I wrote a Triton kernel that beats cuDNN FlashAttention on sliding-window attention.

6.24x on A100. 3.56x on T4. 8 out of 8 shapes win on A100.

To my knowledge, this is the first Triton kernel to beat cuDNN on sliding-window attention. The trick: compile-time tile-pruning skips 96-98% of the tiles cuDNN computes densely. ~160 lines of Triton.

Corrections welcome if I'm wrong on the novelty claim.

> [SCREENSHOT: Table 13 from paper — A100 and T4 sliding-window results side by side, showing N/W/H/D/speedup/tile skip ratio]

---

**2/**

How it works: Gemma 3/4 runs 5 of 6 attention layers as 1024-token local windows. At W=64 and N=8192, each query touches 64 of 8192 key positions. cuDNN FlashAttention computes the full causal triangle — it doesn't exploit the window constraint at tile level.

Our kernel's compile-time tile bounds restrict the K-tile loop to only the overlapping tiles. At these parameters, 98% of cuDNN's work is wasted.

The win is specific: long sequences, small windows, small head_dim. At W=1024 or head_dim=256, cuDNN wins. I'm explicit about this because the result is not "Triton beats cuDNN on attention" — it's "workload-specific pruning beats dense iteration in its weak regime."

---

**3/**

But the bigger story is the system that found this kernel.

Noeris is an autonomous kernel search system I've been building. 20 parameterized Triton operators. One consistent interface. Every benchmark result persists in a shape-indexed database keyed by (operator, shape, hardware).

The operator list: RMSNorm, LayerNorm, softmax, cross-entropy, RoPE, fused QK-RMSNorm+RoPE (forward + backward), attention (GQA + sliding-window + QK-norm + K=V sharing), GeGLU, MoE router, grouped GEMM, PLE gather, paged-KV decode, fused norm+matmul, matmul.

> [SCREENSHOT: operator table from paper showing all 20 operators with their parameter spaces and metrics]

---

**4/**

The headline fusion kernel: QK-RMSNorm+RoPE.

Every LLM inference engine launches ~40 CUDA kernels for the attention prologue. torch.compile gets it to 9. Noeris: 1.

40 → 9 → 1.

torch.compile sees the full graph (0 graph breaks!) but Inductor splits at the RMSNorm reduction boundary — it emits 4 separate Triton kernels because it won't fuse across a row-wise reduce. We fuse across the reduction in registers.

Load once, norm once, rotate once, store once.

> [SCREENSHOT: bar chart — 40 / 9 / 1 kernel launches, with torch.compile graph annotation showing the 4 output kernels and the reduction boundary split]

---

**5/**

Prologue fusion results:

A100: 10.2-12.9x
H100: 10.4-11.9x, peak 1,627 GB/s (49% HBM3 peak)
T4: 6-9x

HBM cost model predicts ~2x. Measured 10-13x. The gap is launch overhead — at sub-100us per op, the 5-10us CUDA dispatch latency eats 10-20% of each call. Fusing 40 launches into 1 eliminates almost all of it.

19 models. 13 families. Zero failures. Gemma, LLaMA 3/4, Qwen 3, Mistral, Mixtral, Phi-3/4, Falcon 3, DBRX, OLMo 2, InternLM 3. All 6-9x fusion speedup.

> [SCREENSHOT: Table 14 — A100 19-model fusion speedup table, sorted by speedup descending]

---

**6/**

The vLLM situation, because credit matters:

vLLM has `enable_qk_norm_rope_fusion`. It exists. Their team recognized the opportunity and built a solution. But it's disabled by default — H100 perf regression (issue #34391). Reported benefit: 2-3% E2E.

The problem isn't the idea. The problem is making it fast across hardware. That's an autotuning problem. Their kernel uses 1 head per warp — at batch size, the grid explodes on H100. Our bandit discovered that packing multiple heads per warp is critical on H100 but not on A100. Hardware-specific configs, found automatically.

---

**7/**

End-to-end: 1.18x on a full 26-layer Gemma 4 forward pass. 322.7ms → 274.4ms.

The fused operations (RMSNorm, QK-norm+RoPE, GeGLU) account for less than 20% of per-layer compute. 18% E2E improvement from fusing the minority of the workload. That's 6-9x more E2E impact than vLLM's 2-3%.

The matmuls and SDPA run identical PyTorch code in both configurations. This is purely the fusion wins compounding across 26 layers.

---

**8/**

Two things nobody else exploits at kernel level (to my knowledge):

**K=V sharing.** Gemma 4's global attention layers have K=V — literally the same tensor. Nobody exploits this in the attention kernel. We do: halve the KV-cache reads.

**PLE fusion.** Gemma 4 E2B/E4B uses per-layer embeddings — a gather operation every layer. We fuse the PLE gather into the operator pipeline instead of launching it as a separate kernel.

Also shipped: fused norm+matmul. RMSNorm fused with the subsequent linear projection. To my knowledge, the first Triton implementation — Mirage (OSDI 2025) showed the algebraic insight, we provide the portable kernel.

---

**9/**

Cross-hardware transfer: the search infrastructure result that matters most for practicality.

Train the cost model on A100 data only. Use it to predict A100 configs from T4 measurements: Spearman rho = 0.907 (p < 1e-7). Zero-shot. No target-device fine-tuning.

The A100-trained cost model's operator-level rankings transfer to H100 with rho = 0.967.

This means you can tune on cheap hardware and transfer. TenSet (Chen et al. 2021) requires target-device fine-tuning. We don't.

> [SCREENSHOT: Table 8 — cross-hardware transfer showing per-operator Spearman rho values, R^2 vs Spearman gap]

---

**10/**

What this is NOT:

- Not "6x faster than cuDNN on all attention." The sliding-window win is specific to narrow-window, long-sequence, small head_dim. I'm explicit about where cuDNN still wins.
- Not "13x faster inference." The prologue fusion is prologue-only. Full model E2E is 1.18x.
- Not production-deployed. Correct, fast, reproducible — but not wired into a serving stack yet.
- Not magic on T4. Triton codegen on Turing (SM 7.5) has a 3-5x per-op overhead vs PyTorch native. The fusion wins are real, the absolute throughput is not.

---

**11/**

The system behind this:

- 20 parameterized Triton operators, 110 shape buckets, 606 unit tests
- Shape-indexed cross-run config database
- Gradient-boosted cost model (R^2 = 0.94)
- Multi-armed bandit + adaptive meta-bandit router
- Cross-hardware transfer: rho = 0.907 (A100 configs from T4 data)
- Backward pass fusion for training, not just inference
- ~$0.01 per search iteration on Modal

The fused kernels are the headline. But the real contribution is the search infrastructure that found them.

---

**12/**

Open source, MIT license.

Reproduce the sliding-window result in ~3 min on Modal (~$0.20):
```
git clone https://github.com/0sec-labs/noeris
pip install -e . modal numpy scikit-learn
python scripts/smoke_modal.py --full --a100 --write-results
```

Paper draft, full data, kernel source all in the repo. Every number in this thread has a reproduction script.

Questions welcome. Corrections especially welcome — if the numbers are wrong, the novelty claims are overstated, or the framing is unfair to prior art, I want to know.

github.com/0sec-labs/noeris

> [SCREENSHOT: repo README header or paper abstract]
