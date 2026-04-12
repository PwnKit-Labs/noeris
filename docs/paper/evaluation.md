# §4. Evaluation

This section reports all empirical results for Noeris. Every number is drawn
directly from raw benchmark artifacts in `docs/results/`. No smoothing or
post-hoc selection has been applied: FAIL rows count as zero speedup in the
fast_p denominator.

---

## §4.1 Experimental Setup

**Hardware.** All benchmarks run on cloud GPUs provisioned through
[Modal](https://modal.com). The A100 target is an `NVIDIA A100-SXM4-40GB`
with 40 GB HBM2e. The H100 target is an `NVIDIA H100 SXM5-80GB`. Containers
are warm-started; the first call may incur a ~10 s cold-start overhead that
is excluded from timing.

**Software.** Triton kernels are written against Triton 2.x (matching the
Modal image). PyTorch eager baselines use the version bundled in the same
Modal image. We do not pin exact version strings in the artifact metadata, but
both the Triton and PyTorch versions present in the Modal `noeris-gpu` image
at the time of evaluation are used uniformly across all problems.

**Problem set.** We evaluate on a curated 53-problem subset of KernelBench-style
shapes drawn from real LLM workloads at difficulty Levels 1–3:

| Operator | Problems | Levels | Representative shapes |
|---|---|---|---|
| matmul | 19 | L1–L2 | 128³–4096³; GPT-2, BERT, LLaMA-7B, Mistral MLP dims |
| rmsnorm | 7 | L1–L2 | hidden dim 768–8192 (GPT-2 through LLaMA-70B) |
| softmax | 8 | L1–L2 | width 1k–128k (attention rows and vocab projections) |
| layernorm | 6 | L1–L2 | hidden dim 768–8192, up to long sequence variants |
| cross_entropy | 7 | L1–L2 | vocab 50k–128k (BERT through LLaMA-3) |
| attention | 6 | L2–L3 | seq 512–4096, head_dim 64–128 (causal and non-causal) |

All problems use FP16 tensors. Warmup and measurement repetitions match the
timing conventions used by the PyTorch eager baseline in the same harness.

**Baselines.** Primary baseline is **PyTorch eager**: `F.linear` → cuBLAS;
`F.rms_norm`/`F.layer_norm` → fused ATen; `F.softmax` → eager;
`F.cross_entropy` → fused; `F.scaled_dot_product_attention` → FlashAttention-2/3.
We do not run `torch.compile` for every problem; where AutoKernel uses it as
baseline, we note the discrepancy in §4.4.

**Configs per problem.** For the KernelBench evaluation we submit **4 curated
starter configurations per problem** and take the best result. No search
iterations are run; the curated configs represent the initial state of the
Noeris search loop, not the outcome of any LLM-guided exploration.

**Cost.** The full two-GPU evaluation (53 problems × 2 hardware targets = 106
problem×hardware pairs, each with 4 configs) consumed approximately **$1.44**
in Modal GPU credits, including overhead from the ablation runs described in
§4.5. Individual KernelBench eval runs cost approximately $0.20 (A100) and
$0.40 (H100).

---

## §4.2 Aggregate Results

We report **fast_p** scores following KernelBench convention: fast_p(t) is the
fraction of problems where Noeris achieves a speedup ≥ t× over the PyTorch
eager baseline using any of the 4 curated configs.

**Table 1. fast_p scores — NVIDIA A100-SXM4-40GB (53 problems)**

| Threshold | Overall | Level 1 (24 probs) | Level 2 (26 probs) |
|---|---|---|---|
| fast_1.0 (any speedup) | **56.6%** | 58.3% | 61.5% |
| fast_1.5 | 41.5% | 45.8% | 42.3% |
| fast_2.0 | 37.7% | 37.5% | 42.3% |
| fast_3.0 | 32.1% | 33.3% | 34.6% |

**Table 2. fast_p scores — NVIDIA H100 SXM5 (53 problems)**

| Threshold | Overall | Level 1 (24 probs) | Level 2 (26 probs) |
|---|---|---|---|
| fast_1.0 (any speedup) | **56.6%** | 58.3% | 61.5% |
| fast_1.5 | 43.4% | 45.8% | 46.2% |
| fast_2.0 | **41.5%** | 45.8% | 42.3% |
| fast_3.0 | 30.2% | 33.3% | 30.8% |

fast_1.0 is identical across A100 and H100 because the same problems pass and
fail. H100 gains at fast_1.5 and fast_2.0: higher memory bandwidth (3.35 TB/s
vs 2.0 TB/s) amplifies our memory-bound kernel throughput enough to push more
problems over the 1.5–2.0× thresholds. At fast_3.0 H100 drops slightly
(30.2% vs 32.1%) because cuBLAS and FlashAttention baselines also scale with
H100 compute, pulling Noeris's matmul and attention results below 3.0× on a
handful of problems. The 6 attention problems (Level 2–3) score 0% at every
threshold on both GPUs; our simplified kernel does not beat PyTorch's SDPA
(§4.3).

---

## §4.3 Per-Operator Deep Dive

### RMSNorm

RMSNorm is the clearest win. On H100 the Mixtral hidden-dim shape
(hidden_dim=4096, rows=2048) reaches **2625.1 GB/s at 11.66× PyTorch eager**;
LLaMA-13B and LLaMA-7B shapes achieve 11.20× and 11.11× respectively. On A100
the same shapes yield 10.43×, 10.03×, and 10.11×.

PyTorch's eager `F.rms_norm` launches a generic ATen kernel that cannot
fully exploit HBM bandwidth for the reduction-then-scale pattern. Our Triton
kernel fuses squared-mean reduction, reciprocal-sqrt, and element-wise scale
into a single pass with `BLOCK_SIZE=2048, num_warps=8`, saturating HBM
bandwidth for LLaMA-scale hidden dimensions. Smaller GPT-2 hidden dims (768)
achieve 4.60–5.60× because short rows hit L2 effects and require fewer warps.

### Softmax

Softmax shows the largest shape-sensitivity of any operator. On H100, speedups
range from 2.18× (tiny, width=256) to **6.38× (vocab_llama, width=32000)**.
Wide vocab-projection rows fit in a single large block without thread
divergence; our fused online-max-then-exp-normalize pattern reaches 2526.4 GB/s
vs 396.1 GB/s eager. Narrower attention-score rows (width 512–4096) are only
2.55–4.86× because PyTorch's eager softmax is already reasonably efficient at
those widths. H100 delivers roughly 1.9–2.2× the raw GB/s of A100 on
equivalent problems, consistent with the hardware bandwidth ratio.

### Cross-Entropy

Cross-entropy delivers our best absolute numbers: 2407.1 GB/s at **9.65×**
over eager on H100 (`long_llama`), and 2266.3 GB/s at 9.08× (Mistral). On
A100 the same shapes reach 10.95× and 10.17× — A100 relative speedups exceed
H100 because PyTorch's fused cross-entropy scales well with H100's extra
bandwidth while our kernel's gains are bounded by shared-memory tile size.
The outlier is `llama3_128k` (vocab=131072): only 2.38× on H100 and 2.28×
on A100. At this vocab width the required BLOCK_SIZE overflows shared memory
and we fall back to a smaller tile, defeating vectorized reduction. Multi-pass
tiling would fix this but is not yet implemented.

### LayerNorm

LayerNorm is our weakest operator relative to the baseline. On H100 speedups
range from 1.25–1.53× (`long_seq`: 1666.8 GB/s vs 1088.4 GB/s eager); on A100,
1.22–1.34×. The gap relative to rmsnorm is expected: LayerNorm requires both
mean and variance passes (Welford), whereas RMSNorm only needs the squared
mean. PyTorch's fused ATen LayerNorm kernel is therefore far more competitive.
We do not yet implement a fused two-pass Welford kernel; this is the primary
target for search iterations and the main reason we trail AutoKernel on this
operator (§4.4).

### Matmul

Matmul tells an expected story: cuBLAS is hard to beat. On A100, Noeris
achieves 0.66–0.98× across 17 successful problems (2 FAILs counted as 0× in
fast_p). On H100, the range is 0.44–1.01×. The two LLaMA-7B shapes (QKV
and MLP-up) reach **1.01× on H100** with config
`bm128_bn256_bk64_gm8_w8_s3`, tying cuBLAS at 691.9 and 632.8 TFLOPS —
the only operator where we match the baseline on a real production shape.
Small-M shapes (M≤256) and K-heavy shapes (`matmul_deep`, 0.44× on H100) are
the worst cases; our default tile configs are calibrated for large square
matrices and degrade on extreme aspect ratios.

### Attention

Our simplified FlashAttention-style kernel achieves 0.68–0.83× of
`F.scaled_dot_product_attention` on H100 and 0.80–1.05× on A100 across the
six attention problems. The one A100 win (1.05×, `attn_short_64`) does not
hold on H100 (0.68×). `torch.nn.functional.scaled_dot_product_attention`
dispatches to FlashAttention-2/3 on NVIDIA hardware, implementing loop-order
optimizations and register-pressure tuning that our 120-line reference does
not replicate. We include attention to establish a starting point for the
search loop, not to claim competitiveness with production implementations.

---

## §4.4 Comparison to AutoKernel

AutoKernel [arXiv:2603.21331] is the closest published comparison point: it
also runs Triton kernels on H100 and reports PyTorch baselines per-operator.
We reproduce their Table 4 numbers alongside ours.

**Table 3. Best-shape comparison to AutoKernel on H100 (Triton, PyTorch eager baseline)**

| Kernel | **Noeris best** | **AutoKernel Table 4** | Delta | Note |
|---|---|---|---|---|
| RMSNorm | **11.66×** (Mixtral, 2625 GB/s) | 5.29× | **+120%** | same baseline type |
| Cross-entropy | **9.65×** (long\_llama, 2407 GB/s) | 2.94× | **+228%** | same baseline type |
| Softmax | **6.38×** (vocab\_llama, 2526 GB/s) | 3.44× | **+85%** | same baseline type |
| LayerNorm | 1.53× (long\_seq, 1667 GB/s) | 3.21× | **-52%** | ⚠ different baselines |
| matmul | 1.01× (llama7b\_qkv, 692 TFLOPS) | not reported | — | — |

On RMSNorm, Cross-entropy, and Softmax, Noeris exceeds AutoKernel's published
figures by large margins. These operators are all memory-bound and highly
sensitive to the fused tiling strategy; our curated starter configs already
implement near-optimal block sizes for the LLaMA-scale hidden dimensions that
appear to be AutoKernel's test shapes.

**LayerNorm: an apples-to-oranges comparison.** AutoKernel's 3.21× LayerNorm
figure is measured against `torch.compile` as the baseline, not PyTorch eager.
AutoKernel's paper reports that `torch.compile` achieves 3.0× over eager on
the same LayerNorm shape. Our 1.53× is against eager directly. Scaling our
result to the same reference: 1.53× / 3.0× ≈ 0.51× of compile — we are
meaningfully slower. AutoKernel's LayerNorm kernel is more optimized than ours
on large shapes, and LayerNorm remains our primary improvement target. It is
also worth noting that AutoKernel's paper reports 1.07× against eager on one
specific LayerNorm shape, which is comparable to our 1.25–1.34× range for L1
shapes.

These comparisons use curated starter configs with no search iterations. A
proper head-to-head would require running both systems with matched compute
budget and identical random seeds on a shared problem set. The comparison
above is directional, not a controlled benchmark.

---

## §4.5 Cross-Run Learning Ablation (Honest Negative Result)

The central architectural claim of Noeris is that a persistent shape-indexed
configuration database accelerates kernel search by surfacing historical winners
as cross-run priors. We test this claim with a controlled multi-trial ablation.

**Protocol.** For each operator, we run 3 independent trials. Within each
trial, we execute 5 iterations with a live database (the `with_database`
condition, where each iteration's winners are written back and visible to the
next) and 5 iterations with a reset empty database each time (the
`without_database` condition). The LLM proposer runs in both conditions with
the same prompt template; the only difference is whether it receives historical
winners as context. We report the **final best metric** at the end of 5
iterations for each trial.

**Results.**

**Table 4. Multi-trial ablation results (3 trials × 5 iterations, A100)**

| Operator | with\_database (mean ± σ) | without\_database (mean ± σ) | Relative Δ |
|---|---|---|---|
| matmul | 135.52 ± 3.12 TFLOPS | 138.51 ± 1.53 TFLOPS | **-2.16%** |
| rmsnorm | 784.39 ± 6.30 GB/s | 794.33 ± 15.49 GB/s | **-1.25%** |

For matmul, the individual trial finals are: `with_database` = [132.11, 138.24,
136.22] TFLOPS; `without_database` = [140.08, 137.03, 138.42] TFLOPS. For
rmsnorm: `with_database` = [788.76, 777.17, 787.24] GB/s; `without_database` =
[783.14, 812.01, 787.83] GB/s.

Neither result is statistically significant. The -2.16% and -1.25% relative
changes are smaller than the within-condition standard deviation and fall
squarely within the ~2–3% GPU runner noise floor. A t-test against H₀: Δ=0
would not reject the null at any conventional significance level.

We also ran a single-trial ablation for softmax on A100 (5 iterations per
condition). The `with_database` condition reached a final best of 864.40 GB/s;
the `without_database` condition reached 969.61 GB/s. The softmax result is
directionally reversed and more pronounced than the noise floor — but with n=1
trial it is not reproducible. We include it to motivate the multi-trial design
and note that a higher-powered ablation is needed before any conclusion can be
drawn.

**Why the negative result?** We believe the dominant factor is the strength of
the curated starter configs. Both conditions begin with 4 hand-picked
configurations that are known to work well across LLaMA-scale shapes. Both
converge to a near-optimal plateau within 1–2 iterations, leaving little
room for the database's cross-run priors to add value. In the limit of
infinitely strong curated priors, the database is redundant.

A secondary factor is iteration budget: 5 iterations is short enough that
neither condition has time to explore substantially beyond the starter configs.
The database's value compounds over many iterations, not within 5.

**Experimental designs that would give the approach more room to show value:**

1. **Weaker priors.** Remove curated configs and seed only from the systematic
   parameter grid. Without hand-picked starters, database historical winners
   should be significantly more valuable than random grid exploration.

2. **Longer budgets.** Run 20–50 iterations per condition. Short budgets favor
   already-good configs; longer budgets favor approaches that steer exploration
   efficiently toward unexplored but promising regions.

3. **Harder operators.** Attention and large matmul have 6–7 dimensional
   parameter spaces vs 3 for the norm operators. Random exploration is less
   likely to stumble on good configs in high-dimensional spaces, giving the
   database more leverage.

4. **Cold novel shapes.** Test on shapes not seen in any previous run but
   similar to shapes that are in the database. This isolates bucket-level
   generalization: does the database produce better starting points for new
   shapes than the curated starters alone?

The framework fully supports all four experiments via the `ablation` and
`triton-iterate` CLI commands. We leave them as explicit future work.

**Mitigation via the learned cost model.** The noise-floor problem is avoided
by the learned cost model (§2.5), which operates at prediction time —
deterministically re-ranking grid candidates by predicted throughput before
any GPU call is made. Its ablation is clean: measure wall-clock time to
first-within-5%-of-best with and without cost model ranking, a comparison
not contaminated by runner variance. With R²=0.535 on 144 training points,
the model is already useful; at 1000+ points we expect substantially tighter
predictions.

---

## §4.6 Reproducibility

**Cost breakdown.**

| Experiment | GPU type | Approx cost | Approx time |
|---|---|---|---|
| KernelBench eval — A100 (53 problems, 4 configs) | A100-SXM4-40GB | ~$0.20 | ~10 min |
| KernelBench eval — H100 (53 problems, 4 configs) | H100 SXM5 | ~$0.40 | ~8 min |
| Ablation — matmul, 3 trials × 5 iters each | A100-SXM4-40GB | ~$0.12 | ~15 min |
| Ablation — rmsnorm, 3 trials × 5 iters each | A100-SXM4-40GB | ~$0.12 | ~15 min |
| Ablation — softmax, 1 trial × 5 iters each | A100-SXM4-40GB | ~$0.06 | ~8 min |
| Miscellaneous dev/debug calls | A100/H100 | ~$0.54 | — |
| **Total** | | **~$1.44** | — |

All results are reproducible with:

```bash
git clone https://github.com/PwnKit-Labs/noeris
cd noeris
pip install -e ".[dev]"
pip install modal scikit-learn datasets
modal token new   # requires a Modal account

# KernelBench eval on A100 (~$0.20, ~10 min)
python -m research_engine.cli kernelbench-eval --gpu A100

# KernelBench eval on H100 (~$0.40, ~8 min)
python -m research_engine.cli kernelbench-eval --gpu H100

# Multi-trial cross-run ablation on rmsnorm
python -m research_engine.cli ablation --operator rmsnorm \
    --trials 3 --iterations 5 --fast

# Multi-trial cross-run ablation on matmul
python -m research_engine.cli ablation --operator matmul \
    --trials 3 --iterations 5 --fast
```

**Data availability.** Raw benchmark results are in `docs/results/` in both
JSON (per-problem shapes, configs, raw metric values) and Markdown form:
`kernelbench-{a100,h100}-53problems.{json,md}`, `ablation-matmul-multitrial.{json,md}`,
`ablation-rmsnorm-multitrial.{json,md}`, `ablation-softmax-a100.md`.
All tables in this section can be reconstructed from these artifacts. No
numbers were computed outside them.
