# KernelBench Level 4 via Op-Substitution: Feasibility Assessment

**Issue:** Noeris #33
**Date:** 2026-04-11
**Scope:** Can we attack KernelBench Level 4 (20 HF model forward passes) by walking the module tree and swapping `nn.Linear`, `nn.LayerNorm`, `F.scaled_dot_product_attention`, etc. for Noeris Triton kernels?
**Verdict:** Yes, but only for ~11/20 problems, gated by a **fp32→fp16 dtype adapter decision** and a **Conv1D shim for GPT-2/GPT-Neo**. Realistic ceiling is **1.2×–1.5× end-to-end** on the GPT-2 / OPT / BART / Electra family. BigBird and Reformer should be skipped.

---

## 1. The 20 L4 problems

Every L4 problem file is a **thin wrapper** around `AutoModelForCausalLM.from_pretrained(name, config).forward(x).logits`, confirmed by inspecting [`16_gpt2_bs1_seq1023.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/16_gpt2_bs1_seq1023.py), [`5_google-bigbird-roberta-base_bs1_seq4095.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/5_google-bigbird-roberta-base_bs1_seq4095.py), and [`13_google-reformer-enwik8_bs32_seq256.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/13_google-reformer-enwik8_bs32_seq256.py) — the wrapper pattern is identical; only `model_name`, `batch_size`, and `sequence_length` change. Inputs are `torch.randint(vocab)`; default dtype is **fp32** (no `.half()` call in any wrapper). There are **6 distinct base architectures** across 20 configs:

| # | File | Model | (B, S) | Arch family |
|---|---|---|---|---|
| 1 | [`1_…gpt-neo-2p7B_bs32_seq256.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/1_EleutherAI-gpt-neo-2p7B_bs32_seq256.py) | gpt-neo-2.7B | (32, 256) | GPT-Neo |
| 2 | [`2_…opt-1p3b_bs1_seq2047.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/2_facebook-opt-1p3b_bs1_seq2047.py) | OPT-1.3B | (1, 2047) | OPT |
| 3 | [`3_…gpt-neo-2p7B_bs1_seq2047.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/3_EleutherAI-gpt-neo-2p7B_bs1_seq2047.py) | gpt-neo-2.7B | (1, 2047) | GPT-Neo |
| 4 | [`4_…opt-1p3b_bs32_seq256.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/4_facebook-opt-1p3b_bs32_seq256.py) | OPT-1.3B | (32, 256) | OPT |
| 5 | [`5_…bigbird-roberta-base_bs1_seq4095.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/5_google-bigbird-roberta-base_bs1_seq4095.py) | BigBird | (1, 4095) | BigBird |
| 6 | [`6_…bart-large_bs1_seq1023.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/6_facebook-bart-large_bs1_seq1023.py) | BART-large | (1, 1023) | BART |
| 7 | [`7_gpt2_bs32_seq256.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/7_gpt2_bs32_seq256.py) | gpt2 | (32, 256) | GPT-2 |
| 8 | [`8_…opt-1p3b_bs512_seq32.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/8_facebook-opt-1p3b_bs512_seq32.py) | OPT-1.3B | (512, 32) | OPT |
| 9 | [`9_…bigbird-roberta-base_bs32_seq256.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/9_google-bigbird-roberta-base_bs32_seq256.py) | BigBird | (32, 256) | BigBird |
| 10 | [`10_…bigbird-roberta-base_bs1024_seq32.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/10_google-bigbird-roberta-base_bs1024_seq32.py) | BigBird | (1024, 32) | BigBird |
| 11 | [`11_…electra-small_bs1_seq511.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/11_google-electra-small-discriminator_bs1_seq511.py) | Electra-small | (1, 511) | Electra (BERT) |
| 12 | [`12_…electra-small_bs1024_seq32.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/12_google-electra-small-discriminator_bs1024_seq32.py) | Electra-small | (1024, 32) | Electra |
| 13 | [`13_…reformer-enwik8_bs32_seq256.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/13_google-reformer-enwik8_bs32_seq256.py) | Reformer | (32, 256) | Reformer |
| 14 | [`14_…electra-small_bs32_seq256.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/14_google-electra-small-discriminator_bs32_seq256.py) | Electra-small | (32, 256) | Electra |
| 15 | [`15_…reformer-enwik8_bs1024_seq32.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/15_google-reformer-enwik8_bs1024_seq32.py) | Reformer | (1024, 32) | Reformer |
| 16 | [`16_gpt2_bs1_seq1023.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/16_gpt2_bs1_seq1023.py) | gpt2 | (1, 1023) | GPT-2 |
| 17 | [`17_…bart-large_bs1024_seq32.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/17_facebook-bart-large_bs1024_seq32.py) | BART-large | (1024, 32) | BART |
| 18 | [`18_…gpt-neo-2p7B_bs512_seq32.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/18_EleutherAI-gpt-neo-2p7B_bs512_seq32.py) | gpt-neo-2.7B | (512, 32) | GPT-Neo |
| 19 | [`19_gpt2_bs1024_seq32.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/19_gpt2_bs1024_seq32.py) | gpt2 | (1024, 32) | GPT-2 |
| 20 | [`20_…bart-large_bs32_seq256.py`](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level4/20_facebook-bart-large_bs32_seq256.py) | BART-large | (32, 256) | BART |

Architecture breakdown: OPT ×3, GPT-Neo ×3, GPT-2 ×3, BART ×3, BigBird ×3, Electra ×3, Reformer ×2 = 20.

---

## 2. Op inventory per architecture

I checked each family against the HF `transformers` modeling file. Noeris kernels (from `src/research_engine/triton_*.py`): `matmul`, `layernorm`, `rmsnorm`, `softmax`, `flash_attn` (causal + sliding-window + QK-norm), `rotary`, `geglu`, `cross_entropy`. All kernels are **fp16-only** at their output boundaries — I grepped the stores: every `tl.store` casts to `tl.float16`. This is a hard constraint.

| Family | HF source | Linear | LayerNorm | Softmax | Attention | Activation | Noeris-foreign ops |
|---|---|---|---|---|---|---|---|
| **GPT-2** | [`modeling_gpt2.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py) | uses **`Conv1D`** (kernel-size-1) for qkv & mlp, *not* `nn.Linear` | yes (pre-LN) | yes (in attn) | manual `matmul → scale → mask → softmax → matmul` | GELU (new) | Embedding, Conv1D, Dropout |
| **GPT-Neo** | [`modeling_gpt_neo.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neo/modeling_gpt_neo.py) | `nn.Linear` | yes | yes | manual + local attention mask | GELU | Embedding, Dropout, positional bias |
| **OPT** | [`modeling_opt.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py) | `nn.Linear` | yes | yes | `nn.MultiheadAttention`-style manual | ReLU | Embedding, LearnedPosEmb, Dropout |
| **BART** | [`modeling_bart.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py) | `nn.Linear` (self + cross-attn) | yes | yes | manual, non-causal encoder + causal decoder + **cross-attention** | GELU | Embedding, sinusoidal pos, encoder-decoder plumbing |
| **Electra** | [`modeling_electra.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/electra/modeling_electra.py) (BERT clone) | `nn.Linear` | yes (post-LN BERT style) | yes | manual non-causal | GELU | Embedding, TokenType, Dropout |
| **BigBird** | [`modeling_big_bird.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/big_bird/modeling_big_bird.py) | `nn.Linear` | yes | yes | **block-sparse / ITC / ETC** — custom gather/scatter over random+window+global tokens | GELU | Large hand-written sparse attention kernels |
| **Reformer** | [`modeling_reformer.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/reformer/modeling_reformer.py) | `nn.Linear` | **no** — uses reversible residuals | custom | **LSH + local attention**, hashing, chunking, reversible blocks | ReLU/GELU | Axial pos-emb, LSH hashing, reversible autograd |

Classification per op (for the Linear/LN/Attn/Softmax subset only; everything else is left as torch):

| Op | Status | Notes |
|---|---|---|
| `nn.Linear` | **Direct** (OPT/GPT-Neo/BART/Electra) | Flatten `(B,S,H)→(B·S,H)`, matmul, reshape back. Bias handled by fused epilogue or `+b`. |
| `Conv1D` (GPT-2) | **Adapter** | Transformers' `Conv1D` stores weight `(in,out)` not `(out,in)` — it's a transposed linear. Needs explicit `NoerisConv1D` wrapper. |
| `nn.LayerNorm` | **Direct** | Noeris `layernorm(x, w, b, config, eps)` matches exactly after `(B,S,H)→(B·S,H)` reshape. |
| `torch.softmax(dim=-1)` inside attention | **Skip** | Already fused inside Noeris `flash_attn`; if substituting at attention level, no standalone softmax kernel needed. |
| SDPA / manual attention | **Adapter** (GPT-2, GPT-Neo, OPT, BART, Electra) | Detect `matmul→scale→mask→softmax→matmul` idiom in each `*Attention` module and replace the whole forward with `flash_attn`. Requires per-arch attention wrappers — not a generic substitution. |
| Embedding / pos-emb / dropout | **Unnecessary** | Memory-bound lookups; leave as torch. |
| GELU/ReLU activation | **Unnecessary** | Noeris has `geglu` but no standalone GELU; torch GELU is already fused-epilogue-eligible. |
| Reformer LSH, BigBird block-sparse | **Not supported** | Would require writing new kernels from scratch. |
| Cross-attention (BART decoder) | **Adapter** | Noeris `flash_attn` is non-causal-capable; cross-attn has separate KV lengths — must verify mask shape path. |

---

## 3. Top-5 most-addressable problems

I'm ranking by **(fraction of FLOPs in Noeris-owned ops) × (shape favorability for Triton matmul)**. Small-batch short-seq configs have too little matmul work for kernel launch to pay off; the sweet spot is medium-large matmul tiles.

| Rank | Problem | Why | Noeris coverage | Est. ceiling |
|---|---|---|---|---|
| **1** | **#4 OPT-1.3B (32, 256)** | 24 layers × `(8192, 2048) @ (2048, 8192)` MLP matmuls dominate. OPT uses plain `nn.Linear` + `nn.LayerNorm` — cleanest substitution target. B·S = 8192, fits Noeris matmul sweet spot. | ~88% (matmul + LN + attention) | **~1.5×** if matmul ≥1.3× and flash-attn ≥2× on these shapes |
| **2** | **#20 BART-large (32, 256)** | 12 enc + 12 dec layers, `d_model=1024`, `ffn=4096`. B·S=8192. Plain `nn.Linear` + `LayerNorm`. Cross-attention is the only wrinkle. | ~82% (decoder cross-attn needs adapter) | **~1.4×** |
| **3** | **#1 GPT-Neo-2.7B (32, 256)** | 32 layers, `d_model=2560`, `ffn=10240`. Huge matmuls. Local-attention windows match Noeris sliding-window path. | ~85% | **~1.4×** (limited by 32-layer launch overhead on short seq) |
| **4** | **#14 Electra-small (32, 256)** | BERT backbone: 12 layers, d=256, ffn=1024. Small matmuls hurt, but LayerNorm (post-LN BERT) and softmax dominate in relative terms. Good regression target — fast to iterate. | ~80% | **~1.25×** |
| **5** | **#7 GPT-2 (32, 256)** | 12 layers, d=768. Requires `Conv1D` shim (see §5). Once shim lands, clean target. | ~82% (after Conv1D adapter) | **~1.3×** |

Estimates assume: Noeris matmul @ 1.3× cuBLAS on these shapes (consistent with our A100/H100 matmul ablation in `docs/results/ablation-matmul-multitrial.md`), Noeris layernorm @ 2.5× `nn.LayerNorm` (consistent with rmsnorm/layernorm ablations), and flash_attn @ 2× manual unfused attention. Amdahl's-law math: if 85% of time is in Noeris-owned ops at avg 1.5× speedup, end-to-end = 1 / (0.15 + 0.85/1.5) ≈ **1.36×**. These are **not** eligible for the 5× headline numbers on KernelBench leaderboard — L4 speedups are structurally bounded by embedding, dropout, and python dispatch overhead.

---

## 4. `NoerisOpSubstitutor` design

Pseudocode:

```
class NoerisOpSubstitutor:
    RULES = {
        nn.Linear:    LinearWrapper,          # Noeris matmul + bias epilogue
        nn.LayerNorm: LayerNormWrapper,       # Noeris layernorm
        Conv1D:       Conv1DWrapper,          # transformers.pytorch_utils.Conv1D
        # attention handled at Module-class level, not op level:
        GPT2Attention:    FlashAttnWrapper.gpt2,
        OPTAttention:     FlashAttnWrapper.opt,
        BartAttention:    FlashAttnWrapper.bart,
        GPTNeoSelfAttention: FlashAttnWrapper.gpt_neo,
        ElectraSelfAttention: FlashAttnWrapper.bert_style,
    }

    def substitute(self, model):
        for name, module in list(model.named_modules()):
            for cls, Wrapper in self.RULES.items():
                if type(module) is cls:             # exact type, not isinstance
                    parent, attr = _resolve(model, name)
                    wrapped = Wrapper.from_module(module)   # copies weights, no clone
                    wrapped.original = module               # keep ref for correctness gate
                    setattr(parent, attr, wrapped)
                    break

    def validate(self, orig, subst, sample_input, atol=1e-3, rtol=1e-3):
        with torch.no_grad():
            a = orig(sample_input); b = subst(sample_input)
        assert torch.allclose(a, b, atol=atol, rtol=rtol), ...

    def time(self, model, input, n_iters=200):
        return cuda_event_timer_with_l2_flush(model, input, n_iters)   # issue #23
```

Each wrapper's `forward` looks like:
```
class LinearWrapper(nn.Module):
    def __init__(self, lin):  self.weight, self.bias = lin.weight, lin.bias
    def forward(self, x):
        x2 = x.reshape(-1, x.shape[-1]).contiguous().to(torch.float16)
        out = noeris_matmul(x2, self.weight.t().contiguous().to(torch.float16), cfg)
        if self.bias is not None: out = out + self.bias.to(torch.float16)
        return out.reshape(*x.shape[:-1], -1).to(x.dtype)
```

**Python/Torch primitives used:**
- `model.named_modules()` + `setattr(parent, attr, new)` for in-place tree rewrite (no `register_forward_pre_hook` — hooks can't replace outputs cleanly and add dispatch overhead).
- `type(m) is cls` (not `isinstance`) so subclasses don't accidentally match.
- `torch.allclose` with `atol=1e-3, rtol=1e-3` as the correctness gate (relaxed from default 1e-5 because of fp16 intermediate precision).
- `torch.cuda.Event` pair with L2 flush — already on the backlog as issue #23.
- `functools.wraps` is **not** needed here; we subclass `nn.Module`, not wrap functions.

The wrapper caches its fp16 weight once at init (not per forward) so the `.to(float16)` copy happens at substitution time, not in the hot path. This is mandatory for fair benchmarking.

---

## 5. Blockers and risks

**Hopeless problems — skip.**
- **#5, #9, #10 BigBird** — [modeling_big_bird.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/big_bird/modeling_big_bird.py) implements block-sparse attention over a random + window + global token graph. There is no Noeris kernel for this, it dominates runtime, and writing a block-sparse flash attention is a multi-week project.
- **#13, #15 Reformer** — [modeling_reformer.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/reformer/modeling_reformer.py) uses LSH hashing + reversible residuals. No LayerNorm to substitute, custom autograd graph, completely foreign to Noeris.

That takes **5 problems off the table** immediately; realistic scope is **15 problems**.

**Ops we have kernels for but should NOT substitute.**
- **Standalone softmax** inside attention: once `flash_attn` replaces the full attention module, the explicit softmax is gone. Don't double-substitute.
- **Cross-entropy at the head**: all L4 problems return `.logits`, not `loss`. There's no cross-entropy in the forward path. Skip.
- **Tiny matmuls for `(1024, 32)` configs (#10, #12, #15, #17, #19)**: B·S = 1024, hidden=256–1024. These are launch-overhead-bound; `nn.Linear` already calls cuBLAS `gemv`-ish paths that are hard to beat. Expected speedup < 1.1×; probably a loss once we add the fp16 cast overhead. **Deprioritize the `bs1024_seq32` configs.**

**The Conv1D footgun.**
HF GPT-2's attention QKV and MLP projection are [`transformers.pytorch_utils.Conv1D`](https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py), a 1×1 "conv" whose `weight` is stored as `(nf, nx)` with `y = x @ w + b` — it's a transposed linear. A naive "replace all `nn.Linear`" walk **will miss every GPT-2 projection**. We need an explicit `Conv1D` rule in the substitutor, and we must transpose the weight at substitution time (or pass `b` on the correct side to matmul). This also affects #7, #16, #19.

**The fp32 dtype mismatch — the big one.**
All 20 L4 wrappers run in fp32 by default (no `.half()`, no `torch.autocast`). Every Noeris `tl.store` casts to `tl.float16`. Two options:

1. **Cast at the boundary** (fast): wrapper does `x.to(fp16)` → `noeris_op` → `.to(orig_dtype)`. Pros: zero kernel changes. Cons: reference-vs-substituted numerical drift; `atol=1e-3` may be too tight for a 24-layer OPT forward (errors compound). Risk: KernelBench's `atol` defaults are `1e-2` in the harness — we need to confirm this.
2. **Make Noeris kernels dtype-polymorphic** (correct): parameterize `tl.float16` → `OUT_DTYPE` constexpr. Pros: clean correctness. Cons: **every kernel + every autotune cache needs re-keying**; breaks the shape-indexed learning store. This is the bigger refactor and we should avoid it for L4 v1.

Recommendation: **cast at boundary with atol=5e-3, rtol=5e-3**. If end-to-end drift breaks correctness on OPT-1.3B at 24 layers, fall back to running the whole wrapper under `torch.autocast("cuda", dtype=torch.float16)` (which is what KernelBench-level5 grads already do) and accept that we're benchmarking an fp16 model vs. an fp32 reference — we'd need to argue that's fair play. This is a **judgment call for the maintainer before any L4 submission**.

**Cross-attention in BART.**
[BART decoder](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py) has encoder-decoder attention where `q` comes from decoder hidden states but `k, v` come from encoder output with different seq length. Noeris `flash_attn` signature `(q, k, v, config, is_causal, ...)` already supports differing k/v length (that's the standard flash-attention shape), but I have not verified the causal-mask path handles `q_len != k_len` correctly. **Needs a unit test before #6/#17/#20 substitution.**

---

## 6. Concrete next steps

1. **Build `NoerisOpSubstitutor` skeleton with Linear + LayerNorm rules only.** No attention yet. Add correctness gate and cuda-event timing. Validate on Electra-small first — smallest model, fastest iteration. **~250 LOC**, 1 day.
2. **Add `Conv1D` and `GPT2Attention` rules.** Requires unit tests for the weight-transpose and for the `(q @ k.T / sqrt(d) + causal_mask).softmax() @ v` → `flash_attn` rewrite. **~300 LOC**, 2 days.
3. **Add `OPTAttention`, `GPTNeoSelfAttention`, `BartAttention` rules.** Each is an arch-specific attention wrapper that extracts q/k/v projections, calls `flash_attn`, and fuses back. BART cross-attention gets a dedicated test. **~400 LOC**, 3 days.
4. **Validation run: the top-5 from §3, in this order — #14 Electra (smallest, fastest), #4 OPT-1.3B (the headline target), #20 BART, #1 GPT-Neo, #7 GPT-2.** Report per-problem end-to-end speedup, per-op breakdown from nsys or torch profiler, and correctness margin. **~1 day** of benchmarking once rules land.

Total estimated effort: **~7 dev-days + ~1 benchmark day ≈ 1.5 weeks** to a first L4 number.

**Go/no-go gate after step 1:** if Electra-small end-to-end is *slower* than baseline even after step-1 substitution, stop and re-examine. The per-Linear fp16-cast cost dominates small-matmul problems and that would invalidate the whole op-substitution thesis for L4. Do not proceed to step 2 until step 1 shows a measurable win.

---

## What I could not verify from available sources

- **KernelBench L4 atol/rtol defaults**: I did not read the L4 harness code; the "cast-at-boundary" plan depends on it being ≥ 1e-3. This must be checked before any submission.
- **Noeris `flash_attn` behavior when `q_len != k_len`** (BART cross-attn): the kernel grid generator accepts separate shapes but I did not trace the mask path.
- **Exact matmul speedup on these specific (M, N, K) shapes**: our ablation data is on square shapes; OPT MLP matmuls are `(8192, 2048) × (2048, 8192)` — needs a targeted shape sweep before committing to the 1.5× ceiling estimate.
- **Reformer / BigBird skip decision is based on HF source inspection only**; I did not profile them. If someone wants to sanity-check, a `torch.profiler` run would confirm that ≥80% of their time is in ops Noeris doesn't own.
