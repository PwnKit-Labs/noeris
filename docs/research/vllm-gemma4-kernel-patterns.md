# vLLM Gemma 4 Kernel Patterns (deep read)

_Research pass 2026-04-11. Source: `vllm-project/vllm` main branch, Gemma 4 enabling PR #38826 (merged, commits `68b04400…a377ddc0`, 2026-04-02). HF transformers `models/gemma4/modeling_gemma4.py`._

## TL;DR for Noeris

**The single most important finding:** vLLM's Gemma 4 Python layer does **not** call a custom fused "QK-RMSNorm + RoPE + KV-write" kernel. It calls **three separate launches** — `RMSNorm` (q_norm/k_norm/v_norm per head), then `ops.rotary_embedding`, then `Attention` (which internally does `reshape_and_cache` → paged attention). The only Python-level fusion vLLM exposes is `fused_add_rms_norm` (RMSNorm + residual add). Everything else labelled "fusion" in the Gemma 4 pipeline audit is actually done **inside** the CUDA kernels (`pos_encoding_kernels.cu`, `reshape_and_cache_kernel`, `paged_attention_v1/v2`), not at the Python graph level.

This means:

- Noeris issue #31 ("fused QK-RMSNorm+RoPE prologue") is a **genuine gap vs. vLLM**, not a re-port — vLLM doesn't do it either. The prior art to follow is Unsloth and FlashAttention-3, not vLLM.
- Noeris issues #36/#37 (FusedMoE grouped GEMM) map 1:1 onto vLLM's `fused_moe_kernel` in `vllm/model_executor/layers/fused_moe/fused_moe.py`. Port it directly.
- Noeris issue #38 (PagedAttention decode) → `csrc/attention/paged_attention_v1.cu` is CUDA-only. The Python-level `PagedAttention` wrapper at `vllm/v1/attention/ops/paged_attn.py` is 51 lines and just dispatches to `ops.reshape_and_cache` + the C++ attention kernel. There is no Triton reference to copy.

## 1. Per-layer forward pass (annotated)

From `vllm/model_executor/models/gemma4.py::Gemma4DecoderLayer.forward` (lines 586–631, file is 1613 lines total, 1239 additions in PR #38826):

```
# Each arrow = separate kernel launch in vLLM today.
residual = hidden_states

# --- Attention sub-block ---
h = RMSNorm(residual, gamma_in)                      # kernel 1: ops.rms_norm
qkv = QKVParallelLinear(h)                           # kernel 2: cuBLAS GEMM
q, k, v = split(qkv)

# Gemma4 per-head RMSNorm (Q/K have learnable gamma, V doesn't)
q = q.unflatten(-1, (H_q, D)); q = RMSNorm(q, g_q); q = q.flatten()  # kernel 3
if not is_kv_shared_layer:
    k = k.unflatten(-1, (H_kv, D)); k = RMSNorm(k, g_k); k = k.flatten()  # kernel 4
    q, k = rotary_emb(positions, q, k)               # kernel 5: ops.rotary_embedding
    v = v.unflatten(-1, (H_kv, D)); v = RMSNorm(v, None); v = v.flatten()  # kernel 6 (no gamma)
else:
    q, _ = rotary_emb(positions, q, k)               # Q-only rotary; K/V reused from target layer

attn_out = Attention(q, k, v)                        # kernels 7+8: reshape_and_cache + paged_attention
o = o_proj(attn_out)                                 # kernel 9
h = RMSNorm(o, gamma_post_attn)                      # kernel 10
h = h + residual                                     # kernel 11 (often fused into RMSNorm via fused_add_rms_norm)

# --- MLP / MoE sub-block ---
residual = h
h = RMSNorm(h, gamma_pre_ff)                         # kernel 12
mlp_out = MLP(h)                                     # 3 GEMMs + gelu (kernels 13–15, or fused SwiGLU)

if enable_moe_block:                                 # Gemma 4 26B-A4B path
    h1 = RMSNorm(mlp_out, gamma_post_ff_1)
    router_logits = Router(residual)                 # RMSNorm(no-γ) → *root_size → *scale → GateLinear (fp32 out)
    h2 = RMSNorm(residual, gamma_pre_ff_2)
    h2 = MoE.experts(h2, router_logits)              # fused_moe_kernel (Triton) — see §3
    h2 = RMSNorm(h2, gamma_post_ff_2)
    mlp_out = h1 + h2                                # dense-MLP + MoE run in parallel, summed

h = RMSNorm(mlp_out, gamma_post_ff)
h = h + residual

# --- Per-Layer Embedding (PLE) tail ---
if hidden_size_per_layer_input > 0:
    pli = per_layer_input_gate(h) * per_layer_input  # gated projection
    pli = per_layer_projection(pli)
    h = post_per_layer_input_norm(h + pli)

h = h * layer_scalar                                 # per-layer learned gate (loaded buffer)
```

Per-layer variation:

| feature | sliding (`sliding_attention`) | full (`full_attention`) | k_eq_v (laptop variant) | KV-shared (last N layers) |
|---|---|---|---|---|
| `head_dim` | `config.head_dim` | `config.global_head_dim` (may differ) | same as full | inherits from target |
| `sliding_window` | set | None | None | inherits |
| RoPE | `rope_parameters["sliding_attention"]` | `rope_parameters["full_attention"]` | full | inherits |
| V-proj weights | loaded | loaded | K weights reused in both K and V slots of `QKVParallelLinear` | none (target layer's cache) |
| Q/K/V norm applied | yes | yes | yes | **Q only**; K/V norms skipped (K/V not recomputed) |
| MLP width | `intermediate_size` | `intermediate_size` | same | `2× intermediate_size` if `use_double_wide_mlp` |

MoE is only present on layers where `enable_moe_block` or `use_second_mlp_block` is true (i.e. every Nth layer in 26B-A4B; not every layer).

## 2. Fused QK-RMSNorm + RoPE prologue (Noeris #31 spec)

**vLLM status:** not fused. vLLM issues 4 separate launches per attention prologue (q_norm, k_norm, v_norm, rotary). This is a Noeris win opportunity, not a port.

Relevant vLLM sources to cross-reference against:

- `vllm/model_executor/models/gemma4.py` lines 395–427 — the unfused Python sequence.
- `vllm/model_executor/layers/layernorm.py::RMSNorm.forward_cuda` (line 262) — calls `ops.rms_norm(out, x, weight, eps)` or `ops.fused_add_rms_norm(x, residual, weight, eps)`. No Q/K-aware variant.
- `vllm/model_executor/layers/rotary_embedding/gemma4_rope.py` (84 lines) — only overrides `_compute_inv_freq` for "proportional" RoPE (exponent denominator = `head_size` not `rotary_dim`, zero-pad non-rotated dims). The actual rotation goes through the base `RotaryEmbedding.forward_cuda` which calls `ops.rotary_embedding`.
- `csrc/pos_encoding_kernels.cu::rotary_embedding_kernel` — one CUDA block per token; iterates all heads; reads cos/sin from `cos_sin_cache` of shape `[max_position, 2, rot_dim//2]` (first half cos, second half sin for neox style). Signature: `(positions, query, key, cos_sin_cache, rot_dim, query_stride, key_stride, head_stride, num_heads, num_kv_heads, head_size)`. Neox style rotates pairs `(x[i], x[i+embed_dim])`.

**HF reference (what you must match bitwise):** `src/transformers/models/gemma4/modeling_gemma4.py`:

- `Gemma4RMSNorm.forward` (line 171): `x = x.float(); var = x.pow(2).mean(-1, keepdim=True); x = x * rsqrt(var + eps); return (x * (1 + weight)).to(orig_dtype)` — **note the `(1 + weight)`**, Gemma-style, not pure `weight`. Easy to miss.
- `apply_rotary_pos_emb` (line 734): standard neox rotate_half.
- `Gemma4Attention.forward` (line 881): `q = q_norm(q)`, then `k = k_norm(k)`, then `apply_rotary_pos_emb(q, cos, sin)`, then `apply_rotary_pos_emb(k, cos, sin)`. V-norm applied separately, no rotation.

**Pseudocode Triton kernel spec for Noeris**:

```python
@triton.jit
def fused_qk_norm_rope(
    q_ptr, k_ptr, v_ptr,                 # [T, H_q*D], [T, H_kv*D], [T, H_kv*D]
    q_gamma_ptr, k_gamma_ptr,            # [D] each (Gemma: "1 + gamma" applied)
    cos_sin_ptr,                         # [max_pos, rot_dim] (cos[:rot_dim/2] | sin[rot_dim/2:])
    positions_ptr,                       # [T] int64
    T,                                   # num tokens (dynamic)
    EPS: tl.constexpr,
    D: tl.constexpr,                     # head_dim (power of 2, e.g. 128 or 256)
    ROT_DIM: tl.constexpr,               # == D for Gemma 4 (partial_rotary_factor=1)
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    APPLY_V_NORM: tl.constexpr,          # True for Gemma 4
    V_HAS_GAMMA: tl.constexpr,           # False for Gemma 4 (v_norm is pure)
    IS_KV_SHARED: tl.constexpr,          # if True, skip K/V paths entirely
    BLOCK_H: tl.constexpr,               # heads per program (e.g. 4)
):
    pid_t = tl.program_id(0)             # one token
    pid_h = tl.program_id(1)             # head group
    # Grid: (T, cdiv(max(H_Q, H_KV), BLOCK_H))

    pos = tl.load(positions_ptr + pid_t)
    cos = tl.load(cos_sin_ptr + pos*ROT_DIM + tl.arange(0, ROT_DIM//2))
    sin = tl.load(cos_sin_ptr + pos*ROT_DIM + ROT_DIM//2 + tl.arange(0, ROT_DIM//2))

    # --- Q path ---
    for h in range(pid_h*BLOCK_H, min((pid_h+1)*BLOCK_H, H_Q)):
        x = tl.load(q_ptr + pid_t*H_Q*D + h*D + tl.arange(0, D))
        # RMSNorm in fp32
        xf = x.to(tl.float32)
        rms = tl.sqrt(tl.sum(xf*xf)/D + EPS)
        gamma = tl.load(q_gamma_ptr + tl.arange(0, D)).to(tl.float32)
        xf = xf * (1.0 + gamma) / rms          # Gemma: (1 + γ)
        # NeoX rotate_half on first ROT_DIM elements
        x_rot1 = xf[:ROT_DIM//2]; x_rot2 = xf[ROT_DIM//2:ROT_DIM]
        y_rot1 = x_rot1*cos - x_rot2*sin
        y_rot2 = x_rot2*cos + x_rot1*sin
        out = tl.cat([y_rot1, y_rot2, xf[ROT_DIM:]])
        tl.store(q_ptr + pid_t*H_Q*D + h*D + tl.arange(0, D), out.to(x.dtype))

    if not IS_KV_SHARED:
        # --- K path: same as Q with k_gamma ---
        # --- V path: RMSNorm only, no rotation, optional γ ---
        ...
```

**Correctness invariants:**

- Numerics: match HF's `Gemma4RMSNorm` exactly. That means `(1 + weight)`, fp32 intermediate, `.to(orig_dtype)` at the end. The vLLM CUDA `ops.rms_norm` is registered via `gemma_rms_norm` CustomOp (`layernorm.py` line 358, `class GemmaRMSNorm`) which handles the `1+weight` convention. Noeris must mirror this or q_norm/k_norm outputs will be off by ~10x.
- Tolerance expected: the HF reference is in fp32 upcast, so absolute tol ≤ 1e-3, relative ≤ 1e-2 at bf16.
- Rotation style: NeoX (pair `[i, i+D/2]`), not GPT-J (pair `[2i, 2i+1]`). vLLM Gemma 4 explicitly passes `is_neox_style=True`.
- `partial_rotary_factor`: Gemma 4 passes `1.0` for both layer types per `gemma4_rope.py` line 40–42 comment; the HF `_compute_proportional_rope_parameters` uses `head_dim` (not `rotary_dim`) as the denominator in the inv_freq exponent. Noeris must reuse vLLM's proportional inv_freq formula verbatim.
- Q and K **can** be in the same kernel (both operate on the same token slot, same positions, same cos/sin); V adds a separate path but can share the outer token loop.
- GQA: `H_Q != H_KV`. Use `min(H_Q, H_KV)` for the shared inner loop and branch on `h < H_KV` for K/V writes.
- **Do not** fuse the KV-cache write (`reshape_and_cache`) into this kernel. vLLM keeps it separate because (a) the paged layout depends on `slot_mapping` which is a full attention-metadata object, and (b) decode and prefill call different attention backends that each want the post-norm K/V in the standard `[T, H_kv*D]` layout before cache insertion. Keep #31 as norm+rope only; the cache write is its own kernel.

## 3. FusedMoE grouped GEMM (Noeris #36 + #37 spec)

**vLLM source:** `vllm/model_executor/layers/fused_moe/fused_moe.py` (2319 lines).

Entry points:
- `fused_moe_kernel` (line 311) — the unquantized/fp8/int8 Triton grouped GEMM. **This is the one to port.**
- `fused_moe_kernel_gptq_awq` (line 77) — int4/int8 weight-only; skip for first port.
- `fused_experts` (line 1556) → `fused_experts_impl` (line 1633) → `invoke_fused_moe_kernel` (line 624) — the Python driver. Two invocations: one for `w1` (up+gate), one for `w2` (down).
- `moe_align_block_size` is imported from `vllm/model_executor/layers/fused_moe/moe_align_block_size.py` — **this is the token-to-expert permutation step** and is a separate kernel run before the GEMM. It produces `sorted_token_ids`, `expert_ids`, `num_tokens_post_padded`.
- Gemma 4 glue is at `vllm/model_executor/models/gemma4.py::Gemma4MoE` (line 183). It wires `FusedMoE(num_experts, top_k, hidden_size, intermediate_size, renormalize=True, custom_routing_function=routing_function, activation="gelu")`.

**Signature of `fused_moe_kernel`** (lines 311–365):

```python
@triton.jit
def fused_moe_kernel(
    a_ptr, b_ptr, c_ptr,
    b_bias_ptr, a_scale_ptr, b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    # dims
    N, K, EM, num_valid_tokens,
    # strides
    stride_am, stride_ak, stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask, stride_bse, stride_bsk, stride_bsn,
    stride_bbe, stride_bbn,
    # constexprs
    group_n: tl.constexpr, group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr, use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr, per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
)
```

Tensor shapes:
- `A` (input): `[M, K]`, M = num_tokens.
- `B` (stacked expert weights): `[E, N, K]`.
- `C` (output cache): `[M, top_k, N]`.
- `sorted_token_ids`: `[EM]` where `EM ≈ M * top_k + E * (BLOCK_SIZE_M - 1)` (padding per expert so each expert's rows are divisible by `BLOCK_SIZE_M`).
- `expert_ids`: `[EM / BLOCK_SIZE_M]` — one expert index per M-block.
- `num_tokens_post_padded`: scalar.

**Kernel structure** (lines 395–572):

1. **Grid**: `pid ∈ [0, cdiv(EM, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N))`. Grouped ordering on M-blocks (`GROUP_SIZE_M`) for L2 reuse.
2. **Token lookup**: load `sorted_token_ids[pid_m*BLOCK_SIZE_M : +BLOCK_SIZE_M]`; divide by `top_k` to map back to real input row. `offs_token // top_k` is used as the row index into `A` — **this is the scatter-gather trick**: no explicit permutation of A, just indexed loads.
3. **Expert dispatch**: `off_experts = expert_ids[pid_m]`. If `-1` (expert not on this EP rank), write zeros and return.
4. **Accumulate** in fp32 (`tl.dot(a, b, acc=accumulator)`).
5. **Top-k weight rescale**: if `MUL_ROUTED_WEIGHT`, `accumulator *= topk_weights[offs_token][:, None]` in fp32 before down-cast. Applied only to the **second** GEMM (`w2`, down-projection) so the sum over experts is a simple scatter-add.
6. **Output**: `c_ptrs = c + stride_cm*offs_token[:, None] + stride_cn*offs_cn[None, :]`. The scatter writes each token's expert contribution to a `[M, top_k, N]` buffer; a separate reduction sums across `top_k`.

**Default config** (`get_default_config`, line 1223, bf16 non-quantized path):

| M range | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | GROUP_SIZE_M | num_warps |
|---|---|---|---|---|---|
| ≤ 32 (decode) | 16 | 64 | 128 (if M ≤ 64) else 64 | 1 (unless `tokens_per_expert > 128`) | 4 |
| 33–96 | 32 | 64 | 128/64 | 1 | 4 |
| 97–512 | 64 | 128 | 64 | 1 | 4 |
| > 512 | 128 | 128 | 64 | 16 | 8 |
| `num_stages` | 3 (CUDA) / 2 (ROCm) |

Batch-invariant mode override (line 1232): `{BLOCK_M:64, BLOCK_N:64, BLOCK_K:32, GROUP_M:8, SPLIT_K:1}` — use this as the Noeris default for determinism.

**Gemma 4 specifics (from `Gemma4MoE`, lines 183–255):**

- `activation="gelu"` (not silu) — the MLP between the two GEMMs is `up * gelu(gate)`. Port the `invoke_fused_moe_kernel`'s activation switch.
- `renormalize=True` at the routing level, but Gemma 4 uses a **custom routing function** (line 213) that does `softmax over ALL experts → top-k → gate_weights = one_hot(topk_ids) * probs → renorm`, **then folds `per_expert_scale[topk_ids]` into `topk_weights`**. The kernel itself treats this as opaque `topk_weights` — no Gemma-specific kernel change needed.
- **No shared-expert** path in Gemma4MoE. The audit claim that "1 extra expert always active" appears to refer to the dense MLP running in parallel (`hidden_states_1 + hidden_states_2` at line 626), which is a **separate FFN kernel**, not an expert. Noeris can model this as "dense MLP result + MoE result" at the Python level; no kernel fusion needed.

**`moe_align_block_size`** (separate file): generates sorted_token_ids + expert_ids + pad. Implemented as a custom CUDA op; there is a Triton fallback. For Noeris, this is a small-M pre-pass — not the perf hot spot; port last.

**What Noeris needs:**

1. A Triton kernel with the same signature (can drop `a_scale`, `b_scale`, `use_fp8*`, `use_int8*`, `HAS_BIAS`, `naive_block_assignment` for first port → keep only the bf16 path).
2. A `moe_align_block_size` equivalent (can be pure PyTorch for first port — it's a sort + cumsum).
3. Two invocations per MoE layer: w1 (GEMM → gelu), w2 (GEMM with `MUL_ROUTED_WEIGHT=True`).
4. An `index_add` over top_k dim after w2 to collapse `[M, top_k, hidden]` → `[M, hidden]`.

## 4. PagedAttention decode (Noeris #38 spec)

**vLLM status:** CUDA-only hand-written kernel. No Triton reference in vLLM proper.

Files:
- `vllm/v1/attention/ops/paged_attn.py` — 51 lines. Just defines `split_kv_cache` (view-manipulation) and `write_to_paged_cache` which calls `ops.reshape_and_cache`. No Python-level attention implementation.
- The decode kernel lives in `csrc/attention/paged_attention_v1.cu` / `paged_attention_v2.cu` (not fetched here; compiled into `vllm._C`).
- The Gemma 4 path calls it via `vllm/model_executor/layers/attention.py::Attention.forward` (the high-level wrapper that picks a backend: FlashAttention, FlashInfer, xFormers, or vLLM's own paged kernel depending on `AttentionBackend`).

**KV cache layout** (from `split_kv_cache`, line 17):

```
kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
  key_cache view:   [num_blocks, num_kv_heads, head_size/x, block_size, x]   where x = 16 / elem_size
  value_cache view: [num_blocks, num_kv_heads, head_size, block_size]
```

The `x` packing on the key cache is the "16-byte vector load" tile that the CUDA kernel uses — preserving this is important for vLLM-binary compat but **Noeris can choose a simpler layout** if it's self-contained.

**Block size:** default 16 tokens per block (vLLM convention; configurable via `CacheConfig.block_size`, common values 16 / 32).

**GQA + sliding window + paging** combination: handled by `per_layer_sliding_window` passed into `Attention(...)` (gemma4.py line 390) plus the `kv_sharing_target_layer_name`. In the CUDA kernel, sliding window is a mask bound applied per-query to the block-table walk; paging indirection is `block_table[seq_id][i] → physical_block_idx`.

**Pseudocode Noeris decode kernel (single-query paged attention, GQA-aware):**

```python
@triton.jit
def paged_attn_decode(
    q_ptr,                         # [B, H_q, D]
    k_cache_ptr,                   # [num_blocks, block_size, H_kv, D]   (Noeris layout)
    v_cache_ptr,                   # [num_blocks, block_size, H_kv, D]
    block_tables_ptr,              # [B, max_blocks_per_seq] int32
    seq_lens_ptr,                  # [B] int32
    out_ptr,                       # [B, H_q, D]
    softmax_scale,                 # 1 / sqrt(D) (but Gemma 4 uses scaling=1.0; norms absorb it)
    sliding_window,                # int or -1
    B, H_Q, H_KV,
    D: tl.constexpr, BLOCK_SIZE: tl.constexpr,  # block_size, e.g. 16
    MAX_BLOCKS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hq = tl.program_id(1)
    hkv = pid_hq // (H_Q // H_KV)   # GQA head mapping

    seq_len = tl.load(seq_lens_ptr + pid_b)
    start = max(0, seq_len - sliding_window) if sliding_window > 0 else 0

    q = tl.load(q_ptr + pid_b*H_Q*D + pid_hq*D + tl.arange(0, D))
    m = -inf; l = 0; acc = zeros(D)

    for blk_idx in range(start // BLOCK_SIZE, cdiv(seq_len, BLOCK_SIZE)):
        physical = tl.load(block_tables_ptr + pid_b*MAX_BLOCKS + blk_idx)
        k = tl.load(k_cache_ptr + physical*BLOCK_SIZE*H_KV*D + hkv*D + ...)  # [BLOCK_SIZE, D]
        v = tl.load(v_cache_ptr + physical*BLOCK_SIZE*H_KV*D + hkv*D + ...)
        s = tl.sum(q[None, :] * k, axis=1) * softmax_scale
        # apply per-position mask (sliding window, seq_len bound)
        s = tl.where(token_valid, s, -inf)
        # online softmax
        m_new = max(m, tl.max(s)); alpha = exp(m - m_new)
        acc = acc*alpha + tl.sum(exp(s - m_new)[:, None] * v, axis=0)
        l = l*alpha + tl.sum(exp(s - m_new)); m = m_new

    tl.store(out_ptr + ..., acc / l)
```

**Key design decisions from vLLM (observed from the Python wrapper + the C++ header):**

- **One CTA per (batch, query-head)** for v1; v2 adds partitions across the K dimension for long contexts. Noeris can start with v1.
- **Block table indirection lives in shared memory** in the CUDA kernel (pre-loaded for the whole seq). In Triton, just load from global each iteration — the L1 will catch it.
- **Soft-cap support**: Gemma 4 passes `attn_logit_softcapping`; apply as `s = softcap * tanh(s / softcap)` before the mask.
- **Scaling**: Gemma 4 uses `scaling=1.0` (gemma4.py line 295) because the Q/K RMSNorm gammas absorb the pre-attention scalar. Do NOT apply `1/sqrt(D)`.
- **No `reshape_and_cache` fusion** with the attention kernel. They are two launches.

## 5. Out of scope for Noeris

Things in vLLM that are either CUDA-only or too runtime-coupled to port:

1. **`paged_attention_v1.cu` / `paged_attention_v2.cu`** — hand-written CUDA with warp-level reductions, 16-byte vectorized loads, shared memory block-table caching. Re-implement in Triton from scratch rather than porting.
2. **`reshape_and_cache_flash`** — KV cache write kernel with the `[num_blocks, block_size, H, D/x, x]` pack. The `x`-packing is specific to the CUDA kernel; Noeris should pick a flat layout.
3. **`moe_align_block_size` CUDA op** — small pre-pass, pure PyTorch is fine.
4. **FlashInfer / FlashAttention-3 backends** — vLLM delegates most attention to external libraries. Don't try to port the wrappers; port the math instead.
5. **`QKVParallelLinear`, `RowParallelLinear`, tensor-parallel sharding** — these are vLLM runtime concerns, not kernels.
6. **The Gemma4 mm/vision tower** (`gemma4_mm.py`, 1341 lines) — out of scope; text pipeline only.
7. **Per-Layer Embedding (PLE) tail** — two small ReplicatedLinears + a norm. Not a fusion target.
8. **`fused_moe_kernel_gptq_awq`** — quantization variant; port only after the bf16 path works.
9. **`Oink` Blackwell custom-op fast path** (`layernorm.py` lines 147–189) — SM100-specific external dependency.
10. **`rocm_aiter_ops`, `_xpu_ops`** — non-NVIDIA paths.

## 6. References

- Gemma 4 decoder layer: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gemma4.py (PR #38826, commit `68b04400abde1c7762cda9e0e2c72da8a37a1858`, 2026-04-02)
- Gemma 4 RoPE: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding/gemma4_rope.py
- Rotary CUDA kernel: https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding_kernels.cu
- RMSNorm + `fused_add_rms_norm` + `GemmaRMSNorm`: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/layernorm.py (lines 56–102, 358–406)
- FusedMoE Triton kernel: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py (lines 311–572 for `fused_moe_kernel`, 1223–1310 for `get_default_config`, 1556–1600 for `fused_experts`)
- `moe_align_block_size`: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/moe_align_block_size.py
- PagedAttention Python wrapper: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/paged_attn.py (51 lines)
- PR #38826 (Gemma 4 enabling): https://github.com/vllm-project/vllm/pull/38826 — status: merged; author: @lucianommartins + @Isotr0py; commits dated 2026-04-02.
- HF transformers Gemma 4 reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py (`Gemma4RMSNorm` line 157 — note `(1 + weight)`; `apply_rotary_pos_emb` line 734; `Gemma4Attention.forward` line 881)
- Unsloth Gemma 3 blog: https://unsloth.ai/blog/gemma3 — **could not verify specific Triton kernel details from the blog content; it focuses on training fixes (float handling) rather than kernel-level fusions**. Check Unsloth's GitHub (`unsloth/models/gemma3.py`) directly for the actual patches.

---

_Verification notes:_ All file paths and line numbers above are from files fetched 2026-04-11 via `gh api`. The statement that "vLLM does not fuse QK-RMSNorm+RoPE" is verified by reading `Gemma4Attention.forward` (gemma4.py lines 395–427) and confirming four distinct Python-level op calls. The FusedMoE kernel signature and block-size table are quoted directly from `fused_moe.py` lines 311–365 and 1223–1310 respectively. The PagedAttention claim that "there is no Triton reference" is based on `paged_attn.py` being 51 lines of pure dispatch — confirmed by reading the full file. Any detail about the `.cu` attention kernel beyond its existence was not directly verified from source in this pass; the `csrc/attention/*.cu` files should be read separately when Noeris implements #38.
