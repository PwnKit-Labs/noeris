# GQA Implementation Spec — `triton_attention.py`

**Status:** spec only, no code changes. Unblocks Noeris issues #28 and #29.
**Date:** 2026-04-11.
**Author:** research pass following Gemma 4 audit (`docs/research/gemma4-pipeline-audit.md`).
**Target file:** `src/research_engine/triton_attention.py` (line numbers below reference current HEAD).

## Background

Audit on 2026-04-11 found the Triton attention kernel (`attn_fwd_kernel`, defined at `triton_attention.py:219`) assumes `q.shape[1] == k.shape[1] == v.shape[1]` (i.e. MHA). Every modern LLM uses GQA: Gemma 4 31B is 32:4 global / 32:16 local, Gemma 4 26B-A4B is 16:2 global / 16:8 local, Gemma 4 E2B is 8:1, Llama 3 70B is 64:8, Mistral is 32:8. This spec adds a `NUM_KV_HEADS` constexpr and the minimal launcher + bucket + test machinery around it.

Key design invariant: **the Q program-id grid does not shrink**. Each Q head still owns its own `(BLOCK_M)` program; GQA only changes which K/V rows that program reads. This matches FlashAttention-2 §3.2 and the HuggingFace `repeat_kv` pattern in `modeling_llama.py` (except the replication happens by pointer arithmetic, never by materializing the replicated tensor).

---

## Part 1 — Kernel diff (pseudocode)

### 1.1 New constexpr

Insert `NUM_KV_HEADS: tl.constexpr` into the signature at `triton_attention.py:219-235`, **between `HEAD_DIM` and `BLOCK_M`**:

```python
@triton.jit
def attn_fwd_kernel(
    Q, K, V, Out,
    QScale, KScale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B, H, M, N,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,   # NEW — total KV heads (= H for MHA, H/group for GQA)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    USE_QK_NORM: tl.constexpr,
):
```

Placement rationale: constexprs ordered roughly by "shape-like → block-size → feature flags", so `NUM_KV_HEADS` pairs with `HEAD_DIM`. Downstream `num_warps` / `num_stages` are keyword args and do not care about position.

### 1.2 Derive `kv_head_idx`

At `triton_attention.py:262-270`, the kernel currently does:

```python
pid = tl.program_id(0)
off_bh = tl.program_id(1)
off_b = off_bh // H
off_h = off_bh % H          # Q head index, range [0, H)

q_base = Q + off_b * stride_qb + off_h * stride_qh
k_base = K + off_b * stride_kb + off_h * stride_kh   # BUG under GQA
v_base = V + off_b * stride_vb + off_h * stride_vh   # BUG under GQA
o_base = Out + off_b * stride_ob + off_h * stride_oh
```

Replace with:

```python
pid = tl.program_id(0)
off_bh = tl.program_id(1)
off_b = off_bh // H
off_h = off_bh % H                                  # Q head, [0, H)
# NUM_KV_HEADS is a compile-time constant; integer division folds.
GROUP_SIZE: tl.constexpr = H // NUM_KV_HEADS
# NOTE: Triton constexpr from a runtime value (H) is not legal. See §1.2b.
off_kvh = off_h // GROUP_SIZE                       # KV head, [0, NUM_KV_HEADS)

q_base = Q   + off_b * stride_qb + off_h  * stride_qh
k_base = K   + off_b * stride_kb + off_kvh * stride_kh
v_base = V   + off_b * stride_vb + off_kvh * stride_vh
o_base = Out + off_b * stride_ob + off_h  * stride_oh
```

### 1.2b Group size — where the division happens

`H` is a kernel runtime arg (see `B, H, M, N,` at `triton_attention.py:227`). You cannot declare `GROUP_SIZE` as `tl.constexpr` from a runtime int. Two clean options; **recommend option A**:

**Option A — pass `GROUP_SIZE` directly as constexpr (preferred).** The launcher knows `H` and `num_kv_heads` at Python time, so it passes `GROUP_SIZE = H // num_kv_heads` directly. Replace the constexpr as:

```python
HEAD_DIM:     tl.constexpr,
NUM_KV_HEADS: tl.constexpr,   # kept for codegen clarity / config caching
GROUP_SIZE:   tl.constexpr,   # = H // NUM_KV_HEADS, integer, >= 1
BLOCK_M:      tl.constexpr,
...
```

And inside the kernel:

```python
off_kvh = off_h // GROUP_SIZE
```

Integer division of an `int32` by a `constexpr` is legal and compiles to a shift or mul-hi depending on GROUP_SIZE. Both `NUM_KV_HEADS` and `GROUP_SIZE` are kept as constexprs so the bandit's per-(shape, config) cache key distinguishes GQA regimes.

**Option B — compute `off_kvh` in the launcher and pass as a grid dim.** Clean semantically but requires reshaping the grid; **rejected** because it breaks the existing `grid = (cdiv(M, BLOCK_M), B * H, 1)` invariant that lots of downstream bandit/cost-model machinery assumes.

### 1.3 Stride semantics (`stride_kh`, `stride_vh`)

The caller (PyTorch) passes `k.stride(1)` / `v.stride(1)` unchanged. These are the *bytes-per-KV-head stride of the K/V tensor*. Because K and V are allocated with shape `(B, NUM_KV_HEADS, N, D)`, `k.stride(1) == N * D` (for contiguous tensors), which is exactly what we want to multiply `off_kvh` by. **No launcher-side pre-multiplication; no kernel-side division.** This is the same convention as the HF `repeat_kv`-free kernel path in FlashAttention-2.

Q and Out continue to use `stride_qh = q.stride(1)` with `off_h` (Q head). No changes to Q/Out pointer math.

### 1.4 Causal + sliding-window masks — unchanged

Masks at `triton_attention.py:336-344` operate over `(offs_m, curr_n)` — sequence positions only. They are **head-agnostic** and therefore unchanged by GQA. Confirmed by inspection: no mask code references `off_h`.

### 1.5 QK-norm + GQA interaction — unchanged, one subtlety

Q-norm (`triton_attention.py:281-289`) is per-row (`axis=1` over HEAD_DIM) with a single shared `QScale[HEAD_DIM]` weight vector. K-norm (`triton_attention.py:316-324`) is identically structured. Both are per-token RMSNorm with a per-head-dim affine; they do **not** index head or kv-head. So under GQA:

- `QScale` and `KScale` remain `[HEAD_DIM]` float32.
- The kernel still normalizes each loaded K tile. Since the tile loaded for Q-head `h` now comes from kv-head `h // GROUP_SIZE`, the same kv-row gets normalized `GROUP_SIZE` times across different programs — **redundant but correct**. Optimizing this (pre-norm K once) is out of scope.
- Verify at implementation time: the K-norm reads the same `KScale[offs_k]` regardless of kv-head, which is correct because RMSNorm affine is over the head-dim axis, not the head axis. **Gemma 3/4 publishes per-head-dim scale; no per-head scale.** (HF `Gemma3RMSNorm` has shape `[head_dim]`.)

No changes needed.

---

## Part 2 — Launcher diff

Current `flash_attn` at `triton_attention.py:367-412`. Changes:

```python
def flash_attn(
    q, k, v, config, is_causal=False, sm_scale=None, window_size=-1,
    use_qk_norm=False, q_scale=None, k_scale=None,
    num_kv_heads=None,               # NEW
):
    B, H, M, D = q.shape
    _, Hk, N, Dk = k.shape
    _, Hv, Nv, Dv = v.shape
    if num_kv_heads is None:
        num_kv_heads = H             # MHA fallback (backwards compat)
    assert num_kv_heads > 0, "num_kv_heads must be positive"
    assert H % num_kv_heads == 0, f"H={H} not divisible by num_kv_heads={num_kv_heads}"
    assert Hk == num_kv_heads and Hv == num_kv_heads, (
        f"K/V must have shape[1] == num_kv_heads; got Hk={Hk} Hv={Hv} num_kv_heads={num_kv_heads}"
    )
    assert Dk == D and Dv == D, "head_dim must match across Q/K/V"
    assert Nv == N, "K/V seq_len must match"
    group_size = H // num_kv_heads
    ...
    grid = (triton.cdiv(M, BLOCK_M), B * H, 1)   # UNCHANGED — grid is over Q heads
    attn_fwd_kernel[grid](
        q, k, v, out,
        q_scale, k_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),   # stride(1) is per-KV-head
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, M, N,
        sm_scale,
        HEAD_DIM=D,
        NUM_KV_HEADS=num_kv_heads,   # NEW
        GROUP_SIZE=group_size,       # NEW
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
        WINDOW_SIZE=window_size,
        USE_QK_NORM=use_qk_norm,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out
```

Grid is unchanged: `(cdiv(M, BLOCK_M), B * H, 1)`. GQA saves no work on the Q side — it saves memory for KV cache and reduces KV projection FLOPs upstream.

`benchmark_one(...)` (at `triton_attention.py:437`) needs a new `num_kv_heads` kwarg (default to `heads`) plumbed through to both the allocation lines and the `flash_attn(...)` calls. See Part 6 for allocation and reference.

---

## Part 3 — Shape bucket additions

Current buckets at `triton_attention.py:57-78`. **Backwards-compat rule:** every existing bucket gets an explicit `"num_kv_heads": <same as heads>` field. This is the only MHA signal we can rely on since we cannot distinguish MHA from "GQA with group=1" any other way.

Add after the existing QK-norm buckets:

```python
ATTENTION_SHAPE_BUCKETS = [
    # ... existing buckets, each with num_kv_heads == heads added ...
    {"name": "short_64",             "batch": 4, "heads": 32, "num_kv_heads": 32, "seq_len": 512,  "head_dim": 64,  "is_causal": False},
    {"name": "short_128",            "batch": 2, "heads": 16, "num_kv_heads": 16, "seq_len": 1024, "head_dim": 128, "is_causal": False},
    # ... (same pattern for all 15 existing buckets) ...

    # --- New GQA buckets ---
    {"name": "gemma4_31b_local",      "batch": 1, "heads": 32, "num_kv_heads": 16, "seq_len": 4096, "head_dim": 256, "is_causal": True, "window_size": 1024, "use_qk_norm": True},
    {"name": "gemma4_31b_global",     "batch": 1, "heads": 32, "num_kv_heads": 4,  "seq_len": 4096, "head_dim": 512, "is_causal": True, "window_size": -1,   "use_qk_norm": True},
    {"name": "gemma4_26b_a4b_local",  "batch": 1, "heads": 16, "num_kv_heads": 8,  "seq_len": 4096, "head_dim": 256, "is_causal": True, "window_size": 1024, "use_qk_norm": True},
    {"name": "gemma4_26b_a4b_global", "batch": 1, "heads": 16, "num_kv_heads": 2,  "seq_len": 4096, "head_dim": 512, "is_causal": True, "window_size": -1,   "use_qk_norm": True},
    {"name": "llama3_70b_gqa",        "batch": 1, "heads": 64, "num_kv_heads": 8,  "seq_len": 4096, "head_dim": 128, "is_causal": True, "window_size": -1},
    {"name": "mistral_gqa",           "batch": 1, "heads": 32, "num_kv_heads": 8,  "seq_len": 8192, "head_dim": 128, "is_causal": True, "window_size": -1},
]
```

Note `head_dim=512` on the two global Gemma 4 buckets — this is new in Gemma 4 (Gemma 3 used 256 throughout). Source: `docs/research/gemma4-pipeline-audit.md:13-20` and HF transformers issue #45201.

---

## Part 4 — `attention_shape_bucket_key` additions

Current function at `triton_attention.py:85-116`. Read `num_kv_heads` and introduce GQA branch **before** the QK-norm branch so Gemma-4 GQA + QK-norm shapes route correctly:

```python
def attention_shape_bucket_key(shape: dict[str, int]) -> str:
    seq = shape.get("seq_len", 0)
    hd = shape.get("head_dim", 0)
    heads = shape.get("heads", 0)
    nkv = shape.get("num_kv_heads", heads)   # default = MHA
    ws = shape.get("window_size", -1)
    use_qk_norm = bool(shape.get("use_qk_norm", False))
    is_gqa = nkv > 0 and nkv < heads

    # GQA buckets take precedence — these are the LLM-shaped routes.
    if is_gqa:
        if use_qk_norm:
            # Gemma 4. head_dim=512 signals global, 256 signals local.
            if hd >= 512:
                # 32:4 vs 16:2 distinguished by heads
                return "gemma4_31b_global" if heads >= 32 else "gemma4_26b_a4b_global"
            # local layers: distinguish 31B (32:16) vs 26B (16:8) by heads
            return "gemma4_31b_local" if heads >= 32 else "gemma4_26b_a4b_local"
        # Non-QK-norm GQA: Llama 3 / Mistral
        if seq >= 8192:
            return "mistral_gqa"
        if heads >= 64:
            return "llama3_70b_gqa"
        # Fallback: treat as mistral_gqa for now
        return "mistral_gqa"

    # ... existing QK-norm / window / MHA logic unchanged ...
```

**Open decision — flag for the reviewer:** the current QK-norm branch at `triton_attention.py:93-96` does not check `nkv`, so MHA + QK-norm shapes (the existing `gemma4_qknorm` bucket — `heads=16, num_kv_heads=16`) still route to `gemma4_qknorm`. That's correct under the "is_gqa" gate above. Verify at implementation time that no existing test shape has `heads > num_kv_heads` with `use_qk_norm=True` under a different label.

---

## Part 5 — Test diff (`tests/test_triton_attention.py`)

### 5.1 `WindowSizeMinusOneRegressionTests` / `WindowShapeBucketTests` (lines 132-276)

No change needed — all existing shapes default `num_kv_heads = heads`. Add an explicit regression test:

```python
def test_mha_shapes_still_route_to_mha_buckets(self):
    """GQA gate must not misroute MHA shapes."""
    shape = {"seq_len": 4096, "head_dim": 128, "heads": 16}  # no num_kv_heads
    self.assertEqual(attention_shape_bucket_key(shape), "long_128")
    shape = {"seq_len": 4096, "head_dim": 128, "heads": 16, "num_kv_heads": 16}
    self.assertEqual(attention_shape_bucket_key(shape), "long_128")
```

### 5.2 New `GQAShapeBucketTests` class

```python
class GQAShapeBucketTests(unittest.TestCase):
    def test_gemma4_31b_local_bucket(self):
        shape = {"seq_len": 4096, "head_dim": 256, "heads": 32, "num_kv_heads": 16,
                 "window_size": 1024, "use_qk_norm": True, "is_causal": True}
        self.assertEqual(attention_shape_bucket_key(shape), "gemma4_31b_local")

    def test_gemma4_31b_global_bucket(self):
        shape = {"seq_len": 4096, "head_dim": 512, "heads": 32, "num_kv_heads": 4,
                 "window_size": -1, "use_qk_norm": True, "is_causal": True}
        self.assertEqual(attention_shape_bucket_key(shape), "gemma4_31b_global")

    def test_gemma4_26b_a4b_local_bucket(self):  # analogous
    def test_gemma4_26b_a4b_global_bucket(self): # analogous
    def test_llama3_70b_gqa_bucket(self):
        shape = {"seq_len": 4096, "head_dim": 128, "heads": 64, "num_kv_heads": 8,
                 "is_causal": True}
        self.assertEqual(attention_shape_bucket_key(shape), "llama3_70b_gqa")
    def test_mistral_gqa_bucket(self): # analogous
    def test_extreme_gqa_num_kv_heads_1(self):
        """Gemma 4 E2B: 8 Q heads, 1 KV head."""
        shape = {"seq_len": 4096, "head_dim": 256, "heads": 8, "num_kv_heads": 1,
                 "window_size": 1024, "use_qk_norm": True, "is_causal": True}
        # Should not crash and should return *some* GQA bucket.
        key = attention_shape_bucket_key(shape)
        self.assertIn("gemma", key)
    def test_all_new_gqa_buckets_present_in_shape_list(self):
        names = {s["name"] for s in ATTENTION_SHAPE_BUCKETS}
        for n in ["gemma4_31b_local", "gemma4_31b_global",
                  "gemma4_26b_a4b_local", "gemma4_26b_a4b_global",
                  "llama3_70b_gqa", "mistral_gqa"]:
            self.assertIn(n, names)
    def test_new_gqa_buckets_have_num_kv_heads_lt_heads(self):
        for s in ATTENTION_SHAPE_BUCKETS:
            if "gqa" in s["name"] or s["name"].startswith("gemma4_31b") or s["name"].startswith("gemma4_26b"):
                self.assertLess(s["num_kv_heads"], s["heads"])
                self.assertEqual(s["heads"] % s["num_kv_heads"], 0)
```

### 5.3 `BenchmarkScriptWindowTests` — compile-time test for one GQA shape

Add:

```python
def test_script_with_gqa_shape_is_valid_python(self):
    shapes = [{
        "name": "gemma4_31b_local", "batch": 1, "heads": 32, "num_kv_heads": 16,
        "seq_len": 4096, "head_dim": 256, "is_causal": True,
        "window_size": 1024, "use_qk_norm": True,
    }]
    script = self._make_script(shapes)
    compile(script, "<benchmark_gqa>", "exec")

def test_script_embeds_num_kv_heads(self):
    shapes = [{"name": "x", "batch": 1, "heads": 8, "num_kv_heads": 2,
               "seq_len": 512, "head_dim": 64}]
    script = self._make_script(shapes)
    self.assertIn("NUM_KV_HEADS", script)
    self.assertIn("num_kv_heads", script)
```

### 5.4 `GridSharedMemoryTests` — no change

The grid generator doesn't know about shapes; still a no-op shmem check. The bandit learns which (BLOCK_M, BLOCK_N, num_stages) configs OOM at `head_dim=512` from runtime failures (reward=0), as already designed.

### 5.5 New test: `num_kv_heads=1` routes

Already covered in `test_extreme_gqa_num_kv_heads_1` above. Plus, add an assertion in a new `LauncherAssertionsTests` (skip-if-no-torch):

```python
@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class LauncherAssertionsTests(unittest.TestCase):
    def test_flash_attn_rejects_mismatched_kv_heads(self):
        from research_engine.triton_attention import flash_attn
        q = torch.zeros(1, 8, 16, 64, dtype=torch.float16)
        k = torch.zeros(1, 4, 16, 64, dtype=torch.float16)  # 4 kv heads
        v = torch.zeros(1, 4, 16, 64, dtype=torch.float16)
        cfg = {"BLOCK_M": 16, "BLOCK_N": 16, "num_warps": 2, "num_stages": 2}
        with self.assertRaises(AssertionError):
            flash_attn(q, k, v, cfg, num_kv_heads=2)  # k/v have 4, not 2
    def test_flash_attn_rejects_nondivisible(self):
        q = torch.zeros(1, 6, 16, 64, dtype=torch.float16)
        k = torch.zeros(1, 4, 16, 64, dtype=torch.float16)
        v = torch.zeros(1, 4, 16, 64, dtype=torch.float16)
        cfg = {"BLOCK_M": 16, "BLOCK_N": 16, "num_warps": 2, "num_stages": 2}
        with self.assertRaises(AssertionError):
            flash_attn(q, k, v, cfg, num_kv_heads=4)  # 6 % 4 != 0
```

(These assertions fire before the kernel launches, so they run without CUDA. The kernel compile step does not fire on an assertion-only path.)

---

## Part 6 — Correctness reference (inside `benchmark_one`)

Current reference at `triton_attention.py:437-466`. Update the allocation and the PyTorch reference:

```python
def benchmark_one(batch, heads, seq_len, head_dim, config, is_causal=False,
                  window_size=-1, use_qk_norm=False, num_kv_heads=None):
    try:
        if num_kv_heads is None:
            num_kv_heads = heads
        q = torch.randn((batch, heads,        seq_len, head_dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, num_kv_heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        v = torch.randn((batch, num_kv_heads, seq_len, head_dim), device="cuda", dtype=torch.float16)

        # For PyTorch reference: SDPA in PyTorch >= 2.5 supports GQA natively
        # (K/V with fewer heads than Q — internally broadcasts). If the installed
        # torch is older, fall back to repeat_interleave.
        # VERIFY-AT-IMPLEMENTATION: check torch.__version__ and the SDPA docs for
        # `enable_gqa=True` or automatic broadcasting. Safest is to always
        # repeat_interleave for the reference — it costs nothing (reference is
        # not benchmarked for speed) and removes the version dependency.
        group_size = heads // num_kv_heads
        k_ref = k.repeat_interleave(group_size, dim=1) if group_size > 1 else k
        v_ref = v.repeat_interleave(group_size, dim=1) if group_size > 1 else v

        if use_qk_norm:
            q_ref = torch.nn.functional.rms_norm(q.float(), (head_dim,)).half()
            k_ref = torch.nn.functional.rms_norm(k_ref.float(), (head_dim,)).half()
        else:
            q_ref = q

        if window_size > 0:
            ws_mask = make_sliding_window_mask(seq_len, window_size, is_causal).to(q.device)
            ref = torch.nn.functional.scaled_dot_product_attention(
                q_ref, k_ref, v_ref, attn_mask=ws_mask, is_causal=False,
            )
        else:
            ref = torch.nn.functional.scaled_dot_product_attention(
                q_ref, k_ref, v_ref, is_causal=is_causal,
            )

        out = flash_attn(q, k, v, config, is_causal=is_causal,
                         window_size=window_size, use_qk_norm=use_qk_norm,
                         num_kv_heads=num_kv_heads)
        ...
```

**Recommendation: always `repeat_interleave` the reference** and skip the torch-version check. The reference path is run once per (config, shape) for correctness; its cost is irrelevant. This also makes the test deterministic across torch versions and avoids depending on whether `F.scaled_dot_product_attention` gained `enable_gqa` in 2.5 (could not verify in this research pass — flagged as VERIFY-AT-IMPLEMENTATION, but the `repeat_interleave` path sidesteps it).

Also update `main()` at `triton_attention.py:491-520` to read `num_kv_heads` from the shape dict and pass it through:

```python
num_kv_heads = int(shape.get("num_kv_heads", heads))
result = benchmark_one(
    batch, heads, seq_len, head_dim, config,
    is_causal=is_causal, window_size=window_size,
    use_qk_norm=use_qk_norm, num_kv_heads=num_kv_heads,
)
```

And add `result["num_kv_heads"] = num_kv_heads` to the result dict for downstream analysis.

---

## Part 7 — Edge cases and risks

### 7.1 `head_dim=512` on Gemma 4 global layers

Shared memory budget roughly doubles vs the previous 256 ceiling (and is 4x the 128 Llama baseline). The previous hardcoded shmem filter `(bm * 128 + 2 * bn * 128) * 2 * ns + 2048 <= 192_000` (see `test_grid_size_strictly_larger_without_filter` at `tests/test_triton_attention.py:319`) is already removed — feasibility is learned from runtime failures (`triton_attention.py:119-126`). Expected to **fail at launch** on A100 (48KB shmem/SM default, 164KB opt-in) and possibly H100:

- `BLOCK_M=128, BLOCK_N=128, num_stages>=3` at `HEAD_DIM=512` → `(128*512 + 2*128*512) * 2 * 3 + 2048 ≈ 1.18 MB` — definitely OOM.
- `BLOCK_M=64, BLOCK_N=64, num_stages=3` → `(64*512 + 2*64*512) * 2 * 3 ≈ 590 KB` — still OOM on A100 (164 KB cap).
- `BLOCK_M=32, BLOCK_N=32, num_stages=2` → `(32*512 + 2*32*512) * 2 * 2 ≈ 196 KB` — borderline; probably still too big on A100.
- `BLOCK_M=16, BLOCK_N=16, num_stages=2` → `~49 KB` — should fit.

The bandit will learn this. Call out: **expect most `head_dim=512` configs to reward=0 for the first trial**, and expect the explored set to collapse to tiny tiles. This is the desired behavior under the "learned feasibility" refactor.

### 7.2 `num_kv_heads=1` (Gemma 4 E2B global)

8 Q heads replicated from 1 KV head. Under the `off_kvh = off_h // GROUP_SIZE` scheme with `GROUP_SIZE=8`, all 8 programs compute identical K/V pointer offsets (`off_kvh=0`) and re-load the same K/V tiles. This is correct but redundant — an L2-friendly optimization (cooperative loading) is a follow-up. No materialization of the 8x replicated K/V occurs.

Assertion `H % num_kv_heads == 0` covers this case (8 % 1 == 0). `GROUP_SIZE` becomes 8, which Triton compiles to a 3-bit right shift.

### 7.3 GQA + QK-norm

See §1.5 — K-norm scale is `[head_dim]`, not `[num_kv_heads, head_dim]`, so the kernel needs no per-kv-head scale indexing. Verified by inspection of `triton_attention.py:316-324` and the Gemma 3/4 HF config (`Gemma3RMSNorm` weight is `[head_dim]`). **No changes to the QScale/KScale pointer math.**

### 7.4 Risks to call out explicitly

- **Constexpr explosion.** Adding `GROUP_SIZE` as a constexpr multiplies the Triton compile-cache by the number of distinct group sizes seen. In practice this is ~6 (1, 2, 4, 8, 16, plus MHA = H). Acceptable.
- **Bandit bucket cache invalidation.** Adding `num_kv_heads` to the bucket key string will invalidate all existing incumbents for MHA buckets unless the new key function produces the same string for MHA shapes (which §4 does — MHA shapes bypass the `is_gqa` branch entirely). Confirmed: no regression.
- **Grid feasibility on H100 with head_dim=512.** H100 has 228 KB shmem/SM. Even there, large tiles will OOM. Bandit will learn. Monitor first-trial success rate when running Gemma 4 global benchmarks and escalate if < 5% of configs compile.

---

## Part 8 — LOC and time estimates

| Area | LOC delta | Est time |
|---|---|---|
| `attn_fwd_kernel` signature + pointer math | +8 / -4 | 20 min |
| `flash_attn` launcher (assertions + kwarg + plumbing) | +15 / -2 | 20 min |
| `benchmark_one` + `main()` in f-string | +15 / -5 | 25 min |
| `ATTENTION_SHAPE_BUCKETS` (15 existing + 6 new) | +21 / -15 | 15 min |
| `attention_shape_bucket_key` GQA branch | +20 / 0 | 20 min |
| Tests (GQA bucket class, launcher assertions, benchmark compile) | +90 / 0 | 40 min |
| **Total** | **~170 LOC net** | **~2h implementation** |

**Test time:** ~30 min running locally (pure-Python tests + assertion-only launcher tests; no CUDA required for the main suite).

**Validation time:** ~1.5-2h on A100 / H100 to benchmark at least `gemma4_31b_local` and `llama3_70b_gqa` end-to-end against a PyTorch SDPA reference and confirm correctness + non-zero TFLOPs. Expect a first-run flurry of `reward=0` failures on `head_dim=512` configs (see §7.1) — this is not a bug.

**Total effort: ~4-5h from spec to merged PR**, assuming no unexpected Triton compile-error rabbit holes.

---

## Could-not-verify / VERIFY-AT-IMPLEMENTATION flags

1. **`F.scaled_dot_product_attention` GQA support in PyTorch 2.5+.** I was unable to fetch the PyTorch 2.5 release notes in this research pass. Recommendation in Part 6 is to sidestep this entirely by using `repeat_interleave` in the reference. If the reviewer wants to use native SDPA GQA, verify `enable_gqa` kwarg exists in the installed torch version before merging.
2. **`NUM_KV_HEADS` vs `GROUP_SIZE` constexpr choice.** I recommend passing both (see §1.2b option A) for cache-key clarity. If bandit cache churn is a concern, `NUM_KV_HEADS` can be dropped and only `GROUP_SIZE` kept — functionally equivalent, but less readable in compiled kernel names.
3. **Exact H100 shmem ceiling for `head_dim=512`.** I gave order-of-magnitude estimates in §7.1 but did not run an actual sweep. The bandit will produce ground truth on first run.

## References

- Current kernel: `src/research_engine/triton_attention.py:219-412`.
- Current bucket logic: `src/research_engine/triton_attention.py:57-116`.
- Gemma 4 architectural facts: `docs/research/gemma4-pipeline-audit.md:9-27`.
- Operator spec protocol: `src/research_engine/triton_operators.py:21-49`.
- Test patterns to mirror: `tests/test_triton_attention.py:250-276` (bucket tests), `:373-431` (benchmark compile tests), `:438-497` (QK-norm routing tests).
- HF transformers `modeling_llama.py` — `repeat_kv` pattern (not referenced directly; we replicate by pointer arithmetic instead of materializing).
- FlashAttention-2 (Dao 2023) §3.2 — GQA grid is over Q heads.
- FlashAttention tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
