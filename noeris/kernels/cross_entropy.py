"""Fused cross-entropy with online log-sum-exp.

Single-pass kernel: computes max, log-sum-exp, and per-target loss without
materializing the full softmax probabilities.
"""

from __future__ import annotations

_kernel = None


def _ensure_compiled():
    global _kernel
    if _kernel is not None:
        return
    import triton
    import triton.language as tl

    @triton.jit
    def _cross_entropy_kernel(
        logits_ptr, target_ptr, loss_ptr,
        logits_row_stride, n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        logits_ptr += row_idx * logits_row_stride
        target = tl.load(target_ptr + row_idx)

        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        logits = tl.load(logits_ptr + offs, mask=mask, other=-float("inf")).to(tl.float32)

        row_max = tl.max(logits, axis=0)
        log_sum_exp = row_max + tl.log(tl.sum(tl.exp(logits - row_max), axis=0))

        target_logit = tl.load(logits_ptr + target).to(tl.float32)
        loss = log_sum_exp - target_logit

        tl.store(loss_ptr + row_idx, loss)

    _kernel = _cross_entropy_kernel


def cross_entropy_forward(logits, targets):
    """Fused cross-entropy forward. Returns per-row loss in fp32.

    Args:
        logits: (n_rows, vocab_size) fp16 tensor.
        targets: (n_rows,) int64 tensor.

    Returns:
        (n_rows,) fp32 loss tensor.
    """
    import torch
    import triton

    _ensure_compiled()
    n_rows, n_cols = logits.shape
    loss = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    _kernel[(n_rows,)](
        logits, targets, loss,
        logits.stride(0), n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=min(16, max(1, BLOCK_SIZE // 256)),
        num_stages=1,
    )
    return loss
