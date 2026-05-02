"""Integration tests for noeris.patch() on real HuggingFace models.

These tests verify:
1. Architecture detection works for supported model types
2. Patching replaces the right modules
3. Forward pass still produces valid outputs
4. Backward pass works (gradients flow through patched modules)
5. Unpatch restores original behavior

Requires: transformers, torch, triton (GPU tests marked with @pytest.mark.gpu)
"""

import pytest


def _require_torch_cuda() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU noeris tests")


# ---- Architecture detection tests (no GPU needed) ----

class TestDetection:
    def test_detect_llama(self):
        from noeris._detect import detect_architecture

        class FakeConfig:
            model_type = "llama"

        class FakeModel:
            config = FakeConfig()

        arch = detect_architecture(FakeModel())
        assert arch is not None
        assert arch.model_type == "llama"
        assert arch.has_qk_norm is False
        assert arch.affine_mode == 0
        assert arch.activation == "swiglu"

    def test_detect_gemma2(self):
        from noeris._detect import detect_architecture

        class FakeConfig:
            model_type = "gemma2"

        class FakeModel:
            config = FakeConfig()

        arch = detect_architecture(FakeModel())
        assert arch is not None
        assert arch.model_type == "gemma2"
        assert arch.has_qk_norm is True
        assert arch.affine_mode == 1
        assert arch.activation == "geglu"

    def test_detect_qwen3(self):
        from noeris._detect import detect_architecture

        class FakeConfig:
            model_type = "qwen3"

        class FakeModel:
            config = FakeConfig()

        arch = detect_architecture(FakeModel())
        assert arch is not None
        assert arch.has_qk_norm is True
        assert arch.affine_mode == 0
        assert arch.activation == "swiglu"

    def test_detect_unknown_returns_none(self):
        from noeris._detect import detect_architecture

        class FakeConfig:
            model_type = "totally_unknown_model"

        class FakeModel:
            config = FakeConfig()

        assert detect_architecture(FakeModel()) is None

    def test_detect_no_config_returns_none(self):
        from noeris._detect import detect_architecture

        class FakeModel:
            pass

        assert detect_architecture(FakeModel()) is None


# ---- Kernel correctness tests (require GPU) ----

@pytest.mark.gpu
class TestKernelCorrectness:
    @pytest.fixture(autouse=True)
    def _require_gpu_stack(self):
        _require_torch_cuda()

    def test_rmsnorm_forward(self):
        import torch
        from noeris.kernels.rmsnorm import rmsnorm_forward

        x = torch.randn(128, 1024, device="cuda", dtype=torch.float16)
        w = torch.randn(1024, device="cuda", dtype=torch.float16)

        out = rmsnorm_forward(x, w, eps=1e-6, affine_mode=0)

        # PyTorch reference
        x_f = x.float()
        var = x_f.pow(2).mean(-1, keepdim=True)
        ref = (x_f * torch.rsqrt(var + 1e-6) * w.float()).half()

        assert (out - ref).abs().max().item() < 0.05

    def test_rmsnorm_gemma_mode(self):
        import torch
        from noeris.kernels.rmsnorm import rmsnorm_forward

        x = torch.randn(128, 1536, device="cuda", dtype=torch.float16)
        w = torch.randn(1536, device="cuda", dtype=torch.float16) * 0.1

        out = rmsnorm_forward(x, w, eps=1e-6, affine_mode=1)

        x_f = x.float()
        var = x_f.pow(2).mean(-1, keepdim=True)
        ref = (x_f * torch.rsqrt(var + 1e-6) * (1.0 + w.float())).half()

        assert (out - ref).abs().max().item() < 0.05

    def test_geglu_forward(self):
        import torch
        from noeris.kernels.geglu import geglu_forward

        gate = torch.randn(256, 2048, device="cuda", dtype=torch.float16)
        up = torch.randn(256, 2048, device="cuda", dtype=torch.float16)

        out = geglu_forward(gate, up)
        ref = torch.nn.functional.gelu(up.float(), approximate="tanh").half() * gate

        assert (out - ref).abs().max().item() < 0.02

    def test_swiglu_forward(self):
        import torch
        from noeris.kernels.geglu import swiglu_forward

        gate = torch.randn(256, 2048, device="cuda", dtype=torch.float16)
        up = torch.randn(256, 2048, device="cuda", dtype=torch.float16)

        out = swiglu_forward(gate, up)
        ref = (torch.nn.functional.silu(up.float()) * gate.float()).half()

        assert (out - ref).abs().max().item() < 0.02

    def test_cross_entropy_forward(self):
        import torch
        from noeris.kernels.cross_entropy import cross_entropy_forward

        logits = torch.randn(64, 32000, device="cuda", dtype=torch.float16)
        targets = torch.randint(0, 32000, (64,), device="cuda")

        out = cross_entropy_forward(logits, targets)
        ref = torch.nn.functional.cross_entropy(
            logits.float(), targets, reduction="none",
        )

        assert (out - ref).abs().max().item() < 0.5


# ---- Autograd tests (require GPU) ----

@pytest.mark.gpu
class TestAutograd:
    @pytest.fixture(autouse=True)
    def _require_gpu_stack(self):
        _require_torch_cuda()

    def test_rmsnorm_autograd(self):
        import torch
        from noeris._autograd import TritonRMSNorm

        x = torch.randn(32, 512, device="cuda", dtype=torch.float16, requires_grad=True)
        w = torch.randn(512, device="cuda", dtype=torch.float16, requires_grad=True)

        out = TritonRMSNorm.apply(x, w, 1e-6, 0)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert w.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(w.grad).any()

    def test_swiglu_autograd(self):
        import torch
        from noeris._autograd import TritonSwiGLU

        gate = torch.randn(32, 1024, device="cuda", dtype=torch.float16, requires_grad=True)
        up = torch.randn(32, 1024, device="cuda", dtype=torch.float16, requires_grad=True)

        out = TritonSwiGLU.apply(gate, up)
        loss = out.sum()
        loss.backward()

        assert gate.grad is not None
        assert up.grad is not None
        assert not torch.isnan(gate.grad).any()
        assert not torch.isnan(up.grad).any()

    def test_qk_norm_rope_autograd(self):
        import torch
        from noeris._autograd import TritonQKNormRoPE

        B, H, S, D = 1, 8, 64, 128
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True)
        cos = torch.randn(S, D // 2, device="cuda", dtype=torch.float32)
        sin = torch.randn(S, D // 2, device="cuda", dtype=torch.float32)
        q_scale = torch.randn(D, device="cuda", dtype=torch.float32, requires_grad=True) * 0.1
        k_scale = torch.randn(D, device="cuda", dtype=torch.float32, requires_grad=True) * 0.1

        q_out, k_out = TritonQKNormRoPE.apply(q, k, cos, sin, q_scale, k_scale, 1e-6, 1)
        loss = q_out.sum() + k_out.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert q_scale.grad is not None
        assert k_scale.grad is not None


# ---- Patch integration tests (require GPU + transformers) ----

@pytest.mark.gpu
class TestPatchIntegration:
    @pytest.fixture(autouse=True)
    def _require_gpu_stack(self):
        _require_torch_cuda()

    @pytest.fixture
    def _check_transformers(self):
        pytest.importorskip("transformers")

    def test_patch_unpatch_roundtrip(self, _check_transformers):
        """Verify patch/unpatch doesn't break model structure."""
        import torch
        import noeris

        # Create a minimal model-like structure with fake config
        from noeris._detect import _ARCH_REGISTRY

        class FakeRMSNorm(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(dim))
                self.variance_epsilon = 1e-6

            def forward(self, x):
                var = x.float().pow(2).mean(-1, keepdim=True)
                return (x * torch.rsqrt(var + self.variance_epsilon) * self.weight).to(x.dtype)

        # Register fake class name temporarily
        FakeRMSNorm.__name__ = "LlamaRMSNorm"
        FakeRMSNorm.__qualname__ = "LlamaRMSNorm"

        class FakeConfig:
            model_type = "llama"

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = FakeConfig()
                self.norm = FakeRMSNorm(64)

        model = FakeModel()
        original_class = type(model.norm).__name__

        noeris.patch(model, kernels=["rmsnorm"], verbose=True)
        assert hasattr(model, "_noeris_patched")
        assert type(model.norm).__name__ == "NoerisRMSNorm"

        noeris.unpatch(model, verbose=True)
        assert not hasattr(model, "_noeris_patched")
        assert type(model.norm).__name__ == original_class
