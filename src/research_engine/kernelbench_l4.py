"""KernelBench Level 4 op-substitution framework.

Level 4 consists of 20 HuggingFace model forward passes. This module
defines the addressable L4 problems (excluding BigBird and Reformer),
provides a ``NoerisOpSubstitutor`` that walks an ``nn.Module`` tree and
replaces Linear/LayerNorm/GELU modules with Noeris Triton kernel
wrappers, and generates self-contained benchmark scripts for GPU
execution.

Design notes
------------
* GPU-free at import time — ``torch`` is imported lazily so tests and
  script generation work on CPU-only CI environments.
* We do NOT inline Triton kernel source — L4 benchmarks run on the same
  GPU as the HF model, so ``research_engine.triton_*`` can be imported
  directly.  This differs from the L1 upstream runner which inlines
  everything for Modal isolation.
* Replacement uses ``type(m) is cls`` (exact match) to avoid subclass
  surprises.
* GPT-2 uses ``transformers.pytorch_utils.Conv1D`` (a transposed linear,
  NOT ``nn.Conv1d``).  The substitutor detects and replaces it.
* All Noeris kernels output fp16; wrappers cast at the boundary
  (fp32 -> fp16 -> kernel -> fp16 -> fp32) with weights cached in fp16
  at substitution time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


# ---------------------------------------------------------------------------
# L4 Problem definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class L4Problem:
    """One KernelBench Level-4 benchmark problem."""

    problem_id: int
    filename: str
    model_name: str
    batch_size: int
    seq_len: int
    arch_family: str
    expected_ops: tuple[str, ...]


# Full set of 15 addressable problems (BigBird x3 + Reformer x2 excluded).
L4_PROBLEMS: list[L4Problem] = [
    L4Problem(
        problem_id=1,
        filename="1_EleutherAI-gpt-neo-2p7B_bs32_seq256.py",
        model_name="EleutherAI/gpt-neo-2.7B",
        batch_size=32,
        seq_len=256,
        arch_family="GPT-Neo",
        expected_ops=("Linear", "LayerNorm", "GELU"),
    ),
    L4Problem(
        problem_id=2,
        filename="2_facebook-opt-1p3b_bs1_seq2047.py",
        model_name="facebook/opt-1.3b",
        batch_size=1,
        seq_len=2047,
        arch_family="OPT",
        expected_ops=("Linear", "LayerNorm"),
    ),
    L4Problem(
        problem_id=3,
        filename="3_EleutherAI-gpt-neo-2p7B_bs1_seq2047.py",
        model_name="EleutherAI/gpt-neo-2.7B",
        batch_size=1,
        seq_len=2047,
        arch_family="GPT-Neo",
        expected_ops=("Linear", "LayerNorm", "GELU"),
    ),
    L4Problem(
        problem_id=4,
        filename="4_facebook-opt-1p3b_bs32_seq256.py",
        model_name="facebook/opt-1.3b",
        batch_size=32,
        seq_len=256,
        arch_family="OPT",
        expected_ops=("Linear", "LayerNorm"),
    ),
    # 5, 9, 10 — BigBird: block-sparse attention, skipped
    L4Problem(
        problem_id=6,
        filename="6_facebook-bart-large_bs1_seq1023.py",
        model_name="facebook/bart-large",
        batch_size=1,
        seq_len=1023,
        arch_family="BART",
        expected_ops=("Linear", "LayerNorm", "GELU"),
    ),
    L4Problem(
        problem_id=7,
        filename="7_gpt2_bs32_seq256.py",
        model_name="gpt2",
        batch_size=32,
        seq_len=256,
        arch_family="GPT-2",
        expected_ops=("Linear", "LayerNorm", "GELU", "Conv1D"),
    ),
    L4Problem(
        problem_id=8,
        filename="8_facebook-opt-1p3b_bs512_seq32.py",
        model_name="facebook/opt-1.3b",
        batch_size=512,
        seq_len=32,
        arch_family="OPT",
        expected_ops=("Linear", "LayerNorm"),
    ),
    L4Problem(
        problem_id=11,
        filename="11_google-electra-small-discriminator_bs1_seq511.py",
        model_name="google/electra-small-discriminator",
        batch_size=1,
        seq_len=511,
        arch_family="Electra",
        expected_ops=("Linear", "LayerNorm", "GELU"),
    ),
    L4Problem(
        problem_id=12,
        filename="12_google-electra-small-discriminator_bs1024_seq32.py",
        model_name="google/electra-small-discriminator",
        batch_size=1024,
        seq_len=32,
        arch_family="Electra",
        expected_ops=("Linear", "LayerNorm", "GELU"),
    ),
    # 13, 15 — Reformer: LSH attention, skipped
    L4Problem(
        problem_id=14,
        filename="14_google-electra-small-discriminator_bs32_seq256.py",
        model_name="google/electra-small-discriminator",
        batch_size=32,
        seq_len=256,
        arch_family="Electra",
        expected_ops=("Linear", "LayerNorm", "GELU"),
    ),
    L4Problem(
        problem_id=16,
        filename="16_gpt2_bs1_seq1023.py",
        model_name="gpt2",
        batch_size=1,
        seq_len=1023,
        arch_family="GPT-2",
        expected_ops=("Linear", "LayerNorm", "GELU", "Conv1D"),
    ),
    L4Problem(
        problem_id=17,
        filename="17_facebook-bart-large_bs1024_seq32.py",
        model_name="facebook/bart-large",
        batch_size=1024,
        seq_len=32,
        arch_family="BART",
        expected_ops=("Linear", "LayerNorm", "GELU"),
    ),
    L4Problem(
        problem_id=18,
        filename="18_EleutherAI-gpt-neo-2p7B_bs512_seq32.py",
        model_name="EleutherAI/gpt-neo-2.7B",
        batch_size=512,
        seq_len=32,
        arch_family="GPT-Neo",
        expected_ops=("Linear", "LayerNorm", "GELU"),
    ),
    L4Problem(
        problem_id=19,
        filename="19_gpt2_bs1024_seq32.py",
        model_name="gpt2",
        batch_size=1024,
        seq_len=32,
        arch_family="GPT-2",
        expected_ops=("Linear", "LayerNorm", "GELU", "Conv1D"),
    ),
    L4Problem(
        problem_id=20,
        filename="20_facebook-bart-large_bs32_seq256.py",
        model_name="facebook/bart-large",
        batch_size=32,
        seq_len=256,
        arch_family="BART",
        expected_ops=("Linear", "LayerNorm", "GELU"),
    ),
]

# Attack order from the feasibility study (top-5).
L4_ATTACK_ORDER: list[int] = [14, 4, 20, 1, 7]

# Skipped problem IDs (BigBird x3 + Reformer x2).
L4_SKIPPED_IDS: list[int] = [5, 9, 10, 13, 15]


def get_l4_problems() -> list[L4Problem]:
    """Return all 15 addressable L4 problems."""
    return list(L4_PROBLEMS)


def get_l4_attack_problems() -> list[L4Problem]:
    """Return the top-5 attack-order problems."""
    by_id = {p.problem_id: p for p in L4_PROBLEMS}
    return [by_id[pid] for pid in L4_ATTACK_ORDER]


# ---------------------------------------------------------------------------
# Op-substitutor wrappers
#
# All wrapper classes import torch/nn lazily.  The classes themselves are
# defined as plain objects; the ``_make_wrapper_classes()`` factory builds
# proper ``nn.Module`` subclasses at first use (when torch is guaranteed
# importable).
# ---------------------------------------------------------------------------

_WRAPPER_CLASSES_CACHE: dict[str, type] | None = None


def _make_wrapper_classes() -> dict[str, type]:
    """Build nn.Module wrapper subclasses (requires torch at call time)."""
    global _WRAPPER_CLASSES_CACHE
    if _WRAPPER_CLASSES_CACHE is not None:
        return _WRAPPER_CLASSES_CACHE

    import torch
    import torch.nn as nn

    class NoerisLinearWrapper(nn.Module):
        """Drop-in replacement for ``nn.Linear`` using Noeris matmul kernel."""

        def __init__(self, linear: nn.Linear) -> None:
            super().__init__()
            self.weight_fp16 = nn.Parameter(
                linear.weight.data.to(torch.float16).contiguous(), requires_grad=False
            )
            self.bias_fp16: torch.Tensor | None = None
            if linear.bias is not None:
                self.bias_fp16 = nn.Parameter(
                    linear.bias.data.to(torch.float16).contiguous(), requires_grad=False
                )
            self.out_features = linear.out_features
            self.in_features = linear.in_features

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            orig_shape = x.shape
            orig_dtype = x.dtype
            x2 = x.reshape(-1, x.shape[-1]).contiguous().to(torch.float16)
            from research_engine.triton_kernels import noeris_matmul, TRITON_MATMUL_CURATED_CONFIGS
            cfg = TRITON_MATMUL_CURATED_CONFIGS[0]
            out = noeris_matmul(x2, self.weight_fp16.t().contiguous(), cfg)
            if self.bias_fp16 is not None:
                out = out + self.bias_fp16
            return out.reshape(*orig_shape[:-1], self.out_features).to(orig_dtype)

    class NoerisConv1DWrapper(nn.Module):
        """Drop-in replacement for ``transformers.pytorch_utils.Conv1D``.

        Conv1D stores weight as ``(in_features, out_features)`` and computes
        ``x @ weight + bias`` — it is a transposed linear, NOT ``nn.Conv1d``.
        """

        def __init__(self, conv1d) -> None:
            super().__init__()
            self.weight_fp16 = nn.Parameter(
                conv1d.weight.data.to(torch.float16).contiguous(), requires_grad=False
            )
            self.bias_fp16: torch.Tensor | None = None
            if conv1d.bias is not None:
                self.bias_fp16 = nn.Parameter(
                    conv1d.bias.data.to(torch.float16).contiguous(), requires_grad=False
                )
            self.nf = conv1d.nf  # out_features

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            orig_shape = x.shape
            orig_dtype = x.dtype
            x2 = x.reshape(-1, x.shape[-1]).contiguous().to(torch.float16)
            from research_engine.triton_kernels import noeris_matmul, TRITON_MATMUL_CURATED_CONFIGS
            cfg = TRITON_MATMUL_CURATED_CONFIGS[0]
            out = noeris_matmul(x2, self.weight_fp16, cfg)
            if self.bias_fp16 is not None:
                out = out + self.bias_fp16
            return out.reshape(*orig_shape[:-1], self.nf).to(orig_dtype)

    class NoerisLayerNormWrapper(nn.Module):
        """Drop-in replacement for ``nn.LayerNorm`` using Noeris layernorm kernel."""

        def __init__(self, ln: nn.LayerNorm) -> None:
            super().__init__()
            self.normalized_shape = ln.normalized_shape
            self.eps = ln.eps
            self.weight_fp16: torch.Tensor | None = None
            self.bias_fp16: torch.Tensor | None = None
            if ln.weight is not None:
                self.weight_fp16 = nn.Parameter(
                    ln.weight.data.to(torch.float16).contiguous(), requires_grad=False
                )
            if ln.bias is not None:
                self.bias_fp16 = nn.Parameter(
                    ln.bias.data.to(torch.float16).contiguous(), requires_grad=False
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            orig_dtype = x.dtype
            orig_shape = x.shape
            hidden = self.normalized_shape[-1]
            x2 = x.reshape(-1, hidden).contiguous().to(torch.float16)
            from research_engine.triton_layernorm import noeris_layernorm
            out = noeris_layernorm(x2, self.weight_fp16, self.bias_fp16, self.eps)
            return out.reshape(orig_shape).to(orig_dtype)

    class NoerisGELUWrapper(nn.Module):
        """Drop-in replacement for ``nn.GELU`` using Noeris geglu kernel with gate=1."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            orig_dtype = x.dtype
            orig_shape = x.shape
            x2 = x.reshape(-1, x.shape[-1]).contiguous().to(torch.float16)
            from research_engine.triton_geglu import noeris_geglu
            gate = torch.ones_like(x2)
            out = noeris_geglu(x2, gate)
            return out.reshape(orig_shape).to(orig_dtype)

    _WRAPPER_CLASSES_CACHE = {
        "NoerisLinearWrapper": NoerisLinearWrapper,
        "NoerisConv1DWrapper": NoerisConv1DWrapper,
        "NoerisLayerNormWrapper": NoerisLayerNormWrapper,
        "NoerisGELUWrapper": NoerisGELUWrapper,
    }
    return _WRAPPER_CLASSES_CACHE


# Convenience accessors so callers can do ``from kernelbench_l4 import NoerisLinearWrapper``.
# These are lazy — they import torch on first access.

def __getattr__(name: str) -> Any:
    """Module-level __getattr__ for lazy wrapper class access.

    Allows ``from kernelbench_l4 import NoerisLinearWrapper`` to work
    while deferring the torch import until actually needed.
    """
    _wrapper_names = {
        "NoerisLinearWrapper",
        "NoerisConv1DWrapper",
        "NoerisLayerNormWrapper",
        "NoerisGELUWrapper",
    }
    if name in _wrapper_names:
        return _make_wrapper_classes()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# NoerisOpSubstitutor
# ---------------------------------------------------------------------------

def _resolve_parent(root, dotted_name: str):
    """Resolve a dotted module name to (parent_module, attr_name)."""
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


class NoerisOpSubstitutor:
    """Walk an ``nn.Module`` tree and replace supported ops with Noeris wrappers.

    Replacement rules (exact type match via ``type(m) is cls``):
      * ``nn.Linear``              -> ``NoerisLinearWrapper``
      * ``nn.LayerNorm``           -> ``NoerisLayerNormWrapper``
      * ``nn.modules.activation.GELU`` -> ``NoerisGELUWrapper``
      * ``transformers.pytorch_utils.Conv1D`` -> ``NoerisConv1DWrapper``

    Requires torch at instantiation time.
    """

    def __init__(self) -> None:
        import torch.nn as nn
        wrappers = _make_wrapper_classes()
        self._rules: list[tuple[type, type]] = [
            (nn.Linear, wrappers["NoerisLinearWrapper"]),
            (nn.LayerNorm, wrappers["NoerisLayerNormWrapper"]),
            (nn.modules.activation.GELU, wrappers["NoerisGELUWrapper"]),
        ]
        # Try to add Conv1D rule if transformers is installed.
        try:
            from transformers.pytorch_utils import Conv1D
            self._rules.append((Conv1D, wrappers["NoerisConv1DWrapper"]))
            self._conv1d_cls: type | None = Conv1D
        except ImportError:
            self._conv1d_cls = None

    @property
    def rules(self) -> list[tuple[type, type]]:
        """Return the active replacement rules."""
        return list(self._rules)

    def substitute(self, model) -> dict[str, int]:
        """Replace all matched modules in-place. Returns op -> count mapping."""
        counts: dict[str, int] = {}
        for name, module in list(model.named_modules()):
            for cls, wrapper_cls in self._rules:
                if type(module) is cls:
                    parent, attr = _resolve_parent(model, name)
                    wrapped = wrapper_cls(module)
                    setattr(parent, attr, wrapped)
                    op_name = cls.__name__
                    counts[op_name] = counts.get(op_name, 0) + 1
                    break
        return counts


# ---------------------------------------------------------------------------
# Benchmark script generation
# ---------------------------------------------------------------------------

def generate_l4_benchmark_script(
    problems: list[L4Problem] | None = None,
) -> str:
    """Generate a self-contained Python script for L4 benchmarking.

    The generated script:
      1. Installs ``transformers`` if needed.
      2. For each problem, loads the HF model, runs original forward, applies
         ``NoerisOpSubstitutor``, runs substituted forward, checks correctness.
      3. Reports JSON results to stdout.

    Unlike the L1 upstream runner, kernels are imported directly from
    ``research_engine.triton_*`` (same GPU, no Modal isolation needed).
    """
    if problems is None:
        problems = get_l4_attack_problems()

    problems_json = json.dumps(
        [
            {
                "problem_id": p.problem_id,
                "filename": p.filename,
                "model_name": p.model_name,
                "batch_size": p.batch_size,
                "seq_len": p.seq_len,
                "arch_family": p.arch_family,
                "expected_ops": list(p.expected_ops),
            }
            for p in problems
        ],
        indent=2,
    )

    script = f'''#!/usr/bin/env python3
"""Auto-generated: Noeris KernelBench Level 4 op-substitution benchmark."""
import json
import subprocess
import sys
import time
import traceback

# Ensure transformers is installed.
try:
    import transformers
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers', '-q'])
    import transformers

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModel, AutoConfig

# ---------------------------------------------------------------------------
# Noeris op-substitutor (imported directly — same GPU, no Modal)
# ---------------------------------------------------------------------------
from research_engine.kernelbench_l4 import NoerisOpSubstitutor

PROBLEMS = {problems_json}


# ---------------------------------------------------------------------------
# Timing helpers (cuda_event with L2 flush)
# ---------------------------------------------------------------------------

def noeris_time(fn, n_warmup=3, n_iters=10):
    """Time a callable using CUDA events with L2 cache flush."""
    # Warmup
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    # L2 flush buffer (allocate once)
    try:
        l2_flush = torch.empty(40 * 1024 * 1024, dtype=torch.int8, device="cuda")
    except Exception:
        l2_flush = None

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n_iters):
        # L2 flush
        if l2_flush is not None:
            l2_flush.zero_()
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    return sum(times) / len(times)


# ---------------------------------------------------------------------------
# Per-problem benchmark
# ---------------------------------------------------------------------------

def _load_model(model_name, arch_family):
    """Load HF model on CUDA in eval mode."""
    if arch_family in ("BART",):
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval().cuda()
    elif arch_family in ("Electra",):
        model = AutoModel.from_pretrained(model_name).eval().cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).eval().cuda()
    return model


def benchmark_l4_problem(problem):
    """Benchmark one L4 problem. Returns a result dict."""
    model_name = problem["model_name"]
    batch_size = problem["batch_size"]
    seq_len = problem["seq_len"]
    arch_family = problem["arch_family"]

    print(f"  Loading {{model_name}} ({{arch_family}})...", flush=True)
    model = _load_model(model_name, arch_family)

    # Random input tokens
    vocab_size = model.config.vocab_size if hasattr(model.config, "vocab_size") else 30522
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

    # ---- Original forward ----
    with torch.no_grad():
        orig_out = model(input_ids)
        orig_logits = orig_out.logits if hasattr(orig_out, "logits") else orig_out.last_hidden_state

    original_ms = noeris_time(lambda: model(input_ids))

    # ---- Substitute ops ----
    substitutor = NoerisOpSubstitutor()
    ops_replaced = substitutor.substitute(model)
    print(f"    Replaced: {{ops_replaced}}", flush=True)

    # ---- Substituted forward ----
    with torch.no_grad():
        subst_out = model(input_ids)
        subst_logits = subst_out.logits if hasattr(subst_out, "logits") else subst_out.last_hidden_state

    substituted_ms = noeris_time(lambda: model(input_ids))

    # ---- Correctness ----
    correct = torch.allclose(
        orig_logits.float(), subst_logits.float(), atol=5e-3, rtol=5e-3
    )

    speedup = original_ms / substituted_ms if substituted_ms > 0 else 0.0

    return {{
        "model": model_name,
        "arch_family": arch_family,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "original_ms": round(original_ms, 3),
        "substituted_ms": round(substituted_ms, 3),
        "speedup": round(speedup, 4),
        "correct": correct,
        "ops_replaced": ops_replaced,
    }}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Noeris KernelBench L4 Op-Substitution Benchmark")
    print("=" * 60)
    results = []
    for problem in PROBLEMS:
        pid = problem["problem_id"]
        print(f"\\n[Problem #{{pid}}] {{problem['filename']}}")
        try:
            result = benchmark_l4_problem(problem)
            result["problem_id"] = pid
            results.append(result)
            status = "PASS" if result["correct"] else "FAIL"
            print(
                f"    {{status}} | orig={{result['original_ms']:.1f}}ms "
                f"subst={{result['substituted_ms']:.1f}}ms "
                f"speedup={{result['speedup']:.3f}}x"
            )
        except Exception:
            print(f"    ERROR: {{traceback.format_exc()}}")
            results.append({{"problem_id": pid, "error": traceback.format_exc()}})

    print("\\n" + "=" * 60)
    print("JSON results:")
    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
'''
    return script
