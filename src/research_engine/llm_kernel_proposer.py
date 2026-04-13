"""LLM-guided Triton kernel variant proposal.

Uses an LLM (Anthropic Claude or OpenAI) to propose modifications to
existing Triton kernel source code.  The bandit evaluates each variant
for correctness and performance.  If the LLM discovers an optimization
a human didn't write, that's a headline result.

Supports a "dry run" mode (no API key) that prints prompts without
calling any external service -- usable on Kaggle free tier.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

VARIANT_PROPOSAL_PROMPT = """\
You are an expert GPU kernel engineer specialising in Triton (OpenAI's \
Python-to-PTX compiler).  Your goal is to propose ONE specific, \
self-contained optimisation to the Triton kernel shown below.

## Hardware target
{hardware_info}

## Base kernel source
```python
{kernel_source}
```

## Current best performance
{performance_summary}

## Task
Propose exactly ONE modification that could improve throughput.  \
Pick from strategies such as:
- Vectorised loads / stores (tl.load with a wider element type)
- Different reduction strategy (two-pass, warp-shuffle)
- Shared-memory caching of weights or intermediate values
- Different BLOCK_SIZE / tile shape for better occupancy
- Loop unrolling or software pipelining hints
- Fusing an adjacent elementwise op into the same kernel
- Replacing divisions with reciprocal multiplications
- Re-ordering arithmetic to reduce register pressure

## Output format
Return ONLY a fenced Python code block containing the COMPLETE \
modified kernel function (including the @triton.jit decorator). \
Do NOT include driver code, benchmarks, or imports beyond \
`import triton` and `import triton.language as tl`.  \
Before the code block, write ONE sentence explaining the change.
"""

T4_HARDWARE_INFO = (
    "NVIDIA T4: 40 SMs, 65 TFLOPS fp16, 300 GB/s HBM, "
    "64 KB shared memory per SM, compute capability 7.5"
)
A100_HARDWARE_INFO = (
    "NVIDIA A100-80G: 108 SMs, 312 TFLOPS fp16, 2039 GB/s HBM2e, "
    "164 KB shared memory per SM, compute capability 8.0"
)
H100_HARDWARE_INFO = (
    "NVIDIA H100-SXM: 132 SMs, 989 TFLOPS fp16, 3352 GB/s HBM3, "
    "228 KB shared memory per SM, compute capability 9.0"
)

HARDWARE_PROFILES = {
    "t4": T4_HARDWARE_INFO,
    "a100": A100_HARDWARE_INFO,
    "h100": H100_HARDWARE_INFO,
}


# ---------------------------------------------------------------------------
# Lightweight LLM client (stdlib only -- no pip dependency)
# ---------------------------------------------------------------------------

def _call_anthropic(prompt: str, api_key: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Call the Anthropic Messages API and return the assistant text."""
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": model,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    data = json.dumps(payload).encode()
    req = Request(url, data=data, headers=headers, method="POST")
    with urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    # Messages API returns content blocks
    return "".join(b["text"] for b in body["content"] if b["type"] == "text")


def _call_openai(prompt: str, api_key: str, model: str = "gpt-4.1-mini") -> str:
    """Call the OpenAI Chat Completions API and return the assistant text."""
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = json.dumps(payload).encode()
    req = Request(url, data=data, headers=headers, method="POST")
    with urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Core proposer
# ---------------------------------------------------------------------------

@dataclass
class LLMKernelProposer:
    """Uses an LLM to propose Triton kernel variants."""

    operator_name: str
    provider: str = "anthropic"  # "anthropic" or "openai"
    api_key: str | None = None
    hardware: str = "t4"
    model: str | None = None  # override default model per provider

    def __post_init__(self) -> None:
        if self.api_key is None:
            if self.provider == "anthropic":
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            else:
                self.api_key = os.environ.get("OPENAI_API_KEY")

    @property
    def dry_run(self) -> bool:
        return self.api_key is None

    def _build_prompt(
        self,
        kernel_source: str,
        performance_history: list[dict[str, Any]] | None = None,
    ) -> str:
        hw = HARDWARE_PROFILES.get(self.hardware, T4_HARDWARE_INFO)

        if performance_history:
            lines = []
            for entry in performance_history[:10]:
                cfg = entry.get("config", "?")
                metric = entry.get("metric_value", "?")
                metric_name = entry.get("metric_name", "throughput")
                lines.append(f"  config={cfg}  {metric_name}={metric}")
            perf_summary = "\n".join(lines)
        else:
            perf_summary = "(no performance data yet)"

        return VARIANT_PROPOSAL_PROMPT.format(
            hardware_info=hw,
            kernel_source=kernel_source,
            performance_summary=perf_summary,
        )

    def propose_variant(
        self,
        kernel_source: str,
        performance_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, str]:
        """Ask the LLM to propose a modified kernel.

        Returns dict with keys: prompt, response (None if dry_run),
        variant_source (extracted code block or None).
        """
        prompt = self._build_prompt(kernel_source, performance_history)
        if self.dry_run:
            return {"prompt": prompt, "response": None, "variant_source": None}

        model_id = self.model
        if self.provider == "anthropic":
            response = _call_anthropic(prompt, self.api_key, model=model_id or "claude-sonnet-4-20250514")
        else:
            response = _call_openai(prompt, self.api_key, model=model_id or "gpt-4.1-mini")

        variant_source = _extract_code_block(response)
        return {
            "prompt": prompt,
            "response": response,
            "variant_source": variant_source,
        }

    def evaluate_variant(
        self,
        variant_source: str,
        test_shapes: list[dict[str, Any]],
        base_kernel_source: str | None = None,
    ) -> dict[str, Any]:
        """Write variant to a temp file, execute, check correctness + perf.

        Returns dict with: correct, throughput, error, variant_source.
        """
        script = _build_eval_script(variant_source, test_shapes, base_kernel_source)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="noeris_variant_"
        ) as f:
            f.write(script)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                return {
                    "correct": False,
                    "throughput": 0.0,
                    "error": result.stderr[-2000:] if result.stderr else "non-zero exit",
                    "variant_source": variant_source,
                }
            # Expect last line to be JSON
            out_lines = result.stdout.strip().splitlines()
            for line in reversed(out_lines):
                line = line.strip()
                if line.startswith("{"):
                    data = json.loads(line)
                    return {
                        "correct": data.get("correct", False),
                        "throughput": data.get("throughput", 0.0),
                        "error": data.get("error"),
                        "variant_source": variant_source,
                    }
            return {
                "correct": False,
                "throughput": 0.0,
                "error": "no JSON output found",
                "variant_source": variant_source,
            }
        except subprocess.TimeoutExpired:
            return {
                "correct": False,
                "throughput": 0.0,
                "error": "timeout (120s)",
                "variant_source": variant_source,
            }
        except Exception as exc:
            return {
                "correct": False,
                "throughput": 0.0,
                "error": str(exc),
                "variant_source": variant_source,
            }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_code_block(text: str) -> str | None:
    """Pull the first ```python ... ``` block from LLM output."""
    import re
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _build_eval_script(
    variant_source: str,
    test_shapes: list[dict[str, Any]],
    base_kernel_source: str | None = None,
) -> str:
    """Build a self-contained script that checks one kernel variant."""
    shapes_json = json.dumps(test_shapes)
    # Indent the variant source for embedding
    return f'''#!/usr/bin/env python3
"""Auto-generated variant evaluation script."""
import json, time, math, sys

try:
    import torch
    import triton
    import triton.language as tl
except ImportError:
    print(json.dumps({{"correct": False, "throughput": 0.0, "error": "triton/torch not available"}}))
    sys.exit(0)

# ---------- variant kernel ----------
{variant_source}

# ---------- reference (PyTorch RMSNorm) ----------
def ref_rmsnorm(x, w, eps=1e-6, affine_mode=0):
    x_f32 = x.float()
    rstd = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    if affine_mode == 0:
        return (x_f32 * rstd * w.float()).half()
    return (x_f32 * rstd * (1.0 + w.float())).half()

shapes = {shapes_json}
all_correct = True
total_throughput = 0.0
n_shapes = 0
error_msg = None

for shape in shapes:
    n_rows = shape.get("n_rows", 1024)
    hidden = shape.get("hidden_dim", 768)
    affine = shape.get("affine_mode", 0)
    try:
        x = torch.randn(n_rows, hidden, device="cuda", dtype=torch.float16)
        w = torch.randn(hidden, device="cuda", dtype=torch.float16)
        ref = ref_rmsnorm(x, w, affine_mode=affine)

        # Try to call the variant kernel -- look for the jit function
        import inspect
        variant_fn = None
        for name, obj in list(globals().items()):
            if hasattr(obj, "run") and hasattr(obj, "warmup"):
                variant_fn = obj
                break
        if variant_fn is None:
            raise RuntimeError("No @triton.jit kernel found in variant")

        # Launch
        y = torch.empty_like(x)
        BLOCK_SIZE = max(128, triton.next_power_of_2(hidden))
        grid = (n_rows,)
        # Attempt launch with standard RMSNorm signature
        variant_fn[grid](
            x, w, y,
            x.stride(0), y.stride(0),
            hidden, 1e-6,
            BLOCK_SIZE=BLOCK_SIZE,
            AFFINE_MODE=affine,
            num_warps=4, num_stages=1,
        )
        torch.cuda.synchronize()

        if not torch.allclose(ref, y, atol=1e-2, rtol=1e-2):
            all_correct = False
            error_msg = f"Correctness failed for shape {{n_rows}}x{{hidden}}"
            break

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            variant_fn[grid](
                x, w, y,
                x.stride(0), y.stride(0),
                hidden, 1e-6,
                BLOCK_SIZE=BLOCK_SIZE,
                AFFINE_MODE=affine,
                num_warps=4, num_stages=1,
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 100
        nbytes = (x.numel() + w.numel() + y.numel()) * 2  # fp16
        gb_per_s = nbytes / elapsed / 1e9
        total_throughput += gb_per_s
        n_shapes += 1
    except Exception as e:
        all_correct = False
        error_msg = str(e)[:500]
        break

avg_tp = total_throughput / max(n_shapes, 1)
print(json.dumps({{"correct": all_correct, "throughput": round(avg_tp, 2), "error": error_msg}}))
'''
