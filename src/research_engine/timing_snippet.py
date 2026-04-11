"""Reusable timing helper embedded into every generated benchmark script.

The upstream KernelBench methodology is:
  * torch.cuda.Event start/end markers
  * 3 warmup trials + 10 measurement trials
  * L2 cache flush before every measurement trial (256 MB int64 dummy fill)
  * Return the median of the measurement trials (in ms)

The Noeris default historically was triton.testing.do_bench with warmup=25ms,
rep=100ms and *no* L2 flush. On small-tensor kernels that fit in L2 the hot
cache inflates measured bandwidth 2-5x, which in turn inflates reported
speedups. Task 2 in the P0 credibility pass switches the default to match
upstream so our numbers are apples-to-apples comparable.

Two timers are exposed via ``noeris_time(fn, timer=...)``:

    timer="cuda_event"  -> upstream-compatible (new default)
    timer="do_bench"    -> legacy Triton adaptive (preserved for
                          backwards-compatibility of old result artifacts)

The body of this module is also exposed as a raw string
(``TIMING_HELPER_SOURCE``) so each generated benchmark script can embed it
verbatim at the top, without taking a runtime dependency on Noeris being
importable inside the Modal container.
"""

from __future__ import annotations


# IMPORTANT: this source is copy-pasted verbatim into every generated
# benchmark script. It must be self-contained: only `torch` and
# `triton.testing` imports are assumed already present.
TIMING_HELPER_SOURCE = r'''
# ---------------------------------------------------------------------------
# Noeris timing helper (auto-embedded). Matches upstream KernelBench
# methodology by default: cuda_event timer, 3 warmup + 10 trials, L2 flush
# between trials, median in ms. The do_bench path is preserved for
# backwards-compatibility with old result artifacts.
# ---------------------------------------------------------------------------
import statistics as _noeris_stats

def _noeris_clear_l2_cache(device=None):
    """Thrash L2 by filling a ~256 MB int64 dummy tensor.

    Matches upstream src/kernelbench/timing.py::clear_l2_cache (32*1024*1024
    int64 = 256 MB, larger than every current-gen L2). Cold-cache timings
    are what KernelBench reports; do_bench does not do this.
    """
    if device is None and torch.cuda.is_available():
        device = torch.cuda.current_device()
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    dummy.fill_(42)
    del dummy

def _noeris_time_cuda_event(fn, *, num_warmup=3, num_trials=10, device=None):
    """Upstream-compatible cuda_event timer.

    Runs `fn` ``num_warmup`` times (warmup, not recorded), then ``num_trials``
    measurement trials with an L2 flush before each, and returns the median
    elapsed time in milliseconds across measurement trials.
    """
    if device is None and torch.cuda.is_available():
        device = torch.cuda.current_device()
    with torch.cuda.device(device):
        for _ in range(num_warmup):
            fn()
            torch.cuda.synchronize(device=device)
        torch.cuda.empty_cache()
        times_ms = []
        for _ in range(num_trials):
            torch.cuda.synchronize(device=device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)
            _noeris_clear_l2_cache(device=device)
            start_event.record()
            _ = fn()
            end_event.record()
            torch.cuda.synchronize(device=device)
            times_ms.append(start_event.elapsed_time(end_event))
    return float(_noeris_stats.median(times_ms))

def noeris_time(fn, *, warmup=25, rep=100, timer=None, num_warmup=3, num_trials=10):
    """Unified timer entry point used by every generated benchmark script.

    Signature intentionally accepts the old do_bench kwargs (``warmup``,
    ``rep``) so mechanical replacement is safe. If ``timer`` is not
    supplied explicitly, the script-level ``NOERIS_TIMER`` constant
    decides; that constant is set by ``make_timing_prelude`` at
    script-generation time and defaults to "cuda_event".

    Returns a median time in milliseconds in both modes.
    """
    if timer is None:
        timer = globals().get("NOERIS_TIMER", "cuda_event")
    if timer == "cuda_event":
        return _noeris_time_cuda_event(fn, num_warmup=num_warmup, num_trials=num_trials)
    if timer == "do_bench":
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    raise ValueError(f"unknown timer={timer!r}; expected 'cuda_event' or 'do_bench'")

# Default timer honored by every downstream call site in this script.
# Override at script-generation time (NOERIS_TIMER) to switch.
# ---------------------------------------------------------------------------
'''


def make_timing_prelude(timer: str = "cuda_event") -> str:
    """Return the embeddable timing helper source + a top-level ``NOERIS_TIMER``
    constant set to the requested timer.
    """
    if timer not in ("cuda_event", "do_bench"):
        raise ValueError(f"unknown timer={timer!r}")
    return TIMING_HELPER_SOURCE + f'\nNOERIS_TIMER = "{timer}"\n'


def install_noeris_timing(script: str, *, timer: str = "cuda_event") -> str:
    """Post-process a generated benchmark script so its timing calls use
    the Noeris unified timer instead of ``triton.testing.do_bench``.

    The approach is deliberately conservative:
      1. Inject the timing helper source + the ``NOERIS_TIMER`` constant
         right after the first ``import triton`` line (so the helper sees
         ``torch`` and ``triton`` symbols).
      2. Mechanically replace ``triton.testing.do_bench(`` with
         ``noeris_time(`` everywhere in the script. The existing kwargs
         ``warmup=`` and ``rep=`` are accepted as pass-throughs by
         ``noeris_time`` (ignored under cuda_event, honored under do_bench)
         so no further edits are needed.
      3. Append ``, timer=NOERIS_TIMER`` to each replaced call — this is
         done by detecting balanced parentheses in the replaced callsite.
         If the detection is unsure, we fall back to leaving the default
         (cuda_event) in place, which is the right behavior anyway.

    This lets us switch the default timer with zero edits to 8 operator-
    specific f-string generators and their baseline snippets, which
    sharply reduces the blast radius of the P0 credibility pass.
    """
    # Mechanical replace of do_bench call sites in the *user* script only.
    # We must do this BEFORE injecting the prelude, because the prelude's
    # own do_bench-fallback branch uses ``triton.testing.do_bench(`` and
    # must NOT be rewritten (otherwise timer="do_bench" would infinitely
    # recurse into noeris_time).
    rewritten_user = script.replace("triton.testing.do_bench(", "noeris_time(")

    prelude = make_timing_prelude(timer)

    # Inject the prelude right after the first "import triton" block so the
    # helper sees ``torch`` and ``triton`` symbols.
    lines = rewritten_user.splitlines()
    inject_idx = None
    for i, line in enumerate(lines):
        if line.startswith("import triton") or line.startswith("import triton."):
            inject_idx = i + 1
            while inject_idx < len(lines) and (
                lines[inject_idx].startswith("import triton")
                or lines[inject_idx].startswith("from triton")
                or lines[inject_idx].startswith("import triton.language")
            ):
                inject_idx += 1
            break
    if inject_idx is None:
        inject_idx = 0
    lines.insert(inject_idx, prelude)
    return "\n".join(lines)
