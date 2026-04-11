"""Tests for timing_snippet.py (Task 2: upstream KernelBench timing).

These run offline: we can't exercise the CUDA paths without a GPU, but we
verify the helper source is syntactically valid, install_noeris_timing
rewrites do_bench call sites, and both timer modes are reachable via
their dispatcher branches (monkeypatched).
"""

from __future__ import annotations

import types
import unittest

from tests import _pathfix  # noqa: F401

from research_engine.timing_snippet import (
    TIMING_HELPER_SOURCE,
    install_noeris_timing,
    make_timing_prelude,
)


class TestHelperSourceSyntax(unittest.TestCase):
    def test_helper_source_is_valid_python(self) -> None:
        """The embedded helper text must compile on its own."""
        code = (
            "import torch\nimport triton\nimport triton.testing\n"
            + TIMING_HELPER_SOURCE
            + '\nNOERIS_TIMER = "cuda_event"\n'
        )
        try:
            compile(code, "<timing_snippet>", "exec")
        except SyntaxError as exc:
            self.fail(f"Helper source has a syntax error: {exc}")

    def test_make_timing_prelude_default_is_cuda_event(self) -> None:
        prelude = make_timing_prelude()
        self.assertIn('NOERIS_TIMER = "cuda_event"', prelude)

    def test_make_timing_prelude_do_bench(self) -> None:
        prelude = make_timing_prelude(timer="do_bench")
        self.assertIn('NOERIS_TIMER = "do_bench"', prelude)

    def test_make_timing_prelude_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            make_timing_prelude(timer="wall_clock")

    def test_prelude_contains_clear_l2_cache(self) -> None:
        self.assertIn("_noeris_clear_l2_cache", TIMING_HELPER_SOURCE)

    def test_prelude_contains_cuda_event_timer(self) -> None:
        self.assertIn("_noeris_time_cuda_event", TIMING_HELPER_SOURCE)

    def test_prelude_hardcodes_3_warmup_10_trials_defaults(self) -> None:
        self.assertIn("num_warmup=3", TIMING_HELPER_SOURCE)
        self.assertIn("num_trials=10", TIMING_HELPER_SOURCE)


class TestInstallNoerisTiming(unittest.TestCase):
    _BASE = (
        "#!/usr/bin/env python3\n"
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "def main():\n"
        "    x = torch.randn(1024, 1024, device='cuda')\n"
        "    ms = triton.testing.do_bench(lambda: torch.relu(x), warmup=25, rep=100)\n"
        "    print(ms)\n"
    )

    def test_rewrites_do_bench_call_to_noeris_time(self) -> None:
        out = install_noeris_timing(self._BASE)
        # Exactly one occurrence of triton.testing.do_bench remains — the
        # one inside the injected prelude's do_bench-fallback branch. All
        # user-side call sites must be rewritten to noeris_time.
        self.assertEqual(out.count("triton.testing.do_bench("), 1)
        self.assertIn("noeris_time(", out)
        # The user's original call site must be gone.
        user_section = out.split('NOERIS_TIMER = "cuda_event"', 1)[1]
        self.assertNotIn("triton.testing.do_bench(", user_section)

    def test_rewritten_script_is_valid_python(self) -> None:
        out = install_noeris_timing(self._BASE)
        compile(out, "<rewritten>", "exec")

    def test_rewritten_script_sets_cuda_event_by_default(self) -> None:
        out = install_noeris_timing(self._BASE)
        self.assertIn('NOERIS_TIMER = "cuda_event"', out)

    def test_rewritten_script_can_select_do_bench(self) -> None:
        out = install_noeris_timing(self._BASE, timer="do_bench")
        self.assertIn('NOERIS_TIMER = "do_bench"', out)

    def test_helper_injected_after_triton_imports(self) -> None:
        out = install_noeris_timing(self._BASE)
        triton_pos = out.index("import triton.language")
        helper_pos = out.index("_noeris_clear_l2_cache")
        self.assertGreater(helper_pos, triton_pos)

    def test_multiple_do_bench_call_sites_all_rewritten(self) -> None:
        script = self._BASE + (
            "    ms2 = triton.testing.do_bench(lambda: torch.tanh(x), warmup=10, rep=50)\n"
        )
        out = install_noeris_timing(script)
        # The prelude defines `def noeris_time(` once; the script body
        # now has two rewritten call sites. Total: 3 occurrences of
        # "noeris_time(". Exactly one triton.testing.do_bench( remains:
        # the one inside the prelude's do_bench fallback branch.
        self.assertEqual(out.count("noeris_time("), 3)
        self.assertEqual(out.count("triton.testing.do_bench("), 1)


class TestNoerisTimeDispatch(unittest.TestCase):
    """Execute the helper source in an isolated namespace and verify the
    dispatcher correctly routes between cuda_event and do_bench without
    touching real CUDA. We stub torch.cuda and triton.testing entirely.
    """

    def _build_namespace(self, timer: str):
        # Minimal torch.cuda + triton.testing stubs.
        fake_torch = types.ModuleType("torch")

        class _Event:
            def __init__(self, enable_timing=True): self.t = 0.0
            def record(self): self.t = 1.23
            def elapsed_time(self, other): return 1.23

        def _ctx(device=None):
            class C:
                def __enter__(self_): return self_
                def __exit__(self_, *a): return False
            return C()

        fake_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            synchronize=lambda device=None: None,
            empty_cache=lambda: None,
            Event=_Event,
            device=_ctx,
        )
        fake_torch.cuda = fake_cuda
        fake_torch.empty = lambda *args, **kwargs: types.SimpleNamespace(fill_=lambda v: None)
        fake_torch.int64 = object()
        fake_triton = types.ModuleType("triton")
        fake_triton.testing = types.SimpleNamespace(
            do_bench=lambda fn, warmup=25, rep=100: 7.77,
        )
        ns: dict = {"torch": fake_torch, "triton": fake_triton}
        exec(TIMING_HELPER_SOURCE + f'\nNOERIS_TIMER = "{timer}"\n', ns)
        return ns

    def test_cuda_event_dispatch_returns_stub_median(self) -> None:
        ns = self._build_namespace(timer="cuda_event")
        result = ns["noeris_time"](lambda: None)
        self.assertAlmostEqual(result, 1.23, places=2)

    def test_do_bench_dispatch_routes_to_triton(self) -> None:
        ns = self._build_namespace(timer="do_bench")
        result = ns["noeris_time"](lambda: None)
        self.assertAlmostEqual(result, 7.77, places=2)

    def test_unknown_timer_raises(self) -> None:
        ns = self._build_namespace(timer="cuda_event")
        with self.assertRaises(ValueError):
            ns["noeris_time"](lambda: None, timer="not_a_timer")


class TestL2FlushCalledBetweenTrials(unittest.TestCase):
    """Verify _noeris_time_cuda_event calls _noeris_clear_l2_cache once per
    measurement trial (and not in the warmup loop)."""

    def test_flush_called_once_per_trial(self) -> None:
        fake_torch = types.ModuleType("torch")
        flush_calls = {"count": 0}

        class _Event:
            def record(self): pass
            def elapsed_time(self, other): return 0.5

        def _ctx(device=None):
            class C:
                def __enter__(self_): return self_
                def __exit__(self_, *a): return False
            return C()

        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            synchronize=lambda device=None: None,
            empty_cache=lambda: None,
            Event=lambda enable_timing=True: _Event(),
            device=_ctx,
        )
        fake_torch.int64 = object()

        # Instrumented empty() counts flush invocations.
        def _empty(*args, **kwargs):
            class T:
                def fill_(self_, v):
                    flush_calls["count"] += 1
            return T()

        fake_torch.empty = _empty
        fake_triton = types.ModuleType("triton")
        fake_triton.testing = types.SimpleNamespace(do_bench=lambda fn, **kw: 0.0)
        ns = {"torch": fake_torch, "triton": fake_triton}
        exec(TIMING_HELPER_SOURCE + '\nNOERIS_TIMER = "cuda_event"\n', ns)

        ns["noeris_time"](lambda: None, num_warmup=3, num_trials=10)
        self.assertEqual(
            flush_calls["count"], 10,
            f"L2 flush should fire once per measurement trial; got {flush_calls['count']}",
        )


if __name__ == "__main__":
    unittest.main()
