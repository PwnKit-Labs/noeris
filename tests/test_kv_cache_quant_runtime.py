from __future__ import annotations

import unittest
from unittest.mock import patch

from tests import _pathfix  # noqa: F401

from research_engine.kv_cache_quant_runtime import (
    quantize_kv_write,
    resolve_kv_quant_backend,
)


class KvCacheQuantRuntimeTests(unittest.TestCase):
    def test_resolve_backend(self) -> None:
        self.assertEqual(resolve_kv_quant_backend(prefer="separated"), "separated")
        self.assertEqual(resolve_kv_quant_backend(prefer="fused"), "fused")
        self.assertEqual(resolve_kv_quant_backend(prefer="auto", external_fn=None), "fused")
        self.assertEqual(
            resolve_kv_quant_backend(prefer="auto", external_fn=lambda x: (x, x)),
            "external",
        )

    def test_invalid_preference(self) -> None:
        with self.assertRaises(ValueError):
            resolve_kv_quant_backend(prefer="bad")

    def test_fallback_from_external_to_fused(self) -> None:
        with patch(
            "research_engine.triton_kv_quant_write.kv_quantize_write_fused",
            return_value=("fused_q", "fused_s"),
        ), patch(
            "research_engine.triton_kv_quant_write.kv_quantize_separated",
            return_value=("sep_q", "sep_s"),
        ):
            q, s, backend = quantize_kv_write(
                object(),
                prefer="auto",
                external_fn=lambda x: (_ for _ in ()).throw(RuntimeError("fail")),
            )
        self.assertEqual((q, s, backend), ("fused_q", "fused_s", "fused"))

    def test_fallback_from_fused_to_separated(self) -> None:
        with patch(
            "research_engine.triton_kv_quant_write.kv_quantize_write_fused",
            side_effect=RuntimeError("fused fail"),
        ), patch(
            "research_engine.triton_kv_quant_write.kv_quantize_separated",
            return_value=("sep_q", "sep_s"),
        ):
            q, s, backend = quantize_kv_write(object(), prefer="fused")
        self.assertEqual((q, s, backend), ("sep_q", "sep_s", "separated"))


if __name__ == "__main__":
    unittest.main()
