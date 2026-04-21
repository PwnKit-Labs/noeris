from __future__ import annotations

import unittest
from unittest.mock import patch

from tests import _pathfix  # noqa: F401

from research_engine.spec_decode_runtime import (
    resolve_verify_accept_backend,
    verify_accept_tokens,
)


class SpecDecodeRuntimeSelectionTests(unittest.TestCase):
    def test_resolve_explicit(self) -> None:
        self.assertEqual(resolve_verify_accept_backend(prefer="separated"), "separated")
        self.assertEqual(resolve_verify_accept_backend(prefer="fused"), "fused")
        self.assertEqual(resolve_verify_accept_backend(prefer="flashinfer", flashinfer_fn=lambda a, b: (a, b)), "flashinfer")

    def test_resolve_auto(self) -> None:
        self.assertEqual(resolve_verify_accept_backend(prefer="auto", flashinfer_fn=None), "fused")
        self.assertEqual(resolve_verify_accept_backend(prefer="auto", flashinfer_fn=lambda a, b: (a, b)), "flashinfer")

    def test_invalid_preference(self) -> None:
        with self.assertRaises(ValueError):
            resolve_verify_accept_backend(prefer="unknown")

    def test_verify_accept_tokens_prefers_flashinfer_when_provided(self) -> None:
        fallback_len = object()
        fallback_mask = object()
        with patch(
            "research_engine.triton_spec_decode_verify_accept.verify_accept_fused",
            return_value=(fallback_len, fallback_mask),
        ), patch(
            "research_engine.triton_spec_decode_verify_accept.verify_accept_reference",
            return_value=(fallback_len, fallback_mask),
        ):
            out_len, out_mask, backend = verify_accept_tokens(
                object(),
                object(),
                prefer="auto",
                flashinfer_fn=lambda a, b: ("fi_len", "fi_mask"),
            )
        self.assertEqual((out_len, out_mask, backend), ("fi_len", "fi_mask", "flashinfer"))

    def test_verify_accept_tokens_falls_back_from_flashinfer_to_fused(self) -> None:
        with patch(
            "research_engine.triton_spec_decode_verify_accept.verify_accept_fused",
            return_value=("fused_len", "fused_mask"),
        ), patch(
            "research_engine.triton_spec_decode_verify_accept.verify_accept_reference",
            return_value=("ref_len", "ref_mask"),
        ):
            out_len, out_mask, backend = verify_accept_tokens(
                object(),
                object(),
                prefer="auto",
                flashinfer_fn=lambda a, b: (_ for _ in ()).throw(RuntimeError("flashinfer fail")),
            )
        self.assertEqual((out_len, out_mask, backend), ("fused_len", "fused_mask", "fused"))

    def test_verify_accept_tokens_falls_back_from_fused_to_separated(self) -> None:
        with patch(
            "research_engine.triton_spec_decode_verify_accept.verify_accept_fused",
            side_effect=RuntimeError("fused fail"),
        ), patch(
            "research_engine.triton_spec_decode_verify_accept.verify_accept_reference",
            return_value=("ref_len", "ref_mask"),
        ):
            out_len, out_mask, backend = verify_accept_tokens(
                object(),
                object(),
                prefer="fused",
            )
        self.assertEqual((out_len, out_mask, backend), ("ref_len", "ref_mask", "separated"))


if __name__ == "__main__":
    unittest.main()
